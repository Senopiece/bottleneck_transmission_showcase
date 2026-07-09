from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from receiver_tools.dataset import default_dataset_dir, repo_root_from_tools

from led_reader.dataset import LedReaderDataset, bad_video_negative_samples, led_video_samples, split_samples, synthetic_led_samples
from led_reader.model import CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH, create_model


def parse_args() -> argparse.Namespace:
    root = repo_root_from_tools()
    parser = argparse.ArgumentParser(description="Train shared LED crop scorer.")
    parser.add_argument("--dataset", type=Path, default=default_dataset_dir())
    parser.add_argument("--out", type=Path, default=root / "receiver" / "models" / "led_reader" / "led_reader_crop_v003_gate.pt")
    parser.add_argument("--include", nargs="*", default=None, help="Optional good video stems.")
    parser.add_argument("--max-frames", type=int, default=360)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--synthetic-samples", type=int, default=2500)
    parser.add_argument("--bad-negative-samples", type=int, default=160)
    parser.add_argument("--rates", nargs="*", type=int, default=[8])
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.14)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def pick_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weighted_bce_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
    return (loss * weight).sum() / torch.clamp(weight.sum(), min=1.0)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    correct = 0
    count = 0
    on_correct = 0
    on_count = 0
    off_correct = 0
    off_count = 0
    margins: list[float] = []
    all_logits: list[float] = []
    all_expected: list[bool] = []
    raw_correct = 0
    raw_margins: list[float] = []
    gate_values: list[float] = []
    for x, likelihood, y, w in loader:
        x = x.to(device)
        likelihood = likelihood.to(device)
        y = y.to(device)
        w = w.to(device)
        logits = model(x, likelihood)
        raw_logits = model.raw_logits(x) if hasattr(model, "raw_logits") else logits
        gates = model.evidence_gate(raw_logits, likelihood) if hasattr(model, "evidence_gate") else torch.ones_like(logits)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
        total_loss += float((loss * w).sum().item())
        total_weight += float(w.sum().item())
        pred = logits >= 0
        raw_pred = raw_logits >= 0
        expected = y >= 0.5
        correct_mask = pred == expected
        raw_correct += int((raw_pred == expected).sum().item())
        correct += int(correct_mask.sum().item())
        count += int(y.numel())
        on_mask = expected
        off_mask = ~expected
        if bool(on_mask.any()):
            on_correct += int(correct_mask[on_mask].sum().item())
            on_count += int(on_mask.sum().item())
        if bool(off_mask.any()):
            off_correct += int(correct_mask[off_mask].sum().item())
            off_count += int(off_mask.sum().item())
        signed_margin = torch.where(expected, logits, -logits)
        raw_signed_margin = torch.where(expected, raw_logits, -raw_logits)
        margins.extend(float(v) for v in signed_margin.detach().cpu().tolist())
        raw_margins.extend(float(v) for v in raw_signed_margin.detach().cpu().tolist())
        gate_values.extend(float(v) for v in gates.detach().cpu().tolist())
        all_logits.extend(float(v) for v in logits.detach().cpu().tolist())
        all_expected.extend(bool(v) for v in expected.detach().cpu().tolist())
    margins_arr = np.array(margins, dtype=np.float32) if margins else np.zeros(1, dtype=np.float32)
    raw_margins_arr = np.array(raw_margins, dtype=np.float32) if raw_margins else np.zeros(1, dtype=np.float32)
    gate_arr = np.array(gate_values, dtype=np.float32) if gate_values else np.ones(1, dtype=np.float32)
    threshold, balanced_accuracy = best_threshold(all_logits, all_expected)
    return {
        "loss": total_loss / max(1.0, total_weight),
        "accuracy": correct / max(1, count),
        "raw_accuracy": raw_correct / max(1, count),
        "on_accuracy": on_correct / max(1, on_count),
        "off_accuracy": off_correct / max(1, off_count),
        "margin_mean": float(np.mean(margins_arr)),
        "margin_p05": float(np.percentile(margins_arr, 5)),
        "raw_margin_p05": float(np.percentile(raw_margins_arr, 5)),
        "gate_mean": float(np.mean(gate_arr)),
        "gate_p05": float(np.percentile(gate_arr, 5)),
        "threshold_logit": threshold,
        "balanced_accuracy": balanced_accuracy,
    }


def best_threshold(logits: list[float], expected: list[bool]) -> tuple[float, float]:
    if not logits or not expected:
        return 0.0, 0.0
    values = np.array(logits, dtype=np.float32)
    labels = np.array(expected, dtype=bool)
    candidates = np.unique(values)
    if len(candidates) > 512:
        candidates = np.percentile(values, np.linspace(1.0, 99.0, 511)).astype(np.float32)
    best_t = 0.0
    best_score = -1.0
    for threshold in candidates:
        pred = values >= threshold
        on_mask = labels
        off_mask = ~labels
        on_acc = float(np.mean(pred[on_mask] == labels[on_mask])) if bool(on_mask.any()) else 1.0
        off_acc = float(np.mean(pred[off_mask] == labels[off_mask])) if bool(off_mask.any()) else 1.0
        score = 0.5 * (on_acc + off_acc)
        if score > best_score:
            best_score = score
            best_t = float(threshold)
    return best_t, best_score


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    samples = synthetic_led_samples(args.synthetic_samples, args.seed)
    samples.extend(
        led_video_samples(
            args.dataset,
            include=set(args.include) if args.include else None,
            max_frames_per_video=args.max_frames,
            stride=args.stride,
            rates=tuple(args.rates),
            seed=args.seed + 1,
        )
    )
    if args.bad_negative_samples > 0:
        samples.extend(
            bad_video_negative_samples(
                args.dataset,
                max_samples_per_video=args.bad_negative_samples,
                seed=args.seed + 5,
            )
        )
    if not samples:
        raise SystemExit("No LED training samples generated.")
    train_samples, val_samples = split_samples(samples, args.validation_fraction, args.seed + 2)
    pos = sum(1 for sample in train_samples if sample.target >= 0.5)
    neg = len(train_samples) - pos
    pos_weight = torch.tensor(neg / max(1, pos), dtype=torch.float32, device=device).clamp(0.50, 4.0)
    print(f"label balance train on={pos} off={neg} pos_weight={float(pos_weight.item()):.2f}")
    train_loader = DataLoader(
        LedReaderDataset(train_samples, augment=True, seed=args.seed + 3),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        LedReaderDataset(val_samples, augment=False, seed=args.seed + 4),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = create_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    best_val = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_weight = 0.0
        progress = tqdm(train_loader, desc=f"led epoch {epoch}/{args.epochs}", leave=False)
        for x, likelihood, y, w in progress:
            x = x.to(device)
            likelihood = likelihood.to(device)
            y = y.to(device)
            w = w.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, likelihood)
            loss = weighted_bce_loss(logits, y, w, pos_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            batch_weight = float(w.sum().item())
            running_loss += float(loss.item()) * batch_weight
            running_weight += batch_weight
            progress.set_postfix(loss=running_loss / max(1.0, running_weight))
        scheduler.step()
        train_loss = running_loss / max(1.0, running_weight)
        metrics = evaluate(model, val_loader, device)
        row = {"epoch": float(epoch), "train_loss": train_loss, **metrics}
        history.append(row)
        print(
            f"epoch={epoch} train={train_loss:.4f} val={metrics['loss']:.4f} "
            f"acc={metrics['accuracy']:.3f} on={metrics['on_accuracy']:.3f} "
            f"off={metrics['off_accuracy']:.3f} margin05={metrics['margin_p05']:.2f} "
            f"gate={metrics['gate_mean']:.2f}/{metrics['gate_p05']:.2f} "
            f"rawAcc={metrics['raw_accuracy']:.3f} thr={metrics['threshold_logit']:.2f} "
            f"bacc={metrics['balanced_accuracy']:.3f}"
        )
        if metrics["loss"] < best_val:
            best_val = metrics["loss"]
            save_checkpoint(args.out, model, args, history)
    save_checkpoint(args.out.with_name(args.out.stem + "_last.pt"), model, args, history)
    print(f"saved best {args.out}")
    print(f"samples train={len(train_samples)} val={len(val_samples)} device={device}")


def save_checkpoint(path: Path, model: nn.Module, args: argparse.Namespace, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": "LedCropNet",
            "input_shape": [CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH],
            "inputs": ["led_crop", "detector_likelihood"],
            "args": vars(args),
            "history": history,
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "arch": "LedCropNet",
                "input_shape": [CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH],
                "inputs": ["led_crop", "detector_likelihood"],
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "last_metrics": history[-1] if history else {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
