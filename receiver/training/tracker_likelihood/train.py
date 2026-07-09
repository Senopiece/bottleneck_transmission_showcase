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

from tracker_likelihood.dataset import TrackerLikelihoodDataset, split_samples, synthetic_samples, video_mined_samples
from tracker_likelihood.model import PATCH_CHANNELS, PATCH_HEIGHT, PATCH_WIDTH, create_model


def parse_args() -> argparse.Namespace:
    root = repo_root_from_tools()
    parser = argparse.ArgumentParser(description="Train tracker likelihood patch CNN.")
    parser.add_argument("--dataset", type=Path, default=default_dataset_dir())
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "receiver" / "models" / "tracker_likelihood" / "tracker_likelihood_fast_v003.pt",
    )
    parser.add_argument("--synthetic-samples", type=int, default=12000)
    parser.add_argument("--real-max-frames", type=int, default=420)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.12)
    parser.add_argument("--hard-negative-min-score", type=float, default=0.40)
    parser.add_argument("--include", nargs="*", default=None, help="Optional video stems for weak-label mining.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def pick_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weighted_bce_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * weight).sum() / torch.clamp(weight.sum(), min=1.0)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    correct = 0
    hard_count = 0
    soft_abs_error = 0.0
    for x, y, w in loader:
        x = x.to(device)
        y = y.to(device)
        w = w.to(device)
        logits = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
        total_loss += float((loss * w).sum().item())
        total_weight += float(w.sum().item())
        prob = torch.sigmoid(logits)
        hard_mask = (y <= 0.05) | (y >= 0.95)
        if bool(hard_mask.any()):
            expected = y[hard_mask] >= 0.5
            actual = prob[hard_mask] >= 0.5
            correct += int((expected == actual).sum().item())
            hard_count += int(hard_mask.sum().item())
        soft_abs_error += float((torch.abs(prob - y) * w).sum().item())
    return {
        "loss": total_loss / max(1.0, total_weight),
        "hard_accuracy": correct / max(1, hard_count),
        "weighted_mae": soft_abs_error / max(1.0, total_weight),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    samples = []
    samples.extend(synthetic_samples(args.synthetic_samples, args.seed))
    samples.extend(
        video_mined_samples(
            args.dataset,
            max_frames_per_video=args.real_max_frames,
            stride=args.stride,
            hard_negative_min_score=args.hard_negative_min_score,
            include=set(args.include) if args.include else None,
            seed=args.seed + 1,
        )
    )
    if not samples:
        raise SystemExit("No training samples generated.")

    train_samples, val_samples = split_samples(samples, args.validation_fraction, args.seed + 2)
    train_loader = DataLoader(
        TrackerLikelihoodDataset(train_samples, augment=True, seed=args.seed + 3),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TrackerLikelihoodDataset(val_samples, augment=False, seed=args.seed + 4),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = create_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_weight = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for x, y, w in progress:
            x = x.to(device)
            y = y.to(device)
            w = w.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = weighted_bce_loss(logits, y, w)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            batch_weight = float(w.sum().item())
            running_loss += float(loss.item()) * batch_weight
            running_weight += batch_weight
            progress.set_postfix(loss=running_loss / max(1.0, running_weight))
        scheduler.step()
        train_loss = running_loss / max(1.0, running_weight)
        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(row)
        print(
            f"epoch={epoch} train={train_loss:.4f} val={val_metrics['loss']:.4f} "
            f"acc={val_metrics['hard_accuracy']:.3f} mae={val_metrics['weighted_mae']:.4f}"
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(args.out, model, args, history)

    save_checkpoint(args.out.with_name(args.out.stem + "_last.pt"), model, args, history)
    print(f"saved best {args.out}")
    print(f"samples train={len(train_samples)} val={len(val_samples)} device={device}")


def save_checkpoint(path: Path, model: nn.Module, args: argparse.Namespace, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": "FastMarkerLikelihoodNet",
            "input_shape": [PATCH_CHANNELS, PATCH_HEIGHT, PATCH_WIDTH],
            "args": vars(args),
            "history": history,
        },
        path,
    )
    sidecar = path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "arch": "FastMarkerLikelihoodNet",
                "input_shape": [PATCH_CHANNELS, PATCH_HEIGHT, PATCH_WIDTH],
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "last_metrics": history[-1] if history else {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
