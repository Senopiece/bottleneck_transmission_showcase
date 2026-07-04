from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from receiver_tools.dataset import repo_root_from_tools
from receiver_tools.geometry import patch_to_feature
from receiver_tools.simple_mlp import Mlp, train_bce
from receiver_tools.synthetic import render_marker_patch


def parse_args() -> argparse.Namespace:
    root = repo_root_from_tools()
    parser = argparse.ArgumentParser(description="Bootstrap LED reader model.")
    parser.add_argument("--out", type=Path, default=root / "receiver" / "models" / "led_reader" / "led_reader_bootstrap_v001.npz")
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(123)
    x: list[np.ndarray] = []
    y: list[np.ndarray] = []
    for _ in range(args.samples):
        bits = rng.integers(0, 2, size=5).astype(np.float32)
        jitter = (
            float(rng.normal(0, 0.015)),
            float(rng.normal(0, 0.018)),
            float(rng.normal(0, 0.04)),
            float(rng.normal(0, 0.03)),
        )
        patch = render_marker_patch(bits, size=(96, 24), rng=rng, pose_jitter=jitter)
        x.append(patch_to_feature(patch, (64, 16)))
        y.append(bits)

    x_arr = np.stack(x).astype(np.float32)
    y_arr = np.stack(y).astype(np.float32)
    model = Mlp.create(input_dim=x_arr.shape[1], hidden_dim=96, output_dim=5, seed=9)
    train_bce(model, x_arr, y_arr, epochs=args.epochs, batch_size=128, lr=2e-3, seed=10)
    model.save(args.out, input_width=np.array([64]), input_height=np.array([16]), kind=np.array(["led_reader"]))
    print(f"saved {args.out} samples={len(x_arr)}")


if __name__ == "__main__":
    main()
