from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from receiver_tools.cv_marker import CvMarkerDetector
from receiver_tools.dataset import default_dataset_dir, default_derived_dir, discover_videos, repo_root_from_tools
from receiver_tools.geometry import patch_to_feature, warp_marker_patch
from receiver_tools.simple_mlp import Mlp, train_bce
from receiver_tools.synthetic import render_marker_patch


def parse_args() -> argparse.Namespace:
    root = repo_root_from_tools()
    parser = argparse.ArgumentParser(description="Bootstrap tracker likelihood model.")
    parser.add_argument("--dataset", type=Path, default=default_dataset_dir())
    parser.add_argument("--out", type=Path, default=root / "receiver" / "models" / "tracker_likelihood" / "tracker_likelihood_bootstrap_v001.npz")
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--real-max-frames", type=int, default=220)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--hard-negative-min-score", type=float, default=0.42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(42)
    x: list[np.ndarray] = []
    y: list[list[float]] = []

    for _ in range(args.samples):
        bits = rng.integers(0, 2, size=5).astype(np.float32)
        jitter = (
            float(rng.normal(0, 0.035)),
            float(rng.normal(0, 0.04)),
            float(rng.normal(0, 0.10)),
            float(rng.normal(0, 0.08)),
        )
        patch = render_marker_patch(bits, rng=rng, pose_jitter=jitter)
        x.append(patch_to_feature(patch, (64, 24)))
        pose_error = (jitter[0] / 0.08) ** 2 + (jitter[1] / 0.08) ** 2 + (jitter[2] / 0.18) ** 2 + (jitter[3] / 0.14) ** 2
        y.append([float(np.exp(-pose_error))])

    detector = CvMarkerDetector()
    for item in discover_videos(args.dataset):
        cap = cv2.VideoCapture(str(item.path))
        frame_index = 0
        accepted = 0
        hard_negative = 0
        while cap.isOpened() and frame_index < args.real_max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % args.stride == 0:
                if item.kind == "good":
                    candidate = detector.detect(frame)
                    if candidate and candidate.score >= 0.55:
                        patch = warp_marker_patch(frame, candidate.pose)
                        x.append(patch_to_feature(patch, (64, 24)))
                        y.append([1.0])
                        accepted += 1
                else:
                    candidate = detector.detect(frame)
                    if candidate and candidate.score >= args.hard_negative_min_score:
                        patch = warp_marker_patch(frame, candidate.pose)
                        x.append(patch_to_feature(patch, (64, 24)))
                        y.append([0.0])
                        hard_negative += 1
                    h, w = frame.shape[:2]
                    for _ in range(2):
                        crop_w = int(rng.integers(max(48, w // 10), max(64, w // 3)))
                        crop_h = max(24, crop_w * 3 // 8)
                        x0 = int(rng.integers(0, max(1, w - crop_w)))
                        y0 = int(rng.integers(0, max(1, h - crop_h)))
                        crop = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
                        x.append(patch_to_feature(crop, (64, 24)))
                        y.append([0.0])
            frame_index += 1
        cap.release()
        print(f"{item.kind} {item.name}: pseudo_positive={accepted} hard_negative={hard_negative}")

    x_arr = np.stack(x).astype(np.float32)
    y_arr = np.array(y, dtype=np.float32)
    model = Mlp.create(input_dim=x_arr.shape[1], hidden_dim=96, output_dim=1, seed=7)
    train_bce(model, x_arr, y_arr, epochs=args.epochs, batch_size=128, lr=2e-3, seed=8)
    model.save(args.out, input_width=np.array([64]), input_height=np.array([24]), kind=np.array(["tracker_likelihood"]))
    print(f"saved {args.out} samples={len(x_arr)}")


if __name__ == "__main__":
    main()
