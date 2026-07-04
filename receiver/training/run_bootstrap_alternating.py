from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Bootstrap alternating tracker/LED training and render predictions.")
    parser.add_argument("--max-frames", type=int, default=450)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--tracker-samples", type=int, default=3000)
    parser.add_argument("--led-samples", type=int, default=6000)
    parser.add_argument("--tracker-epochs", type=int, default=14)
    parser.add_argument("--led-epochs", type=int, default=18)
    parser.add_argument("--accept-score", type=float, default=0.48)
    return parser.parse_args()


def run(cmd: list[str], cwd: Path) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def uv_python_cmd(script: Path, *args: str) -> list[str]:
    return ["uv", "run", "python", str(script), *args]


def main() -> None:
    args = parse_args()
    receiver = Path(__file__).resolve().parents[1]
    training = receiver / "training"
    tools = receiver / "tools"
    tracker_model = receiver / "models" / "tracker_likelihood" / "tracker_likelihood_bootstrap_v001.npz"
    led_model = receiver / "models" / "led_reader" / "led_reader_bootstrap_v001.npz"
    overlays = receiver / "datasets" / "derived" / "overlays" / "tracker_bootstrap_v001"
    metrics = receiver / "datasets" / "derived" / "metrics" / "tracker_bootstrap_v001"

    run(
        [
            *uv_python_cmd(
                training / "tracker_likelihood" / "train.py",
                "--samples",
                str(args.tracker_samples),
                "--epochs",
                str(args.tracker_epochs),
                "--real-max-frames",
                str(args.max_frames),
                "--stride",
                str(args.stride),
                "--out",
                str(tracker_model),
            ),
        ],
        cwd=training,
    )
    run(
        [
            *uv_python_cmd(
                training / "led_reader" / "train.py",
                "--samples",
                str(args.led_samples),
                "--epochs",
                str(args.led_epochs),
                "--out",
                str(led_model),
            ),
        ],
        cwd=training,
    )
    run(
        [
            *uv_python_cmd(
                tools / "run_tracker_dataset" / "run_tracker_dataset.py",
                "--backend",
                "cv",
                "--tracker-model",
                str(tracker_model),
                "--overlay-out",
                str(overlays),
                "--metrics-out",
                str(metrics),
                "--stride",
                str(args.stride),
                "--max-frames",
                str(args.max_frames),
                "--accept-score",
                str(args.accept_score),
            ),
        ],
        cwd=training,
    )
    print(f"tracker_model={tracker_model}")
    print(f"led_model={led_model}")
    print(f"overlays={overlays}")
    print(f"metrics={metrics}")


if __name__ == "__main__":
    main()
