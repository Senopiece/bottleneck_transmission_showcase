from __future__ import annotations

import argparse
from pathlib import Path

from receiver_tools.dataset import default_dataset_dir, default_derived_dir, discover_videos
from receiver_tools.runner import run_video, write_summary
from receiver_tools.cv_marker import CvTrackerBackend
from receiver_tools.tracker_backend import StubTrackerBackend


def parse_args() -> argparse.Namespace:
    derived = default_derived_dir()
    parser = argparse.ArgumentParser(description="Run tracker/acquirer over receiver dataset videos.")
    parser.add_argument("--dataset", type=Path, default=default_dataset_dir())
    parser.add_argument("--overlay-out", type=Path, default=derived / "overlays" / "tracker_v003")
    parser.add_argument("--metrics-out", type=Path, default=derived / "metrics" / "tracker_v003")
    parser.add_argument("--backend", choices=("stub", "cv"), default="cv")
    parser.add_argument("--tracker-model", type=Path, default=None)
    parser.add_argument("--accept-score", type=float, default=0.48)
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Optional video stems to process, for example: good4 good5 bad1",
    )
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend == "cv":
        backend = CvTrackerBackend(tracker_model=args.tracker_model, accept_score=args.accept_score)
    else:
        backend = StubTrackerBackend()

    videos = discover_videos(args.dataset)
    if args.include:
        wanted = set(args.include)
        videos = [item for item in videos if item.name in wanted]
    if not videos:
        raise SystemExit(f"No videos found in {args.dataset}")

    summary = []
    for item in videos:
        overlay_path = None if args.no_overlays else args.overlay_out / item.kind / f"{item.name}_overlay.mp4"
        csv_path = args.metrics_out / item.kind / f"{item.name}.csv"
        summary.append(
            run_video(
                item=item,
                backend=backend,
                overlay_path=overlay_path,
                csv_path=csv_path,
                max_frames=args.max_frames,
                stride=max(1, args.stride),
            )
        )

    write_summary(summary, args.metrics_out / "summary.csv")
    print(f"Wrote metrics to {args.metrics_out}")
    if not args.no_overlays:
        print(f"Wrote overlays to {args.overlay_out}")


if __name__ == "__main__":
    main()
