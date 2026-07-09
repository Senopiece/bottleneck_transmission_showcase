from __future__ import annotations

import argparse
import csv
from pathlib import Path

from receiver_tools.dataset import default_derived_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize tracker dataset metrics.")
    parser.add_argument("--summary", type=Path, default=default_derived_dir() / "metrics" / "tracker_v003" / "summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.summary.exists():
        raise SystemExit(f"Summary file not found: {args.summary}")
    with args.summary.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit("Summary is empty")

    total_bad_fp = sum(int(row["false_positive_frames"]) for row in rows if row["kind"] == "bad")
    print(f"videos={len(rows)}")
    print(f"bad_false_positive_frames={total_bad_fp}")
    for row in rows:
        print(
            f"{row['kind']:4s} {row['video']}: "
            f"hit_rate={float(row['hit_rate']):.4f} "
            f"frames={row['processed_frames']} fp={row['false_positive_frames']}"
        )


if __name__ == "__main__":
    main()
