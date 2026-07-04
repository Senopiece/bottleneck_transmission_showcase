from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import cv2
from tqdm import tqdm

from receiver_tools.dataset import VideoItem
from receiver_tools.overlay import draw_result
from receiver_tools.tracker_backend import TrackerBackend, TrackerResult


CSV_FIELDS = [
    "video",
    "kind",
    "frame",
    "timestamp_ns",
    "hit",
    "mode",
    "confidence",
    "x",
    "y",
    "angle_rad",
    "scale_px",
    "square_x",
    "square_y",
    "triangle_x",
    "triangle_y",
    "led_scores",
    "debug",
]


def run_video(
    item: VideoItem,
    backend: TrackerBackend,
    overlay_path: Path | None,
    csv_path: Path,
    max_frames: int | None = None,
    stride: int = 1,
) -> dict[str, float | int | str]:
    capture = cv2.VideoCapture(str(item.path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {item.path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    writer = None
    if overlay_path is not None:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(overlay_path), fourcc, fps / max(1, stride), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create overlay video: {overlay_path}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    backend.reset()

    processed = 0
    hits = 0
    false_positive_frames = 0
    previous_hit = False
    tracking_drops = 0
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        csv_writer.writeheader()

        progress_total = frame_count if max_frames is None else min(frame_count, max_frames)
        with tqdm(total=progress_total, desc=item.name, unit="frame") as progress:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if max_frames is not None and frame_index >= max_frames:
                    break

                if frame_index % stride == 0:
                    timestamp_ns = int(frame_index / fps * 1_000_000_000)
                    result = backend.process(frame, timestamp_ns)
                    processed += 1
                    if previous_hit and not result.hit:
                        tracking_drops += 1
                    previous_hit = result.hit
                    if result.hit:
                        hits += 1
                        if item.kind == "bad":
                            false_positive_frames += 1
                    csv_writer.writerow(result_to_row(item, frame_index, timestamp_ns, result))
                    if writer is not None:
                        writer.write(draw_result(frame, result, frame_index, timestamp_ns, tracking_drops))

                frame_index += 1
                progress.update(1)

    capture.release()
    if writer is not None:
        writer.release()

    return {
        "video": item.name,
        "kind": item.kind,
        "processed_frames": processed,
        "hit_frames": hits,
        "false_positive_frames": false_positive_frames,
        "hit_rate": hits / processed if processed else 0.0,
    }


def result_to_row(item: VideoItem, frame_index: int, timestamp_ns: int, result: TrackerResult) -> dict[str, object]:
    square_x, square_y = unpack_point(result.square)
    triangle_x, triangle_y = unpack_point(result.triangle)
    return {
        "video": item.name,
        "kind": item.kind,
        "frame": frame_index,
        "timestamp_ns": timestamp_ns,
        "hit": int(result.hit),
        "mode": result.mode,
        "confidence": f"{result.confidence:.6f}",
        "x": f"{result.x:.3f}",
        "y": f"{result.y:.3f}",
        "angle_rad": f"{result.angle_rad:.6f}",
        "scale_px": f"{result.scale_px:.3f}",
        "square_x": f"{square_x:.3f}" if square_x is not None else "",
        "square_y": f"{square_y:.3f}" if square_y is not None else "",
        "triangle_x": f"{triangle_x:.3f}" if triangle_x is not None else "",
        "triangle_y": f"{triangle_y:.3f}" if triangle_y is not None else "",
        "led_scores": "|".join(f"{value:.6f}" for value in result.led_scores),
        "debug": result.debug,
    }


def unpack_point(point: tuple[float, float] | None) -> tuple[float | None, float | None]:
    if point is None:
        return None, None
    return point


def write_summary(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
