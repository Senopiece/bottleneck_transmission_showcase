from __future__ import annotations

import cv2
import numpy as np

from receiver_tools.geometry import LED_RADIUS_FRACTION, Pose, square_corners, triangle_corners
from receiver_tools.tracker_backend import TrackerResult


GOOD_COLOR = (0, 255, 255)
MISS_COLOR = (80, 80, 255)
TEXT_COLOR = (245, 245, 245)
TEXT_BG = (20, 20, 20)


def draw_result(
    frame: np.ndarray,
    result: TrackerResult,
    frame_index: int,
    timestamp_ns: int,
    tracking_drops: int = 0,
) -> np.ndarray:
    out = frame.copy()
    status = "HIT" if result.hit else "MISS"
    color = GOOD_COLOR if result.hit else MISS_COLOR
    text = (
        f"{status} mode={result.mode} conf={result.confidence:.3f} "
        f"frame={frame_index} t={timestamp_ns / 1e9:.3f}s {result.debug}"
    )
    draw_label(out, text, (16, 24), color)
    draw_label(out, f"tracking drops: {tracking_drops}", (16, out.shape[0] - 18), color)

    if result.hit:
        draw_pose(out, result, color)

    return out


def draw_label(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(frame, (x - 6, y - height - 8), (x + width + 6, y + baseline + 6), TEXT_BG, -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_pose(frame: np.ndarray, result: TrackerResult, color: tuple[int, int, int]) -> None:
    pose = Pose(center=(result.x, result.y), angle_rad=result.angle_rad, scale_px=result.scale_px)
    draw_marker_box(frame, pose, color)
    draw_triangle(frame, pose, color)
    for index, point in enumerate(result.leds):
        score = result.led_scores[index] if index < len(result.led_scores) else 0.0
        center = (round(point[0]), round(point[1]))
        radius = max(3, round(result.scale_px * LED_RADIUS_FRACTION))
        cv2.circle(frame, center, radius, color, 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"{index}:{score:.2f}",
            (center[0] + radius + 3, center[1] - radius),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_marker_box(frame: np.ndarray, pose: Pose, color: tuple[int, int, int]) -> None:
    points = np.array([(round(x), round(y)) for x, y in square_corners(pose)], dtype=np.int32)
    cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)


def draw_triangle(
    frame: np.ndarray,
    pose: Pose,
    color: tuple[int, int, int],
) -> None:
    points = np.array([(round(x), round(y)) for x, y in triangle_corners(pose)], dtype=np.int32)
    cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
