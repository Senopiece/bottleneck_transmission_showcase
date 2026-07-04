from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


LED_MM = 3.0
LED_GAP_MM = 2.5
MARKER_MM = 4.0
SQUARE_MM = LED_MM + 0.45
TRIANGLE_BOX_MM = LED_MM + 2.0
TRIANGLE_VISIBLE_FRACTION = 8.0 / 12.0
TRIANGLE_MM = TRIANGLE_BOX_MM * TRIANGLE_VISIBLE_FRACTION
MARKER_GAP_MM = 4.0
STEP_MM = LED_MM + LED_GAP_MM
MARKER_DISTANCE_MM = MARKER_MM + MARKER_GAP_MM * 2.0 + LED_MM + STEP_MM * 4.0
FIRST_LED_OFFSET_MM = MARKER_MM * 0.5 + MARKER_GAP_MM + LED_MM * 0.5

LED_FRACTIONS = np.array(
    [(FIRST_LED_OFFSET_MM + index * STEP_MM) / MARKER_DISTANCE_MM for index in range(5)],
    dtype=np.float32,
)
LED_RADIUS_FRACTION = (LED_MM * 0.5) / MARKER_DISTANCE_MM
SQUARE_SIZE_FRACTION = SQUARE_MM / MARKER_DISTANCE_MM
TRIANGLE_SIZE_FRACTION = TRIANGLE_MM / MARKER_DISTANCE_MM


@dataclass
class Pose:
    center: tuple[float, float]
    angle_rad: float
    scale_px: float

    @property
    def axis(self) -> tuple[float, float]:
        return math.cos(self.angle_rad), math.sin(self.angle_rad)

    @property
    def normal(self) -> tuple[float, float]:
        ax, ay = self.axis
        return -ay, ax

    @property
    def square(self) -> tuple[float, float]:
        ax, ay = self.axis
        return self.center[0] - ax * self.scale_px * 0.5, self.center[1] - ay * self.scale_px * 0.5

    @property
    def triangle(self) -> tuple[float, float]:
        ax, ay = self.axis
        return self.center[0] + ax * self.scale_px * 0.5, self.center[1] + ay * self.scale_px * 0.5

    @property
    def leds(self) -> list[tuple[float, float]]:
        sx, sy = self.square
        ax, ay = self.axis
        return [(sx + ax * self.scale_px * float(t), sy + ay * self.scale_px * float(t)) for t in LED_FRACTIONS]


def local_point(pose: Pose, along_fraction: float, normal_fraction: float = 0.0) -> tuple[float, float]:
    ax, ay = pose.axis
    nx, ny = pose.normal
    return (
        pose.center[0] + pose.scale_px * (along_fraction * ax + normal_fraction * nx),
        pose.center[1] + pose.scale_px * (along_fraction * ay + normal_fraction * ny),
    )


def square_corners(pose: Pose) -> list[tuple[float, float]]:
    half = SQUARE_SIZE_FRACTION * 0.5
    cx = -0.5
    return [
        local_point(pose, cx - half, -half),
        local_point(pose, cx + half, -half),
        local_point(pose, cx + half, half),
        local_point(pose, cx - half, half),
    ]


def triangle_corners(pose: Pose) -> list[tuple[float, float]]:
    half = TRIANGLE_SIZE_FRACTION * 0.5
    cx = 0.5
    return [
        local_point(pose, cx, -half),
        local_point(pose, cx + half, half),
        local_point(pose, cx - half, half),
    ]


def pose_from_markers(square: tuple[float, float], triangle: tuple[float, float]) -> Pose:
    dx = triangle[0] - square[0]
    dy = triangle[1] - square[1]
    scale = math.hypot(dx, dy)
    return Pose(
        center=((square[0] + triangle[0]) * 0.5, (square[1] + triangle[1]) * 0.5),
        angle_rad=math.atan2(dy, dx),
        scale_px=scale,
    )


def warp_marker_patch(frame_bgr: np.ndarray, pose: Pose, size: tuple[int, int] = (128, 48)) -> np.ndarray:
    width, height = size
    xs = (np.arange(width, dtype=np.float32) / max(1, width - 1) - 0.5) * 1.35
    ys = (np.arange(height, dtype=np.float32) / max(1, height - 1) - 0.5) * 0.46
    grid_x, grid_y = np.meshgrid(xs, ys)
    ax, ay = pose.axis
    nx, ny = pose.normal
    map_x = pose.center[0] + pose.scale_px * (grid_x * ax + grid_y * nx)
    map_y = pose.center[1] + pose.scale_px * (grid_x * ay + grid_y * ny)
    return cv2.remap(
        frame_bgr,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def warp_led_patch(frame_bgr: np.ndarray, pose: Pose, size: tuple[int, int] = (96, 24)) -> np.ndarray:
    width, height = size
    xs = (np.arange(width, dtype=np.float32) / max(1, width - 1) - 0.5) * 0.72
    ys = (np.arange(height, dtype=np.float32) / max(1, height - 1) - 0.5) * 0.22
    grid_x, grid_y = np.meshgrid(xs, ys)
    ax, ay = pose.axis
    nx, ny = pose.normal
    map_x = pose.center[0] + pose.scale_px * (grid_x * ax + grid_y * nx)
    map_y = pose.center[1] + pose.scale_px * (grid_x * ay + grid_y * ny)
    return cv2.remap(
        frame_bgr,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def patch_to_feature(patch_bgr: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    patch = cv2.resize(patch_bgr, out_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mean = float(gray.mean())
    std = float(gray.std())
    normalized = np.clip((gray - mean) / (std + 1e-4), -3.0, 3.0) / 3.0
    return normalized.reshape(-1).astype(np.float32)


def led_scores_from_pose(frame_bgr: np.ndarray, pose: Pose) -> list[float]:
    scores: list[float] = []
    h, w = frame_bgr.shape[:2]
    for x, y in pose.leds:
        xi = int(round(x))
        yi = int(round(y))
        r = max(2, int(round(pose.scale_px * LED_RADIUS_FRACTION * 0.48)))
        x0, x1 = max(0, xi - r), min(w, xi + r + 1)
        y0, y1 = max(0, yi - r), min(h, yi + r + 1)
        if x0 >= x1 or y0 >= y1:
            scores.append(0.0)
            continue
        roi = frame_bgr[y0:y1, x0:x1].astype(np.float32)
        b = roi[:, :, 0].mean()
        g = roi[:, :, 1].mean()
        red = roi[:, :, 2].mean()
        gray = (b + g + red) / 3.0
        blueish = (b - 0.5 * g - 0.5 * red) / 255.0
        bright = gray / 255.0
        scores.append(float(np.clip(0.65 * bright + 0.55 * blueish, 0.0, 1.5)))
    return scores
