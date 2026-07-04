from __future__ import annotations

import math

import cv2
import numpy as np

from receiver_tools.geometry import LED_FRACTIONS, LED_RADIUS_FRACTION, SQUARE_SIZE_FRACTION, TRIANGLE_SIZE_FRACTION


def render_marker_patch(
    bits: np.ndarray,
    size: tuple[int, int] = (128, 48),
    rng: np.random.Generator | None = None,
    pose_jitter: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    width, height = size
    bg = rng.uniform(8, 45)
    patch = np.full((height, width, 3), bg, dtype=np.float32)
    tint = np.array([rng.uniform(0.8, 1.25), rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2)], dtype=np.float32)
    patch *= tint

    dx, dy, da, ds = pose_jitter
    scale = math.exp(ds)
    center = np.array([width * (0.5 + dx), height * (0.5 + dy)], dtype=np.float32)
    marker_w = width * 0.78 * scale
    angle = da
    axis = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)

    def pt(local_x: float, local_y: float) -> tuple[int, int]:
        p = center + axis * (local_x * marker_w) + normal * (local_y * marker_w)
        return int(round(p[0])), int(round(p[1]))

    marker_color = rng.uniform(190, 255)
    led_off = np.array([rng.uniform(45, 95), rng.uniform(30, 70), rng.uniform(12, 40)], dtype=np.float32)
    led_on = np.array([rng.uniform(210, 255), rng.uniform(210, 255), rng.uniform(210, 255)], dtype=np.float32)

    square_half = SQUARE_SIZE_FRACTION * 0.5
    square = np.array(
        [
            pt(-0.5 - square_half, -square_half),
            pt(-0.5 + square_half, -square_half),
            pt(-0.5 + square_half, square_half),
            pt(-0.5 - square_half, square_half),
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(patch, square, (marker_color,) * 3)

    tri_half = TRIANGLE_SIZE_FRACTION * 0.5
    tri = np.array(
        [
            pt(0.5, -tri_half),
            pt(0.5 + tri_half, tri_half),
            pt(0.5 - tri_half, tri_half),
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(patch, tri, (marker_color,) * 3)

    radius = max(2, int(round(marker_w * LED_RADIUS_FRACTION)))
    for index, fraction in enumerate(LED_FRACTIONS):
        c = pt(float(fraction) - 0.5, 0.0)
        color = led_on if bits[index] > 0.5 else led_off
        cv2.circle(patch, c, radius, tuple(float(v) for v in color), -1, cv2.LINE_AA)
        if bits[index] > 0.5:
            cv2.GaussianBlur(patch, (0, 0), sigmaX=1.2, dst=patch)
            cv2.circle(patch, c, radius, tuple(float(v) for v in color), -1, cv2.LINE_AA)

    if rng.random() < 0.8:
        sigma = rng.uniform(0.0, 1.1)
        if sigma > 0.05:
            patch = cv2.GaussianBlur(patch, (0, 0), sigmaX=sigma)
    noise = rng.normal(0.0, rng.uniform(0.0, 5.0), size=patch.shape)
    patch = np.clip(patch + noise, 0, 255).astype(np.uint8)
    return patch
