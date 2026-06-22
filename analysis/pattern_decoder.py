from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class DecodeResult:
    frame: str
    score: float
    square: float
    triangle: float
    bits: str | None


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def blue_score(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip((b - 0.45 * r - 0.35 * g + (y - 45) * 0.08) / 145, 0, 1.5)


def marker_score(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = np.abs(r - g) + np.abs(g - b) + np.abs(b - r)
    neutral = np.clip(1 - chroma / 220, 0, 1)
    return np.clip((y - 80) / 160, 0, 1) * (0.35 + 0.65 * neutral)


def sample_bilinear(img: np.ndarray, x: float, y: float) -> float:
    h, w = img.shape[:2]
    if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
        return 0.0
    x0 = int(x)
    y0 = int(y)
    fx = x - x0
    fy = y - y0
    return float(
        img[y0, x0] * (1 - fx) * (1 - fy)
        + img[y0, x0 + 1] * fx * (1 - fy)
        + img[y0 + 1, x0] * (1 - fx) * fy
        + img[y0 + 1, x0 + 1] * fx * fy
    )


def local(cx: float, cy: float, ux: float, uy: float, vx: float, vy: float, lx: float, ly: float, scale: float):
    return cx + (ux * lx + vx * ly) * scale, cy + (uy * lx + vy * ly) * scale


SQUARE_IN = [(0, 0), (-0.32, -0.32), (0.32, -0.32), (-0.32, 0.32), (0.32, 0.32)]
SQUARE_OUT = [(-0.72, 0), (0.72, 0), (0, -0.72), (0, 0.72), (-0.62, -0.62), (0.62, 0.62)]
TRI_IN = [(0, -0.34), (0, -0.08), (0, 0.18), (-0.22, 0.32), (0.22, 0.32)]
TRI_EMPTY = [(-0.42, -0.16), (0.42, -0.16), (-0.58, 0.08), (0.58, 0.08), (0, -0.70)]
SLOT_FRACTIONS = np.array([0.242, 0.401, 0.559, 0.718, 0.877], dtype=np.float32)


def mean_template(score_img: np.ndarray, cx: float, cy: float, ux: float, uy: float, vx: float, vy: float, pts, scale: float) -> float:
    vals = [sample_bilinear(score_img, *local(cx, cy, ux, uy, vx, vy, px, py, scale)) for px, py in pts]
    return float(np.mean(vals)) if vals else 0.0


def evaluate_pose(rgb: np.ndarray, cx: float, cy: float, angle: float, distance: float) -> tuple[float, float, float, str]:
    h, w = rgb.shape[:2]
    ux = float(np.cos(angle))
    uy = float(np.sin(angle))
    vx = -uy
    vy = ux
    marker = distance / 9.875
    start = (cx - ux * distance * 0.5, cy - uy * distance * 0.5)
    end = (cx + ux * distance * 0.5, cy + uy * distance * 0.5)
    margin = marker * 0.7
    if min(start[0], end[0]) < margin or max(start[0], end[0]) >= w - margin:
        return -1e6, 0, 0, ""
    if min(start[1], end[1]) < margin or max(start[1], end[1]) >= h - margin:
        return -1e6, 0, 0, ""

    marker_img = marker_score(rgb)
    led_img = blue_score(rgb)
    sq_fill = mean_template(marker_img, start[0], start[1], ux, uy, vx, vy, SQUARE_IN, marker)
    sq_out = mean_template(marker_img, start[0], start[1], ux, uy, vx, vy, SQUARE_OUT, marker)
    tri_fill = mean_template(marker_img, end[0], end[1], ux, uy, vx, vy, TRI_IN, marker * 1.25)
    tri_empty = mean_template(marker_img, end[0], end[1], ux, uy, vx, vy, TRI_EMPTY, marker * 1.25)
    square = np.clip(sq_fill * 0.85 + max(0, sq_fill - sq_out) * 0.55, 0, 1)
    triangle = np.clip(tri_fill * 0.95 + max(0, tri_fill - tri_empty) * 0.75, 0, 1)
    score = square * 3.0 + triangle * 4.6

    bits = []
    for fraction in SLOT_FRACTIONS:
        x = start[0] + ux * distance * float(fraction)
        y = start[1] + uy * distance * float(fraction)
        on = sample_bilinear(led_img, x, y) > 0.48
        bits.append("1" if on else "0")
    return float(score), float(square), float(triangle), "".join(bits)


def refine(rgb: np.ndarray, seed: tuple[float, float, float, float]) -> tuple[float, float, float, float, float, float, str]:
    cx, cy, angle, distance = seed
    best = evaluate_pose(rgb, cx, cy, angle, distance)
    best_pose = (cx, cy, angle, distance)
    for dp, da, ds in [(18, 0.10, 0.08), (10, 0.055, 0.045), (5, 0.030, 0.022), (2.5, 0.015, 0.010)]:
        improved = True
        while improved:
            improved = False
            candidates = [
                (cx + dp, cy, angle, distance),
                (cx - dp, cy, angle, distance),
                (cx, cy + dp, angle, distance),
                (cx, cy - dp, angle, distance),
                (cx, cy, angle + da, distance),
                (cx, cy, angle - da, distance),
                (cx, cy, angle, distance * (1 + ds)),
                (cx, cy, angle, distance * (1 - ds)),
            ]
            for cand in candidates:
                ev = evaluate_pose(rgb, *cand)
                if ev[0] > best[0]:
                    best = ev
                    best_pose = cand
                    cx, cy, angle, distance = cand
                    improved = True
    return (*best_pose, *best)


def decode_frame(path: Path) -> DecodeResult:
    rgb = load_rgb(path)
    h, w = rgb.shape[:2]
    seed = (w * 0.5, h * 0.5, 0.0, w * 0.58 * 0.82)
    cx, cy, angle, distance, score, square, triangle, bits = refine(rgb, seed)
    if score < 4.75 or square < 0.72 or triangle < 0.52:
        bits = None
    return DecodeResult(path.name, score, square, triangle, bits)


def decode_frame_dir(input_dir: Path) -> list[DecodeResult]:
    paths = sorted([*input_dir.glob("*.jpg"), *input_dir.glob("*.png")])
    return [decode_frame(path) for path in paths]


def draw_overlay(input_path: Path, output_path: Path) -> None:
    rgb = load_rgb(input_path)
    h, w = rgb.shape[:2]
    cx, cy, angle, distance, score, square, triangle, bits = refine(rgb, (w * 0.5, h * 0.5, 0.0, w * 0.58 * 0.82))
    ux, uy = float(np.cos(angle)), float(np.sin(angle))
    vx, vy = -uy, ux
    start = (cx - ux * distance * 0.5, cy - uy * distance * 0.5)
    end = (cx + ux * distance * 0.5, cy + uy * distance * 0.5)
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for fraction in SLOT_FRACTIONS:
        x = start[0] + ux * distance * float(fraction)
        y = start[1] + uy * distance * float(fraction)
        r = distance / 9.875 * 0.48
        draw.ellipse((x - r, y - r, x + r, y + r), outline="yellow", width=3)
    draw.rectangle((start[0] - 10, start[1] - 10, start[0] + 10, start[1] + 10), outline="white", width=3)
    draw.polygon([(end[0], end[1] - 12), (end[0] + 12, end[1] + 12), (end[0] - 12, end[1] + 12)], outline="white")
    draw.text((12, 12), f"score={score:.2f} sq={square:.2f} tri={triangle:.2f} bits={bits}", fill="yellow")
    img.save(output_path)


if __name__ == "__main__":
    import argparse
    import csv

    parser = argparse.ArgumentParser()
    parser.add_argument("frames", type=Path)
    parser.add_argument("--csv", type=Path, default=Path("analysis/data/pattern_decode.csv"))
    args = parser.parse_args()

    rows = decode_frame_dir(args.frames)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "score", "square", "triangle", "bits"])
        for row in rows:
            writer.writerow([row.frame, row.score, row.square, row.triangle, row.bits or "null"])
