from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from receiver_tools.cv_marker import CvTrackerBackend
from receiver_tools.dataset import default_dataset_dir, default_derived_dir, discover_videos, repo_root_from_tools
from receiver_tools.geometry import LED_FRACTIONS, Pose, square_corners, triangle_corners


CROP_SIZE = 28
MARKER_PATCH_SIZE = (160, 64)


class OnnxLedScorer:
    def __init__(self, model_path: Path):
        self.net = cv2.dnn.readNetFromONNX(str(model_path))

    def logits(self, crops_bgr: list[np.ndarray], detector_likelihood: float = 1.0) -> np.ndarray:
        if not crops_bgr:
            return np.zeros(0, dtype=np.float32)
        tensors = [crop_to_tensor(crop) for crop in crops_bgr]
        blob = np.stack(tensors, axis=0).astype(np.float32)
        likelihood = np.full((len(crops_bgr),), float(np.clip(detector_likelihood, 0.0, 1.0)), dtype=np.float32)
        self.net.setInput(blob, "led_crop")
        self.net.setInput(likelihood, "detector_likelihood")
        return np.reshape(self.net.forward(), -1).astype(np.float32)


def crop_to_tensor(crop_bgr: np.ndarray) -> np.ndarray:
    crop = cv2.resize(crop_bgr, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
    bgr = crop.astype(np.float32) / 255.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    luma = np.clip((gray - float(gray.mean())) / (float(gray.std()) + 1e-4), -3.0, 3.0) / 3.0
    blue_excess = np.clip(bgr[:, :, 0] - 0.5 * bgr[:, :, 1] - 0.5 * bgr[:, :, 2], -1.0, 1.0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    edge = np.clip(edge / (float(np.percentile(edge, 95)) + 1e-4), 0.0, 1.0)
    return np.stack([luma, blue_excess, edge], axis=0)


def warp_marker_patch(frame_bgr: np.ndarray, pose: Pose, size: tuple[int, int] = MARKER_PATCH_SIZE) -> np.ndarray:
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


def crop_led_from_marker_patch(patch_bgr: np.ndarray, bit_index: int) -> np.ndarray:
    height, width = patch_bgr.shape[:2]
    x_norm = (float(LED_FRACTIONS[bit_index]) - 0.5) / 1.35 + 0.5
    cx = x_norm * (width - 1)
    cy = 0.5 * (height - 1)
    side = int(round(np.clip(min(width / 11.0, height / 2.2), 18, 38)))
    x0 = int(round(cx - side * 0.5))
    y0 = int(round(cy - side * 0.5))
    padded = cv2.copyMakeBorder(patch_bgr, side, side, side, side, cv2.BORDER_REPLICATE)
    return padded[y0 + side : y0 + side + side, x0 + side : x0 + side + side]


def pose_from_result(result) -> Pose:
    return Pose(center=(float(result.x), float(result.y)), angle_rad=float(result.angle_rad), scale_px=float(result.scale_px))


def draw_pose(frame: np.ndarray, pose: Pose, logits: np.ndarray, drop_count: int) -> None:
    for corners in (square_corners(pose), triangle_corners(pose)):
        pts = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)
    for index, ((x, y), logit) in enumerate(zip(pose.leds, logits)):
        prob = 1.0 / (1.0 + math.exp(-float(np.clip(logit, -40.0, 40.0))))
        radius = max(6, int(round(pose.scale_px * 0.030)))
        color = (0, int(150 + prob * 105), 255)
        cv2.circle(frame, (int(round(x)), int(round(y))), radius, color, 2, cv2.LINE_AA)
        label = f"{index}:{prob:.2f}"
        cv2.putText(frame, label, (int(round(x)) - radius, int(round(y)) - radius - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"drops={drop_count}", (24, frame.shape[0] - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)


def render_video(video_path: Path, out_path: Path, tracker: CvTrackerBackend, scorer: OnnxLedScorer, max_frames: int | None, stride: int) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps / max(1, stride)), (width, height))
    tracker.reset()
    drop_count = 0
    was_hit = False
    written = 0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    progress = tqdm(total=total if max_frames is None else min(total, max_frames), desc=video_path.stem, leave=False)
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if max_frames is not None and frame_index >= max_frames:
            break
        progress.update(1)
        if frame_index % stride != 0:
            frame_index += 1
            continue
        timestamp_ns = int(frame_index / max(1e-6, fps) * 1_000_000_000)
        result = tracker.process(frame, timestamp_ns)
        if was_hit and not result.hit:
            drop_count += 1
        was_hit = bool(result.hit)
        if result.hit:
            pose = pose_from_result(result)
            patch = warp_marker_patch(frame, pose)
            crops = [crop_led_from_marker_patch(patch, index) for index in range(5)]
            logits = scorer.logits(crops, result.confidence)
            draw_pose(frame, pose, logits, drop_count)
        else:
            cv2.putText(frame, f"MISS drops={drop_count}", (24, frame.shape[0] - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 90, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        written += 1
        frame_index += 1
    progress.close()
    writer.release()
    capture.release()
    if written == 0:
        raise RuntimeError(f"No frames rendered for {video_path}")


def parse_args() -> argparse.Namespace:
    root = repo_root_from_tools()
    parser = argparse.ArgumentParser(description="Render LED scorer predictions on tracked videos.")
    parser.add_argument("--dataset", type=Path, default=default_dataset_dir())
    parser.add_argument("--model", type=Path, default=root / "receiver" / "models" / "led_reader" / "led_reader_crop_v003_gate.onnx")
    parser.add_argument("--tracker-model", type=Path, default=root / "receiver" / "models" / "tracker_likelihood" / "tracker_likelihood_fast_v003.onnx")
    parser.add_argument("--out", type=Path, default=default_derived_dir() / "overlays" / "led_reader_v003_gate")
    parser.add_argument("--include", nargs="*", default=None)
    parser.add_argument("--kind", choices=("good", "bad", "all"), default="good")
    parser.add_argument("--max-frames", type=int, default=260)
    parser.add_argument("--stride", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include = set(args.include) if args.include else None
    videos = [item for item in discover_videos(args.dataset) if args.kind in ("all", item.kind)]
    if include:
        videos = [item for item in videos if item.name in include]
    if not videos:
        raise SystemExit("No videos selected.")
    tracker = CvTrackerBackend(tracker_model=args.tracker_model)
    scorer = OnnxLedScorer(args.model)
    for item in videos:
        out_path = args.out / item.kind / f"{item.name}_led.mp4"
        render_video(item.path, out_path, tracker, scorer, args.max_frames, max(1, args.stride))
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
