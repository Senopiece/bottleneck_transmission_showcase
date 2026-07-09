from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from receiver_tools.cv_marker import CvMarkerDetector, CvTrackerBackend
from receiver_tools.dataset import VideoItem, default_dataset_dir, discover_videos
from receiver_tools.geometry import Pose, warp_marker_patch
from receiver_tools.geometry import led_scores_from_pose
from receiver_tools.synthetic import render_marker_patch

from tracker_likelihood.model import PATCH_HEIGHT, PATCH_WIDTH


@dataclass(frozen=True)
class TrackerSample:
    patch_bgr: np.ndarray
    target: float
    weight: float = 1.0


def patch_to_tensor(patch_bgr: np.ndarray) -> torch.Tensor:
    patch = cv2.resize(patch_bgr, (PATCH_WIDTH, PATCH_HEIGHT), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mean = float(gray.mean())
    std = float(gray.std())
    luma = np.clip((gray - mean) / (std + 1e-4), -3.0, 3.0) / 3.0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    edge = edge / (float(np.percentile(edge, 95)) + 1e-4)
    edge = np.clip(edge, 0.0, 1.0)

    stacked = np.stack([luma, edge], axis=0).astype(np.float32)
    return torch.from_numpy(stacked)


def augment_patch(patch_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    patch = patch_bgr.astype(np.float32)
    contrast = float(rng.uniform(0.70, 1.35))
    brightness = float(rng.uniform(-24.0, 24.0))
    patch = (patch - 127.5) * contrast + 127.5 + brightness
    if rng.random() < 0.45:
        gamma = float(rng.uniform(0.75, 1.45))
        patch = 255.0 * np.power(np.clip(patch / 255.0, 0.0, 1.0), gamma)
    if rng.random() < 0.55:
        sigma = float(rng.uniform(0.0, 1.25))
        if sigma > 0.05:
            patch = cv2.GaussianBlur(patch, (0, 0), sigmaX=sigma)
    if rng.random() < 0.65:
        patch += rng.normal(0.0, float(rng.uniform(0.0, 7.0)), size=patch.shape)
    return np.clip(patch, 0, 255).astype(np.uint8)


class TrackerLikelihoodDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: list[TrackerSample], *, augment: bool, seed: int):
        self.samples = samples
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        patch = sample.patch_bgr
        if self.augment:
            patch = augment_patch(patch, self.rng)
        x = patch_to_tensor(patch)
        y = torch.tensor(sample.target, dtype=torch.float32)
        w = torch.tensor(sample.weight, dtype=torch.float32)
        return x, y, w


def synthetic_samples(count: int, seed: int) -> list[TrackerSample]:
    rng = np.random.default_rng(seed)
    samples: list[TrackerSample] = []
    for _ in range(count):
        bits = rng.integers(0, 2, size=5).astype(np.float32)
        jitter = (
            float(rng.normal(0, 0.045)),
            float(rng.normal(0, 0.050)),
            float(rng.normal(0, 0.13)),
            float(rng.normal(0, 0.10)),
        )
        patch = render_marker_patch(bits, size=(128, 48), rng=rng, pose_jitter=jitter)
        pose_error = (
            (jitter[0] / 0.075) ** 2
            + (jitter[1] / 0.075) ** 2
            + (jitter[2] / 0.17) ** 2
            + (jitter[3] / 0.13) ** 2
        )
        target = float(np.exp(-0.5 * pose_error))
        samples.append(TrackerSample(patch_bgr=patch, target=target, weight=0.75))
    return samples


def random_bad_crops(frame_bgr: np.ndarray, rng: np.random.Generator, count: int) -> list[TrackerSample]:
    h, w = frame_bgr.shape[:2]
    samples: list[TrackerSample] = []
    for _ in range(count):
        crop_w = int(rng.integers(max(48, w // 12), max(64, w // 3)))
        crop_h = max(24, int(crop_w * rng.uniform(0.30, 0.55)))
        x0 = int(rng.integers(0, max(1, w - crop_w)))
        y0 = int(rng.integers(0, max(1, h - crop_h)))
        crop = frame_bgr[y0 : y0 + crop_h, x0 : x0 + crop_w]
        samples.append(TrackerSample(patch_bgr=crop, target=0.0, weight=1.0))
    return samples


def video_mined_samples(
    dataset_dir: Path = default_dataset_dir(),
    *,
    max_frames_per_video: int = 420,
    stride: int = 8,
    hard_negative_min_score: float = 0.40,
    include: set[str] | None = None,
    seed: int = 2,
) -> list[TrackerSample]:
    rng = np.random.default_rng(seed)
    tracker_model = latest_tracker_model()
    detector = CvMarkerDetector(tracker_model)
    samples: list[TrackerSample] = []
    for item in discover_videos(dataset_dir):
        if include is not None and item.name not in include:
            continue
        samples.extend(
            samples_from_video(
                item,
                detector,
                rng,
                max_frames_per_video,
                stride,
                hard_negative_min_score,
                tracker_model,
            )
        )
    return samples


def latest_tracker_model() -> Path | None:
    model_dir = Path(__file__).resolve().parents[2] / "models" / "tracker_likelihood"
    models = sorted(model_dir.glob("tracker_likelihood_fast_v*.onnx"))
    return models[-1] if models else None


def samples_from_video(
    item: VideoItem,
    detector: CvMarkerDetector,
    rng: np.random.Generator,
    max_frames: int,
    stride: int,
    hard_negative_min_score: float,
    tracker_model: Path | None,
) -> list[TrackerSample]:
    cap = cv2.VideoCapture(str(item.path))
    samples: list[TrackerSample] = []
    frame_index = 0
    accepted = 0
    soft_positive = 0
    local_samples = 0
    hard_negative = 0
    tracker = CvTrackerBackend(tracker_model=tracker_model) if item.kind == "good" else None
    while cap.isOpened() and frame_index < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % stride == 0:
            if item.kind == "good":
                assert tracker is not None
                timestamp_ns = int(frame_index / 30.0 * 1_000_000_000)
                result = tracker.process(frame, timestamp_ns)
                pose = pose_from_tracker_result(result)
                if result.hit and pose is not None:
                    patch = warp_marker_patch(frame, pose)
                    samples.append(TrackerSample(patch_bgr=patch, target=1.0, weight=1.35))
                    accepted += 1
                    nearby = perturb_pose_samples(frame, pose, rng, strong=True)
                    samples.extend(nearby)
                    local_samples += len(nearby)
                elif result.mode == "track_predict" and pose is not None and float(np.mean(led_scores_from_pose(frame, pose))) >= 0.18:
                    patch = warp_marker_patch(frame, pose)
                    samples.append(TrackerSample(patch_bgr=patch, target=0.72, weight=0.85))
                    soft_positive += 1
                    nearby = perturb_pose_samples(frame, pose, rng, strong=False)
                    samples.extend(nearby)
                    local_samples += len(nearby)
                samples.extend(random_bad_crops(frame, rng, 1))
            else:
                candidate = detector.detect(frame)
                if candidate is not None and candidate.score >= hard_negative_min_score:
                    patch = warp_marker_patch(frame, candidate.pose)
                    samples.append(TrackerSample(patch_bgr=patch, target=0.0, weight=2.0))
                    hard_negative += 1
                samples.extend(random_bad_crops(frame, rng, 3))
        frame_index += 1
    cap.release()
    print(
        f"{item.kind} {item.name}: pseudo_positive={accepted} soft_positive={soft_positive} "
        f"local={local_samples} hard_negative={hard_negative}"
    )
    return samples


def pose_from_tracker_result(result) -> Pose | None:
    if result.scale_px <= 1.0:
        return None
    return Pose(center=(float(result.x), float(result.y)), angle_rad=float(result.angle_rad), scale_px=float(result.scale_px))


def perturb_pose_samples(frame_bgr: np.ndarray, pose: Pose, rng: np.random.Generator, *, strong: bool) -> list[TrackerSample]:
    samples: list[TrackerSample] = []
    count = 5 if strong else 3
    sigma_along = 0.030 if strong else 0.045
    sigma_normal = 0.035 if strong else 0.050
    sigma_angle = 0.070 if strong else 0.095
    sigma_scale = 0.045 if strong else 0.060
    for _ in range(count):
        along = float(rng.normal(0.0, sigma_along))
        normal = float(rng.normal(0.0, sigma_normal))
        angle = float(rng.normal(0.0, sigma_angle))
        log_scale = float(rng.normal(0.0, sigma_scale))
        ax = np.cos(pose.angle_rad)
        ay = np.sin(pose.angle_rad)
        nx = -ay
        ny = ax
        variant = Pose(
            center=(
                pose.center[0] + pose.scale_px * (along * ax + normal * nx),
                pose.center[1] + pose.scale_px * (along * ay + normal * ny),
            ),
            angle_rad=pose.angle_rad + angle,
            scale_px=pose.scale_px * float(np.exp(log_scale)),
        )
        patch = warp_marker_patch(frame_bgr, variant)
        pose_error = (
            (along / 0.060) ** 2
            + (normal / 0.060) ** 2
            + (angle / 0.130) ** 2
            + (log_scale / 0.100) ** 2
        )
        target = float(np.exp(-0.5 * pose_error))
        samples.append(TrackerSample(patch_bgr=patch, target=target, weight=1.10 if strong else 0.80))
    return samples


def split_samples(samples: list[TrackerSample], validation_fraction: float, seed: int) -> tuple[list[TrackerSample], list[TrackerSample]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(samples))
    val_count = max(1, int(round(len(samples) * validation_fraction)))
    val_ids = set(int(i) for i in order[:val_count])
    train = [sample for index, sample in enumerate(samples) if index not in val_ids]
    val = [sample for index, sample in enumerate(samples) if index in val_ids]
    return train, val
