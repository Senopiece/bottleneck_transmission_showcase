from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from receiver_tools.cv_marker import CvMarkerDetector, CvTrackerBackend, OnnxPatchScorer, tracker_patch_blob
from receiver_tools.dataset import VideoItem, default_dataset_dir, discover_videos
from receiver_tools.geometry import Pose, warp_marker_patch
from receiver_tools.geometry import led_scores_from_pose
from receiver_tools.synthetic import render_marker_patch

from tracker_likelihood.model import PATCH_HEIGHT, PATCH_WIDTH


FINE_TARGET_POWER = 4.0
FINE_BROAD_WEIGHT = 0.22
ACTIVE_PATCH_WIDTH = PATCH_WIDTH
ACTIVE_PATCH_HEIGHT = PATCH_HEIGHT


@dataclass(frozen=True)
class TrackerSample:
    patch_bgr: np.ndarray
    target: float
    weight: float = 1.0
    group: str = "synthetic"


def patch_to_tensor(patch_bgr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(tracker_patch_blob(patch_bgr, ACTIVE_PATCH_WIDTH, ACTIVE_PATCH_HEIGHT)[0])


def configure_patch_size(width: int, height: int) -> None:
    global ACTIVE_PATCH_WIDTH, ACTIVE_PATCH_HEIGHT
    ACTIVE_PATCH_WIDTH = width
    ACTIVE_PATCH_HEIGHT = height


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
    def __init__(
        self,
        samples: list[TrackerSample],
        *,
        augment: bool,
        seed: int,
        broad_target_power: float = 1.0,
    ):
        self.samples = samples
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.broad_target_power = broad_target_power

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        patch = sample.patch_bgr
        if self.augment:
            patch = augment_patch(patch, self.rng)
        x = patch_to_tensor(patch)
        base_target = float(np.clip(sample.target, 0.0, 1.0))
        broad_target = base_target**self.broad_target_power
        fine_target = FINE_BROAD_WEIGHT * base_target + (1.0 - FINE_BROAD_WEIGHT) * base_target**FINE_TARGET_POWER
        y = torch.tensor((broad_target, fine_target), dtype=torch.float32)
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
        samples.append(TrackerSample(patch_bgr=patch, target=target, weight=0.75, group=f"synthetic-{len(samples) % 24}"))
    return samples


def random_bad_crops(
    frame_bgr: np.ndarray,
    rng: np.random.Generator,
    count: int,
    *,
    group: str,
) -> list[TrackerSample]:
    h, w = frame_bgr.shape[:2]
    samples: list[TrackerSample] = []
    for _ in range(count):
        crop_w = int(rng.integers(max(48, w // 12), max(64, w // 3)))
        crop_h = max(24, int(crop_w * rng.uniform(0.30, 0.55)))
        x0 = int(rng.integers(0, max(1, w - crop_w)))
        y0 = int(rng.integers(0, max(1, h - crop_h)))
        crop = frame_bgr[y0 : y0 + crop_h, x0 : x0 + crop_w]
        samples.append(TrackerSample(patch_bgr=crop, target=0.0, weight=1.0, group=group))
    return samples


def hard_pose_negative_samples(
    frame_bgr: np.ndarray,
    rng: np.random.Generator,
    scorer: OnnxPatchScorer | None,
    *,
    count: int,
    keep: int,
    group: str,
) -> list[TrackerSample]:
    """Mine marker-shaped negatives instead of unrelated rectangular crops."""
    h, w = frame_bgr.shape[:2]
    min_side = float(min(h, w))
    candidates: list[tuple[float, np.ndarray]] = []
    seeds: list[Pose] = []
    for index in range(count):
        if index == 0:
            center = (w * 0.5, h * 0.5)
            angle = 0.0
            scale = min_side * 0.48
        else:
            center = (float(rng.uniform(0.12, 0.88) * w), float(rng.uniform(0.12, 0.88) * h))
            angle = float(rng.uniform(-np.pi, np.pi))
            scale = float(rng.uniform(0.18, 0.72) * min_side)
        seeds.append(Pose(center=center, angle_rad=angle, scale_px=scale))

    # Follow the same failure mechanism as the mobile coordinate ascent: a
    # mediocre seed climbs toward a text/icon-shaped local maximum.
    poses: list[Pose] = []
    for seed in seeds:
        pose = seed
        if scorer is not None:
            score = scorer.score(warp_marker_patch(frame_bgr, pose))
            for translation, angle_step, scale_step in (
                (0.060, 0.095, 0.070),
                (0.032, 0.052, 0.038),
                (0.017, 0.028, 0.020),
                (0.009, 0.014, 0.010),
            ):
                ax, ay = pose.axis
                nx, ny = pose.normal
                d = pose.scale_px
                variants = (
                    Pose((pose.center[0] + d * translation * ax, pose.center[1] + d * translation * ay), pose.angle_rad, d),
                    Pose((pose.center[0] - d * translation * ax, pose.center[1] - d * translation * ay), pose.angle_rad, d),
                    Pose((pose.center[0] + d * translation * nx, pose.center[1] + d * translation * ny), pose.angle_rad, d),
                    Pose((pose.center[0] - d * translation * nx, pose.center[1] - d * translation * ny), pose.angle_rad, d),
                    Pose(pose.center, pose.angle_rad + angle_step, d),
                    Pose(pose.center, pose.angle_rad - angle_step, d),
                    Pose(pose.center, pose.angle_rad, d * float(np.exp(scale_step))),
                    Pose(pose.center, pose.angle_rad, d * float(np.exp(-scale_step))),
                )
                variant_patches = [warp_marker_patch(frame_bgr, variant) for variant in variants]
                variant_scores = scorer.scores(variant_patches)
                for variant, variant_score in zip(variants, variant_scores, strict=True):
                    if variant_score > score:
                        pose = variant
                        score = float(variant_score)
        poses.append(pose)

    patches = [warp_marker_patch(frame_bgr, pose) for pose in poses]
    scores = scorer.scores(patches) if scorer is not None else np.zeros(len(patches), dtype=np.float32)
    candidates.extend((float(score), patch) for score, patch in zip(scores, patches, strict=True))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [TrackerSample(patch_bgr=patch, target=0.0, weight=3.2, group=group) for _, patch in candidates[:keep]]


def decoy_pose_samples(
    frame_bgr: np.ndarray,
    pose: Pose,
    rng: np.random.Generator,
    scorer: OnnxPatchScorer | None,
    *,
    keep: int = 2,
    group: str,
) -> list[TrackerSample]:
    """Mine wrong poses on real patterns, including LED-as-marker alignments."""
    ax, ay = pose.axis
    nx, ny = pose.normal
    variants: list[Pose] = []
    for along in (-0.30, -0.18, 0.18, 0.30):
        variants.append(
            Pose(
                center=(pose.center[0] + pose.scale_px * along * ax, pose.center[1] + pose.scale_px * along * ay),
                angle_rad=pose.angle_rad,
                scale_px=pose.scale_px * float(rng.uniform(0.72, 1.20)),
            )
        )
    variants.extend(
        [
            Pose(pose.center, pose.angle_rad + np.pi, pose.scale_px),
            Pose(
                center=(pose.center[0] + pose.scale_px * 0.16 * nx, pose.center[1] + pose.scale_px * 0.16 * ny),
                angle_rad=pose.angle_rad + 0.20,
                scale_px=pose.scale_px * 0.82,
            ),
            Pose(
                center=(pose.center[0] - pose.scale_px * 0.16 * nx, pose.center[1] - pose.scale_px * 0.16 * ny),
                angle_rad=pose.angle_rad - 0.20,
                scale_px=pose.scale_px * 1.18,
            ),
        ]
    )
    patches = [warp_marker_patch(frame_bgr, variant) for variant in variants]
    scores = scorer.scores(patches, sharp=True) if scorer is not None else np.zeros(len(patches), dtype=np.float32)
    candidates = [(float(score), patch) for score, patch in zip(scores, patches, strict=True)]
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [TrackerSample(patch_bgr=patch, target=0.0, weight=3.4, group=group) for _, patch in candidates[:keep]]


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
    pose_labels = load_pose_labels(item.name) if item.kind == "good" else {}
    tracker = CvTrackerBackend(tracker_model=tracker_model) if item.kind == "good" and not pose_labels else None
    negative_scorer = detector.onnx_model
    while cap.isOpened() and frame_index < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % stride == 0:
            frame_group = f"{item.name}:{frame_index // 60}"
            if item.kind == "good":
                result = None
                pose = pose_labels.get(frame_index)
                if pose is None and tracker is not None:
                    timestamp_ns = int(frame_index / 30.0 * 1_000_000_000)
                    result = tracker.process(frame, timestamp_ns)
                    pose = pose_from_tracker_result(result)
                if pose is not None and (result is None or result.hit):
                    patch = warp_marker_patch(frame, pose)
                    samples.append(TrackerSample(patch_bgr=patch, target=1.0, weight=1.35, group=frame_group))
                    accepted += 1
                    nearby = perturb_pose_samples(frame, pose, rng, strong=True, group=frame_group)
                    samples.extend(nearby)
                    local_samples += len(nearby)
                    decoys = decoy_pose_samples(frame, pose, rng, negative_scorer, group=frame_group)
                    samples.extend(decoys)
                    hard_negative += len(decoys)
                elif result is not None and result.mode == "track_predict" and pose is not None and float(np.mean(led_scores_from_pose(frame, pose))) >= 0.18:
                    patch = warp_marker_patch(frame, pose)
                    samples.append(TrackerSample(patch_bgr=patch, target=0.72, weight=0.85, group=frame_group))
                    soft_positive += 1
                    nearby = perturb_pose_samples(frame, pose, rng, strong=False, group=frame_group)
                    samples.extend(nearby)
                    local_samples += len(nearby)
                samples.extend(random_bad_crops(frame, rng, 1, group=frame_group))
            else:
                candidate = detector.detect(frame)
                if candidate is not None and candidate.score >= hard_negative_min_score:
                    patch = warp_marker_patch(frame, candidate.pose)
                    samples.append(TrackerSample(patch_bgr=patch, target=0.0, weight=2.0, group=frame_group))
                    hard_negative += 1
                pose_negatives = hard_pose_negative_samples(
                    frame,
                    rng,
                    negative_scorer,
                    count=5,
                    keep=3,
                    group=frame_group,
                )
                samples.extend(pose_negatives)
                hard_negative += len(pose_negatives)
                samples.extend(random_bad_crops(frame, rng, 1, group=frame_group))
        frame_index += 1
    cap.release()
    print(
        f"{item.kind} {item.name}: pseudo_positive={accepted} soft_positive={soft_positive} "
        f"local={local_samples} hard_negative={hard_negative}"
    )
    return samples


def load_pose_labels(video_name: str) -> dict[int, Pose]:
    datasets_dir = Path(__file__).resolve().parents[2] / "datasets"
    candidates = (
        datasets_dir / "derived" / "metrics" / "anchor_localization_refine3" / "good" / f"{video_name}.csv",
        datasets_dir / "derived" / "metrics" / "anchored_acquire_all" / "good" / f"{video_name}.csv",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return {}
    labels: dict[int, Pose] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("hit") != "1":
                continue
            scale = float(row["scale_px"])
            if scale <= 1.0:
                continue
            labels[int(row["frame"])] = Pose(
                center=(float(row["x"]), float(row["y"])),
                angle_rad=float(row["angle_rad"]),
                scale_px=scale,
            )
    return labels


def pose_from_tracker_result(result) -> Pose | None:
    if result.scale_px <= 1.0:
        return None
    return Pose(center=(float(result.x), float(result.y)), angle_rad=float(result.angle_rad), scale_px=float(result.scale_px))


def perturb_pose_samples(
    frame_bgr: np.ndarray,
    pose: Pose,
    rng: np.random.Generator,
    *,
    strong: bool,
    group: str = "synthetic",
) -> list[TrackerSample]:
    samples: list[TrackerSample] = []
    count = 8 if strong else 5
    sigma_along = 0.030 if strong else 0.045
    sigma_normal = 0.035 if strong else 0.050
    sigma_angle = 0.070 if strong else 0.095
    sigma_scale = 0.045 if strong else 0.060
    for index in range(count):
        if index < (count + 1) // 2:
            spread = 1.0
        elif index < count - 2:
            spread = 2.3
        else:
            spread = 4.5
        along = float(rng.normal(0.0, sigma_along * spread))
        normal = float(rng.normal(0.0, sigma_normal * spread))
        angle = float(rng.normal(0.0, sigma_angle * spread))
        log_scale = float(rng.normal(0.0, sigma_scale * spread))
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
        samples.append(
            TrackerSample(
                patch_bgr=patch,
                target=target,
                weight=1.10 if strong else 0.80,
                group=group,
            )
        )
    return samples


def split_samples(samples: list[TrackerSample], validation_fraction: float, seed: int) -> tuple[list[TrackerSample], list[TrackerSample]]:
    rng = np.random.default_rng(seed)
    groups = sorted({sample.group for sample in samples})
    order = rng.permutation(len(groups))
    val_count = max(1, int(round(len(groups) * validation_fraction)))
    val_groups = {groups[int(i)] for i in order[:val_count]}
    train = [sample for sample in samples if sample.group not in val_groups]
    val = [sample for sample in samples if sample.group in val_groups]
    return train, val
