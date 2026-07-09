from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from receiver_tools.cv_marker import CvTrackerBackend
from receiver_tools.dataset import VideoItem, default_dataset_dir, discover_videos, repo_root_from_tools
from receiver_tools.geometry import LED_FRACTIONS, Pose, warp_marker_patch
from receiver_tools.synthetic import render_marker_patch

from led_reader.model import CROP_HEIGHT, CROP_WIDTH


PREAMBLE = ("00000", "01010", "10101", "11111")
MESSAGE_WIDTH = 8
MESSAGE_HEIGHT = 8
MESSAGE_BITS = MESSAGE_WIDTH * MESSAGE_HEIGHT
PARITY_BITS = 32
CODEWORD_BITS = MESSAGE_BITS + PARITY_BITS
PACKET_BITS = 5
LDGM_SEED = 0x12345678
LDPC_SEED = 0xB0771E
MIX_GOLDEN_RATIO = 0x9E3779B9
MIX_MURMUR_1 = 0x85EBCA6B
MIX_MURMUR_2 = 0xC2B2AE35
UINT_MASK = 0xFFFFFFFF
PARITY_CHECK_DEGREE = 8


@dataclass(frozen=True)
class LedSample:
    crop_bgr: np.ndarray
    target: float
    weight: float = 1.0
    detector_likelihood: float = 1.0


@dataclass(frozen=True)
class FrameObservation:
    frame_bgr: np.ndarray
    timestamp_s: float
    pose: Pose
    led_scores: np.ndarray
    confidence: float


def crop_to_tensor(crop_bgr: np.ndarray) -> torch.Tensor:
    crop = cv2.resize(crop_bgr, (CROP_WIDTH, CROP_HEIGHT), interpolation=cv2.INTER_AREA)
    bgr = crop.astype(np.float32) / 255.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mean = float(gray.mean())
    std = float(gray.std())
    luma = np.clip((gray - mean) / (std + 1e-4), -3.0, 3.0) / 3.0
    blue = np.clip(bgr[:, :, 0] - 0.5 * bgr[:, :, 1] - 0.5 * bgr[:, :, 2], -1.0, 1.0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    edge = edge / (float(np.percentile(edge, 95)) + 1e-4)
    edge = np.clip(edge, 0.0, 1.0)
    return torch.from_numpy(np.stack([luma, blue, edge], axis=0).astype(np.float32))


def augment_crop(crop_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    crop = crop_bgr.astype(np.float32)
    contrast = float(rng.uniform(0.72, 1.35))
    brightness = float(rng.uniform(-22.0, 22.0))
    crop = (crop - 127.5) * contrast + 127.5 + brightness
    if rng.random() < 0.40:
        gamma = float(rng.uniform(0.78, 1.38))
        crop = 255.0 * np.power(np.clip(crop / 255.0, 0.0, 1.0), gamma)
    if rng.random() < 0.45:
        sigma = float(rng.uniform(0.0, 1.05))
        if sigma > 0.05:
            crop = cv2.GaussianBlur(crop, (0, 0), sigmaX=sigma)
    if rng.random() < 0.55:
        crop += rng.normal(0.0, float(rng.uniform(0.0, 6.0)), size=crop.shape)
    return np.clip(crop, 0, 255).astype(np.uint8)


class LedReaderDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: list[LedSample], *, augment: bool, seed: int):
        self.samples = samples
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        crop = sample.crop_bgr
        if self.augment:
            crop = augment_crop(crop, self.rng)
        return (
            crop_to_tensor(crop),
            torch.tensor(sample.detector_likelihood, dtype=torch.float32),
            torch.tensor(sample.target, dtype=torch.float32),
            torch.tensor(sample.weight, dtype=torch.float32),
        )


def message_bits_from_json(path: Path | None = None) -> str:
    if path is None:
        path = repo_root_from_tools() / "receiver" / "datasets" / "raw" / "expected" / "default_creeper_8x8.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]
    return "".join(rows).replace(" ", "").strip()


def expected_packet_stream(message_bits: str, count: int) -> list[str]:
    codeword = encode_precode(message_bits)
    packets: list[str] = []
    for packet_index in range(count):
        if packet_index < len(PREAMBLE):
            packets.append(PREAMBLE[packet_index])
            continue
        payload_index = packet_index - len(PREAMBLE)
        bits = []
        for bit_index in range(PACKET_BITS):
            measurement = 0
            for variable in measurement_neighbors(payload_index * PACKET_BITS + bit_index):
                measurement ^= codeword[variable]
            bits.append(str(measurement))
        packets.append("".join(bits))
    return packets


def encode_precode(message_bits: str) -> list[int]:
    payload = message_bits[:MESSAGE_BITS].ljust(MESSAGE_BITS, "0")
    parity = []
    for check in range(PARITY_BITS):
        bit = 0
        for payload_index in parity_group(check):
            bit ^= 1 if payload[payload_index] == "1" else 0
        parity.append(str(bit))
    return [1 if c == "1" else 0 for c in (payload + "".join(parity)).ljust(CODEWORD_BITS, "0")]


class SeededRng:
    def __init__(self, seed: int):
        self.state = seed & UINT_MASK

    def next(self) -> int:
        self.state = (self.state * 1103515245 + 12345) & UINT_MASK
        return self.state

    def next_neighbors(self, max_index: int) -> list[int]:
        r = self.next()
        if r < 858993459:
            degree = 1
        elif r < 2147483648:
            degree = 2
        elif r < 3221225472:
            degree = 3
        elif r < 3865470566:
            degree = 4
        elif r < 4166118277:
            degree = 5
        else:
            degree = 6
        return self.next_fixed_neighbors(max_index, degree)

    def next_fixed_neighbors(self, max_index: int, degree: int) -> list[int]:
        neighbors: list[int] = []
        while len(neighbors) < degree:
            candidate = self.next() % max_index
            if candidate not in neighbors:
                neighbors.append(candidate)
        return neighbors


def measurement_neighbors(measurement_index: int) -> list[int]:
    return SeededRng(mix_seed(LDGM_SEED, measurement_index)).next_neighbors(CODEWORD_BITS)


def parity_group(check_index: int) -> list[int]:
    return SeededRng(mix_seed(LDPC_SEED, check_index)).next_fixed_neighbors(MESSAGE_BITS, PARITY_CHECK_DEGREE)


def mix_seed(seed: int, index: int) -> int:
    x = (seed + (index & UINT_MASK) * MIX_GOLDEN_RATIO) & UINT_MASK
    x = ((x ^ (x >> 16)) * MIX_MURMUR_1) & UINT_MASK
    x = ((x ^ (x >> 13)) * MIX_MURMUR_2) & UINT_MASK
    return (x ^ (x >> 16)) & UINT_MASK


def led_video_samples(
    dataset_dir: Path = default_dataset_dir(),
    *,
    include: set[str] | None = None,
    max_frames_per_video: int = 360,
    stride: int = 2,
    rates: tuple[int, ...] = (8,),
    seed: int = 4,
) -> list[LedSample]:
    rng = np.random.default_rng(seed)
    samples: list[LedSample] = []
    message = message_bits_from_json()
    packets = expected_packet_stream(message, 512)
    for item in discover_videos(dataset_dir):
        if item.kind != "good":
            continue
        if include is not None and item.name not in include:
            continue
        observations = collect_observations(item, max_frames=max_frames_per_video, stride=stride)
        if len(observations) < 24:
            print(f"{item.name}: skip observations={len(observations)}")
            continue
        start_s, rate, align_score = align_stream_start(observations, packets, rates)
        video_samples = samples_from_observations(observations, packets, start_s, rate, rng)
        samples.extend(video_samples)
        print(
            f"{item.name}: obs={len(observations)} rate={rate} start={start_s:.3f}s "
            f"align={align_score:.2f} samples={len(video_samples)}"
        )
    return samples


def bad_video_negative_samples(
    dataset_dir: Path = default_dataset_dir(),
    *,
    max_frames_per_video: int = 180,
    stride: int = 5,
    max_samples_per_video: int = 160,
    seed: int = 9,
) -> list[LedSample]:
    rng = np.random.default_rng(seed)
    samples: list[LedSample] = []
    for item in discover_videos(dataset_dir):
        if item.kind != "bad":
            continue
        observations = collect_observations(item, max_frames=max_frames_per_video, stride=stride)
        rng.shuffle(observations)
        video_samples: list[LedSample] = []
        for obs in observations:
            patch = warp_marker_patch(obs.frame_bgr, obs.pose, size=(160, 64))
            for bit_index in range(PACKET_BITS):
                crop = crop_led_from_marker_patch(patch, bit_index, rng)
                video_samples.append(
                    LedSample(
                        crop_bgr=crop,
                        target=0.0,
                        weight=0.35,
                        detector_likelihood=float(np.clip(obs.confidence, 0.0, 1.0)),
                    )
                )
                if len(video_samples) >= max_samples_per_video:
                    break
            if len(video_samples) >= max_samples_per_video:
                break
        samples.extend(video_samples)
        print(f"{item.name}: bad negatives obs={len(observations)} samples={len(video_samples)}")
    return samples


def synthetic_led_samples(count: int, seed: int) -> list[LedSample]:
    rng = np.random.default_rng(seed)
    samples: list[LedSample] = []
    for _ in range(count):
        bits = rng.integers(0, 2, size=PACKET_BITS).astype(np.float32)
        jitter = (
            float(rng.normal(0.0, 0.020)),
            float(rng.normal(0.0, 0.026)),
            float(rng.normal(0.0, 0.055)),
            float(rng.normal(0.0, 0.045)),
        )
        patch = render_marker_patch(bits, size=(160, 64), rng=rng, pose_jitter=jitter)
        for bit_index, bit in enumerate(bits):
            crop = crop_led_from_marker_patch(patch, bit_index, rng)
            samples.append(LedSample(crop_bgr=crop, target=float(bit), weight=0.55, detector_likelihood=1.0))
    return samples


def collect_observations(item: VideoItem, *, max_frames: int, stride: int) -> list[FrameObservation]:
    cap = cv2.VideoCapture(str(item.path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    tracker = CvTrackerBackend()
    observations: list[FrameObservation] = []
    frame_index = 0
    while cap.isOpened() and frame_index < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % stride == 0:
            timestamp_s = frame_index / fps
            result = tracker.process(frame, int(timestamp_s * 1_000_000_000))
            if result.hit and result.confidence >= 0.70 and result.scale_px > 20.0:
                pose = Pose((float(result.x), float(result.y)), float(result.angle_rad), float(result.scale_px))
                observations.append(
                    FrameObservation(
                        frame_bgr=frame.copy(),
                        timestamp_s=timestamp_s,
                        pose=pose,
                        led_scores=np.array(result.led_scores, dtype=np.float32),
                        confidence=float(result.confidence),
                    )
                )
        frame_index += 1
    cap.release()
    return observations


def align_stream_start(
    observations: list[FrameObservation],
    packets: list[str],
    rates: tuple[int, ...],
) -> tuple[float, int, float]:
    ts = np.array([obs.timestamp_s for obs in observations], dtype=np.float64)
    scores = np.stack([obs.led_scores for obs in observations]).astype(np.float32)
    center = float(np.median(scores))
    spread = float(np.percentile(scores, 85) - np.percentile(scores, 15))
    spread = max(spread, 0.08)
    norm = np.clip((scores - center) / spread, -3.0, 3.0)
    best: tuple[float, int, float] | None = None
    first = float(ts.min())
    last = float(ts.max())
    for rate in rates:
        period = 1.0 / rate
        start_min = first - 0.75
        step = min(period / 12.0, 0.010)
        start_max = max(start_min + step, last - len(PREAMBLE) * period * 0.70)
        count = max(1, int(math.ceil((start_max - start_min) / step)))
        for i in range(count):
            start = start_min + i * step
            score = alignment_score(ts, norm, packets, start, rate)
            if best is None or score > best[2]:
                best = (start, rate, score)
    assert best is not None
    return best


def alignment_score(ts: np.ndarray, norm_scores: np.ndarray, packets: list[str], start_s: float, rate: int) -> float:
    period = 1.0 / rate
    relative = (ts - start_s) / period
    packet_indices = np.floor(relative).astype(np.int32)
    phase = relative - packet_indices
    mask = (packet_indices >= 0) & (packet_indices < min(len(packets), 96)) & (phase >= 0.25) & (phase <= 0.75)
    if int(mask.sum()) < 8:
        return -1e9
    score = 0.0
    used = 0
    for row_index in np.nonzero(mask)[0]:
        packet = packets[int(packet_indices[row_index])]
        for bit_index, bit in enumerate(packet):
            sign = 1.0 if bit == "1" else -1.0
            score += sign * float(norm_scores[row_index, bit_index])
            used += 1
    return score / max(1, used)


def samples_from_observations(
    observations: list[FrameObservation],
    packets: list[str],
    start_s: float,
    rate: int,
    rng: np.random.Generator,
) -> list[LedSample]:
    samples: list[LedSample] = []
    period = 1.0 / rate
    for obs in observations:
        relative = (obs.timestamp_s - start_s) / period
        packet_index = int(math.floor(relative))
        if packet_index < 0 or packet_index >= len(packets):
            continue
        phase = relative - packet_index
        if phase < 0.30 or phase > 0.70:
            continue
        phase_weight = 1.0 - abs(phase - 0.5) / 0.20
        packet = packets[packet_index]
        patch = warp_marker_patch(obs.frame_bgr, obs.pose, size=(160, 64))
        for bit_index, bit in enumerate(packet):
            crop = crop_led_from_marker_patch(patch, bit_index, rng)
            samples.append(
                LedSample(
                    crop_bgr=crop,
                    target=1.0 if bit == "1" else 0.0,
                    weight=float(0.75 + 0.45 * max(0.0, phase_weight)),
                    detector_likelihood=float(np.clip(obs.confidence, 0.0, 1.0)),
                )
            )
    return samples


def crop_led_from_marker_patch(patch_bgr: np.ndarray, bit_index: int, rng: np.random.Generator | None = None) -> np.ndarray:
    height, width = patch_bgr.shape[:2]
    x_norm = (float(LED_FRACTIONS[bit_index]) - 0.5) / 1.35 + 0.5
    y_norm = 0.5
    jitter_x = 0.0
    jitter_y = 0.0
    if rng is not None:
        jitter_x = float(rng.normal(0.0, 0.012))
        jitter_y = float(rng.normal(0.0, 0.016))
    cx = (x_norm + jitter_x) * (width - 1)
    cy = (y_norm + jitter_y) * (height - 1)
    side = int(round(min(width, height) * 0.46))
    side = max(18, min(38, side))
    return crop_square_replicate(patch_bgr, cx, cy, side)


def crop_square_replicate(frame: np.ndarray, cx: float, cy: float, side: int) -> np.ndarray:
    half = side // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + side
    y1 = y0 + side
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - frame.shape[1])
    pad_bottom = max(0, y1 - frame.shape[0])
    if pad_left or pad_top or pad_right or pad_bottom:
        padded = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top
        return padded[y0:y1, x0:x1]
    return frame[y0:y1, x0:x1]


def split_samples(samples: list[LedSample], validation_fraction: float, seed: int) -> tuple[list[LedSample], list[LedSample]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(samples))
    val_count = max(1, int(round(len(samples) * validation_fraction)))
    val_ids = set(int(i) for i in order[:val_count])
    train = [sample for index, sample in enumerate(samples) if index not in val_ids]
    val = [sample for index, sample in enumerate(samples) if index in val_ids]
    return train, val
