from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from receiver_tools.cv_marker import OnnxPatchScorer
from receiver_tools.geometry import Pose, led_scores_from_pose, warp_marker_patch
from receiver_tools.tracker_backend import TrackerResult


@dataclass(frozen=True)
class Step:
    translation_px: float
    angle_rad: float
    log_scale: float


ACQUIRE_STEPS = (
    Step(32.0, math.radians(5.2), 0.120),
    Step(18.0, math.radians(3.0), 0.075),
    Step(10.0, math.radians(1.7), 0.045),
    Step(5.5, math.radians(0.9), 0.025),
    Step(3.0, math.radians(0.45), 0.012),
    Step(1.5, math.radians(0.24), 0.006),
)
TRACK_STEPS = (
    Step(9.0, math.radians(3.0), 0.035),
    Step(4.5, math.radians(1.5), 0.018),
    Step(2.2, math.radians(0.7), 0.008),
    Step(1.0, math.radians(0.3), 0.0035),
)
VERIFY_STEPS = (
    Step(3.0, math.radians(1.1), 0.014),
    Step(1.5, math.radians(0.55), 0.007),
    Step(0.7, math.radians(0.25), 0.003),
)


class NeuralPoseTrackerBackend:
    """Desktop mirror of Android neural acquire/verify/track loop."""

    def __init__(
        self,
        acquire_model: Path,
        precise_model: Path,
        *,
        coarse_threshold: float = 0.42,
        acquire_threshold: float = 0.72,
        track_threshold: float = 0.56,
    ):
        self.acquire = OnnxPatchScorer(acquire_model)
        self.precise = OnnxPatchScorer(precise_model)
        self.coarse_threshold = coarse_threshold
        self.acquire_threshold = acquire_threshold
        self.track_threshold = track_threshold
        self.reset()

    def reset(self) -> None:
        self.previous: Pose | None = None
        self.older: Pose | None = None
        self.previous_timestamp_ns = 0
        self.older_timestamp_ns = 0
        self.missed = 0

    def process(self, frame_bgr: np.ndarray, timestamp_ns: int) -> TrackerResult:
        tracking = self.previous is not None and self.missed < 4
        if tracking:
            seeds = [self.previous]
            predicted = self._predicted(timestamp_ns)
            if predicted is not None:
                seeds.append(predicted)
            pose, score = self._refine(frame_bgr, seeds, TRACK_STEPS, self.precise, sharp=True)
            mode = "tracking"
            accepted = score >= self.track_threshold
        else:
            pose, score = self._acquire(frame_bgr)
            mode = "acquire"
            accepted = score >= self.acquire_threshold

        if tracking and not accepted:
            pose, score = self._acquire(frame_bgr)
            mode = "reacquire"
            accepted = score >= self.acquire_threshold

        if not accepted:
            self.missed += 1
            if self.missed >= 4:
                self.reset()
            return TrackerResult(hit=False, mode=mode, confidence=score, debug=f"neural score={score:.3f}")

        if mode == "tracking":
            self.older = self.previous
            self.older_timestamp_ns = self.previous_timestamp_ns
        else:
            self.older = None
            self.older_timestamp_ns = 0
        self.previous = pose
        self.previous_timestamp_ns = timestamp_ns
        self.missed = 0
        return TrackerResult(
            hit=True,
            mode=mode,
            confidence=score,
            x=pose.center[0],
            y=pose.center[1],
            angle_rad=pose.angle_rad,
            scale_px=pose.scale_px,
            square=pose.square,
            triangle=pose.triangle,
            leds=pose.leds,
            led_scores=led_scores_from_pose(frame_bgr, pose),
            debug=f"neural score={score:.3f}",
        )

    def _acquire(self, frame_bgr: np.ndarray) -> tuple[Pose, float]:
        height, width = frame_bgr.shape[:2]
        guide_width = width * 0.58
        seed = Pose((width * 0.5, height * 0.5), 0.0, guide_width * 0.82)
        coarse_pose, coarse_score = self._refine(frame_bgr, [seed], ACQUIRE_STEPS, self.acquire, sharp=False)
        if coarse_score < self.coarse_threshold:
            return coarse_pose, coarse_score
        return self._refine(frame_bgr, [coarse_pose], VERIFY_STEPS, self.precise, sharp=True)

    def _refine(
        self,
        frame_bgr: np.ndarray,
        seeds: list[Pose],
        steps: tuple[Step, ...],
        scorer: OnnxPatchScorer,
        *,
        sharp: bool,
    ) -> tuple[Pose, float]:
        height, width = frame_bgr.shape[:2]
        guide_width = width * 0.58
        min_scale = max(32.0, guide_width * 0.30)
        max_scale = guide_width * 1.02
        seeds = [self._normalized(seed, min_scale, max_scale) for seed in seeds]
        scores = scorer.scores([warp_marker_patch(frame_bgr, pose) for pose in seeds], sharp=sharp)
        best_index = int(np.argmax(scores))
        pose = seeds[best_index]
        score = float(scores[best_index])

        for step in steps:
            for _ in range(2):
                variants = self._variants(pose, step, min_scale, max_scale)
                variant_scores = scorer.scores(
                    [warp_marker_patch(frame_bgr, variant) for variant in variants],
                    sharp=sharp,
                )
                index = int(np.argmax(variant_scores))
                candidate_score = float(variant_scores[index])
                if candidate_score <= score + 0.0015:
                    break
                pose = variants[index]
                score = candidate_score
        return pose, score

    @staticmethod
    def _variants(pose: Pose, step: Step, min_scale: float, max_scale: float) -> list[Pose]:
        x, y = pose.center
        variants = [
            Pose((x + step.translation_px, y), pose.angle_rad, pose.scale_px),
            Pose((x - step.translation_px, y), pose.angle_rad, pose.scale_px),
            Pose((x, y + step.translation_px), pose.angle_rad, pose.scale_px),
            Pose((x, y - step.translation_px), pose.angle_rad, pose.scale_px),
            Pose(pose.center, pose.angle_rad + step.angle_rad, pose.scale_px),
            Pose(pose.center, pose.angle_rad - step.angle_rad, pose.scale_px),
            Pose(pose.center, pose.angle_rad, pose.scale_px * math.exp(step.log_scale)),
            Pose(pose.center, pose.angle_rad, pose.scale_px * math.exp(-step.log_scale)),
        ]
        return [NeuralPoseTrackerBackend._normalized(item, min_scale, max_scale) for item in variants]

    @staticmethod
    def _normalized(pose: Pose, min_scale: float, max_scale: float) -> Pose:
        return Pose(pose.center, math.atan2(math.sin(pose.angle_rad), math.cos(pose.angle_rad)), float(np.clip(pose.scale_px, min_scale, max_scale)))

    def _predicted(self, timestamp_ns: int) -> Pose | None:
        if self.previous is None or self.older is None:
            return None
        history_dt = self.previous_timestamp_ns - self.older_timestamp_ns
        prediction_dt = timestamp_ns - self.previous_timestamp_ns
        if history_dt <= 0 or prediction_dt <= 0:
            return None
        ratio = float(np.clip(prediction_dt / history_dt, 0.0, 2.2))
        dx = (self.previous.center[0] - self.older.center[0]) * ratio
        dy = (self.previous.center[1] - self.older.center[1]) * ratio
        limit = self.previous.scale_px * 0.12
        length = math.hypot(dx, dy)
        if length > limit:
            dx *= limit / length
            dy *= limit / length
        angle_delta = math.atan2(
            math.sin(self.previous.angle_rad - self.older.angle_rad),
            math.cos(self.previous.angle_rad - self.older.angle_rad),
        )
        angle_delta = float(np.clip(angle_delta * ratio, -0.12, 0.12))
        log_delta = math.log(self.previous.scale_px / self.older.scale_px) * ratio
        log_delta = float(np.clip(log_delta, -0.08, 0.08))
        return Pose(
            (self.previous.center[0] + dx, self.previous.center[1] + dy),
            self.previous.angle_rad + angle_delta,
            self.previous.scale_px * math.exp(log_delta),
        )
