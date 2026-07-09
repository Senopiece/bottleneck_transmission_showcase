from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class TrackerResult:
    hit: bool
    mode: str = "acquire"
    confidence: float = 0.0
    x: float = 0.0
    y: float = 0.0
    angle_rad: float = 0.0
    scale_px: float = 0.0
    square: tuple[float, float] | None = None
    triangle: tuple[float, float] | None = None
    leds: list[tuple[float, float]] = field(default_factory=list)
    led_scores: list[float] = field(default_factory=list)
    debug: str = ""


class TrackerBackend(Protocol):
    def reset(self) -> None:
        ...

    def process(self, frame_bgr: np.ndarray, timestamp_ns: int) -> TrackerResult:
        ...


class StubTrackerBackend:
    """No-op backend for checking dataset runner plumbing."""

    def reset(self) -> None:
        return None

    def process(self, frame_bgr: np.ndarray, timestamp_ns: int) -> TrackerResult:
        height, width = frame_bgr.shape[:2]
        return TrackerResult(
            hit=False,
            mode="acquire",
            x=width * 0.5,
            y=height * 0.5,
            debug="stub_backend",
        )
