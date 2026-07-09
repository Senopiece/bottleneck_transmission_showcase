from __future__ import annotations

import torch
from torch import nn


CROP_WIDTH = 28
CROP_HEIGHT = 28
CROP_CHANNELS = 3
DEFAULT_DETECTOR_LIKELIHOOD = 1.0


class ConvBnAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class LedCropNet(nn.Module):
    """Shared one-LED crop scorer.

    Input: canonical LED crop, channels are normalized luma, blue excess, edge.
    Output: one logit. Positive means LED ON, negative means LED OFF.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBnAct(CROP_CHANNELS, 12, stride=1),
            DepthwiseSeparable(12, 18, stride=2),
            DepthwiseSeparable(18, 24, stride=2),
            DepthwiseSeparable(24, 32, stride=2),
            DepthwiseSeparable(32, 40, stride=1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.04),
            nn.Linear(40, 1),
        )
        # Learnable evidence gate. The crop classifier still decides ON/OFF,
        # but weak marker likelihood suppresses overconfident LED evidence.
        self.gate_slope = nn.Parameter(torch.tensor(2.0))
        self.gate_center = nn.Parameter(torch.tensor(0.56))

    def forward(self, x: torch.Tensor, detector_likelihood: torch.Tensor | None = None) -> torch.Tensor:
        raw = self.raw_logits(x)
        gate = self.evidence_gate(raw, detector_likelihood)
        return raw * gate

    def raw_logits(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = features.mean(dim=(2, 3))
        pooled = self.head[2](pooled)
        return self.head[3](pooled).squeeze(1)

    def evidence_gate(self, raw: torch.Tensor, detector_likelihood: torch.Tensor | None = None) -> torch.Tensor:
        if detector_likelihood is None:
            detector_likelihood = torch.full_like(raw, DEFAULT_DETECTOR_LIKELIHOOD)
        likelihood = detector_likelihood.reshape_as(raw)
        slope = self.gate_slope
        center = self.gate_center
        gate = torch.sigmoid(slope * (likelihood - center))
        return gate


def create_model() -> LedCropNet:
    return LedCropNet()
