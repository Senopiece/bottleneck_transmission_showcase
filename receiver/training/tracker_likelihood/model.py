from __future__ import annotations

import torch
from torch import nn


PATCH_WIDTH = 96
PATCH_HEIGHT = 36
PATCH_CHANNELS = 2


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


class FastMarkerLikelihoodNet(nn.Module):
    """Tiny patch likelihood model for mobile candidate scoring.

    Input is a canonical marker patch generated from one pose hypothesis:
    channel 0 is locally normalized luma, channel 1 is normalized edge magnitude.
    Output is one logit: higher means "this pose aligns with the marker".
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBnAct(PATCH_CHANNELS, 16, stride=2),      # 48 x 18
            DepthwiseSeparable(16, 24, stride=2),        # 24 x 9
            DepthwiseSeparable(24, 32, stride=2),        # 12 x 5
            DepthwiseSeparable(32, 48, stride=2),        # 6 x 3
            DepthwiseSeparable(48, 64, stride=1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.05),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(1)


def create_model() -> FastMarkerLikelihoodNet:
    return FastMarkerLikelihoodNet()
