from __future__ import annotations

import torch
from torch import nn


PATCH_WIDTH = 96
PATCH_HEIGHT = 36
PATCH_CHANNELS = 2
LIKELIHOOD_HEADS = 2


class ConvBnAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
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
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class FastMarkerLikelihoodNet(nn.Module):
    """Tiny patch likelihood model for mobile candidate scoring.

    Input is a canonical marker patch generated from one pose hypothesis:
    channel 0 is locally normalized luma, channel 1 is normalized edge magnitude.
    Output 0 is a broad acquire likelihood. Output 1 is a sharper tracking
    likelihood over the same pose, so acquisition basin width and final pose
    precision do not fight in one scalar objective.
    """

    def __init__(self, patch_width: int = PATCH_WIDTH, patch_height: int = PATCH_HEIGHT):
        super().__init__()
        self.features = nn.Sequential(
            ConvBnAct(PATCH_CHANNELS, 12, stride=2),
            DepthwiseSeparable(12, 16, stride=2),
            DepthwiseSeparable(16, 24, stride=2),
            DepthwiseSeparable(24, 32, stride=2),
            DepthwiseSeparable(32, 40, stride=1),
        )
        # Preserve canonical spatial layout. Global average pooling made the
        # score nearly translation-invariant inside the patch, producing broad
        # optima and allowing LEDs/text to substitute for endpoint markers.
        grid_height = downsampled_size(patch_height, stages=4)
        grid_width = downsampled_size(patch_width, stages=4)
        self.head = nn.Sequential(
            nn.Identity(),
            nn.Conv2d(40, LIKELIHOOD_HEADS, kernel_size=(grid_height, grid_width), bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).flatten(1)


def downsampled_size(size: int, stages: int) -> int:
    for _ in range(stages):
        size = (size + 1) // 2
    return size


def create_model(patch_width: int = PATCH_WIDTH, patch_height: int = PATCH_HEIGHT) -> FastMarkerLikelihoodNet:
    return FastMarkerLikelihoodNet(patch_width, patch_height)
