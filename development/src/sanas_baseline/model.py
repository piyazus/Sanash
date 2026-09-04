"""The baseline models.

Two of them, and the trivial one matters as much as the trained one.

`DensityCNN` is a ResNet-18 trunk truncated at stride 8 plus a two-layer
density head. ResNet-18 is chosen because it ships with torchvision, is small
enough to train on a CPU, and is a reference point rather than a proposal:
`weights=None` means no ImageNet checkpoint is downloaded, so the run carries
no third party weight licence and no network dependency.

`ConstantPredictor` always answers the train split mean count. It is the floor.
Any candidate architecture, CSRNet + PFCASA included, has to beat it on the
same split before its numbers mean anything.
"""

import torch
import torch.nn as nn
import torchvision


class DensityCNN(nn.Module):
    """Randomly initialised ResNet-18 trunk with a density head at stride 8."""

    stride = 8

    def __init__(self) -> None:
        super().__init__()
        backbone = torchvision.models.resnet18(weights=None)
        self.trunk = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
        )
        final = nn.Conv2d(64, 1, kernel_size=1)
        # A small positive bias keeps the output off the flat side of the final
        # ReLU at initialisation, so the density head still has gradient.
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, 0.01)
        self.head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            final,
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, H/8, W/8) non-negative density."""
        return self.head(self.trunk(x)).squeeze(1)


def counts_from_density(density: torch.Tensor, density_scale: float) -> torch.Tensor:
    """Sum a predicted density map back to a count in people."""
    return density.sum(dim=(1, 2)) / density_scale


class ConstantPredictor:
    """Predicts the train split mean count for every image.

    The lower bound every future candidate must clear. A model that cannot beat
    this has learned nothing about the image.
    """

    def __init__(self, value: float = 0.0) -> None:
        self.value = float(value)

    def fit(self, train_counts) -> "ConstantPredictor":
        self.value = float(sum(train_counts) / len(train_counts))
        return self

    def predict(self, n: int):
        return [self.value] * int(n)
