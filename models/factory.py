"""
Model factory for the Sanash crowd analysis project.

Unifies three families of models under a simple, consistent interface:

- Ultralytics YOLO11 (detection-style head localization)
- P2PNet-style point prediction network
- CSRNet-style density map regression network (dilated convolutions)

Design goals
------------
- Provide a clean abstraction so the training engine can switch models
  without changing its core logic.
- Use modern, actively maintained APIs (Ultralytics YOLO11, PyTorch 2.x).
- Keep implementations lightweight but faithful enough for research use.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from torchvision import models as tv_models


class ModelKind(str, Enum):
    YOLO11 = "yolo11"
    P2PNET = "p2pnet"
    CSRNET = "csrnet"


class TaskType(str, Enum):
    DETECTION = "detection"
    DENSITY = "density"
    POINTS = "points"


@dataclass
class ModelConfig:
    """Configuration shared across different model types."""

    kind: ModelKind
    # Path to a checkpoint (for YOLO) or backbone weights.
    weights: Optional[str] = None
    # Number of output channels for density/point maps. For crowd counting this
    # is typically 1.
    out_channels: int = 1
    # Number of classes for detection; for head detection this is usually 1.
    num_classes: int = 1


class BaseCrowdModel(nn.Module):
    """
    Base class for internally trained models (P2PNet, CSRNet).

    All subclasses must implement:
    - task_type: TaskType
    - forward(images: Tensor) -> Tensor (task-specific output)
    """

    task_type: TaskType

    def __init__(self) -> None:
        super().__init__()


class CSRNet(BaseCrowdModel):
    """
    CSRNet implementation for density map regression.

    Architecture follows the original paper:
    "CSRNet: Dilated Convolutional Neural Networks for Understanding
    the Highly Congested Scenes" (CVPR 2018).

    Front-end: first 10 layers of VGG16.
    Back-end: dilated convolutions with increasing receptive field.
    """

    def __init__(self, out_channels: int = 1, pretrained_backbone: bool = True) -> None:
        super().__init__()
        self.task_type = TaskType.DENSITY

        vgg16 = tv_models.vgg16_bn(weights=tv_models.VGG16_BN_Weights.DEFAULT if pretrained_backbone else None)
        features = list(vgg16.features.children())

        # Front-end: conv1_1 to conv4_3 (up to, but not including, last maxpool).
        self.frontend = nn.Sequential(*features[:33])

        # Back-end: dilated convs
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1),
        )

        self._initialize_backend()

    def _initialize_backend(self) -> None:
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (N, 3, H, W)

        Returns
        -------
        density : Tensor of shape (N, 1, H/8, W/8)
        """
        x = self.frontend(x)
        x = self.backend(x)
        return F.relu(x)  # density should be non-negative


class P2PNet(BaseCrowdModel):
    """
    Lightweight P2PNet-style network for point-based crowd counting.

    The original P2PNet uses a VGG16 backbone and a two-branch head to predict
    point proposals and their confidences. Here we implement a simplified but
    compatible variant:

    - Backbone: ResNet-18 features.
    - FPN-style upsampling to a stride-8 feature map.
    - Output: per-pixel confidence map indicating head locations.

    Training is typically done against a pseudo-binary target map derived from
    head annotations (points), using a combination of focal / BCE losses.
    """

    def __init__(self, out_channels: int = 1, pretrained_backbone: bool = True) -> None:
        super().__init__()
        self.task_type = TaskType.POINTS

        backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT if pretrained_backbone else None)

        # Extract layers as in a basic FPN setup.
        self.layer1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Lateral and upsample layers to produce stride-8 feature maps.
        self.lat4 = nn.Conv2d(512, 256, kernel_size=1)
        self.lat3 = nn.Conv2d(256, 256, kernel_size=1)
        self.lat2 = nn.Conv2d(128, 256, kernel_size=1)

        self.head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1),
        )

    def _upsample_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Upsample x to y's spatial size and add."""
        return F.interpolate(x, size=y.shape[-2:], mode="bilinear", align_corners=False) + y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (N, 3, H, W)

        Returns
        -------
        points_map : Tensor of shape (N, 1, H/8, W/8)
            Per-pixel confidence scores; threshold/peak extraction can
            be used to obtain final point locations.
        """
        c1 = self.layer1(x)  # stride 4
        c2 = self.layer2(c1)  # stride 8
        c3 = self.layer3(c2)  # stride 16
        c4 = self.layer4(c3)  # stride 32

        p4 = self.lat4(c4)
        p3 = self._upsample_add(p4, self.lat3(c3))
        p2 = self._upsample_add(p3, self.lat2(c2))  # stride 8 features

        out = self.head(p2)
        return out


class YOLO11Wrapper:
    """
    Lightweight wrapper around Ultralytics YOLO11 to integrate with the engine.

    Instead of rewriting Ultralytics' internal training loops, this class
    exposes a small, explicit subset of the official API that the engine can
    call in a unified way.
    """

    task_type: TaskType = TaskType.DETECTION

    def __init__(self, weights: str, num_classes: int = 1) -> None:
        """
        Parameters
        ----------
        weights : str
            Path to YOLO11 checkpoint (e.g., 'yolo11n.pt' or a fine-tuned .pt).
        num_classes : int
            Number of detection classes (heads = 1).
        """
        self.model = YOLO(weights)
        # Optionally adjust number of classes if starting from a base model.
        # This is supported in YOLO11 via overloading model.model[-1].nc, but
        # we keep it simple and assume matching checkpoints by default.
        self.num_classes = num_classes

    def train(self, **kwargs: Any) -> Any:
        """
        Proxy to `YOLO.train()`. Typical arguments:

        - data: path to data.yaml
        - epochs, imgsz, batch, lr0, device, project, name, etc.
        """
        return self.model.train(**kwargs)

    def validate(self, **kwargs: Any) -> Any:
        """Proxy to `YOLO.val()`."""
        return self.model.val(**kwargs)

    def predict(self, **kwargs: Any) -> Any:
        """Proxy to `YOLO.predict()` for inference."""
        return self.model.predict(**kwargs)


def create_model(config: ModelConfig) -> Any:
    """
    Factory entry-point for model construction.

    Returns
    -------
    model : BaseCrowdModel | YOLO11Wrapper
        The caller (training engine) can branch on type or `kind` to decide
        how to run training for each model.
    """
    kind = config.kind

    if kind is ModelKind.CSRNET:
        return CSRNet(out_channels=config.out_channels)

    if kind is ModelKind.P2PNET:
        return P2PNet(out_channels=config.out_channels)

    if kind is ModelKind.YOLO11:
        if not config.weights:
            raise ValueError("YOLO11 requires `weights` to be specified in ModelConfig.")
        return YOLO11Wrapper(weights=config.weights, num_classes=config.num_classes)

    raise ValueError(f"Unsupported model kind: {kind}")


def model_from_name(
    name: str,
    weights: Optional[str] = None,
    num_classes: int = 1,
    out_channels: int = 1,
) -> Any:
    """
    Convenience helper that parses a string name into a concrete model.

    Examples
    --------
    - 'yolo11'  -> YOLO11Wrapper
    - 'p2pnet'  -> P2PNet
    - 'csrnet'  -> CSRNet
    """
    name_lower = name.lower()
    if name_lower in {"yolo11", "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"}:
        cfg = ModelConfig(kind=ModelKind.YOLO11, weights=weights, num_classes=num_classes)
    elif name_lower in {"p2pnet", "p2p"}:
        cfg = ModelConfig(kind=ModelKind.P2PNET, out_channels=out_channels)
    elif name_lower in {"csrnet", "csr"}:
        cfg = ModelConfig(kind=ModelKind.CSRNET, out_channels=out_channels)
    else:
        raise ValueError(f"Unrecognized model name: {name}")

    return create_model(cfg)


__all__ = [
    "ModelKind",
    "TaskType",
    "ModelConfig",
    "BaseCrowdModel",
    "CSRNet",
    "P2PNet",
    "YOLO11Wrapper",
    "create_model",
    "model_from_name",
]

