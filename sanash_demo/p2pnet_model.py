from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _make_vgg_layers(batch_norm: bool = True) -> nn.Sequential:
    """Build the VGG16 feature stack used by the original P2PNet checkpoint."""
    cfg = [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ]
    layers: list[nn.Module] = []
    in_channels = 3
    for item in cfg:
        if item == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        out_channels = int(item)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        if batch_norm:
            layers.extend([conv, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)])
        else:
            layers.extend([conv, nn.ReLU(inplace=True)])
        in_channels = out_channels
    return nn.Sequential(*layers)


class Vgg16BnBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        features = list(_make_vgg_layers(batch_norm=True).children())
        self.body1 = nn.Sequential(*features[:13])
        self.body2 = nn.Sequential(*features[13:23])
        self.body3 = nn.Sequential(*features[23:33])
        self.body4 = nn.Sequential(*features[33:43])

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        x = image
        for layer in (self.body1, self.body2, self.body3, self.body4):
            x = layer(x)
            outputs.append(x)
        return outputs


class RegressionHead(nn.Module):
    def __init__(self, in_channels: int, num_anchor_points: int, feature_size: int = 256) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The released P2PNet code defines conv3/conv4 but does not use them in
        # the forward pass. Keeping this behavior is required for checkpoint parity.
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.output(x)
        x = x.permute(0, 2, 3, 1)
        return x.contiguous().view(x.shape[0], -1, 2)


class ClassificationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_anchor_points: int,
        num_classes: int = 2,
        feature_size: int = 256,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points
        self.conv1 = nn.Conv2d(in_channels, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.output(x)
        x = x.permute(0, 2, 3, 1)
        batch_size, width, height, _ = x.shape
        x = x.view(batch_size, width, height, self.num_anchor_points, self.num_classes)
        return x.contiguous().view(batch_size, -1, self.num_classes)


def _generate_anchor_points(stride: int, row: int, line: int) -> np.ndarray:
    row_step = stride / row
    line_step = stride / line
    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    return np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()


def _shift_grid(shape: np.ndarray, stride: int, anchor_points: np.ndarray) -> np.ndarray:
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()
    anchor_count = anchor_points.shape[0]
    cell_count = shifts.shape[0]
    all_points = anchor_points.reshape((1, anchor_count, 2)) + shifts.reshape((1, cell_count, 2)).transpose(
        (1, 0, 2)
    )
    return all_points.reshape((cell_count * anchor_count, 2))


class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels: list[int] | None = None, row: int = 2, line: int = 2) -> None:
        super().__init__()
        self.pyramid_levels = pyramid_levels if pyramid_levels is not None else [3]
        self.strides = [2**level for level in self.pyramid_levels]
        self.row = row
        self.line = line

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image_shape = np.array(image.shape[2:])
        image_shapes = [(image_shape + 2**level - 1) // (2**level) for level in self.pyramid_levels]
        all_anchor_points = np.zeros((0, 2), dtype=np.float32)
        for shape, stride in zip(image_shapes, self.strides):
            anchor_points = _generate_anchor_points(stride, row=self.row, line=self.line)
            shifted = _shift_grid(shape, stride, anchor_points)
            all_anchor_points = np.append(all_anchor_points, shifted, axis=0)
        all_anchor_points = np.expand_dims(all_anchor_points, axis=0).astype(np.float32)
        return torch.from_numpy(all_anchor_points).to(device=image.device)


class FpnDecoder(nn.Module):
    def __init__(self, c3_size: int = 256, c4_size: int = 512, c5_size: int = 512, feature_size: int = 256) -> None:
        super().__init__()
        self.P5_1 = nn.Conv2d(c5_size, feature_size, kernel_size=1)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.P4_1 = nn.Conv2d(c4_size, feature_size, kernel_size=1)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.P3_1 = nn.Conv2d(c3_size, feature_size, kernel_size=1)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        c3, c4, c5 = inputs
        p5_x = self.P5_1(c5)
        p5_upsampled = self.P5_upsampled(p5_x)
        p5_x = self.P5_2(p5_x)

        p4_x = self.P4_1(c4)
        p4_x = p5_upsampled + p4_x
        p4_upsampled = self.P4_upsampled(p4_x)
        p4_x = self.P4_2(p4_x)

        p3_x = self.P3_1(c3)
        p3_x = p3_x + p4_upsampled
        p3_x = self.P3_2(p3_x)
        return [p3_x, p4_x, p5_x]


class P2PNet(nn.Module):
    def __init__(self, row: int = 2, line: int = 2) -> None:
        super().__init__()
        num_anchor_points = row * line
        self.backbone = Vgg16BnBackbone()
        self.regression = RegressionHead(256, num_anchor_points)
        self.classification = ClassificationHead(256, num_anchor_points, num_classes=2)
        self.anchor_points = AnchorPoints(pyramid_levels=[3], row=row, line=line)
        self.fpn = FpnDecoder(256, 512, 512)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(image)
        pyramid = self.fpn([features[1], features[2], features[3]])
        batch_size = image.shape[0]
        regression = self.regression(pyramid[1]) * 100.0
        classification = self.classification(pyramid[1])
        anchor_points = self.anchor_points(image).repeat(batch_size, 1, 1)
        return {"pred_logits": classification, "pred_points": regression + anchor_points}


def build_p2pnet(row: int = 2, line: int = 2) -> P2PNet:
    return P2PNet(row=row, line=line)
