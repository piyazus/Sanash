import numpy as np
import torch
from torch import nn

from .backbone import build_backbone


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, feature_size=256):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.output(out)
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 2)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, num_classes=2, feature_size=256):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.output(out)
        out = out.permute(0, 2, 3, 1)
        batch_size, width, height, _ = out.shape
        out = out.view(batch_size, width, height, self.num_anchor_points, self.num_classes)
        return out.contiguous().view(x.shape[0], -1, self.num_classes)


def generate_anchor_points(stride=16, row=2, line=2):
    row_step = stride / row
    line_step = stride / line
    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    return np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()


def shift(shape, stride, anchor_points):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()
    anchor_count = anchor_points.shape[0]
    shift_count = shifts.shape[0]
    all_anchor_points = (
        anchor_points.reshape((1, anchor_count, 2))
        + shifts.reshape((1, shift_count, 2)).transpose((1, 0, 2))
    )
    return all_anchor_points.reshape((shift_count * anchor_count, 2))


class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, row=2, line=2):
        super().__init__()
        self.pyramid_levels = pyramid_levels or [3, 4, 5, 6, 7]
        self.strides = strides or [2**level for level in self.pyramid_levels]
        self.row = row
        self.line = line

    def forward(self, image):
        image_shape = np.array(image.shape[2:])
        image_shapes = [(image_shape + 2**level - 1) // (2**level) for level in self.pyramid_levels]
        all_anchor_points = np.zeros((0, 2), dtype=np.float32)
        for index, level in enumerate(self.pyramid_levels):
            base_anchor_points = generate_anchor_points(2**level, row=self.row, line=self.line)
            shifted_anchor_points = shift(image_shapes[index], self.strides[index], base_anchor_points)
            all_anchor_points = np.append(all_anchor_points, shifted_anchor_points, axis=0)
        all_anchor_points = np.expand_dims(all_anchor_points, axis=0).astype(np.float32)
        return torch.from_numpy(all_anchor_points).to(image.device)


class Decoder(nn.Module):
    def __init__(self, c3_size, c4_size, c5_size, feature_size=256):
        super().__init__()
        self.P5_1 = nn.Conv2d(c5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_1 = nn.Conv2d(c4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_1 = nn.Conv2d(c3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        c3, c4, c5 = inputs
        p5_x = self.P5_1(c5)
        p5_upsampled_x = self.P5_upsampled(p5_x)
        p5_x = self.P5_2(p5_x)

        p4_x = self.P4_1(c4)
        p4_x = p5_upsampled_x + p4_x
        p4_upsampled_x = self.P4_upsampled(p4_x)
        p4_x = self.P4_2(p4_x)

        p3_x = self.P3_1(c3)
        p3_x = p3_x + p4_upsampled_x
        p3_x = self.P3_2(p3_x)
        return [p3_x, p4_x, p5_x]


class P2PNet(nn.Module):
    def __init__(self, backbone, row=2, line=2):
        super().__init__()
        self.backbone = backbone
        self.num_classes = 2
        num_anchor_points = row * line
        self.regression = RegressionModel(256, num_anchor_points=num_anchor_points)
        self.classification = ClassificationModel(
            256,
            num_anchor_points=num_anchor_points,
            num_classes=self.num_classes,
        )
        self.anchor_points = AnchorPoints(pyramid_levels=[3], row=row, line=line)
        self.fpn = Decoder(256, 512, 512)

    def forward(self, samples):
        features = self.backbone(samples)
        features_fpn = self.fpn([features[1], features[2], features[3]])
        batch_size = features[0].shape[0]
        regression = self.regression(features_fpn[1]) * 100
        classification = self.classification(features_fpn[1])
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)
        return {
            "pred_logits": classification,
            "pred_points": regression + anchor_points,
        }


def build(args, training=False):
    if training:
        raise NotImplementedError("This local setup only supports inference.")
    backbone = build_backbone(args)
    return P2PNet(backbone, args.row, args.line)
