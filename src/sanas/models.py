"""PHASE 2 ASSET - DORMANT, NOT DEAD. Do not delete.

Unused in Phase 1 (door-mounted APC counting, which is depth geometry and needs
no learned model). This becomes active in Phase 2: RGB whole-frame cabin
classification, once camera permission for a real bus lands. Kept intact for
the same reason as notebooks/depth_occupancy_multizone/ - dormant, not gone.

Frozen backbone + CORN ordinal head.

Two backbones, both permissively licensed (checked 2026-08-09):

  dinov2_vits14   timm 'vit_small_patch14_dinov2.lvd142m', 21M params.
                  DINOv2 README: "DINOv2 code and model weights are released
                  under the Apache License 2.0."
  convnext_tiny   timm 'convnext_tiny.fb_in1k'. facebookresearch/ConvNeXt is
                  MIT; timm is Apache-2.0. This is the edge-deployable
                  comparison: plain convolutions export to ONNX/TensorRT far
                  more predictably than ViT attention.

Deliberately NOT here: P2PNet. Its licence restricts use to "the purpose of
academic research", which excludes shipping inside Avtobys. See
experiments/log.md.

The backbone is frozen and only the head trains, so a smoke test is minutes of
compute rather than hours.
"""

from __future__ import annotations

import torch
import torch.nn as nn

BACKBONES = {
    "dinov2_vits14": {
        "timm_name": "vit_small_patch14_dinov2.lvd142m",
        "input_size": 224,  # must be a multiple of the patch size (14)
        "license": "Apache-2.0 (code and weights)",
        "params_m": 21,
    },
    "convnext_tiny": {
        "timm_name": "convnext_tiny.fb_in1k",
        "input_size": 224,
        "license": "MIT (ConvNeXt) / Apache-2.0 (timm)",
        "params_m": 28,
    },
}


class OrdinalHead(nn.Module):
    """LayerNorm + dropout + linear, producing one logit per ordinal threshold."""

    def __init__(self, in_features: int, num_thresholds: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_thresholds)
        nn.init.zeros_(self.fc.bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(self.norm(feats)))


class FrozenBackboneOrdinal(nn.Module):
    def __init__(self, backbone: nn.Module, head: OrdinalHead, frozen: bool):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.frozen = frozen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.frozen:
            with torch.no_grad():
                feats = self.backbone(x)
        else:
            feats = self.backbone(x)
        return self.head(feats.float())

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)


def build_model(
    backbone: str,
    num_thresholds: int,
    pretrained: bool = True,
    freeze: bool = True,
    dropout: float = 0.1,
) -> tuple[FrozenBackboneOrdinal, dict]:
    """Build a frozen-backbone CORN model. Returns (model, backbone_info)."""
    import timm  # imported lazily so the module imports without torch extras

    if backbone not in BACKBONES:
        raise ValueError(
            f"unknown backbone {backbone!r}; choose from {sorted(BACKBONES)}"
        )
    info = dict(BACKBONES[backbone])

    net = timm.create_model(info["timm_name"], pretrained=pretrained, num_classes=0)
    if freeze:
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)

    dim = getattr(net, "num_features", None)
    if dim is None:
        raise RuntimeError(f"cannot determine feature dim for {info['timm_name']}")

    model = FrozenBackboneOrdinal(
        net, OrdinalHead(dim, num_thresholds, dropout), freeze
    )
    info["feature_dim"] = dim
    info["backbone_key"] = backbone
    info["frozen"] = freeze
    info["trainable_params"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    info["total_params"] = sum(p.numel() for p in model.parameters())
    return model, info


def train_mode(model: FrozenBackboneOrdinal) -> None:
    """Put the head in train mode while keeping a frozen backbone in eval.

    Matters: a frozen backbone left in train() would keep updating BatchNorm
    running statistics on ConvNeXt-style models.
    """
    model.train()
    if model.frozen:
        model.backbone.eval()
