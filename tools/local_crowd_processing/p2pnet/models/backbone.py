from torch import nn
from torchvision import models as tv_models


def _build_vgg(name: str) -> nn.Module:
    if name == "vgg16_bn":
        try:
            return tv_models.vgg16_bn(weights=None)
        except TypeError:
            return tv_models.vgg16_bn(pretrained=False)
    if name == "vgg16":
        try:
            return tv_models.vgg16(weights=None)
        except TypeError:
            return tv_models.vgg16(pretrained=False)
    raise ValueError(f"Unsupported backbone: {name}")


class BackboneBaseVGG(nn.Module):
    def __init__(self, backbone: nn.Module, name: str, return_interm_layers: bool):
        super().__init__()
        features = list(backbone.features.children())
        if return_interm_layers:
            if name == "vgg16_bn":
                self.body1 = nn.Sequential(*features[:13])
                self.body2 = nn.Sequential(*features[13:23])
                self.body3 = nn.Sequential(*features[23:33])
                self.body4 = nn.Sequential(*features[33:43])
            else:
                self.body1 = nn.Sequential(*features[:9])
                self.body2 = nn.Sequential(*features[9:16])
                self.body3 = nn.Sequential(*features[16:23])
                self.body4 = nn.Sequential(*features[23:30])
        else:
            if name == "vgg16_bn":
                self.body = nn.Sequential(*features[:44])
            else:
                self.body = nn.Sequential(*features[:30])
        self.return_interm_layers = return_interm_layers

    def forward(self, x):
        outputs = []
        if self.return_interm_layers:
            current = x
            for layer in (self.body1, self.body2, self.body3, self.body4):
                current = layer(current)
                outputs.append(current)
            return outputs

        return [self.body(x)]


class BackboneVGG(BackboneBaseVGG):
    def __init__(self, name: str, return_interm_layers: bool):
        super().__init__(_build_vgg(name), name, return_interm_layers)


def build_backbone(args):
    return BackboneVGG(args.backbone, True)
