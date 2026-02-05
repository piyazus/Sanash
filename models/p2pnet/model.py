
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class P2PNet(nn.Module):
    def __init__(self, backbone='vgg16'):
        super(P2PNet, self).__init__()
        
        # Backbone (VGG16 without last pool and fc)
        vgg = models.vgg16(pretrained=True)
        self.backbone = nn.Sequential(*list(vgg.features.children())[:-1]) # Stripped
        # VGG output stride is 16 usually? Need to check.
        # If we remove last pool, it's stride 16.
        
        self.conv_out = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        
        # Regression Head (x, y)
        self.reg_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 2, 3, padding=1)
        )
        
        # Classification Head (Confidence)
        self.cls_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, 3, padding=1) # Logits
        )
        
    def forward(self, x):
        features = self.backbone(x)
        features = self.conv_out(features)
        
        # Predictions map: B x 256 x H/16 x W/16
        
        # Regression: Offset or Coordinates?
        # Usually predict offset relative to anchor or pixel.
        # P2PNet paper uses direct point prediction per pixel location.
        # Output shape: B x 2 x H' x W'
        
        reg_map = self.reg_head(features)
        cls_map = self.cls_head(features)
        
        # Flatten for matching
        B, _, H, W = reg_map.shape
        points = reg_map.permute(0, 2, 3, 1).reshape(B, -1, 2) # B x N x 2
        logits = cls_map.permute(0, 2, 3, 1).reshape(B, -1, 1) # B x N x 1
        
        return points, logits
        
# Note: P2PNet usually adds coordinate grid to reg_map to get absolute coords
# We will do that in loss computation or just assume network learns offsets + grid
# But standard P2PNet explicitly adds grid.

    def predict(self, x):
        points, logits = self.forward(x)
        scores = torch.sigmoid(logits)
        return points, scores

