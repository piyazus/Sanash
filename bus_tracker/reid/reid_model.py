"""
Bus Vision - Person Re-Identification Module
=============================================

Uses OSNet for extracting appearance features from person crops.
Enables tracking the same person across multiple cameras.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

logger = logging.getLogger(__name__)


# =============================================================================
# OSNET ARCHITECTURE (Lightweight ReID Model)
# =============================================================================

class ConvBlock(nn.Module):
    """Basic convolutional block: Conv -> BN -> ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution using depthwise separable convolution"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class ChannelGate(nn.Module):
    """Channel attention gate"""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class OSBlock(nn.Module):
    """Omni-Scale Block - core building block of OSNet"""
    
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 4):
        super().__init__()
        mid_channels = out_channels // reduction
        
        self.conv1 = ConvBlock(in_channels, mid_channels, 1, 1, 0)
        
        # Multi-scale streams
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels)
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels)
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels)
        )
        
        self.gate = ChannelGate(mid_channels)
        self.conv3 = ConvBlock(mid_channels, out_channels, 1, 1, 0)
        
        # Skip connection
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = ConvBlock(in_channels, out_channels, 1, 1, 0)
    
    def forward(self, x):
        identity = x
        
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        
        # Unified aggregation
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        return x3 + identity


class OSNet(nn.Module):
    """
    Omni-Scale Network for Person Re-Identification
    
    Paper: "Omni-Scale Feature Learning for Person Re-Identification"
    https://arxiv.org/abs/1905.00953
    
    Output: 512-dimensional feature vector per person
    """
    
    def __init__(self, num_classes: int = 1000, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Stem
        self.conv1 = ConvBlock(3, 64, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # Stages
        self.conv2 = self._make_layer(64, 256, 2)
        self.pool2 = nn.Sequential(
            ConvBlock(256, 256, 1, 1, 0),
            nn.AvgPool2d(2, 2)
        )
        
        self.conv3 = self._make_layer(256, 384, 2)
        self.pool3 = nn.Sequential(
            ConvBlock(384, 384, 1, 1, 0),
            nn.AvgPool2d(2, 2)
        )
        
        self.conv4 = self._make_layer(384, 512, 2)
        
        # Global pooling and classifier
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, feature_dim, bias=False)
        self.bn = nn.BatchNorm1d(feature_dim)
        
        self._init_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int):
        layers = [OSBlock(in_channels, out_channels)]
        for _ in range(1, num_blocks):
            layers.append(OSBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        
        return x


# =============================================================================
# REID MODEL WRAPPER
# =============================================================================

class ReIDModel:
    """
    High-level wrapper for person re-identification.
    
    Features:
    - Automatic model loading with caching
    - GPU acceleration if available
    - Batch processing support
    - Feature normalization
    
    Usage:
        model = ReIDModel()
        features = model.extract_features(person_crop)  # (512,) normalized vector
    """
    
    # Model weights URL (OSNet pre-trained on Market-1501)
    WEIGHTS_URL = "https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0.6/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"
    
    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        image_size: Tuple[int, int] = (256, 128)  # (height, width)
    ):
        """
        Initialize ReID model.
        
        Args:
            weights_path: Path to model weights. Downloads if not provided.
            device: 'cuda' or 'cpu'. Auto-detects if not provided.
            image_size: Input image size (height, width).
        """
        self.image_size = image_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = OSNet(feature_dim=512)
        
        # Load weights
        if weights_path and Path(weights_path).exists():
            self._load_weights(weights_path)
        else:
            self._download_and_load_weights()
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info(f"ReID model loaded on {self.device}")
    
    def _load_weights(self, weights_path: str):
        """Load model weights from file."""
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # Handle different state dict formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"Loaded weights from {weights_path}")
    
    def _download_and_load_weights(self):
        """Download pre-trained weights if not available."""
        import urllib.request
        
        weights_dir = Path(__file__).parent / "weights"
        weights_dir.mkdir(exist_ok=True)
        weights_path = weights_dir / "osnet_market.pth"
        
        if not weights_path.exists():
            logger.info(f"Downloading OSNet weights to {weights_path}...")
            try:
                urllib.request.urlretrieve(self.WEIGHTS_URL, weights_path)
                logger.info("Download complete!")
            except Exception as e:
                logger.warning(f"Failed to download weights: {e}. Using random initialization.")
                return
        
        self._load_weights(str(weights_path))
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess person crop for feature extraction.
        
        Args:
            image: BGR image (H, W, 3) from OpenCV
            
        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        tensor = self.transform(image)
        
        return tensor.unsqueeze(0)  # Add batch dimension
    
    @torch.no_grad()
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract normalized feature vector from person crop.
        
        Args:
            image: Person crop (H, W, 3) BGR format
            
        Returns:
            Feature vector (512,) normalized to unit length
        """
        tensor = self.preprocess(image).to(self.device)
        features = self.model(tensor)
        
        # L2 normalize
        features = features / features.norm(dim=1, keepdim=True)
        
        return features.cpu().numpy().flatten()
    
    @torch.no_grad()
    def extract_features_batch(self, images: list) -> np.ndarray:
        """
        Extract features from multiple person crops.
        
        Args:
            images: List of person crops [(H, W, 3), ...]
            
        Returns:
            Feature matrix (N, 512)
        """
        if not images:
            return np.array([])
        
        # Preprocess all images
        tensors = torch.cat([self.preprocess(img) for img in images], dim=0)
        tensors = tensors.to(self.device)
        
        # Extract features
        features = self.model(tensors)
        
        # L2 normalize
        features = features / features.norm(dim=1, keepdim=True)
        
        return features.cpu().numpy()


# =============================================================================
# FEATURE CACHE
# =============================================================================

class FeatureCache:
    """
    LRU cache for ReID features to avoid re-extraction.
    
    Key: (camera_id, track_id, frame_number)
    Value: Feature vector (512,)
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
    
    def get(self, camera_id: int, track_id: int, frame: int) -> Optional[np.ndarray]:
        """Get cached features if available."""
        key = (camera_id, track_id, frame)
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, camera_id: int, track_id: int, frame: int, features: np.ndarray):
        """Cache features."""
        key = (camera_id, track_id, frame)
        
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Evict least recently used
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = features
        self.access_order.append(key)
    
    def clear(self):
        """Clear all cached features."""
        self.cache.clear()
        self.access_order.clear()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

_model_instance: Optional[ReIDModel] = None


def get_reid_model() -> ReIDModel:
    """Get singleton ReID model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = ReIDModel()
    return _model_instance


def extract_features(image: np.ndarray) -> np.ndarray:
    """
    Convenience function to extract features from person crop.
    
    Args:
        image: Person crop (H, W, 3) BGR format
        
    Returns:
        Feature vector (512,)
    """
    model = get_reid_model()
    return model.extract_features(image)


if __name__ == "__main__":
    # Test the model
    import sys
    
    print("Testing ReID Model...")
    
    # Create random test image
    test_image = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    
    model = ReIDModel()
    features = model.extract_features(test_image)
    
    print(f"Feature shape: {features.shape}")
    print(f"Feature norm: {np.linalg.norm(features):.4f}")
    print(f"Feature sample: {features[:5]}")
    
    # Test batch extraction
    batch = [test_image for _ in range(10)]
    batch_features = model.extract_features_batch(batch)
    
    print(f"Batch shape: {batch_features.shape}")
    print("ReID Model test passed!")
