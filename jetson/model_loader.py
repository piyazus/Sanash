"""
Model Loader
============

Loads the CSRNet model for inference on Jetson Nano.
Priority order:
  1. TensorRT FP16 engine (fastest — ~15ms)
  2. PyTorch checkpoint (fallback — ~80ms)
  3. Mock model (testing / no GPU)

Usage:
    from jetson.model_loader import load_model
    model = load_model("/opt/sanash/models/csrnet_fp16.trt")
    count = model.predict(frame_rgb)
"""
import logging
import os
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

# ImageNet normalization constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class InferenceModel(Protocol):
    """Common interface for all model backends."""

    def predict(self, frame_rgb: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Run inference on an RGB frame.

        Parameters
        ----------
        frame_rgb : np.ndarray
            Shape (H, W, 3) uint8 RGB image.

        Returns
        -------
        count : float
            Estimated passenger count (sum of density map).
        density_map : np.ndarray
            Float32 density map (H/8, W/8).
        """
        ...


def _preprocess(frame_rgb: np.ndarray, width: int = 512, height: int = 384) -> np.ndarray:
    """Resize + normalize frame to (1, 3, H, W) float32 tensor."""
    try:
        from PIL import Image
        img = Image.fromarray(frame_rgb).resize((width, height), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - _MEAN) / _STD
        return arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)
    except ImportError:
        # Minimal resize without PIL
        import cv2
        img = cv2.resize(frame_rgb, (width, height))
        arr = img.astype(np.float32) / 255.0
        arr = (arr - _MEAN) / _STD
        return arr.transpose(2, 0, 1)[np.newaxis]


class TensorRTModel:
    """TensorRT FP16 inference engine for Jetson Nano."""

    def __init__(self, engine_path: str, width: int = 512, height: int = 384):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401

        self.width = width
        self.height = height
        logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self._allocate_buffers()
        log.info(f"TensorRT engine loaded: {engine_path}")

    def _allocate_buffers(self):
        import pycuda.driver as cuda

        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()

        for binding in self.engine:
            size = (
                abs(self.engine.get_binding_volume(self.engine.get_binding_index(binding)))
            )
            dtype = np.float32
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(self.engine.get_binding_index(binding)):
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})

    def predict(self, frame_rgb: np.ndarray) -> tuple[float, np.ndarray]:
        import pycuda.driver as cuda

        tensor = _preprocess(frame_rgb, self.width, self.height)
        np.copyto(self.inputs[0]["host"], tensor.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]["device"], self.inputs[0]["host"], self.stream
        )
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle
        )
        cuda.memcpy_dtoh_async(
            self.outputs[0]["host"], self.outputs[0]["device"], self.stream
        )
        self.stream.synchronize()

        out_h, out_w = self.height // 8, self.width // 8
        density = self.outputs[0]["host"].reshape(out_h, out_w)
        count = float(density.sum())
        return count, density


class PyTorchModel:
    """PyTorch CSRNet model (CPU or CUDA) — fallback when TensorRT unavailable."""

    def __init__(self, checkpoint_path: str, width: int = 512, height: int = 384):
        import torch
        import torch.nn as nn

        self.width = width
        self.height = height
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_csrnet().to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get("state_dict", ckpt)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        log.info(f"PyTorch model loaded on {self.device}: {checkpoint_path}")

    def _build_csrnet(self):
        """Minimal CSRNet: VGG frontend (3 blocks) + dilated conv backend."""
        import torch.nn as nn
        import torchvision.models as tvm

        vgg = tvm.vgg16(weights=None)
        frontend = nn.Sequential(*list(vgg.features.children())[:23])

        backend = nn.Sequential(
            nn.Conv2d(512, 256, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

        class CSRNet(nn.Module):
            def forward(self, x):
                return backend(frontend(x))

        return CSRNet()

    def predict(self, frame_rgb: np.ndarray) -> tuple[float, np.ndarray]:
        import torch

        tensor = torch.from_numpy(_preprocess(frame_rgb, self.width, self.height)).to(
            self.device
        )
        with torch.no_grad():
            out = self.model(tensor)
        density = out.squeeze().cpu().numpy()
        count = float(density.sum())
        return count, density


class MockModel:
    """Synthetic model for testing without GPU or model files."""

    def __init__(self, width: int = 512, height: int = 384):
        self.width = width
        self.height = height
        self._rng = np.random.default_rng(42)
        log.warning("Using MockModel — returns synthetic density maps")

    def predict(self, frame_rgb: np.ndarray) -> tuple[float, np.ndarray]:
        H, W = self.height // 8, self.width // 8
        n_people = self._rng.integers(5, 55)
        density = np.zeros((H, W), dtype=np.float32)
        for _ in range(int(n_people)):
            y = self._rng.integers(0, H)
            x = self._rng.integers(0, W)
            density[y, x] += 1.0
        from scipy.ndimage import gaussian_filter
        density = gaussian_filter(density, sigma=2.0)
        count = float(density.sum())
        return count, density


def load_model(
    trt_path: str | None = None,
    pth_path: str | None = None,
    width: int = 512,
    height: int = 384,
) -> InferenceModel:
    """
    Load the best available model.

    Returns TensorRT > PyTorch > Mock in order of preference.
    """
    # Try TensorRT
    if trt_path and Path(trt_path).exists():
        try:
            return TensorRTModel(trt_path, width, height)
        except Exception as exc:
            log.warning(f"TensorRT load failed ({exc}), trying PyTorch...")

    # Try PyTorch
    if pth_path and Path(pth_path).exists():
        try:
            return PyTorchModel(pth_path, width, height)
        except Exception as exc:
            log.warning(f"PyTorch load failed ({exc}), using mock model...")

    return MockModel(width, height)
