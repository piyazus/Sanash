"""
ShanghaiTech data utilities for the Sanash project.

Provides:
- Efficient loading of .mat head annotations.
- Conversions to:
  - YOLO-style bounding boxes (delegated to existing script logic).
  - Density maps for CSRNet training.
  - Point maps for P2PNet-style training.
- A unified PyTorch Dataset for density / point supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple

import numpy as np
import torch
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image


def load_shanghaitech_points(mat_path: Path) -> np.ndarray:
    """
    Load head center points from a ShanghaiTech-style .mat file.

    Returns
    -------
    points : (N, 2) float32 array of (x, y) coordinates in pixels.
    """
    mat = loadmat(str(mat_path))

    if "image_info" in mat:
        info = mat["image_info"]
        try:
            loc = info[0, 0]["location"][0, 0]
            pts = np.asarray(loc, dtype=np.float32)
            if pts.ndim == 2 and pts.shape[1] == 2:
                return pts
        except Exception:
            pass

    for v in mat.values():
        arr = np.asarray(v)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr.astype(np.float32)

    raise ValueError(f"Could not infer head locations from {mat_path}")


def points_to_density_map(
    points: np.ndarray,
    image_size: Tuple[int, int],
    sigma: float = 4.0,
    downsample: int = 4,
) -> np.ndarray:
    """
    Convert head center points to a density map using Gaussian kernels.

    Implementation:
    - Create an impulse map with ones at (x, y) locations.
    - Optionally downsample to a coarser grid for efficiency.
    - Apply a Gaussian filter with standard deviation `sigma`.
    """
    h, w = image_size
    if downsample > 1:
        h_ds = h // downsample
        w_ds = w // downsample
        density = np.zeros((h_ds, w_ds), dtype=np.float32)
        for x, y in points:
            ix = min(w_ds - 1, max(0, int(x / downsample)))
            iy = min(h_ds - 1, max(0, int(y / downsample)))
            density[iy, ix] += 1.0
    else:
        density = np.zeros((h, w), dtype=np.float32)
        for x, y in points:
            ix = min(w - 1, max(0, int(x)))
            iy = min(h - 1, max(0, int(y)))
            density[iy, ix] += 1.0

    density = gaussian_filter(density, sigma=sigma)
    return density.astype(np.float32)


def points_to_point_map(
    points: np.ndarray,
    image_size: Tuple[int, int],
    downsample: int = 4,
) -> np.ndarray:
    """
    Convert head centers to a sparse point map on a downsampled grid.

    This is useful as a supervision signal for P2PNet-style point prediction.
    """
    h, w = image_size
    h_ds = h // downsample
    w_ds = w // downsample

    point_map = np.zeros((h_ds, w_ds), dtype=np.float32)
    for x, y in points:
        ix = min(w_ds - 1, max(0, int(x / downsample)))
        iy = min(h_ds - 1, max(0, int(y / downsample)))
        point_map[iy, ix] = 1.0
    return point_map


@dataclass
class ShanghaiSample:
    image_path: Path
    mat_path: Path
    points: np.ndarray


# ShanghaiTech convention: annotations are often GT_IMG_1.mat, images are IMG_1.jpg
GT_PREFIX = "GT_"


def _image_stem_from_mat_stem(mat_stem: str) -> str:
    """Map .mat file stem to image file stem (strip GT_ prefix if present)."""
    if mat_stem.startswith(GT_PREFIX):
        return mat_stem[len(GT_PREFIX) :]
    return mat_stem


def iter_shanghaitech_samples(
    image_dir: Path,
    mat_dir: Path,
    image_ext: str = ".jpg",
) -> Iterable[ShanghaiSample]:
    """
    Yield ShanghaiSample objects by matching stems between images and .mat files.
    Handles ShanghaiTech naming: .mat files may be GT_IMG_1.mat while images are IMG_1.jpg.
    """
    mat_paths = sorted(p for p in mat_dir.glob("*.mat") if p.is_file())
    for mat_path in mat_paths:
        stem = mat_path.stem
        image_stem = _image_stem_from_mat_stem(stem)
        img_path = image_dir / f"{image_stem}{image_ext}"
        if not img_path.is_file():
            continue
        points = load_shanghaitech_points(mat_path)
        yield ShanghaiSample(image_path=img_path, mat_path=mat_path, points=points)


TargetType = Literal["density", "points"]


class ShanghaiTechDataset(Dataset):
    """
    Unified dataset for ShanghaiTech with multiple target types:

    - 'density': Gaussian-smoothed density maps (CSRNet).
    - 'points' : Sparse point maps (P2PNet-style).
    """

    def __init__(
        self,
        image_dir: str | Path,
        mat_dir: str | Path,
        target_type: TargetType = "density",
        sigma: float = 4.0,
        downsample: int = 4,
        image_ext: str = ".jpg",
        transform: T.Compose | None = None,
    ) -> None:
        super().__init__()
        self.image_dir = Path(image_dir)
        self.mat_dir = Path(mat_dir)
        self.target_type = target_type
        self.sigma = sigma
        self.downsample = downsample
        self.image_ext = image_ext

        self.transform = transform or T.Compose(
            [
                T.ToTensor(),
            ]
        )

        self.samples: List[ShanghaiSample] = list(
            iter_shanghaitech_samples(self.image_dir, self.mat_dir, self.image_ext)
        )
        if not self.samples:
            raise RuntimeError(f"No ShanghaiTech samples found in {self.image_dir} / {self.mat_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = self._load_image(sample.image_path)

        _, h, w = image.shape
        points = sample.points

        if self.target_type == "density":
            target_np = points_to_density_map(
                points, image_size=(h, w), sigma=self.sigma, downsample=self.downsample
            )
        else:  # "points"
            target_np = points_to_point_map(
                points, image_size=(h, w), downsample=self.downsample
            )

        target = torch.from_numpy(target_np).unsqueeze(0)  # (1, H', W')

        return {
            "image": image,
            "target": target,
        }


__all__ = [
    "load_shanghaitech_points",
    "points_to_density_map",
    "points_to_point_map",
    "ShanghaiSample",
    "iter_shanghaitech_samples",
    "ShanghaiTechDataset",
]

