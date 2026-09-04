"""DISCO dataset for the counting baseline.

Splits are the ones shipped in `disco_density_maps.zip` and materialised by
prepare.py. Nothing here re-partitions the data.

The one non-obvious piece is the density resize. A DISCO density map is people
per pixel: the label is the sum of the array, not its mean. Bilinear or area
interpolation treats the map as an image and rescales that sum by the area
ratio, so a naive resize silently divides the count target by ~68 at our
geometry. `resize_density_preserving_sum` therefore aggregates by area and then
renormalises back to the original total, which keeps the count invariant.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

# ImageNet statistics. The trunk is randomly initialised, so these are just a
# fixed normalisation today; they are kept so that swapping in pretrained
# weights later does not change the input pipeline.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def resize_density_preserving_sum(
    density: np.ndarray, out_hw: tuple[int, int]
) -> torch.Tensor:
    """Resize a density map so that its total is unchanged.

    Area interpolation averages over each source region, which is the correct
    local aggregation but scales the total by the pixel-count ratio. The exact
    ratio only holds when the output divides the input evenly, so the sum is
    restored explicitly instead of by a closed-form factor.
    """
    src = torch.from_numpy(np.ascontiguousarray(density, dtype=np.float32))[None, None]
    total = src.sum()
    out = F.interpolate(src, size=out_hw, mode="area")[0, 0]
    scaled = out.sum()
    if scaled > 0:
        out = out * (total / scaled)
    return out


def load_image(path: Path, out_hw: tuple[int, int]) -> torch.Tensor:
    """Read a JPEG, resize to (H, W) and normalise to a CHW float tensor."""
    with Image.open(path) as img:
        img = img.convert("RGB").resize((out_hw[1], out_hw[0]), Image.BILINEAR)
        array = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (tensor - mean) / std


class DiscoDensityDataset(Dataset):
    """One extracted DISCO split: image, density map and scalar count."""

    def __init__(
        self,
        root: Path,
        split: str,
        image_hw: tuple[int, int],
        density_hw: tuple[int, int],
    ):
        self.dir = Path(root) / split
        if not self.dir.is_dir():
            raise FileNotFoundError(
                f"{self.dir} does not exist, run sanas_baseline.prepare first"
            )
        self.ids = sorted(p.stem for p in self.dir.glob("*.npy"))
        if not self.ids:
            raise FileNotFoundError(f"no samples in {self.dir}")
        self.image_hw = image_hw
        self.density_hw = density_hw

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        image_id = self.ids[index]
        density = np.load(self.dir / f"{image_id}.npy")
        # The count is taken from the full resolution annotation, before any
        # resize, so it is the dataset's own label and not a resampling result.
        # DISCO sums are floats and not exactly integral (min 0.3 over the
        # annotated set), and the rounding policy is still open, so the raw
        # float sum is kept as the regression target.
        count = float(density.sum())
        image = load_image(self.dir / f"{image_id}.jpg", self.image_hw)
        target = resize_density_preserving_sum(density, self.density_hw)
        return image, target, torch.tensor(count, dtype=torch.float32)

    def counts(self) -> np.ndarray:
        return np.array(
            [float(np.load(self.dir / f"{i}.npy").sum()) for i in self.ids],
            dtype=np.float64,
        )


def read_manifest(root: Path) -> dict:
    path = Path(root) / "manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
