"""Configuration for the DISCO counting baseline.

Every default here is sized for a CPU smoke test on the local Windows machine:
torch 2.13.0+cpu, no CUDA device. Nothing in this package assumes a GPU, a
Kaggle kernel or any paid backend. Raising `epochs` or dropping the `limit_*`
fields will make a run slow, not illegal, but a real training run is a separate
decision that has to be logged before it is started.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# development/src/sanas_baseline/config.py -> repository root
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class BaselineConfig:
    """Paths, geometry and optimisation settings for one baseline run."""

    # Source archives. Read-only, never unpacked in full.
    images_zip: Path = REPO_ROOT / "data" / "external" / "disco_images.zip"
    density_zip: Path = REPO_ROOT / "data" / "external" / "disco_density_maps.zip"

    # Extracted working subset and run artifacts. Both are gitignored.
    subset_dir: Path = REPO_ROOT / "data" / "interim" / "disco_baseline"
    output_dir: Path = REPO_ROOT / "outputs" / "baseline"

    # DISCO images are 1920x1080. 384x216 keeps that aspect ratio exactly and
    # divides by the trunk stride of 8, so the density target is 48x27.
    image_height: int = 216
    image_width: int = 384
    density_stride: int = 8

    # The sum preserving resize in data.py already lifts the target out of the
    # 1e-2 per pixel range, since one 8x8 output cell absorbs 1600 source
    # pixels, so no extra scaling is needed. The knob stays because the loss is
    # sensitive to it and a different output stride would change the argument.
    # Counts are always reported in people, never in scaled units.
    density_scale: float = 1.0

    batch_size: int = 2
    learning_rate: float = 1e-4
    epochs: int = 2
    seed: int = 2026
    device: str = "cpu"
    num_workers: int = 0

    def density_hw(self) -> tuple[int, int]:
        return (
            self.image_height // self.density_stride,
            self.image_width // self.density_stride,
        )

    def to_json(self) -> str:
        payload = {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()
        }
        payload["density_hw"] = list(self.density_hw())
        return json.dumps(payload, indent=1)
