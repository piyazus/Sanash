from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageOps


VALID_SPLITS = {"train", "val"}


@dataclass(frozen=True)
class AlmatySample:
    filename: str
    image_path: Path
    gt_path: Path
    image_id: int


def _read_split_file(root: Path, split: str) -> list[str]:
    split_file = root / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    filenames = [
        line.strip()
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not filenames:
        raise ValueError(f"Split file is empty: {split_file}")
    return filenames


def _load_points(gt_path: Path) -> torch.Tensor:
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth points file not found: {gt_path}")

    points = np.load(gt_path)
    if points.size == 0:
        return torch.empty((0, 2), dtype=torch.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected [N, 2] points in {gt_path}, got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError(f"Found NaN or inf point coordinates in {gt_path}")

    return torch.as_tensor(points, dtype=torch.float32)


def _pil_size_hw(image: Image.Image) -> torch.Tensor:
    width, height = image.size
    return torch.tensor([height, width], dtype=torch.int64)


def _image_hw(image: Any) -> tuple[int, int] | None:
    if isinstance(image, Image.Image):
        width, height = image.size
        return height, width
    if torch.is_tensor(image):
        if image.ndim >= 2:
            return int(image.shape[-2]), int(image.shape[-1])
    if isinstance(image, np.ndarray):
        if image.ndim >= 2:
            return int(image.shape[0]), int(image.shape[1])
    return None


def _validate_points(points: torch.Tensor, height: int, width: int, source: str) -> None:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected [N, 2] points for {source}, got {tuple(points.shape)}")
    if torch.isnan(points).any() or torch.isinf(points).any():
        raise ValueError(f"Found NaN or inf point coordinates for {source}")
    if points.numel() == 0:
        return
    if (points < 0).any():
        raise ValueError(f"Found negative point coordinates for {source}")

    x = points[:, 0]
    y = points[:, 1]
    if (x > float(width)).any() or (y > float(height)).any():
        raise ValueError(
            f"Found point coordinates outside image bounds for {source}: "
            f"height={height}, width={width}"
        )


def _maybe_scale_points_for_size_change(
    points: torch.Tensor,
    old_hw: tuple[int, int],
    new_hw: tuple[int, int],
) -> torch.Tensor:
    old_h, old_w = old_hw
    new_h, new_w = new_hw
    if old_h == new_h and old_w == new_w:
        return points

    scaled = points.clone()
    if scaled.numel():
        scaled[:, 0] *= float(new_w) / float(old_w)
        scaled[:, 1] *= float(new_h) / float(old_h)
    return scaled


class ResizeWithPoints:
    """Resize a PIL image and scale P2PNet point coordinates with it."""

    def __init__(
        self,
        size: int | tuple[int, int],
        interpolation: Image.Resampling = Image.Resampling.BILINEAR,
    ) -> None:
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.interpolation = interpolation

    def __call__(
        self, image: Image.Image, target: dict[str, Any]
    ) -> tuple[Image.Image, dict[str, Any]]:
        old_w, old_h = image.size
        new_h, new_w = self.size
        image = image.resize((new_w, new_h), self.interpolation)

        points = target["points"].clone()
        if points.numel():
            points[:, 0] *= float(new_w) / float(old_w)
            points[:, 1] *= float(new_h) / float(old_h)

        target = dict(target)
        target["points"] = points
        target["size"] = torch.tensor([new_h, new_w], dtype=torch.int64)
        return image, target


class AlmatyTransitDataset(torch.utils.data.Dataset):
    """Sanash/Almaty transit crowd-counting dataset for P2PNet point training."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transforms: Callable[..., Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transforms = transforms

        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {split!r}")
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.image_dir = self.root / "images" / split
        self.gt_dir = self.root / "gt" / split
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image folder not found: {self.image_dir}")
        if not self.gt_dir.exists():
            raise FileNotFoundError(f"GT folder not found: {self.gt_dir}")

        self.samples = self._build_samples()
        self._validate_file_pairs()

    def _build_samples(self) -> list[AlmatySample]:
        samples: list[AlmatySample] = []
        for image_id, filename in enumerate(_read_split_file(self.root, self.split)):
            image_path = self.image_dir / filename
            gt_path = self.gt_dir / f"{Path(filename).stem}.npy"
            samples.append(
                AlmatySample(
                    filename=filename,
                    image_path=image_path,
                    gt_path=gt_path,
                    image_id=image_id,
                )
            )
        return samples

    def _validate_file_pairs(self) -> None:
        image_stems = {sample.image_path.stem for sample in self.samples}
        gt_stems = {sample.gt_path.stem for sample in self.samples}
        missing_images = [sample.filename for sample in self.samples if not sample.image_path.exists()]
        missing_gt = [sample.gt_path.name for sample in self.samples if not sample.gt_path.exists()]

        disk_gt_stems = {
            path.stem for path in self.gt_dir.glob("*.npy") if path.is_file()
        }
        extra_gt = sorted(disk_gt_stems - image_stems)

        if missing_images:
            raise FileNotFoundError(
                f"{len(missing_images)} image(s) listed in {self.split}.txt are missing; "
                f"first: {missing_images[:5]}"
            )
        if missing_gt:
            raise FileNotFoundError(
                f"{len(missing_gt)} GT file(s) listed in {self.split}.txt are missing; "
                f"first: {missing_gt[:5]}"
            )
        if image_stems != gt_stems:
            raise ValueError(f"Image/GT stem mismatch in split {self.split}")
        if extra_gt:
            raise ValueError(
                f"{len(extra_gt)} extra GT file(s) not listed in {self.split}.txt; "
                f"first: {extra_gt[:5]}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        sample = self.samples[index]
        if not sample.image_path.exists():
            raise FileNotFoundError(f"Image not found: {sample.image_path}")
        if not sample.gt_path.exists():
            raise FileNotFoundError(f"GT file not found: {sample.gt_path}")

        with Image.open(sample.image_path) as raw_image:
            image = ImageOps.exif_transpose(raw_image).convert("RGB")

        orig_w, orig_h = image.size
        points = _load_points(sample.gt_path)
        _validate_points(points, orig_h, orig_w, sample.filename)

        target: dict[str, Any] = {
            "points": points,
            "labels": torch.ones((points.shape[0],), dtype=torch.int64),
            "image_id": torch.tensor(sample.image_id, dtype=torch.int64),
            "filename": sample.filename,
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
            "size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
        }

        before_hw = (orig_h, orig_w)
        if self.transforms is not None:
            try:
                transformed = self.transforms(image, target)
            except TypeError:
                transformed = self.transforms(image)

            if isinstance(transformed, tuple) and len(transformed) == 2:
                image, target = transformed
            else:
                image = transformed
                new_hw = _image_hw(image)
                if new_hw is not None:
                    target["points"] = _maybe_scale_points_for_size_change(
                        target["points"], before_hw, new_hw
                    )
                    target["size"] = torch.tensor(new_hw, dtype=torch.int64)

        final_hw = _image_hw(image)
        if final_hw is not None:
            final_h, final_w = final_hw
            if "size" not in target or tuple(target["size"].tolist()) != final_hw:
                target["size"] = torch.tensor([final_h, final_w], dtype=torch.int64)
            _validate_points(target["points"], final_h, final_w, sample.filename)

        target["labels"] = torch.ones(
            (target["points"].shape[0],), dtype=torch.int64
        )
        return image, target


def collate_fn(batch: list[tuple[Any, dict[str, Any]]]) -> tuple[list[Any], list[dict[str, Any]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)
