from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NUMERIC_GROUP_RE = re.compile(r"(\d+)")


@dataclass(frozen=True)
class Sample:
    filename: str
    image_path: Path
    gt_path: Path
    frame_number: int
    point_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a temporal train/val split for a prepared P2PNet dataset."
    )
    parser.add_argument("--images", required=True, type=Path, help="Prepared image folder")
    parser.add_argument("--gt", required=True, type=Path, help="Prepared .npy GT folder")
    parser.add_argument("--out", required=True, type=Path, help="Output split dataset folder")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of temporally sorted samples to put in validation",
    )
    parser.add_argument(
        "--split-mode",
        choices=["temporal_tail"],
        default="temporal_tail",
        help="temporal_tail uses the final val-ratio of sorted frames for validation",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy image and GT files into the output split folders",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output contents",
    )
    parser.add_argument(
        "--allow-empty-gt",
        action="store_true",
        help="Allow samples whose .npy point file has zero points",
    )
    return parser.parse_args()


def extract_frame_number(filename: str) -> int:
    matches = NUMERIC_GROUP_RE.findall(Path(filename).stem)
    if not matches:
        raise ValueError(f"Could not extract a frame number from {filename!r}")
    return int(matches[-1])


def image_files(images_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name.casefold(),
    )


def load_point_count(gt_path: Path) -> int:
    try:
        points = np.load(gt_path)
    except Exception as exc:
        raise ValueError(f"Failed to load GT file {gt_path}: {exc}") from exc

    if points.size == 0:
        return 0
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"GT file {gt_path} has unexpected shape {tuple(points.shape)}")
    return int(points.shape[0])


def validate_and_load_samples(
    images_dir: Path, gt_dir: Path, allow_empty_gt: bool
) -> list[Sample]:
    images = image_files(images_dir)
    if not images:
        raise ValueError(f"No image files found in {images_dir}")

    image_stems = {path.stem for path in images}
    gt_files = sorted(gt_dir.glob("*.npy"), key=lambda path: path.name.casefold())
    gt_stems = {path.stem for path in gt_files}

    missing_gt = sorted(image_stems - gt_stems)
    extra_gt = sorted(gt_stems - image_stems)
    if missing_gt:
        preview = ", ".join(missing_gt[:10])
        raise ValueError(
            f"{len(missing_gt)} image(s) are missing matching .npy files: {preview}"
        )
    if extra_gt:
        preview = ", ".join(extra_gt[:10])
        raise ValueError(
            f"{len(extra_gt)} .npy file(s) have no matching image: {preview}"
        )

    samples: list[Sample] = []
    empty_gt: list[str] = []
    for image_path in images:
        gt_path = gt_dir / f"{image_path.stem}.npy"
        point_count = load_point_count(gt_path)
        if point_count == 0:
            empty_gt.append(image_path.name)
        samples.append(
            Sample(
                filename=image_path.name,
                image_path=image_path,
                gt_path=gt_path,
                frame_number=extract_frame_number(image_path.name),
                point_count=point_count,
            )
        )

    if empty_gt and not allow_empty_gt:
        preview = ", ".join(empty_gt[:10])
        raise ValueError(
            f"{len(empty_gt)} sample(s) have empty GT. Pass --allow-empty-gt to allow: {preview}"
        )

    return sorted(samples, key=lambda sample: (sample.frame_number, sample.filename.casefold()))


def split_temporal_tail(samples: list[Sample], val_ratio: float) -> tuple[list[Sample], list[Sample]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be greater than 0 and less than 1")

    val_count = int(round(len(samples) * val_ratio))
    val_count = max(1, min(len(samples) - 1, val_count))
    train_count = len(samples) - val_count
    return samples[:train_count], samples[train_count:]


def count_stats(samples: list[Sample]) -> dict[str, float | int]:
    counts = [sample.point_count for sample in samples]
    if not counts:
        return {
            "min_count": 0,
            "max_count": 0,
            "mean_count": 0.0,
            "median_count": 0.0,
            "total_points": 0,
        }

    return {
        "min_count": min(counts),
        "max_count": max(counts),
        "mean_count": float(statistics.mean(counts)),
        "median_count": float(statistics.median(counts)),
        "total_points": int(sum(counts)),
    }


def frame_range(samples: list[Sample]) -> list[int | None]:
    if not samples:
        return [None, None]
    return [samples[0].frame_number, samples[-1].frame_number]


def validate_split(train: list[Sample], val: list[Sample]) -> None:
    train_names = {sample.filename for sample in train}
    val_names = {sample.filename for sample in val}
    overlap = sorted(train_names & val_names)
    if overlap:
        preview = ", ".join(overlap[:10])
        raise ValueError(f"Train/val overlap detected: {preview}")

    if train and val and train[-1].frame_number >= val[0].frame_number:
        raise ValueError(
            "Temporal split is not ordered: train tail frame is not before val head frame"
        )


def clear_output_contents(out_dir: Path) -> None:
    for path in sorted(out_dir.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file() or path.is_symlink():
            try:
                path.unlink()
            except PermissionError:
                pass
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def prepare_output(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {out_dir}. Pass --overwrite to replace it."
            )
        clear_output_contents(out_dir)

    for subdir in (
        out_dir / "images" / "train",
        out_dir / "images" / "val",
        out_dir / "gt" / "train",
        out_dir / "gt" / "val",
    ):
        subdir.mkdir(parents=True, exist_ok=True)


def write_lines(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def copy_split(samples: list[Sample], out_dir: Path, split_name: str) -> None:
    image_out = out_dir / "images" / split_name
    gt_out = out_dir / "gt" / split_name
    for sample in samples:
        shutil.copy2(sample.image_path, image_out / sample.image_path.name)
        shutil.copy2(sample.gt_path, gt_out / sample.gt_path.name)


def split_summary(
    samples: list[Sample],
    train: list[Sample],
    val: list[Sample],
    val_ratio: float,
    split_mode: str,
) -> dict[str, object]:
    train_stats = count_stats(train)
    val_stats = count_stats(val)
    all_stats = count_stats(samples)
    return {
        "split_mode": split_mode,
        "total_samples": len(samples),
        "train_samples": len(train),
        "val_samples": len(val),
        "val_ratio": val_ratio,
        "actual_val_ratio": len(val) / len(samples),
        "total_points": all_stats["total_points"],
        "train_total_points": train_stats["total_points"],
        "val_total_points": val_stats["total_points"],
        "train_count_min": train_stats["min_count"],
        "train_count_max": train_stats["max_count"],
        "train_count_mean": train_stats["mean_count"],
        "train_count_median": train_stats["median_count"],
        "val_count_min": val_stats["min_count"],
        "val_count_max": val_stats["max_count"],
        "val_count_mean": val_stats["mean_count"],
        "val_count_median": val_stats["median_count"],
        "train_frame_range": frame_range(train),
        "val_frame_range": frame_range(val),
    }


def print_distribution(name: str, samples: list[Sample]) -> None:
    stats = count_stats(samples)
    print(
        f"  {name}: "
        f"samples={len(samples)}, "
        f"points={stats['total_points']}, "
        f"min={stats['min_count']}, "
        f"max={stats['max_count']}, "
        f"mean={stats['mean_count']:.3f}, "
        f"median={stats['median_count']:.3f}, "
        f"frame_range={frame_range(samples)}"
    )


def main() -> int:
    args = parse_args()

    print("P2PNet temporal split")
    print(f"  Images: {args.images}")
    print(f"  GT: {args.gt}")
    print(f"  Output: {args.out}")
    print(f"  Val ratio: {args.val_ratio}")
    print(f"  Split mode: {args.split_mode}")
    print(f"  Copy files: {args.copy}")
    print(f"  Overwrite: {args.overwrite}")

    if not args.images.exists():
        print(f"ERROR: images folder not found: {args.images}", file=sys.stderr)
        return 1
    if not args.gt.exists():
        print(f"ERROR: gt folder not found: {args.gt}", file=sys.stderr)
        return 1

    try:
        samples = validate_and_load_samples(args.images, args.gt, args.allow_empty_gt)
        if args.split_mode == "temporal_tail":
            train, val = split_temporal_tail(samples, args.val_ratio)
        else:
            raise ValueError(f"Unsupported split mode: {args.split_mode}")

        validate_split(train, val)

        prepare_output(args.out, args.overwrite)
        write_lines(args.out / "train.txt", [sample.filename for sample in train])
        write_lines(args.out / "val.txt", [sample.filename for sample in val])

        if args.copy:
            copy_split(train, args.out, "train")
            copy_split(val, args.out, "val")

        summary = split_summary(samples, train, val, args.val_ratio, args.split_mode)
        (args.out / "split_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        print("")
        print("Split summary")
        print_distribution("train", train)
        print_distribution("val", val)
        print(f"  train.txt: {args.out / 'train.txt'}")
        print(f"  val.txt: {args.out / 'val.txt'}")
        print(f"  split_summary.json: {args.out / 'split_summary.json'}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
