from __future__ import annotations

import argparse
import csv
import itertools
import json
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


@dataclass
class Block:
    block_id: str
    block_index: int
    samples: list[Sample]
    density_group: str = ""
    split: str = "train"

    @property
    def start_frame(self) -> int:
        return self.samples[0].frame_number

    @property
    def end_frame(self) -> int:
        return self.samples[-1].frame_number

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def counts(self) -> list[int]:
        return [sample.point_count for sample in self.samples]

    @property
    def total_points(self) -> int:
        return int(sum(self.counts))

    @property
    def mean_count(self) -> float:
        return float(statistics.mean(self.counts))

    @property
    def median_count(self) -> float:
        return float(statistics.median(self.counts))

    @property
    def min_count(self) -> int:
        return int(min(self.counts))

    @property
    def max_count(self) -> int:
        return int(max(self.counts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a block-stratified temporal split for P2PNet data."
    )
    parser.add_argument("--images", required=True, type=Path, help="Prepared image folder")
    parser.add_argument("--gt", required=True, type=Path, help="Prepared .npy GT folder")
    parser.add_argument("--out", required=True, type=Path, help="Output split dataset folder")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Target validation ratio")
    parser.add_argument(
        "--block-size",
        type=int,
        default=30,
        help="Number of temporally sorted samples per block",
    )
    parser.add_argument("--copy", action="store_true", help="Copy files into split folders")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output contents",
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


def validate_and_load_samples(images_dir: Path, gt_dir: Path) -> list[Sample]:
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

    if empty_gt:
        preview = ", ".join(empty_gt[:10])
        raise ValueError(f"{len(empty_gt)} sample(s) have empty GT: {preview}")

    return sorted(samples, key=lambda sample: (sample.frame_number, sample.filename.casefold()))


def build_blocks(samples: list[Sample], block_size: int) -> list[Block]:
    if block_size <= 0:
        raise ValueError("--block-size must be greater than 0")

    blocks: list[Block] = []
    for start in range(0, len(samples), block_size):
        block_samples = samples[start : start + block_size]
        block_index = len(blocks)
        blocks.append(
            Block(
                block_id=f"block_{block_index:03d}",
                block_index=block_index,
                samples=block_samples,
            )
        )
    return blocks


def assign_density_groups(blocks: list[Block]) -> None:
    ranked_blocks = sorted(blocks, key=lambda block: (block.mean_count, block.block_index))
    total = len(ranked_blocks)
    for rank, block in enumerate(ranked_blocks):
        rank_fraction = rank / max(1, total)
        if rank_fraction < 1.0 / 3.0:
            block.density_group = "low"
        elif rank_fraction < 2.0 / 3.0:
            block.density_group = "medium"
        else:
            block.density_group = "high"


def sample_stats(samples: list[Sample]) -> dict[str, float | int]:
    counts = [sample.point_count for sample in samples]
    if not counts:
        return {
            "total_points": 0,
            "mean_count": 0.0,
            "median_count": 0.0,
            "min_count": 0,
            "max_count": 0,
        }

    return {
        "total_points": int(sum(counts)),
        "mean_count": float(statistics.mean(counts)),
        "median_count": float(statistics.median(counts)),
        "min_count": int(min(counts)),
        "max_count": int(max(counts)),
    }


def block_combo_stats(blocks: tuple[Block, ...]) -> tuple[int, int, float, float]:
    sample_count = sum(block.sample_count for block in blocks)
    total_points = sum(block.total_points for block in blocks)
    counts = [count for block in blocks for count in block.counts]
    mean_count = total_points / sample_count if sample_count else 0.0
    median_count = float(statistics.median(counts)) if counts else 0.0
    return sample_count, total_points, mean_count, median_count


def select_validation_blocks(blocks: list[Block], val_ratio: float) -> list[Block]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be greater than 0 and less than 1")

    total_samples = sum(block.sample_count for block in blocks)
    target_samples = total_samples * val_ratio
    all_samples = [sample for block in blocks for sample in block.samples]
    overall_stats = sample_stats(all_samples)
    overall_mean = max(float(overall_stats["mean_count"]), 1.0)
    overall_median = max(float(overall_stats["median_count"]), 1.0)
    required_groups = {block.density_group for block in blocks}

    if len(blocks) <= 24:
        best_key: tuple[float, float, float, float, int, tuple[int, ...]] | None = None
        best_blocks: tuple[Block, ...] = ()
        for size in range(1, len(blocks)):
            for combo in itertools.combinations(blocks, size):
                groups = {block.density_group for block in combo}
                if not required_groups.issubset(groups):
                    continue

                sample_count, _, mean_count, median_count = block_combo_stats(combo)
                sample_distance = abs(sample_count - target_samples) / target_samples
                mean_distance = abs(mean_count - float(overall_stats["mean_count"])) / overall_mean
                median_distance = (
                    abs(median_count - float(overall_stats["median_count"])) / overall_median
                )
                block_ids = tuple(block.block_index for block in combo)
                key = (
                    sample_distance * 2.0 + mean_distance + median_distance * 0.25,
                    sample_distance,
                    mean_distance,
                    median_distance,
                    len(combo),
                    block_ids,
                )

                if best_key is None or key < best_key:
                    best_key = key
                    best_blocks = combo

        if best_blocks:
            return sorted(best_blocks, key=lambda block: block.block_index)

    return greedy_select_validation_blocks(blocks, target_samples, required_groups)


def greedy_select_validation_blocks(
    blocks: list[Block], target_samples: float, required_groups: set[str]
) -> list[Block]:
    selected: list[Block] = []
    selected_ids: set[int] = set()
    groups = sorted(required_groups)

    for group in groups:
        group_blocks = [block for block in blocks if block.density_group == group]
        center = len(group_blocks) // 2
        block = sorted(group_blocks, key=lambda item: abs(group_blocks.index(item) - center))[0]
        selected.append(block)
        selected_ids.add(block.block_index)

    while sum(block.sample_count for block in selected) < target_samples:
        candidates = [block for block in blocks if block.block_index not in selected_ids]
        if not candidates:
            break
        block = min(
            candidates,
            key=lambda item: abs(
                sum(existing.sample_count for existing in selected)
                + item.sample_count
                - target_samples
            ),
        )
        selected.append(block)
        selected_ids.add(block.block_index)

    return sorted(selected, key=lambda block: block.block_index)


def apply_split(blocks: list[Block], val_blocks: list[Block]) -> tuple[list[Sample], list[Sample]]:
    val_block_ids = {block.block_index for block in val_blocks}
    train: list[Sample] = []
    val: list[Sample] = []
    for block in blocks:
        if block.block_index in val_block_ids:
            block.split = "val"
            val.extend(block.samples)
        else:
            block.split = "train"
            train.extend(block.samples)
    return train, val


def validate_split(train: list[Sample], val: list[Sample]) -> None:
    train_names = {sample.filename for sample in train}
    val_names = {sample.filename for sample in val}
    overlap = sorted(train_names & val_names)
    if overlap:
        preview = ", ".join(overlap[:10])
        raise ValueError(f"Train/val overlap detected: {preview}")


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


def block_range(block: Block) -> dict[str, int | str]:
    return {
        "block_id": block.block_id,
        "start_frame": block.start_frame,
        "end_frame": block.end_frame,
        "sample_count": block.sample_count,
        "density_group": block.density_group,
    }


def split_frame_ranges(blocks: list[Block], split_name: str) -> list[dict[str, int | str]]:
    return [block_range(block) for block in blocks if block.split == split_name]


def selected_val_blocks(blocks: list[Block]) -> list[dict[str, int | float | str]]:
    selected: list[dict[str, int | float | str]] = []
    for block in blocks:
        if block.split != "val":
            continue
        selected.append(
            {
                **block_range(block),
                "total_points": block.total_points,
                "mean_count": block.mean_count,
                "median_count": block.median_count,
                "min_count": block.min_count,
                "max_count": block.max_count,
            }
        )
    return selected


def split_summary(
    samples: list[Sample],
    train: list[Sample],
    val: list[Sample],
    blocks: list[Block],
    val_ratio: float,
    block_size: int,
) -> dict[str, object]:
    train_stats = sample_stats(train)
    val_stats = sample_stats(val)
    return {
        "split_mode": "block_stratified",
        "block_size": block_size,
        "total_blocks": len(blocks),
        "total_samples": len(samples),
        "train_samples": len(train),
        "val_samples": len(val),
        "target_val_ratio": val_ratio,
        "actual_val_ratio": len(val) / len(samples),
        "train_total_points": train_stats["total_points"],
        "val_total_points": val_stats["total_points"],
        "train_mean_count": train_stats["mean_count"],
        "val_mean_count": val_stats["mean_count"],
        "train_median_count": train_stats["median_count"],
        "val_median_count": val_stats["median_count"],
        "train_min_count": train_stats["min_count"],
        "val_min_count": val_stats["min_count"],
        "train_max_count": train_stats["max_count"],
        "val_max_count": val_stats["max_count"],
        "train_frame_ranges": split_frame_ranges(blocks, "train"),
        "val_frame_ranges": split_frame_ranges(blocks, "val"),
        "selected_val_blocks": selected_val_blocks(blocks),
    }


def write_block_summary(blocks: list[Block], path: Path) -> None:
    fieldnames = [
        "block_id",
        "block_index",
        "split",
        "density_group",
        "start_frame",
        "end_frame",
        "sample_count",
        "total_points",
        "mean_count",
        "median_count",
        "min_count",
        "max_count",
        "first_filename",
        "last_filename",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for block in blocks:
            writer.writerow(
                {
                    "block_id": block.block_id,
                    "block_index": block.block_index,
                    "split": block.split,
                    "density_group": block.density_group,
                    "start_frame": block.start_frame,
                    "end_frame": block.end_frame,
                    "sample_count": block.sample_count,
                    "total_points": block.total_points,
                    "mean_count": f"{block.mean_count:.6f}",
                    "median_count": f"{block.median_count:.6f}",
                    "min_count": block.min_count,
                    "max_count": block.max_count,
                    "first_filename": block.samples[0].filename,
                    "last_filename": block.samples[-1].filename,
                }
            )


def print_distribution(name: str, samples: list[Sample]) -> None:
    stats = sample_stats(samples)
    print(
        f"  {name}: "
        f"samples={len(samples)}, "
        f"points={stats['total_points']}, "
        f"min={stats['min_count']}, "
        f"max={stats['max_count']}, "
        f"mean={stats['mean_count']:.3f}, "
        f"median={stats['median_count']:.3f}"
    )


def print_selected_blocks(blocks: list[Block]) -> None:
    print("  selected val blocks:")
    for block in blocks:
        if block.split != "val":
            continue
        print(
            f"    {block.block_id} "
            f"{block.start_frame}-{block.end_frame} "
            f"{block.density_group} "
            f"n={block.sample_count} "
            f"mean={block.mean_count:.2f}"
        )


def main() -> int:
    args = parse_args()

    print("P2PNet block-stratified split")
    print(f"  Images: {args.images}")
    print(f"  GT: {args.gt}")
    print(f"  Output: {args.out}")
    print(f"  Val ratio: {args.val_ratio}")
    print(f"  Block size: {args.block_size}")
    print(f"  Copy files: {args.copy}")
    print(f"  Overwrite: {args.overwrite}")

    if not args.images.exists():
        print(f"ERROR: images folder not found: {args.images}", file=sys.stderr)
        return 1
    if not args.gt.exists():
        print(f"ERROR: gt folder not found: {args.gt}", file=sys.stderr)
        return 1

    try:
        samples = validate_and_load_samples(args.images, args.gt)
        blocks = build_blocks(samples, args.block_size)
        assign_density_groups(blocks)
        val_blocks = select_validation_blocks(blocks, args.val_ratio)
        train, val = apply_split(blocks, val_blocks)
        validate_split(train, val)

        prepare_output(args.out, args.overwrite)
        write_lines(args.out / "train.txt", [sample.filename for sample in train])
        write_lines(args.out / "val.txt", [sample.filename for sample in val])
        write_block_summary(blocks, args.out / "block_summary.csv")

        if args.copy:
            copy_split(train, args.out, "train")
            copy_split(val, args.out, "val")

        summary = split_summary(samples, train, val, blocks, args.val_ratio, args.block_size)
        (args.out / "split_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        print("")
        print("Split summary")
        print_distribution("train", train)
        print_distribution("val", val)
        print_selected_blocks(blocks)
        print(f"  train.txt: {args.out / 'train.txt'}")
        print(f"  val.txt: {args.out / 'val.txt'}")
        print(f"  block_summary.csv: {args.out / 'block_summary.csv'}")
        print(f"  split_summary.json: {args.out / 'split_summary.json'}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
