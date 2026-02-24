"""
Dataset Split Utility
======================

Splits a custom image dataset into train/val/test subsets.
Copies image files and matching annotation files into structured output directories.
Saves a JSON manifest listing all files per split.

Usage:
    python data_pipeline/split_dataset.py --data-dir data/custom/ --train 0.7 --val 0.15 --test 0.15
    python data_pipeline/split_dataset.py \\
        --data-dir data/custom/ \\
        --images-subdir images/ \\
        --annotations-subdir ground_truth/ \\
        --ann-prefix GT_ --ann-ext .mat \\
        --output data/custom_split/ \\
        --seed 42
"""

import argparse
import json
import logging
import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def find_image_annotation_pairs(
    data_dir: str,
    images_subdir: str = "images",
    annotations_subdir: str = "ground_truth",
    ann_prefix: str = "GT_",
    ann_ext: str = ".mat",
) -> List[Tuple[Path, Optional[Path]]]:
    """
    Find matching image–annotation file pairs.

    Parameters
    ----------
    data_dir : str
        Root data directory.
    images_subdir : str
        Subdirectory containing images (relative to data_dir).
    annotations_subdir : str
        Subdirectory containing annotations.
    ann_prefix : str
        Annotation filename prefix (e.g., 'GT_' → GT_IMG_001.mat).
    ann_ext : str
        Annotation file extension.

    Returns
    -------
    list of (image_path, annotation_path_or_None) tuples
    """
    img_dir = Path(data_dir) / images_subdir
    ann_dir = Path(data_dir) / annotations_subdir

    if not img_dir.exists():
        log.error(f"Image directory not found: {img_dir}")
        return []

    img_paths = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    pairs = []
    missing_ann = 0

    for img_path in img_paths:
        stem = img_path.stem
        ann_path = ann_dir / f"{ann_prefix}{stem}{ann_ext}"

        if not ann_path.exists():
            # Try without prefix
            ann_path_no_prefix = ann_dir / f"{stem}{ann_ext}"
            if ann_path_no_prefix.exists():
                ann_path = ann_path_no_prefix
            else:
                log.debug(f"No annotation for {img_path.name}")
                missing_ann += 1
                ann_path = None

        pairs.append((img_path, ann_path))

    log.info(f"Found {len(pairs)} images ({missing_ann} without annotations)")
    return pairs


def stratified_split(
    pairs: List[Tuple[Path, Optional[Path]]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """
    Randomly shuffle and split pairs into train/val/test.

    Parameters
    ----------
    pairs : list
    train_ratio, val_ratio, test_ratio : float
        Must sum to 1.0.
    seed : int

    Returns
    -------
    tuple: (train_pairs, val_pairs, test_pairs)
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total:.3f}")

    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_pairs = shuffled[:n_train]
    val_pairs = shuffled[n_train:n_train + n_val]
    test_pairs = shuffled[n_train + n_val:]

    log.info(f"Split: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
    return train_pairs, val_pairs, test_pairs


def copy_split(
    pairs: List[Tuple[Path, Optional[Path]]],
    output_dir: Path,
    split_name: str,
    images_subdir: str = "images",
    annotations_subdir: str = "ground_truth",
    ann_prefix: str = "GT_",
) -> List[dict]:
    """
    Copy image and annotation files to split output directory.

    Parameters
    ----------
    pairs : list of (img_path, ann_path) tuples
    output_dir : Path
    split_name : str
        'train', 'val', or 'test'
    images_subdir : str
    annotations_subdir : str
    ann_prefix : str

    Returns
    -------
    list of dicts describing copied files (for manifest)
    """
    img_out = output_dir / split_name / images_subdir
    ann_out = output_dir / split_name / annotations_subdir
    img_out.mkdir(parents=True, exist_ok=True)
    ann_out.mkdir(parents=True, exist_ok=True)

    manifest_entries = []

    for img_path, ann_path in pairs:
        dst_img = img_out / img_path.name
        shutil.copy2(img_path, dst_img)

        entry = {
            "image": str(dst_img.relative_to(output_dir)),
            "annotation": None,
        }

        if ann_path is not None and ann_path.exists():
            dst_ann = ann_out / ann_path.name
            shutil.copy2(ann_path, dst_ann)
            entry["annotation"] = str(dst_ann.relative_to(output_dir))

        manifest_entries.append(entry)

    log.info(f"Copied {len(pairs)} files to {output_dir / split_name}")
    return manifest_entries


def create_split_manifest(splits: dict, output_path: str) -> None:
    """
    Save a JSON manifest listing all files in each split.

    Parameters
    ----------
    splits : dict
        {'train': [...], 'val': [...], 'test': [...]}
    output_path : str
        Destination .json path.
    """
    manifest = {
        "splits": {
            name: {"count": len(entries), "files": entries}
            for name, entries in splits.items()
        },
        "total": sum(len(e) for e in splits.values()),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Manifest saved: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Split dataset into train/val/test with annotation pairing"
    )
    parser.add_argument("--data-dir", required=True,
                        help="Root data directory containing images/ and ground_truth/")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: <data-dir>_split)")
    parser.add_argument("--train", type=float, default=0.70,
                        help="Training set ratio (default: 0.70)")
    parser.add_argument("--val", type=float, default=0.15,
                        help="Validation set ratio (default: 0.15)")
    parser.add_argument("--test", type=float, default=0.15,
                        help="Test set ratio (default: 0.15)")
    parser.add_argument("--images-subdir", default="images",
                        help="Images subdirectory (default: images)")
    parser.add_argument("--annotations-subdir", default="ground_truth",
                        help="Annotations subdirectory (default: ground_truth)")
    parser.add_argument("--ann-prefix", default="GT_",
                        help="Annotation filename prefix (default: GT_)")
    parser.add_argument("--ann-ext", default=".mat",
                        help="Annotation file extension (default: .mat)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show split sizes without copying files")
    return parser.parse_args()


def main() -> None:
    """Run dataset split pipeline."""
    args = parse_args()

    total_ratio = args.train + args.val + args.test
    if abs(total_ratio - 1.0) > 0.01:
        raise SystemExit(
            f"ERROR: Ratios must sum to 1.0 (got {total_ratio:.3f}). "
            f"Adjust --train, --val, --test."
        )

    output_dir = Path(args.output) if args.output else Path(args.data_dir + "_split")

    pairs = find_image_annotation_pairs(
        args.data_dir,
        images_subdir=args.images_subdir,
        annotations_subdir=args.annotations_subdir,
        ann_prefix=args.ann_prefix,
        ann_ext=args.ann_ext,
    )

    if not pairs:
        raise SystemExit("No image-annotation pairs found. Check --data-dir and subdirectory args.")

    train_pairs, val_pairs, test_pairs = stratified_split(
        pairs, args.train, args.val, args.test, args.seed
    )

    if args.dry_run:
        log.info("DRY RUN — no files copied")
        log.info(f"  Train: {len(train_pairs)} images")
        log.info(f"  Val:   {len(val_pairs)} images")
        log.info(f"  Test:  {len(test_pairs)} images")
        log.info(f"  Output: {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    split_manifests = {}
    for split_name, split_pairs in [
        ("train", train_pairs), ("val", val_pairs), ("test", test_pairs)
    ]:
        entries = copy_split(
            split_pairs, output_dir, split_name,
            args.images_subdir, args.annotations_subdir, args.ann_prefix
        )
        split_manifests[split_name] = entries

    manifest_path = output_dir / "split_manifest.json"
    create_split_manifest(split_manifests, str(manifest_path))

    log.info(f"\nDataset split complete:")
    log.info(f"  Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")
    log.info(f"  Output: {output_dir}")
    log.info(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
