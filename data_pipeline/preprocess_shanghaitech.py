"""
ShanghaiTech Dataset Preprocessor
====================================

Converts .mat point annotation files to Gaussian kernel density maps
stored in HDF5 format (.h5), suitable for CSRNet training.

Usage:
    python data_pipeline/preprocess_shanghaitech.py --part B --split train
    python data_pipeline/preprocess_shanghaitech.py --part A --split all
    python data_pipeline/preprocess_shanghaitech.py --part B --split test --no-adaptive
"""

import argparse
import logging
from pathlib import Path

import numpy as np

try:
    from scipy.io import loadmat
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import KDTree
    import h5py
except ImportError as e:
    raise SystemExit(f"Missing dependency: {e}. Run: pip install scipy h5py")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def load_mat_annotations(mat_path: str) -> np.ndarray:
    """
    Load point annotations from a ShanghaiTech .mat ground truth file.

    Handles two .mat format variants:
    - Old format: image_info[0,0][0,0][0] — array of (x, y) points
    - New format: annPoints — direct Nx2 array

    Parameters
    ----------
    mat_path : str
        Path to .mat file.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with columns [x, y].
        Returns empty array (0, 2) if file is unreadable.
    """
    try:
        mat = loadmat(mat_path)
    except Exception as e:
        log.warning(f"Could not load {mat_path}: {e}")
        return np.zeros((0, 2), dtype=np.float32)

    # Try ShanghaiTech Part A/B standard format
    if "image_info" in mat:
        try:
            points = mat["image_info"][0, 0][0, 0][0]
            return np.array(points, dtype=np.float32)
        except (IndexError, KeyError, TypeError):
            pass

    # Try alternative key names
    for key in ["annPoints", "ann_points", "points", "gt_points"]:
        if key in mat:
            return np.array(mat[key], dtype=np.float32)

    # Last resort: return largest 2D array value
    for val in mat.values():
        if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[1] in [2, 3]:
            return val[:, :2].astype(np.float32)

    log.warning(f"No point annotations found in {mat_path}")
    return np.zeros((0, 2), dtype=np.float32)


def gaussian_density_map(
    img_shape: tuple,
    points: np.ndarray,
    adaptive: bool = True,
    k_neighbors: int = 3,
    fixed_sigma: float = 15.0,
    min_sigma: float = 1.0,
    max_sigma: float = 25.0,
) -> np.ndarray:
    """
    Convert point annotations to a Gaussian kernel density map.

    If adaptive=True, sigma for each point is set proportional to the
    average distance to its k nearest neighbors (Gaussian-adaptive kernel).
    If adaptive=False, a fixed sigma is applied to all points.

    Parameters
    ----------
    img_shape : tuple
        (H, W) image dimensions.
    points : np.ndarray
        (N, 2) array of (x, y) coordinates.
    adaptive : bool
        Use adaptive sigma based on k-NN distance.
    k_neighbors : int
        Number of nearest neighbors for adaptive sigma.
    fixed_sigma : float
        Fixed sigma to use when adaptive=False.
    min_sigma, max_sigma : float
        Clamping range for adaptive sigma.

    Returns
    -------
    np.ndarray
        Float32 density map of shape (H, W).
    """
    H, W = img_shape[:2]
    density = np.zeros((H, W), dtype=np.float32)

    if len(points) == 0:
        return density

    # Clip points to image bounds
    xs = np.clip(points[:, 0], 0, W - 1)
    ys = np.clip(points[:, 1], 0, H - 1)
    pts = np.column_stack([xs, ys])

    if adaptive and len(pts) >= k_neighbors + 1:
        # Build KD-tree in (x, y) space
        tree = KDTree(pts)
        dists, _ = tree.query(pts, k=k_neighbors + 1)  # +1 includes self
        avg_dists = dists[:, 1:].mean(axis=1)  # exclude self

        for i, (x, y) in enumerate(pts):
            sigma = float(np.clip(avg_dists[i] * 0.3, min_sigma, max_sigma))
            px, py = int(round(x)), int(round(y))
            density[py, px] += 1.0
            # Apply point-wise Gaussian (approximate by placing on integer grid)
            if sigma > 1.0:
                # Use small local Gaussian for efficiency
                r = int(sigma * 3)
                y0, y1 = max(0, py - r), min(H, py + r + 1)
                x0, x1 = max(0, px - r), min(W, px + r + 1)
                if y1 > y0 and x1 > x0:
                    patch = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
                    cy = py - y0
                    cx = px - x0
                    if 0 <= cy < patch.shape[0] and 0 <= cx < patch.shape[1]:
                        patch[cy, cx] = 1.0
                    density[y0:y1, x0:x1] += gaussian_filter(patch, sigma=sigma)
                    density[py, px] -= 1.0  # Remove the initial spike
    else:
        # Non-adaptive: place all points then apply single Gaussian
        for x, y in pts:
            px, py = int(round(x)), int(round(y))
            if 0 <= py < H and 0 <= px < W:
                density[py, px] += 1.0
        density = gaussian_filter(density, sigma=fixed_sigma)

    return density


def process_split(
    data_dir: str,
    output_dir: str,
    split: str,
    part: str,
    adaptive: bool = True,
) -> int:
    """
    Process all images in a dataset split, saving density maps as .h5 files.

    Parameters
    ----------
    data_dir : str
        Root of ShanghaiTech dataset (e.g., data/shanghaitech/).
    output_dir : str
        Where to save .h5 density maps.
    split : str
        'train' or 'test'.
    part : str
        'A' or 'B'.
    adaptive : bool

    Returns
    -------
    int : Number of images processed.
    """
    split_name = f"train_data" if split == "train" else "test_data"
    img_dir = Path(data_dir) / f"part_{part}_final" / split_name / "images"
    gt_dir = Path(data_dir) / f"part_{part}_final" / split_name / "ground_truth"
    out_dir = Path(output_dir) / f"part_{part}_final" / split_name / "ground_truth_h5"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        log.error(f"Image directory not found: {img_dir}")
        return 0

    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not img_paths:
        log.warning(f"No images found in {img_dir}")
        return 0

    log.info(f"Processing Part {part} {split}: {len(img_paths)} images...")
    processed = 0
    counts = []

    for img_path in img_paths:
        stem = img_path.stem
        mat_path = gt_dir / f"GT_{stem}.mat"

        if not mat_path.exists():
            log.warning(f"No GT file for {img_path.name}: skipping")
            continue

        # Load image shape
        try:
            from PIL import Image
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception:
            H, W = 768, 1024  # Fallback shape

        points = load_mat_annotations(str(mat_path))
        density = gaussian_density_map((H, W), points, adaptive=adaptive)

        gt_count = len(points)
        density_sum = float(density.sum())
        counts.append(gt_count)

        h5_path = out_dir / f"{stem}.h5"
        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("density", data=density, compression="gzip")
            hf.attrs["gt_count"] = gt_count
            hf.attrs["image_file"] = img_path.name

        processed += 1

        if processed % 50 == 0:
            log.info(f"  {processed}/{len(img_paths)} processed, avg count: {np.mean(counts):.1f}")

    if counts:
        log.info(f"Part {part} {split} complete: {processed} images, "
                 f"counts: min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.1f}")
    return processed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ShanghaiTech .mat → HDF5 density map preprocessor"
    )
    parser.add_argument("--part", choices=["A", "B", "both"], default="B",
                        help="Dataset part: A, B, or both (default: B)")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all",
                        help="Split to process: train, test, or all (default: all)")
    parser.add_argument("--data-dir", default="data/shanghaitech/",
                        help="ShanghaiTech dataset root (default: data/shanghaitech/)")
    parser.add_argument("--output-dir", default="data/shanghaitech/",
                        help="Output root for .h5 files (default: same as data-dir)")
    parser.add_argument("--no-adaptive", action="store_true",
                        help="Use fixed-sigma Gaussian instead of adaptive k-NN")
    return parser.parse_args()


def main() -> None:
    """Run preprocessing pipeline."""
    args = parse_args()
    adaptive = not args.no_adaptive

    parts = ["A", "B"] if args.part == "both" else [args.part]
    splits = ["train", "test"] if args.split == "all" else [args.split]

    total = 0
    for part in parts:
        for split in splits:
            n = process_split(args.data_dir, args.output_dir, split, part, adaptive)
            total += n

    log.info(f"Preprocessing complete: {total} density maps created.")


if __name__ == "__main__":
    main()
