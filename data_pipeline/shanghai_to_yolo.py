"""
Domain adaptation script for converting ShanghaiTech-style head annotations
into YOLO bounding boxes with size driven by local crowd density.

The algorithm:
1. Read .mat annotation files with head center coordinates.
2. For each image, fit k-Nearest Neighbors on head points.
3. Estimate local scale as the average distance to the k nearest neighbors.
4. Invert this scale to get a bounding box side length
   (denser areas -> smaller boxes, sparse areas -> larger boxes).
5. Export YOLO txt labels with normalized coordinates.

Usage
-----
python data_pipeline/shanghai_to_yolo.py \
    --mat_dir path/to/annotations \
    --image_dir path/to/images \
    --output_dir path/to/yolo_labels
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors

try:
    import cv2  # type: ignore
except ImportError as e:  # pragma: no cover - runtime import check
    raise ImportError(
        "OpenCV (opencv-python) is required for reading image sizes. "
        "Install it via `pip install opencv-python`."
    ) from e


def load_shanghaitech_points(mat_path: Path) -> np.ndarray:
    """
    Load head center points from a ShanghaiTech-style .mat file.

    Many ShanghaiTech annotations store locations under:
        mat['image_info'][0, 0]['location'][0, 0]  # (N, 2) array of (x, y)

    This function attempts that layout first and falls back to a few
    common alternatives. It returns an (N, 2) float array.
    """
    mat = loadmat(str(mat_path))

    # Common ShanghaiTech structure
    if "image_info" in mat:
        info = mat["image_info"]
        # Typical nesting: image_info[0, 0]['location'][0, 0]
        try:
            loc = info[0, 0]["location"][0, 0]
            pts = np.asarray(loc, dtype=np.float32)
            if pts.ndim == 2 and pts.shape[1] == 2:
                return pts
        except Exception:
            pass

    # Fallbacks: search for any (N, 2) array
    for v in mat.values():
        arr = np.asarray(v)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr.astype(np.float32)

    raise ValueError(f"Could not infer head locations from {mat_path}")


def compute_local_scales(points: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Compute a local scale for each point as the mean distance
    to its k nearest neighbors (excluding itself).

    Parameters
    ----------
    points : (N, 2) array of (x, y) coordinates in pixels.
    k : int
        Number of nearest neighbors to consider.

    Returns
    -------
    scales : (N,) array of mean distances.
    """
    if points.shape[0] < 2:
        # Degenerate case: single head; assign arbitrary scale
        return np.full(points.shape[0], 50.0, dtype=np.float32)

    # Fit KNN on all points
    n_neighbors = min(k + 1, len(points))  # include self
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="kd_tree")
    knn.fit(points)
    distances, _ = knn.kneighbors(points)

    # distances[:, 0] is distance to self (0); ignore it
    neighbor_distances = distances[:, 1:]  # shape (N, k')
    scales = neighbor_distances.mean(axis=1)
    return scales.astype(np.float32)


def scales_to_box_sizes(
    scales: np.ndarray,
    image_size: Tuple[int, int],
    density_factor: float = 0.15,
    min_rel: float = 0.01,
    max_rel: float = 0.25,
) -> np.ndarray:
    """
    Convert local scales (average neighbor distance) to bounding box side lengths.

    We model box side length as inversely proportional to the local scale:

        box_side ~ density_factor * max_dim / (scale + eps)

    and then clip to [min_rel * max_dim, max_rel * max_dim].

    Parameters
    ----------
    scales : (N,) array
        Local neighbor distances (pixels).
    image_size : (height, width)
        Image dimensions in pixels.
    density_factor : float
        Global scaling factor controlling how aggressively density shrinks boxes.
    min_rel, max_rel : float
        Min and max relative size of box side w.r.t max(image_height, image_width).

    Returns
    -------
    side_lengths : (N,) array of side lengths in pixels.
    """
    h, w = image_size
    max_dim = float(max(h, w))
    eps = 1e-6

    raw = density_factor * max_dim / (scales + eps)
    side_min = min_rel * max_dim
    side_max = max_rel * max_dim
    side_lengths = np.clip(raw, side_min, side_max)
    return side_lengths.astype(np.float32)


def points_to_yolo_boxes(
    points: np.ndarray,
    image_size: Tuple[int, int],
    k: int = 3,
) -> np.ndarray:
    """
    Convert head centers to YOLO-format bounding boxes using local density.

    Parameters
    ----------
    points : (N, 2) array
        Head centers in pixel coordinates (x, y).
    image_size : (height, width)
        Image size in pixels.
    k : int
        Number of neighbors to use for density estimation.

    Returns
    -------
    yolo_boxes : (N, 5) array
        Each row: [class_id, x_center, y_center, width, height] in [0, 1] coords.
    """
    if points.size == 0:
        return np.empty((0, 5), dtype=np.float32)

    h, w = image_size
    scales = compute_local_scales(points, k=k)
    side_lengths = scales_to_box_sizes(scales, image_size=image_size)

    # Convert to normalized YOLO format
    cx = points[:, 0] / float(w)
    cy = points[:, 1] / float(h)
    bw = side_lengths / float(w)
    bh = side_lengths / float(h)

    # Single class (e.g., head = 0)
    class_ids = np.zeros_like(cx, dtype=np.float32)
    yolo = np.stack([class_ids, cx, cy, bw, bh], axis=1)
    return yolo


def write_yolo_labels(label_path: Path, boxes: np.ndarray) -> None:
    """
    Write YOLO labels to disk.

    Parameters
    ----------
    label_path : Path
        Output .txt path.
    boxes : (N, 5) array
        YOLO boxes [class, x_center, y_center, width, height].
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as f:
        for row in boxes:
            cls, x, y, w, h = row.tolist()
            f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def infer_image_size(image_path: Path) -> Tuple[int, int]:
    """Return (height, width) for an image file using OpenCV."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    h, w = img.shape[:2]
    return int(h), int(w)


def process_sample(
    mat_path: Path,
    image_dir: Path,
    output_dir: Path,
    image_ext: str = ".jpg",
    k: int = 3,
) -> None:
    """
    Process a single .mat annotation file and write its YOLO label file.

    The script assumes that the image name shares the same stem as the .mat file
    (e.g., IMG_1.mat -> IMG_1.jpg).
    """
    points = load_shanghaitech_points(mat_path)
    image_path = image_dir / (mat_path.stem + image_ext)
    image_size = infer_image_size(image_path)

    boxes = points_to_yolo_boxes(points, image_size=image_size, k=k)
    label_path = output_dir / f"{mat_path.stem}.txt"
    write_yolo_labels(label_path, boxes)


def iter_mat_files(mat_dir: Path) -> Iterable[Path]:
    """Yield all .mat files in a directory (non-recursive)."""
    for p in sorted(mat_dir.glob("*.mat")):
        if p.is_file():
            yield p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ShanghaiTech .mat head annotations into YOLO bounding boxes "
            "using a k-NN-based local density model."
        )
    )
    parser.add_argument(
        "--mat_dir",
        type=str,
        required=True,
        help="Directory containing .mat annotation files.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing corresponding crowd images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where YOLO .txt labels will be written.",
    )
    parser.add_argument(
        "--image_ext",
        type=str,
        default=".jpg",
        help="File extension of images (default: .jpg).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of nearest neighbors for local density (default: 3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mat_dir = Path(args.mat_dir)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    if not mat_dir.is_dir():
        raise NotADirectoryError(f"mat_dir does not exist: {mat_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"image_dir does not exist: {image_dir}")

    os.makedirs(output_dir, exist_ok=True)

    mat_files = list(iter_mat_files(mat_dir))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {mat_dir}")

    for mat_path in mat_files:
        process_sample(
            mat_path=mat_path,
            image_dir=image_dir,
            output_dir=output_dir,
            image_ext=args.image_ext,
            k=args.k,
        )

    print(f"Converted {len(mat_files)} annotation files to YOLO format in {output_dir}")


if __name__ == "__main__":
    main()

