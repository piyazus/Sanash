"""
VIA JSON → ShanghaiTech .mat Annotation Converter
===================================================

Converts point annotations exported from the VGG Image Annotator (VIA)
tool into ShanghaiTech-compatible .mat format for CSRNet training.

VIA JSON structure:
    {
      "filename_key": {
        "filename": "IMG_001.jpg",
        "regions": [
          {"shape_attributes": {"name": "point", "cx": 120, "cy": 200}},
          {"shape_attributes": {"name": "circle", "cx": 300, "cy": 150, "r": 5}}
        ]
      }
    }

Usage:
    python data_pipeline/convert_annotations.py \\
        --input annotations/via_export.json \\
        --output data/custom/ground_truth/ \\
        --images data/custom/images/
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

try:
    from scipy.io import savemat
except ImportError:
    raise SystemExit("scipy required: pip install scipy")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def load_via_json(json_path: str) -> dict:
    """
    Load and validate VIA-exported JSON annotation file.

    Parameters
    ----------
    json_path : str

    Returns
    -------
    dict : VIA annotations keyed by filename_key
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # VIA 2.x wraps annotations under "_via_img_metadata"
    if "_via_img_metadata" in data:
        data = data["_via_img_metadata"]

    log.info(f"Loaded {len(data)} annotated images from {json_path}")
    return data


def via_to_points(regions: list) -> np.ndarray:
    """
    Extract (x, y) point coordinates from VIA region list.

    Supported shape types:
    - 'point': uses cx, cy
    - 'circle': uses cx, cy (centre)
    - 'rect': uses x + w/2, y + h/2 (centre of rectangle)
    - 'polygon': uses centroid of vertices

    Parameters
    ----------
    regions : list
        VIA regions list from one image entry.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with columns [x, y].
    """
    points = []

    for region in regions:
        shape = region.get("shape_attributes", {})
        name = shape.get("name", "")

        try:
            if name in ("point", "circle"):
                x = float(shape.get("cx", shape.get("x", 0)))
                y = float(shape.get("cy", shape.get("y", 0)))
                points.append([x, y])

            elif name == "rect":
                x = float(shape.get("x", 0)) + float(shape.get("width", 0)) / 2
                y = float(shape.get("y", 0)) + float(shape.get("height", 0)) / 2
                points.append([x, y])

            elif name == "polygon":
                xs = shape.get("all_points_x", [])
                ys = shape.get("all_points_y", [])
                if xs and ys:
                    points.append([np.mean(xs), np.mean(ys)])

            elif name == "ellipse":
                points.append([float(shape.get("cx", 0)), float(shape.get("cy", 0))])

        except (TypeError, ValueError) as e:
            log.debug(f"Skipping malformed region ({name}): {e}")

    return np.array(points, dtype=np.float32) if points else np.zeros((0, 2), dtype=np.float32)


def convert_to_mat(
    points: np.ndarray,
    image_filename: str,
    output_path: str,
) -> None:
    """
    Save point annotations in ShanghaiTech .mat format.

    The output .mat file has the structure:
        image_info[0,0][0,0][0] = Nx2 array of (x, y) points

    Parameters
    ----------
    points : np.ndarray (N, 2)
    image_filename : str
        Original image filename (stored as attribute).
    output_path : str
        Destination .mat file path.
    """
    # Build the nested structure expected by ShanghaiTech loaders
    ann = np.zeros((1,), dtype=object)
    ann[0] = points

    inner = np.zeros((1, 1), dtype=object)
    inner[0, 0] = ann

    image_info = np.zeros((1, 1), dtype=object)
    image_info[0, 0] = inner

    savemat(output_path, {
        "image_info": image_info,
        "image_filename": image_filename,
        "num_annotations": len(points),
    })

    log.debug(f"Saved {len(points)} points to {output_path}")


def main() -> None:
    """Convert VIA JSON annotations to ShanghaiTech .mat files."""
    parser = argparse.ArgumentParser(
        description="Convert VIA JSON annotations to ShanghaiTech .mat format"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="VIA exported JSON file path")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for .mat ground truth files")
    parser.add_argument("--images", default=None,
                        help="Image directory (for verification only)")
    parser.add_argument("--prefix", default="GT_",
                        help="Prefix for output .mat files (default: GT_)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_via_json(args.input)

    converted = 0
    skipped = 0
    all_counts = []

    for key, entry in annotations.items():
        filename = entry.get("filename", key)
        regions = entry.get("regions", [])

        if not regions:
            log.warning(f"No regions for {filename} — skipping")
            skipped += 1
            continue

        points = via_to_points(regions)
        stem = Path(filename).stem
        mat_path = out_dir / f"{args.prefix}{stem}.mat"

        convert_to_mat(points, filename, str(mat_path))

        count = len(points)
        all_counts.append(count)
        log.info(f"  {filename}: {count} annotations -> {mat_path.name}")
        converted += 1

    log.info(f"\nConversion complete:")
    log.info(f"  Converted: {converted} files")
    log.info(f"  Skipped:   {skipped} files (no annotations)")
    if all_counts:
        log.info(f"  Counts:    min={min(all_counts)}, max={max(all_counts)}, "
                 f"mean={np.mean(all_counts):.1f}")
    log.info(f"  Output:    {out_dir}")


if __name__ == "__main__":
    main()
