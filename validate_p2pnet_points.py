from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ValidationRecord:
    image_path: Path
    gt_path: Path
    filename: str
    image_width: int
    image_height: int
    point_count: int
    gt_exists: bool
    points_in_bounds: bool
    out_of_bounds_count: int
    suspicious_low_count: bool
    suspicious_high_count: bool
    load_error: str
    overlay_path: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create visual overlays and contact sheets for P2PNet point files."
    )
    parser.add_argument("--images", required=True, type=Path, help="Prepared image folder")
    parser.add_argument("--gt", required=True, type=Path, help="Folder of .npy point files")
    parser.add_argument("--out", required=True, type=Path, help="Overlay output folder")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=80,
        help="Number of images to render as overlays. Use 0 or a negative value for all images.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random sample seed")
    parser.add_argument(
        "--point-radius", type=int, default=6, help="Radius of drawn point circles"
    )
    parser.add_argument(
        "--draw-indices",
        action="store_true",
        help="Draw a small point index beside each point",
    )
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ("arial.ttf", "DejaVuSans.ttf", "calibri.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def image_files(images_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name.casefold(),
    )


def load_points(gt_path: Path) -> tuple[np.ndarray, str]:
    if not gt_path.exists():
        return np.empty((0, 2), dtype=np.float32), ""

    try:
        points = np.load(gt_path)
    except Exception as exc:
        return np.empty((0, 2), dtype=np.float32), f"failed_to_load_gt: {exc}"

    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32), ""

    if points.ndim != 2 or points.shape[1] != 2:
        return (
            np.empty((0, 2), dtype=np.float32),
            f"unexpected_gt_shape: {tuple(points.shape)}",
        )

    return points.astype(np.float32, copy=False), ""


def point_bounds(points: np.ndarray, width: int, height: int) -> tuple[np.ndarray, int]:
    if points.size == 0:
        return np.empty((0,), dtype=bool), 0

    finite = np.isfinite(points).all(axis=1)
    inside = (
        finite
        & (points[:, 0] >= 0.0)
        & (points[:, 0] <= float(width))
        & (points[:, 1] >= 0.0)
        & (points[:, 1] <= float(height))
    )
    return inside, int((~inside).sum())


def inspect_image(image_path: Path, gt_dir: Path) -> tuple[ValidationRecord, np.ndarray]:
    gt_path = gt_dir / f"{image_path.stem}.npy"
    points, load_error = load_points(gt_path)

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size

    gt_exists = gt_path.exists()
    inside_mask, out_of_bounds_count = point_bounds(points, width, height)
    point_count = int(points.shape[0])
    record = ValidationRecord(
        image_path=image_path,
        gt_path=gt_path,
        filename=image_path.name,
        image_width=width,
        image_height=height,
        point_count=point_count,
        gt_exists=gt_exists,
        points_in_bounds=gt_exists and not load_error and out_of_bounds_count == 0,
        out_of_bounds_count=out_of_bounds_count,
        suspicious_low_count=gt_exists and not load_error and point_count <= 3,
        suspicious_high_count=gt_exists and not load_error and point_count >= 38,
        load_error=load_error,
    )
    return record, points


def record_is_suspicious(record: ValidationRecord) -> bool:
    return (
        not record.gt_exists
        or bool(record.load_error)
        or record.point_count == 0
        or record.out_of_bounds_count > 0
        or record.suspicious_low_count
        or record.suspicious_high_count
    )


def selected_for_overlays(
    records: list[ValidationRecord], sample_size: int, seed: int
) -> set[str]:
    if sample_size <= 0 or sample_size >= len(records):
        return {record.filename for record in records}

    suspicious = [record for record in records if record_is_suspicious(record)]
    normal = [record for record in records if not record_is_suspicious(record)]
    rng = random.Random(seed)

    if len(suspicious) >= sample_size:
        selected = rng.sample(suspicious, sample_size)
    else:
        selected = list(suspicious)
        selected.extend(rng.sample(normal, sample_size - len(selected)))

    return {record.filename for record in selected}


def draw_corner_label(
    draw: ImageDraw.ImageDraw,
    label: str,
    width: int,
    font: ImageFont.ImageFont,
) -> None:
    padding = 12
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = min(text_bbox[2] - text_bbox[0], width - padding * 4)
    text_height = text_bbox[3] - text_bbox[1]
    box = (0, 0, text_width + padding * 2, text_height + padding * 2)
    draw.rectangle(box, fill=(0, 0, 0, 180))
    draw.text((padding, padding), label, fill=(255, 255, 255, 255), font=font)


def draw_points_overlay(
    image_path: Path,
    points: np.ndarray,
    out_path: Path,
    point_radius: int,
    draw_indices: bool,
) -> None:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGBA")

    width, height = image.size
    draw = ImageDraw.Draw(image, "RGBA")
    label_font = load_font(28)
    index_font = load_font(16)

    label = f"{image_path.name} | count: {len(points)}"
    draw_corner_label(draw, label, width, label_font)

    for index, point in enumerate(points, start=1):
        x = float(point[0])
        y = float(point[1])
        in_bounds = math.isfinite(x) and math.isfinite(y) and 0 <= x <= width and 0 <= y <= height

        if in_bounds:
            cx = int(round(x))
            cy = int(round(y))
            outer = (
                cx - point_radius - 2,
                cy - point_radius - 2,
                cx + point_radius + 2,
                cy + point_radius + 2,
            )
            inner = (
                cx - point_radius,
                cy - point_radius,
                cx + point_radius,
                cy + point_radius,
            )
            draw.ellipse(outer, fill=(0, 0, 0, 210))
            draw.ellipse(inner, fill=(255, 225, 0, 235), outline=(255, 40, 40, 255), width=2)
            draw.line((cx - point_radius - 3, cy, cx + point_radius + 3, cy), fill=(255, 40, 40, 255), width=2)
            draw.line((cx, cy - point_radius - 3, cx, cy + point_radius + 3), fill=(255, 40, 40, 255), width=2)
        else:
            cx = int(min(max(x if math.isfinite(x) else 0.0, 0.0), float(width)))
            cy = int(min(max(y if math.isfinite(y) else 0.0, 0.0), float(height)))
            size = point_radius + 8
            draw.line((cx - size, cy - size, cx + size, cy + size), fill=(255, 0, 255, 255), width=5)
            draw.line((cx - size, cy + size, cx + size, cy - size), fill=(255, 0, 255, 255), width=5)

        if draw_indices:
            draw.text(
                (cx + point_radius + 4, cy - point_radius - 4),
                str(index),
                fill=(255, 255, 255, 255),
                stroke_width=2,
                stroke_fill=(0, 0, 0, 255),
                font=index_font,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out_path, quality=92)


def clean_previous_outputs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("overlay_*.jpg", "contact_sheet_*.jpg", "validation_summary.csv"):
        for path in out_dir.glob(pattern):
            try:
                path.unlink()
            except PermissionError:
                pass


def write_summary_csv(records: list[ValidationRecord], out_path: Path) -> None:
    fieldnames = [
        "filename",
        "image_width",
        "image_height",
        "point_count",
        "gt_exists",
        "points_in_bounds",
        "overlay_path",
        "out_of_bounds_count",
        "empty_gt",
        "suspicious_low_count",
        "suspicious_high_count",
        "load_error",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "filename": record.filename,
                    "image_width": record.image_width,
                    "image_height": record.image_height,
                    "point_count": record.point_count,
                    "gt_exists": record.gt_exists,
                    "points_in_bounds": record.points_in_bounds,
                    "overlay_path": record.overlay_path,
                    "out_of_bounds_count": record.out_of_bounds_count,
                    "empty_gt": record.gt_exists and record.point_count == 0,
                    "suspicious_low_count": record.suspicious_low_count,
                    "suspicious_high_count": record.suspicious_high_count,
                    "load_error": record.load_error,
                }
            )


def make_contact_sheets(
    records: list[ValidationRecord],
    out_dir: Path,
    per_sheet: int = 20,
    columns: int = 4,
) -> int:
    overlay_records = [record for record in records if record.overlay_path]
    if not overlay_records:
        return 0

    thumb_width = 480
    thumb_height = 270
    caption_height = 34
    gutter = 12
    rows = math.ceil(per_sheet / columns)
    sheet_width = columns * thumb_width + (columns + 1) * gutter
    sheet_height = rows * (thumb_height + caption_height) + (rows + 1) * gutter
    caption_font = load_font(16)
    sheets = 0

    for sheet_index in range(0, len(overlay_records), per_sheet):
        chunk = overlay_records[sheet_index : sheet_index + per_sheet]
        sheet = Image.new("RGB", (sheet_width, sheet_height), color=(245, 245, 245))
        draw = ImageDraw.Draw(sheet)

        for tile_index, record in enumerate(chunk):
            row = tile_index // columns
            col = tile_index % columns
            x0 = gutter + col * (thumb_width + gutter)
            y0 = gutter + row * (thumb_height + caption_height + gutter)

            with Image.open(record.overlay_path) as overlay:
                overlay = overlay.convert("RGB")
                overlay.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
                paste_x = x0 + (thumb_width - overlay.width) // 2
                paste_y = y0 + (thumb_height - overlay.height) // 2
                sheet.paste(overlay, (paste_x, paste_y))

            caption = f"{record.filename} | n={record.point_count}"
            caption_y = y0 + thumb_height + 6
            draw.text(
                (x0, caption_y),
                caption[:72],
                fill=(20, 20, 20),
                font=caption_font,
            )

        sheets += 1
        sheet_path = out_dir / f"contact_sheet_{sheets:03d}.jpg"
        sheet.save(sheet_path, quality=92)

    return sheets


def main() -> int:
    args = parse_args()

    if not args.images.exists():
        print(f"ERROR: images folder not found: {args.images}", file=sys.stderr)
        return 1
    if not args.gt.exists():
        print(f"ERROR: gt folder not found: {args.gt}", file=sys.stderr)
        return 1
    if args.point_radius < 1:
        print("ERROR: --point-radius must be >= 1", file=sys.stderr)
        return 1

    print("P2PNet point visual validation")
    print(f"  Images: {args.images}")
    print(f"  GT: {args.gt}")
    print(f"  Output: {args.out}")
    print(f"  Sample size: {args.sample_size}")
    print(f"  Seed: {args.seed}")
    print(f"  Point radius: {args.point_radius}")

    images = image_files(args.images)
    if not images:
        print(f"ERROR: no images found in {args.images}", file=sys.stderr)
        return 1

    clean_previous_outputs(args.out)

    points_by_filename: dict[str, np.ndarray] = {}
    records: list[ValidationRecord] = []
    for image_path in images:
        record, points = inspect_image(image_path, args.gt)
        records.append(record)
        points_by_filename[record.filename] = points

    selected = selected_for_overlays(records, args.sample_size, args.seed)
    overlays_saved = 0
    for record in records:
        if record.filename not in selected:
            continue
        overlay_path = args.out / f"overlay_{record.image_path.stem}.jpg"
        draw_points_overlay(
            record.image_path,
            points_by_filename[record.filename],
            overlay_path,
            args.point_radius,
            args.draw_indices,
        )
        record.overlay_path = str(overlay_path)
        overlays_saved += 1

    contact_sheets_saved = make_contact_sheets(records, args.out)
    write_summary_csv(records, args.out / "validation_summary.csv")

    missing_gt = sum(1 for record in records if not record.gt_exists)
    empty_gt = sum(1 for record in records if record.gt_exists and record.point_count == 0)
    out_of_bounds_points = sum(record.out_of_bounds_count for record in records)
    low_count_images = sum(1 for record in records if record.suspicious_low_count)
    high_count_images = sum(1 for record in records if record.suspicious_high_count)

    summary = {
        "total_checked": len(records),
        "missing_gt": missing_gt,
        "empty_gt": empty_gt,
        "out_of_bounds_points": out_of_bounds_points,
        "low_count_images": low_count_images,
        "high_count_images": high_count_images,
        "overlays_saved": overlays_saved,
        "contact_sheets_saved": contact_sheets_saved,
    }
    (args.out / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("")
    print("Validation summary")
    print(f"  Total checked: {summary['total_checked']}")
    print(f"  Missing gt: {summary['missing_gt']}")
    print(f"  Empty gt: {summary['empty_gt']}")
    print(f"  Out-of-bounds points: {summary['out_of_bounds_points']}")
    print(f"  Low-count images <= 3: {summary['low_count_images']}")
    print(f"  High-count images >= 38: {summary['high_count_images']}")
    print(f"  Overlays saved: {summary['overlays_saved']}")
    print(f"  Contact sheets saved: {summary['contact_sheets_saved']}")
    print(f"  Summary CSV: {args.out / 'validation_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
