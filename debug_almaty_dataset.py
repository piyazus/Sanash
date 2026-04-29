from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import DataLoader

from datasets.almaty_transit import AlmatyTransitDataset, collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug the Almaty P2PNet dataset.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("p2pnet_almaty_dataset_block_stratified"),
        help="Dataset root containing images/{train,val}, gt/{train,val}, train.txt, val.txt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("debug_outputs/almaty_dataset_overlays"),
        help="Output folder for debug overlays",
    )
    parser.add_argument("--num-overlays", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--point-radius", type=int, default=6)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ("arial.ttf", "DejaVuSans.ttf", "calibri.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_overlay(
    image_path: Path,
    points: torch.Tensor,
    out_path: Path,
    point_radius: int,
) -> None:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGBA")

    draw = ImageDraw.Draw(image, "RGBA")
    font = load_font(24)
    label = f"{image_path.name} | gt: {points.shape[0]}"
    draw.rectangle((0, 0, 720, 46), fill=(0, 0, 0, 185))
    draw.text((12, 10), label, fill=(255, 255, 255, 255), font=font)

    for point in points:
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        outer = (
            x - point_radius - 2,
            y - point_radius - 2,
            x + point_radius + 2,
            y + point_radius + 2,
        )
        inner = (
            x - point_radius,
            y - point_radius,
            x + point_radius,
            y + point_radius,
        )
        draw.ellipse(outer, fill=(0, 0, 0, 220))
        draw.ellipse(inner, fill=(0, 255, 120, 230), outline=(255, 255, 255, 255), width=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out_path, quality=92)


def print_first_samples(dataset: AlmatyTransitDataset, split_name: str) -> None:
    print(f"{split_name} first 5 samples:")
    for index in range(min(5, len(dataset))):
        image, target = dataset[index]
        width, height = image.size
        print(
            f"  {index:03d}: "
            f"{target['filename']} | "
            f"image_size=({width}x{height}) | "
            f"points={target['points'].shape[0]}"
        )


def main() -> int:
    args = parse_args()

    train_dataset = AlmatyTransitDataset(args.root, split="train")
    val_dataset = AlmatyTransitDataset(args.root, split="val")

    print("Almaty dataset debug")
    print(f"  root: {args.root}")
    print(f"  train size: {len(train_dataset)}")
    print(f"  val size: {len(val_dataset)}")
    print_first_samples(train_dataset, "train")
    print_first_samples(val_dataset, "val")

    loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    images, targets = next(iter(loader))
    print("")
    print(
        "DataLoader smoke batch: "
        f"images={len(images)}, "
        f"target_point_counts={[target['points'].shape[0] for target in targets]}"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    for old_overlay in args.out.glob("*.jpg"):
        try:
            old_overlay.unlink()
        except PermissionError:
            pass

    rng = random.Random(args.seed)
    combined = [("train", index) for index in range(len(train_dataset))]
    combined.extend(("val", index) for index in range(len(val_dataset)))
    selected = rng.sample(combined, min(args.num_overlays, len(combined)))

    for overlay_index, (split, sample_index) in enumerate(selected, start=1):
        dataset = train_dataset if split == "train" else val_dataset
        _, target = dataset[sample_index]
        sample = dataset.samples[sample_index]
        out_path = args.out / f"{overlay_index:03d}_{split}_{sample.image_path.stem}.jpg"
        draw_overlay(sample.image_path, target["points"], out_path, args.point_radius)

    print(f"  saved overlays: {len(selected)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
