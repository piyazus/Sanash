from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont, ImageOps

from datasets.almaty_transit import AlmatyTransitDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw P2PNet predictions and GT points on validation images."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("p2pnet_almaty_dataset_block_stratified"),
        help="Dataset root",
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="Trained checkpoint")
    parser.add_argument(
        "--model-module",
        required=True,
        help="Python module containing the model factory, e.g. models or models.p2pnet",
    )
    parser.add_argument(
        "--model-factory",
        default="build_model",
        help="Factory function in --model-module",
    )
    parser.add_argument(
        "--model-args-json",
        type=Path,
        default=None,
        help="Optional JSON file converted into an argparse Namespace for the model factory",
    )
    parser.add_argument("--out", type=Path, default=Path("debug_outputs/prediction_overlays"))
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def namespace_from_json(path: Path | None) -> Namespace:
    if path is None:
        return Namespace()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--model-args-json must contain a JSON object")
    return Namespace(**data)


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    module = importlib.import_module(args.model_module)
    factory = getattr(module, args.model_factory)
    model_args = namespace_from_json(args.model_args_json)

    try:
        built = factory(model_args)
    except TypeError:
        built = factory()

    if isinstance(built, tuple):
        model = built[0]
    else:
        model = built

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            state_dict = checkpoint[key]
            break

    if not isinstance(state_dict, dict):
        raise ValueError(f"Could not find a state dict in {args.checkpoint}")

    cleaned = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"WARNING: missing model keys: {len(missing)}")
    if unexpected:
        print(f"WARNING: unexpected checkpoint keys: {len(unexpected)}")

    model.to(args.device)
    model.eval()
    return model


def call_model(model: torch.nn.Module, image_tensor: torch.Tensor) -> Any:
    batch = image_tensor.unsqueeze(0)
    try:
        return model(batch)
    except Exception:
        return model([image_tensor])


def _first_batch_item(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim >= 3:
        return tensor[0]
    return tensor


def _scores_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logits = _first_batch_item(logits.detach().float().cpu())
    if logits.ndim == 1:
        return logits.sigmoid()
    if logits.shape[-1] == 1:
        return logits[..., 0].sigmoid()
    return logits.softmax(dim=-1)[..., 1]


def extract_pred_points(
    outputs: Any, width: int, height: int, score_threshold: float
) -> np.ndarray:
    scores: torch.Tensor | None = None
    points: torch.Tensor | None = None

    if isinstance(outputs, dict):
        for score_key in ("pred_logits", "logits", "scores", "pred_scores"):
            if score_key in outputs and torch.is_tensor(outputs[score_key]):
                raw_scores = outputs[score_key]
                scores = (
                    _scores_from_logits(raw_scores)
                    if score_key in {"pred_logits", "logits"}
                    else _first_batch_item(raw_scores.detach().float().cpu())
                )
                break

        for point_key in ("pred_points", "points", "pred_coords", "coords"):
            if point_key in outputs and torch.is_tensor(outputs[point_key]):
                points = _first_batch_item(outputs[point_key].detach().float().cpu())
                break

    elif torch.is_tensor(outputs):
        points = _first_batch_item(outputs.detach().float().cpu())
    elif isinstance(outputs, (list, tuple)):
        for item in outputs:
            if torch.is_tensor(item) and item.shape[-1] == 2:
                points = _first_batch_item(item.detach().float().cpu())
                break

    if points is None:
        raise ValueError(
            "Could not extract predicted points. Expected an output key like "
            "'pred_points' with shape [B, N, 2]."
        )
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Predicted points must have shape [N, 2], got {tuple(points.shape)}")

    if scores is not None:
        scores = scores.reshape(-1)
        keep = scores >= score_threshold
        if keep.shape[0] == points.shape[0]:
            points = points[keep]

    if points.numel() and float(points.max()) <= 1.5:
        points = points.clone()
        points[:, 0] *= float(width)
        points[:, 1] *= float(height)

    return points.numpy()


def load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ("arial.ttf", "DejaVuSans.ttf", "calibri.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_overlay(
    image_path: Path,
    gt_points: torch.Tensor,
    pred_points: np.ndarray,
    out_path: Path,
) -> None:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGBA")

    draw = ImageDraw.Draw(image, "RGBA")
    font = load_font(24)
    label = f"{image_path.name} | gt={gt_points.shape[0]} pred={len(pred_points)}"
    draw.rectangle((0, 0, 860, 48), fill=(0, 0, 0, 190))
    draw.text((12, 10), label, fill=(255, 255, 255, 255), font=font)

    for point in gt_points:
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(0, 255, 100, 255), width=3)

    for point in pred_points:
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        size = 8
        draw.line((x - size, y - size, x + size, y + size), fill=(255, 60, 60, 255), width=4)
        draw.line((x - size, y + size, x + size, y - size), fill=(255, 60, 60, 255), width=4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out_path, quality=92)


def main() -> int:
    args = parse_args()
    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    dataset = AlmatyTransitDataset(args.root, split="val")
    model = load_model(args)
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), min(args.num_images, len(dataset)))

    args.out.mkdir(parents=True, exist_ok=True)
    absolute_errors: list[float] = []
    squared_errors: list[float] = []
    pred_counts: list[int] = []
    gt_counts: list[int] = []

    with torch.inference_mode():
        for output_index, sample_index in enumerate(indices, start=1):
            image, target = dataset[sample_index]
            image_tensor = TF.to_tensor(image).to(args.device)
            outputs = call_model(model, image_tensor)
            width, height = image.size
            pred_points = extract_pred_points(
                outputs, width, height, args.score_threshold
            )

            gt_count = int(target["points"].shape[0])
            pred_count = int(len(pred_points))
            error = float(abs(pred_count - gt_count))
            absolute_errors.append(error)
            squared_errors.append(error * error)
            pred_counts.append(pred_count)
            gt_counts.append(gt_count)

            sample = dataset.samples[sample_index]
            out_path = args.out / f"{output_index:03d}_{sample.image_path.stem}.jpg"
            draw_overlay(sample.image_path, target["points"], pred_points, out_path)

    mae = float(np.mean(absolute_errors)) if absolute_errors else 0.0
    rmse = float(math.sqrt(np.mean(squared_errors))) if squared_errors else 0.0
    avg_pred = float(np.mean(pred_counts)) if pred_counts else 0.0
    avg_gt = float(np.mean(gt_counts)) if gt_counts else 0.0

    print("Prediction validation metrics")
    print(f"  images: {len(indices)}")
    print(f"  MAE: {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  average predicted count: {avg_pred:.3f}")
    print(f"  average ground truth count: {avg_gt:.3f}")
    print(f"  overlays: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
