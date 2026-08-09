from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .inference import PassengerCounter, find_default_model, load_rgb_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Sanash passenger-counting MAE on labeled images.")
    parser.add_argument("--model", type=Path, default=None, help="Path to .pth/.onnx or Sanash artifact .zip")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("p2pnet_almaty_dataset_block_stratified"),
        help="Dataset root with images/<split>, gt/<split>, and <split>.txt",
    )
    parser.add_argument("--split", default="val", help="Dataset split name, usually val or train")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of images to evaluate")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="PyTorch inference device")
    parser.add_argument(
        "--thresholds",
        default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70",
        help="Comma-separated confidence thresholds to sweep",
    )
    parser.add_argument("--csv-out", type=Path, default=None, help="Optional per-image CSV output")
    return parser.parse_args()


def _read_filenames(root: Path, split: str, limit: int) -> list[str]:
    split_file = root / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    names = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return names[:limit] if limit > 0 else names


def main() -> int:
    args = parse_args()
    model_path = args.model or find_default_model(Path.cwd())
    if model_path is None:
        print("ERROR: no model was provided and no default model was found.")
        return 1
    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    if not thresholds:
        print("ERROR: provide at least one threshold.")
        return 1

    image_dir = args.root / "images" / args.split
    gt_dir = args.root / "gt" / args.split
    if not image_dir.exists():
        print(f"ERROR: image folder not found: {image_dir}")
        return 1
    if not gt_dir.exists():
        print(f"ERROR: GT folder not found: {gt_dir}")
        return 1

    counter = PassengerCounter(device=args.device)
    summary = counter.load_model(model_path)
    print(summary.details)
    print()

    filenames = _read_filenames(args.root, args.split, args.limit)
    errors: dict[float, list[int]] = {threshold: [] for threshold in thresholds}
    biases: dict[float, list[int]] = {threshold: [] for threshold in thresholds}
    rows: list[dict[str, object]] = []

    for index, filename in enumerate(filenames, start=1):
        image_path = image_dir / filename
        gt_path = gt_dir / f"{Path(filename).stem}.npy"
        gt_count = int(np.load(gt_path).shape[0])
        image = load_rgb_image(image_path)
        _points, scores, inference_ms, _diag = counter.backend.predict(image, threshold=0.0)  # type: ignore[union-attr]

        row: dict[str, object] = {
            "filename": filename,
            "gt": gt_count,
            "inference_ms": f"{inference_ms:.1f}",
        }
        parts = []
        for threshold in thresholds:
            pred_count = int((scores >= threshold).sum()) if scores is not None and len(scores) else 0
            error = abs(pred_count - gt_count)
            errors[threshold].append(error)
            biases[threshold].append(pred_count - gt_count)
            row[f"pred@{threshold:.2f}"] = pred_count
            row[f"abs_error@{threshold:.2f}"] = error
            parts.append(f"{threshold:.2f}:{pred_count}")
        rows.append(row)
        print(f"{index:03d}/{len(filenames)} {filename} gt={gt_count} " + " ".join(parts), flush=True)

    print()
    print(f"Summary on {len(filenames)} {args.split} image(s)")
    best_threshold = None
    best_mae = float("inf")
    for threshold in thresholds:
        mae = float(np.mean(errors[threshold])) if errors[threshold] else 0.0
        bias = float(np.mean(biases[threshold])) if biases[threshold] else 0.0
        print(f"threshold={threshold:.2f} MAE={mae:.3f} bias={bias:+.3f}")
        if mae < best_mae:
            best_mae = mae
            best_threshold = threshold
    print(f"Recommended threshold on this set: {best_threshold:.2f} (MAE={best_mae:.3f})")

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys()) if rows else ["filename", "gt"]
        with args.csv_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV: {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
