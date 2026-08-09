from __future__ import annotations

import argparse
from pathlib import Path

from .inference import PassengerCounter, find_default_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Sanash passenger-counting model inputs and outputs.")
    parser.add_argument("--model", type=Path, default=None, help="Path to .onnx/.pth/.pt/.ckpt or artifact .zip")
    parser.add_argument("--image", type=Path, default=None, help="Optional image to run through the model")
    parser.add_argument("--threshold", type=float, default=0.50, help="Confidence threshold for optional image inference")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="PyTorch inference device")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model or find_default_model(Path.cwd())
    if model_path is None:
        print("ERROR: no model was provided and no default model was found.")
        return 1
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}")
        return 1

    counter = PassengerCounter(device=args.device)
    try:
        summary = counter.load_model(model_path)
    except Exception as exc:
        print(f"ERROR: could not load model: {exc}")
        return 1

    print(summary.details)
    print(f"Resolved model file: {summary.resolved_path}")

    if args.image is not None:
        if not args.image.exists():
            print(f"ERROR: image not found: {args.image}")
            return 1
        try:
            result = counter.predict_image(args.image, threshold=args.threshold)
        except Exception as exc:
            print(f"ERROR: inference failed: {exc}")
            return 1
        print()
        print("Sample inference")
        print(f"Image: {result.image_path}")
        print(f"Original size: {result.original_image.size[0]}x{result.original_image.size[1]}")
        print(f"Count: {result.count}")
        print(f"Points shape: {result.points.shape}")
        print(f"Scores shape: {result.scores.shape if result.scores is not None else None}")
        print(f"Inference time: {result.inference_ms:.1f} ms")
        print(f"Diagnostics: {result.diagnostics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
