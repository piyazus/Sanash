from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as standard_transforms
from PIL import Image

from models import build_model


SCRIPT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_ROOT.parent
DEFAULT_INPUT_DIR = WORKSPACE_ROOT / "frames_with_people"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "output_predictions"
DEFAULT_CHECKPOINT = SCRIPT_ROOT / "weights" / "SHTechA.pth"
DEFAULT_COUNTS_CSV = DEFAULT_OUTPUT_DIR / "predicted_counts.csv"
DEFAULT_BATCH_SIZE = 50
DEFAULT_INFERENCE_BATCH_SIZE = 4
THRESHOLD = 0.3
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run P2PNet inference over extracted frames.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weight-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--counts-csv", type=Path, default=DEFAULT_COUNTS_CSV)
    parser.add_argument("--backbone", default="vgg16_bn")
    parser.add_argument("--row", default=2, type=int)
    parser.add_argument("--line", default=2, type=int)
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument("--inference-batch-size", default=DEFAULT_INFERENCE_BATCH_SIZE, type=int)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--worker-manifest", type=Path)
    return parser.parse_args()


def iter_images(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_completed(counts_csv: Path) -> set[str]:
    if not counts_csv.exists():
        return set()
    with counts_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["relative_path"] for row in reader if row.get("relative_path")}


def read_manifest(manifest_path: Path) -> list[Path]:
    return [Path(line.strip()) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_rows(counts_csv: Path, rows: list[tuple[str, int]]) -> None:
    counts_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = counts_csv.exists()
    with counts_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(["relative_path", "predicted_count"])
        writer.writerows(rows)


def resize_for_inference(image: Image.Image) -> Image.Image:
    width, height = image.size
    new_width = max(128, (width // 128) * 128)
    new_height = max(128, (height // 128) * 128)
    if (new_width, new_height) == image.size:
        return image
    resampling = getattr(Image, "Resampling", Image)
    return image.resize((new_width, new_height), resampling.LANCZOS)


def build_transform():
    return standard_transforms.Compose(
        [
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_model(args) -> torch.nn.Module:
    device = torch.device(args.device)
    model = build_model(args)
    checkpoint = torch.load(args.weight_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def relative_image_path(image_path: Path, input_root: Path) -> str:
    return str(image_path.resolve().relative_to(input_root.resolve())).replace("\\", "/")


def save_prediction(
    relative_path: str,
    image_resized: Image.Image,
    kept_points: np.ndarray,
    predicted_count: int,
    output_root: Path,
) -> None:
    image_to_draw = cv2.cvtColor(np.array(image_resized), cv2.COLOR_RGB2BGR)
    for point in kept_points:
        cv2.circle(image_to_draw, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
    cv2.putText(
        image_to_draw,
        f"Count: {predicted_count}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    destination = output_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), image_to_draw)


def worker_main(args) -> None:
    device = torch.device(args.device)
    transform = build_transform()
    model = load_model(args)
    image_paths = read_manifest(args.worker_manifest)
    completed = load_completed(args.counts_csv)
    if completed:
        image_paths = [
            path for path in image_paths if relative_image_path(path, args.input_dir) not in completed
        ]
    inference_batch_size = max(1, args.inference_batch_size)

    for start in range(0, len(image_paths), inference_batch_size):
        batch_paths = image_paths[start : start + inference_batch_size]
        prepared = []
        for image_path in batch_paths:
            image_raw = Image.open(image_path).convert("RGB")
            image_resized = resize_for_inference(image_raw)
            prepared.append(
                {
                    "path": image_path,
                    "relative_path": relative_image_path(image_path, args.input_dir),
                    "image": image_resized,
                    "tensor": transform(image_resized),
                }
            )

        batch_tensor = torch.stack([item["tensor"] for item in prepared]).to(device)
        with torch.no_grad():
            outputs = model(batch_tensor)
            all_scores = torch.softmax(outputs["pred_logits"], dim=-1)[:, :, 1]
            all_points = outputs["pred_points"]

        for index, item in enumerate(prepared):
            keep = all_scores[index] > THRESHOLD
            kept_points = all_points[index][keep].detach().cpu().numpy()
            predicted_count = int(keep.sum().item())
            save_prediction(
                relative_path=item["relative_path"],
                image_resized=item["image"],
                kept_points=kept_points,
                predicted_count=predicted_count,
                output_root=args.output_dir,
            )
            append_rows(args.counts_csv, [(item["relative_path"], predicted_count)])
            print(f"{item['relative_path']},{predicted_count}", flush=True)


def run_worker(args, manifest_path: Path) -> int:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--input-dir",
        str(args.input_dir),
        "--output-dir",
        str(args.output_dir),
        "--weight-path",
        str(args.weight_path),
        "--counts-csv",
        str(args.counts_csv),
        "--backbone",
        args.backbone,
        "--row",
        str(args.row),
        "--line",
        str(args.line),
        "--inference-batch-size",
        str(args.inference_batch_size),
        "--device",
        args.device,
        "--worker-manifest",
        str(manifest_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=7200)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    return result.returncode


def parent_main(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_images = iter_images(args.input_dir)
    completed = load_completed(args.counts_csv)
    pending_images = [
        path
        for path in all_images
        if str(path.resolve().relative_to(args.input_dir.resolve())).replace("\\", "/") not in completed
    ]
    if args.max_images is not None:
        pending_images = pending_images[: max(0, args.max_images)]

    if not pending_images:
        print("No pending images to process.")
    else:
        total_pending = len(pending_images)
        print(f"Pending images in this run: {total_pending}", flush=True)
        for start in range(0, total_pending, max(1, args.batch_size)):
            end = min(start + max(1, args.batch_size), total_pending)
            chunk = pending_images[start:end]
            with tempfile.NamedTemporaryFile(
                "w",
                suffix=".txt",
                prefix="p2pnet_batch_",
                dir=WORKSPACE_ROOT,
                delete=False,
                encoding="utf-8",
            ) as manifest:
                for image_path in chunk:
                    manifest.write(f"{image_path.resolve()}\n")
                manifest_path = Path(manifest.name)

            try:
                exit_code = run_worker(args, manifest_path)
            finally:
                manifest_path.unlink(missing_ok=True)

            if exit_code != 0:
                raise RuntimeError(f"Worker failed for batch {start}-{end} with exit code {exit_code}.")

            print(f"Processed {end}/{total_pending} pending images", flush=True)

    with args.counts_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        total_count = 0
        row_count = 0
        for row in reader:
            row_count += 1
            total_count += int(row["predicted_count"])
    print(f"Completed images: {row_count}")
    print(f"Total predicted heads: {total_count}")


def main() -> None:
    args = parse_args()
    if args.worker_manifest:
        worker_main(args)
        return
    parent_main(args)


if __name__ == "__main__":
    main()
