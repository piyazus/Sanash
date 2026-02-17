"""
Server-side training script for Sanash (Colab / Kaggle / local).

Features
--------
- Environment awareness:
  - Detects Google Colab and mounts Google Drive to /content/drive.
  - Detects Kaggle kernels and uses /kaggle/working as base.
  - Falls back to local repository paths when neither is detected.
- Sequential training:
  - YOLO11 (detection) via Ultralytics through TrainingEngine.
  - CSRNet (density maps) with native PyTorch loop.
  - P2PNet (point maps) with native PyTorch loop.
- Artifact management:
  - Saves best.pt weights for all three models into a unified weights/ directory.
- Sanity checks:
  - After each epoch for CSRNet and P2PNet, saves one validation image
    visualization to logs/viz/ (handled inside TrainingEngine).
  - After YOLO11 training, saves one image with predicted boxes to logs/viz/.
- WandB:
  - Logging is optional and fail-safe; if wandb is missing or misconfigured,
    training still proceeds.
"""

from __future__ import annotations

import argparse
import os
import sys
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.factory import ModelConfig, ModelKind, create_model, YOLO11Wrapper
from trainers.engine import TrainingEngine, TrainConfig, ExperimentSpec
from data_pipeline.shanghaitech import ShanghaiTechDataset


def detect_environment() -> tuple[bool, bool]:
    """Return (is_colab, is_kaggle)."""
    is_colab = "google.colab" in sys.modules
    is_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    return is_colab, is_kaggle


def get_base_dir() -> Path:
    """Determine the base directory for data, logs, and weights."""
    is_colab, is_kaggle = detect_environment()
    if is_colab:
        try:
            from google.colab import drive  # type: ignore

            drive.mount("/content/drive")
            base = Path("/content/drive/MyDrive/sanash")
            print(f"[env] Detected Colab. Using base dir: {base}")
            base.mkdir(parents=True, exist_ok=True)
            return base
        except Exception as exc:  # pragma: no cover - runtime environment
            print(f"[env] Colab detected but Google Drive mount failed: {exc}")
            return Path("/content/sanash")
    if is_kaggle:
        base = Path("/kaggle/working/sanash")
        print(f"[env] Detected Kaggle. Using base dir: {base}")
        base.mkdir(parents=True, exist_ok=True)
        return base

    # Local development: assume this script lives in repo_root/scripts/
    return Path(__file__).resolve().parents[1]


def build_data_paths(base_dir: Path) -> dict[str, Path]:
    """
    Build standard ShanghaiTech paths (Part A) relative to base_dir.
    Adjust here if your data layout differs on the server.
    """
    root = base_dir / "data" / "ShanghaiTech"
    part = "part_A"
    paths = {
        "train_images": root / part / "train_data" / "images",
        "train_gt": root / part / "train_data" / "ground_truth",
        "val_images": root / part / "test_data" / "images",
        "val_gt": root / part / "test_data" / "ground_truth",
    }
    return paths


def ensure_dirs(base_dir: Path) -> dict[str, Path]:
    weights_dir = base_dir / "weights"
    logs_viz_dir = base_dir / "logs" / "viz"
    runs_dir = base_dir / "runs"

    weights_dir.mkdir(parents=True, exist_ok=True)
    logs_viz_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "weights": weights_dir,
        "logs_viz": logs_viz_dir,
        "runs": runs_dir,
    }


def train_yolo11(
    engine: TrainingEngine,
    device: str,
    base_dir: Path,
    dirs: dict[str, Path],
    use_wandb: bool,
) -> None:
    """
    Train YOLO11 using the unified engine.

    Assumes:
    - A YOLO11 pre-trained checkpoint is available at weights/yolo11n.pt (or similar).
    - A YOLO data.yaml exists at configs/shanghaitech_yolo.yaml.
    """
    weights_dir = dirs["weights"]
    logs_viz_dir = dirs["logs_viz"]

    yolo_pretrained = weights_dir / "yolo11n.pt"
    data_yaml = base_dir / "configs" / "shanghaitech_yolo.yaml"

    if not data_yaml.is_file():
        print(f"[YOLO11] data.yaml not found at {data_yaml}. Skipping YOLO training.")
        return

    if not yolo_pretrained.is_file():
        print(
            f"[YOLO11] Pretrained weights not found at {yolo_pretrained}. "
            "You can download a YOLO11 model (e.g. yolo11n.pt) and place it there."
        )

    spec = ExperimentSpec(
        name="yolo11_shanghaitech",
        model=ModelConfig(
            kind=ModelKind.YOLO11,
            weights=str(yolo_pretrained) if yolo_pretrained.is_file() else "yolo11n.pt",
            num_classes=1,
        ),
        data_yaml=str(data_yaml),
        project=str(base_dir / "runs_yolo"),
        use_wandb=use_wandb,
    )

    train_cfg = TrainConfig(
        epochs=50,
        batch_size=16,
        lr=0.01,
        device=device,
    )

    print("[YOLO11] Starting training...")
    engine.run_yolo_experiment(spec, train_cfg)

    # Copy best.pt to unified weights dir
    run_dir = Path(spec.project) / spec.name
    src_best = run_dir / "weights" / "best.pt"
    dst_best = weights_dir / "yolo11_best.pt"
    if src_best.is_file():
        shutil.copy2(src_best, dst_best)
        print(f"[YOLO11] Saved best weights to {dst_best}")
    else:
        print(f"[YOLO11] best.pt not found at {src_best} (Ultralytics layout may differ).")

    # Sanity-check visualization: run a single prediction and save image.
    try:
        val_paths = build_data_paths(base_dir)
        val_ds = ShanghaiTechDataset(
            image_dir=val_paths["val_images"],
            mat_dir=val_paths["val_gt"],
            target_type="points",
        )
        if len(val_ds) == 0:
            print("[YOLO11] No validation images for visualization.")
            return

        sample = val_ds[0]
        img = sample["image"].unsqueeze(0).to(device)

        # Load best YOLO weights if available
        best_cfg = ModelConfig(
            kind=ModelKind.YOLO11,
            weights=str(dst_best if dst_best.is_file() else spec.model.weights),
            num_classes=1,
        )
        yolo = create_model(best_cfg)
        yolo.model.to(device)

        with torch.no_grad():
            results = yolo.model(img)
        res = results[0]
        plotted = res.plot()  # returns an RGB image (numpy array)

        import cv2  # type: ignore

        out_path = logs_viz_dir / "yolo11_val_last.png"
        cv2.imwrite(str(out_path), plotted[:, :, ::-1])  # RGB -> BGR for OpenCV
        print(f"[YOLO11] Saved visualization to {out_path}")
    except Exception as exc:
        print(f"[YOLO11] Visualization failed (non-fatal): {exc}")


def train_density_model(
    name: str,
    model_kind: ModelKind,
    engine: TrainingEngine,
    device: str,
    base_dir: Path,
    dirs: dict[str, Path],
    use_wandb: bool,
) -> None:
    """
    Train CSRNet or P2PNet using the TrainingEngine's native loop.
    Handles:
    - Dataset creation
    - Dataloaders
    - WandB (opt-in)
    - best.pt checkpointing
    - per-epoch visualization (via engine)
    """
    assert model_kind in (ModelKind.CSRNET, ModelKind.P2PNET)

    data_paths = build_data_paths(base_dir)
    weights_dir = dirs["weights"] / name
    viz_dir = dirs["logs_viz"] / name

    target_type = "density" if model_kind == ModelKind.CSRNET else "points"

    train_ds = ShanghaiTechDataset(
        image_dir=data_paths["train_images"],
        mat_dir=data_paths["train_gt"],
        target_type=target_type,
    )
    val_ds = ShanghaiTechDataset(
        image_dir=data_paths["val_images"],
        mat_dir=data_paths["val_gt"],
        target_type=target_type,
    )

    train_cfg = TrainConfig(
        epochs=200 if model_kind == ModelKind.CSRNET else 150,
        batch_size=8,
        lr=1e-5 if model_kind == ModelKind.CSRNET else 1e-4,
        device=device,
        num_workers=4,
        log_every=10,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=("cuda" in device),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=("cuda" in device),
    )

    spec = ExperimentSpec(
        name=name,
        model=ModelConfig(kind=model_kind, out_channels=1),
        project=str(dirs["runs"] / name),
        use_wandb=use_wandb,
    )

    model = create_model(spec.model)

    print(f"[{name}] Starting training...")
    engine.run_density_experiment(
        spec=spec,
        train_cfg=train_cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=weights_dir,
        viz_dir=viz_dir,
        viz_every=1,
    )

    best_path = weights_dir / "best.pt"
    if best_path.is_file():
        print(f"[{name}] Saved best weights to {best_path}")
    else:
        print(f"[{name}] Warning: best.pt not found in {weights_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Server-side training orchestrator for Sanash (YOLO11, CSRNet, P2PNet)."
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging even if configured.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device string, e.g. "cuda" or "cpu". Defaults to CUDA if available.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = get_base_dir()
    dirs = ensure_dirs(base_dir)

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    use_wandb = not args.no_wandb

    print(f"[main] Base dir: {base_dir}")
    print(f"[main] Device: {device}")
    print(f"[main] WandB enabled: {use_wandb}")

    engine = TrainingEngine()

    # 1. YOLO11 detection training (optional if config/weights missing)
    train_yolo11(engine, device, base_dir, dirs, use_wandb=use_wandb)

    # 2. CSRNet density training
    train_density_model(
        name="csrnet_shanghaitech",
        model_kind=ModelKind.CSRNET,
        engine=engine,
        device=device,
        base_dir=base_dir,
        dirs=dirs,
        use_wandb=use_wandb,
    )

    # 3. P2PNet point training
    train_density_model(
        name="p2pnet_shanghaitech",
        model_kind=ModelKind.P2PNET,
        engine=engine,
        device=device,
        base_dir=base_dir,
        dirs=dirs,
        use_wandb=use_wandb,
    )

    print("[main] Training sequence completed.")


if __name__ == "__main__":
    main()

