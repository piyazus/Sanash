"""
Experiment runner for fine-tuning a YOLO model on the adapted crowd dataset.

Responsibilities:
- Configure and launch training using Ultralytics YOLO (v8+).
- Automatically integrate Weights & Biases (wandb) for:
  - training loss curves
  - validation metrics including mean Average Precision (mAP)

This module is intentionally minimal but fully runnable, serving as a
baseline for more advanced experiment management (sweeps, multiple models, etc.).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import wandb
from ultralytics import YOLO


@dataclass
class ExperimentConfig:
    """Configuration for a single YOLO fine-tuning run."""

    model_path: str  # path to .pt YOLO checkpoint (e.g., 'yolov8n.pt')
    data_yaml: str  # path to data.yaml describing train/val/test

    # Training hyperparameters
    epochs: int = 50
    batch: int = 16
    img_size: int = 640
    lr0: float = 0.01
    device: str = "0"  # "cpu" or CUDA device string

    # Logging / output
    project: str = "sanash-yolo"
    name: str = "exp"
    save: bool = True
    exist_ok: bool = False


class YoloExperimentRunner:
    """
    Simple wrapper around Ultralytics YOLO training with wandb integration.

    Usage
    -----
    runner = YoloExperimentRunner(config, wandb_project="sanash-cv")
    runner.run()
    """

    def __init__(
        self,
        config: ExperimentConfig,
        wandb_project: str = "sanash-cv",
        wandb_entity: Optional[str] = None,
        use_wandb: bool = True,
    ) -> None:
        self.config = config
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.use_wandb = use_wandb
        self._wandb_run: Optional[wandb.sdk.wandb_run.Run] = None

    def _init_wandb(self) -> None:
        if not self.use_wandb:
            return

        self._wandb_run = wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            config=asdict(self.config),
            name=self.config.name,
        )

    def _finish_wandb(self) -> None:
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None

    def run(self) -> None:
        """Launch YOLO training and log metrics to wandb."""
        self._init_wandb()

        try:
            model = YOLO(self.config.model_path)

            # Ultralytics has native wandb integration. Ensuring WANDB project name
            # is propagated for consistency with manual logging.
            if self.use_wandb:
                # These env vars are respected by Ultralytics if set.
                import os

                os.environ.setdefault("WANDB_PROJECT", self.wandb_project)
                if self.wandb_entity:
                    os.environ.setdefault("WANDB_ENTITY", self.wandb_entity)

            # Start training
            results = model.train(
                data=self.config.data_yaml,
                epochs=self.config.epochs,
                batch=self.config.batch,
                imgsz=self.config.img_size,
                lr0=self.config.lr0,
                device=self.config.device,
                project=self.config.project,
                name=self.config.name,
                save=self.config.save,
                exist_ok=self.config.exist_ok,
            )

            # Optionally log any aggregate loss information if available
            if self.use_wandb and results is not None:
                # `results` may be a dict-like object depending on ultralytics version.
                try:
                    if isinstance(results, dict):
                        loss_keys = [k for k in results.keys() if "loss" in k]
                        loss_payload = {f"loss/{k}": results[k] for k in loss_keys}
                        if loss_payload:
                            wandb.log(loss_payload)
                except Exception:
                    # If structure is unexpected, let built-in integration handle details.
                    pass

            # Evaluate and log mAP metrics explicitly
            metrics = model.val(data=self.config.data_yaml, split="val")
            if self.use_wandb and metrics is not None:
                # Ultralytics metrics expose box.map, box.map50, etc.
                payload = {}
                try:
                    payload["metrics/mAP"] = float(getattr(metrics.box, "map", 0.0))
                    payload["metrics/mAP50"] = float(
                        getattr(metrics.box, "map50", payload["metrics/mAP"])
                    )
                    payload["metrics/mAP75"] = float(
                        getattr(metrics.box, "map75", payload["metrics/mAP"])
                    )
                except Exception:
                    # Fallback to dict-like structure
                    try:
                        payload["metrics/mAP"] = float(metrics.get("metrics/mAP", 0.0))
                        payload["metrics/mAP50"] = float(
                            metrics.get("metrics/mAP50", payload["metrics/mAP"])
                        )
                    except Exception:
                        payload = {}

                if payload:
                    wandb.log(payload)

        finally:
            self._finish_wandb()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a YOLO fine-tuning experiment with wandb logging."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to YOLO model checkpoint (e.g., yolov8n.pt).",
    )
    parser.add_argument(
        "--data_yaml",
        type=str,
        required=True,
        help="Path to data.yaml describing train/val/test splits.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--img_size", type=int, default=640, help="Training image size.")
    parser.add_argument(
        "--lr0", type=float, default=0.01, help="Initial learning rate (Ultralytics lr0)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help='Device string, e.g. "0" for first GPU or "cpu".',
    )
    parser.add_argument(
        "--project",
        type=str,
        default="sanash-yolo",
        help="Ultralytics project name (output directory).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="exp",
        help="Experiment/run name for both Ultralytics and wandb.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="sanash-cv",
        help="wandb project name.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="wandb entity (team/user).",
    )
    parser.add_argument(
        "--exist_ok",
        action="store_true",
        help="Allow existing project/name directory without incrementing version.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ExperimentConfig(
        model_path=args.model_path,
        data_yaml=args.data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        img_size=args.img_size,
        lr0=args.lr0,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
    )

    runner = YoloExperimentRunner(
        config=config,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        use_wandb=not args.no_wandb,
    )
    runner.run()


if __name__ == "__main__":
    main()

