"""
Unified training and benchmarking engine for Sanash models.

Supports:
- Ultralytics YOLO11 for detection-based head localization.
- P2PNet-style point prediction.
- CSRNet-style density map regression.

The engine provides:
- Simple configuration dataclasses.
- Native PyTorch training loops for P2PNet and CSRNet.
- A thin adapter around Ultralytics' training API for YOLO11.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.factory import (
    BaseCrowdModel,
    ModelKind,
    TaskType,
    YOLO11Wrapper,
    create_model,
    ModelConfig,
)

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore
    np = None  # type: ignore


@dataclass
class TrainConfig:
    """Generic training hyperparameters."""

    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    log_every: int = 10


@dataclass
class ExperimentSpec:
    """
    High-level description of an experiment for a single model.

    For YOLO11 experiments, `data_yaml` should point to a YOLO-style dataset
    definition. For P2PNet/CSRNet, dataloaders are supplied separately.
    """

    name: str
    model: ModelConfig

    # For YOLO
    data_yaml: Optional[str] = None

    # Logging
    project: str = "sanash-benchmark"
    use_wandb: bool = False


class TrainingEngine:
    """
    Orchestrates training and evaluation for different models.

    Usage patterns
    --------------
    - YOLO11:
        engine.run_yolo_experiment(spec, train_cfg)

    - CSRNet / P2PNet:
        model = create_model(spec.model)
        engine.run_density_experiment(
            spec, train_cfg, model, train_loader, val_loader
        )
    """

    def __init__(self) -> None:
        pass

    # ---- YOLO11 branch -------------------------------------------------

    def run_yolo_experiment(self, spec: ExperimentSpec, train_cfg: TrainConfig) -> Any:
        """
        Launch YOLO11 training using Ultralytics' native trainer.

        Parameters
        ----------
        spec : ExperimentSpec
            With `model.kind == ModelKind.YOLO11` and `data_yaml` defined.
        train_cfg : TrainConfig
            Generic hyperparameters; translated to YOLO kwargs.
        """
        if spec.model.kind is not ModelKind.YOLO11:
            raise ValueError("run_yolo_experiment requires a YOLO11 model config.")
        if not spec.data_yaml:
            raise ValueError("YOLO11 experiment requires `data_yaml` in ExperimentSpec.")

        yolo: YOLO11Wrapper = create_model(spec.model)  # type: ignore[assignment]

        yolo_kwargs: Dict[str, Any] = {
            "data": spec.data_yaml,
            "epochs": train_cfg.epochs,
            "batch": train_cfg.batch_size,
            "lr0": train_cfg.lr,
            "device": train_cfg.device,
            "project": spec.project,
            "name": spec.name,
        }

        return yolo.train(**yolo_kwargs)

    # ---- P2PNet / CSRNet branch ---------------------------------------

    def _init_wandb(self, spec: ExperimentSpec, train_cfg: TrainConfig) -> Optional[Any]:
        if not spec.use_wandb or wandb is None:
            return None
        run = wandb.init(
            project=spec.project,
            name=spec.name,
            config={**asdict(train_cfg), "model": asdict(spec.model)},
        )
        return run

    def _log_wandb(self, payload: Dict[str, Any]) -> None:
        if wandb is None:
            return
        wandb.log(payload)

    def _finish_wandb(self, run: Optional[Any]) -> None:
        if run is not None:
            run.finish()

    def run_density_experiment(
        self,
        spec: ExperimentSpec,
        train_cfg: TrainConfig,
        model: BaseCrowdModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: Optional[str | Path] = None,
        viz_dir: Optional[str | Path] = None,
        viz_every: int = 1,
    ) -> Dict[str, Any]:
        """
        Train P2PNet / CSRNet-style models with a native PyTorch loop.

        Assumes:
        - Input batches are dicts with keys:
            - 'image': Tensor[N, 3, H, W]
            - 'target': Tensor[N, 1, H', W']  (density or point map)
        - Loss: MSE between prediction and target.
        """
        device = torch.device(train_cfg.device)
        model = model.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
        )
        criterion = nn.MSELoss()

        wandb_run = self._init_wandb(spec, train_cfg)

        global_step = 0
        history: Dict[str, Any] = {"train_loss": [], "val_loss": []}

        best_val_loss: Optional[float] = None
        ckpt_path: Optional[Path] = None
        if checkpoint_dir is not None:
            ckpt_path = Path(checkpoint_dir)
            ckpt_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(train_cfg.epochs):
            model.train()
            running_loss = 0.0

            for i, batch in enumerate(train_loader):
                images = batch["image"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                preds = model(images)
                loss = criterion(preds, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                global_step += 1

                if (i + 1) % train_cfg.log_every == 0:
                    avg_loss = running_loss / train_cfg.log_every
                    history["train_loss"].append(avg_loss)
                    print(
                        f"[{spec.name}] Epoch {epoch+1}/{train_cfg.epochs} "
                        f"Step {i+1}/{len(train_loader)} "
                        f"Loss={avg_loss:.4f}"
                    )
                    if wandb_run is not None:
                        self._log_wandb({"train/loss": avg_loss, "step": global_step})
                    running_loss = 0.0

            # Validation
            val_loss = None
            if val_loader is not None:
                model.eval()
                val_running = 0.0
                first_images = None
                first_preds = None
                first_target = None
                with torch.no_grad():
                    for j, batch in enumerate(val_loader):
                        images = batch["image"].to(device, non_blocking=True)
                        target = batch["target"].to(device, non_blocking=True)
                        preds = model(images)
                        loss = criterion(preds, target)
                        val_running += loss.item()

                        if (
                            viz_dir is not None
                            and plt is not None
                            and np is not None
                            and (epoch + 1) % max(1, viz_every) == 0
                            and j == 0
                        ):
                            # Capture the first batch of the epoch for visualization
                            first_images = images.detach().cpu()
                            first_preds = preds.detach().cpu()
                            first_target = target.detach().cpu()

                val_loss = val_running / max(1, len(val_loader))
                history["val_loss"].append(val_loss)
                print(
                    f"[{spec.name}] Epoch {epoch+1}/{train_cfg.epochs} "
                    f"ValLoss={val_loss:.4f}"
                )
                if wandb_run is not None:
                    self._log_wandb({"val/loss": val_loss, "epoch": epoch + 1})

                # Checkpoint best model
                if ckpt_path is not None:
                    if best_val_loss is None or val_loss < best_val_loss:
                        best_val_loss = float(val_loss)
                        torch.save(model.state_dict(), ckpt_path / "best.pt")

                # Save visualization for this epoch
                if (
                    viz_dir is not None
                    and plt is not None
                    and np is not None
                    and first_images is not None
                    and first_preds is not None
                    and first_target is not None
                ):
                    self._save_density_visualization(
                        images=first_images,
                        preds=first_preds,
                        target=first_target,
                        spec=spec,
                        epoch=epoch + 1,
                        viz_dir=viz_dir,
                    )

        self._finish_wandb(wandb_run)
        return history

    def _save_density_visualization(
        self,
        images: torch.Tensor,
        preds: torch.Tensor,
        target: torch.Tensor,
        spec: ExperimentSpec,
        epoch: int,
        viz_dir: str | Path,
    ) -> None:
        """
        Save a side-by-side visualization of:
        - input image
        - predicted map (density or points)
        - ground truth map
        """
        if plt is None or np is None:  # pragma: no cover - optional dependency
            return

        viz_dir_path = Path(viz_dir)
        viz_dir_path.mkdir(parents=True, exist_ok=True)

        img = images[0].permute(1, 2, 0).cpu().numpy()
        pred_map = preds[0, 0].cpu().numpy()
        tgt_map = target[0, 0].cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(np.clip(img, 0, 1))
        axes[0].set_title("Image")
        axes[0].axis("off")

        im1 = axes[1].imshow(pred_map, cmap="magma")
        axes[1].set_title("Prediction")
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(tgt_map, cmap="magma")
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        out_path = viz_dir_path / f"{spec.name}_epoch{epoch:03d}.png"
        fig.suptitle(f"{spec.name} - Epoch {epoch}")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


__all__ = [
    "TrainConfig",
    "ExperimentSpec",
    "TrainingEngine",
]

