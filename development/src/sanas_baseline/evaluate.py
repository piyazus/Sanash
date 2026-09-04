"""Evaluate a saved baseline checkpoint against the constant predictor.

Usage from `development/src`:

    python -m sanas_baseline.evaluate --checkpoint ../../outputs/baseline/<run>/checkpoint.pt --split val

The comparison against `ConstantPredictor` is the point of the script. An MAE
on its own says nothing; an MAE next to the MAE of "always answer the train
mean" says whether the network used the image at all.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sanas_baseline.config import BaselineConfig
from sanas_baseline.data import DiscoDensityDataset
from sanas_baseline.metrics import mae, rmse
from sanas_baseline.model import ConstantPredictor, DensityCNN, counts_from_density


def build_loader(cfg: BaselineConfig, split: str, shuffle: bool = False) -> DataLoader:
    dataset = DiscoDensityDataset(
        cfg.subset_dir, split, (cfg.image_height, cfg.image_width), cfg.density_hw()
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    return loader


@torch.no_grad()
def predict_counts(model: DensityCNN, loader: DataLoader, cfg: BaselineConfig):
    """Predicted and true per-image counts, in people, for one split."""
    model.eval()
    pred, true = [], []
    for images, _, counts in loader:
        density = model(images.to(cfg.device))
        pred.append(counts_from_density(density, cfg.density_scale).cpu().numpy())
        true.append(counts.numpy())
    return np.concatenate(pred), np.concatenate(true)


def evaluate_split(model, loader, cfg, constant: ConstantPredictor) -> dict:
    pred, true = predict_counts(model, loader, cfg)
    const = np.asarray(constant.predict(len(true)), dtype=np.float64)
    return {
        "n": int(len(true)),
        "true_mean_count": float(true.mean()),
        "pred_mean_count": float(pred.mean()),
        "model_mae": mae(pred, true),
        "model_rmse": rmse(pred, true),
        "constant_value": constant.value,
        "constant_mae": mae(const, true),
        "constant_rmse": rmse(const, true),
    }


def main() -> None:
    cfg = BaselineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DensityCNN().to(cfg.device)
    model.load_state_dict(state["model"])
    constant = ConstantPredictor(state["constant_value"])

    loader = build_loader(cfg, args.split)
    result = evaluate_split(model, loader, cfg, constant)
    result["split"] = args.split
    result["checkpoint"] = str(args.checkpoint)

    print(f"split: {args.split}  n={result['n']}")
    print(f"true mean count: {result['true_mean_count']:.2f}")
    print(f"model      MAE {result['model_mae']:.2f}  RMSE {result['model_rmse']:.2f}")
    print(
        f"constant   MAE {result['constant_mae']:.2f}  RMSE {result['constant_rmse']:.2f}"
    )
    beats = result["model_mae"] < result["constant_mae"]
    print(f"model beats constant predictor: {beats}")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
