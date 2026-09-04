"""Train the DISCO counting baseline on the CPU.

Usage from `development/src`:

    python -m sanas_baseline.prepare --train 16 --val 8 --test 8
    python -m sanas_baseline.train

Defaults are a smoke test: a handful of images and two epochs, enough to prove
the pipeline runs end to end and writes an artifact. The numbers it prints are
a wiring check, not model quality, and must not be reported as counting
performance. See development/src/README.md.
"""

import argparse
import json
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sanas_baseline.config import BaselineConfig
from sanas_baseline.data import read_manifest
from sanas_baseline.evaluate import build_loader, evaluate_split
from sanas_baseline.model import ConstantPredictor, DensityCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def train_one_epoch(model, loader, optimizer, criterion, cfg) -> float:
    model.train()
    total, seen = 0.0, 0
    for images, targets, _ in loader:
        images = images.to(cfg.device)
        targets = targets.to(cfg.device) * cfg.density_scale
        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        total += loss.detach().item() * images.shape[0]
        seen += images.shape[0]
    return total / seen


def main() -> None:
    cfg = BaselineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.learning_rate)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--device", default=cfg.device)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    cfg.epochs, cfg.batch_size = args.epochs, args.batch_size
    cfg.learning_rate, cfg.seed, cfg.device = args.lr, args.seed, args.device
    if cfg.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit(
            f"device {cfg.device} requested but no CUDA device is available"
        )

    set_seed(cfg.seed)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = cfg.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_loader(cfg, "train", shuffle=True)
    val_loader = build_loader(cfg, "val")
    train_counts = train_loader.dataset.counts()
    constant = ConstantPredictor().fit(train_counts)

    model = DensityCNN().to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()

    print(f"run id: {run_id}")
    print(
        f"device: {cfg.device}  torch {torch.__version__}  python {platform.python_version()}"
    )
    print(f"train {len(train_loader.dataset)} / val {len(val_loader.dataset)} images")
    print(f"image {cfg.image_height}x{cfg.image_width}  density {cfg.density_hw()}")
    print(f"train mean count: {constant.value:.2f}")

    history, started = [], time.time()
    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg)
        val = evaluate_split(model, val_loader, cfg, constant)
        history.append({"epoch": epoch, "train_loss": loss, **val})
        print(
            f"epoch {epoch}: loss {loss:.4f}  "
            f"val MAE {val['model_mae']:.2f} vs constant {val['constant_mae']:.2f}  "
            f"({time.time() - epoch_start:.1f}s)"
        )

    torch.save(
        {
            "model": model.state_dict(),
            "constant_value": constant.value,
            "run_id": run_id,
        },
        run_dir / "checkpoint.pt",
    )
    results = {
        "run_id": run_id,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "backend": f"local cpu, torch {torch.__version__}, python {platform.python_version()}",
        "config": json.loads(cfg.to_json()),
        "data_manifest": read_manifest(cfg.subset_dir),
        "train_mean_count": constant.value,
        "history": history,
        "final_val": history[-1],
        "wall_seconds": round(time.time() - started, 1),
        "disclaimer": (
            "Smoke test on a subset of DISCO, a public outdoor crowd dataset. "
            "Not bus cabin data, not a validated model, not product performance."
        ),
    }
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8"
    )
    print(f"wall time: {results['wall_seconds']}s")
    print(f"wrote {run_dir / 'checkpoint.pt'}")
    print(f"wrote {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
