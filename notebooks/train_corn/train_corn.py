#!/usr/bin/env python
"""Sanas branch 1: frozen backbone + CORN ordinal head, 1-epoch smoke test.

NEEDS GPU. Burns weekly Kaggle quota, so it does not run until Diyas approves.

What this is: a plumbing check that labels survive extraction and that an
ordinal head can be trained on them. It is NOT a claim about real bus
performance. The substitute dataset never contains more than four occupants
and was recorded in a single 32-minute daytime session, so nothing here
transfers to a crowded or night-time cabin.

Target: RAW COUNT (0-4). No 0-1 normalisation and no 5-level mapping yet, per
the decision recorded in experiments/log.md. Switching to a normalised target
later is a config change: pass --target-mode normalized --capacity N and the
head width, loss and decoder all follow from TargetConfig.

Backbones (both permissively licensed, checked 2026-08-09):
  dinov2_vits14  timm vit_small_patch14_dinov2.lvd142m, Apache-2.0 code+weights
  convnext_tiny  timm convnext_tiny.fb_in1k, MIT / Apache-2.0  (edge comparison)

Prints the train/val split sizes and the count distribution it actually sees,
so the labels can be confirmed end to end.

NOTE ON DUPLICATION: canonical implementations are src/sanas/{config,corn,
models,data}.py. Kaggle pushes one file per kernel, so this is a self-contained
vendored copy. Change src/ first, then re-sync the marked block.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ==========================================================================
# BEGIN VENDORED BLOCK - canonical: src/sanas/{config,corn,models,data}.py
# ==========================================================================

RAW_COUNT = "raw_count"
NORMALIZED = "normalized"


@dataclass
class TargetConfig:
    mode: str = RAW_COUNT
    label_column: str = "count_view"
    max_count: int = 4
    capacity: int | None = None
    num_levels: int | None = 5
    level_edges: list = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])

    def __post_init__(self):
        if self.mode not in (RAW_COUNT, NORMALIZED):
            raise ValueError(f"unknown target mode {self.mode!r}")
        if self.mode == NORMALIZED:
            if not self.capacity or self.capacity <= 0:
                raise ValueError("normalized mode needs a positive capacity")
            if len(self.level_edges) != int(self.num_levels) - 1:
                raise ValueError("level_edges must have num_levels-1 entries")

    @property
    def num_classes(self):
        return self.max_count + 1 if self.mode == RAW_COUNT else int(self.num_levels)

    @property
    def num_thresholds(self):
        return self.num_classes - 1

    def encode(self, count):
        if self.mode == RAW_COUNT:
            return max(0, min(int(count), self.max_count))
        frac = count / self.capacity
        return min(sum(1 for e in self.level_edges if frac >= e), self.num_classes - 1)

    def decode(self, cls):
        cls = max(0, min(int(cls), self.num_classes - 1))
        if self.mode == RAW_COUNT:
            return float(cls)
        return float(([0.0] + list(self.level_edges))[cls])

    def describe(self):
        if self.mode == RAW_COUNT:
            return (
                f"raw_count on '{self.label_column}', classes 0..{self.max_count} "
                f"({self.num_classes} classes, {self.num_thresholds} CORN thresholds)"
            )
        return (
            f"normalized on '{self.label_column}', capacity={self.capacity}, "
            f"{self.num_levels} levels, edges={self.level_edges}"
        )


def corn_loss(logits, targets, num_classes):
    """K-1 binary tasks; task k trained only on samples with y >= k."""
    targets = targets.long()
    total = logits.new_zeros(())
    used = 0
    for k in range(num_classes - 1):
        mask = targets >= k
        if not bool(mask.any()):
            continue
        binary = (targets[mask] > k).float()
        total = total + F.binary_cross_entropy_with_logits(logits[mask, k], binary)
        used += 1
    return total / max(used, 1)


def corn_cumulative_probs(logits):
    return torch.cumprod(torch.sigmoid(logits), dim=1)


def corn_predict(logits, threshold=0.5):
    return (corn_cumulative_probs(logits) > threshold).sum(dim=1)


def corn_expected_class(logits):
    return corn_cumulative_probs(logits).sum(dim=1)


BACKBONES = {
    "dinov2_vits14": {
        "timm_name": "vit_small_patch14_dinov2.lvd142m",
        "input_size": 224,
        "license": "Apache-2.0 (code and weights)",
    },
    "convnext_tiny": {
        "timm_name": "convnext_tiny.fb_in1k",
        "input_size": 224,
        "license": "MIT (ConvNeXt) / Apache-2.0 (timm)",
    },
}


class OrdinalHead(nn.Module):
    def __init__(self, in_features, num_thresholds, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_thresholds)
        nn.init.zeros_(self.fc.bias)

    def forward(self, feats):
        return self.fc(self.drop(self.norm(feats)))


class FrozenBackboneOrdinal(nn.Module):
    def __init__(self, backbone, head, frozen):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.frozen = frozen

    def forward(self, x):
        if self.frozen:
            with torch.no_grad():
                feats = self.backbone(x)
        else:
            feats = self.backbone(x)
        return self.head(feats.float())


def build_model(backbone, num_thresholds, pretrained=True, freeze=True, dropout=0.1):
    import timm

    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}")
    info = dict(BACKBONES[backbone])
    net = timm.create_model(info["timm_name"], pretrained=pretrained, num_classes=0)
    if freeze:
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
    dim = net.num_features
    model = FrozenBackboneOrdinal(
        net, OrdinalHead(dim, num_thresholds, dropout), freeze
    )
    info.update(
        backbone_key=backbone,
        feature_dim=dim,
        frozen=freeze,
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_params=sum(p.numel() for p in model.parameters()),
    )
    return model, info


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size, train):
    from torchvision import transforms

    common = [transforms.Resize((img_size, img_size))]
    if train:
        # No horizontal flip: cabin geometry is fixed, a mirrored bus is not a
        # real input. Photometric jitter only.
        common.append(
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)
        )
    common += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(common)


class OccupancyCountDataset(Dataset):
    def __init__(self, root, rows, target, transform=None):
        self.root, self.rows, self.target, self.transform = (
            root,
            rows,
            target,
            transform,
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(
            os.path.join(self.root, row["image"].replace("/", os.sep))
        ).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        count = int(row[self.target.label_column])
        return (
            img,
            torch.tensor(self.target.encode(count), dtype=torch.long),
            torch.tensor(count),
        )


def split_by_sequence(rows, val_fraction=0.25, seed=0):
    """Hold out whole sub-sequences. Frames are ~0.067 s apart, so a random
    split would put near-duplicate neighbours on both sides and flatter us."""
    by_seq = {}
    for r in rows:
        by_seq.setdefault(int(r["sequence"]), []).append(r)
    seq_ids = sorted(by_seq, key=lambda s: (-len(by_seq[s]), s))
    random.Random(seed).shuffle(seq_ids)
    target_n, running, val_seqs = val_fraction * len(rows), 0, set()
    for s in seq_ids:
        if running >= target_n:
            break
        val_seqs.add(s)
        running += len(by_seq[s])
    if not val_seqs or len(val_seqs) == len(seq_ids):
        val_seqs = {seq_ids[0]}
    train = [r for r in rows if int(r["sequence"]) not in val_seqs]
    val = [r for r in rows if int(r["sequence"]) in val_seqs]
    return train, val, sorted(val_seqs)


# ==========================================================================
# END VENDORED BLOCK
# ==========================================================================


def default_data_dir():
    for c in (
        "/kaggle/input/sanas-extract-subset/subset",
        "/kaggle/input/sanas-extract-subset",
        os.path.join("data", "subset"),
    ):
        if os.path.isdir(c):
            return c
    return os.path.join("data", "subset")


def distribution(rows, column):
    d = {}
    for r in rows:
        k = "missing" if r.get(column) is None else str(r[column])
        d[k] = d.get(k, 0) + 1
    return dict(sorted(d.items(), key=lambda kv: (kv[0] == "missing", kv[0])))


def evaluate(model, loader, target, device):
    model.eval()
    preds, gts, soft = [], [], []
    with torch.no_grad():
        for imgs, cls, _count in loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            preds.append(corn_predict(logits).cpu())
            soft.append(corn_expected_class(logits).cpu())
            gts.append(cls)
    if not preds:
        return {}
    p = torch.cat(preds).numpy()
    g = torch.cat(gts).numpy()
    s = torch.cat(soft).numpy()
    n = len(g)
    k = target.num_classes
    cm = np.zeros((k, k), dtype=int)
    for a, b in zip(g, p):
        cm[int(a), int(b)] += 1
    return {
        "n": int(n),
        "exact_match_accuracy": float((p == g).mean()),
        "within_one_accuracy": float((np.abs(p - g) <= 1).mean()),
        "mae_classes": float(np.abs(p - g).mean()),
        "mae_soft": float(np.abs(s - g).mean()),
        "confusion_matrix": cm.tolist(),
        "pred_distribution": {str(i): int((p == i).sum()) for i in range(k)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Frozen backbone + CORN ordinal head (1-epoch smoke test)"
    )
    ap.add_argument("--data", default=None)
    ap.add_argument("--backbone", default="dinov2_vits14", choices=sorted(BACKBONES))
    ap.add_argument(
        "--compare-backbone",
        default="convnext_tiny",
        help="second backbone to train identically; '' to skip",
    )
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--val-fraction", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-mode", default=RAW_COUNT, choices=[RAW_COUNT, NORMALIZED])
    ap.add_argument(
        "--label-column", default="count_view", choices=["count_view", "count_cabin"]
    )
    ap.add_argument("--max-count", type=int, default=4)
    ap.add_argument(
        "--capacity", type=int, default=None, help="only for --target-mode normalized"
    )
    ap.add_argument(
        "--num-levels", type=int, default=5, help="only for --target-mode normalized"
    )
    ap.add_argument("--limit", type=int, default=0, help="cap frames, for debugging")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    data = args.data or default_data_dir()
    work = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    out_path = args.out or os.path.join(work, "train_metrics.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target = TargetConfig(
        mode=args.target_mode,
        label_column=args.label_column,
        max_count=args.max_count,
        capacity=args.capacity,
        num_levels=args.num_levels,
    )

    print("=" * 72)
    print("Sanas branch 1: frozen backbone + CORN ordinal head")
    print("=" * 72)
    print(f"data    : {data}")
    print(
        f"device  : {device}  ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})"
    )
    print(f"target  : {target.describe()}")
    print(f"torch   : {torch.__version__}")

    labels_path = os.path.join(data, "labels.jsonl")
    if not os.path.exists(labels_path):
        print(
            f"missing {labels_path}. Run the extraction kernel first.", file=sys.stderr
        )
        return 1
    with open(labels_path, encoding="utf-8") as fh:
        rows = [json.loads(x) for x in fh if x.strip()]

    total_rows = len(rows)
    rows = [r for r in rows if r.get(target.label_column) is not None]
    dropped = total_rows - len(rows)
    if args.limit:
        rows = rows[: args.limit]

    train_rows, val_rows, val_seqs = split_by_sequence(
        rows, args.val_fraction, args.seed
    )

    print("\n" + "-" * 72)
    print("LABELS AS ACTUALLY SEEN BY THE TRAINER")
    print("-" * 72)
    print(f"rows in labels.jsonl      : {total_rows:,}")
    print(f"dropped (no label)        : {dropped:,}")
    print(f"usable                    : {len(rows):,}")
    print(f"sub-sequences             : {len({int(r['sequence']) for r in rows})}")
    print(f"train frames              : {len(train_rows):,}")
    print(
        f"val frames                : {len(val_rows):,}   (held-out sequences {val_seqs})"
    )
    print(f"count dist  (all)         : {distribution(rows, target.label_column)}")
    print(
        f"count dist  (train)       : {distribution(train_rows, target.label_column)}"
    )
    print(f"count dist  (val)         : {distribution(val_rows, target.label_column)}")
    print(
        f"class dist  (train)       : "
        f"{distribution([{'c': target.encode(int(r[target.label_column]))} for r in train_rows], 'c')}"
    )
    print(f"other column (count_cabin): {distribution(rows, 'count_cabin')}")
    if not train_rows or not val_rows:
        print("empty split; aborting", file=sys.stderr)
        return 1

    results = {}
    backbones = [args.backbone] + (
        [args.compare_backbone] if args.compare_backbone else []
    )
    for bb in backbones:
        print("\n" + "=" * 72)
        print(f"BACKBONE: {bb}")
        print("=" * 72)
        model, info = build_model(bb, target.num_thresholds)
        model.to(device)
        print(f"  timm name        : {info['timm_name']}")
        print(f"  licence          : {info['license']}")
        print(f"  feature dim      : {info['feature_dim']}")
        print(f"  total params     : {info['total_params']:,}")
        print(f"  trainable params : {info['trainable_params']:,} (head only)")

        tr = DataLoader(
            OccupancyCountDataset(
                data, train_rows, target, build_transforms(args.img_size, True)
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        va = DataLoader(
            OccupancyCountDataset(
                data, val_rows, target, build_transforms(args.img_size, False)
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
        )

        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        t0 = time.time()
        for epoch in range(args.epochs):
            model.train()
            if model.frozen:
                model.backbone.eval()  # keep frozen BN/stat layers frozen too
            running, seen = 0.0, 0
            for i, (imgs, cls, _c) in enumerate(tr, 1):
                imgs = imgs.to(device, non_blocking=True)
                cls = cls.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = model(imgs)
                    loss = corn_loss(logits.float(), cls, target.num_classes)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                running += loss.item() * imgs.size(0)
                seen += imgs.size(0)
                if i % 10 == 0 or i == len(tr):
                    print(
                        f"  epoch {epoch + 1} step {i}/{len(tr)}  loss {running / seen:.4f}",
                        flush=True,
                    )
            print(
                f"  epoch {epoch + 1} done in {time.time() - t0:.0f}s, train loss {running / seen:.4f}"
            )

        metrics = evaluate(model, va, target, device)
        metrics.update(
            backbone=bb,
            timm_name=info["timm_name"],
            license=info["license"],
            trainable_params=info["trainable_params"],
            seconds=round(time.time() - t0, 1),
        )
        print("\n  VALIDATION (held-out sequences)")
        print(f"    exact match      : {metrics['exact_match_accuracy']:.3f}")
        print(f"    within +/-1      : {metrics['within_one_accuracy']:.3f}")
        print(f"    MAE (classes)    : {metrics['mae_classes']:.3f}")
        print(f"    MAE (soft score) : {metrics['mae_soft']:.3f}")
        print(f"    pred dist        : {metrics['pred_distribution']}")
        print("    confusion (rows=true, cols=pred):")
        for r_i, row in enumerate(metrics["confusion_matrix"]):
            print(f"      {r_i}: {row}")
        results[bb] = metrics

        torch.save(
            {
                "head": model.head.state_dict(),
                "backbone": bb,
                "target": target.__dict__,
            },
            os.path.join(work, f"head_{bb}.pt"),
        )

    payload = {
        "branch": "rgb_frozen_backbone_corn",
        "trained": True,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "img_size": args.img_size,
        "target": target.__dict__,
        "data_dir": data,
        "frames_total": total_rows,
        "frames_used": len(rows),
        "train_frames": len(train_rows),
        "val_frames": len(val_rows),
        "val_sequences": val_seqs,
        "count_distribution_all": distribution(rows, target.label_column),
        "count_distribution_train": distribution(train_rows, target.label_column),
        "count_distribution_val": distribution(val_rows, target.label_column),
        "results": results,
        "caveats": [
            "Pipeline check only, not a claim about real bus performance.",
            "Max 4 occupants anywhere in this dataset; upper occupancy levels have zero examples.",
            "Single 32-minute daytime session: low-light robustness is untestable here.",
            "Ground truth is model-generated pseudo-labels, not human annotation.",
        ],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {out_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
