"""PHASE 2 ASSET - DORMANT, NOT DEAD. Do not delete.

Unused in Phase 1 (door-mounted APC counting on PCDS reads depth video, not
extracted Gorelik colour frames). Active in Phase 2: RGB whole-frame cabin
classification. The sub-sequence splitting below stays relevant then, because
near-duplicate neighbouring frames leak across a random split. Dormant, not
gone.

Dataset and splitting for the extracted subset.

Splitting is by recording sub-sequence, not at random. Frames are ~0.067 s
apart, so neighbouring frames are near-duplicates; a random split would put
almost-identical images in train and val and report a meaninglessly good
score. The recording breaks into 11 sub-sequences at gaps > 5 s, and those
are the split unit.
"""

from __future__ import annotations

import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import TargetConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int, train: bool):
    from torchvision import transforms

    if train:
        # Deliberately mild: no horizontal flip, because cabin geometry is
        # fixed and a mirrored bus interior is not a real input.
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class OccupancyCountDataset(Dataset):
    """One extracted frame -> (image tensor, ordinal class, raw count)."""

    def __init__(
        self, root: str, rows: list[dict], target: TargetConfig, transform=None
    ):
        self.root = root
        self.rows = rows
        self.target = target
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        path = os.path.join(self.root, row["image"].replace("/", os.sep))
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        count = int(row[self.target.label_column])
        cls = self.target.encode(count)
        return (
            img,
            torch.tensor(cls, dtype=torch.long),
            torch.tensor(count, dtype=torch.long),
        )


def usable_rows(rows: list[dict], target: TargetConfig) -> tuple[list[dict], int]:
    """Drop rows without a label. Returns (kept, dropped)."""
    kept = [r for r in rows if r.get(target.label_column) is not None]
    return kept, len(rows) - len(kept)


def split_by_sequence(
    rows: list[dict], val_fraction: float = 0.25, seed: int = 0
) -> tuple[list[dict], list[dict], list[int]]:
    """Hold out whole sub-sequences for validation.

    Returns (train, val, held_out_sequence_ids). Sequences are assigned to val
    largest-first until the target fraction is met, which keeps the split
    deterministic and avoids a val set made only of tiny fragments.
    """
    import random

    by_seq: dict[int, list[dict]] = {}
    for r in rows:
        by_seq.setdefault(int(r["sequence"]), []).append(r)

    seq_ids = sorted(by_seq, key=lambda s: (-len(by_seq[s]), s))
    rng = random.Random(seed)
    rng.shuffle(seq_ids)

    target_n = val_fraction * len(rows)
    val_seqs: set[int] = set()
    running = 0
    for s in seq_ids:
        if running >= target_n:
            break
        val_seqs.add(s)
        running += len(by_seq[s])

    # Never let the split collapse to empty on either side.
    if not val_seqs or len(val_seqs) == len(seq_ids):
        val_seqs = {seq_ids[0]}

    train = [r for r in rows if int(r["sequence"]) not in val_seqs]
    val = [r for r in rows if int(r["sequence"]) in val_seqs]
    return train, val, sorted(val_seqs)
