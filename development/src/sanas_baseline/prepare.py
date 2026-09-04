"""Extract a working subset of DISCO from the two zip archives.

`disco_images.zip` is 2.1 GB and holds 8,116 JPEGs; only 1,935 of them carry a
density map. Unpacking the whole archive to train on 1,935 supervised samples
is wasteful, so this script reads the zip central directory and pulls out only
the members it needs.

The train/val/test split is the one shipped inside `disco_density_maps.zip`
(train/ 1,435, val/ 200, test/ 300). It is read from the archive, not invented
here. Selection inside a split is a deterministic shuffle under `--seed`, so
the same flags always produce the same subset.

Each sample is written as `<id>.jpg` (raw archive bytes, untouched) and
`<id>.npy` (the MATLAB `map` variable cast to float32). The cast is a storage
format change only: no resampling, no cropping, no rescaling, so the per-image
count, which is the sum of the map, survives it. Resizing happens in data.py.
"""

import argparse
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.io as sio

from sanas_baseline.config import BaselineConfig

SPLITS = ("train", "val", "test")


def split_ids(density_zip: Path) -> dict[str, list[str]]:
    """Image ids per split, taken from the directory layout of the archive."""
    with zipfile.ZipFile(density_zip) as z:
        names = [n for n in z.namelist() if n.endswith(".mat")]
    out: dict[str, list[str]] = {s: [] for s in SPLITS}
    for name in names:
        split, stem = name.split("/", 1)
        if split not in out:
            raise ValueError(f"unexpected split directory in archive: {name}")
        out[split].append(stem[: -len(".mat")])
    return {s: sorted(ids) for s, ids in out.items()}


def select(ids: list[str], limit: int | None, seed: int) -> list[str]:
    if limit is None or limit >= len(ids):
        return ids
    rng = np.random.default_rng(seed)
    picked = rng.permutation(len(ids))[:limit]
    return sorted(ids[i] for i in picked)


def extract(cfg: BaselineConfig, limits: dict[str, int | None], seed: int) -> dict:
    ids = split_ids(cfg.density_zip)
    chosen = {s: select(ids[s], limits[s], seed) for s in SPLITS}

    with (
        zipfile.ZipFile(cfg.images_zip) as zimg,
        zipfile.ZipFile(cfg.density_zip) as zden,
    ):
        image_members = set(zimg.namelist())
        for split in SPLITS:
            out_dir = cfg.subset_dir / split
            out_dir.mkdir(parents=True, exist_ok=True)
            for image_id in chosen[split]:
                member = f"imgs/{image_id}.jpg"
                if member not in image_members:
                    raise FileNotFoundError(
                        f"{member} missing from {cfg.images_zip.name}"
                    )
                (out_dir / f"{image_id}.jpg").write_bytes(zimg.read(member))
                mat = sio.loadmat(io.BytesIO(zden.read(f"{split}/{image_id}.mat")))
                np.save(
                    out_dir / f"{image_id}.npy",
                    np.asarray(mat["map"], dtype=np.float32),
                )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "source": {
            name: {"path": str(path), "bytes": path.stat().st_size}
            for name, path in (
                ("images_zip", cfg.images_zip),
                ("density_zip", cfg.density_zip),
            )
        },
        "split_sizes_in_archive": {s: len(ids[s]) for s in SPLITS},
        "extracted": {s: chosen[s] for s in SPLITS},
    }
    digest = hashlib.sha256(
        json.dumps(manifest["extracted"], sort_keys=True).encode()
    ).hexdigest()
    manifest["subset_sha256"] = digest
    path = cfg.subset_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main() -> None:
    cfg = BaselineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", type=int, default=16, help="samples, omit value 0 for all"
    )
    parser.add_argument("--val", type=int, default=8)
    parser.add_argument("--test", type=int, default=8)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    args = parser.parse_args()

    limits = {s: (None if getattr(args, s) == 0 else getattr(args, s)) for s in SPLITS}
    manifest = extract(cfg, limits, args.seed)

    print(f"source images: {manifest['source']['images_zip']['bytes']} bytes")
    print(f"source density: {manifest['source']['density_zip']['bytes']} bytes")
    print(f"archive splits: {manifest['split_sizes_in_archive']}")
    for split in SPLITS:
        print(f"  extracted {split}: {len(manifest['extracted'][split])}")
    print(f"subset sha256: {manifest['subset_sha256']}")
    print(f"wrote {cfg.subset_dir}")


if __name__ == "__main__":
    main()
