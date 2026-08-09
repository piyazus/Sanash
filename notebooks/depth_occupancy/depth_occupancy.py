#!/usr/bin/env python
"""Sanas branch 2: geometric occupancy from stereo depth. CPU only, no training.

Consumes the output of the extraction kernel and scores the SAME frames the
RGB branch is trained on, so the two are comparable. No GPU, no epochs: this
deprojects depth to a point cloud, moves it into the cabin frame, rasterises a
BEV occupancy grid, subtracts an empty-cabin background built from the
dataset's own zero-occupant frames, and correlates the resulting score against
the ground-truth raw count.

Why this branch exists: an Ouster OS0-128 in every bus is not a deployable
product, a RealSense-class depth camera is, and IR stereo still works in the
dark - which is the gap this dataset otherwise cannot test at all (single
daytime session).

WHAT IT ALSO CHECKS: the D435i depth module is specified for roughly 0.3-3.0 m
and a bus cabin is longer than that. The kernel measures how much of each
frame actually falls inside that window and prints it, because if most of the
cabin is out of reliable range this branch only ever describes the near field.

LICENSING: written from the sensor model and the archive's own calibration
data. The upstream toolkit (EvgenyGorelik/multiview_incabin_dataset) has NO
licence file, so none of its code is copied. Calibration values are read out
of the archive's YAML/JSON, which is CC-BY-4.0 data.

NOTE ON DUPLICATION: canonical implementation is src/sanas/depth_occupancy.py.
Kaggle pushes one file per kernel, so this is a self-contained vendored copy.
Change src/ first, then re-sync the marked block.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass

import numpy as np

# ==========================================================================
# BEGIN VENDORED BLOCK - canonical: src/sanas/depth_occupancy.py
# ==========================================================================

DEPTH_SCALE_M = 0.001
UINT16_SATURATION = 65535


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple = ()

    @property
    def has_distortion(self) -> bool:
        return any(abs(d) > 1e-9 for d in self.distortion)


def _yaml_floats(text, block):
    m = re.search(rf"{block}:.*?data:\s*((?:\s*-\s*[-\d.eE+]+\s*\n)+)", text, re.S)
    return (
        [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", m.group(1))]
        if m
        else []
    )


def load_intrinsics(path):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    k = _yaml_floats(text, "camera_matrix")
    if len(k) < 9:
        raise ValueError(f"no camera_matrix in {path}")
    w = re.search(r"image_width:\s*(\d+)", text)
    h = re.search(r"image_height:\s*(\d+)", text)
    return Intrinsics(
        int(w.group(1)) if w else 0,
        int(h.group(1)) if h else 0,
        k[0],
        k[4],
        k[2],
        k[5],
        tuple(_yaml_floats(text, "distortion_coefficients")),
    )


def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _joint_matrix(entry):
    xyz = [float(v) for v in str(entry["xyz"]).split()]
    rpy = [float(v) for v in str(entry["rpy"]).split()]
    t = np.eye(4)
    t[:3, :3] = rpy_to_matrix(*rpy)
    t[:3, 3] = xyz
    return t


def load_extrinsics(subset_dir, camera, root_frame="base_link"):
    with open(os.path.join(subset_dir, "frame_ids.json"), encoding="utf-8") as fh:
        frame_ids = json.load(fh)
    with open(
        os.path.join(subset_dir, "target_transforms.json"), encoding="utf-8"
    ) as fh:
        transforms = json.load(fh)
    frame = frame_ids.get(camera)
    if frame is None:
        raise KeyError(f"camera {camera!r} not in frame_ids.json")
    chain = transforms.get(frame)
    if not chain:
        raise KeyError(f"frame {frame!r} has no entry in target_transforms.json")
    t = np.eye(4)
    for entry in reversed(chain):
        t = t @ _joint_matrix(entry)
    parents = [e.get("parent") for e in chain]
    if root_frame not in parents:
        raise ValueError(
            f"chain for {frame!r} does not reach {root_frame!r}; parents={parents}"
        )
    return t


def load_depth_metres(path):
    """Decode by content, not extension: depth frames are PNG named .jpg."""
    from PIL import Image

    arr = np.array(Image.open(path))
    if arr.dtype != np.uint16:
        raise ValueError(f"expected uint16 depth, got {arr.dtype} in {path}")
    out = arr.astype(np.float32) * DEPTH_SCALE_M
    out[arr == UINT16_SATURATION] = 0.0
    return out


def depth_range_stats(depth_m, min_range, max_range):
    total = depth_m.size
    return {
        "pixels": total,
        "frac_zero": float((depth_m <= 0).sum()) / total,
        "frac_in_range": float(((depth_m >= min_range) & (depth_m <= max_range)).sum())
        / total,
        "frac_beyond_max": float((depth_m > max_range).sum()) / total,
        "frac_below_min": float(((depth_m > 0) & (depth_m < min_range)).sum()) / total,
    }


def deproject(depth_m, intr, min_range=0.3, max_range=3.0, pixel_stride=4):
    if intr.has_distortion:
        raise NotImplementedError(
            "non-zero distortion; deprojection assumes rectified depth"
        )
    d = depth_m[::pixel_stride, ::pixel_stride]
    ys, xs = np.nonzero((d >= min_range) & (d <= max_range))
    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    z = d[ys, xs]
    u = xs.astype(np.float32) * pixel_stride
    v = ys.astype(np.float32) * pixel_stride
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    return np.stack([x, y, z], axis=1).astype(np.float32)


def transform_points(points, t):
    if points.size == 0:
        return points
    return (points @ t[:3, :3].T + t[:3, 3]).astype(np.float32)


@dataclass
class GridConfig:
    cell_size: float = 0.10
    x_min: float = -6.0
    x_max: float = 6.0
    y_min: float = -6.0
    y_max: float = 6.0
    z_min: float = 0.6
    z_max: float = 2.0
    min_points_per_cell: int = 3

    @property
    def shape(self):
        return (
            int(round((self.x_max - self.x_min) / self.cell_size)),
            int(round((self.y_max - self.y_min) / self.cell_size)),
        )


def bev_grid(points, cfg):
    nx, ny = cfg.shape
    grid = np.zeros((nx, ny), dtype=np.int32)
    if points.size == 0:
        return grid > 0
    m = (
        (points[:, 2] >= cfg.z_min)
        & (points[:, 2] <= cfg.z_max)
        & (points[:, 0] >= cfg.x_min)
        & (points[:, 0] < cfg.x_max)
        & (points[:, 1] >= cfg.y_min)
        & (points[:, 1] < cfg.y_max)
    )
    p = points[m]
    if p.size == 0:
        return grid > 0
    ix = ((p[:, 0] - cfg.x_min) / cfg.cell_size).astype(np.int32)
    iy = ((p[:, 1] - cfg.y_min) / cfg.cell_size).astype(np.int32)
    np.add.at(grid, (ix, iy), 1)
    return grid >= cfg.min_points_per_cell


def build_background(grids, quantile=0.5):
    if not grids:
        raise ValueError("no zero-occupancy frames available to build a background")
    return np.stack(grids, axis=0).mean(axis=0) >= quantile


def occupancy_score(grid, background):
    foreground = grid & ~background
    denom = int((~background).sum())
    return (int(foreground.sum()) / denom if denom else 0.0), int(foreground.sum())


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rankdata(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < sx.size:
        j = i
        while j + 1 < sx.size and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return ranks


def spearman(a, b):
    return pearson(_rankdata(a), _rankdata(b))


# ==========================================================================
# END VENDORED BLOCK
# ==========================================================================


def default_data_dir() -> str:
    for c in (
        "/kaggle/input/sanas-extract-subset/subset",
        "/kaggle/input/sanas-extract-subset",
        os.path.join("data", "subset"),
    ):
        if os.path.isdir(c):
            return c
    return os.path.join("data", "subset")


_DEPTH_INDEX = None


def find_depth_frame(depth_dir: str, ts: int, tolerance_ns: int) -> str | None:
    """Nearest depth file to a colour timestamp. Extension-agnostic."""
    global _DEPTH_INDEX
    if _DEPTH_INDEX is None:
        idx = {}
        for name in os.listdir(depth_dir):
            stem = os.path.splitext(name)[0].split("_")[0]
            if stem.isdigit() and len(stem) == 19:
                idx[int(stem)] = os.path.join(depth_dir, name)
        _DEPTH_INDEX = (sorted(idx), idx)
    keys, idx = _DEPTH_INDEX
    if not keys:
        return None
    import bisect

    i = bisect.bisect_left(keys, ts)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(keys):
            d = abs(keys[j] - ts)
            if d <= tolerance_ns and (best is None or d < abs(best - ts)):
                best = keys[j]
    return idx[best] if best is not None else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Geometric BEV occupancy from stereo depth (CPU)"
    )
    ap.add_argument("--data", default=None, help="extraction kernel output dir")
    ap.add_argument("--camera", default="front_left_color")
    ap.add_argument(
        "--label-column", default="count_view", choices=["count_view", "count_cabin"]
    )
    ap.add_argument(
        "--min-range", type=float, default=0.3, help="D435i lower spec bound (m)"
    )
    ap.add_argument(
        "--max-range", type=float, default=3.0, help="D435i upper spec bound (m)"
    )
    ap.add_argument("--pixel-stride", type=int, default=4)
    ap.add_argument("--cell-size", type=float, default=0.10)
    ap.add_argument(
        "--z-min",
        type=float,
        default=0.6,
        help="above seat pan height (cabin floor is z~0)",
    )
    ap.add_argument("--z-max", type=float, default=2.0)
    ap.add_argument("--min-points-per-cell", type=int, default=3)
    ap.add_argument("--background-quantile", type=float, default=0.5)
    ap.add_argument("--depth-tolerance-ms", type=int, default=100)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = args.data or default_data_dir()
    depth_cam = args.camera.replace("_color", "_depth")
    out_path = args.out or os.path.join(
        "/kaggle/working" if os.path.isdir("/kaggle/working") else ".",
        "depth_occupancy_metrics.json",
    )

    print("=" * 72)
    print("Sanas depth-occupancy baseline (branch 2, CPU, no training)")
    print("=" * 72)
    print(f"data       : {data}")
    print(f"camera     : {args.camera} -> depth stream {depth_cam}")
    print(f"range gate : {args.min_range}-{args.max_range} m (D435i spec window)")

    labels_path = os.path.join(data, "labels.jsonl")
    if not os.path.exists(labels_path):
        print(
            f"missing {labels_path}. Run the extraction kernel first.", file=sys.stderr
        )
        return 1
    with open(labels_path, encoding="utf-8") as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    rows = [r for r in rows if r.get(args.label_column) is not None]
    if args.max_frames:
        rows = rows[: args.max_frames]
    print(f"frames with a {args.label_column} label: {len(rows):,}")

    depth_dir = os.path.join(data, "cameras", "depth", depth_cam, "images")
    if not os.path.isdir(depth_dir):
        print(
            f"no depth frames at {depth_dir}. Re-run extraction with depth enabled.",
            file=sys.stderr,
        )
        return 1

    intr_path = os.path.join(data, "cameras", "depth", depth_cam, "camera_info.yaml")
    intr = load_intrinsics(intr_path)
    print(
        f"depth intrinsics: {intr.width}x{intr.height} fx={intr.fx:.2f} fy={intr.fy:.2f} "
        f"cx={intr.cx:.2f} cy={intr.cy:.2f} (NOT the colour stream's)"
    )

    try:
        T = load_extrinsics(data, depth_cam)
        frame_ok = True
        print(
            f"extrinsics from archive URDF chain; camera origin in base_link: {np.round(T[:3, 3], 3)}"
        )
    except (KeyError, ValueError, FileNotFoundError) as exc:
        frame_ok = False
        T = np.eye(4)
        print(f"WARNING: no usable camera->cabin transform in the archive ({exc}).")
        print(
            "Falling back to per-camera frustum occupancy in the optical frame; "
            "the BEV axes are then camera-relative, not cabin-relative."
        )

    cfg = GridConfig(
        cell_size=args.cell_size,
        z_min=args.z_min,
        z_max=args.z_max,
        min_points_per_cell=args.min_points_per_cell,
    )
    print(
        f"BEV grid: {cfg.shape[0]}x{cfg.shape[1]} cells of {cfg.cell_size} m, "
        f"z window {cfg.z_min}-{cfg.z_max} m"
    )

    tol = args.depth_tolerance_ms * 1_000_000
    grids, scores, counts, kept, range_acc, missing = {}, [], [], [], [], 0
    for r in rows:
        path = find_depth_frame(depth_dir, int(r["timestamp"]), tol)
        if path is None:
            missing += 1
            continue
        d = load_depth_metres(path)
        range_acc.append(depth_range_stats(d, args.min_range, args.max_range))
        pts = transform_points(
            deproject(d, intr, args.min_range, args.max_range, args.pixel_stride), T
        )
        g = bev_grid(pts, cfg)
        grids[r["timestamp"]] = g
        kept.append(r)

    print(f"\nmatched depth frames: {len(kept):,}   unmatched: {missing:,}")
    if not kept:
        print("nothing to score", file=sys.stderr)
        return 1

    agg = {
        k: float(np.mean([s[k] for s in range_acc]))
        for k in range_acc[0]
        if k != "pixels"
    }
    print("\nDEPTH RANGE REALITY CHECK (mean over scored frames)")
    print(f"  zero / no return          : {100 * agg['frac_zero']:.1f}%")
    print(
        f"  inside {args.min_range}-{args.max_range} m       : {100 * agg['frac_in_range']:.1f}%"
    )
    print(
        f"  beyond {args.max_range} m             : {100 * agg['frac_beyond_max']:.1f}%"
    )
    print(
        f"  closer than {args.min_range} m        : {100 * agg['frac_below_min']:.1f}%"
    )
    if agg["frac_in_range"] < 0.5:
        print("  VERDICT: most of the frame is outside the sensor's reliable window.")
        print(
            "  This baseline describes the NEAR FIELD ONLY, not full-cabin occupancy."
        )

    zero_rows = [r for r in kept if int(r[args.label_column]) == 0]
    print(f"\nzero-occupancy frames for background: {len(zero_rows):,}")
    if not zero_rows:
        print(
            "no empty-cabin frames; cannot subtract cabin structure. Aborting.",
            file=sys.stderr,
        )
        return 1
    background = build_background(
        [grids[r["timestamp"]] for r in zero_rows], args.background_quantile
    )
    print(f"background cells: {int(background.sum()):,} of {background.size:,}")

    for r in kept:
        s, _cells = occupancy_score(grids[r["timestamp"]], background)
        scores.append(s)
        counts.append(int(r[args.label_column]))
    scores_a = np.array(scores)
    counts_a = np.array(counts)

    per_level = {}
    for c in sorted(set(counts)):
        m = counts_a == c
        per_level[str(c)] = {
            "n": int(m.sum()),
            "mean_score": float(scores_a[m].mean()),
            "std_score": float(scores_a[m].std()),
            "median_score": float(np.median(scores_a[m])),
        }

    r_p = pearson(scores_a, counts_a)
    r_s = spearman(scores_a, counts_a)

    print("\nOCCUPANCY SCORE vs GROUND-TRUTH COUNT")
    print(f"  Pearson  r   : {r_p:.3f}")
    print(f"  Spearman rho : {r_s:.3f}")
    print(f"\n  {'count':>6} {'n':>7} {'mean':>10} {'median':>10} {'std':>10}")
    for c, v in per_level.items():
        print(
            f"  {c:>6} {v['n']:>7,} {v['mean_score']:>10.4f} {v['median_score']:>10.4f} {v['std_score']:>10.4f}"
        )

    monotonic = all(
        per_level[a]["mean_score"] <= per_level[b]["mean_score"]
        for a, b in zip(sorted(per_level), sorted(per_level)[1:])
    )
    print(f"\n  mean score monotonic in count: {monotonic}")

    metrics = {
        "branch": "depth_geometric_bev",
        "trained": False,
        "camera": args.camera,
        "depth_stream": depth_cam,
        "label_column": args.label_column,
        "frames_scored": len(kept),
        "frames_unmatched": missing,
        "cabin_frame_resolved": frame_ok,
        "intrinsics": {
            "w": intr.width,
            "h": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.cx,
            "cy": intr.cy,
        },
        "range_gate_m": [args.min_range, args.max_range],
        "depth_range_stats_mean": agg,
        "grid": {
            "cell_size": cfg.cell_size,
            "shape": list(cfg.shape),
            "z_min": cfg.z_min,
            "z_max": cfg.z_max,
            "min_points_per_cell": cfg.min_points_per_cell,
        },
        "background_frames": len(zero_rows),
        "background_cells": int(background.sum()),
        "pearson_r": r_p,
        "spearman_rho": r_s,
        "monotonic_in_count": bool(monotonic),
        "per_count_level": per_level,
        "caveats": [
            "Pseudo-label ground truth, not human annotation.",
            "Max 4 occupants anywhere in this dataset; single daytime session.",
            "Depth beyond the range gate is discarded as unreliable.",
        ],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nwrote {out_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
