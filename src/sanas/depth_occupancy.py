"""Geometric occupancy from stereo depth. No training, no GPU.

Branch 2 of the smoke test. Deprojects a depth frame to a point cloud, moves
it into the cabin frame, rasterises a bird's-eye-view occupancy grid, and
scores occupancy as the fraction of BEV cells that are occupied now but not in
an empty cabin. The empty-cabin reference is built from the dataset's own
zero-occupant frames, so cabin structure (seats, poles, walls) is subtracted
rather than counted as people.

Everything here is written from the sensor model and the archive's own
calibration data. The upstream toolkit
(EvgenyGorelik/multiview_incabin_dataset) carries NO licence file, so none of
its conversion or auto-labeling code is copied. Reading calibration values out
of the archive's YAML/JSON is fine: that is CC-BY-4.0 data, not their code.

Verified against the archive (2026-08-09):
  - depth streams are 848x480 (front/center) or 640x480 (back_right) with
    fx ~422-427, NOT the colour streams' 1280x720 / fx ~920. Using colour
    intrinsics on depth would be wrong by more than a factor of two.
  - depth payloads are 16-bit PNG (PIL mode I;16) named ``*.jpg``.
  - values are uint16 millimetres (RealSense convention, depth_scale 0.001).
  - extrinsics ARE in the archive: frame_ids.json maps a camera to its optical
    frame, target_transforms.json gives the URDF joint chain up to base_link.

RANGE CAVEAT, measured on 20 sampled front_left_depth frames (8,140,800 px):
  13.7% zero/no-return, 41.9% inside the D435i's stated 0.3-3.0 m window,
  24.6% at 3-6 m, 10.4% at 6-12 m, 9.4% beyond 12 m, 1.7% saturated at 65535.
  Depth beyond ~3 m in this data is not trustworthy. Default max_range is
  therefore 3.0 m and the resulting score describes only the near field.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass

import numpy as np

DEPTH_SCALE_M = 0.001  # uint16 millimetres -> metres
UINT16_SATURATION = 65535


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...] = ()

    @property
    def has_distortion(self) -> bool:
        return any(abs(d) > 1e-9 for d in self.distortion)


def _yaml_floats(text: str, block: str) -> list[float]:
    m = re.search(rf"{block}:.*?data:\s*((?:\s*-\s*[-\d.eE+]+\s*\n)+)", text, re.S)
    if not m:
        return []
    return [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", m.group(1))]


def load_intrinsics(path: str) -> Intrinsics:
    """Parse a ROS camera_info.yaml.

    Deliberately a small regex parser rather than a PyYAML dependency: the
    file shape is fixed and this keeps the CPU kernel free of extra installs.
    """
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    k = _yaml_floats(text, "camera_matrix")
    if len(k) < 9:
        raise ValueError(f"no camera_matrix in {path}")
    w = re.search(r"image_width:\s*(\d+)", text)
    h = re.search(r"image_height:\s*(\d+)", text)
    return Intrinsics(
        width=int(w.group(1)) if w else 0,
        height=int(h.group(1)) if h else 0,
        fx=k[0],
        fy=k[4],
        cx=k[2],
        cy=k[5],
        distortion=tuple(_yaml_floats(text, "distortion_coefficients")),
    )


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw -> 3x3 rotation (Rz @ Ry @ Rx)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _joint_matrix(entry: dict) -> np.ndarray:
    xyz = [float(v) for v in str(entry["xyz"]).split()]
    rpy = [float(v) for v in str(entry["rpy"]).split()]
    t = np.eye(4)
    t[:3, :3] = rpy_to_matrix(*rpy)
    t[:3, 3] = xyz
    return t


def load_extrinsics(
    subset_dir: str, camera: str, root_frame: str = "base_link"
) -> np.ndarray:
    """4x4 transform taking points from a camera's optical frame to root_frame.

    Uses only archive data: frame_ids.json (camera -> optical frame name) and
    target_transforms.json (URDF joint chain, child expressed in parent).
    Raises if the chain cannot be resolved, so a caller can fall back to
    per-camera frustum occupancy instead of silently using a wrong frame.
    """
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

    # Entries are ordered child-first: [aruco->cam, base->aruco]. Compose from
    # the root down, so multiply in reverse order.
    t = np.eye(4)
    for entry in reversed(chain):
        t = t @ _joint_matrix(entry)

    parents = [e.get("parent") for e in chain]
    if root_frame not in parents:
        raise ValueError(
            f"chain for {frame!r} does not reach {root_frame!r}; parents={parents}"
        )
    return t


# --------------------------------------------------------------------------
# Depth -> points
# --------------------------------------------------------------------------


def load_depth_metres(path: str) -> np.ndarray:
    """Read a depth frame as float32 metres. Decodes by content, not extension.

    The archive names depth frames ``*.jpg`` while the payload is PNG; PIL
    sniffs the magic bytes so this works either way, but never assume the
    extension elsewhere.
    """
    from PIL import Image

    arr = np.array(Image.open(path))
    if arr.dtype != np.uint16:
        raise ValueError(f"expected uint16 depth, got {arr.dtype} in {path}")
    out = arr.astype(np.float32) * DEPTH_SCALE_M
    out[arr == UINT16_SATURATION] = 0.0  # saturated = no measurement
    return out


def depth_range_stats(depth_m: np.ndarray, min_range: float, max_range: float) -> dict:
    """Per-frame accounting of how much of the frame is actually usable."""
    total = depth_m.size
    zero = int((depth_m <= 0).sum())
    inside = int(((depth_m >= min_range) & (depth_m <= max_range)).sum())
    beyond = int((depth_m > max_range).sum())
    near = int(((depth_m > 0) & (depth_m < min_range)).sum())
    return {
        "pixels": total,
        "frac_zero": zero / total,
        "frac_in_range": inside / total,
        "frac_beyond_max": beyond / total,
        "frac_below_min": near / total,
    }


def deproject(
    depth_m: np.ndarray,
    intr: Intrinsics,
    min_range: float = 0.3,
    max_range: float = 3.0,
    pixel_stride: int = 4,
) -> np.ndarray:
    """Depth image -> (N,3) points in the camera optical frame (x right, y down, z forward)."""
    if intr.has_distortion:
        raise NotImplementedError(
            "intrinsics carry non-zero distortion; this deprojection assumes "
            "rectified depth (all four cameras in this archive report zeros)"
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


def transform_points(points: np.ndarray, t: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    return (points @ t[:3, :3].T + t[:3, 3]).astype(np.float32)


# --------------------------------------------------------------------------
# BEV occupancy
# --------------------------------------------------------------------------


@dataclass
class GridConfig:
    """BEV grid in the cabin frame. Defaults are metres."""

    cell_size: float = 0.10
    x_min: float = -6.0
    x_max: float = 6.0
    y_min: float = -6.0
    y_max: float = 6.0
    z_min: float = 0.6  # above seat height: ignore floor and seat pans
    z_max: float = 2.0  # below ceiling
    min_points_per_cell: int = 3

    @property
    def shape(self) -> tuple[int, int]:
        return (
            int(round((self.x_max - self.x_min) / self.cell_size)),
            int(round((self.y_max - self.y_min) / self.cell_size)),
        )


def bev_grid(points: np.ndarray, cfg: GridConfig) -> np.ndarray:
    """Boolean occupancy grid over the cabin floor plan."""
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


def build_background(grids: list[np.ndarray], quantile: float = 0.5) -> np.ndarray:
    """Empty-cabin reference: cells occupied in at least `quantile` of frames.

    Built from frames whose ground-truth occupant count is zero, so fixed cabin
    structure is subtracted instead of being scored as people.
    """
    if not grids:
        raise ValueError("no zero-occupancy frames available to build a background")
    stack = np.stack(grids, axis=0)
    return stack.mean(axis=0) >= quantile


def occupancy_score(grid: np.ndarray, background: np.ndarray) -> tuple[float, int]:
    """Fraction of non-background cells that are occupied. Returns (score, cells)."""
    foreground = grid & ~background
    denom = int((~background).sum())
    cells = int(foreground.sum())
    return (cells / denom if denom else 0.0), cells


# --------------------------------------------------------------------------
# Metrics (numpy only, no scipy)
# --------------------------------------------------------------------------


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    # average tied groups
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


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(_rankdata(a), _rankdata(b))
