"""Canonical subset selection. One definition, reused by every kernel.

This exists so the 1-camera and 4-camera runs select frames the same way. If
each kernel re-derived its own selection, a later comparison would be
measuring our own inconsistency instead of measuring cameras.

The single-camera default here reproduces exactly what
`sanas-extract-subset` already ran on Kaggle:
    1,289 colour frames, 1,204 matched depth frames, 32,911 annotation files,
    35,404 members, 0.676 GB selected.
`development/scripts/build_kernels.py --check` plus the regression numbers in
`tests_selection_expectations` guard against drift.

DEPTH FRAMES ARE MATCHED, NOT STRIDED. Each depth frame is the nearest one in
time to a selected colour frame, within a tolerance. Independently striding
each depth stream would give four cameras four different sets of instants,
and a person who moves between them would land in four different places in a
merged BEV grid. Matching costs slightly fewer frames and slightly fewer bytes
than independent striding, and is the only version that is temporally
coherent across cameras.
"""

from __future__ import annotations

import bisect
import os
from dataclasses import dataclass, field

ROOT_PREFIX = "beintelli_v1/"

ROOT_METADATA = {
    ROOT_PREFIX + "person_states.json",
    ROOT_PREFIX + "modalities.json",
    ROOT_PREFIX + "frame_ids.json",
    ROOT_PREFIX + "target_transforms.json",
}

COLOR_CAMERAS = [
    "front_left_color",
    "front_right_color",
    "center_left_color",
    "back_right_color",
]

NS_PER_MS = 1_000_000

# What the already-executed single-camera extraction produced. Any change to
# selection logic that moves these numbers is drift and must be deliberate.
SINGLE_CAMERA_EXPECTATION = {
    "images": 1289,
    "depth": 1204,
    "labels": 32911,
    "total_members": 35404,
}


@dataclass
class SubsetSpec:
    """Which archive members a run wants."""

    color_cameras: list = field(default_factory=lambda: ["front_left_color"])
    # Depth streams to pull. None -> the depth twin of each colour camera.
    depth_cameras: list | None = None
    # Colour camera whose frame times define the sampling grid. All depth
    # streams are matched to these instants so the cameras stay in sync.
    timebase_camera: str | None = None
    stride: int = 10
    max_images: int = 0
    include_depth: bool = True
    include_masks: bool = False
    include_poses: bool = False
    depth_tolerance_ms: int = 100

    def resolved_depth_cameras(self) -> list:
        if self.depth_cameras is not None:
            return list(self.depth_cameras)
        if not self.include_depth:
            return []
        return [c.replace("_color", "_depth") for c in self.color_cameras]

    def resolved_timebase(self) -> str:
        return self.timebase_camera or self.color_cameras[0]


def timestamp_of(name: str) -> int | None:
    """19-digit nanosecond stamp from a member name, or None."""
    stem = os.path.splitext(os.path.basename(name))[0].split("_")[0]
    return int(stem) if stem.isdigit() and len(stem) == 19 else None


def nearest(ts: int, sorted_ts: list, tolerance_ns: int) -> int | None:
    if not sorted_ts:
        return None
    i = bisect.bisect_left(sorted_ts, ts)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(sorted_ts):
            d = abs(sorted_ts[j] - ts)
            if d <= tolerance_ns and (best is None or d < abs(best - ts)):
                best = sorted_ts[j]
    return best


def select_members(members, spec: SubsetSpec):
    """Return (selected_members, breakdown_dict) for a SubsetSpec."""
    timebase = spec.resolved_timebase()
    depth_cams = spec.resolved_depth_cameras()

    # ---- colour frames -------------------------------------------------
    images = []
    per_camera_images = {}
    for cam in spec.color_cameras:
        cam_imgs = sorted(
            (
                m
                for m in members
                if m.name.startswith(f"{ROOT_PREFIX}cameras/color/{cam}/images/")
                and m.name.endswith(".jpg")
            ),
            key=lambda m: m.name,
        )
        picked = cam_imgs[:: spec.stride]
        if spec.max_images:
            picked = picked[: spec.max_images]
        per_camera_images[cam] = picked
        images.extend(picked)

    timebase_ts = [timestamp_of(m.name) for m in per_camera_images.get(timebase, [])]
    timebase_ts = [t for t in timebase_ts if t is not None]

    # ---- annotations ---------------------------------------------------
    seg_prefixes = tuple(f"{ROOT_PREFIX}segmentations/{c}/" for c in spec.color_cameras)
    labels = []
    for m in members:
        n = m.name
        if m.is_dir:
            continue
        if n.startswith(seg_prefixes):
            if n.endswith(".json") or (spec.include_masks and n.endswith(".png")):
                labels.append(m)
        elif n.startswith(f"{ROOT_PREFIX}bboxes_3d/") or n.startswith(
            f"{ROOT_PREFIX}states/"
        ):
            labels.append(m)
        elif spec.include_poses and n.startswith(f"{ROOT_PREFIX}poses/"):
            labels.append(m)
        elif n in ROOT_METADATA:
            labels.append(m)
        elif n.startswith(f"{ROOT_PREFIX}cameras/") and n.endswith("camera_info.yaml"):
            labels.append(m)

    # ---- depth frames, matched to the timebase -------------------------
    depth = []
    per_camera_depth = {}
    tol = spec.depth_tolerance_ms * NS_PER_MS
    for dcam in depth_cams:
        cand = {}
        for m in members:
            if m.is_dir:
                continue
            if not m.name.startswith(f"{ROOT_PREFIX}cameras/depth/{dcam}/images/"):
                continue
            t = timestamp_of(m.name)
            if t is not None:
                cand[t] = m
        keys = sorted(cand)
        seen, chosen = set(), []
        for t in timebase_ts:
            hit = nearest(t, keys, tol)
            if hit is not None and hit not in seen:
                seen.add(hit)
                chosen.append(cand[hit])
        per_camera_depth[dcam] = chosen
        depth.extend(chosen)

    selected = images + labels + depth
    breakdown = {
        "timebase_camera": timebase,
        "stride": spec.stride,
        "images": len(images),
        "image_bytes": sum(m.csize for m in images),
        "labels": len(labels),
        "label_bytes": sum(m.csize for m in labels),
        "depth": len(depth),
        "depth_bytes": sum(m.csize for m in depth),
        "total_members": len(selected),
        "total_compressed_bytes": sum(m.csize for m in selected),
        "per_camera_images": {c: len(v) for c, v in per_camera_images.items()},
        "per_camera_depth": {c: len(v) for c, v in per_camera_depth.items()},
        "per_camera_depth_bytes": {
            c: sum(m.csize for m in v) for c, v in per_camera_depth.items()
        },
    }
    return selected, breakdown


def check_single_camera_regression(breakdown: dict, keys=None) -> list:
    """Compare a single-camera breakdown against what already ran.

    `keys` limits the comparison, because a run that deliberately skips depth
    cannot be expected to match the depth-inclusive frame counts. Returns a
    list of human-readable mismatches; empty means no drift.
    """
    problems = []
    for key, expected in SINGLE_CAMERA_EXPECTATION.items():
        if keys is not None and key not in keys:
            continue
        actual = breakdown.get(key)
        if actual != expected:
            problems.append(f"{key}: expected {expected:,}, got {actual:,}")
    return problems
