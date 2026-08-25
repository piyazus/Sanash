"""One way to get a subset onto a machine, shared by every kernel.

Kaggle's `kernel_sources` mounts a source kernel's CODE at
/kaggle/input/notebooks/<owner>/<slug>, not its output, so kernels cannot be
chained to pass data. Every kernel therefore fetches its own subset by HTTP
range request from the HuggingFace mirror. This is the single place that
happens, so the selection cannot drift between the RGB and depth branches.
"""

from __future__ import annotations

import json
import os
import sys

from .labels import build_label_table, count_distribution
from .selection import (
    SubsetSpec,
    check_single_camera_regression,
    select_members,
)
from .ziprange import ARCHIVE_SIZE, MIRRORS, extract_members, plan_runs, read_members


def default_work_dir() -> str:
    return (
        "/kaggle/working/subset"
        if os.path.isdir("/kaggle/working")
        else os.path.join("data", "subset")
    )


def fetch_subset(
    work: str | None = None,
    mirror: str = "hf",
    camera: str = "front_left_color",
    stride: int = 10,
    max_images: int = 0,
    depth_cameras: list | None = None,
    include_depth: bool = True,
    score_threshold: float = 0.0,
    dry_run: bool = False,
    no_fetch: bool = False,
    check_regression: bool = True,
    verbose: bool = True,
) -> tuple[str, dict]:
    """Ensure a usable subset exists at `work`. Returns (work_dir, info)."""
    work = work or default_work_dir()
    labels_path = os.path.join(work, "labels.jsonl")

    if no_fetch:
        if not os.path.exists(labels_path):
            raise SystemExit(f"--no-fetch given but {labels_path} is missing")
        if verbose:
            print(f"--no-fetch: reusing existing subset at {work}")
        return work, {"fetched": False}

    spec = SubsetSpec(
        color_cameras=[camera],
        depth_cameras=depth_cameras,
        timebase_camera=camera,
        stride=stride,
        max_images=max_images,
        include_depth=include_depth,
    )

    url = MIRRORS[mirror]
    if verbose:
        print(f"mirror: {mirror}")
    archive_size, members = read_members(url, verbose=verbose)
    if archive_size != ARCHIVE_SIZE:
        print(
            f"WARNING: archive is {archive_size:,} bytes, expected {ARCHIVE_SIZE:,}",
            file=sys.stderr,
        )

    selected, breakdown = select_members(members, spec)
    if verbose:
        print("\nselection (shared definition, development/src/sanas/selection.py):")
        for k in (
            "images",
            "labels",
            "depth",
            "total_members",
            "total_compressed_bytes",
        ):
            print(f"  {k:<26} {breakdown[k]:,}")
        print(f"  {'GB selected':<26} {breakdown['total_compressed_bytes'] / 1e9:.3f}")
        print(f"  per-camera depth frames    {breakdown['per_camera_depth']}")

    single_default = (
        check_regression
        and stride == 10
        and max_images == 0
        and camera == "front_left_color"
        and len(spec.resolved_depth_cameras()) <= 1
    )
    if single_default and verbose:
        keys = (
            ("images", "labels", "depth", "total_members")
            if include_depth
            else ("images", "labels")
        )
        problems = check_single_camera_regression(breakdown, keys=keys)
        print(
            "  regression vs executed run: "
            + ("CLEAN" if not problems else "DRIFT -> " + "; ".join(problems))
        )

    if dry_run:
        runs = plan_runs(selected, archive_size)
        planned = sum(e - s for s, e, _ in runs)
        print(
            f"\n--dry-run: would issue {len(runs):,} range request(s), "
            f"~{planned / 1e9:.3f} GB. Nothing downloaded."
        )
        raise SystemExit(0)

    os.makedirs(work, exist_ok=True)
    stats = extract_members(
        url, selected, work, archive_size=archive_size, verbose=verbose
    )
    if verbose:
        print(
            f"\nextracted {stats['members']:,} files, "
            f"{stats['bytes_written'] / 1e9:.3f} GB in {stats['seconds']}s "
            f"({stats['renamed_total']:,} extensions corrected by magic-byte sniff)"
        )

    rows = build_label_table(work, camera, score_threshold=score_threshold)
    with open(labels_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    if verbose:
        print(f"labels: {len(rows):,} rows -> {labels_path}")
        print(f"count_view  {count_distribution(rows, 'count_view')}")
        print(f"count_cabin {count_distribution(rows, 'count_cabin')}")

    return work, {"fetched": True, "selection": breakdown, "transfer": stats}
