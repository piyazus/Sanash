"""Shared body for the depth-occupancy kernels. One implementation, two entry points.

`main(default_mode="single")` scores one depth stream.
`main(default_mode="multi")`  scores all four and compares against one.

Both are the same code path with different defaults, so the 1-camera vs
4-camera comparison measures cameras and not two different implementations.

Each kernel is self-contained and fetches its own subset by HTTP range request
from the HuggingFace mirror. `kernel_sources` mounts a source kernel's CODE at
/kaggle/input/notebooks/<owner>/<slug>, not its output, so chaining kernels
does not deliver data. Fetching directly is the fix.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from .depth_occupancy import (
    GridConfig,
    bev_grid,
    bootstrap_ci,
    build_background,
    coverage_report,
    deproject,
    depth_range_stats,
    fragility_warnings,
    load_depth_metres,
    load_extrinsics,
    load_intrinsics,
    merge_grids,
    observed_cells,
    occupancy_score,
    pearson,
    person_region_cells,
    spearman,
    transform_points,
)
from .fetch import default_work_dir, fetch_subset
from .labels import count_distribution
from .selection import COLOR_CAMERAS, nearest, timestamp_of
from .ziprange import MIRRORS

DIAGNOSTIC_INPUT_DIR = "/kaggle/input/notebooks/diypyzsdiyas/sanas-extract-subset"


def on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def report_mounted_kernel_source(path: str = DIAGNOSTIC_INPUT_DIR) -> dict:
    """Print what `kernel_sources` actually mounts. Names and sizes only.

    Cheap diagnostic: if one of these turns out to be a usable artifact the
    design could be simplified later, so it is worth one listing.
    """
    print("\n" + "-" * 72)
    print(f"DIAGNOSTIC: contents of {path}")
    print("-" * 72)
    if not os.path.isdir(path):
        print("  not mounted in this run")
        return {"mounted": False, "files": []}
    files = []
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        kind = "dir" if os.path.isdir(full) else "file"
        size = os.path.getsize(full) if os.path.isfile(full) else 0
        files.append({"name": name, "kind": kind, "bytes": size})
        print(f"  {kind:<5} {size:>12,}  {name}")
    if not files:
        print("  (empty)")
    return {"mounted": True, "files": files}


def ensure_subset(args) -> tuple[str, dict]:
    """Fetch the subset by byte range. Delegates to the shared fetcher so the
    RGB and depth branches cannot select different frames."""
    depth_cams = (
        [c.replace("_color", "_depth") for c in COLOR_CAMERAS]
        if args.mode == "multi"
        else [args.camera.replace("_color", "_depth")]
    )
    return fetch_subset(
        work=args.data,
        mirror=args.mirror,
        camera=args.camera,
        stride=args.stride,
        max_images=args.max_images,
        depth_cameras=depth_cams,
        include_depth=True,
        score_threshold=args.score_threshold,
        dry_run=args.dry_run,
        no_fetch=args.no_fetch,
        check_regression=(args.mode == "single"),
    )


def build_rig(work: str, depth_cams: list) -> dict:
    """Per-camera intrinsics and camera->base_link transform."""
    rig = {}
    for cam in depth_cams:
        intr = load_intrinsics(
            os.path.join(work, "cameras", "depth", cam, "camera_info.yaml")
        )
        try:
            T = load_extrinsics(work, cam)
            ok = True
        except (KeyError, ValueError, FileNotFoundError) as exc:
            print(f"  WARNING: no cabin transform for {cam} ({exc}); using identity")
            T = np.eye(4)
            ok = False
        rig[cam] = {"intrinsics": intr, "T": T, "frame_resolved": ok}
        print(
            f"  {cam:<20} {intr.width}x{intr.height} fx={intr.fx:.1f} "
            f"origin_in_base={np.round(T[:3, 3], 2)}"
        )
    return rig


def index_depth_dir(work: str, cam: str) -> tuple[list, dict]:
    d = os.path.join(work, "cameras", "depth", cam, "images")
    idx = {}
    if os.path.isdir(d):
        for name in os.listdir(d):
            t = timestamp_of(name)
            if t is not None:
                idx[t] = os.path.join(d, name)
    return sorted(idx), idx


def score_frames(rows, work, rig, cfg, args):
    """Per-camera occupancy grids and observation footprints for every frame."""
    depth_index = {cam: index_depth_dir(work, cam) for cam in rig}
    tol = args.depth_tolerance_ms * 1_000_000

    grids = {cam: {} for cam in rig}
    observed = {cam: {} for cam in rig}
    range_acc = {cam: [] for cam in rig}
    matched = {cam: 0 for cam in rig}
    kept = []

    for r in rows:
        ts = int(r["timestamp"])
        per_cam_ok = False
        for cam, spec in rig.items():
            keys, idx = depth_index[cam]
            hit = nearest(ts, keys, tol)
            if hit is None:
                continue
            d = load_depth_metres(idx[hit])
            range_acc[cam].append(depth_range_stats(d, args.min_range, args.max_range))
            pts = transform_points(
                deproject(
                    d,
                    spec["intrinsics"],
                    args.min_range,
                    args.max_range,
                    args.pixel_stride,
                ),
                spec["T"],
            )
            grids[cam][ts] = bev_grid(pts, cfg)
            observed[cam][ts] = observed_cells(pts, cfg)
            matched[cam] += 1
            per_cam_ok = True
        if per_cam_ok:
            kept.append(r)
    return kept, grids, observed, range_acc, matched


def summarize_scores(scores, counts, label):
    """Correlations with bootstrap intervals plus per-level breakdown."""
    s = np.asarray(scores, dtype=float)
    c = np.asarray(counts, dtype=float)
    r_p = pearson(s, c)
    r_s = spearman(s, c)
    p_lo, p_hi = bootstrap_ci(s, c, pearson)
    s_lo, s_hi = bootstrap_ci(s, c, spearman)
    per_level = {}
    for lv in sorted(set(int(v) for v in c)):
        m = c == lv
        per_level[str(lv)] = {
            "n": int(m.sum()),
            "mean_score": float(s[m].mean()),
            "median_score": float(np.median(s[m])),
            "std_score": float(s[m].std()),
        }
    keys = sorted(per_level, key=int)
    monotonic = all(
        per_level[a]["mean_score"] <= per_level[b]["mean_score"]
        for a, b in zip(keys, keys[1:])
    )
    return {
        "label": label,
        "n": int(s.size),
        "pearson_r": r_p,
        "pearson_ci95": [p_lo, p_hi],
        "spearman_rho": r_s,
        "spearman_ci95": [s_lo, s_hi],
        "per_count_level": per_level,
        "monotonic_in_count": bool(monotonic),
    }


def print_summary(sm, warnings):
    print(f"\n  {sm['label']}  (n={sm['n']:,})")
    print(
        f"    Pearson  r   : {sm['pearson_r']:.3f}   95% CI "
        f"[{sm['pearson_ci95'][0]:.3f}, {sm['pearson_ci95'][1]:.3f}]"
    )
    print(
        f"    Spearman rho : {sm['spearman_rho']:.3f}   95% CI "
        f"[{sm['spearman_ci95'][0]:.3f}, {sm['spearman_ci95'][1]:.3f}]"
    )
    print(f"    monotonic in count: {sm['monotonic_in_count']}")
    print(f"    {'count':>6} {'n':>7} {'mean':>10} {'median':>10} {'std':>10}")
    for lv in sorted(sm["per_count_level"], key=int):
        v = sm["per_count_level"][lv]
        print(
            f"    {lv:>6} {v['n']:>7,} {v['mean_score']:>10.4f} "
            f"{v['median_score']:>10.4f} {v['std_score']:>10.4f}"
        )
    for w in warnings:
        print(f"    FRAGILE: {w}")


def build_arg_parser(default_mode: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Geometric BEV occupancy from stereo depth (CPU, no training)"
    )
    ap.add_argument("--mode", choices=["single", "multi"], default=default_mode)
    ap.add_argument(
        "--camera", default="front_left_color", help="colour timebase camera"
    )
    ap.add_argument("--mirror", choices=sorted(MIRRORS), default="hf")
    ap.add_argument("--data", default=None, help="reuse an existing subset dir")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument(
        "--dry-run", action="store_true", help="report the selection, fetch nothing"
    )
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument(
        "--label-column", default="count_cabin", choices=["count_view", "count_cabin"]
    )
    ap.add_argument("--score-threshold", type=float, default=0.0)
    ap.add_argument(
        "--min-range", type=float, default=0.3, help="D435i lower spec bound (m)"
    )
    ap.add_argument(
        "--max-range", type=float, default=3.0, help="D435i upper spec bound (m)"
    )
    ap.add_argument("--pixel-stride", type=int, default=4)
    ap.add_argument("--cell-size", type=float, default=0.10)
    ap.add_argument("--z-min", type=float, default=0.6)
    ap.add_argument("--z-max", type=float, default=2.0)
    ap.add_argument("--min-points-per-cell", type=int, default=3)
    ap.add_argument("--background-quantile", type=float, default=0.5)
    ap.add_argument("--depth-tolerance-ms", type=int, default=100)
    ap.add_argument("--region-dilate-m", type=float, default=0.30)
    ap.add_argument("--coverage-threshold", type=float, default=0.98)
    ap.add_argument(
        "--continue-on-gap",
        action="store_true",
        help="carry on scoring even if the rig leaves occupied floor unseen",
    )
    ap.add_argument("--out", default=None)
    return ap


def main(default_mode: str = "single", argv=None) -> int:
    args = build_arg_parser(default_mode).parse_args(argv)
    t_start = time.time()
    work_out = "/kaggle/working" if on_kaggle() else "."
    out_path = args.out or os.path.join(
        work_out, f"depth_occupancy_{args.mode}_metrics.json"
    )

    print("=" * 72)
    print(f"Sanas depth occupancy - mode: {args.mode.upper()} (CPU, no training)")
    print("=" * 72)

    diagnostic = report_mounted_kernel_source()

    work, fetch_info = ensure_subset(args)

    depth_cams = (
        [c.replace("_color", "_depth") for c in COLOR_CAMERAS]
        if args.mode == "multi"
        else [args.camera.replace("_color", "_depth")]
    )
    print("\nrig:")
    rig = build_rig(work, depth_cams)

    with open(os.path.join(work, "labels.jsonl"), encoding="utf-8") as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    rows = [r for r in rows if r.get(args.label_column) is not None]
    if args.max_frames:
        rows = rows[: args.max_frames]
    print(f"\nframes with a {args.label_column} label: {len(rows):,}")
    print(f"count distribution: {count_distribution(rows, args.label_column)}")

    cfg = GridConfig(
        cell_size=args.cell_size,
        z_min=args.z_min,
        z_max=args.z_max,
        min_points_per_cell=args.min_points_per_cell,
    )
    print(
        f"BEV grid {cfg.shape[0]}x{cfg.shape[1]} cells of {cfg.cell_size} m, "
        f"z {cfg.z_min}-{cfg.z_max} m"
    )

    kept, grids, observed, range_acc, matched = score_frames(rows, work, rig, cfg, args)
    print(
        f"\nframes scored: {len(kept):,}   matched depth frames per camera: {matched}"
    )
    if not kept:
        print("nothing to score", file=sys.stderr)
        return 1

    # ---- depth range reality check ------------------------------------
    print("\nDEPTH RANGE REALITY CHECK (share of pixels, mean over frames)")
    range_summary = {}
    for cam, acc in range_acc.items():
        if not acc:
            continue
        agg = {k: float(np.mean([a[k] for a in acc])) for k in acc[0] if k != "pixels"}
        range_summary[cam] = agg
        print(
            f"  {cam:<20} zero {100 * agg['frac_zero']:4.1f}%  "
            f"in {args.min_range}-{args.max_range}m {100 * agg['frac_in_range']:4.1f}%  "
            f"beyond {100 * agg['frac_beyond_max']:4.1f}%"
        )
    worst = min((v["frac_in_range"] for v in range_summary.values()), default=1.0)
    if worst < 0.5:
        print(
            "  VERDICT: on at least one camera most of the frame lies outside the\n"
            "  sensor's reliable window. These scores describe the NEAR FIELD ONLY."
        )

    # ---- step 1: geometric floor coverage on empty-cabin frames --------
    zero_rows = [r for r in kept if int(r[args.label_column]) == 0]
    print(f"\nSTEP 1: FLOOR COVERAGE (empty-cabin frames: {len(zero_rows):,})")
    if not zero_rows:
        print(
            "  no empty-cabin frames; cannot build a background or measure coverage",
            file=sys.stderr,
        )
        return 1

    region = person_region_cells(
        os.path.join(work, "bboxes_3d"), cfg, dilate_m=args.region_dilate_m
    )
    per_cam_obs = {}
    for cam in rig:
        obs = [
            observed[cam][int(r["timestamp"])]
            for r in zero_rows
            if int(r["timestamp"]) in observed[cam]
        ]
        per_cam_obs[cam] = merge_grids(obs) if obs else np.zeros(cfg.shape, dtype=bool)
    cov = coverage_report(per_cam_obs, region)

    print(
        f"  occupied-region cells (from every 3D box, dilated {args.region_dilate_m} m): "
        f"{cov['region_cells']:,}"
    )
    for cam, v in cov["per_camera"].items():
        print(f"    {cam:<20} covers {100 * v['region_fraction']:5.1f}% of the region")
    print(
        f"  UNION of {len(rig)} camera(s): {100 * cov['coverage_fraction']:.1f}% covered, "
        f"{cov['uncovered_cells']:,} cells unseen"
    )

    blocked = (
        args.mode == "multi" and cov["coverage_fraction"] < args.coverage_threshold
    )
    if blocked and not args.continue_on_gap:
        print("\n" + "!" * 72)
        print("RESULT: BLOCKED - four cameras still leave occupied floor unseen.")
        print(
            f"  covered {100 * cov['coverage_fraction']:.1f}% of the region people "
            f"actually occupy; {cov['uncovered_cells']:,} cells have no depth return\n"
            f"  from any camera within {args.min_range}-{args.max_range} m."
        )
        print("  Merged-BEV occupancy over an incompletely observed floor would")
        print("  understate occupancy in a way that looks like a model error rather")
        print("  than a sensing gap, so scoring stops here as instructed.")
        print("  Re-run with --continue-on-gap to score anyway, or raise --max-range")
        print("  (at the cost of trusting depth the sensor is not specified for).")
        print("!" * 72)
        payload = {
            "branch": "depth_geometric_bev",
            "mode": args.mode,
            "trained": False,
            "status": "blocked_incomplete_floor_coverage",
            "coverage": cov,
            "range_summary": range_summary,
            "frames_scored": len(kept),
            "diagnostic_kernel_source_mount": diagnostic,
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {out_path}")
        return 0

    # ---- step 2: merged BEV + step 3: correlation ----------------------
    print("\nSTEP 2/3: OCCUPANCY SCORES vs GROUND TRUTH")
    backgrounds = {
        cam: build_background(
            [
                grids[cam][int(r["timestamp"])]
                for r in zero_rows
                if int(r["timestamp"]) in grids[cam]
            ],
            args.background_quantile,
        )
        for cam in rig
    }
    merged_background = merge_grids(list(backgrounds.values()))

    counts, per_cam_scores, merged_scores = [], {cam: [] for cam in rig}, []
    for r in kept:
        ts = int(r["timestamp"])
        counts.append(int(r[args.label_column]))
        frame_grids = []
        for cam in rig:
            g = grids[cam].get(ts)
            if g is None:
                g = np.zeros(cfg.shape, dtype=bool)
            frame_grids.append(g)
            s, _ = occupancy_score(g, backgrounds[cam])
            per_cam_scores[cam].append(s)
        ms, _ = occupancy_score(merge_grids(frame_grids), merged_background)
        merged_scores.append(ms)

    warnings = fragility_warnings(np.array(counts))
    summaries = {}
    for cam in rig:
        sm = summarize_scores(per_cam_scores[cam], counts, f"single camera: {cam}")
        summaries[cam] = sm
        print_summary(sm, warnings)
    merged_summary = summarize_scores(
        merged_scores, counts, f"merged {len(rig)} camera(s)"
    )
    print_summary(merged_summary, warnings)

    # ---- step 4: head to head -----------------------------------------
    head_to_head = None
    if args.mode == "multi":
        base_cam = args.camera.replace("_color", "_depth")
        base = summaries.get(base_cam)
        print("\nSTEP 4: 1 CAMERA vs 4 CAMERAS")
        if base:
            d_rho = merged_summary["spearman_rho"] - base["spearman_rho"]
            print(
                f"  {base_cam:<22} Spearman {base['spearman_rho']:.3f} "
                f"CI [{base['spearman_ci95'][0]:.3f}, {base['spearman_ci95'][1]:.3f}]"
            )
            print(
                f"  {'merged 4 cameras':<22} Spearman {merged_summary['spearman_rho']:.3f} "
                f"CI [{merged_summary['spearman_ci95'][0]:.3f}, {merged_summary['spearman_ci95'][1]:.3f}]"
            )
            print(f"  difference: {d_rho:+.3f}")
            overlap = not (
                merged_summary["spearman_ci95"][0] > base["spearman_ci95"][1]
                or base["spearman_ci95"][0] > merged_summary["spearman_ci95"][1]
            )
            print(
                "  the two intervals OVERLAP, so this run does not establish that four\n"
                "  cameras beat one"
                if overlap
                else "  the intervals are disjoint, which is weak evidence of a real difference"
            )
            head_to_head = {
                "baseline_camera": base_cam,
                "baseline_spearman": base["spearman_rho"],
                "merged_spearman": merged_summary["spearman_rho"],
                "difference": d_rho,
                "confidence_intervals_overlap": bool(overlap),
            }

    # ---- step 5: what this does and does not test ----------------------
    scope = [
        "This measures GEOMETRY AND INFRASTRUCTURE: whether the rig sees the floor,",
        "whether depth deprojects into a coherent cabin frame, and whether a BEV",
        "occupancy score moves with occupant count at all.",
        "It does NOT test behaviour in a genuinely full cabin. The dataset never",
        "exceeds 4 occupants and this subset tops out lower, so the crowded regime",
        "that the product exists to measure is entirely unobserved.",
        "Ground truth is model-generated pseudo-labels, not human annotation.",
        "Single 32-minute daytime session: nothing here speaks to low light.",
    ]
    print("\nSTEP 5: SCOPE")
    for line in scope:
        print("  " + line)
    if warnings:
        print("\n  Statistical fragility:")
        for w in warnings:
            print("   - " + w)
        print(
            "  Correlation point estimates above are reported with bootstrap intervals\n"
            "  precisely because the point estimate alone would look more solid than it is."
        )

    payload = {
        "branch": "depth_geometric_bev",
        "mode": args.mode,
        "trained": False,
        "status": "scored",
        "label_column": args.label_column,
        "frames_scored": len(kept),
        "matched_depth_frames": matched,
        "cameras": list(rig),
        "frame_resolved": {c: rig[c]["frame_resolved"] for c in rig},
        "range_gate_m": [args.min_range, args.max_range],
        "range_summary": range_summary,
        "grid": {
            "cell_size": cfg.cell_size,
            "shape": list(cfg.shape),
            "z_min": cfg.z_min,
            "z_max": cfg.z_max,
            "min_points_per_cell": cfg.min_points_per_cell,
        },
        "coverage": cov,
        "coverage_blocked": bool(blocked),
        "background_frames": len(zero_rows),
        "per_camera": summaries,
        "merged": merged_summary,
        "head_to_head": head_to_head,
        "fragility_warnings": warnings,
        "scope": scope,
        "diagnostic_kernel_source_mount": diagnostic,
        "fetch": {k: v for k, v in fetch_info.items() if k != "selection"},
        "seconds": round(time.time() - t_start, 1),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {out_path}")
    print("=" * 72)
    return 0
