"""Run the door-APC validation locally on CPU. No GPU, no Kaggle, no training.

Selects center_left_depth frames around the 25 boarding/alighting episodes,
range-fetches them under a hard 0.5 GB cap, runs the virtual-line counter at
full resolution and as a simulated 8x8 / 4x4 multizone ToF, and evaluates
against the person_states / bboxes_3d ground truth.

FRAME-RATE DEVIATION, on the record: the brief asked for episode windows
+-5 s at the stream's full 15 Hz. Measured selection cost for that is
746 MB compressed before negatives and background; the tightest windows
that still contain the observed crossings (board [-2,+5] s, alight
[-5,+2] s, from the bbox-level dry run) cost 641 MB at 15 Hz, and no
buffer trim brings episodes + 60 s negatives + background under the
0.5 GB cap at 15 Hz. Instead of stopping with nothing, this run samples
every 2nd frame (~7.7 Hz) everywhere and fetches THREE episodes at the
full 15 Hz as a rate-sensitivity spot check. 7.7 Hz moves a 1.5 m/s
walker 0.19 m between frames, well inside the 0.8 m association gate,
and is the frame-rate class a real 8x8 multizone ToF runs at anyway.
The spot check quantifies what the halving does instead of asserting it
is harmless.

Usage:
  python development/scripts/run_door_apc.py --plan          # selection size only
  python development/scripts/run_door_apc.py --fetch         # download (respects cap)
  python development/scripts/run_door_apc.py --evaluate      # process + evaluate + write
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sanas.door_apc import (  # noqa: E402
    ApcConfig,
    Episode,
    build_depth_background,
    build_episodes,
    build_multizone_model,
    foreground_points,
    match_events_to_episodes,
    multizone_background,
    multizone_foreground_points,
    multizone_ranges,
    poisson_rate_ci,
    run_chunk,
    truth_cabin_series,
    wilson_ci,
)
from sanas.depth_occupancy import (  # noqa: E402
    load_depth_metres,
    load_extrinsics,
    load_intrinsics,
)
from sanas.selection import timestamp_of  # noqa: E402
from sanas.ziprange import MIRRORS, extract_members, load_index  # noqa: E402

NS = 1_000_000_000
DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "apc")
INDEX = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "index",
    "beintelli_v1_zip_index.json",
)
CAMERA = "center_left_depth"
HARD_CAP_BYTES = 500_000_000  # 0.5 GB, compressed selected bytes

# Window buffers, seconds. Asymmetric on purpose: measured on the bbox-level
# dry run, boarding crossings occur up to ~+4 s AFTER `enter vehicle` ends
# (the person is still on the kerb during most of the state run) and
# alighting crossings down to ~-4 s BEFORE `exit vehicle` starts.
BOARD_PRE, BOARD_POST = 2.0, 5.0
ALIGHT_PRE, ALIGHT_POST = 5.0, 2.0
FRAME_STEP = 2  # every 2nd frame (~7.7 Hz); see module docstring
# Full-rate (15 Hz) episodes for the rate-sensitivity spot check. Indices
# into the time-sorted episode list; check_spotcheck() hard-verifies that
# each index still names the (pid, kind) recorded here, because the
# 15Hz-vs-7.7Hz equivalence claim in the log rests on these three.
SPOTCHECK_EPISODES = {6: ("1", "board"), 14: ("119", "alight"), 21: ("158", "board")}
N_NEG_CHUNKS = 6
NEG_CHUNK_S = 10.5
EXCLUDE_AROUND_EPISODE_S = 8.0
BG_FRAMES_PER_STRETCH = 12
MATCH_PRE_S, MATCH_POST_S = 6.0, 6.0
# counting-band ablation: zone narrowed to |x - x_line| <= w (metres)
BAND_HALF_WIDTHS_M = [1.0, 0.7, 0.5, 0.35, 0.25]

EXPECTED = {
    "episodes": 25,
    "board": 12,
    "alight": 13,
    "enter_frames": 377,
    "exit_frames": 436,
}


def load_camera_frames() -> tuple[np.ndarray, dict]:
    idx, members = load_index(INDEX)
    frames = {}
    prefix = f"beintelli_v1/cameras/depth/{CAMERA}/images/"
    for m in members:
        if m.is_dir or not m.name.startswith(prefix):
            continue
        t = timestamp_of(m.name)
        if t is not None:
            frames[t] = m
    return np.array(sorted(frames)), frames


def check_episodes(episodes: list[Episode]) -> None:
    board = [e for e in episodes if e.kind == "board"]
    alight = [e for e in episodes if e.kind == "alight"]
    got = {
        "episodes": len(episodes),
        "board": len(board),
        "alight": len(alight),
        "enter_frames": sum(e.n_state_frames for e in board),
        "exit_frames": sum(e.n_state_frames for e in alight),
    }
    if got != EXPECTED:
        raise SystemExit(f"episode cross-check FAILED: expected {EXPECTED}, got {got}")
    print(f"episode cross-check vs log 2026-08-10: CLEAN {got}")


def check_spotcheck(episodes: list[Episode]) -> None:
    """Hard-verify the spot-check indices name the episodes they claim to."""
    for i, (pid, kind) in SPOTCHECK_EPISODES.items():
        got = (episodes[i].pid, episodes[i].kind)
        if got != (pid, kind):
            raise SystemExit(
                f"spot-check episode index {i} expected pid/kind {(pid, kind)}, "
                f"got {got} - episode ordering changed, re-derive the indices"
            )
    print(
        "spot-check episodes verified: "
        + ", ".join(f"[{i}]=pid {p} {k}" for i, (p, k) in SPOTCHECK_EPISODES.items())
    )


def merged_windows(episodes: list[Episode]) -> list[list[int]]:
    iv = []
    for e in episodes:
        pre = BOARD_PRE if e.kind == "board" else ALIGHT_PRE
        post = BOARD_POST if e.kind == "board" else ALIGHT_POST
        iv.append([e.start_ns - int(pre * NS), e.end_ns + int(post * NS)])
    iv.sort()
    out = [iv[0]]
    for a, b in iv[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def build_selection(episodes: list[Episode], ts_all: np.ndarray) -> dict:
    """Deterministic frame selection. Returns manifest dict (no download)."""
    z = np.load(os.path.join(DATA, "bbox_cache.npz"))
    bts, frame_of_box = z["ts"], z["frame"]
    box_counts = np.bincount(frame_of_box, minlength=len(bts))

    wins = merged_windows(episodes)
    spot = set()
    for i in SPOTCHECK_EPISODES:
        e = episodes[i]
        pre = BOARD_PRE if e.kind == "board" else ALIGHT_PRE
        post = BOARD_POST if e.kind == "board" else ALIGHT_POST
        spot.add((e.start_ns - int(pre * NS), e.end_ns + int(post * NS)))

    global_index = {int(t): i for i, t in enumerate(ts_all)}

    def frames_in(a: int, b: int, step: int) -> list[int]:
        i0, i1 = np.searchsorted(ts_all, a), np.searchsorted(ts_all, b, "right")
        return [int(t) for t in ts_all[i0:i1] if global_index[int(t)] % step == 0]

    episode_chunks = []
    for a, b in wins:
        full_rate = any(sa < b and sb > a for sa, sb in spot)
        step = 1 if full_rate else FRAME_STEP
        episode_chunks.append(
            {
                "kind": "episode",
                "start_ns": int(a),
                "end_ns": int(b),
                "full_rate": full_rate,
                "ts": frames_in(a, b, step),
            }
        )

    # exclusion zones for negatives/background
    excl = [
        [
            e.start_ns - int(EXCLUDE_AROUND_EPISODE_S * NS),
            e.end_ns + int(EXCLUDE_AROUND_EPISODE_S * NS),
        ]
        for e in episodes
    ]

    def clear_of_episodes(a: int, b: int) -> bool:
        return all(b <= s or a >= t for s, t in excl)

    # negative chunks: spread across the session, activity preferred
    session_a, session_b = int(ts_all[0]), int(ts_all[-1])
    want = int(NEG_CHUNK_S * NS)
    candidates = []
    tcur = session_a
    while tcur + want < session_b:
        if clear_of_episodes(tcur, tcur + want):
            i0, i1 = np.searchsorted(bts, tcur), np.searchsorted(bts, tcur + want)
            occ = box_counts[i0:i1]
            candidates.append(
                (tcur, tcur + want, float(np.median(occ)) if occ.size else 0.0)
            )
            tcur += want
        else:
            tcur += NS
    if len(candidates) < N_NEG_CHUNKS:
        raise SystemExit(f"only {len(candidates)} negative candidates found")
    picks_idx = np.linspace(0, len(candidates) - 1, N_NEG_CHUNKS).round().astype(int)
    negative_chunks = [
        {
            "kind": "negative",
            "start_ns": candidates[i][0],
            "end_ns": candidates[i][1],
            "median_cabin_occupancy": candidates[i][2],
            "ts": frames_in(candidates[i][0], candidates[i][1], FRAME_STEP),
        }
        for i in picks_idx
    ]

    # background: longest empty-cabin stretches clear of episodes
    empty_ts = bts[box_counts == 0]
    stretches, a0, prev = [], int(empty_ts[0]), int(empty_ts[0])
    for t in empty_ts[1:]:
        t = int(t)
        if t - prev > NS:
            stretches.append((a0, prev))
            a0 = t
        prev = t
    stretches.append((a0, prev))
    stretches = [s for s in stretches if clear_of_episodes(*s)]
    stretches.sort(key=lambda s: s[0] - s[1])  # longest first
    bg_ts = []
    for s0, s1 in stretches[:4]:
        cand = frames_in(s0, s1, FRAME_STEP)
        if len(cand) > BG_FRAMES_PER_STRETCH:
            keep = (
                np.linspace(0, len(cand) - 1, BG_FRAMES_PER_STRETCH).round().astype(int)
            )
            cand = [cand[i] for i in keep]
        bg_ts.extend(cand)

    all_ts = sorted(
        {t for c in episode_chunks + negative_chunks for t in c["ts"]} | set(bg_ts)
    )
    return {
        "camera": CAMERA,
        "frame_step": FRAME_STEP,
        "buffers_s": {
            "board": [BOARD_PRE, BOARD_POST],
            "alight": [ALIGHT_PRE, ALIGHT_POST],
        },
        "episode_chunks": episode_chunks,
        "negative_chunks": negative_chunks,
        "background_ts": bg_ts,
        "all_ts": all_ts,
    }


def selection_bytes(manifest: dict, member_of: dict) -> int:
    return sum(member_of[t].csize for t in manifest["all_ts"])


def frame_path(t: int) -> str:
    return os.path.join(DATA, "cameras", "depth", CAMERA, "images", f"{t}.png")


def fetch(manifest: dict, member_of: dict) -> dict:
    missing = [t for t in manifest["all_ts"] if not os.path.exists(frame_path(t))]
    print(
        f"frames selected: {len(manifest['all_ts']):,}  already local: "
        f"{len(manifest['all_ts']) - len(missing):,}  to fetch: {len(missing):,}"
    )
    if not missing:
        return {"fetched": 0}
    members = [member_of[t] for t in missing]
    total = sum(m.csize for m in members)
    if total > HARD_CAP_BYTES:
        raise SystemExit(
            f"selection {total / 1e9:.3f} GB exceeds hard cap - refusing to download"
        )
    # small gap_merge: adjacent selected frames merge, skipped frames are NOT transferred
    stats = extract_members(MIRRORS["hf"], members, DATA, gap_merge=4096, verbose=True)
    return stats


# --------------------------------------------------------------------------
# Processing
# --------------------------------------------------------------------------


def process_all(manifest: dict, cfg: ApcConfig) -> dict:
    intr = load_intrinsics(
        os.path.join(DATA, "cameras", "depth", CAMERA, "camera_info.yaml")
    )
    T = load_extrinsics(DATA, CAMERA)
    print(
        f"rig: {CAMERA} {intr.width}x{intr.height} fx={intr.fx:.1f} "
        f"origin={np.round(T[:3, 3], 3).tolist()}"
    )

    # completeness gate: --evaluate on a partial fetch must fail loudly, not
    # silently write results from whatever happens to be on disk
    missing_all = [t for t in manifest["all_ts"] if not os.path.exists(frame_path(t))]
    if missing_all:
        raise SystemExit(
            f"{len(missing_all)} of {len(manifest['all_ts'])} selected frames "
            f"missing on disk (first: {missing_all[0]}) - run --fetch to "
            "completion before --evaluate"
        )
    bg_paths = [frame_path(t) for t in manifest["background_ts"]]
    if len(bg_paths) != len(manifest["background_ts"]):
        raise SystemExit("background frame list inconsistent with manifest")
    print(f"background frames: {len(bg_paths)}")
    bg_full = build_depth_background(bg_paths, cfg.pixel_stride)

    mz8 = build_multizone_model(intr, 8, cfg)
    mz4 = build_multizone_model(intr, 4, cfg)
    bg8 = multizone_background(
        [multizone_ranges(load_depth_metres(p), mz8, cfg) for p in bg_paths]
    )
    bg4 = multizone_background(
        [multizone_ranges(load_depth_metres(p), mz4, cfg) for p in bg_paths]
    )

    variants = {
        "full": {"events": [], "chunks": []},
        "mz8": {"events": [], "chunks": []},
        "mz4": {"events": [], "chunks": []},
    }
    spot_full_rate, spot_half_rate = [], []
    # counting-band ablation (coordinator request, node_design.md s.3): rerun
    # the full-res path with the zone narrowed to |x - x_line| <= w. Answers
    # how narrow the effective counting band can get before detection degrades.
    # Oblique-geometry caveat: this narrows the band in base_link, not on a
    # ceiling-mounted sensor's floor footprint.
    band_events = {w: [] for w in BAND_HALF_WIDTHS_M}

    chunks = manifest["episode_chunks"] + manifest["negative_chunks"]
    t0 = time.time()
    n_frames = 0
    for ch in chunks:
        ts_all_local = list(ch["ts"])
        missing = [t for t in ts_all_local if not os.path.exists(frame_path(t))]
        if missing:
            raise SystemExit(
                f"chunk at {ch['start_ns']} is missing {len(missing)} of "
                f"{len(ts_all_local)} frames - partial fetch, refusing to score"
            )
        # uniform ~7.7 Hz for the headline result: full-rate (spot check)
        # chunks are subsampled back to every 2nd frame here
        ts_half = (
            [t for i, t in enumerate(ts_all_local) if i % 2 == 0]
            if ch.get("full_rate")
            else ts_all_local
        )
        ts_half_set = set(ts_half)
        pts_cache: dict[int, np.ndarray] = {}
        pts_mz8, pts_mz4 = [], []
        for t in ts_all_local:
            d = load_depth_metres(frame_path(t))
            pts_cache[t] = foreground_points(d, bg_full, intr, T, cfg)
            if t in ts_half_set:
                r8 = multizone_ranges(d, mz8, cfg)
                pts_mz8.append((t, multizone_foreground_points(r8, bg8, mz8, T, cfg)))
                r4 = multizone_ranges(d, mz4, cfg)
                pts_mz4.append((t, multizone_foreground_points(r4, bg4, mz4, T, cfg)))
            n_frames += 1
        pts_full = [(t, pts_cache[t]) for t in ts_half]
        for name, stream, cell, minpts in (
            ("full", pts_full, cfg.cluster_cell_m, cfg.min_cluster_points),
            ("mz8", pts_mz8, cfg.mz_cluster_cell_m, cfg.mz_min_cluster_points),
            ("mz4", pts_mz4, cfg.mz_cluster_cell_m, cfg.mz_min_cluster_points),
        ):
            evs = run_chunk(stream, cfg, cell, minpts)
            variants[name]["events"].append(
                (ch["kind"], ch["start_ns"], ch["end_ns"], evs)
            )
        for w in BAND_HALF_WIDTHS_M:
            narrowed = [
                (t, p[np.abs(p[:, 0] - cfg.x_line) <= w] if p.size else p)
                for t, p in pts_full
            ]
            evs = run_chunk(narrowed, cfg, cfg.cluster_cell_m, cfg.min_cluster_points)
            band_events[w].append((ch["kind"], ch["start_ns"], ch["end_ns"], evs))
        # rate-sensitivity spot check on full-rate chunks (full-res path only)
        if ch.get("full_rate"):
            stream_15 = [(t, pts_cache[t]) for t in ts_all_local]
            evs_15 = run_chunk(
                stream_15, cfg, cfg.cluster_cell_m, cfg.min_cluster_points
            )
            evs_7 = run_chunk(pts_full, cfg, cfg.cluster_cell_m, cfg.min_cluster_points)
            spot_full_rate.append((ch["start_ns"], evs_15))
            spot_half_rate.append((ch["start_ns"], evs_7))
    print(f"processed {n_frames:,} frames x 3 variants in {time.time() - t0:.1f} s")
    return {
        "variants": variants,
        "band_events": band_events,
        "spot_full": spot_full_rate,
        "spot_half": spot_half_rate,
    }


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def evaluate_variant(
    name: str,
    chunk_events: list,
    episodes: list[Episode],
    manifest: dict,
) -> dict:
    episode_events = [
        ev for kind, _, _, evs in chunk_events if kind == "episode" for ev in evs
    ]
    negative_events = [
        ev for kind, _, _, evs in chunk_events if kind == "negative" for ev in evs
    ]

    m = match_events_to_episodes(episode_events, episodes, MATCH_PRE_S, MATCH_POST_S)
    per_ep = m["per_episode"]
    tp = sum(1 for r in per_ep if r["detected"])
    fn = len(per_ep) - tp
    dir_correct = sum(1 for r in per_ep if r["direction_correct"])
    dup_events = sum(max(0, r["n_events_in_window"] - 1) for r in per_ep)
    stray = len(m["unassigned_events"])

    neg_seconds = sum(
        (c["end_ns"] - c["start_ns"]) / NS for c in manifest["negative_chunks"]
    )
    fp_neg = len(negative_events)

    det_lo, det_hi = wilson_ci(tp, len(per_ep))
    dir_lo, dir_hi = wilson_ci(dir_correct, tp) if tp else (float("nan"), float("nan"))
    fp_lo, fp_hi = poisson_rate_ci(fp_neg, neg_seconds / 60.0)

    not1 = [r for r in per_ep if r["pid"] != "1"]
    tp_not1 = sum(1 for r in not1 if r["detected"])

    return {
        "variant": name,
        "episodes": len(per_ep),
        "true_positives": tp,
        "false_negatives": fn,
        "detection_rate": tp / len(per_ep),
        "detection_ci95": [det_lo, det_hi],
        "direction_correct_of_detected": dir_correct,
        "direction_accuracy": (dir_correct / tp) if tp else None,
        "direction_ci95": [dir_lo, dir_hi],
        "duplicate_events_in_windows": dup_events,
        "stray_events_in_episode_chunks": stray,
        "false_positives_on_negatives": fp_neg,
        "negative_seconds": neg_seconds,
        "fp_per_minute": fp_neg / (neg_seconds / 60.0),
        "fp_per_minute_ci95": [fp_lo, fp_hi],
        "excluding_pid1": {"episodes": len(not1), "true_positives": tp_not1},
        "per_episode": per_ep,
    }


def nt_series(
    name: str,
    chunk_events: list,
    manifest: dict,
    x_line: float,
    episodes: list[Episode],
) -> dict:
    z = np.load(os.path.join(DATA, "bbox_cache.npz"))
    bts, centers, frame_of_box = z["ts"], z["centers"], z["frame"]
    truth = truth_cabin_series(bts, centers, frame_of_box, x_line)

    events = sorted(
        (ev for _, _, _, evs in chunk_events for ev in evs), key=lambda e: e.t_ns
    )
    processed = sorted(
        t
        for c in manifest["episode_chunks"] + manifest["negative_chunks"]
        for t in c["ts"]
    )
    t_first = processed[0]
    i0 = int(np.searchsorted(bts, t_first))
    i0 = min(i0, len(bts) - 1)
    n0 = int(truth[i0])

    est_at = {}
    n = n0
    ei = 0
    for t in processed:
        while ei < len(events) and events[ei].t_ns <= t:
            n += 1 if events[ei].kind == "board" else -1
            ei += 1
        est_at[t] = n

    # DOOR-EVENT truth: cumulative count from the labelled episodes alone
    # (each board +1 / alight -1 applied at the episode midpoint). The
    # line-position truth above and this door truth DIVERGE on this staged
    # recording: actors already inside the bus wander across the x_line
    # plane between episodes (e.g. the wheelchair user around t+262 s), so
    # the line-based series moves without any door event. The line MAE
    # therefore mixes counter error with that semantic gap; the door MAE
    # isolates counting fidelity against the 25 labelled events.
    door_deltas = sorted(
        ((ep.start_ns + ep.end_ns) // 2, 1 if ep.kind == "board" else -1)
        for ep in episodes
    )

    def door_truth_at(t: int) -> int:
        return n0 + sum(d for tt, d in door_deltas if t_first < tt <= t)

    errs, door_errs = [], []
    for t in processed:
        j = int(np.clip(np.searchsorted(bts, t), 0, len(bts) - 1))
        if abs(int(bts[j]) - t) > 200_000_000 and j > 0:
            j -= 1
        errs.append(est_at[t] - int(truth[j]))
        door_errs.append(est_at[t] - door_truth_at(t))
    errs = np.asarray(errs, dtype=float)
    door_errs = np.asarray(door_errs, dtype=float)

    # block bootstrap over chunks for the MAE interval: frames are strongly
    # autocorrelated, so resampling frames would fake precision. Blocks are
    # built by chunk membership (interval containment), not manifest order.
    chunks_sorted = sorted(
        manifest["episode_chunks"] + manifest["negative_chunks"],
        key=lambda c: c["start_ns"],
    )
    bounds = [(c["start_ns"], c["end_ns"]) for c in chunks_sorted]

    def block_of(t: int) -> int:
        for i, (a, b) in enumerate(bounds):
            if a <= t <= b:
                return i
        # defensive only: bounds covers BOTH episode and negative chunks, and
        # every processed frame comes from one of them, so this is unreachable
        # unless the manifest is inconsistent; strays would share one block
        return len(bounds)

    block_ids = np.array([block_of(t) for t in processed])
    rng = np.random.default_rng(0)

    def block_boot_mae(e: np.ndarray) -> tuple[float, float]:
        blocks = [e[block_ids == i] for i in np.unique(block_ids)]
        maes = []
        for _ in range(2000):
            pick = rng.integers(0, len(blocks), len(blocks))
            v = np.concatenate([blocks[i] for i in pick])
            if v.size:
                maes.append(float(np.abs(v).mean()))
        if not maes:
            return (float("nan"), float("nan"))
        return (float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5)))

    mae_lo, mae_hi = block_boot_mae(errs)
    dmae_lo, dmae_hi = block_boot_mae(door_errs)

    # 1 Hz series for figures
    grid = np.arange(int(bts[0]), int(bts[-1]), NS)
    truth_1hz = truth[np.clip(np.searchsorted(bts, grid), 0, len(bts) - 1)]
    est_1hz, n, ei = [], n0, 0
    for t in grid:
        while ei < len(events) and events[ei].t_ns <= t:
            n += 1 if events[ei].kind == "board" else -1
            ei += 1
        est_1hz.append(n)
    return {
        "variant": name,
        "initial_count_anchor": n0,
        "mae_vs_line_truth": float(np.abs(errs).mean()),
        "mae_vs_line_truth_ci95": [float(mae_lo), float(mae_hi)],
        "final_error_vs_line_truth": int(errs[-1]),
        "mae_vs_door_truth": float(np.abs(door_errs).mean()),
        "mae_vs_door_truth_ci95": [float(dmae_lo), float(dmae_hi)],
        "final_error_vs_door_truth": int(door_errs[-1]),
        "final_truth": int(truth[-1]),
        "n_frames_evaluated": int(errs.size),
        "series_1hz": {
            "t_rel_s": [round((int(t) - int(grid[0])) / NS, 1) for t in grid],
            "truth": [int(v) for v in truth_1hz],
            "door_truth": [int(door_truth_at(int(t))) for t in grid],
            "estimate": [int(v) for v in est_1hz],
        },
        "events": [
            {
                "t_ns": e.t_ns,
                "kind": e.kind,
                "x_from": round(e.x_from, 3),
                "x_to": round(e.x_to, 3),
            }
            for e in events
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--plan", action="store_true", help="print selection size, no download"
    )
    ap.add_argument("--fetch", action="store_true", help="download selected frames")
    ap.add_argument(
        "--evaluate", action="store_true", help="process + evaluate + write results"
    )
    args = ap.parse_args()

    episodes = build_episodes(os.path.join(DATA, "person_states.json"))
    check_episodes(episodes)
    check_spotcheck(episodes)
    ts_all, member_of = load_camera_frames()
    manifest = build_selection(episodes, ts_all)
    sel_bytes = selection_bytes(manifest, member_of)
    n_full = sum(len(c["ts"]) for c in manifest["episode_chunks"] if c.get("full_rate"))
    print(
        f"selection: {len(manifest['all_ts']):,} frames "
        f"({len(manifest['episode_chunks'])} episode chunks, "
        f"{len(manifest['negative_chunks'])} negative chunks, "
        f"{len(manifest['background_ts'])} background frames, "
        f"{n_full} at full 15 Hz for the spot check)"
    )
    print(f"selected compressed bytes: {sel_bytes / 1e9:.3f} GB  (hard cap 0.5 GB)")
    if sel_bytes > HARD_CAP_BYTES:
        raise SystemExit("over the hard cap - refusing to proceed")
    with open(os.path.join(DATA, "apc_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)

    if args.plan and not (args.fetch or args.evaluate):
        return 0
    if args.fetch:
        stats = fetch(manifest, member_of)
        print(
            f"fetch stats: {json.dumps({k: v for k, v in stats.items() if k != 'renamed'})}"
        )
    if not args.evaluate:
        return 0

    cfg = ApcConfig()
    results = process_all(manifest, cfg)

    out = {
        "experiment": "door_apc_virtual_line",
        "camera": CAMERA,
        "backend": "local CPU",
        "frame_rate_hz_nominal": 15.4,
        "frame_step": FRAME_STEP,
        "config_untuned_defaults": {k: getattr(cfg, k) for k in vars(cfg)},
        "buffers_s": manifest["buffers_s"],
        "match_window_s": [MATCH_PRE_S, MATCH_POST_S],
        "definitions": {
            "line_truth": (
                "count of bboxes_3d human boxes with center x < x_line, per "
                "annotated frame over the WHOLE session. Moves when anyone "
                "crosses the line plane, door event or not (92 transitions, "
                "range 0-2). NOT chunk-gated, NOT zero-filled."
            ),
            "door_truth": (
                "initial_count_anchor + cumulative sum of the 25 labelled "
                "episodes (+1 board / -1 alight), each applied at the episode "
                "midpoint. The clean 25-step staircase Fig. 1 expects is "
                "series_1hz.door_truth."
            ),
            "series_1hz.estimate": (
                "RAW cumulative count (not an error), anchored at "
                "initial_count_anchor, over a 1 Hz grid spanning the bbox "
                "session. Frozen across intervals outside the processed "
                "chunks, because events can only be observed there."
            ),
            "series_1hz.truth": (
                "line_truth sampled at 1 Hz; each sample takes the nearest "
                "FOLLOWING annotated bbox frame (<=0.1 s ahead in continuous "
                "recording, up to the gap length across the 10 recording "
                "gaps)."
            ),
            "mae_vs_*": (
                "mean |estimate - truth| over the n_frames_evaluated "
                "PROCESSED frames only (episode + negative chunks at the "
                "evaluation rate), NOT over the 1 Hz grid: outside processed "
                "chunks the estimate is frozen on intervals the counter "
                "never saw, so on-grid MAE mixes counter error with "
                "unobservable gap drift and will differ."
            ),
            "final_error_vs_door_truth": (
                "estimate minus door_truth AT THE LAST PROCESSED FRAME (an "
                "error, not a count). series_1hz.estimate endpoints are raw "
                "counts; subtract the session-end door_truth (= anchor + "
                "12 boards - 13 alights = anchor - 1) to reconcile: est "
                "0/-4/+2 (mz8/full/mz4) -> error +1/-3/+3."
            ),
            "final_error_vs_line_truth": (
                "estimate minus line_truth at the last processed frame."
            ),
        },
        "variants": {},
        "nt": {},
    }
    for name in ("full", "mz8", "mz4"):
        ev = evaluate_variant(
            name, results["variants"][name]["events"], episodes, manifest
        )
        nt = nt_series(
            name, results["variants"][name]["events"], manifest, cfg.x_line, episodes
        )
        out["variants"][name] = ev
        out["nt"][name] = nt
        print(f"\n=== {name} ===")
        print(
            f"  detected {ev['true_positives']}/{ev['episodes']} "
            f"(rate {ev['detection_rate']:.2f}, 95% CI "
            f"[{ev['detection_ci95'][0]:.2f}, {ev['detection_ci95'][1]:.2f}])"
        )
        print(
            f"  direction correct: {ev['direction_correct_of_detected']}/{ev['true_positives']}"
        )
        print(
            f"  duplicates in windows: {ev['duplicate_events_in_windows']}  "
            f"stray in episode chunks: {ev['stray_events_in_episode_chunks']}"
        )
        print(
            f"  FPs on {ev['negative_seconds']:.0f} s negatives: "
            f"{ev['false_positives_on_negatives']} "
            f"({ev['fp_per_minute']:.2f}/min, CI [{ev['fp_per_minute_ci95'][0]:.2f}, "
            f"{ev['fp_per_minute_ci95'][1]:.2f}])"
        )
        print(
            f"  N_t MAE vs door truth {nt['mae_vs_door_truth']:.2f} "
            f"CI [{nt['mae_vs_door_truth_ci95'][0]:.2f}, {nt['mae_vs_door_truth_ci95'][1]:.2f}]  "
            f"final {nt['final_error_vs_door_truth']:+d}"
        )
        print(
            f"  N_t MAE vs line truth {nt['mae_vs_line_truth']:.2f} "
            f"CI [{nt['mae_vs_line_truth_ci95'][0]:.2f}, {nt['mae_vs_line_truth_ci95'][1]:.2f}]  "
            f"final {nt['final_error_vs_line_truth']:+d}  "
            "(includes non-door line crossings in unprocessed gaps)"
        )

    # counting-band ablation
    out["band_ablation"] = {}
    print("\ncounting-band ablation (full-res path, zone |x - x_line| <= w):")
    for w in BAND_HALF_WIDTHS_M:
        ev = evaluate_variant(
            f"band_{w}", results["band_events"][w], episodes, manifest
        )
        out["band_ablation"][str(w)] = ev
        print(
            f"  w={w:.2f} m: detected {ev['true_positives']}/{ev['episodes']}, "
            f"direction {ev['direction_correct_of_detected']}/{ev['true_positives']}, "
            f"FPs {ev['false_positives_on_negatives']}, "
            f"duplicates {ev['duplicate_events_in_windows']}"
        )
    print(
        "  caveat: band narrowed in base_link on an OBLIQUE view; a ceiling-"
        "mounted sensor's floor footprint narrows differently."
    )

    # spot check
    spot = []
    for (t0, e15), (_, e7) in zip(results["spot_full"], results["spot_half"]):
        spot.append(
            {
                "chunk_start_ns": t0,
                "events_15hz": [[e.t_ns, e.kind] for e in e15],
                "events_7hz": [[e.t_ns, e.kind] for e in e7],
                "same_count_and_kinds": [e.kind for e in e15] == [e.kind for e in e7],
            }
        )
    out["rate_sensitivity_spot_check"] = spot
    print(
        "\nrate spot check (15 Hz vs 7.7 Hz, full-res path): "
        + ", ".join(str(s["same_count_and_kinds"]) for s in spot)
    )

    print(
        "\nFRAGILE: n=25 events total; one person (pid 1) contributes 8 of 25; "
        "a single miscount moves detection rate by 4 percentage points."
    )
    print(
        "FRAGILE: oblique across-the-aisle viewpoint, not APC top-down; single "
        "door; staged actors; stationary bus; pseudo-label ground truth."
    )

    os.makedirs(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "research", "paper", "results"
        ),
        exist_ok=True,
    )
    out_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "research",
        "paper",
        "results",
        "apc_validation.json",
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
