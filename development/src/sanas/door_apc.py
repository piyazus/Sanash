"""Door-mounted virtual-line APC on center_left_depth. No training, no GPU.

Phase 1 core experiment: a boarding/alighting counter built from depth
geometry alone. Deprojects each depth frame with center_left_depth's OWN
intrinsics (848x480, fx~426.6 -- NOT the colour intrinsics and NOT
front_left's), transforms into `base_link` via the archive's own calibration
chain, background-subtracts against empty-cabin depth, restricts to a
vestibule zone, clusters foreground into per-frame centroids, tracks them
with a nearest-neighbour tracker (1-2 people, not a crowd), and counts
signed crossings of a virtual line as boardings/alightings.

LINE PLACEMENT (Finding C, log 2026-08-10): the labelled door-threshold
positions are 3.24-5.96 m from center_left, outside the D435i 0.3-3.0 m spec
window, so the counting line sits INSIDE the cabin, in the vestibule, within
the 1.2-2.2 m reliable band. This deviates from real APC geometry (top-down
over the aperture); every result carries that caveat.

Geometry, measured from the archive calibration + bboxes_3d (2026-08-10):
  - base_link: x is across the bus (door side positive), y runs along the
    bus, z up, floor at z~0. Occupant bbox centres span x -2.7..0.5 inside
    the cabin; the kerb crowd sits at x 3..7; the door corridor footprint is
    x 0.5-1.5, y 0.05-1.05 (Finding B).
  - center_left_depth origin [-1.335, 0.398, 1.988], optical axis
    (0.922, -0.032, -0.385): mounted on the far wall looking ACROSS the bus
    at the door, 67 deg off vertical. Oblique, not top-down.
  - bbox-level dry run of the line test: boarding crossings occur up to
    ~+4 s AFTER the `enter vehicle` state run ends, alighting crossings down
    to ~-4 s BEFORE `exit vehicle` starts, and a person hovering at the line
    chatters across it for seconds. Hence asymmetric evaluation windows and
    a hysteresis band around the line instead of a bare plane test.

The multizone variant simulates a cheap 64-zone ToF (ST VL53L7CX class) by
cropping the depth frame to a ~60 deg square FoV and pooling it to an 8x8
(or 4x4) grid of per-zone ranges, then running the IDENTICAL zone / line /
tracker logic on the zone points. Simulation fidelity limits: oblique
viewpoint instead of the top-down mount such a sensor would get, D435i
stereo noise instead of ToF noise, per-zone range approximated as a low
percentile of the pixel block, single door, staged actors.

Ground truth comes from person_states.json (`enter vehicle` / `exit
vehicle` per-frame states grouped into episodes) and bboxes_3d (cabin count
over time). Both are pseudo-labels; 25 events total is the entire
validation set and one person contributes 8 of them.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

from .depth_occupancy import Intrinsics, load_depth_metres

NS = 1_000_000_000


# --------------------------------------------------------------------------
# Ground-truth episodes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Episode:
    pid: str
    kind: str  # "board" | "alight"
    start_ns: int
    end_ns: int
    n_state_frames: int

    @property
    def duration_s(self) -> float:
        return (self.end_ns - self.start_ns) / NS


STATE_TO_KIND = {"enter vehicle": "board", "exit vehicle": "alight"}


def build_episodes(person_states_path: str, gap_ns: int = 2 * NS) -> list[Episode]:
    """Group per-frame `enter vehicle`/`exit vehicle` states into episodes.

    Contiguous runs per (person, state) with a 2 s gap tolerance. On this
    archive that yields 12 boardings + 13 alightings = 25 episodes from 377
    `enter vehicle` and 436 `exit vehicle` frames; any other total means the
    input changed and must be investigated, so callers should cross-check.
    """
    with open(person_states_path, encoding="utf-8") as fh:
        frames = json.load(fh)["frames"]
    runs: dict[tuple[str, str], list[int]] = {}
    for ts, fr in frames.items():
        t = int(ts)
        for pid, state in fr.get("states", {}).items():
            if state in STATE_TO_KIND:
                runs.setdefault((pid, state), []).append(t)
    episodes: list[Episode] = []
    for (pid, state), tss in runs.items():
        tss.sort()
        start = prev = tss[0]
        n = 1
        for t in tss[1:]:
            if t - prev > gap_ns:
                episodes.append(Episode(pid, STATE_TO_KIND[state], start, prev, n))
                start, n = t, 0
            prev = t
            n += 1
        episodes.append(Episode(pid, STATE_TO_KIND[state], start, prev, n))
    episodes.sort(key=lambda e: e.start_ns)
    return episodes


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class ApcConfig:
    """Counter parameters. EVERY default below is UNTUNED unless the comment
    says otherwise: values were chosen once from geometry inspection of the
    bboxes_3d trajectories and never optimised against the outcome."""

    # Virtual counting line: plane of constant x in base_link. 0.20 m puts it
    # ~1.8 m from center_left at corridor centre, inside the 1.2-2.2 m
    # reliable band (Finding C). UNTUNED.
    x_line: float = 0.20
    # Hysteresis half-width around the line: a track must clear the band to
    # register a crossing, otherwise doorway hover chatters. UNTUNED.
    hysteresis_m: float = 0.15
    # Vestibule zone in base_link (m). x spans aisle edge to door corridor,
    # y the door corridor plus margin, z above floor clutter and below
    # ceiling. All UNTUNED.
    zone_x: tuple = (-0.6, 1.6)
    zone_y: tuple = (-0.2, 1.3)
    zone_z: tuple = (0.2, 2.0)
    # Depth validity gate: the D435i spec window (paper-stated, not tuned).
    min_range: float = 0.3
    max_range: float = 3.0
    # Background subtraction: pixel is foreground when its range is shorter
    # than the empty-cabin background by this margin, or when the background
    # has no return there. UNTUNED.
    bg_margin_m: float = 0.15
    # Full-resolution path only. Stride 2 keeps ~100k candidate pixels.
    pixel_stride: int = 2
    # Centroid extraction: connected components on a coarse floor-plane grid.
    # UNTUNED. min_cluster_points is at pixel_stride 2; scale if you change
    # the stride.
    cluster_cell_m: float = 0.15
    min_cluster_points: int = 60
    # Multizone path: same clustering but on zone points, so the thresholds
    # are necessarily different. UNTUNED.
    mz_cluster_cell_m: float = 0.40
    mz_min_cluster_points: int = 1
    mz_bg_margin_m: float = 0.20
    # Per-zone range = this percentile of valid pixels in the block, a stand-
    # in for a ToF's strongest/nearest target. UNTUNED, fidelity caveat.
    mz_range_percentile: float = 20.0
    mz_fov_deg: float = 60.0
    # Tracker: nearest-neighbour gate and drop-out. UNTUNED.
    max_assoc_m: float = 0.8
    max_track_gap_s: float = 0.7


# --------------------------------------------------------------------------
# Foreground extraction - full resolution
# --------------------------------------------------------------------------


def build_depth_background(paths: list[str], stride: int) -> np.ndarray:
    """Per-pixel median range over empty-cabin frames (0 = no return)."""
    if not paths:
        raise ValueError("no empty-cabin frames to build a depth background")
    stack = np.stack([load_depth_metres(p)[::stride, ::stride] for p in paths])
    masked = np.where(stack > 0, stack, np.nan)
    with np.errstate(all="ignore"):
        bg = np.nanmedian(masked, axis=0)
    return np.nan_to_num(bg, nan=0.0).astype(np.float32)


def foreground_points(
    depth_m: np.ndarray,
    background: np.ndarray,
    intr: Intrinsics,
    T: np.ndarray,
    cfg: ApcConfig,
) -> np.ndarray:
    """(N,3) base_link points that are foreground AND inside the vestibule zone."""
    s = cfg.pixel_stride
    d = depth_m[::s, ::s]
    valid = (d >= cfg.min_range) & (d <= cfg.max_range)
    fg = valid & ((background <= 0) | (d < background - cfg.bg_margin_m))
    ys, xs = np.nonzero(fg)
    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    z = d[ys, xs]
    u = xs.astype(np.float32) * s
    v = ys.astype(np.float32) * s
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    pts = np.stack([x, y, z], axis=1)
    pts = pts @ T[:3, :3].T + T[:3, 3]
    m = (
        (pts[:, 0] >= cfg.zone_x[0])
        & (pts[:, 0] <= cfg.zone_x[1])
        & (pts[:, 1] >= cfg.zone_y[0])
        & (pts[:, 1] <= cfg.zone_y[1])
        & (pts[:, 2] >= cfg.zone_z[0])
        & (pts[:, 2] <= cfg.zone_z[1])
    )
    return pts[m].astype(np.float32)


# --------------------------------------------------------------------------
# Foreground extraction - simulated multizone ToF
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MultizoneModel:
    """Fixed geometry of the simulated multizone sensor."""

    rows: int
    cols: int
    rays: np.ndarray  # (rows*cols, 3) unit rays in the camera optical frame
    row_edges: np.ndarray
    col_edges: np.ndarray

    @property
    def n_zones(self) -> int:
        return self.rows * self.cols


def build_multizone_model(
    intr: Intrinsics, zones: int, cfg: ApcConfig
) -> MultizoneModel:
    """Crop a ~fov_deg square FoV around the principal point, split into
    zones x zones blocks, one ray through each block centre.

    The D435i frame is 848x480 (~87x58 deg); a 60 deg square crop is 493 px
    wide and taller than the frame, so the vertical FoV is whatever the
    sensor has (~58 deg) - stated rather than pretended away.
    """
    half_w = math.tan(math.radians(cfg.mz_fov_deg / 2)) * intr.fx
    c0 = max(0, int(round(intr.cx - half_w)))
    c1 = min(intr.width, int(round(intr.cx + half_w)))
    col_edges = np.linspace(c0, c1, zones + 1).round().astype(int)
    row_edges = np.linspace(0, intr.height, zones + 1).round().astype(int)
    rays = []
    for r in range(zones):
        v = (row_edges[r] + row_edges[r + 1]) / 2.0
        for c in range(zones):
            u = (col_edges[c] + col_edges[c + 1]) / 2.0
            ray = np.array([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, 1.0])
            rays.append(ray / np.linalg.norm(ray))
    return MultizoneModel(zones, zones, np.array(rays), row_edges, col_edges)


def multizone_ranges(
    depth_m: np.ndarray, mz: MultizoneModel, cfg: ApcConfig
) -> np.ndarray:
    """Pool a depth frame to per-zone ranges (0 = no return in that zone)."""
    out = np.zeros(mz.n_zones, dtype=np.float32)
    k = 0
    for r in range(mz.rows):
        for c in range(mz.cols):
            block = depth_m[
                mz.row_edges[r] : mz.row_edges[r + 1],
                mz.col_edges[c] : mz.col_edges[c + 1],
            ]
            vals = block[(block >= cfg.min_range) & (block <= cfg.max_range)]
            if vals.size:
                out[k] = np.percentile(vals, cfg.mz_range_percentile)
            k += 1
    return out


def multizone_background(range_rows: list[np.ndarray]) -> np.ndarray:
    """Per-zone median range over empty-cabin frames (0 = no return)."""
    stack = np.stack(range_rows)
    masked = np.where(stack > 0, stack, np.nan)
    with np.errstate(all="ignore"):
        bg = np.nanmedian(masked, axis=0)
    return np.nan_to_num(bg, nan=0.0).astype(np.float32)


def multizone_foreground_points(
    ranges: np.ndarray,
    background: np.ndarray,
    mz: MultizoneModel,
    T: np.ndarray,
    cfg: ApcConfig,
) -> np.ndarray:
    """Zone rays whose range is foreground, deprojected into the zone."""
    valid = ranges > 0
    fg = valid & ((background <= 0) | (ranges < background - cfg.mz_bg_margin_m))
    if not fg.any():
        return np.empty((0, 3), dtype=np.float32)
    pts = mz.rays[fg] * ranges[fg, None]
    pts = pts @ T[:3, :3].T + T[:3, 3]
    m = (
        (pts[:, 0] >= cfg.zone_x[0])
        & (pts[:, 0] <= cfg.zone_x[1])
        & (pts[:, 1] >= cfg.zone_y[0])
        & (pts[:, 1] <= cfg.zone_y[1])
        & (pts[:, 2] >= cfg.zone_z[0])
        & (pts[:, 2] <= cfg.zone_z[1])
    )
    return pts[m].astype(np.float32)


# --------------------------------------------------------------------------
# Centroids
# --------------------------------------------------------------------------


def extract_centroids(
    points: np.ndarray, cell_m: float, min_points: int
) -> list[np.ndarray]:
    """Cluster zone points into person centroids on the floor plane.

    Connected components (4-neighbour) over a coarse (x, y) grid. This is a
    1-2 person vestibule, not a crowd; two people brushing shoulders can
    merge into one cluster and that failure is measured, not hidden.
    """
    if points.shape[0] == 0:
        return []
    ij = np.floor(points[:, :2] / cell_m).astype(np.int64)
    keys, inverse, counts = np.unique(
        ij, axis=0, return_inverse=True, return_counts=True
    )
    occupied = {tuple(k): i for i, k in enumerate(keys)}
    parent = list(range(len(keys)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (ci, cj), idx in occupied.items():
        for ni, nj in ((ci + 1, cj), (ci, cj + 1)):
            j = occupied.get((ni, nj))
            if j is not None:
                ra, rb = find(idx), find(j)
                if ra != rb:
                    parent[rb] = ra
    roots = np.array([find(i) for i in range(len(keys))])
    centroids = []
    for root in np.unique(roots):
        member_cells = np.nonzero(roots == root)[0]
        mask = np.isin(inverse, member_cells)
        if int(mask.sum()) >= min_points:
            centroids.append(points[mask].mean(axis=0))
    return centroids


# --------------------------------------------------------------------------
# Tracking and crossing count
# --------------------------------------------------------------------------


@dataclass
class CrossingEvent:
    t_ns: int
    kind: str  # "board" | "alight"
    track_id: int
    x_from: float
    x_to: float


@dataclass
class _Track:
    tid: int
    pos: np.ndarray
    t_ns: int
    side: int  # +1 door side, -1 cabin side, 0 inside the hysteresis band


class NearestNeighbourTracker:
    """Greedy NN association with a distance gate. Enough for 1-2 people.

    A crossing fires only when a track moves from one side of the hysteresis
    band to the other: door side -> cabin side = board, reverse = alight.
    Tracks born inside the band stay side-0 until they clear it, and a
    side-0 excursion that returns to its original side counts nothing.
    """

    def __init__(self, cfg: ApcConfig):
        self.cfg = cfg
        self.tracks: list[_Track] = []
        self._next_id = 0

    def _side(self, x: float) -> int:
        if x > self.cfg.x_line + self.cfg.hysteresis_m:
            return 1
        if x < self.cfg.x_line - self.cfg.hysteresis_m:
            return -1
        return 0

    def step(self, t_ns: int, centroids: list[np.ndarray]) -> list[CrossingEvent]:
        gap_ns = int(self.cfg.max_track_gap_s * NS)
        self.tracks = [tr for tr in self.tracks if t_ns - tr.t_ns <= gap_ns]
        events: list[CrossingEvent] = []
        unmatched = list(range(len(centroids)))
        # greedy: repeatedly take the globally closest (track, centroid) pair
        pairs = []
        for ti, tr in enumerate(self.tracks):
            for ci in unmatched:
                d = float(np.linalg.norm(tr.pos[:2] - centroids[ci][:2]))
                if d <= self.cfg.max_assoc_m:
                    pairs.append((d, ti, ci))
        pairs.sort()
        used_t: set[int] = set()
        used_c: set[int] = set()
        for d, ti, ci in pairs:
            if ti in used_t or ci in used_c:
                continue
            used_t.add(ti)
            used_c.add(ci)
            tr = self.tracks[ti]
            new_side = self._side(float(centroids[ci][0]))
            if new_side != 0 and tr.side != 0 and new_side != tr.side:
                events.append(
                    CrossingEvent(
                        t_ns,
                        "board" if new_side == -1 else "alight",
                        tr.tid,
                        float(tr.pos[0]),
                        float(centroids[ci][0]),
                    )
                )
            if new_side != 0:
                tr.side = new_side
            tr.pos = centroids[ci]
            tr.t_ns = t_ns
        for ci in range(len(centroids)):
            if ci not in used_c:
                x = float(centroids[ci][0])
                self.tracks.append(
                    _Track(self._next_id, centroids[ci], t_ns, self._side(x))
                )
                self._next_id += 1
        return events


def run_chunk(
    chunk: list[tuple[int, np.ndarray]],
    cfg: ApcConfig,
    cell_m: float,
    min_points: int,
) -> list[CrossingEvent]:
    """Track one contiguous chunk. The tracker is reset per chunk: chunks are
    separated by undownloaded gaps, and carrying a track across a gap would
    invent motion that was never observed."""
    tracker = NearestNeighbourTracker(cfg)
    events: list[CrossingEvent] = []
    for t_ns, pts in chunk:
        cents = extract_centroids(pts, cell_m, min_points)
        events.extend(tracker.step(t_ns, cents))
    return events


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def poisson_rate_ci(
    k: int, exposure: float, alpha: float = 0.05
) -> tuple[float, float]:
    """Garwood exact CI for a Poisson rate (events per unit exposure).

    chi-square quantiles via Wilson-Hilferty; adequate at these tiny counts
    and keeps the module scipy-free.
    """

    def chi2_ppf(p: float, df: float) -> float:
        if df <= 0:
            return 0.0
        # Wilson-Hilferty approximation
        z = _norm_ppf(p)
        return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3

    def _norm_ppf(p: float) -> float:
        # Acklam rational approximation
        a = [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
        b = [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
        c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
        d = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]
        plow = 0.02425
        if p < plow:
            q = math.sqrt(-2 * math.log(p))
            return (
                ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
            ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        if p > 1 - plow:
            q = math.sqrt(-2 * math.log(1 - p))
            return -(
                ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
            ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )

    if exposure <= 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else chi2_ppf(alpha / 2, 2 * k) / 2 / exposure
    hi = chi2_ppf(1 - alpha / 2, 2 * (k + 1)) / 2 / exposure
    return (lo, hi)


def match_events_to_episodes(
    events: list[CrossingEvent],
    episodes: list[Episode],
    pre_s: float = 6.0,
    post_s: float = 6.0,
) -> dict:
    """Assign detected crossings to ground-truth episodes.

    Window per episode: [start - pre_s, end + post_s]; from the bbox-level
    dry run, true crossings land within ~4 s of the episode, so 6 s gives
    margin without swallowing neighbours. Each event is assigned to at most
    one episode (nearest window centre). Detection ignores direction;
    direction accuracy is then scored among detected episodes, so a counter
    that sees every event but signs them wrong cannot hide.
    """
    pre, post = int(pre_s * NS), int(post_s * NS)
    assignment: dict[int, list[CrossingEvent]] = {i: [] for i in range(len(episodes))}
    unassigned: list[CrossingEvent] = []
    for ev in sorted(events, key=lambda e: e.t_ns):
        candidates = [
            (abs(ev.t_ns - (ep.start_ns + ep.end_ns) // 2), i)
            for i, ep in enumerate(episodes)
            if ep.start_ns - pre <= ev.t_ns <= ep.end_ns + post
        ]
        if candidates:
            candidates.sort()
            assignment[candidates[0][1]].append(ev)
        else:
            unassigned.append(ev)
    per_episode = []
    for i, ep in enumerate(episodes):
        evs = assignment[i]
        detected = bool(evs)
        first = evs[0] if evs else None
        per_episode.append(
            {
                "pid": ep.pid,
                "kind": ep.kind,
                "start_ns": ep.start_ns,
                "end_ns": ep.end_ns,
                "duration_s": round(ep.duration_s, 2),
                "detected": detected,
                "n_events_in_window": len(evs),
                "first_event_kind": first.kind if first else None,
                "first_event_t_ns": first.t_ns if first else None,
                "direction_correct": (first.kind == ep.kind) if first else None,
                "net_signed": sum(1 if e.kind == "board" else -1 for e in evs),
            }
        )
    return {"per_episode": per_episode, "unassigned_events": unassigned}


def self_test() -> None:
    """Deterministic synthetic checks for the tracker/crossing logic.

    Same convention as the CORN math verification (log 2026-08-09): no test
    framework, a self-contained function that raises on failure. Run with
    ``python -c "import sys; sys.path.insert(0,'src'); from sanas.door_apc
    import self_test; self_test()"`` from the repo root.
    """
    cfg = ApcConfig()

    def run_scripted(xs: list[float]) -> list[CrossingEvent]:
        tracker = NearestNeighbourTracker(cfg)
        events: list[CrossingEvent] = []
        for i, x in enumerate(xs):
            events.extend(tracker.step(int(i * 1.3e8), [np.array([x, 0.5, 1.0])]))
        return events

    # 1. clean inward pass door side -> cabin side: exactly one board
    evs = run_scripted(list(np.linspace(1.0, -0.5, 12)))
    assert [e.kind for e in evs] == ["board"], f"expected one board, got {evs}"

    # 2. clean outward pass cabin side -> door side: exactly one alight
    evs = run_scripted(list(np.linspace(-0.5, 1.0, 12)))
    assert [e.kind for e in evs] == ["alight"], f"expected one alight, got {evs}"

    # 3. hover across the hysteresis boundary without clearing the band on
    #    the far side: no event, however long the chatter
    hover = [0.5, 0.35, 0.22, 0.18, 0.30, 0.12, 0.28, 0.06, 0.33, 0.5]
    evs = run_scripted(hover)
    assert evs == [], f"hysteresis hover must not count, got {evs}"

    # 4. track born INSIDE the band (side 0) then leaving to either side:
    #    no event either way - an unknown starting side cannot cross
    evs = run_scripted([0.20, 0.18, 0.4, 0.8])
    assert evs == [], f"side-0 birth exiting door side must not count, got {evs}"
    evs = run_scripted([0.20, 0.18, -0.1, -0.5])
    assert evs == [], f"side-0 birth exiting cabin side must not count, got {evs}"

    # 5. full round trip in and back out: one board then one alight
    xs = list(np.linspace(1.0, -0.5, 10)) + list(np.linspace(-0.5, 1.0, 10))
    evs = run_scripted(xs)
    assert [e.kind for e in evs] == ["board", "alight"], f"round trip: {evs}"

    print("door_apc self_test: all 6 scripted tracker checks passed")


def truth_cabin_series(
    bbox_ts: np.ndarray, centers: np.ndarray, frame_of_box: np.ndarray, x_line: float
) -> np.ndarray:
    """Ground-truth 'people on the cabin side of the line' per bbox frame.

    Uses the same plane as the counter so the two series measure the same
    quantity: a person mid-doorway counts as outside until they cross.
    """
    inside = centers[:, 0] < x_line
    counts = np.zeros(len(bbox_ts), dtype=np.int32)
    np.add.at(counts, frame_of_box[inside], 1)
    return counts
