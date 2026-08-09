"""Turn the dataset's annotations into per-frame occupant counts.

Two different counts are produced, because they are not the same quantity and
only one of them is observable from a single camera:

  count_view   persons detected in THIS camera view, from
               segmentations/<camera>/<camera_ts>.json. Exact timestamp match.
               This is what a single-view model can actually be asked to predict.
  count_cabin  total occupants in the cabin, from bboxes_3d/<lidar_ts>.json.
               The eventual product target, but not recoverable from one view
               when people are occluded or out of frame.

Verified on the archive index (2026-08-09):
  - front_left_color has 12,890 images and 12,890 segmentation JSONs: every
    frame has a label, so there is no missing-file ambiguity.
  - 2,130 of those JSONs reference no mask PNG, i.e. genuine zero-person frames.
  - bboxes_3d timestamps and camera timestamps have ZERO overlap (separate
    lidar and camera clocks), so count_cabin requires nearest-timestamp
    matching within a tolerance.

The segmentation entries carry a "score" field and the upstream toolkit has a
scripts/autolabeling/ directory: these are model-generated pseudo-labels, not
human ground truth. Treat metrics accordingly.
"""

from __future__ import annotations

import bisect
import json
import os

NS_PER_MS = 1_000_000


def counts_from_segmentation(
    payload: bytes | str, score_threshold: float = 0.0
) -> tuple[int, float | None]:
    """Return (person_count, min_score) for one segmentation JSON."""
    entries = json.loads(payload)
    if not isinstance(entries, list):
        raise ValueError("segmentation JSON is not a list")
    kept = [e for e in entries if float(e.get("score", 1.0)) >= score_threshold]
    min_score = min((float(e.get("score", 1.0)) for e in kept), default=None)
    return len(kept), min_score


def counts_from_bboxes(payload: bytes | str, label: str = "human") -> int:
    """Return the number of oriented 3D boxes with the given label."""
    entries = json.loads(payload)
    if not isinstance(entries, list):
        raise ValueError("bboxes_3d JSON is not a list")
    return sum(1 for e in entries if e.get("bbox_label") == label)


def _ts_from_filename(path: str) -> int | None:
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = stem.split("_")[0]
    return int(stem) if stem.isdigit() and len(stem) == 19 else None


def load_view_counts(
    subset_dir: str, camera: str, score_threshold: float = 0.0
) -> dict[int, tuple[int, float | None]]:
    """Map camera timestamp -> (count_view, min_score) for one camera."""
    seg_dir = os.path.join(subset_dir, "segmentations", camera)
    out: dict[int, tuple[int, float | None]] = {}
    if not os.path.isdir(seg_dir):
        return out
    for name in os.listdir(seg_dir):
        if not name.endswith(".json"):
            continue
        ts = _ts_from_filename(name)
        if ts is None:
            continue
        with open(os.path.join(seg_dir, name), "rb") as fh:
            out[ts] = counts_from_segmentation(fh.read(), score_threshold)
    return out


def load_cabin_counts(subset_dir: str) -> dict[int, int]:
    """Map lidar timestamp -> total occupants from bboxes_3d."""
    box_dir = os.path.join(subset_dir, "bboxes_3d")
    out: dict[int, int] = {}
    if not os.path.isdir(box_dir):
        return out
    for name in os.listdir(box_dir):
        if not name.endswith(".json"):
            continue
        ts = _ts_from_filename(name)
        if ts is None:
            continue
        with open(os.path.join(box_dir, name), "rb") as fh:
            out[ts] = counts_from_bboxes(fh.read())
    return out


def match_nearest(ts: int, sorted_ts: list[int], tolerance_ns: int) -> int | None:
    """Nearest timestamp in sorted_ts within tolerance, else None."""
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


def assign_sequences(sorted_ts: list[int], gap_ns: int = 5 * 10**9) -> dict[int, int]:
    """Split a timestamp list into sub-sequences at recording gaps.

    Frames are ~0.067 s apart, so a random train/val split leaks near-duplicate
    neighbours across the boundary. The recording has 10 gaps > 5 s in
    front_left_color, giving 11 sub-sequences to split on instead.
    """
    seq: dict[int, int] = {}
    current = 0
    for i, ts in enumerate(sorted_ts):
        if i and ts - sorted_ts[i - 1] > gap_ns:
            current += 1
        seq[ts] = current
    return seq


def build_label_table(
    subset_dir: str,
    camera: str,
    score_threshold: float = 0.0,
    cabin_tolerance_ns: int = 100 * NS_PER_MS,
    sequence_gap_ns: int = 5 * 10**9,
) -> list[dict]:
    """Join extracted images with their counts into one row per frame."""
    img_dir = os.path.join(subset_dir, "cameras", "color", camera, "images")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"no extracted images at {img_dir}")

    images: dict[int, str] = {}
    for name in sorted(os.listdir(img_dir)):
        ts = _ts_from_filename(name)
        if ts is not None:
            images[ts] = f"cameras/color/{camera}/images/{name}"

    view = load_view_counts(subset_dir, camera, score_threshold)
    cabin = load_cabin_counts(subset_dir)
    cabin_ts = sorted(cabin)

    ordered = sorted(images)
    seq = assign_sequences(ordered, sequence_gap_ns)

    rows: list[dict] = []
    for ts in ordered:
        v = view.get(ts)
        c_ts = match_nearest(ts, cabin_ts, cabin_tolerance_ns)
        rows.append(
            {
                "timestamp": ts,
                "camera": camera,
                "image": images[ts],
                "sequence": seq[ts],
                "count_view": None if v is None else v[0],
                "min_score": None if v is None else v[1],
                "count_cabin": None if c_ts is None else cabin[c_ts],
                "cabin_dt_ns": None if c_ts is None else c_ts - ts,
            }
        )
    return rows


def count_distribution(rows: list[dict], column: str) -> dict[str, int]:
    dist: dict[str, int] = {}
    for r in rows:
        key = "missing" if r.get(column) is None else str(r[column])
        dist[key] = dist.get(key, 0) + 1
    return dict(sorted(dist.items(), key=lambda kv: (kv[0] == "missing", kv[0])))


def write_labels(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def read_labels(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
