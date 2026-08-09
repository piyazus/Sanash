#!/usr/bin/env python
"""Sanas: extract a smoke-test subset of the beintelli_v1 in-cabin dataset.

CPU only. Needs internet. Does NOT download the 73.5 GB archive: it reads the
ZIP central directory over HTTP range requests, selects the members it wants,
and pulls only those byte ranges from the HuggingFace mirror.

Default selection (~0.5 GB):
  - every annotation needed for counting: segmentations/<camera>/*.json,
    bboxes_3d/, states/, the root metadata JSONs, camera intrinsics
  - front_left_color images at stride 10 (~1,290 frames spanning the whole
    32-minute recording and all 11 sub-sequences)

Outputs, under /kaggle/working/subset (or ./data/subset locally):
  labels.jsonl   one row per frame: timestamp, image path, sequence id,
                 count_view, count_cabin
  manifest.json  what was selected, what was transferred, count distributions

Runs anywhere (Kaggle, Colab, rented box, laptop) with the stdlib only.

NOTE ON DUPLICATION: the ZIP range reader and the label logic below are a
vendored copy of src/sanas/ziprange.py and src/sanas/labels.py. Kaggle pushes
one file per kernel, so a self-contained script is the only option that needs
nothing uploaded alongside it. src/sanas/ is the canonical copy: change it
there first, then re-sync the marked blocks here.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import struct
import sys
import time
import urllib.request
import zlib
from dataclasses import dataclass

# ==========================================================================
# BEGIN VENDORED BLOCK - canonical: src/sanas/ziprange.py
# ==========================================================================

MIRRORS = {
    "hf": (
        "https://huggingface.co/datasets/evgenygorelik96/"
        "multiview_incabin_dataset/resolve/main/beintelli_v1.zip"
    ),
    "zenodo": "https://zenodo.org/api/records/20559664/files/beintelli_v1.zip/content",
}
ARCHIVE_SIZE = 73_515_214_805
ARCHIVE_MD5 = "74924b5be59e706a5a89affba48f6b87"
ROOT_PREFIX = "beintelli_v1/"
# Pad small: the central directory does not record the *local* extra length,
# so a run's end is an estimate. Padding by the legal max (65535) would waste
# ~25% of a stride-10 transfer; instead pad small and re-fetch exactly when a
# member overruns its run. Observed local extra length here is 32 bytes.
_LOCAL_EXTRA_PAD = 1024
_USER_AGENT = "sanas-occupancy/0.1 (+research)"


@dataclass(frozen=True)
class Member:
    name: str
    method: int
    size: int
    csize: int
    offset: int
    crc: int

    @property
    def is_dir(self) -> bool:
        return self.name.endswith("/")


def http_get(url, start=None, end=None, retries=5, timeout=120, backoff=2.0) -> bytes:
    headers = {"User-Agent": _USER_AGENT}
    if start is not None:
        headers["Range"] = f"bytes={start}-{'' if end is None else end}"
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if start is not None and resp.status != 206:
                    raise OSError(f"server ignored Range (status {resp.status})")
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt == retries - 1:
                break
            time.sleep(backoff**attempt)
    raise OSError(f"GET failed after {retries} attempts: {url} ({last})")


def remote_size(url, timeout=60) -> int:
    req = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": _USER_AGENT}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(resp.headers["Content-Length"])


def _find_eocd(tail: bytes):
    i = tail.rfind(b"PK\x05\x06")
    if i == -1:
        raise ValueError("EOCD not found")
    _, _, _, _, n, cd_size, cd_off, _ = struct.unpack("<IHHHHIIH", tail[i : i + 22])
    j = tail.rfind(b"PK\x06\x06")
    if j != -1:
        z = struct.unpack("<IQHHIIQQQQ", tail[j : j + 56])
        n, cd_size, cd_off = z[7], z[8], z[9]
    return n, cd_size, cd_off


def parse_central_directory(blob: bytes):
    members = []
    off, end = 0, len(blob)
    while off + 46 <= end:
        if blob[off : off + 4] != b"PK\x01\x02":
            break
        (
            _s,
            _a,
            _b,
            _f,
            method,
            _t,
            _d,
            crc,
            csize,
            usize,
            nlen,
            elen,
            clen,
            _ds,
            _ia,
            _ea,
            lhoff,
        ) = struct.unpack("<IHHHHHHIIIHHHHHII", blob[off : off + 46])
        name = blob[off + 46 : off + 46 + nlen].decode("utf-8", "replace")
        extra = blob[off + 46 + nlen : off + 46 + nlen + elen]
        eo = 0
        while eo + 4 <= len(extra):
            hid, hsz = struct.unpack("<HH", extra[eo : eo + 4])
            body = extra[eo + 4 : eo + 4 + hsz]
            if hid == 0x0001:
                bo = 0
                if usize == 0xFFFFFFFF and bo + 8 <= len(body):
                    usize = struct.unpack("<Q", body[bo : bo + 8])[0]
                    bo += 8
                if csize == 0xFFFFFFFF and bo + 8 <= len(body):
                    csize = struct.unpack("<Q", body[bo : bo + 8])[0]
                    bo += 8
                if lhoff == 0xFFFFFFFF and bo + 8 <= len(body):
                    lhoff = struct.unpack("<Q", body[bo : bo + 8])[0]
                    bo += 8
            eo += 4 + hsz
        members.append(Member(name, method, usize, csize, lhoff, crc))
        off += 46 + nlen + elen + clen
    return members


def build_index(url, verbose=True):
    total = remote_size(url)
    if verbose:
        print(f"archive size: {total:,} bytes", flush=True)
    tail = http_get(url, total - 65_536, total - 1)
    n_entries, cd_size, cd_off = _find_eocd(tail)
    if verbose:
        print(
            f"central directory: {n_entries:,} entries, {cd_size:,} bytes "
            f"at offset {cd_off:,}",
            flush=True,
        )
    blob = http_get(url, cd_off, cd_off + cd_size - 1)
    members = parse_central_directory(blob)
    if verbose:
        print(f"parsed {len(members):,} members", flush=True)
    return total, members


def sniff_image_ext(data: bytes):
    """Real extension for an image payload. Depth frames are named .jpg but
    are actually PNG (magic 89 50 4E 47) - this is what catches that."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:2] == b"\xff\xd8":
        return ".jpg"
    return None


def _run_end(m: Member, archive_size: int) -> int:
    return min(
        archive_size,
        m.offset + 30 + len(m.name.encode()) + _LOCAL_EXTRA_PAD + m.csize,
    )


def plan_runs(members, archive_size, gap_merge=1 << 20, max_run_bytes=256 << 20):
    ordered = sorted((m for m in members if not m.is_dir), key=lambda m: m.offset)
    runs = []
    for m in ordered:
        m_end = _run_end(m, archive_size)
        if runs:
            start, end, group = runs[-1]
            if m.offset - end <= gap_merge and (m_end - start) <= max_run_bytes:
                group.append(m)
                runs[-1] = (start, max(end, m_end), group)
                continue
        runs.append((m.offset, m_end, [m]))
    return runs


def _member_data(buf: bytes, m: Member, base: int, url: str):
    """Slice a member out of a run buffer, re-fetching exactly what the run
    estimate missed. Returns (data, extra_requests)."""
    extra = 0
    pos = m.offset - base
    if pos < 0 or pos + 30 > len(buf):
        header = http_get(url, m.offset, m.offset + 29)
        extra += 1
    else:
        header = buf[pos : pos + 30]
    sig, _v, _f, _me, _t, _d, _c, _cs, _us, nlen, elen = struct.unpack(
        "<IHHHHHIIIHH", header
    )
    if sig != 0x04034B50:
        raise ValueError(f"bad local header for {m.name}")
    data_start = m.offset + 30 + nlen + elen
    lo = data_start - base
    if lo >= 0 and lo + m.csize <= len(buf):
        raw = buf[lo : lo + m.csize]
    else:
        raw = http_get(url, data_start, data_start + m.csize - 1)
        extra += 1
    if len(raw) != m.csize:
        raise ValueError(f"short read for {m.name}")
    data = zlib.decompress(raw, -15) if m.method == 8 else raw
    if m.crc and zlib.crc32(data) & 0xFFFFFFFF != m.crc:
        raise ValueError(f"CRC mismatch for {m.name}")
    return data, extra


def extract_members(
    url,
    members,
    out_dir,
    archive_size=ARCHIVE_SIZE,
    strip_prefix=ROOT_PREFIX,
    gap_merge=1 << 20,
    max_run_bytes=256 << 20,
    fix_image_extensions=True,
    verbose=True,
):
    runs = plan_runs(members, archive_size, gap_merge, max_run_bytes)
    planned = sum(end - start for start, end, _ in runs)
    if verbose:
        print(
            f"extracting {len(members):,} members in {len(runs):,} range request(s); "
            f"~{planned / 1e9:.3f} GB to transfer",
            flush=True,
        )
    written = bytes_out = extra_requests = 0
    renamed = []
    t0 = time.time()
    for idx, (start, end, group) in enumerate(runs, 1):
        buf = http_get(url, start, end - 1)
        for m in group:
            data, extra = _member_data(buf, m, start, url)
            extra_requests += extra
            rel = (
                m.name[len(strip_prefix) :]
                if m.name.startswith(strip_prefix)
                else m.name
            )
            if fix_image_extensions:
                stem, ext = os.path.splitext(rel)
                real = sniff_image_ext(data[:8])
                if (
                    real
                    and ext.lower() in {".jpg", ".jpeg", ".png"}
                    and real != ext.lower()
                ):
                    renamed.append((rel, stem + real))
                    rel = stem + real
            dst = os.path.join(out_dir, rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as fh:
                fh.write(data)
            written += 1
            bytes_out += len(data)
        if verbose and (idx % 25 == 0 or idx == len(runs)):
            rate = planned / max(time.time() - t0, 1e-6) / 1e6
            print(
                f"  run {idx}/{len(runs)}  files={written:,}  "
                f"out={bytes_out / 1e9:.3f} GB  ~{rate:.1f} MB/s",
                flush=True,
            )
    if renamed and verbose:
        print(
            f"  corrected {len(renamed)} extension(s) after magic-byte sniff "
            f"(e.g. {renamed[0][0]} -> {renamed[0][1]})",
            flush=True,
        )
    return {
        "members": written,
        "bytes_written": bytes_out,
        "bytes_transferred_planned": planned,
        "runs": len(runs),
        "extra_requests": extra_requests,
        "renamed_total": len(renamed),
        "renamed_examples": renamed[:20],
        "seconds": round(time.time() - t0, 1),
    }


# ==========================================================================
# END VENDORED BLOCK (ziprange)
# BEGIN VENDORED BLOCK - canonical: src/sanas/labels.py
# ==========================================================================

NS_PER_MS = 1_000_000


def counts_from_segmentation(payload, score_threshold=0.0):
    entries = json.loads(payload)
    kept = [e for e in entries if float(e.get("score", 1.0)) >= score_threshold]
    min_score = min((float(e.get("score", 1.0)) for e in kept), default=None)
    return len(kept), min_score


def counts_from_bboxes(payload, label="human"):
    entries = json.loads(payload)
    return sum(1 for e in entries if e.get("bbox_label") == label)


def _ts_from_filename(path):
    stem = os.path.splitext(os.path.basename(path))[0].split("_")[0]
    return int(stem) if stem.isdigit() and len(stem) == 19 else None


def match_nearest(ts, sorted_ts, tolerance_ns):
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


def assign_sequences(sorted_ts, gap_ns=5 * 10**9):
    seq, current = {}, 0
    for i, ts in enumerate(sorted_ts):
        if i and ts - sorted_ts[i - 1] > gap_ns:
            current += 1
        seq[ts] = current
    return seq


def count_distribution(rows, column):
    dist = {}
    for r in rows:
        key = "missing" if r.get(column) is None else str(r[column])
        dist[key] = dist.get(key, 0) + 1
    return dict(sorted(dist.items(), key=lambda kv: (kv[0] == "missing", kv[0])))


# ==========================================================================
# END VENDORED BLOCK (labels)
# ==========================================================================

ROOT_METADATA = {
    ROOT_PREFIX + "person_states.json",
    ROOT_PREFIX + "modalities.json",
    ROOT_PREFIX + "frame_ids.json",
    ROOT_PREFIX + "target_transforms.json",
}


def select_members(members, args):
    """Choose the archive members to pull. Returns (selected, breakdown)."""
    cam = args.camera
    depth_cam = cam.replace("_color", "_depth")

    images = sorted(
        (
            m
            for m in members
            if m.name.startswith(f"{ROOT_PREFIX}cameras/color/{cam}/images/")
            and m.name.endswith(".jpg")
        ),
        key=lambda m: m.name,
    )
    picked = images[:: args.stride]
    if args.max_images:
        picked = picked[: args.max_images]
    picked_ts = [_ts_from_filename(m.name) for m in picked]

    labels = []
    for m in members:
        n = m.name
        if m.is_dir:
            continue
        if n.startswith(f"{ROOT_PREFIX}segmentations/{cam}/"):
            if n.endswith(".json") or (args.include_masks and n.endswith(".png")):
                labels.append(m)
        elif n.startswith(f"{ROOT_PREFIX}bboxes_3d/") or n.startswith(
            f"{ROOT_PREFIX}states/"
        ):
            labels.append(m)
        elif args.include_poses and n.startswith(f"{ROOT_PREFIX}poses/"):
            labels.append(m)
        elif n in ROOT_METADATA:
            labels.append(m)
        elif n.startswith(f"{ROOT_PREFIX}cameras/") and n.endswith("camera_info.yaml"):
            labels.append(m)

    depth = []
    if args.include_depth:
        # Depth frames use their own clock, so pick the nearest depth frame to
        # each selected colour frame. Their payloads are PNG despite the .jpg
        # name; extract_members() sniffs and rewrites the extension.
        cand = {
            _ts_from_filename(m.name): m
            for m in members
            if m.name.startswith(f"{ROOT_PREFIX}cameras/depth/{depth_cam}/images/")
            and not m.is_dir
        }
        cand.pop(None, None)
        cand_ts = sorted(cand)
        seen = set()
        for ts in picked_ts:
            hit = match_nearest(ts, cand_ts, args.depth_tolerance_ms * NS_PER_MS)
            if hit is not None and hit not in seen:
                seen.add(hit)
                depth.append(cand[hit])

    selected = picked + labels + depth
    breakdown = {
        "images": len(picked),
        "image_bytes": sum(m.csize for m in picked),
        "labels": len(labels),
        "label_bytes": sum(m.csize for m in labels),
        "depth": len(depth),
        "depth_bytes": sum(m.csize for m in depth),
        "total_members": len(selected),
        "total_compressed_bytes": sum(m.csize for m in selected),
        "available_images_for_camera": len(images),
    }
    return selected, breakdown


def build_label_table(out_dir, camera, args):
    """Join extracted frames with their counts. One row per extracted image."""
    img_dir = os.path.join(out_dir, "cameras", "color", camera, "images")
    images = {}
    for name in sorted(os.listdir(img_dir)):
        ts = _ts_from_filename(name)
        if ts is not None:
            images[ts] = f"cameras/color/{camera}/images/{name}"

    seg_dir = os.path.join(out_dir, "segmentations", camera)
    view = {}
    if os.path.isdir(seg_dir):
        for name in os.listdir(seg_dir):
            if not name.endswith(".json"):
                continue
            ts = _ts_from_filename(name)
            if ts is None:
                continue
            with open(os.path.join(seg_dir, name), "rb") as fh:
                view[ts] = counts_from_segmentation(fh.read(), args.score_threshold)

    box_dir = os.path.join(out_dir, "bboxes_3d")
    cabin = {}
    if os.path.isdir(box_dir):
        for name in os.listdir(box_dir):
            if not name.endswith(".json"):
                continue
            ts = _ts_from_filename(name)
            if ts is None:
                continue
            with open(os.path.join(box_dir, name), "rb") as fh:
                cabin[ts] = counts_from_bboxes(fh.read())
    cabin_ts = sorted(cabin)

    ordered = sorted(images)
    seq = assign_sequences(ordered, args.sequence_gap_s * 10**9)

    rows = []
    for ts in ordered:
        v = view.get(ts)
        c_ts = match_nearest(ts, cabin_ts, args.cabin_tolerance_ms * NS_PER_MS)
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


def default_out_dir() -> str:
    return (
        "/kaggle/working/subset"
        if os.path.isdir("/kaggle/working")
        else os.path.join("data", "subset")
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract a smoke-test subset (range requests only)"
    )
    ap.add_argument("--mirror", choices=sorted(MIRRORS), default="hf")
    ap.add_argument("--camera", default="front_left_color")
    ap.add_argument(
        "--stride", type=int, default=10, help="take every Nth colour frame"
    )
    ap.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="0 = no cap; use a small value to test",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="drop pseudo-label detections below this score",
    )
    ap.add_argument(
        "--cabin-tolerance-ms",
        type=int,
        default=100,
        help="max camera<->lidar clock offset when matching count_cabin",
    )
    ap.add_argument(
        "--sequence-gap-s",
        type=int,
        default=5,
        help="recording gap that starts a new sub-sequence",
    )
    ap.add_argument(
        "--include-masks", action="store_true", help="also pull segmentation mask PNGs"
    )
    ap.add_argument(
        "--include-poses", action="store_true", help="also pull 3D pose JSONs"
    )
    ap.add_argument(
        "--include-depth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pull the depth frame matching each colour frame (branch 2 needs it)",
    )
    ap.add_argument("--depth-tolerance-ms", type=int, default=100)
    ap.add_argument(
        "--dry-run", action="store_true", help="report the selection, download nothing"
    )
    args = ap.parse_args()

    out_dir = args.out or default_out_dir()
    url = MIRRORS[args.mirror]
    t_start = time.time()

    print("=" * 72)
    print("Sanas subset extraction")
    print("=" * 72)
    print(f"mirror   : {args.mirror} -> {url}")
    print(f"camera   : {args.camera}   stride: {args.stride}")
    print(f"out dir  : {out_dir}")
    print()

    archive_size, members = build_index(url)
    if archive_size != ARCHIVE_SIZE:
        print(
            f"WARNING: archive is {archive_size:,} bytes, expected {ARCHIVE_SIZE:,}. "
            "The mirror may have changed; re-verify before trusting results.",
            file=sys.stderr,
        )

    selected, breakdown = select_members(members, args)
    print("\nselection:")
    for k, v in breakdown.items():
        print(f"  {k:<32} {v:,}")
    print(
        f"  {'estimated transfer':<32} {breakdown['total_compressed_bytes'] / 1e9:.3f} GB"
    )

    if not selected:
        print("nothing selected - check --camera", file=sys.stderr)
        return 1

    if args.dry_run:
        runs = plan_runs(selected, archive_size)
        planned = sum(e - s for s, e, _ in runs)
        print(
            f"\n--dry-run: would issue {len(runs):,} range request(s), "
            f"~{planned / 1e9:.3f} GB. Nothing downloaded."
        )
        return 0

    os.makedirs(out_dir, exist_ok=True)
    stats = extract_members(url, selected, out_dir, archive_size=archive_size)
    print(
        f"\nextracted {stats['members']:,} files, "
        f"{stats['bytes_written'] / 1e9:.3f} GB on disk in {stats['seconds']}s"
    )

    rows = build_label_table(out_dir, args.camera, args)
    labels_path = os.path.join(out_dir, "labels.jsonl")
    with open(labels_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    dist_view = count_distribution(rows, "count_view")
    dist_cabin = count_distribution(rows, "count_cabin")
    seqs = sorted({r["sequence"] for r in rows})
    missing_view = sum(1 for r in rows if r["count_view"] is None)

    manifest = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mirror": args.mirror,
        "source_url": url,
        "archive_size": archive_size,
        "archive_md5_expected": ARCHIVE_MD5,
        "camera": args.camera,
        "stride": args.stride,
        "score_threshold": args.score_threshold,
        "selection": breakdown,
        "transfer": stats,
        "frames": len(rows),
        "sequences": len(seqs),
        "frames_missing_view_label": missing_view,
        "count_view_distribution": dist_view,
        "count_cabin_distribution": dist_cabin,
        "labels_file": "labels.jsonl",
        "note": (
            "count_view = persons in this camera view (segmentation pseudo-labels). "
            "count_cabin = total cabin occupants (3D boxes, nearest lidar timestamp). "
            "Pseudo-labels, not human ground truth. Max 4 occupants anywhere in this "
            "dataset, single daytime session."
        ),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print("\n" + "=" * 72)
    print(f"frames                : {len(rows):,}")
    print(f"sub-sequences         : {len(seqs)}  {seqs}")
    print(f"missing view label    : {missing_view}")
    print(f"count_view  dist      : {dist_view}")
    print(f"count_cabin dist      : {dist_cabin}")
    print(f"labels                : {labels_path}")
    print(f"manifest              : {os.path.join(out_dir, 'manifest.json')}")
    print(f"total wall clock      : {time.time() - t_start:.0f}s")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
