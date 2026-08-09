#!/usr/bin/env python
"""Regenerate the byte-range index for the beintelli_v1 ZIP.

Reads only the ZIP central directory over HTTP range requests (~45 MB), never
the 73.5 GB archive. The output is a generated artifact under data/ and is
gitignored: reproduce it by running this script, do not commit it.

Usage:
    python scripts/build_zip_index.py
    python scripts/build_zip_index.py --mirror zenodo --out data/index/idx.json
    python scripts/build_zip_index.py --summary-only     # no write, just report

The resulting index is what lets the extraction kernel pull an arbitrary
subset without downloading the archive.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from sanas.ziprange import MIRRORS, build_index, save_index  # noqa: E402

DEFAULT_OUT = os.path.join("data", "index", "beintelli_v1_zip_index.json")


def summarize(members: list[dict]) -> None:
    """Print the group breakdown so the index can be eyeballed after a rebuild."""
    files = [m for m in members if not m["name"].endswith("/")]
    total_u = sum(m["size"] for m in files)
    total_c = sum(m["csize"] for m in files)
    print(f"\nmembers: {len(members):,} ({len(files):,} files)")
    print(f"uncompressed: {total_u:,} bytes ({total_u / 1e9:.2f} GB)")
    print(f"compressed:   {total_c:,} bytes ({total_c / 1e9:.2f} GB)")

    groups: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    for m in files:
        # Group by the containing directory, collapsed to two levels below the
        # archive root, so flat dirs like bboxes_3d/ stay a single row.
        dirparts = m["name"].split("/")[1:-1]
        key = "/".join(dirparts[:2]) if dirparts else "<archive root>"
        g = groups[key]
        g[0] += 1
        g[1] += m["size"]
        g[2] += m["csize"]

    print(f"\n{'group':<45}{'files':>9}{'uncompressed':>16}{'compressed':>15}")
    for key, (n, u, c) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
        print(f"{key:<45}{n:>9,}{u:>16,}{c:>15,}")

    exts = collections.Counter(os.path.splitext(m["name"])[1].lower() for m in files)
    print("\nextensions:", dict(exts.most_common(10)))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--mirror",
        choices=sorted(MIRRORS),
        default="hf",
        help="hf is ~8.6x faster than zenodo from a residential line",
    )
    ap.add_argument("--url", default=None, help="override the mirror URL entirely")
    ap.add_argument(
        "--out", default=DEFAULT_OUT, help=f"output path (default {DEFAULT_OUT})"
    )
    ap.add_argument(
        "--summary-only", action="store_true", help="do not write the index"
    )
    ap.add_argument(
        "--from-file",
        default=None,
        help="re-summarize an existing index instead of refetching",
    )
    args = ap.parse_args()

    if args.from_file:
        import json

        with open(args.from_file, encoding="utf-8") as fh:
            index = json.load(fh)
        print(f"re-summarizing {args.from_file} (no network)")
        summarize(index["members"])
        return 0

    url = args.url or MIRRORS[args.mirror]
    print(f"mirror: {args.mirror} -> {url}")

    index = build_index(url)
    summarize(index["members"])

    if args.summary_only:
        print("\n--summary-only: nothing written")
        return 0

    save_index(args.out, index)
    size = os.path.getsize(args.out)
    print(f"\nwrote {args.out} ({size:,} bytes, {index['entry_count']:,} entries)")
    print("This file is gitignored on purpose. Rebuild it with this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
