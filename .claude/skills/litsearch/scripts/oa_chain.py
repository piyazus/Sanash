"""Backward and forward citation chaining over OpenAlex.

Given seed works (DOIs or OpenAlex ids), collects:
  backward - what the seeds cite (their reference lists),
  forward  - what cites the seeds,
and ranks candidates by how many distinct seeds point at them. A work that
five seeds all cite is a likely foundational paper the keyword search missed;
this is the part a plain database query cannot do.

Usage:
    python oa_chain.py --seeds seeds.txt --out data/litsearch/chain.jsonl \
        --direction both --min-seeds 2 --max-forward 200

seeds.txt: one DOI or OpenAlex id per line, # comments ignored.
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

API = "https://api.openalex.org/works"
UA = "sanas-litsearch/0.1 (mailto:tleukindias@gmail.com)"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def norm_seed(s):
    s = s.strip()
    if s.lower().startswith("10."):
        return "https://doi.org/" + s
    return s


def abstract_from_index(inv):
    if not inv:
        return None
    pos = []
    for word, idxs in inv.items():
        for i in idxs:
            pos.append((i, word))
    pos.sort()
    return " ".join(w for _, w in pos)


def flatten(work):
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    return {
        "openalex_id": work.get("id"),
        "doi": work.get("doi"),
        "title": work.get("title"),
        "year": work.get("publication_year"),
        "venue": source.get("display_name"),
        "authors": [
            a.get("author", {}).get("display_name")
            for a in (work.get("authorships") or [])[:12]
        ],
        "cited_by_count": work.get("cited_by_count"),
        "abstract": abstract_from_index(work.get("abstract_inverted_index")),
    }


def get_work(ident):
    return fetch(API + "/" + urllib.parse.quote(ident, safe=":/."))


def cited_by(ident, limit, sleep):
    out, cursor = [], "*"
    while len(out) < limit:
        params = {
            "filter": "cites:" + ident.rsplit("/", 1)[-1],
            "per-page": str(min(200, limit - len(out))),
            "cursor": cursor,
        }
        data = fetch(API + "?" + urllib.parse.urlencode(params))
        res = data.get("results") or []
        if not res:
            break
        out.extend(res)
        cursor = data["meta"].get("next_cursor")
        if not cursor:
            break
        time.sleep(sleep)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--direction", choices=["backward", "forward", "both"], default="both"
    )
    ap.add_argument(
        "--min-seeds",
        type=int,
        default=2,
        help="keep candidates reached from at least N seeds",
    )
    ap.add_argument(
        "--max-forward", type=int, default=200, help="per-seed cap on citing works"
    )
    ap.add_argument("--sleep", type=float, default=0.34)
    args = ap.parse_args()

    seeds = [
        norm_seed(x)
        for x in Path(args.seeds).read_text(encoding="utf-8").splitlines()
        if x.strip() and not x.strip().startswith("#")
    ]

    hits = defaultdict(lambda: {"backward": set(), "forward": set()})
    seed_titles = {}
    resolved = []

    for s in seeds:
        try:
            w = get_work(s)
        except Exception as exc:  # noqa: BLE001
            print("seed FAILED %s: %s" % (s, exc))
            continue
        wid = w["id"]
        resolved.append(wid)
        seed_titles[wid] = w.get("title")
        print("seed ok: %s | %s" % (wid, (w.get("title") or "")[:70]))
        if args.direction in ("backward", "both"):
            for ref in w.get("referenced_works") or []:
                hits[ref]["backward"].add(wid)
        if args.direction in ("forward", "both"):
            for c in cited_by(wid, args.max_forward, args.sleep):
                hits[c["id"]]["forward"].add(wid)
        time.sleep(args.sleep)

    candidates = {
        k: v
        for k, v in hits.items()
        if k not in resolved and len(v["backward"] | v["forward"]) >= args.min_seeds
    }
    print("candidates reached from >= %d seeds: %d" % (args.min_seeds, len(candidates)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ordered = sorted(
        candidates.items(),
        key=lambda kv: len(kv[1]["backward"] | kv[1]["forward"]),
        reverse=True,
    )
    with out_path.open("a", encoding="utf-8") as out:
        for wid, v in ordered:
            try:
                rec = flatten(get_work(wid))
            except Exception as exc:  # noqa: BLE001
                print("skip %s: %s" % (wid, exc))
                continue
            rec["reached_by_seeds"] = sorted(v["backward"] | v["forward"])
            rec["seed_count"] = len(v["backward"] | v["forward"])
            rec["direction"] = (
                "both"
                if v["backward"] and v["forward"]
                else ("backward" if v["backward"] else "forward")
            )
            rec["found_by_query"] = "citation-chaining"
            rec["run_at_utc"] = stamp
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            time.sleep(args.sleep)
    print("wrote %s" % out_path)


if __name__ == "__main__":
    main()
