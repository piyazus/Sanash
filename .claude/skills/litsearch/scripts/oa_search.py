"""Reproducible OpenAlex search for the Sanas literature review.

Runs one or more search strings against the OpenAlex works API, records the
exact query, filters, timestamp and result count, and writes one JSON record
per work. No LLM call, no relevance judgment: this is the deterministic
retrieval half of a systematic search.

Every run appends a manifest line so the search is reproducible and reportable
(charter section 9 requires recorded strings, databases, dates and counts).

Usage:
    python oa_search.py --query "real-time crowding information bus" \
        --out data/litsearch/run1.jsonl --max 200
    python oa_search.py --queries-file blocks.txt --out run2.jsonl \
        --from-year 2010 --max 300

blocks.txt: one search string per line, blank lines and # comments ignored.
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API = "https://api.openalex.org/works"
UA = "sanas-litsearch/0.1 (mailto:tleukindias@gmail.com)"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def abstract_from_index(inv):
    if not inv:
        return None
    positions = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def flatten(work):
    inv = work.get("abstract_inverted_index")
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    return {
        "openalex_id": work.get("id"),
        "doi": work.get("doi"),
        "title": work.get("title"),
        "year": work.get("publication_year"),
        "type": work.get("type"),
        "venue": source.get("display_name"),
        "authors": [
            a.get("author", {}).get("display_name")
            for a in (work.get("authorships") or [])[:12]
        ],
        "cited_by_count": work.get("cited_by_count"),
        "referenced_works_count": len(work.get("referenced_works") or []),
        "open_access_url": (work.get("best_oa_location") or {}).get("pdf_url"),
        "abstract": abstract_from_index(inv),
    }


def search(query, max_results, from_year, to_year, work_type, sleep):
    got, cursor, total = [], "*", None
    while len(got) < max_results:
        params = {
            "search": query,
            "per-page": str(min(200, max_results - len(got))),
            "cursor": cursor,
        }
        filters = []
        if from_year:
            filters.append("from_publication_date:%d-01-01" % from_year)
        if to_year:
            filters.append("to_publication_date:%d-12-31" % to_year)
        if work_type:
            filters.append("type:%s" % work_type)
        if filters:
            params["filter"] = ",".join(filters)
        url = API + "?" + urllib.parse.urlencode(params)
        data = fetch(url)
        if total is None:
            total = data["meta"]["count"]
        results = data.get("results") or []
        if not results:
            break
        got.extend(results)
        cursor = data["meta"].get("next_cursor")
        if not cursor:
            break
        time.sleep(sleep)
    return got, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", action="append", default=[])
    ap.add_argument("--queries-file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--from-year", type=int)
    ap.add_argument("--to-year", type=int)
    ap.add_argument("--type", dest="work_type", help="e.g. article, review")
    ap.add_argument("--sleep", type=float, default=0.34)
    args = ap.parse_args()

    queries = list(args.query)
    if args.queries_file:
        for line in Path(args.queries_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                queries.append(line)
    if not queries:
        ap.error("no queries given")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path.with_suffix(".manifest.jsonl")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    seen = set()
    written = 0
    with (
        out_path.open("a", encoding="utf-8") as out,
        manifest_path.open("a", encoding="utf-8") as man,
    ):
        for q in queries:
            works, total = search(
                q, args.max, args.from_year, args.to_year, args.work_type, args.sleep
            )
            man.write(
                json.dumps(
                    {
                        "database": "OpenAlex",
                        "query": q,
                        "filters": {
                            "from_year": args.from_year,
                            "to_year": args.to_year,
                            "type": args.work_type,
                        },
                        "total_hits": total,
                        "retrieved": len(works),
                        "run_at_utc": stamp,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            for w in works:
                rec = flatten(w)
                key = rec["doi"] or rec["openalex_id"]
                if key in seen:
                    continue
                seen.add(key)
                rec["found_by_query"] = q
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            print("query: %s | hits %s | retrieved %d" % (q, total, len(works)))

    print("wrote %d unique records to %s" % (written, out_path))
    print("manifest: %s" % manifest_path)


if __name__ == "__main__":
    main()
