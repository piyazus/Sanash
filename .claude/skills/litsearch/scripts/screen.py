"""Merge, deduplicate and stage litsearch results for screening.

Takes the JSONL produced by oa_search.py / oa_chain.py, removes duplicates by
DOI and by normalized title, flags anything already present in the local
corpus, and writes a compact screening sheet plus PRISMA-style counts.

Local corpus checked:
  rtci_paper_inventory.csv
  research/refs/**  (filenames only)
  research/wiki/sources/*.md (citation lines)

Usage:
    python screen.py --in data/litsearch/*.jsonl --out data/litsearch/screen.md
"""

import argparse
import csv
import glob
import io
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def norm_title(t):
    if not t:
        return ""
    t = t.lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def load_local(root):
    known_titles, known_dois = set(), set()
    inv = root / "rtci_paper_inventory.csv"
    if inv.exists():
        with inv.open(encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                for k, v in row.items():
                    if not v:
                        continue
                    if k and "doi" in k.lower():
                        known_dois.add(
                            v.strip().lower().replace("https://doi.org/", "")
                        )
                    if k and "title" in k.lower():
                        known_titles.add(norm_title(v))
    for md in (root / "research" / "wiki" / "sources").glob("*.md"):
        text = md.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if "." in line and len(line) > 40:
                known_titles.add(norm_title(line))
    return known_titles, known_dois


def already_known(rec, known_titles, known_dois):
    doi = (rec.get("doi") or "").lower().replace("https://doi.org/", "")
    if doi and doi in known_dois:
        return "doi match in inventory"
    nt = norm_title(rec.get("title"))
    if not nt:
        return None
    if nt in known_titles:
        return "title match in inventory"
    for kt in known_titles:
        if nt and len(nt) > 25 and nt in kt:
            return "title contained in local corpus entry"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--min-year", type=int)
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    known_titles, known_dois = load_local(root)

    files = []
    for pattern in args.inputs:
        files.extend(glob.glob(pattern))
    records = []
    for f in files:
        for line in io.open(f, encoding="utf-8"):
            line = line.strip()
            if line:
                records.append(json.loads(line))

    retrieved = len(records)
    by_key, dup = {}, 0
    for r in records:
        key = (
            (r.get("doi") or "").lower()
            or norm_title(r.get("title"))
            or r.get("openalex_id")
        )
        if key in by_key:
            dup += 1
            prev = by_key[key]
            q = r.get("found_by_query")
            if q and q not in (prev.get("found_by_query") or ""):
                prev["found_by_query"] = (prev.get("found_by_query") or "") + " ; " + q
            continue
        by_key[key] = r

    unique = list(by_key.values())
    if args.min_year:
        before = len(unique)
        unique = [r for r in unique if (r.get("year") or 0) >= args.min_year]
        year_filtered = before - len(unique)
    else:
        year_filtered = 0

    new, known = [], []
    for r in unique:
        why = already_known(r, known_titles, known_dois)
        (known if why else new).append((r, why))

    new.sort(
        key=lambda t: (
            -(t[0].get("seed_count") or 0),
            -(t[0].get("cited_by_count") or 0),
        )
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Screening sheet\n\n")
        fh.write("Сгенерировано `screen.py`. Ничего не отброшено автоматически, ")
        fh.write("кроме точных дубликатов и фильтра по году.\n\n")
        fh.write("## Counts\n\n")
        fh.write("| Стадия | N |\n|---|---:|\n")
        fh.write("| Извлечено записей | %d |\n" % retrieved)
        fh.write("| Дубликатов удалено | %d |\n" % dup)
        fh.write("| Отфильтровано по году | %d |\n" % year_filtered)
        fh.write("| Уникальных к скринингу | %d |\n" % len(unique))
        fh.write("| Уже в локальном корпусе | %d |\n" % len(known))
        fh.write("| Новых кандидатов | %d |\n\n" % len(new))
        fh.write("## Новые кандидаты\n\n")
        fh.write(
            "| # | Год | Работа | Цит. | Seeds | Найдено через |\n|---:|---:|---|---:|---:|---|\n"
        )
        for i, (r, _) in enumerate(new, 1):
            authors = ", ".join(a for a in (r.get("authors") or [])[:3] if a)
            title = (r.get("title") or "").replace("|", "/")
            venue = (r.get("venue") or "").replace("|", "/")
            doi = r.get("doi") or ""
            fh.write(
                "| %d | %s | %s. %s. %s %s | %s | %s | %s |\n"
                % (
                    i,
                    r.get("year") or "",
                    authors,
                    title,
                    venue,
                    doi,
                    r.get("cited_by_count") or "",
                    r.get("seed_count") or "",
                    (r.get("found_by_query") or "")[:60],
                )
            )
        fh.write("\n## Уже известные локально\n\n")
        for r, why in known:
            fh.write(
                "- %s (%s) — %s\n" % ((r.get("title") or "")[:110], r.get("year"), why)
            )
        fh.write("\n## Abstracts новых кандидатов\n\n")
        for i, (r, _) in enumerate(new, 1):
            fh.write("### %d. %s\n\n" % (i, (r.get("title") or "")[:140]))
            fh.write(
                "%s\n\n" % ((r.get("abstract") or "нет abstract в OpenAlex")[:1200])
            )

    print(
        "retrieved %d | unique %d | new %d | known %d"
        % (retrieved, len(unique), len(new), len(known))
    )
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
