"""Reconstruct the fielded instrument from the raw response export.

The `survey/instrument/survey_questions_{en,ru,kz}.md` files previously carried
on `origin/main` describe a questionnaire that was never administered: they list
a gender question, three crowding levels and a 15-minute wait level, none of
which appear in the export. The response file is the only authoritative record
of what respondents saw, so the instrument is generated from it.
"""

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "responses.csv"
OUT = ROOT / "instrument" / "fielded_instrument.md"

HEADER = """# Fielded instrument (reconstructed from the response export)

This file is generated from the column headers and observed answer options of
`data/raw/responses.csv`, which is the authoritative record of what respondents
actually saw. The form was bilingual Kazakh/English and was administered through
Google Forms between 2026-02-04 and 2026-02-23.

Regenerate with `python analysis/build_instrument.py`.
"""


def main():
    rows = list(csv.reader(RAW.open(encoding="utf-8-sig", newline="")))
    hdr, data = rows[0], rows[1:]

    lines = [HEADER]
    for i, h in enumerate(hdr):
        if i == 0:
            continue
        lines.append(f"## {' '.join(h.split())}\n")
        opts = Counter(
            " ".join(r[i].split()) for r in data if i < len(r) and r[i].strip()
        )
        for opt, n in sorted(opts.items(), key=lambda kv: -kv[1]):
            # Multi-select answers are stored comma-joined; list atomic options only.
            if opt.count("/") <= 1:
                lines.append(f"- {opt}  _(n={n})_")
        lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
