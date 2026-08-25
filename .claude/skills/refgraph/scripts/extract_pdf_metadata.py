"""Cheap first-pass PDF metadata extraction for refgraph.

Reads every PDF in a directory, pulls PDF metadata (title/author if
present) plus the first ~2 pages of raw text (usually title, authors,
abstract), and writes one JSON record per file. Does not call any LLM
and does not classify relevance — that judgment happens after, over
the compact JSON, not over 60 raw PDFs.

Usage:
    python extract_pdf_metadata.py <input_dir> <output_json>
"""

import json
import sys
from pathlib import Path

from pypdf import PdfReader

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def extract_one(path: Path) -> dict:
    record = {
        "file": path.name,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "error": None,
        "num_pages": None,
        "pdf_title": None,
        "pdf_author": None,
        "first_pages_text": None,
    }
    try:
        reader = PdfReader(str(path))
        record["num_pages"] = len(reader.pages)
        meta = reader.metadata or {}
        record["pdf_title"] = getattr(meta, "title", None)
        record["pdf_author"] = getattr(meta, "author", None)

        pages_to_read = min(2, len(reader.pages))
        text_parts = []
        for i in range(pages_to_read):
            try:
                text_parts.append(reader.pages[i].extract_text() or "")
            except Exception as e:
                text_parts.append(f"[page {i + 1} extract failed: {e}]")
        text = "\n".join(text_parts).strip()
        record["first_pages_text"] = text[:3000]
    except Exception as e:
        record["error"] = str(e)
    return record


def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_pdf_metadata.py <input_dir> <output_json>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_json = Path(sys.argv[2])

    if not input_dir.is_dir():
        print(f"Not a directory: {input_dir}")
        sys.exit(1)

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {input_dir}")
        sys.exit(1)

    records = []
    for i, pdf_path in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {pdf_path.name}")
        records.append(extract_one(pdf_path))

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    errors = [r for r in records if r["error"]]
    print(f"\nDone. {len(records)} files processed, {len(errors)} errors.")
    if errors:
        print("Errors:")
        for r in errors:
            print(f"  {r['file']}: {r['error']}")


if __name__ == "__main__":
    main()
