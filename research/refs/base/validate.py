"""Consistency checks for the SANASH reference base.

Run from anywhere:

    python research/refs/base/validate.py

Exits 0 when every check passes, 1 otherwise. Standard library only.

The point of this script is that the verified reference base cannot silently
drift. It does not check whether a DOI resolves or whether a paper says what an
email claims it says. It checks that the two files agree with each other and
that nothing is missing a link, a key or a legal status.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
REFERENCES_MD = BASE / "references.md"
REFERENCES_BIB = BASE / "references.bib"

# Contact lists. They live outside this directory and may not exist yet; a
# missing file is a warning, not a failure.
CONTACT_FILES = [
    BASE.parents[2] / "business" / "outreach" / "contacts" / "wave2.md",
    BASE.parents[2] / "business" / "outreach" / "contacts" / "wave3.md",
    BASE.parents[2] / "business" / "outreach" / "contacts" / "contacted.md",
]

VALID_STATUSES = {"CITED", "LISTED", "READ"}

KEY_RE = re.compile(r"^[a-z][a-z0-9]*_(?:\d{4}|nodate)_[a-z0-9]+$")
BIB_ENTRY_RE = re.compile(r"^@(\w+)\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)
BIB_FIELD_RE = re.compile(r"^\s*(\w+)\s*=\s*\{(.*)\}\s*,?\s*$")
DOI_IN_MD_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\)\]]+)")
URL_IN_MD_RE = re.compile(r"https?://[^\s\)\]]+")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

errors: list[str] = []
warnings: list[str] = []


def fail(message: str) -> None:
    errors.append(message)


def warn(message: str) -> None:
    warnings.append(message)


def parse_references_md(path: Path) -> list[dict]:
    """Return one record per table row in references.md."""
    rows = []
    section = None
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 4:
            fail(f"{path.name}:{lineno}: table row has {len(cells)} cells, expected 4")
            continue
        key, citation, link, status = cells
        if key == "Key" or set(key) <= set("-: "):
            continue  # header or separator
        rows.append(
            {
                "lineno": lineno,
                "section": section,
                "key": key,
                "citation": citation,
                "link": link,
                "status": status,
            }
        )
    return rows


def parse_references_bib(path: Path) -> dict[str, dict]:
    """Return {key: {'type': ..., 'fields': {...}}} for every bib entry."""
    text = path.read_text(encoding="utf-8")
    entries: dict[str, dict] = {}
    matches = list(BIB_ENTRY_RE.finditer(text))
    for i, match in enumerate(matches):
        entry_type, key = match.group(1).lower(), match.group(2)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        fields: dict[str, str] = {}
        for line in text[start:end].splitlines():
            field = BIB_FIELD_RE.match(line)
            if field:
                fields[field.group(1).lower()] = field.group(2).strip()
        if key in entries:
            fail(f"references.bib: duplicate entry key {key}")
        entries[key] = {"type": entry_type, "fields": fields}
    return entries


def check_keys_unique(rows: list[dict]) -> None:
    counts = Counter(row["key"] for row in rows)
    for key, count in sorted(counts.items()):
        if count > 1:
            fail(f"references.md: citation key {key} appears {count} times")
    for row in rows:
        if not KEY_RE.match(row["key"]):
            fail(
                f"references.md:{row['lineno']}: key {row['key']!r} is not "
                "surname_year_word (year may be 'nodate')"
            )


def check_md_bib_agree(rows: list[dict], entries: dict[str, dict]) -> None:
    md_keys = {row["key"] for row in rows}
    bib_keys = set(entries)
    for key in sorted(md_keys - bib_keys):
        fail(f"key {key} is in references.md but not in references.bib")
    for key in sorted(bib_keys - md_keys):
        fail(f"key {key} is in references.bib but not in references.md")


def check_links(rows: list[dict], entries: dict[str, dict]) -> list[str]:
    """Every entry needs a DOI or a URL. Return the keys that have neither."""
    unlinkable = []
    for row in rows:
        key = row["key"]
        md_has_link = bool(
            DOI_IN_MD_RE.search(row["link"]) or URL_IN_MD_RE.search(row["link"])
        )
        fields = entries.get(key, {}).get("fields", {})
        bib_has_link = bool(fields.get("doi") or fields.get("url"))
        if md_has_link != bib_has_link:
            fail(
                f"{key}: references.md and references.bib disagree on whether a "
                f"link exists (md={md_has_link}, bib={bib_has_link})"
            )
        if not md_has_link and not bib_has_link:
            unlinkable.append(key)
    return unlinkable


def check_duplicate_dois(entries: dict[str, dict]) -> None:
    seen: dict[str, list[str]] = {}
    for key, entry in entries.items():
        doi = entry["fields"].get("doi")
        if doi:
            seen.setdefault(doi.lower(), []).append(key)
    for doi, keys in sorted(seen.items()):
        if len(keys) > 1:
            fail(f"duplicate DOI {doi} on keys: {', '.join(sorted(keys))}")


def check_duplicate_urls(entries: dict[str, dict]) -> None:
    """A shared URL is usually two rows for one paper. Warn, do not fail."""
    seen: dict[str, list[str]] = {}
    for key, entry in entries.items():
        url = entry["fields"].get("url")
        if url:
            seen.setdefault(url.rstrip("/").lower(), []).append(key)
    for url, keys in sorted(seen.items()):
        if len(keys) > 1:
            warn(f"same URL on {len(keys)} entries ({', '.join(sorted(keys))}): {url}")


def check_statuses(rows: list[dict]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        word = row["status"].split()[0] if row["status"].split() else ""
        if word not in VALID_STATUSES:
            fail(
                f"references.md:{row['lineno']}: status starts with {word!r}, "
                f"expected one of {sorted(VALID_STATUSES)}"
            )
            continue
        counts[word] += 1
    return counts


def check_unverified_notes(entries: dict[str, dict]) -> None:
    """An entry missing a field its type requires must say so in note."""
    required = {
        "article": ("author", "title", "year", "journal"),
        "inproceedings": ("author", "title", "year", "booktitle"),
        "incollection": ("author", "title", "year"),
        "book": ("author", "title", "year", "publisher"),
        "misc": ("author", "title"),
    }
    for key, entry in sorted(entries.items()):
        needed = required.get(entry["type"])
        if needed is None:
            warn(f"{key}: unrecognised entry type @{entry['type']}")
            continue
        missing = [f for f in needed if not entry["fields"].get(f)]
        note = entry["fields"].get("note", "")
        if missing and "UNVERIFIED FIELDS" not in note:
            fail(
                f"{key}: @{entry['type']} is missing {', '.join(missing)} "
                "and carries no note = {UNVERIFIED FIELDS}"
            )


def check_contact_emails() -> None:
    present = [p for p in CONTACT_FILES if p.exists()]
    if not present:
        warn(
            "no contact list found at business/outreach/contacts/. The duplicate "
            "email check did not run. This is a gap in the data, not a passing check."
        )
        return
    for path in CONTACT_FILES:
        if not path.exists():
            warn(f"contact list missing: {path.relative_to(BASE.parents[2])}")
    seen: dict[str, list[str]] = {}
    for path in present:
        for email in EMAIL_RE.findall(path.read_text(encoding="utf-8")):
            seen.setdefault(email.lower(), []).append(path.name)
    for email, files in sorted(seen.items()):
        if len(files) > 1:
            fail(f"duplicate email {email} across {', '.join(sorted(set(files)))}")
    print(f"  contact lists checked: {', '.join(p.name for p in present)}")
    print(f"  unique email addresses: {len(seen)}")


def main() -> int:
    for path in (REFERENCES_MD, REFERENCES_BIB):
        if not path.exists():
            print(f"FAIL: {path} does not exist")
            return 1

    rows = parse_references_md(REFERENCES_MD)
    entries = parse_references_bib(REFERENCES_BIB)

    check_keys_unique(rows)
    check_md_bib_agree(rows, entries)
    unlinkable = check_links(rows, entries)
    check_duplicate_dois(entries)
    check_duplicate_urls(entries)
    statuses = check_statuses(rows)
    check_unverified_notes(entries)

    print("SANASH reference base")
    print(f"  references in references.md: {len(rows)}")
    print(f"  entries in references.bib:   {len(entries)}")
    print()
    print("  by status:")
    for status in sorted(VALID_STATUSES):
        print(f"    {status:<7} {statuses.get(status, 0)}")
    print()
    print("  by section:")
    section_counts = Counter(row["section"] for row in rows)
    for section, count in section_counts.items():
        print(f"    {count:>3}  {section}")
    print()
    check_contact_emails()
    print()

    if unlinkable:
        print(f"  entries with neither DOI nor URL ({len(unlinkable)}):")
        for key in unlinkable:
            print(f"    {key}")
        print()

    for message in warnings:
        print(f"WARNING: {message}")
    if warnings:
        print()
    for message in errors:
        print(f"FAIL: {message}")

    if errors:
        print(f"\n{len(errors)} check(s) failed.")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
