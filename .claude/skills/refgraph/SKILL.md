---
name: refgraph
description: Use when Diyas drops a batch of reference PDFs (papers, reports) into the project and asks which are needed for the RTCI research track and which to delete. Cheap first-pass extraction script plus a keep/reject judgment pass against Ground Truth and the RTCI charter. Invoke with /refgraph or let Claude pick it up automatically when a PDF batch shows up.
---

# Refgraph

## What this replaces

A tool named `research/scripts/refgraph.py` existed before the ceiling-RGB
pivot and was deleted. It is not recoverable — checked all branches in git
history, never committed. This skill is a fresh build, not a restore.

## Why this exists

Reading 60 PDFs one at a time by having the model open and reason over each
full document is expensive. Most of that cost buys nothing: a paper's
relevance to the RQ is usually decidable from title + abstract + venue, not
from full text. This skill front-loads a cheap, deterministic extraction
pass (no LLM call) so the model judges relevance from compact text, not raw
PDFs, and only opens full text for genuinely borderline cases.

## Scope

Reference literature triage only. Not for the CV/training dataset track
(see research-trainer) and not for drafting paper prose (see
research-writer). This skill decides keep/reject and updates the reference
index; it does not write literature review paragraphs.

## Procedure

1. Read `GROUND_TRUTH.md` and `research/RTCI_RESEARCH_CHARTER.md` first.
   The keep/reject criterion is relevance to the current RQ (causal effect
   of RTCI display on bus boarding decisions) and its supporting themes:
   willingness-to-wait, crowding valuation, RTCI field experiments, boarding
   choice models, revealed-preference transit behavior, bus bunching /
   operations, sensing and prediction of crowding. Do not invent a
   different criterion.

2. Run the extraction script over the batch directory:
   ```
   python .claude/skills/refgraph/scripts/extract_pdf_metadata.py <input_dir> research/refs/graph.json
   ```
   This produces one JSON record per PDF: filename, page count, PDF
   metadata title/author if embedded, and raw text of the first ~2 pages
   (title/authors/abstract on most papers). No classification happens in
   the script — it is pure extraction, cheap and deterministic.

3. Read `research/refs/graph.json` (it will be small — first-2-pages text
   only, not full papers) and classify each record:
   - **keep**: relevant to RQ/themes above, or a close comparator/novelty
     threat worth tracking even if not directly citable.
   - **reject**: off-topic, duplicate of an already-verified source, or a
     course/business document that isn't literature.
   - **borderline**: abstract text was truncated, garbled (some PDFs
     extract with one-character-per-line spacing, seen before in this
     project), missing, or genuinely ambiguous. For borderline cases only,
     open the specific PDF page range needed to decide — don't default to
     reading full text for everything.

4. Cross-check against the existing corpus before accepting anything as
   new: `rtci_paper_inventory.csv`, `research/refs/rtci-supporting/`, and
   the Obsidian vault `01_Projects/Sanash/paper/literature.md` (~50-60
   existing entries). A title match against an existing entry is a
   duplicate, not a new keep.

5. Write `research/refs/REFGRAPH_REPORT.md`: one row per input PDF —
   filename, verdict (keep/reject/duplicate), one-line reason, and status
   (verified from text extracted / verified from full-text read /
   metadata only). This is the audit trail; don't skip it even for
   obvious rejects.

6. For files verdicted **reject**, delete them only after showing Diyas
   the list and reason per file. Do not delete silently — "delete boldly"
   from Diyas means don't hesitate once the call is made, not skip
   showing the list. Batch-delete only after confirmation on the list as
   a whole; no need to confirm file-by-file.

7. Update `rtci_paper_inventory.csv` for anything kept that isn't already
   an entry there.

## Hard constraints

- Never fabricate a DOI, venue, sample size, or finding for a paper the
  extraction script couldn't get clean text from. Mark it "metadata only"
  or "extraction failed" instead of guessing from the filename.
- Never treat this pass as a substitute for full-text verification before
  a paper is actually cited in the paper draft. Keep/reject here is
  triage, not citation-ready verification — that bar is set in
  research-writer's citation rules.
- If the extracted text for a PDF is empty or clearly broken (e.g. a
  scanned image PDF with no text layer), say so explicitly rather than
  marking it reject by default — a scanned paper can still be relevant,
  it just needs OCR or manual title lookup.
