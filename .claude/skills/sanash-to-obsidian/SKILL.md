---
name: sanash-to-obsidian
description: Curate raw Sanash agent captures from the Obsidian inbox into clean project notes. Use when Diyas says save/curate agent work to Obsidian, or asks to process the agent captures. Reads 00_Inbox/sanash-agents/, writes atomic notes into the right project space, archives consumed captures.
---

# Sanash → Obsidian curate

Turns the raw agent output the SubagentStop hook dropped in the vault inbox
into clean, atomic project notes. This is the manual half of the hybrid
capture system; the hook is the automatic half.

Vault: `C:\Users\User\OneDrive\Desktop\obsidian\DiyasVault`

## Inputs

Raw captures in `00_Inbox\sanash-agents\`:
- `danyshpan-<date>.md`, `donatello-<date>.md`, `cady-<date>.md`,
  `pitch-<date>.md` — one file per agent per day, each with timestamped
  `## HH:MM — <task>` sections.
- `other-<date>.md` — subagents the hook could not identify. Review by hand;
  usually skip.

## Steps

1. List the capture files. Read each recognized one (skip `_index.md` and
   `other-*` unless asked).
2. Show a short plan before writing: which captures become which notes, and
   where. Wait for a go on anything more than one or two notes (vault rule:
   show a plan before a bulk change).
3. For each capture entry, write or update a clean note in the destination
   below, in vault format:
   - frontmatter: `title`, `tags`, `created` (real date), `status: active`
   - H1 repeating the title
   - atomic — one topic per note; split a capture that covers several
   - `[[wikilinks]]`, never raw paths in prose
   - link back to the repo files the agent touched (e.g.
     `development/experiments/log.md`, the paper draft) so the vault stays an index and
     details stay in the repo
4. Never invent. Use only what the capture contains. If the capture is thin,
   write a thin note and say so — do not pad it.
5. After an entry is curated, move its capture file to
   `06_Archive\sanash-agents\` (create the folder if needed). Never delete —
   vault rule. If a file has several entries and only some are curated, leave
   the file and note which entries are done.

## Routing

| Capture | Destination |
| --- | --- |
| danyshpan | `01_Projects\Sanash\paper\` — a note per section/decision; link from `paper\_index.md` |
| donatello | `01_Projects\Sanash\research\` for findings; append run decisions to `01_Projects\Sanash\decisions.md` |
| cady | `01_Projects\Sanash\research\` — diagram / CAD / chart notes |
| pitch | `01_Projects\Sanash\pitch\` — deck / one-pager / demo notes |

Keep the two tracks separate: CV/training (Donatello, CADy) and the paper
(Danyshpan) do not share a note. Pitch pulls real numbers only, no invention.

## Donatello decisions

When a Donatello capture records a model/backend/config choice, add a
`### YYYY-MM-DD — <decision>` block to `decisions.md` with Context / Decision
/ Why / Alternatives rejected, matching the existing journal format there.

## Danyshpan method

Danyshpan notes should already follow the course method
(`03_Learning\Research Writing\`). When curating, keep IEEE numbering and
theme-based structure intact; do not flatten it.
