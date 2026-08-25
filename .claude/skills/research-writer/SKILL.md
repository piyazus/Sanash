---
name: research-writer
description: Use when writing, editing, or reviewing academic paper content for the Sanas/Avtobys RTCI field-experiment paper — intro sections, literature review, discussion, or citations. Invoke with /research-writer or let Claude pick it up automatically for academic writing tasks in this repo.
---

# Research Writer

## Scope
Academic writing track only (the causal RTCI paper). Not for the CV/training
track — use research-trainer for that.

Read `GROUND_TRUTH.md` first. The active academic scope here is RTCI causal
research; the door/ToF device paper and figures were cancelled.

## Citation rules (hard constraints)
1. IEEE citation style, numbered, matching reference list order.
2. Every claim in a discussion paragraph needs an in-text citation.
3. Never fabricate a source detail — volume, issue, city, month, sample
   size. If a detail can't be confirmed, say so instead of guessing.
4. Distinguish explicitly: confirmed from full text vs inferred from
   abstract only. Mark which is which when reporting back.
5. Verify claims and find related work via Scite (connected). Don't invent
   a citation that Scite/search can't surface.
6. Never use Sci-Hub or any other paywall-bypass source. For paywalled
   papers: try Unpaywall, arXiv, DOAJ, PubMed Central, or the school/NIS
   library's institutional access. If none of those work, say the paper
   isn't accessible rather than routing around the paywall.
7. One quote per source maximum, under 15 words, or paraphrase entirely —
   standard copyright limits apply even inside a paper draft.

The former literature graph and `research/scripts/refgraph.py` were deleted
during the architecture pivot. Do not refer to them as available. Use the live
RTCI corpus and verified primary sources.

## Literature review structure
- Organize by cross-cutting theme (e.g. "field vs. simulation evidence,"
  "crowding valuation multipliers"), not one paragraph per article.
- Known verified sources so far: Kapatsila et al. 2025, Drabicki et al.
  2022 & 2023, Chen et al. 2023, Zhang et al. 2017, Yap et al. 2025,
  Fiorista et al. 2025 (arXiv preprint). Check project files/memory before
  treating any of these as settled — details may have been updated.

## Method reference (course notes)
Diyas is taking a research-writing course. The method notes live in the
Obsidian vault at `C:\Users\User\OneDrive\Desktop\obsidian\DiyasVault\03_Learning\Research Writing\`.
Before drafting or revising a section, read them and keep the paper consistent:
- `finding-research-gaps.md` — six gap types (geographic, population,
  methodological, theoretical, time, contradictory findings), where to find
  gaps, organize the literature by theme not by author.
- `intro-research-writing.md` — four-part intro formula, lit-review vs
  empirical structure, body-section/paragraph anatomy, quoting vs
  paraphrasing (~70% paraphrase), IEEE numbering, outline.
These are the conventions Diyas is being taught; match them.

## Intro structure
4-part formula: Background, Problem, Research Gap, Purpose Statement.
Target 300-350 words unless told otherwise. (Full formula in the course note
`intro-research-writing.md` above.)

## Figures/charts
Danyshpan doesn't build figures. When one is needed (e.g. crowding valuation
comparison, RTCI effect sizes across studies), specify it for CADy: which
numbers and which verified sources they come from, what's being compared,
and what type of figure it should be. Assign the figure number and write the
caption per IEEE convention (caption below figure, numbered, referenced in
text); CADy does the actual build under the research-diagrams skill.

Interpretation stays here. CADy hands back the chart and the raw numbers it
plotted; writing what the result means is Danyshpan's job, including for
training-metric charts pulled from Donatello's experiment log.

## Before writing
If Diyas says "tell me what you'd write first" or similar, state the
sources and structure you're planning and wait for approval before
producing prose. Don't skip this when asked for it.

## Style
English for this track. Short, direct, no flattery. Flag uncertainty
plainly rather than smoothing over it.
