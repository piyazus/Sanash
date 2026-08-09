---
name: research-diagrams
description: Use when building diagrams or schematics for either track — CV system architecture (data flow, edge pipeline, model architecture), or figures for the paper (study design, timeline, comparison charts). Invoke with /research-diagrams or let Claude pick it up automatically for diagram requests.
---

# CADy — diagrams and schematics

## Scope
Any visual that isn't prose: system architecture, data pipeline, model
diagrams, study-design schematics, comparison charts for the paper. Not
responsible for the underlying analysis or writing — pull numbers/structure
from Donatello's experiment log or Danyshpan's verified sources, don't
invent them here.

## Tools/format
- System/architecture/flow diagrams: Mermaid or diagrams-as-code (Python
  `diagrams` lib, Graphviz) checked into the repo as source, not just a
  rendered image — so they're editable later.
- Data charts (metrics, comparisons, valuation multipliers): generate with
  real code (matplotlib/plotly) against real numbers, never a mockup
  presented as data.
- Paper figures: match IEEE figure conventions (caption below figure,
  numbered, referenced in text) — coordinate with Danyshpan on numbering.

## Rules
1. Never fabricate data points to fill out a chart. If a number isn't
   confirmed, say so and leave it out or mark it as estimated.
2. Keep diagram source files in the repo (not just exported images) so they
   can be regenerated when numbers change.
3. For CV architecture diagrams, reflect what Donatello's log actually
   shows was chosen/tested — don't diagram a model that was only proposed
   and never run without labeling it as proposed.
4. Ask before starting if the request is ambiguous about which diagram type
   fits (architecture vs. data chart vs. study timeline) rather than
   guessing and producing the wrong shape of output.
