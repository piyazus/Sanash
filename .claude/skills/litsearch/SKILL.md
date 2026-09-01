---
name: litsearch
description: Use when Diyas needs to find scientific papers for the Sanas RTCI or CV track - a deep, reproducible literature search rather than a single web query. Runs OpenAlex keyword blocks plus backward/forward citation chaining, deduplicates against the local corpus, and produces a screening sheet with PRISMA-style counts. Invoke with /litsearch or pick it up when the ask is "find papers on X", "who else studied X", "what am I missing in the literature".
---

# Litsearch

## Why this exists

`refgraph` triages PDFs that already sit on disk. This skill answers the
other question: what papers exist that Diyas has not downloaded yet, and how
to find them without inventing citations.

A single web search finds the popular paper, not the field. Keyword search
alone misses work that uses different vocabulary for the same construct
(crowding valuation, crowding disutility, willingness to wait, comfort
penalty, load factor discomfort). Citation chaining finds those, because a
paper that five of your seeds all cite is central even when its title shares
no words with your query.

Measured on the first real run in this repo: keyword queries returned RAND's
autonomous-vehicle policy report and an airport service-robot paper as top
hits, while chaining from three seed DOIs surfaced Kroes, Tirachini, Wardman,
Hörcher, Yap and the target author's own 2025 Transportation paper. Chaining
is the part that works. Do not skip it.

## Scope

Finding and screening literature. Not triage of existing PDFs (`refgraph`),
not writing prose or final citations (`research-writer`). Output of this
skill is a screening sheet plus wiki pages, never a paragraph of a paper.

## Hard constraints

1. Never write a citation this skill did not retrieve. Every row in the
   output carries a DOI or OpenAlex id from the API response.
2. OpenAlex metadata is metadata. It is enough for keep/reject screening,
   not enough to state what a paper found. Reading the abstract is
   `abstract-only`; reading the PDF is `full-text`. Label which one applies.
3. Record every executed query. The manifest file is written automatically,
   do not delete it: `RTCI_RESEARCH_CHARTER.md` section 9 requires reporting
   the actual strings, databases, dates and counts.
4. Do not silently cap coverage. If a run was limited to N results per query
   or one hop of chaining, say so in the report.
5. Retrieved data goes to `data/litsearch/` (gitignored). Only the screening
   verdicts and wiki pages get committed.
6. Downloading a paper PDF is a separate step and needs Diyas to say so.
   This skill does not fetch full texts by itself.

## Procedure

1. **Fix the question first.** Read `research/RTCI_RESEARCH_CHARTER.md`
   section 9 for the review question and concept blocks, or
   `GROUND_TRUTH.md` if the ask is on the device track. Write down the
   inclusion criterion before searching. If the ask does not map to either
   track, ask Diyas rather than guessing the criterion.

2. **Keyword pass.** Build one search string per concept combination and run:

   ```
   python .claude/skills/litsearch/scripts/oa_search.py \
     --queries-file data/litsearch/queries.txt \
     --out data/litsearch/<topic>.jsonl --max 200 --from-year 2005
   ```

   Each line of `queries.txt` is a full search string. Prefer several narrow
   strings over one broad one: OpenAlex `search` matches full text, so broad
   strings return thousands of weak hits.

3. **Seed selection.** From the keyword pass plus what the repo already
   trusts, pick 3-8 papers that are unambiguously on target. Their DOIs go
   into `data/litsearch/seeds.txt`. Bad seeds poison the whole chain, so
   prefer few and certain over many and hopeful.

4. **Citation chaining, both directions.**

   ```
   python .claude/skills/litsearch/scripts/oa_chain.py \
     --seeds data/litsearch/seeds.txt \
     --out data/litsearch/chain.jsonl --min-seeds 2 --max-forward 200
   ```

   `--min-seeds 2` keeps only works reached from at least two seeds, which
   is the noise filter. Lower it to 1 only for a deliberately wide sweep,
   and say in the report that you did.

   For a deeper sweep, run a second hop: promote the top chained results to
   seeds and chain again. Two hops is usually where new names stop
   appearing. Stop when a round adds nothing new, not at a fixed count.

5. **Deduplicate and screen.**

   ```
   python .claude/skills/litsearch/scripts/screen.py \
     --in "data/litsearch/*.jsonl" --out data/litsearch/screen.md
   ```

   The sheet separates new candidates from work already in
   `rtci_paper_inventory.csv` and `research/wiki/sources/`. Read the
   abstracts section and mark each new candidate include, exclude or unclear
   against the criterion from step 1. Give a reason per exclusion; "not
   relevant" is not a reason.

6. **File the result.** For an included paper that changes something, create
   `research/wiki/sources/<slug>.md` per `research/wiki/WIKI_SCHEMA.md`,
   update the affected concept pages, update `research/wiki/index.md`, and
   append one line to `research/wiki/log.md`. Add new entries to
   `rtci_paper_inventory.csv`.

7. **Report the search itself.** State: databases used, number of query
   strings, hits and retrieved per string, chaining hops and seed count,
   duplicates removed, screened, included. These numbers come from the
   manifest and the screening sheet, never from memory.

## Databases

- **OpenAlex** is the engine here: free, no key, exposes reference lists and
  citing works, which is what makes chaining possible. Confirmed reachable
  from this machine.
- **Crossref** confirmed reachable, useful to verify a DOI's metadata.
- **Semantic Scholar** returns HTTP 429 without an API key. Usable only with
  a key and backoff; do not build a run around it.
- **Scite and Consensus MCP servers** are available in some sessions and
  give citation-context snippets. Use them to check how a paper is cited,
  not as the retrieval backbone: their result sets are not enumerable in a
  way that can be reported as a systematic search.
- Scopus, Web of Science and TRID are named in the charter but need
  institutional access. If Diyas has it, the queries from `queries.txt` are
  reusable there by hand, and the counts belong in the same report.

## Known limits

- OpenAlex `search` covers full text where available, so a common word in
  the query pulls unrelated fields. Narrow strings beat long ones.
- Reference lists in OpenAlex are incomplete for some publishers. A missing
  backward link is not evidence that a paper cites nothing.
- Forward citations lag for very recent papers. A 2026 paper with zero
  citing works is not necessarily ignored.
- The scripts do not read PDFs. Full-text verification is a separate,
  explicit step.
