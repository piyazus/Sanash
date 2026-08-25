# Research Method Grounding

Compiled 2026-08-26. Synthesizes every video and document in
`research/coursework/` for research methodology (research question, gap,
literature review, methodology, design/dataset/metric/validation choices).
Does not touch Sanas product architecture or `GROUND_TRUTH.md`.

This file adds one thing `course_notes.md` (2026-08-20, already in this
directory) does not have: coverage of `video1311882563.mp4`, a 20-slide data
analysis lecture never referenced anywhere else in the repo, and an audit of
`literature_review_methodology.docx`, a generated (not course) artifact.
Everything already in `course_notes.md` is treated as verified prior work and
summarized here, not redone. Read `course_notes.md` directly for full quotes
and the complete argument on each topic.

## Source inventory

| File | Type | Watched/read | Transcript | Limitations |
|---|---|---|---|---|
| `1st lesson- part 1.mp4` (34:56) | recording | Previously transcribed + read (per `course_notes.md`, verified against `transcripts/lesson1_part1.txt` this session) | Yes, faster-whisper small int8 | Names/technical terms occasionally mangled per README |
| `1st lesson- part 2.mp4` (19:00) | recording | Previously transcribed + read | Yes, `transcripts/lesson1_part2.txt` | Same as above |
| `video1768002125.mp4` (24:56) | recording | Previously transcribed + read | Yes, `transcripts/video1768002125.txt` | none noted |
| `video1168216779.mp4` (26:16) | recording | Previously transcribed + read | Yes, `transcripts/video1168216779.txt` | none noted |
| `video1433589796.mp4` (53:55) | recording | Previously transcribed + read | Yes, `transcripts/video1433589796.txt` | none noted |
| `video1101430871.mp4` (46:02) | recording | Previously transcribed + read | Yes, `transcripts/video1101430871.txt` | none noted |
| **`video1311882563.mp4` (32:34)** | recording | **New this session.** No transcript exists and none was generated (out of scope: task says extract from visible slides only when transcript is absent, don't invent speech). Video track sampled at 1 frame/60s (33 frames) plus a scene-detect pass; confirmed via `ffprobe` to have an AAC audio track that was not transcribed. | **No** | **Speech content is unknown.** Only the visible slide deck (`Analysis slides.pdf`, 20 pages, shown via screen share) could be read. Presenter is on camera throughout but slides are the only accessible content. Not listed in `README.md` or `course_notes.md` — this is new material to the repo's documentation, not new material to the disk. |
| `Foundations class- group 3.pdf` | slides, 20 pp | Previously read (page count verified: 20 pp match) | n/a | none |
| `Research - intro.pdf` | slides, 20 pp | Previously read (verified: 20 pp) | n/a | none |
| `2nd class-Reading.pdf` | slides, 19 pp | Previously read (verified: 19 pp) | n/a | none |
| `Lit. review.pdf` | slides, 15 pp | Previously read (verified: 15 pp) | n/a | none |
| `Methodology.pdf` | slides, 17 pp | Previously read (verified: 17 pp) | n/a | none |
| `Methodology for emp. papers.pdf` | slides, 15 pp | Previously read (verified: 15 pp) | n/a | **Truncated at component 5 of 6** (ethics/consent component missing from the file itself, not from the reading) |
| `Mohlaroy--RAS.docx` | worked student paper, 22 refs | Previously read; confirmed this session (title and final reference match `course_notes.md` exactly, 102 non-empty paragraphs) | n/a | none |
| `course_notes.md` | synthesis, ~11.8k words | Read in full this session | n/a | Treated as verified prior work, not re-derived |
| `README.md` | index | Read in full this session | n/a | none |
| **`literature_review_methodology.docx`** | **not course material** — a generated methodology draft | **New this session.** Read in full (10 non-empty paragraphs) | n/a | See note below. Not a Terra course artifact; excluded from the course synthesis sections. Docx properties: `author: python-docx`, `last_modified_by: moneyman`, `modified: 2026-08-25`. This is machine-authored output describing a scoping review already run on a Sanas-adjacent research question ("How effectively do deep learning-based computer vision methods estimate passenger density in public transport vehicles?"), citing Scite, CSRNet-lineage, and "this review's parent project." It is evidence of a prior literature-review exercise, not evidence of course content, and it is not corroborated elsewhere in this repo (no matching entry in `development/experiments/log.md` or `GROUND_TRUTH.md`). Its "Eligibility criteria" section heading has no body text — the file itself is incomplete. **Status: unverified artifact, not authoritative for product or paper claims.** |
| `~$terature_review_methodology.docx` | Word lock file | Not a document — confirmed via `ls`: 162-byte owner-lock file for the `.docx` above, created because that file was open in Word at time of inspection | n/a | Not content; excluded from all synthesis below |

## Lesson-by-lesson notes

Sections 1–6 below are drawn from the six previously-transcribed recordings
and are condensed from `course_notes.md` §§1–16, which remains the source of
full quotes and timestamps. Section 7 is new.

### 1. Foundations (`1st lesson- part 1/2.mp4`, combined ~54 min)

What research is/is not; paraphrase vs copy; AI policy (strict, this tutor);
no personal pronouns; bias vs opinion. Full detail in `course_notes.md` §1.

### 2. Research question, gap, sourcing (same two recordings)

RQ formula `How/Why does [factor] affect [outcome] among [group/context]?`,
10–15 words; red flags; CRAAP; database list; bibliography tag; Zotero.
`course_notes.md` §§2, 4.

### 3. Reading academic literature (`video1768002125.mp4`, `video1168216779.mp4`)

SMART reading order (title→abstract→intro→conclusion→headings→figures→
methods/results last); abstract's four parts; finding RQ and contribution.
`course_notes.md` §5. These two recordings are also the "blueprint/outline"
session: four-part introduction, research-gap types, purpose statement.
`course_notes.md` §§10–11.

### 4. Literature review (`video1433589796.mp4`, 53:55)

Conversation metaphor; theme structure; summary vs synthesis; four-part
paragraph; comparing/contrasting; citation rules; `Mohlaroy--RAS.docx`
dissected live from ~30:00 as the model paragraph and model paper.
`course_notes.md` §§6, 12–13.

### 5. Methodology for literature-review papers (`video1101430871.mp4`, 46:02)

Five components (design, databases, search strategy, eligibility criteria,
analysis strategy) plus screening and word budget; Q&A on paper count and AI
use. `course_notes.md` §8.

### 6. Methodology for empirical papers (slides only, no matching recording)

Six components (design, population, sample, instrument, data collection,
ethics); survey vs interview vs mixed methods; the empirical-method
landscape including "training a model on a public dataset counts as
computational empirical research." `course_notes.md` §9. Ethics component
text is missing from the source PDF itself (truncated at page 15/15).

### 7. Data analysis lecture — `video1311882563.mp4` (32:34) — NEW

**No transcript.** The following is read from the 20-slide deck shown on
screen (`Analysis slides.pdf`, per the visible file path in the recording:
`C:/Users/kamil/Downloads/Analysis-%20slides.pdf`). Slide numbers below are
PDF page numbers, not video timestamps — no reliable timestamp-to-slide map
exists without a transcript, so **all points here are slide-text-only,
speaker commentary is not represented, and nothing below should be quoted as
something a tutor said.** Frame sampling was 1 frame/60s across the full
32:34 (33 samples), so slide dwell time and any brief interstitial slides
between samples cannot be ruled out as missed. Twelve "Step" slides were
found; two intervening slide numbers (14, in the run between Steps 8 and 9;
and any slides between 15–17 title cards) were not individually captured but
their content is inferable from the surrounding steps and step numbering is
otherwise continuous 1–12.

Deck content, by Step:

- **Step 1** (p.8): "Your RQ is your filter." Worked example — RQ *"To what
  extent does real-time bus occupancy information change boarding decisions
  of Almaty commuters?"* — with a 10-item example survey variable list (age,
  gender, income, commute frequency, bus usage, transportation preferences,
  app usage, perceived crowding, willingness to wait, response to occupancy
  information). Explicit instruction: don't analyze all variables, ask "does
  this help answer my RQ?" for each one.
- **Step 2** (p.9): The "so what?" test. A number alone (e.g. "61% of
  respondents are 18–25") is not a finding until tied back to the RQ; a
  comparison across groups (72% vs 49%) becomes a finding once framed as a
  23-point difference relevant to the RQ.
- **Step 3** (p.10): Data cleaning before analysis — check for missing
  responses, duplicate responses, impossible responses ("does the data make
  sense"), inconsistent responses, unusable/meaningless open-ended responses.
- **Step 4** (p.11): Descriptive statistics as a required first pass before
  looking for relationships — frequency, percentage, mean, median, range,
  each illustrated with a worked number.
- **Step 5** (p.12): Three questions to ask of any RQ once the basic
  distribution is understood — "what is happening" (distribution), "who is
  different" (comparison), "what moves together" (relationship) — each
  illustrated with worked survey numbers from the same Almaty bus RQ.
- **Step 6** (p.13): Don't just report a raw percentage difference between
  groups ("72% of frequent commuters said yes") — ask why the difference
  exists and whether it is relevant to the RQ.
- **Step 8** (p.15): "Association ≠ Causation." Explicit worked example:
  finding that transit-app users are more likely to value occupancy
  information does not mean app use *causes* that value — lists confounds
  (prior interest in tech, age, commute frequency, other variables). Gives
  safe language ("X was associated with Y") vs dangerous language ("X caused
  Y").
- **Step 9** (p.16): A null or unexpected result is not a failed study —
  "almost no difference by age" is itself a result. Distinguishes stated
  preference from revealed behavior ("people prefer less crowded buses, but
  few would wait for the next one").
- **Step 10** (p.17): Look for contradictions inside your own data as a
  source of interesting findings — high preference + low behavior, high
  awareness + low usage, strong attitude + weak action.
- **Step 11** (p.18): "Do not graph everything." A graph must answer the RQ;
  worked bad example (pie chart of favorite colors, doesn't answer the RQ)
  vs good example (bar chart comparing two groups' willingness to change
  boarding decisions).
- **Step 12** (p.19): Every figure needs a job — a graph/table should show a
  difference, a trend, a distribution, a relationship, or an important
  pattern, not just exist.
- **p.20**: "Homework to be uploaded in google classroom!" — no further
  detail visible; matches the no-stated-deadlines gap already flagged in
  `course_notes.md` §16.

Steps 1–12 numbering implies at least one earlier "Step 0" / framing slide
before p.8 not distinctly captured by the 60-second sampling; the deck's
opening pages (1–7) were only partly sampled and mostly showed a title/agenda
pattern consistent with the other decks in this directory, not additional
numbered steps. This is a coverage gap, not a claim that no content exists
there — a future pass extracting all 20 pages directly from the PDF (if it
can be located — the PDF itself is not in this repo, only its on-screen
capture in the recording) would close it completely.

**Note on the RQ example used throughout this deck:** the worked RQ ("real-time
bus occupancy information... boarding decisions of Almaty commuters") is
topically adjacent to Sanas/RTCI. This is a generic teaching example
constructed by the deck's author for illustration — there is no evidence in
this deck, in the other recordings, or elsewhere in this repo that it is
connected to the actual Sanas or RTCI research question, and it must not be
treated as such. It is plausible this specific worked example was custom-built
for this student's course submission (unclear from slide content alone,
since no instructor speech was accessible) — flagged as inferred, not
confirmed.

## Presentation and document notes

Slide-by-slide detail for the six PDFs is in `course_notes.md` (§§1–16 cite
specific pages throughout; the deck-by-deck table is in that file's header
and `README.md`'s file table). Not reproduced here to avoid duplication.
Summary of what each contributes methodologically:

- `Foundations class- group 3.pdf` (20 pp): research question formula, scope
  control, red flags, CRAAP, annotated bibliography.
- `Research - intro.pdf` (20 pp): scholarly search workflow, bibliography
  tag, credible-vs-reliable distinction, peer-review/author/content signals.
- `2nd class-Reading.pdf` (19 pp): SMART reading order, abstract structure,
  active reading.
- `Lit. review.pdf` (15 pp): conversation metaphor, theme structure,
  summary-vs-synthesis, four-part paragraph, citation rules, IEEE numbering.
- `Methodology.pdf` (17 pp): five-component lit-review methodology, review
  type taxonomy, search-strategy reporting, eligibility criteria rules,
  two-pass screening, PRISMA flow diagram guidance, word budget.
- `Methodology for emp. papers.pdf` (15 pp, truncated): empirical-method
  landscape, why surveys are the default, six-component survey methodology,
  question-type taxonomy, sampling terms and size guidance. Ethics component
  (stated as component 6 of 6) has no body text in the file.
- `Mohlaroy--RAS.docx` (worked example, 22 refs): a complete scoping review
  applying every rule above — PRISMA-ScR, named databases, full search
  string, paired inclusion/exclusion criteria, two-reviewer screening,
  narrative synthesis, absence-as-finding, five specific (non-generic)
  limitations.

## Research-question framework

From `course_notes.md` §2, unchanged:

1. Start from a broad topic, narrow using the formula `How/Why does [factor]
   affect [outcome] among [group/context]?`.
2. Target 10–15 words. Longer is an explicit, named failure mode.
3. Required characteristics: focused, clear, researchable, specific,
   significant.
4. Red flags: too broad, unmeasurable, opinion-based, emotionally loaded,
   requires inaccessible data.
5. Narrow along time, place, population, platform, or variable — pick
   whichever axis actually bounds the question.
6. Read existing literature *before* finalizing the RQ, partly to sharpen it
   and partly to avoid duplicating existing work.

**Research gap types**, from `course_notes.md` §11.4: population gap,
geographical gap, temporal gap, comparative gap, and gaps from age group,
culture, method, theoretical perspective, or technology change. A gap must
be evidenced from the literature you reviewed, not asserted. Overclaiming
("nobody has studied X") is a named failure mode.

**Common mistakes**, consolidated from §§2, 11: RQs that are three sentences
long; questions with no measurable variable; treating "more research is
needed" as sufficient gap justification instead of naming what specifically
is missing; scope so broad it cannot be executed ("effects of technology on
society").

## Literature-review workflow

From `course_notes.md` §§4, 6, unchanged:

1. **Search**: build a keyword list from 3 concepts (topic, population,
   outcome typically), expand each with synonyms, join with OR within a
   concept and AND across concepts, check controlled vocabulary (MeSH, ERIC
   descriptors). Calibrate to 100–800 records; adjust if far outside that
   range.
2. **Inclusion/exclusion criteria**: write before screening, not after,
   to avoid confirmation bias. Must be testable from an abstract alone, each
   tied explicitly to the RQ, tight restrictions justified in one sentence,
   applied consistently.
3. **Screening**: two passes — title/abstract (fast, generous), then full
   text (slow, strict, log every exclusion reason). Deduplicate before
   counting. Report the funnel numbers (total → deduped → screened →
   full-text → included) in three sentences; a PRISMA flow diagram is the
   stronger version of the same disclosure.
4. **Extraction table**: fixed dimensions decided before screening (e.g.
   author/year/country/design/sample/method/findings), consistently applied.
5. **Synthesis by themes**: organize by idea, never by author name; use the
   four-part paragraph (topic sentence → evidence → analysis → transition);
   report agreement, tension, and *why* sources diverge, not just that they
   do.
6. **Citation verification**: cite every quote, paraphrase, specific study
   reference, and statistic; skip citations for common knowledge; IEEE
   numbering is by first appearance, never re-numbered, never alphabetical.

## Methodology framework

From `course_notes.md` §§7–9, unchanged:

- **Choosing design**: the type of evidence collected (numerical, textual,
  or both) must follow from the RQ, not be a style preference. Literature
  review (analyze existing papers) vs empirical (collect original data) are
  the two course-taught routes; empirical subtypes include experimental,
  lab/wet-lab, computational/simulation (explicitly includes "training an ML
  model on a public dataset"), observational, secondary data analysis, case
  study, content analysis.
- **Variables/hypotheses**: not explicitly named as a standalone
  vocabulary block in the course material; embedded in the instrument design
  guidance (state what is measured, why, and what question types capture
  it) and in Step 1 of the analysis lecture (define which collected
  variables actually bear on the RQ before analyzing any of them).
- **Dataset/sampling**: population (who the study is about, and who the
  conclusions can honestly generalize to) vs sample (who actually
  participated) are explicitly distinct. Convenience sampling is
  course-acceptable if disclosed as such. Target ≥100 responses for a
  quantitative survey where feasible; below ~30, percentages become
  unstable.
- **Instrument design**: state what was measured, item count, question
  types, and why those specific questions were chosen; borrowing a
  validated instrument from a cited source is explicitly flagged as a
  strength. Balance closed (analyzable) and open-ended (depth) questions.
- **Baseline/comparison**: not named as a modeling term in this course (it
  teaches survey/lit-review methodology, not ML baselines) — the closest
  analogue is Step 5–6 of the analysis lecture: every comparison between
  groups must be checked for relevance to the RQ, not just reported because
  it exists.
- **Metrics**: descriptive statistics first (frequency, percentage, mean,
  median, range) before any relationship analysis; every reported number
  must pass "does this help answer the RQ" before it is worth including.
- **Validation**: the closest course concept is the reproducibility
  standard stated at the very start of the course (`course_notes.md` §1.1) —
  a stranger reading only the methodology should be able to repeat the study
  and land on comparable results/comparable article set. For lit reviews
  this is operationalized as reporting the exact search string and date; for
  surveys as reporting the exact instrument and sampling method.
- **Association vs causation**: explicit, repeated warning (course_notes.md
  general theme, reinforced heavily in the newly-read analysis lecture's
  Step 8) — an observed association between two survey variables cannot be
  reported as one causing the other without ruling out confounds; use "X was
  associated with Y," never "X caused Y," unless the design supports it.
- **Limitations**: expected even though absent from the Methodology deck's
  own slide list — the tutor singled it out as the best feature of the
  worked example (`course_notes.md` §8.9a). Two kinds: systematic (from your
  own process — e.g. English-only search) and content (from what the
  included sources actually said, or didn't). Limitations should be
  specific, not the generic "small sample size."
- **Reproducibility**: the course's stated bar throughout — see
  "Validation" above. Also underlies the newly-read analysis lecture's
  insistence on reporting the actual search string, actual sample numbers,
  and disclosing partial screening rather than presenting it as full
  coverage.

## What applies to Sanas

Methodological, not architectural or product, conclusions only. No RQ,
target semantics, or tech stack is proposed here — those remain open per
`GROUND_TRUTH.md` §3.2 and §10.

1. **The course's RQ formula and 10–15 word constraint apply directly** to
   whatever RQ eventually gets set for the RTCI field-experiment paper (a
   causal question about crowding information and boarding decisions) —
   this is squarely the empirical-paper track the course describes (survey/
   field experiment collecting original data), not the literature-review
   track.
2. **The "so what" test and the distribution/comparison/relationship
   triad (Steps 2 and 5 of the newly-read analysis lecture)** are a usable
   checklist for whatever RTCI survey or field-experiment data eventually
   gets analyzed, independent of what the final RQ turns out to be.
3. **The association-vs-causation warning is directly load-bearing for
   RTCI**, since RTCI's entire premise is a claimed causal effect of
   real-time crowding information on boarding decisions. The course's "safe
   language" rule (associated with, not caused by) should gate how any
   correlational pilot or survey result is described in the paper, prior to
   and separate from whatever causal identification strategy the field
   experiment itself uses.
4. **The five/six-component methodology templates (lit review and
   empirical)** are directly reusable skeletons for the RTCI paper's
   methodology section once its design is chosen — but the course does not
   resolve which design fits a field experiment specifically; that is
   closer to "experimental" in the empirical-method landscape (§9.1) than to
   "survey," and the course's survey-specific instrument/sampling guidance
   would need adaptation, not direct reuse.
5. **`literature_review_methodology.docx`** describes a scoping review
   already conducted on a CV/crowd-density research question adjacent to
   Sanas, following the course's five-component structure (design,
   databases, search strategy — with full Boolean string and record counts
   — eligibility criteria placeholder, screening funnel, thematic analysis).
   It is unverified (no corroborating log entry, incomplete eligibility
   criteria section, machine-authored) and must not be cited as a completed
   literature review for the Sanas or RTCI paper without independent
   verification of its claimed record counts and included studies.
6. **The limitations-as-a-required-section pattern** (course_notes.md
   §8.9a, reinforced by Mohlaroy's worked example) should be planned into
   the RTCI paper's methodology from the start, with systematic and content
   limitations kept distinct.
7. **Every one of these is contingent on decisions Diyas has not made yet**
   (RQ wording, design type, dataset, split, metric — see next section) —
   the course supplies the *method for making and reporting those choices*,
   not the choices themselves.

## Open questions for Diyas

1. What is the RTCI paper's exact research question, phrased to the
   course's formula and word budget? (Currently open — no RQ text found
   anywhere in this repo.)
2. Is the RTCI paper empirical (field experiment, per its stated causal
   premise) or does it also need a literature-review component analyzing
   existing crowding-information/passenger-behavior research first? The
   course says every paper needs a literature review regardless of route
   (§3) — has that been scoped yet for RTCI?
3. Which empirical-method subtype does the RTCI field experiment map to —
   the course's "experimental" category (§9.1), and if so, what is the
   controlled variable, what is held constant, and what is the outcome
   measure?
4. What is the sampling frame and target sample size for the RTCI field
   experiment, and is it a convenience sample (course-acceptable if
   disclosed) or something more structured?
5. Is `literature_review_methodology.docx` connected to any actual prior
   work session, or is it an unrelated test/generated artifact that should
   be moved out of `research/coursework/` entirely? Its claimed 604/509/40/
   35/19/11-record funnel is currently uncorroborated anywhere else in the
   repo.
6. Does the RTCI paper's eventual methodology section need the course's
   survey-specific six components (population/sample/instrument/etc.), or
   does a field-experiment design need a different component set the course
   doesn't cover (the course's own landscape table names "experimental" but
   only elaborates the survey path in depth)?
7. Should `video1311882563.mp4` be transcribed properly (faster-whisper, as
   the other six were) so its speaker commentary — currently completely
   inaccessible — can be checked against the slide text the way `README.md`
   already flags for the other six recordings?
8. Is the Almaty-bus-occupancy RQ example in the newly-read analysis deck
   this student's own submitted RQ for the course, or a generic instructor
   example — and if it is the student's own, does that create any
   consistency obligation with the actual Sanas/RTCI research question?
9. What acceptance criteria or validation strategy will the RTCI paper use
   to satisfy the course's reproducibility bar (§1.1) — could an independent
   reader repeat the field experiment and expect comparable results?
10. Given `Methodology for emp. papers.pdf`'s truncation at component 5
    (ethics/consent never covered by any source in this directory), has the
    RTCI field experiment's ethics/consent process been defined anywhere
    outside this course material, given it involves real passengers?

## Traceability table

| Conclusion | Source file | Page/slide/timestamp | Status |
|---|---|---|---|
| RQ formula `How/Why does [factor] affect [outcome] among [group/context]?`, 10–15 words | `Foundations class- group 3.pdf`; `1st lesson- part 1/2.mp4` | deck + `course_notes.md` §2.2–2.3 | explicit |
| Research gap = missing/under-examined, not "never studied" | `video1768002125.mp4` / `video1168216779.mp4` | `course_notes.md` §11.4 | explicit (tutor quote) |
| SMART reading order (title→abstract→intro→conclusion→headings→figures→methods/results last) | `2nd class-Reading.pdf` | `course_notes.md` §5.3 | explicit |
| Summary vs synthesis distinction, "single most important habit" | `Lit. review.pdf`; `video1433589796.mp4` | `course_notes.md` §6.4 | explicit |
| Five-component lit-review methodology (design, databases, search strategy, eligibility, analysis) | `Methodology.pdf`; `video1101430871.mp4` | `course_notes.md` §8.2 | explicit |
| Six-component empirical methodology (design, population, sample, instrument, collection, ethics) | `Methodology for emp. papers.pdf` | `course_notes.md` §9.4 | explicit; ethics component's own text is missing from source (p.15/15 cutoff) |
| Association ≠ causation, "X was associated with Y" not "X caused Y" | `video1311882563.mp4`, Step 8 (p.15) | slide text only, no transcript | explicit (slide), speaker commentary unknown |
| RQ-as-filter: only analyze survey variables that answer the RQ | `video1311882563.mp4`, Step 1 (p.8) | slide text only | explicit (slide) |
| Distribution/comparison/relationship as the three analysis questions | `video1311882563.mp4`, Step 5 (p.12) | slide text only | explicit (slide) |
| Null/contradictory results are still valid findings | `video1311882563.mp4`, Steps 9–10 (pp.16–17) | slide text only | explicit (slide) |
| "Do not graph everything," every figure needs a job | `video1311882563.mp4`, Steps 11–12 (pp.18–19) | slide text only | explicit (slide) |
| Worked Almaty-bus-occupancy RQ example may be a generic teaching device, not connected to actual Sanas RQ | `video1311882563.mp4`, throughout | slide text only | inferred — no corroboration found either way |
| Mohlaroy paper as worked model of theme paragraph, absence-as-finding, specific limitations | `Mohlaroy--RAS.docx`; `video1433589796.mp4` ~30:00 | `course_notes.md` §13 | explicit |
| `literature_review_methodology.docx` describes an already-run scoping review on a Sanas-adjacent CV question | `literature_review_methodology.docx` | full document (10 paragraphs) | explicit in the document; **uncorroborated elsewhere in repo** — treat the document's own claims as unverified |
| `literature_review_methodology.docx` is machine-generated, not course material, modified 2026-08-25 | `literature_review_methodology.docx` core properties | docx metadata | explicit (file property) |
| Course teaches survey/lit-review methodology in depth; field-experiment/RTCI-style causal design is only named, not elaborated | `Methodology for emp. papers.pdf` §9.1 list; absence of further detail | `course_notes.md` §9.1 | inferred from absence |
