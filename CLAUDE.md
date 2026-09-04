# Sanas: agent instructions

## Mandatory boot sequence

Before planning or changing anything:

1. Read `GROUND_TRUTH.md` completely.
2. Run `git status --short --branch`.
3. Treat the working tree as transitional until the current pivot is committed.
4. Label material claims as decided, verified, candidate, open, cancelled or
   history, using the definitions in `GROUND_TRUTH.md`.

`GROUND_TRUTH.md` is the only current-state authority. The append-only
`development/experiments/log.md` is evidence and decision history, not a
current spec: later decisions supersede earlier decisions, while old measured
results remain valid only for the experiment that produced them.

## Current context

Sanas is investigating a ceiling-mounted RGB system that estimates bus cabin
crowding and sends five ordinal levels plus a continuous `0..1` score to
Avtobys (Innoforce). Door APC, VL53L7CX/ESP32 and cabin-wide RealSense designs
are cancelled.

The primary scientific objective is the RTCI field experiment: estimate the
causal effect of displaying real-time crowding in Avtobys on boarding the first
arriving bus versus waiting. The device is an enabling measurement layer, not
the scientific endpoint. Read `research/RTCI_RESEARCH_CHARTER.md` for the
working research design.

The present ceiling RGB direction is not implemented or trained. CSRNet +
PFCASA is a model candidate, not a validated or frozen stack. Jetson Orin Nano
Super Developer Kit and Waveshare IMX219-160 are prototype choices that have
not been bench-tested in this repo. Target semantics, level thresholds,
acceptance metrics, camera geometry, data protocol and the Avtobys API remain
open. Do not let model selection hide those blockers.

There is no own in-cabin camera data yet. RPEE-HEADS and DISCO archives are on
disk. Only a subset/index of Gorelik/BeIntelli is local; the full 73.5 GB
archive is not downloaded. Read `development/datasets.md` and Ground Truth
before using any dataset. Results on public substitute data cannot be reported
as real Sanas bus performance.

## Hard rules

1. Never commit, read or print Kaggle/cloud credentials. They live outside the
   repo and are gitignored.
2. Confirm before spending GPU quota, incurring cost, pushing a kernel,
   uploading data or publishing externally.
3. Never fabricate metrics, dataset statistics, citations, hardware
   compatibility or implementation status.
4. Do not run or revive `run_door_apc.py`, `build_zip_index.py`, the old kernel
   generator or deleted door/depth code unless Diyas explicitly reopens that
   architecture. They are historical/orphaned.
5. Before a model run, define target, dataset version, split, metric, backend
   and acceptance criterion. Start with a small smoke test.
6. Log every real run append-only in `development/experiments/log.md` with
   timestamp, commit hash, data version, config, backend/run id, artifacts and
   result or failure.
7. When a decision changes current state, append the history to the log and
   update `GROUND_TRUTH.md` in the same change. Do not rewrite past log entries.
8. Keep the RTCI causal estimand, its CV measurement system and coursework
   methodologically distinct even though RTCI is the primary scientific track.
   Shared literature does not merge their evidence or deliverables.

## Model and compute decisions

Do not silently lock a backbone, head, loss or backend. For a modeling task:

1. Check whether the target and validation protocol are already defined.
2. Compare 2-3 legal candidates on accuracy, calibration, edge latency,
   memory, low-light behavior and implementation licence.
3. State why the backend fits the run and whether quota/cost approval is
   required.
4. Re-open the choice when own-camera data arrives.

Kaggle scripts exist because Kaggle was previously used; Kaggle is not the
mandatory backend. No dedicated server is recorded as available.

## Remote/mobile workflow

Long jobs must be fire-and-forget and independently checkable. When starting
one, state:

- exact backend and run id;
- how to check status with one short command;
- completion/failure artifact;
- quota or cost consumed.

If approval is required and Diyas is unavailable, report the blocker and wait.

## Tracks and ownership

- Donatello: current RGB data/model/training and reproducibility.
- Danyshpan: RTCI academic writing and verified citations.
- CADy: diagrams, CAD, schematics and plotting code from verified inputs.
- Pitch: product communication using only current Ground Truth and real
  measurements.

Agent role files can become stale. This file and every role defer to
`GROUND_TRUTH.md`. A role must not use a missing script, deleted graph or
cancelled architecture merely because its own prompt mentions one.

## Outreach and citations

SANASH is the outreach and reference track of this project. It is a field study
in Almaty testing whether showing bus passengers a five level real time crowding
level before boarding changes whether they board the bus at the stop or wait for
the next one. Diyas cold emails academics and each email cites one specific
paper by that person. In an earlier round the citations were written from memory
and fourteen of sixty were wrong: a finding stated backwards, papers credited to
co-authors instead of first authors, a research topic invented for someone with
no publications. The reference base under `research/refs/base/` is the repaired
result, checked against live pages. The four rules below exist so that it is
never reconstructed from memory again.

**Rule 1, citations.** Never cite a paper in an outreach email unless it is a
row in `research/refs/base/references.md`. Never state a finding from a paper
unless the finding is recorded there or you have read the paper in this session.
If a paper is missing, search for it, verify it against a live page, and add it
with its DOI and a citation key before using it. Recalling a paper from memory
is the specific failure this project has already suffered.

**Rule 2, salutation.** Emails to academics open `Dear Professor [Surname],` or
`Dear Dr [Surname],`. Never `Hi [first name]`, not in a first email, not in a
reply, not when the person signed with their first name. For a PhD student with
no doctorate, `Dear [Full Name],`. Never guess gender and never use a pronoun
for the recipient.

**Rule 3, no overselling.** An email must not promise a Transportation Research
Part C submission, a launch across 25 cities, or any working device. Say what is
true today: a field study of real time bus crowding information in Almaty, run
with the city bus operator.

**Rule 4, status discipline.** `CITED` means the paper was named in an email
that was sent. `LISTED` means it only sits behind a contact's name. `READ` means
Diyas actually read it, and only Diyas marks `READ`. Whenever you draft an email
citing a paper, flip that row to `CITED (Surname)` in the same commit.

### Corrections that must never be reintroduced

- Xiaopeng Hong is the third author of Bayesian Loss, not its originator.
- The personalised crowding information paper is Jenelius, not Antoniou.
- Daniel Graham uses difference in differences and synthetic control, not
  staggered adoption.
- Ali Jadbabaie works on non-Bayesian social learning, not information cascades.
- GEM is about controllability and depth output, not physical plausibility.
- The train carriage density paper is by Tyler's UCL group but not by Tyler
  himself.
- Allcott and Rogers found that the effect persists. It does not decay after
  novelty.

Run `python research/refs/base/validate.py` after any change to the reference
base. It must exit 0.

## Current directory meaning

- `development/src/`, `development/notebooks/`: empty placeholders for the
  new implementation.
- `development/scripts/`: mixed legacy/support scripts; inspect before use.
- `development/experiments/log.md`: append-only history.
- `development/datasets.md`: dataset provenance.
- `ARCHIVE.md`: dead documents merged into one file on 2026-09-03. Holds the
  former master document, the pre-pivot findings and CV-track conclusions, the
  door/APC related-work notes and the superseded email template. History only,
  never a current spec.
- `research/survey/`: stated-preference survey. Raw export, rebuilt analysis
  dataset, fitted model and outputs. The raw file is the authority on what was
  asked; do not trust the superseded survey docs on `origin/main`.
- `research/refs/rtci-supporting/`: live RTCI literature only.
- `research/wiki/`: agent-maintained research wiki over external sources.
  Read `research/wiki/WIKI_SCHEMA.md` before writing there. It never
  overrides `GROUND_TRUTH.md`.
- `research/EXPERT_CONSULTATIONS.md`: append-only external expert opinions.
  An expert opinion is evidence, not a project decision.
- `research/coursework/`: separate coursework. `WRITING_RULES.md` is the
  operational rule set for paper prose, extracted from the course transcripts
  and decks and quote-audited; proposed, not ratified. `course_notes.md` is the
  full conspectus behind it.
- `business/outreach/`: contacts; current ceiling-device pitch is absent.
  `template.md` is the current cold email template and style rules. The
  superseded `email_template.md` is in `ARCHIVE.md` section 5.
- `research/refs/LITERATURE_DISCOVERY_LOG.md`: ResearchRabbit session log and
  the topic evidence map, merged 2026-09-03. Discovery record, not a citation
  authority; its relevance judgements were made without project context.
- `research/refs/academic_writing_conventions.md`: external deep-research report
  on academic writing conventions. Not verified against primary sources and not
  our rules.
- `research/refs/base/`: the verified SANASH reference base.
  `references.md` is the source of truth, `references.bib` is generated from
  it, `validate.py` checks that the two agree and that no row lost its key,
  link or status.
- `data/`, `outputs/`: local gitignored data and artifacts.

Do not create a new top-level folder unless the current structure genuinely
cannot express the artifact.

## Communication style

- Technical and product discussion in Russian; academic prose in English.
- Short, direct and objective. Separate done, measured, proposed and open.
- No em dash, unnecessary headings or praise.
- When a request would cross an unresolved product decision, show the missing
  decision instead of inventing it.
