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

## Current directory meaning

- `development/src/`, `development/notebooks/`: empty placeholders for the
  new implementation.
- `development/scripts/`: mixed legacy/support scripts; inspect before use.
- `development/experiments/log.md`: append-only history.
- `development/datasets.md`: dataset provenance.
- `development/findings.md`, `development/sanas_cv_track_conclusions.md`:
  historical findings, not current architecture.
- `research/refs/rtci-supporting/`: live RTCI literature only.
- `research/paper/related_work_notes.md`: retained door/APC history, not a
  current paper draft.
- `research/coursework/`: separate coursework.
- `business/outreach/`: contacts; current ceiling-device pitch is absent.
- `data/`, `outputs/`: local gitignored data and artifacts.

Do not create a new top-level folder unless the current structure genuinely
cannot express the artifact.

## Communication style

- Technical and product discussion in Russian; academic prose in English.
- Short, direct and objective. Separate done, measured, proposed and open.
- No em dash, unnecessary headings or praise.
- When a request would cross an unresolved product decision, show the missing
  decision instead of inventing it.
