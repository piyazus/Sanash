---
name: research-trainer
description: Use when starting, monitoring, or managing CV model training runs for the Sanas occupancy-estimation model — backend selection (Kaggle/Colab/cloud GPU), dataset prep, kicking off training, checking run status, or pulling results. Invoke with /research-trainer or let Claude pick it up automatically for training-related tasks in this repo.
---

# Research Trainer

## Scope
CV/training track only (occupancy density model). Not for the academic
paper — use research-writer for that.

Read `GROUND_TRUTH.md` first. The current ceiling RGB pipeline is not built;
CSRNet + PFCASA is a candidate, not a frozen stack. Do not revive cancelled
door/depth code.

## Before starting any run
1. State which backend you're using and why (quota left, job size, whether
   interactive access is needed). Don't default to Kaggle without saying so.
2. Propose 2-3 model candidates with tradeoffs if the architecture isn't
   already decided for this run — don't silently reuse a prior choice
   without confirming it still makes sense for the current data.
3. Run a smoke test first: 1 epoch, small data subset, before a full run.
4. If GPU quota is limited or the run will cost money, confirm before
   starting.

## During/after a run
- This should be fire-and-forget: start the job, detach, tell me exactly
  how to check status later (what command, what file, what to expect when
  it's done). I may check from my phone without a live terminal.
- Log to development/experiments/log.md on completion: commit hash, backend/kernel id,
  config (architecture, epochs, batch size, data subset), result metrics,
  notes for next run. Append only, never overwrite past entries.
- Don't fabricate metrics if a run fails or is incomplete — report what
  actually happened.

## Compute backend notes
- Kaggle: no SSH, submit-and-poll via `kaggle` CLI (development/scripts/). Free tier
  ~30 GPU-hrs/week, 12h session cap.
- Colab: similar quota constraints, different auth flow.
- Rented GPU (RunPod/Lambda/etc): costs money per hour, needs explicit
  confirmation before spinning up.
- Pick based on what's actually available and cheap for the job size, not
  out of habit.

## Data
The full 73.5 GB Gorelik archive is not local; only an index and a subset from
the cancelled APC experiment are present. RPEE-HEADS and DISCO archives are
local but have major domain/licence constraints. Read `development/datasets.md`
and define current-use eligibility before building a loader. Substitute-data
results are pipeline checks, not claims about real Sanas bus performance.
