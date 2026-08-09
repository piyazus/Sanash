# Sanas — Bus Occupancy CV Project

## Context
Building an edge CV system (Sanas) that estimates bus cabin occupancy density
(0-1 continuous score, 5 ordinal levels) from RGB cameras, integrated into
the Avtobys transit app (Innoforce). No own in-cabin camera data yet
(municipal approval pending). Using public multi-view in-cabin dataset
(Gorelik et al., Zenodo 10.5281/zenodo.20559664, 73.5GB, CC-BY 4.0,
RGB+depth, 4 cameras, German urban bus, nuScenes format) as substitute data
to build and validate the pipeline before real data arrives.

Parallel academic track: causal field-experiment paper on whether real-time
crowding info changes boarding decisions. Keep the two tracks separate in
commits/logs but both can pull from the same lit review.

## Model stack — not fixed
Don't lock a backbone/head/loss into this file. Approaches differ across
labs and the right choice depends on what the data and constraints look
like at the time. When starting a modeling task:
1. Pull current options (DINOv2, ConvNeXt, CSRNet-style, others) and compare
   on: accuracy for ordinal density estimation, inference latency on edge
   hardware, robustness to low light, license.
2. Propose 2-3 candidates with tradeoffs before committing to a training run.
3. Log the choice + reasoning in experiments/log.md, not in this file —
   this file is context, not a frozen spec.
4. Re-open the choice once real camera data arrives; substitute-data results
   may not transfer.

## Compute — not fixed to one platform
No dedicated server yet. Don't assume Kaggle is the only backend — it might
be Kaggle, Colab, or a rented GPU box (RunPod/Lambda/etc), decided per task
based on cost, quota left, and job length. Concretely:
- Before starting a training run, state which backend you're using and why
  (quota remaining, job size, whether interactive access is needed).
- Don't build tooling that only works on one backend. scripts/ has a Kaggle
  path today because that's what exists; treat it as one option, not the
  default assumption.
- If free-tier quota (Kaggle ~30 GPU-hrs/week, Colab similar) is tight for a
  planned run, say so and ask before burning it, rather than silently
  switching to a paid option.

## Scientific/research skill stack needed
- **Literature search & verification** — Scite (connected) for checking claims
  and finding related work; never fabricate a citation's volume/issue/date —
  flag uncertainty and mark confirmed-from-full-text vs inferred-from-abstract.
- **Dataset discovery** — Zenodo, arXiv, HuggingFace datasets, Kaggle datasets,
  paper GitHub repos.
- **Experiment tracking** — needs a connector once training starts for real
  (W&B or MLflow) so runs are comparable, not just logged as text.
- **Reproducibility** — pin dependency versions, log commit hash + data
  version + config per run in experiments/log.md.
- **Statistical/causal methods** — for the field-experiment paper track:
  power analysis, randomization design, standard causal inference checks.
- **Citation formatting** — IEEE style for the academic paper track.
- **Google Drive/Docs** (connected) — for pulling/writing paper drafts and
  shared lit review docs.

## Remote / mobile workflow (primary requirement)
I need to be able to kick things off and check on them from my phone when
I don't have laptop access, without babysitting a live terminal. This means:
1. Prefer **fire-and-forget jobs** over interactive sessions: start a
   training run, detach, and let me check status later with a short command
   ("what's the status of the training run") rather than requiring me to
   watch output live.
2. Every long-running action (train, download, kernel push) should be
   checkable with one short status command — don't make me reconstruct
   context to ask "did it finish."
3. State clearly, right when a job starts, how I check on it and how I'll
   know it's done (what file, what message, what command).
4. If something needs my approval (spending GPU quota, pushing a kernel,
   downloading a large file) and I'm not around to confirm quickly, say
   what's blocked and wait — don't assume and don't silently skip it either.
5. Keep responses short by default; expand with numbered points only when
   I ask or when a decision needs laying out.

## Rules for Claude Code in this repo
1. Never commit or print Kaggle/cloud API credentials. They live outside the
   repo (e.g. `~/.kaggle/kaggle.json`), already gitignored.
2. Confirm before anything that spends GPU quota, costs money, or pushes
   data externally (kernel push, cloud instance launch, publishing anything).
3. Log every experiment run (config, commit hash, backend/kernel id, result
   metrics) to `experiments/log.md` — append, never overwrite.
4. Don't fabricate dataset statistics, benchmark numbers, or citations.
5. Prefer a small smoke-test run (1 epoch, data subset) before a full run,
   on whichever backend is in use.

## Roles: writer vs trainer
Two focused roles exist so context doesn't blur between tracks:
- `/research-writer` (skill) or the `writer` subagent — academic paper track
- `/research-trainer` (skill) or the `trainer` subagent — CV/training track
Details for each live in `.claude/skills/<name>/SKILL.md`, not here. Use the
subagents (`.claude/agents/`) when you want a genuinely separate context
window (e.g. running a long training job while also drafting the intro);
use the skill/slash-command form when you just want this same session to
focus on one track.

## Directory layout
- `data/` — raw/processed data (gitignored, too large for git)
- `scripts/` — backend setup/run scripts (Kaggle today, add others as used)
- `src/` — model code, data loaders, training loop
- `notebooks/` — kernel/notebook sources actually pushed to a backend
- `experiments/log.md` — append-only run log, source of truth for model
  choices and results (not this file)

## Style
- Short, direct, objective. Say if something's not possible.
- Technical/product discussion in Russian, academic writing in English.
- No em dashes, no unnecessary colons, no flattery.
- Numbered points when expanding on something, not by default.
