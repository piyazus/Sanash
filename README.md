# Setup

1. Unzip this as your project root and `cd` into it. Run `git init` if you
   want version control (recommended — CLAUDE.md rules assume commits).
2. `pip install kaggle --break-system-packages`
3. Get `kaggle.json` from kaggle.com/settings → API → Create New Token,
   place it yourself at `~/.kaggle/kaggle.json` (chmod 600). Do this
   outside Claude Code — don't paste the token into chat.
4. Open the project in Claude Code (`claude` in this directory, or through
   the mobile app for remote sessions).
5. Run `bash scripts/kaggle_setup.sh` to confirm auth works.

## What's here
- `CLAUDE.md` — project context and rules, loads automatically every session
- `.claude/skills/research-writer/` — academic writing rules (IEEE, citations)
- `.claude/skills/research-trainer/` — training/backend rules
- `.claude/skills/research-diagrams/` — diagram/figure rules (diagrams-as-code,
  no fabricated data points)
- `.claude/agents/danyshpan.md`, `.claude/agents/donatello.md`,
  `.claude/agents/cady.md` — subagents with isolated context for each track
- `.claude/settings.json` — permission rules (safe reads auto-allowed,
  destructive/costly actions ask or deny)
- `scripts/` — Kaggle setup/download/push helpers
- `experiments/log.md` — append-only run log, source of truth for what was
  tried and what worked

## Day to day
- For paper work: `/research-writer` then describe the task, or just start
  writing about the paper and Claude will pick the skill up automatically.
- For training work: `/research-trainer` then describe the task.
- For diagrams and paper figures: `/research-diagrams` then describe the task.
- For a genuinely separate context, name the subagent (e.g. train in
  background while writing in the main session):
  - "use donatello to start a training run on a 5% subset" — CV/training
    track, runs with its own context window.
  - "use danyshpan to draft the intro section" — paper track, no training
    context bleeding into it.
  - "use cady to diagram the edge inference pipeline" — diagrams and paper
    figures, built from Donatello's log or Danyshpan's verified sources
    rather than invented on the spot.
- Checking on a long job from your phone: ask "what's the status of the
  training run" — the donatello role is set up to make this answerable
  without reconstructing context.
