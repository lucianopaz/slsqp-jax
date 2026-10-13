# Orchestration: checklist-driven session per cause group

Replaces the first draft (kept in `_superseded_draft/`, can be deleted).

One **session** = one Claude Code session that works through `PROTOCOL.md` for one cause
group of the benchmark diagnosis (e.g. `ip_barrier_stall`). There is no outer driver loop:
the checklist, the helper scripts and the ledger carry the state, so every session starts
cold from three small files and costs little.

## Files

| File | Role |
|---|---|
| `PROTOCOL.md` | The checklist, gates and budget rules a session follows. |
| `queue.md` | Ordered list of work items with status. Human-editable. |
| `ledger.jsonl` | Append-only record of every hypothesis tried (accepted and rejected). |
| `hyp.sh` | Git plumbing: worktree + branch per hypothesis, guarded merge, safe cleanup. |
| `ledger.py` | Add/list ledger entries, session lock. |
| `digest.py` | Compact evidence digest of one cause group from `classified.jsonl`. |
| `gap.py` | Head-to-head against the frozen scipy baselines (the objective). |
| `regress_join.py` | Join two benchmark `results.jsonl` files: solved per config, gains, losses. |
| `CONTEXT.md` | Project state (read at session start, updated at session end). |
| `../.cursor/skills/benchmark-diagnostics/SKILL.md` | How the diagnostic tooling is run. |

## Scope

Objective: each sqpdax config should solve more instances than the scipy baselines. Only `slsqp_jax/sqpdax/**` and
`tests/sqpdax/**` are read or changed; the rest of `slsqp_jax/` is legacy. Benchmark problems and scipy baselines are frozen.

## Git model

- Base branch: `local_bench_cleanup` (override with `SQPDAX_BASE`). The main checkout stays on it.
- Each hypothesis gets `hyp/<cause>-<slug>` in `.worktrees/<cause>-<slug>`, branched from the base tip.
- An accepted hypothesis is merged with `--no-ff` into the base branch by `hyp.sh merge`.
- Rejected hypotheses keep their branch (the code of a failed attempt is evidence). Their worktree is removed.
- Never used: `reset --hard`, `clean`, `stash`, `branch -D`, `checkout --`, force push.

## Launching

Order the first time:

1. **Q0 (you):** commit the tooling so that worktrees contain it: `scripts/`, `benchmarks/validate.py`,
   `tests/benchmarks/test_validate.py`, `.cursor/skills/`, `.claude/agents/`, `orchestration/`. Not the `diagnostics/` data
   (`hyp.sh` adds it to `.git/info/exclude`). `hyp.sh new` refuses to start until these are on `local_bench_cleanup`.
2. **Q1 session** (diagnose the newest benchmark run; mostly waiting, about 744 tasks):
   ```
   cd ~/repos/slsqp-jax          # main checkout, branch local_bench_cleanup
   claude --model sonnet
   > Do queue item Q1 in orchestration/queue.md, following .cursor/skills/benchmark-diagnostics/SKILL.md.
   > Output to diagnostics/v4. Scope and constraints: orchestration/PROTOCOL.md "Objective and scope".
   ```
3. **Cause-group sessions**, one at a time (a lock prevents two at once):
   ```
   claude --model sonnet
   > Follow orchestration/PROTOCOL.md for cause group <CAUSE>. MODE=attended. MAX_HYP=2.
   > BASELINE_DIR=diagnostics/v4 RESULTS=results/new_run/results.jsonl
   ```
   or "pick the next open item in orchestration/queue.md". Look at
   `python orchestration/gap.py results/new_run/results.jsonl` first to choose the cause with the largest deficit.

`MODE=attended` pauses at the gates for you. `MODE=unattended` stops at any gate that needs judgement instead of deciding.
A headless call (`claude -p "<same text>" --model sonnet ...`) is possible for unattended runs; check the flags with
`claude --help` first and keep it to one cause group per call.

Models per step are in `PROTOCOL.md` ("Models per step") and in the frontmatter of `.claude/agents/*.md`.

Permissions: the session needs these commands allowed and no others: `orchestration/hyp.sh`,
`python orchestration/{ledger,digest,gap,regress_join}.py`, read-only `git` plus `add`/`commit`/`worktree`/`merge`,
`uv run` (through `hyp.sh run`), `sleep`, `tail`, `grep`, `wc`.
