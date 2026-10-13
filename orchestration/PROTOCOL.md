# Cause-group session protocol

Work through this checklist top to bottom. Mark each item in the task list. Do not skip a
gate; if a gate fails, record it (step R) and move on or stop as stated.

Notation: `MAIN` = repo root (main checkout, branch `local_bench_cleanup`), `BASE` = that branch,
`H` = `<cause>-<slug>` hypothesis name, `WT` = `MAIN/.worktrees/H`,
`OUT` = `MAIN/diagnostics/v4/<cause>/<slug>` (untracked data, shared by all worktrees).
`hyp` = `orchestration/hyp.sh`.

Parameters (given at start, else defaults): `CAUSE`, `MODE` = attended | unattended (attended),
`MAX_HYP` = 2, `RESULTS` = `results/new_run/results.jsonl`, `BASELINE_DIR` = latest diagnosis
directory holding `classified.jsonl` (see `queue.md`).

## Objective and scope (set by the user, Oct 13)

- **Objective:** raise the success rate of the sqpdax models until each of them solves more benchmark
  instances than the scipy baselines (`scipy-slsqp`, `scipy-trust-constr`). Track it with
  `python orchestration/gap.py RESULTS --overlay OUT/compare.json`: per-config difference to scipy,
  union, and the deficit set (instances scipy solves and no sqpdax config does).
- **Frozen:** problems, catalogue, benchmark code and scipy baselines. Never re-run scipy rows
  (`--ref-max-n 0`; `--configs` lists only sqpdax configs). `RESULTS` is the reference.
- **Code scope:** read and change only `slsqp_jax/sqpdax/**` and its tests `tests/sqpdax/**`.
  Everything else under `slsqp_jax/` (the legacy implementation: `slsqp/`, `inner/`, `qp/`, `lpeca.py`,
  `compat.py`, ...) is out of scope: do not read it for ideas, do not touch it, do not run its tests as a gate.
  The benchmark package, `scripts/` and `orchestration/` are tools: run them, do not change them, except the report
  step R2 (`scripts/diagnose_benchmark.py` narrative constants, on a `hyp/report-*` branch) and the bookkeeping files.
- Causes that are benchmark artefacts (`unbounded`, `reference_mismatch`, `local_minimum_consensus`,
  `problem_infeasible`, `evaluation_error`, `stationarity_scale_f`) are not fixed on the sqpdax side by definition;
  only open them if the digest shows a sqpdax-side defect behind the label.
- `hyp.sh scope <name>` fails if the branch changes anything outside this scope; `hyp.sh merge` runs it.

## Models per step

Main session: **Sonnet** (`claude --model sonnet`). It runs the checklist, the gates, the ledger and git.
Delegated through subagents defined in `.claude/agents/` (model set in each file's frontmatter, change it there):

| Step | Who | Model |
|---|---|---|
| P, D1, gates G0-G3, R (ledger, queue, report text), merge | main session | sonnet |
| D2 tracing of a representative task | main session | sonnet |
| H1 hypothesis design | subagent `hypothesis-designer` | opus (read-only, once per session) |
| I2-I3 implementation and fast checks | subagent `implementer` | sonnet |
| I4, I6, I9 waiting on long jobs | subagent `job-watcher` | haiku |

Optional upgrade for a hard group (for example the interior-point stall): set `hypothesis-designer` to `fable` in its
frontmatter for that session only. Model names are aliases and their availability and cost on your plan are not verified here.
Invoke subagents by name: "Use the hypothesis-designer subagent on ...".

## Reading budget (context hygiene)

Read only: this file, `orchestration/CONTEXT.md`, `orchestration/queue.md`, `ledger.py list --cause CAUSE`,
`AGENTS.md`, the output of `digest.py` and `gap.py`, and source files under `slsqp_jax/sqpdax/` (and `tests/sqpdax/`) that a hypothesis touches.
Never read: `_cursor_export/`, `report.html`, whole `diag.jsonl`/`classified.jsonl`/`results.jsonl`,
whole logs. Use `tail`, `grep`, `python -c`, or `digest.py`. Long jobs are watched by `job-watcher`.

## Never do

`git reset --hard`, `git clean`, `git stash`, `git branch -D`, `git checkout -- <path>`, force push,
edits in `MAIN` other than `hyp.sh merge` and the ledger/queue/context/report files in step R,
commands outside the user's allowed list (if one needs extra permission, skip it and say so),
running a full-tier benchmark (that is a human decision), re-running scipy baselines, editing anything outside `slsqp_jax/sqpdax/**` and `tests/sqpdax/**` (see scope), removing the interior-point slacks,
loosening a solver tolerance to make a task pass, editing the catalogue `f*` to make a task pass.

## Checklist

### P. Preflight
- [ ] P1 `git -C MAIN branch --show-current` is `BASE`; `git status --porcelain --untracked-files=no` is empty (apart from the bookkeeping files `orchestration/{ledger.jsonl,queue.md,CONTEXT.md}`). Else stop and tell the user.
- [ ] P2 `python orchestration/ledger.py lock CAUSE` (fails if another session holds a lock).
- [ ] P3 Read `orchestration/CONTEXT.md` sections 2, 7 and the ledger entries for `CAUSE`. If `queue.md` lists a prerequisite item for `CAUSE`, do that first or stop.
- [ ] P4 `RESULTS` and `BASELINE_DIR/classified.jsonl` exist. A baseline tiny-tier run for the `BASE` tip exists in `MAIN/diagnostics/baselines/tiny_<sha7>/results.jsonl`; if not, note it (created in G2, one-time cost, shared by later sessions).

### D. Evidence
- [ ] D1 `python orchestration/digest.py --classified BASELINE_DIR/classified.jsonl --cause CAUSE > /tmp/digest_CAUSE.md`; read it (about 3k tokens).
- [ ] D2 If the digest does not explain the mechanism, trace 1–2 representative tasks (a small script in `/tmp` that runs one problem/config with logging, as done for HS100/CANTILVR). Limit: 25 tool calls. Write down the observed quantities (mu, radius, predicted reduction, kkt, feas).

### H. Hypotheses
- [ ] H1 Write 1–3 hypotheses to `orchestration/hypotheses/CAUSE.md`, each with: claim; the equation or inequality that should hold if it is true; evidence (problem names, numbers); the change (files, about how many lines); a cheap falsifier; expected solved count in the group; risk to other causes. Prefer the change that is a necessary condition of the others. Respect the "Never do" list and `CONTEXT.md` section 2.
- [ ] H2 **Gate G0.** attended: show the list to the user and wait for a choice and any ordering. unattended: take hypothesis 1 only, and only if it needs no new public API and touches at most three source files; otherwise stop with the hypotheses written down.

### I. Per hypothesis (at most `MAX_HYP`, sequential)
- [ ] I1 `hyp new CAUSE-SLUG` then `hyp check CAUSE-SLUG` (must print a `slsqp_jax` path inside `WT`). All commands below run through `hyp run H -- <cmd>` so that the worktree code is imported.
- [ ] I2 Implement the smallest change that tests the claim, inside `slsqp_jax/sqpdax/` (tests in `tests/sqpdax/`). Follow `AGENTS.md` (numpy-style markdown docstrings, doctests, pytest conventions, tests under the right sub-package). Add or adapt tests for the changed behaviour.
- [ ] I3 Fast checks: `hyp run H -- pytest <touched test files> -q -p no:cacheprovider --no-cov`; `hyp run H -- prek run --files <changed files>`. Fix before measuring. Commit on the hypothesis branch (message: why, then what).
- [ ] I4 Targeted measurement (only the group, never the whole benchmark):
  `hyp run H -- python scripts/diagnose_benchmark.py run RESULTS --out OUT --causes CAUSE --classified BASELINE_DIR/classified.jsonl --fresh --ref-max-n 0 --jobs 4 --timeout 600 > OUT.log 2>&1 &`
  then `hyp run H -- python scripts/diagnose_benchmark.py compare BASELINE_DIR OUT --results RESULTS`.
  Poll the log no more than once every 5–10 minutes with a single `sleep N; tail -2` command.
- [ ] I5 **Gate G1** (cheap kill). Let `N` = tasks in the group, `S` = solved after. Pass iff `S >= max(2, ceil(0.03 N))`. Also read the `after` breakdown: note which new diagnoses appeared. Also run `python orchestration/gap.py RESULTS --overlay OUT/compare.json` and note the change in the per-config difference to scipy-slsqp and in the deficit set. Fail → go to R (decision `rejected`), then next hypothesis.
- [ ] I6 Regression check (tiny tier). Configs: the changed family only (`asls-craig,asls-pcg,asls-pcg-exact,asls-minresqlp,pasls` or `trip,tfip`); all seven sqpdax configs if the diff touches shared code (`minimiser/base.py`, `step_controller/`, `linalg/`, `subproblem/`, `barrier/`). Never include `scipy-*` configs.
  `hyp run H -- python -m benchmarks run --tier tiny --configs CFGS --repeats 1 --jobs 4 --timeout 300 --out OUT/regress_tiny --flat > OUT/regress_tiny.log 2>&1 &`
  Baseline: `MAIN/diagnostics/baselines/tiny_<sha7>/` made the same way from a `hyp base` worktree if it does not exist.
  `python orchestration/regress_join.py BASELINE_TINY OUT/regress_tiny/results.jsonl --out OUT/regress_join.json`.
- [ ] I7 Attribute every lost task: re-run only those tasks in a base worktree (`hyp base`; `hyp run base-<sha7> -- python -m benchmarks run --tier tiny --problems P1,P2 --configs C --repeats 1 --out ...`). Fails there too → `noise` (platform/machine). `f*` or the catalogue label differs → `catalogue`. Otherwise `regression`.
- [ ] I8 **Gate G2.** Pass iff there is no task labelled `regression`. attended: the user may waive a regression if the group gain is at least three times the regressions; record the waiver. unattended: no waiver.
- [ ] I9 **Gate G3** (only for a candidate that passed G1–G2): non-slow sqpdax suite in the worktree `hyp run H -- pytest tests/sqpdax -m "not slow" -q -p no:cacheprovider --no-cov > OUT/pytest.log 2>&1 &` (legacy tests outside `tests/sqpdax` are not a gate), and `hyp run H -- prek run --files <changed files>`. A failing test that also fails at `BASE` is `pre-existing`: list it, do not fix it here.
- [ ] I10 Record the outcome (step R1) for this hypothesis before starting the next one.
- [ ] I11 If accepted: `hyp merge H` (guarded: clean main checkout on `BASE`, `--no-ff`, aborts and reports on conflict). Later hypotheses branch from the new tip; their baseline for G1 is still `BASELINE_DIR`, but G2 uses a baseline taken at the new tip.
- [ ] I12 `hyp drop H` (removes the worktree, keeps the branch).

### R. Record (after each hypothesis, and once at the end)
- [ ] R1 `python orchestration/ledger.py add ...` with: id, cause, title, branch, commit, baseline_dir, compare_dir, tasks, solved, regress_lost (by label), decision (`accepted` | `rejected` | `inconclusive`), notes (what was tried, what the numbers say, what to try next). Rejected attempts are recorded as carefully as accepted ones.
- [ ] R2 At the end of the session, in a worktree `hyp new report-CAUSE-YYYYMMDD`: add one `FIX_ATTEMPTS` entry per ledger entry of this session in `scripts/diagnose_benchmark.py` (title, change, outcome with the measured numbers, `compare_dir`), add a `WORK_LOG` bullet if a round is complete, update any counts the narrative quotes. Run `prek run --files scripts/diagnose_benchmark.py`. Regenerate `report.html` only if a new classification exists. Merge with `hyp merge`.
- [ ] R3 Update `orchestration/CONTEXT.md` (sections 4–7: round table, current numbers, next steps) and `orchestration/queue.md` (status of the item, new items discovered). Keep both short. Then commit exactly those three bookkeeping files (and nothing else) directly on `BASE` in the main checkout.
- [ ] R4 `python orchestration/ledger.py unlock CAUSE`. Final message to the user: per hypothesis, one line `H: group solved S/N, regressions k, decision`; the branch names; what is next.

## Stop and ask (any mode)

- Gate failure that is not covered above, merge conflict, a test that fails at `BASE` but passes in the worktree, a task that changes from fail to false-success (solver says `successful`, `solved` false, and it solved before), or a needed command outside the allowed list.
- More than 60 tool calls on one implementation, or three consecutive failed attempts at the same step.
- A hypothesis that requires deleting slacks, loosening a tolerance, or changing a catalogue reference.

## Defaults a user may change

G1 threshold `max(2, ceil(0.03 N))`; `MAX_HYP = 2`; tiny-tier regression only on the changed family; poll interval 5–10 min.
