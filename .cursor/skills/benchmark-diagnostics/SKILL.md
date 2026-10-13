---
name: benchmark-diagnostics
description: >-
  Produce the failure-mode diagnostics report (report.html + classified.jsonl)
  for a sqpdax benchmark run from its results.jsonl, using the scipy rows of
  the same run as the baseline and an earlier classification for the
  fixed/regressed delta. Use when the user has new benchmark results and asks
  to diagnose, classify or re-diagnose failures, redo the diagnostics report,
  compare against scipy, or measure a fix attempt before/after.
---

# Benchmark failure diagnostics

Everything lives in `scripts/diagnose_benchmark.py` (sub-commands `run`,
`report`, `compare`). Input is the benchmark's `results.jsonl` (one row per
problem × starting point × config; configs include the seven sqpdax ones and
the `scipy-slsqp` / `scipy-trust-constr` baselines). A failure is a row with
`solved == false`. Only sqpdax rows are re-run and diagnosed; the scipy rows
are used as the external reference.

## Workflow

```
- [ ] 1. Preserve the previous classification
- [ ] 2. `run`: re-execute the unsolved sqpdax tasks with trajectory capture
- [ ] 3. `report`: classify + render, with `--previous`
- [ ] 4. Verify the numbers quoted in the narrative against classified.jsonl
- [ ] 5. Refresh the narrative in the script (CAUSES / RECOMMENDATIONS /
         FIX_ATTEMPTS / PRIORITIES) and regenerate
- [ ] 6. prek + written summary
```

### 1. Preserve the previous classification

The delta section needs the *old* `classified.jsonl`. Copy it out of the
way before writing a new diagnostics directory (or write the new run into a
fresh directory such as `diagnostics/v2`):

```bash
cp diagnostics/classified.jsonl /tmp/classified_prev.jsonl
```

### 2. Re-run the unsolved tasks

```bash
nohup uv run --no-sync python scripts/diagnose_benchmark.py run results.jsonl \
    --out diagnostics/v2 --jobs 7 --timeout 900 --ref-max-n 0 \
    > diagnostics/v2/run.log 2>&1 &
```

- `--ref-max-n 0` disables the in-process scipy reference solve; the
  baseline now comes from the `scipy-*` rows in `results.jsonl`.
- Budget: ~1300 tasks take ~30 min on 7 jobs. Launch with `nohup … &` and
  poll `diagnostics/v2/run.log`; do not block the shell on it.
- Produces `diag.jsonl` (one row per task: kkt breakdown, feas, slack
  residuals, mu updates, secant stats, status) and `traj/*.json`.
- Restrict with `--configs`, `--problems`, `--statuses`, `--causes`
  (resolved through `--classified`) for targeted re-runs; `--fresh` ignores
  an existing `diag.jsonl`.

### 3. Classify and render

```bash
MPLCONFIGDIR=/tmp/mpl uv run --no-sync python scripts/diagnose_benchmark.py \
    report diagnostics/v2 --results results.jsonl \
    --previous /tmp/classified_prev.jsonl
```

Writes `diagnostics/v2/report.html` and `diagnostics/v2/classified.jsonl`.
`MPLCONFIGDIR` avoids the matplotlib cache warning in the sandbox. Report
sections: 1 overview, 1b scipy baseline comparison (head-to-head, causes by
baseline outcome, instance-level view), 1c fixed/regressed vs `--previous`,
priorities, 2 per-cause evidence + recommendations + fix attempts, 3a/3b
appendices (unsolved instances solved by scipy / by nobody).

### 4. Verify the narrative numbers

The per-cause texts in the script quote counts ("38 of 88 solved by scipy
SLSQP", "45 never update mu"). Check every such number against the new
`classified.jsonl` before publishing. Useful fields per row: `cause`,
`category`, `config`, `status`, `feas`, `kkt_last`, `slack_resid`,
`mu_updates`, `ref_solved` (scipy-slsqp), `tc_solved` (trust-constr),
`baseline_solved`, `f_gap`.

```python
import json, collections
rows = [json.loads(l) for l in open("diagnostics/v2/classified.jsonl")]
g = [r for r in rows if r["cause"] == "ip_barrier_stall"]
print(len(g), collections.Counter(r["config"] for r in g),
      sum(r.get("ref_solved") or 0 for r in g), sum(r.get("tc_solved") or 0 for r in g))
```

### 5. Refresh the narrative

The report text is data in the script, not generated:

| Constant | Content |
|---|---|
| `CAUSES` | `{cause_id: (category, title, description)}`; categories `algorithmic`, `bug`, `parameter`, `benchmark`, `performance` |
| `RECOMMENDATIONS` | `{cause_id: html}` |
| `FIX_ATTEMPTS` | `{cause_id: [{"title", "change", "outcome", "compare_dir"}]}`; `compare_dir` points at a `compare.json` from the `compare` command |
| `PRIORITIES` | ordered `(title, [cause_ids], modules)` |

Update the description/outcome text where the picture changed (counts,
which configs dominate, what scipy solves), re-order `PRIORITIES` by "share
solved by scipy" (deficits first, benchmark artefacts last), then re-run
step 3. Keep descriptions anchored on concrete evidence from `diag.jsonl`
(a problem name, `kkt`, `mu`, `feas` values).

### 6. Finish

```bash
uv run prek run --files scripts/diagnose_benchmark.py
```

Summarise for the user: head-to-head solved counts (per sqpdax config, the
union, scipy-slsqp, trust-constr), the cause table split into solver
deficits (mostly solved by scipy) vs artefacts (solved by nobody), the
fixed/regressed delta with the regressions worth following up, and the
revised priority order.

## Measuring a fix attempt

Re-run only the tasks of the affected causes in a separate directory and
compare against the baseline classification:

```bash
uv run --no-sync python scripts/diagnose_benchmark.py run results.jsonl \
    --out diagnostics/fixN_name --causes false_convergence \
    --classified diagnostics/classified.jsonl --fresh --ref-max-n 0
uv run --no-sync python scripts/diagnose_benchmark.py compare \
    diagnostics diagnostics/fixN_name --results results.jsonl
```

`compare` writes `diagnostics/fixN_name/compare.json`; register the attempt
in `FIX_ATTEMPTS` with `"compare_dir": "fixN_name"` and the report pulls the
before/after numbers (the directory is looked up under the report directory
first, then its parent; a relative path such as `"v3/fix_b_restoration"` is
fine). One attempt dict can be shared by several causes when the same change
was measured on all of them. `WORK_LOG` holds the summary bullets of the
last fix round (rendered before section 2), including the regression check.

To re-run a single task into an existing directory (e.g. after a guard was
added), drop its line from `diag.jsonl` and run with `--problems NAME`
without `--fresh`; `run` skips tasks already present.

Regression check of a fix round (previously solved tasks must stay solved):

```bash
uv run --no-sync python -m benchmarks run --tier tiny \
    --configs asls-craig,asls-pcg,asls-pcg-exact,asls-minresqlp,pasls,trip,tfip \
    --repeats 1 --jobs 4 --timeout 300 --out diagnostics/v3/regress_tiny --flat
```

then join `diagnostics/v3/regress_tiny/results.jsonl` with the baseline
`results.jsonl` on `(problem, y0_iD, config)` and list the tasks whose
`solved` flag went from true to false. Use a fresh `--out` directory per
pass (`regress_tiny2`, ...) and join successive passes with each other as
well: the second join isolates the effect of the last refinement from the
rest of the round. Attribute each loss before acting on it: re-run the task
at the base commit from a worktree (platform noise if it fails there too),
check whether the catalogue `f*` changed (`fstar_source`), and only then
treat it as a code regression.

## Pitfalls

- A bisect that needs the solver at an older commit must run from a git
  worktree (`git worktree add /tmp/x <commit>`, symlink `.venv`, run with
  `PYTHONPATH=/tmp/x` or pytest inside it). `python /tmp/script.py` imports
  the editable install of HEAD, not the worktree. Never use `git stash` for
  this; the user keeps stashes.
- Classification thresholds: feasible means `feas <= 1e-5`; a scipy
  baseline "solves" when its own `solved` flag is set.
- Two runs from different machines flip a handful of tasks near a tolerance;
  mention this when reporting the delta.
- Add new cause ids to `CAUSES` before classifying with them, otherwise the
  report has no title/description for the group.
