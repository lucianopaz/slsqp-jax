# sqpdax benchmarks

Benchmarks of the `slsqp_jax.sqpdax` minimisers on the CUTEst problems shipped
by [sif2jax](https://github.com/thowell/sif2jax) (pure-JAX ports, no Fortran),
alongside two SciPy baselines (`scipy-slsqp`, `scipy-trust-constr`) that solve
the identical problems through `scipy.optimize.minimize`.

```bash
uv sync --all-extras --all-groups          # installs the `benchmark` group

python -m benchmarks run --tier tiny                       # all 9 configs, n <= 20
python -m benchmarks run --tier small --jobs 4 --device cpu
python -m benchmarks run --problems HS71,HS35 --configs pasls,tfip --repeats 5
python -m benchmarks run --tier tiny --configs trip --chunk 0/3   # CI-style shard
python -m benchmarks run --tier tiny --configs scipy-slsqp,scipy-trust-constr  # baselines only

python -m benchmarks report results/<run_id>               # static HTML report
python -m benchmarks publish results/<run_id> --results-dir site/results --dashboard
python -m benchmarks render --results-dir site/results     # re-export everything
python -m benchmarks rescore in.jsonl out.jsonl --stat-tol 1e-3   # re-evaluate `solved` of stored rows
python -m benchmarks catalog                               # regenerate catalog.csv
python -m benchmarks validate --jobs 4                     # check signs / f* against scipy
```

## Layout

| file | role |
|---|---|
| `catalog.csv`, `catalog.py` | one row per `(problem, y0_iD)` with sizes, collection, size tier (`tiny` ≤ 20, `small` ≤ 100, `medium` ≤ 1000, `large`), reference optimum flags and the validation columns below; `select()` filters and shards (and drops `valid == False` instances unless `include_invalid=True`) |
| `validate.py` | `python -m benchmarks validate`: solves every instance with scipy SLSQP under the candidate inequality sign conventions (documented `ineq >= 0`, the rows the reference point violates, all rows flipped) and with trust-constr on the chosen one; records `ineq_flip` (rows converted as `h = c` instead of `h = -c`), the scoring reference `fstar` with its `fstar_source` (`sif2jax` or `scipy-consensus`), `fstar_sif`, `valid` and a `validation` label (`ok`, `sign_flipped`, `fstar_from_consensus`, `fstar_mismatch`, `unverified`, `unbounded`, `infeasible`, `evaluation_error`, `error`, `timeout`); `catalog` carries these columns over |
| `problems.py` | `to_sqpdax()` adapts a sif2jax problem (`eq == 0`, `ineq >= 0`) to `sqpdax.Problem` (`g == 0`, `h <= 0`) with exact HVPs enabled, applying the catalogue's validated `ineq_flip` / `fstar` overrides |
| `configs.py` | the nine configurations: seven sqpdax ones (`asls-pcg`, `asls-craig`, `asls-minresqlp`, `asls-pcg-exact`, `pasls`, `trip`, `tfip`) and two SciPy baselines (`scipy-slsqp`, `scipy-trust-constr`); each carries a `backend` tag |
| `runners.py` | backend-neutral `SolveOutcome` plus `SqpdaxRunner` (one jitted `minimise` executable with a traced step budget) |
| `baselines.py` | `ScipyRunner`: feeds jitted JAX callbacks (objective, gradient, constraints, Jacobians, HVPs) to `scipy.optimize.minimize`; maps SciPy statuses onto the harness vocabulary and translates SciPy multipliers into the sqpdax convention so the KKT metrics are comparable |
| `worker.py`, `_bootstrap.py` | persistent worker processes; per task: compile (untimed) → one-step warm-up (untimed) → pilot (timed) → adaptive repeats, with a 5-minute deadline |
| `criteria.py` | `solved_reason(row, feas_tol, f_tol, stat_tol)`: the pure *solved* rule shared by the worker, `python -m benchmarks rescore` and `scripts/diagnose_benchmark.py` (see below) |
| `metrics.py` | solver-independent KKT metrics at the returned point (feasibility, `‖∇ₓL‖`, complementarity, gap to the reference optimum) |
| `results.py` | append-only `results.jsonl` + `run.json`; `analysis.py`: pure-pandas helpers (solved mask, Dolan–Moré profiles, summaries) |
| `report.py` | marimo notebook: per-run report, exported to `report.html` |
| `dashboard.py` | marimo notebook: longitudinal dashboard, exported to WebAssembly as the site's `index.html` |

## What counts as solved

A task is **solved** when, checked in this order, the point is finite
(`non_finite`), the minimiser reported success (`unsuccessful`), the maximum
constraint violation is at most `--feas-tol`, default 1e-6 (`infeasible`), the
objective matches the reference optimum, when one exists, to a relative gap of
at most `--f-tol`, default 1e-4 (`wrong_objective`), *and* the point is
stationary (`non_stationary`): `stat_scaled <= --stat-tol`, default 1e-4.
`stat_scaled` is the IPOPT-scaled inf-norm of the Lagrangian gradient,
`‖∇ₓL‖∞ / max(1, ‖∇f‖∞, ‖multipliers‖∞)`, evaluated with the multipliers the
solver returned (`metrics.quality_metrics`). A solver that returned no
multipliers has `stat_scaled = NaN`; the check then passes (benefit of the
doubt). The tolerances are stored in every result row. In the analysis layer
the first three failures and `non_stationary` become the outcomes
`claimed_but_infeasible`, `claimed_but_wrong` and `claimed_but_non_stationary`
for reported successes. `python -m benchmarks rescore IN.jsonl OUT.jsonl`
re-evaluates the flag of an existing results file (it adds `solved_reason` and
the tolerances and prints the solved counts per configuration before and
after).

## SciPy baselines

`scipy-slsqp` (BFGS secant, `ftol = 1e-6`) and `scipy-trust-constr` (exact
Hessian-vector products from the JAX problem, `gtol = 1e-6`) run through the
same worker phases: *compile* evaluates every jitted callback once, *warm-up*
is one `maxiter = 1` call, and the timed solves pass `--max-steps` as
`maxiter`. `status` is `successful` / `max_steps_reached` or a slug of the
SciPy message; `steps` is `nit` and SciPy's counters appear as `stats_nfev`,
`stats_njev`, `stats_nhev`, ... Reported time includes the Python overhead of
the SciPy drivers. SLSQP does not return bound multipliers; they are
recovered from the stationarity residual on the active bounds before the KKT
metrics are evaluated. trust-constr reports success once the Lagrangian
gradient drops below `gtol` regardless of the remaining barrier parameter, so
its returned points can sit O(μ) inside active bounds; such solves show up as
`claimed_but_wrong` when the objective gap exceeds `--f-tol`.

## Repeats

`repeats = clamp(ceil(2 s / t_pilot), floor, 1000)` with `floor = 3` for solves
under a minute and `1` otherwise; `--repeats N` overrides. Timing excludes
compilation and the warm-up call; the pilot solve counts as the first repeat
and provides the status, solver statistics (`stats_*` columns) and quality
metrics. Repeats are skipped when the pilot did not solve the problem
(`repeats_skipped` records `non_finite`, `unsuccessful`, `infeasible`,
`wrong_objective` or `non_stationary`; `--repeat-unsolved` forces timing
anyway).

## Published results

`.github/workflows/benchmark.yml` runs the `tiny` tier after every successful
release and weekly when code under `slsqp_jax/`, `pyproject.toml`, `uv.lock`
or `benchmarks/` changed. Repository admins can also start a run (or a
render-only pass) from the Actions tab via "Run workflow"; the gate job
rejects manual triggers from anyone below admin. Results land on the
`benchmark-results` branch:

```
index.html                 interactive dashboard (marimo + Pyodide)
runs.html                  static list of runs linking to the reports
results/index.json         manifest of runs
results/<ts>_<sha>/results.jsonl, run.json, summary.json, report.html
```

Serve the branch with GitHub Pages (classic "deploy from branch", root).
Local or GPU runs publish into the same tree with `python -m benchmarks publish`;
the row's `device` column lets the dashboard separate them.
