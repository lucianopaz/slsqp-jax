# sqpdax benchmarks

Benchmarks of the `slsqp_jax.sqpdax` minimisers on the CUTEst problems shipped
by [sif2jax](https://github.com/thowell/sif2jax) (pure-JAX ports, no Fortran).

```bash
uv sync --all-extras --all-groups          # installs the `benchmark` group

python -m benchmarks run --tier tiny                       # all 7 configs, n <= 20
python -m benchmarks run --tier small --jobs 4 --device cpu
python -m benchmarks run --problems HS71,HS35 --configs pasls,tfip --repeats 5
python -m benchmarks run --tier tiny --configs trip --chunk 0/3   # CI-style shard

python -m benchmarks report results/<run_id>               # static HTML report
python -m benchmarks publish results/<run_id> --results-dir site/results --dashboard
python -m benchmarks render --results-dir site/results     # re-export everything
python -m benchmarks catalog                               # regenerate catalog.csv
```

## Layout

| file | role |
|---|---|
| `catalog.csv`, `catalog.py` | one row per `(problem, y0_iD)` with sizes, collection, size tier (`tiny` ≤ 20, `small` ≤ 100, `medium` ≤ 1000, `large`), reference optimum flags; `select()` filters and shards |
| `problems.py` | `to_sqpdax()` adapts a sif2jax problem (`eq == 0`, `ineq >= 0`) to `sqpdax.Problem` (`g == 0`, `h <= 0`) with exact HVPs enabled |
| `configs.py` | the seven solver configurations: `asls-pcg`, `asls-craig`, `asls-minresqlp`, `asls-pcg-exact`, `pasls`, `trip`, `tfip` |
| `worker.py`, `_bootstrap.py` | persistent worker processes; per task: compile (untimed) → one-step warm-up (untimed) → pilot (timed) → adaptive repeats, with a 5-minute deadline |
| `metrics.py` | solver-independent KKT metrics at the returned point (feasibility, `‖∇ₓL‖`, complementarity, gap to the reference optimum) |
| `results.py` | append-only `results.jsonl` + `run.json`; `analysis.py`: pure-pandas helpers (solved mask, Dolan–Moré profiles, summaries) |
| `report.py` | marimo notebook: per-run report, exported to `report.html` |
| `dashboard.py` | marimo notebook: longitudinal dashboard, exported to WebAssembly as the site's `index.html` |

## Repeats

`repeats = clamp(ceil(2 s / t_pilot), floor, 1000)` with `floor = 3` for solves
under a minute and `1` otherwise; `--repeats N` overrides. Timing excludes
compilation and the warm-up call; the pilot solve counts as the first repeat
and provides the status, solver statistics (`stats_*` columns) and quality
metrics. Repeats are skipped when the pilot did not solve the problem
(`repeats_skipped` records `unsuccessful`, `infeasible`, `wrong_objective` or
`non_finite`; `--repeat-unsolved` forces timing anyway).

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
