"""Longitudinal benchmark dashboard (marimo notebook, exported to WebAssembly).

The exported notebook runs in the browser (Pyodide), so this file only uses
``pandas``, ``altair`` and the standard library, and loads
``benchmarks/analysis.py`` from the published site when the package is not
importable.

Run locally against a results tree::

    uv run marimo run benchmarks/dashboard.py -- --results-dir results

Export (done by ``python -m benchmarks dashboard``)::

    uv run marimo export html-wasm benchmarks/dashboard.py -o site --mode run
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="sqpdax benchmarks")


@app.cell
def _():
    import json
    import sys
    import types
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import pandas as pd

    alt.data_transformers.disable_max_rows()

    def fetch_text(location) -> str:
        """Read a local file or (under Pyodide) a URL relative to the site."""
        target = str(location)
        if target.startswith(("http://", "https://")):
            try:
                from pyodide.http import open_url  # type: ignore[import-not-found]

                return open_url(target).read()
            except ImportError:
                import urllib.request

                with urllib.request.urlopen(target) as response:
                    return response.read().decode()
        return Path(target).read_text()

    def read_jsonl_text(text: str) -> pd.DataFrame:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return pd.DataFrame(rows)

    return Path, alt, fetch_text, json, mo, pd, read_jsonl_text, sys, types


@app.cell
def _(Path, fetch_text, mo, sys, types):
    _arg = mo.cli_args().get("results-dir")
    if _arg:
        results_loc = Path(_arg).resolve()
        site_loc = results_loc.parent
    else:
        site_loc = mo.notebook_location()
        results_loc = site_loc / "results"

    try:
        _repo = Path(__file__).resolve().parent.parent
        if str(_repo) not in sys.path:
            sys.path.insert(0, str(_repo))
        from benchmarks import analysis
    except Exception:
        try:
            # marimo's wasm export bundles local modules as a top-level wheel.
            import analysis  # type: ignore[no-redef]
        except Exception:  # Fallback: fetch the helper module from the site.
            analysis = types.ModuleType("analysis")
            exec(fetch_text(site_loc / "public" / "analysis.py"), analysis.__dict__)
    return analysis, results_loc


@app.cell
def _(fetch_text, json, mo, pd, results_loc):
    try:
        index = pd.DataFrame(json.loads(fetch_text(results_loc / "index.json")))
    except Exception as _exc:  # noqa: BLE001
        index = pd.DataFrame()
        mo.stop(
            True, mo.md(f"No `results/index.json` found at `{results_loc}` ({_exc}).")
        )
    index = index.sort_values("run_id").reset_index(drop=True)
    index["timestamp"] = pd.to_datetime(index["timestamp"], utc=True, errors="coerce")
    index["short_sha"] = index["sha"].astype(str).str[:7]
    return (index,)


@app.cell
def _(index, mo):
    mo.md(
        f"""# sqpdax benchmarks

{len(index)} published runs, latest `{index["run_id"].iloc[-1]}`. Each run benchmarks
the sqpdax minimiser configurations on the sif2jax (CUTEst) catalogue; see the
[list of runs](runs.html) for links to the detailed per-run reports.

A task counts as **solved** when the minimiser reports success *and* the returned point
is feasible and, when a reference optimum is known, matches it (tolerances 1e-6 and
1e-4 respectively; the per-run reports let you vary them)."""
    )
    return


@app.cell
def _(fetch_text, index, json, pd, results_loc):
    _frames = []
    for _rec in index.to_dict("records"):
        try:
            _summary = json.loads(
                fetch_text(results_loc / _rec["run_id"] / "summary.json")
            )
        except Exception:  # noqa: BLE001
            continue
        _frame = pd.DataFrame(_summary.get("by_config", []))
        _frame["run_id"] = _rec["run_id"]
        _frames.append(_frame)
    trend = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame()
    if not trend.empty:
        _meta = index.set_index("run_id")
        for _col in ("timestamp", "short_sha", "tag", "device", "tier", "trigger"):
            trend[_col] = trend["run_id"].map(_meta[_col])
    return (trend,)


@app.cell
def _(alt, mo, trend):
    mo.stop(
        trend.empty,
        mo.md("No `summary.json` files found; nothing to plot longitudinally."),
    )
    _tooltip = [
        "run_id",
        "short_sha",
        "tag",
        "config",
        "tier",
        "device",
        alt.Tooltip("solve_rate:Q", format=".3f"),
        "n_solved",
        "n_tasks",
        alt.Tooltip("median_time_s:Q", format=".3e"),
    ]
    _base = alt.Chart(trend).encode(
        x=alt.X("timestamp:T", title="run timestamp (UTC)"),
        color=alt.Color("config:N", title="configuration"),
        tooltip=_tooltip,
    )
    _rate = (
        _base.mark_line(point=True)
        .encode(
            y=alt.Y("solve_rate:Q", scale=alt.Scale(domain=[0, 1]), title="solve rate")
        )
        .properties(
            width=700, height=280, title="Solve rate per configuration over time"
        )
    )
    _time = (
        _base.mark_line(point=True)
        .encode(
            y=alt.Y(
                "median_time_s:Q",
                scale=alt.Scale(type="log"),
                title="median solve time over solved tasks [s] (log)",
            )
        )
        .properties(
            width=700, height=280, title="Median solve time per configuration over time"
        )
    )
    mo.vstack(
        [
            mo.md("## Trends across runs"),
            mo.ui.altair_chart(_rate),
            mo.md(
                """**Figure 1.** Fraction of tasks solved by each configuration (y) for every
published run, ordered by run timestamp (x). Runs may differ in tier or hardware - hover a
point to see the run id, commit, tag, tier and device before comparing two points."""
            ),
            mo.ui.altair_chart(_time),
            mo.md(
                """**Figure 2.** Median wall time of one full solve, over the tasks each
configuration solved (y, log), per run (x). Because the median is taken over *solved*
tasks only, a configuration that starts solving harder problems can legitimately get
slower here while improving in Figure 1."""
            ),
        ]
    )
    return


@app.cell
def _(index, mo):
    run_picker = mo.ui.dropdown(
        options=list(index["run_id"].iloc[::-1]),
        value=index["run_id"].iloc[-1],
        label="run",
        searchable=True,
    )
    mo.vstack([mo.md("## Run drill-down"), run_picker])
    return (run_picker,)


@app.cell
def _(analysis, fetch_text, read_jsonl_text, results_loc, run_picker):
    run_rows = analysis.add_solved(
        read_jsonl_text(fetch_text(results_loc / run_picker.value / "results.jsonl"))
    )
    return (run_rows,)


@app.cell
def _(alt, analysis, index, mo, run_picker, run_rows):
    _meta = index.set_index("run_id").loc[run_picker.value]
    _outcomes = analysis.outcome_table(run_rows)
    _bar = (
        alt.Chart(_outcomes)
        .mark_bar()
        .encode(
            x=alt.X("config:N", title="solver configuration"),
            y=alt.Y("count:Q", title="number of tasks"),
            color=alt.Color("outcome:N", title="outcome"),
            order=alt.Order("outcome:N"),
            tooltip=["config", "outcome", "count"],
        )
        .properties(width=420, height=300, title="Outcome per configuration")
    )
    _prof = analysis.performance_profile(run_rows, metric="time_median_s")
    _profile = (
        alt.Chart(_prof)
        .mark_line(interpolate="step-after")
        .encode(
            x=alt.X(
                "tau:Q", scale=alt.Scale(type="log"), title="performance ratio τ (log)"
            ),
            y=alt.Y(
                "rho:Q",
                scale=alt.Scale(domain=[0, 1]),
                title="fraction solved within τ × best",
            ),
            color=alt.Color("config:N", title="configuration"),
            tooltip=[
                "config",
                alt.Tooltip("tau:Q", format=".2f"),
                alt.Tooltip("rho:Q", format=".3f"),
            ],
        )
        .properties(width=420, height=300, title="Performance profile (wall time)")
    )
    mo.vstack(
        [
            mo.md(
                f"Run `{run_picker.value}` - commit `{_meta['short_sha']}`"
                + (
                    f", tag `{_meta['tag']}`"
                    if isinstance(_meta.get("tag"), str)
                    else ""
                )
                + f", tier `{_meta.get('tier')}`, device `{_meta.get('device')}`, {len(run_rows)} tasks. "
                f"[Open the detailed report](results/{run_picker.value}/report.html) · "
                f"[raw results.jsonl](results/{run_picker.value}/results.jsonl)"
            ),
            mo.hstack([mo.ui.altair_chart(_bar), mo.ui.altair_chart(_profile)]),
            mo.md(
                """**Figure 3.** Left: tasks per configuration stacked by outcome (`solved`,
reported successes that are infeasible or off the reference optimum, step-budget hits,
other solver failures, harness timeouts/errors). Right: Dolan-Moré profile of median wall
time - the fraction of instances (y) each configuration solved within τ times the fastest
configuration's time (x, log); the height at τ = 1 is how often it was the fastest and the
right plateau is its solve rate."""
            ),
            mo.ui.table(analysis.summary_table(run_rows).round(6), selection=None),
            mo.md(
                "**Table 1.** Solve counts and median time / iterations over solved tasks for the selected run."
            ),
        ]
    )
    return


@app.cell
def _(index, mo):
    _lines = [
        "| run | timestamp (UTC) | commit | tag | trigger | tier | device | report |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _rec in index.iloc[::-1].to_dict("records"):
        _ts = (
            _rec["timestamp"].strftime("%Y-%m-%d %H:%M")
            if hasattr(_rec["timestamp"], "strftime")
            else ""
        )
        _lines.append(
            f"| `{_rec['run_id']}` | {_ts} | `{_rec['short_sha']}` | {_rec.get('tag') or ''} | "
            f"{_rec.get('trigger') or ''} | {_rec.get('tier') or ''} | {_rec.get('device') or ''} | "
            f"[report](results/{_rec['run_id']}/report.html) |"
        )
    mo.vstack([mo.md("## All runs"), mo.md("\n".join(_lines))])
    return


if __name__ == "__main__":
    app.run()
