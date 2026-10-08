"""Per-run benchmark report (marimo notebook).

Run interactively::

    uv run marimo run benchmarks/report.py -- --run-dir results/<run_id>

Export the static HTML published alongside each run::

    uv run python -m benchmarks report results/<run_id>
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="sqpdax benchmark report")


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import pandas as pd

    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from benchmarks import analysis
    from benchmarks.results import load_run

    alt.data_transformers.disable_max_rows()
    return Path, alt, analysis, json, load_run, mo, pd


@app.cell
def _(Path, load_run, mo):
    _arg = mo.cli_args().get("run-dir")
    run_dir = Path(_arg) if _arg else sorted(Path("results").glob("*_*"))[-1]
    raw, meta = load_run(run_dir)
    return meta, raw, run_dir


@app.cell
def _(meta, mo, raw, run_dir):
    _counts = meta.get("status_counts") or {}
    _lines = [
        f"# Benchmark report `{run_dir.name}`",
        "",
        f"- **commit** `{str(meta.get('sha', ''))[:12]}`"
        + (f" (tag `{meta['tag']}`)" if meta.get("tag") else ""),
        f"- **timestamp** {meta.get('timestamp', '?')} UTC, trigger `{meta.get('trigger', '?')}`",
        f"- **host** {meta.get('hostname', '?')} - {meta.get('platform', '?')} - "
        f"device `{', '.join(map(str, raw['device'].dropna().unique())) if 'device' in raw else '?'}`"
        f" - jax {', '.join(map(str, raw['jax_version'].dropna().unique())) if 'jax_version' in raw else '?'}",
        f"- **tier** `{meta.get('tier', '?')}`, **max_steps** {meta.get('max_steps', '?')}, "
        f"**timeout** {meta.get('timeout_s', '?')} s, **configs** {', '.join(f'`{c}`' for c in meta.get('configs', []))}",
        f"- **tasks** {len(raw)} rows, status counts: "
        + ", ".join(f"`{k}`={v}" for k, v in sorted(_counts.items())),
        "",
        "Every task is one catalogue instance (a CUTEst problem and a starting "
        "point) solved by one solver configuration. Timing excludes compilation "
        "and a one-step warm-up call; the reported time is the median over the "
        "repeated full solves.",
    ]
    mo.md("\n".join(_lines))
    return


@app.cell
def _(mo):
    feas_tol = mo.ui.number(
        value=1e-6, start=1e-12, stop=1.0, step=1e-7, label="feasibility tolerance"
    )
    f_tol = mo.ui.number(
        value=1e-4,
        start=1e-12,
        stop=1.0,
        step=1e-5,
        label="relative objective-gap tolerance",
    )
    mo.hstack([feas_tol, f_tol])
    return f_tol, feas_tol


@app.cell
def _(analysis, f_tol, feas_tol, raw):
    df = analysis.add_solved(raw, feas_tol=feas_tol.value, f_tol=f_tol.value)
    return (df,)


@app.cell
def _(alt, analysis, df, f_tol, feas_tol, mo):
    _outcomes = analysis.outcome_table(df)
    _chart = (
        alt.Chart(_outcomes)
        .mark_bar()
        .encode(
            x=alt.X("config:N", title="solver configuration"),
            y=alt.Y("count:Q", title="number of tasks"),
            color=alt.Color(
                "outcome:N",
                title="outcome",
                sort=list(_outcomes["outcome"].cat.categories)
                if hasattr(_outcomes["outcome"], "cat")
                else None,
            ),
            order=alt.Order("outcome:N"),
            tooltip=["config", "outcome", "count"],
        )
        .properties(width=520, height=300, title="Outcome per configuration")
    )
    mo.vstack(
        [
            mo.md("## Outcomes"),
            mo.ui.altair_chart(_chart),
            mo.md(
                f"""**Figure 1.** Stacked count of tasks per solver configuration (x axis), split by
outcome. `solved` requires the minimiser to report success *and* the returned point to
be feasible (max violation ≤ {feas_tol.value:g}) *and*, when a reference optimum is
known, to match it (relative gap ≤ {f_tol.value:g}). `claimed_but_infeasible` /
`claimed_but_wrong` are reported successes that fail those checks; `max_steps` hit the
iteration budget; `solver_failure` groups every other solver-reported termination;
`timeout`, `compile_error`, `runtime_error` are harness-level failures."""
            ),
        ]
    )
    return


@app.cell
def _(analysis, df, mo):
    _summary = analysis.summary_table(df)
    _by_coll = analysis.summary_table(df, by=("collection", "config"))
    mo.vstack(
        [
            mo.md("## Summary tables"),
            mo.md(
                "**Table 1.** Per configuration: number of tasks, number solved, solve rate, "
                "median wall time and median outer iterations over *solved* tasks, median compile time."
            ),
            mo.ui.table(_summary.round(6), selection=None),
            mo.md("**Table 2.** The same statistics split by catalogue collection."),
            mo.ui.table(_by_coll.round(6), selection=None),
        ]
    )
    return


@app.cell
def _(alt, analysis, df, mo):
    _prof_t = analysis.performance_profile(df, metric="time_median_s")
    _prof_s = analysis.performance_profile(df, metric="steps")

    def _profile_chart(data, title):
        return (
            alt.Chart(data)
            .mark_line(interpolate="step-after")
            .encode(
                x=alt.X(
                    "tau:Q",
                    scale=alt.Scale(type="log"),
                    title="performance ratio τ (log scale)",
                ),
                y=alt.Y(
                    "rho:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    title="fraction of instances solved within τ × best",
                ),
                color=alt.Color("config:N", title="configuration"),
                tooltip=[
                    "config",
                    alt.Tooltip("tau:Q", format=".2f"),
                    alt.Tooltip("rho:Q", format=".3f"),
                ],
            )
            .properties(width=420, height=300, title=title)
        )

    mo.vstack(
        [
            mo.md("## Performance profiles"),
            mo.hstack(
                [
                    mo.ui.altair_chart(_profile_chart(_prof_t, "Wall time")),
                    mo.ui.altair_chart(_profile_chart(_prof_s, "Outer iterations")),
                ]
            ),
            mo.md(
                """**Figure 2.** Dolan-Moré performance profiles. For each instance the cost of a
configuration is divided by the smallest cost among the configurations that solved it;
the curve shows, for every ratio τ on the (logarithmic) x axis, the fraction of all
instances the configuration solved within τ times the best cost. The value at τ = 1 is
the share of instances on which the configuration was the fastest; the right-hand
plateau is its overall solve rate. Left: median wall time per solve. Right: number of
outer iterations."""
            ),
        ]
    )
    return


@app.cell
def _(alt, df, mo):
    _chart = (
        alt.Chart(df.dropna(subset=["time_median_s"]))
        .mark_point(filled=True, opacity=0.7)
        .encode(
            x=alt.X(
                "n:Q", scale=alt.Scale(type="log"), title="number of variables n (log)"
            ),
            y=alt.Y(
                "time_median_s:Q",
                scale=alt.Scale(type="log"),
                title="median wall time per solve [s] (log)",
            ),
            color=alt.Color("config:N", title="configuration"),
            shape=alt.Shape("outcome:N", title="outcome"),
            tooltip=[
                "problem",
                "config",
                "outcome",
                "status",
                "n",
                "steps",
                alt.Tooltip("time_median_s:Q", format=".3e"),
                alt.Tooltip("feas:Q", format=".2e"),
                alt.Tooltip("stat_inf:Q", format=".2e"),
            ],
        )
        .properties(width=700, height=380, title="Solve time versus problem size")
        .interactive()
    )
    _compile = (
        alt.Chart(df.dropna(subset=["compile_s"]))
        .mark_point(filled=True, opacity=0.7)
        .encode(
            x=alt.X(
                "n:Q", scale=alt.Scale(type="log"), title="number of variables n (log)"
            ),
            y=alt.Y(
                "compile_s:Q",
                scale=alt.Scale(type="log"),
                title="compile time [s] (log)",
            ),
            color=alt.Color("config:N", title="configuration"),
            tooltip=[
                "problem",
                "config",
                "n",
                alt.Tooltip("compile_s:Q", format=".2f"),
            ],
        )
        .properties(width=700, height=260, title="Compile time versus problem size")
        .interactive()
    )
    mo.vstack(
        [
            mo.md("## Timing"),
            mo.ui.altair_chart(_chart),
            mo.md(
                """**Figure 3.** Median wall time of one full solve (y, log) against the number of
variables (x, log), one point per task, coloured by configuration and shaped by outcome.
Points for unsolved tasks still show the time the solver took to terminate (or to hit
the step budget). Hover for problem name, status and KKT metrics."""
            ),
            mo.ui.altair_chart(_compile),
            mo.md(
                """**Figure 4.** One-off XLA compile time of `minimise` (y, log) against n (x, log).
Compilation happens once per task and is excluded from every other timing figure."""
            ),
        ]
    )
    return


@app.cell
def _(alt, df, mo):
    _ok = df[df["successful"].fillna(False).astype(bool)].copy()
    _eps = 1e-16
    for _c in ("feas", "stat_inf", "f_gap"):
        _ok[_c] = _ok[_c].clip(lower=_eps)
    _chart = (
        alt.Chart(_ok)
        .mark_point(filled=True, opacity=0.7)
        .encode(
            x=alt.X(
                "feas:Q",
                scale=alt.Scale(type="log"),
                title="max constraint violation (log)",
            ),
            y=alt.Y(
                "stat_inf:Q",
                scale=alt.Scale(type="log"),
                title="‖∇ₓL‖∞ with returned multipliers (log)",
            ),
            color=alt.Color("config:N", title="configuration"),
            shape=alt.Shape("outcome:N", title="outcome"),
            tooltip=[
                "problem",
                "config",
                "outcome",
                alt.Tooltip("feas:Q", format=".2e"),
                alt.Tooltip("stat_inf:Q", format=".2e"),
                alt.Tooltip("f_gap:Q", format=".2e"),
                alt.Tooltip("compl:Q", format=".2e"),
            ],
        )
        .properties(width=700, height=380, title="KKT quality of reported successes")
        .interactive()
    )
    _gap = (
        alt.Chart(_ok[_ok["has_fstar"].fillna(False).astype(bool)])
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("config:N", title="configuration"),
            y=alt.Y(
                "f_gap:Q",
                scale=alt.Scale(type="log"),
                title="|f − f*| / max(1, |f*|) (log)",
            ),
            color=alt.Color("config:N", legend=None),
        )
        .properties(
            width=520, height=300, title="Objective gap to the CUTEst reference"
        )
    )
    mo.vstack(
        [
            mo.md("## Solution quality"),
            mo.ui.altair_chart(_chart),
            mo.md(
                """**Figure 5.** For tasks whose minimiser *reported* success: maximum constraint
violation at the returned point (x, log; equalities, inequalities and bounds) against the
infinity norm of the Lagrangian gradient evaluated with the returned multipliers (y, log).
The same stationarity expression is used for every family, independent of any barrier
term the interior-point methods minimise internally. Points far from the lower-left corner
are successes whose KKT conditions are not actually satisfied. Values are floored at 1e-16
for the log scale."""
            ),
            mo.ui.altair_chart(_gap),
            mo.md(
                """**Figure 6.** Distribution (box = quartiles, whiskers = min/max) of the relative
objective gap to the reference optimum shipped with sif2jax, per configuration, for
reported successes on problems with a known reference. A gap well above the tolerance
flags convergence to a different (possibly local) minimiser or a false success."""
            ),
        ]
    )
    return


@app.cell
def _(analysis, df, mo):
    _cols = [
        "problem",
        "y0_iD",
        "collection",
        "n",
        "config",
        "outcome",
        "status",
        "steps",
        "feas",
        "f_gap",
        "stat_inf",
        "compl",
        "time_median_s",
        "error",
    ]
    _cols = [c for c in _cols if c in df]
    _bad = df[~df["solved"].astype(bool)][_cols].sort_values(["problem", "config"])
    mo.vstack(
        [
            mo.md("## Unsolved tasks"),
            mo.md(
                f"**Table 3.** The {len(_bad)} tasks that did not count as solved, with the raw "
                "termination status and KKT metrics (empty for harness failures)."
            ),
            mo.ui.table(_bad, selection=None, page_size=25),
            mo.md("**Table 4.** Raw termination statuses per configuration."),
            mo.ui.table(analysis.status_table(df).reset_index(), selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
