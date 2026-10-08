"""Pure ``pandas`` / ``numpy`` helpers shared by the marimo notebooks.

Nothing in this module may import ``jax``, ``sif2jax`` or ``slsqp_jax``:
the dashboard runs under Pyodide where only pure-Python wheels are
available.

Terminology
-----------
instance
    One ``(problem, y0_iD)`` pair from the catalogue.
task
    One instance solved with one solver ``config``.
solved
    A task whose minimiser reported success *and* whose returned point is
    feasible (``feas <= feas_tol``) and - when a reference value is known -
    matches the reference objective (``f_gap <= f_tol``). See
    :func:`add_solved`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_FEAS_TOL",
    "DEFAULT_F_TOL",
    "HARNESS_STATUSES",
    "add_solved",
    "classify_outcome",
    "longitudinal",
    "outcome_table",
    "performance_profile",
    "status_table",
    "summary_table",
]

DEFAULT_FEAS_TOL = 1e-6
DEFAULT_F_TOL = 1e-4
HARNESS_STATUSES: tuple[str, ...] = ("timeout", "compile_error", "runtime_error")

_OUTCOME_ORDER = (
    "solved",
    "claimed_but_wrong",
    "claimed_but_infeasible",
    "max_steps",
    "solver_failure",
    "timeout",
    "compile_error",
    "runtime_error",
)


def add_solved(
    frame: pd.DataFrame,
    *,
    feas_tol: float = DEFAULT_FEAS_TOL,
    f_tol: float = DEFAULT_F_TOL,
) -> pd.DataFrame:
    """Return a copy with ``solved`` and ``outcome`` columns.

    Parameters
    ----------
    frame
        Result rows from ``results.jsonl``.
    feas_tol
        Maximum constraint violation (inf-norm) for a point to count.
    f_tol
        Maximum relative objective gap ``|f - f*| / max(1, |f*|)`` when a
        reference ``fstar`` is available.

    Returns
    -------
    pandas.DataFrame
        Copy of ``frame`` with boolean ``solved`` and categorical
        ``outcome``.

    Examples
    --------
    >>> import pandas as pd
    >>> from benchmarks.analysis import add_solved
    >>> rows = pd.DataFrame({
    ...     "status": ["successful", "successful", "max_steps_reached", "timeout"],
    ...     "successful": [True, True, False, False],
    ...     "feas": [0.0, 1e-3, 0.0, None],
    ...     "f_gap": [1e-9, 1e-9, 1e-9, None],
    ...     "has_fstar": [True, True, True, True],
    ... })
    >>> add_solved(rows)["outcome"].tolist()
    ['solved', 'claimed_but_infeasible', 'max_steps', 'timeout']
    """
    out = frame.copy()
    if out.empty:
        out["solved"] = pd.Series(dtype=bool)
        out["outcome"] = pd.Series(dtype=object)
        return out
    successful = (
        out.get("successful", pd.Series(False, index=out.index))
        .fillna(False)
        .astype(bool)
    )
    feas = pd.to_numeric(
        out.get("feas", pd.Series(np.nan, index=out.index)), errors="coerce"
    )
    f_gap = pd.to_numeric(
        out.get("f_gap", pd.Series(np.nan, index=out.index)), errors="coerce"
    )
    has_fstar = (
        out.get("has_fstar", pd.Series(False, index=out.index))
        .fillna(False)
        .astype(bool)
    )
    finite = (
        out.get("finite", pd.Series(True, index=out.index)).fillna(True).astype(bool)
    )

    feasible = feas.le(feas_tol).fillna(False) & finite
    objective_ok = (~has_fstar) | f_gap.le(f_tol).fillna(False)
    solved = successful & feasible & objective_ok
    out["solved"] = solved
    out["outcome"] = classify_outcome(
        out["status"].astype(str), successful, feasible, objective_ok
    )
    out["outcome"] = pd.Categorical(out["outcome"], categories=list(_OUTCOME_ORDER))
    return out


def classify_outcome(
    status: pd.Series,
    successful: pd.Series,
    feasible: pd.Series,
    objective_ok: pd.Series,
) -> pd.Series:
    """Map each task to one of the labels in ``_OUTCOME_ORDER``.

    Parameters
    ----------
    status
        Result-enumeration name reported by the minimiser or the harness.
    successful, feasible, objective_ok
        Boolean series aligned with ``status``.
    """
    outcome = pd.Series("solver_failure", index=status.index, dtype=object)
    outcome[status.isin(HARNESS_STATUSES)] = status[status.isin(HARNESS_STATUSES)]
    outcome[status.str.contains("max_steps", case=False, regex=False)] = "max_steps"
    outcome[successful & ~feasible] = "claimed_but_infeasible"
    outcome[successful & feasible & ~objective_ok] = "claimed_but_wrong"
    outcome[successful & feasible & objective_ok] = "solved"
    return outcome


def performance_profile(
    frame: pd.DataFrame,
    *,
    metric: str = "time_median_s",
    group: str = "config",
    instance: Sequence[str] = ("problem", "y0_iD"),
    solved: str = "solved",
    tau_max: float | None = None,
    n_tau: int = 200,
) -> pd.DataFrame:
    """Dolan-Moré performance-profile curves.

    For every solver *s* and instance *p* the ratio
    \\( r_{p,s} = t_{p,s} / \\min_{s'} t_{p,s'} \\) is computed over the
    solvers that solved *p*; unsolved tasks get \\( r = \\infty \\). The
    profile \\( \\rho_s(\\tau) \\) is the fraction of instances with
    \\( r_{p,s} \\le \\tau \\).

    Parameters
    ----------
    frame
        Rows that already carry the ``solved`` column (see :func:`add_solved`).
    metric
        Column holding the cost (time, steps, ...). Must be positive.
    group
        Column identifying a solver.
    instance
        Columns identifying an instance.
    solved
        Boolean column; unsolved tasks have infinite ratio.
    tau_max
        Right end of the \\( \\tau \\) grid (log-spaced). Defaults to the
        largest finite ratio.
    n_tau
        Number of grid points.

    Returns
    -------
    pandas.DataFrame
        Long table with columns ``[group, "tau", "rho"]``.

    Examples
    --------
    >>> import pandas as pd
    >>> from benchmarks.analysis import performance_profile
    >>> rows = pd.DataFrame({
    ...     "problem": ["A", "A", "B", "B"], "y0_iD": 0,
    ...     "config": ["s1", "s2", "s1", "s2"],
    ...     "time_median_s": [1.0, 2.0, 4.0, 1.0],
    ...     "solved": [True, True, False, True],
    ... })
    >>> prof = performance_profile(rows, n_tau=3)
    >>> prof[prof.config == "s1"]["rho"].tolist()
    [0.5, 0.5, 0.5]
    >>> prof[prof.config == "s2"]["rho"].tolist()
    [0.5, 0.5, 1.0]
    """
    inst = list(instance)
    cost = pd.to_numeric(frame[metric], errors="coerce")
    data = frame.assign(_cost=np.where(frame[solved].astype(bool), cost, np.inf))
    best = data.groupby(inst)["_cost"].transform("min")
    data["_ratio"] = data["_cost"] / best
    data.loc[~np.isfinite(best), "_ratio"] = np.inf
    n_instances = data[inst].drop_duplicates().shape[0]
    finite = data["_ratio"][np.isfinite(data["_ratio"])]
    if tau_max is None:
        tau_max = float(finite.max()) if not finite.empty else 1.0
    tau_max = max(tau_max, 1.0 + 1e-12)
    taus = np.logspace(0, np.log10(tau_max), n_tau)
    records = []
    for name, sub in data.groupby(group, observed=True):
        ratios = sub["_ratio"].to_numpy()
        rho = [(ratios <= tau).sum() / n_instances for tau in taus]
        records.append(pd.DataFrame({group: name, "tau": taus, "rho": rho}))
    if not records:
        return pd.DataFrame(columns=[group, "tau", "rho"])
    return pd.concat(records, ignore_index=True)


def summary_table(frame: pd.DataFrame, by: Sequence[str] = ("config",)) -> pd.DataFrame:
    """Per-group success rate and timing summary.

    Parameters
    ----------
    frame
        Rows carrying ``solved`` (see :func:`add_solved`).
    by
        Grouping columns (``config`` alone, or ``("config", "collection")``).

    Returns
    -------
    pandas.DataFrame
        Columns ``n_tasks``, ``n_solved``, ``solve_rate``, ``median_time_s``
        (over solved tasks), ``median_steps`` (over solved), ``median_compile_s``.
    """
    by = list(by)
    if frame.empty:
        return pd.DataFrame(columns=[*by, "n_tasks", "n_solved", "solve_rate"])
    grouped = frame.groupby(by, observed=True)
    out = grouped.size().rename("n_tasks").to_frame()
    out["n_solved"] = grouped["solved"].sum().astype(int)
    out["solve_rate"] = out["n_solved"] / out["n_tasks"]
    solved = frame[frame["solved"].astype(bool)]
    if not solved.empty:
        sg = solved.groupby(by, observed=True)
        out["median_time_s"] = sg["time_median_s"].median()
        if "steps" in solved:
            out["median_steps"] = sg["steps"].median()
    if "compile_s" in frame:
        out["median_compile_s"] = grouped["compile_s"].median()
    return out.reset_index()


def outcome_table(frame: pd.DataFrame, by: str = "config") -> pd.DataFrame:
    """Counts of each ``outcome`` per group, as a long table for stacked bars."""
    if frame.empty:
        return pd.DataFrame(columns=[by, "outcome", "count"])
    counts = (
        frame.groupby([by, "outcome"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    return counts[counts["count"] > 0].reset_index(drop=True)


def status_table(frame: pd.DataFrame, by: str = "config") -> pd.DataFrame:
    """Cross-tabulation of raw ``status`` names per group."""
    if frame.empty:
        return pd.DataFrame()
    return pd.crosstab(frame["status"], frame[by])


def longitudinal(
    runs: pd.DataFrame,
    *,
    by: Sequence[str] = ("run_id", "config"),
) -> pd.DataFrame:
    """Aggregate several runs for trend plots.

    Parameters
    ----------
    runs
        Concatenated rows from many runs, carrying ``run_id`` and ``solved``.
    by
        Grouping columns; the first must be ``run_id``.

    Returns
    -------
    pandas.DataFrame
        :func:`summary_table` output per run plus ``timestamp`` and ``sha``
        when those columns are present in ``runs``, sorted by ``run_id``.
    """
    table = summary_table(runs, by=by)
    for col in ("timestamp", "sha", "tag", "device"):
        if col in runs:
            first = runs.groupby("run_id", observed=True)[col].first()
            table[col] = table["run_id"].map(first)
    return table.sort_values(list(by)).reset_index(drop=True)
