"""The benchmark's *solved* criterion as a pure function of one result row.

The same rule is applied when a task finishes (:mod:`benchmarks.worker`),
when stored rows are re-scored (``python -m benchmarks rescore``) and when a
diagnostic re-run is compared with the original run
(``scripts/diagnose_benchmark.py``). The vectorised twin used by the
notebooks lives in :func:`benchmarks.analysis.add_solved`; it cannot import
this module because the wasm dashboard ships ``analysis.py`` alone, and
``tests/benchmarks/test_criteria.py`` checks that both agree.

This module must stay free of heavy imports (no ``jax``, no ``pandas``).

A task is **solved** when, in this order of checks:

1. the point is finite (``finite``);
2. the minimiser reported success (``successful``);
3. the point is feasible (``feas <= feas_tol``);
4. the objective matches the reference, when one exists
   (``not has_fstar or f_gap <= f_tol``);
5. the point is stationary: the IPOPT-scaled inf-norm of the Lagrangian
   gradient, evaluated with the multipliers the solver returned
   (``stat_scaled``, see :func:`benchmarks.metrics.quality_metrics`), is at
   most ``stat_tol``.

The first check that fails gives the *reason*. A missing or ``NaN``
``stat_scaled`` means that the solver returned no multipliers; the check then
passes (benefit of the doubt) because there is nothing to evaluate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

__all__ = [
    "DEFAULT_FEAS_TOL",
    "DEFAULT_F_TOL",
    "DEFAULT_STAT_TOL",
    "SOLVED_REASONS",
    "solved_reason",
]

DEFAULT_FEAS_TOL = 1e-6
DEFAULT_F_TOL = 1e-4
DEFAULT_STAT_TOL = 1e-4

SOLVED_REASONS: tuple[str, ...] = (
    "non_finite",
    "unsuccessful",
    "infeasible",
    "wrong_objective",
    "non_stationary",
)
"""Reasons a task can fail, in the order in which they are checked."""


def _num(value: Any) -> float:
    """``float(value)`` with ``None`` and non-numeric values mapped to ``NaN``."""
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _flag(value: Any, default: bool) -> bool:
    """Boolean field where ``None`` and ``NaN`` (a missing value) give ``default``."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return bool(value)


def solved_reason(
    row: Mapping[str, Any],
    feas_tol: float = DEFAULT_FEAS_TOL,
    f_tol: float = DEFAULT_F_TOL,
    stat_tol: float = DEFAULT_STAT_TOL,
) -> str | None:
    """Return why a result row does not count as solved, or ``None`` if it does.

    Parameters
    ----------
    row
        Result row. The fields read are ``finite`` (missing means finite),
        ``successful``, ``feas``, ``has_fstar``, ``f_gap`` and
        ``stat_scaled``; ``None`` and ``NaN`` are treated as missing.
    feas_tol
        Maximum constraint violation (inf-norm). A missing ``feas`` fails.
    f_tol
        Maximum relative objective gap ``|f - f*| / max(1, |f*|)``, checked
        only when ``has_fstar`` is true. A missing ``f_gap`` fails then.
    stat_tol
        Maximum IPOPT-scaled stationarity residual (``stat_scaled``). A
        missing or ``NaN`` value passes: the solver returned no multipliers.

    Returns
    -------
    str or None
        One of :data:`SOLVED_REASONS` (``"non_finite"``, ``"unsuccessful"``,
        ``"infeasible"``, ``"wrong_objective"``, ``"non_stationary"``), the
        first failing check in that order, or ``None`` when the task is
        solved. All comparisons are inclusive (``<=``).

    Examples
    --------
    >>> from benchmarks.criteria import solved_reason
    >>> ok = dict(successful=True, finite=True, feas=0.0, has_fstar=True,
    ...           f_gap=1e-9, stat_scaled=1e-7)
    >>> solved_reason(ok) is None
    True
    >>> solved_reason({**ok, "successful": False})
    'unsuccessful'
    >>> solved_reason({**ok, "feas": 1e-3})
    'infeasible'
    >>> solved_reason({**ok, "f_gap": 1e-2})
    'wrong_objective'

    A reported success at a feasible point with the right objective value
    that is nowhere near stationary (the objective is flat along a valley):

    >>> solved_reason({**ok, "f_gap": 7.6e-6, "stat_scaled": 0.53})
    'non_stationary'

    No multipliers (``NaN``) means the stationarity check cannot fail:

    >>> solved_reason({**ok, "stat_scaled": float("nan")}) is None
    True
    """
    if not _flag(row.get("finite"), True):
        return "non_finite"
    if not _flag(row.get("successful"), False):
        return "unsuccessful"
    if not _num(row.get("feas")) <= feas_tol:
        return "infeasible"
    if _flag(row.get("has_fstar"), False) and not _num(row.get("f_gap")) <= f_tol:
        return "wrong_objective"
    stat = _num(row.get("stat_scaled"))
    if not math.isnan(stat) and not stat <= stat_tol:
        return "non_stationary"
    return None
