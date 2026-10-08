"""Solver-independent quality metrics for a returned primal-dual point.

Every quantity is computed from the original NLP (never from a barrier or
merit reformulation), so the same numbers are comparable across the
active-set and interior-point minimisers.

Conventions follow :class:`slsqp_jax.sqpdax.problem.Problem`: equalities
``g(x) = 0``, inequalities ``h(x) <= 0`` with multipliers ``mu >= 0``,
bounds ``lb <= x <= ub`` with multipliers ``z_lb, z_ub >= 0``. The
stationarity residual is

```
r = grad f(x) + lam^T J_g(x) + mu^T J_h(x) - z_lb + z_ub
```

with the bound terms masked where the corresponding bound is infinite
(this is the formula of ``EvaluatedLagrangian.x_grad`` in
:mod:`slsqp_jax.sqpdax.lagrangian.evaluated`).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.problem import Problem

__all__ = [
    "quality_metrics",
    "result_name",
    "stationarity_residual",
]


def stationarity_residual(problem: Problem, x: Array, dual: Dual) -> Array:
    """Gradient of the NLP Lagrangian with respect to ``x``.

    Parameters
    ----------
    problem
        The NLP.
    x
        Primal point.
    dual
        Multipliers ``(lam, mu, z_lb, z_ub)``.

    Returns
    -------
    jax.Array
        ``grad f + lam^T J_g + mu^T J_h - z_lb + z_ub`` with inactive-bound
        terms zeroed.
    """
    r = problem.grad(x)
    if problem.meq:
        r = r + dual.eq_multipliers @ problem.eq_fn_jac(x)
    if problem.mineq:
        r = r + dual.ineq_multipliers @ problem.ineq_fn_jac(x)
    r = r - jnp.where(problem.null_lb, 0.0, dual.lb_multipliers)
    r = r + jnp.where(problem.null_ub, 0.0, dual.ub_multipliers)
    return r


def _inf(v: Array) -> float:
    return float(jnp.max(jnp.abs(v))) if v.size else 0.0


def _two(v: Array) -> float:
    return float(jnp.linalg.norm(v)) if v.size else 0.0


def _min(v: Array) -> float:
    return float(jnp.min(v)) if v.size else 0.0


def quality_metrics(
    problem: Problem,
    x: Array,
    dual: Dual | None,
    *,
    fstar: float | None = None,
    xstar: Array | None = None,
) -> dict[str, Any]:
    """Objective, feasibility, stationarity and complementarity at ``x``.

    Parameters
    ----------
    problem
        The NLP.
    x
        Returned primal point.
    dual
        Returned multipliers; when ``None`` the stationarity and
        complementarity entries are ``NaN``.
    fstar
        Reference objective value, if known.
    xstar
        Reference minimiser, if known.

    Returns
    -------
    dict[str, Any]
        Flat mapping of scalar metrics (all Python ``float``/``bool``):

        - ``objective``, ``f_gap`` (``|f - f*| / max(1, |f*|)``),
          ``x_err`` (``||x - x*||_inf``);
        - ``feas_eq``, ``feas_ineq``, ``feas_lb``, ``feas_ub``, ``feas``
          (inf-norms of the violations and their maximum);
        - ``stat_inf``, ``stat_2``, ``stat_scaled`` (stationarity residual
          in the inf- and 2-norms and the IPOPT-style scaled inf-norm);
        - ``compl`` (``max |mu_i h_i|`` and bound analogues), ``mult_min``
          (most negative inequality / bound multiplier; should be >= 0);
        - ``finite`` (whether ``x`` and ``f`` are finite).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.dual import Dual
    >>> from slsqp_jax.sqpdax.problem import build_problem
    >>> from benchmarks.metrics import quality_metrics
    >>> # min x0^2 + x1^2  s.t.  x0 + x1 = 1  ->  x* = (0.5, 0.5), lam = -1
    >>> prob = build_problem(lambda x: jnp.sum(x**2), n=2, meq=1,
    ...                      eq_fn=lambda x: jnp.array([x[0] + x[1] - 1.0]),
    ...                      autodiff_mode="jax")
    >>> dual = Dual(jnp.array([-1.0]), jnp.zeros(0), jnp.zeros(2), jnp.zeros(2))
    >>> m = quality_metrics(prob, jnp.array([0.5, 0.5]), dual, fstar=0.5)
    >>> round(m["stat_inf"], 12), round(m["feas"], 12), round(m["f_gap"], 12)
    (0.0, 0.0, 0.0)
    """
    x = jnp.asarray(x)
    f, _ = problem.fn(x)
    f = float(f)
    g = problem.eq_fn(x)
    h = problem.ineq_fn(x)

    feas_eq = _inf(g)
    feas_ineq = _inf(jnp.maximum(h, 0.0))
    feas_lb = _inf(jnp.where(problem.null_lb, 0.0, jnp.maximum(problem.lb - x, 0.0)))
    feas_ub = _inf(jnp.where(problem.null_ub, 0.0, jnp.maximum(x - problem.ub, 0.0)))

    out: dict[str, Any] = {
        "objective": f,
        "f_gap": float("nan"),
        "x_err": float("nan"),
        "feas_eq": feas_eq,
        "feas_ineq": feas_ineq,
        "feas_lb": feas_lb,
        "feas_ub": feas_ub,
        "feas": max(feas_eq, feas_ineq, feas_lb, feas_ub),
        "stat_inf": float("nan"),
        "stat_2": float("nan"),
        "stat_scaled": float("nan"),
        "compl": float("nan"),
        "mult_min": float("nan"),
        "finite": bool(np.isfinite(f) and bool(jnp.all(jnp.isfinite(x)))),
    }
    if fstar is not None and np.isfinite(fstar):
        out["f_gap"] = abs(f - fstar) / max(1.0, abs(fstar))
    if xstar is not None:
        xs = jnp.ravel(jnp.asarray(xstar, dtype=x.dtype))
        if xs.shape == x.shape:
            out["x_err"] = _inf(x - xs)

    if dual is not None:
        r = stationarity_residual(problem, x, dual)
        grad_inf = _inf(problem.grad(x))
        mult_inf = max(
            _inf(dual.eq_multipliers),
            _inf(dual.ineq_multipliers),
            _inf(jnp.where(problem.null_lb, 0.0, dual.lb_multipliers)),
            _inf(jnp.where(problem.null_ub, 0.0, dual.ub_multipliers)),
        )
        out["stat_inf"] = _inf(r)
        out["stat_2"] = _two(r)
        out["stat_scaled"] = _inf(r) / max(1.0, grad_inf, mult_inf)
        compl_terms = [
            _inf(dual.ineq_multipliers * h),
            _inf(
                jnp.where(problem.null_lb, 0.0, dual.lb_multipliers * (x - problem.lb))
            ),
            _inf(
                jnp.where(problem.null_ub, 0.0, dual.ub_multipliers * (problem.ub - x))
            ),
        ]
        out["compl"] = max(compl_terms)
        out["mult_min"] = min(
            _min(dual.ineq_multipliers),
            _min(jnp.where(problem.null_lb, 0.0, dual.lb_multipliers)),
            _min(jnp.where(problem.null_ub, 0.0, dual.ub_multipliers)),
        )
    return out


def result_name(result: Any) -> str:
    """Name of an Equinox ``Enumeration`` item (e.g. ``"successful"``).

    Parameters
    ----------
    result
        An ``EnumerationItem`` with a concrete value, or any object with a
        usable ``str``.

    Returns
    -------
    str
        The attribute name of the item in its enumeration, falling back to
        ``str(result)``.

    Examples
    --------
    >>> from slsqp_jax.sqpdax.minimiser import ACTIVE_SET_LINE_SEARCH_RESULTS as R
    >>> from benchmarks.metrics import result_name
    >>> result_name(R.successful), result_name(R.max_steps_reached)
    ('successful', 'max_steps_reached')
    """
    enum = getattr(result, "_enumeration", None)
    value = getattr(result, "_value", None)
    if enum is None or value is None:
        return str(result)
    try:
        value = int(np.asarray(value))
    except Exception:  # noqa: BLE001 - traced or otherwise non-concrete
        return str(result)
    for name, item in enum._name_to_item.items():
        if int(np.asarray(item._value)) == value:
            return name
    return str(result)
