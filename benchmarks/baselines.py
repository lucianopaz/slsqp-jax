"""SciPy baselines (``SLSQP`` and ``trust-constr``) for the benchmark suite.

The baselines solve exactly the same :class:`~slsqp_jax.sqpdax.problem.Problem`
as the sqpdax configurations: every callback handed to
:func:`scipy.optimize.minimize` is a ``jax.jit``-compiled function of the
problem (objective, gradient, constraint values, Jacobians and - for
``trust-constr`` - Hessian-vector products), so the two backends pay the
same per-evaluation cost and differ only in the optimisation algorithm and
in the Python-level overhead of the SciPy drivers.

Phases map onto the worker's as follows: *compile* evaluates every jitted
callback once (XLA compilation), *warm-up* is one ``maxiter=1`` call and
the timed solves call :func:`scipy.optimize.minimize` with the task's
``max_steps`` as ``maxiter``.

Multipliers are translated into the sqpdax convention
(``grad f + lam^T J_g + mu^T J_h - z_lb + z_ub = 0`` with ``mu, z >= 0``) so
:func:`benchmarks.metrics.quality_metrics` evaluates every backend with the
same stationarity expression:

- SLSQP uses ``L = f - sum_i m_i c_i`` with ``c_eq = 0`` and ``c_ineq >= 0``
  and returns ``m`` for the general constraints only. Hence ``lam = -m_eq``,
  ``mu = m_ineq`` and the bound multipliers are recovered from the
  stationarity residual on the active bounds.
- trust-constr uses ``grad f + sum_c J_c^T v_c`` with the constraints passed
  directly in sqpdax form (``g = 0``, ``h <= 0``), so ``lam = v_g``,
  ``mu = v_h`` and the bounds entry gives ``-z_lb + z_ub``.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from jax import Array

from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.problem import Problem

from .runners import SolveOutcome

__all__ = [
    "METHODS",
    "ScipyBaseline",
    "ScipyRunner",
    "dual_from_slsqp",
    "dual_from_trust_constr",
    "scipy_status_name",
]

METHODS: tuple[str, ...] = ("SLSQP", "trust-constr")
"""SciPy ``minimize`` methods available as baselines."""

_SLSQP_EXIT_MODES: dict[int, str] = {
    0: "Optimization terminated successfully",
    2: "More equality constraints than independent variables",
    3: "More than 3*n iterations in LSQ subproblem",
    4: "Inequality constraints incompatible",
    5: "Singular matrix E in LSQ subproblem",
    6: "Singular matrix C in LSQ subproblem",
    7: "Rank-deficient equality constraint subproblem HFTI",
    8: "Positive directional derivative for linesearch",
    9: "Iteration limit reached",
}

_TRUST_CONSTR_MESSAGES: dict[int, str] = {
    0: "The maximum number of function evaluations is exceeded.",
    1: "`gtol` termination condition is satisfied.",
    2: "`xtol` termination condition is satisfied.",
    3: "`callback` raised `StopIteration`.",
    4: "Constraint violation exceeds 'gtol'",
}

_MAX_STEPS_STATUS: dict[str, int] = {"SLSQP": 9, "trust-constr": 0}

_RESULT_FIELDS_SKIPPED: frozenset[str] = frozenset(
    {
        "x",
        "v",
        "jac",
        "grad",
        "lagrangian_grad",
        "constr",
        "multipliers",
        "message",
        "method",
        "status",
        "success",
        "nit",
        "niter",
        "fun",
    }
)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def scipy_status_name(method: str, status: int, success: bool) -> str:
    """Map a SciPy termination to the status vocabulary of the harness.

    Parameters
    ----------
    method
        ``"SLSQP"`` or ``"trust-constr"``.
    status
        ``OptimizeResult.status`` (SLSQP exit mode / trust-constr code).
    success
        ``OptimizeResult.success``.

    Returns
    -------
    str
        ``"successful"`` for a reported success, ``"max_steps_reached"``
        when the iteration limit stopped the solver (SLSQP exit mode 9,
        trust-constr status 0) and otherwise a slug of SciPy's termination
        message (``"<method>_status_<n>"`` for unknown codes).

    Examples
    --------
    >>> from benchmarks.baselines import scipy_status_name
    >>> scipy_status_name("SLSQP", 0, True)
    'successful'
    >>> scipy_status_name("SLSQP", 9, False)
    'max_steps_reached'
    >>> scipy_status_name("SLSQP", 8, False)
    'positive_directional_derivative_for_linesearch'
    >>> scipy_status_name("trust-constr", 0, False)
    'max_steps_reached'
    >>> scipy_status_name("trust-constr", 4, False)
    'constraint_violation_exceeds_gtol'
    """
    if method not in METHODS:
        raise ValueError(f"unknown SciPy baseline method {method!r}; known: {METHODS}")
    if success:
        return "successful"
    if status == _MAX_STEPS_STATUS[method]:
        return "max_steps_reached"
    messages = _SLSQP_EXIT_MODES if method == "SLSQP" else _TRUST_CONSTR_MESSAGES
    message = messages.get(status)
    if message is None:
        return f"{_slug(method)}_status_{status}"
    return _slug(message)


@dataclass(frozen=True)
class ScipyBaseline:
    """Description of one SciPy baseline solver.

    Attributes
    ----------
    method
        ``"SLSQP"`` or ``"trust-constr"`` (the ``method`` argument of
        :func:`scipy.optimize.minimize`).
    bound_tol
        Absolute distance below which a bound counts as active when SLSQP's
        missing bound multipliers are recovered from the stationarity
        residual.

    Examples
    --------
    >>> from benchmarks.baselines import ScipyBaseline
    >>> ScipyBaseline("SLSQP").method
    'SLSQP'
    """

    method: str
    bound_tol: float = 1e-8

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(
                f"unknown SciPy baseline method {self.method!r}; known: {METHODS}"
            )

    @property
    def exact_curvature(self) -> bool:
        """Whether the method consumes the problem's exact Hessian-vector products."""
        return self.method == "trust-constr"


def _active_bound_multipliers(
    problem: Problem, x: np.ndarray, residual: np.ndarray, tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Bound multipliers that absorb ``residual`` on the active bounds."""
    lb = np.asarray(problem.lb, dtype=float)
    ub = np.asarray(problem.ub, dtype=float)
    null_lb = np.asarray(problem.null_lb, dtype=bool)
    null_ub = np.asarray(problem.null_ub, dtype=bool)
    at_lb = ~null_lb & (x - lb <= tol)
    at_ub = ~null_ub & (ub - x <= tol)
    z_lb = np.where(at_lb, np.maximum(residual, 0.0), 0.0)
    z_ub = np.where(at_ub, np.maximum(-residual, 0.0), 0.0)
    return z_lb, z_ub


def dual_from_slsqp(
    problem: Problem,
    x: Array | np.ndarray,
    multipliers: np.ndarray | None,
    *,
    grad: np.ndarray | None = None,
    eq_jac: np.ndarray | None = None,
    ineq_jac: np.ndarray | None = None,
    bound_tol: float = 1e-8,
) -> Dual | None:
    """Translate SLSQP's ``OptimizeResult.multipliers`` into a :class:`Dual`.

    Parameters
    ----------
    problem
        The NLP (sqpdax convention).
    x
        Returned point.
    multipliers
        SciPy's array ``[m_eq, m_ineq]`` for the Lagrangian
        ``f - m^T c`` with ``c_eq = 0`` and ``c_ineq = -h >= 0``; ``None``
        when the SciPy version does not provide it.
    grad, eq_jac, ineq_jac
        Objective gradient and constraint Jacobians at ``x``. Evaluated
        from ``problem`` when omitted; :class:`ScipyRunner` passes its
        jitted evaluations so the translation stays off the critical path.
    bound_tol
        Absolute activity tolerance for the bounds.

    Returns
    -------
    Dual or None
        ``lam = -m_eq``, ``mu = m_ineq`` and bound multipliers recovered
        from the stationarity residual on the active bounds.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.problem import build_problem
    >>> from benchmarks.baselines import dual_from_slsqp
    >>> # min x0^2 + x1^2  s.t.  x0 + x1 >= 1  (SLSQP multiplier m = 1), x1 >= 0.6
    >>> prob = build_problem(lambda x: jnp.sum(x**2), n=2, mineq=1,
    ...                      ineq_fn=lambda x: jnp.array([1.0 - x[0] - x[1]]),
    ...                      lb=jnp.array([-jnp.inf, 0.6]), ub=jnp.array([jnp.inf, jnp.inf]),
    ...                      autodiff_mode="jax")
    >>> d = dual_from_slsqp(prob, jnp.array([0.4, 0.6]), np.array([0.8]))
    >>> [round(float(v), 6) for v in (d.ineq_multipliers[0], d.lb_multipliers[1])]
    [0.8, 0.4]
    """
    if multipliers is None:
        return None
    import jax.numpy as jnp

    xj = jnp.asarray(x)
    m = np.asarray(multipliers, dtype=float).ravel()
    lam = -m[: problem.meq]
    mu = m[problem.meq : problem.meq + problem.mineq]
    r = np.asarray(problem.grad(xj) if grad is None else grad, dtype=float)
    if problem.meq:
        jac = problem.eq_fn_jac(xj) if eq_jac is None else eq_jac
        r = r + lam @ np.asarray(jac, dtype=float).reshape(problem.meq, -1)
    if problem.mineq:
        jac = problem.ineq_fn_jac(xj) if ineq_jac is None else ineq_jac
        r = r + mu @ np.asarray(jac, dtype=float).reshape(problem.mineq, -1)
    z_lb, z_ub = _active_bound_multipliers(
        problem, np.asarray(x, dtype=float), r, bound_tol
    )
    return Dual(
        eq_multipliers=jnp.asarray(lam),
        ineq_multipliers=jnp.asarray(mu),
        lb_multipliers=jnp.asarray(z_lb),
        ub_multipliers=jnp.asarray(z_ub),
    )


def dual_from_trust_constr(
    problem: Problem,
    v: list[np.ndarray],
    *,
    has_eq: bool,
    has_ineq: bool,
    has_bounds: bool,
) -> Dual:
    """Translate trust-constr's ``OptimizeResult.v`` into a :class:`Dual`.

    Parameters
    ----------
    problem
        The NLP (sqpdax convention).
    v
        One multiplier array per constraint object passed to SciPy, in the
        order ``[equalities, inequalities, bounds]`` restricted to the
        blocks that were present (``has_*`` flags).
    has_eq, has_ineq, has_bounds
        Which blocks were passed to :func:`scipy.optimize.minimize`.

    Returns
    -------
    Dual
        ``lam = v_g``, ``mu = v_h``, ``z_lb = max(-v_b, 0)``,
        ``z_ub = max(v_b, 0)``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.problem import build_problem
    >>> from benchmarks.baselines import dual_from_trust_constr
    >>> prob = build_problem(lambda x: jnp.sum(x**2), n=2, meq=1,
    ...                      eq_fn=lambda x: jnp.array([x[0] + x[1] - 1.0]),
    ...                      autodiff_mode="jax")
    >>> d = dual_from_trust_constr(prob, [np.array([-1.0]), np.array([0.5, -0.25])],
    ...                            has_eq=True, has_ineq=False, has_bounds=True)
    >>> (d.eq_multipliers.tolist(), d.lb_multipliers.tolist(), d.ub_multipliers.tolist())
    ([-1.0], [0.0, 0.25], [0.5, 0.0])
    """
    import jax.numpy as jnp

    blocks = iter(v)
    lam = np.asarray(next(blocks), float).ravel() if has_eq else np.zeros(problem.meq)
    mu = (
        np.asarray(next(blocks), float).ravel() if has_ineq else np.zeros(problem.mineq)
    )
    if has_bounds:
        v_b = np.asarray(next(blocks), dtype=float).ravel()
        z_lb = np.maximum(-v_b, 0.0)
        z_ub = np.maximum(v_b, 0.0)
    else:
        z_lb = np.zeros(problem.n)
        z_ub = np.zeros(problem.n)
    return Dual(
        eq_multipliers=jnp.asarray(lam),
        ineq_multipliers=jnp.asarray(mu),
        lb_multipliers=jnp.asarray(z_lb),
        ub_multipliers=jnp.asarray(z_ub),
    )


def _as_f64(value: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value, dtype=np.float64))


class ScipyRunner:
    """Solve an sqpdax :class:`Problem` with :func:`scipy.optimize.minimize`.

    Parameters
    ----------
    problem
        The NLP; must expose exact HVPs
        (:attr:`~slsqp_jax.sqpdax.problem.Problem.has_exact_curvature`)
        for ``trust-constr``.
    x0
        Starting point.
    baseline
        Which SciPy method to run.
    options
        ``options`` mapping for :func:`scipy.optimize.minimize`; ``maxiter``
        is overwritten by the ``max_steps`` of each solve.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.problem import build_problem
    >>> from benchmarks.baselines import ScipyBaseline, ScipyRunner
    >>> prob = build_problem(lambda x: jnp.sum(x**2), n=2, meq=1,
    ...                      eq_fn=lambda x: jnp.array([x[0] + x[1] - 1.0]),
    ...                      autodiff_mode="jax", force_hvp_in_jax_mode=True)
    >>> runner = ScipyRunner(prob, jnp.array([2.0, 2.0]), ScipyBaseline("SLSQP"))
    >>> _ = runner.compile()
    >>> out = runner.solve(max_steps=50)
    >>> out.status, [round(float(v), 6) for v in out.x]
    ('successful', [0.5, 0.5])
    """

    def __init__(
        self,
        problem: Problem,
        x0: Array,
        baseline: ScipyBaseline,
        options: Mapping[str, Any] | None = None,
    ):
        import jax
        import jax.numpy as jnp

        if baseline.exact_curvature and not problem.has_exact_curvature:
            raise ValueError(
                f"{baseline.method} baseline needs exact Hessian-vector products; "
                "build the problem with force_hvp_in_jax_mode=True or supply HVPs"
            )
        self.problem = problem
        self.baseline = baseline
        self.options = dict(options or {})
        self.x0 = _as_f64(x0)
        n = int(self.x0.size)
        self.n = n
        self.has_eq = problem.meq > 0
        self.has_ineq = problem.mineq > 0
        lb = np.asarray(problem.lb, dtype=float)
        ub = np.asarray(problem.ub, dtype=float)
        self.has_bounds = bool(np.any(np.isfinite(lb)) or np.any(np.isfinite(ub)))
        self._lb, self._ub = lb, ub

        def _f(x):
            return problem.fn(x)[0]

        self._fn = jax.jit(_f)
        self._grad = jax.jit(problem.grad)
        self._eq = jax.jit(problem.eq_fn)
        self._eq_jac = jax.jit(problem.eq_fn_jac)
        self._ineq = jax.jit(problem.ineq_fn)
        self._ineq_jac = jax.jit(problem.ineq_fn_jac)
        self._hvp = self._eq_whvp = self._ineq_whvp = None
        if baseline.exact_curvature:
            hvp, eq_hvp, ineq_hvp = problem.hvp, problem.eq_fn_hvp, problem.ineq_fn_hvp
            assert hvp is not None and eq_hvp is not None and ineq_hvp is not None
            self._hvp = jax.jit(hvp)
            # ``*_fn_hvp(x, p)`` is the directional derivative of the Jacobian
            # along ``p`` with shape (m, n); ``v @ ...`` is the HVP of ``v . c``.
            self._eq_whvp = jax.jit(lambda x, v, p: v @ eq_hvp(x, p))
            self._ineq_whvp = jax.jit(lambda x, v, p: v @ ineq_hvp(x, p))
        self._jnp = jnp
        self._compiled = False

    # ------------------------------------------------------------- callbacks
    def _call(self, fn: Any, *args: Any) -> np.ndarray:
        return np.asarray(fn(*(_as_f64(a) for a in args)), dtype=np.float64)

    def fun(self, x: np.ndarray) -> float:
        return float(self._call(self._fn, x))

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self._call(self._grad, x)

    def eq(self, x: np.ndarray) -> np.ndarray:
        return self._call(self._eq, x).ravel()

    def eq_jac(self, x: np.ndarray) -> np.ndarray:
        return self._call(self._eq_jac, x).reshape(self.problem.meq, self.n)

    def ineq(self, x: np.ndarray) -> np.ndarray:
        return self._call(self._ineq, x).ravel()

    def ineq_jac(self, x: np.ndarray) -> np.ndarray:
        return self._call(self._ineq_jac, x).reshape(self.problem.mineq, self.n)

    def hessp(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return self._call(self._hvp, x, p)

    def _weighted_hess(self, whvp: Any):
        from scipy.sparse.linalg import LinearOperator

        def hess(x: np.ndarray, v: np.ndarray) -> LinearOperator:
            x = _as_f64(x)
            v = _as_f64(v)

            def matvec(p: np.ndarray) -> np.ndarray:
                return self._call(whvp, x, v, np.ravel(p))

            return LinearOperator(
                (self.n, self.n), matvec=matvec, rmatvec=matvec, dtype=np.float64
            )

        return hess

    # ------------------------------------------------------------- problem spec
    def _bounds(self):
        from scipy.optimize import Bounds

        if not self.has_bounds:
            return None
        return Bounds(self._lb, self._ub)

    def _constraints(self) -> list[Any]:
        if self.baseline.method == "SLSQP":
            cons: list[Any] = []
            if self.has_eq:
                cons.append({"type": "eq", "fun": self.eq, "jac": self.eq_jac})
            if self.has_ineq:
                # SciPy wants c(x) >= 0; sqpdax has h(x) <= 0.
                cons.append(
                    {
                        "type": "ineq",
                        "fun": lambda x: -self.ineq(x),
                        "jac": lambda x: -self.ineq_jac(x),
                    }
                )
            return cons

        from scipy.optimize import NonlinearConstraint

        cons = []
        if self.has_eq:
            cons.append(
                NonlinearConstraint(
                    self.eq,
                    0.0,
                    0.0,
                    jac=self.eq_jac,
                    hess=self._weighted_hess(self._eq_whvp),
                )
            )
        if self.has_ineq:
            cons.append(
                NonlinearConstraint(
                    self.ineq,
                    -np.inf,
                    0.0,
                    jac=self.ineq_jac,
                    hess=self._weighted_hess(self._ineq_whvp),
                )
            )
        return cons

    def _minimize(self, max_steps: int):
        from scipy.optimize import minimize

        kwargs: dict[str, Any] = {}
        if self.baseline.exact_curvature:
            kwargs["hessp"] = self.hessp
        return minimize(
            self.fun,
            self.x0,
            jac=self.grad,
            method=self.baseline.method,
            bounds=self._bounds(),
            constraints=self._constraints(),
            options={**self.options, "maxiter": int(max_steps)},
            **kwargs,
        )

    # ------------------------------------------------------------- phases
    def compile(self) -> float:
        """Evaluate every jitted callback once at ``x0`` and return the elapsed seconds."""
        import jax

        t0 = time.perf_counter()
        x = self.x0
        outputs = [self._fn(x), self._grad(x)]
        if self.has_eq:
            outputs += [self._eq(x), self._eq_jac(x)]
        if self.has_ineq:
            outputs += [self._ineq(x), self._ineq_jac(x)]
        if self.baseline.exact_curvature:
            p = np.ones(self.n)
            outputs.append(self._hvp(x, p))
            if self.has_eq:
                outputs.append(self._eq_whvp(x, np.ones(self.problem.meq), p))
            if self.has_ineq:
                outputs.append(self._ineq_whvp(x, np.ones(self.problem.mineq), p))
        jax.block_until_ready(outputs)
        self._compiled = True
        return time.perf_counter() - t0

    def warmup(self) -> None:
        """One ``maxiter=1`` call through the SciPy driver."""
        if not self._compiled:
            self.compile()
        self._minimize(1)

    def solve(self, max_steps: int) -> SolveOutcome:
        """Run :func:`scipy.optimize.minimize` with ``maxiter=max_steps``."""
        if not self._compiled:
            self.compile()
        res = self._minimize(max_steps)
        status = int(res.status)
        success = bool(res.success)
        x = _as_f64(res.x)
        if self.baseline.method == "SLSQP":
            dual = dual_from_slsqp(
                self.problem,
                x,
                getattr(res, "multipliers", None),
                grad=self.grad(x),
                eq_jac=self.eq_jac(x) if self.has_eq else None,
                ineq_jac=self.ineq_jac(x) if self.has_ineq else None,
                bound_tol=self.baseline.bound_tol,
            )
        else:
            dual = dual_from_trust_constr(
                self.problem,
                list(res.v),
                has_eq=self.has_eq,
                has_ineq=self.has_ineq,
                has_bounds=self.has_bounds,
            )
        stats: dict[str, Any] = {
            key: value
            for key, value in res.items()
            if key not in _RESULT_FIELDS_SKIPPED
        }
        stats["num_steps"] = int(res.nit)
        stats["final_objective"] = float(res.fun)
        return SolveOutcome(
            x=self._jnp.asarray(x),
            dual=dual,
            status=scipy_status_name(self.baseline.method, status, success),
            successful=success,
            steps=int(res.nit),
            stats=stats,
        )
