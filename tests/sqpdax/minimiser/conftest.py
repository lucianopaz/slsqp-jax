"""Fixtures shared by :mod:`slsqp_jax.sqpdax.minimiser` tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from slsqp_jax.sqpdax.logging import MemoryDiagnosticsHandler, MemoryHandler
from slsqp_jax.sqpdax.minimiser import (
    ActiveSetLineSearchMinimiser,
    TrustFunnelInteriorPointMinimiser,
    TrustFunnelTerminationMetrics,
    minimise,
)
from slsqp_jax.sqpdax.primal import InteriorPointPrimal
from slsqp_jax.sqpdax.problem import Problem
from slsqp_jax.sqpdax.types import Aux
from tests.sqpdax.conftest import make_shifted_box_quadratic
from tests.sqpdax.lagrangian.conftest import make_problem
from tests.sqpdax.subproblem.solver.conftest import unbounded_box

# Back-compat alias used by the existing CommonMinimiser tests.
ActiveSetLineSearchStub = ActiveSetLineSearchMinimiser


def make_unconstrained_quadratic(*, n: int = 2) -> Problem:
    """``f(x) = ‖x‖²`` with exact HVP and no constraints / bounds."""
    lb, ub = unbounded_box(n)
    return make_problem(n=n, meq=0, mineq=0, lb=lb, ub=ub, with_curvature=True)


def make_equality_quadratic(*, n: int = 2, with_curvature: bool = True) -> Problem:
    """``f(x) = ‖x‖²`` subject to ``x₀ + x₁ = 1`` (unbounded).

    With ``with_curvature=False`` the problem exposes no HVPs, so minimisers
    built on it carry an L-BFGS secant.
    """
    lb, ub = unbounded_box(n)
    return make_problem(
        n=n, meq=1, mineq=0, lb=lb, ub=ub, with_curvature=with_curvature
    )


def make_scaled_quartic(*, with_curvature: bool = True) -> Problem:
    """Badly scaled ``Σ wᵢ xᵢ² + ¼ x₀⁴`` on ``Σ xᵢ = 1`` with ``x₀ >= 0.2``.

    ``w = (1, 10, 100)`` makes the Hessian diagonal span two orders of
    magnitude and the quartic term makes it iterate-dependent, so exact
    curvature, the secant and both preconditioners behave differently while
    sharing one minimiser (the inequality and the box are inactive there).

    Parameters
    ----------
    with_curvature
        Whether the problem exposes exact objective / constraint HVPs.

    Returns
    -------
    Problem
        Three-variable NLP with one equality and one inequality.
    """
    w = jnp.array([1.0, 10.0, 100.0])
    e0 = jnp.array([1.0, 0.0, 0.0])

    def fn(x: Array) -> tuple[Array, Aux]:
        return (jnp.sum(w * x**2) + 0.25 * x[0] ** 4, None)

    def grad(x: Array) -> Array:
        return 2.0 * w * x + x[0] ** 3 * e0

    def hvp(x: Array, v: Array) -> Array:
        return 2.0 * w * v + 3.0 * x[0] ** 2 * v[0] * e0

    def zero_hvp(m: int):
        return lambda x, v: jnp.zeros((m, x.shape[-1]), dtype=x.dtype)

    lb = jnp.full(3, -5.0)
    ub = jnp.full(3, 5.0)
    return Problem(
        fn=fn,
        grad=grad,
        hvp=hvp if with_curvature else None,
        eq_fn=lambda x: jnp.array([jnp.sum(x) - 1.0]),
        ineq_fn=lambda x: jnp.array([0.2 - x[0]]),
        eq_fn_jac=lambda x: jnp.ones((1, 3), dtype=x.dtype),
        ineq_fn_jac=lambda x: -e0[None, :].astype(x.dtype),
        eq_fn_hvp=zero_hvp(1) if with_curvature else None,
        ineq_fn_hvp=zero_hvp(1) if with_curvature else None,
        lb=lb,
        ub=ub,
        null_lb=jnp.zeros(3, dtype=bool),
        null_ub=jnp.zeros(3, dtype=bool),
        n=3,
        meq=1,
        mineq=1,
    )


# --- trust-funnel problems and shared runs -----------------------------------------

QUARTIC_X_STAR = [0.8684468143545903, 0.11959380513219049, 0.01195938051321905]


def make_box_quadratic() -> Problem:
    """``min ‖x − c‖²`` with one active inequality and one active bound."""
    return make_shifted_box_quadratic(n=3)[0]


def make_bound_only_quadratic() -> Problem:
    """``min ‖x‖²`` on ``[0.5, 2] × [−1, 3]``; solution ``(0.5, 0)``."""
    return make_problem(
        n=2,
        meq=0,
        mineq=0,
        lb=jnp.array([0.5, -1.0]),
        ub=jnp.array([2.0, 3.0]),
        with_curvature=True,
    )


def make_infeasible_problem() -> Problem:
    """``min ‖x‖²`` with ``x₀ ≤ 0`` and ``x₀ ≥ 1``: no feasible point."""
    n = 2

    def zero_rows(m: int):
        return lambda x, *a: jnp.zeros((m, n), x.dtype)

    return Problem(
        fn=lambda x: (jnp.sum(x**2), None),
        grad=lambda x: 2.0 * x,
        hvp=lambda x, v: 2.0 * v,
        eq_fn=lambda x: jnp.zeros((0,), x.dtype),
        eq_fn_jac=zero_rows(0),
        eq_fn_hvp=zero_rows(0),
        ineq_fn=lambda x: x[:1],
        ineq_fn_jac=lambda x: jnp.array([[1.0, 0.0]], x.dtype),
        ineq_fn_hvp=zero_rows(1),
        lb=jnp.array([1.0, -jnp.inf]),
        ub=jnp.array([jnp.inf, jnp.inf]),
        null_lb=jnp.array([False, True]),
        null_ub=jnp.array([True, True]),
        n=n,
        meq=0,
        mineq=1,
    )


# ``x0`` / ``x_star`` are plain lists: parametrize arguments are built at
# import time, before the x64 contexts of the convergence tests are entered.
CONVERGENCE_CASES: dict[str, tuple[Callable[[], Problem], list[float], list[float]]] = {
    "box": (make_box_quadratic, [0.5, 0.0, 0.5], [0.9, -1.0, 0.0]),
    # Far outside the box: slacks are reset on the way in.
    "box-far": (make_box_quadratic, [3.0, -5.0, 0.0], [0.9, -1.0, 0.0]),
    "quartic": (make_scaled_quartic, [0.5, 0.3, 0.2], QUARTIC_X_STAR),
    "equality": (make_equality_quadratic, [0.25, 0.25], [0.5, 0.5]),
    "bound": (make_bound_only_quadratic, [1.0, 1.0], [0.5, 0.0]),
    "unconstrained": (make_unconstrained_quadratic, [1.0, 1.0], [0.0, 0.0]),
}


def constraint_residual(problem: Problem, primal: InteriorPointPrimal) -> jax.Array:
    """``c(x, s)`` stacked over inequalities and live bounds."""
    x, slack = primal.x, primal.slack
    return jnp.concatenate(
        [
            problem.ineq_fn(x) + slack.s,
            jnp.where(problem.null_lb, 0.0, problem.lb - x + slack.s_lb),
            jnp.where(problem.null_ub, 0.0, x - problem.ub + slack.s_ub),
        ]
    )


def funnel_minimiser(**kwargs: Any) -> TrustFunnelInteriorPointMinimiser:
    """Funnel minimiser with the ``atol`` / ``μ₀`` used throughout the tests."""
    kwargs.setdefault("atol", 1e-6)
    kwargs.setdefault("initial_mu", 0.1)
    return TrustFunnelInteriorPointMinimiser(**kwargs)


@dataclass(frozen=True)
class FunnelRun:
    """One float64 run of the default funnel minimiser on a convergence case.

    Built once per session by :func:`funnel_run` and shared by every test
    that only inspects a converged trajectory, so the driver is compiled and
    run a single time per problem.
    """

    case: str
    problem: Problem
    x0: list[float]
    x_star: list[float]
    sol: Any
    diagnostics: MemoryDiagnosticsHandler
    metrics: TrustFunnelTerminationMetrics

    @property
    def n_steps(self) -> int:
        return int(self.sol.stats["num_steps"])

    @property
    def steps(self) -> list:
        """Per-step ``minimiser`` diagnostic records, in order."""
        return self.diagnostics.select(name="minimiser", kind="step")

    @property
    def funnel_steps(self) -> list:
        """Per-step ``funnel_step`` records of the subproblem solver."""
        return self.diagnostics.select(name="minimiser.subproblem", kind="funnel_step")


def run_funnel_case(case: str, **options: Any) -> FunnelRun:
    """Run the default funnel minimiser on ``CONVERGENCE_CASES[case]`` in x64.

    A :class:`MemoryDiagnosticsHandler` is attached (it only adds host
    callbacks and leaves the iterates untouched) and the power-iteration
    budget of the diagnostic norm estimates is raised so the Lemma bounds
    reported in the ``funnel_step`` records are tight.
    """
    make_problem_fn, x0, x_star = CONVERGENCE_CASES[case]
    handler = MemoryDiagnosticsHandler()
    with jax.enable_x64(True):
        problem = make_problem_fn()
        sol = minimise(
            problem,
            funnel_minimiser(),
            jnp.asarray(x0),
            max_steps=120,
            throw=False,
            options={
                "logging": {"diagnostics": handler},
                "subproblem": {"norm_estimate_iters": 50},
                **options,
            },
        )
        metrics = sol.state.termination_metrics(
            sol.state._optimisation_context(problem)
        )
    handler.close()
    return FunnelRun(case, problem, x0, x_star, sol, handler, metrics)


@pytest.fixture(scope="session")
def funnel_run() -> Callable[[str], FunnelRun]:
    """Session-cached getter ``funnel_run(case) -> FunnelRun``.

    Each :data:`CONVERGENCE_CASES` entry is run at most once per session,
    whichever module asks first.
    """
    cache: dict[str, FunnelRun] = {}

    def get(case: str) -> FunnelRun:
        if case not in cache:
            cache[case] = run_funnel_case(case)
        return cache[case]

    return get


@dataclass(frozen=True)
class InfeasibleFunnelRun:
    """The float32 funnel run on :func:`make_infeasible_problem` with INFO logging.

    In float32 the incompatible problem ends through the y-iteration fixed
    point (the slacks cannot shrink far enough for ``χᵛ`` to vanish), which
    is the route that emits the streak warning; float64 reaches Step 8
    directly. The generic LSMR multiplier recovery is forced because the
    exact pseudo-inverse paths hand the large least-squares multipliers of
    the incompatible system to the ``κ_y`` cap instead of producing a
    y-streak.
    """

    problem: Problem
    sol: Any
    handler: MemoryHandler
    metrics: TrustFunnelTerminationMetrics


@pytest.fixture(scope="session")
def infeasible_funnel_run() -> InfeasibleFunnelRun:
    """Session-shared :class:`InfeasibleFunnelRun`."""
    handler = MemoryHandler()
    with jax.enable_x64(False):
        problem = make_infeasible_problem()
        sol = minimise(
            problem,
            funnel_minimiser(),
            jnp.array([0.5, 0.3]),
            max_steps=80,
            throw=False,
            options={
                "logging": {"level": "INFO", "handler": handler},
                "subproblem": {
                    "tangential_solver": {"normal_equations": "generic"},
                    "multiplier_recovery": {"normal_equations": "generic"},
                },
            },
        )
        metrics = sol.state.termination_metrics(
            sol.state._optimisation_context(problem)
        )
    return InfeasibleFunnelRun(problem, sol, handler, metrics)
