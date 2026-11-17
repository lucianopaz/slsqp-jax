"""Fixtures shared by :mod:`slsqp_jax.sqpdax.minimiser` tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import Array

from slsqp_jax.sqpdax.barrier import FunnelBarrierUpdate
from slsqp_jax.sqpdax.logging import MemoryDiagnosticsHandler, MemoryHandler
from slsqp_jax.sqpdax.minimiser import (
    ActiveSetLineSearchMinimiser,
    TrustFunnelInteriorPointMinimiser,
    TrustFunnelTerminationMetrics,
    minimise,
)
from slsqp_jax.sqpdax.primal import InteriorPointPrimal, Primal
from slsqp_jax.sqpdax.problem import Problem
from slsqp_jax.sqpdax.step_controller import StepResult
from slsqp_jax.sqpdax.subproblem.solver import ACTIVE_SET_QP_RESULTS, RESULTS
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


# --- Hock-Schittkowski problems ----------------------------------------------------
#
# Written inline, with analytic gradients / Jacobians and no HVPs (the LRH model
# is a secant). Inequalities follow the sqpdax convention ``h(x) <= 0``; ``None``
# bounds are encoded with infinities and the matching ``null_*`` masks.


def _hs_problem(
    *,
    fn: Callable[[Array], Array],
    grad: Callable[[Array], Array],
    eq: tuple[Callable[[Array], Array], Callable[[Array], Array]] | None = None,
    ineq: tuple[Callable[[Array], Array], Callable[[Array], Array]] | None = None,
    lb: list[float],
    ub: list[float],
) -> Problem:
    """Assemble a no-HVP :class:`Problem` from ``(fn, jac)`` pairs."""
    n = len(lb)
    lb_a, ub_a = jnp.asarray(lb), jnp.asarray(ub)
    empty = (
        lambda x: jnp.zeros((0,), x.dtype),
        lambda x: jnp.zeros((0, n), x.dtype),
    )
    eq_fn, eq_jac = eq if eq is not None else empty
    ineq_fn, ineq_jac = ineq if ineq is not None else empty
    return Problem(
        fn=lambda x: (fn(x), None),
        grad=grad,
        hvp=None,
        eq_fn=eq_fn,
        ineq_fn=ineq_fn,
        eq_fn_jac=eq_jac,
        ineq_fn_jac=ineq_jac,
        eq_fn_hvp=None,
        ineq_fn_hvp=None,
        lb=lb_a,
        ub=ub_a,
        null_lb=jnp.isinf(lb_a) & (lb_a < 0),
        null_ub=jnp.isinf(ub_a) & (ub_a > 0),
        n=n,
        meq=0 if eq is None else int(eq_fn(jnp.zeros(n)).shape[0]),
        mineq=0 if ineq is None else int(ineq_fn(jnp.zeros(n)).shape[0]),
    )


def make_hs21() -> Problem:
    """HS21: ``0.01 x₀² + x₁² − 100`` s.t. ``10 x₀ − x₁ ≥ 10``, boxed.

    Solution ``(2, 0)`` with ``f* = −99.96``.
    """
    return _hs_problem(
        fn=lambda x: 0.01 * x[0] ** 2 + x[1] ** 2 - 100.0,
        grad=lambda x: jnp.stack([0.02 * x[0], 2.0 * x[1]]),
        ineq=(
            lambda x: jnp.array([10.0 - 10.0 * x[0] + x[1]]),
            lambda x: jnp.array([[-10.0, 1.0]], x.dtype),
        ),
        lb=[2.0, -50.0],
        ub=[50.0, 50.0],
    )


def make_hs35() -> Problem:
    """HS35 (Beale): convex quadratic on ``x₀ + x₁ + 2 x₂ ≤ 3``, ``x ≥ 0``.

    Solution ``(4/3, 7/9, 4/9)`` with ``f* = 1/9``.
    """
    return _hs_problem(
        fn=lambda x: (
            9.0
            - 8.0 * x[0]
            - 6.0 * x[1]
            - 4.0 * x[2]
            + 2.0 * x[0] ** 2
            + 2.0 * x[1] ** 2
            + x[2] ** 2
            + 2.0 * x[0] * x[1]
            + 2.0 * x[0] * x[2]
        ),
        grad=lambda x: jnp.stack(
            [
                -8.0 + 4.0 * x[0] + 2.0 * x[1] + 2.0 * x[2],
                -6.0 + 4.0 * x[1] + 2.0 * x[0],
                -4.0 + 2.0 * x[2] + 2.0 * x[0],
            ]
        ),
        ineq=(
            lambda x: jnp.array([x[0] + x[1] + 2.0 * x[2] - 3.0]),
            lambda x: jnp.array([[1.0, 1.0, 2.0]], x.dtype),
        ),
        lb=[0.0, 0.0, 0.0],
        ub=[jnp.inf] * 3,
    )


def make_hs71() -> Problem:
    """HS71: ``x₀x₃(x₀+x₁+x₂) + x₂`` s.t. ``Πx ≥ 25``, ``‖x‖² = 40``, ``1 ≤ x ≤ 5``.

    Solution ``(1, 4.74299963, 3.82114998, 1.37940829)`` with
    ``f* = 17.0140173``.
    """

    def grad(x: Array) -> Array:
        return jnp.stack(
            [
                x[3] * (2.0 * x[0] + x[1] + x[2]),
                x[0] * x[3],
                x[0] * x[3] + 1.0,
                x[0] * (x[0] + x[1] + x[2]),
            ]
        )

    def prod_jac(x: Array) -> Array:
        return -jnp.stack(
            [
                x[1] * x[2] * x[3],
                x[0] * x[2] * x[3],
                x[0] * x[1] * x[3],
                x[0] * x[1] * x[2],
            ]
        )[None, :]

    return _hs_problem(
        fn=lambda x: x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2],
        grad=grad,
        eq=(
            lambda x: jnp.array([jnp.sum(x**2) - 40.0]),
            lambda x: 2.0 * x[None, :],
        ),
        ineq=(lambda x: jnp.array([25.0 - jnp.prod(x)]), prod_jac),
        lb=[1.0] * 4,
        ub=[5.0] * 4,
    )


def make_hs76() -> Problem:
    """HS76: convex quadratic with three linear inequalities and ``x ≥ 0``.

    Solution ``(3/11, 23/11, 0, 6/11)`` with ``f* = −4.681818181818``.
    """
    a = jnp.array([[1.0, 2.0, 1.0, 1.0], [3.0, 1.0, 2.0, -1.0], [0.0, -1.0, -4.0, 0.0]])
    b = jnp.array([5.0, 4.0, -1.5])
    q = jnp.array([2.0, 1.0, 2.0, 1.0])
    c = jnp.array([-1.0, -3.0, 1.0, -1.0])
    # Off-diagonal part of the Hessian: -x₀x₂ + x₂x₃.
    return _hs_problem(
        fn=lambda x: (
            0.5 * jnp.sum(q * x**2) - x[0] * x[2] + x[2] * x[3] + jnp.dot(c, x)
        ),
        grad=lambda x: (
            q * x + c + jnp.stack([-x[2], jnp.zeros_like(x[0]), x[3] - x[0], x[2]])
        ),
        ineq=(lambda x: a @ x - b, lambda x: a.astype(x.dtype)),
        lb=[0.0] * 4,
        ub=[jnp.inf] * 4,
    )


# ``(factory, x0, x_star, f_star)``; plain lists for the same reason as below.
HS_CASES: dict[str, tuple[Callable[[], Problem], list[float], list[float], float]] = {
    "hs21": (make_hs21, [-1.0, -1.0], [2.0, 0.0], -99.96),
    "hs35": (
        make_hs35,
        [0.5, 0.5, 0.5],
        [4.0 / 3.0, 7.0 / 9.0, 4.0 / 9.0],
        1.0 / 9.0,
    ),
    "hs71": (
        make_hs71,
        [1.0, 5.0, 5.0, 1.0],
        [1.0, 4.74299963, 3.82114998, 1.37940829],
        17.0140172890,
    ),
    "hs76": (
        make_hs76,
        [0.5, 0.5, 0.5, 0.5],
        [3.0 / 11.0, 23.0 / 11.0, 0.0, 6.0 / 11.0],
        -51.5 / 11.0,
    ),
}


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


def make_zero_jacobian_infeasible_problem() -> Problem:
    """``min ‖x‖²`` subject to ``x₀² + 1 = 0``: infeasible, and at ``x = 0``
    both the constraint Jacobian and ``∇f`` vanish exactly.

    Starting there, every step is a y-iteration (``πᵛ = πᶠ = 0`` with
    ``v = 1``) in any precision, which makes the y-streak route of the funnel
    minimiser deterministic.
    """
    n = 2

    def zero_rows(m: int):
        return lambda x, *a: jnp.zeros((m, n), x.dtype)

    return Problem(
        fn=lambda x: (jnp.sum(x**2), None),
        grad=lambda x: 2.0 * x,
        hvp=lambda x, v: 2.0 * v,
        eq_fn=lambda x: x[:1] ** 2 + 1.0,
        eq_fn_jac=lambda x: jnp.array([[2.0 * x[0], 0.0]], x.dtype),
        eq_fn_hvp=lambda x, v: 2.0 * v[0] * jnp.array([[1.0, 0.0]], x.dtype),
        ineq_fn=lambda x: jnp.zeros((0,), x.dtype),
        ineq_fn_jac=zero_rows(0),
        ineq_fn_hvp=zero_rows(0),
        lb=jnp.array([-jnp.inf, -jnp.inf]),
        ub=jnp.array([jnp.inf, jnp.inf]),
        null_lb=jnp.array([True, True]),
        null_ub=jnp.array([True, True]),
        n=n,
        meq=1,
        mineq=0,
    )


def make_shifted_equality_problem() -> Problem:
    """``min x₀² + x₁² + x₁`` subject to ``x₀ = 4``.

    At ``x = 0`` the least-squares multiplier is ``y = 0``, so
    ``πᶠ = ‖P∇f‖ = 1`` while ``πᵛ = ‖Aᵀc‖ = 4`` and (3.15b) holds. With the
    normal step gated off this gives a y-iteration whose multiplier estimate
    fails the (3.16) health test.
    """
    n = 2

    def zero_rows(m: int):
        return lambda x, *a: jnp.zeros((m, n), x.dtype)

    return Problem(
        fn=lambda x: (jnp.sum(x**2) + x[1], None),
        grad=lambda x: 2.0 * x + jnp.array([0.0, 1.0], x.dtype),
        hvp=lambda x, v: 2.0 * v,
        eq_fn=lambda x: x[:1] - 4.0,
        eq_fn_jac=lambda x: jnp.array([[1.0, 0.0]], x.dtype),
        eq_fn_hvp=zero_rows(1),
        ineq_fn=lambda x: jnp.zeros((0,), x.dtype),
        ineq_fn_jac=zero_rows(0),
        ineq_fn_hvp=zero_rows(0),
        lb=jnp.array([-jnp.inf, -jnp.inf]),
        ub=jnp.array([jnp.inf, jnp.inf]),
        null_lb=jnp.array([True, True]),
        null_ub=jnp.array([True, True]),
        n=n,
        meq=1,
        mineq=0,
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

    The run ends with ``infeasible_stationary_point``, but *which* route gets
    there is rounding-dependent: ``Âᵀĉ`` is cancellation noise near the
    stationary point of the violation, so depending on the platform the
    noise either rounds ``πᵛ`` to exactly zero (y-iterations, ending through
    the streak fallback) or leaves it at ``~eps`` (tiny ``f``/``v`` steps,
    ending through the ``eps``-floored Step 8 test). Tests on this run must
    only assert what holds on both routes; the streak and the unhealthy
    y-iteration diagnostics are exercised deterministically on
    :func:`make_zero_jacobian_infeasible_problem` and
    :func:`make_shifted_equality_problem`. The generic LSMR multiplier
    recovery is forced so the large least-squares multipliers of the
    incompatible system hit the ``κ_y`` cap in a reproducible way, and the
    barrier update's KKT test is disabled (``kappa_eps=0``) so ``μ`` stays
    at ``μ₀`` long enough for them to do so (with the test on, ``μ`` drops
    on the second step and Step 8 ends the run before the cap is reached).
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
            funnel_minimiser(barrier_update=FunnelBarrierUpdate(kappa_eps=0.0)),
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


def _execute_synthetic_step(
    solver,
    problem,
    *,
    accepted: bool,
    qp_success: bool,
    qp_status,
    x,
    merit: float,
    step_size: float | None = None,
    proposed_step_norm: float = 1.0,
    accepted_by_fallback: bool = False,
    rounding_level: bool = False,
):
    """Exercise phase four with a controlled line-search/QP outcome."""
    ctx = solver._init_subproblem(problem)
    qp_result = ACTIVE_SET_QP_RESULTS.where(
        jnp.asarray(qp_success),
        ACTIVE_SET_QP_RESULTS.working_set_converged,
        ACTIVE_SET_QP_RESULTS.where(
            qp_status == RESULTS.max_steps_reached,
            ACTIVE_SET_QP_RESULTS.max_iter_reached,
            ACTIVE_SET_QP_RESULTS.kkt_solver_failure,
        ),
    )
    state = eqx.tree_at(
        lambda s: (s.success, s.status, s.qp_result),
        solver.solver_state,
        (jnp.asarray(qp_success), qp_status, qp_result),
    )
    result = StepResult(
        x=Primal(jnp.asarray(x)),
        accepted=jnp.asarray(accepted),
        merit_val=jnp.asarray(merit),
        solver_state=state,
        step_size=jnp.asarray(
            (1.0 if accepted else 0.0) if step_size is None else step_size
        ),
        proposed_step_norm=jnp.asarray(proposed_step_norm),
        accepted_by_fallback=jnp.asarray(accepted_by_fallback),
        rounding_level=jnp.asarray(rounding_level),
    )
    return solver._execute_step(ctx, solver.dual, result)


def _indefinite_quadratic():
    """``f(x) = 1/2 (x0^2 - 4 x1^2)`` with exact curvature, unconstrained."""
    w = jnp.array([1.0, -4.0])
    lb, ub = unbounded_box(2)
    return Problem(
        fn=lambda x: (0.5 * jnp.sum(w * x**2), None),
        grad=lambda x: w * x,
        hvp=lambda x, v: w * v,
        eq_fn=lambda x: jnp.zeros((0,), x.dtype),
        ineq_fn=lambda x: jnp.zeros((0,), x.dtype),
        eq_fn_jac=lambda x: jnp.zeros((0, 2), x.dtype),
        ineq_fn_jac=lambda x: jnp.zeros((0, 2), x.dtype),
        eq_fn_hvp=lambda x, v: jnp.zeros((0, 2), x.dtype),
        ineq_fn_hvp=lambda x, v: jnp.zeros((0, 2), x.dtype),
        lb=lb,
        ub=ub,
        null_lb=jnp.ones(2, dtype=bool),
        null_ub=jnp.ones(2, dtype=bool),
        n=2,
        meq=0,
        mineq=0,
    )
