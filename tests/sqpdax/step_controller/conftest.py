"""Fixtures shared by :mod:`slsqp_jax.sqpdax.step_controller` tests."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from slsqp_jax.sqpdax.active_set import ActiveSet
from slsqp_jax.sqpdax.barrier import LogBarrier
from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.lagrangian import InteriorPointLagrangian, Lagrangian
from slsqp_jax.sqpdax.merit import ConstraintViolation, NormMerit
from slsqp_jax.sqpdax.primal import InteriorPointPrimal, Primal, Slack
from slsqp_jax.sqpdax.problem import Problem, build_problem
from slsqp_jax.sqpdax.step_controller import (
    ArmijoLineSearch,
    CorrectionResult,
    LeastNormCorrection,
    StepCorrection,
    TrustFunnelManager,
    TrustRegionManager,
)
from slsqp_jax.sqpdax.subproblem.active_set import ActiveSetSubProblem
from slsqp_jax.sqpdax.subproblem.scaled_barrier import ScaledBarrierSubProblem
from slsqp_jax.sqpdax.subproblem.solver import (
    TrustFunnelSolverState,
    TrustRegionSolverState,
)
from tests.sqpdax.lagrangian.conftest import make_problem
from tests.sqpdax.subproblem.solver.conftest import (
    make_trust_funnel_state,
    make_trust_region_state,
    unbounded_box,
)


def make_unconstrained_quadratic(*, n: int = 2) -> Problem:
    """``f(x) = ‖x‖²`` with no constraints / bounds."""
    lb, ub = unbounded_box(n)
    return make_problem(n=n, meq=0, mineq=0, lb=lb, ub=ub)


def make_armijo(*, max_steps: int = 20, backtrack: float = 0.5) -> ArmijoLineSearch:
    """Armijo line search on the unconstrained quadratic merit."""
    return ArmijoLineSearch(
        merit=NormMerit(problem=make_unconstrained_quadratic()),
        max_steps=max_steps,
        backtrack=jnp.asarray(backtrack),
    )


def make_tr_manager() -> TrustRegionManager:
    """Trust-region manager on the unconstrained quadratic merit."""
    return TrustRegionManager(merit=NormMerit(problem=make_unconstrained_quadratic()))


def make_barrier_merit(problem: Problem, mu: float = 0.5) -> NormMerit:
    """Barrier function ``f(x, s) = f(x) − μ Σ log s`` as a feasibility-free merit."""
    barrier = LogBarrier(
        weight=jnp.asarray(mu), null_lb=problem.null_lb, null_ub=problem.null_ub
    )
    return NormMerit(
        problem=problem,
        barrier=barrier,
        feasibility_weight=jnp.asarray(0.0),
        norm=2,
    )


def make_funnel_manager(
    problem: Problem, *, mu: float = 0.5, **kwargs
) -> TrustFunnelManager:
    """Trust-funnel manager on ``problem`` with the barrier merit and ``v``."""
    return TrustFunnelManager(
        barrier_merit=make_barrier_merit(problem, mu),
        violation=ConstraintViolation(problem=problem, norm=2),
        **kwargs,
    )


def make_funnel_state(
    radius_v: float = 1.0,
    radius_f: float = 1.0,
    v_max: float = 10.0,
    **fields,
) -> TrustFunnelSolverState:
    """Cold funnel state with arbitrary fields overridden (``name=value``)."""
    state = make_trust_funnel_state(radius_v, radius_f, v_max)
    if not fields:
        return state
    names = tuple(fields)
    values = tuple(
        jnp.asarray(fields[k], getattr(state, k).dtype)
        if hasattr(getattr(state, k), "dtype")
        else fields[k]
        for k in names
    )
    return eqx.tree_at(lambda s: tuple(getattr(s, k) for k in names), state, values)


def make_tr_state(
    *,
    radius: float = 1.0,
    predicted_reduction: float = 1.0,
    on_boundary: bool = False,
) -> TrustRegionSolverState:
    """Cold trust-region state with prescribed prediction / boundary flag."""
    state = make_trust_region_state(radius)
    return TrustRegionSolverState(
        n_iter=state.n_iter,
        success=state.success,
        status=state.status,
        radius=state.radius,
        predicted_reduction=jnp.asarray(predicted_reduction),
        merit_penalty=state.merit_penalty,
        n_cg_iter=state.n_cg_iter,
        on_boundary=jnp.asarray(on_boundary),
        rho=jnp.asarray(1.0),
    )


# --- second-order correction ------------------------------------------------


def make_circle_problem(
    *,
    inequality: bool = False,
    lb: jax.Array | None = None,
    ub: jax.Array | None = None,
) -> Problem:
    """Nocedal & Wright Example 15.4: ``min 2(x₁²+x₂²−1) − x₁`` on the unit circle.

    The constraint is ``x₁² + x₂² − 1 = 0`` (``≤ 0`` with ``inequality=True``).
    Near ``x* = (1, 0)`` the full SQP step increases the ℓ1 merit: the
    Maratos example.
    """

    def circle(x):
        return jnp.array([x @ x - 1.0])

    rows: dict[str, Any] = (
        {"mineq": 1, "ineq_fn": circle} if inequality else {"meq": 1, "eq_fn": circle}
    )
    return build_problem(
        lambda x: 2.0 * (x @ x - 1.0) - x[0],
        n=2,
        lb=lb,
        ub=ub,
        autodiff_mode="jax",
        force_hvp_in_jax_mode=True,
        **rows,
    )


def zero_dual(problem: Problem) -> Dual:
    """All-zero multipliers sized for ``problem``."""
    n = problem.n
    return Dual(
        jnp.zeros(problem.meq), jnp.zeros(problem.mineq), jnp.zeros(n), jnp.zeros(n)
    )


def bind_active_set_correction(
    problem: Problem,
    x: jax.Array,
    *,
    enabled: bool = True,
    correction: StepCorrection | None = None,
    active_lb: tuple[bool, ...] | None = None,
    subproblem_type: type[ActiveSetSubProblem] = ActiveSetSubProblem,
) -> tuple[Primal, StepCorrection]:
    """``(x0, correction)`` bound to a working-set QP with every inequality active.

    ``active_lb`` adds lower bounds to the working set (none by default);
    ``subproblem_type`` swaps in an :class:`ActiveSetSubProblem` subclass.
    """
    lagrangian = Lagrangian(problem)
    primal = Primal(jnp.asarray(x))
    dual = zero_dual(problem)
    lower = jnp.zeros(problem.n, bool) if active_lb is None else jnp.array(active_lb)
    working_set = ActiveSet(
        meq=problem.meq,
        active_inequalities=jnp.ones(problem.mineq, bool),
        active_lb=lower,
        active_ub=jnp.zeros(problem.n, bool),
    )
    subproblem = subproblem_type(lagrangian(primal, dual), working_set)
    bound = (correction or LeastNormCorrection()).bind(
        subproblem=subproblem,
        lagrangian=lagrangian,
        dual=dual,
        enabled=jnp.asarray(enabled),
        default_tol=1e-8,
    )
    return primal, bound


def bind_interior_point_correction(
    problem: Problem,
    x: jax.Array,
    *,
    slack: float = 0.5,
    enabled: bool = True,
    correction: StepCorrection | None = None,
) -> tuple[InteriorPointPrimal, StepCorrection]:
    """``(x0, correction)`` bound to the scaled-barrier subproblem at ``(x, s)``."""
    barrier = LogBarrier(
        weight=jnp.asarray(0.1), null_lb=problem.null_lb, null_ub=problem.null_ub
    )
    lagrangian = InteriorPointLagrangian(
        problem, secant=None, barrier=barrier, primal_dual=True
    )
    n = problem.n
    primal = InteriorPointPrimal(
        x=jnp.asarray(x),
        slack=Slack(
            s=jnp.full(problem.mineq, slack), s_lb=jnp.ones(n), s_ub=jnp.ones(n)
        ),
    )
    dual = zero_dual(problem)
    subproblem = ScaledBarrierSubProblem(lagrangian(primal, dual))
    bound = (correction or LeastNormCorrection()).bind(
        subproblem=subproblem,
        lagrangian=lagrangian,
        dual=dual,
        enabled=jnp.asarray(enabled),
        default_tol=1e-8,
    )
    return primal, bound


def tangent_step(x0: Primal, t: float) -> Primal:
    """Step of length ``t`` tangent to the circle through ``x0.x`` (slacks fixed)."""
    x = x0.x
    d = t * jnp.array([-x[1], x[0]]) / jnp.linalg.norm(x)
    step = jax.tree.map(jnp.zeros_like, x0)
    return eqx.tree_at(lambda p: p.x, step, d)


def maratos_step(x0: Primal) -> Primal:
    """SQP step of :func:`make_circle_problem` from a point ``x0`` on the circle.

    With ``∇²L = I`` (``λ* = −3/2``) the step is the projected steepest
    descent ``d = −(tᵀ∇f) t`` along the unit tangent ``t``; at
    ``x0 = (cos θ, sin θ)`` it is ``sin θ (sin θ, −cos θ)``. It raises both
    ``f`` and ``|c|`` by ``sin² θ``, so every ℓ1 merit rejects it.
    """
    x = x0.x
    t = jnp.array([-x[1], x[0]]) / jnp.linalg.norm(x)
    grad = 4.0 * x - jnp.array([1.0, 0.0])
    return Primal(-(t @ grad) * t)


class NeverValidCorrection(StepCorrection):
    """Correction that is never valid (a controller must then ignore it)."""

    def bind(self, **kwargs):
        return self

    def __call__(self, x0, direction):
        return CorrectionResult(
            step=jax.tree.map(jnp.zeros_like, direction),
            valid=jnp.asarray(False),
            residual_norm=jnp.asarray(0.0),
            fraction=jnp.asarray(0.0),
        )
