"""Fixtures shared by :mod:`slsqp_jax.sqpdax.subproblem` tests."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import Array

from slsqp_jax.sqpdax.active_set import ActiveSet
from slsqp_jax.sqpdax.barrier import LogBarrier
from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.lagrangian.basic import Lagrangian
from slsqp_jax.sqpdax.lagrangian.evaluated import EvaluatedLagrangian
from slsqp_jax.sqpdax.lagrangian.interior_point import InteriorPointLagrangian
from slsqp_jax.sqpdax.primal import InteriorPointPrimal, Primal, Slack
from slsqp_jax.sqpdax.problem.basic import Problem
from slsqp_jax.sqpdax.subproblem.active_set import ActiveSetSubProblem
from slsqp_jax.sqpdax.subproblem.funnel_barrier import FunnelBarrierSubProblem
from slsqp_jax.sqpdax.subproblem.scaled_barrier import ScaledBarrierSubProblem
from tests.sqpdax.conftest import make_shifted_box_quadratic
from tests.sqpdax.lagrangian.conftest import (
    make_dual,
    make_ip_primal,
    make_primal,
    make_problem,
)

# Problems the trust-funnel component tests are parametrised over.
FUNNEL_PROBLEMS: dict[str, Callable[[], Problem]] = {
    "eq-ineq-finite": lambda: make_problem(),
    "eq-ineq-mixed-null": lambda: make_problem(
        lb=jnp.array([0.0, -jnp.inf]), ub=jnp.array([jnp.inf, 3.0])
    ),
    "shifted-box": lambda: make_shifted_box_quadratic(n=3)[0],
}


@functools.cache
def funnel_problem(name: str) -> Problem:
    """One shared :class:`Problem` instance per :data:`FUNNEL_PROBLEMS` entry.

    The problem callables are static leaves that hash by identity, so jitted
    helpers only reuse their compilation when every test hands them the
    *same* instance. Only for float32 modules: the bound arrays are created
    at first use and keep that dtype.
    """
    return FUNNEL_PROBLEMS[name]()


def make_active_set(
    meq: int,
    mineq: int,
    n: int,
    *,
    active_inequalities: tuple[bool, ...] | None = None,
    active_lb: tuple[bool, ...] | None = None,
    active_ub: tuple[bool, ...] | None = None,
) -> ActiveSet:
    """Build an :class:`ActiveSet` with optional explicit masks."""
    if active_inequalities is None:
        active_inequalities = tuple(i % 2 == 0 for i in range(mineq))
    if active_lb is None:
        active_lb = tuple(i % 2 == 0 for i in range(n))
    if active_ub is None:
        active_ub = tuple(i % 2 == 1 for i in range(n))
    return ActiveSet(
        meq=meq,
        active_inequalities=jnp.asarray(active_inequalities),
        active_lb=jnp.asarray(active_lb),
        active_ub=jnp.asarray(active_ub),
    )


def make_evaluated_lagrangian(
    *,
    problem: Problem | None = None,
    primal: Primal | None = None,
    dual: Dual | None = None,
) -> EvaluatedLagrangian[Primal]:
    """Evaluate a plain :class:`Lagrangian` at a default point."""
    if problem is None:
        problem = make_problem()
    if primal is None:
        primal = make_primal(n=problem.n)
    if dual is None:
        dual = make_dual(problem.n, problem.meq, problem.mineq)
    return Lagrangian(problem)(primal, dual)


def make_active_set_subproblem(
    *,
    problem: Problem | None = None,
    active_inequalities: tuple[bool, ...] | None = None,
    active_lb: tuple[bool, ...] | None = None,
    active_ub: tuple[bool, ...] | None = None,
) -> ActiveSetSubProblem:
    """Build an :class:`ActiveSetSubProblem` on the default quadratic NLP."""
    lag = make_evaluated_lagrangian(problem=problem)
    active = make_active_set(
        lag.meq,
        lag.mineq,
        lag.n,
        active_inequalities=active_inequalities,
        active_lb=active_lb,
        active_ub=active_ub,
    )
    return ActiveSetSubProblem(lag, active)


def make_zero_dual(n: int, meq: int, mineq: int) -> Dual:
    """All-zero dual used as a tangent / RHS block."""
    return Dual(
        eq_multipliers=jnp.zeros((meq,)),
        ineq_multipliers=jnp.zeros((mineq,)),
        lb_multipliers=jnp.zeros((n,)),
        ub_multipliers=jnp.zeros((n,)),
    )


def make_step(
    n: int = 2,
    meq: int = 1,
    mineq: int = 2,
    *,
    dx: Array | None = None,
) -> tuple[Primal, Dual]:
    """Plain primal-dual step for active-set tests."""
    if dx is None:
        dx = jnp.linspace(0.1, -0.2, n)
    return Primal(x=dx), make_dual(n, meq, mineq)


def make_ip_lagrangian(
    problem: Problem,
    *,
    weight: float = 0.5,
    dual_kkt_regularization: float = 0.0,
    primal_dual: bool = True,
) -> InteriorPointLagrangian:
    """Unevaluated interior-point Lagrangian with a log barrier of ``weight``."""
    barrier = LogBarrier(
        weight=jnp.asarray(weight),
        null_lb=problem.null_lb,
        null_ub=problem.null_ub,
    )
    return InteriorPointLagrangian(
        problem,
        secant=None,
        barrier=barrier,
        dual_kkt_regularization=dual_kkt_regularization,
        primal_dual=primal_dual,
    )


def make_ip_evaluated(
    *,
    problem: Problem | None = None,
    weight: float = 0.5,
    dual_kkt_regularization: float = 0.0,
    primal_dual: bool = True,
    primal: InteriorPointPrimal | None = None,
    dual: Dual | None = None,
):
    """Evaluate an interior-point Lagrangian (optionally primal-dual)."""
    if problem is None:
        problem = make_problem()
    lag = make_ip_lagrangian(
        problem,
        weight=weight,
        dual_kkt_regularization=dual_kkt_regularization,
        primal_dual=primal_dual,
    )
    if primal is None:
        primal = make_ip_primal(n=problem.n, mineq=problem.mineq)
    if dual is None:
        dual = make_dual(problem.n, problem.meq, problem.mineq)
    return lag(primal, dual)


def make_funnel_barrier_subproblem(
    *,
    problem: Problem | None = None,
    weight: float = 0.5,
    primal_dual: bool = True,
    kappa_fbn: float = 0.1,
    kappa_fbt: float = 0.1,
    primal: InteriorPointPrimal | None = None,
    dual: Dual | None = None,
) -> FunnelBarrierSubProblem:
    """Build a :class:`FunnelBarrierSubProblem` on an IP Lagrangian."""
    evaluated = make_ip_evaluated(
        problem=problem,
        weight=weight,
        primal_dual=primal_dual,
        primal=primal,
        dual=dual,
    )
    return FunnelBarrierSubProblem(evaluated, kappa_fbn=kappa_fbn, kappa_fbt=kappa_fbt)


@eqx.filter_jit
def _apply_on_funnel_subproblem(
    fn: Callable[..., Any],
    lagrangian: InteriorPointLagrangian,
    primal: InteriorPointPrimal,
    dual: Dual,
    kappa_fbn: float,
    kappa_fbt: float,
    args: tuple,
):
    """Evaluate the Lagrangian, build the funnel subproblem and run ``fn`` under jit."""
    sub = FunnelBarrierSubProblem(
        lagrangian(primal, dual), kappa_fbn=kappa_fbn, kappa_fbt=kappa_fbt
    )
    return fn(sub, *args)


@dataclass(frozen=True)
class FunnelCase:
    """A funnel subproblem together with the pieces needed to rebuild it under jit.

    An evaluated Lagrangian carries per-point closures, so a subproblem can
    never be a cache-friendly jit argument. :meth:`apply` instead re-evaluates
    the (unevaluated) Lagrangian inside the trace; with the shared instances
    of :func:`funnel_problem` every call with the same ``(problem,
    primal_dual, fn, static args)`` reuses one compilation across tests.
    """

    lagrangian: InteriorPointLagrangian
    primal: InteriorPointPrimal
    dual: Dual
    kappa_fbn: float
    kappa_fbt: float

    @functools.cached_property
    def sub(self) -> FunnelBarrierSubProblem:
        """Eagerly evaluated subproblem for reference computations."""
        return FunnelBarrierSubProblem(
            self.lagrangian(self.primal, self.dual),
            kappa_fbn=self.kappa_fbn,
            kappa_fbt=self.kappa_fbt,
        )

    @property
    def problem(self) -> Problem:
        return self.lagrangian.problem  # type: ignore[return-value]

    def zero_warm(self) -> tuple[InteriorPointPrimal, Dual]:
        """All-zero ``(primal, dual)`` warm start."""
        return jax.tree.map(jnp.zeros_like, (self.primal, self.dual))

    def apply(self, fn: Callable[..., Any], *args):
        """Run ``fn(sub, *args)`` under a shared jit.

        ``fn`` must be a module-level function (lambdas defined inside a
        test are new objects on every call and defeat the cache).
        """
        return _apply_on_funnel_subproblem(
            fn,
            self.lagrangian,
            self.primal,
            self.dual,
            self.kappa_fbn,
            self.kappa_fbt,
            args,
        )


def make_funnel_case(
    problem_name: str,
    *,
    weight: float = 0.5,
    primal_dual: bool = True,
    kappa_fbn: float = 0.1,
    kappa_fbt: float = 0.1,
    primal: InteriorPointPrimal | None = None,
    dual: Dual | None = None,
) -> FunnelCase:
    """:class:`FunnelCase` on the shared instance of ``FUNNEL_PROBLEMS[problem_name]``."""
    problem = funnel_problem(problem_name)
    if primal is None:
        primal = make_ip_primal(n=problem.n, mineq=problem.mineq)
    if dual is None:
        dual = make_dual(problem.n, problem.meq, problem.mineq)
    return FunnelCase(
        lagrangian=make_ip_lagrangian(problem, weight=weight, primal_dual=primal_dual),
        primal=primal,
        dual=dual,
        kappa_fbn=kappa_fbn,
        kappa_fbt=kappa_fbt,
    )


def dense_funnel_reference(
    sub: FunnelBarrierSubProblem,
) -> tuple[Array, Array, Array, Array]:
    """Assemble ``(ĝ, Ĥ, Â, ĉ)`` densely from ``P``, ``J(x, s)`` and ``G``.

    Null bound slacks are dead coordinates: ``P`` carries a zero there so
    every scaled object vanishes on them, matching the operator masks.
    """
    lag = sub.lagrangian
    n, meq, mineq = lag.n, lag.meq, lag.mineq
    S = lag.slack
    live_lb = ~lag.null_lb
    live_ub = ~lag.null_ub
    p = jnp.concatenate(
        [
            jnp.ones((n,)),
            S.s,
            jnp.where(live_lb, S.s_lb, 0.0),
            jnp.where(live_ub, S.s_ub, 0.0),
        ]
    )
    P = jnp.diag(p)

    m = meq + mineq + 2 * n
    N = n + mineq + 2 * n
    J = jnp.zeros((m, N))
    J = J.at[:meq, :n].set(lag.eq_fn_jac_val)
    J = J.at[meq : meq + mineq, :n].set(lag.ineq_fn_jac_val)
    J = J.at[meq : meq + mineq, n : n + mineq].set(jnp.eye(mineq))
    r0 = meq + mineq
    J = J.at[r0 : r0 + n, :n].set(-jnp.diag(live_lb.astype(J.dtype)))
    J = J.at[r0 : r0 + n, n + mineq : n + mineq + n].set(
        jnp.diag(live_lb.astype(J.dtype))
    )
    J = J.at[r0 + n :, :n].set(jnp.diag(live_ub.astype(J.dtype)))
    J = J.at[r0 + n :, n + mineq + n :].set(jnp.diag(live_ub.astype(J.dtype)))

    eye_n = jnp.eye(n)
    H_xx = jnp.stack([lag.hvp(eye_n[i]) for i in range(n)], axis=1)
    if lag.primal_dual:
        y = lag.dual
        D = jnp.concatenate(
            [
                y.ineq_multipliers / S.s,
                y.lb_multipliers / S.s_lb,
                y.ub_multipliers / S.s_ub,
            ]
        )
    else:
        eye_s = jnp.eye(mineq + 2 * n)
        D = jnp.stack(
            [
                lag.barrier.hvp(Slack.from_flat(eye_s[i], n, mineq)).flatten()[i]
                for i in range(mineq + 2 * n)
            ]
        )
    live_s = jnp.concatenate([jnp.ones((mineq,), bool), live_lb, live_ub])
    D = jnp.where(live_s, D, 0.0)
    G = jax.scipy.linalg.block_diag(H_xx, jnp.diag(D))

    g = jnp.concatenate([lag.grad_val, lag.barrier.grad_val.flatten()])
    g = jnp.where(p > 0.0, g, 0.0)
    c = lag.dual_grad.flatten()
    return P @ g, P @ G @ P, J @ P, c


def random_funnel_step(sub: FunnelBarrierSubProblem, seed: int) -> InteriorPointPrimal:
    """Random scaled primal step for ``sub`` (deterministic in ``seed``)."""
    lag = sub.lagrangian
    flat = jax.random.normal(jax.random.key(seed), (lag.n + lag.mineq + 2 * lag.n,))
    return InteriorPointPrimal.from_flat(0.3 * flat, lag.n, lag.mineq)


def random_funnel_dual(sub: FunnelBarrierSubProblem, seed: int) -> Dual:
    """Random dual vector for ``sub`` (deterministic in ``seed``)."""
    lag = sub.lagrangian
    flat = jax.random.normal(jax.random.key(seed), (lag.meq + lag.mineq + 2 * lag.n,))
    return Dual.from_flat(flat, lag.n, lag.mineq, lag.meq)


def make_scaled_barrier_subproblem(
    *,
    problem: Problem | None = None,
    weight: float = 0.5,
    dual_kkt_regularization: float = 0.0,
    tau: float = 0.995,
) -> ScaledBarrierSubProblem:
    """Build a :class:`ScaledBarrierSubProblem` on a primal-dual IP Lagrangian."""
    evaluated = make_ip_evaluated(
        problem=problem,
        weight=weight,
        dual_kkt_regularization=dual_kkt_regularization,
        primal_dual=True,
    )
    return ScaledBarrierSubProblem(evaluated, tau=tau)


def make_ip_step(
    n: int = 2,
    mineq: int = 2,
    meq: int = 1,
    *,
    fill: float = 0.1,
) -> tuple[InteriorPointPrimal, Dual]:
    """Interior-point primal-dual step with constant slack fills."""
    primal = InteriorPointPrimal(
        x=jnp.full((n,), fill),
        slack=Slack(
            s=jnp.full((mineq,), fill),
            s_lb=jnp.full((n,), fill),
            s_ub=jnp.full((n,), -fill),
        ),
    )
    return primal, make_dual(n, meq, mineq)


@pytest.fixture
def quadratic_problem() -> Problem:
    """Default 2-variable NLP used across subproblem tests."""
    return make_problem()
