"""Unit tests for :mod:`slsqp_jax.sqpdax.barrier.update`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.barrier import (
    AdaptiveBarrierUpdate,
    BarrierUpdate,
    LogBarrier,
    MonotoneBarrierUpdate,
)
from slsqp_jax.sqpdax.barrier.update import _inf_norm
from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.primal import InteriorPointPrimal, Slack
from tests.sqpdax.lagrangian.conftest import make_problem

from .conftest import make_evaluated_ip_lagrangian, make_log_barrier


@pytest.mark.parametrize(
    ("v", "expected"),
    [
        (jnp.array([1.0, -3.0, 2.0]), 3.0),
        (jnp.zeros((0,)), 0.0),
        (jnp.array([-0.5]), 0.5),
    ],
    ids=["mixed", "empty", "scalar"],
)
def test_inf_norm(v, expected):
    """``_inf_norm`` is the max absolute entry, with empty → 0."""
    assert jnp.allclose(_inf_norm(v), expected)


@pytest.mark.parametrize(
    "kind_cls",
    [MonotoneBarrierUpdate, AdaptiveBarrierUpdate],
    ids=["monotone", "adaptive"],
)
def test_barrier_update_registers_on_family(kind_cls):
    """Concrete policies are discoverable via :meth:`BarrierUpdate.from_spec`."""
    assert BarrierUpdate._registry[kind_cls.kind] is kind_cls
    built = BarrierUpdate.from_spec({"kind": kind_cls.kind, "sigma": 0.3})
    assert isinstance(built, kind_cls)
    assert built.sigma == pytest.approx(0.3)


def test_complementarity_averages_active_pairs_only():
    """Null bound pairs are excluded from average complementarity."""
    problem = make_problem(
        lb=jnp.array([0.0, -jnp.inf]),
        ub=jnp.array([jnp.inf, 3.0]),
    )
    # null_lb = [False, True], null_ub = [True, False] with make_problem masks.
    assert bool(problem.null_lb[1])
    assert bool(problem.null_ub[0])

    primal = InteriorPointPrimal(
        x=jnp.array([0.5, 0.5]),
        slack=Slack(
            s=jnp.array([2.0, 4.0]),
            s_lb=jnp.array([1.0, 10.0]),
            s_ub=jnp.array([10.0, 3.0]),
        ),
    )
    dual = Dual(
        eq_multipliers=jnp.array([0.0]),
        ineq_multipliers=jnp.array([0.5, 0.25]),
        lb_multipliers=jnp.array([2.0, 99.0]),
        ub_multipliers=jnp.array([99.0, 1.0]),
    )
    evaluated, _ = make_evaluated_ip_lagrangian(
        problem=problem, primal=primal, dual=dual, weight=1.0
    )
    policy = MonotoneBarrierUpdate()
    # Active: ineq (2*0.5 + 4*0.25) + lb0 (1*2) + ub1 (3*1) = 1+1+2+3 = 7
    # m = 2 ineq + 1 lb + 1 ub = 4
    assert jnp.allclose(policy.complementarity(evaluated), 7.0 / 4.0)


def test_optimality_residual_matches_manual_inf_norms():
    """``E(x,s;μ)`` is the max of stationarity, feasibility, complementarity."""
    evaluated, barrier = make_evaluated_ip_lagrangian(weight=0.5)
    policy = MonotoneBarrierUpdate()
    mu = barrier.weight

    stationarity = jnp.max(jnp.abs(evaluated.x_grad))
    feas = evaluated.dual_grad
    feasibility = jnp.max(
        jnp.abs(
            jnp.concatenate(
                [
                    feas.eq_multipliers,
                    feas.ineq_multipliers,
                    feas.lb_multipliers,
                    feas.ub_multipliers,
                ]
            )
        )
    )
    slack, dual = evaluated.slack, evaluated.dual
    comps = jnp.concatenate(
        [
            slack.s * dual.ineq_multipliers - mu,
            jnp.where(evaluated.null_lb, 0.0, slack.s_lb * dual.lb_multipliers - mu),
            jnp.where(evaluated.null_ub, 0.0, slack.s_ub * dual.ub_multipliers - mu),
        ]
    )
    complementarity = jnp.max(jnp.abs(comps))
    expected = jnp.max(jnp.stack([stationarity, feasibility, complementarity]))
    assert jnp.allclose(policy.optimality_residual(evaluated, mu), expected)


@pytest.mark.parametrize(
    ("kappa_eps", "sigma", "mu", "expect_reduce"),
    [
        (1e6, 0.2, 1.0, True),  # huge tolerance → always "solved"
        (0.0, 0.2, 1.0, False),  # zero tolerance → never reduce unless E==0
    ],
    ids=["reduce", "hold"],
)
def test_monotone_update_reduce_or_hold(
    kappa_eps: float, sigma: float, mu: float, expect_reduce: bool
):
    """Monotone policy reduces ``μ`` only when ``E ≤ kappa_eps * μ``."""
    evaluated, barrier = make_evaluated_ip_lagrangian(weight=mu)
    policy = MonotoneBarrierUpdate(sigma=sigma, kappa_eps=kappa_eps, mu_min=1e-12)
    updated = policy.update(barrier, evaluated)
    if expect_reduce:
        assert jnp.allclose(updated.weight, sigma * mu)
    else:
        # Residual at a generic point is positive, so μ is held.
        assert policy.optimality_residual(evaluated, mu) > 0
        assert jnp.allclose(updated.weight, mu)
    assert jnp.allclose(barrier.weight, mu)


def test_monotone_update_respects_mu_min():
    """Reduced ``μ`` is floored at ``mu_min``."""
    evaluated, barrier = make_evaluated_ip_lagrangian(weight=1e-10)
    policy = MonotoneBarrierUpdate(sigma=0.1, kappa_eps=1e12, mu_min=1e-9)
    updated = policy.update(barrier, evaluated)
    assert jnp.allclose(updated.weight, 1e-9)


def test_adaptive_update_tracks_complementarity():
    """Adaptive policy sets ``μ = max(σ * (sᵀz/m), mu_min)``."""
    evaluated, barrier = make_evaluated_ip_lagrangian(weight=5.0)
    policy = AdaptiveBarrierUpdate(sigma=0.25, mu_min=1e-12)
    comp = policy.complementarity(evaluated)
    updated = policy.update(barrier, evaluated)
    assert jnp.allclose(updated.weight, 0.25 * comp)
    assert jnp.allclose(barrier.weight, 5.0)


def test_adaptive_update_respects_mu_min():
    """Adaptive ``μ`` is floored at ``mu_min`` when complementarity is tiny."""
    problem = make_problem()
    primal = InteriorPointPrimal(
        x=jnp.array([0.5, 0.5]),
        slack=Slack(
            s=jnp.full((problem.mineq,), 1e-8),
            s_lb=jnp.full((problem.n,), 1e-8),
            s_ub=jnp.full((problem.n,), 1e-8),
        ),
    )
    dual = Dual(
        eq_multipliers=jnp.zeros((problem.meq,)),
        ineq_multipliers=jnp.full((problem.mineq,), 1e-8),
        lb_multipliers=jnp.full((problem.n,), 1e-8),
        ub_multipliers=jnp.full((problem.n,), 1e-8),
    )
    evaluated, barrier = make_evaluated_ip_lagrangian(
        problem=problem, primal=primal, dual=dual, weight=1.0
    )
    policy = AdaptiveBarrierUpdate(sigma=0.2, mu_min=1e-6)
    updated = policy.update(barrier, evaluated)
    assert jnp.allclose(updated.weight, 1e-6)


def test_update_preserves_barrier_masks():
    """``tree_at`` only replaces ``weight``; null masks stay put."""
    problem = make_problem(
        lb=jnp.array([0.0, -jnp.inf]),
        ub=jnp.array([jnp.inf, 3.0]),
    )
    barrier = make_log_barrier(problem, weight=1.0)
    evaluated, _ = make_evaluated_ip_lagrangian(problem=problem, weight=1.0)
    for policy in (
        MonotoneBarrierUpdate(kappa_eps=1e6),
        AdaptiveBarrierUpdate(),
    ):
        updated = policy.update(barrier, evaluated)
        assert isinstance(updated, LogBarrier)
        assert jnp.array_equal(updated.null_lb, barrier.null_lb)
        assert jnp.array_equal(updated.null_ub, barrier.null_ub)
