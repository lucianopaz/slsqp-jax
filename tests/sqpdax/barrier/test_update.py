"""Unit tests for :mod:`slsqp_jax.sqpdax.barrier.update`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.barrier import (
    AdaptiveBarrierUpdate,
    BarrierUpdate,
    FunnelBarrierUpdate,
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
    ("kind_cls", "field"),
    [
        (MonotoneBarrierUpdate, "sigma"),
        (AdaptiveBarrierUpdate, "sigma"),
        (FunnelBarrierUpdate, "gamma_mu"),
    ],
    ids=["monotone", "adaptive", "funnel"],
)
def test_barrier_update_registers_on_family(kind_cls, field):
    """Concrete policies are discoverable via :meth:`BarrierUpdate.from_spec`."""
    assert BarrierUpdate._registry[kind_cls.kind] is kind_cls
    built = BarrierUpdate.from_spec({"kind": kind_cls.kind, field: 0.3})
    assert isinstance(built, kind_cls)
    assert getattr(built, field) == pytest.approx(0.3)


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
    updated, was_updated = policy.update(barrier, evaluated)
    if expect_reduce:
        assert was_updated
        assert jnp.allclose(updated.weight, sigma * mu)
    else:
        # Residual at a generic point is positive, so μ is held.
        assert not was_updated
        assert policy.optimality_residual(evaluated, mu) > 0
        assert jnp.allclose(updated.weight, mu)
    assert jnp.allclose(barrier.weight, mu)


def test_monotone_update_respects_mu_min():
    """Reduced ``μ`` is floored at ``mu_min``."""
    evaluated, barrier = make_evaluated_ip_lagrangian(weight=1e-10)
    policy = MonotoneBarrierUpdate(sigma=0.1, kappa_eps=1e12, mu_min=1e-9)
    updated, _ = policy.update(barrier, evaluated)
    assert jnp.allclose(updated.weight, 1e-9)


def test_adaptive_update_tracks_complementarity():
    """Adaptive policy sets ``μ = max(σ * (sᵀz/m), mu_min)``."""
    evaluated, barrier = make_evaluated_ip_lagrangian(weight=5.0)
    policy = AdaptiveBarrierUpdate(sigma=0.25, mu_min=1e-12)
    comp = policy.complementarity(evaluated)
    updated, was_updated = policy.update(barrier, evaluated)
    assert was_updated
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
    updated, _ = policy.update(barrier, evaluated)
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
        FunnelBarrierUpdate(zeta1=0.99, zeta2=1e6, alpha=1.0, beta=1.0),
    ):
        updated, _ = policy.update(barrier, evaluated)
        assert isinstance(updated, LogBarrier)
        assert jnp.array_equal(updated.null_lb, barrier.null_lb)
        assert jnp.array_equal(updated.null_ub, barrier.null_ub)


# --------------------------------------------------------------------------- #
# FunnelBarrierUpdate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gamma_mu": 1.0},
        {"gamma_mu": 0.0},
        {"mu_min": 0.0},
        {"zeta1": 1.0},
        {"alpha": 0.5},
        {"zeta2": 0.0},
        {"beta": 0.0},
        {"kappa_fb_max": 1.0},
        {"kappa_fb_scale": 0.0},
        {"kappa_fb_power": 0.0},
        {"kappa_y_scale": 0.0},
        {"kappa_D_scale": -1.0},
    ],
    ids=lambda kw: "-".join(kw),
)
def test_funnel_check_init_rejects_invalid_constants(kwargs):
    with pytest.raises(ValueError):
        FunnelBarrierUpdate(**kwargs)


@pytest.mark.parametrize(
    ("pi_f", "v", "expect_solved"),
    [
        (0.0, 0.0, True),
        (0.049, 0.09, True),  # both just inside ε_π = 0.05, ε_v = 0.1
        (0.051, 0.0, False),  # stationarity fails
        (0.0, 0.11, False),  # violation fails
        (1.0, 1.0, False),
    ],
    ids=["exact", "inside", "pi-fails", "v-fails", "both-fail"],
)
def test_funnel_update_reduces_mu_iff_both_tolerances_met(pi_f, v, expect_solved):
    """``μ`` drops by ``γ_μ`` iff ``πᶠ ≤ ζ₁ μ^α`` and ``v ≤ ζ₂ μ^β`` (3.15a)/(5.3)."""
    mu = 0.1
    evaluated, barrier = make_evaluated_ip_lagrangian(weight=mu)
    policy = FunnelBarrierUpdate(
        gamma_mu=0.2, zeta1=0.5, alpha=1.0, zeta2=1.0, beta=1.0
    )
    assert float(policy.eps_pi(mu)) == pytest.approx(0.05)
    assert float(policy.eps_v(mu)) == pytest.approx(0.1)
    updated, solved = policy.update(
        barrier, evaluated, pi_f=jnp.asarray(pi_f), v=jnp.asarray(v)
    )
    assert bool(solved) == expect_solved
    assert bool(policy.solved(mu, jnp.asarray(pi_f), jnp.asarray(v))) == expect_solved
    expected = 0.2 * mu if expect_solved else mu
    assert float(updated.weight) == pytest.approx(expected)
    assert float(barrier.weight) == pytest.approx(mu)  # input untouched


def test_funnel_update_respects_mu_min():
    evaluated, barrier = make_evaluated_ip_lagrangian(weight=1e-10)
    policy = FunnelBarrierUpdate(gamma_mu=0.1, mu_min=1e-9)
    updated, solved = policy.update(
        barrier, evaluated, pi_f=jnp.asarray(0.0), v=jnp.asarray(0.0)
    )
    assert bool(solved)
    assert float(updated.weight) == pytest.approx(1e-9)


def test_funnel_update_defaults_pi_f_and_v_from_lagrangian():
    """Omitted ``πᶠ``/``v`` are recomputed from ``lag`` (Definition 1.2, ``n = 0``)."""
    from slsqp_jax.sqpdax.subproblem.funnel_barrier import FunnelBarrierSubProblem

    mu = 0.5
    evaluated, barrier = make_evaluated_ip_lagrangian(weight=mu)
    sub = FunnelBarrierSubProblem(evaluated)
    pi_f = float(sub.pi_f(sub._zero_primal(), evaluated.dual))
    v = float(sub.violation())
    assert pi_f > 0.0 and v > 0.0

    # Violation tolerance just above / below the recomputed ``v`` (πᶠ supplied).
    loose = FunnelBarrierUpdate(zeta1=0.99, alpha=1.0, zeta2=2.0 * v / mu, beta=1.0)
    tight = FunnelBarrierUpdate(zeta1=0.99, alpha=1.0, zeta2=0.5 * v / mu, beta=1.0)
    assert float(loose.eps_v(mu)) > v > float(tight.eps_v(mu))
    _, solved_loose = loose.update(barrier, evaluated, pi_f=jnp.asarray(0.0))
    _, solved_tight = tight.update(barrier, evaluated, pi_f=jnp.asarray(0.0))
    assert bool(solved_loose) and not bool(solved_tight)
    # Stationarity recomputed (``v`` supplied): solved iff πᶠ ≤ ζ₁ μ.
    _, solved_pi = FunnelBarrierUpdate(zeta1=0.99, zeta2=1e6).update(
        barrier, evaluated, v=jnp.asarray(0.0)
    )
    assert bool(solved_pi) == (pi_f <= 0.99 * mu)


@pytest.mark.parametrize(
    "schedule, limit",
    [
        ("eps_pi", 0.0),
        ("eps_v", 0.0),
        ("kappa_fbn", 0.0),
        ("kappa_fbt", 0.0),
        ("kappa_y", jnp.inf),
        ("kappa_D", jnp.inf),
    ],
)
def test_funnel_schedules_are_monotone_with_the_paper_limits(schedule, limit):
    """Table 1 schedules satisfy (5.1)–(5.2): monotone in ``μ`` with the stated limits."""
    policy = FunnelBarrierUpdate()
    mus = jnp.logspace(0, -10, 11)
    values = jnp.stack([getattr(policy, schedule)(mu) for mu in mus])
    assert jnp.all(values > 0.0)
    if limit == 0.0:
        assert jnp.all(jnp.diff(values) <= 0.0)
        assert float(values[-1]) < 1e-8
    else:
        assert jnp.all(jnp.diff(values) >= 0.0)
        assert float(values[-1]) > 1e8


def test_funnel_fraction_to_boundary_schedule_is_capped():
    policy = FunnelBarrierUpdate(
        kappa_fb_max=0.1, kappa_fb_scale=1.0, kappa_fb_power=1.0
    )
    assert float(policy.kappa_fbn(10.0)) == pytest.approx(0.1)
    assert float(policy.kappa_fbn(0.01)) == pytest.approx(0.01)
    assert float(policy.kappa_fbt(0.01)) == pytest.approx(float(policy.kappa_fbn(0.01)))


def test_funnel_defaults_satisfy_53():
    """Default tolerances have the (5.3) form with ``ζ₁ ∈ (0,1)``, ``α ≥ 1``, ``ζ₂, β > 0``."""
    policy = FunnelBarrierUpdate()
    assert 0.0 < policy.zeta1 < 1.0
    assert policy.alpha >= 1.0
    assert policy.zeta2 > 0.0 and policy.beta > 0.0
    for mu in (1.0, 1e-2, 1e-6):
        assert float(policy.eps_pi(mu)) <= policy.zeta1 * mu**policy.alpha + 1e-12
        assert float(policy.eps_v(mu)) <= policy.zeta2 * mu**policy.beta + 1e-12
