"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.funnel_barrier`."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.primal import InteriorPointPrimal, Slack
from slsqp_jax.sqpdax.problem.basic import Problem
from slsqp_jax.sqpdax.subproblem.funnel_barrier import FunnelBarrierSubProblem
from tests.sqpdax.conftest import make_shifted_box_quadratic
from tests.sqpdax.lagrangian.conftest import make_dual, make_problem

from .conftest import make_funnel_barrier_subproblem, make_ip_evaluated

# ---------------------------------------------------------------------------
# problem factories
# ---------------------------------------------------------------------------

PROBLEMS: dict[str, Callable[[], Problem]] = {
    "eq-ineq-finite": lambda: make_problem(),
    "eq-ineq-mixed-null": lambda: make_problem(
        lb=jnp.array([0.0, -jnp.inf]), ub=jnp.array([jnp.inf, 3.0])
    ),
    "shifted-box": lambda: make_shifted_box_quadratic(n=3)[0],
}

problem_ids = pytest.mark.parametrize("problem_name", list(PROBLEMS))
primal_dual_ids = pytest.mark.parametrize(
    "primal_dual", [True, False], ids=["primal-dual", "primal"]
)


# ---------------------------------------------------------------------------
# dense reference assembly (tests only)
# ---------------------------------------------------------------------------


def dense_reference(
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


def random_step(sub: FunnelBarrierSubProblem, seed: int) -> InteriorPointPrimal:
    lag = sub.lagrangian
    flat = jax.random.normal(jax.random.key(seed), (lag.n + lag.mineq + 2 * lag.n,))
    return InteriorPointPrimal.from_flat(0.3 * flat, lag.n, lag.mineq)


def random_dual(sub: FunnelBarrierSubProblem, seed: int) -> Dual:
    lag = sub.lagrangian
    flat = jax.random.normal(jax.random.key(seed), (lag.meq + lag.mineq + 2 * lag.n,))
    return Dual.from_flat(flat, lag.n, lag.mineq, lag.meq)


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kappa_fbn": 0.0},
        {"kappa_fbn": 1.0},
        {"kappa_fbt": -0.1},
        {"kappa_fbt": 1.5},
    ],
)
def test_constructor_rejects_bad_kappas(kwargs: dict[str, float]):
    """Fraction-to-boundary constants must lie strictly inside ``(0, 1)``."""
    evaluated = make_ip_evaluated()
    with pytest.raises(ValueError, match="kappa_fb"):
        FunnelBarrierSubProblem(evaluated, **kwargs)


@primal_dual_ids
def test_constructor_accepts_both_slack_curvatures(primal_dual: bool):
    """Unlike the base class, both ``primal_dual`` settings are allowed."""
    sub = make_funnel_barrier_subproblem(primal_dual=primal_dual, kappa_fbn=0.25)
    assert sub.lagrangian.primal_dual is primal_dual
    assert sub.tau == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# operators and models against the dense reference
# ---------------------------------------------------------------------------


@problem_ids
@primal_dual_ids
def test_scaled_operators_match_dense(problem_name: str, primal_dual: bool):
    """``Ĥ w``, ``Â w`` and ``Âᵀ y`` equal their dense counterparts."""
    sub = make_funnel_barrier_subproblem(
        problem=PROBLEMS[problem_name](), primal_dual=primal_dual
    )
    g_hat, H_hat, A_hat, c = dense_reference(sub)
    w = random_step(sub, 0)
    y = random_dual(sub, 1)

    assert jnp.allclose(sub.primal_grad().flatten(), g_hat, atol=1e-5)
    assert jnp.allclose(sub.hess_mvp(w).flatten(), H_hat @ w.flatten(), atol=1e-5)
    assert jnp.allclose(sub.jac_mvp(w).flatten(), A_hat @ w.flatten(), atol=1e-5)
    assert jnp.allclose(sub.jac_t_mvp(y).flatten(), A_hat.T @ y.flatten(), atol=1e-5)
    assert jnp.allclose(sub.dual_grad().flatten(), c)


@problem_ids
@primal_dual_ids
def test_models_match_dense(problem_name: str, primal_dual: bool):
    """``m_f`` is the full-primal quadratic and ``m_v`` the linearised norm."""
    sub = make_funnel_barrier_subproblem(
        problem=PROBLEMS[problem_name](), primal_dual=primal_dual
    )
    g_hat, H_hat, A_hat, c = dense_reference(sub)
    w = random_step(sub, 2)
    flat = w.flatten()

    expected_f = g_hat @ flat + 0.5 * flat @ H_hat @ flat
    assert jnp.allclose(sub.model_f(w), expected_f, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(
        sub.model_value((w, random_dual(sub, 3))), expected_f, rtol=1e-5, atol=1e-6
    )
    assert jnp.allclose(
        sub.model_v(w), jnp.linalg.norm(c + A_hat @ flat), rtol=1e-5, atol=1e-6
    )
    assert jnp.allclose(sub.model_f_grad(w).flatten(), g_hat + H_hat @ flat, atol=1e-5)


@problem_ids
@primal_dual_ids
def test_criticality_measures_match_dense(problem_name: str, primal_dual: bool):
    """``v, πᵛ, χᵛ, r̂, πᶠ, χᶠ`` follow eqs. (3.1) and (3.13)–(3.14)."""
    sub = make_funnel_barrier_subproblem(
        problem=PROBLEMS[problem_name](), primal_dual=primal_dual
    )
    g_hat, H_hat, A_hat, c = dense_reference(sub)
    w_n = random_step(sub, 4)
    y = random_dual(sub, 5)

    v = jnp.linalg.norm(c)
    assert v > 0.0
    assert jnp.allclose(sub.violation(), v)
    zero = jax.tree.map(jnp.zeros_like, w_n)
    assert jnp.allclose(sub.model_v(zero), sub.violation())

    pi_v = jnp.linalg.norm(A_hat.T @ c)
    assert jnp.allclose(sub.pi_v(), pi_v, rtol=1e-5)
    assert jnp.allclose(sub.chi_v(), pi_v / v, rtol=1e-5)

    r = g_hat + H_hat @ w_n.flatten() + A_hat.T @ y.flatten()
    assert jnp.allclose(sub.r(w_n, y).flatten(), r, atol=1e-5)
    assert jnp.allclose(sub.pi_f(w_n, y), jnp.linalg.norm(r), rtol=1e-5)
    grad_m = g_hat + H_hat @ w_n.flatten()
    assert jnp.allclose(
        sub.chi_f(w_n, y), grad_m @ r / jnp.linalg.norm(r), rtol=1e-5, atol=1e-6
    )
    # With n = 0 and y = 0 the residual is the model gradient itself.
    zero_dual = jax.tree.map(jnp.zeros_like, y)
    assert jnp.allclose(sub.chi_f(zero, zero_dual), sub.pi_f(zero, zero_dual))


@primal_dual_ids
def test_measures_vanish_at_feasible_point(primal_dual: bool):
    """Slacks closing every constraint give ``v = πᵛ = χᵛ = 0`` without NaNs."""
    problem, _, _ = make_shifted_box_quadratic(n=3)
    x = jnp.array([0.5, 0.0, 0.0])
    primal = InteriorPointPrimal(
        x=x,
        slack=Slack(
            s=-problem.ineq_fn(x),
            s_lb=x - problem.lb,
            s_ub=problem.ub - x,
        ),
    )
    sub = make_funnel_barrier_subproblem(
        problem=problem, primal_dual=primal_dual, primal=primal
    )
    assert jnp.allclose(sub.violation(), 0.0, atol=1e-6)
    assert jnp.allclose(sub.pi_v(), 0.0, atol=1e-6)
    assert jnp.allclose(sub.chi_v(), 0.0)
    assert jnp.isfinite(sub.chi_v())


@primal_dual_ids
def test_chi_f_is_bounded_near_a_stationary_normal_step(primal_dual: bool):
    """``χᶠ`` is an alignment measure: finite and ``≤ ‖ĝ + Ĥ w_n‖`` even as ``πᶠ → 0``."""
    sub = make_funnel_barrier_subproblem(
        problem=make_problem(), primal_dual=primal_dual
    )
    g_hat, H_hat, A_hat, _ = dense_reference(sub)
    y = make_dual(sub.lagrangian.n, sub.lagrangian.meq, sub.lagrangian.mineq)
    w_flat = jnp.linalg.solve(H_hat, -(g_hat + A_hat.T @ y.flatten()))
    w_n = InteriorPointPrimal.from_flat(w_flat, sub.lagrangian.n, sub.lagrangian.mineq)
    assert jnp.allclose(sub.pi_f(w_n, y), 0.0, atol=1e-5)
    chi_f = sub.chi_f(w_n, y)
    assert jnp.isfinite(chi_f)
    grad_norm = jnp.linalg.norm(sub.model_f_grad(w_n).flatten())
    assert jnp.abs(chi_f) <= grad_norm * (1.0 + 1e-5)


@primal_dual_ids
def test_f_measures_guard_exact_zero_residual(primal_dual: bool):
    """With ``ĝ = 0`` exactly (``μ = 0``, ``∇f = 0``) ``πᶠ = χᶠ = 0`` without ``0/0``."""
    problem = make_problem()
    primal = InteriorPointPrimal(
        x=jnp.zeros((problem.n,)),
        slack=Slack(
            s=jnp.full((problem.mineq,), 1.5),
            s_lb=jnp.full((problem.n,), 1.25),
            s_ub=jnp.full((problem.n,), 1.75),
        ),
    )
    sub = make_funnel_barrier_subproblem(
        problem=problem, primal_dual=primal_dual, weight=0.0, primal=primal
    )
    zero_w = jax.tree.map(jnp.zeros_like, sub.lagrangian.ref)
    zero_y = jax.tree.map(jnp.zeros_like, sub.lagrangian.dual)
    assert jnp.array_equal(sub.pi_f(zero_w, zero_y), 0.0)
    assert jnp.array_equal(sub.chi_f(zero_w, zero_y), 0.0)


# ---------------------------------------------------------------------------
# fraction-to-boundary boxes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kappa_fbn", [0.05, 0.3])
def test_primal_box_uses_normal_fraction_to_boundary(kappa_fbn: float):
    """Normal-step box: ``x`` free, live slacks ``≥ −(1 − κ_fbn)``, nulls free."""
    problem = PROBLEMS["eq-ineq-mixed-null"]()
    sub = make_funnel_barrier_subproblem(problem=problem, kappa_fbn=kappa_fbn)
    lo, hi = sub.primal_box()
    n, mineq = problem.n, problem.mineq

    assert jnp.all(jnp.isneginf(lo[:n]))
    assert jnp.all(jnp.isposinf(hi))
    assert jnp.allclose(lo[n : n + mineq], -(1.0 - kappa_fbn))
    lo_lb = lo[n + mineq : n + mineq + n]
    lo_ub = lo[n + mineq + n :]
    assert jnp.allclose(lo_lb[0], -(1.0 - kappa_fbn))
    assert jnp.isneginf(lo_lb[1])
    assert jnp.isneginf(lo_ub[0])
    assert jnp.allclose(lo_ub[1], -(1.0 - kappa_fbn))

    # A slack step sitting on the face leaves exactly κ_fbn s in native scale.
    S = sub.lagrangian.slack
    face = sub._slack_to_orig_scale(
        Slack(
            s=lo[n : n + mineq],
            s_lb=jnp.where(problem.null_lb, 0.0, lo_lb),
            s_ub=jnp.where(problem.null_ub, 0.0, lo_ub),
        )
    )
    assert jnp.allclose(S.s + face.s, kappa_fbn * S.s)
    assert jnp.allclose((S.s_lb + face.s_lb)[0], kappa_fbn * S.s_lb[0])
    assert jnp.allclose((S.s_ub + face.s_ub)[1], kappa_fbn * S.s_ub[1])


@pytest.mark.parametrize("kappa_fbt", [0.05, 0.3])
def test_tangential_box_is_relative_to_normal_step(kappa_fbt: float):
    """Tangential box faces are ``−(1 − κ_fbt)(1 + w_n,s)`` on live slacks."""
    problem = PROBLEMS["eq-ineq-mixed-null"]()
    sub = make_funnel_barrier_subproblem(problem=problem, kappa_fbt=kappa_fbt)
    n, mineq = problem.n, problem.mineq
    w_n = random_step(sub, 6)
    lo, hi = sub.tangential_box(w_n)

    assert lo.shape == (n + mineq + 2 * n,)
    assert jnp.all(jnp.isneginf(lo[:n]))
    assert jnp.all(jnp.isposinf(hi))
    expected = -(1.0 - kappa_fbt) * (1.0 + w_n.flatten()[n:])
    assert jnp.allclose(lo[n : n + mineq], expected[:mineq])
    lo_lb = lo[n + mineq : n + mineq + n]
    lo_ub = lo[n + mineq + n :]
    assert jnp.allclose(lo_lb[0], expected[mineq])
    assert jnp.isneginf(lo_lb[1])
    assert jnp.isneginf(lo_ub[0])
    assert jnp.allclose(lo_ub[1], expected[mineq + n + 1])

    # Zero normal step collapses to the plain −(1 − κ_fbt) face.
    zero = jax.tree.map(jnp.zeros_like, w_n)
    lo0, _ = sub.tangential_box(zero)
    assert jnp.allclose(lo0[n : n + mineq], -(1.0 - kappa_fbt))

    # On the face, s + nˢ + tˢ = κ_fbt (s + nˢ) in native scale (eq. 3.17).
    S = sub.lagrangian.slack
    n_orig = sub._slack_to_orig_scale(w_n.slack)
    t_orig = sub._slack_to_orig_scale(
        Slack(
            s=lo[n : n + mineq],
            s_lb=jnp.where(problem.null_lb, 0.0, lo_lb),
            s_ub=jnp.where(problem.null_ub, 0.0, lo_ub),
        )
    )
    after_normal = S.s + n_orig.s
    assert jnp.allclose(after_normal + t_orig.s, kappa_fbt * after_normal, atol=1e-6)
    after_normal_lb = (S.s_lb + n_orig.s_lb)[0]
    assert jnp.allclose(
        after_normal_lb + t_orig.s_lb[0], kappa_fbt * after_normal_lb, atol=1e-6
    )
