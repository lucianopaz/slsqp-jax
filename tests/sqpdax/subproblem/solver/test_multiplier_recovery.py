"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.multiplier_recovery`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax.flatten_util import ravel_pytree

from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.lagrangian import Lagrangian
from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.secant import LBFGS
from slsqp_jax.sqpdax.subproblem.active_set import ActiveSetSubProblem
from slsqp_jax.sqpdax.subproblem.base import SubProblem
from slsqp_jax.sqpdax.subproblem.solver import (
    BarrierSafeguard,
    ClampSafeguard,
    KKTMultiplierRecovery,
    LeastSquaresMultiplierRecovery,
    MultiplierRecovery,
    SVDProjector,
    TrustRegionInteriorPointSolver,
)
from tests.sqpdax.lagrangian.conftest import make_primal, make_problem
from tests.sqpdax.subproblem.conftest import (
    FUNNEL_PROBLEMS,
    dense_funnel_reference,
    make_funnel_barrier_subproblem,
    make_scaled_barrier_subproblem,
    make_zero_dual,
    random_funnel_step,
)

from .conftest import make_qp_subproblem, make_trust_region_state, unbounded_box

RECOVERIES = {
    "kkt": KKTMultiplierRecovery,
    "least-squares": LeastSquaresMultiplierRecovery,
}


def _as_free() -> ActiveSetSubProblem:
    lb, ub = unbounded_box(2)
    return make_qp_subproblem(
        problem=make_problem(meq=1, mineq=2, lb=lb, ub=ub),
        active_inequalities=(True, False),
    )


def _as_bound() -> ActiveSetSubProblem:
    # ``x0`` pinned at its lower bound, only the equality in the working set:
    # ``A_work = [[0, 1]]`` is full row rank.
    return make_qp_subproblem(active_lb=(True, False))


def _as_degenerate() -> ActiveSetSubProblem:
    # Parallel working rows once ``x0`` is fixed: ``A_work = [[0, 1], [0, -1]]``.
    return make_qp_subproblem(
        active_inequalities=(False, True), active_lb=(True, False)
    )


SUBPROBLEMS = {
    "free": _as_free,
    "bound": _as_bound,
    "degenerate": _as_degenerate,
}


class NewtonViewQP(ActiveSetSubProblem):
    @property
    def is_kkt_dual_increment(self) -> bool:
        return True


def _stationarity_residual(
    sub: SubProblem, recovery: MultiplierRecovery, d, dual: Dual
) -> float:
    target, _ = ravel_pytree(recovery.stationarity_target(sub, d))
    zero_p = jax.tree.map(jnp.zeros_like, d)
    at_lam, _ = ravel_pytree(sub.kkt_mvp_upper_offdiag((zero_p, dual)))
    return float(jnp.linalg.norm(at_lam - target))


@pytest.mark.parametrize("make_sub", SUBPROBLEMS.values(), ids=SUBPROBLEMS.keys())
@pytest.mark.parametrize("recovery_cls", RECOVERIES.values(), ids=RECOVERIES.keys())
def test_both_paths_solve_the_stationarity_row(make_sub, recovery_cls):
    """Projector and matrix-free paths zero the residual with the right sparsity."""
    sub = make_sub()
    recovery = recovery_cls()
    ctx = SVDProjector().build(sub)
    d = Primal(jnp.array([0.1, -0.2]))
    active = sub.active_set

    exact = recovery.recover(sub, ctx, d)
    free = recovery.recover(sub, None, d)
    for dual in (exact, free):
        assert _stationarity_residual(sub, recovery, d, dual) < 1e-5
        # Rows outside the working set carry exact zeros.
        assert jnp.all(dual.ineq_multipliers[~active.active_inequalities] == 0)
        assert jnp.all(dual.lb_multipliers[~active.active_lb] == 0)
        assert jnp.all(dual.ub_multipliers[~active.active_ub] == 0)
    if make_sub is not _as_degenerate:
        # Unique least-squares solution: the two linear-algebra paths agree.
        assert jnp.allclose(exact.flatten(), free.flatten(), atol=1e-5)


@pytest.mark.parametrize("make_sub", SUBPROBLEMS.values(), ids=SUBPROBLEMS.keys())
def test_kkt_and_least_squares_coincide_at_zero_step(make_sub):
    """Without a step the Hessian term vanishes and both targets agree."""
    sub = make_sub()
    zero = Primal(jnp.zeros(2))
    kkt = KKTMultiplierRecovery().recover(sub, None, zero)
    ls = LeastSquaresMultiplierRecovery().recover(sub, None, zero)
    assert jnp.allclose(kkt.flatten(), ls.flatten(), atol=1e-6)
    # With a step they differ by the curvature term.
    d = Primal(jnp.array([0.3, 0.1]))
    kkt_d = KKTMultiplierRecovery().recover(sub, None, d)
    assert not jnp.allclose(kkt_d.flatten(), ls.flatten(), atol=1e-6)


def test_least_squares_recovery_is_secant_invariant():
    """LS multipliers ignore the QP Hessian; KKT multipliers do not."""
    problem = make_problem()
    x = make_primal(n=problem.n)
    dual0 = make_zero_dual(problem.n, problem.meq, problem.mineq)
    exact = make_qp_subproblem(problem=problem, primal=x)
    secant = LBFGS(n=problem.n, memory=4).append(
        jnp.array([0.2, -0.1]), jnp.array([1.0, 0.7])
    )
    lag_secant = Lagrangian(problem, secant=secant)(x, dual0)
    with_secant = ActiveSetSubProblem(lag_secant, exact.active_set)
    d = Primal(jnp.array([0.25, -0.4]))

    ls_a = LeastSquaresMultiplierRecovery().recover(exact, None, d)
    ls_b = LeastSquaresMultiplierRecovery().recover(with_secant, None, d)
    assert jnp.allclose(ls_a.flatten(), ls_b.flatten(), atol=1e-6)
    kkt_a = KKTMultiplierRecovery().recover(exact, None, d)
    kkt_b = KKTMultiplierRecovery().recover(with_secant, None, d)
    assert not jnp.allclose(kkt_a.flatten(), kkt_b.flatten(), atol=1e-6)


def test_refinement_rounds_do_not_increase_the_residual():
    """A loose LSMR solve is tightened round by round."""
    sub = _as_free()
    d = Primal(jnp.array([0.1, -0.2]))
    residuals = []
    for rounds in range(3):
        recovery = LeastSquaresMultiplierRecovery(
            refinement_rounds=rounds, rtol=1e-2, atol=1e-2, max_steps=1
        )
        dual = recovery.recover(sub, None, d)
        residuals.append(_stationarity_residual(sub, recovery, d, dual))
    assert residuals[0] > 1e-6
    assert residuals[1] <= residuals[0] + 1e-12
    assert residuals[2] <= residuals[1] + 1e-12


@pytest.mark.parametrize("recovery_cls", RECOVERIES.values(), ids=RECOVERIES.keys())
def test_empty_working_set_yields_exact_zeros(recovery_cls):
    """No active rows: the least-squares dual is zero on both paths (no LSMR NaN)."""
    lb, ub = unbounded_box(2)
    sub = make_qp_subproblem(problem=make_problem(meq=0, mineq=2, lb=lb, ub=ub))
    d = Primal(jnp.array([0.1, -0.2]))
    ctx = SVDProjector().build(sub)
    for dual in (
        recovery_cls().recover(sub, ctx, d),
        recovery_cls().recover(sub, None, d),
    ):
        assert jnp.array_equal(dual.flatten(), jnp.zeros(6))


def test_projector_path_rejects_non_x_primals():
    """A projector context cannot serve an interior-point primal."""
    sub = make_scaled_barrier_subproblem()
    ctx = SVDProjector().build(sub)
    zero = jax.tree.map(jnp.zeros_like, sub.lagrangian.ref)
    with pytest.raises(TypeError, match="decision variables only"):
        LeastSquaresMultiplierRecovery().recover(sub, ctx, zero)


def test_clamp_safeguard_zeroes_negative_inequality_and_bound_multipliers():
    """Clamp leaves equalities signed and floors the rest at zero."""
    sub = _as_free()
    dual = Dual(
        eq_multipliers=jnp.array([-1.5]),
        ineq_multipliers=jnp.array([-0.3, 0.4]),
        lb_multipliers=jnp.array([0.2, -0.1]),
        ub_multipliers=jnp.array([-0.7, 0.0]),
    )
    out = ClampSafeguard().apply(sub, dual)
    assert jnp.array_equal(out.eq_multipliers, dual.eq_multipliers)
    assert jnp.array_equal(out.ineq_multipliers, jnp.array([0.0, 0.4]))
    assert jnp.array_equal(out.lb_multipliers, jnp.array([0.2, 0.0]))
    assert jnp.array_equal(out.ub_multipliers, jnp.array([0.0, 0.0]))
    # Composed through a recovery: the LS estimate on the working set is clamped.
    d = Primal(jnp.array([0.1, -0.2]))
    raw = LeastSquaresMultiplierRecovery().recover(sub, None, d)
    clamped = LeastSquaresMultiplierRecovery(safeguard=ClampSafeguard()).recover(
        sub, None, d
    )
    assert jnp.all(clamped.ineq_multipliers >= 0)
    assert jnp.allclose(clamped.eq_multipliers, raw.eq_multipliers)


def test_barrier_safeguard_matches_the_trust_region_rule():
    """N&W eq. 19.38 on a scaled-barrier subproblem; rejects active-set models."""
    sub = make_scaled_barrier_subproblem()
    lag = sub.lagrangian
    mu = lag.barrier.weight
    n, mineq = lag.n, lag.mineq
    dual = Dual(
        eq_multipliers=jnp.array([-2.0]),
        ineq_multipliers=jnp.array([-1.0, 0.5]),
        lb_multipliers=-jnp.ones(n),
        ub_multipliers=jnp.full(n, 2.0),
    )
    out = BarrierSafeguard().apply(sub, dual)
    assert jnp.array_equal(out.eq_multipliers, dual.eq_multipliers)
    assert jnp.allclose(
        out.ineq_multipliers,
        jnp.array([jnp.minimum(1e-3, mu / lag.slack.s[0]), 0.5]),
    )
    assert jnp.allclose(
        out.lb_multipliers,
        jnp.where(lag.null_lb, 0.0, jnp.minimum(1e-3, mu / lag.slack.s_lb)),
    )
    assert jnp.allclose(out.ub_multipliers, jnp.where(lag.null_ub, 0.0, 2.0))
    assert mineq == 2
    with pytest.raises(TypeError, match="interior-point"):
        BarrierSafeguard().apply(_as_free(), make_zero_dual(2, 1, 2))


@pytest.mark.parametrize("max_norm", [0.5, 1e3], ids=["active", "inactive"])
def test_barrier_safeguard_max_norm_caps_the_dual_without_breaking_positivity(
    max_norm,
):
    """κ_y of (3.10): rescale onto the 2-norm ball only when it is exceeded."""
    sub = make_scaled_barrier_subproblem()
    n = sub.lagrangian.n
    dual = Dual(
        eq_multipliers=jnp.array([-2.0]),
        ineq_multipliers=jnp.array([-1.0, 0.5]),
        lb_multipliers=-jnp.ones(n),
        ub_multipliers=jnp.full(n, 2.0),
    )
    uncapped = BarrierSafeguard().apply(sub, dual)
    capped = BarrierSafeguard(max_norm=max_norm).apply(sub, dual)
    norm = float(jnp.linalg.norm(uncapped.flatten()))
    assert float(jnp.linalg.norm(capped.flatten())) <= max_norm * (1 + 1e-6)
    if norm <= max_norm:
        assert jnp.array_equal(capped.flatten(), uncapped.flatten())
    else:
        # Uniform shrink: same direction, norm exactly on the ball.
        assert jnp.allclose(
            capped.flatten(), uncapped.flatten() * (max_norm / norm), rtol=1e-6
        )
        assert jnp.isclose(jnp.linalg.norm(capped.flatten()), max_norm, rtol=1e-6)
    # Positivity from the N&W repair survives the rescaling; nulls stay zero.
    lag = sub.lagrangian
    assert jnp.all(capped.ineq_multipliers > 0)
    assert jnp.all(capped.lb_multipliers[~lag.null_lb] > 0)
    assert jnp.all(capped.lb_multipliers[lag.null_lb] == 0)
    assert jnp.all(capped.ub_multipliers[~lag.null_ub] > 0)
    assert jnp.all(capped.ub_multipliers[lag.null_ub] == 0)


def test_barrier_safeguard_rejects_non_positive_max_norm():
    with pytest.raises(ValueError, match="max_norm"):
        BarrierSafeguard(max_norm=0.0)


@pytest.mark.parametrize(
    "make_problem_", FUNNEL_PROBLEMS.values(), ids=FUNNEL_PROBLEMS.keys()
)
def test_kkt_recovery_on_funnel_subproblem_solves_the_scaled_least_squares(
    make_problem_,
):
    """(2.7) of CGRT 2017: ``y = argmin ‖ĝ + Ĥ wₙ + Âᵀ y‖₂`` with ``wₙ`` the normal step.

    Both the dense ``lstsq`` and the matrix-free LSMR path return the
    minimum-norm solution, so they agree even when null bounds leave zero
    rows in ``Â``.
    """
    sub = make_funnel_barrier_subproblem(problem=make_problem_())
    g_hat, h_hat, a_hat, _ = dense_funnel_reference(sub)
    w_n = random_funnel_step(sub, seed=3)
    w_flat, _ = ravel_pytree(w_n)
    rhs = -(g_hat + h_hat @ w_flat)
    expected, *_ = jnp.linalg.lstsq(a_hat.T, rhs)

    y = KKTMultiplierRecovery(rtol=1e-8, atol=1e-8).recover(sub, None, w_n)
    assert jnp.allclose(y.flatten(), expected, atol=1e-4)
    # The subproblem's residual (3.13) at this dual is the least-squares residual.
    assert jnp.allclose(
        ravel_pytree(sub.r(w_n, y))[0], a_hat.T @ expected - rhs, atol=1e-4
    )


@pytest.mark.parametrize(
    "safeguard", [ClampSafeguard(), BarrierSafeguard()], ids=["clamp", "barrier"]
)
def test_safeguards_reject_newton_view_subproblems(safeguard):
    """Sign-interpreting safeguards refuse a ``Δλ`` block at trace time."""
    base = _as_free()
    sub = NewtonViewQP(base.lagrangian, base.active_set)
    with pytest.raises(TypeError, match="SQP-view"):
        safeguard.apply(sub, make_zero_dual(2, 1, 2))
    with pytest.raises(TypeError, match="SQP-view"):
        LeastSquaresMultiplierRecovery(safeguard=safeguard).recover(
            sub, None, Primal(jnp.zeros(2))
        )


@pytest.mark.parametrize(
    "recovery",
    [
        None,
        LeastSquaresMultiplierRecovery(
            refinement_rounds=0, safeguard=BarrierSafeguard()
        ),
        LeastSquaresMultiplierRecovery(safeguard=ClampSafeguard()),
    ],
    ids=["default", "no-refinement", "clamp"],
)
def test_trust_region_interior_point_uses_the_multiplier_recovery_strategy(recovery):
    """The trust-region solver's dual is exactly its ``multiplier_recovery`` output.

    The default is least squares (eq. 19.37) with the barrier safeguard
    (eq. 19.38); any other strategy is honoured verbatim.
    """
    sub = make_scaled_barrier_subproblem()
    lag = sub.lagrangian
    warm = (
        jax.tree.map(jnp.zeros_like, lag.ref),
        jax.tree.map(jnp.zeros_like, lag.dual),
    )
    tr = TrustRegionInteriorPointSolver()
    if recovery is not None:
        tr = tr.init(multiplier_recovery=recovery)
    (_, tr_dual), _ = tr.solve(sub, warm, make_trust_region_state(1.0))
    assert isinstance(tr.multiplier_recovery, LeastSquaresMultiplierRecovery)
    if recovery is None:
        assert isinstance(tr.multiplier_recovery.safeguard, BarrierSafeguard)
    expected = tr.multiplier_recovery.recover(sub, None, warm[0])
    assert jnp.allclose(tr_dual.flatten(), expected.flatten(), atol=1e-6)
    # Every strategy here enforces dual feasibility on inequalities / bounds.
    assert jnp.all(tr_dual.ineq_multipliers >= 0)
    assert jnp.all(tr_dual.lb_multipliers >= 0)
    assert jnp.all(tr_dual.ub_multipliers >= 0)
    # Refinement changes the estimate only within LSMR's own tolerance.
    unrefined = (
        LeastSquaresMultiplierRecovery(
            refinement_rounds=0, safeguard=BarrierSafeguard()
        )
        .recover(sub, None, warm[0])
        .flatten()
    )
    assert jnp.allclose(
        TrustRegionInteriorPointSolver()
        .solve(sub, warm, make_trust_region_state(1.0))[0][1]
        .flatten(),
        unrefined,
        atol=1e-4,
    )
