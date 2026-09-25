"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.proximal_active_set_loop`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.preconditioner import (
    IdentityPreconditioner,
    MatrixPreconditioner,
)
from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.problem import build_problem
from slsqp_jax.sqpdax.subproblem import ProximalActiveSetSubProblem
from slsqp_jax.sqpdax.subproblem.solver import (
    RESULTS,
    ProjectedCGSubProblemSolver,
    ProximalActiveSetQPSolver,
    ProximalActiveSetQPSolverState,
)
from tests.sqpdax.lagrangian.conftest import make_primal, make_problem
from tests.sqpdax.subproblem.conftest import make_zero_dual

from .conftest import (
    make_projected_cg_state,
    make_proximal_state,
    make_qp_subproblem,
    unbounded_box,
)


def _zero_warm(n: int, meq: int, mineq: int):
    return Primal(jnp.zeros(n)), make_zero_dual(n, meq, mineq)


def _stationarity(lag, d, lam):
    """``∇f + B d + A_eqᵀλ_eq + A_ineqᵀλ_ineq − λ_lb + λ_ub`` (null bounds masked)."""
    return (
        lag.grad_val
        + lag.hvp(d)
        + lam.eq_multipliers @ lag.eq_fn_jac_val
        + lam.ineq_multipliers @ lag.ineq_fn_jac_val
        - jnp.where(lag.null_lb, 0.0, lam.lb_multipliers)
        + jnp.where(lag.null_ub, 0.0, lam.ub_multipliers)
    )


@pytest.mark.parametrize(
    ("kkt_residual", "mu_min", "eq_center_from_exact"),
    [(1e-2, 1e-6, False), (1e-12, 1e-3, False), (1e-2, 1e-6, True)],
    ids=["mu-large", "mu-floor", "centred-at-exact"],
)
def test_equality_only_solves_stabilised_kkt(
    kkt_residual: float, mu_min: float, eq_center_from_exact: bool
):
    """Equality-only: ``B d + Aᵀλ = -g`` and ``A d − μ(λ − λ_k) = -c``; one WS iteration."""
    n = 2
    lb, ub = unbounded_box(n)
    problem = make_problem(meq=1, mineq=0, lb=lb, ub=ub)
    sub = make_qp_subproblem(problem=problem, primal=make_primal(n=n))
    lag = sub.lagrangian
    warm = _zero_warm(n, 1, 0)

    (d_pcg, lam_pcg), _ = ProjectedCGSubProblemSolver().solve(
        sub, warm, make_projected_cg_state()
    )
    eq_center = lam_pcg.eq_multipliers if eq_center_from_exact else jnp.zeros((1,))
    state0 = make_proximal_state(1, kkt_residual=kkt_residual, eq_center=eq_center)
    solver = ProximalActiveSetQPSolver(mu_min=mu_min)
    (d, lam), state = solver.solve(sub, warm, state0)

    assert bool(state.success)
    assert state.status == RESULTS.successful
    assert int(state.n_iter) == 1
    mu = state.mu
    assert jnp.allclose(mu, solver.proximal_mu(jnp.asarray(kkt_residual)))
    if kkt_residual < 1e-6:
        assert float(mu) == pytest.approx(mu_min)
    # Tolerances are float32-friendly (the sqpdax suite runs without x64).
    assert jnp.allclose(_stationarity(lag, d.x, lam), 0.0, atol=1e-4)
    A, c = lag.eq_fn_jac_val, lag.eq_fn_val
    assert jnp.allclose(A @ d.x - mu * (lam.eq_multipliers - eq_center), -c, atol=1e-4)
    assert jnp.allclose(state.eq_center, lam.eq_multipliers)
    # The carried dual reflects the *recovered* equality multipliers.
    assert jnp.allclose(state.dual.eq_multipliers, lam.eq_multipliers)
    if eq_center_from_exact:
        # Centred at the exact multipliers the stabilised step *is* the SQP step.
        assert jnp.allclose(d.x, d_pcg.x, atol=1e-4)
    elif kkt_residual < 1e-6:
        # At the μ floor the step is within O(μ ‖λ*‖) of the projected-CG step.
        assert jnp.allclose(d.x, d_pcg.x, atol=10 * mu_min)
    else:
        assert not jnp.allclose(A @ d.x, -c, atol=1e-4)


def test_mixed_problem_kkt_conditions_and_jit():
    """Mixed eq / ineq / bounds: stationarity, complementarity, sign, recovery."""
    problem = make_problem()  # meq=1, mineq=2, finite bounds
    sub = make_qp_subproblem(problem=problem, primal=make_primal(n=problem.n))
    lag = sub.lagrangian
    warm = _zero_warm(problem.n, problem.meq, problem.mineq)
    eq_center = jnp.asarray([0.2])
    state0 = make_proximal_state(1, kkt_residual=1e-3, eq_center=eq_center)
    solver = ProximalActiveSetQPSolver()
    (d, lam), state = solver.solve(sub, warm, state0)

    assert bool(state.success)
    assert jnp.allclose(_stationarity(lag, d.x, lam), 0.0, atol=1e-5)
    x_new = lag.x_ref + d.x
    h_lin = lag.ineq_fn_val + lag.ineq_fn_jac_val @ d.x
    assert jnp.all(h_lin <= 1e-5)
    assert jnp.all(lam.ineq_multipliers >= -1e-6)
    assert jnp.allclose(lam.ineq_multipliers * h_lin, 0.0, atol=1e-5)
    assert jnp.all(x_new >= lag.lb - 1e-5)
    assert jnp.all(x_new <= lag.ub + 1e-5)
    assert jnp.all(lam.lb_multipliers >= -1e-6)
    assert jnp.all(lam.ub_multipliers >= -1e-6)
    assert jnp.allclose(
        lam.eq_multipliers,
        eq_center + (lag.eq_fn_jac_val @ d.x + lag.eq_fn_val) / state.mu,
        atol=1e-5,
    )
    assert jnp.allclose(state.eq_center, lam.eq_multipliers)

    (d_jit, lam_jit), state_jit = jax.jit(solver.solve)(sub, warm, state0)
    assert jnp.allclose(d_jit.x, d.x, atol=1e-6)
    assert jnp.allclose(lam_jit.flatten(), lam.flatten(), atol=1e-6)
    assert jnp.allclose(state_jit.mu, state.mu)


@pytest.mark.parametrize(
    ("kkt_residual", "tau", "mu_min", "mu_max"),
    [
        (1e-2, 0.5, 1e-6, 0.1),
        (1e-30, 0.5, 1e-6, 0.1),
        (0.0, 0.5, 1e-6, 0.1),
        (0.0, 0.0, 1e-6, 0.1),
        (jnp.inf, 0.5, 1e-6, 0.1),
        (1e-2, 0.9, 1e-4, 1.0),
    ],
    ids=["mid", "tiny", "zero", "tau0", "inf", "custom"],
)
def test_mu_schedule_is_clipped(kkt_residual, tau, mu_min, mu_max):
    """``μ = clip(res^τ, μ_min, μ_max)``; ``1/μ`` always finite."""
    solver = ProximalActiveSetQPSolver(tau=tau, mu_min=mu_min, mu_max=mu_max)
    mu = solver.proximal_mu(jnp.asarray(kkt_residual))
    expected = jnp.clip(jnp.power(jnp.asarray(kkt_residual), tau), mu_min, mu_max)
    assert jnp.allclose(mu, expected)
    assert mu >= mu_min
    assert mu <= mu_max
    assert jnp.isfinite(1.0 / mu)

    problem = make_problem(meq=1, mineq=0, lb=unbounded_box()[0], ub=unbounded_box()[1])
    sub = make_qp_subproblem(problem=problem)
    warm = _zero_warm(problem.n, 1, 0)
    (d, _), state = solver.solve(
        sub, warm, make_proximal_state(1, kkt_residual=kkt_residual)
    )
    assert jnp.allclose(state.mu, expected)
    assert jnp.all(jnp.isfinite(d.x))


@pytest.mark.parametrize(
    "kwargs",
    [{"tau": -0.1}, {"tau": 1.0}, {"mu_min": 0.0}, {"mu_min": 1.0, "mu_max": 0.1}],
    ids=["tau-negative", "tau-one", "mu_min-zero", "mu_max-below-min"],
)
def test_invalid_schedule_parameters_raise(kwargs):
    """Schedule parameters are validated at construction (and via ``init``)."""
    with pytest.raises(ValueError):
        ProximalActiveSetQPSolver(**kwargs)
    with pytest.raises(ValueError):
        ProximalActiveSetQPSolver().init(**kwargs)


def test_warm_start_from_kkt_point_is_a_fixed_point():
    """At ``(x*, λ*)`` the stabilised QP returns ``d ≈ 0`` and keeps ``λ*``."""
    n = 2
    lb, ub = unbounded_box(n)
    problem = make_problem(meq=1, mineq=0, lb=lb, ub=ub)
    # min ‖x‖² s.t. x0 + x1 = 1  =>  x* = (0.5, 0.5), λ* = -1 (∇f + λ A = 0).
    sub = make_qp_subproblem(problem=problem, primal=Primal(jnp.array([0.5, 0.5])))
    warm = _zero_warm(n, 1, 0)
    lam_star = jnp.asarray([-1.0])
    solver = ProximalActiveSetQPSolver()
    (d, lam), state = solver.solve(
        sub, warm, make_proximal_state(1, kkt_residual=1e-3, eq_center=lam_star)
    )
    assert jnp.allclose(d.x, 0.0, atol=1e-6)
    assert jnp.allclose(lam.eq_multipliers, lam_star, atol=1e-6)
    assert jnp.allclose(state.eq_center, lam_star, atol=1e-6)

    # Chaining: the state returned is a valid warm start for the next solve.
    (d2, lam2), state2 = solver.solve(sub, warm, state)
    assert jnp.allclose(d2.x, 0.0, atol=1e-6)
    assert jnp.allclose(lam2.eq_multipliers, lam.eq_multipliers, atol=1e-6)
    assert isinstance(state2, ProximalActiveSetQPSolverState)


def _diagonal_quadratic_with_equalities():
    """``min ½ xᵀ Q x + qᵀx`` s.t. ``A x = b`` with ``n = 5``, ``meq = 2``."""
    n, meq = 5, 2
    Q = jnp.diag(jnp.array([1.0, 4.0, 9.0, 16.0, 25.0]))
    q = jnp.array([1.0, -2.0, 0.5, 3.0, -1.0])
    A = jnp.array([[1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, -1.0, 2.0, 0.5]])
    b = jnp.array([1.0, -0.5])
    problem = build_problem(
        fn=lambda x: 0.5 * x @ Q @ x + q @ x,
        n=n,
        meq=meq,
        mineq=0,
        grad=lambda x: Q @ x + q,
        hvp=lambda x, v: Q @ v,
        eq_fn=lambda x: A @ x - b,
        eq_fn_jac=lambda x: A,
        eq_fn_hvp=lambda x, v: jnp.zeros((meq, n), x.dtype),
        lb=jnp.full((n,), -jnp.inf),
        ub=jnp.full((n,), jnp.inf),
        autodiff_mode="custom",
    )
    return problem, Q


def test_secant_preconditioner_gets_woodbury_wrapped():
    """A base preconditioner is Woodbury-corrected: same step, fewer CG iterations."""
    problem, Q = _diagonal_quadratic_with_equalities()
    sub = make_qp_subproblem(problem=problem, primal=Primal(jnp.zeros(problem.n)))
    lag = sub.lagrangian
    warm = _zero_warm(problem.n, problem.meq, 0)
    state0 = make_proximal_state(problem.meq, kkt_residual=1e-4)  # μ = 1e-2

    plain = ProximalActiveSetQPSolver()
    (d_plain, _), s_plain = plain.solve(sub, warm, state0)
    # Exact base ``M = Q``; the Woodbury wrap makes ``M̃`` the exact stabilised operator.
    pre = ProximalActiveSetQPSolver(
        subproblem_solver=ProjectedCGSubProblemSolver(
            preconditioner=MatrixPreconditioner(Q)
        )
    )
    (d_pre, lam_pre), s_pre = pre.solve(sub, warm, state0)

    mu = s_pre.mu
    A, c, g = lag.eq_fn_jac_val, lag.eq_fn_val, lag.grad_val
    d_exact = jnp.linalg.solve(Q + A.T @ A / mu, -(g + A.T @ (c / mu)))
    assert bool(s_plain.success) and bool(s_pre.success)
    assert jnp.allclose(d_pre.x, d_exact, atol=1e-4)
    assert jnp.allclose(lam_pre.eq_multipliers, (A @ d_exact + c) / mu, atol=1e-2)
    # Preconditioning is never less accurate and needs fewer CG iterations
    # (in float32 the unpreconditioned CG stalls on this conditioning).
    err_pre = jnp.linalg.norm(d_pre.x - d_exact)
    err_plain = jnp.linalg.norm(d_plain.x - d_exact)
    assert err_pre <= err_plain + 1e-6
    assert int(s_pre.n_cg_iter) < int(s_plain.n_cg_iter)
    # The solver object itself is untouched (the wrap is local to ``solve``).
    assert isinstance(pre.subproblem_solver.preconditioner, MatrixPreconditioner)


@pytest.mark.parametrize(
    ("preconditioner", "meq", "expect_wrapped"),
    [
        (None, 2, False),
        ("identity", 2, False),
        ("matrix", 0, False),
        ("matrix", 2, True),
    ],
    ids=["no-preconditioner", "identity", "no-equalities", "wrapped"],
)
def test_kkt_solver_hook_wraps_preconditioner_only_when_needed(
    preconditioner, meq, expect_wrapped
):
    """``_kkt_solver`` passes the inner solver through unless a Woodbury wrap applies.

    The hook receives the proximal subproblem, so the wrap uses its ``μ`` and
    ``A_eq``; the solver's own ``subproblem_solver`` is never modified.
    """
    if meq > 0:
        problem, Q = _diagonal_quadratic_with_equalities()
    else:
        problem = make_problem(
            meq=0, mineq=0, lb=unbounded_box()[0], ub=unbounded_box()[1]
        )
        Q = 2.0 * jnp.eye(problem.n)
    sub = make_qp_subproblem(problem=problem, primal=Primal(jnp.zeros(problem.n)))
    pre = {
        None: None,
        "identity": IdentityPreconditioner(jnp.zeros(problem.n)),
        "matrix": MatrixPreconditioner(Q),
    }[preconditioner]
    inner = (
        ProjectedCGSubProblemSolver()
        if pre is None
        else ProjectedCGSubProblemSolver(preconditioner=pre)
    )
    solver = ProximalActiveSetQPSolver(subproblem_solver=inner)
    mu = jnp.asarray(0.05)
    prox = ProximalActiveSetSubProblem(
        sub.lagrangian, sub.active_set, mu, jnp.zeros(problem.meq)
    )

    kkt = solver._kkt_solver(prox)

    if not expect_wrapped:
        assert kkt is inner
        return
    assert kkt is not inner
    assert solver.subproblem_solver is inner
    v = jnp.arange(1.0, problem.n + 1.0)
    A = prox.eq_jac
    expected = Q @ v + A.T @ (A @ v) / mu
    assert jnp.allclose(kkt.preconditioner.pushforward(v), expected, rtol=1e-5)
    assert jnp.allclose(kkt.preconditioner.invert(expected), v, rtol=1e-3, atol=1e-4)


def test_exports_roundtrip():
    """Public re-exports resolve from solver / subproblem / sqpdax packages."""
    from slsqp_jax import sqpdax
    from slsqp_jax.sqpdax import subproblem
    from slsqp_jax.sqpdax.subproblem import solver

    assert solver.ProximalActiveSetQPSolver is ProximalActiveSetQPSolver
    assert subproblem.ProximalActiveSetQPSolver is ProximalActiveSetQPSolver
    assert sqpdax.ProximalActiveSetQPSolver is ProximalActiveSetQPSolver
    assert sqpdax.ProximalActiveSetQPSolverState is ProximalActiveSetQPSolverState
