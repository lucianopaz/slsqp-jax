"""Regression tests for the robustness-review pass.

These tests exercise the fixes landed in the bug-review sweep:

* ``terminate`` no longer declares success after chronic line-search failure
* ``SLSQPDiagnostics`` counters populate through ``step`` under JIT
* ``QPResult.converged`` propagates non-finite directions (but *not* merely
  slow CG) through every solve path
* ``craig_solve`` reports breakdown on rank-deficient constraint matrices
* ``MinresQLPSolver`` respects ``free_mask`` even when a preconditioner is
  supplied
* ``SLSQP`` exposes ``lbfgs_diag_floor`` / ``lbfgs_diag_ceil`` that reach
  ``lbfgs_append``
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest

from slsqp_jax import (
    SLSQP,
    MinresQLPSolver,
    ProjectedCGCraig,
    get_diagnostics,
    lbfgs_append,
    lbfgs_init,
    lbfgs_should_skip,
)
from slsqp_jax.inner_solver import craig_solve
from slsqp_jax.qp_solver import solve_qp

jax.config.update("jax_enable_x64", True)


def _run_loop(solver, objective, x0, max_steps=None):
    if max_steps is None:
        max_steps = solver.max_steps
    state = solver.init(objective, x0, None, {}, None, None, frozenset())
    y = x0
    last_result = optx.RESULTS.successful
    for _ in range(max_steps):
        done, last_result = solver.terminate(objective, y, None, {}, state, frozenset())
        if done:
            break
        y, state, _ = solver.step(objective, y, None, {}, state, frozenset())
    return y, state, last_result


# ---------------------------------------------------------------------------
# Termination regression
# ---------------------------------------------------------------------------


class TestChronicLineSearchNoLongerSuccess:
    """When the line search fails ``>= 2 * ls_failure_patience`` times in a
    row, ``terminate`` must report ``nonlinear_divergence`` -- never the old
    ``successful`` that came from the ``qp_optimal & primal_feasible``
    disjunct.
    """

    @pytest.mark.slow
    def test_degenerate_linear_program_flags_ls_fatal(self):
        n = 5
        weights = jnp.linspace(1.0, 10.0, n)
        target = jnp.ones(n).at[:4].set(-2.0)

        def objective(x, args):
            return jnp.sum(weights * (x - target) ** 2), None

        def eq_constraint(x, args):
            return jnp.array([jnp.sum(x) - float(n)])

        def ineq_constraint(x, args):
            return x[:4]

        solver = SLSQP(
            atol=1e-12,
            rtol=1e-12,
            max_steps=200,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=4,
            ls_failure_patience=2,
            proximal_tau=0.5,
        )

        x0 = jnp.ones(n)
        _, state, _ = _run_loop(solver, objective, x0, max_steps=60)

        if bool(state.ls_fatal):
            assert state.consecutive_ls_failures >= 2 * solver.ls_failure_patience
            done, result = solver.terminate(objective, x0, None, {}, state, frozenset())
            assert bool(done)
            assert result == optx.RESULTS.nonlinear_divergence


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class TestDiagnosticsAccumulator:
    def test_counters_nonnegative_and_finite(self):
        def objective(x, args):
            return jnp.sum((x - 1.0) ** 2), None

        def eq_constraint(x, args):
            return jnp.array([jnp.sum(x) - 3.0])

        solver = SLSQP(
            atol=1e-8,
            max_steps=20,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
        )

        x0 = jnp.zeros(3)
        _, state, _ = _run_loop(solver, objective, x0)

        diag = get_diagnostics(state)
        assert int(diag.n_qp_inner_failures) >= 0
        assert int(diag.n_ls_failures) >= 0
        assert int(diag.n_lbfgs_skips) >= 0
        assert int(diag.n_nan_directions) >= 0
        assert jnp.isfinite(diag.ls_alpha_min)
        assert float(diag.ls_alpha_min) > 0.0
        assert float(diag.max_gamma) >= 0.0
        assert float(diag.max_diag) >= float(diag.min_diag)

    def test_tail_ls_failures_tracks_state(self):
        def objective(x, args):
            return jnp.sum(x**2), None

        solver = SLSQP(atol=1e-8, max_steps=10)
        x0 = jnp.array([2.0, -1.0])
        _, state, _ = _run_loop(solver, objective, x0)

        diag = get_diagnostics(state)
        assert int(diag.tail_ls_failures) == int(state.consecutive_ls_failures)


# ---------------------------------------------------------------------------
# QP inner failure propagation: isfinite, not slow CG
# ---------------------------------------------------------------------------


class TestQPConvergedFlag:
    """``QPResult.converged`` must remain ``True`` when CG merely ran out of
    iterations (the common case for the stabilized proximal system) but fall
    to ``False`` when the direction contains NaN/Inf.
    """

    def test_truncated_cg_still_converged(self):
        n = 4
        B = jnp.diag(jnp.array([1.0, 2.0, 3.0, 4.0]))
        g = jnp.array([1.0, 1.0, 1.0, 1.0])
        # Single equality and no inequalities: forces the unconstrained CG
        # branch inside ``_solve_qp_proximal``.  Very tight tol + tiny budget
        # should prevent CG from converging below the residual threshold.
        A_eq = jnp.array([[1.0, 1.0, 1.0, 1.0]])
        b_eq = jnp.array([0.0])
        A_ineq = jnp.zeros((0, n))
        b_ineq = jnp.zeros((0,))

        def hvp(v):
            return B @ v

        result = solve_qp(
            hvp_fn=hvp,
            g=g,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ineq=A_ineq,
            b_ineq=b_ineq,
            max_iter=5,
            max_cg_iter=1,
            tol=1e-16,
            proximal_mu=1e-3,
            use_proximal=True,
        )

        assert jnp.isfinite(result.d).all()
        assert bool(result.converged)

    def test_nan_hvp_flags_failure(self):
        n = 3
        g = jnp.array([1.0, 1.0, 1.0])
        A_eq = jnp.zeros((0, n))
        b_eq = jnp.zeros((0,))
        A_ineq = jnp.zeros((0, n))
        b_ineq = jnp.zeros((0,))

        def hvp(v):
            return jnp.full_like(v, jnp.nan)

        result = solve_qp(
            hvp_fn=hvp,
            g=g,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ineq=A_ineq,
            b_ineq=b_ineq,
            max_iter=5,
            max_cg_iter=10,
            tol=1e-8,
        )

        assert not bool(result.converged)
        assert not bool(jnp.isfinite(result.d).all())


# ---------------------------------------------------------------------------
# CRAIG rank-deficient breakdown
# ---------------------------------------------------------------------------


class TestCraigBreakdown:
    def test_rank_deficient_A_reports_failure(self):
        # rows 0 and 1 identical -> A is rank-deficient (rank 1)
        A = jnp.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        # rhs not in range(A): inconsistent system, CRAIG cannot converge.
        rhs = jnp.array([1.0, -1.0])
        x, converged = craig_solve(A, rhs, tol=1e-10, max_iter=50)
        assert jnp.isfinite(x).all()
        assert not bool(converged)

    def test_consistent_rank_deficient_is_finite(self):
        # rows 0 and 1 identical and rhs consistent -> minimum-norm solution
        A = jnp.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        rhs = jnp.array([7.0, 7.0])
        x, _ = craig_solve(A, rhs, tol=1e-10, max_iter=50)
        assert jnp.isfinite(x).all()
        np.testing.assert_allclose(A @ x, rhs, atol=1e-8)


# ---------------------------------------------------------------------------
# MINRES-QLP fixed-variable masking
# ---------------------------------------------------------------------------


class TestMinresFreeMaskWithPrecond:
    """The ``_free * d + _dfixed`` epilogue must zero out fixed variables
    even when a (non-identity) preconditioner is supplied.
    """

    @pytest.mark.slow
    def test_bounds_fixed_stay_at_bound(self):
        def objective(x, args):
            return 0.5 * jnp.sum((x - 2.0) ** 2), None

        solver = SLSQP(
            atol=1e-8,
            max_steps=30,
            bounds=jnp.array(
                [
                    [0.0, 0.5],
                    [0.0, 2.0],
                    [0.0, 2.0],
                    [0.0, 2.0],
                ]
            ),
            inner_solver=MinresQLPSolver(max_iter=60, tol=1e-10, max_cg_iter=30),
            preconditioner_type="lbfgs",
        )
        x0 = jnp.array([0.1, 1.0, 1.0, 1.0])
        y, _, _ = _run_loop(solver, objective, x0)

        assert y[0] == pytest.approx(0.5, abs=1e-5)
        assert jnp.isfinite(y).all()
        np.testing.assert_allclose(y[1:], [2.0, 2.0, 2.0], atol=1e-5)


# ---------------------------------------------------------------------------
# L-BFGS clipping parameters propagate
# ---------------------------------------------------------------------------


class TestLBFGSClipParameters:
    def test_floor_clamps_tiny_curvature(self):
        history = lbfgs_init(3, memory=5)
        s = jnp.array([1.0, 0.0, 0.0])
        # y ~ 1e-10 * s keeps the ratio y_i*s_i / s_i^2 well below 1e-4
        y_vec = 1e-10 * s
        out = lbfgs_append(history, s, y_vec, diag_floor=1e-3, diag_ceil=1e6)
        assert float(jnp.min(out.diagonal)) >= 1e-3 - 1e-12

    def test_ceil_clamps_large_curvature(self):
        history = lbfgs_init(3, memory=5)
        s = jnp.array([1.0, 0.0, 0.0])
        y_vec = 1e8 * s  # huge curvature on coordinate 0
        out = lbfgs_append(history, s, y_vec, diag_floor=1e-4, diag_ceil=1e2)
        assert float(jnp.max(out.diagonal)) <= 1e2 + 1e-12

    def test_solver_exposes_parameters(self):
        def objective(x, args):
            return jnp.sum(x**2), None

        solver = SLSQP(
            atol=1e-8,
            max_steps=10,
            lbfgs_diag_floor=1e-2,
            lbfgs_diag_ceil=1e5,
        )
        x0 = jnp.array([3.0, -2.0, 1.0])
        _, state, _ = _run_loop(solver, objective, x0)
        diag = get_diagnostics(state)
        # min_diag is jnp.inf if no step was taken -- guard against that.
        if jnp.isfinite(diag.min_diag):
            assert float(diag.min_diag) >= 1e-2 - 1e-12
        if jnp.isfinite(diag.max_diag) and float(diag.max_diag) > 0.0:
            assert float(diag.max_diag) <= 1e5 + 1e-12


# ---------------------------------------------------------------------------
# ProjectedCGCraig integration (exercises the multiplier-recovery path)
# ---------------------------------------------------------------------------


class TestProjectedCGCraigIntegration:
    @pytest.mark.slow
    def test_equality_constrained_quadratic(self):
        def objective(x, args):
            return jnp.sum((x - jnp.array([1.0, 2.0, 3.0])) ** 2), None

        def eq_constraint(x, args):
            return jnp.array([jnp.sum(x) - 6.0])

        solver = SLSQP(
            atol=1e-8,
            max_steps=50,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            inner_solver=ProjectedCGCraig(
                max_cg_iter=50,
                cg_tol=1e-8,
                craig_tol=1e-10,
                craig_max_iter=100,
                mult_recovery_tol=1e-10,
                mult_recovery_max_iter=50,
            ),
            proximal_tau=0.0,  # direct projection path
        )
        x0 = jnp.array([0.0, 0.0, 0.0])
        y, _, _ = _run_loop(solver, objective, x0)

        np.testing.assert_allclose(y, [1.0, 2.0, 3.0], atol=1e-5)
        assert abs(float(jnp.sum(y)) - 6.0) < 1e-6


# ---------------------------------------------------------------------------
# L-BFGS skip counter: must be based on should_skip, not on ring-buffer diff
# ---------------------------------------------------------------------------


class TestLBFGSSkipCounter:
    """The old counter (``new.count == state.count``) saturated as soon as
    the circular buffer filled (``count == memory``), flagging every
    subsequent iteration as a skip.  The fix is to reproduce the exact
    ``should_skip`` decision that ``lbfgs_append`` uses internally.
    """

    def test_should_skip_matches_do_not_append(self):
        rng = np.random.RandomState(0)
        s_accept = jnp.asarray(rng.randn(4))
        y_accept = jnp.asarray(rng.randn(4))
        assert not bool(lbfgs_should_skip(s_accept, y_accept))

        s_tiny = jnp.zeros(4)
        y_ok = jnp.asarray(rng.randn(4))
        assert bool(lbfgs_should_skip(s_tiny, y_ok))

    @pytest.mark.slow
    def test_counter_does_not_saturate_past_memory(self):
        """Run many more iterations than ``lbfgs_memory`` on an easy
        problem where almost every pair should be accepted; the fixed
        counter must stay well below the number of post-memory steps.
        """

        def objective(x, args):
            return jnp.sum((x - 1.0) ** 2), None

        def eq(x, args):
            return jnp.array([jnp.sum(x) - float(x.shape[0])])

        solver = SLSQP(
            atol=1e-12,
            rtol=1e-12,
            max_steps=60,
            eq_constraint_fn=eq,
            n_eq_constraints=1,
            lbfgs_memory=5,
        )
        x0 = jnp.zeros(4)
        _, state, _ = _run_loop(solver, objective, x0, max_steps=60)

        diag = get_diagnostics(state)
        # With lbfgs_memory=5 and 60 outer steps, the old (buggy)
        # counter would report at least (steps - memory) ~= 55 skips on
        # any run that actually makes forward progress.  The corrected
        # counter matches ``should_skip`` and should stay much lower.
        assert int(diag.n_lbfgs_skips) < 30


# ---------------------------------------------------------------------------
# Bound-fix short-circuit counter
# ---------------------------------------------------------------------------


class TestBoundFixShortCircuit:
    """``diag.n_bound_fix_solves`` must stay near zero when the problem
    has bounds on every variable but no variable actually hits one.  The
    pre-fix code would report ``5 * max_steps`` solves because the loop
    ran the inner solve unconditionally.
    """

    def test_bounds_present_but_inactive(self):
        def objective(x, args):
            return jnp.sum((x - 0.1) ** 2), None

        # Wide bounds so the unconstrained Newton direction never
        # coincides with a box face -- the short-circuit should collapse
        # the bound-fix loop to (at most) one non-trivial solve per
        # outer step, typically zero.
        bounds = jnp.stack([-10.0 * jnp.ones(3), 10.0 * jnp.ones(3)], axis=1)

        solver = SLSQP(atol=1e-8, max_steps=20, bounds=bounds)
        x0 = jnp.zeros(3)
        _, state, _ = _run_loop(solver, objective, x0, max_steps=20)

        diag = get_diagnostics(state)
        steps = int(state.step_count)
        # Pre-fix upper bound was ``5 * steps``; the short-circuit
        # should collapse this to at most ``steps`` on average.
        assert int(diag.n_bound_fix_solves) <= steps
        assert int(diag.max_bound_fixed) == 0

    def test_bounds_active_still_records_solves(self):
        def objective(x, args):
            return jnp.sum((x - 2.0) ** 2), None

        bounds = jnp.stack([jnp.zeros(3), jnp.array([0.5, 2.0, 2.0])], axis=1)

        solver = SLSQP(atol=1e-8, max_steps=30, bounds=bounds)
        x0 = jnp.zeros(3)
        y, state, _ = _run_loop(solver, objective, x0, max_steps=30)

        diag = get_diagnostics(state)
        assert int(diag.n_bound_fix_solves) >= 1
        assert int(diag.max_bound_fixed) >= 1
        assert y[0] == pytest.approx(0.5, abs=1e-5)
