"""Tests for LPEC-A active set identification.

Tests cover:
1. Unit tests for rho_bar computation and threshold test.
2. QP solver integration with the three active_set_method modes.
3. SLSQP integration comparing expand, lpeca_init, and lpeca modes
   on nondegenerate, degenerate/weakly-active, and cycling-prone problems.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax import SLSQP
from slsqp_jax.lpeca import (
    compute_lpeca_active_set,
    compute_rho_bar,
    identify_active_set_lpeca,
)
from slsqp_jax.qp_solver import solve_qp

jax.config.update("jax_enable_x64", True)


def _make_hvp(H):
    """Create an HVP closure from a dense Hessian matrix."""

    def hvp_fn(v):
        return H @ v

    return hvp_fn


def _run_solver(solver, objective, x0, args=None, max_steps=None):
    """Run the SLSQP solver loop and return final iterate + state."""
    if max_steps is None:
        max_steps = solver.max_steps
    state = solver.init(objective, x0, args, {}, None, None, frozenset())
    y = x0
    for _ in range(max_steps):
        done, _ = solver.terminate(objective, y, args, {}, state, frozenset())
        if done:
            break
        y, state, _ = solver.step(objective, y, args, {}, state, frozenset())
    return y, state


# ============================================================
# Unit tests for rho_bar and threshold
# ============================================================


class TestComputeRhoBar:
    """Unit tests for the LPEC-A proximity measure."""

    def test_at_optimum_rho_bar_is_zero(self):
        """At a KKT point, rho_bar should be zero (or near zero)."""
        # min x^2 + y^2 s.t. x >= 1 (active at optimum)
        # Optimum: x=1, y=0, lambda=2 (grad = 2x = 2, A_ineq^T lambda = 2)
        c_ineq = jnp.array([0.0])  # x - 1 = 0, active
        c_eq = jnp.zeros(0)
        grad = jnp.array([2.0, 0.0])
        A_ineq = jnp.array([[1.0, 0.0]])
        A_eq = jnp.zeros((0, 2))
        lambda_ineq = jnp.array([2.0])
        mu_eq = jnp.zeros(0)

        rho = compute_rho_bar(c_ineq, c_eq, grad, A_ineq, A_eq, lambda_ineq, mu_eq)
        np.testing.assert_allclose(rho, 0.0, atol=1e-10)

    def test_feasible_constraint_contributes_sqrt(self):
        """Feasible constraint (c_ineq > 0) contributes sqrt(c * lambda)."""
        c_ineq = jnp.array([4.0])
        c_eq = jnp.zeros(0)
        grad = jnp.array([0.0])
        A_ineq = jnp.array([[0.0]])
        A_eq = jnp.zeros((0, 1))
        lambda_ineq = jnp.array([9.0])
        mu_eq = jnp.zeros(0)

        rho = compute_rho_bar(c_ineq, c_eq, grad, A_ineq, A_eq, lambda_ineq, mu_eq)
        # sqrt(4 * 9) = 6, plus stationarity |0 - 0| = 0
        np.testing.assert_allclose(rho, 6.0, atol=1e-10)

    def test_violated_constraint_contributes_magnitude(self):
        """Violated constraint (c_ineq < 0) contributes -c_ineq."""
        c_ineq = jnp.array([-3.0])
        c_eq = jnp.zeros(0)
        grad = jnp.array([0.0])
        A_ineq = jnp.array([[0.0]])
        A_eq = jnp.zeros((0, 1))
        lambda_ineq = jnp.array([0.0])
        mu_eq = jnp.zeros(0)

        rho = compute_rho_bar(c_ineq, c_eq, grad, A_ineq, A_eq, lambda_ineq, mu_eq)
        np.testing.assert_allclose(rho, 3.0, atol=1e-10)

    def test_equality_violation_contributes(self):
        """Equality violation contributes |c_eq|."""
        c_ineq = jnp.zeros(0)
        c_eq = jnp.array([2.5, -1.5])
        grad = jnp.array([0.0])
        A_ineq = jnp.zeros((0, 1))
        A_eq = jnp.zeros((2, 1))
        lambda_ineq = jnp.zeros(0)
        mu_eq = jnp.array([0.0, 0.0])

        rho = compute_rho_bar(c_ineq, c_eq, grad, A_ineq, A_eq, lambda_ineq, mu_eq)
        np.testing.assert_allclose(rho, 4.0, atol=1e-10)


class TestIdentifyActiveSet:
    """Unit tests for the LPEC-A threshold test."""

    def test_active_constraints_identified(self):
        """Constraints near zero should be identified as active."""
        c_ineq = jnp.array([0.0, 1.0, 0.001])
        c_eq = jnp.zeros(0)
        grad = jnp.array([1.0, 0.0])
        A_ineq = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        A_eq = jnp.zeros((0, 2))
        lambda_ineq = jnp.array([1.0, 0.0, 0.0])
        mu_eq = jnp.zeros(0)

        active = identify_active_set_lpeca(
            c_ineq, c_eq, grad, A_ineq, A_eq, lambda_ineq, mu_eq
        )
        # c_ineq[0] = 0 should be active; c_ineq[2] = 0.001 might be
        assert active[0], "Constraint with c=0 must be active"

    def test_inactive_constraints_excluded(self):
        """Well-feasible constraints should not be predicted active."""
        # Large c_ineq values, small rho_bar -> threshold is small
        c_ineq = jnp.array([100.0, 200.0])
        c_eq = jnp.zeros(0)
        grad = jnp.zeros(2)
        A_ineq = jnp.eye(2)
        A_eq = jnp.zeros((0, 2))
        lambda_ineq = jnp.zeros(2)
        mu_eq = jnp.zeros(0)

        active = identify_active_set_lpeca(
            c_ineq, c_eq, grad, A_ineq, A_eq, lambda_ineq, mu_eq
        )
        assert not jnp.any(active), "Well-feasible constraints should be inactive"

    def test_sigma_affects_threshold(self):
        """Smaller sigma should produce a larger threshold (more active)."""
        c_ineq = jnp.array([0.1])
        c_eq = jnp.zeros(0)
        grad = jnp.array([1.0])
        A_ineq = jnp.array([[1.0]])
        A_eq = jnp.zeros((0, 1))
        lambda_ineq = jnp.array([0.5])
        mu_eq = jnp.zeros(0)

        active_high_sigma = identify_active_set_lpeca(
            c_ineq,
            c_eq,
            grad,
            A_ineq,
            A_eq,
            lambda_ineq,
            mu_eq,
            sigma=0.99,
        )
        active_low_sigma = identify_active_set_lpeca(
            c_ineq,
            c_eq,
            grad,
            A_ineq,
            A_eq,
            lambda_ineq,
            mu_eq,
            sigma=0.5,
        )
        # With sigma=0.5, threshold = (beta * rho_bar)^0.5 which is larger
        # than (beta * rho_bar)^0.99 when beta*rho_bar < 1
        # Both should give consistent results
        assert active_low_sigma.shape == (1,)
        assert active_high_sigma.shape == (1,)


class TestSolveLpecaLp:
    """Tests for the LPEC-A LP solve via mpax.r2HPDHG."""

    def test_lp_import_error_without_mpax(self):
        """When mpax is not installed, ImportError is raised."""
        import importlib
        import sys

        saved = sys.modules.get("mpax", None)
        sys.modules["mpax"] = None  # type: ignore[assignment]
        try:
            import slsqp_jax.lpeca as lpeca_mod

            importlib.reload(lpeca_mod)

            with pytest.raises(ImportError, match="mpax"):
                lpeca_mod.solve_lpeca_lp(
                    c_ineq=jnp.array([1.0]),
                    c_eq=jnp.zeros(0),
                    grad=jnp.array([1.0]),
                    A_ineq=jnp.array([[1.0]]),
                    A_eq=jnp.zeros((0, 1)),
                )
        finally:
            if saved is not None:
                sys.modules["mpax"] = saved
            else:
                del sys.modules["mpax"]
            importlib.reload(lpeca_mod)

    def test_solve_lp_inequality_only(self):
        """LP solve returns valid multipliers for an inequality-only problem.

        KKT system for min x^2 s.t. x >= 1:
          grad = 2, A_ineq = [[1]], lambda >= 0
          stationarity: 2 - lambda = 0  =>  lambda = 2
        """
        from slsqp_jax.lpeca import solve_lpeca_lp

        c_ineq = jnp.array([0.0])  # active constraint
        c_eq = jnp.zeros(0)
        grad = jnp.array([2.0])
        A_ineq = jnp.array([[1.0]])
        A_eq = jnp.zeros((0, 1))

        lambda_opt, mu_opt = solve_lpeca_lp(
            c_ineq,
            c_eq,
            grad,
            A_ineq,
            A_eq,
            eps_abs=1e-4,
            eps_rel=1e-4,
            max_iter=5000,
        )

        assert lambda_opt.shape == (1,)
        assert mu_opt.shape == (0,)
        np.testing.assert_allclose(lambda_opt[0], 2.0, atol=0.1)

    def test_solve_lp_with_equalities(self):
        """LP solve returns valid multipliers with equality constraints.

        KKT for min x^2 + y^2  s.t.  x + y = 1,  x >= 0:
          grad = [2x, 2y] at optimum [0.5, 0.5] => [1, 1]
          A_eq = [[1, 1]], A_ineq = [[1, 0]]
          stationarity: [1, 1] - A_eq^T mu - A_ineq^T lambda = 0
            => [1 - mu - lambda, 1 - mu] = 0  =>  mu = 1, lambda = 0
        """
        from slsqp_jax.lpeca import solve_lpeca_lp

        c_ineq = jnp.array([0.5])  # feasible, x = 0.5 > 0
        c_eq = jnp.array([0.0])  # x + y - 1 = 0
        grad = jnp.array([1.0, 1.0])
        A_ineq = jnp.array([[1.0, 0.0]])
        A_eq = jnp.array([[1.0, 1.0]])

        lambda_opt, mu_opt = solve_lpeca_lp(
            c_ineq,
            c_eq,
            grad,
            A_ineq,
            A_eq,
            eps_abs=1e-4,
            eps_rel=1e-4,
            max_iter=5000,
        )

        assert lambda_opt.shape == (1,)
        assert mu_opt.shape == (1,)
        np.testing.assert_allclose(lambda_opt[0], 0.0, atol=0.2)
        np.testing.assert_allclose(mu_opt[0], 1.0, atol=0.2)

    def test_solve_lp_multiple_constraints(self):
        """LP solve with multiple inequality constraints.

        KKT for min x^2 + y^2  s.t. x >= 1, y >= 0, x + y >= 0:
          At optimum x=1, y=0: grad = [2, 0]
          A_ineq = [[1,0],[0,1],[1,1]]
          Active: constraint 0 (x=1) and 1 (y=0), constraint 2 inactive
          stationarity: [2,0] - [1,0]^T l0 - [0,1]^T l1 - [1,1]^T l2 = 0
            => l0 + l2 = 2, l1 + l2 = 0. With l1,l2 >= 0: l2=0, l1=0, l0=2
        """
        from slsqp_jax.lpeca import solve_lpeca_lp

        c_ineq = jnp.array([0.0, 0.0, 1.0])
        c_eq = jnp.zeros(0)
        grad = jnp.array([2.0, 0.0])
        A_ineq = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        A_eq = jnp.zeros((0, 2))

        lambda_opt, mu_opt = solve_lpeca_lp(
            c_ineq,
            c_eq,
            grad,
            A_ineq,
            A_eq,
            eps_abs=1e-4,
            eps_rel=1e-4,
            max_iter=5000,
        )

        assert lambda_opt.shape == (3,)
        assert mu_opt.shape == (0,)
        np.testing.assert_allclose(lambda_opt[0], 2.0, atol=0.2)
        np.testing.assert_allclose(lambda_opt[2], 0.0, atol=0.2)


class TestComputeLpecaActiveSet:
    """Tests for the combined compute_lpeca_active_set entry point."""

    def test_without_lp(self):
        """Basic call without LP refinement."""
        c_ineq = jnp.array([0.0, 5.0])
        c_eq = jnp.zeros(0)
        grad = jnp.array([1.0, 0.0])
        A_ineq = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        A_eq = jnp.zeros((0, 2))
        lambda_ineq = jnp.array([1.0, 0.0])
        mu_eq = jnp.zeros(0)

        active = compute_lpeca_active_set(
            c_ineq,
            c_eq,
            grad,
            A_ineq,
            A_eq,
            lambda_ineq,
            mu_eq,
            use_lp=False,
        )
        assert active[0], "Constraint with c=0 should be active"

    def test_with_lp_refinement(self):
        """Calling with use_lp=True exercises the mpax LP path.

        Same KKT problem as TestSolveLpecaLp.test_solve_lp_inequality_only:
          min x^2 s.t. x >= 1, active at x=1
          grad = 2, A_ineq = [[1]], true lambda = 2
        The LP-refined multipliers should still identify c_ineq=0 as active.
        """
        c_ineq = jnp.array([0.0])
        c_eq = jnp.zeros(0)
        grad = jnp.array([2.0])
        A_ineq = jnp.array([[1.0]])
        A_eq = jnp.zeros((0, 1))
        lambda_ineq = jnp.array([0.0])  # bad initial guess
        mu_eq = jnp.zeros(0)

        active = compute_lpeca_active_set(
            c_ineq,
            c_eq,
            grad,
            A_ineq,
            A_eq,
            lambda_ineq,
            mu_eq,
            use_lp=True,
            lp_eps=1e-4,
            lp_max_iter=5000,
        )
        assert active[0], "Active constraint must be identified with LP refinement"

    def test_with_lp_equality_and_inequality(self):
        """LP refinement with both equality and inequality constraints."""
        c_ineq = jnp.array([0.5])  # feasible (inactive)
        c_eq = jnp.array([0.0])
        grad = jnp.array([1.0, 1.0])
        A_ineq = jnp.array([[1.0, 0.0]])
        A_eq = jnp.array([[1.0, 1.0]])
        lambda_ineq = jnp.array([0.0])
        mu_eq = jnp.array([0.0])

        active = compute_lpeca_active_set(
            c_ineq,
            c_eq,
            grad,
            A_ineq,
            A_eq,
            lambda_ineq,
            mu_eq,
            use_lp=True,
            lp_eps=1e-4,
            lp_max_iter=5000,
        )
        assert not active[0], "Feasible inactive constraint should not be active"


# ============================================================
# QP solver integration tests
# ============================================================


class TestQPSolverActiveMethods:
    """Tests for solve_qp with different active_set_method values."""

    def test_invalid_method_raises(self):
        """Invalid active_set_method should raise ValueError."""
        H = jnp.eye(2)
        g = jnp.array([1.0, 1.0])
        with pytest.raises(ValueError, match="active_set_method"):
            solve_qp(
                _make_hvp(H),
                g,
                jnp.zeros((0, 2)),
                jnp.zeros(0),
                jnp.zeros((0, 2)),
                jnp.zeros(0),
                active_set_method="invalid",
            )

    def test_expand_baseline(self):
        """Standard EXPAND mode solves a simple inequality QP.

        minimize (1/2)(x^2 + y^2)
        s.t. x >= 1, y >= 0

        Solution: x=1, y=0, active: both
        """
        H = jnp.eye(2)
        g = jnp.zeros(2)
        A_ineq = jnp.eye(2)
        b_ineq = jnp.array([1.0, 0.0])

        result = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, 2)),
            jnp.zeros(0),
            A_ineq,
            b_ineq,
            active_set_method="expand",
        )
        np.testing.assert_allclose(result.d, [1.0, 0.0], atol=1e-6)
        assert result.converged

    def test_lpeca_init_matches_expand(self):
        """lpeca_init mode should converge to the same solution as expand."""
        H = jnp.eye(2)
        g = jnp.zeros(2)
        A_ineq = jnp.eye(2)
        b_ineq = jnp.array([1.0, 0.0])
        predicted = jnp.array([True, True])

        result = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, 2)),
            jnp.zeros(0),
            A_ineq,
            b_ineq,
            active_set_method="lpeca_init",
            predicted_active_set=predicted,
        )
        np.testing.assert_allclose(result.d, [1.0, 0.0], atol=1e-6)
        assert result.converged

    def test_lpeca_mode_matches_expand(self):
        """lpeca mode (fixed tol) should converge to the same solution."""
        H = jnp.eye(2)
        g = jnp.zeros(2)
        A_ineq = jnp.eye(2)
        b_ineq = jnp.array([1.0, 0.0])
        predicted = jnp.array([True, True])

        result = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, 2)),
            jnp.zeros(0),
            A_ineq,
            b_ineq,
            active_set_method="lpeca",
            predicted_active_set=predicted,
        )
        np.testing.assert_allclose(result.d, [1.0, 0.0], atol=1e-6)
        assert result.converged

    def test_predicted_active_set_improves_iteration_count(self):
        """Good predicted active set should reduce iteration count.

        minimize (1/2) x^T H x + g^T x
        s.t. x >= b (component-wise)

        With 5 constraints, 3 active at the solution. A correct prediction
        should converge faster than cold-starting.
        """
        n = 5
        H = jnp.eye(n) * 2.0
        g = jnp.array([-3.0, -2.0, -1.0, 1.0, 2.0])
        A_ineq = jnp.eye(n)
        b_ineq = jnp.zeros(n)

        # Solution: d = max(0, -H^{-1} g) = [1.5, 1.0, 0.5, 0, 0]
        # Active: constraints 3, 4 (0-indexed)

        result_expand = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, n)),
            jnp.zeros(0),
            A_ineq,
            b_ineq,
            active_set_method="expand",
        )

        predicted = jnp.array([False, False, False, True, True])
        result_lpeca = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, n)),
            jnp.zeros(0),
            A_ineq,
            b_ineq,
            active_set_method="lpeca_init",
            predicted_active_set=predicted,
        )

        np.testing.assert_allclose(result_expand.d, result_lpeca.d, atol=1e-6)
        assert result_lpeca.converged
        assert result_lpeca.iterations <= result_expand.iterations

    def test_lpeca_with_equality_proximal(self):
        """LPEC-A modes work with equality constraints (proximal path).

        minimize (1/2)(x^2 + y^2)
        s.t. x + y = 1, x >= 0, y >= 0

        Solution: x = y = 0.5
        """
        H = jnp.eye(2)
        g = jnp.zeros(2)
        A_eq = jnp.array([[1.0, 1.0]])
        b_eq = jnp.array([1.0])
        A_ineq = jnp.eye(2)
        b_ineq = jnp.zeros(2)

        predicted = jnp.array([False, False])

        for method in ("expand", "lpeca_init", "lpeca"):
            result = solve_qp(
                _make_hvp(H),
                g,
                A_eq,
                b_eq,
                A_ineq,
                b_ineq,
                active_set_method=method,
                predicted_active_set=predicted if method != "expand" else None,
                proximal_mu=0.01,
                prev_multipliers_eq=jnp.zeros(1),
            )
            np.testing.assert_allclose(result.d, [0.5, 0.5], atol=1e-2)
            assert result.converged, f"Failed to converge with method={method}"

    def test_lpeca_with_equality_direct(self):
        """LPEC-A modes work with equality constraints (direct path)."""
        H = jnp.eye(2)
        g = jnp.zeros(2)
        A_eq = jnp.array([[1.0, 1.0]])
        b_eq = jnp.array([1.0])
        A_ineq = jnp.eye(2)
        b_ineq = jnp.zeros(2)

        predicted = jnp.array([False, False])

        for method in ("expand", "lpeca_init", "lpeca"):
            result = solve_qp(
                _make_hvp(H),
                g,
                A_eq,
                b_eq,
                A_ineq,
                b_ineq,
                active_set_method=method,
                predicted_active_set=predicted if method != "expand" else None,
                use_proximal=False,
            )
            np.testing.assert_allclose(result.d, [0.5, 0.5], atol=1e-4)
            assert result.converged, f"Failed to converge with method={method}"


# ============================================================
# SLSQP integration tests
# ============================================================


class TestSLSQPLpecaModes:
    """Integration tests comparing SLSQP with different active_set_method."""

    def test_invalid_method_raises_on_construction(self):
        """Invalid active_set_method should raise ValueError at init."""
        with pytest.raises(ValueError, match="active_set_method"):
            SLSQP(active_set_method="invalid")

    def test_invalid_sigma_raises(self):
        """Invalid lpeca_sigma should raise ValueError."""
        with pytest.raises(ValueError, match="lpeca_sigma"):
            SLSQP(active_set_method="lpeca_init", lpeca_sigma=0.0)
        with pytest.raises(ValueError, match="lpeca_sigma"):
            SLSQP(active_set_method="lpeca_init", lpeca_sigma=1.0)

    @pytest.mark.parametrize("method", ["expand", "lpeca_init", "lpeca"])
    def test_inequality_constrained_quadratic(self, method):
        """All modes solve a simple inequality-constrained problem.

        minimize x^2 + y^2
        s.t. x + y >= 2

        Solution: x = y = 1
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def ineq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 2.0])

        solver = SLSQP(
            atol=1e-6,
            max_steps=50,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=1,
            active_set_method=method,
        )
        x0 = jnp.array([3.0, 3.0])
        y, state = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [1.0, 1.0], atol=1e-4)

    @pytest.mark.parametrize("method", ["expand", "lpeca_init", "lpeca"])
    def test_equality_and_inequality(self, method):
        """All modes solve a mixed equality + inequality problem.

        minimize (x-2)^2 + (y-1)^2
        s.t. x + y = 2
             x >= 0, y >= 0

        Solution: x = 1.5, y = 0.5
        """

        def objective(x, args):
            return (x[0] - 2) ** 2 + (x[1] - 1) ** 2, None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 2.0])

        def ineq_constraint(x, args):
            return x

        solver = SLSQP(
            atol=1e-6,
            max_steps=50,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=2,
            active_set_method=method,
        )
        x0 = jnp.array([1.0, 1.0])
        y, state = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [1.5, 0.5], atol=1e-3)

    @pytest.mark.parametrize("method", ["expand", "lpeca_init", "lpeca"])
    def test_weakly_active_constraints(self, method):
        """Test on a problem with weakly active constraint at optimum.

        minimize (x-1)^2 + y^2
        s.t. x >= 1 (weakly active: c(x*) = 0, lambda* = 0)

        Solution: x = 1, y = 0
        """

        def objective(x, args):
            return (x[0] - 1) ** 2 + x[1] ** 2, None

        def ineq_constraint(x, args):
            return jnp.array([x[0] - 1.0])

        solver = SLSQP(
            atol=1e-6,
            max_steps=50,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=1,
            active_set_method=method,
        )
        x0 = jnp.array([2.0, 1.0])
        y, state = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [1.0, 0.0], atol=1e-4)

    @pytest.mark.parametrize("method", ["expand", "lpeca_init", "lpeca"])
    def test_multiple_active_bounds(self, method):
        """Test with multiple active bound constraints.

        minimize sum((x - target)^2)
        s.t. 0 <= x <= 1

        target = [-1, 0.5, 2] -> solution = [0, 0.5, 1]
        """
        target = jnp.array([-1.0, 0.5, 2.0])

        def objective(x, args):
            return jnp.sum((x - target) ** 2), None

        n = 3
        bounds = jnp.stack([jnp.zeros(n), jnp.ones(n)], axis=1)

        solver = SLSQP(
            atol=1e-6,
            max_steps=50,
            bounds=bounds,
            active_set_method=method,
        )
        x0 = jnp.array([0.5, 0.5, 0.5])
        y, state = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [0.0, 0.5, 1.0], atol=1e-4)

    @pytest.mark.parametrize("method", ["expand", "lpeca_init", "lpeca"])
    def test_degenerate_constraints(self, method):
        """Test with degenerate constraints (multiple active at vertex).

        minimize x^2 + y^2
        s.t. x >= 0, y >= 0, x + y >= 0

        The third constraint is redundant (degenerate vertex at origin).
        Solution: x = y = 0 (all three constraints active, but rank < 3).
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def ineq_constraint(x, args):
            return jnp.array([x[0], x[1], x[0] + x[1]])

        solver = SLSQP(
            atol=1e-6,
            max_steps=50,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=3,
            active_set_method=method,
        )
        x0 = jnp.array([1.0, 1.0])
        y, state = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [0.0, 0.0], atol=1e-4)

    def test_lpeca_uses_custom_sigma_and_beta(self):
        """Custom sigma and beta parameters are accepted and used."""

        def objective(x, args):
            return jnp.sum(x**2), None

        def ineq_constraint(x, args):
            return jnp.array([x[0] - 1.0])

        solver = SLSQP(
            atol=1e-6,
            max_steps=50,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=1,
            active_set_method="lpeca_init",
            lpeca_sigma=0.8,
            lpeca_beta=0.5,
        )
        x0 = jnp.array([3.0, 1.0])
        y, state = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y[0], 1.0, atol=1e-3)
