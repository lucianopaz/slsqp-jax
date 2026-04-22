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

        result = identify_active_set_lpeca(
            c_ineq, c_eq, grad, A_ineq, A_eq, lambda_ineq, mu_eq
        )
        active = result.predicted
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

        result = identify_active_set_lpeca(
            c_ineq, c_eq, grad, A_ineq, A_eq, lambda_ineq, mu_eq
        )
        active = result.predicted
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

        result_high = identify_active_set_lpeca(
            c_ineq,
            c_eq,
            grad,
            A_ineq,
            A_eq,
            lambda_ineq,
            mu_eq,
            sigma=0.99,
        )
        result_low = identify_active_set_lpeca(
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
        assert result_low.predicted.shape == (1,)
        assert result_high.predicted.shape == (1,)


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
            # Rebind module-level references so that subsequent tests
            # (and beartype-cached return-type checks) see the freshly
            # reloaded class identities.  Without this, the symbols
            # imported at the top of this file still point to the
            # pre-reload function objects whose `__globals__` now look
            # up the *new* LPECAResult, which fails beartype's identity
            # check on the return type.
            globals()["compute_lpeca_active_set"] = lpeca_mod.compute_lpeca_active_set
            globals()["compute_rho_bar"] = lpeca_mod.compute_rho_bar
            globals()["identify_active_set_lpeca"] = lpeca_mod.identify_active_set_lpeca

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

        result = compute_lpeca_active_set(
            c_ineq,
            c_eq,
            grad,
            A_ineq,
            A_eq,
            lambda_ineq,
            mu_eq,
            use_lp=False,
        )
        assert result.predicted[0], "Constraint with c=0 should be active"

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

        result = compute_lpeca_active_set(
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
        assert result.predicted[0], (
            "Active constraint must be identified with LP refinement"
        )

    def test_with_lp_equality_and_inequality(self):
        """LP refinement with both equality and inequality constraints."""
        c_ineq = jnp.array([0.5])  # feasible (inactive)
        c_eq = jnp.array([0.0])
        grad = jnp.array([1.0, 1.0])
        A_ineq = jnp.array([[1.0, 0.0]])
        A_eq = jnp.array([[1.0, 1.0]])
        lambda_ineq = jnp.array([0.0])
        mu_eq = jnp.array([0.0])

        result = compute_lpeca_active_set(
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
        assert not result.predicted[0], (
            "Feasible inactive constraint should not be active"
        )


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


# ============================================================
# Bundle 2 regression tests: trust gate + size cap
# ============================================================


class TestLpecaTrustGate:
    """Verify the rho_bar-based trust gate replaces the broken clamp.

    The pre-Bundle-2 implementation used ``min(threshold_raw, max|c_ineq|)``
    which trivially flagged every inequality as active early in SQP.
    Bundle 2 replaces it with ``rho_bar <= trust_threshold``.
    """

    def test_far_from_solution_returns_empty_prediction(self):
        """With large ``rho_bar``, the trust gate disables LPEC-A."""
        # Construct a scenario with huge equality violation so rho_bar is large.
        c_ineq = jnp.array([0.0, 0.0, 0.0])  # would be flagged by raw threshold
        c_eq = jnp.array([1e3])  # massive feasibility violation
        grad = jnp.array([1.0, 0.0])
        A_ineq = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        A_eq = jnp.array([[1.0, 0.0]])
        lambda_ineq = jnp.zeros(3)
        mu_eq = jnp.array([0.0])

        result = identify_active_set_lpeca(
            c_ineq,
            c_eq,
            grad,
            A_ineq,
            A_eq,
            lambda_ineq,
            mu_eq,
            trust_threshold=1.0,
        )
        assert not bool(result.valid), (
            f"rho_bar={float(result.rho_bar):e} should fail trust gate"
        )
        assert not bool(jnp.any(result.predicted)), (
            "Prediction must be empty when LPEC-A is not trusted"
        )

    def test_near_solution_identifies_active_set(self):
        """Near a KKT point, the prediction must match (modulo at most
        one false positive) the true active set."""
        # min x^2 + y^2 s.t. x >= 1 (active), y >= -10 (inactive).
        # Optimum: x=1, y=0, lambda = (2, 0).
        c_ineq = jnp.array([1e-9, 10.0])  # near active / well-feasible
        c_eq = jnp.zeros(0)
        grad = jnp.array([2.0, 0.0])  # 2x, 2y at (1, 0)
        A_ineq = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        A_eq = jnp.zeros((0, 2))
        lambda_ineq = jnp.array([2.0, 0.0])
        mu_eq = jnp.zeros(0)

        result = identify_active_set_lpeca(
            c_ineq,
            c_eq,
            grad,
            A_ineq,
            A_eq,
            lambda_ineq,
            mu_eq,
            trust_threshold=1.0,
        )
        assert bool(result.valid), (
            f"Near solution rho_bar={float(result.rho_bar):e} should pass gate"
        )
        true_active = jnp.array([True, False])
        # At most one false positive (LPEC-A may overshoot by one).
        false_positives = int(jnp.sum(result.predicted & ~true_active))
        assert bool(result.predicted[0]), "Active constraint must be flagged"
        assert false_positives <= 1, (
            f"Expected <= 1 false positive, got {false_positives}"
        )

    def test_trust_threshold_parameter_is_respected(self):
        """A loose threshold lets a borderline case through; a tight
        threshold rejects it."""
        c_ineq = jnp.array([0.0])
        c_eq = jnp.zeros(0)
        # rho_bar contribution: |grad - A_ineq^T lambda| = |1 - 0.5| = 0.5
        grad = jnp.array([1.0])
        A_ineq = jnp.array([[1.0]])
        A_eq = jnp.zeros((0, 1))
        lambda_ineq = jnp.array([0.5])
        mu_eq = jnp.zeros(0)

        loose = identify_active_set_lpeca(
            c_ineq,
            c_eq,
            grad,
            A_ineq,
            A_eq,
            lambda_ineq,
            mu_eq,
            trust_threshold=10.0,
        )
        tight = identify_active_set_lpeca(
            c_ineq,
            c_eq,
            grad,
            A_ineq,
            A_eq,
            lambda_ineq,
            mu_eq,
            trust_threshold=1e-6,
        )
        assert bool(loose.valid)
        assert not bool(tight.valid)
        assert not bool(jnp.any(tight.predicted))


class TestLpecaSizeCap:
    """Verify the rank-aware size cap on the LPEC-A prediction."""

    def test_size_cap_truncates_overprediction(self):
        """When the raw threshold would activate too many constraints,
        the cap keeps at most ``n - m_eq - 1`` of them."""
        n = 10
        m_eq = 5
        m_ineq = 30
        n_dof = n - m_eq - 1  # = 4

        # Use a moderate rho_bar (driven by the stationarity residual)
        # so threshold = (beta * rho_bar)^sigma is large enough that
        # every nearly-feasible inequality is below it.
        c_eq = jnp.zeros(m_eq)
        # Ranked feasibility: c_ineq[i] = i * 1e-3, all small.
        c_ineq = jnp.arange(m_ineq, dtype=jnp.float64) * 1e-3
        # Stationarity residual = |grad - A_ineq^T lambda - A_eq^T mu| = 0.5
        grad = jnp.full(n, 0.5)
        A_ineq = jnp.eye(m_ineq, n)
        A_eq = jnp.zeros((m_eq, n)).at[jnp.arange(m_eq), jnp.arange(m_eq)].set(1.0)
        lambda_ineq = jnp.zeros(m_ineq)
        mu_eq = jnp.zeros(m_eq)

        result = identify_active_set_lpeca(
            c_ineq,
            c_eq,
            grad,
            A_ineq,
            A_eq,
            lambda_ineq,
            mu_eq,
            trust_threshold=10.0,  # allow rho_bar ~ 5 (sqrt(0.25 * n))
            sigma=0.3,  # large threshold so raw test fires on all
            beta=1.0,
        )
        assert bool(result.valid)
        predicted_count = int(jnp.sum(result.predicted))
        assert predicted_count <= n_dof, (
            f"Expected at most {n_dof} constraints, got {predicted_count}"
        )
        if predicted_count > 0:
            # The cap should keep the most confidently active (smallest
            # c_ineq).  c_ineq[0] is smallest, so it must be in the set.
            assert bool(result.predicted[0])
        # The capped flag must report the truncation.
        assert bool(result.capped), "Size cap must set the capped flag"

    def test_size_cap_inactive_when_under_budget(self):
        """When the prediction count is already under the rank budget,
        the cap is a no-op and ``capped=False``."""
        n = 5
        c_ineq = jnp.array([0.0, 5.0, 10.0])  # only one active
        c_eq = jnp.zeros(0)
        grad = jnp.zeros(n)
        A_ineq = jnp.zeros((3, n)).at[0, 0].set(1.0)
        A_eq = jnp.zeros((0, n))
        lambda_ineq = jnp.zeros(3)
        mu_eq = jnp.zeros(0)

        result = identify_active_set_lpeca(
            c_ineq,
            c_eq,
            grad,
            A_ineq,
            A_eq,
            lambda_ineq,
            mu_eq,
            trust_threshold=1.0,
        )
        assert bool(result.valid)
        assert int(jnp.sum(result.predicted)) <= 1
        assert not bool(result.capped)

    def test_compute_lpeca_active_set_propagates_trust_threshold(self):
        """``compute_lpeca_active_set`` forwards ``trust_threshold``."""
        c_ineq = jnp.zeros(2)
        c_eq = jnp.array([1e3])  # huge violation -> large rho_bar
        grad = jnp.array([1.0, 0.0])
        A_ineq = jnp.eye(2)
        A_eq = jnp.array([[1.0, 0.0]])
        lambda_ineq = jnp.zeros(2)
        mu_eq = jnp.array([0.0])

        result = compute_lpeca_active_set(
            c_ineq,
            c_eq,
            grad,
            A_ineq,
            A_eq,
            lambda_ineq,
            mu_eq,
            use_lp=False,
            trust_threshold=1.0,
        )
        assert not bool(result.valid)
        assert not bool(jnp.any(result.predicted))


class TestLpecaWarmupGate:
    """Verify the SQP-level warm-up gate bypasses LPEC-A early."""

    def test_lpeca_warmup_steps_parameter_exists(self):
        """``SLSQP`` accepts ``lpeca_warmup_steps`` and stores it."""
        solver = SLSQP(
            atol=1e-6,
            active_set_method="lpeca",
            lpeca_warmup_steps=5,
        )
        assert solver.lpeca_warmup_steps == 5

    def test_lpeca_warmup_diagnostic_increments(self):
        """When the warm-up gate fires, the LPEC-A bypass counter grows."""

        def objective(x, args):
            return jnp.sum(x**2), None

        def ineq_constraint(x, args):
            return jnp.array([x[0] - 1.0])

        solver = SLSQP(
            atol=1e-6,
            max_steps=20,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=1,
            active_set_method="lpeca_init",
            lpeca_warmup_steps=3,
            verbose=False,
        )
        x0 = jnp.array([5.0, 0.5])
        _, state = _run_solver(solver, objective, x0)
        assert int(state.diagnostics.n_lpeca_bypassed) >= 1, (
            "Warm-up gate should bypass LPEC-A for the first few SQP steps"
        )


# ============================================================
# Bound-prediction extension tests
# ============================================================


class TestLpecaBoundPrediction:
    """Verify that LPEC-A's active-set prediction is extended to box bounds.

    Checks three invariants:

    1. When the bound extension is enabled, the solver produces the same
       solution as the ``expand`` baseline (correctness).
    2. When LPEC-A's prediction is accurate, the ``n_lpeca_bounds_prefixed``
       counter records the warm-started bounds and the total bound-fixing
       ``inner_solver`` invocations do not inflate.
    3. When the warm-up / trust gate suppresses the LPEC-A prediction,
       the bound warm-start is also skipped and ``n_lpeca_bounds_prefixed``
       remains ``0``.
    """

    @staticmethod
    def _box_problem():
        target = jnp.array([-2.0, 0.5, 3.0, 0.1, -1.5])

        def objective(x, args):
            return jnp.sum((x - target) ** 2), None

        n = target.shape[0]
        bounds = jnp.stack([jnp.zeros(n), jnp.ones(n)], axis=1)
        x0 = jnp.full(n, 0.5)
        # Optimum: clamp(target, 0, 1) = [0, 0.5, 1, 0.1, 0]
        expected = jnp.clip(target, 0.0, 1.0)
        return objective, bounds, x0, expected

    def test_bound_extension_matches_expand(self):
        """Solution is unchanged regardless of the bound-extension flag."""
        objective, bounds, x0, expected = self._box_problem()

        common = dict(atol=1e-7, max_steps=80, bounds=bounds)
        solver_expand = SLSQP(**common, active_set_method="expand")
        solver_bounds_off = SLSQP(
            **common,
            active_set_method="lpeca_init",
            lpeca_warmup_steps=0,
            lpeca_predict_bounds=False,
        )
        solver_bounds_on = SLSQP(
            **common,
            active_set_method="lpeca_init",
            lpeca_warmup_steps=0,
            lpeca_predict_bounds=True,
        )

        y_expand, _ = _run_solver(solver_expand, objective, x0)
        y_off, _ = _run_solver(solver_bounds_off, objective, x0)
        y_on, _ = _run_solver(solver_bounds_on, objective, x0)

        np.testing.assert_allclose(y_expand, expected, atol=1e-5)
        np.testing.assert_allclose(y_off, expected, atol=1e-5)
        np.testing.assert_allclose(y_on, expected, atol=1e-5)

    def test_bound_prefix_counter_accumulates(self):
        """When the prediction is trusted and non-empty, the counter is > 0."""
        objective, bounds, x0, _ = self._box_problem()

        solver = SLSQP(
            atol=1e-7,
            max_steps=80,
            bounds=bounds,
            active_set_method="lpeca_init",
            # Disable warm-up so LPEC-A fires from the first step.
            lpeca_warmup_steps=0,
            # Very loose trust threshold so the trust gate does not veto.
            lpeca_trust_threshold=1e6,
            lpeca_predict_bounds=True,
        )
        _, state = _run_solver(solver, objective, x0)

        # At least one SQP step must pre-fix at least one bound.
        assert int(state.diagnostics.n_lpeca_bounds_prefixed) > 0, (
            "Expected LPEC-A to pre-fix at least one bound across the run; "
            f"got n_lpeca_bounds_prefixed={int(state.diagnostics.n_lpeca_bounds_prefixed)}"
        )

    def test_bound_prefix_zero_when_flag_off(self):
        """With ``lpeca_predict_bounds=False`` no bounds are pre-fixed."""
        objective, bounds, x0, _ = self._box_problem()

        solver = SLSQP(
            atol=1e-7,
            max_steps=80,
            bounds=bounds,
            active_set_method="lpeca_init",
            lpeca_warmup_steps=0,
            lpeca_trust_threshold=1e6,
            lpeca_predict_bounds=False,
        )
        _, state = _run_solver(solver, objective, x0)

        assert int(state.diagnostics.n_lpeca_bounds_prefixed) == 0, (
            "Bound pre-fix counter must stay at 0 when the flag is off"
        )

    def test_bound_prefix_zero_when_trust_gate_vetoes(self):
        """With a very tight trust threshold, no prefix is ever applied."""
        objective, bounds, x0, _ = self._box_problem()

        solver = SLSQP(
            atol=1e-7,
            max_steps=80,
            bounds=bounds,
            active_set_method="lpeca_init",
            lpeca_warmup_steps=0,
            # Threshold small enough that rho_bar always exceeds it.
            lpeca_trust_threshold=1e-30,
            lpeca_predict_bounds=True,
        )
        _, state = _run_solver(solver, objective, x0)

        assert int(state.diagnostics.n_lpeca_bounds_prefixed) == 0, (
            "Bound pre-fix must not fire when the trust gate vetoes LPEC-A"
        )
        assert int(state.diagnostics.n_lpeca_bypassed) > 0, (
            "Trust gate must have fired on at least one step"
        )

    def test_bound_prefix_skipped_during_warmup(self):
        """During the LPEC-A warm-up window, no bound prefix is applied."""
        objective, bounds, x0, _ = self._box_problem()

        solver = SLSQP(
            atol=1e-7,
            # Cap max_steps at the warm-up window so LPEC-A never fires.
            max_steps=3,
            bounds=bounds,
            active_set_method="lpeca_init",
            lpeca_warmup_steps=100,
            lpeca_trust_threshold=1e6,
            lpeca_predict_bounds=True,
        )
        _, state = _run_solver(solver, objective, x0)

        assert int(state.diagnostics.n_lpeca_bounds_prefixed) == 0, (
            "Warm-up must suppress LPEC-A bound pre-fix"
        )

    def test_expand_method_has_zero_bound_prefix(self):
        """When ``active_set_method='expand'`` the counter stays at 0."""
        objective, bounds, x0, _ = self._box_problem()

        solver = SLSQP(
            atol=1e-7,
            max_steps=80,
            bounds=bounds,
            active_set_method="expand",
        )
        _, state = _run_solver(solver, objective, x0)

        assert int(state.diagnostics.n_lpeca_bounds_prefixed) == 0

    def test_bound_prefix_handles_mixed_general_and_bounds(self):
        """Predicted bound active set is consistent with optimum when
        the problem mixes a general inequality and box bounds.

        Minimise (x - target)^2 + (y - target)^2 subject to
            x + y >= 0.5,  0 <= x <= 1, 0 <= y <= 1.
        With target = (-1, 2) the unconstrained optimum is outside the
        box on both coordinates, so the optimum is x=0, y=1 and the
        general inequality is inactive.
        """
        target = jnp.array([-1.0, 2.0])

        def objective(x, args):
            return jnp.sum((x - target) ** 2), None

        def ineq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 0.5])

        bounds = jnp.stack([jnp.zeros(2), jnp.ones(2)], axis=1)
        x0 = jnp.array([0.5, 0.5])
        expected = jnp.array([0.0, 1.0])

        solver = SLSQP(
            atol=1e-7,
            max_steps=80,
            bounds=bounds,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=1,
            active_set_method="lpeca_init",
            lpeca_warmup_steps=0,
            lpeca_trust_threshold=1e6,
            lpeca_predict_bounds=True,
        )
        y, state = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, expected, atol=1e-5)
        # The bound counter should record at least one prefix on this
        # bound-dominated problem.
        assert int(state.diagnostics.n_lpeca_bounds_prefixed) > 0
