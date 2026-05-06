"""Unit tests for the QP subproblem solver.

These tests verify that the HVP-based Active Set QP solver correctly solves
quadratic programming problems with equality and inequality constraints
using projected conjugate gradient.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax.qp import solve_qp

# Enable 64-bit precision for numerical accuracy
jax.config.update("jax_enable_x64", True)


def _make_hvp(H):
    """Create an HVP closure from a dense Hessian matrix (for testing)."""

    def hvp_fn(v):
        return H @ v

    return hvp_fn


class TestEqualityQP:
    """Tests for equality-constrained QP via solve_qp (no inequalities)."""

    def test_unconstrained_qp(self):
        """Test QP with no constraints.

        minimize (1/2) x^T H x + g^T x

        Solution: x = -H^{-1} g
        """
        H = jnp.array([[2.0, 0.0], [0.0, 4.0]])
        g = jnp.array([2.0, 4.0])

        result = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, 2)),
            jnp.zeros((0,)),
            jnp.zeros((0, 2)),
            jnp.zeros((0,)),
        )

        expected = -jnp.linalg.solve(H, g)
        np.testing.assert_allclose(result.d, expected, rtol=1e-6)

    def test_single_equality_constraint(self):
        """Test QP with single equality constraint.

        minimize (1/2)(x^2 + y^2)
        subject to x + y = 1

        Solution: x = y = 0.5
        """
        H = jnp.eye(2)
        g = jnp.zeros(2)
        A_eq = jnp.array([[1.0, 1.0]])
        b_eq = jnp.array([1.0])

        result = solve_qp(
            _make_hvp(H),
            g,
            A_eq,
            b_eq,
            jnp.zeros((0, 2)),
            jnp.zeros((0,)),
        )

        expected_d = jnp.array([0.5, 0.5])
        np.testing.assert_allclose(result.d, expected_d, rtol=1e-6)

    def test_multiple_equality_constraints(self):
        """Test QP with multiple equality constraints.

        minimize (1/2)(x^2 + y^2 + z^2)
        subject to x + y = 1
                   y + z = 1

        Solution: x = z = 1/3, y = 2/3
        """
        H = jnp.eye(3)
        g = jnp.zeros(3)
        A_eq = jnp.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
        b_eq = jnp.array([1.0, 1.0])

        result = solve_qp(
            _make_hvp(H),
            g,
            A_eq,
            b_eq,
            jnp.zeros((0, 3)),
            jnp.zeros((0,)),
        )

        expected_d = jnp.array([1 / 3, 2 / 3, 1 / 3])
        np.testing.assert_allclose(result.d, expected_d, rtol=1e-5)

        # Verify constraints are satisfied
        np.testing.assert_allclose(A_eq @ result.d, b_eq, rtol=1e-5)


class TestInequalityQP:
    """Tests for QP with inequality constraints."""

    def test_inactive_inequality(self):
        """Test QP where inequality is not active at solution.

        minimize (1/2)((x-1)^2 + (y-1)^2)
        subject to x + y >= 0

        Unconstrained minimum at (1, 1) satisfies x + y = 2 >= 0.
        """
        H = jnp.eye(2)
        g = jnp.array([-1.0, -1.0])
        A_eq = jnp.zeros((0, 2))
        b_eq = jnp.zeros((0,))
        A_ineq = jnp.array([[1.0, 1.0]])
        b_ineq = jnp.array([0.0])

        result = solve_qp(_make_hvp(H), g, A_eq, b_eq, A_ineq, b_ineq)

        expected_d = jnp.array([1.0, 1.0])
        np.testing.assert_allclose(result.d, expected_d, rtol=1e-5)
        assert result.converged

    def test_active_inequality(self):
        """Test QP where inequality is active at solution.

        minimize (1/2)(x^2 + y^2)
        subject to x + y >= 2

        Constrained minimum: x = y = 1.
        """
        H = jnp.eye(2)
        g = jnp.zeros(2)
        A_eq = jnp.zeros((0, 2))
        b_eq = jnp.zeros((0,))
        A_ineq = jnp.array([[1.0, 1.0]])
        b_ineq = jnp.array([2.0])

        result = solve_qp(_make_hvp(H), g, A_eq, b_eq, A_ineq, b_ineq)

        expected_d = jnp.array([1.0, 1.0])
        np.testing.assert_allclose(result.d, expected_d, rtol=1e-5)

        constraint_val = A_ineq @ result.d
        np.testing.assert_allclose(constraint_val, b_ineq, rtol=1e-5)
        assert result.converged

    def test_multiple_inequalities_one_active(self):
        """Test QP with multiple inequalities, one active.

        minimize (1/2)(x^2 + y^2)
        subject to x >= 1
                   y >= 0

        Solution: x = 1, y = 0
        """
        H = jnp.eye(2)
        g = jnp.zeros(2)
        A_eq = jnp.zeros((0, 2))
        b_eq = jnp.zeros((0,))
        A_ineq = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        b_ineq = jnp.array([1.0, 0.0])

        result = solve_qp(_make_hvp(H), g, A_eq, b_eq, A_ineq, b_ineq)

        expected_d = jnp.array([1.0, 0.0])
        np.testing.assert_allclose(result.d, expected_d, rtol=1e-5)
        assert result.converged

    def test_box_constraints(self):
        """Test QP with box constraints.

        minimize (1/2)((x-3)^2 + (y-3)^2)
        subject to 0 <= x <= 2
                   0 <= y <= 2

        Constrained minimum: (2, 2).
        """
        H = 2.0 * jnp.eye(2)
        g = jnp.array([-6.0, -6.0])
        A_eq = jnp.zeros((0, 2))
        b_eq = jnp.zeros((0,))
        A_ineq = jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ]
        )
        b_ineq = jnp.array([0.0, 0.0, -2.0, -2.0])

        result = solve_qp(_make_hvp(H), g, A_eq, b_eq, A_ineq, b_ineq)

        expected_d = jnp.array([2.0, 2.0])
        np.testing.assert_allclose(result.d, expected_d, rtol=1e-4)
        assert result.converged


class TestMixedConstraintsQP:
    """Tests for QP with both equality and inequality constraints."""

    def test_equality_and_inequality(self):
        """Test QP with equality and inequality constraints.

        minimize (1/2)(x^2 + y^2 + z^2)
        subject to x + y + z = 3
                   x >= 0, y >= 0, z >= 0

        Solution: x = y = z = 1
        """
        H = jnp.eye(3)
        g = jnp.zeros(3)
        A_eq = jnp.array([[1.0, 1.0, 1.0]])
        b_eq = jnp.array([3.0])
        A_ineq = jnp.eye(3)
        b_ineq = jnp.zeros(3)

        result = solve_qp(_make_hvp(H), g, A_eq, b_eq, A_ineq, b_ineq)

        expected_d = jnp.array([1.0, 1.0, 1.0])
        np.testing.assert_allclose(result.d, expected_d, rtol=1e-4)

        np.testing.assert_allclose(A_eq @ result.d, b_eq, rtol=1e-5)
        assert jnp.all(A_ineq @ result.d >= b_ineq - 1e-5)
        assert result.converged

    def test_equality_with_active_inequality(self):
        """Test where some inequalities become active.

        minimize (1/2)((x-2)^2 + (y-2)^2)
        subject to x + y = 2
                   x >= 1
        """
        H = jnp.eye(2)
        g = jnp.array([-2.0, -2.0])
        A_eq = jnp.array([[1.0, 1.0]])
        b_eq = jnp.array([2.0])
        A_ineq = jnp.array([[1.0, 0.0]])
        b_ineq = jnp.array([1.0])

        result = solve_qp(_make_hvp(H), g, A_eq, b_eq, A_ineq, b_ineq)

        np.testing.assert_allclose(A_eq @ result.d, b_eq, rtol=1e-5)
        assert result.d[0] >= 1.0 - 1e-5
        assert result.converged


class TestQPEdgeCases:
    """Edge cases for the QP solver."""

    def test_zero_gradient(self):
        """Test QP with zero gradient (solution at origin if feasible)."""
        H = jnp.eye(2)
        g = jnp.zeros(2)

        result = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, 2)),
            jnp.zeros((0,)),
            jnp.zeros((0, 2)),
            jnp.zeros((0,)),
        )

        np.testing.assert_allclose(result.d, jnp.zeros(2), atol=1e-8)
        assert result.converged

    def test_identity_hessian(self):
        """Test QP with identity Hessian (steepest descent direction)."""
        H = jnp.eye(3)
        g = jnp.array([1.0, 2.0, 3.0])

        result = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, 3)),
            jnp.zeros((0,)),
            jnp.zeros((0, 3)),
            jnp.zeros((0,)),
        )

        expected_d = -g
        np.testing.assert_allclose(result.d, expected_d, rtol=1e-6)
        assert result.converged

    def test_scaled_hessian(self):
        """Test QP with scaled Hessian."""
        H = jnp.diag(jnp.array([1.0, 10.0, 100.0]))
        g = jnp.array([1.0, 1.0, 1.0])

        result = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, 3)),
            jnp.zeros((0,)),
            jnp.zeros((0, 3)),
            jnp.zeros((0,)),
        )

        expected_d = jnp.array([-1.0, -0.1, -0.01])
        np.testing.assert_allclose(result.d, expected_d, rtol=1e-6)
        assert result.converged


class TestQPJIT:
    """Test that QP solver works with JAX JIT compilation."""

    def test_jit_compilation(self):
        """Test that solve_qp can be JIT compiled."""
        H = jnp.eye(2)

        @jax.jit
        def solve_qp_jit(g, A_eq, b_eq, A_ineq, b_ineq):
            return solve_qp(_make_hvp(H), g, A_eq, b_eq, A_ineq, b_ineq)

        g = jnp.array([1.0, 1.0])
        A_eq = jnp.zeros((0, 2))
        b_eq = jnp.zeros((0,))
        A_ineq = jnp.array([[1.0, 0.0]])
        b_ineq = jnp.array([0.5])

        result = solve_qp_jit(g, A_eq, b_eq, A_ineq, b_ineq)

        assert result.d[0] >= 0.5 - 1e-5
        assert result.converged


class TestQPExpandAntiCycling:
    """Tests for the EXPAND anti-cycling mechanism in the QP active-set solver."""

    def test_degenerate_vertex_converges(self):
        """Degenerate QP where all constraints are active at the solution.

        minimize 0.5 * ||d||^2 + g^T d   with g = [1, 1, ..., 1]
        subject to d_i >= 0 for all i
                   sum(d) >= 0  (redundant)
                   d_1 + d_2 >= 0 (redundant)

        The unconstrained minimum violates all constraints.  The solution
        d = 0 has all constraints active simultaneously -- a maximally
        degenerate vertex.  Without anti-cycling guards, the redundant
        constraints can cause the active-set loop to oscillate.
        """
        n = 4
        H = jnp.eye(n)
        g = jnp.ones(n)

        A_ineq = jnp.concatenate(
            [
                jnp.eye(n),
                jnp.ones((1, n)),
                jnp.array([[1.0, 1.0, 0.0, 0.0]]),
            ],
            axis=0,
        )
        b_ineq = jnp.zeros(n + 2)

        result = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, n)),
            jnp.zeros((0,)),
            A_ineq,
            b_ineq,
            max_iter=200,
        )

        assert result.converged
        np.testing.assert_allclose(result.d, jnp.zeros(n), atol=1e-6)

    def test_nearly_parallel_constraints(self):
        """QP with nearly parallel inequality constraints.

        Two constraints that differ only by a tiny epsilon create a
        near-degenerate configuration.  The active-set solver must
        resolve which constraint to keep active without cycling between
        them.

        minimize 0.5 * (x^2 + y^2)
        subject to  x + y >= 1
                    x + y >= 1 - 1e-10   (nearly identical)
                    x >= 0
                    y >= 0
        Solution: (0.5, 0.5) with only x + y >= 1 active.
        """
        H = jnp.eye(2)
        g = jnp.zeros(2)

        eps = 1e-10
        A_ineq = jnp.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        b_ineq = jnp.array([1.0, 1.0 - eps, 0.0, 0.0])

        result = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, 2)),
            jnp.zeros((0,)),
            A_ineq,
            b_ineq,
            max_iter=200,
        )

        assert result.converged
        np.testing.assert_allclose(result.d, jnp.array([0.5, 0.5]), atol=1e-4)

    def test_expand_disabled_vs_enabled(self):
        """Compare convergence with and without EXPAND on a degenerate QP.

        With expand_factor=0 (disabled), the solver relies solely on the
        iteration limit.  With expand_factor=1 (default), the expanding
        tolerance should help convergence.  We verify the EXPAND version
        converges and uses no more iterations.
        """
        n = 3
        H = jnp.eye(n)
        g = jnp.ones(n)

        A_ineq = jnp.concatenate(
            [
                jnp.eye(n),
                jnp.ones((1, n)),
            ],
            axis=0,
        )
        b_ineq = jnp.zeros(n + 1)

        result_expand = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, n)),
            jnp.zeros((0,)),
            A_ineq,
            b_ineq,
            max_iter=100,
            expand_factor=1.0,
        )

        result_no_expand = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, n)),
            jnp.zeros((0,)),
            A_ineq,
            b_ineq,
            max_iter=100,
            expand_factor=0.0,
        )

        assert result_expand.converged
        np.testing.assert_allclose(result_expand.d, jnp.zeros(n), atol=1e-6)

        # Both should converge on this problem, but EXPAND should not
        # use more iterations than the non-EXPAND version
        assert result_expand.iterations <= result_no_expand.iterations + 1


class TestQPActiveSetWarmStart:
    """Tests for active-set warm-starting via initial_active_set."""

    def test_qp_result_has_active_set(self):
        """QPResult should include the final active set."""
        H = jnp.eye(2)
        g = jnp.zeros(2)
        A_ineq = jnp.array([[1.0, 1.0]])
        b_ineq = jnp.array([2.0])

        result = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, 2)),
            jnp.zeros((0,)),
            A_ineq,
            b_ineq,
        )

        assert hasattr(result, "active_set")
        assert result.active_set.shape == (1,)
        assert result.active_set.dtype == jnp.bool_

    def test_active_set_reflects_solution(self):
        """Active set should be True for constraints active at the solution.

        minimize (1/2)(x^2 + y^2)
        subject to x + y >= 2

        Solution: (1, 1) with the constraint active.
        """
        H = jnp.eye(2)
        g = jnp.zeros(2)
        A_ineq = jnp.array([[1.0, 1.0]])
        b_ineq = jnp.array([2.0])

        result = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, 2)),
            jnp.zeros((0,)),
            A_ineq,
            b_ineq,
        )

        assert result.converged
        assert result.active_set[0], "Constraint should be active at the solution"

    def test_inactive_constraint_not_in_active_set(self):
        """Inactive constraints should be False in the returned active set.

        minimize (1/2)((x-1)^2 + (y-1)^2)
        subject to x + y >= 0

        Unconstrained min at (1, 1) satisfies the constraint => inactive.
        """
        H = jnp.eye(2)
        g = jnp.array([-1.0, -1.0])
        A_ineq = jnp.array([[1.0, 1.0]])
        b_ineq = jnp.array([0.0])

        result = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, 2)),
            jnp.zeros((0,)),
            A_ineq,
            b_ineq,
        )

        assert result.converged
        assert not result.active_set[0], "Constraint should be inactive"

    def test_warm_start_reduces_iterations(self):
        """Warm-starting with the correct active set should not require
        more iterations than cold start, and ideally fewer.

        minimize (1/2)(x^2 + y^2)
        subject to x >= 1, y >= 0

        Solution: (1, 0) with x >= 1 active, y >= 0 active.
        """
        H = jnp.eye(2)
        g = jnp.zeros(2)
        A_ineq = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        b_ineq = jnp.array([1.0, 0.0])

        result_cold = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, 2)),
            jnp.zeros((0,)),
            A_ineq,
            b_ineq,
        )
        assert result_cold.converged

        result_warm = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, 2)),
            jnp.zeros((0,)),
            A_ineq,
            b_ineq,
            initial_active_set=result_cold.active_set,
        )
        assert result_warm.converged
        np.testing.assert_allclose(result_warm.d, result_cold.d, rtol=1e-5)
        assert result_warm.iterations <= result_cold.iterations

    def test_warm_start_wrong_active_set_still_converges(self):
        """Even with an incorrect warm-start, the solver should still converge.

        minimize (1/2)((x-1)^2 + (y-1)^2)
        subject to x + y >= 0, x >= -10

        Solution: (1, 1) with both constraints inactive.
        Warm-start with both constraints active (incorrect).
        """
        H = jnp.eye(2)
        g = jnp.array([-1.0, -1.0])
        A_ineq = jnp.array([[1.0, 1.0], [1.0, 0.0]])
        b_ineq = jnp.array([0.0, -10.0])

        wrong_active = jnp.array([True, True])
        result = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, 2)),
            jnp.zeros((0,)),
            A_ineq,
            b_ineq,
            initial_active_set=wrong_active,
        )

        assert result.converged
        np.testing.assert_allclose(result.d, jnp.array([1.0, 1.0]), rtol=1e-4)

    def test_no_inequality_active_set_empty(self):
        """When there are no inequality constraints, active_set should be empty."""
        H = jnp.eye(2)
        g = jnp.array([1.0, 1.0])

        result = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, 2)),
            jnp.zeros((0,)),
            jnp.zeros((0, 2)),
            jnp.zeros((0,)),
        )

        assert result.active_set.shape == (0,)


class TestQPAdaptiveTolerance:
    """Tests for KKT-residual-based adaptive EXPAND tolerance."""

    def test_kkt_residual_zero_matches_default(self):
        """With kkt_residual=0, behavior should match default."""
        H = jnp.eye(2)
        g = jnp.zeros(2)
        A_ineq = jnp.array([[1.0, 1.0]])
        b_ineq = jnp.array([2.0])

        result_default = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, 2)),
            jnp.zeros((0,)),
            A_ineq,
            b_ineq,
        )

        result_zero = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, 2)),
            jnp.zeros((0,)),
            A_ineq,
            b_ineq,
            kkt_residual=0.0,
        )

        np.testing.assert_allclose(result_default.d, result_zero.d, rtol=1e-10)
        assert result_default.iterations == result_zero.iterations

    def test_large_kkt_residual_converges(self):
        """With a large KKT residual the solver should still converge.

        minimize (1/2)(x^2 + y^2) subject to x + y >= 2
        """
        H = jnp.eye(2)
        g = jnp.zeros(2)
        A_ineq = jnp.array([[1.0, 1.0]])
        b_ineq = jnp.array([2.0])

        result = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, 2)),
            jnp.zeros((0,)),
            A_ineq,
            b_ineq,
            kkt_residual=100.0,
        )

        assert result.converged
        np.testing.assert_allclose(result.d, jnp.array([1.0, 1.0]), rtol=1e-4)

    def test_kkt_residual_capped_at_one(self):
        """kkt_residual > 1 should produce the same tolerance as kkt_residual=1."""
        H = jnp.eye(2)
        g = jnp.zeros(2)
        A_ineq = jnp.array([[1.0, 1.0]])
        b_ineq = jnp.array([2.0])

        result_one = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, 2)),
            jnp.zeros((0,)),
            A_ineq,
            b_ineq,
            kkt_residual=1.0,
        )

        result_large = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, 2)),
            jnp.zeros((0,)),
            A_ineq,
            b_ineq,
            kkt_residual=1000.0,
        )

        np.testing.assert_allclose(result_one.d, result_large.d, rtol=1e-10)
        assert result_one.iterations == result_large.iterations


class TestProximalStabilization:
    """Tests for proximal multiplier stabilization (sSQP) in solve_qp."""

    def test_degenerate_qp_converges_with_proximal(self):
        """Proximal should converge on a QP that cycles without it.

        Constructs a degenerate QP where 4 inequality constraints meet
        at the same vertex with near-linearly-dependent normals. The
        standard active-set method cycles (hits max_iter), but sSQP
        absorbs the equality constraint and converges.
        """
        n = 3
        H = jnp.eye(n)
        g = -jnp.array([1.0, 1.0, 1.0])
        A_eq = jnp.array([[1.0, 1.0, 1.0]])
        b_eq = jnp.array([1.0])

        # Near-degenerate inequalities: all nearly parallel normals
        # meeting close to the same vertex
        A_ineq = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
        b_ineq = jnp.array([0.0, 0.0, 0.0, 0.0])

        result_proximal = solve_qp(
            _make_hvp(H),
            g,
            A_eq,
            b_eq,
            A_ineq,
            b_ineq,
            max_iter=100,
            proximal_mu=0.01,
            prev_multipliers_eq=jnp.zeros(1),
        )
        assert result_proximal.converged

    def test_solution_quality_with_different_mu(self):
        """Proximal solution should converge with different mu values.

        With similar mu values the solutions should be close; very
        different mu values lead to different regularization strength
        and can produce different solutions on constrained problems.
        """
        H = 2.0 * jnp.eye(2)
        g = jnp.array([-2.0, -5.0])
        A_eq = jnp.array([[1.0, 1.0]])
        b_eq = jnp.array([1.0])
        A_ineq = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        b_ineq = jnp.array([0.0, 0.0])

        result_mu1 = solve_qp(
            _make_hvp(H),
            g,
            A_eq,
            b_eq,
            A_ineq,
            b_ineq,
            proximal_mu=0.01,
            prev_multipliers_eq=jnp.zeros(1),
        )
        result_mu2 = solve_qp(
            _make_hvp(H),
            g,
            A_eq,
            b_eq,
            A_ineq,
            b_ineq,
            proximal_mu=0.02,
            prev_multipliers_eq=jnp.zeros(1),
        )

        assert result_mu1.converged
        assert result_mu2.converged
        np.testing.assert_allclose(
            result_mu1.d,
            result_mu2.d,
            atol=0.1,
        )

    def test_multiplier_recovery_formula(self):
        """Equality multipliers should satisfy
        lambda = lambda_k - (1/mu)(A_eq d - b_eq)."""
        H = jnp.eye(2)
        g = jnp.array([-1.0, -2.0])
        A_eq = jnp.array([[1.0, 1.0]])
        b_eq = jnp.array([1.0])
        A_ineq = jnp.zeros((0, 2))
        b_ineq = jnp.zeros(0)

        mu = 0.01
        prev_mult = jnp.array([0.5])

        result = solve_qp(
            _make_hvp(H),
            g,
            A_eq,
            b_eq,
            A_ineq,
            b_ineq,
            proximal_mu=mu,
            prev_multipliers_eq=prev_mult,
        )

        expected_mult = prev_mult - (1.0 / mu) * (A_eq @ result.d - b_eq)
        np.testing.assert_allclose(
            result.multipliers_eq,
            expected_mult,
            rtol=1e-6,
        )

    def test_smaller_mu_enforces_equality_more_tightly(self):
        """Smaller mu (stronger penalty 1/mu) enforces the equality
        constraint more tightly than larger mu."""
        H = jnp.eye(2)
        g = jnp.array([-1.0, -2.0])
        A_eq = jnp.array([[1.0, 1.0]])
        b_eq = jnp.array([1.0])
        A_ineq = jnp.zeros((0, 2))
        b_ineq = jnp.zeros(0)

        result_small_mu = solve_qp(
            _make_hvp(H),
            g,
            A_eq,
            b_eq,
            A_ineq,
            b_ineq,
            proximal_mu=0.001,
            prev_multipliers_eq=jnp.zeros(1),
        )
        result_large_mu = solve_qp(
            _make_hvp(H),
            g,
            A_eq,
            b_eq,
            A_ineq,
            b_ineq,
            proximal_mu=1.0,
            prev_multipliers_eq=jnp.zeros(1),
        )

        assert result_small_mu.converged
        assert result_large_mu.converged

        eq_resid_small = float(jnp.linalg.norm(A_eq @ result_small_mu.d - b_eq))
        eq_resid_large = float(jnp.linalg.norm(A_eq @ result_large_mu.d - b_eq))
        assert eq_resid_small < eq_resid_large

    def test_no_equalities_proximal_passthrough(self):
        """With no equalities, proximal_mu is irrelevant — uses ineq-only path."""
        H = jnp.eye(2)
        g = jnp.array([-1.0, -2.0])
        A_eq = jnp.zeros((0, 2))
        b_eq = jnp.zeros(0)
        A_ineq = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        b_ineq = jnp.array([0.0, 0.0])

        result_standard = solve_qp(
            _make_hvp(H),
            g,
            A_eq,
            b_eq,
            A_ineq,
            b_ineq,
        )
        result_with_mu = solve_qp(
            _make_hvp(H),
            g,
            A_eq,
            b_eq,
            A_ineq,
            b_ineq,
            proximal_mu=0.01,
        )

        np.testing.assert_allclose(
            result_with_mu.d,
            result_standard.d,
            rtol=1e-10,
        )


class TestPreconditionedCG:
    """Tests for preconditioned conjugate gradient in the QP solver."""

    def test_pcg_matches_unpreconditioned_well_conditioned(self):
        """On a well-conditioned Hessian, PCG and CG produce the same result."""
        H = jnp.array([[3.0, 1.0], [1.0, 4.0]])
        g = jnp.array([1.0, -2.0])
        A_eq = jnp.zeros((0, 2))
        b_eq = jnp.zeros(0)
        A_ineq = jnp.zeros((0, 2))
        b_ineq = jnp.zeros(0)

        H_inv = jnp.linalg.inv(H)

        result_no_precond = solve_qp(_make_hvp(H), g, A_eq, b_eq, A_ineq, b_ineq)
        result_precond = solve_qp(
            _make_hvp(H),
            g,
            A_eq,
            b_eq,
            A_ineq,
            b_ineq,
            precond_fn=lambda v: H_inv @ v,
        )

        np.testing.assert_allclose(result_precond.d, result_no_precond.d, atol=1e-10)

    def test_pcg_converges_ill_conditioned(self):
        """PCG converges on an ill-conditioned H where unpreconditioned CG struggles."""
        n = 20
        rng = np.random.RandomState(42)
        Q, _ = np.linalg.qr(rng.randn(n, n))
        eigenvalues = np.logspace(-3, 2, n)
        H = jnp.array(Q @ np.diag(eigenvalues) @ Q.T)
        g = jnp.array(rng.randn(n))
        A_eq = jnp.zeros((0, n))
        b_eq = jnp.zeros(0)
        A_ineq = jnp.zeros((0, n))
        b_ineq = jnp.zeros(0)

        expected = -jnp.linalg.solve(H, g)

        H_inv = jnp.linalg.inv(H)
        result_precond = solve_qp(
            _make_hvp(H),
            g,
            A_eq,
            b_eq,
            A_ineq,
            b_ineq,
            max_cg_iter=5,
            precond_fn=lambda v: H_inv @ v,
        )
        np.testing.assert_allclose(result_precond.d, expected, atol=1e-6)

        result_no_precond = solve_qp(
            _make_hvp(H),
            g,
            A_eq,
            b_eq,
            A_ineq,
            b_ineq,
            max_cg_iter=5,
        )
        error_no_precond = float(jnp.linalg.norm(result_no_precond.d - expected))
        error_precond = float(jnp.linalg.norm(result_precond.d - expected))
        assert error_precond < error_no_precond * 0.01

    def test_pcg_with_proximal_stabilization(self):
        """PCG works with proximal stabilization (sSQP)."""
        H = jnp.eye(3) * 0.001
        g = jnp.array([1.0, -1.0, 0.5])
        A_eq = jnp.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
        b_eq = jnp.array([1.0, 0.5])
        A_ineq = jnp.zeros((0, 3))
        b_ineq = jnp.zeros(0)

        H_inv = jnp.linalg.inv(H)

        result = solve_qp(
            _make_hvp(H),
            g,
            A_eq,
            b_eq,
            A_ineq,
            b_ineq,
            proximal_mu=0.01,
            precond_fn=lambda v: H_inv @ v,
        )
        residual = A_eq @ result.d - b_eq
        np.testing.assert_allclose(residual, 0.0, atol=0.1)

    def test_pcg_projected_equality_constrained(self):
        """PCG with equality constraints in projected null space."""
        H = jnp.array([[10.0, 0.0], [0.0, 0.1]])
        g = jnp.array([1.0, 1.0])
        A_eq = jnp.array([[1.0, 1.0]])
        b_eq = jnp.array([0.5])
        A_ineq = jnp.zeros((0, 2))
        b_ineq = jnp.zeros(0)

        H_inv = jnp.linalg.inv(H)

        result = solve_qp(
            _make_hvp(H),
            g,
            A_eq,
            b_eq,
            A_ineq,
            b_ineq,
            precond_fn=lambda v: H_inv @ v,
        )
        np.testing.assert_allclose(A_eq @ result.d, b_eq, atol=1e-6)

    def test_pcg_identity_preconditioner(self):
        """Identity preconditioner behaves like no preconditioning."""
        H = jnp.array([[2.0, 0.5], [0.5, 3.0]])
        g = jnp.array([1.0, -1.0])
        A_eq = jnp.zeros((0, 2))
        b_eq = jnp.zeros(0)
        A_ineq = jnp.zeros((0, 2))
        b_ineq = jnp.zeros(0)

        result_none = solve_qp(_make_hvp(H), g, A_eq, b_eq, A_ineq, b_ineq)
        result_identity = solve_qp(
            _make_hvp(H),
            g,
            A_eq,
            b_eq,
            A_ineq,
            b_ineq,
            precond_fn=lambda v: v,
        )
        np.testing.assert_allclose(result_identity.d, result_none.d, atol=1e-10)

    def test_pcg_with_active_set(self):
        """Full QP with inequality constraints and preconditioner."""
        n = 10
        rng = np.random.RandomState(123)
        Q, _ = np.linalg.qr(rng.randn(n, n))
        eigenvalues = np.logspace(-2, 2, n)
        H = jnp.array(Q @ np.diag(eigenvalues) @ Q.T)
        g = jnp.array(rng.randn(n))
        A_eq = jnp.zeros((0, n))
        b_eq = jnp.zeros(0)
        A_ineq = jnp.eye(n)
        b_ineq = jnp.zeros(n)

        H_inv = jnp.linalg.inv(H)

        result = solve_qp(
            _make_hvp(H),
            g,
            A_eq,
            b_eq,
            A_ineq,
            b_ineq,
            precond_fn=lambda v: H_inv @ v,
        )
        np.testing.assert_array_less(-1e-6, A_ineq @ result.d - b_ineq)


class TestCGRegularization:
    """Tests for the SNOPT-style CG regularization parameter.

    The regularization replaces the hard absolute curvature threshold
    ``pBp <= 1e-8`` with a relative one ``pBp <= delta^2 * ||p||^2``,
    where ``delta^2 = cg_regularization``.  This is scale-invariant:
    CG stops when the effective eigenvalue along p falls below delta^2,
    regardless of how small ||p|| is.
    """

    def test_regularization_allows_progress_on_small_eigenvalue(self):
        """With eigenvalue 1e-5 (above delta^2 = 1e-6), CG should continue
        and find the correct solution.  Under the old absolute threshold
        (1e-8), this would have been stopped when ||p|| < ~0.003."""
        n = 3
        H = jnp.diag(jnp.array([1e-5, 1.0, 1.0]))
        g = jnp.array([1.0, 1.0, 1.0])

        A_eq = jnp.zeros((0, n))
        b_eq = jnp.zeros(0)
        A_ineq = jnp.zeros((0, n))
        b_ineq = jnp.zeros(0)

        result = solve_qp(
            _make_hvp(H),
            g,
            A_eq,
            b_eq,
            A_ineq,
            b_ineq,
            cg_regularization=1e-6,
        )
        expected = -jnp.linalg.solve(H, g)
        np.testing.assert_allclose(result.d, expected, rtol=1e-4)

    def test_regularization_negligible_on_well_conditioned(self):
        """On a well-conditioned Hessian, the regularization has no effect
        on the solution (CG converges normally without hitting the guard)."""
        n = 5
        H = jnp.diag(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        g = jnp.array([1.0, -2.0, 3.0, -4.0, 5.0])

        A_eq = jnp.zeros((0, n))
        b_eq = jnp.zeros(0)
        A_ineq = jnp.zeros((0, n))
        b_ineq = jnp.zeros(0)

        result_reg = solve_qp(
            _make_hvp(H),
            g,
            A_eq,
            b_eq,
            A_ineq,
            b_ineq,
            cg_regularization=1e-6,
        )
        result_no_reg = solve_qp(
            _make_hvp(H),
            g,
            A_eq,
            b_eq,
            A_ineq,
            b_ineq,
            cg_regularization=0.0,
        )

        expected = -jnp.linalg.solve(H, g)
        np.testing.assert_allclose(result_reg.d, expected, atol=1e-6)
        np.testing.assert_allclose(result_no_reg.d, expected, atol=1e-6)
        np.testing.assert_allclose(result_reg.d, result_no_reg.d, atol=1e-8)

    def test_negative_curvature_stops_cg(self):
        """Truly negative curvature still triggers early stopping."""
        n = 3
        H = jnp.diag(jnp.array([-1.0, 1.0, 1.0]))
        g = jnp.array([1.0, 0.0, 0.0])

        A_eq = jnp.zeros((0, n))
        b_eq = jnp.zeros(0)
        A_ineq = jnp.zeros((0, n))
        b_ineq = jnp.zeros(0)

        result = solve_qp(
            _make_hvp(H),
            g,
            A_eq,
            b_eq,
            A_ineq,
            b_ineq,
            cg_regularization=1e-6,
        )
        assert jnp.linalg.norm(result.d) < 1e-10

    def test_regularization_with_projected_cg(self):
        """CG regularization works correctly with projected CG."""
        n = 5
        H = jnp.diag(jnp.array([0.1, 1.0, 2.0, 5.0, 10.0]))
        g = jnp.array([1.0, -2.0, 0.5, 3.0, -1.0])

        A_eq = jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
        b_eq = jnp.array([0.0])
        A_ineq = jnp.zeros((0, n))
        b_ineq = jnp.zeros(0)

        result = solve_qp(
            _make_hvp(H),
            g,
            A_eq,
            b_eq,
            A_ineq,
            b_ineq,
            cg_regularization=1e-6,
        )
        np.testing.assert_allclose(A_eq @ result.d, b_eq, atol=1e-6)
        assert jnp.linalg.norm(result.d) > 1e-8

    def test_regularization_with_inequality_constraints(self):
        """CG regularization does not degrade QP solutions with box
        constraints (regression guard for the null-space projection
        leakage issue)."""
        H = 2.0 * jnp.eye(2)
        g = jnp.array([-6.0, -6.0])
        A_eq = jnp.zeros((0, 2))
        b_eq = jnp.zeros((0,))
        A_ineq = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        b_ineq = jnp.array([0.0, 0.0, -2.0, -2.0])

        result = solve_qp(
            _make_hvp(H),
            g,
            A_eq,
            b_eq,
            A_ineq,
            b_ineq,
            cg_regularization=1e-6,
        )
        np.testing.assert_allclose(result.d, jnp.array([2.0, 2.0]), rtol=1e-4)


class TestQPCyclingDiagnostics:
    """Regression tests for Bundle 1 (EXPAND, ping-pong, mult-noise)."""

    def test_qpresult_exposes_new_diagnostic_fields(self):
        """``QPResult`` carries the Bundle-1 diagnostic fields."""
        H = jnp.eye(2)
        g = jnp.zeros(2)
        result = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, 2)),
            jnp.zeros(0),
            jnp.eye(2),
            jnp.array([1.0, 0.0]),
        )
        assert hasattr(result, "ping_ponged")
        assert hasattr(result, "reached_max_iter")
        assert hasattr(result, "final_working_tol")
        assert bool(result.converged)
        assert not bool(result.ping_ponged), (
            "Clean QP should not flag ping-pong on a non-degenerate problem"
        )
        assert not bool(result.reached_max_iter), (
            "Clean QP should not exhaust the iteration budget"
        )

    def test_degenerate_constraints_break_within_half_budget(self):
        """SNOPT-style EXPAND breaks degenerate ties well within the budget.

        Three inequality constraints that are all active (with linearly
        dependent gradients) at the optimum (0, 0).  Without the
        recalibrated EXPAND ramp, ``working_tol`` barely moves and the
        active-set loop can churn for the full budget.  With the new
        ramp, the loop should converge inside ``max_iter // 2``.
        """
        H = jnp.eye(2)
        g = jnp.zeros(2)
        # Constraints x >= 0, y >= 0, x + y >= 0 (third is degenerate).
        A_ineq = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        b_ineq = jnp.zeros(3)
        max_iter = 20
        result = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, 2)),
            jnp.zeros(0),
            A_ineq,
            b_ineq,
            max_iter=max_iter,
        )
        np.testing.assert_allclose(result.d, [0.0, 0.0], atol=1e-6)
        assert bool(result.converged)
        assert int(result.iterations) <= max_iter // 2, (
            f"Degenerate QP took {int(result.iterations)} iterations, "
            f"expected <= {max_iter // 2}"
        )
        assert not bool(result.reached_max_iter)

    def test_mult_drop_floor_prevents_noise_oscillation(self):
        """A loose ``mult_drop_floor`` keeps noisy multipliers from
        cycling on a poorly-conditioned ``AAᵀ``.

        We construct an inequality system where two constraint gradients
        are nearly parallel (cond(AAᵀ) ~ 1e10), which inflates the
        Cholesky-projection multiplier-recovery error.  Both default
        and very-loose floors should converge; the very-tight floor
        is a worst-case proxy for the pre-fix behaviour.
        """
        # Constraints with nearly parallel rows: c0 = x0,
        # c1 = x0 + 1e-5 * x1.  The optimum 0 is on both.
        H = jnp.eye(2)
        g = jnp.array([-1.0, 0.0])
        A_ineq = jnp.array([[1.0, 0.0], [1.0, 1e-5]])
        b_ineq = jnp.zeros(2)
        max_iter = 30

        result_default = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, 2)),
            jnp.zeros(0),
            A_ineq,
            b_ineq,
            max_iter=max_iter,
        )
        assert bool(result_default.converged), (
            "Default mult_drop_floor should converge on noisy projection"
        )
        # With a generous floor, the loop must terminate without
        # exhausting the iteration budget.
        result_loose = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, 2)),
            jnp.zeros(0),
            A_ineq,
            b_ineq,
            max_iter=max_iter,
            mult_drop_floor=1e-2,
        )
        assert bool(result_loose.converged)
        assert not bool(result_loose.reached_max_iter)

    def test_ping_pong_threshold_short_circuits_loop(self):
        """A small ``ping_pong_threshold`` short-circuits oscillation.

        We force the oscillation potential by giving an LP-like QP
        (degenerate vertex) and asking the solver to use a very tight
        feasibility tolerance with EXPAND disabled (``expand_factor=0``)
        plus a very tight ``mult_drop_floor`` so noisy multipliers can
        cycle.  The ping-pong detector should either converge on the
        cycle (``ping_ponged=True``) or converge cleanly; in either
        case ``reached_max_iter`` must remain False with ample budget.
        """
        H = jnp.eye(3)
        g = jnp.array([1e-3, 1e-3, 1e-3])
        # Three competing constraints meeting at the origin.
        A_ineq = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1e-8],
            ]
        )
        b_ineq = jnp.zeros(3)
        result = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, 3)),
            jnp.zeros(0),
            A_ineq,
            b_ineq,
            max_iter=50,
            mult_drop_floor=0.0,  # disable noise floor
            ping_pong_threshold=2,
        )
        # Convergence must be reported; either clean or ping-pong.
        assert bool(result.converged)
        assert not bool(result.reached_max_iter), (
            "ping-pong detector should fire before budget exhaustion"
        )

    def test_expand_ramp_default_increases_working_tol(self):
        """The v0.9.2 EXPAND ramp grows ``working_tol`` monotonically.

        ``working_tol = base_tol + k * tau`` with ``tau =
        base_tol * expand_factor / max_iter``.  With the default
        ``expand_factor=1.0`` the final tolerance sits in
        ``[base_tol, 2 * base_tol]``.
        """
        n = 4
        H = jnp.diag(jnp.array([1.0, 2.0, 3.0, 4.0]))
        g = jnp.array([-1.0, -2.0, 1.0, 2.0])
        A_ineq = jnp.eye(n)
        b_ineq = jnp.zeros(n)
        tol = 1e-8
        result = solve_qp(
            _make_hvp(H),
            g,
            jnp.zeros((0, n)),
            jnp.zeros(0),
            A_ineq,
            b_ineq,
            tol=tol,
            max_iter=20,
        )
        assert bool(result.converged)
        wt = float(result.final_working_tol)
        # ``base_tol = tol + min(kkt_res, 1.0) * tol`` so it sits in
        # ``[tol, 2 * tol]`` even at iter=0; the ramp can then push
        # ``working_tol`` up to ``2 * base_tol``.
        assert wt >= tol * 0.999, (
            f"final_working_tol {wt:e} below base_tol lower bound {tol:e}"
        )
        assert wt <= 4.0 * tol * 1.001, (
            f"final_working_tol {wt:e} above 4*tol {4.0 * tol:e} (ramp overshoot)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
