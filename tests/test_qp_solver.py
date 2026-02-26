"""Unit tests for the QP subproblem solver.

These tests verify that the HVP-based Active Set QP solver correctly solves
quadratic programming problems with equality and inequality constraints
using projected conjugate gradient.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax.qp_solver import solve_qp

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
