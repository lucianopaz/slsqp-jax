"""Unit tests for the QP subproblem solver.

These tests verify that the Active Set QP solver correctly solves
quadratic programming problems with equality and inequality constraints.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax.qp_solver import solve_equality_qp, solve_qp

# Enable 64-bit precision for numerical accuracy
jax.config.update("jax_enable_x64", True)


class TestEqualityQP:
    """Tests for equality-constrained QP solver."""

    def test_unconstrained_qp(self):
        """Test QP with no constraints.

        minimize (1/2) x^T H x + g^T x

        Solution: x = -H^{-1} g
        """
        H = jnp.array([[2.0, 0.0], [0.0, 4.0]])
        g = jnp.array([2.0, 4.0])
        A = jnp.zeros((0, 2))
        b = jnp.zeros((0,))

        d, mult = solve_equality_qp(H, g, A, b)

        expected = -jnp.linalg.solve(H, g)
        np.testing.assert_allclose(d, expected, rtol=1e-10)

    def test_single_equality_constraint(self):
        """Test QP with single equality constraint.

        minimize (1/2)(x^2 + y^2)
        subject to x + y = 1

        Solution: x = y = 0.5
        """
        H = jnp.eye(2)
        g = jnp.zeros(2)
        A = jnp.array([[1.0, 1.0]])
        b = jnp.array([1.0])

        d, mult = solve_equality_qp(H, g, A, b)

        # At origin, we want d such that d[0] + d[1] = 1
        # and minimizes ||d||^2. Solution: d = [0.5, 0.5]
        expected_d = jnp.array([0.5, 0.5])
        np.testing.assert_allclose(d, expected_d, rtol=1e-10)

    def test_multiple_equality_constraints(self):
        """Test QP with multiple equality constraints.

        minimize (1/2)(x^2 + y^2 + z^2)
        subject to x + y = 1
                   y + z = 1

        Solution: x = z, and x + y = y + z = 1
        => x = z, and 2y + x = 2 - x => 2y = 2 - 2x
        => y = 1 - x. Also x + y = 1 => x + 1 - x = 1 ✓

        Minimize x^2 + (1-x)^2 + x^2 = 3x^2 - 2x + 1
        d/dx = 6x - 2 = 0 => x = 1/3
        So x = z = 1/3, y = 2/3
        """
        H = jnp.eye(3)
        g = jnp.zeros(3)
        A = jnp.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
        b = jnp.array([1.0, 1.0])

        d, mult = solve_equality_qp(H, g, A, b)

        expected_d = jnp.array([1 / 3, 2 / 3, 1 / 3])
        np.testing.assert_allclose(d, expected_d, rtol=1e-10)

        # Verify constraints are satisfied
        np.testing.assert_allclose(A @ d, b, rtol=1e-10)


class TestInequalityQP:
    """Tests for QP with inequality constraints."""

    def test_inactive_inequality(self):
        """Test QP where inequality is not active at solution.

        minimize (1/2)((x-1)^2 + (y-1)^2)
        subject to x + y >= 0

        Unconstrained minimum at (1, 1) satisfies x + y = 2 >= 0.
        """
        # Expand: (1/2)(x^2 - 2x + 1 + y^2 - 2y + 1)
        # = (1/2)(x^2 + y^2) - x - y + 1
        # = (1/2) d^T I d + [-1, -1]^T d (ignoring constant)
        H = jnp.eye(2)
        g = jnp.array([-1.0, -1.0])
        A_eq = jnp.zeros((0, 2))
        b_eq = jnp.zeros((0,))
        A_ineq = jnp.array([[1.0, 1.0]])
        b_ineq = jnp.array([0.0])  # x + y >= 0

        result = solve_qp(H, g, A_eq, b_eq, A_ineq, b_ineq)

        # Unconstrained solution: d = -H^{-1} g = [1, 1]
        expected_d = jnp.array([1.0, 1.0])
        np.testing.assert_allclose(result.d, expected_d, rtol=1e-6)
        assert result.converged

    def test_active_inequality(self):
        """Test QP where inequality is active at solution.

        minimize (1/2)(x^2 + y^2)
        subject to x + y >= 2

        Unconstrained minimum (0, 0) violates constraint.
        Constrained minimum: on the line x + y = 2, minimize distance to origin.
        Solution: x = y = 1.
        """
        H = jnp.eye(2)
        g = jnp.zeros(2)
        A_eq = jnp.zeros((0, 2))
        b_eq = jnp.zeros((0,))
        A_ineq = jnp.array([[1.0, 1.0]])
        b_ineq = jnp.array([2.0])  # x + y >= 2

        result = solve_qp(H, g, A_eq, b_eq, A_ineq, b_ineq)

        expected_d = jnp.array([1.0, 1.0])
        np.testing.assert_allclose(result.d, expected_d, rtol=1e-6)

        # Constraint should be active (tight)
        constraint_val = A_ineq @ result.d
        np.testing.assert_allclose(constraint_val, b_ineq, rtol=1e-6)
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
        b_ineq = jnp.array([1.0, 0.0])  # x >= 1, y >= 0

        result = solve_qp(H, g, A_eq, b_eq, A_ineq, b_ineq)

        expected_d = jnp.array([1.0, 0.0])
        np.testing.assert_allclose(result.d, expected_d, rtol=1e-6)
        assert result.converged

    def test_box_constraints(self):
        """Test QP with box constraints.

        minimize (1/2)((x-3)^2 + (y-3)^2)
        subject to 0 <= x <= 2
                   0 <= y <= 2

        Unconstrained minimum at (3, 3) violates both upper bounds.
        Constrained minimum: (2, 2).
        """
        # (x-3)^2 + (y-3)^2 = x^2 - 6x + 9 + y^2 - 6y + 9
        # = x^2 + y^2 - 6x - 6y + 18
        # Gradient: [2x - 6, 2y - 6] at origin = [-6, -6]
        # Hessian: 2*I
        H = 2.0 * jnp.eye(2)
        g = jnp.array([-6.0, -6.0])
        A_eq = jnp.zeros((0, 2))
        b_eq = jnp.zeros((0,))
        # Box constraints as inequalities:
        # x >= 0, y >= 0, -x >= -2, -y >= -2
        A_ineq = jnp.array(
            [
                [1.0, 0.0],  # x >= 0
                [0.0, 1.0],  # y >= 0
                [-1.0, 0.0],  # -x >= -2 i.e. x <= 2
                [0.0, -1.0],  # -y >= -2 i.e. y <= 2
            ]
        )
        b_ineq = jnp.array([0.0, 0.0, -2.0, -2.0])

        result = solve_qp(H, g, A_eq, b_eq, A_ineq, b_ineq)

        expected_d = jnp.array([2.0, 2.0])
        np.testing.assert_allclose(result.d, expected_d, rtol=1e-5)
        assert result.converged


class TestMixedConstraintsQP:
    """Tests for QP with both equality and inequality constraints."""

    def test_equality_and_inequality(self):
        """Test QP with equality and inequality constraints.

        minimize (1/2)(x^2 + y^2 + z^2)
        subject to x + y + z = 3
                   x >= 0, y >= 0, z >= 0

        Solution: x = y = z = 1 (symmetric, sum = 3, all positive)
        """
        H = jnp.eye(3)
        g = jnp.zeros(3)
        A_eq = jnp.array([[1.0, 1.0, 1.0]])
        b_eq = jnp.array([3.0])
        A_ineq = jnp.eye(3)  # x >= 0, y >= 0, z >= 0
        b_ineq = jnp.zeros(3)

        result = solve_qp(H, g, A_eq, b_eq, A_ineq, b_ineq)

        expected_d = jnp.array([1.0, 1.0, 1.0])
        np.testing.assert_allclose(result.d, expected_d, rtol=1e-5)

        # Verify equality constraint
        np.testing.assert_allclose(A_eq @ result.d, b_eq, rtol=1e-6)

        # Verify inequalities
        assert jnp.all(A_ineq @ result.d >= b_ineq - 1e-6)
        assert result.converged

    def test_equality_with_active_inequality(self):
        """Test where some inequalities become active.

        minimize (1/2)((x-2)^2 + (y-2)^2)
        subject to x + y = 2
                   x >= 1

        Minimize on line x + y = 2.
        Without inequality: x = y = 1
        With x >= 1: still x = y = 1 (constraint barely active)
        """
        H = jnp.eye(2)
        g = jnp.array([-2.0, -2.0])  # gradient of (x-2)^2 + (y-2)^2 at origin
        A_eq = jnp.array([[1.0, 1.0]])
        b_eq = jnp.array([2.0])
        A_ineq = jnp.array([[1.0, 0.0]])
        b_ineq = jnp.array([1.0])

        result = solve_qp(H, g, A_eq, b_eq, A_ineq, b_ineq)

        # Equality: x + y = 2
        np.testing.assert_allclose(A_eq @ result.d, b_eq, rtol=1e-6)

        # Inequality: x >= 1
        assert result.d[0] >= 1.0 - 1e-6
        assert result.converged


class TestQPEdgeCases:
    """Edge cases for the QP solver."""

    def test_zero_gradient(self):
        """Test QP with zero gradient (solution at origin if feasible)."""
        H = jnp.eye(2)
        g = jnp.zeros(2)
        A_eq = jnp.zeros((0, 2))
        b_eq = jnp.zeros((0,))
        A_ineq = jnp.zeros((0, 2))
        b_ineq = jnp.zeros((0,))

        result = solve_qp(H, g, A_eq, b_eq, A_ineq, b_ineq)

        np.testing.assert_allclose(result.d, jnp.zeros(2), atol=1e-10)
        assert result.converged

    def test_identity_hessian(self):
        """Test QP with identity Hessian (steepest descent direction)."""
        H = jnp.eye(3)
        g = jnp.array([1.0, 2.0, 3.0])
        A_eq = jnp.zeros((0, 3))
        b_eq = jnp.zeros((0,))
        A_ineq = jnp.zeros((0, 3))
        b_ineq = jnp.zeros((0,))

        result = solve_qp(H, g, A_eq, b_eq, A_ineq, b_ineq)

        # With H = I, solution is d = -g
        expected_d = -g
        np.testing.assert_allclose(result.d, expected_d, rtol=1e-10)
        assert result.converged

    def test_scaled_hessian(self):
        """Test QP with scaled Hessian."""
        H = jnp.diag(jnp.array([1.0, 10.0, 100.0]))
        g = jnp.array([1.0, 1.0, 1.0])
        A_eq = jnp.zeros((0, 3))
        b_eq = jnp.zeros((0,))
        A_ineq = jnp.zeros((0, 3))
        b_ineq = jnp.zeros((0,))

        result = solve_qp(H, g, A_eq, b_eq, A_ineq, b_ineq)

        # d = -H^{-1} g = [-1, -0.1, -0.01]
        expected_d = jnp.array([-1.0, -0.1, -0.01])
        np.testing.assert_allclose(result.d, expected_d, rtol=1e-10)
        assert result.converged


class TestQPJIT:
    """Test that QP solver works with JAX JIT compilation."""

    def test_jit_compilation(self):
        """Test that solve_qp can be JIT compiled."""

        @jax.jit
        def solve_qp_jit(H, g, A_eq, b_eq, A_ineq, b_ineq):
            return solve_qp(H, g, A_eq, b_eq, A_ineq, b_ineq)

        H = jnp.eye(2)
        g = jnp.array([1.0, 1.0])
        A_eq = jnp.zeros((0, 2))
        b_eq = jnp.zeros((0,))
        A_ineq = jnp.array([[1.0, 0.0]])
        b_ineq = jnp.array([0.5])

        result = solve_qp_jit(H, g, A_eq, b_eq, A_ineq, b_ineq)

        # Should find d such that d[0] >= 0.5 (active) and d[1] = -1
        assert result.d[0] >= 0.5 - 1e-6
        assert result.converged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
