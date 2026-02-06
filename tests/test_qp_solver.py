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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
