"""Integration tests for the SLSQP solver.

These tests verify that the full SLSQP implementation works correctly
on various optimization problems and can be used with JAX transformations.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize as scipy_minimize

from slsqp_jax import SLSQP

# Enable 64-bit precision for numerical accuracy
jax.config.update("jax_enable_x64", True)


class TestSLSQPUnconstrained:
    """Tests for unconstrained optimization with SLSQP."""

    def test_simple_quadratic(self):
        """Test on a simple quadratic function.

        minimize x^2 + y^2

        Solution: (0, 0)
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        solver = SLSQP(rtol=1e-8, atol=1e-8, max_steps=50)
        x0 = jnp.array([3.0, -2.0])

        # Manual iteration loop (since we're testing the solver directly)
        state = solver.init(objective, x0, None, {}, None, None, frozenset())
        y = x0

        for _ in range(solver.max_steps):
            done, _ = solver.terminate(objective, y, None, {}, state, frozenset())
            if done:
                break
            y, state, _ = solver.step(objective, y, None, {}, state, frozenset())

        np.testing.assert_allclose(y, [0.0, 0.0], atol=1e-6)

    def test_shifted_quadratic(self):
        """Test on a shifted quadratic.

        minimize (x-1)^2 + (y-2)^2

        Solution: (1, 2)
        """

        def objective(x, args):
            return (x[0] - 1) ** 2 + (x[1] - 2) ** 2, None

        solver = SLSQP(rtol=1e-8, atol=1e-8, max_steps=50)
        x0 = jnp.array([0.0, 0.0])

        state = solver.init(objective, x0, None, {}, None, None, frozenset())
        y = x0

        for _ in range(solver.max_steps):
            done, _ = solver.terminate(objective, y, None, {}, state, frozenset())
            if done:
                break
            y, state, _ = solver.step(objective, y, None, {}, state, frozenset())

        np.testing.assert_allclose(y, [1.0, 2.0], atol=1e-6)

    def test_rosenbrock_2d(self):
        """Test on 2D Rosenbrock function.

        minimize (1-x)^2 + 100(y-x^2)^2

        Solution: (1, 1)
        """

        def objective(x, args):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2, None

        solver = SLSQP(rtol=1e-8, atol=1e-8, max_steps=100)
        x0 = jnp.array([-1.0, 1.0])

        state = solver.init(objective, x0, None, {}, None, None, frozenset())
        y = x0

        for _ in range(solver.max_steps):
            done, _ = solver.terminate(objective, y, None, {}, state, frozenset())
            if done:
                break
            y, state, _ = solver.step(objective, y, None, {}, state, frozenset())

        # Rosenbrock is harder, use looser tolerance
        np.testing.assert_allclose(y, [1.0, 1.0], rtol=1e-4)


class TestSLSQPEqualityConstraints:
    """Tests for equality-constrained optimization."""

    def test_sphere_linear_equality(self):
        """Minimize sphere with linear equality.

        minimize x^2 + y^2 + z^2
        subject to x + y + z = 3

        Solution: (1, 1, 1)
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] + x[2] - 3.0])

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
        )
        x0 = jnp.array([1.0, 1.0, 1.0])

        state = solver.init(objective, x0, None, {}, None, None, frozenset())
        y = x0

        for _ in range(solver.max_steps):
            done, _ = solver.terminate(objective, y, None, {}, state, frozenset())
            if done:
                break
            y, state, _ = solver.step(objective, y, None, {}, state, frozenset())

        np.testing.assert_allclose(y, [1.0, 1.0, 1.0], rtol=1e-5)

        # Verify constraint is satisfied
        constraint_val = eq_constraint(y, None)
        np.testing.assert_allclose(constraint_val, 0.0, atol=1e-6)

    def test_quadratic_circle_constraint(self):
        """Minimize distance to point with circle constraint.

        minimize (x-2)^2 + (y-2)^2
        subject to x^2 + y^2 = 1

        Solution: (1/sqrt(2), 1/sqrt(2))
        """

        def objective(x, args):
            return (x[0] - 2) ** 2 + (x[1] - 2) ** 2, None

        def eq_constraint(x, args):
            return jnp.array([x[0] ** 2 + x[1] ** 2 - 1.0])

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
        )
        x0 = jnp.array([0.5, 0.5])

        state = solver.init(objective, x0, None, {}, None, None, frozenset())
        y = x0

        for _ in range(solver.max_steps):
            done, _ = solver.terminate(objective, y, None, {}, state, frozenset())
            if done:
                break
            y, state, _ = solver.step(objective, y, None, {}, state, frozenset())

        expected = jnp.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
        np.testing.assert_allclose(y, expected, rtol=1e-5)


class TestSLSQPInequalityConstraints:
    """Tests for inequality-constrained optimization."""

    def test_sphere_linear_inequality(self):
        """Minimize sphere with linear inequality.

        minimize x^2 + y^2
        subject to x + y >= 2

        Solution: (1, 1)
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def ineq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 2.0])  # >= 0

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=1,
        )
        x0 = jnp.array([2.0, 2.0])

        state = solver.init(objective, x0, None, {}, None, None, frozenset())
        y = x0

        for _ in range(solver.max_steps):
            done, _ = solver.terminate(objective, y, None, {}, state, frozenset())
            if done:
                break
            y, state, _ = solver.step(objective, y, None, {}, state, frozenset())

        np.testing.assert_allclose(y, [1.0, 1.0], rtol=1e-5)

        # Verify constraint is satisfied
        constraint_val = ineq_constraint(y, None)
        assert constraint_val[0] >= -1e-6

    def test_box_constraints(self):
        """Minimize with box constraints.

        minimize (x-3)^2 + (y-3)^2
        subject to 0 <= x <= 2
                   0 <= y <= 2

        Solution: (2, 2) - at the corner
        """

        def objective(x, args):
            return (x[0] - 3) ** 2 + (x[1] - 3) ** 2, None

        def ineq_constraint(x, args):
            return jnp.array(
                [
                    x[0],  # x >= 0
                    x[1],  # y >= 0
                    2.0 - x[0],  # x <= 2
                    2.0 - x[1],  # y <= 2
                ]
            )

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=4,
        )
        x0 = jnp.array([1.0, 1.0])

        state = solver.init(objective, x0, None, {}, None, None, frozenset())
        y = x0

        for _ in range(solver.max_steps):
            done, _ = solver.terminate(objective, y, None, {}, state, frozenset())
            if done:
                break
            y, state, _ = solver.step(objective, y, None, {}, state, frozenset())

        np.testing.assert_allclose(y, [2.0, 2.0], rtol=1e-4)


class TestSLSQPMixedConstraints:
    """Tests for mixed equality and inequality constraints."""

    def test_sphere_plane_and_halfspace(self):
        """Minimize with equality and inequality.

        minimize x^2 + y^2 + z^2
        subject to x + y + z = 3  (equality)
                   x >= 0, y >= 0, z >= 0  (inequalities)

        Solution: (1, 1, 1)
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] + x[2] - 3.0])

        def ineq_constraint(x, args):
            return x  # x >= 0, y >= 0, z >= 0

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            eq_constraint_fn=eq_constraint,
            ineq_constraint_fn=ineq_constraint,
            n_eq_constraints=1,
            n_ineq_constraints=3,
        )
        x0 = jnp.array([1.0, 1.0, 1.0])

        state = solver.init(objective, x0, None, {}, None, None, frozenset())
        y = x0

        for _ in range(solver.max_steps):
            done, _ = solver.terminate(objective, y, None, {}, state, frozenset())
            if done:
                break
            y, state, _ = solver.step(objective, y, None, {}, state, frozenset())

        np.testing.assert_allclose(y, [1.0, 1.0, 1.0], rtol=1e-5)


class TestSLSQPJIT:
    """Test that SLSQP works with JAX JIT compilation."""

    def test_step_is_jittable(self):
        """Test that the step function can be JIT compiled."""

        def objective(x, args):
            return jnp.sum(x**2), None

        solver = SLSQP(rtol=1e-6, atol=1e-6, max_steps=10)
        x0 = jnp.array([1.0, 2.0])

        state = solver.init(objective, x0, None, {}, None, None, frozenset())

        # JIT compile the step function
        @jax.jit
        def jit_step(y, state):
            return solver.step(objective, y, None, {}, state, frozenset())

        # Run a few steps
        y = x0
        for _ in range(5):
            y, state, _ = jit_step(y, state)

        # Should have moved toward (0, 0)
        assert jnp.linalg.norm(y) < jnp.linalg.norm(x0)


class TestSLSQPComparisonWithSciPy:
    """Compare SLSQP-JAX results with SciPy's SLSQP."""

    def test_vs_scipy_unconstrained(self):
        """Compare unconstrained optimization with SciPy."""

        def objective_scipy(x):
            return x[0] ** 2 + 2 * x[1] ** 2

        def objective_jax(x, args):
            return x[0] ** 2 + 2 * x[1] ** 2, None

        x0 = np.array([3.0, -2.0])

        # SciPy result
        result_scipy = scipy_minimize(objective_scipy, x0, method="SLSQP")

        # JAX result
        solver = SLSQP(rtol=1e-8, atol=1e-8, max_steps=50)
        x0_jax = jnp.array(x0)
        state = solver.init(objective_jax, x0_jax, None, {}, None, None, frozenset())
        y = x0_jax

        for _ in range(solver.max_steps):
            done, _ = solver.terminate(objective_jax, y, None, {}, state, frozenset())
            if done:
                break
            y, state, _ = solver.step(objective_jax, y, None, {}, state, frozenset())

        # Both should find approximately (0, 0)
        # Note: the exact values may differ at the numerical precision level
        np.testing.assert_allclose(y, [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(result_scipy.x, [0.0, 0.0], atol=1e-6)

    def test_vs_scipy_equality_constrained(self):
        """Compare equality-constrained optimization with SciPy."""

        def objective_scipy(x):
            return x[0] ** 2 + x[1] ** 2

        def constraint_scipy(x):
            return x[0] + x[1] - 1.0

        def objective_jax(x, args):
            return x[0] ** 2 + x[1] ** 2, None

        def constraint_jax(x, args):
            return jnp.array([x[0] + x[1] - 1.0])

        x0 = np.array([0.0, 0.0])

        # SciPy result
        result_scipy = scipy_minimize(
            objective_scipy,
            x0,
            method="SLSQP",
            constraints={"type": "eq", "fun": constraint_scipy},
        )

        # JAX result
        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            eq_constraint_fn=constraint_jax,
            n_eq_constraints=1,
        )
        x0_jax = jnp.array(x0)
        state = solver.init(objective_jax, x0_jax, None, {}, None, None, frozenset())
        y = x0_jax

        for _ in range(solver.max_steps):
            done, _ = solver.terminate(objective_jax, y, None, {}, state, frozenset())
            if done:
                break
            y, state, _ = solver.step(objective_jax, y, None, {}, state, frozenset())

        # Both should find (0.5, 0.5)
        np.testing.assert_allclose(y, result_scipy.x, rtol=1e-4)
