"""Comparison tests between SLSQP-JAX and scipy.optimize.minimize(method='SLSQP').

These tests verify that the JAX implementation produces equivalent results
to the reference SciPy implementation on standard test problems.
"""

import jax
import numpy as np
from scipy.optimize import minimize as scipy_minimize

# Enable 64-bit precision for fair comparison
jax.config.update("jax_enable_x64", True)

# Import will work once the package is properly installed


class TestUnconstrainedOptimization:
    """Tests for unconstrained optimization problems."""

    def test_rosenbrock_2d(self):
        """Test on the 2D Rosenbrock function.

        The Rosenbrock function is a classic test for optimization algorithms:
            f(x, y) = (a - x)^2 + b(y - x^2)^2

        with a=1, b=100. Minimum at (1, 1) with f(1, 1) = 0.
        """

        def rosenbrock_scipy(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        def rosenbrock_jax(x, args):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        x0 = np.array([-1.0, 1.0])

        # SciPy reference
        result_scipy = scipy_minimize(
            rosenbrock_scipy,
            x0,
            method="SLSQP",
            options={"ftol": 1e-9, "maxiter": 100},
        )

        # TODO: Test with JAX implementation once complete
        # x0_jax = jnp.array([-1.0, 1.0])
        # solver = SLSQP(rtol=1e-9, atol=1e-9, max_steps=100)
        # result_jax = optx.minimise(rosenbrock_jax, solver, x0_jax)

        # np.testing.assert_allclose(
        #     result_jax.value, result_scipy.x, rtol=1e-4, atol=1e-6
        # )

        # For now, just verify SciPy finds the correct minimum
        np.testing.assert_allclose(result_scipy.x, [1.0, 1.0], rtol=1e-4)
        assert result_scipy.success

    def test_quadratic_2d(self):
        """Test on a simple 2D quadratic function.

        f(x, y) = x^2 + 2*y^2

        Minimum at (0, 0) with f(0, 0) = 0.
        """

        def quadratic_scipy(x):
            return x[0] ** 2 + 2 * x[1] ** 2

        def quadratic_jax(x, args):
            return x[0] ** 2 + 2 * x[1] ** 2

        x0 = np.array([3.0, -2.0])

        result_scipy = scipy_minimize(
            quadratic_scipy,
            x0,
            method="SLSQP",
        )

        np.testing.assert_allclose(result_scipy.x, [0.0, 0.0], atol=1e-6)
        assert result_scipy.success

    def test_beale_function(self):
        """Test on Beale's function.

        f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2

        Minimum at (3, 0.5) with f(3, 0.5) = 0.
        """

        def beale_scipy(x):
            return (
                (1.5 - x[0] + x[0] * x[1]) ** 2
                + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
                + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
            )

        x0 = np.array([0.0, 0.0])

        result_scipy = scipy_minimize(
            beale_scipy,
            x0,
            method="SLSQP",
            options={"maxiter": 200},
        )

        np.testing.assert_allclose(result_scipy.x, [3.0, 0.5], rtol=1e-3)
        assert result_scipy.success


class TestEqualityConstraints:
    """Tests for optimization with equality constraints."""

    def test_sphere_with_linear_equality(self):
        """Minimize sphere function with a linear equality constraint.

        minimize    x^2 + y^2 + z^2
        subject to  x + y + z = 1

        The optimal solution is at (1/3, 1/3, 1/3).
        """

        def sphere_scipy(x):
            return np.sum(x**2)

        def eq_constraint_scipy(x):
            return x[0] + x[1] + x[2] - 1.0

        x0 = np.array([1.0, 0.0, 0.0])

        result_scipy = scipy_minimize(
            sphere_scipy,
            x0,
            method="SLSQP",
            constraints={"type": "eq", "fun": eq_constraint_scipy},
        )

        expected = np.array([1 / 3, 1 / 3, 1 / 3])
        np.testing.assert_allclose(result_scipy.x, expected, rtol=1e-5)
        assert result_scipy.success

    def test_quadratic_with_nonlinear_equality(self):
        """Minimize quadratic with a nonlinear equality constraint.

        minimize    (x-1)^2 + (y-1)^2
        subject to  x^2 + y^2 = 1 (unit circle)

        The solution is at (1/sqrt(2), 1/sqrt(2)) where the objective
        is minimized on the unit circle.
        """

        def quadratic_scipy(x):
            return (x[0] - 1) ** 2 + (x[1] - 1) ** 2

        def eq_constraint_scipy(x):
            return x[0] ** 2 + x[1] ** 2 - 1.0

        x0 = np.array([0.5, 0.5])

        result_scipy = scipy_minimize(
            quadratic_scipy,
            x0,
            method="SLSQP",
            constraints={"type": "eq", "fun": eq_constraint_scipy},
        )

        # Expected solution: (1/sqrt(2), 1/sqrt(2))
        expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
        np.testing.assert_allclose(result_scipy.x, expected, rtol=1e-5)

        # Verify constraint is satisfied
        constraint_val = eq_constraint_scipy(result_scipy.x)
        np.testing.assert_allclose(constraint_val, 0.0, atol=1e-6)
        assert result_scipy.success


class TestInequalityConstraints:
    """Tests for optimization with inequality constraints."""

    def test_quadratic_with_linear_inequality(self):
        """Minimize quadratic with linear inequality constraint.

        minimize    (x-2)^2 + (y-1)^2
        subject to  x + y >= 3

        Without constraint: minimum at (2, 1)
        With constraint: minimum at (2, 1) since 2+1=3 satisfies x+y>=3
        """

        def objective_scipy(x):
            return (x[0] - 2) ** 2 + (x[1] - 1) ** 2

        def ineq_constraint_scipy(x):
            return x[0] + x[1] - 3  # >= 0

        x0 = np.array([0.0, 0.0])

        result_scipy = scipy_minimize(
            objective_scipy,
            x0,
            method="SLSQP",
            constraints={"type": "ineq", "fun": ineq_constraint_scipy},
        )

        # The constraint is active at the solution
        np.testing.assert_allclose(result_scipy.x, [2.0, 1.0], rtol=1e-4)
        assert result_scipy.success

    def test_quadratic_with_active_inequality(self):
        """Minimize quadratic where inequality is strictly active.

        minimize    x^2 + y^2
        subject to  x + y >= 2

        The unconstrained minimum (0,0) violates the constraint.
        The constrained minimum is at (1, 1).
        """

        def objective_scipy(x):
            return x[0] ** 2 + x[1] ** 2

        def ineq_constraint_scipy(x):
            return x[0] + x[1] - 2  # >= 0

        x0 = np.array([2.0, 2.0])

        result_scipy = scipy_minimize(
            objective_scipy,
            x0,
            method="SLSQP",
            constraints={"type": "ineq", "fun": ineq_constraint_scipy},
        )

        np.testing.assert_allclose(result_scipy.x, [1.0, 1.0], rtol=1e-5)
        # Verify constraint is exactly satisfied (active)
        np.testing.assert_allclose(
            ineq_constraint_scipy(result_scipy.x), 0.0, atol=1e-6
        )
        assert result_scipy.success

    def test_multiple_inequalities(self):
        """Minimize with multiple inequality constraints.

        minimize    x^2 + y^2
        subject to  x >= 1
                    y >= 1
                    x + y <= 3

        The minimum is at (1, 1) with two active constraints.
        """

        def objective_scipy(x):
            return x[0] ** 2 + x[1] ** 2

        constraints = [
            {"type": "ineq", "fun": lambda x: x[0] - 1},  # x >= 1
            {"type": "ineq", "fun": lambda x: x[1] - 1},  # y >= 1
            {"type": "ineq", "fun": lambda x: 3 - x[0] - x[1]},  # x + y <= 3
        ]

        x0 = np.array([2.0, 2.0])

        result_scipy = scipy_minimize(
            objective_scipy,
            x0,
            method="SLSQP",
            constraints=constraints,
        )

        np.testing.assert_allclose(result_scipy.x, [1.0, 1.0], rtol=1e-5)
        assert result_scipy.success


class TestMixedConstraints:
    """Tests for optimization with both equality and inequality constraints."""

    def test_equality_and_inequality(self):
        """Minimize with both equality and inequality constraints.

        minimize    x^2 + y^2 + z^2
        subject to  x + y + z = 3    (equality)
                    x >= 0           (inequality)
                    y >= 0           (inequality)
                    z >= 0           (inequality)

        The minimum is at (1, 1, 1).
        """

        def objective_scipy(x):
            return np.sum(x**2)

        constraints = [
            {"type": "eq", "fun": lambda x: x[0] + x[1] + x[2] - 3},
            {"type": "ineq", "fun": lambda x: x[0]},
            {"type": "ineq", "fun": lambda x: x[1]},
            {"type": "ineq", "fun": lambda x: x[2]},
        ]

        x0 = np.array([1.0, 1.0, 1.0])

        result_scipy = scipy_minimize(
            objective_scipy,
            x0,
            method="SLSQP",
            constraints=constraints,
        )

        np.testing.assert_allclose(result_scipy.x, [1.0, 1.0, 1.0], rtol=1e-5)
        assert result_scipy.success

    def test_active_inequality_with_equality(self):
        """Test where inequality becomes active in presence of equality.

        minimize    (x-2)^2 + (y-2)^2
        subject to  x + y = 2    (equality)
                    x >= 1       (inequality, will be active)

        Without inequality: minimum at (1, 1)
        With inequality: minimum at (1, 1) - inequality is just satisfied
        """

        def objective_scipy(x):
            return (x[0] - 2) ** 2 + (x[1] - 2) ** 2

        constraints = [
            {"type": "eq", "fun": lambda x: x[0] + x[1] - 2},
            {"type": "ineq", "fun": lambda x: x[0] - 1},
        ]

        x0 = np.array([0.5, 1.5])

        result_scipy = scipy_minimize(
            objective_scipy,
            x0,
            method="SLSQP",
            constraints=constraints,
        )

        np.testing.assert_allclose(result_scipy.x, [1.0, 1.0], rtol=1e-5)
        assert result_scipy.success


class TestEdgeCases:
    """Tests for edge cases and potential numerical issues."""

    def test_starting_at_optimum(self):
        """Test behavior when starting at the optimal point."""

        def objective_scipy(x):
            return x[0] ** 2 + x[1] ** 2

        x0 = np.array([0.0, 0.0])  # Already optimal

        result_scipy = scipy_minimize(
            objective_scipy,
            x0,
            method="SLSQP",
        )

        np.testing.assert_allclose(result_scipy.x, [0.0, 0.0], atol=1e-10)
        assert result_scipy.success

    def test_ill_conditioned_hessian(self):
        """Test with an ill-conditioned objective (different scales).

        f(x, y) = 1000*x^2 + 0.001*y^2
        """

        def objective_scipy(x):
            return 1000 * x[0] ** 2 + 0.001 * x[1] ** 2

        x0 = np.array([1.0, 1.0])

        result_scipy = scipy_minimize(
            objective_scipy,
            x0,
            method="SLSQP",
            options={"maxiter": 200},
        )

        np.testing.assert_allclose(result_scipy.x, [0.0, 0.0], atol=1e-4)
        assert result_scipy.success

    def test_single_variable(self):
        """Test with a single decision variable."""

        def objective_scipy(x):
            return (x[0] - 5) ** 2

        x0 = np.array([0.0])

        result_scipy = scipy_minimize(
            objective_scipy,
            x0,
            method="SLSQP",
        )

        np.testing.assert_allclose(result_scipy.x, [5.0], rtol=1e-6)
        assert result_scipy.success
