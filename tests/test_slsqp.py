"""Integration tests for the SLSQP solver.

These tests verify that the full SLSQP implementation works correctly
on various optimization problems with the L-BFGS Hessian approximation
and projected CG QP solver.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize as scipy_minimize

from slsqp_jax import SLSQP

# Enable 64-bit precision for numerical accuracy
jax.config.update("jax_enable_x64", True)


def _run_solver(solver, objective, x0, args=None, max_steps=None):
    """Run the SLSQP solver loop and return final iterate."""
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


class TestSLSQPUnconstrained:
    """Tests for unconstrained optimization with SLSQP."""

    def test_simple_quadratic(self):
        """minimize x^2 + y^2  =>  (0, 0)"""

        def objective(x, args):
            return jnp.sum(x**2), None

        solver = SLSQP(rtol=1e-8, atol=1e-8, max_steps=50)
        x0 = jnp.array([3.0, -2.0])
        y, _ = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y, [0.0, 0.0], atol=1e-5)

    def test_shifted_quadratic(self):
        """minimize (x-1)^2 + (y-2)^2  =>  (1, 2)"""

        def objective(x, args):
            return (x[0] - 1) ** 2 + (x[1] - 2) ** 2, None

        solver = SLSQP(rtol=1e-8, atol=1e-8, max_steps=50)
        x0 = jnp.array([0.0, 0.0])
        y, _ = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y, [1.0, 2.0], atol=1e-5)

    def test_rosenbrock_2d(self):
        """minimize (1-x)^2 + 100(y-x^2)^2  =>  (1, 1)"""

        def objective(x, args):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2, None

        solver = SLSQP(rtol=1e-8, atol=1e-8, max_steps=200)
        x0 = jnp.array([-1.0, 1.0])
        y, _ = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y, [1.0, 1.0], rtol=1e-3)


class TestSLSQPEqualityConstraints:
    """Tests for equality-constrained optimization."""

    def test_sphere_linear_equality(self):
        """minimize x^2+y^2+z^2 s.t. x+y+z=3  =>  (1,1,1)"""

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
        y, _ = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [1.0, 1.0, 1.0], rtol=1e-4)
        np.testing.assert_allclose(eq_constraint(y, None), 0.0, atol=1e-5)

    def test_quadratic_circle_constraint(self):
        """minimize (x-2)^2+(y-2)^2 s.t. x^2+y^2=1  =>  (1/sqrt2, 1/sqrt2)"""

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
        y, _ = _run_solver(solver, objective, x0)

        expected = jnp.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
        np.testing.assert_allclose(y, expected, rtol=1e-4)


class TestSLSQPInequalityConstraints:
    """Tests for inequality-constrained optimization."""

    def test_sphere_linear_inequality(self):
        """minimize x^2+y^2 s.t. x+y>=2  =>  (1,1)"""

        def objective(x, args):
            return jnp.sum(x**2), None

        def ineq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 2.0])

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=1,
        )
        x0 = jnp.array([2.0, 2.0])
        y, _ = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [1.0, 1.0], rtol=1e-4)
        assert ineq_constraint(y, None)[0] >= -1e-5

    def test_box_constraints(self):
        """minimize (x-3)^2+(y-3)^2 s.t. 0<=x,y<=2  =>  (2,2)"""

        def objective(x, args):
            return (x[0] - 3) ** 2 + (x[1] - 3) ** 2, None

        def ineq_constraint(x, args):
            return jnp.array([x[0], x[1], 2.0 - x[0], 2.0 - x[1]])

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=4,
        )
        x0 = jnp.array([1.0, 1.0])
        y, _ = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y, [2.0, 2.0], rtol=1e-3)


class TestSLSQPMixedConstraints:
    """Tests for mixed equality and inequality constraints."""

    def test_sphere_plane_and_halfspace(self):
        """minimize x^2+y^2+z^2 s.t. x+y+z=3, x,y,z>=0  =>  (1,1,1)"""

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] + x[2] - 3.0])

        def ineq_constraint(x, args):
            return x

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
        y, _ = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y, [1.0, 1.0, 1.0], rtol=1e-4)


class TestSLSQPJIT:
    """Test that SLSQP works with JAX JIT compilation."""

    def test_step_is_jittable(self):
        """Test that the step function can be JIT compiled."""

        def objective(x, args):
            return jnp.sum(x**2), None

        solver = SLSQP(rtol=1e-6, atol=1e-6, max_steps=10)
        x0 = jnp.array([1.0, 2.0])
        state = solver.init(objective, x0, None, {}, None, None, frozenset())

        @jax.jit
        def jit_step(y, state):
            return solver.step(objective, y, None, {}, state, frozenset())

        y = x0
        for _ in range(5):
            y, state, _ = jit_step(y, state)

        assert jnp.linalg.norm(y) < jnp.linalg.norm(x0)


class TestSLSQPUserSuppliedDerivatives:
    """Tests for user-supplied gradient, Jacobian, and HVP functions."""

    def test_user_gradient(self):
        """Test with user-supplied gradient function."""

        def objective(x, args):
            return jnp.sum(x**2), None

        def obj_grad(x, args):
            return 2.0 * x

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            obj_grad_fn=obj_grad,
        )
        x0 = jnp.array([3.0, -2.0])
        y, _ = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y, [0.0, 0.0], atol=1e-5)

    def test_user_jacobian(self):
        """Test with user-supplied constraint Jacobian."""

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] + x[2] - 3.0])

        def eq_jac(x, args):
            return jnp.array([[1.0, 1.0, 1.0]])

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            eq_jac_fn=eq_jac,
        )
        x0 = jnp.array([1.0, 1.0, 1.0])
        y, _ = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y, [1.0, 1.0, 1.0], rtol=1e-4)

    def test_user_hvp_unconstrained(self):
        """Test with user-supplied HVP for unconstrained problem."""

        def objective(x, args):
            return jnp.sum(x**2), None

        def obj_hvp(x, v, args):
            # Hessian of sum(x^2) is 2*I, so HVP is 2*v
            return 2.0 * v

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            obj_hvp_fn=obj_hvp,
        )
        x0 = jnp.array([3.0, -2.0])
        y, _ = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y, [0.0, 0.0], atol=1e-5)

    def test_user_hvp_equality_constrained(self):
        """Test with user-supplied HVP for equality-constrained problem.

        minimize x^2 + y^2 + z^2  s.t. x + y + z = 3
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] + x[2] - 3.0])

        def obj_hvp(x, v, args):
            return 2.0 * v

        def eq_hvp(x, v, args):
            # Hessian of linear constraint is zero
            return jnp.zeros((1, x.shape[0]))

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            obj_hvp_fn=obj_hvp,
            eq_hvp_fn=eq_hvp,
        )
        x0 = jnp.array([1.0, 1.0, 1.0])
        y, _ = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y, [1.0, 1.0, 1.0], rtol=1e-4)

    def test_user_hvp_ad_fallback_for_constraints(self):
        """Test with user-supplied objective HVP but AD fallback for constraints.

        The solver should compute constraint HVP via forward-over-reverse AD.
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            # Nonlinear constraint: x^2 + y^2 + z^2 = 3
            return jnp.array([jnp.sum(x**2) - 3.0])

        def obj_hvp(x, v, args):
            return 2.0 * v

        # No eq_hvp_fn provided -> AD fallback
        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            obj_hvp_fn=obj_hvp,
        )
        x0 = jnp.array([1.0, 1.0, 1.0])
        y, _ = _run_solver(solver, objective, x0)

        # Constraint: sum(x^2) = 3, and minimizing sum(x^2)
        # Solution: x = y = z = 1 on the sphere
        np.testing.assert_allclose(jnp.sum(y**2), 3.0, rtol=1e-4)


class TestSLSQPModerateScale:
    """Tests at moderate scale to verify scalability."""

    @pytest.mark.slow
    def test_quadratic_n100(self):
        """Test on a 100-dimensional quadratic problem.

        minimize sum_i (w_i * x_i^2)
        where w_i = i (different curvatures, condition number 100)
        """
        n = 100
        weights = jnp.arange(1, n + 1, dtype=jnp.float64)

        def objective(x, args):
            return jnp.sum(weights * x**2), None

        solver = SLSQP(rtol=1e-8, atol=1e-8, max_steps=300, lbfgs_memory=15)
        key = jax.random.PRNGKey(42)
        x0 = jax.random.normal(key, (n,))

        y, _ = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y, jnp.zeros(n), atol=1e-2)

    @pytest.mark.slow
    def test_quadratic_n100_with_equality(self):
        """Test 100-dimensional quadratic with a linear equality.

        minimize sum(x^2)
        s.t. sum(x) = n  (mean = 1)

        Solution: all x_i = 1
        """
        n = 100

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([jnp.sum(x) - float(n)])

        solver = SLSQP(
            rtol=1e-6,
            atol=1e-6,
            max_steps=100,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            lbfgs_memory=10,
        )
        x0 = jnp.ones(n)
        y, _ = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, jnp.ones(n), rtol=1e-3)
        np.testing.assert_allclose(jnp.sum(y), float(n), atol=1e-4)

    @pytest.mark.slow
    def test_quadratic_n100_with_hvp(self):
        """Test 100-dimensional quadratic with user-supplied HVP.

        minimize sum_i (w_i * x_i^2)
        """
        n = 100
        weights = jnp.arange(1, n + 1, dtype=jnp.float64)

        def objective(x, args):
            return jnp.sum(weights * x**2), None

        def obj_hvp(x, v, args):
            return 2.0 * weights * v

        solver = SLSQP(
            rtol=1e-6,
            atol=1e-6,
            max_steps=100,
            obj_hvp_fn=obj_hvp,
        )
        key = jax.random.PRNGKey(42)
        x0 = jax.random.normal(key, (n,))

        y, _ = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y, jnp.zeros(n), atol=1e-4)


class TestSLSQPComparisonWithSciPy:
    """Compare SLSQP-JAX results with SciPy's SLSQP."""

    def test_vs_scipy_unconstrained(self):
        """Compare unconstrained optimization with SciPy."""

        def objective_scipy(x):
            return x[0] ** 2 + 2 * x[1] ** 2

        def objective_jax(x, args):
            return x[0] ** 2 + 2 * x[1] ** 2, None

        x0 = np.array([3.0, -2.0])
        result_scipy = scipy_minimize(objective_scipy, x0, method="SLSQP")

        solver = SLSQP(rtol=1e-8, atol=1e-8, max_steps=50)
        y, _ = _run_solver(solver, objective_jax, jnp.array(x0))

        np.testing.assert_allclose(y, [0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(result_scipy.x, [0.0, 0.0], atol=1e-5)

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
        result_scipy = scipy_minimize(
            objective_scipy,
            x0,
            method="SLSQP",
            constraints={"type": "eq", "fun": constraint_scipy},
        )

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            eq_constraint_fn=constraint_jax,
            n_eq_constraints=1,
        )
        y, _ = _run_solver(solver, objective_jax, jnp.array(x0))

        np.testing.assert_allclose(y, result_scipy.x, rtol=1e-3)
