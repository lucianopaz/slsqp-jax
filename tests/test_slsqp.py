"""Integration tests for the SLSQP solver.

These tests verify that the full SLSQP implementation works correctly
on various optimization problems with the L-BFGS Hessian approximation
and projected CG QP solver.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
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

    def test_user_hvp_ad_fallback_for_ineq_constraints(self):
        """Test with user-supplied objective HVP but AD fallback for ineq constraints.

        The solver should compute inequality constraint HVP via forward-over-reverse AD.

        minimize x^2 + y^2  s.t. x + y >= 2
        Unconstrained minimum at (0, 0), but constraint pushes to (1, 1).
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def ineq_constraint(x, args):
            # Nonlinear inequality: x^2 + y >= 1 (so x^2 + y - 1 >= 0)
            return jnp.array([x[0] ** 2 + x[1] - 1.0])

        def obj_hvp(x, v, args):
            return 2.0 * v

        # No ineq_hvp_fn provided -> AD fallback
        solver = SLSQP(
            rtol=1e-6,
            atol=1e-6,
            max_steps=50,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=1,
            obj_hvp_fn=obj_hvp,
        )
        x0 = jnp.array([1.0, 1.0])
        y, _ = _run_solver(solver, objective, x0)

        # Constraint should be satisfied: x^2 + y >= 1
        constraint_val = ineq_constraint(y, None)[0]
        assert constraint_val >= -1e-4, f"Constraint violated: {constraint_val}"


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


class TestOptimistixMinimise:
    """Tests using the optimistix.minimise high-level interface."""

    def test_unconstrained_auto_derivatives(self):
        """Unconstrained problem with all derivatives computed by AD."""

        def objective(x, args):
            return jnp.sum(x**2), None

        solver = SLSQP(rtol=1e-8, atol=1e-8)
        x0 = jnp.array([3.0, -2.0])
        sol = optx.minimise(objective, solver, x0, has_aux=True, max_steps=50)

        np.testing.assert_allclose(sol.value, [0.0, 0.0], atol=1e-5)

    def test_equality_constrained_auto_derivatives(self):
        """Equality-constrained problem with all derivatives computed by AD.

        minimize x^2+y^2+z^2 s.t. x+y+z=3  =>  (1,1,1)
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] + x[2] - 3.0])

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
        )
        x0 = jnp.array([1.0, 1.0, 1.0])
        sol = optx.minimise(objective, solver, x0, has_aux=True, max_steps=50)

        np.testing.assert_allclose(sol.value, [1.0, 1.0, 1.0], rtol=1e-4)
        np.testing.assert_allclose(eq_constraint(sol.value, None), 0.0, atol=1e-5)

    def test_inequality_constrained_auto_derivatives(self):
        """Inequality-constrained problem with all derivatives computed by AD.

        minimize x^2+y^2 s.t. x+y>=2  =>  (1,1)
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def ineq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 2.0])

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=1,
        )
        x0 = jnp.array([2.0, 2.0])
        sol = optx.minimise(objective, solver, x0, has_aux=True, max_steps=50)

        np.testing.assert_allclose(sol.value, [1.0, 1.0], rtol=1e-4)

    def test_mixed_constraints_auto_derivatives(self):
        """Mixed equality and inequality constraints with AD derivatives.

        minimize x^2+y^2+z^2 s.t. x+y+z=3, x,y,z>=0  =>  (1,1,1)
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] + x[2] - 3.0])

        def ineq_constraint(x, args):
            return x

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            eq_constraint_fn=eq_constraint,
            ineq_constraint_fn=ineq_constraint,
            n_eq_constraints=1,
            n_ineq_constraints=3,
        )
        x0 = jnp.array([1.0, 1.0, 1.0])
        sol = optx.minimise(objective, solver, x0, has_aux=True, max_steps=50)

        np.testing.assert_allclose(sol.value, [1.0, 1.0, 1.0], rtol=1e-4)

    def test_user_supplied_gradient(self):
        """Test optimistix.minimise with user-supplied gradient."""

        def objective(x, args):
            return jnp.sum(x**2), None

        def obj_grad(x, args):
            return 2.0 * x

        solver = SLSQP(rtol=1e-8, atol=1e-8, obj_grad_fn=obj_grad)
        x0 = jnp.array([3.0, -2.0])
        sol = optx.minimise(objective, solver, x0, has_aux=True, max_steps=50)

        np.testing.assert_allclose(sol.value, [0.0, 0.0], atol=1e-5)

    def test_user_supplied_gradient_and_jacobian(self):
        """Test optimistix.minimise with user-supplied gradient and Jacobian.

        minimize x^2+y^2 s.t. x+y=1  =>  (0.5, 0.5)
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 1.0])

        def obj_grad(x, args):
            return 2.0 * x

        def eq_jac(x, args):
            return jnp.array([[1.0, 1.0]])

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            obj_grad_fn=obj_grad,
            eq_jac_fn=eq_jac,
        )
        x0 = jnp.array([0.5, 0.5])
        sol = optx.minimise(objective, solver, x0, has_aux=True, max_steps=50)

        np.testing.assert_allclose(sol.value, [0.5, 0.5], rtol=1e-4)

    def test_user_supplied_hvp_unconstrained(self):
        """Test optimistix.minimise with user-supplied HVP (exact Hessian mode)."""

        def objective(x, args):
            return jnp.sum(x**2), None

        def obj_hvp(x, v, args):
            return 2.0 * v

        solver = SLSQP(rtol=1e-8, atol=1e-8, obj_hvp_fn=obj_hvp)
        x0 = jnp.array([3.0, -2.0])
        sol = optx.minimise(objective, solver, x0, has_aux=True, max_steps=50)

        np.testing.assert_allclose(sol.value, [0.0, 0.0], atol=1e-5)

    def test_user_supplied_hvp_equality_constrained(self):
        """Test optimistix.minimise with full user-supplied HVPs and constraints.

        minimize x^2+y^2+z^2 s.t. x+y+z=3  =>  (1,1,1)
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] + x[2] - 3.0])

        def obj_hvp(x, v, args):
            return 2.0 * v

        def eq_hvp(x, v, args):
            return jnp.zeros((1, x.shape[0]))

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            obj_hvp_fn=obj_hvp,
            eq_hvp_fn=eq_hvp,
        )
        x0 = jnp.array([1.0, 1.0, 1.0])
        sol = optx.minimise(objective, solver, x0, has_aux=True, max_steps=50)

        np.testing.assert_allclose(sol.value, [1.0, 1.0, 1.0], rtol=1e-4)
        np.testing.assert_allclose(eq_constraint(sol.value, None), 0.0, atol=1e-5)

    def test_user_supplied_hvp_with_ad_constraint_fallback(self):
        """Test optimistix.minimise with user objective HVP but AD for constraints.

        minimize x^2+y^2+z^2 s.t. x^2+y^2+z^2=3  =>  x=y=z=1
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([jnp.sum(x**2) - 3.0])

        def obj_hvp(x, v, args):
            return 2.0 * v

        # No eq_hvp_fn — solver uses forward-over-reverse AD for constraint HVP
        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            obj_hvp_fn=obj_hvp,
        )
        x0 = jnp.array([1.0, 1.0, 1.0])
        sol = optx.minimise(objective, solver, x0, has_aux=True, max_steps=50)

        np.testing.assert_allclose(jnp.sum(sol.value**2), 3.0, rtol=1e-4)

    def test_all_derivatives_user_supplied(self):
        """Test optimistix.minimise with every derivative function user-supplied.

        minimize (x-1)^2+(y-2)^2 s.t. x+y>=2  =>  (1,2) if feasible, else on boundary
        Unconstrained min at (1,2): x+y=3>=2, so constraint inactive.
        """

        def objective(x, args):
            return (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2, None

        def ineq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 2.0])

        def obj_grad(x, args):
            return jnp.array([2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)])

        def ineq_jac(x, args):
            return jnp.array([[1.0, 1.0]])

        def obj_hvp(x, v, args):
            return 2.0 * v

        def ineq_hvp(x, v, args):
            return jnp.zeros((1, x.shape[0]))

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=1,
            obj_grad_fn=obj_grad,
            ineq_jac_fn=ineq_jac,
            obj_hvp_fn=obj_hvp,
            ineq_hvp_fn=ineq_hvp,
        )
        x0 = jnp.array([0.0, 0.0])
        sol = optx.minimise(objective, solver, x0, has_aux=True, max_steps=50)

        np.testing.assert_allclose(sol.value, [1.0, 2.0], atol=1e-4)

    def test_solution_object_fields(self):
        """Verify the optimistix.Solution object has the expected fields."""

        def objective(x, args):
            return jnp.sum(x**2), None

        solver = SLSQP(rtol=1e-8, atol=1e-8)
        x0 = jnp.array([1.0, 2.0])
        sol = optx.minimise(objective, solver, x0, has_aux=True, max_steps=50)

        # sol.value is the optimum
        assert sol.value.shape == (2,)
        # sol.aux is the auxiliary output (None in our case)
        assert sol.aux is None
        # sol.stats should be a dict
        assert isinstance(sol.stats, dict)

    def test_throw_false_returns_result(self):
        """Test that throw=False returns a Solution even if not converged."""

        def objective(x, args):
            return jnp.sum(x**2), None

        solver = SLSQP(rtol=1e-15, atol=1e-15)  # impossibly tight tolerance
        x0 = jnp.array([3.0, -2.0])
        # Should not raise, just return with a non-successful result
        sol = optx.minimise(
            objective,
            solver,
            x0,
            has_aux=True,
            max_steps=2,
            throw=False,
        )

        # Value should still be a valid array (even if not fully converged)
        assert sol.value.shape == (2,)


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


class TestSLSQPBoxConstraints:
    """Tests for box constraints (bounds) on decision variables."""

    def test_simple_lower_bound(self):
        """Minimize x^2 subject to x >= 2  =>  x = 2"""

        def objective(x, args):
            return x[0] ** 2, None

        bounds = jnp.array([[2.0, jnp.inf]])  # x >= 2

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            bounds=bounds,
        )
        x0 = jnp.array([5.0])
        y, _ = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [2.0], rtol=1e-4)

    def test_simple_upper_bound(self):
        """Minimize -x subject to x <= 3  =>  x = 3"""

        def objective(x, args):
            return -x[0], None

        bounds = jnp.array([[-jnp.inf, 3.0]])  # x <= 3

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            bounds=bounds,
        )
        x0 = jnp.array([0.0])
        y, _ = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [3.0], rtol=1e-4)

    def test_box_bounds(self):
        """Minimize (x-5)^2 + (y-5)^2 subject to 0 <= x,y <= 3  =>  (3, 3)"""

        def objective(x, args):
            return (x[0] - 5) ** 2 + (x[1] - 5) ** 2, None

        bounds = jnp.array(
            [
                [0.0, 3.0],
                [0.0, 3.0],
            ]
        )

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            bounds=bounds,
        )
        x0 = jnp.array([1.0, 1.0])
        y, _ = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [3.0, 3.0], rtol=1e-3)
        # Check bounds are satisfied
        assert jnp.all(y >= 0.0 - 1e-5)
        assert jnp.all(y <= 3.0 + 1e-5)

    def test_bounds_with_equality_constraint(self):
        """Minimize x^2 + y^2 subject to x + y = 4, 0 <= x,y <= 3  =>  (2, 2)"""

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 4.0])

        bounds = jnp.array(
            [
                [0.0, 3.0],
                [0.0, 3.0],
            ]
        )

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            bounds=bounds,
        )
        x0 = jnp.array([2.0, 2.0])
        y, _ = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [2.0, 2.0], rtol=1e-3)
        np.testing.assert_allclose(y[0] + y[1], 4.0, atol=1e-5)

    def test_bounds_with_inequality_constraint(self):
        """Minimize (x-5)^2 + (y-5)^2 subject to x + y >= 3, 0 <= x,y <= 2

        The optimal unconstrained solution is (5, 5), but bounds limit to (2, 2).
        The sum constraint x + y >= 3 is satisfied at (2, 2) since 4 >= 3.
        """

        def objective(x, args):
            return (x[0] - 5) ** 2 + (x[1] - 5) ** 2, None

        def ineq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 3.0])

        bounds = jnp.array(
            [
                [0.0, 2.0],
                [0.0, 2.0],
            ]
        )

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=1,
            bounds=bounds,
        )
        x0 = jnp.array([1.0, 1.0])
        y, _ = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [2.0, 2.0], rtol=1e-3)
        # Check constraints
        assert y[0] + y[1] >= 3.0 - 1e-5
        assert jnp.all(y >= 0.0 - 1e-5)
        assert jnp.all(y <= 2.0 + 1e-5)

    def test_bounds_inactive(self):
        """Test when bounds exist but are not active at the solution.

        Minimize (x-1)^2 + (y-1)^2 with bounds 0 <= x,y <= 10
        Solution: (1, 1) which is interior to the box.
        """

        def objective(x, args):
            return (x[0] - 1) ** 2 + (x[1] - 1) ** 2, None

        bounds = jnp.array(
            [
                [0.0, 10.0],
                [0.0, 10.0],
            ]
        )

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            bounds=bounds,
        )
        x0 = jnp.array([5.0, 5.0])
        y, _ = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [1.0, 1.0], rtol=1e-4)

    def test_partial_bounds(self):
        """Test with some variables bounded and some unbounded.

        Minimize x^2 + y^2 subject to x >= 2 (y unbounded)
        Solution: (2, 0)
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        bounds = jnp.array(
            [
                [2.0, jnp.inf],  # x >= 2
                [-jnp.inf, jnp.inf],  # y unbounded
            ]
        )

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            bounds=bounds,
        )
        x0 = jnp.array([5.0, 5.0])
        y, _ = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [2.0, 0.0], rtol=1e-3, atol=1e-6)

    def test_mixed_bounds(self):
        """Test with different bound types on each variable.

        Minimize (x-5)^2 + (y+5)^2 + z^2
        subject to x >= 0, y <= 0, -1 <= z <= 1

        Solution:
        - x: minimize (x-5)^2 with x >= 0 -> x = 5
        - y: minimize (y+5)^2 with y <= 0 -> y = -5 (since -5 <= 0)
        - z: minimize z^2 with -1 <= z <= 1 -> z = 0
        """

        def objective(x, args):
            return (x[0] - 5) ** 2 + (x[1] + 5) ** 2 + x[2] ** 2, None

        bounds = jnp.array(
            [
                [0.0, jnp.inf],  # x >= 0 only
                [-jnp.inf, 0.0],  # y <= 0 only
                [-1.0, 1.0],  # z in [-1, 1]
            ]
        )

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            bounds=bounds,
        )
        x0 = jnp.array([0.0, 0.0, 0.0])
        y, _ = _run_solver(solver, objective, x0)

        # y = -5 is optimal because (y+5)^2 is minimized at y=-5, and -5 <= 0 satisfies the bound
        np.testing.assert_allclose(y, [5.0, -5.0, 0.0], rtol=1e-3)

    def test_no_bounds(self):
        """Test that bounds=None behaves the same as before."""

        def objective(x, args):
            return jnp.sum(x**2), None

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            bounds=None,
        )
        x0 = jnp.array([3.0, -2.0])
        y, _ = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [0.0, 0.0], atol=1e-5)

    def test_all_infinite_bounds(self):
        """Test with explicit bounds that are all infinite (no constraints)."""

        def objective(x, args):
            return jnp.sum(x**2), None

        bounds = jnp.array(
            [
                [-jnp.inf, jnp.inf],
                [-jnp.inf, jnp.inf],
            ]
        )

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            bounds=bounds,
        )
        x0 = jnp.array([3.0, -2.0])
        y, _ = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [0.0, 0.0], atol=1e-5)

    def test_bounds_with_optimistix(self):
        """Test bounds work with optimistix.minimise API."""

        def objective(x, args):
            return (x[0] - 5) ** 2 + (x[1] - 5) ** 2, None

        bounds = jnp.array(
            [
                [0.0, 2.0],
                [0.0, 2.0],
            ]
        )

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            bounds=bounds,
        )
        x0 = jnp.array([1.0, 1.0])
        sol = optx.minimise(objective, solver, x0, has_aux=True, max_steps=50)

        np.testing.assert_allclose(sol.value, [2.0, 2.0], rtol=1e-3)


class TestSLSQPBoundsComparisonWithSciPy:
    """Compare SLSQP-JAX bounds with SciPy's bounds parameter."""

    def test_bounds_match_scipy_simple(self):
        """Compare simple bounded optimization with SciPy."""

        def objective_scipy(x):
            return (x[0] - 5) ** 2 + (x[1] - 5) ** 2

        def objective_jax(x, args):
            return (x[0] - 5) ** 2 + (x[1] - 5) ** 2, None

        scipy_bounds = [(0, 3), (0, 3)]
        jax_bounds = jnp.array([[0.0, 3.0], [0.0, 3.0]])

        x0 = np.array([1.0, 1.0])

        result_scipy = scipy_minimize(
            objective_scipy,
            x0,
            method="SLSQP",
            bounds=scipy_bounds,
        )

        solver = SLSQP(rtol=1e-8, atol=1e-8, max_steps=50, bounds=jax_bounds)
        y, _ = _run_solver(solver, objective_jax, jnp.array(x0))

        np.testing.assert_allclose(y, result_scipy.x, rtol=1e-3)
        np.testing.assert_allclose(y, [3.0, 3.0], rtol=1e-3)

    def test_bounds_match_scipy_with_constraint(self):
        """Compare bounded + constrained optimization with SciPy."""

        def objective_scipy(x):
            return x[0] ** 2 + x[1] ** 2

        def constraint_scipy(x):
            return x[0] + x[1] - 4.0

        def objective_jax(x, args):
            return x[0] ** 2 + x[1] ** 2, None

        def constraint_jax(x, args):
            return jnp.array([x[0] + x[1] - 4.0])

        scipy_bounds = [(0, 3), (0, 3)]
        jax_bounds = jnp.array([[0.0, 3.0], [0.0, 3.0]])

        x0 = np.array([2.0, 2.0])

        result_scipy = scipy_minimize(
            objective_scipy,
            x0,
            method="SLSQP",
            bounds=scipy_bounds,
            constraints={"type": "eq", "fun": constraint_scipy},
        )

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            eq_constraint_fn=constraint_jax,
            n_eq_constraints=1,
            bounds=jax_bounds,
        )
        y, _ = _run_solver(solver, objective_jax, jnp.array(x0))

        np.testing.assert_allclose(y, result_scipy.x, rtol=1e-3)

    def test_bounds_match_scipy_partial(self):
        """Compare partially bounded optimization with SciPy."""

        def objective_scipy(x):
            return x[0] ** 2 + x[1] ** 2 + x[2] ** 2

        def objective_jax(x, args):
            return jnp.sum(x**2), None

        # x >= 1, y unrestricted, z <= -1
        scipy_bounds = [(1, None), (None, None), (None, -1)]
        jax_bounds = jnp.array(
            [
                [1.0, jnp.inf],
                [-jnp.inf, jnp.inf],
                [-jnp.inf, -1.0],
            ]
        )

        x0 = np.array([5.0, 5.0, -5.0])

        result_scipy = scipy_minimize(
            objective_scipy,
            x0,
            method="SLSQP",
            bounds=scipy_bounds,
        )

        solver = SLSQP(rtol=1e-8, atol=1e-8, max_steps=50, bounds=jax_bounds)
        y, _ = _run_solver(solver, objective_jax, jnp.array(x0))

        np.testing.assert_allclose(y, result_scipy.x, rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(y, [1.0, 0.0, -1.0], rtol=1e-3, atol=1e-6)


class TestFrozenHVP:
    """Tests verifying that user-supplied HVPs use frozen L-BFGS in the QP solver.

    When a user provides exact HVP functions, the solver should:
    1. Use the frozen L-BFGS approximation inside the QP inner loop.
    2. Call the exact HVP only once per main iteration (to probe the step
       direction for the L-BFGS secant update).
    """

    def test_hvp_call_count_unconstrained(self):
        """Verify exact HVP is called once per step, not per CG iteration."""
        call_count = {"n": 0}

        def objective(x, args):
            return jnp.sum(x**2), None

        def obj_hvp(x, v, args):
            call_count["n"] += 1
            return 2.0 * v

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=10,
            obj_hvp_fn=obj_hvp,
        )
        x0 = jnp.array([3.0, -2.0])

        state = solver.init(objective, x0, None, {}, None, None, frozenset())
        y = x0

        call_count["n"] = 0
        y, state, _ = solver.step(objective, y, None, {}, state, frozenset())

        # Exact HVP should be called exactly once per step (for the secant probe)
        assert call_count["n"] == 1, (
            f"Expected 1 exact HVP call per step, got {call_count['n']}"
        )

    def test_hvp_call_count_equality_constrained(self):
        """Verify exact HVP call count with equality constraints."""
        call_count = {"obj": 0, "eq": 0}

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] + x[2] - 3.0])

        def obj_hvp(x, v, args):
            call_count["obj"] += 1
            return 2.0 * v

        def eq_hvp(x, v, args):
            call_count["eq"] += 1
            return jnp.zeros((1, x.shape[0]))

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=10,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            obj_hvp_fn=obj_hvp,
            eq_hvp_fn=eq_hvp,
        )
        x0 = jnp.array([2.0, 0.5, 0.5])

        state = solver.init(objective, x0, None, {}, None, None, frozenset())
        y = x0

        call_count["obj"] = 0
        call_count["eq"] = 0
        y, state, _ = solver.step(objective, y, None, {}, state, frozenset())

        assert call_count["obj"] == 1, (
            f"Expected 1 obj HVP call per step, got {call_count['obj']}"
        )
        assert call_count["eq"] == 1, (
            f"Expected 1 eq HVP call per step, got {call_count['eq']}"
        )

    def test_frozen_hvp_gives_correct_solution(self):
        """Verify that using frozen L-BFGS with exact HVP probes converges.

        minimize x^2 + 2*y^2  s.t. x + y = 1
        Solution: x = 2/3, y = 1/3
        """

        def objective(x, args):
            return x[0] ** 2 + 2 * x[1] ** 2, None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 1.0])

        def obj_hvp(x, v, args):
            return jnp.array([2.0 * v[0], 4.0 * v[1]])

        def eq_hvp(x, v, args):
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
        x0 = jnp.array([0.0, 0.0])
        y, _ = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [2.0 / 3.0, 1.0 / 3.0], rtol=1e-4)

    def test_frozen_hvp_inequality_constrained(self):
        """Verify frozen HVP works with inequality constraints.

        minimize x^2 + y^2  s.t. x + y >= 2
        Solution: (1, 1)
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def ineq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 2.0])

        def obj_hvp(x, v, args):
            return 2.0 * v

        def ineq_hvp(x, v, args):
            return jnp.zeros((1, x.shape[0]))

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=1,
            obj_hvp_fn=obj_hvp,
            ineq_hvp_fn=ineq_hvp,
        )
        x0 = jnp.array([2.0, 2.0])
        y, _ = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [1.0, 1.0], rtol=1e-4)

    def test_frozen_hvp_multi_step_convergence(self):
        """Verify L-BFGS builds up curvature over multiple steps.

        The L-BFGS starts as the identity (B_0 = I). With exact HVP probes,
        it should gradually improve, and the solver should converge even
        for problems where the true Hessian differs from the identity.
        """

        def objective(x, args):
            return x[0] ** 2 + 10 * x[1] ** 2 + 100 * x[2] ** 2, None

        def obj_hvp(x, v, args):
            return jnp.array([2.0 * v[0], 20.0 * v[1], 200.0 * v[2]])

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            obj_hvp_fn=obj_hvp,
        )
        x0 = jnp.array([5.0, 5.0, 5.0])
        y, state = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [0.0, 0.0, 0.0], atol=1e-5)
        assert state.lbfgs_history.count > 0, "L-BFGS should have accumulated pairs"


class TestEarlyTerminationFix:
    """Tests for the premature termination fix.

    Verifies that the solver does not terminate prematurely when:
    1. The initial point exactly satisfies equality constraints.
    2. The objective gradient is small at the initial point.
    3. Multipliers haven't been computed yet (zero-initialized before fix).
    """

    def test_equality_on_constraint_surface(self):
        """The pathological case: initial point satisfies constraint, optimizer
        should still explore.

        minimize (x-3)^2 + (y+3)^2  s.t. x + y = 2
        x0 = (1, 1) satisfies x+y=2 but is not optimal.
        Solution: (4, -2) with f = 2 vs f(1,1) = 20.

        To verify this, substitute y = 2-x:
        minimize (x-3)^2 + (2-x+3)^2 = (x-3)^2 + (5-x)^2
        = x^2-6x+9 + 25-10x+x^2 = 2x^2 - 16x + 34
        d/dx = 4x - 16 = 0 -> x = 4, y = -2.
        f(4,-2) = 1+1 = 2.
        f(1,1) = 4+16 = 20. So the optimizer should move from 20 to 2.
        """

        def objective(x, args):
            return (x[0] - 3) ** 2 + (x[1] + 3) ** 2, None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 2.0])

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
        )
        x0 = jnp.array([1.0, 1.0])  # Satisfies x+y=2 but not optimal
        y, state = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [4.0, -2.0], rtol=1e-4)
        np.testing.assert_allclose(eq_constraint(y, None), 0.0, atol=1e-5)
        assert state.step_count > 0, "Solver should have taken at least one step"

    def test_small_gradient_on_constraint(self):
        """Pathological case with small objective gradient on constraint surface.

        minimize 1e-3 * ((x-10)^2 + y^2)  s.t. x + y = 2
        x0 = (1, 1) exactly on constraint.

        grad_f at (1,1) = 1e-3 * (-18, 2) which has norm ~= 1.81e-2.
        With zero multipliers, this would be the Lagrangian gradient,
        potentially triggering false convergence in a poorly calibrated solver.

        Optimal: substitute y=2-x -> minimize (x-10)^2 + (2-x)^2
        = x^2-20x+100 + x^2-4x+4 = 2x^2 - 24x + 104
        d/dx = 4x-24=0 -> x=6, y=-4.
        """

        def objective(x, args):
            return 1e-3 * ((x[0] - 10) ** 2 + x[1] ** 2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 2.0])

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=200,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
        )
        x0 = jnp.array([1.0, 1.0])
        y, state = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [6.0, -4.0], rtol=1e-3)
        assert state.step_count > 0, "Should not terminate at initial point"

    def test_min_steps_prevents_zero_step_convergence(self):
        """Verify min_steps parameter prevents convergence before any step."""

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 2.0])

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            min_steps=1,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
        )
        x0 = jnp.array([1.0, 1.0])

        state = solver.init(objective, x0, None, {}, None, None, frozenset())

        # Before any step, terminate should return False (min_steps=1)
        done, _ = solver.terminate(objective, x0, None, {}, state, frozenset())
        assert not done, "Should not terminate before min_steps"

    def test_min_steps_zero_allows_immediate_convergence(self):
        """Verify min_steps=0 allows convergence at step 0 if KKT conditions hold.

        This tests that the min_steps parameter actually controls the behavior.
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        solver = SLSQP(
            rtol=1.0,  # Very loose tolerance
            atol=1.0,
            max_steps=50,
            min_steps=0,
        )
        x0 = jnp.array([0.1, 0.1])

        state = solver.init(objective, x0, None, {}, None, None, frozenset())
        done, _ = solver.terminate(objective, x0, None, {}, state, frozenset())

        # With very loose tolerance and min_steps=0, should converge immediately
        assert done, "With min_steps=0 and loose tol, should converge at step 0"

    def test_initial_multipliers_not_zero(self):
        """Verify initial equality multipliers are computed via least-squares."""

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 2.0])

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
        )
        x0 = jnp.array([1.0, 1.0])

        state = solver.init(objective, x0, None, {}, None, None, frozenset())

        # Initial multipliers should not be zero (least-squares estimate)
        assert not jnp.allclose(state.multipliers_eq, 0.0), (
            "Initial eq multipliers should be estimated via least-squares, not zero"
        )

    def test_feasible_start_nonlinear_constraint(self):
        """Test with nonlinear equality constraint satisfied at start.

        minimize x^2 + y^2  s.t. x^2 + y^2 = 2
        x0 = (1, 1) satisfies constraint.
        Solution: any point on the circle with radius sqrt(2), but with min norm.
        Since the objective IS the constraint, f* = 2.
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] ** 2 + x[1] ** 2 - 2.0])

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
        )
        x0 = jnp.array([1.0, 1.0])
        y, state = _run_solver(solver, objective, x0)

        # The constraint should be satisfied
        np.testing.assert_allclose(eq_constraint(y, None), 0.0, atol=1e-5)
        # Solution should be on the sphere
        np.testing.assert_allclose(jnp.sum(y**2), 2.0, rtol=1e-4)
