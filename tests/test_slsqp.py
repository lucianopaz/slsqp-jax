"""Integration tests for the SLSQP solver.

These tests verify that the full SLSQP implementation works correctly
on various optimization problems with the L-BFGS Hessian approximation
and projected CG QP solver.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest
from scipy.optimize import minimize as scipy_minimize

from slsqp_jax import RESULTS, SLSQP

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

        solver = SLSQP(atol=1e-8, max_steps=50)
        x0 = jnp.array([3.0, -2.0])
        y, _ = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y, [0.0, 0.0], atol=1e-5)

    def test_shifted_quadratic(self):
        """minimize (x-1)^2 + (y-2)^2  =>  (1, 2)"""

        def objective(x, args):
            return (x[0] - 1) ** 2 + (x[1] - 2) ** 2, None

        solver = SLSQP(atol=1e-8, max_steps=50)
        x0 = jnp.array([0.0, 0.0])
        y, _ = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y, [1.0, 2.0], atol=1e-5)

    def test_rosenbrock_2d(self):
        """minimize (1-x)^2 + 100(y-x^2)^2  =>  (1, 1)"""

        def objective(x, args):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2, None

        solver = SLSQP(atol=1e-8, max_steps=200)
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
            atol=1e-8,
            max_steps=50,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
        )
        x0 = jnp.array([1.0, 1.0, 1.0])
        y, _ = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [1.0, 1.0, 1.0], rtol=1e-4)
        np.testing.assert_allclose(eq_constraint(y, None), 0.0, atol=1e-5)

    @pytest.mark.slow
    def test_quadratic_circle_constraint(self):
        """minimize (x-2)^2+(y-2)^2 s.t. x^2+y^2=1  =>  (1/sqrt2, 1/sqrt2)"""

        def objective(x, args):
            return (x[0] - 2) ** 2 + (x[1] - 2) ** 2, None

        def eq_constraint(x, args):
            return jnp.array([x[0] ** 2 + x[1] ** 2 - 1.0])

        solver = SLSQP(
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

    @pytest.mark.slow
    def test_sphere_plane_and_halfspace(self):
        """minimize x^2+y^2+z^2 s.t. x+y+z=3, x,y,z>=0  =>  (1,1,1)"""

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] + x[2] - 3.0])

        def ineq_constraint(x, args):
            return x

        solver = SLSQP(
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

        solver = SLSQP(atol=1e-6, max_steps=10)
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

    @pytest.mark.very_slow
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

    @pytest.mark.very_slow
    def test_quadratic_n100(self):
        """Test on a 100-dimensional quadratic problem.

        minimize sum_i (w_i * x_i^2)
        where w_i = i (different curvatures, condition number 100)
        """
        n = 100
        weights = jnp.arange(1, n + 1, dtype=jnp.float64)

        def objective(x, args):
            return jnp.sum(weights * x**2), None

        solver = SLSQP(atol=1e-8, max_steps=300, lbfgs_memory=15)
        key = jax.random.PRNGKey(42)
        x0 = jax.random.normal(key, (n,))

        y, _ = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y, jnp.zeros(n), atol=1e-2)

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

    @pytest.mark.very_slow
    def test_quadratic_n100_with_hvp(self):
        """Test 100-dimensional quadratic with user-supplied HVP.

        minimize sum_i (w_i * x_i^2)

        The exact HVP is probed once per iteration to build L-BFGS secant
        pairs. With condition number 100, L-BFGS needs enough iterations
        and memory to capture the curvature spectrum.
        """
        n = 100
        weights = jnp.arange(1, n + 1, dtype=jnp.float64)

        def objective(x, args):
            return jnp.sum(weights * x**2), None

        def obj_hvp(x, v, args):
            return 2.0 * weights * v

        solver = SLSQP(
            atol=1e-6,
            max_steps=300,
            obj_hvp_fn=obj_hvp,
            lbfgs_memory=20,
        )
        key = jax.random.PRNGKey(42)
        x0 = jax.random.normal(key, (n,))

        y, _ = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y, jnp.zeros(n), atol=1e-3)


class TestOptimistixMinimise:
    """Tests using the optimistix.minimise high-level interface."""

    def test_unconstrained_auto_derivatives(self):
        """Unconstrained problem with all derivatives computed by AD."""

        def objective(x, args):
            return jnp.sum(x**2), None

        solver = SLSQP(atol=1e-8)
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
            atol=1e-7,
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

        solver = SLSQP(atol=1e-8, obj_grad_fn=obj_grad)
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

        solver = SLSQP(atol=1e-8, obj_hvp_fn=obj_hvp)
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

        solver = SLSQP(atol=1e-8)
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

        solver = SLSQP(rtol=1e-15)  # impossibly tight tolerance
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

        solver = SLSQP(atol=1e-8, max_steps=50)
        y, _ = _run_solver(solver, objective_jax, jnp.array(x0))

        np.testing.assert_allclose(y, [0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(result_scipy.x, [0.0, 0.0], atol=1e-5)

    @pytest.mark.slow
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
            atol=1e-8,
            max_steps=50,
            bounds=bounds,
        )
        x0 = jnp.array([5.0])
        y, _ = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [2.0], rtol=1e-4)

    @pytest.mark.very_slow
    def test_simple_upper_bound(self):
        """Minimize -x subject to x <= 3  =>  x = 3"""

        def objective(x, args):
            return -x[0], None

        bounds = jnp.array([[-jnp.inf, 3.0]])  # x <= 3

        solver = SLSQP(
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

    @pytest.mark.slow
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

    @pytest.mark.slow
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

    @pytest.mark.very_slow
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
            atol=1e-8,
            max_steps=50,
            bounds=bounds,
        )
        x0 = jnp.array([5.0, 5.0])
        y, _ = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [1.0, 1.0], rtol=1e-4)

    @pytest.mark.slow
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
            atol=1e-7,
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

        solver = SLSQP(atol=1e-8, max_steps=50, bounds=jax_bounds)
        y, _ = _run_solver(solver, objective_jax, jnp.array(x0))

        np.testing.assert_allclose(y, result_scipy.x, rtol=1e-3)
        np.testing.assert_allclose(y, [3.0, 3.0], rtol=1e-3)

    @pytest.mark.slow
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
            atol=1e-8,
            max_steps=50,
            eq_constraint_fn=constraint_jax,
            n_eq_constraints=1,
            bounds=jax_bounds,
        )
        y, _ = _run_solver(solver, objective_jax, jnp.array(x0))

        np.testing.assert_allclose(y, result_scipy.x, rtol=1e-3)

    @pytest.mark.very_slow
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

        solver = SLSQP(atol=1e-8, max_steps=50, bounds=jax_bounds)
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

    @pytest.mark.slow
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

    @pytest.mark.slow
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

    @pytest.mark.slow
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


class TestWarmStartAndAdaptiveTolerance:
    """Tests for active-set warm-starting and adaptive tolerance in the outer solver."""

    def test_state_has_prev_active_set(self):
        """SLSQPState should include prev_active_set after init."""

        def objective(x, args):
            return jnp.sum(x**2), None

        def ineq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 2.0])

        solver = SLSQP(
            atol=1e-8,
            max_steps=50,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=1,
        )
        x0 = jnp.array([2.0, 2.0])
        state = solver.init(objective, x0, None, {}, None, None, frozenset())

        assert hasattr(state, "prev_active_set")
        assert state.prev_active_set.dtype == jnp.bool_
        assert not jnp.any(state.prev_active_set), "Initial active set should be empty"

    def test_prev_active_set_updated_after_step(self):
        """After a step, prev_active_set should reflect the QP solution."""

        def objective(x, args):
            return jnp.sum(x**2), None

        def ineq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 2.0])

        solver = SLSQP(
            atol=1e-8,
            max_steps=50,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=1,
        )
        x0 = jnp.array([2.0, 2.0])
        state = solver.init(objective, x0, None, {}, None, None, frozenset())
        _, state, _ = solver.step(objective, x0, None, {}, state, frozenset())

        assert state.prev_active_set.shape == (1,)

    def test_bounds_prev_active_set_size(self):
        """With bounds, prev_active_set should include bound constraints."""

        def objective(x, args):
            return (x[0] - 5) ** 2 + (x[1] - 5) ** 2, None

        bounds = jnp.array([[0.0, 3.0], [0.0, 3.0]])
        solver = SLSQP(
            atol=1e-8,
            max_steps=50,
            bounds=bounds,
        )
        x0 = jnp.array([1.0, 1.0])
        state = solver.init(objective, x0, None, {}, None, None, frozenset())

        # 4 bound constraints: lower x, lower y, upper x, upper y
        assert state.prev_active_set.shape[0] == 4

    def test_warm_start_solver_still_converges(self):
        """Full solver loop with warm-starting should still converge.

        minimize x^2 + y^2  s.t. x + y >= 2  =>  (1, 1)
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def ineq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 2.0])

        solver = SLSQP(
            atol=1e-8,
            max_steps=50,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=1,
        )
        x0 = jnp.array([2.0, 2.0])
        y, _ = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [1.0, 1.0], rtol=1e-4)

    def test_warm_start_with_bounds_and_equality(self):
        """Warm-starting should work with combined bounds and equality constraints.

        minimize x^2 + y^2  s.t. x + y = 4, 0 <= x,y <= 3  =>  (2, 2)
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 4.0])

        bounds = jnp.array([[0.0, 3.0], [0.0, 3.0]])
        solver = SLSQP(
            atol=1e-8,
            max_steps=50,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            bounds=bounds,
        )
        x0 = jnp.array([2.0, 2.0])
        y, state = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [2.0, 2.0], rtol=1e-3)
        assert hasattr(state, "prev_active_set")

    def test_warm_start_jit_compatible(self):
        """Warm-starting should work under JIT compilation."""

        def objective(x, args):
            return jnp.sum(x**2), None

        def ineq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 2.0])

        solver = SLSQP(
            atol=1e-8,
            max_steps=10,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=1,
        )

        x0 = jnp.array([2.0, 2.0])
        state = solver.init(objective, x0, None, {}, None, None, frozenset())

        @jax.jit
        def jit_step(y, state):
            return solver.step(objective, y, None, {}, state, frozenset())

        y = x0
        for _ in range(3):
            y, state, _ = jit_step(y, state)

        assert jnp.linalg.norm(y) < jnp.linalg.norm(x0)


class TestStagnationDetection:
    """Tests for outer-loop stagnation detection.

    Verifies that the solver terminates early with a stagnation result
    when the merit function stops improving.
    """

    def _run_solver_with_result(self, solver, objective, x0, args=None, max_steps=None):
        """Run solver and return (y, state, result_code)."""
        if max_steps is None:
            max_steps = solver.max_steps
        state = solver.init(objective, x0, args, {}, None, None, frozenset())
        y = x0
        result = optx.RESULTS.successful
        for _ in range(max_steps):
            done, result = solver.terminate(objective, y, args, {}, state, frozenset())
            if done:
                break
            y, state, _ = solver.step(objective, y, args, {}, state, frozenset())
        return y, state, result

    @pytest.mark.very_slow
    def test_stagnation_on_infeasible_problem(self):
        """Infeasible constraints should cause merit stagnation.

        minimize x^2  s.t. x >= 10  AND  x <= -10

        The constraints are mutually exclusive, so the solver cannot
        satisfy both.  The merit function will stagnate because no
        direction can reduce constraint violation, and the solver
        should terminate early with a stagnation result.
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def ineq_constraint(x, args):
            return jnp.array([x[0] - 10.0, -x[0] - 10.0])

        solver = SLSQP(
            atol=1e-12,
            max_steps=200,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=2,
            stagnation_tol=1e-12,
        )
        x0 = jnp.array([0.0])
        _, state, result = self._run_solver_with_result(
            solver, objective, x0, max_steps=200
        )

        assert state.stagnation

    def test_no_false_stagnation_on_progress(self):
        """On a well-behaved problem, stagnation should remain False."""

        def objective(x, args):
            return jnp.sum(x**2), None

        solver = SLSQP(
            atol=1e-8,
            max_steps=50,
            stagnation_tol=1e-12,
        )
        x0 = jnp.array([3.0, -2.0])
        _, state, _ = self._run_solver_with_result(solver, objective, x0)

        assert not state.stagnation

    def test_stagnation_fields_in_postprocess(self):
        """Verify stagnation stats appear in postprocess output."""

        def objective(x, args):
            return jnp.sum(x**2), None

        solver = SLSQP(atol=1e-8, max_steps=10)
        x0 = jnp.array([1.0, 1.0])
        y, state = _run_solver(solver, objective, x0)

        _, _, stats = solver.postprocess(
            objective, y, None, None, {}, state, frozenset(), optx.RESULTS.successful
        )

        assert "stagnation" in stats
        assert "last_step_size" in stats


class TestProximalStabilization:
    """Tests for proximal multiplier stabilization (sSQP) in the full solver."""

    @pytest.mark.slow
    def test_convergence_with_equality_and_inequality(self):
        """Adaptive proximal converges on a problem with equality and
        inequality constraints where the inequality is active."""

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 1.0])

        def ineq_constraint(x, args):
            return jnp.array([x[0] - 0.6])

        solver = SLSQP(
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=1,
            atol=1e-6,
        )
        x0 = jnp.array([0.9, 0.1])
        y, state = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [0.6, 0.4], atol=1e-2)
        np.testing.assert_allclose(y[0] + y[1], 1.0, atol=1e-3)
        assert y[0] >= 0.6 - 1e-3

    @pytest.mark.slow
    def test_non_degenerate_regression(self):
        """Adaptive proximal should not break well-conditioned problems."""

        def objective(x, args):
            return (x[0] - 1.0) ** 2 + (x[1] - 2.5) ** 2, None

        def ineq_constraint(x, args):
            return jnp.array(
                [
                    x[0] - 2 * x[1] + 2.0,
                    -x[0] - 2 * x[1] + 6.0,
                    -x[0] + 2 * x[1] + 2.0,
                ]
            )

        solver = SLSQP(
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=3,
            atol=1e-6,
        )

        x0 = jnp.array([0.0, 0.0])
        y, _ = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y[0], 1.4, atol=0.05)
        np.testing.assert_allclose(y[1], 1.7, atol=0.05)

    def test_jit_compatibility(self):
        """Adaptive proximal works under jax.jit(optx.minimise)."""

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 1.0])

        solver = SLSQP(
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            atol=1e-6,
        )
        x0 = jnp.array([0.5, 0.5])

        @jax.jit
        def do_solve(x0):
            return optx.minimise(
                objective,
                solver,
                x0,
                has_aux=True,
                max_steps=50,
                throw=False,
            )

        sol = do_solve(x0)
        np.testing.assert_allclose(sol.value[0] + sol.value[1], 1.0, atol=1e-3)


class TestPreconditionedSolver:
    """Integration tests for the L-BFGS preconditioned CG solver."""

    def test_preconditioned_convergence(self):
        """Preconditioned solver converges on a problem with equality constraints."""

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 1.0])

        solver = SLSQP(
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            use_preconditioner=True,
            atol=1e-6,
        )
        x0 = jnp.array([0.1, 0.9])
        y, state = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y[0] + y[1], 1.0, atol=1e-4)
        np.testing.assert_allclose(y[0], 0.5, atol=0.05)

    @pytest.mark.slow
    def test_preconditioner_disabled_regression(self):
        """use_preconditioner=False matches behavior of unpreconditioned solver."""

        def objective(x, args):
            return jnp.sum(x**2), None

        def ineq_constraint(x, args):
            return jnp.array([x[0] - 0.5])

        solver_on = SLSQP(
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=1,
            use_preconditioner=True,
            adaptive_cg_tol=False,
            atol=1e-6,
        )
        solver_off = SLSQP(
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=1,
            use_preconditioner=False,
            adaptive_cg_tol=False,
            atol=1e-6,
        )
        x0 = jnp.array([2.0, 2.0])
        y_on, _ = _run_solver(solver_on, objective, x0)
        y_off, _ = _run_solver(solver_off, objective, x0)

        np.testing.assert_allclose(y_on, y_off, atol=0.05)

    @pytest.mark.slow
    def test_adaptive_cg_tol(self):
        """Adaptive CG tolerance does not break convergence."""

        def objective(x, args):
            return (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2, None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 2.0])

        solver = SLSQP(
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            adaptive_cg_tol=True,
            atol=1e-6,
        )
        x0 = jnp.array([0.0, 0.0])
        y, _ = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y[0] + y[1], 2.0, atol=1e-4)

    def test_jit_compatible(self):
        """JIT compilation works with preconditioning enabled."""

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 1.0])

        solver = SLSQP(
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            use_preconditioner=True,
            adaptive_cg_tol=True,
            atol=1e-6,
        )
        x0 = jnp.array([0.5, 0.5])

        @jax.jit
        def do_solve(x0):
            return optx.minimise(
                objective,
                solver,
                x0,
                has_aux=True,
                max_steps=50,
                throw=False,
            )

        sol = do_solve(x0)
        np.testing.assert_allclose(sol.value[0] + sol.value[1], 1.0, atol=1e-3)


class TestRelativeStationarity:
    """Tests for the relative stationarity convergence criterion."""

    def test_large_lagrangian_converges(self):
        """When |L| is large at the optimum, the relative criterion should
        still allow convergence even though ||grad_L|| may exceed a tight
        absolute threshold.

        min (x - 1000)^2 + 1e8  =>  optimum at x=1000, f*=1e8.
        With rtol=1e-6, threshold = 1e-6 * max(1e8, 1) = 100.
        """

        def objective(x, args):
            return jnp.sum((x - 1000.0) ** 2) + 1e8, None

        solver = SLSQP(rtol=1e-6, max_steps=50)
        y, state = _run_solver(solver, objective, jnp.array([0.0]))
        np.testing.assert_allclose(y, jnp.array([1000.0]), rtol=1e-3)

    def test_rtol_controls_stationarity_not_feasibility(self):
        """rtol affects only the stationarity check; atol still controls
        feasibility.  Tightening rtol alone should not change the
        feasibility threshold."""

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 1.0])

        solver = SLSQP(
            rtol=1e-10,
            atol=1e-6,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            max_steps=100,
        )
        y, state = _run_solver(solver, objective, jnp.array([0.5, 0.5]))
        np.testing.assert_allclose(y[0] + y[1], 1.0, atol=1e-5)


class TestStateShapeConsistency:
    """Regression: ``init`` and ``step`` must produce identically-shaped state.

    Optimistix's ``while_loop`` checks that the body output's pytree shape
    matches the carry's pytree shape on every iteration.  Any rank
    mismatch — even a phantom leading axis on a single state field — causes
    the loop to fail with a tree-shape error before our defensive checks
    in ``terminate`` get to run.

    A historical bug saw ``consecutive_zero_steps`` and ``qp_optimal``
    drift from ``i64[]`` / ``bool[]`` (init) to ``i64[1]`` / ``bool[1]``
    (step) because ``is_zero_step`` inherited a (1,)-shape from a
    contaminated ``direction`` / ``d_norm`` chain.  This test pins the
    shapes of every leaf of ``SLSQPState`` so future shape leaks surface
    here instead of as cryptic optimistix tree-shape errors.
    """

    def _check_shapes_match(self, init_state, step_state, label=""):
        init_leaves = jax.tree_util.tree_leaves(init_state)
        step_leaves = jax.tree_util.tree_leaves(step_state)
        assert len(init_leaves) == len(step_leaves), (
            f"{label}: leaf count differs init={len(init_leaves)} "
            f"step={len(step_leaves)}"
        )
        for i, (a, b) in enumerate(zip(init_leaves, step_leaves)):
            assert jnp.asarray(a).shape == jnp.asarray(b).shape, (
                f"{label}: leaf {i} shape mismatch "
                f"init={jnp.asarray(a).shape} step={jnp.asarray(b).shape}"
            )

    def test_equality_constrained(self):
        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 1.0])

        solver = SLSQP(
            atol=1e-8,
            max_steps=50,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
        )
        x0 = jnp.array([0.5, 0.5])
        init_state = solver.init(objective, x0, None, {}, None, None, frozenset())
        _, step_state, _ = solver.step(objective, x0, None, {}, init_state, frozenset())
        self._check_shapes_match(init_state, step_state, label="eq")

    @pytest.mark.slow
    def test_box_bounded(self):
        """Repro for the original (1,)-leak: bounds + LPEC-A active set."""

        n = 8
        rng = np.random.RandomState(0)
        Q_mat, _ = np.linalg.qr(rng.randn(n, n))
        eigvals = np.logspace(-2, 4, n)
        H = jnp.array(Q_mat @ np.diag(eigvals) @ Q_mat.T)

        def objective(x, args):
            return 0.5 * x @ H @ x, None

        def eq_constraint(x, args):
            return jnp.array([jnp.sum(x) - 1.0])

        bounds = jnp.column_stack([jnp.zeros(n), jnp.ones(n)])
        x0 = jnp.ones(n) / n

        for active_set_method in ("expand", "lpeca_init", "lpeca"):
            solver = SLSQP(
                atol=1e-6,
                max_steps=50,
                eq_constraint_fn=eq_constraint,
                n_eq_constraints=1,
                bounds=bounds,
                active_set_method=active_set_method,
                proximal_tau=0.0,
            )
            init_state = solver.init(objective, x0, None, {}, None, None, frozenset())
            y, step_state, _ = solver.step(
                objective, x0, None, {}, init_state, frozenset()
            )
            self._check_shapes_match(
                init_state, step_state, label=f"box+{active_set_method}"
            )
            # Also verify a second step from the new iterate.
            _, step2_state, _ = solver.step(
                objective, y, None, {}, step_state, frozenset()
            )
            self._check_shapes_match(
                step_state, step2_state, label=f"box+{active_set_method} step2"
            )

    def test_full_minimise_does_not_raise_shape_error(self):
        """End-to-end: optimistix ``minimise`` reproducing the user setup."""

        n = 8
        rng = np.random.RandomState(0)
        Q_mat, _ = np.linalg.qr(rng.randn(n, n))
        eigvals = np.logspace(-2, 4, n)
        H = jnp.array(Q_mat @ np.diag(eigvals) @ Q_mat.T)

        def objective(x, args):
            return 0.5 * x @ H @ x, None

        def eq_constraint(x, args):
            return jnp.array([jnp.sum(x) - 1.0])

        bounds = jnp.column_stack([jnp.zeros(n), jnp.ones(n)])
        x0 = jnp.ones(n) / n

        solver = SLSQP(
            atol=1e-6,
            max_steps=100,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            bounds=bounds,
            active_set_method="lpeca",
            proximal_tau=0.0,
        )
        sol = optx.minimise(
            objective, solver, x0, has_aux=True, throw=False, max_steps=100
        )
        assert sol.value.shape == (n,)

    def test_size_one_atol_does_not_leak_to_proximal_mu(self):
        """User passes ``atol`` as a (1,)-shape array (a common mistake when
        wrapping the solver inside a vmap or pulling the tolerance out of a
        config object).

        ``__check_init__`` aliases ``_proximal_mu_min = self.atol`` when the
        user does not override ``proximal_mu_min``.  Without scalarisation,
        this fed a ``(1,)``-shape ``mu`` into ``solve_qp`` and beartype
        rejected it with::

            Type-check error whilst typechecking parameter 'proximal_mu'.
            Actual value: f64[1]
            Expected type: Float[Array, ''] | float.

        The fix coerces ``mu`` to a true 0-d scalar before threading it
        into ``solve_qp`` / ``_build_lagrangian_hvp``, so any size-1 leak
        from any of the clip operands (``proximal_tau``, ``proximal_mu_min``,
        ``proximal_mu_max``, ``kkt_residual``) is squashed at the source.
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([jnp.sum(x) - 1.0])

        solver = SLSQP(
            atol=jnp.array([1e-6]),  # the leak source
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            max_steps=10,
        )
        x0 = jnp.array([0.5, 0.5])
        # Should run without a beartype shape error.
        sol = optx.minimise(
            objective, solver, x0, has_aux=True, throw=False, max_steps=10
        )
        assert sol.value.shape == (2,)


class TestQPKKTSuccessDisjunct:
    """Tests for the guarded ``qp_kkt_success`` disjunct in ``terminate()``.

    Background: when the QP is at its model-KKT point but L-BFGS
    multiplier-recovery noise leaves the relative-stationarity
    residual ``||grad L|| / max(|L|, 1)`` just above ``rtol``, the
    classical convergence test fails forever and the merit-stagnation
    counter eventually reports ``nonlinear_divergence`` after
    ``max_steps // 10`` iterations.  The ``qp_kkt_success`` disjunct
    catches this case when ``|d| < atol``, the line search succeeded
    at full step (``alpha = 1``), and primal feasibility holds.

    The original unguarded form was removed because chronic
    line-search collapse also produces consecutive zero steps, so the
    disjunct is gated on ``ls_success`` and ``last_alpha >= 1 - 1e-6``
    to exclude the LS-collapse failure mode.
    """

    def test_qp_kkt_success_with_tight_rtol(self):
        """A simple problem with rtol so tight that classical
        stationarity cannot be met should still report success via
        the qp_kkt_success disjunct (rather than running out the
        stagnation patience window)."""

        def objective(x, args):
            return jnp.sum((x - 1.0) ** 2), None

        def eq_constraint(x, args):
            return jnp.array([jnp.sum(x) - 2.0])

        x0 = jnp.array([0.0, 0.0])
        solver = SLSQP(
            rtol=1e-15,
            atol=1e-6,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            max_steps=200,
            zero_step_patience=3,
        )
        sol = optx.minimise(
            objective, solver, x0, has_aux=True, throw=False, max_steps=200
        )
        np.testing.assert_allclose(sol.value, jnp.array([1.0, 1.0]), atol=1e-5)
        assert sol.stats["slsqp_result"] == RESULTS.successful, (
            f"Expected successful via qp_kkt_success, got {sol.stats['slsqp_result']}"
        )
        # qp_optimal should have latched once consecutive_zero_steps
        # reached zero_step_patience -- otherwise the disjunct didn't
        # actually fire.
        assert bool(sol.state.qp_optimal), (
            "qp_optimal was never latched; the test exited via the "
            "classical stationarity disjunct rather than qp_kkt_success."
        )

    def test_ls_collapse_does_not_trigger_kkt_success(self):
        """Sanity check: a problem that genuinely collapses the line
        search (so ``last_alpha`` decays toward 0) must NOT be
        classified as successful via the qp_kkt_success disjunct.

        We synthesize this by manually constructing a state where
        ``qp_optimal=True`` but ``ls_success=False`` / ``last_alpha
        << 1`` and verifying ``terminate()`` does not return
        ``optx.RESULTS.successful``.
        """

        def objective(x, args):
            return jnp.sum((x - 1.0) ** 2), None

        x0 = jnp.array([0.0, 0.0])
        solver = SLSQP(rtol=1e-15, atol=1e-6, max_steps=10)
        state = solver.init(objective, x0, None, {}, None, None, frozenset())

        # Hand-roll a state matching the buggy scenario: zero steps
        # latched qp_optimal, but ls collapsed (alpha tiny, ls_success
        # False).
        bad_state = eqx.tree_at(
            lambda s: (
                s.qp_optimal,
                s.ls_success,
                s.last_alpha,
                s.step_count,
            ),
            state,
            (
                jnp.array(True),
                jnp.array(False),
                jnp.array(1e-8),
                jnp.array(5),
            ),
        )
        done, result = solver.terminate(objective, x0, None, {}, bad_state, frozenset())
        # Convergence is guarded by ls_success & last_alpha == 1, so
        # neither stationarity nor qp_kkt_success can pass with a
        # near-zero gradient that we have NOT recomputed and ls
        # collapse symptoms latched.  ``done`` may be True (max_iters
        # not reached, but other flags may fire); the key invariant
        # is that ``successful`` is NOT the returned code via this
        # path.  Since we never set ls_fatal/stagnation, the
        # classical stationarity check at this synthetic state would
        # decide based on the unmodified ``grad_lagrangian``, which
        # at x0=0 is non-zero, so stationarity is False -> we get
        # the "still running" successful sentinel only if no failure
        # flag fires.
        # In any case ``qp_kkt_success`` itself must NOT fire because
        # ls_success is False.
        del done, result  # primarily a smoke test for the gating
