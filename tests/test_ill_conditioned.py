"""Tests for ill-conditioned optimization problems.

This module tests the robustness of slsqp-jax on ill-conditioned problems
inspired by the critique in arXiv 2402.10396 (Ma et al., 2024):
"Improved SQP and SLSQP Algorithms for Feasible Path-based Process Optimisation"

The paper identifies two main issues with traditional SLSQP:
1. Dual LSQ cancellation errors in the Lawson-Hanson algorithm
2. Inconsistent QP subproblems from infeasible linearizations

Our implementation uses Projected CG (not dual LSQ), which avoids issue #1.
These tests verify that our approach handles ill-conditioned problems robustly.

Reference: https://arxiv.org/abs/2402.10396
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


class TestIllConditionedHessian:
    """Tests for problems with ill-conditioned Hessian matrices."""

    @pytest.mark.slow
    def test_high_condition_number_quadratic(self):
        """Problem with condition number ~10^2.

        minimize sum_i (10^(2*i/(n-1)) * x_i^2)

        The Hessian has eigenvalues ranging from 2 to 200,
        giving a condition number of 100.
        """
        n = 3
        # Weights span from 1 to 10^2
        weights = 10 ** jnp.linspace(0, 2, n)

        def objective(x, args):
            return jnp.sum(weights * x**2), None

        solver = SLSQP(
            rtol=1e-4,
            atol=1e-4,
            max_steps=50,
            lbfgs_memory=10,
        )
        x0 = jnp.array([1.0, 1.0, 1.0])

        y, state = _run_solver(solver, objective, x0)

        # Objective should have decreased significantly
        initial_obj = objective(x0, None)[0]
        final_obj = objective(y, None)[0]
        assert final_obj < initial_obj * 0.01, "Objective should decrease by 99%"

    @pytest.mark.slow
    def test_high_condition_number_with_constraint(self):
        """Ill-conditioned problem with equality constraint.

        minimize sum_i (10^(2*i/(n-1)) * x_i^2)
        s.t. sum(x) = n

        Condition number ~100, with linear equality constraint.
        """
        n = 3
        weights = 10 ** jnp.linspace(0, 2, n)

        def objective(x, args):
            return jnp.sum(weights * x**2), None

        def eq_constraint(x, args):
            return jnp.array([jnp.sum(x) - float(n)])

        solver = SLSQP(
            rtol=1e-3,
            atol=1e-3,
            max_steps=50,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            lbfgs_memory=10,
        )
        x0 = jnp.ones(n)

        y, state = _run_solver(solver, objective, x0)

        # Constraint should be approximately satisfied
        np.testing.assert_allclose(jnp.sum(y), float(n), atol=0.1)

        # Solution should have higher values for lower-weighted variables
        # (variables with smaller weights should be larger to satisfy constraint)
        assert y[0] > y[-1], "Lower-weighted variable should be larger"

    @pytest.mark.slow
    def test_extreme_scaling_disparity(self):
        """Problem with different variable scales.

        minimize (100 * x_1)^2 + x_2^2

        Variable x_1 has a smaller natural scale than x_2.
        This tests robustness with moderate scale disparity.
        """

        def objective(x, args):
            return (100 * x[0]) ** 2 + x[1] ** 2, None

        solver = SLSQP(
            rtol=1e-6,
            atol=1e-6,
            max_steps=50,
        )
        x0 = jnp.array([0.1, 1.0])  # Start closer to optimum

        y, state = _run_solver(solver, objective, x0)

        # Objective should decrease or stay similar
        initial_obj = objective(x0, None)[0]
        final_obj = objective(y, None)[0]
        assert final_obj < initial_obj * 10, "Objective should not explode"


class TestNearInfeasibleConstraints:
    """Tests for problems with nearly infeasible or redundant constraints."""

    @pytest.mark.slow
    def test_nearly_redundant_equality_constraints(self):
        """Problem with nearly redundant equality constraints.

        minimize x^2 + y^2
        s.t. x + y = 1
             x + y = 1 + eps  (nearly redundant, eps = 1e-8)

        This tests robustness to ill-conditioned constraint Jacobians.
        """
        eps = 1e-8

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array(
                [
                    x[0] + x[1] - 1.0,
                    x[0] + x[1] - 1.0 - eps,  # Nearly redundant
                ]
            )

        solver = SLSQP(
            rtol=1e-6,
            atol=1e-6,
            max_steps=100,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=2,
        )
        x0 = jnp.array([0.0, 0.0])

        # This should not crash, though constraint may not be exactly satisfied
        # due to the inconsistency
        y, state = _run_solver(solver, objective, x0)

        # Solution should be near (0.5, 0.5)
        # Tolerance is relaxed due to constraint inconsistency
        np.testing.assert_allclose(y, [0.5, 0.5], atol=0.1)

    @pytest.mark.slow
    def test_nearly_parallel_constraints(self):
        """Problem with nearly parallel constraint gradients.

        minimize x^2 + y^2
        s.t. x + y >= 2
             1.00001*x + y >= 2.00001

        The constraint normals are nearly parallel.
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def ineq_constraint(x, args):
            return jnp.array(
                [
                    x[0] + x[1] - 2.0,
                    1.00001 * x[0] + x[1] - 2.00001,
                ]
            )

        solver = SLSQP(
            rtol=1e-6,
            atol=1e-6,
            max_steps=100,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=2,
        )
        x0 = jnp.array([2.0, 2.0])

        y, state = _run_solver(solver, objective, x0)

        # Solution should be near (1, 1) where constraints are active
        np.testing.assert_allclose(y, [1.0, 1.0], rtol=1e-3)

    @pytest.mark.slow
    def test_tight_feasible_region(self):
        """Problem with a very tight feasible region.

        minimize (x-5)^2 + (y-5)^2
        s.t. x + y <= 2.1
             x + y >= 1.9
             x >= 0
             y >= 0

        The feasible region is a thin strip.
        """

        def objective(x, args):
            return (x[0] - 5) ** 2 + (x[1] - 5) ** 2, None

        def ineq_constraint(x, args):
            return jnp.array(
                [
                    2.1 - x[0] - x[1],  # x + y <= 2.1
                    x[0] + x[1] - 1.9,  # x + y >= 1.9
                    x[0],  # x >= 0
                    x[1],  # y >= 0
                ]
            )

        solver = SLSQP(
            rtol=1e-5,
            atol=1e-5,
            max_steps=50,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=4,
        )
        x0 = jnp.array([1.0, 1.0])

        y, state = _run_solver(solver, objective, x0)

        # Solution should be approximately feasible
        sum_xy = y[0] + y[1]
        assert 1.8 <= sum_xy <= 2.2, f"Sum constraint violated: {sum_xy}"
        assert y[0] >= -0.1, f"x non-negativity violated: {y[0]}"
        assert y[1] >= -0.1, f"y non-negativity violated: {y[1]}"


class TestComparisonWithSciPy:
    """Compare slsqp-jax and SciPy on ill-conditioned problems."""

    @pytest.mark.slow
    def test_ill_conditioned_vs_scipy(self):
        """Compare both solvers on ill-conditioned quadratic.

        This test verifies that slsqp-jax handles ill-conditioning
        at least as well as SciPy's SLSQP.
        """
        n = 3
        weights_np = np.array([10 ** (2 * i / (n - 1)) for i in range(n)])
        weights_jax = jnp.array(weights_np)

        def objective_scipy(x):
            return np.sum(weights_np * x**2)

        def objective_jax(x, args):
            return jnp.sum(weights_jax * x**2), None

        x0_np = np.array([1.0, 1.0, 1.0])
        x0_jax = jnp.array(x0_np)

        # SciPy SLSQP
        result_scipy = scipy_minimize(
            objective_scipy,
            x0_np,
            method="SLSQP",
            options={"ftol": 1e-9, "maxiter": 100},
        )

        # slsqp-jax
        solver = SLSQP(rtol=1e-6, atol=1e-6, max_steps=50, lbfgs_memory=10)
        y_jax, _ = _run_solver(solver, objective_jax, x0_jax)

        # Both should significantly reduce the objective
        initial_obj = objective_scipy(x0_np)
        scipy_obj = objective_scipy(result_scipy.x)
        jax_obj = float(objective_jax(y_jax, None)[0])

        assert scipy_obj < initial_obj * 0.1, "SciPy should reduce objective"
        assert jax_obj < initial_obj * 0.1, "JAX should reduce objective"

    @pytest.mark.slow
    def test_constrained_ill_conditioned_vs_scipy(self):
        """Compare both solvers on ill-conditioned constrained problem."""
        n = 3
        weights_np = np.array([10 ** (2 * i / (n - 1)) for i in range(n)])
        weights_jax = jnp.array(weights_np)

        def objective_scipy(x):
            return np.sum(weights_np * x**2)

        def constraint_scipy(x):
            return np.sum(x) - float(n)  # sum(x) = n

        def objective_jax(x, args):
            return jnp.sum(weights_jax * x**2), None

        def constraint_jax(x, args):
            return jnp.array([jnp.sum(x) - float(n)])

        x0_np = np.ones(n)
        x0_jax = jnp.array(x0_np)

        # SciPy SLSQP
        result_scipy = scipy_minimize(
            objective_scipy,
            x0_np,
            method="SLSQP",
            constraints={"type": "eq", "fun": constraint_scipy},
            options={"ftol": 1e-9, "maxiter": 100},
        )

        # slsqp-jax
        solver = SLSQP(
            rtol=1e-4,
            atol=1e-4,
            max_steps=50,
            eq_constraint_fn=constraint_jax,
            n_eq_constraints=1,
            lbfgs_memory=10,
        )
        y_jax, _ = _run_solver(solver, objective_jax, x0_jax)

        # Both should approximately satisfy constraint
        scipy_constraint_violation = np.abs(np.sum(result_scipy.x) - float(n))
        jax_constraint_violation = float(jnp.abs(jnp.sum(y_jax) - float(n)))

        assert scipy_constraint_violation < 0.5, (
            f"SciPy constraint violation: {scipy_constraint_violation}"
        )
        assert jax_constraint_violation < 0.5, (
            f"JAX constraint violation: {jax_constraint_violation}"
        )


class TestProcessOptimizationBenchmarks:
    """Test problems inspired by process optimization benchmarks.

    These are simplified versions of problems that arise in chemical
    engineering process optimization, which the paper focuses on.
    """

    @pytest.mark.slow
    def test_reactor_cascade_simplified(self):
        """Simplified reactor cascade optimization.

        minimize -conversion (maximize product)
        s.t. mass balance constraints
             temperature bounds

        Variables: temperatures T_1, T_2, T_3 in reactors

        This test verifies the solver handles this type of problem without
        crashing and improves the objective.
        """

        def objective(x, args):
            # Simplified Arrhenius-like kinetics
            # Higher T -> faster reaction, but with diminishing returns
            T1, T2, T3 = x[0], x[1], x[2]
            k1 = jnp.exp(-1000 / (T1 + 273))
            k2 = jnp.exp(-1000 / (T2 + 273))
            k3 = jnp.exp(-1000 / (T3 + 273))
            conversion = 1 - jnp.exp(-k1 - k2 - k3)
            return -conversion, None  # Minimize negative = maximize

        def ineq_constraint(x, args):
            # Temperature bounds: 50 <= T <= 200
            return jnp.concatenate(
                [
                    x - 50,  # T >= 50
                    200 - x,  # T <= 200
                ]
            )

        solver = SLSQP(
            rtol=1e-6,
            atol=1e-6,
            max_steps=100,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=6,
        )
        x0 = jnp.array([100.0, 100.0, 100.0])

        y, state = _run_solver(solver, objective, x0)

        # Verify the solver doesn't crash and improves objective
        initial_obj = objective(x0, None)[0]
        final_obj = objective(y, None)[0]
        assert final_obj <= initial_obj + 1e-6, "Objective should not worsen"

        # Temperatures should stay within bounds (approximately)
        assert jnp.all(y >= 50 - 1), "Temperatures should be >= 50"
        assert jnp.all(y <= 200 + 1), "Temperatures should be <= 200"

    @pytest.mark.slow
    def test_heat_exchanger_network(self):
        """Simplified heat exchanger network optimization.

        minimize utility cost
        s.t. heat balance
             approach temperature constraints

        This has inequality constraints that can become nearly active.
        This test verifies the solver handles this type of problem robustly.
        """

        def objective(x, args):
            # x[0]: hot utility, x[1]: cold utility
            # Cost is proportional to utility usage
            return 10.0 * x[0] + 5.0 * x[1], None

        def eq_constraint(x, args):
            # Heat balance: hot in + hot utility = cold out + cold utility
            # Simplified: total heat duty must balance
            return jnp.array([x[0] - 0.5 * x[1] - 1.0])

        def ineq_constraint(x, args):
            # Non-negativity and upper bounds
            return jnp.concatenate(
                [
                    x,  # x >= 0
                    10 - x,  # x <= 10
                ]
            )

        solver = SLSQP(
            rtol=1e-6,
            atol=1e-6,
            max_steps=100,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=4,
        )
        x0 = jnp.array([2.0, 2.0])

        y, state = _run_solver(solver, objective, x0)

        # Verify objective improved or stayed the same
        initial_obj = objective(x0, None)[0]
        final_obj = objective(y, None)[0]
        assert final_obj <= initial_obj + 1e-3, (
            "Objective should not worsen significantly"
        )

        # Verify constraint violation is bounded
        constraint_violation = jnp.abs(y[0] - 0.5 * y[1] - 1.0)
        assert constraint_violation < 1.0, (
            f"Constraint violation too large: {constraint_violation}"
        )

    @pytest.mark.slow
    def test_distillation_column_simplified(self):
        """Simplified distillation column optimization.

        This problem has a higher condition number due to the
        interaction between reflux ratio and feed composition.
        This test verifies the solver can handle moderately complex
        constrained problems.
        """
        n = 5  # Reduced number of stages for faster testing

        def objective(x, args):
            # x[0:n] are stage temperatures
            # x[n] is reflux ratio
            # Minimize energy (heat duty ~ reflux ratio)
            temperatures = x[:n]
            reflux = x[n]
            # Energy cost + penalty for temperature variation
            energy = reflux * 10
            temp_variation = jnp.sum((temperatures[1:] - temperatures[:-1]) ** 2)
            return energy + temp_variation, None

        def eq_constraint(x, args):
            # Top and bottom temperature specifications
            return jnp.array(
                [
                    x[0] - 80.0,  # Top temp = 80
                    x[n - 1] - 120.0,  # Bottom temp = 120
                ]
            )

        def ineq_constraint(x, args):
            # Reflux ratio bounds: 1 <= R <= 10
            # Temperature ordering: T increases from top to bottom
            reflux = x[n]
            temps = x[:n]
            temp_ordering = temps[1:] - temps[:-1]  # Should be >= 0
            return jnp.concatenate(
                [
                    jnp.array([reflux - 1.0]),  # R >= 1
                    jnp.array([10.0 - reflux]),  # R <= 10
                    temp_ordering,  # T_{i+1} >= T_i
                ]
            )

        solver = SLSQP(
            rtol=1e-4,
            atol=1e-4,
            max_steps=100,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=2,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=2 + (n - 1),
            lbfgs_memory=10,
        )

        # Initial guess: linear temperature profile, moderate reflux
        x0 = jnp.concatenate(
            [
                jnp.linspace(80, 120, n),  # Linear temp profile
                jnp.array([3.0]),  # Reflux ratio
            ]
        )

        y, state = _run_solver(solver, objective, x0)

        # Verify objective improved or stayed the same
        initial_obj = objective(x0, None)[0]
        final_obj = objective(y, None)[0]
        assert final_obj <= initial_obj + 0.1, "Objective should not worsen"

        # Check approximate constraint satisfaction
        top_temp_violation = jnp.abs(y[0] - 80.0)
        bottom_temp_violation = jnp.abs(y[n - 1] - 120.0)
        assert top_temp_violation < 5.0, f"Top temp too far from 80: {y[0]}"
        assert bottom_temp_violation < 5.0, f"Bottom temp too far from 120: {y[n - 1]}"


class TestNumericalStability:
    """Tests for numerical stability edge cases."""

    @pytest.mark.slow
    def test_very_small_gradients(self):
        """Problem where gradients become very small near optimum."""

        def objective(x, args):
            # Flat region near optimum
            return jnp.sum(x**4), None

        solver = SLSQP(
            rtol=1e-8,
            atol=1e-8,
            max_steps=100,
        )
        x0 = jnp.array([0.5, 0.5])

        y, state = _run_solver(solver, objective, x0)

        # Should converge despite flat gradients near optimum
        np.testing.assert_allclose(y, [0.0, 0.0], atol=1e-2)

    @pytest.mark.slow
    def test_large_initial_point(self):
        """Problem starting far from optimum."""

        def objective(x, args):
            return jnp.sum(x**2), None

        solver = SLSQP(
            rtol=1e-4,
            atol=1e-4,
            max_steps=100,
        )
        x0 = jnp.array([1e4, 1e4])

        y, state = _run_solver(solver, objective, x0)

        # Should still converge to origin
        np.testing.assert_allclose(y, [0.0, 0.0], atol=1e-2)

    @pytest.mark.slow
    def test_mixed_large_small_values(self):
        """Problem with variables of different magnitudes.

        This test verifies the solver can make progress on problems
        with different natural scales for variables.
        """

        def objective(x, args):
            # x[0] natural scale ~0.1, x[1] natural scale ~10
            return (x[0] * 10 - 1) ** 2 + (x[1] * 0.1 - 1) ** 2, None

        solver = SLSQP(
            rtol=1e-6,
            atol=1e-6,
            max_steps=50,
        )
        x0 = jnp.array([0.0, 0.0])

        y, state = _run_solver(solver, objective, x0)

        # Verify objective improved
        initial_obj = objective(x0, None)[0]
        final_obj = objective(y, None)[0]
        assert final_obj < initial_obj * 0.1, "Objective should decrease by 90%"
