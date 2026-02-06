"""Test the benchmark problem to compare SLSQP-JAX with SciPy.

This module tests that our SLSQP implementation produces results that match
scipy.optimize.minimize(method='SLSQP') on a constrained quadratic problem.

The test problem:
    minimize    sum_i w_i * (x_i - t_i)^2
    subject to  sum(x) = n
                x[0], ..., x[3] >= 0

where w_i are linearly spaced in [1, 10] and t_i = -2 for the first 4 variables
and t_i = 1 for the rest. This makes the inequality constraints active at the
solution.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest
from scipy.optimize import minimize as scipy_minimize

from slsqp_jax import SLSQP

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

N_INEQ = 4  # fixed number of box constraints


def make_benchmark_problem(n):
    """Create a constrained quadratic with 1 equality + 4 inequality constraints.

    Returns (solver, objective, x0, scipy_problem).
    """
    weights = jnp.linspace(1.0, 10.0, n)
    target = jnp.ones(n).at[: min(N_INEQ, n)].set(-2.0)

    weights_np = np.asarray(weights)
    target_np = np.asarray(target)

    def objective(x, args):
        return jnp.sum(weights * (x - target) ** 2), None

    def eq_constraint(x, args):
        return jnp.array([jnp.sum(x) - float(n)])

    n_ineq = min(N_INEQ, n)

    def ineq_constraint(x, args):
        return x[:n_ineq]

    solver = SLSQP(
        rtol=1e-6,
        atol=1e-6,
        max_steps=200,
        eq_constraint_fn=eq_constraint,
        n_eq_constraints=1,
        ineq_constraint_fn=ineq_constraint,
        n_ineq_constraints=n_ineq,
        lbfgs_memory=10,
    )

    # SciPy problem
    def scipy_objective(x):
        return np.sum(weights_np * (x - target_np) ** 2)

    def scipy_eq_constraint(x):
        return np.sum(x) - float(n)

    scipy_constraints = [{"type": "eq", "fun": scipy_eq_constraint}]
    for i in range(n_ineq):
        scipy_constraints.append({"type": "ineq", "fun": lambda x, i=i: x[i]})

    scipy_problem = {"fun": scipy_objective, "constraints": scipy_constraints}

    x0 = jnp.ones(n)
    return solver, objective, x0, scipy_problem


class TestBenchmarkProblem:
    """Test the benchmark optimization problem."""

    @pytest.mark.parametrize("n", [5, 20, 100, 500])
    def test_benchmark_jax_matches_scipy(self, n):
        """Compare JAX SLSQP with SciPy on benchmark problems of various sizes."""
        solver, objective, x0, scipy_problem = make_benchmark_problem(n)
        x0_np = np.asarray(x0)
        n_ineq = min(N_INEQ, n)

        # Solve with SciPy
        scipy_result = scipy_minimize(
            scipy_problem["fun"],
            x0_np,
            method="SLSQP",
            constraints=scipy_problem["constraints"],
            options={"ftol": 1e-9, "maxiter": 200},
        )
        scipy_sol = scipy_result.x

        assert scipy_result.success, f"SciPy failed: {scipy_result.message}"

        # Solve with JAX
        jax_result = optx.minimise(
            objective,
            solver,
            x0,
            has_aux=True,
            max_steps=200,
            throw=False,
        )
        jax_sol = np.asarray(jax_result.value)

        # Check for NaN
        assert not np.any(np.isnan(jax_sol)), f"JAX solution contains NaN for n={n}"

        # Check solution is not huge (no divergence)
        assert np.max(np.abs(jax_sol)) < 1e6, f"JAX solution diverged for n={n}"

        # Check equality constraint
        jax_eq_viol = np.abs(np.sum(jax_sol) - n)
        assert jax_eq_viol < 1e-4, f"Equality constraint violated: {jax_eq_viol}"

        # Check inequality constraints (allowing small numerical violations)
        jax_ineq_min = np.min(jax_sol[:n_ineq])
        assert jax_ineq_min >= -1e-6, f"Inequality constraint violated: {jax_ineq_min}"

        # Compare solutions with reasonable tolerance
        max_diff = np.max(np.abs(jax_sol - scipy_sol))
        assert max_diff < 1e-2, f"Solutions differ by {max_diff} for n={n}"

    @pytest.mark.parametrize("n", [5, 20, 100])
    def test_benchmark_jit_matches_scipy(self, n):
        """Test that JIT-compiled JAX SLSQP also matches SciPy."""
        solver, objective, x0, scipy_problem = make_benchmark_problem(n)
        x0_np = np.asarray(x0)

        # Solve with SciPy
        scipy_result = scipy_minimize(
            scipy_problem["fun"],
            x0_np,
            method="SLSQP",
            constraints=scipy_problem["constraints"],
            options={"ftol": 1e-9, "maxiter": 200},
        )
        scipy_sol = scipy_result.x

        # Solve with JIT-compiled JAX
        @jax.jit
        def solve_jit(x0, _solver=solver, _objective=objective):
            return optx.minimise(
                _objective,
                _solver,
                x0,
                has_aux=True,
                max_steps=200,
                throw=False,
            ).value

        jax_sol = np.asarray(solve_jit(x0))

        # Check for NaN
        assert not np.any(np.isnan(jax_sol)), f"JIT JAX solution contains NaN for n={n}"

        # Check solution is not huge (no divergence)
        assert np.max(np.abs(jax_sol)) < 1e6, f"JIT JAX solution diverged for n={n}"

        # Check equality constraint
        jax_eq_viol = np.abs(np.sum(jax_sol) - n)
        assert jax_eq_viol < 1e-4, f"Equality constraint violated: {jax_eq_viol}"

        # Compare solutions
        max_diff = np.max(np.abs(jax_sol - scipy_sol))
        assert max_diff < 1e-2, f"JIT solutions differ by {max_diff} for n={n}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
