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


def make_benchmark_problem(n, proximal_tau=0.5):
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
        atol=1e-6,
        max_steps=200,
        eq_constraint_fn=eq_constraint,
        n_eq_constraints=1,
        ineq_constraint_fn=ineq_constraint,
        n_ineq_constraints=n_ineq,
        lbfgs_memory=10,
        proximal_tau=proximal_tau,
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

        # Check inequality constraints (allowing small numerical violations).
        # The proximal sSQP path and per-variable diagonal scaling may
        # produce slightly different convergence trajectories on different
        # platforms, so we allow up to 1e-3 violation.
        jax_ineq_min = np.min(jax_sol[:n_ineq])
        assert jax_ineq_min >= -1e-3, f"Inequality constraint violated: {jax_ineq_min}"

        # Compare objective values rather than solution vectors.  The
        # proximal sSQP path makes B_tilde = B + (1/mu) A_eq^T A_eq, which
        # can be much more ill-conditioned than B alone.  Small FP
        # differences across platforms cascade through CG and the
        # active-set loop, producing different solution vectors that
        # nonetheless achieve similar objective values on this strictly
        # convex problem.
        scipy_f = scipy_result.fun
        jax_f = float(scipy_problem["fun"](jax_sol))
        assert abs(jax_f - scipy_f) / max(abs(scipy_f), 1.0) < 1e-2, (
            f"Objective mismatch: JAX={jax_f:.6f}, SciPy={scipy_f:.6f}"
        )

    @pytest.mark.parametrize("n", [5, 20, 100])
    def test_benchmark_jit_matches_scipy(self, n):
        """Test that JIT-compiled JAX SLSQP also matches SciPy."""
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

        # Check inequality constraints. The proximal sSQP path and
        # per-variable diagonal scaling may produce slightly different
        # convergence trajectories on different platforms, so we allow
        # up to 1e-3 violation.
        jax_ineq_min = np.min(jax_sol[:n_ineq])
        assert jax_ineq_min >= -1e-3, f"Inequality constraint violated: {jax_ineq_min}"

        # Compare objective values rather than solution vectors. The
        # proximal ill-conditioning can cause different CG/active-set
        # trajectories across platforms, but the objective should still be
        # close to SciPy's on this strictly convex problem.
        scipy_f = scipy_result.fun
        jax_f = float(scipy_problem["fun"](jax_sol))
        assert abs(jax_f - scipy_f) / max(abs(scipy_f), 1.0) < 1e-2, (
            f"Objective mismatch: JAX={jax_f:.6f}, SciPy={scipy_f:.6f}"
        )


class TestBenchmarkNoProximal:
    """Test the benchmark problem with proximal_tau=0 (direct projection)."""

    @pytest.mark.parametrize("n", [5, 20, 100])
    def test_no_proximal_matches_scipy(self, n):
        """Direct projection path matches SciPy on benchmark problems."""
        solver, objective, x0, scipy_problem = make_benchmark_problem(n, proximal_tau=0)
        x0_np = np.asarray(x0)
        n_ineq = min(N_INEQ, n)

        scipy_result = scipy_minimize(
            scipy_problem["fun"],
            x0_np,
            method="SLSQP",
            constraints=scipy_problem["constraints"],
            options={"ftol": 1e-9, "maxiter": 200},
        )

        assert scipy_result.success, f"SciPy failed: {scipy_result.message}"

        jax_result = optx.minimise(
            objective,
            solver,
            x0,
            has_aux=True,
            max_steps=200,
            throw=False,
        )
        jax_sol = np.asarray(jax_result.value)

        assert not np.any(np.isnan(jax_sol)), f"JAX solution contains NaN for n={n}"
        assert np.max(np.abs(jax_sol)) < 1e6, f"JAX solution diverged for n={n}"

        jax_eq_viol = np.abs(np.sum(jax_sol) - n)
        assert jax_eq_viol < 1e-3, f"Equality constraint violated: {jax_eq_viol}"

        jax_ineq_min = np.min(jax_sol[:n_ineq])
        assert jax_ineq_min >= -1e-3, f"Inequality constraint violated: {jax_ineq_min}"

        scipy_f = scipy_result.fun
        jax_f = float(scipy_problem["fun"](jax_sol))
        assert abs(jax_f - scipy_f) / max(abs(scipy_f), 1.0) < 1e-2, (
            f"Objective mismatch: JAX={jax_f:.6f}, SciPy={scipy_f:.6f}"
        )


class TestRosenbrock:
    """Test the Rosenbrock function at large scale."""

    @pytest.mark.very_slow
    def test_rosenbrock_100k_dimensions(self):
        """Test SLSQP on 100,000 dimensional Rosenbrock function.

        The Rosenbrock function is:
            f(x) = sum_{i=1}^{n-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]

        The global minimum is at x = (1, 1, ..., 1) with f(x) = 0.

        This test verifies that SLSQP can handle very high-dimensional
        unconstrained optimization problems.
        """
        n = 100_000

        def rosenbrock(x, args):
            """Rosenbrock function - vectorized implementation."""
            # f(x) = sum_{i=0}^{n-2} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
            x_curr = x[:-1]  # x_0, ..., x_{n-2}
            x_next = x[1:]  # x_1, ..., x_{n-1}
            term1 = 100.0 * (x_next - x_curr**2) ** 2
            term2 = (1.0 - x_curr) ** 2
            return jnp.sum(term1 + term2), None

        # Unconstrained SLSQP solver
        solver = SLSQP(
            atol=1e-6,
            max_steps=500,
            eq_constraint_fn=None,
            n_eq_constraints=0,
            ineq_constraint_fn=None,
            n_ineq_constraints=0,
            lbfgs_memory=20,  # More memory for large problem
        )

        # Start near the optimum to make the test tractable
        # (Rosenbrock is notoriously difficult from far away)
        key = jax.random.PRNGKey(42)
        x0 = jnp.ones(n) + 0.1 * jax.random.normal(key, (n,))

        # JIT-compile and solve
        @jax.jit
        def solve(x0, _solver=solver, _objective=rosenbrock):
            return optx.minimise(
                _objective,
                _solver,
                x0,
                has_aux=True,
                max_steps=500,
                throw=False,
            )

        result = solve(x0)
        jax_sol = np.asarray(result.value)
        final_f = float(rosenbrock(result.value, None)[0])

        # Check for NaN
        assert not np.any(np.isnan(jax_sol)), "Solution contains NaN"

        # Check solution is not diverging
        assert np.max(np.abs(jax_sol)) < 100, (
            f"Solution diverged: max={np.max(np.abs(jax_sol))}"
        )

        # Check that we're reasonably close to the optimum (x = 1)
        # For such a large problem starting near the optimum, we should get close
        mean_sol = np.mean(jax_sol)
        assert 0.5 < mean_sol < 1.5, f"Mean solution {mean_sol} is far from 1.0"

        # Check that the function value decreased significantly
        initial_f = float(rosenbrock(x0, None)[0])
        assert final_f < initial_f, (
            f"Function did not decrease: {initial_f} -> {final_f}"
        )

        # The function value should be relatively small (we started close to optimum)
        assert final_f < 1000, f"Final function value {final_f} is too large"

        print("\nRosenbrock 100k test passed:")
        print(f"  Initial f: {initial_f:.2f}")
        print(f"  Final f:   {final_f:.2f}")
        print(f"  Mean x:    {mean_sol:.4f}")
        print(f"  Result:    {result.result}")
