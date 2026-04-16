"""Tests for pluggable inner QP solvers.

Includes both SLSQP-level integration tests (verifying that
ProjectedCGCraig and MinresQLPSolver work when configured via the
SLSQP.inner_solver attribute) and direct unit tests that exercise
specific code paths in the inner solver implementations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax import SLSQP, MinresQLPSolver, ProjectedCGCholesky, ProjectedCGCraig

jax.config.update("jax_enable_x64", True)


def _run_solver(solver, objective, x0, args=None, max_steps=None):
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


CRAIG_SOLVER = ProjectedCGCraig(
    max_cg_iter=50,
    cg_tol=1e-8,
    craig_tol=1e-12,
    craig_max_iter=200,
)
MINRES_SOLVER = MinresQLPSolver(max_iter=200, tol=1e-10, max_cg_iter=50)


@pytest.fixture(params=["craig", "minres"], ids=["craig", "minres"])
def inner_solver(request):
    if request.param == "craig":
        return CRAIG_SOLVER
    return MINRES_SOLVER


class TestInnerSolverUnconstrained:
    """Unconstrained problems to verify inner solver doesn't break the default path."""

    def test_simple_quadratic(self, inner_solver):
        def objective(x, args):
            return jnp.sum(x**2), None

        solver = SLSQP(atol=1e-8, max_steps=50, inner_solver=inner_solver)
        x0 = jnp.array([3.0, -2.0])
        y, _ = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y, [0.0, 0.0], atol=1e-5)

    def test_rosenbrock_2d(self, inner_solver):
        def objective(x, args):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2, None

        solver = SLSQP(atol=1e-8, max_steps=200, inner_solver=inner_solver)
        x0 = jnp.array([-1.0, 1.0])
        y, _ = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y, [1.0, 1.0], rtol=1e-3)


class TestInnerSolverEquality:
    """Equality-constrained problems."""

    def test_sphere_linear_equality(self, inner_solver):
        """minimize x^2+y^2+z^2 s.t. x+y+z=3"""

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] + x[2] - 3.0])

        solver = SLSQP(
            atol=1e-8,
            max_steps=50,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            inner_solver=inner_solver,
        )
        x0 = jnp.array([1.0, 1.0, 1.0])
        y, _ = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y, [1.0, 1.0, 1.0], rtol=1e-4)
        np.testing.assert_allclose(eq_constraint(y, None), 0.0, atol=1e-5)

    def test_quadratic_two_equalities(self, inner_solver):
        """minimize x^2+y^2+z^2 s.t. x+y=2, y+z=2  =>  (2/3, 4/3, 2/3)"""

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 2.0, x[1] + x[2] - 2.0])

        solver = SLSQP(
            atol=1e-8,
            max_steps=50,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=2,
            inner_solver=inner_solver,
        )
        x0 = jnp.array([0.5, 0.5, 0.5])
        y, _ = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y, [2 / 3, 4 / 3, 2 / 3], rtol=1e-3)
        np.testing.assert_allclose(eq_constraint(y, None), 0.0, atol=1e-5)


class TestInnerSolverInequality:
    """Inequality-constrained problems."""

    def test_sphere_linear_inequality(self, inner_solver):
        """minimize x^2+y^2 s.t. x+y>=2"""

        def objective(x, args):
            return jnp.sum(x**2), None

        def ineq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 2.0])

        solver = SLSQP(
            atol=1e-8,
            max_steps=50,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=1,
            inner_solver=inner_solver,
        )
        x0 = jnp.array([2.0, 2.0])
        y, _ = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y, [1.0, 1.0], rtol=1e-4)
        assert ineq_constraint(y, None)[0] >= -1e-5


class TestInnerSolverMixed:
    """Mixed equality + inequality constraints."""

    def test_sphere_plane_and_halfspace(self, inner_solver):
        """minimize x^2+y^2+z^2 s.t. x+y+z=3, x,y,z>=0"""

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
            inner_solver=inner_solver,
        )
        x0 = jnp.array([1.0, 1.0, 1.0])
        y, _ = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y, [1.0, 1.0, 1.0], rtol=1e-4)
        np.testing.assert_allclose(eq_constraint(y, None), 0.0, atol=1e-5)
        assert jnp.all(ineq_constraint(y, None) >= -1e-5)


class TestInnerSolverBoxConstraints:
    """Box-constrained problems."""

    def test_box_bounded_quadratic(self, inner_solver):
        """minimize (x-3)^2+(y-3)^2 s.t. 0<=x,y<=2  =>  (2,2)"""

        def objective(x, args):
            return (x[0] - 3) ** 2 + (x[1] - 3) ** 2, None

        bounds = jnp.array([[0.0, 2.0], [0.0, 2.0]])
        solver = SLSQP(
            atol=1e-8,
            max_steps=50,
            bounds=bounds,
            inner_solver=inner_solver,
        )
        x0 = jnp.array([1.0, 1.0])
        y, _ = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y, [2.0, 2.0], atol=1e-4)


class TestInnerSolverJIT:
    """Verify that inner solvers work under JIT."""

    def test_jit_with_inner_solver(self, inner_solver):
        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 1.0])

        solver = SLSQP(
            atol=1e-6,
            max_steps=10,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            inner_solver=inner_solver,
        )
        x0 = jnp.array([0.5, 0.5])
        state = solver.init(objective, x0, None, {}, None, None, frozenset())

        @jax.jit
        def jit_step(y, state):
            return solver.step(objective, y, None, {}, state, frozenset())

        y = x0
        for _ in range(5):
            y, state, _ = jit_step(y, state)

        assert jnp.isfinite(jnp.sum(y))


# -----------------------------------------------------------------------
# Direct unit tests for inner solver code paths
# -----------------------------------------------------------------------
#
# These call the .solve() method directly on a small equality-constrained
# QP to exercise specific branches not reached through the SLSQP
# integration tests above.
#
# QP problem:
#   min  0.5 d^T B d + g^T d   s.t. A d = b
#
# With B = diag(2, 4), g = [-2, -6], A = [[1, 1]], b = [0]:
#   KKT solution: d = (-2/3, 2/3), multiplier = 10/3
#   (in the solver's sign convention: L = ... - λ^T(Ad-b), λ >= 0)


def _make_qp():
    """Return (hvp_fn, g, A, b, active_mask, d_expected, mult_expected)."""
    B = jnp.diag(jnp.array([2.0, 4.0]))

    def hvp_fn(v):
        return B @ v

    g = jnp.array([-2.0, -6.0])
    A = jnp.array([[1.0, 1.0]])
    b = jnp.array([0.0])
    active_mask = jnp.array([True])
    d_expected = jnp.array([-2.0 / 3.0, 2.0 / 3.0])
    # The KKT system Bd + g = A^T λ_kkt gives λ_kkt = 10/3.
    # The inner solver uses L = 0.5 d^T B d + g^T d - μ^T(Ad - b),
    # so its returned multiplier μ = -λ_kkt = -10/3.
    mult_expected = jnp.array([-10.0 / 3.0])
    return hvp_fn, g, A, b, active_mask, d_expected, mult_expected


class TestCraigUnpreconditioned:
    """ProjectedCGCraig with precond_fn=None (covers lines 830-831)."""

    def test_solve_no_preconditioner(self):
        hvp_fn, g, A, b, active_mask, d_exp, mult_exp = _make_qp()
        solver = ProjectedCGCraig(
            max_cg_iter=50, cg_tol=1e-10, craig_tol=1e-12, craig_max_iter=200
        )
        result = solver.solve(hvp_fn, g, A, b, active_mask, precond_fn=None)
        np.testing.assert_allclose(result.d, d_exp, atol=1e-8)
        np.testing.assert_allclose(result.multipliers, mult_exp, atol=1e-6)
        assert result.converged


class TestCraigConstraintPreconditioner:
    """ProjectedCGCraig with use_constraint_preconditioner=True (covers lines 783-805)."""

    def test_solve_with_constraint_preconditioner(self):
        hvp_fn, g, A, b, active_mask, d_exp, mult_exp = _make_qp()
        B_diag = jnp.array([2.0, 4.0])

        def precond_fn(v):
            return v / B_diag

        solver = ProjectedCGCraig(
            max_cg_iter=50,
            cg_tol=1e-10,
            craig_tol=1e-12,
            craig_max_iter=200,
            use_constraint_preconditioner=True,
        )
        result = solver.solve(hvp_fn, g, A, b, active_mask, precond_fn=precond_fn)
        np.testing.assert_allclose(result.d, d_exp, atol=1e-8)
        np.testing.assert_allclose(result.multipliers, mult_exp, atol=1e-6)
        assert result.converged


class TestMinresQLPUnpreconditioned:
    """MinresQLPSolver with precond_fn=None (covers lines 955-956, 995-996, 1195)."""

    def test_solve_no_preconditioner(self):
        hvp_fn, g, A, b, active_mask, d_exp, mult_exp = _make_qp()
        solver = MinresQLPSolver(max_iter=200, tol=1e-10, max_cg_iter=50)
        result = solver.solve(hvp_fn, g, A, b, active_mask, precond_fn=None)
        np.testing.assert_allclose(result.d, d_exp, atol=1e-8)
        np.testing.assert_allclose(result.multipliers, mult_exp, atol=1e-6)
        assert result.converged


class TestCholeskySolveDirect:
    """Direct call to ProjectedCGCholesky.solve() for baseline comparison."""

    def test_solve_matches_expected(self):
        hvp_fn, g, A, b, active_mask, d_exp, mult_exp = _make_qp()
        solver = ProjectedCGCholesky(max_cg_iter=50, cg_tol=1e-10)
        result = solver.solve(hvp_fn, g, A, b, active_mask, precond_fn=None)
        np.testing.assert_allclose(result.d, d_exp, atol=1e-8)
        np.testing.assert_allclose(result.multipliers, mult_exp, atol=1e-6)
        assert result.converged
