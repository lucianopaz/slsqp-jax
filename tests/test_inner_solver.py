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


class TestMinresQLPFeasibilityProjection:
    """Posterior M-metric projection inside ``_solve_kkt_minres_qlp``.

    PMINRES-QLP minimises the Euclidean residual of the full KKT
    system, with no separate guarantee on the constraint subvector
    ``A d - b``.  When the Lanczos recurrence is truncated by
    ``max_iter``, the SLSQP step would otherwise leave the linearised
    feasible region.  The posterior projection restores
    ``A_work d = b_work`` to working precision in one back-solve;
    these tests exercise both code paths (with and without an SPD
    preconditioner) and the bound-fixed subspace branch.
    """

    @staticmethod
    def _make_truncation_qp(n: int = 20, m: int = 3, seed: int = 0):
        """A medium QP whose KKT system needs more than 5 Lanczos
        iterations to converge.  Returns matvec / data that the inner
        solver expects, plus the dense KKT matrix for sanity checks.
        """
        rng = np.random.RandomState(seed)
        eigvals = jnp.asarray(np.logspace(-1.0, 2.0, n), dtype=jnp.float64)
        Q, _ = jnp.linalg.qr(jnp.asarray(rng.randn(n, n), dtype=jnp.float64))
        B = (Q * eigvals) @ Q.T
        B = 0.5 * (B + B.T)
        A = jnp.asarray(rng.randn(m, n), dtype=jnp.float64)
        g = jnp.asarray(rng.randn(n), dtype=jnp.float64)
        b = jnp.asarray(rng.randn(m), dtype=jnp.float64)

        def hvp_fn(v):
            return B @ v

        active_mask = jnp.ones(m, dtype=bool)
        return hvp_fn, g, A, b, active_mask, B

    @staticmethod
    def _unprojected_minres_d(hvp_fn, g, A, b, max_iter, precond_fn=None):
        """Reference: pure PMINRES-QLP on the KKT system, no projection.

        Used to confirm that truncation actually leaves the unprojected
        direction infeasible, so the projection-pass tests are exercising
        a real failure mode and not a vacuously-feasible iterate.
        """
        from slsqp_jax.inner_solver import pminres_qlp_solve

        n = g.shape[0]
        m = A.shape[0]

        def kkt_matvec(z):
            return jnp.concatenate([hvp_fn(z[:n]) + A.T @ z[n:], A @ z[:n]])

        rhs = jnp.concatenate([-g, b])

        if precond_fn is not None:
            B_diag_arr = jnp.diag(jnp.eye(n))  # unused, just keeps mypy happy
            del B_diag_arr
            M_AT = jax.vmap(precond_fn)(A).T
            S = A @ M_AT + 1e-8 * jnp.eye(m)
            S_chol = jnp.linalg.cholesky(S)

            def kkt_pre(z):
                return jnp.concatenate(
                    [
                        precond_fn(z[:n]),
                        jax.scipy.linalg.cho_solve((S_chol, True), z[n:]),
                    ]
                )

            sol, _ = pminres_qlp_solve(
                kkt_matvec, rhs, tol=1e-12, max_iter=max_iter, precond=kkt_pre
            )
        else:
            sol, _ = pminres_qlp_solve(kkt_matvec, rhs, tol=1e-12, max_iter=max_iter)
        return sol[:n]

    def test_truncated_minres_with_preconditioner_is_feasible(self):
        """5-iteration MINRES on a 20-variable problem cannot solve the
        KKT system, but the posterior projection must drive ``||A d - b||``
        down by many orders of magnitude.

        The achievable feasibility floor is ``O(eps * cond(A M A^T) *
        ||r_dual||)`` because the Schur Cholesky carries a ``1e-8 * I``
        regularisation; we check both the absolute floor and the
        improvement over the un-projected baseline.
        """
        hvp_fn, g, A, b, active_mask, B = self._make_truncation_qp()

        B_diag = jnp.diag(B)

        def precond_fn(v):
            return v / B_diag

        solver = MinresQLPSolver(max_iter=5, tol=1e-12, max_cg_iter=50)
        result = solver.solve(hvp_fn, g, A, b, active_mask, precond_fn=precond_fn)

        d_raw = self._unprojected_minres_d(
            hvp_fn, g, A, b, max_iter=5, precond_fn=precond_fn
        )
        feasibility_raw = float(jnp.max(jnp.abs(A @ d_raw - b)))
        feasibility = float(jnp.max(jnp.abs(A @ result.d - b)))

        assert feasibility_raw > 1e-3, (
            "test setup is degenerate: un-projected MINRES is already feasible "
            f"(||A d - b||_inf = {feasibility_raw:.2e}); pick a harder problem"
        )
        assert feasibility < 1e-7, (
            f"posterior projection failed: ||A d - b||_inf = {feasibility:.2e} "
            f"(raw {feasibility_raw:.2e})"
        )
        # Projection should improve feasibility by at least 4 orders of magnitude.
        assert feasibility < feasibility_raw * 1e-4, (
            f"projection insufficient: raw {feasibility_raw:.2e} vs "
            f"projected {feasibility:.2e}"
        )
        assert jnp.all(jnp.isfinite(result.d))

    def test_truncated_minres_without_preconditioner_is_feasible(self):
        """No-preconditioner branch must build its own ``A A^T`` Cholesky
        and produce a near-feasible direction even after truncation.
        """
        hvp_fn, g, A, b, active_mask, _ = self._make_truncation_qp(seed=1)

        solver = MinresQLPSolver(max_iter=5, tol=1e-12, max_cg_iter=50)
        result = solver.solve(hvp_fn, g, A, b, active_mask, precond_fn=None)

        d_raw = self._unprojected_minres_d(hvp_fn, g, A, b, max_iter=5)
        feasibility_raw = float(jnp.max(jnp.abs(A @ d_raw - b)))
        feasibility = float(jnp.max(jnp.abs(A @ result.d - b)))

        assert feasibility_raw > 1e-3, (
            "test setup is degenerate: un-projected MINRES is already feasible "
            f"(||A d - b||_inf = {feasibility_raw:.2e}); pick a harder problem"
        )
        assert feasibility < 1e-7, (
            f"posterior projection failed (no-precond): "
            f"||A d - b||_inf = {feasibility:.2e} (raw {feasibility_raw:.2e})"
        )
        assert feasibility < feasibility_raw * 1e-4, (
            f"projection insufficient (no-precond): raw {feasibility_raw:.2e} "
            f"vs projected {feasibility:.2e}"
        )
        assert jnp.all(jnp.isfinite(result.d))

    def test_projection_does_not_break_full_convergence(self):
        """When MINRES does converge, the projection must be a near-no-op:
        the iterate already satisfies ``A d = b`` to MINRES tolerance, so
        the M-metric correction should leave the unconstrained KKT
        solution intact (multipliers and direction).
        """
        hvp_fn, g, A, b, active_mask, d_exp, mult_exp = _make_qp()

        solver = MinresQLPSolver(max_iter=200, tol=1e-12, max_cg_iter=50)
        result = solver.solve(hvp_fn, g, A, b, active_mask, precond_fn=None)
        np.testing.assert_allclose(result.d, d_exp, atol=1e-8)
        np.testing.assert_allclose(result.multipliers, mult_exp, atol=1e-6)
        assert bool(result.converged)

    def test_projection_respects_bound_fixed_subspace(self):
        """With ``free_mask`` / ``d_fixed`` supplied, the projected
        direction must (a) leave the fixed entries equal to ``d_fixed``
        and (b) restore ``A_work d - b_work = 0`` on the active rows,
        where ``A_work = A * free_mask[None, :]`` and
        ``b_work = b - A @ d_fixed``.
        """
        n, m = 30, 4
        hvp_fn, g, A, b, active_mask, B = self._make_truncation_qp(n=n, m=m, seed=2)

        rng = np.random.RandomState(7)
        free_mask = jnp.asarray(rng.rand(n) > 0.5, dtype=bool)
        # Make sure at least n - m + 1 variables stay free so the reduced
        # KKT system is still consistent.
        if int(jnp.sum(free_mask)) < n - m + 1:
            free_mask = free_mask.at[: n - m + 1].set(True)
        d_fixed = jnp.asarray(rng.randn(n) * 0.1, dtype=jnp.float64)
        d_fixed = jnp.where(free_mask, 0.0, d_fixed)

        B_diag = jnp.diag(B)

        def precond_fn(v):
            return v / B_diag

        solver = MinresQLPSolver(max_iter=4, tol=1e-12, max_cg_iter=50)
        result = solver.solve(
            hvp_fn,
            g,
            A,
            b,
            active_mask,
            precond_fn=precond_fn,
            free_mask=free_mask,
            d_fixed=d_fixed,
        )

        # (a) Fixed entries are pinned exactly.
        fixed_err = float(
            jnp.max(jnp.abs(jnp.where(free_mask, 0.0, result.d - d_fixed)))
        )
        assert fixed_err == 0.0, (
            f"fixed entries drifted: max|d - d_fixed| over !free = {fixed_err:.2e}"
        )

        # (b) Reduced-space dual residual is at the projection floor
        #     (~1e-7, bounded by the 1e-8 Schur regularisation).
        A_work = A * free_mask[None, :]
        b_work = b - A @ d_fixed
        red_feas = float(jnp.max(jnp.abs(A_work @ result.d - b_work)))
        assert red_feas < 1e-7, (
            f"reduced-space feasibility broken: "
            f"||A_work d - b_work||_inf = {red_feas:.2e}"
        )


class TestNonScalarAdaptiveTol:
    """Regression: a (1,)-shaped tolerance must not crash the inner solvers.

    Optimistix's ``ImplicitAdjoint.apply`` re-traces the primal step
    function during the adjoint computation (and on the forward pass with
    the default adjoint).  If ``state.prev_grad_lagrangian`` ever carries
    an unexpected leading axis, the Eisenstat-Walker tolerance computed
    in ``solver.py`` becomes shape ``(1,)``.  That used to broadcast every
    ``r0_norm_sq < cg_tol ** 2`` comparison to ``(1,)`` and trigger
    ``TypeError: Pred must be a scalar`` deep inside ``jax.lax.cond``
    in ``cg_step``.  The inner solvers must coerce the tolerance to a
    true 0-d scalar at entry; this test simulates the leak directly.
    """

    @pytest.mark.parametrize(
        "make_solver",
        [
            lambda tol: ProjectedCGCholesky(max_cg_iter=50, cg_tol=tol),
            lambda tol: ProjectedCGCraig(
                max_cg_iter=50, cg_tol=tol, craig_tol=1e-12, craig_max_iter=200
            ),
        ],
        ids=["cholesky", "craig"],
    )
    def test_size_one_adaptive_tol(self, make_solver):
        hvp_fn, g, A, b, active_mask, d_exp, mult_exp = _make_qp()
        solver = make_solver(1e-10)
        leaked_tol = jnp.array([1e-10])
        result = solver.solve(
            hvp_fn,
            g,
            A,
            b,
            active_mask,
            precond_fn=None,
            adaptive_tol=leaked_tol,
        )
        np.testing.assert_allclose(result.d, d_exp, atol=1e-8)
        np.testing.assert_allclose(result.multipliers, mult_exp, atol=1e-6)
        assert result.converged.shape == ()
        assert bool(result.converged)
