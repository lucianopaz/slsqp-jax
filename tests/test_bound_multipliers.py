"""Tests for the NLP-level bound-multiplier recovery in SLSQP.

These tests target the post-line-search bound-multiplier splice added in
``solver.py``: after the line search, the bound block of
``multipliers_ineq`` is recomputed from the partial Lagrangian gradient
at ``x_{k+1}`` and used both for the stationarity check and for the
warm-start of the next QP.  By construction this should:

* Cancel ``∇L`` exactly on bound-active indices (up to the residual on
  the equality / general-inequality multipliers).
* Make ``state.multipliers_ineq[m_general:]`` reproduce the analytic
  bound multipliers of a closed-form QP to (close to) machine
  precision.
* Lower the relative-stationarity ratio ``||∇L|| / max(|L|, 1)`` for
  bound-heavy problems where the QP-level recovery used to leave the
  ratio above ``rtol``.

References:

* Plan: ``docs/plans/nlp-bound-multiplier-recovery_da6490f9.plan.md``.
* Algebraic justification (constant bound Jacobian): the bound block of
  the L-BFGS secant pair is invariant under the splice, matching the
  L-BFGS-B precedent (Byrd, Lu, Nocedal, Zhu, SIAM J. Sci. Comput.
  16(5), 1995).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest

from slsqp_jax import (
    SLSQP,
    MinresQLPSolver,
    compute_partial_lagrangian_gradient,
    get_diagnostics,
)

jax.config.update("jax_enable_x64", True)


def _run_solver(solver, objective, x0, args=None, max_steps=None):
    """Run the SLSQP solver loop and return the final iterate plus state."""
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


class TestPartialLagrangianGradient:
    """Sanity checks for ``compute_partial_lagrangian_gradient``."""

    def test_no_constraints_returns_grad_f(self):
        grad_f = jnp.array([1.0, -2.0, 0.5])
        eq_jac = jnp.zeros((0, 3))
        gen_jac = jnp.zeros((0, 3))
        out = compute_partial_lagrangian_gradient(
            grad_f, eq_jac, jnp.zeros((0,)), gen_jac, jnp.zeros((0,))
        )
        np.testing.assert_allclose(out, grad_f)

    def test_eq_only(self):
        grad_f = jnp.array([1.0, 2.0, 3.0])
        eq_jac = jnp.array([[1.0, 1.0, 1.0]])
        mu_eq = jnp.array([0.5])
        gen_jac = jnp.zeros((0, 3))
        out = compute_partial_lagrangian_gradient(
            grad_f, eq_jac, mu_eq, gen_jac, jnp.zeros((0,))
        )
        np.testing.assert_allclose(out, grad_f - eq_jac.T @ mu_eq)

    def test_gen_only(self):
        grad_f = jnp.array([1.0, 2.0, 3.0])
        eq_jac = jnp.zeros((0, 3))
        gen_jac = jnp.array([[1.0, 0.0, 0.0]])
        mu_gen = jnp.array([0.25])
        out = compute_partial_lagrangian_gradient(
            grad_f, eq_jac, jnp.zeros((0,)), gen_jac, mu_gen
        )
        np.testing.assert_allclose(out, grad_f - gen_jac.T @ mu_gen)

    def test_eq_and_gen(self):
        grad_f = jnp.array([1.0, 2.0, 3.0])
        eq_jac = jnp.array([[1.0, 1.0, 1.0]])
        gen_jac = jnp.array([[1.0, 0.0, 0.0]])
        mu_eq = jnp.array([0.7])
        mu_gen = jnp.array([0.3])
        out = compute_partial_lagrangian_gradient(
            grad_f, eq_jac, mu_eq, gen_jac, mu_gen
        )
        expected = grad_f - eq_jac.T @ mu_eq - gen_jac.T @ mu_gen
        np.testing.assert_allclose(out, expected)


class TestNLPBoundMultiplierRecovery:
    """End-to-end checks of the post-line-search bound-multiplier splice."""

    @pytest.mark.slow
    def test_diagonal_qp_lower_bound_multipliers_match_analytic(self):
        """Closed-form ``min ½ x^T diag(d) x − c^T x  s.t.  x ≥ 0``.

        The KKT conditions give ``x_i = max(c_i / d_i, 0)`` and a bound
        multiplier ``μ_i = max(−c_i, 0)`` (which is non-zero exactly when
        ``c_i ≤ 0``, i.e. when ``x_i`` is at the lower bound).  The
        post-line-search bound-multiplier recovery should reproduce
        these analytic multipliers and zero the bound-active components
        of ``∇L`` to (close to) machine precision.
        """
        d = jnp.array([1.0, 2.0, 0.5, 4.0])
        c = jnp.array([2.0, -1.0, 3.0, -0.5])
        n = d.shape[0]

        def objective(x, args):
            return 0.5 * jnp.dot(d * x, x) - jnp.dot(c, x), None

        bounds = jnp.column_stack([jnp.zeros(n), jnp.full(n, jnp.inf)])

        # Analytic optimum
        x_star = jnp.maximum(c / d, 0.0)
        mu_star_lower = jnp.maximum(-c, 0.0)

        solver = SLSQP(
            atol=1e-10,
            rtol=1e-10,
            max_steps=50,
            bounds=bounds,
        )
        x0 = jnp.full(n, 0.5)
        y, state = _run_solver(solver, objective, x0)

        # Primal solution
        np.testing.assert_allclose(y, x_star, atol=1e-7)

        # Bounds are layout [general; lower; upper]; here m_general = 0,
        # n_lower = n, n_upper = 0 (bounds are [0, +inf]).
        m_general = 0
        bound_mult_lower = state.multipliers_ineq[m_general : m_general + n]
        np.testing.assert_allclose(bound_mult_lower, mu_star_lower, atol=1e-7)

        # All recovered bound multipliers must be non-negative.
        assert jnp.all(bound_mult_lower >= -1e-12)

        # Stationarity: ||∇L|| / max(|L|, 1) should be tiny.
        grad_L = state.grad_lagrangian
        # Lagrangian value at the solution
        L = state.f_val - state.multipliers_ineq @ state.ineq_val
        rel_stationarity = jnp.linalg.norm(grad_L) / jnp.maximum(jnp.abs(L), 1.0)
        assert float(rel_stationarity) < 1e-8, (
            f"relative stationarity {float(rel_stationarity):.3e} exceeds 1e-8"
        )

    def test_diagonal_qp_upper_bound_multipliers_match_analytic(self):
        """Same problem but with ``x ≤ ub`` only.

        For ``min ½ x^T diag(d) x − c^T x  s.t.  x ≤ ub`` and large
        positive ``c``, the unconstrained optimum ``c/d`` exceeds the
        upper bound and the upper-bound multiplier is non-zero.

        Sign convention: bound Jacobian is ``+I`` for lower rows and
        ``−I`` for upper rows, so ``∇f − (−I^T) μ_upper = ∇f + μ_upper``
        and the recovery returns ``μ_upper = −∂f/∂x_i`` at the active
        upper-bound indices.
        """
        d = jnp.array([1.0, 2.0])
        c = jnp.array([5.0, 8.0])
        ub = jnp.array([1.0, 2.0])
        n = d.shape[0]

        def objective(x, args):
            return 0.5 * jnp.dot(d * x, x) - jnp.dot(c, x), None

        bounds = jnp.column_stack([jnp.full(n, -jnp.inf), ub])

        # Analytic optimum: x_i = ub_i (since c_i / d_i > ub_i)
        x_star = ub
        # ∇f at the optimum: d * ub − c
        # μ_upper = −∂f/∂x_i = c − d * ub
        mu_star_upper = c - d * ub

        solver = SLSQP(
            atol=1e-10,
            rtol=1e-10,
            max_steps=50,
            bounds=bounds,
        )
        x0 = jnp.zeros(n)
        y, state = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, x_star, atol=1e-7)

        # Layout: [general; lower; upper] with m_general = 0, n_lower = 0.
        bound_mult_upper = state.multipliers_ineq
        np.testing.assert_allclose(bound_mult_upper, mu_star_upper, atol=1e-7)
        assert jnp.all(bound_mult_upper >= -1e-12)

    @pytest.mark.slow
    def test_bound_mult_with_equality_constraint(self):
        """Mix bound + equality so the recovery exercises the partial gradient.

        ``min ½ ||x||^2 − c^T x  s.t.  sum(x) = s,  x ≥ 0``.

        With the convention ``L = f − λ · c_eq − μ · c_ineq`` (matching
        ``compute_lagrangian_gradient``) and ``c_ineq = x``, the
        stationarity condition reads ``x_i = c_i + λ + μ_i`` with
        ``μ_i ≥ 0`` and ``μ_i x_i = 0``.

        Picking ``c = [1, 0, −2]`` and ``s = 2``: assuming index 2 is
        the only active bound, ``λ = 0.5`` solves ``sum(x*) = 2`` and
        ``μ = [0, 0, 1.5]``, giving ``x* = [1.5, 0.5, 0]``.  All
        primals satisfy the active set assumption.
        """
        c = jnp.array([1.0, 0.0, -2.0])
        s_target = 2.0
        n = c.shape[0]

        def objective(x, args):
            return 0.5 * jnp.dot(x, x) - jnp.dot(c, x), None

        def eq_constraint(x, args):
            return jnp.array([jnp.sum(x) - s_target])

        bounds = jnp.column_stack([jnp.zeros(n), jnp.full(n, jnp.inf)])

        solver = SLSQP(
            atol=1e-10,
            rtol=1e-10,
            max_steps=80,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            bounds=bounds,
        )
        x0 = jnp.array([1.0, 1.0, 0.0])
        y, state = _run_solver(solver, objective, x0)

        x_star = jnp.array([1.5, 0.5, 0.0])
        mu_star = jnp.array([0.0, 0.0, 1.5])

        np.testing.assert_allclose(y, x_star, atol=5e-6)

        # Bound multipliers (no general inequalities here, m_general = 0).
        bound_mult_lower = state.multipliers_ineq[:n]
        np.testing.assert_allclose(bound_mult_lower, mu_star, atol=5e-6)
        assert jnp.all(bound_mult_lower >= -1e-10)

        # Lagrangian gradient at active bound (index 2): the recovered
        # bound multiplier should make ∇L[2] ≈ 0.  The free indices have
        # ∇L ≈ 0 from the QP.
        assert jnp.linalg.norm(state.grad_lagrangian) < 1e-6

    @pytest.mark.slow
    def test_bound_mult_only_active_bounds_get_nonzero(self):
        """Only variables actually at a bound at ``x_{k+1}`` get a non-zero μ.

        Free bound-feasible variables (interior to their bound) should
        get ``μ = 0`` from the recovery, regardless of the QP-level
        guess.  Use a problem where exactly one of three bounded
        variables is at the bound at the optimum.
        """
        # min (x-2)^2 + (y-3)^2 + (z+1)^2 with z >= 0
        # Optimum: (2, 3, 0). Only z is bound-active; μ_z = 2 (since
        # ∂f/∂z = 2*(z+1) = 2 at z=0 and the lower-bound row is +I).

        def objective(x, args):
            return (
                (x[0] - 2.0) ** 2 + (x[1] - 3.0) ** 2 + (x[2] + 1.0) ** 2,
                None,
            )

        bounds = jnp.array(
            [
                [-jnp.inf, jnp.inf],
                [-jnp.inf, jnp.inf],
                [0.0, jnp.inf],
            ]
        )

        solver = SLSQP(
            atol=1e-10,
            rtol=1e-10,
            max_steps=50,
            bounds=bounds,
        )
        x0 = jnp.array([0.0, 0.0, 0.5])
        y, state = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [2.0, 3.0, 0.0], atol=1e-6)

        # Layout: [general; lower; upper].  m_general = 0; only z has a
        # finite lower bound, so n_lower = 1.  The recovered multiplier
        # should equal +∂f/∂z = 2 at the optimum.
        bound_mult_lower = state.multipliers_ineq
        np.testing.assert_allclose(bound_mult_lower, [2.0], atol=1e-6)
        assert jnp.all(bound_mult_lower >= -1e-12)

    def test_no_bounds_path_unchanged(self):
        """When ``bounds=None`` the recovery is a no-op.

        ``state.multipliers_ineq`` then carries only general-inequality
        multipliers and the splice should never touch it.
        """

        def objective(x, args):
            return jnp.sum((x - 1.0) ** 2), None

        def ineq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 1.0])

        solver = SLSQP(
            atol=1e-8,
            max_steps=50,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=1,
            bounds=None,
        )
        x0 = jnp.array([0.0, 0.0])
        y, state = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [1.0, 1.0], rtol=1e-5)
        assert state.multipliers_ineq.shape == (1,)


class TestNLPBoundMultiplierStagnationRegression:
    """Regression: bound-heavy problems should not stagnate above ``rtol``.

    These exercise the original failure mode: on a Portfolio-style
    problem with many active bounds, the QP-level bound-mult recovery
    used to leave ``||∇L|| / max(|L|, 1)`` pinned above ``rtol`` even
    though the constraints were satisfied and ``alpha = 1``.  With the
    NLP-level recovery the ratio should drop sharply.

    The ``sol.result`` field is intentionally NOT used for the
    convergence assertion: best-iterate rollback / merit-stagnation can
    flag a run as ``nonlinear_divergence`` even when the iterate
    satisfies KKT to high accuracy, so the meaningful regression target
    is the *stationarity ratio at the returned iterate*.
    """

    @staticmethod
    def _make_portfolio(n: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        k_factors = min(10, n // 5)
        F = jnp.array(rng.randn(n, k_factors) * 0.1)
        d = jnp.array(np.abs(rng.randn(n)) * 0.05 + 0.01)
        mu = jnp.array(rng.rand(n) * 0.1 + 0.02)
        r_target = float(jnp.mean(mu)) * 1.2

        def objective(x, args):
            Fx = F.T @ x
            variance = jnp.dot(Fx, Fx) + jnp.dot(d * x, x)
            return 0.5 * variance, None

        def eq_constraint(x, args):
            return jnp.array([jnp.sum(x) - 1.0])

        def ineq_constraint(x, args):
            return jnp.array([jnp.dot(mu, x) - r_target])

        bounds = jnp.column_stack([jnp.zeros(n), jnp.ones(n)])
        x0 = jnp.ones(n) / n
        return objective, eq_constraint, ineq_constraint, bounds, x0

    def _assert_iterate_stationary(
        self,
        state,
        x_sol,
        eq_fn,
        ineq_fn,
        rtol_target=1e-3,
        atol_target=1e-5,
    ):
        """Assert KKT-like conditions on the *returned iterate*.

        Checks (1) primal feasibility (eq, ineq, bounds), (2) dual
        feasibility on the bound multipliers (≥ 0), and (3) the
        relative-stationarity ratio
        ``||∇L|| / max(|L|, 1) <= rtol_target``.

        ``rtol_target`` is intentionally looser than the solver's own
        ``rtol`` so we are not just round-tripping the convergence
        check; what we care about is "the bound-multiplier recovery
        kept the ratio at small values" rather than "the solver
        terminated with rtol satisfied".
        """
        eq_viol = float(jnp.max(jnp.abs(eq_fn(x_sol, None))))
        ineq_viol = float(jnp.max(jnp.maximum(0.0, -ineq_fn(x_sol, None))))
        assert eq_viol < atol_target, (
            f"eq_viol = {eq_viol:.3e} exceeds {atol_target:.0e}"
        )
        assert ineq_viol < atol_target, (
            f"ineq_viol = {ineq_viol:.3e} exceeds {atol_target:.0e}"
        )
        # Bounds: x in [0, 1] for the portfolio problem
        assert float(jnp.min(x_sol)) >= -1e-7
        assert float(jnp.max(x_sol)) <= 1.0 + 1e-7

        # Bound multipliers are dual-feasible (μ ≥ 0).  General-ineq +
        # bound multipliers all live in ``state.multipliers_ineq``;
        # all of them must be non-negative under our sign convention.
        bound_mults = state.multipliers_ineq
        assert float(jnp.min(bound_mults)) >= -1e-8, (
            f"min(bound mult) = {float(jnp.min(bound_mults)):.3e}"
        )

        # Relative stationarity at the returned iterate.
        L = state.f_val
        if state.multipliers_eq.shape[0] > 0:
            L = L - state.multipliers_eq @ state.eq_val
        if state.multipliers_ineq.shape[0] > 0:
            L = L - state.multipliers_ineq @ state.ineq_val
        rel_stationarity = jnp.linalg.norm(state.grad_lagrangian) / jnp.maximum(
            jnp.abs(L), 1.0
        )
        assert float(rel_stationarity) < rtol_target, (
            f"||∇L||/|L| = {float(rel_stationarity):.3e} exceeds {rtol_target:.0e}"
        )

    @pytest.mark.slow
    def test_portfolio_small_minres_qlp_iterate_is_stationary(self):
        """``Portfolio(n=100) + MinresQLPSolver``: iterate satisfies KKT.

        Sanity / regression guard at the smallest size where
        MINRES-QLP already cleared the convergence test in baseline
        (see ``scripts/test_portfolio_convergence.py``).
        """
        n = 100
        objective, eq_fn, ineq_fn, bounds, x0 = self._make_portfolio(n)

        solver = SLSQP(
            atol=1e-6,
            rtol=1e-6,
            max_steps=300,
            eq_constraint_fn=eq_fn,
            ineq_constraint_fn=ineq_fn,
            n_eq_constraints=1,
            n_ineq_constraints=1,
            bounds=bounds,
            inner_solver=MinresQLPSolver(),
            proximal_tau=0.0,
        )

        sol = optx.minimise(
            objective,
            solver,
            x0,
            has_aux=True,
            throw=False,
            max_steps=300,
        )

        self._assert_iterate_stationary(
            sol.state,
            sol.value,
            eq_fn,
            ineq_fn,
            rtol_target=1e-3,
        )

    @pytest.mark.very_slow
    def test_portfolio_n500_minres_qlp_iterate_is_stationary(self):
        """``Portfolio(n=500) + MinresQLPSolver``: iterate satisfies KKT.

        Baseline (pre-fix) terminated this run with
        ``nonlinear_divergence`` because the merit oscillated even
        though primal feasibility was good.  The bound-multiplier
        recovery does not by itself fix the ``sol.result`` route
        (best-iterate rollback can still trigger), but the *returned
        iterate* should satisfy primal + dual feasibility and the
        relative stationarity ratio should be small.
        """
        n = 500
        objective, eq_fn, ineq_fn, bounds, x0 = self._make_portfolio(n)

        solver = SLSQP(
            atol=1e-6,
            rtol=1e-6,
            max_steps=400,
            eq_constraint_fn=eq_fn,
            ineq_constraint_fn=ineq_fn,
            n_eq_constraints=1,
            n_ineq_constraints=1,
            bounds=bounds,
            inner_solver=MinresQLPSolver(),
            proximal_tau=0.0,
        )

        sol = optx.minimise(
            objective,
            solver,
            x0,
            has_aux=True,
            throw=False,
            max_steps=400,
        )

        self._assert_iterate_stationary(
            sol.state,
            sol.value,
            eq_fn,
            ineq_fn,
            rtol_target=1e-2,
        )

    @pytest.mark.slow
    def test_portfolio_n5000_minres_qlp_does_not_stagnate(self):
        """Large bound-heavy regression test (n=5000) with MINRES-QLP.

        Originally reported failure mode: the iterate stalled with
        ``||∇L|| / |L|`` pinned above ``rtol`` despite constraints
        being satisfied and ``alpha = 1``.  Assert primal + dual
        feasibility and that the stationarity ratio at the returned
        iterate is below a generous target (the solver may still
        time-out or trigger best-iterate rollback before the strict
        ``rtol`` is met, but the iterate should be near-KKT).
        """
        n = 5000
        objective, eq_fn, ineq_fn, bounds, x0 = self._make_portfolio(n)

        solver = SLSQP(
            atol=1e-6,
            rtol=1e-6,
            max_steps=500,
            eq_constraint_fn=eq_fn,
            ineq_constraint_fn=ineq_fn,
            n_eq_constraints=1,
            n_ineq_constraints=1,
            bounds=bounds,
            inner_solver=MinresQLPSolver(),
            proximal_tau=0.0,
            use_exact_hvp_in_qp=True,
        )

        sol = optx.minimise(
            objective,
            solver,
            x0,
            has_aux=True,
            throw=False,
            max_steps=500,
        )

        self._assert_iterate_stationary(
            sol.state,
            sol.value,
            eq_fn,
            ineq_fn,
            rtol_target=1e-2,
            atol_target=5e-5,
        )

        # Diagnostic visibility.  ``max_proj_residual`` is the
        # high-water mark of ``||A d - b||`` after MINRES-QLP's
        # M-metric refinement; use it as a "the projection is
        # actually converging" smoke check.  No strict assertion
        # because the value depends on the inner-solver iteration
        # budget; we only print it for debugging if the test ever
        # regresses.
        diag = get_diagnostics(sol.state)
        assert jnp.isfinite(diag.max_proj_residual), f"diagnostics: {diag}"


class TestCoveragePaths:
    """Small targeted tests that exercise otherwise-uncovered code paths.

    These hit defensive guards and rarely-used user toggles that the
    end-to-end SLSQP tests do not naturally reach.  Each test is
    intentionally small so it stays in the fast suite.
    """

    def test_recover_bound_multipliers_returns_empty_when_no_bounds(self):
        """Direct call exercises the ``bounds is None`` early-return guard.

        ``_recover_bound_multipliers`` is gated by ``m_bounds_static > 0``
        in ``step()``, so the early-return inside the helper is dead
        code at runtime.  Calling it directly with ``bounds=None``
        keeps the safety guard test-covered.
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        solver = SLSQP(atol=1e-6, max_steps=5, bounds=None)
        x0 = jnp.array([1.0, 2.0])
        n = x0.shape[0]
        grad_new = 2 * x0
        eq_jac_new = jnp.zeros((0, n))
        ineq_jac_new = jnp.zeros((0, n))

        mu_lower, mu_upper = solver._recover_bound_multipliers(
            y_new=x0,
            grad_new=grad_new,
            eq_jac_new=eq_jac_new,
            ineq_jac_new=ineq_jac_new,
            mult_eq=jnp.zeros((0,)),
            mult_ineq_general=jnp.zeros((0,)),
        )
        assert mu_lower.shape == (0,)
        assert mu_upper.shape == (0,)
        assert mu_lower.dtype == x0.dtype
        assert mu_upper.dtype == x0.dtype

    def test_use_preconditioner_false(self):
        """``use_preconditioner=False`` covers the early-return path
        in ``_build_preconditioner``.

        Default is ``True``; turning it off forces the inner CG to run
        unpreconditioned and is the only way to hit the ``return None``
        branch.  The problem is intentionally trivial so the test stays
        in the fast suite.
        """

        def objective(x, args):
            return (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2, None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 3.0])

        solver = SLSQP(
            atol=1e-8,
            max_steps=30,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            use_preconditioner=False,
        )
        x0 = jnp.array([0.0, 0.0])
        y, _ = _run_solver(solver, objective, x0)
        # Equality-projected optimum: x = y = 1.5 by symmetry of the
        # objective around the constraint plane.  Wait — actually
        # ``min (x-1)^2 + (y-2)^2`` projected on ``x+y=3`` gives the
        # closest point: solve d/dx[(x-1)^2 + (3-x-2)^2] = 0
        # ⇒ 2(x-1) - 2(1-x) = 0 ⇒ x = 1, y = 2.
        np.testing.assert_allclose(y, [1.0, 2.0], atol=1e-5)

    def test_ineq_hvp_via_jvp_path(self):
        """Cover the ``ineq_hvp_contrib`` AD path in ``__check_init__``.

        Triggered when ``ineq_constraint_fn`` is provided without an
        explicit ``ineq_hvp_fn`` and an exact Lagrangian HVP is
        actually evaluated (here via ``use_exact_hvp_in_qp=True``).
        Without this exercise, the closure body
        ``def weighted(x): return jnp.dot(multipliers, ineq_con_fn(x, args))``
        and the ``jax.jvp(jax.grad(weighted), ...)`` call inside it
        are defined but never executed.
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def ineq_constraint(x, args):
            # Nonlinear so the HVP is non-trivial.
            return jnp.array([1.0 - jnp.sum(x**2)])

        solver = SLSQP(
            atol=1e-6,
            max_steps=30,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=1,
            use_exact_hvp_in_qp=True,
        )
        x0 = jnp.array([0.5, 0.5])
        y, _ = _run_solver(solver, objective, x0)
        # The unconstrained minimum is at the origin, which satisfies
        # ``1 - ||x||^2 >= 0``, so the constraint is inactive at the
        # optimum and the solver should return ``y ≈ 0``.
        np.testing.assert_allclose(y, [0.0, 0.0], atol=1e-5)
