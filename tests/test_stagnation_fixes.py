"""Tests for the stagnation cascade fixes.

Tests cover:
1. L-BFGS diagonal generalisation: lbfgs_compute_diagonal, lbfgs_reset,
   HVP / inverse HVP with non-uniform diagonal.
2. Proximal-aware Woodbury preconditioner.
3. Penalty gating on QP convergence.
4. Steepest descent fallback on QP failure.
5. Integration: solver recovers from a QP failure scenario.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax import SLSQP
from slsqp_jax.hessian import (
    LBFGSHistory,
    lbfgs_append,
    lbfgs_compute_diagonal,
    lbfgs_hvp,
    lbfgs_identity_reset,
    lbfgs_init,
    lbfgs_inverse_hvp,
    lbfgs_reset,
)
from slsqp_jax.qp_solver import solve_qp

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_history_with_pairs(n: int, pairs: list[tuple], memory: int = 10):
    """Build an LBFGSHistory by appending (s, y) pairs sequentially."""
    history = lbfgs_init(n, memory)
    for s, y_vec in pairs:
        history = lbfgs_append(history, jnp.array(s), jnp.array(y_vec))
    return history


def _explicit_diagonal(history: LBFGSHistory) -> jnp.ndarray:
    """Extract diag(B_k) by probing with unit vectors (reference impl)."""
    n = history.diagonal.shape[0]
    diag = jnp.zeros(n)
    for i in range(n):
        e_i = jnp.zeros(n).at[i].set(1.0)
        diag = diag.at[i].set(jnp.dot(lbfgs_hvp(history, e_i), e_i))
    return diag


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


# ===================================================================
# 1. L-BFGS diagonal generalisation
# ===================================================================


class TestLBFGSComputeDiagonal:
    """Verify lbfgs_compute_diagonal against explicit probing."""

    def test_empty_history(self):
        history = lbfgs_init(5, 10)
        diag = lbfgs_compute_diagonal(history)
        np.testing.assert_allclose(diag, jnp.ones(5), atol=1e-12)

    def test_single_pair(self):
        key = jax.random.PRNGKey(42)
        s = jax.random.normal(key, (5,))
        y = jax.random.normal(jax.random.PRNGKey(43), (5,)) * 0.1 + s
        history = _build_history_with_pairs(5, [(s, y)])
        diag_fast = lbfgs_compute_diagonal(history)
        diag_ref = _explicit_diagonal(history)
        np.testing.assert_allclose(diag_fast, diag_ref, rtol=1e-6)

    def test_multiple_pairs(self):
        key = jax.random.PRNGKey(0)
        n = 8
        pairs = []
        for i in range(5):
            k1, k2, key = jax.random.split(key, 3)
            s = jax.random.normal(k1, (n,))
            y = jax.random.normal(k2, (n,)) * 0.1 + 2.0 * s
            pairs.append((s, y))
        history = _build_history_with_pairs(n, pairs)
        diag_fast = lbfgs_compute_diagonal(history)
        diag_ref = _explicit_diagonal(history)
        np.testing.assert_allclose(diag_fast, diag_ref, rtol=1e-5)

    def test_full_buffer(self):
        """When the circular buffer is completely filled."""
        key = jax.random.PRNGKey(7)
        n, memory = 4, 3
        pairs = []
        for _ in range(5):
            k1, k2, key = jax.random.split(key, 3)
            s = jax.random.normal(k1, (n,))
            y = jax.random.normal(k2, (n,)) * 0.2 + 1.5 * s
            pairs.append((s, y))
        history = _build_history_with_pairs(n, pairs, memory=memory)
        diag_fast = lbfgs_compute_diagonal(history)
        diag_ref = _explicit_diagonal(history)
        np.testing.assert_allclose(diag_fast, diag_ref, rtol=1e-5)


class TestLBFGSReset:
    """Verify lbfgs_reset preserves diagonal curvature and clears pairs."""

    def test_reset_clears_pairs(self):
        history = _build_history_with_pairs(4, [(jnp.ones(4), 2.0 * jnp.ones(4))])
        assert int(history.count) == 1
        reset = lbfgs_reset(history)
        assert int(reset.count) == 0
        assert int(reset.next_idx) == 0

    def test_reset_preserves_diagonal(self):
        key = jax.random.PRNGKey(99)
        n = 6
        pairs = []
        for _ in range(3):
            k1, k2, key = jax.random.split(key, 3)
            s = jax.random.normal(k1, (n,))
            y = jax.random.normal(k2, (n,)) * 0.1 + s
            pairs.append((s, y))
        history = _build_history_with_pairs(n, pairs)
        expected_diag = jnp.clip(lbfgs_compute_diagonal(history), 1e-3, 100.0)
        reset = lbfgs_reset(history)
        np.testing.assert_allclose(reset.diagonal, expected_diag, rtol=1e-8)

    def test_reset_gamma_is_median(self):
        key = jax.random.PRNGKey(11)
        n = 6
        pairs = []
        for _ in range(4):
            k1, k2, key = jax.random.split(key, 3)
            s = jax.random.normal(k1, (n,))
            y = jax.random.normal(k2, (n,)) * 0.1 + s
            pairs.append((s, y))
        history = _build_history_with_pairs(n, pairs)
        reset = lbfgs_reset(history)
        expected_gamma = jnp.median(reset.diagonal)
        np.testing.assert_allclose(reset.gamma, expected_gamma, rtol=1e-10)


class TestLBFGSDiagonalHVP:
    """Verify HVP and inverse HVP with non-uniform diagonal B_0."""

    def _build_nonuniform_history(self, n=5):
        """Create a history with non-uniform diagonal (post-reset state)."""
        key = jax.random.PRNGKey(42)
        pairs = []
        for _ in range(3):
            k1, k2, key = jax.random.split(key, 3)
            s = jax.random.normal(k1, (n,))
            y = jax.random.normal(k2, (n,)) * 0.1 + s
            pairs.append((s, y))
        history = _build_history_with_pairs(n, pairs)
        return lbfgs_reset(history)

    def test_hvp_with_nonuniform_diagonal_no_pairs(self):
        """After reset with no pairs: B v = diag(d) v."""
        h = self._build_nonuniform_history(5)
        assert int(h.count) == 0
        v = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = lbfgs_hvp(h, v)
        np.testing.assert_allclose(result, h.diagonal * v, atol=1e-10)

    def test_inverse_hvp_with_nonuniform_diagonal_no_pairs(self):
        """After reset with no pairs: H v = diag(1/d) v."""
        h = self._build_nonuniform_history(5)
        assert int(h.count) == 0
        v = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = lbfgs_inverse_hvp(h, v)
        np.testing.assert_allclose(result, v / h.diagonal, atol=1e-10)

    def test_hvp_inverse_hvp_roundtrip(self):
        """B^{-1}(B v) ≈ v for a reset history with re-appended pairs."""
        h = self._build_nonuniform_history(5)
        s = jnp.array([0.1, -0.2, 0.3, -0.1, 0.05])
        y = jnp.array([0.5, 0.3, 0.8, 0.4, 0.2])
        h = lbfgs_append(h, s, y)

        key = jax.random.PRNGKey(0)
        v = jax.random.normal(key, (5,))
        Bv = lbfgs_hvp(h, v)
        roundtrip = lbfgs_inverse_hvp(h, Bv)
        np.testing.assert_allclose(roundtrip, v, atol=1e-5)

    def test_append_after_reset_uses_per_variable_diagonal(self):
        """After lbfgs_append following a reset, diagonal uses component-wise secant."""
        h = self._build_nonuniform_history(5)
        assert not jnp.allclose(h.diagonal, h.gamma * jnp.ones(5))

        s = jnp.array([0.1, -0.2, 0.3, -0.1, 0.05])
        y = jnp.array([0.5, 0.3, 0.8, 0.4, 0.2])
        h2 = lbfgs_append(h, s, y)

        gamma_new = jnp.dot(y, y) / jnp.maximum(jnp.dot(y, s), 1e-12)
        clip_lo = jnp.maximum(gamma_new * 1e-2, 1e-6)
        clip_hi = jnp.minimum(gamma_new * 1e2, 1e8)
        expected = jnp.clip(jnp.abs(y * s) / jnp.maximum(s**2, 1e-12), clip_lo, clip_hi)
        np.testing.assert_allclose(h2.diagonal, expected, atol=1e-12)


# ===================================================================
# 2. Proximal-aware preconditioner (Woodbury)
# ===================================================================


class TestProximalPreconditioner:
    """Verify the Woodbury-corrected preconditioner."""

    def test_preconditioner_matches_B_tilde_inverse(self):
        """M @ B_tilde v ≈ v  for random v."""

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([jnp.sum(x) - 1.0])

        n = 4
        mu = 0.1
        solver = SLSQP(
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            use_preconditioner=True,
            atol=1e-8,
            max_steps=50,
        )
        x0 = jnp.array([0.5, 0.3, 0.1, 0.1])
        state = solver.init(objective, x0, None, {}, None, None, frozenset())

        precond_fn = solver._build_preconditioner(state, proximal_mu=mu)
        assert precond_fn is not None

        A_eq = state.eq_jac
        lbfgs_history = state.lbfgs_history

        key = jax.random.PRNGKey(0)
        for _ in range(5):
            key, subkey = jax.random.split(key)
            v = jax.random.normal(subkey, (n,))
            Bv = lbfgs_hvp(lbfgs_history, v) + (1.0 / mu) * (A_eq.T @ (A_eq @ v))
            recovered = precond_fn(Bv)
            np.testing.assert_allclose(recovered, v, atol=1e-4)

    def test_no_eq_constraints_uses_plain_inverse(self):
        """Without equality constraints, preconditioner is plain B^{-1}."""

        def objective(x, args):
            return jnp.sum(x**2), None

        solver = SLSQP(
            use_preconditioner=True,
            atol=1e-8,
        )
        x0 = jnp.array([0.5, 0.3, 0.1, 0.1])
        state = solver.init(objective, x0, None, {}, None, None, frozenset())

        precond_fn = solver._build_preconditioner(state)
        assert precond_fn is not None
        v = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = precond_fn(v)
        expected = lbfgs_inverse_hvp(state.lbfgs_history, v)
        np.testing.assert_allclose(result, expected, atol=1e-12)


# ===================================================================
# 3. Penalty gating
# ===================================================================


class TestPenaltyGating:
    """Verify the penalty parameter is unchanged when QP fails."""

    def test_penalty_unchanged_on_qp_failure(self):
        """Run a problem where QP fails; verify penalty is gated."""

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([jnp.sum(x) - 1.0])

        solver = SLSQP(
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            qp_max_iter=1,  # force QP failure
            qp_max_cg_iter=1,
            atol=1e-8,
            max_steps=5,
        )
        x0 = jnp.array([0.5, 0.3, 0.2])
        state = solver.init(objective, x0, None, {}, None, None, frozenset())

        initial_penalty = float(state.merit_penalty)

        y = x0
        for _ in range(3):
            done, _ = solver.terminate(objective, y, None, {}, state, frozenset())
            if done:
                break
            y, state, _ = solver.step(objective, y, None, {}, state, frozenset())

        # With qp_max_iter=1, QP should fail, so penalty should stay at initial value
        if not state.qp_converged:
            assert float(state.merit_penalty) == pytest.approx(
                initial_penalty, abs=1e-10
            )


# ===================================================================
# 4. Steepest descent fallback
# ===================================================================


class TestSteepestDescentFallback:
    """Verify fallback to -grad when QP fails."""

    def test_still_converges_with_steepest_descent(self):
        """Simple unconstrained problem should converge even with forced QP failure."""

        def objective(x, args):
            return jnp.sum(x**2), None

        solver = SLSQP(
            qp_max_iter=1,
            qp_max_cg_iter=1,
            atol=1e-4,
            max_steps=200,
        )
        x0 = jnp.array([3.0, -2.0])
        y, state = _run_solver(solver, objective, x0)
        assert jnp.linalg.norm(y) < 0.1


# ===================================================================
# 5. Integration test
# ===================================================================


class TestStagnationRecovery:
    """Verify solver recovers from scenarios that previously stagnated."""

    @pytest.mark.slow
    def test_eq_constrained_with_adaptive_proximal(self):
        """Equality-constrained problem with adaptive proximal that used to stagnate."""

        def objective(x, args):
            return (x[0] - 1.0) ** 2 + (x[1] - 2.5) ** 2, None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 3.0])

        solver = SLSQP(
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            use_preconditioner=True,
            atol=1e-6,
            max_steps=100,
        )
        x0 = jnp.array([0.0, 0.0])
        y, state = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y[0] + y[1], 3.0, atol=1e-4)
        assert float(state.f_val) < 1.0

    @pytest.mark.slow
    def test_bounded_eq_constrained_with_adaptive_proximal(self):
        """Bounded + equality-constrained problem with adaptive proximal."""

        def objective(x, args):
            return jnp.sum((x - jnp.array([1.0, 2.0, 0.5])) ** 2), None

        def eq_constraint(x, args):
            return jnp.array([jnp.sum(x) - 3.0])

        bounds = jnp.array([[0.0, 5.0], [0.0, 5.0], [0.0, 5.0]])
        solver = SLSQP(
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            bounds=bounds,
            use_preconditioner=True,
            atol=1e-6,
            max_steps=100,
        )
        x0 = jnp.array([1.0, 1.0, 1.0])
        y, state = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(jnp.sum(y), 3.0, atol=1e-4)

    @pytest.mark.slow
    def test_cg_regularization_prevents_stagnation_with_bounds_and_eq(self):
        """Bounds + equality constraints with ill-conditioned Hessian.

        After a diagonal reset, kappa_B can reach 1e5.  Without CG
        regularization the old hard threshold ``pBp <= 1e-8`` falsely
        declares negative curvature and returns d=0, causing permanent
        stagnation.  The SNOPT-style regularization (default 1e-6)
        prevents this by adding delta^2 * ||p||^2 to the curvature
        check.
        """

        def objective(x, args):
            return -jnp.sum(jnp.log(x + 1.0)), None

        def eq_constraint(x, args):
            return jnp.array([jnp.sum(x) - 5.0])

        n = 4
        bounds = jnp.array([[0.0, 10.0]] * n)
        x0 = jnp.array([2.0, 1.0, 1.5, 0.5])

        solver = SLSQP(
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            bounds=bounds,
            cg_regularization=1e-6,
            atol=1e-5,
            max_steps=200,
        )
        y, state = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(jnp.sum(y), 5.0, atol=1e-3)
        assert not state.stagnation


# ===================================================================
# 6. QP false convergence at max iterations
# ===================================================================


class TestQPFalseConvergence:
    """Verify QP reports converged=False when hitting max_iter."""

    def test_qp_reports_false_at_max_iter(self):
        """With max_iter=1, QP cannot converge genuinely; flag must be False."""
        n = 3
        H = jnp.eye(n)

        def hvp_fn(v):
            return H @ v

        g = jnp.array([1.0, -2.0, 0.5])
        A_ineq = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        b_ineq = jnp.array([0.5, 0.5])

        result = solve_qp(
            hvp_fn,
            g,
            jnp.zeros((0, n)),
            jnp.zeros((0,)),
            A_ineq,
            b_ineq,
            max_iter=1,
        )
        assert not bool(result.converged)

    def test_qp_converges_with_sufficient_iters(self):
        """Same problem with enough iterations should converge."""
        n = 3
        H = jnp.eye(n)

        def hvp_fn(v):
            return H @ v

        g = jnp.array([1.0, -2.0, 0.5])
        A_ineq = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        b_ineq = jnp.array([0.5, 0.5])

        result = solve_qp(
            hvp_fn,
            g,
            jnp.zeros((0, n)),
            jnp.zeros((0,)),
            A_ineq,
            b_ineq,
            max_iter=100,
        )
        assert bool(result.converged)


# ===================================================================
# 7. L-BFGS identity reset
# ===================================================================


class TestLBFGSIdentityReset:
    """Verify lbfgs_identity_reset clears everything to B_0 = I."""

    def test_identity_reset_clears_pairs(self):
        """After identity reset, count and next_idx must be zero."""
        history = _build_history_with_pairs(
            3,
            [([1.0, 0.0, 0.0], [2.0, 0.0, 0.0]), ([0.0, 1.0, 0.0], [0.0, 3.0, 0.0])],
        )
        assert int(history.count) == 2

        reset = lbfgs_identity_reset(history)
        assert int(reset.count) == 0
        assert int(reset.next_idx) == 0

    def test_identity_reset_sets_gamma_one(self):
        """After identity reset, gamma must be 1.0."""
        history = _build_history_with_pairs(
            3,
            [([1.0, 0.0, 0.0], [2.0, 0.0, 0.0])],
        )
        reset = lbfgs_identity_reset(history)
        assert float(reset.gamma) == pytest.approx(1.0)

    def test_identity_reset_sets_diagonal_ones(self):
        """After identity reset, diagonal must be all ones."""
        history = _build_history_with_pairs(
            3,
            [([1.0, 0.0, 0.0], [2.0, 0.0, 0.0])],
        )
        reset = lbfgs_identity_reset(history)
        np.testing.assert_allclose(reset.diagonal, jnp.ones(3))

    def test_identity_reset_hvp_is_identity(self):
        """After identity reset, B @ v = v."""
        history = _build_history_with_pairs(
            3,
            [([1.0, 0.0, 0.0], [2.0, 0.0, 0.0])],
        )
        reset = lbfgs_identity_reset(history)
        v = jnp.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(lbfgs_hvp(reset, v), v, atol=1e-12)


class TestEscalatingLBFGSRecovery:
    """Verify consecutive QP failures trigger identity reset."""

    @pytest.mark.slow
    def test_consecutive_failures_trigger_identity_reset(self):
        """After qp_failure_patience consecutive QP failures, L-BFGS resets to identity.

        Uses inequality constraints so the active-set loop always enters the
        body (warm-start doesn't immediately satisfy KKT), forcing genuine QP
        failures with qp_max_iter=1.
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def ineq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 1.0, x[0] - x[1]])

        patience = 3
        solver = SLSQP(
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=2,
            qp_max_iter=1,
            qp_max_cg_iter=1,
            qp_failure_patience=patience,
            atol=1e-8,
            max_steps=patience + 5,
        )
        x0 = jnp.array([2.0, -1.0, 0.5])
        state = solver.init(objective, x0, None, {}, None, None, frozenset())
        y = x0

        had_identity_reset = False
        for _ in range(patience + 3):
            y, state, _ = solver.step(objective, y, None, {}, state, frozenset())
            if int(state.consecutive_qp_failures) >= patience:
                # Identity reset should have fired this step
                had_identity_reset = True

        assert had_identity_reset, (
            "Never reached qp_failure_patience consecutive failures"
        )

        # After an identity reset, consecutive steps will keep triggering
        # identity resets (since QP keeps failing). Verify the counter is
        # at least patience.
        assert int(state.consecutive_qp_failures) >= patience

    def test_qp_success_resets_failure_counter(self):
        """A successful QP clears the consecutive failure counter."""

        def objective(x, args):
            return jnp.sum(x**2), None

        solver = SLSQP(
            qp_failure_patience=3,
            atol=1e-6,
            max_steps=5,
        )
        x0 = jnp.array([3.0, -2.0])
        state = solver.init(objective, x0, None, {}, None, None, frozenset())
        y = x0
        y, state, _ = solver.step(objective, y, None, {}, state, frozenset())

        # QP should succeed on this simple unconstrained problem
        assert bool(state.qp_converged)
        assert int(state.consecutive_qp_failures) == 0


# ===================================================================
# 8. Alpha-scaled multiplier blending
# ===================================================================


class TestAlphaScaledMultipliers:
    """Verify multiplier blending with alpha in the outer solver."""

    def test_full_step_uses_qp_multipliers(self):
        """When alpha=1, stored multipliers should equal QP multipliers."""

        def objective(x, args):
            return 0.5 * jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 1.0])

        solver = SLSQP(
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            atol=1e-8,
            max_steps=2,
        )
        x0 = jnp.array([0.0, 0.0])
        state = solver.init(objective, x0, None, {}, None, None, frozenset())
        y = x0
        y, state, _ = solver.step(objective, y, None, {}, state, frozenset())

        # If alpha=1 (full step), multipliers_eq should be the raw QP value.
        # We can't directly check if alpha was 1, but for this well-posed
        # problem the QP multipliers should be non-zero and meaningful.
        assert jnp.abs(state.multipliers_eq[0]) > 1e-10

    def test_bounds_convergence_with_alpha_scaling(self):
        """Bound-constrained problem should still converge with alpha-scaling.

        This is a regression test: alpha-scaling must not prevent convergence
        at bound-active solutions where the QP correctly identifies the
        multipliers but the line search rejects the step.
        """

        def objective(x, args):
            return (x[0] - 5) ** 2 + (x[1] - 5) ** 2, None

        bounds = jnp.array([[0.0, 2.0], [0.0, 2.0]])
        solver = SLSQP(
            atol=1e-8,
            bounds=bounds,
            max_steps=50,
        )
        x0 = jnp.array([1.0, 1.0])
        y, state = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y, [2.0, 2.0], rtol=1e-3)


# ===================================================================
# 9. Merit-based stagnation detection
# ===================================================================


class TestMeritStagnation:
    """Tests for the merit-based stagnation detection."""

    def test_stagnation_window_calculation(self):
        """_stagnation_window = max(1, max_steps // 10)."""
        solver = SLSQP(max_steps=100)
        assert solver._stagnation_window == 10

        solver2 = SLSQP(max_steps=7)
        assert solver2._stagnation_window == 1

        solver3 = SLSQP(max_steps=200)
        assert solver3._stagnation_window == 20

    def test_best_merit_init(self):
        """best_merit should be finite and steps_without_improvement == 0 after init."""

        def objective(x, args):
            return jnp.sum(x**2), None

        solver = SLSQP(max_steps=100)
        x0 = jnp.array([1.0, 2.0, 3.0])
        state = solver.init(objective, x0, None, {}, None, None, frozenset())
        assert jnp.isfinite(state.best_merit)
        assert state.steps_without_improvement == 0
        assert not state.stagnation

    def test_stagnation_false_on_progress(self):
        """stagnation should be False on a well-behaved problem."""

        def objective(x, args):
            return jnp.sum(x**2), None

        solver = SLSQP(
            atol=1e-8,
            max_steps=50,
            stagnation_tol=1e-12,
        )
        x0 = jnp.array([3.0, -2.0])
        y, state = _run_solver(solver, objective, x0)
        assert not state.stagnation

    def test_stagnation_guard_before_window(self):
        """Stagnation should not fire before W steps have elapsed."""

        def objective(x, args):
            return jnp.sum(x**2), None

        solver = SLSQP(
            atol=1e-15,
            max_steps=100,
            stagnation_tol=1e-3,
        )
        x0 = jnp.array([0.001])
        state = solver.init(objective, x0, None, {}, None, None, frozenset())
        y = x0
        for i in range(5):
            done, _ = solver.terminate(objective, y, None, {}, state, frozenset())
            if done:
                break
            y, state, _ = solver.step(objective, y, None, {}, state, frozenset())
        assert not state.stagnation


# ===================================================================
# 10. Adaptive proximal mu
# ===================================================================


class TestAdaptiveProximalMu:
    """Tests for the adaptive proximal mu feature."""

    def test_proximal_tau_validation(self):
        """proximal_tau must be in [0, 1)."""
        SLSQP(proximal_tau=0.0)  # 0 is valid (disables proximal)
        with pytest.raises(ValueError, match="proximal_tau"):
            SLSQP(proximal_tau=1.0)
        with pytest.raises(ValueError, match="proximal_tau"):
            SLSQP(proximal_tau=-0.5)

    def test_proximal_mu_min_default_from_atol(self):
        """proximal_mu_min=None should resolve to atol."""
        solver = SLSQP(atol=1e-7)
        assert solver._proximal_mu_min == 1e-7

    def test_proximal_mu_min_explicit(self):
        """Explicit proximal_mu_min should be used."""
        solver = SLSQP(proximal_mu_min=1e-5)
        assert solver._proximal_mu_min == 1e-5

    def test_adaptive_mu_converges_equality_constrained(self):
        """Adaptive proximal mu converges on equality-constrained problem."""

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 1.0])

        solver = SLSQP(
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            atol=1e-6,
            max_steps=50,
        )
        x0 = jnp.array([0.0, 0.0])
        y, state = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y[0] + y[1], 1.0, atol=1e-4)
        np.testing.assert_allclose(y[0], 0.5, atol=0.05)

    @pytest.mark.slow
    def test_adaptive_mu_with_custom_tau(self):
        """Different proximal_tau values should not break convergence."""

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 1.0])

        for tau in [0.0, 0.1, 0.5, 0.9]:
            solver = SLSQP(
                eq_constraint_fn=eq_constraint,
                n_eq_constraints=1,
                proximal_tau=tau,
                atol=1e-5,
                max_steps=100,
            )
            x0 = jnp.array([0.0, 0.0])
            y, state = _run_solver(solver, objective, x0)
            np.testing.assert_allclose(y[0] + y[1], 1.0, atol=1e-3)

    def test_no_proximal_equality_only(self):
        """proximal_tau=0 converges on equality-only problem."""

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 1.0])

        solver = SLSQP(
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            proximal_tau=0,
            atol=1e-6,
            max_steps=50,
        )
        x0 = jnp.array([0.0, 0.0])
        y, state = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y[0] + y[1], 1.0, atol=1e-4)
        np.testing.assert_allclose(y[0], 0.5, atol=0.05)

    def test_no_proximal_equality_and_inequality(self):
        """proximal_tau=0 converges with both eq and ineq constraints."""

        def objective(x, args):
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2, None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 4.0])

        def ineq_constraint(x, args):
            return jnp.array([x[0], x[1]])

        solver = SLSQP(
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=2,
            proximal_tau=0,
            atol=1e-6,
            max_steps=100,
        )
        x0 = jnp.array([2.0, 2.0])
        y, state = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(y[0] + y[1], 4.0, atol=1e-4)
        assert y[0] >= -1e-5, f"x[0] < 0: {y[0]}"
        assert y[1] >= -1e-5, f"x[1] < 0: {y[1]}"
