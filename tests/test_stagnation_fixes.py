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
    lbfgs_init,
    lbfgs_inverse_hvp,
    lbfgs_reset,
)

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

    def test_append_after_reset_restores_uniform_diagonal(self):
        """After lbfgs_append following a reset, diagonal = gamma * 1."""
        h = self._build_nonuniform_history(5)
        assert not jnp.allclose(h.diagonal, h.gamma * jnp.ones(5))

        s = jnp.array([0.1, -0.2, 0.3, -0.1, 0.05])
        y = jnp.array([0.5, 0.3, 0.8, 0.4, 0.2])
        h2 = lbfgs_append(h, s, y)
        np.testing.assert_allclose(h2.diagonal, h2.gamma * jnp.ones(5), atol=1e-12)


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
        solver = SLSQP(
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            proximal_sigma=0.1,
            use_preconditioner=True,
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
        )
        x0 = jnp.array([0.5, 0.3, 0.1, 0.1])
        state = solver.init(objective, x0, None, {}, None, None, frozenset())

        precond_fn = solver._build_preconditioner(state)
        assert precond_fn is not None

        A_eq = state.eq_jac
        sigma = solver.proximal_sigma
        lbfgs_history = state.lbfgs_history

        key = jax.random.PRNGKey(0)
        for _ in range(5):
            key, subkey = jax.random.split(key)
            v = jax.random.normal(subkey, (n,))
            Bv = lbfgs_hvp(lbfgs_history, v) + (1.0 / sigma) * (A_eq.T @ (A_eq @ v))
            recovered = precond_fn(Bv)
            np.testing.assert_allclose(recovered, v, atol=1e-4)

    def test_no_proximal_uses_plain_inverse(self):
        """Without proximal term, preconditioner is plain B^{-1}."""

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([jnp.sum(x) - 1.0])

        solver = SLSQP(
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            proximal_sigma=0.0,
            use_preconditioner=True,
            rtol=1e-8,
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
            proximal_sigma=0.1,
            qp_max_iter=1,  # force QP failure
            qp_max_cg_iter=1,
            rtol=1e-8,
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
            rtol=1e-4,
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

    def test_eq_constrained_with_proximal(self):
        """Equality-constrained problem with proximal_sigma that used to stagnate."""

        def objective(x, args):
            return (x[0] - 1.0) ** 2 + (x[1] - 2.5) ** 2, None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 3.0])

        solver = SLSQP(
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            proximal_sigma=0.1,
            use_preconditioner=True,
            rtol=1e-6,
            atol=1e-6,
            max_steps=100,
        )
        x0 = jnp.array([0.0, 0.0])
        y, state = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(y[0] + y[1], 3.0, atol=1e-4)
        assert float(state.f_val) < 1.0

    def test_bounded_eq_constrained_with_proximal(self):
        """Bounded + equality-constrained problem with proximal sigma."""

        def objective(x, args):
            return jnp.sum((x - jnp.array([1.0, 2.0, 0.5])) ** 2), None

        def eq_constraint(x, args):
            return jnp.array([jnp.sum(x) - 3.0])

        bounds = jnp.array([[0.0, 5.0], [0.0, 5.0], [0.0, 5.0]])
        solver = SLSQP(
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            bounds=bounds,
            proximal_sigma=0.1,
            use_preconditioner=True,
            rtol=1e-6,
            atol=1e-6,
            max_steps=100,
        )
        x0 = jnp.array([1.0, 1.0, 1.0])
        y, state = _run_solver(solver, objective, x0)
        np.testing.assert_allclose(jnp.sum(y), 3.0, atol=1e-4)

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
            rtol=1e-5,
            atol=1e-5,
            max_steps=200,
        )
        y, state = _run_solver(solver, objective, x0)

        np.testing.assert_allclose(jnp.sum(y), 5.0, atol=1e-3)
        assert state.stagnation_count < solver.stagnation_patience
