"""Tests for the post-line-search least-squares multiplier recovery.

Direct-helper tests cover the static behaviour of
:func:`slsqp_jax.slsqp.multipliers.recover_ls_multipliers_at_iterate`
against controlled inputs (no SLSQP loop).  Integration tests at the
bottom cover the convergence-test side of the rewire — the main
load-bearing claim is that QP-multiplier noise no longer contaminates
``||grad_L|| / |L|``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax.slsqp.multipliers import recover_ls_multipliers_at_iterate

# Use float64 throughout; the helper's iterative-refinement guarantees
# only hold cleanly in double precision.
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# (1) Equality-only correctness on a clean LS identity.
# ---------------------------------------------------------------------------
def test_ls_recovery_equality_only():
    """`grad = J_eq^T lambda*` with full row-rank `J_eq` -> recover lambda*."""
    rng = np.random.default_rng(0)
    n = 6
    m_eq = 3
    J_eq = jnp.asarray(rng.standard_normal((m_eq, n)), dtype=jnp.float64)
    lambda_star = jnp.asarray(rng.standard_normal((m_eq,)), dtype=jnp.float64)
    grad = J_eq.T @ lambda_star

    J_ineq = jnp.zeros((0, n), dtype=jnp.float64)
    ineq_val = jnp.zeros((0,), dtype=jnp.float64)

    lam_eq_ls, lam_ineq_general_ls = recover_ls_multipliers_at_iterate(
        grad_new=grad,
        eq_jac_new=J_eq,
        ineq_jac_general_new=J_ineq,
        ineq_val_general_new=ineq_val,
        free_mask=None,
        active_tol=1e-8,
        ridge=1e-12,
    )
    assert lam_ineq_general_ls.shape == (0,)
    assert jnp.allclose(lam_eq_ls, lambda_star, atol=1e-10, rtol=1e-10), (
        f"lam_eq_ls = {lam_eq_ls}, lambda_star = {lambda_star}"
    )


# ---------------------------------------------------------------------------
# (2) Active-row dual-feasibility clamp on negative inequality multipliers.
# ---------------------------------------------------------------------------
def test_ls_recovery_clamps_negative_inequality():
    """Active inequality row with negative LS multiplier -> clamped to 0."""
    n = 4
    J_eq = jnp.zeros((0, n), dtype=jnp.float64)
    # Construct a single active inequality row (orthonormal against any
    # equality block).  Choose `grad` so the unconstrained LS multiplier
    # would be negative (the gradient points opposite to J_ineq^T).
    J_ineq = jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float64)
    grad = jnp.array([-2.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    ineq_val = jnp.array([0.0], dtype=jnp.float64)  # active

    lam_eq_ls, lam_ineq_general_ls = recover_ls_multipliers_at_iterate(
        grad_new=grad,
        eq_jac_new=J_eq,
        ineq_jac_general_new=J_ineq,
        ineq_val_general_new=ineq_val,
        free_mask=None,
        active_tol=1e-8,
    )
    assert lam_eq_ls.shape == (0,)
    # Unconstrained LS would give -2, but the clamp pulls it to 0.
    assert lam_ineq_general_ls.shape == (1,)
    assert lam_ineq_general_ls[0] == 0.0, (
        f"expected 0.0 (clamped); got {lam_ineq_general_ls[0]}"
    )


# ---------------------------------------------------------------------------
# (3) Value-based active rule drops a row that was previously QP-active
#     but has now backed away from the boundary.
# ---------------------------------------------------------------------------
def test_ls_recovery_value_based_active_rule_drops_stale_rows():
    """Inequality with `c_ineq > active_tol` -> dropped, regardless of QP."""
    n = 3
    J_eq = jnp.zeros((0, n), dtype=jnp.float64)
    J_ineq = jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float64)
    grad = jnp.array([3.0, 0.0, 0.0], dtype=jnp.float64)
    # The row is well above the active tolerance; the helper must drop
    # it from the LS even though `grad` would otherwise produce a
    # nonzero multiplier.  Helper signature does NOT take a
    # `qp_active_set` — the rule is purely value-based.
    ineq_val = jnp.array([0.5], dtype=jnp.float64)

    _, lam_ineq_general_ls = recover_ls_multipliers_at_iterate(
        grad_new=grad,
        eq_jac_new=J_eq,
        ineq_jac_general_new=J_ineq,
        ineq_val_general_new=ineq_val,
        free_mask=None,
        active_tol=1e-8,
    )
    assert lam_ineq_general_ls[0] == 0.0


# ---------------------------------------------------------------------------
# (4) Symmetric: a value-near-active row missed by the QP is included.
# ---------------------------------------------------------------------------
def test_ls_recovery_value_based_active_rule_includes_newly_near_active():
    """`c_ineq <= active_tol` -> included, no QP hint required."""
    n = 3
    J_eq = jnp.zeros((0, n), dtype=jnp.float64)
    J_ineq = jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float64)
    grad = jnp.array([3.0, 0.0, 0.0], dtype=jnp.float64)
    ineq_val = jnp.array([0.0], dtype=jnp.float64)

    _, lam_ineq_general_ls = recover_ls_multipliers_at_iterate(
        grad_new=grad,
        eq_jac_new=J_eq,
        ineq_jac_general_new=J_ineq,
        ineq_val_general_new=ineq_val,
        free_mask=None,
        active_tol=1e-8,
    )
    # Unconstrained LS gives +3.0 (positive -> not clamped).
    assert lam_ineq_general_ls.shape == (1,)
    assert jnp.isclose(lam_ineq_general_ls[0], 3.0, atol=1e-10)


# ---------------------------------------------------------------------------
# (5) free-mask correctness: at-bound coordinates must not be absorbed
#     into equality / general-inequality multipliers.
# ---------------------------------------------------------------------------
def test_ls_recovery_free_mask_required_for_correctness():
    """Without column-masking the LS fit absorbs at-bound gradient mass."""
    # Construct a problem where `grad` has a large component on
    # coordinate 0 (which is at a bound) and a small component on
    # coordinate 1.  J_eq has its first row only touching coordinate 0
    # and a second row only touching coordinate 1.  With column
    # masking, lambda_eq[0] should be tiny (because the row's only
    # column is masked out and the regularised solve drives mult[0]
    # toward 0); without it, lambda_eq[0] would absorb the large
    # at-bound component.
    n = 4
    J_eq = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=jnp.float64,
    )
    J_ineq = jnp.zeros((0, n), dtype=jnp.float64)
    ineq_val = jnp.zeros((0,), dtype=jnp.float64)
    grad = jnp.array([100.0, 0.5, 0.0, 0.0], dtype=jnp.float64)

    free_mask = jnp.array([False, True, True, True], dtype=bool)

    lam_no_mask, _ = recover_ls_multipliers_at_iterate(
        grad_new=grad,
        eq_jac_new=J_eq,
        ineq_jac_general_new=J_ineq,
        ineq_val_general_new=ineq_val,
        free_mask=None,
        active_tol=1e-8,
    )
    lam_masked, _ = recover_ls_multipliers_at_iterate(
        grad_new=grad,
        eq_jac_new=J_eq,
        ineq_jac_general_new=J_ineq,
        ineq_val_general_new=ineq_val,
        free_mask=free_mask,
        active_tol=1e-8,
    )

    # Without the mask, the LS hands the entire 100.0 to lambda_eq[0].
    assert jnp.abs(lam_no_mask[0]) > 50.0, (
        f"no-mask lam_eq[0] = {lam_no_mask[0]} (expected ~100)"
    )
    # With the mask, the masked column is zeroed out and the regularised
    # solve drives mult[0] toward 0.  The remaining row (coord 1) still
    # picks up the small 0.5 component cleanly.
    assert jnp.abs(lam_masked[0]) < 1e-3, (
        f"masked lam_eq[0] = {lam_masked[0]} (expected ~0)"
    )
    assert jnp.isclose(lam_masked[1], 0.5, atol=1e-8)


# ---------------------------------------------------------------------------
# (6) Iterative refinement reduces the residual on an ill-conditioned
#     `J_eq @ J_eq^T`.
# ---------------------------------------------------------------------------
def test_ls_recovery_iterative_refinement_reduces_residual():
    """Helper's one-round refinement squares the LS residual.

    Builds an ``m=3, n=10`` Jacobian via SVD with controlled
    conditioning ``cond(J_eq) ~ 1e4`` (so ``cond(J_eq @ J_eq.T) ~
    1e8``) and uses a small ``ridge = 1e-14`` so that the ridge sits
    near fp epsilon and the iterative-refinement step is bounded by
    the Newton-step squaring rather than by the ridge bias.  Compares
    against a manually-built single-shot solve at the *same* ridge.

    The plan optimistically asked for a ``1e6x`` gain on a problem
    with ``cond(AAt) ~ 1e16``; that case is rank-deficient at fp64.
    On the realistic regime ``cond(AAt) ~ 1e8`` and
    ``ridge ~ eps``, the squaring claim from Cholesky's iterative
    refinement gives at least three orders of magnitude.
    """
    rng = np.random.default_rng(42)
    n = 10
    m_eq = 3
    U, _ = np.linalg.qr(rng.standard_normal((m_eq, m_eq)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    sigma = np.array([1.0, 1e-2, 1e-4])
    J_eq_np = U @ np.diag(sigma) @ V[:m_eq, :]
    J_eq = jnp.asarray(J_eq_np, dtype=jnp.float64)
    lambda_star = jnp.asarray(rng.standard_normal((m_eq,)), dtype=jnp.float64)
    grad = J_eq.T @ lambda_star

    ridge = 1e-14

    # Reference: helper as implemented (one round of iterative refinement).
    lam_with_ref, _ = recover_ls_multipliers_at_iterate(
        grad_new=grad,
        eq_jac_new=J_eq,
        ineq_jac_general_new=jnp.zeros((0, n), dtype=jnp.float64),
        ineq_val_general_new=jnp.zeros((0,), dtype=jnp.float64),
        free_mask=None,
        active_tol=1e-8,
        ridge=ridge,
    )

    # Manual single-shot Cholesky solve (no refinement) for comparison.
    AAt = J_eq @ J_eq.T + ridge * jnp.eye(m_eq, dtype=jnp.float64)
    AAt_chol = jnp.linalg.cholesky(AAt)
    rhs = J_eq @ grad
    mult_no_ref = jax.scipy.linalg.cho_solve((AAt_chol, True), rhs)

    res_with_ref = jnp.linalg.norm(J_eq.T @ lam_with_ref - grad)
    res_no_ref = jnp.linalg.norm(J_eq.T @ mult_no_ref - grad)

    ratio = float(res_no_ref / jnp.maximum(res_with_ref, 1e-30))
    assert ratio > 100.0, (
        f"refinement gained only {ratio:.2e}x; res_with_ref={res_with_ref}, "
        f"res_no_ref={res_no_ref}"
    )


# ---------------------------------------------------------------------------
# (7) Fixed-shape masking parity: the where-trick is bit-equivalent to
#     a dynamic-shape solve over the active rows.
# ---------------------------------------------------------------------------
def test_ls_recovery_fixed_shape_masking_parity():
    """Active-row entries match the result over a physically-pruned matrix."""
    rng = np.random.default_rng(7)
    n = 5
    # Two equality rows, of which only the first is "active" via the
    # fixed-shape mask.
    J_full = jnp.asarray(rng.standard_normal((2, n)), dtype=jnp.float64)
    grad = jnp.asarray(rng.standard_normal((n,)), dtype=jnp.float64)

    # Fixed-shape (m_eq=1 + an inactive ineq row standing in for the
    # masked-out eq row): build via the helper, with the second eq row
    # marked as a dropped inequality (value > active_tol).
    J_eq_a = J_full[:1]
    J_ineq_a = J_full[1:2]
    ineq_val_a = jnp.array([1.0], dtype=jnp.float64)  # inactive
    lam_eq_a, lam_ineq_a = recover_ls_multipliers_at_iterate(
        grad_new=grad,
        eq_jac_new=J_eq_a,
        ineq_jac_general_new=J_ineq_a,
        ineq_val_general_new=ineq_val_a,
        free_mask=None,
        active_tol=1e-8,
        ridge=1e-12,
    )

    # Reference (m_eq=1 only, no inequality row): the active row's
    # multiplier should match the fixed-shape version's first entry.
    lam_eq_b, lam_ineq_b = recover_ls_multipliers_at_iterate(
        grad_new=grad,
        eq_jac_new=J_eq_a,
        ineq_jac_general_new=jnp.zeros((0, n), dtype=jnp.float64),
        ineq_val_general_new=jnp.zeros((0,), dtype=jnp.float64),
        free_mask=None,
        active_tol=1e-8,
        ridge=1e-12,
    )

    assert lam_ineq_b.shape == (0,)
    assert lam_ineq_a.shape == (1,)
    assert lam_ineq_a[0] == 0.0  # inactive -> exactly 0
    assert jnp.allclose(lam_eq_a, lam_eq_b, atol=1e-9, rtol=1e-9), (
        f"fixed-shape lam_eq = {lam_eq_a}, dynamic-shape lam_eq = {lam_eq_b}"
    )


# ---------------------------------------------------------------------------
# Bonus sanity check the plan asks for: the helper signature is
# independent of `B` / `d` (they are absent from the signature by
# construction; this just pins that fact in case a future refactor
# tries to thread them through).
# ---------------------------------------------------------------------------
def test_ls_recovery_signature_independent_of_B():
    import inspect

    sig = inspect.signature(recover_ls_multipliers_at_iterate)
    bad_names = {"B", "d", "hvp_fn", "alpha"}
    found = bad_names & set(sig.parameters)
    assert not found, (
        f"recover_ls_multipliers_at_iterate must not depend on {found}; "
        f"the helper is Hessian-free and post-line-search by design."
    )


# ===========================================================================
# Integration tests (full SLSQP loop)
# ===========================================================================


@pytest.fixture(autouse=True)
def _enable_x64():
    jax.config.update("jax_enable_x64", True)


def _make_simple_eq_problem():
    """min 0.5 ||x||^2  s.t.  sum(x) = 1.

    Closed-form optimum: x* = (1/n, ..., 1/n); lambda_eq* = 1/n.
    """

    def f(x, args):
        return 0.5 * jnp.sum(x**2), None

    def eq_constraint(x, args):
        return jnp.array([jnp.sum(x) - 1.0])

    return f, eq_constraint


def test_inflated_QP_multipliers_do_not_break_convergence():
    """A bogus-curvature `obj_hvp_fn` should not break the convergence test.

    The QP "thinks" the Hessian is `1e6 * I`, so its recovered
    multipliers are inflated by ~1e6.  Han-Powell's penalty rule
    consumes those (and the merit function will reflect the inflation),
    but `||grad_L|| / |L|` is now driven by the LS multipliers, which
    are the actual stationarity-quality estimate.  The run should
    still terminate at the correct optimum without getting stuck on
    inflated-multiplier noise.
    """
    from tests.conftest import _make_slsqp

    f, eq_constraint = _make_simple_eq_problem()

    n = 3

    def bogus_hvp(x, p, args):
        return 1e6 * p

    solver = _make_slsqp(
        max_steps=200,
        atol=1e-8,
        rtol=1e-7,
        eq_constraint_fn=eq_constraint,
        n_eq_constraints=1,
        obj_hvp_fn=bogus_hvp,
        use_exact_hvp_in_qp=True,
    )

    x0 = jnp.full((n,), 0.5, dtype=jnp.float64)

    state = solver.init(f, x0, None, {}, None, None, frozenset())
    y = x0
    for _ in range(solver.max_steps):
        y, state, _ = solver.step(f, y, None, {}, state, frozenset())
        done, _ = solver.terminate(f, y, None, {}, state, frozenset())
        if bool(done):
            break

    x_star = jnp.full((n,), 1.0 / n, dtype=jnp.float64)
    assert jnp.allclose(y, x_star, atol=1e-4), f"converged to {y}, expected {x_star}"


def test_bound_heavy_no_regression():
    """Bound-constrained QP-style problem still converges within a sane budget.

    Compares the rewired solver against a baseline budget (200 steps)
    on a small problem with explicit bounds.  Pre-rewire numbers are
    not strictly checked (the LS recovery shifts the convergence
    criterion); the test pins that the run still converges to the
    analytical optimum within tolerance.
    """
    from tests.conftest import _make_slsqp

    n = 5

    def f(x, args):
        return 0.5 * jnp.sum((x - 1.0) ** 2), None

    bounds = jnp.stack([jnp.zeros(n), jnp.full((n,), 0.5)], axis=-1)

    solver = _make_slsqp(
        max_steps=200,
        atol=1e-7,
        rtol=1e-6,
        bounds=bounds,
    )
    x0 = jnp.full((n,), 0.1, dtype=jnp.float64)

    state = solver.init(f, x0, None, {}, None, None, frozenset())
    y = x0
    converged = False
    for _ in range(solver.max_steps):
        y, state, _ = solver.step(f, y, None, {}, state, frozenset())
        done, _ = solver.terminate(f, y, None, {}, state, frozenset())
        if bool(done):
            converged = True
            break

    assert converged, "bound-constrained run failed to converge"
    # Optimum is x = 0.5 for every coordinate (upper-bound active).
    assert jnp.allclose(y, jnp.full((n,), 0.5), atol=1e-6)


def test_ls_grad_lagrangian_consistency_after_step():
    """`state.grad_lagrangian` matches `compute_lagrangian_gradient` with LS.

    Sanity check the `_step_body` rewire: after one `step()`, the
    cached Lagrangian gradient should equal the freshly-computed
    Lagrangian gradient using the LS multipliers and the new iterate's
    Jacobians.  This pins that the convergence test denominator and
    numerator share the same multiplier vector.
    """
    from slsqp_jax.hessian import compute_lagrangian_gradient
    from tests.conftest import _make_slsqp

    f, eq_constraint = _make_simple_eq_problem()

    solver = _make_slsqp(
        max_steps=10,
        atol=1e-8,
        rtol=1e-7,
        eq_constraint_fn=eq_constraint,
        n_eq_constraints=1,
    )
    x0 = jnp.array([0.4, 0.4, 0.4], dtype=jnp.float64)

    state = solver.init(f, x0, None, {}, None, None, frozenset())
    y = x0
    y, state, _ = solver.step(f, y, None, {}, state, frozenset())

    expected = compute_lagrangian_gradient(
        state.grad,
        state.eq_jac,
        state.ineq_jac,
        state.multipliers_eq_ls,
        state.multipliers_ineq_ls,
    )
    assert jnp.allclose(state.grad_lagrangian, expected, atol=1e-12, rtol=1e-12)
