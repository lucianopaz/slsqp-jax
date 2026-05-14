"""Hessian-free least-squares multiplier recovery at the accepted iterate.

This module implements the post-line-search least-squares multiplier
estimate used for the L-BFGS secant pair and the convergence-test
Lagrangian:

    lambda_LS(x_{k+1}) = (J(x_{k+1}) J(x_{k+1})^T)^{-1} J(x_{k+1}) grad(x_{k+1})

where ``J = [J_eq; J_ineq_general[active]]`` with the active set
defined *purely by value* at the new iterate.  The recovery is
independent of the QP Hessian approximation ``B``, of the search
direction ``d``, and of the line-search step size ``alpha``.  It is the
multiplier vector that minimises the actual stationarity residual at
``x_{k+1}``.

The Han-Powell merit-penalty rule, the LPEC-A predictor, and the QP
active-set warm-start continue to consume the QP-recovered multipliers
``lambda_QP`` (carried on the renamed ``state.multipliers_*_qp`` fields).
The asymmetric routing is the load-bearing point of this module: it
prevents QP-multiplier noise (typically ``O(s_f / s_eq * cond(B))``
when auto-scaling is on) from contaminating the convergence test, which
otherwise produces dramatic ``||grad_L||/|L|`` swing-ups even when the
iterate is near a true KKT point.

Bound multipliers are *not* recovered by this module — they continue to
come from :func:`slsqp_jax.slsqp.bounds.recover_bound_multipliers`,
which is itself already a post-step LS recovery from the partial
Lagrangian gradient at ``x_{k+1}``.  This module only handles the
equality block and the general-inequality block; the bound block of
``state.multipliers_ineq_ls`` is filled in by the existing bound
recovery path, which is fed *this module's* equality / general-inequality
multipliers (instead of the alpha-blended QP multipliers it consumed
before) so the whole stationarity-side multiplier vector is consistent.

See ``AGENTS.md`` (Multiplier stability section) for the full motivation
and the QP-vs-LS routing matrix.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jaxtyping import Array, Bool, Float, jaxtyped

from slsqp_jax.types import Vector


@jaxtyped(typechecker=beartype)
def compute_at_lower_mask(
    y: Vector,
    bounds: Float[Array, "n 2"] | None,
    lower_indices: tuple[int, ...],
) -> Bool[Array, " n"]:
    """Per-coordinate at-lower-bound mask at ``x_{k+1}``.

    Mirrors the predicate inside
    :func:`slsqp_jax.slsqp.bounds.recover_bound_multipliers` so both
    call sites use the identical test:
    ``(y - lb) <= 1e-12 + eps * (1 + |y|)``.

    Returns an all-``False`` mask when ``bounds`` is ``None`` or no
    finite lower bounds exist.  The mask covers all ``n`` variables
    (entries for variables without a finite lower bound stay ``False``).
    """
    n = y.shape[0]
    if bounds is None or len(lower_indices) == 0:
        return jnp.zeros((n,), dtype=bool)
    eps = jnp.asarray(jnp.finfo(y.dtype).eps, dtype=y.dtype)
    bound_tol = jnp.asarray(1e-12, dtype=y.dtype) + eps * (1.0 + jnp.abs(y))
    lower_idx = jnp.asarray(np.asarray(lower_indices, dtype=np.int32))
    lb = bounds[lower_idx, 0]
    at_lower_local = (y[lower_idx] - lb) <= bound_tol[lower_idx]
    mask = jnp.zeros((n,), dtype=bool)
    return mask.at[lower_idx].set(at_lower_local)


@jaxtyped(typechecker=beartype)
def compute_at_upper_mask(
    y: Vector,
    bounds: Float[Array, "n 2"] | None,
    upper_indices: tuple[int, ...],
) -> Bool[Array, " n"]:
    """Per-coordinate at-upper-bound mask at ``x_{k+1}``.

    Symmetric to :func:`compute_at_lower_mask`; uses ``(ub - y) <=
    1e-12 + eps * (1 + |y|)``.
    """
    n = y.shape[0]
    if bounds is None or len(upper_indices) == 0:
        return jnp.zeros((n,), dtype=bool)
    eps = jnp.asarray(jnp.finfo(y.dtype).eps, dtype=y.dtype)
    bound_tol = jnp.asarray(1e-12, dtype=y.dtype) + eps * (1.0 + jnp.abs(y))
    upper_idx = jnp.asarray(np.asarray(upper_indices, dtype=np.int32))
    ub = bounds[upper_idx, 1]
    at_upper_local = (ub - y[upper_idx]) <= bound_tol[upper_idx]
    mask = jnp.zeros((n,), dtype=bool)
    return mask.at[upper_idx].set(at_upper_local)


def recover_ls_multipliers_at_iterate(
    grad_new: Vector,
    eq_jac_new: Float[Array, "m_eq n"],
    ineq_jac_general_new: Float[Array, "m_ineq_general n"],
    ineq_val_general_new: Float[Array, " m_ineq_general"],
    free_mask: Bool[Array, " n"] | None = None,
    active_tol: float = 1e-8,
    ridge: float = 1e-8,
) -> tuple[Float[Array, " m_eq"], Float[Array, " m_ineq_general"]]:
    """Least-squares multiplier estimate at the accepted iterate.

    Returns ``(lambda_eq_ls, lambda_ineq_general_ls)`` — the multiplier
    vectors for the equality block and the *general* (non-bound) part
    of the inequality block.  The bound block is *not* recovered here;
    callers must pass it through
    :func:`slsqp_jax.slsqp.bounds.recover_bound_multipliers` (already
    a post-step LS) using the equality / general-inequality multipliers
    returned by this function.

    Active-set rule
    ---------------
    The general-inequality active set is *value-based at* ``x_{k+1}``:

        ``active_new[i] = (ineq_val_general_new[i] <= active_tol)``

    The QP active set is intentionally not consumed: it is computed at
    ``x_k``'s linearisation and is stale at ``x_{k+1}`` after an
    ``alpha < 1`` line search.  Pinning stale rows would inflate their
    multipliers; symmetrically, value-near-active rows the QP missed
    would be dropped.  This mirrors the existing bound-recovery
    convention, which uses ``(y_new - lb) <= ...`` purely on the new
    iterate's bound values.

    Equality rows are always active.  Inactive inequality slots are
    returned as exactly ``0`` (the helper does not invent multipliers
    for inactive constraints).

    Fixed-shape masking pattern (JIT-compatible)
    --------------------------------------------
    Mirrors the projector-construction prefix of
    :func:`slsqp_jax.inner.cholesky._make_cholesky_projection_ctx`:
    inactive rows of the stacked Jacobian ``A = [J_eq; J_ineq_general]``
    are zeroed out, and the corresponding diagonal entries of ``A Aᵀ``
    are replaced by ``1`` (not ``0``) so the Cholesky stays SPD at
    fixed shape.  The ``ridge * I`` perturbation handles near-rank-
    deficient active rows; one round of iterative refinement absorbs
    the resulting ``O(eps · cond(A Aᵀ))`` bias.

    ``free_mask`` (column mask)
    ---------------------------
    When bounds are present the caller MUST pass ``free_mask = ~(at_lower
    | at_upper)`` so the LS fit is restricted to the free subspace.
    Without column-masking, ``A`` columns at at-bound coordinates would
    let the LS absorb the gradient component on those coordinates into
    equality / general-inequality multipliers (those gradient
    components belong to the subsequent bound-multiplier recovery via
    :func:`slsqp_jax.slsqp.bounds.recover_bound_multipliers`).

    Pass ``free_mask=None`` only when no bound constraints exist.

    Dual-feasibility clamp
    ----------------------
    Active-row inequality multipliers are clamped at ``max(0, ·)`` —
    a first-pass approximation of dual feasibility.  Equality
    multipliers are signed (no clamp).  The clamp is applied
    uniformly: when the unconstrained LS multiplier of an active
    inequality row is negative, the residual gradient component the
    negative multiplier was algebraically absorbing remains in
    ``||grad_L||``, just no longer attributed to the wrong-sign row.
    The convergence test becomes slightly more conservative as a
    result.  For affine constraints (the dominant case here:
    bounds linear, most user inequalities near-affine) the residual
    is small.  See the *Defended design points* section of the
    decoupling plan for the upgrade path to a fixed-shape
    prune-and-resolve when the simple clamp is insufficient.

    Cost
    ----
    One ``m × m`` Cholesky factorisation + two back-solves + one
    iterative-refinement back-solve per SLSQP step, where
    ``m = m_eq + m_ineq_general``.  Bound rows are *not* in this
    matrix.  The cost is fixed regardless of how many rows are
    actually active.  For the regime SLSQP-JAX targets
    (``m << n``) this is negligible compared to the QP solve.

    Args
    ----
    grad_new: Objective gradient ``∇f(x_{k+1})``.
    eq_jac_new: Equality Jacobian ``J_eq(x_{k+1})``, shape ``(m_eq, n)``.
    ineq_jac_general_new: *General* (non-bound) inequality Jacobian
        ``J_ineq_general(x_{k+1})``, shape ``(m_ineq_general, n)``.
    ineq_val_general_new: General inequality constraint values
        ``c_ineq_general(x_{k+1})``, shape ``(m_ineq_general,)``.
    free_mask: Per-coordinate free mask (``True`` = not at any bound).
        REQUIRED when bounds are present.  Pass ``None`` only when no
        bound constraints exist.
    active_tol: Threshold for the value-based active-set rule.  An
        inequality row is treated as active when ``c_ineq <=
        active_tol``.  Defaults to ``1e-8``; the caller typically
        passes ``self.atol``.
    ridge: Tikhonov ridge added to the masked ``A Aᵀ`` matrix for
        numerical stability.  Absorbed by the single round of
        iterative refinement.  Defaults to ``1e-8``.
    """
    m_eq = eq_jac_new.shape[0]
    m_general = ineq_jac_general_new.shape[0]
    m_total = m_eq + m_general
    dtype = grad_new.dtype

    if m_total == 0:
        return (
            jnp.zeros((m_eq,), dtype=dtype),
            jnp.zeros((m_general,), dtype=dtype),
        )

    eq_active = jnp.ones((m_eq,), dtype=bool)
    if m_general > 0:
        ineq_active = ineq_val_general_new <= jnp.asarray(active_tol, dtype=dtype)
    else:
        ineq_active = jnp.zeros((0,), dtype=bool)
    full_active = jnp.concatenate([eq_active, ineq_active], axis=0)

    A_full = jnp.concatenate([eq_jac_new, ineq_jac_general_new], axis=0)

    A_work = jnp.where(full_active[:, None], A_full, jnp.asarray(0.0, dtype=dtype))
    if free_mask is not None:
        A_work = A_work * free_mask[None, :].astype(dtype)
        rhs_grad = jnp.where(free_mask, grad_new, jnp.asarray(0.0, dtype=dtype))
    else:
        rhs_grad = grad_new

    reg_diag = jnp.where(
        full_active,
        jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
    )
    AAt = (
        A_work @ A_work.T
        + jnp.diag(reg_diag)
        + jnp.asarray(ridge, dtype=dtype) * jnp.eye(m_total, dtype=dtype)
    )
    AAt_chol = jnp.linalg.cholesky(AAt)

    rhs = A_work @ rhs_grad
    mult = jax.scipy.linalg.cho_solve((AAt_chol, True), rhs)
    mult = jnp.where(full_active, mult, jnp.asarray(0.0, dtype=dtype))

    residual = A_work @ rhs_grad - A_work @ (A_work.T @ mult)
    delta = jax.scipy.linalg.cho_solve((AAt_chol, True), residual)
    mult = mult + delta
    mult = jnp.where(full_active, mult, jnp.asarray(0.0, dtype=dtype))

    if m_eq > 0:
        lambda_eq_ls = mult[:m_eq]
    else:
        lambda_eq_ls = jnp.zeros((0,), dtype=dtype)

    if m_general > 0:
        lambda_ineq_general_ls = mult[m_eq:]
        lambda_ineq_general_ls = jnp.where(
            ineq_active,
            jnp.maximum(lambda_ineq_general_ls, jnp.asarray(0.0, dtype=dtype)),
            jnp.asarray(0.0, dtype=dtype),
        )
    else:
        lambda_ineq_general_ls = jnp.zeros((0,), dtype=dtype)

    return lambda_eq_ls, lambda_ineq_general_ls


__all__ = [
    "compute_at_lower_mask",
    "compute_at_upper_mask",
    "recover_ls_multipliers_at_iterate",
]
