"""NLP-level box-bound machinery for the SLSQP outer loop.

These helpers handle the three points where bounds enter the *outer*
SQP iteration (the reduced-space QP-level bound-fixing loop lives in
:mod:`slsqp_jax.qp.bound_fixing` instead):

* :func:`clip_to_bounds` — defensively project an iterate onto the
  feasible box (used in ``init`` and after every line search).
* :func:`compute_bound_constraint_values` — evaluate ``c(x) = x − lb``
  / ``ub − x`` for the finite bound rows; the result is appended to
  the user inequality vector.
* :func:`build_bound_jacobian` — build the constant ``[I; −I]``
  Jacobian rows for the bound constraints.  Computed once during
  ``init`` and stored on ``SLSQPState``.
* :func:`recover_bound_multipliers` — post-line-search refresh of the
  bound multipliers from the *partial* Lagrangian gradient at
  ``x_{k+1}``.  See ``AGENTS.md`` for the rationale.

All functions are pure and free of solver-class state; they take the
precomputed ``lower_indices`` / ``upper_indices`` tuples and the
``bounds`` array directly so they can be reused outside :class:`SLSQP`
(e.g. by the eventual standalone QP solver).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from slsqp_jax.hessian import compute_partial_lagrangian_gradient
from slsqp_jax.types import Vector


def clip_to_bounds(
    y: Vector,
    bounds: Float[Array, "n 2"] | None,
) -> Vector:
    """Project ``y`` onto the box defined by ``bounds``.

    Returns ``y`` unchanged when ``bounds`` is ``None``.
    """
    if bounds is None:
        return y
    return jnp.clip(y, bounds[:, 0], bounds[:, 1])


def compute_bound_constraint_values(
    y: Vector,
    bounds: Float[Array, "n 2"] | None,
    lower_indices: tuple[int, ...],
    upper_indices: tuple[int, ...],
) -> Float[Array, " m_bounds"]:
    """Compute bound constraint values ``c(x) >= 0`` for finite bounds.

    Returns the empty vector when no finite bounds are present.
    """
    if bounds is None or (len(lower_indices) == 0 and len(upper_indices) == 0):
        return jnp.zeros((0,))

    lower_idx = np.array(lower_indices)
    upper_idx = np.array(upper_indices)

    lower_vals = (
        y[lower_idx] - bounds[lower_idx, 0] if len(lower_idx) > 0 else jnp.zeros((0,))
    )
    upper_vals = (
        bounds[upper_idx, 1] - y[upper_idx] if len(upper_idx) > 0 else jnp.zeros((0,))
    )
    return jnp.concatenate([lower_vals, upper_vals])


def build_bound_jacobian(
    n: int,
    bounds: Float[Array, "n 2"] | None,
    lower_indices: tuple[int, ...],
    upper_indices: tuple[int, ...],
) -> Float[Array, "m_bounds n"]:
    """Constant Jacobian of the bound constraints.

    ``+I`` rows for lower bounds, ``−I`` rows for upper bounds.  Empty
    matrix when no finite bounds are present.
    """
    if bounds is None or (len(lower_indices) == 0 and len(upper_indices) == 0):
        return jnp.zeros((0, n))

    lower_idx = np.array(lower_indices)
    upper_idx = np.array(upper_indices)
    identity = jnp.eye(n)
    J_lower = identity[lower_idx] if len(lower_idx) > 0 else jnp.zeros((0, n))
    J_upper = -identity[upper_idx] if len(upper_idx) > 0 else jnp.zeros((0, n))
    return jnp.concatenate([J_lower, J_upper], axis=0)


def recover_bound_multipliers(
    *,
    y_new: Vector,
    grad_new: Vector,
    eq_jac_new: Float[Array, "m_eq n"],
    ineq_jac_new: Float[Array, "m_ineq n"],
    mult_eq: Float[Array, " m_eq"],
    mult_ineq_general: Float[Array, " m_general"],
    bounds: Float[Array, "n 2"] | None,
    lower_indices: tuple[int, ...],
    upper_indices: tuple[int, ...],
    m_general: int,
) -> tuple[Float[Array, " n_lower"], Float[Array, " n_upper"]]:
    """Post-line-search NLP-level bound-multiplier refresh.

    Reads off the bound multiplier at the active bounds from the
    partial Lagrangian gradient at ``x_{k+1}``, with the sign
    convention ``μ_lower = +partial_grad_L`` / ``μ_upper = -partial_grad_L``
    inherited from :func:`build_bound_jacobian` (``+I`` / ``−I``).
    Clamped to ``≥ 0`` for dual feasibility.

    See ``AGENTS.md`` for the full motivation; in short, the QP-level
    bound multipliers were recovered from ``B d + g − Aᵀ λ`` at
    ``x_k`` using the L-BFGS HVP and the QP active set, so they
    inherit an ``O(L-BFGS) + O(line-search) + O(active-set)`` error
    budget that on bound-heavy problems pins ``||∇L|| / |L|`` above
    ``rtol`` even at a true KKT point.  Splicing the partial-gradient
    recovery zeros that residual exactly at the bound-active indices
    by construction.
    """
    n_lower = len(lower_indices)
    n_upper = len(upper_indices)

    if bounds is None or (n_lower == 0 and n_upper == 0):
        return jnp.zeros((0,), dtype=y_new.dtype), jnp.zeros((0,), dtype=y_new.dtype)

    gen_jac_new = (
        ineq_jac_new[:m_general]
        if m_general > 0
        else jnp.zeros((0, y_new.shape[0]), dtype=y_new.dtype)
    )
    partial_grad_L = compute_partial_lagrangian_gradient(
        grad_new,
        eq_jac_new,
        mult_eq,
        gen_jac_new,
        mult_ineq_general,
    )

    # Per-variable active-bound tolerance: 1e-12 absolute floor + a
    # relative ``eps · (1 + |y_new|)`` term so the test still fires
    # for variables whose magnitude pushes 1e-12 below local fp
    # spacing.  Variables that ``clip_to_bounds`` snapped to the bound
    # satisfy the test exactly; the relative term only kicks in for
    # variables the line search drove to within fp precision of a
    # bound without explicit clipping.
    eps = jnp.asarray(jnp.finfo(y_new.dtype).eps, dtype=y_new.dtype)
    bound_tol = jnp.asarray(1e-12, dtype=y_new.dtype) + eps * (1.0 + jnp.abs(y_new))

    if n_lower > 0:
        lower_idx = jnp.asarray(lower_indices, dtype=jnp.int32)
        lb_at_idx = bounds[lower_idx, 0]
        at_lower = (y_new[lower_idx] - lb_at_idx) <= bound_tol[lower_idx]
        mu_lower_corr = jnp.maximum(
            jnp.where(at_lower, partial_grad_L[lower_idx], 0.0),
            0.0,
        )
    else:
        mu_lower_corr = jnp.zeros((0,), dtype=y_new.dtype)

    if n_upper > 0:
        upper_idx = jnp.asarray(upper_indices, dtype=jnp.int32)
        ub_at_idx = bounds[upper_idx, 1]
        at_upper = (ub_at_idx - y_new[upper_idx]) <= bound_tol[upper_idx]
        mu_upper_corr = jnp.maximum(
            jnp.where(at_upper, -partial_grad_L[upper_idx], 0.0),
            0.0,
        )
    else:
        mu_upper_corr = jnp.zeros((0,), dtype=y_new.dtype)

    return mu_lower_corr, mu_upper_corr


__all__ = [
    "build_bound_jacobian",
    "clip_to_bounds",
    "compute_bound_constraint_values",
    "recover_bound_multipliers",
]
