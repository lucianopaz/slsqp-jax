"""Preconditioner factories for the QP inner solver.

Two preconditioner types are offered, selected by
``PreconditionerConfig.type``:

* ``"lbfgs"`` — L-BFGS inverse Hessian via the two-loop recursion
  (Algorithm 7.4 of Nocedal & Wright).
* ``"diagonal"`` — stochastic diagonal Hessian estimate
  (Bekas, Kokiopoulou & Saad, 2007), probed each step with
  ``diagonal_n_probes`` Rademacher vectors.

When equality constraints are present *and* sSQP proximal stabilisation
is active (``ProximalConfig.tau > 0``), both options apply a Woodbury
correction to deliver ``B̃⁻¹`` for ``B̃ = B + (1/μ) Aᵀ A``.  The
``mu × mu`` inner block is factored once per call.

These functions are pure factories: they take the precomputed
``LBFGSHistory`` / Lagrangian HVP and return a closure ``v -> M v``.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from slsqp_jax.hessian import (
    LBFGSHistory,
    estimate_hessian_diagonal,
    lbfgs_inverse_hvp,
)
from slsqp_jax.types import Scalar, Vector


def build_lbfgs_preconditioner(
    *,
    lbfgs_history: LBFGSHistory,
    eq_jac: jnp.ndarray | None,
    proximal_active: bool,
    proximal_mu: Scalar | float,
) -> Callable[[Vector], Vector]:
    """L-BFGS inverse Hessian preconditioner.

    With proximal stabilisation enabled and equality constraints
    present, applies the Woodbury identity to deliver
    ``(B + (1/μ) Aᵀ A)⁻¹``.  Otherwise returns a plain ``B⁻¹`` apply.
    """
    if proximal_active and eq_jac is not None and eq_jac.shape[0] > 0:
        A_eq = eq_jac
        mu = proximal_mu
        m_eq = A_eq.shape[0]

        Hinv_AT = jax.vmap(
            lambda a: lbfgs_inverse_hvp(lbfgs_history, a),
        )(A_eq)
        gram = Hinv_AT @ A_eq.T
        inner = mu * jnp.eye(m_eq) + gram
        inner_factor = jnp.linalg.cholesky(inner + 1e-10 * jnp.eye(m_eq))

        def preconditioner(v: Vector) -> Vector:
            Hinv_v = lbfgs_inverse_hvp(lbfgs_history, v)
            A_Hinv_v = A_eq @ Hinv_v
            w = jax.scipy.linalg.cho_solve((inner_factor, True), A_Hinv_v)
            correction = Hinv_AT.T @ w
            return Hinv_v - correction

        return preconditioner

    def preconditioner(v: Vector) -> Vector:
        return lbfgs_inverse_hvp(lbfgs_history, v)

    return preconditioner


def build_diagonal_preconditioner(
    *,
    lagrangian_hvp_fn: Callable[[Vector], Vector],
    n: int,
    step_count: jnp.ndarray,
    n_probes: int,
    eq_jac: jnp.ndarray | None,
    proximal_active: bool,
    proximal_mu: Scalar | float,
) -> Callable[[Vector], Vector]:
    """Stochastic diagonal Hessian preconditioner (Bekas et al., 2007).

    Estimates ``diag(H_L)`` via Rademacher probing of the exact
    Lagrangian HVP, with a deterministic key derived from the step
    count.  Small / negative entries are clamped to a positive floor
    so the preconditioner stays SPD.
    """
    key = jax.random.fold_in(jax.random.PRNGKey(42), step_count)
    diag_est = estimate_hessian_diagonal(lagrangian_hvp_fn, n, key, n_probes=n_probes)
    abs_diag = jnp.abs(diag_est)
    floor = jnp.maximum(1e-8, 1e-6 * jnp.median(abs_diag))
    diag_safe = jnp.maximum(abs_diag, floor)
    inv_diag = 1.0 / diag_safe

    if proximal_active and eq_jac is not None and eq_jac.shape[0] > 0:
        A_eq = eq_jac
        mu = proximal_mu
        m_eq = A_eq.shape[0]
        Dinv_AT = (A_eq * inv_diag[None, :]).T
        gram = A_eq @ Dinv_AT
        inner = mu * jnp.eye(m_eq) + gram
        inner_factor = jnp.linalg.cholesky(inner + 1e-10 * jnp.eye(m_eq))

        def preconditioner(v: Vector) -> Vector:
            Dinv_v = inv_diag * v
            A_Dinv_v = A_eq @ Dinv_v
            w = jax.scipy.linalg.cho_solve((inner_factor, True), A_Dinv_v)
            correction = Dinv_AT @ w
            return Dinv_v - correction

        return preconditioner

    def preconditioner(v: Vector) -> Vector:
        return inv_diag * v

    return preconditioner


__all__ = [
    "build_diagonal_preconditioner",
    "build_lbfgs_preconditioner",
]
