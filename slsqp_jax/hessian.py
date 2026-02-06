"""BFGS Hessian Approximation Updates for SLSQP.

This module implements the damped BFGS update for maintaining a positive
definite approximation to the Hessian of the Lagrangian function.

The damped BFGS update uses Powell's modification to ensure positive
definiteness is preserved even when the curvature condition s^T y > 0
is not satisfied (which can happen for constrained optimization).
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped


@jaxtyped(typechecker=beartype)
def damped_bfgs_update(
    B: Float[Array, "n n"],
    s: Float[Array, " n"],
    y: Float[Array, " n"],
    damping_threshold: float = 0.2,
) -> Float[Array, "n n"]:
    """Perform damped BFGS update of the Hessian approximation.

    The standard BFGS update is:
        B_{k+1} = B_k - (B_k s s^T B_k) / (s^T B_k s) + (y y^T) / (s^T y)

    This requires s^T y > 0 (curvature condition) for positive definiteness.
    For constrained optimization, this may not hold, so we use Powell's
    damped BFGS modification:

        r = θ * y + (1 - θ) * B_k s

    where θ ∈ [0, 1] is chosen to ensure s^T r ≥ 0.2 * s^T B_k s.

    Then the update becomes:
        B_{k+1} = B_k - (B_k s s^T B_k) / (s^T B_k s) + (r r^T) / (s^T r)

    Args:
        B: Current Hessian approximation (n x n), positive definite.
        s: Step vector s = x_{k+1} - x_k.
        y: Gradient difference y = ∇L_{k+1} - ∇L_k.
        damping_threshold: Threshold for Powell damping (default 0.2).

    Returns:
        Updated Hessian approximation B_{k+1}.
    """

    # Compute key quantities
    Bs = B @ s
    sTBs = jnp.dot(s, Bs)
    sTy = jnp.dot(s, y)

    # Ensure sTBs is positive (should be if B is positive definite)
    sTBs_safe = jnp.maximum(sTBs, 1e-12)

    # Powell's damping: choose θ to ensure s^T r >= threshold * s^T B s
    # If s^T y >= threshold * s^T B s, use θ = 1 (standard BFGS)
    # Otherwise, θ = (1 - threshold) * s^T B s / (s^T B s - s^T y)

    use_damping = sTy < damping_threshold * sTBs_safe

    theta = jax.lax.cond(
        use_damping,
        lambda: (1.0 - damping_threshold) * sTBs_safe / (sTBs_safe - sTy + 1e-12),
        lambda: jnp.array(1.0),
    )

    # Ensure theta is in [0, 1]
    theta = jnp.clip(theta, 0.0, 1.0)

    # Damped gradient difference
    r = theta * y + (1.0 - theta) * Bs

    # Compute s^T r (should be positive by construction)
    sTr = jnp.dot(s, r)
    sTr_safe = jnp.maximum(sTr, 1e-12)

    # BFGS update with damped r
    # B_new = B - (Bs)(Bs)^T / (s^T B s) + (r)(r)^T / (s^T r)
    B_new = B - jnp.outer(Bs, Bs) / sTBs_safe + jnp.outer(r, r) / sTr_safe

    return B_new


@jaxtyped(typechecker=beartype)
def bfgs_update_with_skip(
    B: Float[Array, "n n"],
    s: Float[Array, " n"],
    y: Float[Array, " n"],
    skip_threshold: float = 1e-12,
    damping_threshold: float = 0.2,
) -> Float[Array, "n n"]:
    """Perform BFGS update with option to skip if step is too small.

    This is the main entry point for Hessian updates. It:
    1. Checks if the step is too small (skip update)
    2. Applies damped BFGS update otherwise

    Args:
        B: Current Hessian approximation.
        s: Step vector s = x_{k+1} - x_k.
        y: Gradient difference y = ∇L_{k+1} - ∇L_k.
        skip_threshold: Minimum step norm for update (default 1e-12).
        damping_threshold: Powell damping threshold (default 0.2).

    Returns:
        Updated Hessian approximation.
    """
    s_norm = jnp.linalg.norm(s)

    def do_update():
        return damped_bfgs_update(B, s, y, damping_threshold)

    def skip_update():
        return B

    return jax.lax.cond(
        s_norm > skip_threshold,
        do_update,
        skip_update,
    )


@jaxtyped(typechecker=beartype)
def scale_initial_hessian(
    B: Float[Array, "n n"],
    s: Float[Array, " n"],
    y: Float[Array, " n"],
) -> Float[Array, "n n"]:
    """Scale the initial Hessian approximation based on curvature.

    After the first step, it's often beneficial to scale the initial
    Hessian (typically identity) to better match the problem's curvature:

        B_scaled = (y^T s) / (y^T y) * I

    This scaling is applied before the first BFGS update.

    Args:
        B: Current Hessian approximation (typically identity).
        s: First step vector.
        y: First gradient difference.

    Returns:
        Scaled Hessian approximation.
    """
    n = B.shape[0]

    yTy = jnp.dot(y, y)
    yTs = jnp.dot(y, s)

    # Compute scaling factor
    # Want to scale B so that s^T B s ≈ s^T y
    # If B = γI, then s^T B s = γ s^T s, so γ = s^T y / s^T s
    # But standard approach uses γ = y^T s / y^T y

    gamma = jax.lax.cond(
        yTy > 1e-12,
        lambda: yTs / yTy,
        lambda: jnp.array(1.0),
    )

    # Ensure positive scaling
    gamma = jnp.maximum(gamma, 1e-3)
    gamma = jnp.minimum(gamma, 1e3)

    return gamma * jnp.eye(n)


@jaxtyped(typechecker=beartype)
def compute_lagrangian_gradient(
    grad_f: Float[Array, " n"],
    eq_jac: Float[Array, "m_eq n"],
    ineq_jac: Float[Array, "m_ineq n"],
    multipliers_eq: Float[Array, " m_eq"],
    multipliers_ineq: Float[Array, " m_ineq"],
) -> Float[Array, " n"]:
    """Compute the gradient of the Lagrangian function.

    The Lagrangian is:
        L(x, λ, μ) = f(x) - λ^T c_eq(x) - μ^T c_ineq(x)

    Its gradient with respect to x is:
        ∇_x L = ∇f(x) - Σ λ_i ∇c_eq_i(x) - Σ μ_j ∇c_ineq_j(x)
              = ∇f(x) - J_eq^T λ - J_ineq^T μ

    Args:
        grad_f: Gradient of objective function ∇f(x).
        eq_jac: Jacobian of equality constraints (m_eq x n).
        ineq_jac: Jacobian of inequality constraints (m_ineq x n).
        multipliers_eq: Lagrange multipliers for equality constraints.
        multipliers_ineq: Lagrange multipliers for inequality constraints.

    Returns:
        Gradient of Lagrangian ∇_x L.
    """
    grad_L = grad_f

    m_eq = eq_jac.shape[0]
    m_ineq = ineq_jac.shape[0]

    if m_eq > 0:
        grad_L = grad_L - eq_jac.T @ multipliers_eq

    if m_ineq > 0:
        grad_L = grad_L - ineq_jac.T @ multipliers_ineq

    return grad_L
