"""L-BFGS Hessian Approximation for SLSQP.

This module implements the Limited-memory BFGS (L-BFGS) algorithm for
maintaining a matrix-free approximation to the Hessian of the Lagrangian.

Instead of storing a dense n x n matrix (O(n^2) memory), L-BFGS stores
the last k (s, y) pairs and computes Hessian-vector products in O(kn) time
using the compact representation (Byrd, Nocedal, Schnabel 1994):

    B = gamma * I - [gamma*S, Y] @ N^{-1} @ [gamma*S^T; Y^T]

where N is a small 2k x 2k matrix built from inner products of the
stored vectors.

Powell's damping is applied to each (s, y) pair before storage to ensure
positive definiteness is preserved even when the standard curvature
condition s^T y > 0 is not satisfied (common in constrained optimization).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, Int, jaxtyped


class LBFGSHistory(eqx.Module):
    """L-BFGS history buffer for matrix-free Hessian approximation.

    Stores the last k (s, y) pairs in a circular buffer and provides
    efficient Hessian-vector products via the compact representation.

    Attributes:
        s_history: Stored step vectors s_i = x_{i+1} - x_i.
        y_history: Stored (damped) gradient differences y_i.
        gamma: Initial Hessian scaling (B_0 = gamma * I).
        count: Number of valid pairs stored (0 to memory size).
        next_idx: Next write position in the circular buffer.
    """

    s_history: Float[Array, "memory n"]
    y_history: Float[Array, "memory n"]
    gamma: Float[Array, ""]
    count: Int[Array, ""]
    next_idx: Int[Array, ""]


def lbfgs_init(n: int, memory: int) -> LBFGSHistory:
    """Initialize an empty L-BFGS history buffer.

    Args:
        n: Dimension of the parameter space.
        memory: Maximum number of (s, y) pairs to store (typically 5-20).

    Returns:
        An initialized LBFGSHistory with no stored pairs and gamma=1.
    """
    return LBFGSHistory(
        s_history=jnp.zeros((memory, n)),
        y_history=jnp.zeros((memory, n)),
        gamma=jnp.array(1.0),
        count=jnp.array(0),
        next_idx=jnp.array(0),
    )


@jaxtyped(typechecker=beartype)
def lbfgs_hvp(
    history: LBFGSHistory,
    v: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Compute B @ v using the L-BFGS compact representation.

    Uses the compact form (Nocedal & Wright, Theorem 7.4):

        B = gamma * I - [gamma*S, Y] @ N^{-1} @ [gamma*S^T; Y^T]

    where:
        N = [[gamma * S^T S, L], [L^T, -D]]   (2k x 2k)
        L_{ij} = s_i^T y_j for i > j           (strictly lower triangular)
        D = diag(s_i^T y_i)                    (diagonal)

    When no pairs are stored (count=0), this reduces to B = gamma * I.

    Complexity: O(kn) where k is the number of stored pairs.

    Args:
        history: L-BFGS history buffer.
        v: Vector to multiply by the Hessian approximation.

    Returns:
        B @ v, the Hessian-vector product.
    """
    k = history.s_history.shape[0]
    gamma = history.gamma
    count = history.count

    # Reorder to chronological order from the circular buffer
    start = (history.next_idx - count + k) % k
    indices = (start + jnp.arange(k)) % k
    S = history.s_history[indices]  # (k, n)
    Y = history.y_history[indices]  # (k, n)

    # Zero out invalid entries (positions >= count are not valid)
    valid_mask = (jnp.arange(k) < count)[:, None]  # (k, 1)
    S = S * valid_mask
    Y = Y * valid_mask

    # Build compact form inner matrices
    SY = S @ Y.T  # (k, k): SY[i,j] = s_i^T y_j
    SS = S @ S.T  # (k, k): SS[i,j] = s_i^T s_j

    L = jnp.tril(SY, k=-1)  # strictly lower triangular
    D_diag = jnp.diag(SY)  # diagonal s_i^T y_i

    # Regularize invalid entries to keep N non-singular
    invalid_diag = jnp.where(jnp.arange(k) < count, 0.0, 1.0)

    # Build N = [[gamma * S^T S, L], [L^T, -D]] with regularization
    # Use concatenate instead of multiple .at[].set() for efficiency
    top_left = gamma * SS + jnp.diag(invalid_diag)
    top_right = L
    bottom_left = L.T
    bottom_right = -jnp.diag(D_diag) + jnp.diag(invalid_diag)

    top = jnp.concatenate([top_left, top_right], axis=1)
    bottom = jnp.concatenate([bottom_left, bottom_right], axis=1)
    N = jnp.concatenate([top, bottom], axis=0)

    # Small regularization for numerical stability
    N = N + 1e-10 * jnp.eye(2 * k)

    # Compute p = [gamma * S @ v; Y @ v]
    Sv = S @ v  # (k,)
    Yv = Y @ v  # (k,)
    p = jnp.concatenate([gamma * Sv, Yv])

    # Solve N @ q = p using regularized solve (faster than lstsq on GPU)
    q = jnp.linalg.solve(N, p)

    # B @ v = gamma * v - [gamma*S^T, Y^T] @ q
    #       = gamma * v - gamma * S^T @ q[:k] - Y^T @ q[k:]
    result = gamma * v - gamma * (S.T @ q[:k]) - Y.T @ q[k:]

    # Guard against NaN/inf or unreasonable values: fall back to identity scaling
    # For well-conditioned problems, the Hessian eigenvalues should be O(1) to O(100)
    v_norm = jnp.linalg.norm(v)
    result_norm = jnp.linalg.norm(result)

    # The ratio result_norm / v_norm should be bounded by max eigenvalue of B
    # For most problems this is < 100; use a threshold of 1000
    is_bad = jnp.any(~jnp.isfinite(result)) | (
        result_norm > 1000.0 * jnp.maximum(v_norm, 1e-10)
    )

    # When fallback triggers, use identity scaling (1.0 * v) not gamma * v
    # because gamma can be very small and would create ill-conditioned QP
    result = jnp.where(is_bad, v, result)

    return result


@jaxtyped(typechecker=beartype)
def lbfgs_append(
    history: LBFGSHistory,
    s: Float[Array, " n"],
    y: Float[Array, " n"],
    damping_threshold: float = 0.2,
    skip_threshold: float = 1e-8,
) -> LBFGSHistory:
    """Append a new (s, y) pair to the L-BFGS history with Powell damping.

    Powell's damping modifies y to ensure the curvature condition:
        s^T y_damped >= threshold * s^T B s

    The damped gradient difference is:
        y_damped = theta * y + (1 - theta) * B s

    where theta in [0, 1] is chosen to satisfy the condition above.
    This is essential for constrained optimization where the standard
    curvature condition s^T y > 0 may not hold.

    If ||s|| is too small or the curvature ratio is too extreme,
    the update is skipped entirely to avoid numerical issues.

    After appending, the initial Hessian scaling gamma is updated to
    y_damped^T s / (y_damped^T y_damped), clipped to [1e-3, 100].

    Args:
        history: Current L-BFGS history.
        s: Step vector s = x_{k+1} - x_k.
        y: Gradient difference y = nabla L_{k+1} - nabla L_k.
        damping_threshold: Powell damping threshold (default 0.2).
        skip_threshold: Minimum step norm for update (default 1e-8).

    Returns:
        Updated L-BFGS history with the new pair appended.
    """
    s_norm = jnp.linalg.norm(s)
    y_norm = jnp.linalg.norm(y)

    # Skip if step or gradient difference is too small (absolute threshold)
    step_too_small = s_norm < skip_threshold
    grad_diff_too_small = y_norm < skip_threshold

    # Skip if curvature ratio is too extreme (y_norm / s_norm too large or too small)
    # This prevents ill-conditioning when curvature doesn't match step size
    curvature_ratio = y_norm / jnp.maximum(s_norm, 1e-30)
    curvature_too_extreme = (curvature_ratio > 1e6) | (curvature_ratio < 1e-6)

    # Also check that s^T y is not too small relative to ||s|| ||y||
    sTy_raw = jnp.dot(s, y)
    relative_curvature = jnp.abs(sTy_raw) / jnp.maximum(s_norm * y_norm, 1e-30)
    curvature_too_small = relative_curvature < 1e-6

    # Skip if any values are non-finite
    has_bad_values = ~(
        jnp.isfinite(s_norm) & jnp.isfinite(y_norm) & jnp.isfinite(sTy_raw)
    )

    should_skip = (
        step_too_small
        | grad_diff_too_small
        | curvature_too_extreme
        | curvature_too_small
        | has_bad_values
    )

    def do_append():
        # Compute B @ s using current L-BFGS approximation for damping
        Bs = lbfgs_hvp(history, s)
        sTBs = jnp.dot(s, Bs)
        sTy = jnp.dot(s, y)
        sTBs_safe = jnp.maximum(sTBs, 1e-12)

        # Powell's damping: choose theta to ensure s^T r >= threshold * s^T B s
        use_damping = sTy < damping_threshold * sTBs_safe
        theta = jax.lax.cond(
            use_damping,
            lambda: (1.0 - damping_threshold) * sTBs_safe / (sTBs_safe - sTy + 1e-12),
            lambda: jnp.array(1.0),
        )
        theta = jnp.clip(theta, 0.0, 1.0)
        y_damped = theta * y + (1.0 - theta) * Bs

        # Update gamma (initial Hessian scaling) based on new curvature
        # gamma = s^T y / (y^T y) is the Barzilai-Borwein step length approximation
        yTy = jnp.dot(y_damped, y_damped)
        yTs = jnp.dot(y_damped, s)
        gamma_candidate = yTs / jnp.maximum(yTy, 1e-12)

        # Clip to a reasonable range - for most problems eigenvalues are O(1) to O(100)
        gamma_new = jax.lax.cond(
            (yTy > 1e-12) & jnp.isfinite(gamma_candidate),
            lambda: jnp.clip(gamma_candidate, 1e-3, 100.0),
            lambda: history.gamma,
        )

        # Write to circular buffer at next_idx
        k = history.s_history.shape[0]
        idx = history.next_idx
        new_s_history = history.s_history.at[idx].set(s)
        new_y_history = history.y_history.at[idx].set(y_damped)
        new_count = jnp.minimum(history.count + 1, jnp.array(k))
        new_idx = (idx + 1) % k

        return LBFGSHistory(
            s_history=new_s_history,
            y_history=new_y_history,
            gamma=gamma_new,
            count=new_count,
            next_idx=new_idx,
        )

    def skip():
        return history

    return jax.lax.cond(~should_skip, do_append, skip)


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
        L(x, lambda, mu) = f(x) - lambda^T c_eq(x) - mu^T c_ineq(x)

    Its gradient with respect to x is:
        nabla_x L = nabla f(x) - J_eq^T lambda - J_ineq^T mu

    Args:
        grad_f: Gradient of objective function nabla f(x).
        eq_jac: Jacobian of equality constraints (m_eq x n).
        ineq_jac: Jacobian of inequality constraints (m_ineq x n).
        multipliers_eq: Lagrange multipliers for equality constraints.
        multipliers_ineq: Lagrange multipliers for inequality constraints.

    Returns:
        Gradient of Lagrangian nabla_x L.
    """
    grad_L = grad_f

    m_eq = eq_jac.shape[0]
    m_ineq = ineq_jac.shape[0]

    if m_eq > 0:
        grad_L = grad_L - eq_jac.T @ multipliers_eq

    if m_ineq > 0:
        grad_L = grad_L - ineq_jac.T @ multipliers_ineq

    return grad_L
