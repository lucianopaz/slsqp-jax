"""L-BFGS Hessian Approximation for SLSQP.

This module implements the Limited-memory BFGS (L-BFGS) algorithm for
maintaining a matrix-free approximation to the Hessian of the Lagrangian.

Instead of storing a dense n x n matrix (O(n^2) memory), L-BFGS stores
the last k (s, y) pairs and computes Hessian-vector products in O(kn) time
using the compact representation (Byrd, Nocedal, Schnabel 1994):

    B = B_0 - [B_0 S, Y] @ M^{-1} @ [S^T B_0; Y^T]

where B_0 = diag(diagonal) is the initial Hessian (per-variable scaling)
and M is a small 2k x 2k matrix built from inner products of the stored
vectors.  During normal operation ``diagonal = gamma * ones(n)`` and this
reduces to the scalar-scaled form.  After an SNOPT-style diagonal reset,
``diagonal`` captures per-variable curvature from the discarded history.

Powell's damping is applied to each (s, y) pair before storage to ensure
positive definiteness is preserved even when the standard curvature
condition s^T y > 0 is not satisfied (common in constrained optimization).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, Int, jaxtyped

from slsqp_jax.types import Scalar, Vector


class LBFGSHistory(eqx.Module):
    """L-BFGS history buffer for matrix-free Hessian approximation.

    Stores the last k (s, y) pairs in a circular buffer and provides
    efficient Hessian-vector products via the compact representation.

    Attributes:
        s_history: Stored step vectors s_i = x_{i+1} - x_i.
        y_history: Stored (damped) gradient differences y_i.
        gamma: Scalar summary of the initial Hessian scaling.
        diagonal: Per-variable initial Hessian scaling (B_0 = diag(d)).
            During normal operation this equals ``gamma * ones(n)``.
            After an SNOPT-style reset it stores per-variable curvature.
        count: Number of valid pairs stored (0 to memory size).
        next_idx: Next write position in the circular buffer.
    """

    s_history: Float[Array, "memory n"]
    y_history: Float[Array, "memory n"]
    gamma: Scalar
    diagonal: Float[Array, " n"]
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
        diagonal=jnp.ones(n),
        count=jnp.array(0),
        next_idx=jnp.array(0),
    )


@jaxtyped(typechecker=beartype)
def lbfgs_hvp(
    history: LBFGSHistory,
    v: Vector,
) -> Vector:
    """Compute B @ v using the L-BFGS compact representation.

    Uses the compact form with diagonal initial Hessian B_0 = diag(d)
    (Byrd, Nocedal & Schnabel, 1994, Theorem 2.2):

        B = B_0 - [B_0 S, Y] @ M^{-1} @ [S^T B_0; Y^T]

    where:
        M = [[S^T B_0 S, L], [L^T, -D_sy]]   (2k x 2k)
        L_{ij} = s_i^T y_j for i > j           (strictly lower triangular)
        D_sy = diag(s_i^T y_i)                 (diagonal)

    When no pairs are stored (count=0), this reduces to B = diag(d).

    Complexity: O(k^2 n) where k is the number of stored pairs.

    Args:
        history: L-BFGS history buffer.
        v: Vector to multiply by the Hessian approximation.

    Returns:
        B @ v, the Hessian-vector product.
    """
    k = history.s_history.shape[0]
    d = history.diagonal
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

    # Compute the diagonal part of the Hessian approximation
    # DS[i, :] = d * s_i  (B_0 applied row-wise)
    DS = S * d[None, :]  # (k, n)

    # Build compact form inner matrices
    SY = S @ Y.T  # (k, k)
    SDSS = DS @ S.T  # (k, k): S^T B_0 S

    L = jnp.tril(SY, k=-1)
    D_diag = jnp.diag(SY)

    invalid_diag = jnp.where(jnp.arange(k) < count, 0.0, 1.0)

    top_left = SDSS + jnp.diag(invalid_diag)
    top_right = L
    bottom_left = L.T
    bottom_right = -jnp.diag(D_diag) + jnp.diag(invalid_diag)

    top = jnp.concatenate([top_left, top_right], axis=1)
    bottom = jnp.concatenate([bottom_left, bottom_right], axis=1)
    M = jnp.concatenate([top, bottom], axis=0)
    # Small regularization for numerical stability
    M = M + 1e-10 * jnp.eye(2 * k)

    # p = [S^T B_0 v; Y^T v] = [DS @ v; Y @ v]  but DS rows are d*s_i
    # so DS @ v would be wrong shape.  We need S @ (d * v).
    dv = d * v
    p = jnp.concatenate([S @ dv, Y @ v])

    q = jnp.linalg.solve(M, p)

    # B v = B_0 v - [B_0 S, Y]^T @ q = d*v - DS^T @ q[:k] - Y^T @ q[k:]
    result = dv - DS.T @ q[:k] - Y.T @ q[k:]

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
def lbfgs_inverse_hvp(
    history: LBFGSHistory,
    v: Vector,
) -> Vector:
    """Compute H @ v = B^{-1} @ v via the L-BFGS two-loop recursion.

    Implements Nocedal & Wright Algorithm 7.4 with diagonal initial
    scaling H_0 = diag(1/diagonal) instead of scalar (1/gamma) I.

    Complexity: O(kn) where k is the number of stored pairs.

    Args:
        history: L-BFGS history buffer.
        v: Vector to multiply by the inverse Hessian approximation.

    Returns:
        H @ v = B^{-1} @ v, the inverse Hessian-vector product.
    """
    k = history.s_history.shape[0]
    d = history.diagonal
    count = history.count

    start = (history.next_idx - count + k) % k
    indices = (start + jnp.arange(k)) % k
    S = history.s_history[indices]  # (k, n) chronological
    Y = history.y_history[indices]  # (k, n) chronological

    valid_mask = (jnp.arange(k) < count)[:, None]
    S = S * valid_mask
    Y = Y * valid_mask

    sTy = jnp.sum(S * Y, axis=1)  # (k,)
    pair_ok = (jnp.arange(k) < count) & (sTy > 1e-12)
    rho = jnp.where(pair_ok, 1.0 / sTy, 0.0)

    # Backward loop: q = v; for i = k-1,...,0: alpha_i = rho_i s_i^T q; q -= alpha_i y_i
    alphas_init = jnp.zeros(k)

    def backward_step(carry, idx):
        q, alphas = carry
        rev_idx = k - 1 - idx
        s_i = S[rev_idx]
        y_i = Y[rev_idx]
        rho_i = rho[rev_idx]
        is_valid = rev_idx < count
        alpha_i = jnp.where(is_valid, rho_i * jnp.dot(s_i, q), 0.0)
        q = q - alpha_i * y_i
        alphas = alphas.at[rev_idx].set(alpha_i)
        return (q, alphas), None

    (q, alphas), _ = jax.lax.scan(backward_step, (v, alphas_init), jnp.arange(k))

    # Apply initial inverse Hessian: H_0 = diag(1/d)
    d_safe = jnp.maximum(d, 1e-30)
    r = q / d_safe

    # Forward loop: for i = 0,...,k-1: beta = rho_i y_i^T r; r += s_i (alpha_i - beta)
    def forward_step(r, idx):
        s_i = S[idx]
        y_i = Y[idx]
        rho_i = rho[idx]
        alpha_i = alphas[idx]
        is_valid = idx < count
        beta = jnp.where(is_valid, rho_i * jnp.dot(y_i, r), 0.0)
        r = r + s_i * (alpha_i - beta)
        return r, None

    r, _ = jax.lax.scan(forward_step, r, jnp.arange(k))

    v_norm = jnp.linalg.norm(v)
    r_norm = jnp.linalg.norm(r)

    # Fall back to identity if the result is non-finite or unreasonably large.
    # Use plain v so the fallback always has unit spectral radius.
    is_bad = jnp.any(~jnp.isfinite(r)) | (r_norm > 1000.0 * jnp.maximum(v_norm, 1e-10))
    r = jnp.where(is_bad, v, r)

    return r


@jaxtyped(typechecker=beartype)
def lbfgs_append(
    history: LBFGSHistory,
    s: Vector,
    y: Vector,
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

    n = history.diagonal.shape[0]

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
            diagonal=jnp.full(n, gamma_new),
            count=new_count,
            next_idx=new_idx,
        )

    def skip():
        return history

    return jax.lax.cond(~should_skip, do_append, skip)


@jaxtyped(typechecker=beartype)
def lbfgs_compute_diagonal(
    history: LBFGSHistory,
) -> Float[Array, " n"]:
    """Extract diag(B_k) from the L-BFGS compact representation.

    From the compact form ``B = B_0 - W M^{-1} W^T``, the diagonal is

        diag(B_k) = diagonal - diag(W M^{-1} W^T)

    where ``W = [B_0 S, Y]`` is ``(n, 2k)`` and ``M`` is the ``2k x 2k``
    inner matrix.  The correction term is computed in O(k^2 n) by forming
    ``Q = W M^{-1}`` and summing ``(Q * W)`` row-wise.

    This is used by :func:`lbfgs_reset` to implement the SNOPT diagonal
    reset strategy (Gill, Murray & Saunders, 2005, Section 3.3).
    """
    k = history.s_history.shape[0]
    d = history.diagonal
    count = history.count

    start = (history.next_idx - count + k) % k
    indices = (start + jnp.arange(k)) % k
    S = history.s_history[indices]
    Y = history.y_history[indices]

    valid_mask = (jnp.arange(k) < count)[:, None]
    S = S * valid_mask
    Y = Y * valid_mask

    DS = S * d[None, :]  # (k, n): B_0 applied row-wise

    SY = S @ Y.T
    SDSS = DS @ S.T

    L_mat = jnp.tril(SY, k=-1)
    D_diag = jnp.diag(SY)
    invalid_diag = jnp.where(jnp.arange(k) < count, 0.0, 1.0)

    top_left = SDSS + jnp.diag(invalid_diag)
    top_right = L_mat
    bottom_left = L_mat.T
    bottom_right = -jnp.diag(D_diag) + jnp.diag(invalid_diag)

    top = jnp.concatenate([top_left, top_right], axis=1)
    bottom = jnp.concatenate([bottom_left, bottom_right], axis=1)
    M = jnp.concatenate([top, bottom], axis=0)
    M = M + 1e-10 * jnp.eye(2 * k)

    M_inv = jnp.linalg.inv(M)  # (2k, 2k) — tiny

    # W is (n, 2k): columns are [d*s_0, ..., d*s_{k-1}, y_0, ..., y_{k-1}]
    W = jnp.concatenate([DS.T, Y.T], axis=1)  # (n, 2k)

    # Q = W M^{-1}, shape (n, 2k)
    Q = W @ M_inv

    # diag(W M^{-1} W^T) = row-wise sum of Q * W
    diag_correction = jnp.sum(Q * W, axis=1)  # (n,)

    return d - diag_correction


@jaxtyped(typechecker=beartype)
def lbfgs_reset(
    history: LBFGSHistory,
) -> LBFGSHistory:
    """SNOPT-style diagonal reset of the L-BFGS history.

    Extracts ``diag(B_k)`` from the current approximation, discards all
    stored ``(s, y)`` pairs, and restarts with ``B_0 = diag(diag(B_k))``.
    This preserves per-variable curvature information across the reset,
    preventing the "everything is flat" effect that occurs when the scalar
    ``gamma`` becomes very small.

    Based on the SNOPT limited-memory reset strategy (Gill, Murray &
    Saunders, *SIAM Review*, 47(1), 2005, Section 3.3).
    """
    diag_B = lbfgs_compute_diagonal(history)
    diag_clipped = jnp.clip(diag_B, 1e-3, 100.0)

    # Ensure all values are finite; fall back to 1.0 otherwise
    diag_safe = jnp.where(jnp.isfinite(diag_clipped), diag_clipped, 1.0)

    gamma_new = jnp.median(diag_safe)

    k, n = history.s_history.shape
    return LBFGSHistory(
        s_history=jnp.zeros((k, n)),
        y_history=jnp.zeros((k, n)),
        gamma=gamma_new,
        diagonal=diag_safe,
        count=jnp.array(0),
        next_idx=jnp.array(0),
    )


@jaxtyped(typechecker=beartype)
def compute_lagrangian_gradient(
    grad_f: Vector,
    eq_jac: Float[Array, "m_eq n"],
    ineq_jac: Float[Array, "m_ineq n"],
    multipliers_eq: Float[Array, " m_eq"],
    multipliers_ineq: Float[Array, " m_ineq"],
) -> Vector:
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
