"""Dense kernels for the limited-memory reduced-Hessian (LRH) secant.

Small, pure building blocks used by the LRH update (Gill & Leonard, 2003,
*SIAM J. Optim.* 14): a QR factorisation with a non-negative diagonal, the
twice-iterated classical Gram–Schmidt step that orthogonalises a new step
against the current basis, the BFGS update of the triangular Cholesky factor
of the reduced Hessian, and the removal of the oldest basis vector. Every
array that is not an ``n``-vector has at most ``m + 1`` columns (``m`` = LRH
memory), so no ``n × n`` object is ever formed.
"""

from typing import Literal

from jax import numpy as jnp
from jaxtyping import Array, Float

from ..types import Scalar, Vector_n

__all__ = [
    "positive_qr",
    "orthogonalize_against",
    "bfgs_factor_update",
    "drop_oldest_basis_vector",
]


def positive_qr(
    a: Float[Array, "p q"],
    mode: Literal["reduced", "complete"] = "reduced",
) -> tuple[Float[Array, "p k"], Float[Array, "k q"]]:
    """QR factorisation with a non-negative diagonal of ``R``.

    Wraps :func:`jax.numpy.linalg.qr` and flips the sign of every column of
    ``Q`` (and the matching row of ``R``) whose diagonal entry of ``R`` is
    negative, removing the sign ambiguity of Householder QR. A zero diagonal
    entry keeps its sign (``+1``). In ``"complete"`` mode the extra columns of
    ``Q`` beyond ``min(p, q)`` are left untouched.

    Parameters
    ----------
    a
        Matrix to factorise, shape ``(p, q)``.
    mode
        ``"reduced"`` gives ``Q`` of shape ``(p, k)`` and ``R`` of shape
        ``(k, q)`` with ``k = min(p, q)``; ``"complete"`` gives a square
        ``Q`` of shape ``(p, p)`` and ``R`` of shape ``(p, q)``. Use
        ``"complete"`` only for small inputs (the ``(m + 1)``-sized matrices
        of the LRH update), never for matrices with ``n`` rows, since the
        ``p × p`` factor ``Q`` would be an ``O(n²)`` object.

    Returns
    -------
    tuple of Array
        ``(q, r)`` with ``q @ r == a``, ``q`` having orthonormal columns,
        ``r`` upper triangular and ``diag(r) >= 0``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.linalg import positive_qr
    >>> a = jnp.array([[-1.0, 2.0], [0.0, -3.0], [0.0, 0.0]])
    >>> q, r = positive_qr(a)
    >>> bool(jnp.all(jnp.diagonal(r) >= 0)), bool(jnp.allclose(q @ r, a))
    (True, True)
    """
    if mode not in ("reduced", "complete"):
        raise ValueError(f"mode must be 'reduced' or 'complete', got {mode!r}.")
    q, r = jnp.linalg.qr(a, mode=mode)
    k = min(a.shape)
    sign = jnp.where(jnp.diagonal(r) < 0, -1.0, 1.0).astype(r.dtype)
    q_sign = jnp.ones(q.shape[1], dtype=r.dtype).at[:k].set(sign)
    r_sign = jnp.ones(r.shape[0], dtype=r.dtype).at[:k].set(sign)
    return q * q_sign[None, :], r * r_sign[:, None]


def orthogonalize_against(
    basis: Float[Array, "n k"],
    v: Vector_n,
) -> tuple[Float[Array, " k"], Vector_n, Scalar]:
    """Orthogonalise ``v`` against the columns of ``basis`` (CGS2).

    Classical Gram–Schmidt applied twice (the DGKS "twice is enough"
    criterion of Daniel, Gragg, Kaufman & Stewart, 1976), so the residual is
    orthogonal to ``basis`` to working precision even when ``v`` is nearly in
    its span. The columns of ``basis`` must be orthonormal or identically
    zero; zero (padding) columns receive a zero coefficient and never
    produce a ``NaN``. Costs ``O(n k)`` and forms no ``n × n`` array.

    Parameters
    ----------
    basis
        Matrix with ``k`` columns that are orthonormal or zero.
    v
        Vector to orthogonalise.

    Returns
    -------
    coeffs
        ``basisᵀ v`` (accumulated over both passes), shape ``(k,)``.
    residual
        ``v - basis @ coeffs``, orthogonal to ``basis``.
    rho
        ``‖residual‖``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.linalg import orthogonalize_against
    >>> basis = jnp.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])  # 2nd col: padding
    >>> coeffs, residual, rho = orthogonalize_against(basis, jnp.array([2.0, 3.0, 4.0]))
    >>> coeffs.tolist(), residual.tolist(), float(rho)
    ([2.0, 0.0], [0.0, 3.0, 4.0], 5.0)
    """
    coeffs = basis.T @ v
    residual = v - basis @ coeffs
    correction = basis.T @ residual
    coeffs = coeffs + correction
    residual = residual - basis @ correction
    return coeffs, residual, jnp.linalg.norm(residual)


def bfgs_factor_update(
    r: Float[Array, "k k"],
    s_r: Float[Array, " k"],
    y_r: Float[Array, " k"],
) -> Float[Array, "k k"]:
    """BFGS update of the upper-triangular factor of the reduced Hessian.

    Given ``B = RᵀR`` this returns the upper-triangular ``R_new`` (with a
    non-negative diagonal) such that

    ``R_newᵀ R_new = B - B s sᵀ B / (sᵀ B s) + y yᵀ / (yᵀ s)``

    where ``s = s_r`` and ``y = y_r`` are the step and curvature vectors in
    the reduced (basis) coordinates. The rank-one-modified matrix
    ``R + u vᵀ`` with ``u = R s / ‖R s‖`` and
    ``v = y / sqrt(yᵀ s) - Rᵀ u`` satisfies
    ``(R + u vᵀ)ᵀ (R + u vᵀ) = R_newᵀ R_new``; triangularity is restored with
    a small dense QR (Gill & Leonard, 2003, sec. 2). Padded rows/columns of
    ``R`` that are the identity (with zero entries of ``s_r`` and ``y_r``)
    are left unchanged; a zero padded block stays zero, so callers must keep
    identity padding. Costs ``O(k³)`` for the ``k × k`` factor.

    The caller must guarantee ``y_rᵀ s_r > 0`` and ``‖R s_r‖ > 0``; no guard
    is applied here.

    Parameters
    ----------
    r
        Upper-triangular factor, shape ``(k, k)``.
    s_r
        Step in reduced coordinates, shape ``(k,)``.
    y_r
        Curvature vector in reduced coordinates, shape ``(k,)``.

    Returns
    -------
    Array
        Updated upper-triangular factor, shape ``(k, k)``, ``diag >= 0``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.linalg import bfgs_factor_update
    >>> r = jnp.eye(2)
    >>> s, y = jnp.array([1.0, 0.0]), jnp.array([4.0, 0.0])
    >>> r_new = bfgs_factor_update(r, s, y)
    >>> bool(jnp.allclose(r_new.T @ r_new, jnp.diag(jnp.array([4.0, 1.0]))))
    True
    """
    a = r @ s_r
    u = a / jnp.linalg.norm(a)
    v = y_r / jnp.sqrt(y_r @ s_r) - r.T @ u
    _, r_new = positive_qr(r + jnp.outer(u, v), mode="reduced")
    return r_new


def drop_oldest_basis_vector(
    basis: Float[Array, "n k1"],
    tri: Float[Array, "k1 k1"],
    chol: Float[Array, "k1 k1"],
) -> tuple[Float[Array, "n k"], Float[Array, "k k"], Float[Array, "k k"]]:
    """Drop the oldest step from an expanded LRH basis (``k1 = k + 1`` slots).

    The inputs describe a full basis of ``k + 1`` orthonormal columns
    ``Z'`` (``basis``), the upper-triangular ``T''`` (``tri``) such that
    ``Z' T''`` holds the last ``k + 1`` steps (oldest first), and the
    upper-triangular Cholesky factor ``R''`` (``chol``) of the reduced
    Hessian in the basis ``Z'``. Removing the first column of ``T''`` leaves
    an upper-Hessenberg ``(k + 1) × k`` matrix; a complete QR
    ``T''[:, 1:] = Q_s [T̄; 0]`` (Gill & Leonard, 2003, sec. 3.2) gives the
    new basis ``Z̄ = (Z' Q_s)[:, :k]`` (one ``n × (k+1)`` by ``(k+1) × k``
    product), the new triangle ``T̄`` and the new factor ``R̄``, the leading
    ``k × k`` block of the triangular factor of ``R'' Q_s``. Then

    * ``Z̄ T̄`` equals the newest ``k`` steps and ``Z̄ᵀ Z̄ = I``;
    * ``R̄ᵀ R̄ = Z̄ᵀ B'' Z̄`` where ``B'' = Z' R''ᵀR'' Z'ᵀ + σ (I - Z' Z'ᵀ)``.

    Only ``n × (k + 1)`` and ``(k + 1) × (k + 1)`` arrays are formed. The
    basis is assumed full (no padded slot); a padded input still returns
    finite arrays (so a ``lax.cond`` over this kernel that ``vmap`` turns
    into a select is safe), but the result is then not meaningful.

    Parameters
    ----------
    basis
        Orthonormal basis ``Z'``, shape ``(n, k + 1)``.
    tri
        Upper-triangular ``T''``, shape ``(k + 1, k + 1)``.
    chol
        Upper-triangular reduced-Hessian factor ``R''``, shape
        ``(k + 1, k + 1)``.

    Returns
    -------
    basis_new, tri_new, chol_new
        Arrays of shapes ``(n, k)``, ``(k, k)`` and ``(k, k)``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.linalg import drop_oldest_basis_vector
    >>> basis = jnp.eye(4)[:, :3]
    >>> tri = jnp.eye(3)
    >>> z, t, r = drop_oldest_basis_vector(basis, tri, jnp.eye(3))
    >>> z.shape, t.shape, r.shape
    ((4, 2), (2, 2), (2, 2))
    """
    k = basis.shape[1] - 1
    q_s, h = positive_qr(tri[:, 1:], mode="complete")
    basis_new = basis @ q_s[:, :k]
    tri_new = h[:k, :]
    _, r_s = positive_qr(chol @ q_s, mode="reduced")
    return basis_new, tri_new, r_s[:k, :k]
