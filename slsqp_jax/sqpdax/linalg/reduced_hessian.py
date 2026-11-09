"""Dense kernels for the limited-memory reduced-Hessian (LRH) secant.

Small, pure building blocks used by the LRH update (Gill & Leonard, 2003,
*SIAM J. Optim.* 14): a QR factorisation with a non-negative diagonal and the
twice-iterated classical Gram–Schmidt step that orthogonalises a new step
against the current basis. Every array that is not an ``n``-vector has at
most ``m + 1`` columns (``m`` = LRH memory), so no ``n × n`` object is ever
formed.
"""

from typing import Literal

from jax import numpy as jnp
from jaxtyping import Array, Float

from ..types import Scalar, Vector_n

__all__ = [
    "positive_qr",
    "orthogonalize_against",
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
        ``Q`` of shape ``(p, p)`` and ``R`` of shape ``(p, q)``.

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
