"""Matrix-free orthogonal projection onto ``null(A)``."""

from collections.abc import Callable

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

__all__ = ["cg_normal_equations", "null_space_projector"]

Operator = Callable[[Float[Array, " n"]], Float[Array, " m"]]


def cg_normal_equations(
    normal_op: Callable[[Float[Array, " m"]], Float[Array, " m"]],
    rhs: Float[Array, " m"],
    *,
    tol: float,
    max_iter: int,
) -> Float[Array, " m"]:
    """Conjugate gradients on a symmetric positive semidefinite system.

    Solves ``normal_op(y) = rhs`` from ``y = 0`` and stops once
    ``‖r‖² ≤ tol² max{‖rhs‖², 1}`` or after ``max_iter`` iterations. On a
    consistent singular system the iterates stay in ``range(normal_op)``,
    which is what the projector below relies on.

    Parameters
    ----------
    normal_op
        Matrix-free product ``y ↦ M y`` with ``M`` symmetric PSD.
    rhs
        Right-hand side.
    tol
        Relative residual tolerance (with an absolute floor of ``tol``).
    max_iter
        Iteration cap (a static Python ``int``).

    Returns
    -------
    Float[Array, " m"]
        Approximate solution.
    """
    dtype = rhs.dtype
    tol_sq = jnp.asarray(tol, dtype) ** 2 * jnp.maximum(jnp.dot(rhs, rhs), 1.0)

    def body(_j, carry):
        y, r, p, rz, done = carry

        def do(c):
            y, r, p, rz, _d = c
            Ap = normal_op(p)
            pAp = jnp.dot(p, Ap)
            alpha = jnp.where(pAp > 1e-30, rz / jnp.maximum(pAp, 1e-30), 0.0)
            y_new = y + alpha * p
            r_new = r - alpha * Ap
            rz_new = jnp.dot(r_new, r_new)
            beta = jnp.where(rz > 1e-30, rz_new / jnp.maximum(rz, 1e-30), 0.0)
            return (y_new, r_new, r_new + beta * p, rz_new, rz_new < tol_sq)

        return jax.lax.cond(done, lambda c: c, do, carry)

    rz0 = jnp.dot(rhs, rhs)
    init = (jnp.zeros_like(rhs), rhs, rhs, rz0, rz0 < tol_sq)
    y, *_ = jax.lax.fori_loop(0, max_iter, body, init)
    return y


def null_space_projector(
    A: Operator,
    At: Operator,
    *,
    free_mask: Float[Array, " n"] | None = None,
    reg: float = 0.0,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> Callable[[Float[Array, " n"]], Float[Array, " n"]]:
    """Build ``v ↦ (I − Aᵀ(A Aᵀ + reg I)⁻¹ A) v`` from matrix-free products.

    The normal equations are solved by :func:`cg_normal_equations`, so the
    dense ``A`` is never assembled. When ``free_mask`` is given, columns with
    a zero mask are frozen: the projector acts on ``free_mask ⊙ v`` with the
    restricted operator ``A_free = A ∘ diag(free_mask)`` and returns a vector
    that is zero on frozen coordinates.

    Parameters
    ----------
    A
        Product ``v ↦ A v`` (``n → m``).
    At
        Product ``y ↦ Aᵀ y`` (``m → n``).
    free_mask
        Optional ``0/1`` float mask of the free columns.
    reg
        Tikhonov regularisation added to ``A Aᵀ``.
    tol, max_iter
        Inner CG controls, see :func:`cg_normal_equations`.

    Returns
    -------
    Callable
        The projector ``proj(v)``; ``A proj(v) ≈ 0`` (up to ``reg`` and the
        CG tolerance) for every ``v``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.linalg import null_space_projector
    >>> M = jnp.array([[1.0, 1.0, 0.0]])
    >>> proj = null_space_projector(lambda v: M @ v, lambda y: M.T @ y)
    >>> v = proj(jnp.array([1.0, 0.0, 2.0]))
    >>> bool(jnp.allclose(M @ v, 0.0, atol=1e-6)), bool(jnp.isclose(v[2], 2.0))
    (True, True)
    """

    def A_free(v: Float[Array, " n"]) -> Float[Array, " m"]:
        return A(v if free_mask is None else free_mask * v)

    def At_free(y: Float[Array, " m"]) -> Float[Array, " n"]:
        out = At(y)
        return out if free_mask is None else free_mask * out

    def normal_op(y: Float[Array, " m"]) -> Float[Array, " m"]:
        return A_free(At_free(y)) + jnp.asarray(reg, y.dtype) * y

    def proj(v: Float[Array, " n"]) -> Float[Array, " n"]:
        v = v if free_mask is None else free_mask * v
        y = cg_normal_equations(normal_op, A_free(v), tol=tol, max_iter=max_iter)
        return v - At_free(y)

    return proj
