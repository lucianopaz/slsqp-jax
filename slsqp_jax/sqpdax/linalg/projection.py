"""Matrix-free orthogonal projection onto ``null(A)``."""

from collections.abc import Callable

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int

__all__ = ["cg_normal_equations", "null_space_projector", "pcg"]

Operator = Callable[[Float[Array, " n"]], Float[Array, " m"]]


def pcg(
    op: Callable[[Float[Array, " m"]], Float[Array, " m"]],
    rhs: Float[Array, " m"],
    *,
    tol: float,
    max_iter: int,
    preconditioner: Callable[[Float[Array, " m"]], Float[Array, " m"]] | None = None,
) -> tuple[Float[Array, " m"], Int[Array, ""]]:
    """Preconditioned conjugate gradients on a symmetric positive semidefinite system.

    Solves ``op(y) = rhs`` from ``y = 0`` and stops once
    ``‖r‖² ≤ tol² max{‖rhs‖², 1}`` or after ``max_iter`` iterations. On a
    consistent singular system the iterates stay in ``range(op)`` (in
    ``range(M⁻¹ op)`` with a preconditioner ``M``), so the minimum-norm
    solution is returned without any regularisation.

    Parameters
    ----------
    op
        Matrix-free product ``y ↦ M y`` with ``M`` symmetric PSD.
    rhs
        Right-hand side.
    tol
        Relative residual tolerance (with an absolute floor of ``tol``).
    max_iter
        Iteration cap (a static Python ``int``).
    preconditioner
        Optional product ``r ↦ P⁻¹ r`` with ``P`` symmetric positive
        definite; ``None`` is plain CG.

    Returns
    -------
    y
        Approximate solution.
    n_iter
        Number of iterations performed.
    """
    dtype = rhs.dtype
    tol_sq = jnp.asarray(tol, dtype) ** 2 * jnp.maximum(jnp.dot(rhs, rhs), 1.0)
    prec = (lambda r: r) if preconditioner is None else preconditioner

    def body(_j, carry):
        y, r, p, rz, rr, k, done = carry

        def do(c):
            y, r, p, rz, _rr, k, _d = c
            Ap = op(p)
            pAp = jnp.dot(p, Ap)
            alpha = jnp.where(pAp > 1e-30, rz / jnp.maximum(pAp, 1e-30), 0.0)
            y_new = y + alpha * p
            r_new = r - alpha * Ap
            z_new = prec(r_new)
            rz_new = jnp.dot(r_new, z_new)
            rr_new = jnp.dot(r_new, r_new)
            beta = jnp.where(rz > 1e-30, rz_new / jnp.maximum(rz, 1e-30), 0.0)
            return (
                y_new,
                r_new,
                z_new + beta * p,
                rz_new,
                rr_new,
                k + 1,
                rr_new < tol_sq,
            )

        return jax.lax.cond(done, lambda c: c, do, carry)

    z0 = prec(rhs)
    rr0 = jnp.dot(rhs, rhs)
    init = (
        jnp.zeros_like(rhs),
        rhs,
        z0,
        jnp.dot(rhs, z0),
        rr0,
        jnp.asarray(0, jnp.int32),
        rr0 < tol_sq,
    )
    y, _r, _p, _rz, _rr, n_iter, _done = jax.lax.fori_loop(0, max_iter, body, init)
    return y, n_iter


def cg_normal_equations(
    normal_op: Callable[[Float[Array, " m"]], Float[Array, " m"]],
    rhs: Float[Array, " m"],
    *,
    tol: float,
    max_iter: int,
    preconditioner: Callable[[Float[Array, " m"]], Float[Array, " m"]] | None = None,
) -> Float[Array, " m"]:
    """Conjugate gradients on a symmetric positive semidefinite system.

    Thin wrapper around :func:`pcg` returning only the solution. Solves
    ``normal_op(y) = rhs`` from ``y = 0`` and stops once
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
    preconditioner
        Optional ``r ↦ P⁻¹ r``; see :func:`pcg`.

    Returns
    -------
    Float[Array, " m"]
        Approximate solution.
    """
    y, _ = pcg(
        normal_op, rhs, tol=tol, max_iter=max_iter, preconditioner=preconditioner
    )
    return y


def null_space_projector(
    A: Operator,
    At: Operator,
    *,
    free_mask: Float[Array, " n"] | None = None,
    reg: float = 0.0,
    tol: float = 1e-10,
    max_iter: int = 100,
    solve: Callable[[Float[Array, " m"]], Float[Array, " m"]] | None = None,
) -> Callable[[Float[Array, " n"]], Float[Array, " n"]]:
    """Build ``v ↦ (I − Aᵀ(A Aᵀ + reg I)⁻¹ A) v`` from matrix-free products.

    By default the normal equations are solved by :func:`cg_normal_equations`,
    so the dense ``A`` is never assembled; an external ``solve`` for
    ``(A Aᵀ)⁺`` (e.g. a
    :class:`~slsqp_jax.sqpdax.linalg.scaled_normal_equations.SchurNormalEquations`)
    replaces the inner CG, in which case ``reg``, ``tol`` and ``max_iter``
    are ignored. When ``free_mask`` is given, columns with a zero mask are
    frozen: the projector acts on ``free_mask ⊙ v`` with the restricted
    operator ``A_free = A ∘ diag(free_mask)`` and returns a vector that is
    zero on frozen coordinates (an external ``solve`` must then be built for
    the same restricted operator).

    Parameters
    ----------
    A
        Product ``v ↦ A v`` (``n → m``).
    At
        Product ``y ↦ Aᵀ y`` (``m → n``).
    free_mask
        Optional ``0/1`` float mask of the free columns.
    reg
        Tikhonov regularisation added to ``A Aᵀ`` (inner-CG path only).
    tol, max_iter
        Inner CG controls, see :func:`cg_normal_equations`.
    solve
        Optional external solve ``b ↦ (A_free A_freeᵀ)⁺ b``.

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
        if solve is None:
            y = cg_normal_equations(normal_op, A_free(v), tol=tol, max_iter=max_iter)
        else:
            y = solve(A_free(v))
        return v - At_free(y)

    return proj
