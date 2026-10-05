"""Steihaug–Toint truncated conjugate gradients and its boundary rules."""

from collections.abc import Callable
from typing import cast

import jax
from equinox import Module
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Scalar

__all__ = [
    "boundary_step_length",
    "steihaug_step",
    "SteihaugCGResult",
    "steihaug_cg",
]


def boundary_step_length(
    w: Float[Array, " n"], p: Float[Array, " n"], radius: Scalar
) -> Scalar:
    """Nonnegative ``β`` with ``‖w + β p‖₂ = radius`` (Steihaug 1983, eq. 2.7).

    Parameters
    ----------
    w
        Current iterate, assumed inside or on the ball ``‖w‖₂ ≤ radius``.
    p
        Search direction.
    radius
        Trust-region radius.

    Returns
    -------
    Scalar
        The positive root of ``‖w + β p‖² = radius²``; ``0`` when ``p = 0``.
        When ``w`` is already outside the ball the discriminant is clipped
        at zero and the result is the (nonnegative) tangent point along
        ``p``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.linalg import boundary_step_length
    >>> w = jnp.array([0.0, 0.5])
    >>> p = jnp.array([1.0, 0.0])
    >>> beta = boundary_step_length(w, p, jnp.asarray(1.0))
    >>> bool(jnp.isclose(jnp.linalg.norm(w + beta * p), 1.0))
    True
    """
    tiny = jnp.asarray(jnp.finfo(w.dtype).tiny, w.dtype)
    pp = jnp.dot(p, p)
    ww = jnp.dot(w, w)
    wp = jnp.dot(w, p)
    disc = jnp.sqrt(jnp.maximum(wp * wp + pp * (radius**2 - ww), 0.0))
    beta = jnp.where(pp > tiny, (-wp + disc) / jnp.maximum(pp, tiny), 0.0)
    # Roundoff can make the root marginally negative when ``w`` sits on the ball
    # and ``p`` points outward; the contract is a nonnegative step.
    return jnp.maximum(beta, 0.0)


def steihaug_step(
    w: Float[Array, " n"],
    p: Float[Array, " n"],
    alpha: Scalar,
    radius: Scalar,
    *,
    force_boundary: Bool[Array, ""] | bool = False,
) -> tuple[Float[Array, " n"], Bool[Array, ""]]:
    """Apply the Steihaug–Toint update rule to one Krylov iteration.

    Takes the conjugate-gradient step ``w + α p`` when it stays inside the
    ball; otherwise — or when the caller requests it, e.g. on nonpositive
    curvature — moves from ``w`` along ``p`` to the boundary instead
    (Steihaug 1983; Nocedal & Wright Algorithm 7.2). The caller stops the
    iteration when the returned flag is set.

    Parameters
    ----------
    w
        Current iterate inside the ball.
    p
        Search direction.
    alpha
        Conjugate-gradient step length along ``p``.
    radius
        Trust-region radius.
    force_boundary
        When ``True`` the boundary step is taken regardless of ``α``
        (negative-curvature exit).

    Returns
    -------
    w_new
        ``w + α p`` or the boundary point ``w + β p``.
    hit_boundary
        ``True`` when the boundary point was taken (iteration must stop).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.linalg import steihaug_step
    >>> w = jnp.zeros(2)
    >>> p = jnp.array([1.0, 0.0])
    >>> w_in, hit = steihaug_step(w, p, jnp.asarray(0.5), jnp.asarray(1.0))
    >>> (w_in.tolist(), bool(hit))
    ([0.5, 0.0], False)
    >>> w_out, hit = steihaug_step(w, p, jnp.asarray(3.0), jnp.asarray(1.0))
    >>> (w_out.tolist(), bool(hit))
    ([1.0, 0.0], True)
    """
    w_full = w + alpha * p
    crosses = jnp.dot(w_full, w_full) >= radius**2
    hit_boundary = crosses | jnp.asarray(force_boundary)
    beta = boundary_step_length(w, p, radius)
    w_new = jnp.where(hit_boundary, w + beta * p, w_full)
    return w_new, hit_boundary


class SteihaugCGResult(Module):
    """Outcome of :func:`steihaug_cg`.

    Attributes
    ----------
    w
        Final iterate (inside or on the ball).
    n_iter
        Number of conjugate-gradient iterations performed.
    on_boundary
        ``True`` when the iteration stopped by moving to the boundary
        (trust-region crossing or nonpositive curvature).
    converged
        ``True`` when the residual test ``‖r‖² ≤ tol_sq`` was met (or the
        search direction vanished).
    """

    w: Float[Array, " n"]
    n_iter: Array
    on_boundary: Bool[Array, ""]
    converged: Bool[Array, ""]


def steihaug_cg(
    hvp: Callable[[Float[Array, " n"]], Float[Array, " n"]],
    r0: Float[Array, " n"],
    w0: Float[Array, " n"],
    radius: Scalar,
    *,
    tol_sq: Scalar,
    max_iter: int,
    residual: Callable[[Float[Array, " n"]], Float[Array, " n"]] | None = None,
    project: Callable[[Float[Array, " n"]], Float[Array, " n"]] | None = None,
    curvature_floor: float = 0.0,
    done: Bool[Array, ""] | bool = False,
) -> SteihaugCGResult:
    """Truncated conjugate gradients on ``q(w) = ½ wᵀ H w − bᵀ w`` inside a ball.

    Steihaug (1983) / Toint (1981), Nocedal & Wright Algorithm 7.2, with the
    optional projection of Algorithm 16.2: every search direction lies in
    the range of ``project`` (e.g. ``null(Â)``), so a warm start ``w0`` in an
    affine subspace stays there. The iteration stops when

    * ``‖r‖² ≤ tol_sq`` (or the search direction vanishes) — converged;
    * the CG step leaves the ball or the curvature along ``p`` is at most
      ``curvature_floor ‖p‖²`` — the iterate is moved to the boundary along
      ``p`` and the iteration stops;
    * ``max_iter`` iterations were performed.

    Parameters
    ----------
    hvp
        Matrix-free product ``p ↦ H p`` (symmetric ``H``).
    r0
        Initial residual ``−∇q(w0)`` (already projected when ``project`` is
        used).
    w0
        Starting point inside the ball ``‖w0‖₂ ≤ radius``.
    radius
        Trust-region radius.
    tol_sq
        Convergence threshold on the squared residual norm.
    max_iter
        Static iteration cap.
    residual
        Optional ``w ↦ −∇q(w)`` recomputed from scratch at every new iterate
        (robust against drift of an inexact ``project``). Without it the
        standard recurrence ``r − α H p`` is used, projected when
        ``project`` is given.
    project
        Optional orthogonal projector applied to the recursive residual.
        Ignored when ``residual`` is given (that callable is expected to
        project itself).
    curvature_floor
        Nonpositive-curvature test ``pᵀ H p ≤ curvature_floor ‖p‖²``. Use
        ``0`` for a least-squares Hessian ``ÂᵀÂ`` (CGLS), a small positive
        scale-invariant value for indefinite Hessians.
    done
        Start in the terminated state and return ``w0`` untouched (for
        callers whose own gate says no step is possible).

    Returns
    -------
    SteihaugCGResult
        Final iterate and the stop reason.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.linalg import steihaug_cg
    >>> H = jnp.array([[2.0, 0.0], [0.0, 1.0]])
    >>> b = jnp.array([2.0, 1.0])  # unconstrained minimiser is (1, 1)
    >>> out = steihaug_cg(lambda p: H @ p, b, jnp.zeros(2), jnp.asarray(10.0),
    ...                   tol_sq=1e-12, max_iter=5)
    >>> (out.w.round(6).tolist(), bool(out.converged), bool(out.on_boundary))
    ([1.0, 1.0], True, False)
    >>> out = steihaug_cg(lambda p: H @ p, b, jnp.zeros(2), jnp.asarray(0.5),
    ...                   tol_sq=1e-12, max_iter=5)
    >>> (bool(jnp.isclose(jnp.linalg.norm(out.w), 0.5)), bool(out.on_boundary))
    (True, True)
    """
    dtype = w0.dtype
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype)
    floor = jnp.asarray(curvature_floor, dtype)

    def refresh(w_new, r, alpha, Hp):
        if residual is not None:
            return residual(w_new)
        r_new = r - alpha * Hp
        return r_new if project is None else project(r_new)

    def body(_i, carry):
        w, r, p, rz, done, ncg, on_bnd = carry

        def do(c):
            w, r, p, rz, _done, ncg, on_bnd = c
            Hp = hvp(p)
            pHp = jnp.dot(p, Hp)
            pp = jnp.dot(p, p)
            moving = pp > tiny
            neg_curv = moving & (pHp <= floor * pp)
            alpha = jnp.where(neg_curv, 0.0, rz / jnp.maximum(pHp, tiny))
            w_new, to_boundary = steihaug_step(
                w, p, alpha, radius, force_boundary=neg_curv
            )
            r_new = refresh(w_new, r, alpha, Hp)
            rz_new = jnp.dot(r_new, r_new)
            beta = jnp.where(rz > tiny, rz_new / jnp.maximum(rz, tiny), 0.0)
            p_new = r_new + beta * p
            conv = (rz_new <= tol_sq) | ~moving
            return (
                w_new,
                r_new,
                p_new,
                rz_new,
                to_boundary | conv,
                ncg + 1,
                on_bnd | to_boundary,
            )

        return jax.lax.cond(jnp.reshape(done, ()), lambda c: c, do, carry)

    rz0 = jnp.dot(r0, r0)
    init = (
        w0,
        r0,
        r0,
        rz0,
        jnp.reshape((rz0 <= tol_sq) | jnp.asarray(done), ()),
        jnp.zeros((), jnp.int32),
        jnp.asarray(False),
    )
    w, _, _, rz, _, n_iter, on_bnd = jax.lax.fori_loop(0, max_iter, body, init)
    return cast(
        SteihaugCGResult,
        SteihaugCGResult(
            w=w, n_iter=n_iter, on_boundary=on_bnd, converged=rz <= tol_sq
        ),
    )
