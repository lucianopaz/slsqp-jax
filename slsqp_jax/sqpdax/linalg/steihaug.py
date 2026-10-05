"""Trust-region boundary rules shared by the Steihaug–Toint style Krylov solvers."""

from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Scalar

__all__ = [
    "boundary_step_length",
    "steihaug_step",
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
