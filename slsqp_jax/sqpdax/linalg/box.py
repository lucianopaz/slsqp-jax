"""Ray / box intersection rules shared by the fraction-to-boundary backtracks."""

from jax import numpy as jnp
from jaxtyping import Array, Float, Scalar

__all__ = ["box_ray_length", "box_fraction"]


def box_ray_length(w: Float[Array, " n"], lo: Float[Array, " n"]) -> Scalar:
    """Largest ``α ≥ 0`` with ``α w ≥ lo`` componentwise.

    Only coordinates with ``w_i < 0`` and a finite face ``lo_i`` can bind;
    the result is ``+inf`` when none does. Faces are assumed non-positive
    (``lo ≤ 0``), as produced by the fraction-to-boundary rules, so the
    origin always lies in the box.

    Parameters
    ----------
    w
        Ray direction.
    lo
        Lower faces of the box (``−inf`` on free coordinates).

    Returns
    -------
    Scalar
        Ray length to the first face hit, or ``+inf``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.linalg import box_ray_length
    >>> w = jnp.array([1.0, -2.0, -0.5])
    >>> lo = jnp.array([-jnp.inf, -1.0, -1.0])
    >>> float(box_ray_length(w, lo))  # binds on the second coordinate: 1/2
    0.5
    """
    binding = (w < 0.0) & jnp.isfinite(lo)
    safe_w = jnp.where(binding, w, -1.0)
    ratios = jnp.where(binding, lo / safe_w, jnp.inf)
    return jnp.min(ratios, initial=jnp.asarray(jnp.inf, w.dtype))


def box_fraction(w: Float[Array, " n"], lo: Float[Array, " n"]) -> Scalar:
    """Largest ``β ∈ [0, 1]`` with ``β w ≥ lo`` componentwise.

    The fraction-to-boundary backtracking factor applied to a step ``w``:
    ``1`` when ``w`` already lies in the box, otherwise the ray length of
    :func:`box_ray_length`.

    Parameters
    ----------
    w
        Candidate step.
    lo
        Lower faces of the box (``−inf`` on free coordinates).

    Returns
    -------
    Scalar
        Backtracking factor ``β``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.linalg import box_fraction
    >>> lo = jnp.array([-jnp.inf, -1.0])
    >>> float(box_fraction(jnp.array([3.0, -0.5]), lo))
    1.0
    >>> float(box_fraction(jnp.array([3.0, -4.0]), lo))
    0.25
    """
    return jnp.minimum(jnp.asarray(1.0, w.dtype), box_ray_length(w, lo))
