"""Ray / box intersection rules shared by the fraction-to-boundary backtracks."""

from jax import numpy as jnp
from jaxtyping import Array, Float, Scalar

__all__ = ["box_ray_length", "box_fraction", "box_segment_fraction"]


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


def box_segment_fraction(
    start: Float[Array, " n"],
    w: Float[Array, " n"],
    lo: Float[Array, " n"],
    hi: Float[Array, " n"],
) -> Scalar:
    """Largest ``t ∈ [0, 1]`` with ``lo ≤ start + t w ≤ hi`` componentwise.

    The box scaling of a correction ``w`` appended to a step ``start``
    (scipy's ``box_intersections`` for the second-order correction of
    ``trust-constr``). Unlike :func:`box_fraction` the segment starts at
    ``start`` rather than at the origin and both faces can bind.

    Parameters
    ----------
    start
        Point the segment starts from.
    w
        Segment direction (``t = 1`` is the full segment).
    lo, hi
        Lower and upper faces of the box (``∓inf`` on free coordinates).

    Returns
    -------
    Scalar
        The largest admissible ``t``; ``0`` when ``start`` itself is outside
        the box, so the segment never enters it from there.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.linalg import box_segment_fraction
    >>> lo, hi = jnp.array([0.0, -jnp.inf]), jnp.array([1.0, 2.0])
    >>> start = jnp.array([0.5, 0.0])
    >>> float(box_segment_fraction(start, jnp.array([0.25, 1.0]), lo, hi))
    1.0
    >>> float(box_segment_fraction(start, jnp.array([1.0, 1.0]), lo, hi))
    0.5
    >>> float(box_segment_fraction(start, jnp.array([0.0, 4.0]), lo, hi))
    0.5
    >>> float(box_segment_fraction(jnp.array([1.5, 0.0]), jnp.zeros(2), lo, hi))
    0.0
    """
    inside = jnp.all((start >= lo) & (start <= hi))
    to_lower = box_ray_length(w, lo - start)
    to_upper = box_ray_length(-w, start - hi)
    t = jnp.minimum(jnp.asarray(1.0, w.dtype), jnp.minimum(to_lower, to_upper))
    return jnp.where(inside, t, jnp.asarray(0.0, w.dtype))
