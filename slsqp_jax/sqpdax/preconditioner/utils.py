"""Helpers for building preconditioners from other sqpdax components."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import cast

import jax
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float
from lineax import FunctionLinearOperator, symmetric_tag

from ..secant.base import Secant
from .base import GenericPreconditioner

__all__ = [
    "linear_adjoint",
    "preconditioner_from_secant",
]


def linear_adjoint(
    fn: Callable[[Float[Array, " n"]], Float[Array, " m"]],
    in_structure: jax.ShapeDtypeStruct,
    tags: object | Iterable[object] = (),
) -> Callable[[Float[Array, " m"]], Float[Array, " n"]]:
    """Return the adjoint of a linear map via lineax.

    Wraps ``fn`` in a :class:`~lineax.FunctionLinearOperator` and returns
    the ``mv`` of its transpose. That is the same mechanism lineax uses
    internally:

    * if ``symmetric_tag`` (or another symmetry-implying tag) is present,
      the adjoint is ``fn`` itself;
    * otherwise the adjoint is built with :func:`jax.linear_transpose`
      at ``in_structure``.

    Prefer tagging symmetric operators (e.g. L-BFGS ``B`` / ``H``) rather
    than relying on :func:`jax.linear_transpose`: many matrix-free
    implementations contain data-dependent guards (``jnp.where``, …) that
    are linear in exact arithmetic but are not reverse-mode transposeable.

    Parameters
    ----------
    fn
        Linear map from the space described by ``in_structure`` to some
        output space.
    in_structure
        Abstract shape/dtype of vectors in the domain of ``fn``.
    tags
        Optional lineax tag or iterable of tags forwarded to
        :class:`~lineax.FunctionLinearOperator` (e.g. ``symmetric_tag``).

    Returns
    -------
    Callable
        Adjoint map ``v ↦ fnᵀ v``.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.preconditioner.utils import linear_adjoint
    >>> M = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> push = lambda x: M @ x
    >>> pull = linear_adjoint(push, jax.ShapeDtypeStruct((2,), jnp.float32))
    >>> pull(jnp.array([1.0, 0.0, -1.0])).tolist()
    [-4.0, -4.0]
    """
    return cast(
        FunctionLinearOperator, FunctionLinearOperator(fn, in_structure, tags=tags)
    ).T.mv


def preconditioner_from_secant(
    secant: Secant,
    n: int,
    dtype: jnp.dtype | None = None,
    *,
    inverse_as_forward: bool = True,
) -> GenericPreconditioner:
    """Build a :class:`~slsqp_jax.sqpdax.preconditioner.base.GenericPreconditioner` from a secant.

    Uses the secant's matrix-free ``hvp`` / ``inverse_hvp`` as the forward
    and inverse maps. Adjoints (``pullback``, ``invert_transpose``) are
    obtained with :func:`linear_adjoint` under
    :data:`~lineax.symmetric_tag`, matching
    :meth:`lineax.FunctionLinearOperator.transpose` for a symmetric
    operator (``B`` and ``H`` are symmetric by construction).

    Parameters
    ----------
    secant
        Curvature approximation exposing :meth:`~slsqp_jax.sqpdax.secant.base.Secant.hvp`
        and :meth:`~slsqp_jax.sqpdax.secant.base.Secant.inverse_hvp`.
    n
        Decision dimension (length of vectors accepted by the secant).
    dtype
        Floating dtype for the preconditioner's
        :class:`~jax.ShapeDtypeStruct` structures. Defaults to JAX's
        current default float dtype (``float64`` when x64 is enabled).
    inverse_as_forward
        If ``True`` (default), the forward map ``M`` is
        ``secant.inverse_hvp`` (``H``) and ``M⁻¹`` is ``secant.hvp``
        (``B``). If ``False``, the roles are swapped: ``M = B`` and
        ``M⁻¹ = H``.

    Returns
    -------
    GenericPreconditioner
        Preconditioner whose pushforward / invert wrap the secant
        operators and whose adjoints come from :func:`linear_adjoint`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.secant import LBFGS
    >>> from slsqp_jax.sqpdax.preconditioner.utils import preconditioner_from_secant
    >>> secant = LBFGS(n=2, memory=2)
    >>> prec = preconditioner_from_secant(secant, n=2)
    >>> v = jnp.array([1.0, -1.0])
    >>> # Empty L-BFGS: H = I, so M = H and M⁻¹ = B = I.
    >>> float(jnp.max(jnp.abs(prec.pushforward(v) - v)))
    0.0
    """
    if dtype is None:
        dtype = jnp.result_type(float)

    if inverse_as_forward:
        pushforward = secant.inverse_hvp
        invert = secant.hvp
    else:
        pushforward = secant.hvp
        invert = secant.inverse_hvp

    input_structure = jax.ShapeDtypeStruct(shape=(n,), dtype=dtype)
    output_structure = jax.ShapeDtypeStruct(shape=(n,), dtype=dtype)

    # Secant operators are symmetric: lineax's transpose short-circuits to
    # the forward map (same as tagging FunctionLinearOperator with
    # symmetric_tag). This avoids jax.linear_transpose on implementations
    # that contain non-transposeable guards.
    pullback = linear_adjoint(pushforward, input_structure, tags=symmetric_tag)
    invert_transpose = linear_adjoint(invert, output_structure, tags=symmetric_tag)

    return cast(
        GenericPreconditioner,
        GenericPreconditioner(
            input_structure=input_structure,
            output_structure=output_structure,
            pushforward=pushforward,
            pullback=pullback,
            invert=invert,
            invert_transpose=invert_transpose,
        ),
    )
