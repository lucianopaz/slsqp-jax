"""Lineax adapters for matrix-free preconditioner callables.

:class:`GenericLinearOperator` wraps a pushforward / pullback pair as a
:class:`~lineax.AbstractLinearOperator` and registers the lineax
singledispatch predicates (symmetry, diagonal, materialise, …) so
:class:`~lineax.AutoLinearSolver` can route tagged operators to direct
solvers.
"""

from typing import Callable, Iterable, cast

import equinox as eqx
import jax
from equinox import field
from jax import numpy as jnp
from jaxtyping import Array, Float
from lineax import (
    conj,
    diagonal,
    diagonal_tag,
    is_diagonal,
    is_lower_triangular,
    is_negative_semidefinite,
    is_positive_semidefinite,
    is_symmetric,
    is_tridiagonal,
    is_upper_triangular,
    linearise,
    lower_triangular_tag,
    materialise,
    negative_semidefinite_tag,
    positive_semidefinite_tag,
    symmetric_tag,
    tridiagonal_tag,
    upper_triangular_tag,
)
from lineax._operator import AbstractLinearOperator, MatrixLinearOperator

__all__ = [
    "GenericLinearOperator",
]


class GenericLinearOperator(AbstractLinearOperator):
    """Lineax operator whose ``mv`` / transpose are arbitrary callables.

    Callables are dynamic pytree leaves after
    :func:`equinox.filter_closure_convert` (same pattern as
    :class:`~lineax.FunctionLinearOperator`). Structures and tags are
    static so abstract evaluation does not promote them to
    ``ShapedArray``.

    Parameters
    ----------
    pushforward
        Map from the input space to the output space (``mv``).
    pullback
        Map from the output space to the input space (``mv`` of the
        transpose).
    input_structure
        Abstract shape/dtype of input vectors.
    output_structure
        Abstract shape/dtype of output vectors.
    tags
        Optional lineax tag or frozenset of tags
        (``symmetric_tag``, ``diagonal_tag``, …).
    closure_convert
        When ``True`` (default), closure-convert the callables so
        captured arrays remain traced under JIT / grad / vmap. Pass
        ``False`` when the callables are already converted (e.g. after
        :meth:`transpose`).

    Attributes
    ----------
    input_structure
        Static input :class:`~jax.ShapeDtypeStruct`.
    output_structure
        Static output :class:`~jax.ShapeDtypeStruct`.
    tags
        Static frozenset of lineax property tags.
    """

    _pushforward: Callable[[Float[Array, " a"]], Float[Array, " b"]]
    _pullback: Callable[[Float[Array, " b"]], Float[Array, " a"]]
    input_structure: jax.ShapeDtypeStruct = field(static=True)
    output_structure: jax.ShapeDtypeStruct = field(static=True)
    tags: frozenset[object] = field(static=True)

    def __init__(
        self,
        pushforward: Callable[[Float[Array, " a"]], Float[Array, " b"]],
        pullback: Callable[[Float[Array, " b"]], Float[Array, " a"]],
        input_structure: jax.ShapeDtypeStruct,
        output_structure: jax.ShapeDtypeStruct,
        tags: object | frozenset[object] = frozenset(),
        closure_convert: bool = True,
    ):
        if closure_convert:
            # Hoist closed-over tracers so the operator survives jit / grad /
            # vmap (``pushforward`` maps in→out, ``pullback`` out→in).
            pushforward = eqx.filter_closure_convert(pushforward, input_structure)
            pullback = eqx.filter_closure_convert(pullback, output_structure)
        object.__setattr__(self, "_pushforward", pushforward)
        object.__setattr__(self, "_pullback", pullback)
        object.__setattr__(self, "input_structure", input_structure)
        object.__setattr__(self, "output_structure", output_structure)
        if isinstance(tags, frozenset):
            tag_set = tags
        else:
            try:
                tag_set = frozenset(cast(Iterable[object], tags))
            except TypeError:
                tag_set = frozenset({tags})
        object.__setattr__(self, "tags", tag_set)

    def mv(self, vector: Float[Array, " a"]) -> Float[Array, " b"]:
        """Apply the forward map to ``vector``."""
        return self._pushforward(vector)

    def as_matrix(self) -> Float[Array, " out in"]:
        """Materialise ``M`` by mapping the identity through ``pushforward``.

        Returns
        -------
        jax.Array
            Dense matrix of shape ``(out, in)``.
        """
        # ``vmap(pushforward)`` over identity rows gives ``[M e_0, ...]`` of
        # shape (in, out); transpose for the (out, in) layout.
        eye = jnp.eye(self.input_structure.shape[0], dtype=self.input_structure.dtype)
        return jax.vmap(self._pushforward)(eye).T

    def transpose(self) -> "GenericLinearOperator":
        """Return the adjoint operator (swapped structures and callables)."""
        return cast(
            GenericLinearOperator,
            GenericLinearOperator(
                pushforward=self._pullback,
                pullback=self._pushforward,
                input_structure=self.output_structure,
                output_structure=self.input_structure,
                tags=self.tags,
                closure_convert=False,
            ),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        """Input :class:`~jax.ShapeDtypeStruct`."""
        return self.input_structure

    def out_structure(self) -> jax.ShapeDtypeStruct:
        """Output :class:`~jax.ShapeDtypeStruct`."""
        return self.output_structure


# lineax exposes operator properties through ``functools.singledispatch``
# predicates. A custom operator that does not register them raises
# ``NotImplementedError`` — including at construction, because
# ``AbstractLinearOperator.__check_init__`` calls ``is_symmetric(self)``.
# We mirror lineax's ``MatrixLinearOperator`` / ``FunctionLinearOperator``
# and read properties off ``tags``.


@is_symmetric.register(GenericLinearOperator)
def _(operator: GenericLinearOperator) -> bool:
    """True when any symmetry-implying tag is present."""
    return any(
        tag in operator.tags
        for tag in (
            symmetric_tag,
            positive_semidefinite_tag,
            negative_semidefinite_tag,
            diagonal_tag,
        )
    )


@is_diagonal.register(GenericLinearOperator)
def _(operator: GenericLinearOperator) -> bool:
    """True when ``diagonal_tag`` is set."""
    return diagonal_tag in operator.tags


@is_tridiagonal.register(GenericLinearOperator)
def _(operator: GenericLinearOperator) -> bool:
    """True when ``tridiagonal_tag`` is set."""
    return tridiagonal_tag in operator.tags


@is_lower_triangular.register(GenericLinearOperator)
def _(operator: GenericLinearOperator) -> bool:
    """True when ``lower_triangular_tag`` is set."""
    return lower_triangular_tag in operator.tags


@is_upper_triangular.register(GenericLinearOperator)
def _(operator: GenericLinearOperator) -> bool:
    """True when ``upper_triangular_tag`` is set."""
    return upper_triangular_tag in operator.tags


@is_positive_semidefinite.register(GenericLinearOperator)
def _(operator: GenericLinearOperator) -> bool:
    """True when ``positive_semidefinite_tag`` is set."""
    return positive_semidefinite_tag in operator.tags


@is_negative_semidefinite.register(GenericLinearOperator)
def _(operator: GenericLinearOperator) -> bool:
    """True when ``negative_semidefinite_tag`` is set."""
    return negative_semidefinite_tag in operator.tags


@conj.register(GenericLinearOperator)
def _(operator: GenericLinearOperator) -> GenericLinearOperator:
    """Return an operator that conjugates through the action.

    Keeps lineax's complex code paths (e.g. GMRES) well-typed for real
    operators by folding conjugation into pushforward / pullback.
    """
    return cast(
        GenericLinearOperator,
        GenericLinearOperator(
            pushforward=lambda v: jnp.conj(operator._pushforward(jnp.conj(v))),
            pullback=lambda v: jnp.conj(operator._pullback(jnp.conj(v))),
            input_structure=operator.input_structure,
            output_structure=operator.output_structure,
            tags=operator.tags,
        ),
    )


@linearise.register(GenericLinearOperator)
def _(operator: GenericLinearOperator) -> GenericLinearOperator:
    """No-op: the operator is already linear."""
    return operator


@materialise.register(GenericLinearOperator)
def _(operator: GenericLinearOperator) -> MatrixLinearOperator:
    """Dense :class:`~lineax.MatrixLinearOperator` from :meth:`~GenericLinearOperator.as_matrix`."""
    return cast(
        MatrixLinearOperator,
        MatrixLinearOperator(operator.as_matrix(), operator.tags),
    )


@diagonal.register(GenericLinearOperator)
def _(operator: GenericLinearOperator) -> Array:
    """Main diagonal of the materialised matrix."""
    return jnp.diagonal(operator.as_matrix())
