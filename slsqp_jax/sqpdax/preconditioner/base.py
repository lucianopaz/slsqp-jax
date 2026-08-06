"""Matrix-free preconditioners for SQP / interior-point linear solves.

A :class:`Preconditioner` is written in terms of its forward map ``M``
(``pushforward`` / ``pullback``). The inverse action ``M⁻¹`` used by Krylov
solvers is either supplied directly by a subclass or obtained by solving
``M y = x`` against the forward
:class:`~lineax.AbstractLinearOperator` via lineax
(:func:`generic_invert`).

:class:`~slsqp_jax.sqpdax.preconditioner.lineax_compat.GenericLinearOperator`
bridges these callables into lineax so tagged operators can use direct
solvers (LU / Cholesky / diagonal / …).
"""

from abc import abstractmethod
from typing import Callable, ClassVar, cast

import equinox as eqx
import jax
from equinox import Module, field
from jax import numpy as jnp
from jaxtyping import Array, Float
from lineax import AbstractLinearSolver, AutoLinearSolver, linear_solve
from lineax._operator import (
    AbstractLinearOperator,
    IdentityLinearOperator,
    MatrixLinearOperator,
)

from ..registry import KindRegistryMixin
from .lineax_compat import GenericLinearOperator

__all__ = [
    "generic_invert",
    "generic_invert_transpose",
    "Preconditioner",
    "IdentityPreconditioner",
    "MatrixPreconditioner",
    "GenericPreconditioner",
]


def generic_invert(
    preconditioner: "Preconditioner", x: Float[Array, " m"]
) -> Float[Array, " m"]:
    """Apply ``M⁻¹`` by solving ``M y = x`` with lineax.

    Uses :class:`~lineax.AutoLinearSolver` on
    :meth:`Preconditioner.as_linear_operator`, so a well-tagged forward
    operator inverts cheaply (diagonal / triangular / SPD / …).

    Parameters
    ----------
    preconditioner
        Preconditioner whose forward operator is ``M``.
    x
        Right-hand side in the **output** space of ``M``.

    Returns
    -------
    jax.Array
        Solution ``y`` in the **input** space of ``M``.
    """
    return linear_solve(
        preconditioner.as_linear_operator(),
        x,
        solver=cast(AbstractLinearSolver, AutoLinearSolver(well_posed=None)),
    ).value


def generic_invert_transpose(
    preconditioner: "Preconditioner", x: Float[Array, " m"]
) -> Float[Array, " m"]:
    """Apply ``(Mᵀ)⁻¹`` by solving ``Mᵀ y = x`` with lineax.

    Lineax Krylov methods that transpose / conjugate the preconditioner
    need the inverse operator's adjoint; this supplies it from the
    forward transpose.

    Parameters
    ----------
    preconditioner
        Preconditioner whose forward operator is ``M``.
    x
        Right-hand side in the **input** space of ``M``.

    Returns
    -------
    jax.Array
        Solution ``y`` in the **output** space of ``M``.
    """
    return linear_solve(
        preconditioner.as_linear_operator().T,
        x,
        solver=cast(AbstractLinearSolver, AutoLinearSolver(well_posed=None)),
    ).value


class Preconditioner(KindRegistryMixin, Module):
    """Abstract forward map ``M`` with optional inverse for linear solvers.

    Subclasses implement :meth:`pushforward` (``M``) and :meth:`pullback`
    (``Mᵀ``). Default :meth:`invert` / :meth:`invert_transpose` solve
    against the forward operator via :func:`generic_invert`; subclasses
    with a cheap direct inverse override those methods.

    Concrete kinds register themselves on :attr:`_registry` through
    :class:`~slsqp_jax.sqpdax.registry.KindRegistryMixin` when they declare
    a ``kind`` class attribute.

    Attributes
    ----------
    input_structure
        Abstract shape/dtype of vectors in the domain of ``M``.
        Static so JAX does not promote it to a traced ``ShapedArray``.
    output_structure
        Abstract shape/dtype of vectors in the codomain of ``M``.
    """

    _registry: ClassVar[dict] = {}

    input_structure: jax.ShapeDtypeStruct = field(static=True)
    output_structure: jax.ShapeDtypeStruct = field(static=True)

    @abstractmethod
    def pushforward(self, x: Float[Array, " m"]) -> Float[Array, " m"]:
        """Apply the forward map ``M`` (domain → codomain).

        Parameters
        ----------
        x
            Vector matching :attr:`input_structure`.

        Returns
        -------
        jax.Array
            ``M x``, matching :attr:`output_structure`.
        """
        ...

    @abstractmethod
    def pullback(self, x: Float[Array, " m"]) -> Float[Array, " m"]:
        """Apply the adjoint ``Mᵀ`` (codomain → domain).

        Parameters
        ----------
        x
            Vector matching :attr:`output_structure`.

        Returns
        -------
        jax.Array
            ``Mᵀ x``, matching :attr:`input_structure`.
        """
        ...

    def as_linear_operator(self) -> AbstractLinearOperator:
        """Return the forward operator ``M`` as a lineax linear operator.

        ``mv`` is :meth:`pushforward`. This must **not** be the inverse —
        :meth:`invert` solves against this operator, so wrapping ``invert``
        here would recurse forever.

        Returns
        -------
        lineax.AbstractLinearOperator
            Forward operator with matching in/out structures.
        """
        return cast(
            AbstractLinearOperator,
            GenericLinearOperator(
                pushforward=self.pushforward,
                pullback=self.pullback,
                input_structure=self.input_structure,
                output_structure=self.output_structure,
            ),
        )

    def invert(self, x: Float[Array, " m"]) -> Float[Array, " m"]:
        """Apply ``M⁻¹`` (default: lineax solve of ``M y = x``).

        Parameters
        ----------
        x
            Vector matching :attr:`output_structure`.

        Returns
        -------
        jax.Array
            ``M⁻¹ x``, matching :attr:`input_structure`.
        """
        return generic_invert(self, x)

    def invert_transpose(self, x: Float[Array, " m"]) -> Float[Array, " m"]:
        """Apply ``(Mᵀ)⁻¹`` (default: lineax solve of ``Mᵀ y = x``).

        Parameters
        ----------
        x
            Vector matching :attr:`input_structure`.

        Returns
        -------
        jax.Array
            ``(Mᵀ)⁻¹ x``, matching :attr:`output_structure`.
        """
        return generic_invert_transpose(self, x)

    def as_inverse_linear_operator(self) -> AbstractLinearOperator:
        """Return the inverse operator whose ``mv`` applies ``M⁻¹``.

        This is the form lineax Krylov methods expect from a
        *preconditioner* (``preconditioner.mv(residual)`` applies the
        inverse action). The adjoint is ``(Mᵀ)⁻¹``, so non-symmetric
        ``M`` is handled correctly.

        Returns
        -------
        lineax.AbstractLinearOperator
            Inverse operator (codomain → domain).
        """
        return cast(
            AbstractLinearOperator,
            GenericLinearOperator(
                pushforward=self.invert,
                pullback=self.invert_transpose,
                input_structure=self.output_structure,
                output_structure=self.input_structure,
            ),
        )


class IdentityPreconditioner(Preconditioner):
    """Identity preconditioner ``M = I``.

    All four maps (forward, adjoint, inverse, inverse-transpose) are the
    identity. Lineax operators are
    :class:`~lineax.IdentityLinearOperator` instances.

    Parameters
    ----------
    x
        Prototype vector; its shape and dtype define
        :attr:`~Preconditioner.input_structure` /
        :attr:`~Preconditioner.output_structure`.
    """

    kind: ClassVar[str] = "identity"

    def __init__(self, x: Float[Array, " m"]):
        self.input_structure = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
        self.output_structure = self.input_structure

    def pushforward(self, x: Float[Array, " m"]) -> Float[Array, " m"]:
        """Return ``x`` unchanged."""
        return x

    def pullback(self, x: Float[Array, " m"]) -> Float[Array, " m"]:
        """Return ``x`` unchanged."""
        return x

    def invert(self, x: Float[Array, " m"]) -> Float[Array, " m"]:
        """Return ``x`` unchanged."""
        return x

    def invert_transpose(self, x: Float[Array, " m"]) -> Float[Array, " m"]:
        """Return ``x`` unchanged."""
        return x

    def as_linear_operator(self) -> AbstractLinearOperator:
        """Identity forward operator."""
        return cast(
            AbstractLinearOperator, IdentityLinearOperator(self.input_structure)
        )

    def as_inverse_linear_operator(self) -> AbstractLinearOperator:
        """Identity inverse operator."""
        return cast(
            AbstractLinearOperator, IdentityLinearOperator(self.output_structure)
        )


class MatrixPreconditioner(Preconditioner):
    """Dense-matrix preconditioner ``M`` with least-squares inverse.

    Forward / adjoint use matrix-vector products. :meth:`invert` /
    :meth:`invert_transpose` use :func:`jax.numpy.linalg.lstsq` (so
    rectangular ``M`` is allowed). The forward lineax operator is a
    :class:`~lineax.MatrixLinearOperator`.

    Parameters
    ----------
    matrix
        Dense matrix of shape ``(out, in)`` representing ``M``.

    Attributes
    ----------
    matrix
        The stored dense matrix ``M``.
    """

    kind: ClassVar[str] = "matrix"

    matrix: Float[Array, " a b"]

    def __init__(self, matrix: Float[Array, " a b"]):
        self.matrix = matrix
        self.input_structure = jax.ShapeDtypeStruct(
            shape=(matrix.shape[1],), dtype=matrix.dtype
        )
        self.output_structure = jax.ShapeDtypeStruct(
            shape=(matrix.shape[0],), dtype=matrix.dtype
        )

    def pushforward(self, x: Float[Array, " b"]) -> Float[Array, " a"]:
        """Return ``M @ x``."""
        return self.matrix @ x

    def pullback(self, x: Float[Array, " a"]) -> Float[Array, " b"]:
        """Return ``M.T @ x``."""
        return self.matrix.T @ x

    def invert(self, x: Float[Array, " a"]) -> Float[Array, " b"]:
        """Return a least-squares solution of ``M y = x``."""
        return jnp.linalg.lstsq(self.matrix, x)[0]

    def invert_transpose(self, x: Float[Array, " b"]) -> Float[Array, " a"]:
        """Return a least-squares solution of ``M.T y = x``."""
        return jnp.linalg.lstsq(self.matrix.T, x)[0]

    def as_linear_operator(self) -> AbstractLinearOperator:
        """Dense matrix operator wrapping :attr:`matrix`."""
        return cast(AbstractLinearOperator, MatrixLinearOperator(self.matrix))


class GenericPreconditioner(Preconditioner):
    """Preconditioner built from user callables for ``M`` and optionally ``M⁻¹``.

    ``pushforward`` / ``pullback`` are always required. When ``invert`` /
    ``invert_transpose`` are omitted, the base lineax-solve fallbacks are
    used. Callables are closure-converted so captured arrays remain
    traced leaves under JIT / grad / vmap.

    Parameters
    ----------
    input_structure
        Abstract shape/dtype of the domain of ``M``.
    output_structure
        Abstract shape/dtype of the codomain of ``M``.
    pushforward
        Callable implementing ``M``.
    pullback
        Callable implementing ``Mᵀ``.
    invert
        Optional direct ``M⁻¹``. ``None`` selects :func:`generic_invert`.
    invert_transpose
        Optional direct ``(Mᵀ)⁻¹``. ``None`` selects
        :func:`generic_invert_transpose`.
    """

    kind: ClassVar[str] = "generic"

    _pushforward: Callable[[Float[Array, " a"]], Float[Array, " b"]]
    _pullback: Callable[[Float[Array, " b"]], Float[Array, " a"]]
    _invert: Callable[[Float[Array, " b"]], Float[Array, " a"]] | None
    _invert_transpose: Callable[[Float[Array, " a"]], Float[Array, " b"]] | None

    def __init__(
        self,
        input_structure: jax.ShapeDtypeStruct,
        output_structure: jax.ShapeDtypeStruct,
        pushforward: Callable[[Float[Array, " a"]], Float[Array, " b"]],
        pullback: Callable[[Float[Array, " b"]], Float[Array, " a"]],
        invert: Callable[[Float[Array, " b"]], Float[Array, " a"]] | None = None,
        invert_transpose: Callable[[Float[Array, " a"]], Float[Array, " b"]]
        | None = None,
    ):
        self._pushforward = eqx.filter_closure_convert(pushforward, input_structure)
        self._pullback = eqx.filter_closure_convert(pullback, output_structure)
        # A ``partial(generic_invert, self)`` cannot be closure-converted here:
        # it would trace ``self.as_linear_operator()`` before the structures
        # exist, and storing a closure over ``self`` inside ``self`` makes the
        # pytree self-referential. Keep ``None`` and fall back at call time.
        self._invert = (
            None
            if invert is None
            else eqx.filter_closure_convert(invert, output_structure)
        )
        self._invert_transpose = (
            None
            if invert_transpose is None
            else eqx.filter_closure_convert(invert_transpose, input_structure)
        )
        self.input_structure = input_structure
        self.output_structure = output_structure

    def pushforward(self, x: Float[Array, " a"]) -> Float[Array, " b"]:
        """Apply the user-supplied forward map."""
        return self._pushforward(x)

    def pullback(self, x: Float[Array, " b"]) -> Float[Array, " a"]:
        """Apply the user-supplied adjoint map."""
        return self._pullback(x)

    def invert(self, x: Float[Array, " b"]) -> Float[Array, " a"]:
        """Apply the direct inverse, or :func:`generic_invert` if none given."""
        if self._invert is None:
            return generic_invert(self, x)
        return self._invert(x)

    def invert_transpose(self, x: Float[Array, " a"]) -> Float[Array, " b"]:
        """Apply the direct inverse-transpose, or the lineax fallback."""
        if self._invert_transpose is None:
            return generic_invert_transpose(self, x)
        return self._invert_transpose(x)

    def as_linear_operator(self) -> AbstractLinearOperator:
        """Forward :class:`GenericLinearOperator` (no re-closure-convert)."""
        return cast(
            AbstractLinearOperator,
            GenericLinearOperator(
                pushforward=self._pushforward,
                pullback=self._pullback,
                input_structure=self.input_structure,
                output_structure=self.output_structure,
                closure_convert=False,
            ),
        )
