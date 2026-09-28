from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Protocol, Self

from equinox import Module
from jaxtyping import Array, Float

__all__ = [
    "Aux",
    "Scalar",
    "Vector_n",
    "Vector_meq",
    "Vector_mineq",
    "Matrix_meqn",
    "Matrix_mineqn",
    "InitializableModule",
    "RawObjectiveFn",
    "ObjectiveFn",
    "ObjectiveGradFn",
    "ObjectiveHVPFn",
    "EqConstraintFn",
    "EqConstraintJacFn",
    "EqConstraintHVPFn",
    "IneqConstraintFn",
    "IneqConstraintJacFn",
    "IneqConstraintHVPFn",
]

Aux = Any

Scalar = Float[Array, ""]
Vector_n = Float[Array, " n"]
Vector_meq = Float[Array, " meq"]
Vector_mineq = Float[Array, " mineq"]
Matrix_meqn = Float[Array, " meq n"]
Matrix_mineqn = Float[Array, " mineq n"]


class InitializableModule(Module):
    """Equinox module that supports deferred, recursive field initialization.

    Subclasses typically construct with placeholder fields (often ``None``)
    and later call :meth:`init` to fill them in. Nested fields that are
    themselves :class:`InitializableModule` instances are initialized
    recursively from nested keyword mappings, so a single top-level
    ``init`` call can configure an entire tree of components.
    """

    def init(self, **kwargs) -> Self:
        """Return a copy of this module with selected fields replaced.

        Keyword arguments must name existing fields of ``self``. For a
        field whose current value is an :class:`InitializableModule`, a
        mapping value is forwarded as keyword arguments to that field's
        :meth:`init`; any other value (e.g. a different module instance)
        replaces the field wholesale, as for all other fields.

        Fields may be updated whether their current value is ``None`` or
        already set. The original module is left unchanged; replacement
        uses :func:`dataclasses.replace`.

        Parameters
        ----------
        **kwargs
            Field names mapped to replacement values, or to nested
            ``init`` keyword mappings for :class:`InitializableModule`
            children.

        Returns
        -------
        Self
            A new module of the same type with the requested fields
            updated.
        """
        keyvals = {}
        for key, val in kwargs.items():
            current = getattr(self, key)
            if isinstance(current, InitializableModule) and isinstance(val, Mapping):
                keyvals[key] = current.init(**val)
            else:
                keyvals[key] = val
        return replace(self, **keyvals)


class RawObjectiveFn(Protocol):
    def __call__(self, x: Vector_n, *args, **kwargs) -> Scalar: ...


class ObjectiveFn(Protocol):
    def __call__(self, x: Vector_n, *args, **kwargs) -> tuple[Scalar, Aux]: ...


class ObjectiveGradFn(Protocol):
    def __call__(self, x: Vector_n, *args, **kwargs) -> Vector_n: ...


class ObjectiveHVPFn(Protocol):
    def __call__(self, x: Vector_n, tangent: Vector_n, *args, **kwargs) -> Vector_n: ...


class EqConstraintFn(Protocol):
    def __call__(self, x: Vector_n, *args, **kwargs) -> Vector_meq: ...


class EqConstraintJacFn(Protocol):
    def __call__(self, x: Vector_n, *args, **kwargs) -> Matrix_meqn: ...


class EqConstraintHVPFn(Protocol):
    def __call__(
        self, x: Vector_n, tangent: Vector_n, *args, **kwargs
    ) -> Matrix_meqn: ...


class IneqConstraintFn(Protocol):
    def __call__(self, x: Vector_n, *args, **kwargs) -> Vector_mineq: ...


class IneqConstraintJacFn(Protocol):
    def __call__(self, x: Vector_n, *args, **kwargs) -> Matrix_mineqn: ...


class IneqConstraintHVPFn(Protocol):
    def __call__(
        self, x: Vector_n, tangent: Vector_n, *args, **kwargs
    ) -> Matrix_mineqn: ...
