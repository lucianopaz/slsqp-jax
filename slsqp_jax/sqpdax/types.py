from dataclasses import replace
from typing import Self

from equinox import Module
from jaxtyping import Array, Float

__all__ = [
    "Vector_n",
    "Vector_meq",
    "Vector_mineq",
    "Matrix_meqn",
    "Matrix_mineqn",
    "InitializableModule",
]

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
        field whose current value is an :class:`InitializableModule`, the
        corresponding value must be a mapping of keyword arguments that is
        forwarded to that field's :meth:`init`. All other fields are
        replaced wholesale with the provided value.

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
        nested_initializable_fields = {
            k for k in kwargs if isinstance(getattr(self, k), InitializableModule)
        }
        keyvals = {}
        for key, val in kwargs.items():
            if key in nested_initializable_fields:
                keyvals[key] = getattr(self, key).init(**val)
            else:
                keyvals[key] = val
        return replace(self, **keyvals)
