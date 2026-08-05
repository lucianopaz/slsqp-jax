"""Kind registries and hashable option bags for sqpdax components."""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Mapping
from dataclasses import fields
from typing import ClassVar, get_args, get_type_hints

__all__ = [
    "KindRegistryMixin",
    "FrozenDict",
    "static_field_names",
    "kind_family_field_names",
    "freeze",
]


class KindRegistryMixin:
    """Auto-registering family base for constructible-by-``kind`` components.

    A concrete member declares ``kind: ClassVar[str]``; ``__init_subclass__``
    then registers it in the *family root's* ``_registry``. Each family
    declares its own ``_registry`` dict so the registries stay disjoint.

    :meth:`from_spec` builds a member from a ``{"kind": name, **params}``
    mapping, validating ``params`` against the target's ``__init__``
    signature (minus any runtime-``injected`` arguments) and warning on
    unknown params.

    Attributes
    ----------
    kind
        ClassVar string identifying this concrete member within its family.
        Only subclasses that define ``kind`` on their own class body are
        registered.
    _registry
        ClassVar mapping from kind name to concrete subclass. Each family
        root must supply its own mutable dict.
    """

    kind: ClassVar[str]
    _registry: ClassVar[dict[str, type]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        k = cls.__dict__.get("kind")
        if k is not None:
            cls._registry[k] = cls

    @classmethod
    def from_spec(cls, spec: Mapping, **injected):
        """Construct a registered family member from a kind specification.

        Parameters
        ----------
        spec
            Mapping that must contain a ``"kind"`` key naming a registered
            member. Remaining entries are forwarded as constructor keyword
            arguments after unknown keys are dropped (with a warning).
        **injected
            Runtime keyword arguments merged into the constructor call.
            Their names are excluded from the allowed ``spec`` keys so
            callers cannot override them via the spec mapping.

        Returns
        -------
        object
            An instance of the concrete class registered under
            ``spec["kind"]``.

        Raises
        ------
        ValueError
            If ``spec`` has no ``"kind"`` key, or the kind is not
            registered on this family's ``_registry``.
        """
        params = dict(spec)
        name = params.pop("kind", None)
        if name is None:
            raise ValueError(
                f"{cls.__name__} spec is missing a 'kind' key: {dict(spec)!r}"
            )
        try:
            target = cls._registry[name]
        except KeyError:
            raise ValueError(
                f"unknown {cls.__name__} kind '{name}'; "
                f"registered: {sorted(cls._registry)}"
            ) from None
        allowed = set(inspect.signature(target).parameters) - set(injected) - {"self"}
        unknown = set(params) - allowed
        if unknown:
            warnings.warn(
                f"{target.__name__}: ignoring unknown params {sorted(unknown)}"
            )
            params = {k: v for k, v in params.items() if k in allowed}
        return target(**params, **injected)


class FrozenDict(Mapping):
    """Hashable, immutable ``Mapping`` for storing option bags on a static field.

    Equinox flattens static fields into the pytree ``treedef`` (aux data),
    which must be hashable for ``jit`` / ``lax.while_loop`` caching -- a
    plain ``dict`` is not. Values are frozen recursively (nested mappings
    become :class:`FrozenDict`, lists/tuples become tuples).
    """

    def __init__(self, data: Mapping):
        """Freeze ``data`` into an immutable mapping.

        Parameters
        ----------
        data
            Mapping whose values are recursively passed through
            :func:`freeze`.
        """
        self._d = {k: freeze(v) for k, v in dict(data).items()}

    def __getitem__(self, k):
        return self._d[k]

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __hash__(self):
        return hash(tuple(sorted(self._d.items())))

    def __eq__(self, other):
        return isinstance(other, FrozenDict) and self._d == other._d

    def __repr__(self):
        return f"FrozenDict({self._d!r})"


def freeze(v):
    """Recursively convert mappings and sequences into hashable containers.

    Parameters
    ----------
    v
        Arbitrary value. Mappings become :class:`FrozenDict`; lists and
        tuples become tuples of frozen elements; all other values are
        returned unchanged.

    Returns
    -------
    object
        A hashable (when the leaves are hashable) frozen view of ``v``.
    """
    if isinstance(v, Mapping):
        return FrozenDict(v)
    if isinstance(v, (list, tuple)):
        return tuple(freeze(x) for x in v)
    return v


def static_field_names(cls) -> set[str]:
    """Return names of ``eqx.field(static=True)`` fields declared on ``cls``.

    Parameters
    ----------
    cls
        An equinox :class:`~equinox.Module` (or other dataclass) type.

    Returns
    -------
    set of str
        Field names whose dataclass metadata marks them as static.
    """
    return {f.name for f in fields(cls) if f.metadata.get("static", False)}


def kind_family_field_names(cls) -> set[str]:
    """Return field names typed as a :class:`KindRegistryMixin` family.

    A field matches when its annotated type is a subclass of
    :class:`KindRegistryMixin`, or a union that includes such a subclass
    (for example ``secant: Secant | None``).

    Parameters
    ----------
    cls
        A type whose annotations are inspected via
        :func:`typing.get_type_hints`.

    Returns
    -------
    set of str
        Names of fields whose type hint references a kind-registry family.
    """
    out: set[str] = set()
    for _name, hint in get_type_hints(cls).items():
        for arg in get_args(hint) or (hint,):
            if isinstance(arg, type) and issubclass(arg, KindRegistryMixin):
                out.add(_name)
    return out
