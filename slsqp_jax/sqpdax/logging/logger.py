"""Hierarchical, JIT-safe logger built on :func:`jax.debug.callback`."""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Self, cast

import equinox as eqx
import jax
import numpy as np
from equinox._enum import EnumerationItem
from jax import numpy as jnp

from ..registry import FrozenDict
from .handlers import DiagnosticsHandler, Handler, StreamHandler
from .levels import DEBUG, DISABLED, ERROR, INFO, WARNING, parse_level
from .record import (
    DEFAULT_FORMAT,
    DiagnosticRecord,
    LogRecord,
    template_fields,
    validate_format,
)

__all__ = [
    "Logger",
    "format_fields",
]

_RESERVED_ROOT_KEYS = frozenset(
    {"level", "handler", "format", "indent", "axis_name", "ordered", "diagnostics"}
)
# Keys a *child* entry may carry besides its grandchildren.
_CHILD_SETTING_KEYS = frozenset({"level", "diagnostics"})
_STDOUT = StreamHandler()


def _enum_name(item: EnumerationItem) -> str:
    """Attribute name of an ``Enumeration`` item (its message as fallback)."""
    enumeration = item._enumeration
    index = int(np.asarray(item._value).item())
    for name, member in getattr(enumeration, "_name_to_item", {}).items():
        if int(np.asarray(member._value).item()) == index:
            return name
    return enumeration[item]  # pragma: no cover


def _leaf_to_host(value: Any) -> Any:
    """Convert one callback leaf into a Python scalar, numpy array or name."""
    # ``None`` never reaches here: it is an empty pytree node for ``jax.tree.map``.
    if isinstance(value, EnumerationItem):
        return _enum_name(value)
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    return arr


def _is_host_leaf(value: Any) -> bool:
    return isinstance(value, EnumerationItem)


def _to_host(value: Any) -> Any:
    """Convert a callback argument (any pytree) leaf-wise with :func:`_leaf_to_host`."""
    return jax.tree.map(_leaf_to_host, value, is_leaf=_is_host_leaf)


def _as_predicate(when: Any) -> jax.Array:
    """Validate a ``when`` argument at trace time and return it as a 0-d array.

    ``None`` means "always", and is returned as a constant ``True``.
    """
    when_arr = jnp.asarray(True) if when is None else jnp.asarray(when)
    if when_arr.ndim != 0:
        raise TypeError(
            f"`when` must be a scalar boolean, got an array of shape "
            f"{when_arr.shape}; reduce it first with jnp.any or jnp.all"
        )
    return when_arr


def _is_true(when: Any) -> bool:
    # ``when`` is always 0-d here: it is validated by :func:`_as_predicate` and
    # ``jax.debug.callback`` maps over batch dimensions under ``vmap``.
    return bool(np.asarray(when).item())


def _run_to_host(run: Any) -> int | None:
    return None if run is None else int(np.asarray(run).item())


@dataclass(frozen=True)
class _EmitSpec:
    """Static part of one emission, closed over by the host callback."""

    name: str
    level: int
    msg: str
    fmt: str
    indent: str
    handler: Handler


def _emit(spec: _EmitSpec, when: Any, run: Any, values: dict[str, Any]) -> None:
    if not _is_true(when):
        return
    host_values = {k: _to_host(v) for k, v in values.items()}
    record = LogRecord(
        name=spec.name,
        levelno=spec.level,
        message=spec.msg.format(**host_values),
        run=_run_to_host(run),
        values=host_values,
    )
    spec.handler.emit(record, record.render(spec.fmt, spec.indent))


@dataclass(frozen=True)
class _DiagnosticSpec:
    """Static part of one diagnostic emission, closed over by the host callback."""

    name: str
    kind: str
    handler: DiagnosticsHandler


def _emit_diagnostic(
    spec: _DiagnosticSpec, when: Any, run: Any, values: dict[str, Any]
) -> None:
    # Re-check on the host: under ``vmap`` the device-side ``cond`` lowers to a
    # ``select`` and both branches run, so ``when`` can be ``False`` here.
    if not _is_true(when):
        return
    record = DiagnosticRecord(
        name=spec.name,
        kind=spec.kind,
        run=_run_to_host(run),
        values={k: _to_host(v) for k, v in values.items()},
    )
    spec.handler.emit(record)


class Logger(eqx.Module):
    """Per-component logger whose calls compile into ordered debug callbacks.

    The design mimics the standard library: loggers form a dotted hierarchy
    (``minimiser`` → ``minimiser.subproblem``), each with an effective
    level inherited from its parent unless overridden, and a message is a
    :meth:`str.format` template whose placeholders name traced values.

    Every field is static, so a logger can sit on an ``eqx.field(static=True)``
    of any component without changing the pytree structure; reconfiguring a
    level changes the treedef and therefore triggers the recompilation that
    makes the new configuration take effect.

    Two properties matter under ``jit``:

    * the level test runs *at trace time*, so a call below the threshold
      adds nothing to the jaxpr;
    * an enabled call binds :func:`jax.debug.callback` with
      ``ordered=True`` (by default), so records appear in program order even
      inside :func:`jax.lax.while_loop`.

    Under :func:`jax.vmap` the callback is unrolled over the batch. To tag
    each record with its batch index, give the ``vmap`` an ``axis_name`` and
    pass the same name as :attr:`axis_name`; outside such a ``vmap`` the
    index is simply omitted.

    Only arrays (and Equinox ``Enumeration`` items, rendered as their item
    name, e.g. ``successful``) can travel through the callback; string-valued
    fields must be baked into the template.

    Besides the text channel, a logger carries a binary **diagnostics
    channel** (:meth:`diagnostic`) that ships structured array payloads to a
    :class:`~slsqp_jax.sqpdax.logging.handlers.DiagnosticsHandler`. The
    handler is shared by the whole tree; the per-node :attr:`diagnostics`
    flag is inherited like :attr:`level` and can be overridden per child.

    Attributes
    ----------
    name
        Dotted logger name.
    level
        Threshold; calls with a lower level are dropped at trace time.
    handler
        Destination of rendered records; ``None`` writes to :data:`sys.stdout`.
    fmt
        Record template, see :meth:`~slsqp_jax.sqpdax.logging.record.LogRecord.render`.
    indent
        Indentation unit applied once per nesting level.
    axis_name
        Name of the ``vmap`` axis whose index identifies the run.
    ordered
        Whether to request ordered side effects from
        :func:`jax.debug.callback`.
    diagnostics_handler
        Shared sink for :meth:`diagnostic` records, or ``None``.
    diagnostics
        Whether *this* node emits diagnostics (only effective together with
        a :attr:`diagnostics_handler`).
    children
        Level / diagnostics configuration for descendants (see
        :meth:`from_options`).

    Examples
    --------
    >>> import jax, jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.logging import Logger, MemoryHandler
    >>> handler = MemoryHandler()
    >>> log = Logger.from_options(
    ...     {"level": "INFO", "handler": handler, "subproblem": "DEBUG"}
    ... )
    >>> @jax.jit
    ... def f(x):
    ...     log.info("x={x:.2f}", x=x)
    ...     log.child("subproblem").debug("inner x={x:.1f}", x=2 * x)
    ...     log.debug("dropped at trace time", x=x)
    ...     return x
    >>> _ = f(jnp.asarray(1.5))
    >>> handler.lines
    ['minimiser INFO: x=1.50', '  minimiser.subproblem DEBUG: inner x=3.0']
    """

    name: str = eqx.field(static=True, default="minimiser")
    level: int = eqx.field(static=True, default=DISABLED)
    handler: Handler | None = eqx.field(static=True, default=None)
    fmt: str = eqx.field(static=True, default=DEFAULT_FORMAT)
    indent: str = eqx.field(static=True, default="  ")
    axis_name: str | None = eqx.field(static=True, default=None)
    ordered: bool = eqx.field(static=True, default=True)
    diagnostics_handler: DiagnosticsHandler | None = eqx.field(
        static=True, default=None
    )
    diagnostics: bool = eqx.field(static=True, default=False)
    children: FrozenDict = eqx.field(
        static=True, default_factory=lambda: FrozenDict({})
    )

    def __check_init__(self) -> None:
        validate_format(self.fmt)
        if not isinstance(self.diagnostics, bool):
            raise TypeError(
                f"diagnostics must be a bool; got {type(self.diagnostics).__name__}"
            )
        if self.diagnostics_handler is not None and not isinstance(
            self.diagnostics_handler, DiagnosticsHandler
        ):
            raise TypeError(
                "diagnostics_handler must be a DiagnosticsHandler; got "
                f"{type(self.diagnostics_handler).__name__}"
            )

    # ------------------------------------------------------------ building

    @classmethod
    def disabled(cls) -> Logger:
        """The shared logger that never emits.

        Returns
        -------
        Logger
            Module-level singleton with ``level == DISABLED``.
        """
        return _DISABLED

    @classmethod
    def from_options(cls, spec: Any, *, name: str = "minimiser") -> Logger:
        """Build a root logger from an ``options['logging']`` entry.

        Parameters
        ----------
        spec
            One of

            * ``None`` / ``False`` — the disabled logger;
            * ``True`` — ``INFO`` to stdout;
            * a level (name or int) — that level to stdout;
            * a :class:`~slsqp_jax.sqpdax.logging.handlers.Handler` — ``INFO``
              to that handler;
            * a :class:`~slsqp_jax.sqpdax.logging.handlers.DiagnosticsHandler`
              — text logging disabled, diagnostics enabled everywhere and
              written to that handler;
            * a mapping. Reserved keys are ``level`` (default ``INFO``),
              ``handler``, ``format``, ``indent``, ``axis_name``, ``ordered``
              and ``diagnostics``; every other key names a child logger and
              maps to either a level or a nested mapping (with its own
              ``level``, ``diagnostics`` flag and children). Dotted keys such
              as ``"subproblem.kkt"`` are expanded into the nested form.

              The root ``diagnostics`` entry is ``None`` / ``False`` (off), a
              ``DiagnosticsHandler`` (on everywhere) or a mapping
              ``{"handler": DiagnosticsHandler, "enabled": bool}`` whose
              ``enabled`` (default ``True``) sets the root flag only. A child
              entry's ``diagnostics`` is a bool overriding the inherited flag;
              handlers can only be set at the root.
        name
            Root logger name.

        Returns
        -------
        Logger
            Configured root logger.

        Raises
        ------
        TypeError
            If ``spec`` is of an unsupported type, or a child entry uses a
            root-only reserved key.

        Examples
        --------
        >>> from slsqp_jax.sqpdax.logging import DEBUG, INFO, WARNING, Logger
        >>> log = Logger.from_options(
        ...     {"level": "INFO", "subproblem": {"level": "DEBUG", "kkt": "WARNING"}}
        ... )
        >>> log.level == INFO
        True
        >>> log.child("subproblem").level == DEBUG
        True
        >>> log.child("subproblem").child("kkt").level == WARNING
        True
        >>> log.child("step_controller").level == INFO  # inherited
        True
        >>> Logger.from_options({"subproblem.kkt": "DEBUG"}).child("subproblem").child("kkt").level == DEBUG
        True

        Diagnostics for the subproblem solver only:

        >>> from slsqp_jax.sqpdax.logging import MemoryDiagnosticsHandler
        >>> handler = MemoryDiagnosticsHandler()
        >>> log = Logger.from_options(
        ...     {
        ...         "diagnostics": {"handler": handler, "enabled": False},
        ...         "subproblem": {"diagnostics": True},
        ...     }
        ... )
        >>> log.diagnostics_enabled, log.child("subproblem").diagnostics_enabled
        (False, True)
        """
        # ``None`` / ``True`` / ``False`` are matched by identity (PEP 634),
        # so ``0`` and ``1`` fall through to the level branch as intended.
        match spec:
            case None | False:
                return _DISABLED
            case True:
                return cls(name=name, level=INFO)
            case Handler():
                return cls(name=name, level=INFO, handler=spec)
            case DiagnosticsHandler():
                return cls(
                    name=name,
                    level=DISABLED,
                    diagnostics_handler=spec,
                    diagnostics=True,
                )
            case int() | str():
                return cls(name=name, level=parse_level(spec))
            case Mapping():
                return cls._from_mapping(spec, name=name)
            case _:
                raise TypeError(
                    "options['logging'] must be a bool, a level, a Handler, a "
                    f"DiagnosticsHandler or a mapping; got {type(spec).__name__}"
                )

    @classmethod
    def _from_mapping(cls, spec: Mapping, *, name: str) -> Logger:
        """Build a root logger from the mapping form of ``options['logging']``."""
        tree = _normalise_tree(dict(spec), root=True)
        kwargs: dict[str, Any] = {"name": name}
        kwargs["level"] = parse_level(tree.pop("level", INFO))
        if "handler" in tree:
            kwargs["handler"] = tree.pop("handler")
        if "format" in tree:
            kwargs["fmt"] = str(tree.pop("format"))
        if "indent" in tree:
            kwargs["indent"] = str(tree.pop("indent"))
        if "axis_name" in tree:
            kwargs["axis_name"] = tree.pop("axis_name")
        if "ordered" in tree:
            kwargs["ordered"] = bool(tree.pop("ordered"))
        if "diagnostics" in tree:
            handler, enabled = _parse_root_diagnostics(tree.pop("diagnostics"))
            kwargs["diagnostics_handler"] = handler
            kwargs["diagnostics"] = enabled
        kwargs["children"] = FrozenDict(tree)
        return cls(**kwargs)

    def child(self, name: str) -> Self:
        """Return the descendant logger ``self.name + "." + name``.

        The child inherits every setting; its level and diagnostics flag are
        the entries recorded for it in :attr:`children` when present and the
        parent's otherwise. The diagnostics handler is always shared.

        Parameters
        ----------
        name
            Single path component (no dots).

        Returns
        -------
        Logger
            Child logger.
        """
        spec = self.children.get(name)
        level = self.level
        diagnostics = self.diagnostics
        sub: Mapping = {}
        if isinstance(spec, Mapping):
            if "level" in spec:
                level = parse_level(spec["level"])
            if "diagnostics" in spec:
                diagnostics = bool(spec["diagnostics"])
            sub = {k: v for k, v in spec.items() if k not in _CHILD_SETTING_KEYS}
        elif spec is not None:
            level = parse_level(spec)
        return replace(
            self,
            name=f"{self.name}.{name}",
            level=level,
            diagnostics=diagnostics,
            children=FrozenDict(sub),
        )

    # ------------------------------------------------------------- emitting

    @property
    def enabled(self) -> bool:
        """Whether any call at all can emit (``level < DISABLED``)."""
        return self.level < DISABLED

    @property
    def diagnostics_enabled(self) -> bool:
        """Whether :meth:`diagnostic` emits (flag set *and* a handler exists)."""
        return self.diagnostics and self.diagnostics_handler is not None

    def is_enabled_for(self, level: int) -> bool:
        """Whether a call at ``level`` would emit.

        Parameters
        ----------
        level
            Severity of the prospective call.

        Returns
        -------
        bool
            ``True`` when ``level >= self.level`` and the logger is enabled.
        """
        return self.enabled and level >= self.level

    def log(self, level: int, msg: str, *, when: Any = None, **values: Any) -> None:
        """Emit ``msg`` at ``level`` if enabled.

        Parameters
        ----------
        level
            Severity.
        msg
            :meth:`str.format` template. Every placeholder must be a
            keyword naming an entry of ``values``; format specs such as
            ``{f:.3e}`` are applied host-side to the concrete value.
        when
            Optional traced scalar boolean; when it evaluates to ``False``
            at run time the record is dropped. Lets a warning fire
            conditionally without :func:`jax.lax.cond`.
        **values
            Traced arrays (or Equinox ``Enumeration`` items, rendered by
            item name) interpolated into ``msg``. Scalars are converted to
            Python scalars.

        Raises
        ------
        KeyError
            If ``msg`` references a placeholder missing from ``values``.
        TypeError
            If ``when`` is not a scalar.
        """
        if not self.is_enabled_for(level):
            return
        missing = template_fields(msg, what="log message") - set(values)
        if missing:
            raise KeyError(
                f"log message {msg!r} references {sorted(missing)} but only "
                f"{sorted(values)} were supplied"
            )
        spec = _EmitSpec(
            name=self.name,
            level=level,
            msg=msg,
            fmt=self.fmt,
            indent=self.indent,
            handler=_STDOUT if self.handler is None else self.handler,
        )
        jax.debug.callback(
            functools.partial(_emit, spec),
            _as_predicate(when),
            self._run_index(),
            dict(values),
            ordered=self.ordered,
        )

    def debug(self, msg: str, *, when: Any = None, **values: Any) -> None:
        """Shorthand for :meth:`log` at ``DEBUG``."""
        self.log(DEBUG, msg, when=when, **values)

    def info(self, msg: str, *, when: Any = None, **values: Any) -> None:
        """Shorthand for :meth:`log` at ``INFO``."""
        self.log(INFO, msg, when=when, **values)

    def warning(self, msg: str, *, when: Any = None, **values: Any) -> None:
        """Shorthand for :meth:`log` at ``WARNING``."""
        self.log(WARNING, msg, when=when, **values)

    def error(self, msg: str, *, when: Any = None, **values: Any) -> None:
        """Shorthand for :meth:`log` at ``ERROR``."""
        self.log(ERROR, msg, when=when, **values)

    def diagnostic(
        self,
        kind: str,
        fields: Mapping[str, Any] | Callable[[], Mapping[str, Any]],
        *,
        when: Any = None,
    ) -> None:
        """Ship a structured payload to the diagnostics handler if enabled.

        Nothing is traced unless :attr:`diagnostics_enabled`. When it is, the
        payload travels through an ordered :func:`jax.debug.callback` and
        reaches the handler as a
        :class:`~slsqp_jax.sqpdax.logging.record.DiagnosticRecord` whose
        leaves have been converted to numpy (0-d arrays to Python scalars,
        Equinox ``Enumeration`` items to their names).

        With ``when``, the callback is wrapped in :func:`jax.lax.cond` so the
        device-to-host transfer only happens on iterations where the
        predicate is true, and ``fields`` (if callable) is evaluated *inside*
        the true branch: a costly diagnostic computation placed in the thunk
        therefore executes only when the predicate fires. Under
        :func:`jax.vmap` with a batched predicate the ``cond`` lowers to a
        ``select``, both branches run for every batch element and the cost
        is the same as for an unconditional record; records stay correct
        because the host re-checks ``when``.

        Parameters
        ----------
        kind
            Tag identifying the payload schema (``"step"``, ``"qp"``, ...).
        fields
            Mapping from identifier-valued names to traced values — any
            pytree of arrays (modules, tuples, enumeration items), but no
            non-array leaves such as closures — or a zero-argument callable
            returning such a mapping.
        when
            Optional traced scalar boolean gating the emission.

        Raises
        ------
        ValueError
            If a field name is not a valid identifier.
        TypeError
            If ``when`` is not a scalar.

        Examples
        --------
        >>> import jax, jax.numpy as jnp
        >>> from slsqp_jax.sqpdax.logging import Logger, MemoryDiagnosticsHandler
        >>> handler = MemoryDiagnosticsHandler()
        >>> log = Logger.from_options(handler)
        >>> @jax.jit
        ... def f(x):
        ...     log.diagnostic("point", {"x": x, "norm": jnp.linalg.norm(x)})
        ...     log.diagnostic("large", lambda: {"x2": x * 2}, when=x[0] > 1)
        ...     return x
        >>> _ = f(jnp.array([0.5, 0.5]))
        >>> _ = f(jnp.array([2.0, 0.0]))
        >>> handler.close()
        >>> [rec.kind for rec in handler]
        ['point', 'point', 'large']
        >>> handler.records[-1].values["x2"].tolist()
        [4.0, 0.0]
        """
        if not self.diagnostics_enabled:
            return
        when_arr = _as_predicate(when)
        spec = _DiagnosticSpec(
            name=self.name,
            kind=kind,
            handler=cast(DiagnosticsHandler, self.diagnostics_handler),
        )
        run = self._run_index()

        def resolve() -> dict[str, Any]:
            values = dict(fields if isinstance(fields, Mapping) else fields())
            for key in values:
                if not isinstance(key, str) or not key.isidentifier():
                    raise ValueError(
                        f"diagnostic field name {key!r} is not a valid identifier"
                    )
            return values

        if when is None:
            jax.debug.callback(
                functools.partial(_emit_diagnostic, spec),
                when_arr,
                run,
                resolve(),
                ordered=self.ordered,
            )
            return

        def emit_branch(when_arr: Any, run: Any) -> None:
            jax.debug.callback(
                functools.partial(_emit_diagnostic, spec),
                when_arr,
                run,
                resolve(),
                ordered=self.ordered,
            )

        def noop(when_arr: Any, run: Any) -> None:
            return None

        jax.lax.cond(when_arr, emit_branch, noop, when_arr, run)

    def _run_index(self) -> Any:
        """Batch index along :attr:`axis_name`, or ``None`` outside its ``vmap``."""
        if self.axis_name is None:
            return None
        try:
            return jax.lax.axis_index(self.axis_name)
        except NameError:
            return None


_DISABLED: Logger = cast(Logger, Logger())


def _normalise_tree(spec: dict[str, Any], *, root: bool) -> dict[str, Any]:
    """Expand dotted keys and validate reserved keys of a config mapping."""
    out: dict[str, Any] = {}
    for key, value in spec.items():
        if not isinstance(key, str):
            raise TypeError(f"logging option keys must be strings; got {key!r}")
        if "." in key:
            head, tail = key.split(".", 1)
            nested = out.setdefault(head, {})
            if not isinstance(nested, dict):
                nested = {"level": nested}
                out[head] = nested
            nested[tail] = value
            continue
        if key in _RESERVED_ROOT_KEYS and key not in _CHILD_SETTING_KEYS and not root:
            raise TypeError(
                f"logging option {key!r} may only be set on the root logger"
            )
        out[key] = value
    # Recurse into children (everything that is not a reserved key).
    for key, value in list(out.items()):
        if key in _RESERVED_ROOT_KEYS:
            continue
        if isinstance(value, Mapping):
            out[key] = _normalise_tree(dict(value), root=False)
        else:
            out[key] = parse_level(value)
    if "level" in out:
        out["level"] = parse_level(out["level"])
    if "diagnostics" in out and not root:
        flag = out["diagnostics"]
        if not isinstance(flag, bool):
            raise TypeError(
                "a child's 'diagnostics' entry must be a bool (the handler may "
                f"only be set on the root logger); got {type(flag).__name__}"
            )
    return out


def _parse_root_diagnostics(spec: Any) -> tuple[DiagnosticsHandler | None, bool]:
    """Interpret the root ``diagnostics`` option as ``(handler, enabled)``."""
    match spec:
        case None | False:
            return None, False
        case DiagnosticsHandler():
            return spec, True
        case Mapping():
            unknown = set(spec) - {"handler", "enabled"}
            if unknown:
                raise TypeError(
                    "logging option 'diagnostics' accepts only 'handler' and "
                    f"'enabled'; got {sorted(unknown)}"
                )
            handler = spec.get("handler")
            if not isinstance(handler, DiagnosticsHandler):
                raise TypeError(
                    "logging option 'diagnostics' needs a DiagnosticsHandler under "
                    f"'handler'; got {type(handler).__name__}"
                )
            enabled = spec.get("enabled", True)
            if not isinstance(enabled, bool):
                raise TypeError(
                    "logging option 'diagnostics.enabled' must be a bool; got "
                    f"{type(enabled).__name__}"
                )
            return handler, enabled
        case _:
            raise TypeError(
                "logging option 'diagnostics' must be None/False, a "
                "DiagnosticsHandler or {'handler': ..., 'enabled': ...}; got "
                f"{type(spec).__name__}"
            )


def format_fields(
    fields: Mapping[str, tuple[Any, str | None] | Any], *, sep: str = " "
) -> tuple[str, dict[str, Any]]:
    """Build a ``key=value`` message template from labelled values.

    This is the building block for tabular per-step summaries: each entry
    becomes ``"{key}={{{key}:{fmt}}}"`` in the template and the value is
    returned under ``key`` so the pair can be passed straight to
    :meth:`Logger.log`.

    Parameters
    ----------
    fields
        Mapping from placeholder name (a valid identifier) to either a bare
        value or a ``(value, fmt)`` tuple; ``fmt`` is a :meth:`str.format`
        spec such as ``".3e"`` or ``None`` for the default rendering.
    sep
        Separator between entries.

    Returns
    -------
    msg
        Template string.
    values
        Values keyed by placeholder, ready for ``**values``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.logging import format_fields
    >>> msg, values = format_fields({"step": jnp.asarray(3), "f": (jnp.asarray(2.0), ".2e")})
    >>> msg
    'step={step} f={f:.2e}'
    >>> sorted(values)
    ['f', 'step']
    """
    pieces: list[str] = []
    values: dict[str, Any] = {}
    for key, entry in fields.items():
        if not key.isidentifier():
            raise ValueError(f"field name {key!r} is not a valid identifier")
        if isinstance(entry, tuple):
            value, fmt = entry
        else:
            value, fmt = entry, None
        spec = "" if fmt is None else f":{fmt}"
        pieces.append(f"{key}={{{key}{spec}}}")
        values[key] = value
    return sep.join(pieces), values
