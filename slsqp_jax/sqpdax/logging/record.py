"""Host-side log records and their rendering."""

from __future__ import annotations

import string
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .levels import level_name

__all__ = [
    "DEFAULT_FORMAT",
    "RECORD_FIELDS",
    "DiagnosticRecord",
    "LogRecord",
    "template_fields",
    "validate_format",
]

DEFAULT_FORMAT = "{run_prefix}{indent}{name} {levelname}: {message}"
"""Default record format; see :meth:`LogRecord.render` for the fields."""

RECORD_FIELDS = frozenset(
    {"name", "levelname", "levelno", "depth", "indent", "run", "run_prefix", "message"}
)
"""Placeholder names a record format string may reference."""


def template_fields(template: str, *, what: str = "template") -> set[str]:
    """Top-level field names referenced by a :meth:`str.format` template.

    Parameters
    ----------
    template
        A :meth:`str.format` string.
    what
        Noun used in the error message (``"log message"``, ``"record
        format"``, ...).

    Returns
    -------
    set of str
        Field names, with attribute / index accessors stripped (``{a.b}``
        and ``{a[0]}`` both report ``a``).

    Raises
    ------
    ValueError
        If the template uses a positional placeholder (``{}`` or ``{0}``) or
        is not a valid format string.

    Examples
    --------
    >>> from slsqp_jax.sqpdax.logging.record import template_fields
    >>> sorted(template_fields("{name}: {x:.3e} {x.shape}"))
    ['name', 'x']
    """
    names: set[str] = set()
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as err:
        raise ValueError(f"invalid {what} {template!r}: {err}") from None
    for _literal, field_name, _spec, _conv in parsed:
        if field_name is None:
            continue
        if field_name == "" or field_name.isdigit():
            raise ValueError(
                f"{what} {template!r} uses a positional placeholder; "
                "only keyword placeholders are supported"
            )
        names.add(field_name.split(".", 1)[0].split("[", 1)[0])
    return names


def validate_format(fmt: str) -> None:
    """Check that a record format string only references known fields.

    Called when a :class:`~slsqp_jax.sqpdax.logging.logger.Logger` is built,
    so a typo in ``options['logging']['format']`` fails at configuration
    time instead of inside the compiled callback.

    Parameters
    ----------
    fmt
        Candidate :meth:`LogRecord.render` template.

    Raises
    ------
    ValueError
        If ``fmt`` uses a positional placeholder or a field outside
        :data:`RECORD_FIELDS`.

    Examples
    --------
    >>> from slsqp_jax.sqpdax.logging import validate_format
    >>> validate_format("{levelname:<7} {message}")
    >>> validate_format("{levelName} {message}")
    Traceback (most recent call last):
        ...
    ValueError: record format '{levelName} {message}' references unknown field(s) ['levelName']; allowed: ['depth', 'indent', 'levelname', 'levelno', 'message', 'name', 'run', 'run_prefix']
    """
    unknown = template_fields(fmt, what="record format") - RECORD_FIELDS
    if unknown:
        raise ValueError(
            f"record format {fmt!r} references unknown field(s) {sorted(unknown)}; "
            f"allowed: {sorted(RECORD_FIELDS)}"
        )


@dataclass(frozen=True)
class LogRecord:
    """One emitted log message, materialised on the host.

    Records are created inside the :func:`jax.debug.callback` that a
    :class:`~slsqp_jax.sqpdax.logging.logger.Logger` binds, after the traced
    values have been converted to Python scalars / numpy arrays. Handlers
    receive both the record and its rendered line, so a custom handler can
    re-format or filter on the structured fields.

    Attributes
    ----------
    name
        Dotted logger name (``"minimiser.subproblem"``).
    levelno
        Numeric severity.
    message
        The already-interpolated message.
    run
        Index along the configured ``vmap`` axis, or ``None`` when the
        record was not emitted under such a ``vmap``.
    values
        The interpolated values, keyed by placeholder name.
    """

    name: str
    levelno: int
    message: str
    run: int | None = None
    values: Mapping[str, Any] = field(default_factory=dict)

    @property
    def levelname(self) -> str:
        """Canonical name of :attr:`levelno`."""
        return level_name(self.levelno)

    @property
    def depth(self) -> int:
        """Nesting depth of :attr:`name` (number of dots)."""
        return self.name.count(".")

    def render(self, fmt: str = DEFAULT_FORMAT, indent: str = "  ") -> str:
        """Render the record through a :meth:`str.format` template.

        Parameters
        ----------
        fmt
            Template with any of the fields ``name``, ``levelname``,
            ``levelno``, ``depth``, ``indent`` (``indent`` repeated
            ``depth`` times), ``run`` (the run index or ``""``),
            ``run_prefix`` (``"[run {run}] "`` or ``""``) and ``message``
            (:data:`RECORD_FIELDS`). Fields absent from the template are
            simply not rendered; unknown fields raise ``KeyError`` here, so
            callers should pass ``fmt`` through :func:`validate_format`
            first (the logger does this on construction).
        indent
            Indentation unit repeated once per nesting level.

        Returns
        -------
        str
            The rendered line (without a trailing newline).

        Examples
        --------
        >>> from slsqp_jax.sqpdax.logging import INFO, LogRecord
        >>> rec = LogRecord("minimiser.subproblem", INFO, "qp done", run=2)
        >>> rec.render()
        '[run 2]   minimiser.subproblem INFO: qp done'
        >>> rec.render("{levelname:<7}|{name}|{message}")
        'INFO   |minimiser.subproblem|qp done'
        """
        run_prefix = "" if self.run is None else f"[run {self.run}] "
        return fmt.format(
            name=self.name,
            levelname=self.levelname,
            levelno=self.levelno,
            depth=self.depth,
            indent=indent * self.depth,
            run="" if self.run is None else self.run,
            run_prefix=run_prefix,
            message=self.message,
        )


@dataclass(frozen=True)
class DiagnosticRecord:
    """One structured diagnostic payload, materialised on the host.

    Produced by :meth:`~slsqp_jax.sqpdax.logging.logger.Logger.diagnostic`
    and consumed by a
    :class:`~slsqp_jax.sqpdax.logging.handlers.DiagnosticsHandler`. Unlike
    :class:`LogRecord` there is no rendered message: the payload is kept as
    data (numpy arrays, Python scalars, Equinox modules whose leaves have
    been converted to numpy, enumeration items as their names) so it can be
    post-processed into tables, reports or plots.

    Attributes
    ----------
    name
        Dotted logger name of the emitter (``"minimiser.subproblem"``).
    kind
        Tag given at the call site identifying the payload schema
        (``"step"``, ``"qp"``, ``"qp_iter_failure"``, ...).
    run
        Index along the configured ``vmap`` axis, or ``None`` when the
        record was not emitted under such a ``vmap``.
    values
        Host-side payload keyed by the field names given at the call site.

    Examples
    --------
    >>> import numpy as np
    >>> from slsqp_jax.sqpdax.logging import DiagnosticRecord
    >>> rec = DiagnosticRecord("minimiser.subproblem", "qp", None, {"iters": 3})
    >>> rec.depth, rec.values["iters"]
    (1, 3)
    """

    name: str
    kind: str
    run: int | None = None
    values: Mapping[str, Any] = field(default_factory=dict)

    @property
    def depth(self) -> int:
        """Nesting depth of :attr:`name` (number of dots)."""
        return self.name.count(".")
