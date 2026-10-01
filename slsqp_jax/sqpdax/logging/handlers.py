"""Sinks that receive log records on the host.

Handlers are deliberately *plain* Python objects rather than Equinox
modules: they live in ``static`` fields, so they only need to be hashable,
and identity hashing is exactly right for a stateful sink such as
:class:`MemoryHandler` or an open stream.

Two families exist: :class:`Handler` receives rendered text records
(:class:`~slsqp_jax.sqpdax.logging.record.LogRecord`) and
:class:`DiagnosticsHandler` receives structured payloads
(:class:`~slsqp_jax.sqpdax.logging.record.DiagnosticRecord`). They are kept
as unrelated classes so a logger configuration can tell them apart.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import IO, Self

import jax

from .record import DiagnosticRecord, LogRecord

__all__ = [
    "Handler",
    "StreamHandler",
    "CallableHandler",
    "MemoryHandler",
    "DiagnosticsHandler",
    "MemoryDiagnosticsHandler",
]


class Handler(ABC):
    """Abstract destination for log records.

    Subclasses implement :meth:`emit`; equality and hashing are by identity
    so a handler can be stored on an ``eqx.field(static=True)``.
    """

    @abstractmethod
    def emit(self, record: LogRecord, line: str) -> None:
        """Consume one record.

        Parameters
        ----------
        record
            Structured record.
        line
            The record rendered through the logger's format string.
        """
        ...


class StreamHandler(Handler):
    """Write each rendered line to a text stream.

    Parameters
    ----------
    stream
        Any object with ``write`` (and optionally ``flush``). ``None``
        (default) resolves :data:`sys.stdout` *at emit time*, so output
        redirection set up after the logger was built (for example pytest's
        ``capsys``) is honoured.

    Examples
    --------
    >>> import io
    >>> from slsqp_jax.sqpdax.logging import INFO, LogRecord, StreamHandler
    >>> buf = io.StringIO()
    >>> StreamHandler(buf).emit(LogRecord("minimiser", INFO, "hi"), "line")
    >>> buf.getvalue()
    'line\\n'
    """

    def __init__(self, stream: IO[str] | None = None):
        self.stream = stream

    def emit(self, record: LogRecord, line: str) -> None:
        stream = sys.stdout if self.stream is None else self.stream
        stream.write(line + "\n")
        flush = getattr(stream, "flush", None)
        if flush is not None:
            flush()


class CallableHandler(Handler):
    """Forward each record to a user callable ``fn(record, line)``.

    Parameters
    ----------
    fn
        Called once per record with the structured record and the rendered
        line.
    """

    def __init__(self, fn: Callable[[LogRecord, str], None]):
        self.fn = fn

    def emit(self, record: LogRecord, line: str) -> None:
        self.fn(record, line)


class MemoryHandler(Handler):
    """Accumulate records and rendered lines in memory.

    Useful in tests and for post-run inspection when the emission order
    (for example under ``vmap``) matters less than the content.

    Attributes
    ----------
    records
        Records in emission order.
    lines
        Rendered lines in emission order.

    Examples
    --------
    >>> from slsqp_jax.sqpdax.logging import INFO, LogRecord, MemoryHandler
    >>> handler = MemoryHandler()
    >>> handler.emit(LogRecord("minimiser", INFO, "hi"), "minimiser INFO: hi")
    >>> handler.lines
    ['minimiser INFO: hi']
    >>> handler.clear(); handler.records
    []
    """

    def __init__(self) -> None:
        self.records: list[LogRecord] = []
        self.lines: list[str] = []

    def emit(self, record: LogRecord, line: str) -> None:
        self.records.append(record)
        self.lines.append(line)

    def clear(self) -> None:
        """Drop everything collected so far."""
        self.records.clear()
        self.lines.clear()


class DiagnosticsHandler(ABC):
    """Abstract, closable destination for structured diagnostic records.

    Records arrive from :func:`jax.debug.callback`, which may run
    asynchronously with respect to the Python thread that launched the
    computation. :meth:`close` therefore waits for every outstanding
    callback with :func:`jax.effects_barrier` before marking the handler
    closed; once closed, the collected data is complete and any further
    :meth:`emit` raises ``RuntimeError``. A handler instance is one-shot: it
    cannot be reopened. Use it as a context manager to close it on exit.

    Subclasses implement :meth:`handle`; equality and hashing are by identity
    so a handler can be stored on an ``eqx.field(static=True)``.
    """

    def __init__(self) -> None:
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has been called."""
        return self._closed

    def emit(self, record: DiagnosticRecord) -> None:
        """Consume one record, or raise if the handler is closed.

        Parameters
        ----------
        record
            Structured payload.

        Raises
        ------
        RuntimeError
            If :meth:`close` was already called.
        """
        if self._closed:
            raise RuntimeError(
                f"{type(self).__name__} is closed; create a new handler for a new run"
            )
        self.handle(record)

    @abstractmethod
    def handle(self, record: DiagnosticRecord) -> None:
        """Store or forward one record (the handler is known to be open)."""
        ...

    def close(self) -> None:
        """Wait for pending callbacks and seal the handler. Idempotent."""
        if self._closed:
            return
        jax.effects_barrier()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class MemoryDiagnosticsHandler(DiagnosticsHandler):
    """Accumulate diagnostic records in memory, in emission order.

    Attributes
    ----------
    records
        Records in emission order.

    Examples
    --------
    >>> from slsqp_jax.sqpdax.logging import DiagnosticRecord, MemoryDiagnosticsHandler
    >>> with MemoryDiagnosticsHandler() as handler:
    ...     handler.emit(DiagnosticRecord("minimiser", "step", None, {"f": 1.0}))
    ...     handler.emit(DiagnosticRecord("minimiser.subproblem", "qp", None, {}))
    >>> len(handler), handler.closed
    (2, True)
    >>> [rec.kind for rec in handler.select(name="minimiser")]
    ['step']
    >>> handler.emit(DiagnosticRecord("minimiser", "step", None, {}))
    Traceback (most recent call last):
        ...
    RuntimeError: MemoryDiagnosticsHandler is closed; create a new handler for a new run
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[DiagnosticRecord] = []

    def handle(self, record: DiagnosticRecord) -> None:
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[DiagnosticRecord]:
        return iter(self.records)

    def select(
        self, *, name: str | None = None, kind: str | None = None
    ) -> list[DiagnosticRecord]:
        """Records matching the given emitter name and/or kind.

        Parameters
        ----------
        name
            Exact dotted logger name to match, or ``None`` for any.
        kind
            Record kind to match, or ``None`` for any.

        Returns
        -------
        list[DiagnosticRecord]
            Matching records in emission order.
        """
        return [
            rec
            for rec in self.records
            if (name is None or rec.name == name) and (kind is None or rec.kind == kind)
        ]
