"""JIT-safe hierarchical logging for sqpdax components.

The package mirrors the shape of the standard-library :mod:`logging`
module, reduced to what a compiled optimiser needs:

* :class:`~slsqp_jax.sqpdax.logging.logger.Logger` — a fully static Equinox
  module with a dotted name, a level, a handler and a record format; calls
  below the level vanish at trace time, enabled calls compile into ordered
  :func:`jax.debug.callback` side effects.
* :mod:`~slsqp_jax.sqpdax.logging.handlers` — host-side sinks
  (stdout / any stream, a callable, or an in-memory buffer).
* :class:`~slsqp_jax.sqpdax.logging.record.LogRecord` — the structured
  record handlers receive.
* a separate, binary **diagnostics channel**:
  :meth:`~slsqp_jax.sqpdax.logging.logger.Logger.diagnostic` ships rich
  array payloads (:class:`~slsqp_jax.sqpdax.logging.record.DiagnosticRecord`)
  to a closable
  :class:`~slsqp_jax.sqpdax.logging.handlers.DiagnosticsHandler`,
  independently of the text level.

Configuration is driven by ``options["logging"]`` on the minimiser: a
nested mapping of levels keyed by component name, see
:meth:`~slsqp_jax.sqpdax.logging.logger.Logger.from_options`.
"""

from . import handlers, levels, logger, record
from .handlers import (
    CallableHandler,
    DiagnosticsHandler,
    Handler,
    MemoryDiagnosticsHandler,
    MemoryHandler,
    StreamHandler,
)
from .levels import DEBUG, DISABLED, ERROR, INFO, WARNING, level_name, parse_level
from .logger import Logger, format_fields
from .record import (
    DEFAULT_FORMAT,
    RECORD_FIELDS,
    DiagnosticRecord,
    LogRecord,
    validate_format,
)

__all__ = [
    "handlers",
    "levels",
    "logger",
    "record",
    "Logger",
    "LogRecord",
    "DiagnosticRecord",
    "Handler",
    "StreamHandler",
    "CallableHandler",
    "MemoryHandler",
    "DiagnosticsHandler",
    "MemoryDiagnosticsHandler",
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "DISABLED",
    "DEFAULT_FORMAT",
    "RECORD_FIELDS",
    "validate_format",
    "parse_level",
    "level_name",
    "format_fields",
]
