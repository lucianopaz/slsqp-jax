"""Fixtures shared by :mod:`slsqp_jax.sqpdax.logging` tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from slsqp_jax.sqpdax.logging import Logger, MemoryDiagnosticsHandler, MemoryHandler

MakeLogger = Callable[..., tuple[Logger, MemoryHandler]]
MakeDiagLogger = Callable[..., tuple[Logger, MemoryDiagnosticsHandler]]


@pytest.fixture
def make_logger() -> MakeLogger:
    """Factory building a root logger backed by a fresh :class:`MemoryHandler`.

    Keyword arguments are forwarded as the ``options['logging']`` mapping
    (``level`` defaults to ``"DEBUG"``); the handler is injected.
    """

    def factory(**spec: Any) -> tuple[Logger, MemoryHandler]:
        handler = MemoryHandler()
        spec.setdefault("level", "DEBUG")
        spec["handler"] = handler
        return Logger.from_options(spec), handler

    return factory


@pytest.fixture
def make_diag_logger() -> MakeDiagLogger:
    """Factory building a root logger with diagnostics on a fresh handler.

    Keyword arguments are forwarded as the ``options['logging']`` mapping;
    ``level`` defaults to ``"OFF"`` and a :class:`MemoryDiagnosticsHandler`
    is injected under ``diagnostics`` (as ``{"handler": h, "enabled": ...}``
    when ``enabled`` is given, bare otherwise).
    """

    def factory(
        *, enabled: bool | None = None, **spec: Any
    ) -> tuple[Logger, MemoryDiagnosticsHandler]:
        handler = MemoryDiagnosticsHandler()
        spec.setdefault("level", "OFF")
        spec["diagnostics"] = (
            handler if enabled is None else {"handler": handler, "enabled": enabled}
        )
        return Logger.from_options(spec), handler

    return factory
