"""Unit tests for levels, :class:`LogRecord` rendering and the handlers."""

from __future__ import annotations

import io

import jax
import pytest

from slsqp_jax.sqpdax.logging import (
    DEBUG,
    DEFAULT_FORMAT,
    DISABLED,
    ERROR,
    INFO,
    RECORD_FIELDS,
    WARNING,
    CallableHandler,
    DiagnosticRecord,
    LogRecord,
    MemoryDiagnosticsHandler,
    MemoryHandler,
    StreamHandler,
    level_name,
    parse_level,
    validate_format,
)
from slsqp_jax.sqpdax.logging.record import template_fields


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("debug", DEBUG),
        ("INFO", INFO),
        (" warning ", WARNING),
        ("warn", WARNING),
        ("Error", ERROR),
        ("off", DISABLED),
        ("disabled", DISABLED),
        (25, 25),
    ],
)
def test_parse_level_accepts_names_and_ints(spec, expected):
    assert parse_level(spec) == expected


@pytest.mark.parametrize(
    ("spec", "error"),
    [("verbose", ValueError), (1.5, TypeError), (True, TypeError)],
)
def test_parse_level_rejects_bad_input(spec, error):
    with pytest.raises(error):
        parse_level(spec)


@pytest.mark.parametrize(
    ("level", "expected"),
    [(DEBUG, "DEBUG"), (INFO, "INFO"), (WARNING, "WARNING"), (35, "Level 35")],
)
def test_level_name(level, expected):
    assert level_name(level) == expected


@pytest.mark.parametrize(
    ("name", "run", "fmt", "indent", "expected"),
    [
        ("minimiser", None, None, "  ", "minimiser INFO: hello"),
        ("minimiser.a.b", None, None, "..", "....minimiser.a.b INFO: hello"),
        ("minimiser.a", 3, None, "  ", "[run 3]   minimiser.a INFO: hello"),
        ("minimiser", 0, "{run}|{depth}|{levelno}|{message}", "", "0|0|20|hello"),
        ("x.y", None, "{run}|{depth}|{levelname:<8}|", "", "|1|INFO    |"),
    ],
)
def test_record_render_fields(name, run, fmt, indent, expected):
    record = LogRecord(name, INFO, "hello", run=run)
    rendered = record.render(**({} if fmt is None else {"fmt": fmt}), indent=indent)
    assert rendered == expected
    assert record.levelname == "INFO"
    assert record.depth == name.count(".")


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        ("plain text", set()),
        ("{name}", {"name"}),
        ("{a:.3e} {b!r} {a.shape} {c[0]}", {"a", "b", "c"}),
        ("{{escaped}} {x}", {"x"}),
    ],
)
def test_template_fields(template, expected):
    assert template_fields(template) == expected


@pytest.mark.parametrize(
    ("template", "match"),
    [
        ("{} pos", "positional"),
        ("{0} pos", "positional"),
        ("{unclosed", "invalid record format"),
        ("stray }", "invalid record format"),
    ],
)
def test_template_fields_rejects_malformed_templates(template, match):
    with pytest.raises(ValueError, match=match):
        template_fields(template, what="record format")


@pytest.mark.parametrize(
    "fmt",
    [
        DEFAULT_FORMAT,
        "{message}",
        "{levelname:<8}|{name}|{run}|{depth}|{levelno}|{indent}|{run_prefix}",
        "literal only",
    ],
)
def test_validate_format_accepts_known_fields(fmt):
    validate_format(fmt)
    # Anything that validates must also render without error.
    LogRecord("minimiser.a", INFO, "m", run=1).render(fmt)


@pytest.mark.parametrize(
    ("fmt", "unknown"),
    [
        ("{levelName} {message}", ["levelName"]),
        ("{name} {bogus} {other:>3}", ["bogus", "other"]),
        ("{message.upper}", []),  # attribute access on a known field is fine
    ],
)
def test_validate_format_reports_unknown_fields(fmt, unknown):
    if not unknown:
        validate_format(fmt)
        return
    with pytest.raises(ValueError) as excinfo:
        validate_format(fmt)
    message = str(excinfo.value)
    assert str(unknown) in message
    assert str(sorted(RECORD_FIELDS)) in message


def _emit_through(handler, message: str = "hello") -> LogRecord:
    record = LogRecord("minimiser", WARNING, message)
    handler.emit(record, f"rendered {message}")
    return record


def test_stream_handler_defaults_to_current_stdout(capsys):
    """``stream=None`` resolves ``sys.stdout`` lazily, so ``capsys`` sees it."""
    _emit_through(StreamHandler())
    assert capsys.readouterr().out == "rendered hello\n"


def test_stream_handler_writes_to_given_stream():
    buf = io.StringIO()
    _emit_through(StreamHandler(buf), "a")
    _emit_through(StreamHandler(buf), "b")
    assert buf.getvalue() == "rendered a\nrendered b\n"


def test_callable_handler_receives_record_and_line():
    seen: list[tuple[LogRecord, str]] = []
    handler = CallableHandler(lambda record, line: seen.append((record, line)))
    record = _emit_through(handler)
    assert seen == [(record, "rendered hello")]


def test_memory_handler_collects_and_clears():
    handler = MemoryHandler()
    records = [_emit_through(handler, m) for m in ("a", "b")]
    assert handler.records == records
    assert handler.lines == ["rendered a", "rendered b"]
    handler.clear()
    assert handler.records == [] and handler.lines == []


@pytest.mark.parametrize(
    "make", [MemoryHandler, MemoryDiagnosticsHandler], ids=["text", "diagnostics"]
)
def test_handlers_hash_by_identity(make):
    """Identity hashing is what lets a handler live on a static field."""
    a, b = make(), make()
    assert hash(a) != hash(b) or a is not b
    assert a != b
    assert {a, b} == {a, b}


# --- diagnostics ------------------------------------------------------------


def _diag(name: str = "minimiser", kind: str = "step", **values) -> DiagnosticRecord:
    return DiagnosticRecord(name, kind, None, values)


def test_diagnostic_record_basics():
    rec = DiagnosticRecord("minimiser.subproblem", "qp", 2, {"iters": 3})
    assert (rec.name, rec.kind, rec.run, rec.depth) == (
        "minimiser.subproblem",
        "qp",
        2,
        1,
    )
    assert rec.values == {"iters": 3}
    assert _diag().run is None and _diag().values == {}


def test_memory_diagnostics_handler_collects_selects_and_iterates():
    handler = MemoryDiagnosticsHandler()
    records = [
        _diag("minimiser", "step", i=0),
        _diag("minimiser.subproblem", "qp", i=1),
        _diag("minimiser", "step", i=2),
    ]
    for rec in records:
        handler.emit(rec)
    assert handler.records == records
    assert list(handler) == records and len(handler) == 3
    assert handler.select(name="minimiser") == [records[0], records[2]]
    assert handler.select(kind="qp") == [records[1]]
    assert handler.select(name="minimiser", kind="qp") == []
    assert handler.select() == records


@pytest.mark.parametrize("via_context_manager", [False, True], ids=["close", "with"])
def test_diagnostics_handler_close_is_a_barrier_and_seals(
    monkeypatch, via_context_manager
):
    barriers: list[int] = []
    monkeypatch.setattr(jax, "effects_barrier", lambda: barriers.append(1))
    if via_context_manager:
        with MemoryDiagnosticsHandler() as handler:
            handler.emit(_diag())
            assert handler.closed is False and barriers == []
    else:
        handler = MemoryDiagnosticsHandler()
        handler.emit(_diag())
        handler.close()
    assert handler.closed is True and barriers == [1]
    assert len(handler) == 1
    with pytest.raises(RuntimeError, match="closed"):
        handler.emit(_diag())
    assert len(handler) == 1
    # Closing again is a no-op (no second barrier).
    handler.close()
    assert barriers == [1]
