"""Unit tests for :class:`slsqp_jax.sqpdax.logging.logger.Logger`."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax.sqpdax.logging import (
    DEBUG,
    DISABLED,
    ERROR,
    INFO,
    WARNING,
    Logger,
    MemoryDiagnosticsHandler,
    MemoryHandler,
    format_fields,
)
from slsqp_jax.sqpdax.registry import FrozenDict


class _STATUS(eqx.Enumeration):
    ok = "everything is fine"
    bad = "something went wrong"


def _has_callback(fn, *args) -> bool:
    return "debug_callback" in str(jax.make_jaxpr(fn)(*args))


# --- construction -----------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "level", "has_handler"),
    [
        (None, DISABLED, False),
        (False, DISABLED, False),
        (True, INFO, False),
        ("debug", DEBUG, False),
        (WARNING, WARNING, False),
        (MemoryHandler(), INFO, True),
        ({"level": "error"}, ERROR, False),
        ({}, INFO, False),
    ],
)
def test_from_options_scalar_specs(spec, level, has_handler):
    logger = Logger.from_options(spec)
    assert logger.name == "minimiser"
    assert logger.level == level
    assert (logger.handler is not None) is has_handler
    if level == DISABLED:
        assert logger is Logger.disabled()


@pytest.mark.parametrize(
    "spec",
    [
        {"level": "INFO", "subproblem": {"level": "DEBUG", "kkt": "WARNING"}},
        {"level": "INFO", "subproblem": "DEBUG", "subproblem.kkt": "WARNING"},
        {"level": "INFO", "subproblem.level": "DEBUG", "subproblem.kkt": "WARNING"},
        FrozenDict(
            {"level": "INFO", "subproblem": {"level": "DEBUG", "kkt": "WARNING"}}
        ),
    ],
    ids=["nested", "dotted", "dotted-level", "frozen"],
)
def test_child_levels_from_nested_and_dotted_config(spec):
    root = Logger.from_options(spec)
    sub = root.child("subproblem")
    kkt = sub.child("kkt")
    other = root.child("step_controller")
    assert (root.level, sub.level, kkt.level, other.level) == (
        INFO,
        DEBUG,
        WARNING,
        INFO,
    )
    assert (sub.name, kkt.name) == ("minimiser.subproblem", "minimiser.subproblem.kkt")
    # Unconfigured grandchildren inherit from their nearest configured parent.
    assert kkt.child("deeper").level == WARNING
    assert other.child("deeper").level == INFO
    # Every setting other than name / level / children is shared.
    assert (sub.handler, sub.fmt, sub.indent, sub.axis_name, sub.ordered) == (
        root.handler,
        root.fmt,
        root.indent,
        root.axis_name,
        root.ordered,
    )


@pytest.mark.parametrize(
    ("spec", "error"),
    [
        (1.5, TypeError),
        ({"subproblem": {"handler": MemoryHandler()}}, TypeError),
        ({"level": "loud"}, ValueError),
        ({"subproblem": "loud"}, ValueError),
        ({1: "INFO"}, TypeError),
    ],
)
def test_from_options_rejects_bad_specs(spec, error):
    with pytest.raises(error):
        Logger.from_options(spec)


@pytest.mark.parametrize(
    "build",
    [
        lambda fmt: Logger(fmt=fmt),
        lambda fmt: Logger.from_options({"format": fmt}),
    ],
    ids=["constructor", "from_options"],
)
@pytest.mark.parametrize("fmt", ["{levelName} {message}", "{} {message}", "{oops"])
def test_bad_record_format_fails_at_construction(build, fmt):
    with pytest.raises(ValueError):
        build(fmt)


def test_from_options_reads_root_settings():
    handler = MemoryHandler()
    logger = Logger.from_options(
        {
            "handler": handler,
            "format": "{message}",
            "indent": "\t",
            "axis_name": "batch",
            "ordered": False,
        }
    )
    assert logger.handler is handler
    assert logger.fmt == "{message}"
    assert logger.indent == "\t"
    assert logger.axis_name == "batch"
    assert logger.ordered is False


def test_logger_is_fully_static():
    """No array leaves, so it fits on ``eqx.field(static=True)``."""
    logger = Logger.from_options({"level": "INFO", "handler": MemoryHandler()})
    assert jax.tree.leaves(logger) == []
    assert isinstance(hash(logger), int)
    assert logger != logger.child("x")
    assert Logger.from_options("INFO") == Logger.from_options("info")
    assert hash(Logger.from_options("INFO")) == hash(Logger.from_options("info"))
    assert Logger.from_options("INFO") != Logger.from_options("DEBUG")


def test_static_logger_field_triggers_retrace():
    """Reconfiguring the level changes the treedef and hence recompiles."""

    class Holder(eqx.Module):
        logger: Logger = eqx.field(static=True)
        x: jax.Array

    traces: list[int] = []

    @jax.jit
    def f(holder):
        traces.append(1)
        holder.logger.info("x={x}", x=holder.x)
        return holder.x

    handler = MemoryHandler()
    off = Holder(Logger.disabled(), jnp.asarray(1.0))
    on = Holder(Logger.from_options({"level": "INFO", "handler": handler}), off.x)
    f(off), f(off)
    assert len(traces) == 1 and handler.lines == []
    f(on)
    assert len(traces) == 2 and handler.lines == ["minimiser INFO: x=1.0"]


# --- trace-time gating ------------------------------------------------------


@pytest.mark.parametrize(
    ("logger_level", "call_level", "expected"),
    [
        (DISABLED, ERROR, False),
        (WARNING, INFO, False),
        (WARNING, WARNING, True),
        (INFO, ERROR, True),
        (DEBUG, DEBUG, True),
    ],
)
def test_calls_below_level_add_nothing_to_the_jaxpr(logger_level, call_level, expected):
    logger = Logger(level=logger_level, handler=MemoryHandler())
    assert logger.is_enabled_for(call_level) is expected

    def fn(x):
        logger.log(call_level, "x={x}", x=x)
        return x

    assert _has_callback(fn, jnp.asarray(1.0)) is expected


# --- emission ---------------------------------------------------------------


def test_ordered_emission_inside_jit_while_loop(make_logger):
    logger, handler = make_logger()
    child = logger.child("inner")

    @jax.jit
    def f(x):
        def body(carry):
            i, x = carry
            logger.info("i={i} x={x:.3e}", i=i, x=x)
            child.debug("half={h:.1f}", h=x / 2)
            return i + 1, x * 0.5

        return jax.lax.while_loop(lambda c: c[0] < 3, body, (jnp.int32(0), x))

    f(jnp.asarray(1.0))
    assert handler.lines == [
        "minimiser INFO: i=0 x=1.000e+00",
        "  minimiser.inner DEBUG: half=0.5",
        "minimiser INFO: i=1 x=5.000e-01",
        "  minimiser.inner DEBUG: half=0.2",
        "minimiser INFO: i=2 x=2.500e-01",
        "  minimiser.inner DEBUG: half=0.1",
    ]
    assert [r.levelno for r in handler.records] == [INFO, DEBUG] * 3
    assert handler.records[0].values == {"i": 0, "x": 1.0}


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_when_gates_emission_at_runtime(make_logger, jit):
    logger, handler = make_logger()

    def fn(x):
        logger.warning("negative x={x}", when=x < 0, x=x)
        logger.info("always x={x}", x=x)
        return x

    fn = jax.jit(fn) if jit else fn
    fn(jnp.asarray(1.0))
    fn(jnp.asarray(-1.0))
    assert handler.lines == [
        "minimiser INFO: always x=1.0",
        "minimiser WARNING: negative x=-1.0",
        "minimiser INFO: always x=-1.0",
    ]


@pytest.mark.parametrize(
    ("template", "value", "expected"),
    [
        ("{v}", _STATUS.ok, "ok"),
        ("{v}", _STATUS.where(jnp.asarray(False), _STATUS.ok, _STATUS.bad), "bad"),
        ("{v}", jnp.asarray(True), "True"),
        ("{v:.2e}", jnp.asarray(1234.5), "1.23e+03"),
        ("{v:>4}", jnp.asarray(7, jnp.int32), "   7"),
        ("{v}", jnp.asarray([1.0, 2.0]), "[1. 2.]"),
        ("{v}", 3, "3"),
    ],
    ids=["enum", "traced-enum", "bool", "float-fmt", "int-fmt", "vector", "python"],
)
def test_value_rendering(make_logger, template, value, expected):
    logger, handler = make_logger(format="{message}")
    jax.jit(lambda v: logger.info(template, v=v))(value)
    assert handler.lines == [expected]


def test_custom_format_indent_and_child_depth(make_logger):
    logger, handler = make_logger(
        format="{indent}<{name}|{levelname}> {message}", indent=".."
    )
    logger.child("a").child("b").warning("deep")
    logger.error("top")
    assert handler.lines == [
        "....<minimiser.a.b|WARNING> deep",
        "<minimiser|ERROR> top",
    ]


@pytest.mark.parametrize(
    ("axis_name", "vmap_axis", "expected_runs"),
    [
        ("run", "run", [0, 1, 2]),
        ("run", None, [None] * 3),
        (None, "run", [None] * 3),
        ("run", "other", [None] * 3),
    ],
    ids=["matching", "no-vmap-axis-name", "no-logger-axis-name", "mismatch"],
)
def test_vmap_run_identification(make_logger, axis_name, vmap_axis, expected_runs):
    logger, handler = make_logger(
        **({} if axis_name is None else {"axis_name": axis_name})
    )

    def fn(x):
        logger.info("x={x}", x=x)
        return x

    xs = jnp.arange(3.0)
    jax.jit(jax.vmap(fn, axis_name=vmap_axis))(xs)
    assert [r.run for r in handler.records] == expected_runs
    assert sorted(r.values["x"] for r in handler.records) == [0.0, 1.0, 2.0]
    if axis_name == vmap_axis:
        assert handler.lines[0].startswith("[run 0] ")
    else:
        assert not handler.lines[0].startswith("[run")


@pytest.mark.parametrize(
    ("template", "error"),
    [("x={x} y={y}", KeyError), ("{} positional", ValueError), ("{0}", ValueError)],
)
def test_bad_templates_fail_at_trace_time(make_logger, template, error):
    logger, handler = make_logger()
    with pytest.raises(error):
        jax.jit(lambda x: logger.info(template, x=x))(jnp.asarray(1.0))
    assert handler.lines == []


def test_default_handler_writes_to_stdout(capsys):
    logger = Logger.from_options(True)
    jax.jit(lambda x: logger.info("x={x}", x=x))(jnp.asarray(2.0))
    assert capsys.readouterr().out == "minimiser INFO: x=2.0\n"


# --- format_fields ----------------------------------------------------------


@pytest.mark.parametrize(
    ("fields", "expected_msg", "expected_render"),
    [
        ({"step": jnp.asarray(3)}, "step={step}", "step=3"),
        (
            {"f": (jnp.asarray(2.0), ".2e"), "ok": (jnp.asarray(True), None)},
            "f={f:.2e} ok={ok}",
            "f=2.00e+00 ok=True",
        ),
        (
            {"a": (np.float64(0.5), ".1f"), "b": 1},
            "a={a:.1f} b={b}",
            "a=0.5 b=1",
        ),
    ],
)
def test_format_fields_builds_template(
    make_logger, fields, expected_msg, expected_render
):
    msg, values = format_fields(fields)
    assert msg == expected_msg
    assert set(values) == set(fields)
    logger, handler = make_logger(format="{message}")
    logger.info(msg, **values)
    assert handler.lines == [expected_render]


def test_format_fields_rejects_non_identifier_keys():
    with pytest.raises(ValueError, match="identifier"):
        format_fields({"|c|": 1.0})


# --- diagnostics: configuration ---------------------------------------------


def test_diagnostics_off_by_default():
    logger = Logger.from_options({"level": "DEBUG"})
    assert logger.diagnostics_handler is None
    assert logger.diagnostics is False
    assert logger.diagnostics_enabled is False
    assert Logger.disabled().diagnostics_enabled is False


@pytest.mark.parametrize(
    ("make_spec", "level", "root_enabled"),
    [
        (lambda h: h, DISABLED, True),
        (lambda h: {"diagnostics": h}, INFO, True),
        (lambda h: {"diagnostics": {"handler": h}}, INFO, True),
        (lambda h: {"diagnostics": {"handler": h, "enabled": True}}, INFO, True),
        (lambda h: {"diagnostics": {"handler": h, "enabled": False}}, INFO, False),
        (lambda h: {"level": "DEBUG", "diagnostics": h}, DEBUG, True),
        (lambda h: {"diagnostics": None}, INFO, False),
        (lambda h: {"diagnostics": False}, INFO, False),
    ],
    ids=[
        "bare",
        "key",
        "mapping",
        "mapping-enabled",
        "mapping-disabled",
        "with-level",
        "none",
        "false",
    ],
)
def test_from_options_diagnostics_specs(make_spec, level, root_enabled):
    handler = MemoryDiagnosticsHandler()
    spec = make_spec(handler)
    logger = Logger.from_options(spec)
    has_handler = spec not in ({"diagnostics": None}, {"diagnostics": False})
    assert logger.level == level
    assert (logger.diagnostics_handler is handler) is has_handler
    assert logger.diagnostics_enabled is root_enabled
    # The text channel is unaffected by the diagnostics channel.
    assert logger.handler is None


@pytest.mark.parametrize(
    "make_spec",
    [
        lambda h: {"diagnostics": True},
        lambda h: {"diagnostics": 1.5},
        lambda h: {"diagnostics": {"enabled": True}},
        lambda h: {"diagnostics": {"handler": h, "enabled": 1}},
        lambda h: {"diagnostics": {"handler": h, "level": "INFO"}},
        lambda h: {"diagnostics": {"handler": MemoryHandler()}},
        lambda h: {"subproblem": {"diagnostics": h}},
        lambda h: {"subproblem": {"diagnostics": {"handler": h}}},
        lambda h: {"subproblem": {"diagnostics": 1}},
    ],
    ids=[
        "bare-true",
        "float",
        "no-handler",
        "non-bool-enabled",
        "unknown-key",
        "text-handler",
        "child-handler",
        "child-mapping",
        "child-non-bool",
    ],
)
def test_from_options_rejects_bad_diagnostics_specs(make_spec):
    with pytest.raises(TypeError):
        Logger.from_options(make_spec(MemoryDiagnosticsHandler()))


@pytest.mark.parametrize("diagnostics", [1, "yes", None], ids=["int", "str", "none"])
def test_constructor_rejects_non_bool_diagnostics_flag(diagnostics):
    with pytest.raises(TypeError):
        Logger(diagnostics=diagnostics)  # type: ignore[arg-type]


def test_constructor_rejects_text_handler_as_diagnostics_handler():
    with pytest.raises(TypeError):
        Logger(diagnostics_handler=MemoryHandler())  # type: ignore[arg-type]


@pytest.mark.parametrize("root_enabled", [True, False], ids=["root-on", "root-off"])
@pytest.mark.parametrize(
    "child_override", [None, True, False], ids=["inherit", "child-on", "child-off"]
)
def test_diagnostics_flag_inheritance(make_diag_logger, root_enabled, child_override):
    child_spec = (
        {"level": "DEBUG"}
        if child_override is None
        else {
            "level": "DEBUG",
            "diagnostics": child_override,
        }
    )
    logger, handler = make_diag_logger(
        enabled=root_enabled, subproblem=child_spec, **{"subproblem.kkt": "INFO"}
    )
    sub = logger.child("subproblem")
    kkt = sub.child("kkt")
    other = logger.child("step_controller")
    expected_sub = root_enabled if child_override is None else child_override
    assert logger.diagnostics_enabled is root_enabled
    assert sub.diagnostics_enabled is expected_sub
    # Grandchildren inherit from the nearest configured ancestor.
    assert kkt.diagnostics_enabled is expected_sub
    assert other.diagnostics_enabled is root_enabled
    # The handler is shared and the text settings are untouched.
    assert all(
        node.diagnostics_handler is handler for node in (logger, sub, kkt, other)
    )
    assert (sub.level, kkt.level, other.level) == (DEBUG, INFO, DISABLED)


def test_child_diagnostics_flag_without_root_handler_is_inert():
    logger = Logger.from_options({"subproblem": {"diagnostics": True}})
    sub = logger.child("subproblem")
    assert sub.diagnostics is True
    assert sub.diagnostics_enabled is False
    assert not _has_callback(lambda x: sub.diagnostic("k", {"x": x}), jnp.asarray(1.0))


def test_diagnostics_keep_logger_static_and_hashable():
    handler = MemoryDiagnosticsHandler()
    a = Logger.from_options({"diagnostics": handler})
    b = Logger.from_options({"diagnostics": handler})
    assert jax.tree.leaves(a) == []
    assert a == b and hash(a) == hash(b)
    assert a != Logger.from_options(
        {"diagnostics": {"handler": handler, "enabled": False}}
    )
    assert a != Logger.from_options({"diagnostics": MemoryDiagnosticsHandler()})


# --- diagnostics: emission ----------------------------------------------------


@pytest.mark.parametrize("when", [None, True], ids=["unconditional", "conditional"])
def test_disabled_diagnostic_adds_nothing_to_the_jaxpr(when):
    for logger in (
        Logger.disabled(),
        Logger.from_options({"level": "DEBUG", "handler": MemoryHandler()}),
        Logger.from_options(
            {"diagnostics": {"handler": MemoryDiagnosticsHandler(), "enabled": False}}
        ),
    ):
        assert not _has_callback(
            lambda x: logger.diagnostic(  # noqa: B023
                "k", lambda: {"x": x}, when=None if when is None else x > 0
            ),
            jnp.asarray(1.0),
        )


def test_diagnostic_emits_ordered_records_inside_jit_while_loop(make_diag_logger):
    logger, handler = make_diag_logger()
    child = logger.child("inner")

    @jax.jit
    def f(x):
        def body(carry):
            i, x = carry
            logger.diagnostic("outer", {"i": i, "x": x})
            child.diagnostic("inner", {"half": x / 2})
            return i + 1, x * 0.5

        return jax.lax.while_loop(lambda c: c[0] < 3, body, (jnp.int32(0), x))

    f(jnp.asarray(1.0))
    handler.close()
    assert [(r.name, r.kind) for r in handler] == [
        ("minimiser", "outer"),
        ("minimiser.inner", "inner"),
    ] * 3
    assert [r.values for r in handler.select(kind="outer")] == [
        {"i": 0, "x": 1.0},
        {"i": 1, "x": 0.5},
        {"i": 2, "x": 0.25},
    ]
    assert all(r.run is None for r in handler)


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("thunk", [False, True], ids=["mapping", "callable"])
def test_diagnostic_when_gates_emission(make_diag_logger, jit, thunk):
    logger, handler = make_diag_logger()

    def fn(x):
        fields = {"x": x}
        logger.diagnostic("negative", (lambda: fields) if thunk else fields, when=x < 0)
        logger.diagnostic("always", {"x": x})
        return x

    fn = jax.jit(fn) if jit else fn
    fn(jnp.asarray(1.0))
    fn(jnp.asarray(-1.0))
    handler.close()
    assert [(r.kind, r.values["x"]) for r in handler] == [
        ("always", 1.0),
        ("negative", -1.0),
        ("always", -1.0),
    ]


def test_conditional_diagnostic_is_a_device_side_cond(make_diag_logger):
    """``when`` wraps the callback in ``lax.cond`` and the thunk runs in the branch."""
    logger, handler = make_diag_logger()
    executed: list[int] = []

    def fn(x):
        def costly():
            jax.debug.callback(lambda: executed.append(1), ordered=True)
            return {"x2": x * 2}

        logger.diagnostic("k", costly, when=x > 0)
        return x

    jaxpr = str(jax.make_jaxpr(fn)(jnp.asarray(1.0)))
    assert "cond" in jaxpr and "debug_callback" in jaxpr
    f = jax.jit(fn)
    f(jnp.asarray(-1.0))
    f(jnp.asarray(-2.0))
    f(jnp.asarray(3.0))
    handler.close()
    # The costly thunk only executed on the call where ``when`` was true.
    assert executed == [1]
    assert [r.values["x2"] for r in handler] == [6.0]


def test_unconditional_diagnostic_has_no_cond(make_diag_logger):
    logger, _ = make_diag_logger()
    jaxpr = str(jax.make_jaxpr(lambda x: logger.diagnostic("k", {"x": x}))(1.0))
    assert "debug_callback" in jaxpr and "cond" not in jaxpr


class _Payload(eqx.Module):
    vec: jax.Array
    status: _STATUS
    tag: str = eqx.field(static=True, default="t")


def test_diagnostic_values_arrive_as_host_pytrees(make_diag_logger):
    logger, handler = make_diag_logger()

    @jax.jit
    def f(x):
        payload = _Payload(
            vec=x, status=_STATUS.where(x[0] > 0, _STATUS.ok, _STATUS.bad)
        )
        logger.diagnostic(
            "k",
            {
                "scalar": x[0],
                "vec": x,
                "flag": x[0] > 0,
                "enum": _STATUS.bad,
                "module": payload,
                "tuple": (x, x[1]),
                "nothing": None,
            },
        )
        return x

    f(jnp.asarray([1.0, 2.0]))
    handler.close()
    values = handler.records[0].values
    assert values["scalar"] == 1.0 and isinstance(values["scalar"], float)
    assert isinstance(values["vec"], np.ndarray) and values["vec"].tolist() == [
        1.0,
        2.0,
    ]
    assert values["flag"] is True
    assert values["enum"] == "bad"
    assert isinstance(values["module"], _Payload)
    assert values["module"].vec.tolist() == [1.0, 2.0]
    assert values["module"].status == "ok" and values["module"].tag == "t"
    assert values["tuple"][0].tolist() == [1.0, 2.0] and values["tuple"][1] == 2.0
    assert values["nothing"] is None


@pytest.mark.parametrize("when", [None, True], ids=["unconditional", "conditional"])
def test_diagnostic_rejects_non_identifier_field_names(make_diag_logger, when):
    logger, handler = make_diag_logger()
    with pytest.raises(ValueError, match="identifier"):
        jax.jit(
            lambda x: logger.diagnostic(
                "k", {"bad key": x}, when=None if when is None else x > 0
            )
        )(jnp.asarray(1.0))
    handler.close()
    assert len(handler) == 0


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda logger, x, when: logger.info("x={x}", x=x, when=when), id="log"
        ),
        pytest.param(
            lambda logger, x, when: logger.diagnostic("k", {"x": x}, when=when),
            id="diagnostic",
        ),
    ],
)
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_non_scalar_when_is_rejected_at_trace_time(make_diag_logger, call, jit):
    """A vector ``when`` is a caller bug (missing ``jnp.any``/``jnp.all``);
    both channels reject it before anything reaches the host callback."""
    logger, handler = make_diag_logger(level="INFO")

    def fn(x):
        call(logger, x, x > 0)
        return x

    with pytest.raises(TypeError, match="scalar"):
        (jax.jit(fn) if jit else fn)(jnp.array([1.0, -1.0]))
    # A scalar predicate is still accepted.
    (jax.jit(fn) if jit else fn)(jnp.asarray(1.0))
    jax.effects_barrier()
    handler.close()


@pytest.mark.parametrize(
    ("axis_name", "vmap_axis", "expected_runs"),
    [("run", "run", [0, 1, 2]), ("run", None, [None] * 3), (None, "run", [None] * 3)],
    ids=["matching", "no-vmap-axis-name", "no-logger-axis-name"],
)
def test_diagnostic_vmap_run_identification(
    make_diag_logger, axis_name, vmap_axis, expected_runs
):
    logger, handler = make_diag_logger(
        **({} if axis_name is None else {"axis_name": axis_name})
    )

    def fn(x):
        logger.diagnostic("k", {"x": x})
        logger.diagnostic("odd", {"x": x}, when=(x % 2) == 1)
        return x

    jax.jit(jax.vmap(fn, axis_name=vmap_axis))(jnp.arange(3.0))
    handler.close()
    plain = handler.select(kind="k")
    assert [r.run for r in plain] == expected_runs
    assert sorted(r.values["x"] for r in plain) == [0.0, 1.0, 2.0]
    # Under vmap the cond runs both branches; the host re-check keeps records
    # correct.
    assert [r.values["x"] for r in handler.select(kind="odd")] == [1.0]


def test_diagnostic_after_close_raises_from_the_callback(make_diag_logger):
    logger, handler = make_diag_logger()
    f = jax.jit(lambda x: logger.diagnostic("k", {"x": x}))
    f(jnp.asarray(1.0))
    handler.close()
    with pytest.raises(Exception, match="closed"):
        f(jnp.asarray(2.0))
        jax.effects_barrier()
    assert len(handler) == 1


def test_diagnostics_and_text_channels_are_independent(make_diag_logger, capsys):
    logger, handler = make_diag_logger(level="INFO")
    text = MemoryHandler()
    both = Logger.from_options(
        {"level": "INFO", "handler": text, "diagnostics": handler}
    )
    jax.jit(lambda x: (logger.info("x={x}", x=x), logger.diagnostic("k", {"x": x})))(
        jnp.asarray(1.0)
    )
    jax.jit(lambda x: (both.info("x={x}", x=x), both.diagnostic("k", {"x": x})))(
        jnp.asarray(2.0)
    )
    handler.close()
    assert capsys.readouterr().out == "minimiser INFO: x=1.0\n"
    assert text.lines == ["minimiser INFO: x=2.0"]
    assert [r.values["x"] for r in handler] == [1.0, 2.0]
