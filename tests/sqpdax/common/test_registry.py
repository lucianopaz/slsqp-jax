"""Unit tests for :mod:`slsqp_jax.sqpdax.common.registry`."""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import pytest

from slsqp_jax.sqpdax.common.registry import (
    FrozenDict,
    KindRegistryMixin,
    freeze,
    kind_family_field_names,
    static_field_names,
)


class Widget(KindRegistryMixin):
    """Isolated family root for registry tests."""

    _registry: ClassVar[dict[str, type]] = {}


class Gadget(KindRegistryMixin):
    """Second family root to verify registries stay disjoint."""

    _registry: ClassVar[dict[str, type]] = {}


class Alpha(Widget):
    kind: ClassVar[str] = "alpha"

    def __init__(self, scale: float = 1.0, n: int = 0):
        self.scale = scale
        self.n = n


class Beta(Widget):
    kind: ClassVar[str] = "beta"

    def __init__(self, scale: float = 2.0, n: int = 0):
        self.scale = scale
        self.n = n


class AbstractWidget(Widget):
    """Intermediate base without ``kind`` -- must not register."""


class Gamma(AbstractWidget):
    kind: ClassVar[str] = "gamma"

    def __init__(self, scale: float = 3.0):
        self.scale = scale


class Bolt(Gadget):
    kind: ClassVar[str] = "bolt"

    def __init__(self, torque: float = 1.0):
        self.torque = torque


@pytest.mark.parametrize(
    ("kind", "cls"),
    [
        ("alpha", Alpha),
        ("beta", Beta),
        ("gamma", Gamma),
    ],
)
def test_kind_registry_registers_concrete_subclasses(kind: str, cls: type):
    """Subclasses that declare ``kind`` appear in the family ``_registry``."""
    assert Widget._registry[kind] is cls
    assert "bolt" not in Widget._registry
    assert AbstractWidget.__dict__.get("kind") is None
    assert "abstract" not in Widget._registry


def test_kind_registries_are_disjoint_across_families():
    """Each family root keeps its own ``_registry`` mapping."""
    assert sorted(Widget._registry) == ["alpha", "beta", "gamma"]
    assert sorted(Gadget._registry) == ["bolt"]
    assert Gadget._registry["bolt"] is Bolt


@pytest.mark.parametrize(
    ("spec", "injected", "expected_cls", "expected_attrs"),
    [
        ({"kind": "alpha"}, {}, Alpha, {"scale": 1.0, "n": 0}),
        ({"kind": "alpha", "scale": 4.5}, {}, Alpha, {"scale": 4.5, "n": 0}),
        ({"kind": "beta", "scale": 0.5}, {"n": 7}, Beta, {"scale": 0.5, "n": 7}),
        ({"kind": "gamma"}, {}, Gamma, {"scale": 3.0}),
        ({"kind": "bolt", "torque": 9.0}, {}, Bolt, {"torque": 9.0}),
    ],
)
def test_from_spec_builds_registered_member(
    spec: dict,
    injected: dict,
    expected_cls: type,
    expected_attrs: dict,
):
    """``from_spec`` constructs the member named by ``kind``."""
    family = Widget if spec["kind"] != "bolt" else Gadget
    obj = family.from_spec(spec, **injected)
    assert isinstance(obj, expected_cls)
    for name, value in expected_attrs.items():
        assert getattr(obj, name) == value


def test_from_spec_missing_kind_raises():
    """Specs without a ``kind`` key are rejected."""
    with pytest.raises(ValueError, match="missing a 'kind' key"):
        Widget.from_spec({"scale": 1.0})


def test_from_spec_unknown_kind_raises():
    """Unknown kind names raise with the registered kinds listed."""
    with pytest.raises(ValueError, match="unknown Widget kind 'nope'"):
        Widget.from_spec({"kind": "nope"})


def test_from_spec_warns_and_drops_unknown_params():
    """Unknown spec keys are ignored after a warning."""
    with pytest.warns(UserWarning, match="ignoring unknown params"):
        obj = Widget.from_spec({"kind": "alpha", "scale": 2.0, "extra": 99})
    assert isinstance(obj, Alpha)
    assert obj.scale == 2.0
    assert not hasattr(obj, "extra")


def test_from_spec_treats_injected_names_as_disallowed_in_spec():
    """Injected argument names cannot be supplied via the spec mapping."""
    with pytest.warns(UserWarning, match="ignoring unknown params"):
        obj = Widget.from_spec({"kind": "alpha", "n": 3}, n=5)
    assert obj.n == 5


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [
        (1, int),
        ("x", str),
        ({"a": 1}, FrozenDict),
        ([1, 2], tuple),
        ((1, 2), tuple),
        ({"nested": {"b": [3, 4]}}, FrozenDict),
    ],
)
def test_freeze_converts_containers(value, expected_type: type):
    """``freeze`` leaves scalars alone and freezes mappings/sequences."""
    frozen = freeze(value)
    assert isinstance(frozen, expected_type)


def test_freeze_recurses_into_nested_structures():
    """Nested mappings and lists become FrozenDict / tuple trees."""
    frozen = freeze({"opts": {"tol": 1e-6, "flags": ["a", "b"]}, "n": 2})
    assert isinstance(frozen, FrozenDict)
    assert isinstance(frozen["opts"], FrozenDict)
    assert frozen["opts"]["tol"] == 1e-6
    assert frozen["opts"]["flags"] == ("a", "b")
    assert frozen["n"] == 2


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"a": 1},
        {"a": 1, "b": "x"},
        {"nested": {"c": [1, 2]}},
    ],
)
def test_frozen_dict_mapping_protocol_and_hash(data: dict):
    """FrozenDict behaves as a Mapping and is hashable / equality-comparable."""
    fd = FrozenDict(data)
    assert len(fd) == len(data)
    assert set(fd) == set(data)
    for key in data:
        assert fd[key] == freeze(data[key])

    twin = FrozenDict(data)
    assert fd == twin
    assert hash(fd) == hash(twin)
    assert {fd: "ok"}[twin] == "ok"
    assert fd != data
    assert "FrozenDict" in repr(fd)


def test_static_field_names_reports_static_equinox_fields():
    """Only ``static=True`` equinox fields are returned."""

    class Model(eqx.Module):
        weight: float
        name: str = eqx.field(static=True)
        kind: str = eqx.field(static=True, default="linear")

    assert static_field_names(Model) == {"name", "kind"}


def test_kind_family_field_names_detects_registry_annotations():
    """Fields typed as a kind family (or union therewith) are detected."""

    class Holder(eqx.Module):
        widget: Widget
        optional_widget: Widget | None
        gadget: Gadget | None
        plain: int
        label: str

    assert kind_family_field_names(Holder) == {
        "widget",
        "optional_widget",
        "gadget",
    }
