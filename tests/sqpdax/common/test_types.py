"""Unit tests for :class:`~slsqp_jax.sqpdax.common.types.InitializableModule`."""

from __future__ import annotations

import equinox as eqx
import pytest

from slsqp_jax.sqpdax.common.types import InitializableModule


class Leaf(InitializableModule):
    """Minimal leaf module with a single scalar field."""

    value: int | None = None


class Nested(InitializableModule):
    """Module whose fields include another :class:`InitializableModule`."""

    name: str | None = None
    leaf: Leaf = eqx.field(default_factory=Leaf)


class Deep(InitializableModule):
    """Two-level nesting for recursive ``init`` coverage."""

    scale: float | None = None
    nested: Nested = eqx.field(default_factory=Nested)


@pytest.fixture
def leaf() -> Leaf:
    return Leaf()


@pytest.fixture
def nested() -> Nested:
    return Nested()


@pytest.fixture
def deep() -> Deep:
    return Deep()


@pytest.mark.parametrize(
    ("kwargs", "expected_value"),
    [
        ({"value": 1}, 1),
        ({"value": 0}, 0),
        ({"value": -7}, -7),
    ],
)
def test_init_replaces_flat_fields(leaf: Leaf, kwargs: dict, expected_value: int):
    """``init`` replaces ordinary (non-nested) fields."""
    result = leaf.init(**kwargs)
    assert isinstance(result, Leaf)
    assert result is not leaf
    assert result.value == expected_value
    assert leaf.value is None


@pytest.mark.parametrize(
    ("first", "second", "expected_value"),
    [
        ({"value": 1}, {"value": 2}, 2),
        ({"value": 0}, {"value": -3}, -3),
        ({"value": 5}, {"value": None}, None),
    ],
)
def test_init_overwrites_already_set_flat_fields(
    leaf: Leaf,
    first: dict,
    second: dict,
    expected_value: int | None,
):
    """``init`` can replace fields that are already non-``None``."""
    once = leaf.init(**first)
    twice = once.init(**second)
    assert twice.value == expected_value
    assert once.value == first["value"]
    assert leaf.value is None


@pytest.mark.parametrize(
    ("first", "second", "expected_name", "expected_leaf_value"),
    [
        (
            {"name": "a", "leaf": {"value": 1}},
            {"name": "b"},
            "b",
            1,
        ),
        (
            {"name": "a", "leaf": {"value": 1}},
            {"leaf": {"value": 9}},
            "a",
            9,
        ),
        (
            {"name": "a", "leaf": {"value": 1}},
            {"name": "c", "leaf": {"value": 4}},
            "c",
            4,
        ),
    ],
)
def test_init_overwrites_already_set_nested_fields(
    nested: Nested,
    first: dict,
    second: dict,
    expected_name: str,
    expected_leaf_value: int,
):
    """Nested ``init`` can overwrite fields that were set by a prior ``init``."""
    once = nested.init(**first)
    twice = once.init(**second)
    assert twice.name == expected_name
    assert twice.leaf.value == expected_leaf_value
    assert once.name == first["name"]
    assert once.leaf.value == first["leaf"]["value"]


@pytest.mark.parametrize(
    ("kwargs", "expected_name", "expected_leaf_value"),
    [
        ({"name": "a"}, "a", None),
        ({"leaf": {"value": 3}}, None, 3),
        ({"name": "b", "leaf": {"value": 5}}, "b", 5),
    ],
)
def test_init_recurses_into_nested_initializable_modules(
    nested: Nested,
    kwargs: dict,
    expected_name: str | None,
    expected_leaf_value: int | None,
):
    """Nested :class:`InitializableModule` fields are initialized recursively.

    When a kwarg names a nested ``InitializableModule`` field, its value
    must be a mapping of kwargs forwarded to that field's ``init``.
    """
    result = nested.init(**kwargs)
    assert isinstance(result, Nested)
    assert result is not nested
    assert result.name == expected_name
    assert isinstance(result.leaf, Leaf)
    assert result.leaf.value == expected_leaf_value
    # Original tree is left untouched.
    assert nested.name is None
    assert nested.leaf.value is None


def test_init_with_empty_kwargs_returns_equivalent_module(leaf: Leaf):
    """Calling ``init()`` with no kwargs leaves the module contents unchanged."""
    result = leaf.init()
    assert isinstance(result, Leaf)
    assert result.value is leaf.value


@pytest.mark.parametrize(
    ("kwargs", "expected_scale", "expected_name", "expected_leaf_value"),
    [
        ({"scale": 2.5}, 2.5, None, None),
        (
            {"nested": {"name": "inner", "leaf": {"value": 9}}},
            None,
            "inner",
            9,
        ),
        (
            {
                "scale": 1.5,
                "nested": {"name": "both", "leaf": {"value": 4}},
            },
            1.5,
            "both",
            4,
        ),
    ],
)
def test_init_recurses_through_multiple_nesting_levels(
    deep: Deep,
    kwargs: dict,
    expected_scale: float | None,
    expected_name: str | None,
    expected_leaf_value: int | None,
):
    """``init`` recurses through arbitrarily nested ``InitializableModule`` trees."""
    result = deep.init(**kwargs)
    assert isinstance(result, Deep)
    assert result.scale == expected_scale
    assert result.nested.name == expected_name
    assert result.nested.leaf.value == expected_leaf_value
    assert deep.scale is None
    assert deep.nested.name is None
    assert deep.nested.leaf.value is None


def test_init_does_not_treat_plain_modules_as_nested():
    """Only fields that are ``InitializableModule`` instances recurse.

    A plain equinox ``Module`` (or other object) assigned to a field is
    replaced wholesale, not recursively initialized.
    """

    class Plain(eqx.Module):
        value: int | None = None

    class Holder(InitializableModule):
        plain: Plain = eqx.field(default_factory=Plain)

    holder = Holder()
    replacement = Plain(value=42)
    result = holder.init(plain=replacement)
    assert eqx.tree_equal(result.plain, replacement)
    assert result.plain.value == 42
    assert holder.plain.value is None
