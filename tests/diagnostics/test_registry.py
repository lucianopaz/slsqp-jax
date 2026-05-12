"""Cap-policy enforcement: every registered signal has matching tests.

The plan's cap policy (see ``AGENTS.md`` documentation update) is:
no signal lands in :data:`SIGNAL_REGISTRY` without (a) a synthetic
test in ``test_signals_synthetic.py`` and (b) an integration test in
``test_signals_integration.py``.  This test makes that policy
machine-checkable: CI fails when a registration lacks the matching
test functions.

Skipped tests are accepted by the cap policy (the synthetic test
carries the unit-level fidelity contract; the integration test
exists as a slot for the future when a stable real-problem trigger
is found).  An empty test slot or a renamed test function would
fail this enforcement.
"""

from __future__ import annotations

import importlib

import pytest

from slsqp_jax.diagnostics.signals import SIGNAL_REGISTRY


@pytest.fixture(scope="module")
def synthetic_test_names() -> set[str]:
    mod = importlib.import_module("tests.diagnostics.test_signals_synthetic")
    return {name for name in dir(mod) if name.startswith("test_signal_")}


@pytest.fixture(scope="module")
def integration_test_names() -> set[str]:
    mod = importlib.import_module("tests.diagnostics.test_signals_integration")
    return {name for name in dir(mod) if name.startswith("test_signal_")}


def test_every_registered_signal_has_a_synthetic_test(synthetic_test_names):
    """Cap policy: every signal name has a ``test_signal_<name>_synthetic``."""
    missing: list[str] = []
    for reg in SIGNAL_REGISTRY:
        expected = f"test_signal_{reg.name}_synthetic"
        if expected not in synthetic_test_names:
            missing.append(expected)
    assert not missing, (
        "The following registered signals are missing a synthetic test "
        f"in tests/diagnostics/test_signals_synthetic.py: {missing}.  "
        "Per the cap policy in AGENTS.md, every signal must have a "
        "named synthetic test.  Add the missing function (a "
        "``raise NotImplementedError`` body is acceptable as a "
        "placeholder while the threshold is being tuned)."
    )


def test_every_registered_signal_has_an_integration_test(integration_test_names):
    """Cap policy: every signal name has a ``test_signal_<name>_integration``."""
    missing: list[str] = []
    for reg in SIGNAL_REGISTRY:
        expected = f"test_signal_{reg.name}_integration"
        if expected not in integration_test_names:
            missing.append(expected)
    assert not missing, (
        "The following registered signals are missing an integration "
        "test in tests/diagnostics/test_signals_integration.py: "
        f"{missing}.  Per the cap policy in AGENTS.md, every signal "
        "must have a named integration test.  A skipped test (with "
        "``@pytest.mark.skip(reason=...)``) is acceptable while a "
        "stable real-problem trigger is being found."
    )


def test_signal_registry_is_non_empty():
    """Sanity check: the registry should not be silently empty."""
    assert len(SIGNAL_REGISTRY) >= 1


def test_registered_specificity_is_a_known_label():
    """Every registration must declare one of the documented specificities."""
    valid = {"specific", "ambiguous", "generic"}
    for reg in SIGNAL_REGISTRY:
        assert reg.specificity in valid, (
            f"signal {reg.name!r} declared specificity={reg.specificity!r}; "
            f"must be one of {valid}"
        )


def test_registered_flavour_is_a_known_label():
    """Every registration must declare one of the documented flavours."""
    valid = {"per_step", "end_of_run"}
    for reg in SIGNAL_REGISTRY:
        assert reg.flavour in valid, (
            f"signal {reg.name!r} declared flavour={reg.flavour!r}; "
            f"must be one of {valid}"
        )
