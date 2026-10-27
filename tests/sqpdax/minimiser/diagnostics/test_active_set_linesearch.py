"""Unit tests for :mod:`slsqp_jax.sqpdax.minimiser.diagnostics.active_set_linesearch`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.minimiser.diagnostics import ActiveSetLineSearchDiagnostics
from tests.sqpdax.minimiser.diagnostics.conftest import make_prediction

COUNTERS = (
    "n_armijo_accepts",
    "n_fallback_accepts",
    "n_soc_accepts",
    "n_lpeca_bypassed",
    "n_lpeca_capped",
    "n_lpeca_bounds_prefixed",
)


def test_zero_carry_is_all_zero(zero_als_diagnostics: ActiveSetLineSearchDiagnostics):
    """``zero()`` starts every counter at zero with the fallback flag cleared."""
    assert not bool(zero_als_diagnostics.last_ls_fallback)
    assert not bool(zero_als_diagnostics.last_soc)
    for name in COUNTERS:
        value = getattr(zero_als_diagnostics, name)
        assert int(value) == 0
        assert value.dtype == jnp.int32
    assert set(zero_als_diagnostics.stats()) == {
        "last_ls_fallback",
        "last_soc",
        *COUNTERS,
    }


@pytest.mark.parametrize(
    (
        "steps",
        "expected_armijo",
        "expected_fallback",
        "expected_last",
        "expected_soc",
        "expected_last_soc",
    ),
    [
        # (accepted, accepted_by_fallback, corrected) per step
        ([(True, False, False)], 1, 0, False, 0, False),
        ([(True, True, False)], 0, 1, True, 0, False),
        ([(False, True, False)], 0, 0, False, 0, False),
        (
            [(True, False, False), (True, True, False), (False, False, False)],
            1,
            1,
            False,
            0,
            False,
        ),
        ([(True, False, False), (True, True, False)], 1, 1, True, 0, False),
        # A corrected acceptance is an Armijo acceptance that is also counted
        # as a correction; a rejected corrected trial counts nowhere.
        ([(True, False, True)], 1, 0, False, 1, True),
        ([(False, False, True)], 0, 0, False, 0, False),
        ([(True, False, True), (True, False, False)], 2, 0, False, 1, False),
    ],
)
def test_record_step_counts_acceptance_kind(
    zero_als_diagnostics: ActiveSetLineSearchDiagnostics,
    steps: list[tuple[bool, bool, bool]],
    expected_armijo: int,
    expected_fallback: int,
    expected_last: bool,
    expected_soc: int,
    expected_last_soc: bool,
):
    """Armijo, fallback and corrected acceptances are counted; rejections count nowhere."""
    diag = zero_als_diagnostics
    for accepted, by_fallback, corrected in steps:
        diag = diag.record_step(
            accepted=jnp.asarray(accepted),
            accepted_by_fallback=jnp.asarray(by_fallback),
            corrected=jnp.asarray(corrected),
        )
    assert int(diag.n_armijo_accepts) == expected_armijo
    assert int(diag.n_fallback_accepts) == expected_fallback
    assert bool(diag.last_ls_fallback) is expected_last
    assert int(diag.n_soc_accepts) == expected_soc
    assert bool(diag.last_soc) is expected_last_soc
    for name in COUNTERS[3:]:
        assert int(getattr(diag, name)) == 0


@pytest.mark.parametrize(
    ("predictions", "expected"),
    [
        ([dict(valid=True)], (0, 0, 0)),
        ([dict(valid=False)], (1, 0, 0)),
        ([dict(valid=True, capped=True, n_bounds_prefixed=3)], (0, 1, 3)),
        (
            [
                dict(valid=False, capped=True, n_bounds_prefixed=2),
                dict(valid=True, n_bounds_prefixed=5),
                dict(valid=False),
            ],
            (2, 1, 7),
        ),
    ],
)
def test_record_prediction_accumulates_lpeca_counters(
    zero_als_diagnostics: ActiveSetLineSearchDiagnostics,
    predictions: list[dict],
    expected: tuple[int, int, int],
):
    """Bypassed / capped predictions and prefixed bounds accumulate across calls."""
    diag = zero_als_diagnostics
    for spec in predictions:
        diag = diag.record_prediction(make_prediction(**spec))
    assert (
        int(diag.n_lpeca_bypassed),
        int(diag.n_lpeca_capped),
        int(diag.n_lpeca_bounds_prefixed),
    ) == expected
    assert int(diag.n_armijo_accepts) == int(diag.n_fallback_accepts) == 0


def test_updates_are_jittable(zero_als_diagnostics: ActiveSetLineSearchDiagnostics):
    """Both update methods trace under ``jax.jit`` and return the same carry type."""

    @jax.jit
    def advance(diag: ActiveSetLineSearchDiagnostics) -> ActiveSetLineSearchDiagnostics:
        diag = diag.record_prediction(make_prediction(valid=False, n_bounds_prefixed=1))
        return diag.record_step(
            accepted=jnp.asarray(True), accepted_by_fallback=jnp.asarray(True)
        )

    diag = advance(zero_als_diagnostics)
    assert isinstance(diag, ActiveSetLineSearchDiagnostics)
    assert int(diag.n_lpeca_bypassed) == 1
    assert int(diag.n_lpeca_bounds_prefixed) == 1
    assert int(diag.n_fallback_accepts) == 1
    assert bool(diag.last_ls_fallback)
