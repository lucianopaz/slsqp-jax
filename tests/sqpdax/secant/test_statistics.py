"""Unit tests for :mod:`slsqp_jax.sqpdax.secant.statistics`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax.sqpdax.secant import LBFGS, SecantStatistics

jax.config.update("jax_enable_x64", True)

S = jnp.array([1.0, 0.0, 0.0])
PAIRS = {
    "accept": (S, jnp.array([2.0, 0.5, 0.0])),
    "damped": (S, jnp.array([-1.0, 0.0, 0.0])),
    "skip": (jnp.zeros(3), jnp.ones(3)),
}


def _replay(names: list[str]) -> tuple[LBFGS, SecantStatistics]:
    secant = LBFGS(n=3, memory=4)
    stats = SecantStatistics.initial(secant, jnp.float64)
    for name in names:
        s, y = PAIRS[name]
        diag = secant.diagnostics(s, y)
        secant = secant.append(s, y)
        stats = stats.record_append(diag, secant)
    return secant, stats


def test_initial_is_empty_with_identity_conditioning():
    """Fresh statistics have zero counters and ``κ = 1`` for identity ``B₀``."""
    stats = SecantStatistics.initial(LBFGS(n=3, memory=2), jnp.float64)
    assert int(stats.n_appends) == int(stats.n_skips) == int(stats.n_damped) == 0
    np.testing.assert_array_equal(stats.n_resets, jnp.zeros(3, jnp.int32))
    assert float(stats.last_condition) == float(stats.max_condition) == 1.0
    assert float(stats.min_damping_theta) == 1.0


@pytest.mark.parametrize(
    "sequence",
    [
        ["accept"],
        ["skip", "skip"],
        ["accept", "damped", "skip"],
        ["damped", "damped", "accept", "skip"],
    ],
    ids=["one-accept", "all-skips", "mixed", "damped-heavy"],
)
def test_record_append_counts_and_extrema(sequence: list[str]):
    """Counters match the replayed pairs; extrema bracket the final diagonal."""
    secant, stats = _replay(sequence)
    assert int(stats.n_skips) == sequence.count("skip")
    assert int(stats.n_appends) == len(sequence) - sequence.count("skip")
    assert int(stats.n_damped) == sequence.count("damped")
    assert int(stats.n_appends) == int(secant.num_pairs)

    lower, upper = secant.curvature_bounds()
    assert float(stats.min_diagonal) <= float(lower)
    assert float(stats.max_diagonal) >= float(upper)
    np.testing.assert_allclose(stats.last_condition, upper / lower)
    assert float(stats.max_condition) >= float(stats.last_condition)
    if "damped" in sequence:
        assert float(stats.min_damping_theta) < 1.0
    else:
        assert float(stats.min_damping_theta) == 1.0


@pytest.mark.parametrize(
    ("severity", "expected"),
    [(-1, [0, 0, 0]), (0, [1, 0, 0]), (1, [0, 1, 0]), (2, [0, 0, 1]), (5, [0, 0, 1])],
    ids=["none", "soft", "diagonal", "identity", "above-identity"],
)
def test_record_reset_counts_by_severity(severity: int, expected: list[int]):
    """Each fired reset increments its severity bucket; ``-1`` records nothing."""
    secant, stats = _replay(["accept", "accept"])
    reset = secant.reset(max(severity, 0)) if severity >= 0 else secant
    out = stats.record_reset(jnp.asarray(severity), reset)
    np.testing.assert_array_equal(out.n_resets, jnp.asarray(expected))
    lower, upper = reset.curvature_bounds()
    np.testing.assert_allclose(out.last_condition, upper / lower)


def test_as_stats_prefixes_every_field():
    """``as_stats`` exposes every field under the requested prefix."""
    _, stats = _replay(["accept"])
    flat = stats.as_stats(prefix="lbfgs_")
    assert "lbfgs_n_skips" in flat
    assert "lbfgs_max_condition" in flat
    assert all(key.startswith("lbfgs_") for key in flat)


def test_record_append_is_jittable():
    """Recording runs under :func:`jax.jit` with traced diagnostics."""
    secant = LBFGS(n=3, memory=4)
    stats = SecantStatistics.initial(secant, jnp.float64)

    @jax.jit
    def record(stats, secant, s, y):
        diag = secant.diagnostics(s, y)
        return stats.record_append(diag, secant.append(s, y))

    out = record(stats, secant, *PAIRS["damped"])
    assert int(out.n_damped) == 1
