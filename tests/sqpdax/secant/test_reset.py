"""Unit tests for :mod:`slsqp_jax.sqpdax.secant.reset`."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax.sqpdax.secant import (
    LBFGS,
    FailureRecoverySchedule,
    SecantRecoveryState,
    SecantResetPolicy,
    SecantResetSignals,
)
from tests.sqpdax.secant.conftest import build_lbfgs_with_pairs

jax.config.update("jax_enable_x64", True)

N = 4


def _secant(*, n_pairs: int, ill_conditioned: bool) -> LBFGS:
    pairs = [(jnp.eye(N)[i], 2.0 * jnp.eye(N)[i]) for i in range(n_pairs)]
    secant = build_lbfgs_with_pairs(N, pairs, memory=4)
    if ill_conditioned:
        secant = eqx.tree_at(
            lambda h: h.diagonal, secant, jnp.array([1e-4, 1.0, 1.0, 1e4])
        )
    return secant


def _signals(sub: int, step: int, model: int = 0) -> SecantResetSignals:
    return SecantResetSignals(
        subproblem_streak=jnp.asarray(sub),
        step_streak=jnp.asarray(step),
        model_streak=jnp.asarray(model),
    )


def test_default_schedule_is_soft_diagonal_identity_then_fatal_under_jit():
    """One global episode advances through the default 1/2/3/4 stages."""
    policy = SecantResetPolicy()
    secant = _secant(n_pairs=2, ill_conditioned=False)
    state = SecantRecoveryState()

    expected = [
        (0, 1, 0, False, 1),
        (1, 2, 1, False, 0),
        (2, 3, 2, False, 0),
        (-1, 4, 2, True, 0),
    ]
    for severity_expected, streak, stage, fatal, pairs in expected:
        secant, state, severity = jax.jit(policy.apply)(
            secant, _signals(streak, 0), state
        )
        assert int(severity) == severity_expected
        assert int(state.failure_streak) == streak
        assert int(state.stage) == stage
        assert bool(state.fatal) is fatal
        assert int(secant.num_pairs) == pairs


def test_mixed_channel_failures_count_once_and_never_downgrade():
    """Changing failure source cannot restart or blur a recovery episode."""
    policy = SecantResetPolicy()
    secant = _secant(n_pairs=2, ill_conditioned=False)
    state = SecantRecoveryState()
    episodes = [
        (_signals(1, 1, 1), 0, 1, 0),
        (_signals(0, 2, 1), 1, 2, 1),
        (_signals(1, 0, 3), 2, 3, 2),
    ]
    for signals, expected_severity, expected_streak, expected_stage in episodes:
        secant, state, severity = policy.apply(secant, signals, state)
        assert int(severity) == expected_severity
        assert int(state.failure_streak) == expected_streak
        assert int(state.stage) == expected_stage


def test_healthy_iteration_clears_episode_and_next_failure_restarts_soft():
    """A genuinely healthy iteration starts a fresh recovery epoch."""
    policy = SecantResetPolicy()
    secant = _secant(n_pairs=2, ill_conditioned=False)
    secant, state, _ = policy.apply(secant, _signals(1, 0), SecantRecoveryState())
    secant, state, severity = policy.apply(secant, _signals(0, 0), state)
    assert int(severity) == -1
    assert int(state.failure_streak) == 0
    assert int(state.stage) == -1
    assert not bool(state.fatal)

    _, state, severity = policy.apply(secant, _signals(0, 1), state)
    assert int(severity) == 0
    assert int(state.failure_streak) == 1
    assert int(state.stage) == 0


def test_custom_global_schedule_has_one_shot_stage_transitions():
    """Configured gaps retry the current stage without reapplying it."""
    policy = SecantResetPolicy(
        recovery_schedule=FailureRecoverySchedule(
            soft=1, diagonal=3, identity=5, fatal=6
        )
    )
    secant = _secant(n_pairs=2, ill_conditioned=False)
    state = SecantRecoveryState()
    expected = {1: 0, 3: 1, 5: 2}
    for streak in range(1, 7):
        secant, state, severity = policy.apply(secant, _signals(streak, 0), state)
        assert int(severity) == expected.get(streak, -1)
    assert bool(state.fatal)


def test_conditioning_and_channel_failures_share_the_same_clock():
    """Ill-conditioning starts recovery; a later channel failure advances it."""
    policy = SecantResetPolicy()
    secant = _secant(n_pairs=2, ill_conditioned=True)
    secant, state, severity = policy.apply(
        secant, _signals(0, 0), SecantRecoveryState()
    )
    assert int(severity) == 0
    assert int(state.failure_streak) == 1

    _, state, severity = policy.apply(secant, _signals(0, 1), state)
    assert int(severity) == 1
    assert int(state.failure_streak) == 2


def test_disabled_policy_clears_recovery_without_changing_secant():
    """Disabling recovery suppresses resets and fatal state."""
    policy = SecantResetPolicy(enabled=False)
    secant = _secant(n_pairs=2, ill_conditioned=True)
    state = SecantRecoveryState(
        failure_streak=jnp.asarray(4),
        stage=jnp.asarray(2),
        fatal=jnp.asarray(True),
    )
    # No channel can fire while disabled, however bad the inputs look.
    assert not bool(policy.recovery_event(secant, _signals(4, 4, 4)))
    out, state, severity = policy.apply(secant, _signals(4, 4, 4), state)
    assert int(severity) == -1
    assert int(state.failure_streak) == 0
    assert int(state.stage) == -1
    assert not bool(state.fatal)
    np.testing.assert_allclose(out.s_history, secant.s_history)


def test_none_signals_report_no_failures():
    """``none`` reports no channels while preserving conditioning recovery."""
    policy = SecantResetPolicy()
    quiet = SecantResetSignals.none()
    assert int(quiet.subproblem_streak) == 0
    assert int(quiet.step_streak) == 0
    assert int(quiet.model_streak) == 0
    state = SecantRecoveryState()
    assert (
        int(policy.severity(_secant(n_pairs=2, ill_conditioned=False), quiet, state))
        == -1
    )
    assert (
        int(policy.severity(_secant(n_pairs=2, ill_conditioned=True), quiet, state))
        == 0
    )


@pytest.mark.parametrize(
    "values",
    [(0, 3, 6, 9), (1, 1, 6, 9), (1, 3, 3, 9), (1, 3, 6, 6)],
)
def test_schedule_rejects_nonpositive_or_unordered_thresholds(values):
    """Every reset stage and the fatal threshold must be strictly ordered."""
    with pytest.raises(ValueError, match="0 < soft < diagonal < identity < fatal"):
        FailureRecoverySchedule(
            soft=values[0],
            diagonal=values[1],
            identity=values[2],
            fatal=values[3],
        )
