"""Unit tests for :mod:`slsqp_jax.sqpdax.secant.reset`."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax.sqpdax.secant import LBFGS, SecantResetPolicy, SecantResetSignals
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


def _signals(sub: int, step: int, patience: int = 3) -> SecantResetSignals:
    return SecantResetSignals(
        subproblem_streak=jnp.asarray(sub),
        step_streak=jnp.asarray(step),
        subproblem_patience=patience,
        step_patience=patience,
    )


@pytest.mark.parametrize(
    ("n_pairs", "ill", "sub", "step", "policy_kwargs", "expected"),
    [
        (2, False, 0, 0, {}, -1),
        (2, True, 0, 0, {}, 0),
        (1, True, 0, 0, {}, -1),
        (2, False, 1, 0, {}, 0),
        (2, False, 0, 1, {}, 0),
        (2, False, 2, 0, {}, -1),
        (2, False, 3, 0, {}, 2),
        (2, False, 0, 4, {}, 2),
        (2, True, 3, 1, {}, 2),
        (2, True, 0, 0, {"condition_severity": 1}, 1),
        (2, False, 1, 0, {"first_failure_severity": 1}, 1),
        (2, True, 3, 3, {"enabled": False}, -1),
    ],
    ids=[
        "quiet",
        "ill-conditioned",
        "ill-but-one-pair",
        "first-qp-failure",
        "first-ls-failure",
        "mid-streak",
        "qp-patience",
        "ls-beyond-patience",
        "strongest-wins",
        "custom-condition-severity",
        "custom-first-failure-severity",
        "disabled",
    ],
)
def test_severity_truth_table(n_pairs, ill, sub, step, policy_kwargs, expected):
    """Each trigger maps to its configured severity; the strongest one wins."""
    policy = SecantResetPolicy().init(**policy_kwargs)
    secant = _secant(n_pairs=n_pairs, ill_conditioned=ill)
    assert int(policy.severity(secant, _signals(sub, step))) == expected


def test_patience_equal_one_escalates_first_failure_to_patience_level():
    """With ``patience=1`` the first failure already is the patience failure."""
    policy = SecantResetPolicy()
    secant = _secant(n_pairs=2, ill_conditioned=False)
    assert int(policy.severity(secant, _signals(1, 0, patience=1))) == 2


@pytest.mark.parametrize(
    ("sub", "expected_count"),
    [(0, 2), (1, 1), (3, 0)],
    ids=["no-reset", "soft", "identity"],
)
def test_apply_resets_at_selected_severity_under_jit(sub: int, expected_count: int):
    """``apply`` resets iff a trigger fired and reports the applied severity."""
    policy = SecantResetPolicy()
    secant = _secant(n_pairs=2, ill_conditioned=False)
    out, severity = jax.jit(policy.apply)(secant, _signals(sub, 0))
    assert int(out.num_pairs) == expected_count
    expected = policy.severity(secant, _signals(sub, 0))
    assert int(severity) == int(expected)
    if int(severity) < 0:
        np.testing.assert_allclose(out.s_history, secant.s_history)


def test_none_signals_report_no_failures():
    """``SecantResetSignals.none()`` only leaves the conditioning trigger."""
    policy = SecantResetPolicy()
    quiet = SecantResetSignals.none()
    assert int(policy.severity(_secant(n_pairs=2, ill_conditioned=False), quiet)) == -1
    assert int(policy.severity(_secant(n_pairs=2, ill_conditioned=True), quiet)) == 0


def test_signals_reject_nonpositive_patience():
    """A patience below one would fire on a zero streak, so it is rejected."""
    with pytest.raises(ValueError, match="patience must be at least 1"):
        SecantResetSignals(
            subproblem_streak=jnp.asarray(0),
            step_streak=jnp.asarray(0),
            subproblem_patience=0,
        )
