"""Unit tests for :mod:`slsqp_jax.sqpdax.secant.lbfgs`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax.sqpdax.secant import LBFGS, CurvatureDiagnostics, Secant
from tests.sqpdax.secant.conftest import build_lbfgs_with_pairs, explicit_diagonal

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Construction / registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, {"memory": 4, "skip_threshold": 1e-8, "regularization": 0.0}),
        (
            {"skip_threshold": 1e-6, "damping_threshold": 0.1, "diag_floor": 1e-3},
            {"skip_threshold": 1e-6, "damping_threshold": 0.1, "diag_floor": 1e-3},
        ),
    ],
)
def test_lbfgs_init_defaults_and_overrides(n: int, memory: int, kwargs, expected):
    """Constructor stores static config and starts with identity ``B₀``."""
    hist = LBFGS(n=n, memory=memory, **kwargs)
    assert hist.n == n
    assert hist.memory == memory
    assert hist.kind == "lbfgs"
    assert int(hist.count) == 0
    assert int(hist.next_idx) == 0
    np.testing.assert_allclose(hist.diagonal, jnp.ones(n))
    np.testing.assert_allclose(hist.s_history, jnp.zeros((memory, n)))
    np.testing.assert_allclose(hist.y_history, jnp.zeros((memory, n)))
    for key, value in expected.items():
        assert getattr(hist, key) == pytest.approx(value)


def test_lbfgs_registers_on_secant_family():
    """``LBFGS`` is discoverable via :meth:`Secant.from_spec`."""
    assert Secant._registry["lbfgs"] is LBFGS
    hist = Secant.from_spec({"kind": "lbfgs", "memory": 3}, n=2)
    assert isinstance(hist, LBFGS)
    assert hist.n == 2
    assert hist.memory == 3


# ---------------------------------------------------------------------------
# Empty history operators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "v",
    [
        jnp.array([1.0, -2.0, 0.5, 3.0, -1.0]),
        jnp.zeros(5),
        jnp.ones(5),
    ],
    ids=["mixed", "zero", "ones"],
)
def test_empty_hvp_and_inverse_are_diagonal(empty_lbfgs: LBFGS, v: jnp.ndarray):
    """With no pairs, ``B = diag(d)`` and ``H = diag(1/d)``."""
    np.testing.assert_allclose(empty_lbfgs.hvp(v), empty_lbfgs.diagonal * v)
    np.testing.assert_allclose(
        empty_lbfgs.inverse_hvp(v), v / empty_lbfgs.diagonal, atol=1e-14
    )


# ---------------------------------------------------------------------------
# Skip / diagnostics / append
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("s", "y", "expect_skip"),
    [
        (jnp.ones(4), 2.0 * jnp.ones(4), False),
        (jnp.zeros(4), jnp.ones(4), True),
        (jnp.ones(4), jnp.zeros(4), True),
        (jnp.ones(4) * 1e-12, jnp.ones(4), True),
        (jnp.array([1.0, 0.0, 0.0, 0.0]), jnp.array([1e10, 0.0, 0.0, 0.0]), True),
        (jnp.ones(4), jnp.full(4, jnp.nan), True),
    ],
    ids=[
        "accept",
        "zero-s",
        "zero-y",
        "tiny-s",
        "extreme-ratio",
        "nan-y",
    ],
)
def test_should_skip_and_append_respect_predicate(s, y, expect_skip: bool):
    """``should_skip`` and ``append`` agree on accept / reject."""
    hist = LBFGS(n=4, memory=5)
    skipped = bool(hist.should_skip(s, y))
    assert skipped is expect_skip

    out = hist.append(s, y)
    if expect_skip:
        assert int(out.count) == 0
        assert int(out.next_idx) == 0
    else:
        assert int(out.count) == 1
        assert int(out.next_idx) == 1
        np.testing.assert_allclose(out.s_history[0], s)


def test_diagnostics_matches_should_skip_and_inner_product():
    """Diagnostics expose ``sᵀy``, relative curvature, and the skip flag."""
    hist = LBFGS(n=3, memory=4)
    s = jnp.array([1.0, 2.0, -1.0])
    y = jnp.array([2.0, 4.0, -2.0])
    diag = hist.diagnostics(s, y)
    assert isinstance(diag, CurvatureDiagnostics)
    np.testing.assert_allclose(diag.raw_curvature, jnp.dot(s, y))
    expected_rel = jnp.abs(jnp.dot(s, y)) / (jnp.linalg.norm(s) * jnp.linalg.norm(y))
    np.testing.assert_allclose(diag.relative_curvature, expected_rel)
    assert bool(diag.skipped) is bool(hist.should_skip(s, y))


@pytest.mark.parametrize(
    ("diag_floor", "diag_ceil", "y_scale", "check"),
    [
        (1e-3, 1e6, 1e-10, "floor"),
        (1e-4, 1e2, 1e8, "ceil"),
    ],
    ids=["floor", "ceil"],
)
def test_append_clips_diagonal_to_absolute_bracket(
    diag_floor: float, diag_ceil: float, y_scale: float, check: str
):
    """Absolute ``diag_floor`` / ``diag_ceil`` bound the updated diagonal."""
    hist = LBFGS(
        n=3,
        memory=5,
        diag_floor=diag_floor,
        diag_ceil=diag_ceil,
    )
    s = jnp.array([1.0, 0.0, 0.0])
    y = y_scale * s
    out = hist.append(s, y)
    if check == "floor":
        assert float(jnp.min(out.diagonal)) >= diag_floor - 1e-12
    else:
        assert float(jnp.max(out.diagonal)) <= diag_ceil + 1e-12


def test_append_damps_negative_curvature_pairs():
    """Pairs with ``sᵀy`` below the damping threshold are Powell-damped."""
    hist = LBFGS(n=3, memory=4, damping_threshold=0.2)
    s = jnp.array([1.0, 0.0, 0.0])
    # sᵀy = -1, sᵀ B₀ s = 1 → damping must fire
    y = jnp.array([-1.0, 0.0, 0.0])
    out = hist.append(s, y)
    assert int(out.count) == 1
    stored_y = out.y_history[0]
    # Damped y should satisfy sᵀ y_damped ≥ threshold · sᵀ B₀ s
    sTy = float(jnp.dot(s, stored_y))
    sTB0s = float(jnp.dot(s, hist.diagonal * s))
    assert sTy >= 0.2 * sTB0s - 1e-10
    # And must differ from the raw (negative-curvature) y
    assert not jnp.allclose(stored_y, y)


def test_circular_buffer_overwrites_oldest_pairs():
    """Once full, further appends keep ``count == memory`` and wrap ``next_idx``."""
    memory = 3
    n = 2
    pairs = [
        (jnp.array([float(i + 1), 0.0]), jnp.array([2.0 * (i + 1), 0.0]))
        for i in range(5)
    ]
    hist = build_lbfgs_with_pairs(n, pairs, memory=memory)
    assert int(hist.count) == memory
    assert int(hist.next_idx) == 5 % memory
    # Newest three steps should be i=2,3,4 stored chronologically
    start = (hist.next_idx - hist.count + memory) % memory
    indices = (start + jnp.arange(memory)) % memory
    stored_s0 = hist.s_history[indices[0]]
    np.testing.assert_allclose(stored_s0, jnp.array([3.0, 0.0]))


# ---------------------------------------------------------------------------
# Compact-form operators with history
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regularization", [0.0, 1e-8], ids=["exact", "ridge"])
def test_hvp_inverse_hvp_roundtrip_after_append(regularization: float):
    """``H (B v) ≈ v`` after one well-conditioned pair."""
    hist = LBFGS(n=5, memory=6, regularization=regularization)
    s = jnp.array([0.1, -0.2, 0.3, -0.1, 0.05])
    y = jnp.array([0.5, 0.3, 0.8, 0.4, 0.2])
    hist = hist.append(s, y)
    v = jax.random.normal(jax.random.PRNGKey(0), (5,))
    atol = 1e-4 if regularization > 0 else 1e-5
    np.testing.assert_allclose(hist.inverse_hvp(hist.hvp(v)), v, atol=atol)


@pytest.mark.parametrize("n_pairs", [1, 3, 5], ids=["one", "three", "five"])
def test_compute_diagonal_matches_unit_probe(n_pairs: int):
    """``_compute_diagonal`` agrees with probing ``eᵢᵀ B eᵢ``."""
    key = jax.random.PRNGKey(0)
    n, memory = 6, 4
    pairs = []
    for _ in range(n_pairs):
        k1, k2, key = jax.random.split(key, 3)
        s = jax.random.normal(k1, (n,))
        y = jax.random.normal(k2, (n,)) * 0.1 + 1.5 * s
        pairs.append((s, y))
    hist = build_lbfgs_with_pairs(n, pairs, memory=memory)
    np.testing.assert_allclose(
        hist._compute_diagonal(),
        explicit_diagonal(hist),
        rtol=1e-5,
    )


def test_estimate_condition_from_diagonal():
    """Condition bounds are ``1/max(d)`` and ``1/min(d)``."""
    hist = LBFGS(n=3, memory=2)
    hist = hist.append(jnp.array([1.0, 0.0, 0.0]), jnp.array([2.0, 0.0, 0.0]))
    # Force a known non-uniform diagonal via tree-level replacement is awkward;
    # instead check the formula on the post-append diagonal.
    lo, hi = hist.estimate_condition()
    d_safe = jnp.maximum(hist.diagonal, 1e-30)
    np.testing.assert_allclose(lo, jnp.min(1.0 / d_safe))
    np.testing.assert_allclose(hi, jnp.max(1.0 / d_safe))
    np.testing.assert_allclose(hist.inv_eig_lower, lo)
    np.testing.assert_allclose(hist.inv_eig_upper, hi)


# ---------------------------------------------------------------------------
# Resets
# ---------------------------------------------------------------------------


@pytest.fixture
def filled_lbfgs() -> LBFGS:
    """History with several pairs for reset tests."""
    key = jax.random.PRNGKey(99)
    n = 5
    pairs = []
    for _ in range(4):
        k1, k2, key = jax.random.split(key, 3)
        s = jax.random.normal(k1, (n,))
        y = jax.random.normal(k2, (n,)) * 0.1 + s
        pairs.append((s, y))
    return build_lbfgs_with_pairs(n, pairs, memory=6)


@pytest.mark.parametrize(
    ("severity", "expect_count", "expect_identity_diag"),
    [
        (0, 1, False),
        (1, 0, False),
        (2, 0, True),
    ],
    ids=["soft", "snopt-diagonal", "identity"],
)
def test_reset_escalation(
    filled_lbfgs: LBFGS,
    severity: int,
    expect_count: int,
    expect_identity_diag: bool,
):
    """``reset(severity)`` maps to soft / diagonal / identity behaviour."""
    before = filled_lbfgs
    assert int(before.count) > 1
    out = before.reset(severity)
    assert int(out.count) == expect_count
    assert out.n == before.n
    assert out.memory == before.memory

    if severity == 0:
        # Soft reset keeps the newest pair at slot 0
        newest_idx = (before.next_idx - 1 + before.memory) % before.memory
        np.testing.assert_allclose(out.s_history[0], before.s_history[newest_idx])
        np.testing.assert_allclose(out.diagonal, before.diagonal)
    elif severity == 1:
        expected_diag = jnp.clip(
            before._compute_diagonal(), before.diag_floor, before.diag_ceil
        )
        expected_diag = jnp.where(jnp.isfinite(expected_diag), expected_diag, 1.0)
        np.testing.assert_allclose(out.diagonal, expected_diag, rtol=1e-8)
        assert int(out.next_idx) == 0
        np.testing.assert_allclose(out.s_history, jnp.zeros_like(out.s_history))
    else:
        np.testing.assert_allclose(out.diagonal, jnp.ones(out.n))
        assert expect_identity_diag
        assert int(out.next_idx) == 0


def test_soft_reset_on_empty_history_is_noop_shape(empty_lbfgs: LBFGS):
    """Soft-resetting an empty history leaves ``count == 0``."""
    out = empty_lbfgs.reset(0)
    assert int(out.count) == 0
    assert int(out.next_idx) == 0


def test_hvp_after_diagonal_reset_uses_extracted_diagonal(filled_lbfgs: LBFGS):
    """After severity-1 reset, ``B v = diag(d) v`` with no pairs."""
    reset = filled_lbfgs.reset(1)
    assert int(reset.count) == 0
    v = jnp.arange(1.0, reset.n + 1.0)
    np.testing.assert_allclose(reset.hvp(v), reset.diagonal * v, atol=1e-10)
    np.testing.assert_allclose(reset.inverse_hvp(v), v / reset.diagonal, atol=1e-10)


def test_append_after_diagonal_reset_updates_componentwise_secant(filled_lbfgs: LBFGS):
    """Post-reset append refreshes the diagonal via component-wise secant."""
    reset = filled_lbfgs.reset(1)
    s = jnp.array([0.1, -0.2, 0.3, -0.1, 0.05])
    y = jnp.array([0.5, 0.3, 0.8, 0.4, 0.2])
    out = reset.append(s, y)
    assert int(out.count) == 1
    # No damping expected: sᵀy is comfortably above threshold · sᵀ B₀ s
    curvature_scale = jnp.dot(y, y) / jnp.maximum(jnp.dot(y, s), 1e-12)
    curvature_scale = jnp.clip(curvature_scale, out.diag_floor, out.diag_ceil)
    clip_lo = jnp.maximum(curvature_scale * 1e-2, out.diag_floor)
    clip_hi = jnp.minimum(curvature_scale * 1e2, out.diag_ceil)
    expected = jnp.clip(jnp.abs(y * s) / jnp.maximum(s**2, 1e-12), clip_lo, clip_hi)
    np.testing.assert_allclose(out.diagonal, expected, atol=1e-12)
