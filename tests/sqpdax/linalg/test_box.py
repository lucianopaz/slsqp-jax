"""Unit tests for :mod:`slsqp_jax.sqpdax.linalg.box`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.linalg import box_fraction, box_ray_length, box_segment_fraction

from .conftest import random_direction

INF = jnp.inf

CASES = {
    # (w, lo, expected ray length)
    "free": (jnp.array([1.0, -2.0]), jnp.array([-INF, -INF]), INF),
    "positive-only": (jnp.array([1.0, 2.0]), jnp.array([-1.0, -1.0]), INF),
    "single-binding": (
        jnp.array([1.0, -2.0, -0.5]),
        jnp.array([-INF, -1.0, -1.0]),
        0.5,
    ),
    "tightest-wins": (jnp.array([-1.0, -4.0]), jnp.array([-2.0, -1.0]), 0.25),
    "zero-direction": (jnp.zeros(2), jnp.array([-1.0, -1.0]), INF),
    "empty": (jnp.zeros(0), jnp.zeros(0), INF),
}


@pytest.mark.parametrize("w, lo, expected", CASES.values(), ids=CASES.keys())
def test_box_ray_length(w, lo, expected):
    """Ray length to the first lower face along ``w``; ``+inf`` when none binds."""
    alpha = box_ray_length(w, lo)
    assert alpha == expected
    if jnp.isfinite(expected) and w.shape[0] > 0:
        # Exactly one coordinate sits on its face, none below it.
        assert jnp.all(alpha * w >= lo - 1e-7)
        assert jnp.any(jnp.isclose(alpha * w, lo))


@pytest.mark.parametrize("w, lo, expected", CASES.values(), ids=CASES.keys())
def test_box_fraction_clamps_to_one(w, lo, expected):
    """``β = min{1, α}`` keeps steps already inside the box untouched."""
    beta = box_fraction(w, lo)
    assert beta == jnp.minimum(1.0, expected)
    assert 0.0 <= beta <= 1.0
    if w.shape[0] > 0:
        assert jnp.all(beta * w >= lo - 1e-7)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_box_fraction_is_scale_consistent(dim: int, seed: int):
    """Scaling the step by ``c > 1`` scales a binding ``β`` by ``1/c``."""
    w = -jnp.abs(random_direction(seed, dim)) * 4.0
    lo = -jnp.ones(dim)
    beta = box_fraction(w, lo)
    assert beta < 1.0
    assert jnp.isclose(box_fraction(2.0 * w, lo), beta / 2.0)


SEGMENT_CASES = {
    # (start, w, lo, hi, expected t)
    "inside": ([0.5, 0.0], [0.25, 1.0], [0.0, -INF], [1.0, 2.0], 1.0),
    "lower-face": ([0.5, 0.0], [-1.0, 0.0], [0.0, -INF], [1.0, 2.0], 0.5),
    "upper-face": ([0.5, 0.0], [0.0, 4.0], [0.0, -INF], [1.0, 2.0], 0.5),
    "tightest-wins": ([0.5, 0.0], [1.0, 8.0], [0.0, -INF], [1.0, 2.0], 0.25),
    "start-on-face": ([1.0, 0.0], [1.0, 0.0], [0.0, -INF], [1.0, 2.0], 0.0),
    "start-outside": ([1.5, 0.0], [-1.0, 0.0], [0.0, -INF], [1.0, 2.0], 0.0),
    "free": ([3.0, -3.0], [5.0, -5.0], [-INF, -INF], [INF, INF], 1.0),
}


@pytest.mark.parametrize(
    "start, w, lo, hi, expected", SEGMENT_CASES.values(), ids=SEGMENT_CASES.keys()
)
def test_box_segment_fraction(start, w, lo, hi, expected):
    """Largest ``t ∈ [0, 1]`` keeping ``start + t w`` in the box (0 from outside)."""
    start, w, lo, hi = (jnp.asarray(v, jnp.float32) for v in (start, w, lo, hi))
    t = box_segment_fraction(start, w, lo, hi)
    assert t == pytest.approx(expected)
    point = start + t * w
    if expected > 0.0:
        assert jnp.all((point >= lo - 1e-7) & (point <= hi + 1e-7))
