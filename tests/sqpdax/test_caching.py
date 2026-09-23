"""Tests for sqpdax's identity-based evaluation caches."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from slsqp_jax.sqpdax.caching import _CachedEvaluator, _CachedEvaluator2


def test_cached_evaluator_hits_only_for_same_object() -> None:
    calls = 0

    def fn(x):
        nonlocal calls
        calls += 1
        return 2 * x

    cached = _CachedEvaluator(fn)
    x1 = jnp.array([1.0])
    x2 = jnp.array([2.0])
    np.testing.assert_array_equal(cached(x1), cached(x1))
    cached(x2)
    assert calls == 2


def test_cached_evaluator_forwards_extra_arguments_on_miss() -> None:
    calls = 0

    def fn(x, scale, *, shift):
        nonlocal calls
        calls += 1
        return scale * x + shift

    cached = _CachedEvaluator(fn)
    x = jnp.array([2.0])
    np.testing.assert_allclose(cached(x, 3.0, shift=1.0), [7.0])
    np.testing.assert_allclose(cached(x, 3.0, shift=1.0), [7.0])
    assert calls == 1


def test_cached_evaluator2_keys_on_both_objects() -> None:
    calls = 0

    def fn(x, v):
        nonlocal calls
        calls += 1
        return x + v

    cached = _CachedEvaluator2(fn)
    x = jnp.array([1.0])
    v1 = jnp.array([2.0])
    v2 = jnp.array([3.0])
    np.testing.assert_array_equal(cached(x, v1), cached(x, v1))
    cached(x, v2)
    assert calls == 2
