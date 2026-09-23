"""Small identity-based caches for repeated problem evaluations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class _CachedEvaluator:
    """Cache the most recent evaluation, keyed by the identity of ``x``."""

    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn
        self._cache_key: int | None = None
        self._cache_val: Any = None

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        key = id(x)
        if key != self._cache_key:
            self._cache_val = self._fn(x, *args, **kwargs)
            self._cache_key = key
        return self._cache_val


class _CachedEvaluator2:
    """Cache the most recent evaluation, keyed by identities of ``x`` and ``v``."""

    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn
        self._cache_key: tuple[int, int] | None = None
        self._cache_val: Any = None

    def __call__(self, x: Any, v: Any, *args: Any, **kwargs: Any) -> Any:
        key = (id(x), id(v))
        if key != self._cache_key:
            self._cache_val = self._fn(x, v, *args, **kwargs)
            self._cache_key = key
        return self._cache_val
