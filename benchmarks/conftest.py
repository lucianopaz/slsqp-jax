"""Pytest hooks for the doctests collected from this package.

``benchmarks/problems.py`` imports sif2jax (about 10 s), so its doctests are
marked ``slow`` like the rest of the sif2jax-dependent tests.
"""

from __future__ import annotations

import pytest

_SLOW_DOCTEST_MODULES = ("problems.py",)

# Marimo notebooks are scripts, not libraries; nothing to doctest.
collect_ignore = ["report.py", "dashboard.py", "__main__.py"]


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.path is not None and item.path.name in _SLOW_DOCTEST_MODULES:
            item.add_marker(pytest.mark.slow)
