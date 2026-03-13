import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Make ``very_slow`` imply ``slow`` so that ``-m "not slow"`` deselects both."""
    slow_marker = pytest.mark.slow
    for item in items:
        if item.get_closest_marker("very_slow") is not None:
            item.add_marker(slow_marker)
