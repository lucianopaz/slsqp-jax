#!/usr/bin/env python
"""Identify tests that should be marked ``@pytest.mark.slow``.

Parses the JUnit XML produced by ``pytest`` (written to ``.test_durations.xml``
by the default ``addopts`` in ``pyproject.toml``) and selects the slowest tests
such that the remaining (non-slow) tests finish within a given time budget.

Usage::

    uv run pytest                                     # generates .test_durations.xml
    python scripts/update_slow_marks.py               # default: 300s budget
    python scripts/update_slow_marks.py --budget 120  # custom budget in seconds
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DURATIONS_FILE = Path(__file__).resolve().parent.parent / ".test_durations.xml"
DEFAULT_BUDGET_SECONDS = 300
FULL_SUITE_CMD = (
    'uv run pytest --override-ini="addopts=" --junit-xml=.test_durations.xml -v'
)


def _count_all_tests() -> int | None:
    """Run ``pytest --collect-only`` to discover the total number of tests."""
    try:
        result = subprocess.run(
            ["uv", "run", "pytest", "--override-ini=addopts=", "--collect-only", "-q"],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            # e.g. "284 tests collected in 0.96s"
            if "tests collected" in line:
                return int(line.split()[0])
    except FileNotFoundError:
        pass
    return None


def _file_from_classname(classname: str) -> str:
    """Convert a JUnit classname like ``tests.test_slsqp.TestFoo`` to ``tests/test_slsqp.py``."""
    parts = classname.split(".")
    # The classname is ``tests.<module>.<ClassName>``; the file is ``tests/<module>.py``.
    return "/".join(parts[:2]) + ".py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budget",
        type=float,
        default=DEFAULT_BUDGET_SECONDS,
        help="Maximum wall-time (seconds) allowed for non-slow tests (default: %(default)s)",
    )
    parser.add_argument(
        "--durations-file",
        type=Path,
        default=DURATIONS_FILE,
        help="Path to JUnit XML durations file (default: %(default)s)",
    )
    args = parser.parse_args()

    if not args.durations_file.exists():
        print(
            f"ERROR: {args.durations_file} not found.\n"
            f"Run the full test suite first:\n\n  {FULL_SUITE_CMD}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    tree = ET.parse(args.durations_file)
    root = tree.getroot()

    tests: list[tuple[str, str, str, float]] = []
    for tc in root.iter("testcase"):
        classname = tc.attrib["classname"]
        name = tc.attrib["name"]
        duration = float(tc.attrib["time"])
        filepath = _file_from_classname(classname)
        tests.append((filepath, classname, name, duration))

    expected = _count_all_tests()
    if expected is not None and len(tests) < expected:
        print(
            f"ERROR: {args.durations_file} contains {len(tests)} tests "
            f"but the full suite has {expected}.\n"
            "The file appears to be from a partial run (e.g. with -m filters).\n"
            f"Re-run the full test suite:\n\n  {FULL_SUITE_CMD}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    total = sum(d for _, _, _, d in tests)
    slow_target = total - args.budget
    tests.sort(key=lambda t: t[3], reverse=True)

    cumulative = 0.0
    slow_tests: list[tuple[str, str, str, float]] = []
    for filepath, classname, name, duration in tests:
        if cumulative >= slow_target:
            break
        cumulative += duration
        slow_tests.append((filepath, classname, name, duration))

    remaining = total - cumulative
    print(f"Total suite time:      {total:.1f}s ({total / 60:.1f}m)")
    print(f"Non-slow budget:       {args.budget:.0f}s ({args.budget / 60:.1f}m)")
    print(f"Slow tests:            {len(slow_tests)} (cumulative {cumulative:.1f}s)")
    print(f"Remaining (non-slow):  {remaining:.1f}s ({remaining / 60:.1f}m)")
    print()
    print(f"{'Duration':>10s}  {'Cumulative':>10s}  {'File':<45s}  Name")
    print("-" * 120)

    running = 0.0
    for filepath, classname, name, duration in slow_tests:
        running += duration
        print(f"{duration:10.1f}s  {running:10.1f}s  {filepath:<45s}  {name}")

    print()
    print("Tests that should be marked @pytest.mark.slow:")
    by_file: dict[str, list[str]] = {}
    for filepath, _, name, _ in slow_tests:
        by_file.setdefault(filepath, []).append(name)
    for filepath in sorted(by_file):
        print(f"\n  {filepath}:")
        for name in by_file[filepath]:
            print(f"    - {name}")


if __name__ == "__main__":
    main()
