"""Tests for the pure *solved* criterion and its consumers.

``benchmarks.criteria.solved_reason`` is the scalar rule; ``analysis.add_solved``
is its vectorised twin (the wasm dashboard ships ``analysis.py`` alone) and the
worker applies it with the task's tolerances. They must agree.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import itertools
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

from benchmarks import analysis, criteria
from benchmarks.__main__ import build_parser
from benchmarks.criteria import solved_reason
from benchmarks.worker import TaskSpec, _solved

NAN = float("nan")
FEAS_TOL, F_TOL, STAT_TOL = 1e-6, 1e-4, 1e-4


@pytest.fixture
def solved_row() -> dict:
    """A row that passes every check."""
    return {
        "successful": True,
        "finite": True,
        "feas": 0.0,
        "has_fstar": True,
        "f_gap": 1e-9,
        "stat_scaled": 1e-7,
    }


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, None),
        ({"finite": False}, "non_finite"),
        ({"successful": False}, "unsuccessful"),
        ({"feas": 1e-3}, "infeasible"),
        ({"feas": NAN}, "infeasible"),
        ({"feas": None}, "infeasible"),
        ({"f_gap": 1e-2}, "wrong_objective"),
        ({"f_gap": NAN}, "wrong_objective"),
        ({"f_gap": 1e-2, "has_fstar": False}, None),
        ({"stat_scaled": 0.53}, "non_stationary"),
        ({"stat_scaled": math.inf}, "non_stationary"),
        # PALMER7E: successful, feasible, objective right, nowhere near stationary.
        ({"f_gap": 7.6e-6, "stat_scaled": 0.53}, "non_stationary"),
        # No multipliers -> nothing to evaluate -> benefit of the doubt.
        ({"stat_scaled": NAN}, None),
        ({"stat_scaled": None}, None),
        ({"stat_scaled": None, "feas": 1e-3}, "infeasible"),
        # Boundary equality is inclusive.
        ({"feas": FEAS_TOL}, None),
        ({"f_gap": F_TOL}, None),
        ({"stat_scaled": STAT_TOL}, None),
        ({"stat_scaled": math.nextafter(STAT_TOL, 1.0)}, "non_stationary"),
        # Order of the checks: the first failing one is reported.
        ({"finite": False, "successful": False}, "non_finite"),
        ({"successful": False, "feas": 1.0}, "unsuccessful"),
        ({"feas": 1.0, "f_gap": 1.0, "stat_scaled": 1.0}, "infeasible"),
        ({"f_gap": 1.0, "stat_scaled": 1.0}, "wrong_objective"),
    ],
)
def test_solved_reason(solved_row, overrides, expected):
    row = {**solved_row, **overrides}
    assert solved_reason(row, FEAS_TOL, F_TOL, STAT_TOL) == expected
    assert expected is None or expected in criteria.SOLVED_REASONS


@pytest.mark.parametrize(
    ("missing", "expected"),
    [
        # Absent means "not evaluated": finite, no reference, no multipliers.
        ("finite", None),
        ("has_fstar", None),
        ("stat_scaled", None),
        # Absent success flag, violation or gap cannot be taken as a pass.
        ("successful", "unsuccessful"),
        ("feas", "infeasible"),
        ("f_gap", "wrong_objective"),
    ],
)
def test_missing_fields(solved_row, missing, expected):
    row = {k: v for k, v in solved_row.items() if k != missing}
    assert solved_reason(row, FEAS_TOL, F_TOL, STAT_TOL) == expected


@pytest.mark.parametrize(
    ("tolerances", "expected"),
    [
        ({}, "non_stationary"),
        ({"stat_tol": 1.0}, None),
        ({"stat_tol": 0.53}, None),
        ({"stat_tol": 0.5}, "non_stationary"),
        ({"stat_tol": 1.0, "f_tol": 1e-6}, "wrong_objective"),
        ({"stat_tol": 1.0, "feas_tol": -1.0}, "infeasible"),
    ],
)
def test_tolerances_are_arguments(solved_row, tolerances, expected):
    row = {**solved_row, "f_gap": 7.6e-6, "stat_scaled": 0.53}
    assert solved_reason(row, **tolerances) == expected


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, (True, None)),
        ({"successful": False}, (False, "unsuccessful")),
        ({"stat_scaled": 0.53}, (False, "non_stationary")),
    ],
)
def test_worker_solved_uses_the_spec_tolerances(solved_row, overrides, expected):
    spec = TaskSpec(problem="X", feas_tol=FEAS_TOL, f_tol=F_TOL, stat_tol=STAT_TOL)
    assert _solved({**solved_row, **overrides}, spec) == expected
    loose = dataclasses.replace(spec, stat_tol=1.0)
    assert _solved({**solved_row, "stat_scaled": 0.53}, loose) == (True, None)


@pytest.fixture(scope="module")
def diagnose():
    """``scripts/diagnose_benchmark.py`` imported by path."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "diagnose_benchmark.py"
    spec = importlib.util.spec_from_file_location("diagnose_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up by name
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.mark.parametrize(
    ("overrides", "expected", "expected_loose"),
    [
        ({}, True, True),
        ({"status": "max_steps_reached"}, False, False),  # overrides ``successful``
        ({"feas": "inf"}, False, False),  # diag rows serialise infinities as strings
        ({"f_gap": None}, False, False),
        ({"stat_scaled": 0.53}, False, True),  # passes with stat_tol = 1
        # diag rows serialise NaN as null: no multipliers, passes
        ({"stat_scaled": None}, True, True),
        # the least-squares measure is diagnostic only
        ({"stat_ls": 0.53}, True, True),
    ],
)
def test_diagnose_solved(diagnose, solved_row, overrides, expected, expected_loose):
    row = {**solved_row, "status": "successful", **overrides}
    assert diagnose._solved(row, FEAS_TOL, F_TOL, STAT_TOL) is expected
    assert diagnose._solved(row, FEAS_TOL, F_TOL) is expected  # default 1e-4
    assert diagnose._solved(row, FEAS_TOL, F_TOL, 1.0) is expected_loose


def _grid() -> list[dict]:
    """Every combination of the interesting values of each field."""
    values = {
        "successful": [True, False],
        "finite": [True, False],
        "feas": [0.0, FEAS_TOL, 1e-3, NAN],
        "has_fstar": [True, False],
        "f_gap": [0.0, F_TOL, 1e-2, NAN],
        "stat_scaled": [0.0, STAT_TOL, 0.53, math.inf, NAN],
    }
    return [
        dict(zip(values, combo, strict=True))
        for combo in itertools.product(*values.values())
    ]


# Outcome of a reported success per criteria reason (a non-finite point is a
# feasibility failure for the analysis layer).
_OUTCOME_OF_REASON = {
    None: "solved",
    "non_finite": "claimed_but_infeasible",
    "infeasible": "claimed_but_infeasible",
    "wrong_objective": "claimed_but_wrong",
    "non_stationary": "claimed_but_non_stationary",
}


def test_analysis_agrees_with_the_scalar_criterion():
    rows = _grid()
    frame = pd.DataFrame(rows)
    frame["status"] = frame["successful"].map(
        {True: "successful", False: "solver_failure"}
    )
    scored = analysis.add_solved(
        frame, feas_tol=FEAS_TOL, f_tol=F_TOL, stat_tol=STAT_TOL
    )
    reasons = [solved_reason(r, FEAS_TOL, F_TOL, STAT_TOL) for r in rows]
    assert scored["solved"].tolist() == [reason is None for reason in reasons]
    for outcome, reason, row in zip(scored["outcome"], reasons, rows, strict=True):
        expected = _OUTCOME_OF_REASON[reason] if row["successful"] else "solver_failure"
        assert outcome == expected, (row, reason)


def test_default_tolerances_are_consistent():
    defaults = (1e-6, 1e-4, 1e-4)
    assert defaults == (
        criteria.DEFAULT_FEAS_TOL,
        criteria.DEFAULT_F_TOL,
        criteria.DEFAULT_STAT_TOL,
    )
    assert defaults == (
        analysis.DEFAULT_FEAS_TOL,
        analysis.DEFAULT_F_TOL,
        analysis.DEFAULT_STAT_TOL,
    )
    spec = {f.name: f.default for f in dataclasses.fields(TaskSpec)}
    assert defaults == (spec["feas_tol"], spec["f_tol"], spec["stat_tol"])
    parser = build_parser()
    for argv in (["run"], ["rescore", "in.jsonl", "out.jsonl"]):
        args = parser.parse_args(argv)
        assert defaults == (args.feas_tol, args.f_tol, args.stat_tol)
