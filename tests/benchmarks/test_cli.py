"""Tests for the ``python -m benchmarks`` command line (the light commands)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.__main__ import build_parser, main

NAN = float("nan")
# A line without ``problem`` (e.g. a metadata record): must be copied verbatim.
NO_PROBLEM_LINE = '{"run":  "meta",   "n_tasks": 5}'


def _row(problem: str, config: str, **fields) -> dict:
    row = {
        "problem": problem,
        "y0_iD": 0,
        "config": config,
        "status": "successful",
        "successful": True,
        "finite": True,
        "feas": 0.0,
        "has_fstar": True,
        "f_gap": 1e-9,
        "stat_scaled": 1e-8,
        "solved": True,
        "repeats_skipped": None,
        "feas_tol": 1e-6,
        "f_tol": 1e-4,
        "steps": 7,
    }
    row.update(fields)
    return row


@pytest.fixture
def results_file(tmp_path: Path) -> Path:
    """One line per scenario, serialised like ``results.append_row`` (``NaN`` kept)."""
    rows = [
        _row("A", "c1"),  # stationary success
        _row("PALMER7E", "c1", f_gap=7.6e-6, stat_scaled=0.53),  # old rule: solved
        _row("A", "c2", stat_scaled=NAN),  # no multipliers
        _row("B", "c2", status="max_steps_reached", successful=False, solved=False),
        _row("B", "c3", status="timeout", successful=None, feas=None, solved=False),
    ]
    lines = [json.dumps(row) for row in rows]
    lines.insert(2, NO_PROBLEM_LINE)
    lines.insert(4, "")  # blank lines are dropped
    path = tmp_path / "in.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


def _counts(stdout: str, configs: list[str]) -> dict[str, list[int]]:
    """Parse the per-config ``tasks before after delta`` table."""
    table = {}
    for line in stdout.splitlines():
        words = line.split()
        if words and words[0] in configs:
            table[words[0]] = [int(v) for v in words[1:]]
    return table


@pytest.mark.parametrize(
    ("tolerances", "solved_after", "reason_of_palmer", "table"),
    [
        # defaults: the PALMER7E-like row (right objective, stat_scaled 0.53) is lost
        (
            {},
            [True, False, True, False, False],
            "non_stationary",
            {"c1": [2, 2, 1, -1], "c2": [2, 1, 1, 0], "c3": [1, 0, 0, 0]},
        ),
        (
            {"stat_tol": 1.0},
            [True, True, True, False, False],
            None,
            {"c1": [2, 2, 2, 0], "c2": [2, 1, 1, 0], "c3": [1, 0, 0, 0]},
        ),
        # stat_scaled = 1e-8 fails a tighter tolerance; NaN still passes
        (
            {"stat_tol": 1e-9},
            [False, False, True, False, False],
            "non_stationary",
            {"c1": [2, 2, 0, -2], "c2": [2, 1, 1, 0], "c3": [1, 0, 0, 0]},
        ),
        (
            {"f_tol": 1e-6, "stat_tol": 1.0},
            [True, False, True, False, False],
            "wrong_objective",
            {"c1": [2, 2, 1, -1], "c2": [2, 1, 1, 0], "c3": [1, 0, 0, 0]},
        ),
        (
            {"feas_tol": -1.0},
            [False, False, False, False, False],
            "infeasible",
            {"c1": [2, 2, 0, -2], "c2": [2, 1, 0, -1], "c3": [1, 0, 0, 0]},
        ),
    ],
)
def test_rescore(
    results_file, tmp_path, capsys, tolerances, solved_after, reason_of_palmer, table
):
    out = tmp_path / "nested" / "out.jsonl"
    argv = [
        arg
        for key, value in tolerances.items()
        for arg in (f"--{key.replace('_', '-')}", str(value))
    ]
    assert main(["rescore", str(results_file), str(out), *argv]) == 0

    old_lines = [ln for ln in results_file.read_text().splitlines() if ln]
    new_lines = out.read_text().splitlines()
    assert len(new_lines) == len(old_lines)
    assert new_lines[2] == NO_PROBLEM_LINE  # untouched, byte for byte

    old = [json.loads(ln) for i, ln in enumerate(old_lines) if i != 2]
    new = [json.loads(ln) for i, ln in enumerate(new_lines) if i != 2]
    tol = {"feas_tol": 1e-6, "f_tol": 1e-4, "stat_tol": 1e-4, **tolerances}
    updated = {"solved", "solved_reason", *tol}
    for before, after, solved in zip(old, new, solved_after, strict=True):
        assert after["solved"] is solved
        assert (after["solved_reason"] is None) is solved
        assert {k: after[k] for k in tol} == tol
        # everything else, including key order, is untouched
        assert json.dumps({k: v for k, v in before.items() if k not in updated}) == (
            json.dumps({k: v for k, v in after.items() if k not in updated})
        )
        assert list(after)[: len(before)] == list(before)
    assert new[1]["solved_reason"] == reason_of_palmer
    assert new[2]["stat_scaled"] != new[2]["stat_scaled"]  # NaN survives the round trip

    stdout = capsys.readouterr().out
    assert _counts(stdout, ["c1", "c2", "c3"]) == table


def test_rescore_refuses_to_overwrite_its_input(results_file, capsys):
    before = results_file.read_text()
    assert main(["rescore", str(results_file), str(results_file)]) == 2
    assert results_file.read_text() == before
    assert "must differ" in capsys.readouterr().err


def test_run_parser_stat_tol():
    parser = build_parser()
    assert parser.parse_args(["run", "--stat-tol", "1e-3"]).stat_tol == 1e-3
    assert parser.parse_args(["rescore", "a", "b", "--stat-tol", "0.5"]).stat_tol == 0.5
