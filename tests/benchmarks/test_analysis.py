"""Tests for the pure-pandas ``benchmarks.analysis`` and ``benchmarks.results``."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from benchmarks import analysis, results


def test_add_solved_outcomes(toy_results):
    df = analysis.add_solved(toy_results)
    assert df["outcome"].tolist() == [
        "solved",
        "claimed_but_infeasible",
        "max_steps",
        "claimed_but_wrong",
        "timeout",
        "solved",
    ]
    assert df["solved"].tolist() == [True, False, False, False, False, True]


@pytest.mark.parametrize(
    ("feas_tol", "f_tol", "expected_solved"),
    [(1e-6, 1e-4, 2), (1e-2, 1e-4, 3), (1e-6, 1e-1, 3), (1e-2, 1e-1, 4)],
)
def test_add_solved_tolerances(toy_results, feas_tol, f_tol, expected_solved):
    df = analysis.add_solved(toy_results, feas_tol=feas_tol, f_tol=f_tol)
    assert int(df["solved"].sum()) == expected_solved


def test_add_solved_empty():
    df = analysis.add_solved(pd.DataFrame())
    assert list(df.columns) == ["solved", "outcome"]
    assert df.empty


def test_performance_profile_properties(toy_results):
    df = analysis.add_solved(toy_results)
    prof = analysis.performance_profile(df, metric="time_median_s", n_tau=50)
    assert set(prof.columns) == {"config", "tau", "rho"}
    for _, sub in prof.groupby("config"):
        rho = sub.sort_values("tau")["rho"].to_numpy()
        assert np.all(np.diff(rho) >= 0)  # monotone
        assert 0.0 <= rho.min() and rho.max() <= 1.0
    # s1 solved A (fastest) -> rho(1) = 1/2; s3 solved B (fastest) -> 1/2; s2 solved nothing.
    at_one = prof[prof["tau"] == 1.0].set_index("config")["rho"]
    assert at_one["s1"] == pytest.approx(0.5)
    assert at_one["s3"] == pytest.approx(0.5)
    assert at_one["s2"] == pytest.approx(0.0)
    # right plateau equals the solve rate
    plateau = prof.groupby("config")["rho"].max()
    rate = df.groupby("config")["solved"].mean()
    pd.testing.assert_series_equal(plateau, rate, check_names=False)


def test_summary_and_outcome_tables(toy_results):
    df = analysis.add_solved(toy_results)
    summary = analysis.summary_table(df).set_index("config")
    assert summary.loc["s1", "n_tasks"] == 2
    assert summary.loc["s1", "n_solved"] == 1
    assert summary.loc["s1", "median_time_s"] == pytest.approx(1.0)
    assert summary.loc["s3", "median_steps"] == pytest.approx(20)
    assert np.isnan(summary.loc["s2", "median_time_s"])
    by_coll = analysis.summary_table(df, by=("collection", "config"))
    assert len(by_coll) == 3

    outcomes = analysis.outcome_table(df)
    assert outcomes["count"].sum() == len(df)
    assert (outcomes["count"] > 0).all()


def test_longitudinal(toy_results):
    runs = pd.concat(
        [
            analysis.add_solved(toy_results).assign(
                run_id="r1", timestamp="2026-01-01", sha="a" * 40
            ),
            analysis.add_solved(toy_results).assign(
                run_id="r2", timestamp="2026-02-01", sha="b" * 40
            ),
        ]
    )
    table = analysis.longitudinal(runs)
    assert table["run_id"].tolist() == ["r1"] * 3 + ["r2"] * 3
    assert table["timestamp"].iloc[-1] == "2026-02-01"
    assert set(table.columns) >= {"solve_rate", "sha", "timestamp"}


def test_results_roundtrip(tmp_path):
    path = tmp_path / "results.jsonl"
    results.append_row(
        path, {"problem": "A", "value": np.float64(1.5), "arr": np.arange(2)}
    )
    results.append_row(path, {"problem": "B", "value": 2.0})
    frame = results.read_jsonl(path)
    assert frame["problem"].tolist() == ["A", "B"]
    assert frame["arr"].iloc[0] == [0, 1]

    merged = tmp_path / "merged.jsonl"
    assert results.merge_jsonl([path, path, tmp_path / "missing.jsonl"], merged) == 4
    assert len(results.read_jsonl(merged)) == 4

    run_dir = tmp_path / "run"
    results.write_run_metadata(run_dir, {"sha": "abc", "tier": "tiny"})
    results.merge_jsonl([path], run_dir / results.RESULTS_FILE)
    frame, meta = results.load_run(run_dir)
    assert meta["tier"] == "tiny"
    assert (frame["run_id"] == "run").all()
    assert results.read_jsonl(tmp_path / "nothing.jsonl").empty


def test_new_run_id_sorts_by_time():
    first = results.new_run_id(
        datetime(2026, 1, 1, tzinfo=timezone.utc), "deadbeefcafe"
    )
    later = results.new_run_id(datetime(2026, 1, 2, tzinfo=timezone.utc), "0000000")
    assert first == "20260101T000000Z_deadbee"
    assert first < later


def test_run_metadata_fields():
    meta = results.run_metadata(tier="tiny")
    assert {"sha", "timestamp", "hostname", "python", "trigger", "tier"} <= set(meta)
    json.dumps(meta)  # serialisable
