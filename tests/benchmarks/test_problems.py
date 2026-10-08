"""Tests for the sif2jax -> sqpdax adapter and the end-to-end worker (slow: imports sif2jax)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from benchmarks.problems import collection_of, constraint_sizes, to_sqpdax
from benchmarks.worker import TaskSpec, run_task_inprocess, run_tasks

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("HS71", dict(n=4, meq=1, mineq=1, nbounds=8, collection="constrained")),
        ("HS35", dict(n=3, meq=0, mineq=1, nbounds=3, collection="constrained")),
        ("HS3", dict(n=2, meq=0, mineq=0, nbounds=1, collection="bounded")),
    ],
)
def test_to_sqpdax_metadata(sif2jax, name, expected):
    problem = sif2jax.cutest.get_problem(name)
    sq, x0, meta = to_sqpdax(problem)
    assert meta.name == name
    for key, value in expected.items():
        assert getattr(meta, key) == value, key
    assert x0.shape == (meta.n,)
    assert (sq.meq, sq.mineq) == (meta.meq, meta.mineq)
    assert collection_of(problem) == expected["collection"]
    assert constraint_sizes(problem, x0) == (meta.meq, meta.mineq)


def test_to_sqpdax_sign_conventions(sif2jax):
    """sif2jax ``ineq >= 0`` must become sqpdax ``h <= 0``; bounds and f pass through."""
    problem = sif2jax.cutest.get_problem("HS71")
    sq, x0, _ = to_sqpdax(problem)
    eq, ineq = problem.constraint(x0)
    assert jnp.allclose(sq.eq_fn(x0), jnp.ravel(eq))
    assert jnp.allclose(sq.ineq_fn(x0), -jnp.ravel(ineq))
    assert jnp.allclose(sq.fn(x0)[0], problem.objective(x0, problem.args))
    lb, ub = problem.bounds
    assert jnp.allclose(sq.lb, lb) and jnp.allclose(sq.ub, ub)
    # exact curvature was forced on, so HVPs exist even for linear pieces
    v = jnp.ones_like(x0)
    assert sq.has_exact_curvature
    assert sq.ineq_fn_hvp is not None
    assert jnp.all(jnp.isfinite(sq.ineq_fn_hvp(x0, v)))


@pytest.mark.parametrize("config", ["pasls", "tfip"])
def test_run_task_inprocess_solves_hs71(sif2jax, config):
    jax.config.update("jax_enable_x64", True)
    spec = TaskSpec(problem="HS71", config=config, repeats=2, timeout_s=120.0)
    row = run_task_inprocess(spec)
    assert row["phase"] == "done"
    assert row["status"] == "successful" and row["successful"]
    assert row["solved"] and row["repeats_skipped"] is None
    assert row["repeats"] == 2 and row["repeats_requested"] == 2
    assert row["feas"] < 1e-5 and row["f_gap"] < 1e-5
    assert row["compile_s"] > 0 and row["time_median_s"] > 0
    assert "times_s" not in row
    assert any(k.startswith("stats_") for k in row)


def test_run_task_inprocess_skips_repeats_when_unsolved(sif2jax):
    """An impossibly tight objective tolerance makes every solve 'wrong' -> pilot only."""
    jax.config.update("jax_enable_x64", True)
    spec = TaskSpec(
        problem="HS71", config="pasls", repeats=5, timeout_s=120.0, f_tol=0.0
    )
    row = run_task_inprocess(spec)
    assert row["status"] == "successful"
    assert not row["solved"] and row["repeats_skipped"] == "wrong_objective"
    assert row["repeats"] == 1 and row["repeats_requested"] == 1
    forced = run_task_inprocess(TaskSpec(**{**spec.__dict__, "repeat_unsolved": True}))
    assert forced["repeats"] == 5


def test_worker_pool_runs_and_reports_errors():
    specs = [
        TaskSpec(problem="HS35", config="pasls", repeats=1, timeout_s=120.0),
        TaskSpec(problem="DOESNOTEXIST", config="pasls", repeats=1, timeout_s=120.0),
    ]
    rows = {r["problem"]: r for r in run_tasks(specs, jobs=1)}
    assert rows["HS35"]["status"] == "successful"
    assert rows["DOESNOTEXIST"]["status"] == "compile_error"
    assert "error" in rows["DOESNOTEXIST"]
