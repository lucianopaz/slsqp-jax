"""LRH line-search minimiser on the Hock-Schittkowski problems (HS21/35/71/76)."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize as scipy_minimize

from slsqp_jax.sqpdax.minimiser import LRHLineSearchMinimiser, minimise
from slsqp_jax.sqpdax.minimiser.active_set_linesearch import (
    ACTIVE_SET_LINE_SEARCH_RESULTS,
)
from slsqp_jax.sqpdax.problem import Problem

from .conftest import HS_CASES

TOL = 1e-6


@pytest.fixture(autouse=True)
def x64():
    """Run every test in float64 (the tolerances below assume it)."""
    with jax.enable_x64(True):
        yield


def _solve(problem: Problem, x0, *, max_steps: int = 200, **kwargs):
    kwargs.setdefault("rtol", TOL)
    kwargs.setdefault("atol", TOL)
    sol = minimise(
        problem,
        LRHLineSearchMinimiser(**kwargs),
        jnp.asarray(x0),
        max_steps=max_steps,
        throw=False,
    )
    return sol, sol.state._postprocess_stats(problem, sol.result)


def _scipy_slsqp(problem: Problem, x0) -> tuple[float, np.ndarray]:
    """Reference SLSQP solve (tiny n, so dense NumPy callables are fine)."""

    def np_fn(fn):
        return lambda x: np.asarray(fn(jnp.asarray(x)))

    constraints = []
    if problem.meq:
        constraints.append(
            {
                "type": "eq",
                "fun": np_fn(problem.eq_fn),
                "jac": np_fn(problem.eq_fn_jac),
            }
        )
    if problem.mineq:
        # sqpdax uses ``h(x) <= 0``; scipy uses ``g(x) >= 0``.
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x: -np_fn(problem.ineq_fn)(x),
                "jac": lambda x: -np_fn(problem.ineq_fn_jac)(x),
            }
        )
    lb, ub = np.asarray(problem.lb), np.asarray(problem.ub)
    bounds = [
        (None if np.isinf(lo) else lo, None if np.isinf(hi) else hi)
        for lo, hi in zip(lb, ub)
    ]
    res = scipy_minimize(
        lambda x: float(problem.fn(jnp.asarray(x))[0]),
        np.clip(np.asarray(x0, dtype=float), lb, ub),
        jac=np_fn(problem.grad),
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 500},
    )
    return float(res.fun), res.x


@pytest.fixture(scope="module")
def scipy_reference() -> Callable[[str], tuple[float, np.ndarray]]:
    """Module-cached ``scipy_reference(case) -> (f, x)`` of SLSQP."""
    cache: dict[str, tuple[float, np.ndarray]] = {}

    def get(case: str) -> tuple[float, np.ndarray]:
        if case not in cache:
            with jax.enable_x64(True):
                factory, x0, _, _ = HS_CASES[case]
                cache[case] = _scipy_slsqp(factory(), x0)
        return cache[case]

    return get


@pytest.mark.parametrize("case", list(HS_CASES))
def test_known_optimum_matches_scipy(case, scipy_reference):
    """The literature optimum used below agrees with scipy SLSQP (builder check)."""
    _, _, x_star, f_star = HS_CASES[case]
    f_scipy, x_scipy = scipy_reference(case)
    assert f_scipy == pytest.approx(f_star, rel=1e-6)
    assert np.allclose(x_scipy, x_star, atol=1e-4)


@pytest.mark.parametrize("secant_memory", [2, 5, 10], ids=lambda m: f"m{m}")
@pytest.mark.parametrize("qp_warm_start", [False, True], ids=["cold", "warm"])
@pytest.mark.parametrize("case", list(HS_CASES))
def test_hs_problems_solve_and_match_references(
    case, qp_warm_start, secant_memory, scipy_reference
):
    """Success status, KKT <= tol, feasibility, and the optimum of SLSQP / literature."""
    factory, x0, x_star, f_star = HS_CASES[case]
    problem = factory()
    sol, stats = _solve(
        problem, x0, qp_warm_start=qp_warm_start, secant_memory=secant_memory
    )
    assert bool(sol.state.result_adapter.is_successful(sol.result)), sol.result
    assert bool(jnp.all(jnp.isfinite(sol.value)))
    assert float(stats["kkt_ratio"]) <= TOL

    x = sol.value
    assert bool(jnp.all(x >= problem.lb - 1e-6))
    assert bool(jnp.all(x <= problem.ub + 1e-6))
    assert float(jnp.max(problem.ineq_fn(x), initial=-jnp.inf)) <= 1e-6
    assert float(jnp.max(jnp.abs(problem.eq_fn(x)), initial=0.0)) <= 1e-6

    f = float(stats["final_objective"])
    f_scipy, _ = scipy_reference(case)
    assert f == pytest.approx(f_scipy, rel=1e-6)
    assert f == pytest.approx(f_star, rel=1e-6)
    assert np.allclose(np.asarray(x), x_star, atol=1e-3)


# Outcome ``(status, secant resets recorded)`` of each ``(case, secant_memory)``
# with a one-trial line-search budget. A run fails loudly (resets, then the
# recovery-failure status) here, when some step needs a backtrack: HS21 always
# does, and HS35 does with the shortest memory (m = 2). Otherwise every step is
# accepted at ``alpha = 1``, no backtrack is needed and the run converges with
# no reset at all (HS71, HS76, HS35 with m >= 5).
# The HS35 memory-2 cell is a characterisation, not a claim that failing is
# desirable: with memory < n the complement ``sigma = s'y / s's`` underestimates
# the tangent curvature (about 6 versus ``sigma <= 1`` for every reset target),
# so ``alpha = 1`` is rejected and resets cannot stiffen the model. The cell may
# flip to ``_CONVERGES`` if the ``sigma`` rule changes; it should then be
# updated, not defended.
_FAILS = (ACTIVE_SET_LINE_SEARCH_RESULTS.secant_recovery_failure, True)
_CONVERGES = (ACTIVE_SET_LINE_SEARCH_RESULTS.successful, False)
TINY_BUDGET_OUTCOMES = {
    ("hs21", 2): _FAILS,
    ("hs21", 5): _FAILS,
    ("hs21", 10): _FAILS,
    ("hs35", 2): _FAILS,
    ("hs35", 5): _CONVERGES,
    ("hs35", 10): _CONVERGES,
    ("hs71", 2): _CONVERGES,
    ("hs71", 5): _CONVERGES,
    ("hs71", 10): _CONVERGES,
    ("hs76", 2): _CONVERGES,
    ("hs76", 5): _CONVERGES,
    ("hs76", 10): _CONVERGES,
}


@pytest.mark.parametrize("qp_warm_start", [False, True], ids=["cold", "warm"])
@pytest.mark.parametrize(
    "case, secant_memory",
    list(TINY_BUDGET_OUTCOMES),
    ids=[f"{c}-m{m}" for c, m in TINY_BUDGET_OUTCOMES],
)
def test_tiny_line_search_budget_outcome(case, secant_memory, qp_warm_start):
    """With a one-trial line-search budget each case ends as recorded.

    The iterate is always finite and inside the box and the run takes at most
    60 steps. The status and whether secant resets were recorded are the
    outcome of ``TINY_BUDGET_OUTCOMES``: a run that needs no backtrack still
    converges without a reset, one that does fails loudly, never silently.
    """
    factory, x0, _, _ = HS_CASES[case]
    expected_result, expect_resets = TINY_BUDGET_OUTCOMES[case, secant_memory]
    problem = factory()
    sol, stats = _solve(
        problem,
        x0,
        max_steps=60,
        line_search_max_steps=1,
        qp_warm_start=qp_warm_start,
        secant_memory=secant_memory,
    )
    assert bool(jnp.all(jnp.isfinite(sol.value)))
    assert bool(jnp.isfinite(stats["final_objective"]))
    assert bool(jnp.all(sol.value >= problem.lb - 1e-8))
    assert bool(jnp.all(sol.value <= problem.ub + 1e-8))
    assert int(sol.stats["num_steps"]) <= 60
    assert bool(sol.result == expected_result), sol.result
    n_resets = int(jnp.sum(sol.state.secant_stats.n_resets))
    assert (n_resets > 0) == expect_resets, n_resets


@pytest.mark.parametrize("secant_memory", [2, 5, 10], ids=lambda m: f"m{m}")
@pytest.mark.parametrize("qp_warm_start", [False, True], ids=["cold", "warm"])
def test_forced_line_search_failures_trigger_secant_resets(
    qp_warm_start, secant_memory
):
    """HS21's first step needs a backtrack; with one trial every step fails.

    The failures walk the recovery schedule (one reset per channel, then the
    identity model) and end with the recovery-failure status, never NaN, and
    without moving off the projected start (``f = -98.96``). The resets cannot
    help: the identity-reset target equals the initial identity model, and the
    rejected steps add no secant pairs, so the next trial fails the same way.
    """
    factory, x0, _, _ = HS_CASES["hs21"]
    problem = factory()
    sol, stats = _solve(
        problem,
        x0,
        max_steps=60,
        line_search_max_steps=1,
        qp_warm_start=qp_warm_start,
        secant_memory=secant_memory,
    )
    assert bool(sol.result == ACTIVE_SET_LINE_SEARCH_RESULTS.secant_recovery_failure)
    assert bool(jnp.all(sol.state.secant_stats.n_resets >= 1))
    assert int(sol.state.consecutive_ls_failures) >= 1
    assert float(stats["final_objective"]) == pytest.approx(-98.96, rel=1e-9)
    assert bool(jnp.all(jnp.isfinite(sol.value)))
