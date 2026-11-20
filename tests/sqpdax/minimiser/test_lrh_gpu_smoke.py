"""Large-scale smoke tests of the LRH minimiser (``n = 1e5``, slow).

Two checks share one synthetic problem with all ``2n`` bounds finite, ~30 %
of them active at the solution, 5 linear equalities and 5 nonlinear
inequalities (its callables are ``O(n)``: elementwise objective, dense
``m_c x n`` Jacobians):

* a GPU smoke test (skipped without a GPU) that solves it with both
  ``sigma_rule`` values and bounds the device's peak memory;
* a backend-agnostic check that the compiled ``init`` + ``step`` needs
  ``O(n (memory + m_c))`` temporary memory, from XLA's ``memory_analysis``.

The memory budget is ``BUDGET_FACTOR * itemsize * n * (memory + m_c + 8)``
bytes (:func:`memory_budget`). At ``n = 1e5`` in float64 that is 179 MB,
against ``n^2 * itemsize = 80 GB`` for a single dense ``n x n`` array: the
budget is more than two orders of magnitude below it.
"""

from __future__ import annotations

import gc
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax.sqpdax.minimiser import LRHLineSearchMinimiser, minimise
from slsqp_jax.sqpdax.problem import Problem

from .conftest import _hs_problem

N = 100_000
MEQ = 5
MINEQ = 5
M_C = MEQ + MINEQ
MEMORY = 10
# Generous constant: the elastic augmentation, line-search trial points, the
# carried best iterate, bound masks and multiplier vectors are all n-vectors.
BUDGET_FACTOR = 8
BUDGET_CONST = 8
SIGMA_RULES = ("sy_over_ss", "yy_over_sy")
FEAS_TOL = 1e-4


@pytest.fixture(autouse=True)
def x64():
    """Run every test in float64."""
    with jax.enable_x64(True):
        yield


def memory_budget(n: int, itemsize: int, memory: int = MEMORY, m_c: int = M_C) -> int:
    """Byte budget ``C * itemsize * n * (memory + m_c + const)``.

    A single dense ``n x n`` array needs ``n^2 * itemsize`` bytes, i.e. a
    factor ``n / (C (memory + m_c + const))`` (about 450 at ``n = 1e5``) more
    than the whole budget. The compiled-memory check built on it only catches
    super-linear or gross blow-ups (an extra ``n x K`` buffer is flagged only
    for ``K`` of the order of 19 ``memory``); the per-array ``n x r`` and
    stacked-bounds patterns are covered by the C21 jaxpr audit in
    ``test_lrh_memory_budget.py``.
    """
    return BUDGET_FACTOR * itemsize * n * (memory + m_c + BUDGET_CONST)


def make_smoke_problem(n: int = N) -> tuple[Problem, jax.Array]:
    """Box-constrained quadratic with 5 linear equalities, 5 nonlinear inequalities.

    ``f = 0.5 sum a_i (x_i - c_i)^2`` on ``[-1, 1]^n`` with ``c_i = 2`` (above
    the box) for 30 % of the entries, so ~30 % of the upper bounds are active.
    Equality ``j``: ``w_j . x = 0.1`` with ``w_j`` fixed random weights;
    inequality ``k``: ``mean(v_k x^2) <= 0.7`` (mildly active at most).
    """
    rng = np.random.default_rng(0)
    i = jnp.arange(n)
    a = 1.0 + 9.0 * (i % 7) / 6.0
    c = jnp.where(i % 10 < 3, 2.0, 0.6 * jnp.cos(0.37 * i))
    w = jnp.asarray(rng.uniform(-1.0, 1.0, (MEQ, n)) / n)
    v = jnp.asarray(rng.uniform(0.5, 1.5, (MINEQ, n)))

    problem = _hs_problem(
        fn=lambda x: 0.5 * jnp.sum(a * (x - c) ** 2),
        grad=lambda x: a * (x - c),
        eq=(lambda x: w @ x - 0.1, lambda x: w.astype(x.dtype)),
        ineq=(
            lambda x: v @ x**2 / n - 0.7,
            lambda x: 2.0 * v * x[None, :] / n,
        ),
        lb=[-1.0] * n,
        ub=[1.0] * n,
    )
    return problem, jnp.zeros(n)


def _peak_verdict(
    before: Mapping[str, int] | None,
    after: Mapping[str, int] | None,
    budget: int,
) -> tuple[str, str]:
    """Decide ``("skip" | "pass" | "fail", message)`` from device memory stats.

    ``peak_bytes_in_use`` is a process-lifetime high-water mark, so it is
    baselined: the solve's own peak is ``peak_after - bytes_in_use_before``,
    attributable only if the earlier peak ``p0`` did not already exceed
    ``b0 + budget``.
    """
    keys = ("bytes_in_use", "peak_bytes_in_use")
    if not before or not after or any(k not in before for k in keys):
        return "skip", "device exposes no bytes_in_use / peak_bytes_in_use"
    if "peak_bytes_in_use" not in after:
        return "skip", "device exposes no bytes_in_use / peak_bytes_in_use"
    b0, p0 = before["bytes_in_use"], before["peak_bytes_in_use"]
    if p0 > b0 + budget:
        return "skip", (
            f"earlier allocations already peaked at {p0} bytes (> baseline {b0} "
            f"+ budget {budget}); the solve's peak cannot be attributed, run "
            "this test alone"
        )
    used = after["peak_bytes_in_use"] - b0
    msg = f"peak above baseline {used} bytes vs budget {budget} (b0={b0}, p0={p0})"
    return ("pass" if used <= budget else "fail"), msg


def feasibility(problem: Problem, x) -> float:
    """Largest violation of the bounds, equalities and inequalities at ``x``."""
    viol = [
        jnp.max(problem.lb - x, initial=0.0),
        jnp.max(x - problem.ub, initial=0.0),
        jnp.max(problem.ineq_fn(x), initial=0.0),
        jnp.max(jnp.abs(problem.eq_fn(x)), initial=0.0),
    ]
    return float(jnp.max(jnp.stack(viol)))


def lrh_options(rule: str) -> dict:
    """Init options selecting the LRH secant with ``rule`` and the test memory."""
    return {"minimiser": {"secant": {"kind": "lrh", "sigma_rule": rule}}}


@pytest.fixture(scope="module")
def smoke():
    """The ``n = 1e5`` problem and its starting point (built once)."""
    with jax.enable_x64(True):
        return make_smoke_problem()


@pytest.mark.slow
@pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="GPU smoke test: no GPU backend available"
)
@pytest.mark.parametrize("rule", SIGMA_RULES)
def test_gpu_smoke_n1e5_solves_within_the_memory_budget(smoke, rule):
    """``n = 1e5`` solves, is feasible and peaks at ``O(n (memory + m_c))`` bytes."""
    problem, x0 = smoke
    solver = LRHLineSearchMinimiser(secant_memory=MEMORY)
    device = next(iter(x0.devices()))
    # Free cyclic garbage of earlier parametrisations now, so it is not released
    # during the solve (which would understate the solve's own peak).
    gc.collect()
    before = device.memory_stats()
    sol = minimise(
        problem, solver, x0, max_steps=500, throw=False, options=lrh_options(rule)
    )
    jax.block_until_ready(sol.value)
    assert bool(sol.state.result_adapter.is_successful(sol.result))
    assert feasibility(problem, sol.value) <= FEAS_TOL

    budget = memory_budget(N, sol.value.dtype.itemsize)
    verdict, msg = _peak_verdict(before, device.memory_stats(), budget)
    if verdict == "skip":
        pytest.skip(msg)
    assert verdict == "pass", msg
    print(f"[gpu smoke, {rule}] {msg}")


@pytest.mark.parametrize(
    ("before", "after", "verdict"),
    [
        (None, {"peak_bytes_in_use": 5}, "skip"),
        ({"bytes_in_use": 0}, {"peak_bytes_in_use": 5}, "skip"),
        ({"bytes_in_use": 0, "peak_bytes_in_use": 0}, {}, "skip"),
        (
            {"bytes_in_use": 10, "peak_bytes_in_use": 111},
            {"peak_bytes_in_use": 111},
            "skip",
        ),
        (
            {"bytes_in_use": 10, "peak_bytes_in_use": 110},
            {"peak_bytes_in_use": 110},
            "pass",
        ),
        (
            {"bytes_in_use": 10, "peak_bytes_in_use": 50},
            {"peak_bytes_in_use": 110},
            "pass",
        ),
        (
            {"bytes_in_use": 10, "peak_bytes_in_use": 50},
            {"peak_bytes_in_use": 111},
            "fail",
        ),
    ],
    ids=[
        "no-stats",
        "missing-key",
        "missing-after-key",
        "prior-peak-above-window",
        "prior-peak-at-window",
        "pass",
        "fail",
    ],
)
def test_peak_verdict(before, after, verdict):
    """The skip / pass / fail logic of the GPU peak check (budget 100, CPU-only)."""
    assert _peak_verdict(before, after, 100)[0] == verdict


def compiled_memory(problem: Problem, x0, rule: str):
    """XLA ``memory_analysis`` of the jitted ``init`` + one ``step``, or ``None``."""
    solver = LRHLineSearchMinimiser(secant_memory=MEMORY)

    def init_and_step(x):
        return solver.init(problem, x, lrh_options(rule)).step(problem)

    compiled = jax.jit(init_and_step).lower(x0).compile()
    return compiled.memory_analysis()


def total_bytes(analysis) -> int:
    """Temporary + argument + output bytes of a compiled computation."""
    return int(
        analysis.temp_size_in_bytes
        + analysis.argument_size_in_bytes
        + analysis.output_size_in_bytes
    )


@pytest.mark.slow
@pytest.mark.parametrize("rule", SIGMA_RULES)
def test_compiled_step_memory_is_linear_in_n(smoke, rule):
    """The compiled ``init`` + ``step`` fits ``C itemsize n (memory + m_c + const)``."""
    problem, x0 = smoke
    analysis = compiled_memory(problem, x0, rule)
    if analysis is None:
        pytest.skip("memory_analysis() is not available on this backend")
    budget = memory_budget(N, x0.dtype.itemsize)
    assert total_bytes(analysis) <= budget, (analysis, budget)


@pytest.mark.slow
def test_memory_budget_flags_a_quadratic_allocation():
    """Negative control: an ``n x n`` allocation inside a loop body is counted."""

    def dense(x):
        def body(carry):
            i, v = carry
            return i + 1, jnp.outer(v, v) @ v / jnp.sum(v**2)

        return jax.lax.while_loop(lambda c: c[0] < 3, body, (0, x))[1]

    n = 2048
    x = jnp.ones(n)
    analysis = jax.jit(dense).lower(x).compile().memory_analysis()
    if analysis is None:
        pytest.skip("memory_analysis() is not available on this backend")
    assert total_bytes(analysis) > memory_budget(n, x.dtype.itemsize)
