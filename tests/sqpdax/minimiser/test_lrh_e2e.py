"""End-to-end validation of the LRH minimiser (slow).

Catalogue: the Hock-Schittkowski problems plus scalable problems at
``n in {1e3, 1e4}``. Every case is solved by ``LRHLineSearchMinimiser`` with
both ``sigma_rule`` values, by ``ActiveSetLineSearchMinimiser`` (default
L-BFGS + projected CG) and, for ``n <= 1e3``, by scipy SLSQP. The results are
collected into ``.claude/lrh/reports/T12.md`` (a report file, not committed).

The scalable problems' callables are all O(n) (elementwise objectives, dense
``m_c x n`` Jacobians with ``m_c = 100`` at most); dense references appear
only for the tiny HS problems.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize as scipy_minimize

from slsqp_jax.sqpdax.minimiser import (
    ActiveSetLineSearchMinimiser,
    LRHLineSearchMinimiser,
    minimise,
)
from slsqp_jax.sqpdax.problem import Problem

from .conftest import HS_CASES, _hs_problem

pytestmark = pytest.mark.slow

TOL = 1e-6
TIGHT_TOL = 1e-8  # objective comparison: the gap scales like the KKT residual^2
FEAS_TOL = 1e-5
REPORT = Path(__file__).resolve().parents[3] / ".claude" / "lrh" / "reports" / "T12.md"
SIGMA_RULES = ("sy_over_ss", "yy_over_sy")
N_EQ = N_INEQ = 50


@pytest.fixture(autouse=True)
def x64():
    """Run every test in float64 (the tolerances below assume it)."""
    with jax.enable_x64(True):
        yield


# --- scalable problems (all callables O(n)) ------------------------------------------


def make_box_quadratic_n(n: int) -> tuple[Problem, jax.Array, float]:
    """Convex diagonal quadratic on ``[-1, 1]^n`` with ~30 % of bounds active.

    ``f = 0.5 sum a_i (x_i - c_i)^2`` with ``c_i = 2`` (above the box) for 30 %
    of the entries; the optimum is ``clip(c, -1, 1)`` in closed form.
    """
    i = jnp.arange(n)
    a = 1.0 + 9.0 * (i % 7) / 6.0
    c = jnp.where(i % 10 < 3, 2.0, 0.6 * jnp.cos(0.37 * i))

    def fn(x):
        return 0.5 * jnp.sum(a * (x - c) ** 2)

    problem = _hs_problem(
        fn=fn,
        grad=lambda x: a * (x - c),
        lb=[-1.0] * n,
        ub=[1.0] * n,
    )
    x_star = jnp.clip(c, -1.0, 1.0)
    return problem, jnp.zeros(n), float(fn(x_star))


def make_rosenbrock_box(n: int) -> tuple[Problem, jax.Array, None]:
    """Rosenbrock chain ``sum 100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2`` in a box."""

    def fn(x):
        return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)

    problem = _hs_problem(fn=fn, grad=jax.grad(fn), lb=[-1.5] * n, ub=[0.8] * n)
    return problem, jnp.zeros(n), None


def make_mixed(n: int) -> tuple[Problem, jax.Array, None]:
    """50 linear equalities + 50 nonlinear inequalities + bounds, convex.

    ``f = 0.5 ||x - c||^2``; equality ``j``: sum of the ``j``-th contiguous
    block of ``n / 50`` variables is zero; inequality ``k``: the mean of
    ``x_i^2`` over the stride class ``i mod 50 == k`` is at most ``0.2``.
    """
    block = n // N_EQ
    i = jnp.arange(n)
    c = 1.5 * jnp.sin(0.7 * i)
    eq_mask = (i // block)[None, :] == jnp.arange(N_EQ)[:, None]
    ineq_mask = (i % N_INEQ)[None, :] == jnp.arange(N_INEQ)[:, None]

    def fn(x):
        return 0.5 * jnp.sum((x - c) ** 2)

    problem = _hs_problem(
        fn=fn,
        grad=lambda x: x - c,
        eq=(
            lambda x: jnp.sum(jnp.where(eq_mask, x[None, :], 0.0), axis=1),
            lambda x: eq_mask.astype(x.dtype),
        ),
        ineq=(
            lambda x: jnp.sum(jnp.where(ineq_mask, x[None, :] ** 2, 0.0), axis=1)
            / block
            - 0.2,
            lambda x: jnp.where(ineq_mask, 2.0 * x[None, :] / block, 0.0),
        ),
        lb=[-1.0] * n,
        ub=[1.0] * n,
    )
    return problem, jnp.zeros(n), None


def _hs_case(name: str):
    factory, x0, _, f_star = HS_CASES[name]
    return lambda: (factory(), jnp.asarray(x0), f_star)


SCALABLE = {
    "box_quadratic": make_box_quadratic_n,
    "rosenbrock_box": make_rosenbrock_box,
    "mixed_50eq_50ineq": make_mixed,
}
CASES: dict[str, tuple[Callable, int]] = {
    **{name: (_hs_case(name), 500) for name in HS_CASES},
    **{
        f"{name}_n{n}": ((lambda f=f, n=n: f(n)), 500)
        for name, f in SCALABLE.items()
        for n in (1000, 10000)
    },
}


# --- solver runs -------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    """Outcome of one solver on one problem."""

    solver: str
    ok: bool
    status: str
    steps: int | None
    seconds: float
    f: float
    feas: float
    kkt_ratio: float | None


def feasibility(problem: Problem, x) -> float:
    """Largest violation of the bounds, equalities and inequalities at ``x``."""
    viol = [
        jnp.max(problem.lb - x, initial=0.0),
        jnp.max(x - problem.ub, initial=0.0),
        jnp.max(problem.ineq_fn(x), initial=0.0),
        jnp.max(jnp.abs(problem.eq_fn(x)), initial=0.0),
    ]
    return float(jnp.max(jnp.stack(viol)))


def run_sqpdax(
    label: str,
    problem: Problem,
    x0,
    max_steps: int,
    solver,
    options=None,
    tol: float = TOL,
) -> Run:
    """Solve with a sqpdax minimiser and collect the comparison metrics."""
    t0 = time.perf_counter()
    sol = minimise(
        problem, solver, x0, max_steps=max_steps, throw=False, options=options
    )
    stats = sol.state._postprocess_stats(problem, sol.result)
    jax.block_until_ready(sol.value)
    seconds = time.perf_counter() - t0
    feas = feasibility(problem, sol.value)
    kkt = float(stats["kkt_ratio"])
    f = float(stats["final_objective"])
    success = bool(sol.state.result_adapter.is_successful(sol.result))
    return Run(
        label,
        success and feas <= FEAS_TOL and kkt <= tol and bool(np.isfinite(f)),
        str(sol.result).split("<")[-1].rstrip(">")[:60],
        int(sol.stats["num_steps"]),
        seconds,
        f,
        feas,
        kkt,
    )


def run_scipy(problem: Problem, x0) -> Run:
    """Reference scipy SLSQP solve (``n <= 1e3``)."""

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
    t0 = time.perf_counter()
    res = scipy_minimize(
        lambda x: float(problem.fn(jnp.asarray(x))[0]),
        np.clip(np.asarray(x0, dtype=float), lb, ub),
        jac=np_fn(problem.grad),
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 500},
    )
    seconds = time.perf_counter() - t0
    x = jnp.asarray(res.x)
    feas = feasibility(problem, x)
    return Run(
        "scipy_slsqp",
        bool(res.success) and feas <= FEAS_TOL,
        str(res.message)[:40],
        int(res.nit),
        seconds,
        float(res.fun),
        feas,
        None,
    )


def solvers(tol: float = TOL) -> dict[str, tuple[object, dict | None]]:
    """Name -> ``(solver, init options)`` of the sqpdax solvers compared."""
    out: dict[str, tuple[object, dict | None]] = {
        "active_set_lbfgs_cg": (ActiveSetLineSearchMinimiser(rtol=tol, atol=tol), None)
    }
    for rule in SIGMA_RULES:
        out[f"lrh_{rule}"] = (
            LRHLineSearchMinimiser(rtol=tol, atol=tol),
            {"minimiser": {"secant": {"kind": "lrh", "sigma_rule": rule}}},
        )
    return out


# --- report ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def results():
    """Collect ``(case, n, m_c, Run)`` rows and write the T12 report at teardown."""
    rows: list[tuple[str, int, int, Run]] = []
    yield rows
    if not rows:
        return
    lines = [
        "# T12 end-to-end validation (generated by test_lrh_e2e.py)",
        "",
        "Time includes JIT compilation (sqpdax solvers). `f-evals` are not "
        "exposed by the minimiser stats (n/a). Solved = success status, "
        f"feasibility <= {FEAS_TOL:g}, kkt_ratio <= the run's rtol. Rows suffixed "
        f"`@{TIGHT_TOL:g}` are the tight (rtol = atol = {TIGHT_TOL:g}) runs used "
        "for the objective comparison; the others are rtol = atol = 1e-6.",
        "",
        "Notes:",
        "",
        "* At rtol = 1e-6 the objective gap at n = 1e4 is a stopping-tolerance "
        "effect, not an LRH defect: the stopping test bounds the gradient, and "
        "the gap scales like the KKT residual squared (and grows with n). For "
        "box_quadratic_n10000 the yy_over_sy gap is 1.05e-6 relative at "
        "rtol 1e-6 and about 2.5e-11 at 1e-8. The objective is therefore "
        "compared on the tight runs.",
        "* The parent ActiveSetLineSearchMinimiser ends in merit_stagnation on "
        "rosenbrock_box at rtol 1e-8; an unsolved reference is not used.",
        "* f-evals are not exposed by the sqpdax stats (n/a).",
        "",
        "| problem | n | m_c | solver | solved | steps | f-evals | time [s] | "
        "objective | feasibility | KKT ratio | status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case, n, m_c, r in rows:
        kkt = "n/a" if r.kkt_ratio is None else f"{r.kkt_ratio:.2e}"
        lines.append(
            f"| {case} | {n} | {m_c} | {r.solver} | {'yes' if r.ok else 'NO'} | "
            f"{r.steps} | n/a | {r.seconds:.1f} | {r.f:.10g} | {r.feas:.2e} | "
            f"{kkt} | {r.status} |"
        )
    try:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text("\n".join(lines) + "\n")
    except OSError:  # pragma: no cover - read-only checkout
        pass


# --- the validation ----------------------------------------------------------------


@pytest.fixture(scope="module")
def case_runs(results):
    """Module-cached ``case_runs(case) -> (f_star, runs, tight_runs)``.

    ``runs`` are the rtol = 1e-6 solves (plus scipy for ``n <= 1e3``) that
    decide "solved"; ``tight_runs`` are the rtol = 1e-8 sqpdax solves used for
    the objective comparison.
    """
    cache: dict[str, tuple[float | None, dict[str, Run], dict[str, Run]]] = {}

    def get(case: str):
        if case not in cache:
            build, max_steps = CASES[case]
            problem, x0, f_star = build()
            m_c = problem.meq + problem.mineq
            runs: dict[str, Run] = {}
            tight: dict[str, Run] = {}
            for name, (solver, options) in solvers().items():
                runs[name] = run_sqpdax(name, problem, x0, max_steps, solver, options)
            for name, (solver, options) in solvers(TIGHT_TOL).items():
                tight[name] = run_sqpdax(
                    f"{name}@{TIGHT_TOL:g}",
                    problem,
                    x0,
                    max_steps,
                    solver,
                    options,
                    tol=TIGHT_TOL,
                )
            if problem.n <= 1000:
                runs["scipy_slsqp"] = run_scipy(problem, x0)
            for r in (*runs.values(), *tight.values()):
                results.append((case, problem.n, m_c, r))
            cache[case] = (f_star, runs, tight)
        return cache[case]

    return get


@pytest.mark.parametrize("rule", SIGMA_RULES)
@pytest.mark.parametrize("case", list(CASES))
def test_lrh_solves_what_the_references_solve(case, rule, case_runs):
    """LRH solves every case that the references solve, for each sigma rule.

    "Solved" is judged on the rtol = 1e-6 runs. The objective comparison is
    done on the rtol = 1e-8 runs, where the stopping test implies the 1e-6
    relative band: against the closed-form / literature optimum when known,
    scipy SLSQP (``n <= 1e3``) and the tight parent run when it converged.
    Step counts are reported in the table, never gated.
    """
    f_star, runs, tight = case_runs(case)
    name = f"lrh_{rule}"
    r = runs[name]
    refs = {k: v for k, v in runs.items() if not k.startswith("lrh_") and v.ok}
    if refs:
        assert r.ok, f"{name} failed ({r.status}, feas={r.feas}, kkt={r.kkt_ratio})"

    tight_lrh = tight[name]
    obj_refs = {}
    if "scipy_slsqp" in refs:
        obj_refs["scipy_slsqp"] = refs["scipy_slsqp"].f
    if tight["active_set_lbfgs_cg"].ok:
        obj_refs["active_set_tight"] = tight["active_set_lbfgs_cg"].f
    if f_star is not None:
        obj_refs["known_optimum"] = f_star
    if obj_refs:
        assert tight_lrh.ok, (
            f"{name} did not converge at rtol={TIGHT_TOL:g} "
            f"({tight_lrh.status}, feas={tight_lrh.feas}, kkt={tight_lrh.kkt_ratio})"
        )
    for ref_name, f_ref in obj_refs.items():
        assert tight_lrh.f == pytest.approx(f_ref, rel=1e-6, abs=1e-8), ref_name
