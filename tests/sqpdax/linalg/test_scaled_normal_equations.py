"""Unit tests for :mod:`slsqp_jax.sqpdax.linalg.scaled_normal_equations`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from slsqp_jax.sqpdax.linalg import (
    SchurNormalEquations,
    null_space_projector,
    pcg,
    resolve_normal_equations_strategy,
    slack_eliminated_normal_solver,
    strategy_code,
)
from tests.sqpdax.lagrangian.conftest import make_problem
from tests.sqpdax.subproblem.conftest import (
    FUNNEL_PROBLEMS,
    dense_funnel_reference,
    make_funnel_barrier_subproblem,
)

PROBLEMS = dict(FUNNEL_PROBLEMS) | {
    "no-eq": lambda: make_problem(meq=0),
    "no-ineq": lambda: make_problem(mineq=0),
    "bounds-only": lambda: make_problem(meq=0, mineq=0),
}
problem_ids = pytest.mark.parametrize("problem_name", list(PROBLEMS))
strategy_ids = pytest.mark.parametrize("strategy", ["schur", "matrix-free"])
# Machine-precision claim for the explicit factor, Krylov tolerance otherwise.
TOL = {"schur": 1e-10, "matrix-free": 1e-7}


@pytest.fixture(autouse=True)
def _x64():
    with jax.enable_x64(True):
        yield


def dense_operator(
    jac_eq: Array,
    jac_ineq: Array,
    s: Array,
    s_lb: Array,
    s_ub: Array,
    live_lb: Array,
    live_ub: Array,
    free_x: Array | None = None,
) -> Array:
    """Dense ``Â`` (rows ``[eq | ineq | lb | ub]``, columns ``[x | s | s_lb | s_ub]``)."""
    meq, n = jac_eq.shape
    mineq = jac_ineq.shape[0]
    A = jnp.zeros((meq + mineq + 2 * n, 3 * n + mineq))
    A = A.at[:meq, :n].set(jac_eq)
    A = A.at[meq : meq + mineq, :n].set(jac_ineq)
    A = A.at[meq : meq + mineq, n : n + mineq].set(jnp.diag(s))
    r0 = meq + mineq
    A = A.at[r0 : r0 + n, :n].set(-jnp.diag(live_lb.astype(float)))
    A = A.at[r0 : r0 + n, n + mineq : n + mineq + n].set(
        jnp.diag(jnp.where(live_lb, s_lb, 0.0))
    )
    A = A.at[r0 + n :, :n].set(jnp.diag(live_ub.astype(float)))
    A = A.at[r0 + n :, n + mineq + n :].set(jnp.diag(jnp.where(live_ub, s_ub, 0.0)))
    if free_x is not None:
        A = A.at[:, :n].multiply(free_x.astype(float))
    return A


def pieces(sub):
    """Jacobians, slacks and live masks of a subproblem."""
    lag = sub.lagrangian
    return (
        lag.eq_fn_jac_val,
        lag.ineq_fn_jac_val,
        lag.slack.s,
        lag.slack.s_lb,
        lag.slack.s_ub,
        ~lag.null_lb,
        ~lag.null_ub,
    )


def structured_solver(strategy, *args, **kwargs):
    """Build ``solve(rhs)`` of either structured path from raw pieces."""
    jac_eq, jac_ineq, s, s_lb, s_ub, live_lb, live_ub = args
    if strategy == "schur":
        return SchurNormalEquations.build(
            jac_eq, jac_ineq, s, s_lb, s_ub, live_lb=live_lb, live_ub=live_ub, **kwargs
        ).solve
    return slack_eliminated_normal_solver(
        lambda v: jac_eq @ v,
        lambda y: jac_eq.T @ y,
        lambda v: jac_ineq @ v,
        lambda y: jac_ineq.T @ y,
        s,
        s_lb,
        s_ub,
        live_lb=live_lb,
        live_ub=live_ub,
        meq=jac_eq.shape[0],
        **kwargs,
    )


def pinv_solution(A: Array, b: Array) -> Array:
    return jnp.linalg.pinv(A @ A.T, rtol=1e-12) @ b


# --------------------------------------------------------------------------- #
# agreement with the dense pseudo-inverse
# --------------------------------------------------------------------------- #


@problem_ids
@strategy_ids
def test_structured_solves_match_dense_pseudo_inverse(problem_name, strategy):
    """Both eliminations return ``pinv(Â Âᵀ) b`` for general and in-range ``b``."""
    sub = make_funnel_barrier_subproblem(problem=PROBLEMS[problem_name]())
    args = pieces(sub)
    A = dense_operator(*args)
    _, _, A_ref, _ = dense_funnel_reference(sub)
    assert jnp.allclose(A, A_ref)  # the dense helper mirrors the operator
    solve = sub.normal_equations_solver(strategy)
    assert jnp.allclose(solve(jnp.zeros(A.shape[0])), 0.0)
    for seed in range(3):
        b = jax.random.normal(jax.random.key(seed), (A.shape[0],))
        b_range = A @ jax.random.normal(jax.random.key(10 + seed), (A.shape[1],))
        for rhs in (b, b_range):
            y = solve(rhs)
            assert jnp.allclose(y, pinv_solution(A, rhs), atol=TOL[strategy])
    # Dead bound rows carry exact zeros.
    meq, mineq = sub.lagrangian.meq, sub.lagrangian.mineq
    y = solve(jnp.ones(A.shape[0]))
    dead = jnp.concatenate(
        [jnp.zeros(meq + mineq, bool), sub.lagrangian.null_lb, sub.lagrangian.null_ub]
    )
    assert jnp.all(y[dead] == 0.0)


@strategy_ids
def test_rank_deficient_rows_use_the_pseudo_inverse(strategy):
    """A duplicated equality row is handled by the pseudo-inverse, with no drift."""
    key = jax.random.key(4)
    n, mineq = 5, 2
    jac_eq = jax.random.normal(key, (2, n))
    jac_eq_dup = jnp.concatenate([jac_eq, jac_eq[:1]], axis=0)
    jac_ineq = jax.random.normal(jax.random.fold_in(key, 1), (mineq, n))
    s = jnp.array([0.5, 2.0])
    s_lb = jnp.linspace(0.3, 1.5, n)
    s_ub = jnp.linspace(1.0, 0.4, n)
    live_lb = jnp.array([True, True, False, True, False])
    live_ub = jnp.array([True, False, True, True, True])
    base = (jac_eq, jac_ineq, s, s_lb, s_ub, live_lb, live_ub)
    dup = (jac_eq_dup, jac_ineq, s, s_lb, s_ub, live_lb, live_ub)
    A, A_dup = dense_operator(*base), dense_operator(*dup)
    solve, solve_dup = (
        structured_solver(strategy, *base),
        structured_solver(strategy, *dup),
    )
    b = jax.random.normal(jax.random.fold_in(key, 2), (A.shape[0],))
    b_dup = jnp.concatenate([b[:2], b[:1], b[2:]])

    y_dup = solve_dup(b_dup)
    assert jnp.all(jnp.isfinite(y_dup))
    assert jnp.allclose(y_dup, pinv_solution(A_dup, b_dup), atol=TOL[strategy])
    # Removing the duplicate changes nothing but the split of its multiplier.
    y = solve(b)
    assert jnp.allclose(A_dup.T @ y_dup, A.T @ y, atol=TOL[strategy])
    assert jnp.allclose(y_dup[0] + y_dup[2], y[0], atol=TOL[strategy])
    assert jnp.allclose(y_dup[0], y_dup[2], atol=TOL[strategy])
    assert jnp.allclose(y_dup[3:], y[2:], atol=TOL[strategy])
    if strategy == "schur":
        factor = SchurNormalEquations.build(*dup[:5], live_lb=live_lb, live_ub=live_ub)
        schur = jnp.linalg.matrix_rank(A_dup @ A_dup.T) - int(
            jnp.sum(live_lb) + jnp.sum(live_ub)
        )
        assert int(factor.rank) == schur == 4


@strategy_ids
@pytest.mark.parametrize(
    ("s_lb0", "s_ub0", "s0"),
    [
        pytest.param(1e-8, 0.5, 1.0, id="lower-active"),
        pytest.param(0.3, 1e-9, 1e-6, id="upper-active-tiny-general-slack"),
        pytest.param(1e-8, 0.5, 1e-8, id="lower-and-general-active"),
        pytest.param(1e-7, 1e-7, 1.0, id="both-bounds-active"),
    ],
)
def test_nearly_active_slacks_are_solved_to_roundoff(strategy, s_lb0, s_ub0, s0):
    """Tiny slacks (``1/s² ~ 1e16``) do not amplify rounding: the residual of
    a consistent system stays at roundoff level, as for the dense solve."""
    key = jax.random.key(7)
    n = 4
    jac_eq = jax.random.normal(key, (1, n))
    jac_ineq = jax.random.normal(jax.random.fold_in(key, 1), (2, n))
    s = jnp.array([s0, 0.3])
    s_lb = jnp.array([s_lb0, 0.5, 1.0, 2.0])
    s_ub = jnp.array([s_ub0, 1.0, 0.5, 1.0])
    live = jnp.ones(n, bool)
    args = (jac_eq, jac_ineq, s, s_lb, s_ub, live, live)
    A = dense_operator(*args)
    solve = structured_solver(strategy, *args)
    b = A @ jax.random.normal(jax.random.fold_in(key, 2), (A.shape[1],))
    y = solve(b)
    assert jnp.all(jnp.isfinite(y))
    assert jnp.linalg.norm(A @ (A.T @ y) - b) <= 1e-10 * jnp.linalg.norm(b)


@strategy_ids
def test_masks_match_the_restricted_operator(strategy):
    """Frozen ``x`` columns / dead slacks reproduce ``pinv(A_free A_freeᵀ)``."""
    key = jax.random.key(11)
    n, mineq = 5, 2
    jac_eq = jax.random.normal(key, (1, n))
    jac_ineq = jax.random.normal(jax.random.fold_in(key, 1), (mineq, n))
    s = jnp.array([0.7, 1.3])
    s_lb = jnp.linspace(0.2, 1.0, n)
    s_ub = jnp.linspace(0.9, 0.3, n)
    free_x = jnp.array([True, False, True, True, False])
    live_lb = jnp.array([True, False, True, False, True])  # x_1 frozen + slack frozen
    live_ub = jnp.array([True, True, False, True, True])  # x_4 frozen, slack live
    args = (jac_eq, jac_ineq, s, s_lb, s_ub, live_lb, live_ub)
    A = dense_operator(*args, free_x=free_x)
    solve = structured_solver(strategy, *args, free_x=free_x)
    for seed in range(2):
        b = jax.random.normal(jax.random.fold_in(key, 20 + seed), (A.shape[0],))
        assert jnp.allclose(solve(b), pinv_solution(A, b), atol=TOL[strategy])


# --------------------------------------------------------------------------- #
# projector and iteration structure
# --------------------------------------------------------------------------- #


@problem_ids
@strategy_ids
def test_projector_with_structured_solve_is_exact_and_idempotent(
    problem_name, strategy
):
    """``Â proj(v) ≈ 0`` and ``proj∘proj = proj`` with the external solve."""
    sub = make_funnel_barrier_subproblem(problem=PROBLEMS[problem_name]())
    _, _, A, _ = dense_funnel_reference(sub)
    solve = sub.normal_equations_solver(strategy)
    proj = null_space_projector(lambda v: A @ v, lambda y: A.T @ y, solve=solve)
    scale = jnp.linalg.norm(A, 2)
    for seed in range(3):
        v = jax.random.normal(jax.random.key(seed), (A.shape[1],))
        pv = proj(v)
        assert jnp.linalg.norm(A @ pv) <= TOL[strategy] * scale * jnp.linalg.norm(v)
        assert jnp.allclose(proj(pv), pv, atol=TOL[strategy])
        # Orthogonal projection: v − pv ∈ range(Âᵀ).
        assert jnp.allclose(pv, v - A.T @ pinv_solution(A, A @ v), atol=TOL[strategy])


@problem_ids
def test_matrix_free_iteration_count_is_bounded_by_the_general_rows(problem_name):
    """CG on the ``m × m`` Schur complement finishes in at most ``m`` steps."""
    sub = make_funnel_barrier_subproblem(problem=PROBLEMS[problem_name]())
    lag = sub.lagrangian
    solve = sub.normal_equations_solver("matrix-free")
    b = jax.random.normal(jax.random.key(0), (lag.meq + lag.mineq + 2 * lag.n,))
    _, n_iter = solve(b, with_info=True)
    assert int(n_iter) <= lag.meq + lag.mineq


def test_cost_does_not_grow_with_the_number_of_bounds():
    """Bound-heavy problems (``m_E = 1``, ``m_I = 2``): iteration counts and the
    explicit solve's operation graph do not depend on ``n``."""
    eqn_counts = []
    for n in (50, 200):
        key = jax.random.key(n)
        jac_eq = jax.random.normal(key, (1, n))
        jac_ineq = jax.random.normal(jax.random.fold_in(key, 1), (2, n))
        s = jnp.array([0.4, 1.6])
        s_lb = jax.random.uniform(
            jax.random.fold_in(key, 2), (n,), minval=0.1, maxval=2.0
        )
        s_ub = jax.random.uniform(
            jax.random.fold_in(key, 3), (n,), minval=0.1, maxval=2.0
        )
        live = jnp.ones(n, bool)
        args = (jac_eq, jac_ineq, s, s_lb, s_ub, live, live)
        b = jax.random.normal(jax.random.fold_in(key, 4), (3 + 2 * n,))
        A = dense_operator(*args)

        mf = structured_solver("matrix-free", *args)
        y, n_iter = mf(b, with_info=True)
        assert int(n_iter) <= 3
        assert jnp.allclose(y, pinv_solution(A, b), atol=1e-7)

        factor = SchurNormalEquations.build(*args[:5], live_lb=live, live_ub=live)
        assert jnp.allclose(factor.solve(b), pinv_solution(A, b), atol=1e-10)
        jaxpr = jax.make_jaxpr(factor.solve)(b)
        # No loops: a fixed sequence of GEMVs and diagonal operations.
        assert not any(eqn.primitive.name in ("while", "scan") for eqn in jaxpr.eqns)
        eqn_counts.append(len(jaxpr.eqns))
    assert eqn_counts[0] == eqn_counts[1]


# --------------------------------------------------------------------------- #
# strategy resolution, pcg and the subproblem hooks
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("strategy", "m", "expected"),
    [
        ("auto", 3, "schur"),
        ("auto", 99, "schur"),
        ("auto", 100, "matrix-free"),
        ("schur", 500, "schur"),
        ("matrix-free", 1, "matrix-free"),
        ("generic", 1, "generic"),
    ],
)
def test_resolve_strategy_and_codes(strategy, m, expected):
    resolved = resolve_normal_equations_strategy(strategy, m, 100)
    assert resolved == expected
    assert (
        int(strategy_code(resolved))
        == {"generic": 0, "schur": 1, "matrix-free": 2}[expected]
    )


def test_resolve_strategy_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown normal-equations strategy"):
        resolve_normal_equations_strategy("dense", 3, 100)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [0, 1])
def test_pcg_matches_plain_cg_and_reports_iterations(seed):
    """Preconditioned and plain CG agree; the preconditioner cuts iterations."""
    key = jax.random.key(seed)
    B = jax.random.normal(key, (12, 12))
    scale = jnp.logspace(0, 4, 12)
    S = B @ B.T + jnp.diag(scale)
    rhs = jax.random.normal(jax.random.fold_in(key, 1), (12,))
    y_cg, k_cg = pcg(lambda z: S @ z, rhs, tol=1e-12, max_iter=200)
    y_pcg, k_pcg = pcg(
        lambda z: S @ z,
        rhs,
        tol=1e-12,
        max_iter=200,
        preconditioner=lambda r: r / jnp.diag(S),
    )
    expected = jnp.linalg.solve(S, rhs)
    assert jnp.allclose(y_cg, expected, atol=1e-8)
    assert jnp.allclose(y_pcg, expected, atol=1e-8)
    assert 0 < int(k_pcg) <= int(k_cg)
    _, k_zero = pcg(lambda z: S @ z, jnp.zeros(12), tol=1e-12, max_iter=200)
    assert int(k_zero) == 0


def test_subproblem_hooks_share_and_validate_the_factor():
    """``with_schur_normal_equations`` caches; masks / cutoff changes rebuild; generic rejected."""
    sub = make_funnel_barrier_subproblem(problem=PROBLEMS["eq-ineq-finite"]())
    assert sub.schur_cache is None
    cached = sub.with_schur_normal_equations()
    assert cached.schur_cache is not None
    assert cached.schur_normal_equations() is cached.schur_cache
    assert cached.schur_normal_equations(rcond=cached.schur_cache.rcond) is (
        cached.schur_cache
    )
    assert cached.schur_normal_equations(rcond=1e-3) is not cached.schur_cache
    n = sub.lagrangian.n
    assert (
        cached.schur_normal_equations(free_x=jnp.ones(n, bool))
        is not cached.schur_cache
    )
    assert cached.normal_equations_solver("schur").__self__ is cached.schur_cache
    with pytest.raises(ValueError, match="'schur' and 'matrix-free'"):
        sub.normal_equations_solver("generic")
