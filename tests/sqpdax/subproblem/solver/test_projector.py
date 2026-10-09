"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.projector`."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.preconditioner import IdentityPreconditioner, MatrixPreconditioner
from slsqp_jax.sqpdax.problem import build_problem
from slsqp_jax.sqpdax.subproblem.base import SubProblem
from slsqp_jax.sqpdax.subproblem.solver import (
    CraigProjectionContext,
    CraigProjector,
    Projector,
    SVDProjectionContext,
    SVDProjector,
)
from tests.sqpdax.lagrangian.conftest import make_problem
from tests.sqpdax.preconditioner.conftest import make_spd_matrix
from tests.sqpdax.subproblem.conftest import make_scaled_barrier_subproblem

from .conftest import make_qp_subproblem, unbounded_box


def _active_set_free() -> SubProblem:
    lb, ub = unbounded_box(2)
    return make_qp_subproblem(
        problem=make_problem(meq=1, mineq=2, lb=lb, ub=ub),
        active_inequalities=(True, False),
    )


def _active_set_with_bounds() -> SubProblem:
    # Fixing x0 at its lower bound and activating the ``-x1`` inequality makes
    # ``A_work = [[0, 1], [0, -1]]``: two parallel working rows (rank 1).
    return make_qp_subproblem(
        active_inequalities=(False, True),
        active_lb=(True, False),
    )


SUBPROBLEMS = {
    "active-set-free": _active_set_free,
    "active-set-bounds": _active_set_with_bounds,
    "scaled-barrier": lambda: make_scaled_barrier_subproblem(),
}

PRECONDITIONERS = {
    "none": lambda: None,
    "identity": lambda: IdentityPreconditioner(jnp.zeros(2)),
    "matrix": lambda: MatrixPreconditioner(make_spd_matrix(2)),
}


def _rhs(sub: SubProblem) -> jnp.ndarray:
    _, dual_rhs = sub.kkt_rhs()
    return jnp.concatenate([dual_rhs.eq_multipliers, dual_rhs.ineq_multipliers])


def _apply_M(pre, v):
    if pre is None:
        return v
    return pre.pushforward(v)


@pytest.mark.parametrize("make_sub", SUBPROBLEMS.values(), ids=SUBPROBLEMS.keys())
def test_working_geometry_masks_rows_and_columns(make_sub):
    """Inactive rows are zeroed, fixed columns are dropped from ``A_work``."""
    sub = make_sub()
    A, A_work, active_rows, free_mask, d_fixed = Projector.working_geometry(sub)
    full = sub.nonbound_constraint_jac()
    exp_free, exp_fixed = sub.free_subspace()

    assert jnp.array_equal(active_rows, sub.active_constraint_rows())
    assert jnp.array_equal(free_mask, exp_free)
    assert jnp.array_equal(d_fixed, exp_fixed)
    assert jnp.allclose(A[active_rows], full[active_rows])
    assert jnp.all(A[~active_rows] == 0)
    assert jnp.all(A_work[:, ~free_mask] == 0)
    assert jnp.allclose(A_work[:, free_mask], A[:, free_mask])
    assert jnp.all(d_fixed[free_mask] == 0)


@pytest.mark.parametrize(
    "make_pre", PRECONDITIONERS.values(), ids=PRECONDITIONERS.keys()
)
@pytest.mark.parametrize("make_sub", SUBPROBLEMS.values(), ids=SUBPROBLEMS.keys())
def test_svd_projection_is_an_m_orthogonal_null_space_projector(make_sub, make_pre):
    """``P`` is idempotent, maps into ``null(A_work)`` and kills fixed components."""
    sub = make_sub()
    pre = make_pre()
    ctx = SVDProjector().build(sub, pre)
    assert isinstance(ctx, SVDProjectionContext)
    assert bool(ctx.converged)
    assert int(ctx.n_iter) == 0
    assert ctx.is_preconditioned == (
        pre is not None and not isinstance(pre, IdentityPreconditioner)
    )

    v = jnp.array([0.7, -1.3])
    p = ctx.project(v)
    assert jnp.allclose(ctx.project(p), p, atol=1e-6)
    assert jnp.allclose(ctx.A_work @ p, 0.0, atol=1e-6)
    assert jnp.all(p[~ctx.free_mask] == 0)
    # ``M``-self-adjointness: ``⟨u, M P v⟩ = ⟨P u, M v⟩`` on the free subspace.
    u = jnp.array([-0.4, 0.9]) * ctx.free_f
    lhs = jnp.vdot(u, _apply_M(pre, ctx.project(v * ctx.free_f)))
    rhs = jnp.vdot(ctx.project(u), _apply_M(pre, v * ctx.free_f))
    assert jnp.isclose(lhs, rhs, atol=1e-6)


@pytest.mark.parametrize(
    "make_pre", PRECONDITIONERS.values(), ids=PRECONDITIONERS.keys()
)
@pytest.mark.parametrize("make_sub", SUBPROBLEMS.values(), ids=SUBPROBLEMS.keys())
def test_particular_solution_and_correction_satisfy_the_working_rows(
    make_sub, make_pre
):
    """``A d_p = b`` on active rows (least squares when rank deficient)."""
    sub = make_sub()
    ctx = SVDProjector().build(sub, make_pre())
    b = _rhs(sub)

    d_p = ctx.particular_solution(b)
    assert jnp.allclose(d_p[~ctx.free_mask], ctx.d_fixed[~ctx.free_mask])
    b_eff = ctx.effective_rhs(b)
    assert jnp.all(b_eff[~ctx.active_rows] == 0)
    # Least-squares residual of the free block is the best achievable one.
    lstsq_resid = jnp.linalg.lstsq(ctx.A_work, b_eff)[1]
    achieved = ctx.feasibility_residual(d_p, b)
    best = jnp.sqrt(jnp.sum(lstsq_resid)) if lstsq_resid.size else jnp.asarray(0.0)
    assert jnp.isclose(achieved, best, atol=1e-5)

    # A perturbed step is pulled back to the same residual floor.
    d = d_p + ctx.project(jnp.array([0.3, 0.2])) + 0.5 * ctx.free_f
    corrected = ctx.feasibility_correction(d, b)
    assert jnp.isclose(ctx.feasibility_residual(corrected, b), best, atol=1e-5)


def test_rank_cut_drops_dependent_working_rows():
    """Parallel working rows collapse to rank one and stay finite."""
    ctx = SVDProjector().build(_active_set_with_bounds())
    assert int(jnp.sum(ctx.inv_s2 > 0)) == 1
    assert jnp.all(jnp.isfinite(ctx.inv_s2))
    assert ctx.AMAt_pinv is None
    # Full-rank case keeps every singular value.
    full = SVDProjector().build(_active_set_free())
    assert int(jnp.sum(full.inv_s2 > 0)) == int(jnp.sum(full.active_rows))


def test_explicit_rcond_controls_the_rank_cut():
    """A large ``rcond`` discards the small singular value of a full-rank ``A``."""
    sub = _active_set_free()
    A_work = SVDProjector.working_geometry(sub)[1]
    s = jnp.linalg.svd(A_work, compute_uv=False)
    ratio = float(s.min() / s.max())
    loose = SVDProjector(rcond=0.5 * ratio).build(sub)
    tight = SVDProjector(rcond=2.0 * ratio).build(sub)
    assert int(jnp.sum(loose.inv_s2 > 0)) == 2
    assert int(jnp.sum(tight.inv_s2 > 0)) == 1


def test_preconditioned_normal_solve_reduces_to_plain_without_m():
    """Without ``M`` the two range-space solves coincide."""
    ctx = SVDProjector().build(_active_set_free())
    rhs = jnp.array([0.3, -0.2, 0.9])
    assert jnp.allclose(ctx.solve_preconditioned_normal(rhs), ctx.solve_normal(rhs))
    assert jnp.allclose(ctx.apply_Minv(rhs[:2]), rhs[:2])


# ---------------------------------------------------------------------------
# CRAIG backend
# ---------------------------------------------------------------------------


def _active_set_bound_full_rank() -> SubProblem:
    # ``x0`` pinned at its lower bound with only the equality active:
    # ``A_work = [[0, 1]]`` has full row rank.
    return make_qp_subproblem(active_lb=(True, False))


FULL_RANK = {
    "active-set-free": _active_set_free,
    "active-set-bound": _active_set_bound_full_rank,
}
RANK_DEFICIENT = {
    "active-set-parallel-rows": _active_set_with_bounds,
    "scaled-barrier-overdetermined": lambda: make_scaled_barrier_subproblem(),
}


def _wide_active_set(fix_first: bool) -> SubProblem:
    # Four variables, one equality row and (optionally) ``x0`` pinned at its
    # lower bound: ``A_work = [[0, 1, 1, 1]]`` leaves a two-dimensional null
    # space, so ``P v`` is not round-off and the metric identity is testable.
    lb = jnp.array([0.0, -jnp.inf, -jnp.inf, -jnp.inf])
    problem = build_problem(
        lambda x: jnp.sum(x**2),
        n=4,
        meq=1,
        eq_fn=lambda x: jnp.array([jnp.sum(x) - 1.0]),
        lb=lb,
        ub=jnp.full(4, jnp.inf),
        autodiff_mode="jax",
        force_hvp_in_jax_mode=True,
    )
    return make_qp_subproblem(
        problem=problem, active_lb=(fix_first, False, False, False)
    )


WIDE_SUBPROBLEMS = {
    "all-free": lambda: _wide_active_set(fix_first=False),
    "one-fixed": lambda: _wide_active_set(fix_first=True),
}
WIDE_PRECONDITIONERS = {
    "none": lambda: None,
    "identity": lambda: IdentityPreconditioner(jnp.zeros(4)),
    "matrix": lambda: MatrixPreconditioner(make_spd_matrix(4)),
}


@pytest.mark.parametrize("projector", [SVDProjector(), CraigProjector()])
@pytest.mark.parametrize(
    "make_pre", WIDE_PRECONDITIONERS.values(), ids=WIDE_PRECONDITIONERS.keys()
)
@pytest.mark.parametrize(
    "make_sub", WIDE_SUBPROBLEMS.values(), ids=WIDE_SUBPROBLEMS.keys()
)
def test_project_pair_satisfies_the_masked_metric_identity(
    make_sub, make_pre, projector
):
    """``(P v, Q v)`` reproduces ``project`` and obeys ``vᵀ P v = (Q v)ᵀ P v``.

    The identity is the projected-CG round-off floor detector.  It must hold
    with a fixed variable under a non-diagonal ``M`` (``one-fixed`` ×
    ``matrix``), where the naive ``(P v)ᵀ M (P v)`` does *not*: the inverse
    of a principal block of ``M⁻¹`` is not the principal block of ``M``.
    """
    sub = make_sub()
    pre = make_pre()
    ctx = projector.build(sub, pre)

    v = jnp.array([0.7, -1.3, 0.4, 2.1])
    pv, qv = ctx.project_pair(v)
    assert jnp.allclose(pv, ctx.project(v), atol=1e-5)
    assert jnp.allclose(ctx.A_work @ pv, 0.0, atol=1e-5)
    vpv = jnp.vdot(v, pv)
    assert float(vpv) > 0.1  # non-trivial null-space component
    assert jnp.isclose(vpv, jnp.vdot(qv, pv), rtol=1e-4)
    if ctx.is_preconditioned and not bool(jnp.all(ctx.free_mask)):
        assert pre is not None
        naive = jnp.vdot(pv, pre.pushforward(pv))
        assert not jnp.isclose(vpv, naive, rtol=1e-2)


@pytest.mark.parametrize(
    "make_pre", PRECONDITIONERS.values(), ids=PRECONDITIONERS.keys()
)
@pytest.mark.parametrize("make_sub", FULL_RANK.values(), ids=FULL_RANK.keys())
def test_craig_matches_svd_on_full_rank_working_sets(make_sub, make_pre):
    """Matrix-free CRAIG / CG reproduce the direct SVD context."""
    sub = make_sub()
    pre = make_pre()
    craig = CraigProjector().build(sub, pre)
    svd = SVDProjector().build(sub, pre)
    assert isinstance(craig, CraigProjectionContext)
    assert bool(craig.converged)
    assert 1 <= int(craig.n_iter) <= int(jnp.sum(craig.active_rows))

    v = jnp.array([0.7, -1.3])
    b = _rhs(sub)
    assert jnp.allclose(craig.project(v), svd.project(v), atol=1e-5)
    assert jnp.allclose(
        craig.particular_solution(b), svd.particular_solution(b), atol=1e-5
    )
    # Range-space solves on right-hand sides in the range of ``A_work``.
    rhs = craig.A_work @ v
    assert jnp.allclose(craig.solve_normal(rhs), svd.solve_normal(rhs), atol=1e-5)
    rhs_m = craig.A_work @ craig.apply_Minv(v)
    assert jnp.allclose(
        craig.solve_preconditioned_normal(rhs_m),
        svd.solve_preconditioned_normal(rhs_m),
        atol=1e-4,
    )
    d = craig.particular_solution(b) + 0.5 * craig.free_f
    assert jnp.allclose(
        craig.feasibility_correction(d, b), svd.feasibility_correction(d, b), atol=1e-5
    )
    assert (
        float(craig.feasibility_residual(craig.feasibility_correction(d, b), b)) < 1e-5
    )


@pytest.mark.parametrize(
    "make_pre", PRECONDITIONERS.values(), ids=PRECONDITIONERS.keys()
)
@pytest.mark.parametrize("make_sub", RANK_DEFICIENT.values(), ids=RANK_DEFICIENT.keys())
def test_craig_reports_breakdown_on_rank_deficient_rows(make_sub, make_pre):
    """Inconsistent working rows: ``converged`` is False, outputs stay finite."""
    sub = make_sub()
    pre = make_pre()
    craig = CraigProjector().build(sub, pre)
    svd = SVDProjector().build(sub, pre)
    assert not bool(craig.converged)
    assert int(craig.n_iter) >= 1
    v = jnp.array([0.7, -1.3])
    b = _rhs(sub)
    for out in (craig.project(v), craig.particular_solution(b), craig.solve_normal(b)):
        assert jnp.all(jnp.isfinite(out))
    # The projection only needs consistent right-hand sides and still agrees.
    assert jnp.allclose(craig.project(v), svd.project(v), atol=1e-5)
    assert jnp.allclose(craig.A_work @ craig.project(v), 0.0, atol=1e-5)


def test_craig_tolerances_and_budget_are_honoured():
    """Budget caps the residual; the absolute floor decides ``converged``."""
    sub = _active_set_free()
    b = _rhs(sub)
    tight = CraigProjector(max_iter=1).build(sub)
    loose = CraigProjector(max_iter=1, atol=2.0).build(sub)
    full = CraigProjector().build(sub)
    r_tight = float(tight.feasibility_residual(tight.particular_solution(b), b))
    r_full = float(full.feasibility_residual(full.particular_solution(b), b))
    assert int(tight.n_iter) == 1 and int(full.n_iter) == 2
    assert r_full < 1e-5 < r_tight < 2.0
    assert not bool(tight.converged)
    assert bool(loose.converged)
    # A relative tolerance scales with ``‖b‖``.
    rel = CraigProjector(max_iter=1, rtol=10.0, atol=0.0).build(sub)
    assert bool(rel.converged)
    # Zero right-hand side: exact at zero iterations of the recurrence body.
    sub0 = eqx.tree_at(lambda s: s.lagrangian.evaluated.eq_fn_val, sub, jnp.zeros((1,)))
    sub0 = eqx.tree_at(lambda s: s.L_k.evaluated.eq_fn_val, sub0, jnp.zeros((1,)))
    sub0 = eqx.tree_at(lambda s: s.L_k.evaluated.ineq_fn_val, sub0, jnp.zeros((2,)))
    ctx0 = CraigProjector().build(sub0)
    assert bool(ctx0.converged)
    assert jnp.array_equal(ctx0.particular_solution(_rhs(sub0)), jnp.zeros(2))
