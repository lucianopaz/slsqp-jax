"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.projector`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.preconditioner import IdentityPreconditioner, MatrixPreconditioner
from slsqp_jax.sqpdax.subproblem.base import SubProblem
from slsqp_jax.sqpdax.subproblem.solver import (
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
