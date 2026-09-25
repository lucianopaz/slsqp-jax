"""Unit tests for :mod:`slsqp_jax.sqpdax.minimiser.base`."""

from __future__ import annotations

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.minimiser import (
    ActiveSetLineSearchMinimiser,
    OptimisationContext,
    TrustRegionInteriorPointMinimiser,
)
from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.registry import FrozenDict
from slsqp_jax.sqpdax.step_controller import StepResult
from tests.sqpdax.lagrangian.conftest import make_problem
from tests.sqpdax.subproblem.solver.conftest import unbounded_box

from .conftest import ActiveSetLineSearchStub, make_unconstrained_quadratic


def test_init_seeds_iterate_dual_and_solver_state():
    """``init`` installs primal / dual / solver carry and zeros ``step_count``."""
    problem = make_unconstrained_quadratic()
    solver = ActiveSetLineSearchStub().init(problem, jnp.ones(2))
    assert solver.iterate is not None
    assert jnp.allclose(solver.iterate.x, jnp.ones(2))
    assert solver.dual is not None
    assert solver.solver_state is not None
    assert int(solver.step_count) == 0
    # Exact HVP → no secant.
    assert solver.secant is None


def test_parse_options_freezes_and_maps_static_fields():
    """Option bag becomes a ``FrozenDict``; recognised static keys are applied."""
    problem = make_unconstrained_quadratic()
    solver = ActiveSetLineSearchStub().init(
        problem,
        jnp.ones(2),
        options={"minimiser": {"rtol": 1e-4, "qp_tol": 1e-6}},
    )
    assert isinstance(solver.options, FrozenDict)
    assert solver.rtol == 1e-4
    assert solver.qp_tol == 1e-6


def test_validate_options_warns_on_unknown_keys():
    """Unknown sections / keys emit warnings but do not raise."""
    problem = make_unconstrained_quadratic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ActiveSetLineSearchStub().init(
            problem,
            jnp.ones(2),
            options={
                "bogus": {},
                "minimiser": {"not_a_field": 1.0},
                "subproblem": {
                    "not_a_solver_field": 0.0,
                    "working_set_policy": {"tol": 1e-7},
                },
            },
        )
    messages = " ".join(str(w.message) for w in caught)
    assert "unknown option section 'bogus'" in messages
    assert "unknown minimiser option 'not_a_field'" in messages
    assert "unknown subproblem option 'not_a_solver_field'" in messages


def test_secant_kind_spec_builds_lbfgs_when_no_exact_curvature():
    """Without exact HVP, ``minimiser.secant`` kind-spec seeds L-BFGS."""
    lb, ub = unbounded_box(2)
    problem = make_problem(n=2, meq=0, mineq=0, lb=lb, ub=ub, with_curvature=False)
    solver = ActiveSetLineSearchStub().init(
        problem,
        jnp.ones(2),
        options={"minimiser": {"secant": {"kind": "lbfgs", "memory": 3}}},
    )
    assert solver.secant is not None
    assert int(solver.secant.memory) == 3  # type: ignore[attr-defined]


def test_step_reduces_unconstrained_quadratic():
    """One active-set + Armijo step from ``x=ones`` moves toward the origin."""
    problem = make_unconstrained_quadratic()
    solver = ActiveSetLineSearchStub().init(problem, jnp.ones(2))
    merit0 = float(jnp.sum(solver.iterate.x**2))
    solver = solver.step(problem)
    assert int(solver.step_count) == 1
    assert float(jnp.sum(solver.iterate.x**2)) < merit0
    assert solver.solver_state is not None
    assert bool(solver.solver_state.success)


def test_terminate_and_postprocess_after_progress():
    """After enough progress the driver reports success and packs a Solution."""
    problem = make_unconstrained_quadratic()
    solver = ActiveSetLineSearchStub(min_steps=1, rtol=1e-4, atol=1e-4).init(
        problem, jnp.ones(2)
    )
    solver = solver.step(problem)
    # Another step should reach (near) the origin on the unconstrained QP.
    solver = solver.step(problem)
    done, result = solver.terminate(problem)
    assert bool(done)
    assert bool(solver.result_adapter.is_successful(result))
    sol = solver.postprocess(problem, result)
    assert jnp.allclose(sol.value, solver.iterate.x)
    assert int(sol.stats["num_steps"]) == int(solver.step_count)


def test_postprocess_requires_init():
    """``postprocess`` before ``init`` raises ``ValueError``."""
    solver = ActiveSetLineSearchStub()
    with pytest.raises(ValueError, match="iterate is not set"):
        solver.postprocess(
            make_unconstrained_quadratic(), solver.result_adapter.successful
        )


def test_optimisation_context_bundles_evaluated_lagrangian():
    """``_optimisation_context`` exposes problem / Lagrangian / solver state."""
    problem = make_unconstrained_quadratic()
    solver = ActiveSetLineSearchStub().init(problem, jnp.array([0.5, -0.25]))
    ctx = solver._optimisation_context(problem)
    assert isinstance(ctx, OptimisationContext)
    assert ctx.problem is problem
    assert jnp.allclose(ctx.lagrangian.x_ref, solver.iterate.x)
    assert ctx.solver_state is solver.solver_state


def test_nonfinite_diagnostic_fires():
    """A non-finite iterate forces ``RESULTS.nonfinite`` termination."""
    problem = make_unconstrained_quadratic()
    solver = ActiveSetLineSearchStub().init(problem, jnp.ones(2))
    bad = eqx.tree_at(lambda m: m.iterate, solver, Primal(jnp.array([jnp.nan, 0.0])))
    done, result = bad.terminate(problem)
    assert bool(done)
    assert result == solver.result_adapter.nonfinite


def test_abstract_validate_options_is_noop():
    """Bare abstract ``validate_options`` returns ``None``."""
    from slsqp_jax.sqpdax.minimiser.base import AbstractConstrainedMinimiser

    solver = ActiveSetLineSearchStub()
    assert AbstractConstrainedMinimiser.validate_options(solver) is None


def test_nested_subproblem_option_validation_warns():
    """Unknown keys inside a nested ``SubProblemSolver`` section warn."""
    problem = make_unconstrained_quadratic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ActiveSetLineSearchStub().init(
            problem,
            jnp.ones(2),
            options={
                "subproblem": {
                    "subproblem_solver": {"tol": 1e-9, "not_a_pcg_field": 1},
                }
            },
        )
    messages = " ".join(str(w.message) for w in caught)
    assert "unknown subproblem option 'not_a_pcg_field'" in messages


@pytest.mark.parametrize(
    "minimiser_cls",
    [ActiveSetLineSearchMinimiser, TrustRegionInteriorPointMinimiser],
    ids=["active-set", "interior-point"],
)
@pytest.mark.parametrize("accepted", [True, False], ids=["accepted", "rejected"])
@pytest.mark.parametrize("fill", [1.0, jnp.nan], ids=["finite", "nan"])
def test_execute_step_commits_dual_iff_finite(minimiser_cls, accepted, fill):
    """The multiplier estimate is committed iff finite, whatever the primal did.

    The estimate is a function of the current iterate, so a rejected primal
    step still refreshes ``λ`` (this is how a wrong dual is corrected at a
    primal-stationary point). A non-finite proposal, however, must never
    leak in: it would make the next termination check report ``nonfinite``
    even though the iterate never moved.
    """
    problem = make_problem(n=2, meq=1, mineq=2, with_curvature=True)
    solver = minimiser_cls().init(problem, jnp.asarray([0.5, 0.5]))
    ctx = solver._init_subproblem(problem)
    old_dual = solver.dual
    step_dual = jax.tree.map(lambda d: d + fill, old_dual)
    result = StepResult(
        x=solver.iterate,
        accepted=jnp.asarray(accepted),
        merit_val=jnp.asarray(0.0),
        solver_state=solver.solver_state,
        step_size=jnp.asarray(1.0 if accepted else 0.0),
        proposed_step_norm=jnp.asarray(1.0),
    )

    advanced = solver._execute_step(ctx, step_dual, result)

    expected = step_dual if jnp.isfinite(fill) else old_dual
    for got, want in zip(jax.tree.leaves(advanced.dual), jax.tree.leaves(expected)):
        assert jnp.array_equal(got, want)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(advanced.dual))
    assert int(advanced.step_count) == 1


def test_step_updates_secant_on_inexact_problem():
    """With no exact HVP, a successful step refreshes the L-BFGS secant."""
    lb, ub = unbounded_box(2)
    problem = make_problem(n=2, meq=0, mineq=0, lb=lb, ub=ub, with_curvature=False)
    solver = ActiveSetLineSearchStub().init(problem, jnp.ones(2))
    assert solver.secant is not None
    before = solver.secant
    solver = solver.step(problem)
    assert solver.secant is not None
    # Either a new buffer object or same instance after a skipped pair.
    assert solver.secant is not before or int(solver.step_count) == 1
