"""End-to-end JIT matrix over inner KKT solvers × minimisers × multiplier recovery.

Every combination must compile once, converge on the shifted box quadratic,
recover the exact KKT multipliers with the right signs and expose the
standardised KKT diagnostics in ``stats``.
"""

from __future__ import annotations

import warnings

import equinox as eqx
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.minimiser import (
    ActiveSetLineSearchMinimiser,
    ProximalActiveSetLineSearchMinimiser,
    minimise,
)
from slsqp_jax.sqpdax.subproblem.solver import (
    KKT_SOLVER_RESULTS,
    CraigProjector,
    LeastSquaresMultiplierRecovery,
    MinresQLPSubProblemSolver,
    ProjectedCGSubProblemSolver,
    SVDProjector,
)

from ..conftest import make_shifted_box_quadratic

INNER_SOLVERS = {
    "svd-pcg": lambda: ProjectedCGSubProblemSolver(projector=SVDProjector()),
    "craig-pcg": lambda: ProjectedCGSubProblemSolver(projector=CraigProjector()),
    "minres-qlp": lambda: MinresQLPSubProblemSolver(),
}

MINIMISERS = {
    "active-set": ActiveSetLineSearchMinimiser,
    "proximal": ProximalActiveSetLineSearchMinimiser,
}

RECOVERIES = {
    "kkt": lambda: None,
    "least-squares": lambda: LeastSquaresMultiplierRecovery(),
}

KKT_STATS_KEYS = ("kkt_reason", "kkt_n_refinements", "kkt_feasibility_residual")


@pytest.mark.parametrize("make_recovery", RECOVERIES.values(), ids=RECOVERIES.keys())
@pytest.mark.parametrize("minimiser_cls", MINIMISERS.values(), ids=MINIMISERS.keys())
@pytest.mark.parametrize("make_inner", INNER_SOLVERS.values(), ids=INNER_SOLVERS.keys())
def test_inner_solver_matrix_converges_under_filter_jit(
    make_inner, minimiser_cls, make_recovery
):
    """Each inner solver / minimiser / recovery triple solves the NLP once compiled."""
    problem, x_star, dual_star = make_shifted_box_quadratic()
    subproblem_options: dict = {"subproblem_solver": make_inner()}
    recovery = make_recovery()
    if recovery is not None:
        subproblem_options["multiplier_recovery"] = recovery
    options = {"subproblem": subproblem_options}
    minimiser = minimiser_cls(rtol=1e-6, atol=1e-6, min_steps=2)
    traces = []

    @eqx.filter_jit
    def run(x0):
        traces.append(None)
        return minimise(
            problem, minimiser, x0, max_steps=40, throw=False, options=options
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sol = run(jnp.array([0.5, 0.5, 0.5]))
        sol_again = run(jnp.array([0.2, 0.7, -0.3]))

    # Single compile: the second call with a new starting point reuses the trace.
    assert len(traces) == 1

    for solution in (sol, sol_again):
        assert bool(solution.state.result_adapter.is_successful(solution.result))
        assert jnp.allclose(solution.value, x_star, atol=1e-4)
        stats = solution.stats
        assert jnp.allclose(
            stats["multipliers_ineq"], dual_star.ineq_multipliers, atol=1e-3
        )
        assert jnp.allclose(
            stats["multipliers_lb"], dual_star.lb_multipliers, atol=1e-3
        )
        assert jnp.allclose(
            stats["multipliers_ub"], dual_star.ub_multipliers, atol=1e-3
        )
        # Inequality and bound multipliers are dual feasible.
        for key in ("multipliers_ineq", "multipliers_lb", "multipliers_ub"):
            assert jnp.all(stats[key] >= -1e-6)
        # Standardised KKT diagnostics of the final working set.
        for key in KKT_STATS_KEYS:
            assert key in stats
        assert bool(stats["kkt_reason"] == KKT_SOLVER_RESULTS.converged)
        assert int(stats["kkt_n_refinements"]) >= 0
        assert bool(jnp.isfinite(stats["kkt_feasibility_residual"]))
        assert float(stats["kkt_feasibility_residual"]) < 1e-6
