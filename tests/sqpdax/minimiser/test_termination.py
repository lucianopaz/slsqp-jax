"""Tests for native sqpdax termination classification."""

from __future__ import annotations

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import pytest
from jaxtyping import Array, Bool

from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.lagrangian import Lagrangian
from slsqp_jax.sqpdax.minimiser.active_set_linesearch import (
    ACTIVE_SET_LINE_SEARCH_RESULTS,
    ActiveSetLineSearchResultAdapter,
)
from slsqp_jax.sqpdax.minimiser.termination import (
    TerminationFlags,
    TerminationMetrics,
    classify_termination,
    compute_mu_max,
)
from slsqp_jax.sqpdax.minimiser.trust_region_interior_point import (
    TRUST_REGION_INTERIOR_POINT_RESULTS,
    TrustRegionInteriorPointResultAdapter,
)
from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.results import (
    is_successful,
)
from slsqp_jax.sqpdax.types import Scalar

from .conftest import make_equality_quadratic, make_unconstrained_quadratic


class ExtendedMetrics(TerminationMetrics[ACTIVE_SET_LINE_SEARCH_RESULTS]):
    """Stand-in for an algorithm-specific metrics schema."""

    residual: Scalar
    has_min_steps: Bool[Array, ""]


def make_flags(
    *,
    converged: bool = False,
    nonfinite: bool = False,
    fatal: bool = False,
) -> TerminationFlags[ACTIVE_SET_LINE_SEARCH_RESULTS]:
    """Build active-set flags from Python booleans."""
    return TerminationFlags(
        converged=jnp.asarray(converged),
        nonfinite=jnp.asarray(nonfinite),
        fatal=jnp.asarray(fatal),
        fatal_result=ACTIVE_SET_LINE_SEARCH_RESULTS.qp_subproblem_failure,
    )


@pytest.mark.parametrize(
    ("converged", "nonfinite", "fatal", "expect_done", "expected_name"),
    [
        (False, False, False, False, "running"),
        (True, False, False, True, "successful"),
        (False, True, False, True, "nonfinite"),
        (False, False, True, True, "qp_subproblem_failure"),
        (True, True, True, True, "nonfinite"),
        (True, False, True, True, "successful"),
    ],
)
def test_classify_termination_precedence(
    converged, nonfinite, fatal, expect_done, expected_name
):
    """Priority is nonfinite > converged > fatal > running."""
    adapter = ActiveSetLineSearchResultAdapter()
    done, result = classify_termination(
        make_flags(converged=converged, nonfinite=nonfinite, fatal=fatal),
        adapter,
    )
    assert bool(done) is expect_done
    assert bool(result == getattr(ACTIVE_SET_LINE_SEARCH_RESULTS, expected_name))


def test_classification_is_generic_across_result_types():
    """The same classifier accepts a different algorithm result enumeration."""
    adapter = TrustRegionInteriorPointResultAdapter()
    flags = TerminationFlags(
        converged=jnp.asarray(False),
        nonfinite=jnp.asarray(False),
        fatal=jnp.asarray(True),
        fatal_result=TRUST_REGION_INTERIOR_POINT_RESULTS.subproblem_singular,
    )
    done, result = classify_termination(flags, adapter)
    assert bool(done)
    assert bool(result == TRUST_REGION_INTERIOR_POINT_RESULTS.subproblem_singular)


@pytest.mark.parametrize(
    ("native", "coarse"),
    [
        (
            ACTIVE_SET_LINE_SEARCH_RESULTS.merit_stagnation,
            optx.RESULTS.nonlinear_divergence,
        ),
        (
            ACTIVE_SET_LINE_SEARCH_RESULTS.max_steps_reached,
            optx.RESULTS.nonlinear_max_steps_reached,
        ),
        (
            ACTIVE_SET_LINE_SEARCH_RESULTS.nonfinite,
            optx.RESULTS.nonfinite,
        ),
    ],
)
def test_active_set_adapter_coarsens_only_at_compatibility_boundary(native, coarse):
    """Fine native reasons have an explicit Optimistix dispatch."""
    adapter = ActiveSetLineSearchResultAdapter()
    assert bool(adapter.to_optimistix(native) == coarse)


def test_trust_region_adapter_preserves_known_linear_failure():
    """A native detailed KKT failure maps to its Optimistix counterpart."""
    adapter = TrustRegionInteriorPointResultAdapter()
    coarse = adapter.to_optimistix(
        TRUST_REGION_INTERIOR_POINT_RESULTS.subproblem_singular
    )
    assert bool(coarse == optx.RESULTS.singular)


@pytest.mark.parametrize(
    "result",
    [
        ACTIVE_SET_LINE_SEARCH_RESULTS.successful,
        TRUST_REGION_INTERIOR_POINT_RESULTS.successful,
        optx.RESULTS.successful,
    ],
)
def test_is_successful_handles_native_and_optimistix_enumerations(result):
    """Both sqpdax-local and project-public helpers avoid cross-enum equality."""
    assert bool(is_successful(result))


def test_classify_termination_is_jittable():
    """Native result selection traces under JIT."""
    adapter = ActiveSetLineSearchResultAdapter()

    @eqx.filter_jit
    def run(converged, nonfinite, fatal):
        return classify_termination(
            TerminationFlags(
                converged=converged,
                nonfinite=nonfinite,
                fatal=fatal,
                fatal_result=ACTIVE_SET_LINE_SEARCH_RESULTS.line_search_failure,
            ),
            adapter,
        )

    done, result = run(jnp.asarray(False), jnp.asarray(False), jnp.asarray(True))
    assert bool(done)
    assert bool(result == ACTIVE_SET_LINE_SEARCH_RESULTS.line_search_failure)


def test_termination_modules_round_trip_as_pytrees():
    """Generic flags and metrics remain valid JAX pytrees."""
    values = (
        make_flags(converged=True),
        ExtendedMetrics(
            nonfinite=jnp.asarray(False),
            fatal_result=ACTIVE_SET_LINE_SEARCH_RESULTS.qp_subproblem_failure,
            residual=jnp.asarray(0.5),
            has_min_steps=jnp.asarray(True),
        ),
    )
    for value in values:
        leaves, treedef = jax.tree.flatten(value)
        assert leaves
        assert eqx.tree_equal(jax.tree.unflatten(treedef, leaves), value)


def _zero_dual(n: int, meq: int, mineq: int) -> Dual:
    return Dual(
        eq_multipliers=jnp.zeros(meq),
        ineq_multipliers=jnp.zeros(mineq),
        lb_multipliers=jnp.zeros(n),
        ub_multipliers=jnp.zeros(n),
    )


def test_compute_mu_max_unconstrained_is_objective_gradient_norm():
    """With no multipliers, filterSQP scaling is ``‖∇f‖₂``."""
    problem = make_unconstrained_quadratic()
    lagrangian = Lagrangian(problem, None)(
        Primal(jnp.ones(problem.n)),
        _zero_dual(problem.n, problem.meq, problem.mineq),
    )
    assert float(compute_mu_max(lagrangian)) == pytest.approx(jnp.sqrt(8.0))


def test_compute_mu_max_uses_constraint_row_norm_times_multiplier():
    """A general-constraint contribution can dominate the objective."""
    problem = make_equality_quadratic()
    dual = _zero_dual(problem.n, problem.meq, problem.mineq)
    dual = eqx.tree_at(lambda d: d.eq_multipliers, dual, jnp.asarray([3.0]))
    lagrangian = Lagrangian(problem, None)(Primal(jnp.asarray([0.25, 0.25])), dual)
    assert float(compute_mu_max(lagrangian)) == pytest.approx(3.0 * jnp.sqrt(2.0))


def test_compute_mu_max_uses_only_finite_bound_multipliers():
    """Finite bounds contribute ``|ν|`` while null-bound slots are ignored."""
    problem = replace(
        make_unconstrained_quadratic(),
        lb=jnp.asarray([0.0, -jnp.inf]),
        null_lb=jnp.asarray([False, True]),
    )
    dual = _zero_dual(problem.n, problem.meq, problem.mineq)
    dual = eqx.tree_at(lambda d: d.lb_multipliers, dual, jnp.asarray([5.0, 100.0]))
    lagrangian = Lagrangian(problem, None)(Primal(jnp.ones(problem.n)), dual)
    assert float(compute_mu_max(lagrangian)) == pytest.approx(5.0)
