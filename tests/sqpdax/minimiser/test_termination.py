"""Unit tests for :mod:`slsqp_jax.sqpdax.minimiser.termination`."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import pytest
from jaxtyping import Array, Bool

from slsqp_jax.sqpdax.minimiser.termination import (
    TerminationFlags,
    TerminationMetrics,
    classify_termination,
)
from slsqp_jax.sqpdax.types import Scalar


class ExtendedMetrics(TerminationMetrics):
    """Stand-in for a per-algorithm metrics schema."""

    residual: Scalar
    has_min_steps: Bool[Array, ""]


def make_flags(
    *,
    converged: bool = False,
    nonfinite: bool = False,
    fatal: bool = False,
    subproblem_result: optx.RESULTS = optx.RESULTS.singular,
) -> TerminationFlags:
    """Build :class:`TerminationFlags` from Python booleans."""
    return TerminationFlags(
        converged=jnp.asarray(converged),
        nonfinite=jnp.asarray(nonfinite),
        fatal=jnp.asarray(fatal),
        subproblem_result=subproblem_result,
    )


@pytest.mark.parametrize(
    ("converged", "nonfinite", "fatal", "expect_done", "expect_result"),
    [
        (False, False, False, False, optx.RESULTS.successful),
        (True, False, False, True, optx.RESULTS.successful),
        (False, True, False, True, optx.RESULTS.nonfinite),
        (False, False, True, True, optx.RESULTS.singular),
        (True, True, False, True, optx.RESULTS.nonfinite),
        (True, False, True, True, optx.RESULTS.successful),
        (False, True, True, True, optx.RESULTS.nonfinite),
        (True, True, True, True, optx.RESULTS.nonfinite),
    ],
    ids=[
        "running",
        "converged",
        "nonfinite",
        "fatal",
        "nonfinite-beats-converged",
        "converged-beats-fatal",
        "nonfinite-beats-fatal",
        "nonfinite-beats-all",
    ],
)
def test_classify_termination_precedence(
    converged, nonfinite, fatal, expect_done, expect_result
):
    """``done`` fires on any flag; ``result`` follows nonfinite > converged > fatal.

    ``subproblem_result`` is a distinctive code (``singular``) so the fatal
    branch cannot be mistaken for a generic fallback.
    """
    done, result = classify_termination(
        make_flags(converged=converged, nonfinite=nonfinite, fatal=fatal)
    )
    assert bool(done) is expect_done
    assert bool(result == expect_result)


@pytest.mark.parametrize(
    "subproblem_result",
    [optx.RESULTS.singular, optx.RESULTS.breakdown, optx.RESULTS.stagnation],
)
def test_classify_termination_forwards_subproblem_result(subproblem_result):
    """A fatal outcome reports the subproblem's own code, not a generic one."""
    done, result = classify_termination(
        make_flags(fatal=True, subproblem_result=subproblem_result)
    )
    assert bool(done)
    assert bool(result == subproblem_result)


@pytest.mark.parametrize(
    "subproblem_result",
    [optx.RESULTS.singular, optx.RESULTS.breakdown],
)
@pytest.mark.parametrize("converged", [True, False], ids=["converged", "running"])
def test_classify_termination_hides_subproblem_result_unless_fatal(
    subproblem_result, converged
):
    """``subproblem_result`` only surfaces through the ``fatal`` branch."""
    _, result = classify_termination(
        make_flags(converged=converged, subproblem_result=subproblem_result)
    )
    assert bool(result == optx.RESULTS.successful)


def test_classify_termination_is_jittable():
    """The classifier traces under ``jit`` with traced (non-static) flags."""

    @eqx.filter_jit
    def run(converged, nonfinite, fatal):
        return classify_termination(
            TerminationFlags(
                converged=converged,
                nonfinite=nonfinite,
                fatal=fatal,
                subproblem_result=optx.RESULTS.singular,
            )
        )

    done, result = run(jnp.asarray(False), jnp.asarray(False), jnp.asarray(True))
    assert bool(done)
    assert bool(result == optx.RESULTS.singular)


@pytest.mark.parametrize(
    ("instance_factory", "expect_type"),
    [
        (lambda: make_flags(converged=True), TerminationFlags),
        (
            lambda: ExtendedMetrics(
                nonfinite=jnp.asarray(False),
                subproblem_result=optx.RESULTS.successful,
                residual=jnp.asarray(0.5),
                has_min_steps=jnp.asarray(True),
            ),
            ExtendedMetrics,
        ),
    ],
    ids=["flags", "metrics"],
)
def test_termination_modules_round_trip_as_pytrees(instance_factory, expect_type):
    """Flags / metrics are Equinox modules usable as ``while_loop`` carries."""
    instance = instance_factory()
    leaves, treedef = jax.tree.flatten(instance)
    assert leaves
    rebuilt = jax.tree.unflatten(treedef, leaves)
    assert isinstance(rebuilt, expect_type)
    assert eqx.tree_equal(rebuilt, instance)


def test_termination_metrics_subclass_extends_base_fields():
    """Subclasses add algorithm-specific fields while keeping the shared two."""
    metrics = ExtendedMetrics(
        nonfinite=jnp.asarray(False),
        subproblem_result=optx.RESULTS.successful,
        residual=jnp.asarray(1e-9),
        has_min_steps=jnp.asarray(True),
    )
    assert isinstance(metrics, TerminationMetrics)
    assert not bool(metrics.nonfinite)
    assert bool(metrics.subproblem_result == optx.RESULTS.successful)
    assert float(metrics.residual) == pytest.approx(1e-9)
    assert bool(metrics.has_min_steps)
