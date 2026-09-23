"""Top-level driver for :class:`~slsqp_jax.sqpdax.minimiser.base.AbstractConstrainedMinimiser`."""

from __future__ import annotations

from typing import Any, cast

import equinox as eqx
import jax
import optimistix as optx
from jax import numpy as jnp

from ..problem import ProblemProtocol, bind_problem_args
from ..types import Vector_n
from .base import AbstractConstrainedMinimiser

__all__ = [
    "minimise",
]


def minimise(
    problem: ProblemProtocol[Any],
    solver: AbstractConstrainedMinimiser[Any, Any, Any, Any, Any],
    x0: Vector_n,
    *,
    max_steps: int = 256,
    throw: bool = True,
    options: dict | None = None,
    problem_args: tuple[Any, ...] = (),
    problem_kwargs: dict[str, Any] | None = None,
) -> optx.Solution:
    """Owned driver mirroring ``optimistix.minimise`` for the constrained base.

    ``solver`` is an *already-configured*
    :class:`~slsqp_jax.sqpdax.minimiser.base.AbstractConstrainedMinimiser`
    (an instance, not a class): the algorithm parameters live on it and
    :meth:`~slsqp_jax.sqpdax.minimiser.base.AbstractConstrainedMinimiser.init`
    seeds the dynamic state. The loop is the classical init /
    while-not-done / postprocess triad, reusing optimistix's
    :class:`~optimistix.Solution` container while retaining the concrete
    minimiser's native fine-grained result.

    Parameters
    ----------
    problem
        NLP to minimise.
    solver
        Configured (but typically not yet initialised) constrained
        minimiser.
    x0
        Decision-variable starting point.
    max_steps
        Outer iteration budget.
    throw
        If ``True``, wrap the solution in :func:`equinox.error_if` when
        the status is not ``successful``.
    options
        Optional ``{"minimiser": {...}, "subproblem": {...}}`` bag
        forwarded to :meth:`init`.
    problem_args
        Additional positional arguments forwarded to the problem.
    problem_kwargs
        Additional keyword arguments forwarded to the problem.

    Returns
    -------
    optimistix.Solution
        ``value`` is the final decision vector.

    Examples
    --------
    Concrete algorithms subclass
    :class:`~slsqp_jax.sqpdax.minimiser.base.CommonMinimiser` and are
    passed here as configured instances; see the unit tests for a
    minimal end-to-end stub.
    """
    problem = bind_problem_args(problem, problem_args, problem_kwargs)
    solver = solver.init(problem, x0, options)

    def cond(carry):
        solver, n = carry
        done, _ = solver.terminate(problem)
        return jnp.logical_and(jnp.logical_not(done), n < max_steps)

    def body(carry):
        solver, n = carry
        return solver.step(problem), n + 1

    solver, _ = cast(
        tuple[AbstractConstrainedMinimiser[Any, Any, Any, Any, Any], Any],
        jax.lax.while_loop(cond, body, (solver, jnp.asarray(0, jnp.int32))),
    )
    done, result = solver.terminate(problem)
    result = solver.result_adapter.result_type.where(
        done, result, solver.result_adapter.max_steps_reached
    )
    solution = solver.postprocess(problem, result)
    if throw:
        solution = eqx.error_if(
            solution,
            ~solver.result_adapter.is_successful(result),
            "constrained minimise did not converge",
        )
    return solution
