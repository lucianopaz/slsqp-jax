"""Top-level driver for :class:`~slsqp_jax.sqpdax.minimiser.base.AbstractConstrainedMinimiser`."""

from __future__ import annotations

from typing import Any, cast

import equinox as eqx
import jax
import optimistix as optx
from jax import numpy as jnp

from ..logging import Logger
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
        Optional ``{"minimiser": {...}, "subproblem": {...}, "logging": ...}``
        bag forwarded to :meth:`init`; the ``logging`` entry configures the
        solver's :class:`~slsqp_jax.sqpdax.logging.logger.Logger` (see
        :meth:`~slsqp_jax.sqpdax.logging.logger.Logger.from_options`); a
        :class:`~slsqp_jax.sqpdax.logging.handlers.DiagnosticsHandler`
        under its ``diagnostics`` key additionally collects structured
        per-step / per-solve records and a final ``"run"`` record.
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
    logger = getattr(solver, "logger", Logger.disabled())
    logger.info(
        f"{type(solver).__name__}: n={problem.n} meq={problem.meq} "
        f"mineq={problem.mineq} max_steps={max_steps}"
    )

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
    successful = solver.result_adapter.is_successful(result)
    logger.info(
        "finished: result={result} steps={steps}",
        when=successful,
        result=result,
        steps=solver.step_count,
    )
    logger.warning(
        "finished without success: result={result} steps={steps}",
        when=~successful,
        result=result,
        steps=solver.step_count,
    )
    logger.diagnostic(
        "run",
        {
            "result": result,
            "successful": successful,
            "steps": solver.step_count,
            "x": solver.iterate,
            "dual": solver.dual,
        },
    )
    solution = solver.postprocess(problem, result)
    if throw:
        solution = eqx.error_if(
            solution,
            ~solver.result_adapter.is_successful(result),
            "constrained minimise did not converge",
        )
    return solution
