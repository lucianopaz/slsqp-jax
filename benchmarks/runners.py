"""Backend-neutral solve interface used by :mod:`benchmarks.worker`.

A *runner* wraps one ``(problem, x0, config)`` triple and exposes the three
calls the worker times: :meth:`~SqpdaxRunner.compile` (one-off, excluded
from timing), :meth:`~SqpdaxRunner.warmup` (one short solve, also
excluded) and :meth:`~SqpdaxRunner.solve` (a full solve returning a
:class:`SolveOutcome`). :class:`SqpdaxRunner` drives the native minimisers;
:class:`benchmarks.baselines.ScipyRunner` drives the SciPy baselines.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from jax import Array

from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.problem import Problem

__all__ = [
    "Runner",
    "SolveOutcome",
    "SqpdaxRunner",
]


@dataclass(frozen=True)
class SolveOutcome:
    """Backend-independent summary of one full solve.

    Attributes
    ----------
    x
        Returned primal point.
    dual
        Multipliers in sqpdax convention (see
        :class:`~slsqp_jax.sqpdax.dual.Dual`), or ``None`` when the solver
        does not expose them.
    status
        Termination name. ``"successful"`` for a reported success and a name
        containing ``max_steps`` when the iteration budget was exhausted, so
        :mod:`benchmarks.analysis` classifies every backend the same way.
    successful
        Whether the solver reported success.
    steps
        Outer iterations performed.
    stats
        Scalar solver statistics; flattened into ``stats_*`` row columns.
    """

    x: Any
    dual: Dual | None
    status: str
    successful: bool
    steps: int
    stats: dict[str, Any] = field(default_factory=dict)


class Runner(Protocol):
    """Interface implemented by every benchmark backend."""

    def compile(self) -> float:
        """Prepare executables; return the wall time spent in seconds."""
        ...

    def warmup(self) -> None:
        """Run one short solve to pay any remaining one-off costs."""
        ...

    def solve(self, max_steps: int) -> SolveOutcome:
        """Run one full solve with the given outer-iteration budget."""
        ...


class SqpdaxRunner:
    """Run a native sqpdax minimiser through a single jitted executable.

    ``max_steps`` is a traced ``int32`` argument so the executable compiled
    in :meth:`compile` serves the warm-up (``max_steps=1``) and every full
    solve.

    Parameters
    ----------
    problem
        The NLP.
    x0
        Starting point.
    minimiser
        Un-initialised minimiser instance.
    options
        ``options`` mapping forwarded to
        :func:`~slsqp_jax.sqpdax.minimiser.minimise`.
    """

    def __init__(
        self,
        problem: Problem,
        x0: Array,
        minimiser: Any,
        options: Mapping[str, Any] | None = None,
    ):
        import jax

        from slsqp_jax.sqpdax.minimiser import minimise

        self.problem = problem
        self.x0 = x0
        self.minimiser = minimiser
        self.options = dict(options or {})

        def solve(x, max_steps):
            return minimise(
                problem,
                minimiser,
                x,
                max_steps=max_steps,
                throw=False,
                options=self.options,
            )

        self._jitted = jax.jit(solve)
        self._compiled: Any = None

    def _budget(self, max_steps: int):
        import jax.numpy as jnp

        return jnp.asarray(max_steps, jnp.int32)

    def compile(self) -> float:
        t0 = time.perf_counter()
        self._compiled = self._jitted.lower(self.x0, self._budget(1)).compile()
        return time.perf_counter() - t0

    def _run(self, max_steps: int):
        import jax

        if self._compiled is None:
            self.compile()
        return jax.block_until_ready(self._compiled(self.x0, self._budget(max_steps)))

    def warmup(self) -> None:
        self._run(1)

    def solve(self, max_steps: int) -> SolveOutcome:
        from slsqp_jax.sqpdax.results import is_successful

        from .metrics import result_name

        sol = self._run(max_steps)
        return SolveOutcome(
            x=sol.value,
            dual=getattr(sol.state, "dual", None),
            status=result_name(sol.result),
            successful=bool(is_successful(sol.result)),
            steps=int(sol.stats["num_steps"]),
            stats=dict(sol.stats),
        )
