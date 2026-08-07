"""Adapter exposing a constrained minimiser to ``optimistix.minimise``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import optimistix as optx
from equinox import field
from jax import Array
from jaxtyping import PyTree, Shaped

from ..problem import ProblemProtocol
from .base import AbstractConstrainedMinimiser

__all__ = [
    "OptimistixMinimiser",
    "as_optimistix_minimiser",
]


class OptimistixMinimiser(optx.AbstractMinimiser):
    """Adapter exposing an :class:`AbstractConstrainedMinimiser` to optimistix.

    ``optimistix`` threads its own ``state`` object through ``init`` /
    ``step`` / ``terminate`` / ``postprocess``; here that state *is* our
    constrained-minimiser instance (fused config + state), and ``fn`` /
    ``args`` are ignored because the problem is carried on the adapter.
    ``init`` receives the raw ``y`` (a
    :data:`~slsqp_jax.sqpdax.types.Vector_n`) and delegates the
    :class:`~slsqp_jax.sqpdax.primal.Primal` / slack construction to
    ``inner.init``.

    Attributes
    ----------
    problem
        NLP solved by ``inner``.
    inner
        Configured (pre-init) constrained minimiser.
    rtol
        Relative tolerance required by
        :class:`optimistix.AbstractMinimiser` (unused by the inner
        driver; kept for the optimistix interface).
    atol
        Absolute tolerance required by the optimistix base.
    norm
        Residual norm required by the optimistix base.
    """

    # Parent ``AbstractIterativeSolver`` declares ``rtol`` / ``atol`` / ``norm``
    # with defaults ahead of us in the MRO, so every subclass field needs a
    # default (dataclass field-order rule). Construction always fills these.
    problem: ProblemProtocol[Any] = field(default=cast(Any, None))
    inner: AbstractConstrainedMinimiser[Any, Any, Any] = field(default=cast(Any, None))
    rtol: float = 1e-6
    atol: float = 1e-6
    norm: Callable[[PyTree], Shaped[Array, ""]] = field(
        static=True, default=optx.max_norm
    )

    def init(self, fn, y, args, options, f_struct, aux_struct, tags):
        """Delegate to ``inner.init(problem, y, options)``.

        Parameters
        ----------
        fn, args, f_struct, aux_struct, tags
            Ignored (problem is carried on the adapter).
        y
            Decision-variable starting point.
        options
            Option bag forwarded to the inner minimiser.

        Returns
        -------
        AbstractConstrainedMinimiser
            Initialised inner state.
        """
        return self.inner.init(self.problem, y, options)

    def step(self, fn, y, args, options, state, tags):
        """Take one inner step and return ``(y_new, state_new, aux)``.

        Parameters
        ----------
        fn, y, args, options, tags
            Ignored (``y`` is read from ``state.iterate``).
        state
            Current inner minimiser instance.

        Returns
        -------
        y_new
            Updated decision vector.
        state_new
            Updated inner minimiser.
        aux
            Always ``None``.
        """
        new = state.step(self.problem)
        return new.iterate.x, new, None

    def terminate(self, fn, y, args, options, state, tags):
        """Forward to ``state.terminate(problem)``.

        Parameters
        ----------
        fn, y, args, options, tags
            Ignored.
        state
            Current inner minimiser instance.

        Returns
        -------
        done, result
            Termination flag and status from the inner minimiser.
        """
        return state.terminate(self.problem)

    def postprocess(self, fn, y, aux, args, options, state, tags, result):
        """Forward to ``state.postprocess`` and unpack the solution.

        Parameters
        ----------
        fn, y, args, options, tags
            Ignored.
        aux
            Returned unchanged.
        state
            Final inner minimiser instance.
        result
            Status code.

        Returns
        -------
        value, aux, stats
            Decision vector, aux, and solution stats.
        """
        sol = state.postprocess(self.problem, result)
        return sol.value, aux, sol.stats


def as_optimistix_minimiser(
    minimiser: AbstractConstrainedMinimiser[Any, Any, Any],
    problem: ProblemProtocol[Any],
) -> optx.AbstractMinimiser:
    """Wrap a constrained minimiser for :func:`optimistix.minimise`.

    Parameters
    ----------
    minimiser
        Configured (pre-init) constrained minimiser.
    problem
        NLP to attach to the adapter.

    Returns
    -------
    optimistix.AbstractMinimiser
        Adapter instance usable with ``optimistix.minimise``.
    """
    rtol = float(getattr(minimiser, "rtol", 1e-6))
    atol = float(getattr(minimiser, "atol", 1e-6))
    return cast(
        optx.AbstractMinimiser,
        OptimistixMinimiser(
            problem=problem,
            inner=minimiser,
            rtol=rtol,
            atol=atol,
        ),
    )
