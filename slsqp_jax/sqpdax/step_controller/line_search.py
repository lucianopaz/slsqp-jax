"""Backtracking line-search step controllers."""

from abc import abstractmethod
from typing import cast

import equinox as eqx
import jax
from equinox import Module
from jax import numpy as jnp
from jaxtyping import Array, Bool, Int, Scalar

from ..primal import Primal
from ..subproblem.solver import SubProblemSolverState
from .base import StepController, StepResult


class LineSearchState(Module):
    """Carry for one backtracking line-search episode.

    Built by :meth:`LineSearch.step` and updated by concrete
    :meth:`~LineSearch.trial` / :meth:`~LineSearch.stop_search` hooks inside
    a :func:`jax.lax.while_loop`. Required fields are set at construction;
    defaulted bookkeeping fields follow so Equinox can leave them off the
    constructor call.

    Attributes
    ----------
    x0
        Iterate at the start of the line search.
    step
        Full (``alpha == 1``) proposed direction; never rescaled in place.
    merit0_val
        Merit at ``x0``.
    merit0_grad
        Merit gradient at ``x0`` (primal pytree).
    merit0_grad_dot_step
        Directional derivative ``⟨∇φ(x0), step⟩``.
    merit_val
        Merit at the current trial ``x0 + alpha * step``. Seeded with
        ``+∞`` so :meth:`~LineSearch.stop_search` cannot accept before the
        first trial evaluation.
    merit_grad
        Merit gradient at the current trial (may be left at ``merit0_grad``
        when a concrete search does not need it).
    alpha
        Current step length along ``step`` (starts at ``1``).
    iteration
        Number of trial evaluations completed.
    success
        Optional flag reserved for concrete searches.
    break_loop
        Optional early-exit flag reserved for concrete searches.
    """

    # --- required (populated by LineSearch.step) ---
    x0: Primal
    step: Primal
    merit0_val: Scalar
    merit0_grad: Primal
    merit0_grad_dot_step: Scalar
    merit_val: Scalar
    merit_grad: Primal
    # --- defaulted bookkeeping (must follow the required fields) ---
    alpha: Scalar = eqx.field(default_factory=lambda: jnp.asarray(1.0))
    iteration: Int[Array, ""] = eqx.field(
        default_factory=lambda: jnp.asarray(0, jnp.int32)
    )
    success: Bool[Array, ""] = eqx.field(default_factory=lambda: jnp.asarray(False))
    break_loop: Bool[Array, ""] = eqx.field(default_factory=lambda: jnp.asarray(False))

    @property
    def x(self) -> Primal:
        """Current trial iterate ``x0 + alpha * step``.

        A plain property (not ``cached_property``): caching would write into
        the frozen Module's ``__dict__``, which Equinox folds into the pytree
        metadata and breaks the ``while_loop`` carry-structure invariant.
        """
        scaled = jax.tree.map(lambda s: self.alpha * s, self.step)
        return jax.tree.map(jnp.add, self.x0, scaled)


class LineSearch(StepController[Primal, SubProblemSolverState]):
    """Abstract backtracking line search along a fixed direction.

    Template method: :meth:`step` evaluates the merit and its gradient at
    ``x0``, seeds a :class:`LineSearchState`, and runs
    :func:`jax.lax.while_loop` until :meth:`stop_search` accepts or
    ``max_steps`` trials are exhausted. Concrete subclasses implement
    :meth:`trial` (propose the next ``alpha`` / merit) and
    :meth:`stop_search` (acceptance test).

    The incoming ``solver_state`` is returned unchanged — a line search does
    not own subproblem-solver control such as a trust-region radius.

    Attributes
    ----------
    merit
        Merit used to score trial iterates (inherited).
    max_steps
        Maximum number of trial evaluations inside the search loop.
    """

    max_steps: int = eqx.field(default=20)

    @abstractmethod
    def stop_search(self, state: LineSearchState) -> Bool[Array, ""]:
        """Whether the current trial in ``state`` should be accepted.

        Parameters
        ----------
        state
            Line-search carry after zero or more :meth:`trial` calls.

        Returns
        -------
        Bool[Array, ""]
            ``True`` to exit the search with the current ``alpha``.
        """
        ...

    @abstractmethod
    def trial(self, state: LineSearchState) -> LineSearchState:
        """Propose the next trial step length and evaluate its merit.

        Parameters
        ----------
        state
            Carry from the previous iteration (or the seeded initial state).

        Returns
        -------
        LineSearchState
            Updated carry; typically refreshes ``alpha`` and ``merit_val``.
        """
        ...

    def step(
        self,
        x0: Primal,
        direction: Primal,
        solver_state: SubProblemSolverState | None = None,
    ) -> StepResult[Primal, SubProblemSolverState]:
        """Backtrack along ``direction`` until the acceptance test passes.

        Parameters
        ----------
        x0
            Current primal iterate.
        direction
            Full proposed step (corresponding to ``alpha == 1``).
        solver_state
            Optional subproblem-solver carry; threaded through untouched.

        Returns
        -------
        StepResult
            Trial iterate ``x0 + alpha * direction``, whether
            :meth:`stop_search` accepted it, merit at that point, and the
            unchanged ``solver_state``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from slsqp_jax.sqpdax.merit import NormMerit
        >>> from slsqp_jax.sqpdax.primal import Primal
        >>> from slsqp_jax.sqpdax.problem import build_problem
        >>> from slsqp_jax.sqpdax.step_controller import ArmijoLineSearch
        >>> problem = build_problem(
        ...     n=2, meq=None, mineq=None,
        ...     fn=lambda x: jnp.sum(x**2), grad=lambda x: 2 * x, hvp=None,
        ...     eq_fn=None, ineq_fn=None, eq_fn_jac=None, ineq_fn_jac=None,
        ...     eq_fn_hvp=None, ineq_fn_hvp=None, lb=None, ub=None,
        ... )
        >>> ls = ArmijoLineSearch(merit=NormMerit(problem=problem))
        >>> result = ls.step(Primal(jnp.ones(2)), Primal(-jnp.ones(2)))
        >>> bool(result.accepted)
        True
        >>> float(result.merit_val)  # doctest: +ELLIPSIS
        0.0...
        """
        merit0_val, merit0_grad = eqx.filter_value_and_grad(self.merit)(x0)
        state = LineSearchState(
            x0=x0,
            step=direction,
            merit0_val=merit0_val,
            merit0_grad=merit0_grad,
            merit0_grad_dot_step=jnp.inner(merit0_grad.flatten(), direction.flatten()),
            # No trial evaluated yet: seed with +inf so the loop always runs at
            # least once and stop_search cannot spuriously accept at init (e.g.
            # for a non-descent direction, where merit0_grad_dot_step > 0 would
            # make the sufficient-decrease test trivially true at alpha == 1).
            merit_val=jnp.asarray(jnp.inf, merit0_val.dtype),
            merit_grad=merit0_grad,
        )

        def body_fun(state):
            new_state = self.trial(state)
            new_state = eqx.tree_at(
                lambda s: s.iteration, new_state, state.iteration + 1
            )
            return new_state

        def cond_fun(state):
            return jnp.logical_and(
                jnp.logical_not(self.stop_search(state)),
                state.iteration < self.max_steps,
            )

        state = cast(
            LineSearchState,
            jax.lax.while_loop(
                cond_fun=cond_fun,
                body_fun=body_fun,
                init_val=state,
            ),
        )
        # ``accepted`` records whether we exited via acceptance (True) or budget
        # exhaustion.  ``solver_state`` is threaded through untouched -- a line
        # search does not own any persistent subproblem-solver control.
        accepted = self.stop_search(state)
        return cast(
            StepResult[Primal, SubProblemSolverState],
            StepResult(
                x=state.x,
                accepted=accepted,
                merit_val=state.merit_val,
                solver_state=solver_state,
            ),
        )


class ArmijoLineSearch(LineSearch):
    """Geometric backtracking with the Armijo sufficient-decrease condition.

    Accepts a trial when

    ```
    φ(x0 + α d) ≤ φ(x0) + c₁ α ⟨∇φ(x0), d⟩
    ```

    or, as a fallback, when the merit has decreased at all and ``α < 0.1``.
    The first trial uses ``α = 1``; subsequent trials multiply ``α`` by
    ``backtrack``.

    Attributes
    ----------
    c1
        Armijo constant in ``(0, 1)``.
    backtrack
        Geometric contraction factor in ``(0, 1)`` applied after the first
        unsuccessful trial.
    """

    c1: Scalar = eqx.field(default=1e-4)
    backtrack: Scalar = eqx.field(default=0.5)

    def stop_search(self, state: LineSearchState) -> Bool[Array, ""]:
        """Armijo test (with small-``α`` decrease fallback).

        Parameters
        ----------
        state
            Current line-search carry.

        Returns
        -------
        Bool[Array, ""]
            ``True`` when the trial satisfies Armijo or the fallback.
        """
        sufficient_decrease = state.merit0_val + (
            self.c1 * state.alpha * state.merit0_grad_dot_step
        )

        # Check if current alpha satisfies the condition
        armijo_satisfied = state.merit_val <= sufficient_decrease

        # Also accept if merit decreased at all (fallback)
        merit_decreased = state.merit_val < state.merit0_val

        # Accept if Armijo is satisfied, or if we've improved and alpha is small
        accept = armijo_satisfied | (merit_decreased & (state.alpha < 0.1))
        return accept

    def trial(self, state: LineSearchState) -> LineSearchState:
        """Evaluate the next geometrically contracted trial step.

        The first body call keeps ``alpha`` at its seeded value of ``1``;
        every subsequent call multiplies by ``backtrack``. The invariant
        kept is ``state.merit_val == merit(x0 + state.alpha * step)``.

        Parameters
        ----------
        state
            Carry from the previous iteration.

        Returns
        -------
        LineSearchState
            Carry with updated ``alpha`` and ``merit_val``.
        """
        alpha = jnp.where(
            state.iteration == 0,
            state.alpha,
            state.alpha * self.backtrack,
        )
        state = eqx.tree_at(lambda s: s.alpha, state, alpha)
        merit_val = self.merit(state.x)
        return eqx.tree_at(lambda s: s.merit_val, state, merit_val)
