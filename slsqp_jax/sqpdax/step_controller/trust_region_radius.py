"""Trust-region radius update and step acceptance."""

from typing import cast

import equinox as eqx
import jax
from jax import numpy as jnp

from ..primal import Primal
from ..subproblem.solver.trust_region import TrustRegionSolverState
from .base import StepController, StepResult


class TrustRegionManager(StepController[Primal, TrustRegionSolverState]):
    """Trust-region step controller (N&W eqs. 19.39–19.40).

    Consumes the proposed step and the incoming
    :class:`~slsqp_jax.sqpdax.subproblem.solver.trust_region.TrustRegionSolverState`
    (which carries ``predicted_reduction`` and ``radius`` from the subproblem
    solver), computes the actual/predicted-reduction ratio ``ρ`` from the
    merit, accepts the step iff ``ρ ≥ η``, and returns a new
    :class:`~slsqp_jax.sqpdax.subproblem.solver.trust_region.TrustRegionSolverState`
    with the updated radius. On a reject the iterate is left unchanged
    (``x == x0``); the outer loop re-solves the subproblem at the shrunk
    radius on its next iteration — the retry *is* the outer loop, so there
    is no inner loop here.

    Radius update (with ``Δ`` the incoming radius):

    ```
    Δ₊ = shrink_factor · Δ          if ρ < shrink_threshold
       = min(grow_factor · Δ, max_radius)
                                       if ρ > grow_threshold and on_boundary
       = Δ                             otherwise
    ```

    Because a rejected step leaves ``x`` unchanged, the secant pair
    ``s = x - x0`` is zero and is naturally skipped by
    :meth:`~slsqp_jax.sqpdax.secant.lbfgs.LBFGS.should_skip`; no
    special-casing is needed in the outer loop.

    Attributes
    ----------
    merit
        Merit used for the actual reduction (inherited).
    eta
        Step-acceptance threshold on ``ρ``.
    shrink_threshold
        ``ρ`` below this value shrinks the radius.
    grow_threshold
        ``ρ`` above this value (and ``on_boundary``) grows the radius.
    shrink_factor
        Multiplicative shrink applied when ``ρ`` is too small.
    grow_factor
        Multiplicative grow applied on a successful boundary step.
    max_radius
        Cap on the grown radius.
    """

    eta: float = eqx.field(default=1e-4)  # step-acceptance threshold
    shrink_threshold: float = eqx.field(default=0.25)  # rho below -> shrink radius
    grow_threshold: float = eqx.field(default=0.75)  # rho above (+ boundary) -> grow
    shrink_factor: float = eqx.field(default=0.25)
    grow_factor: float = eqx.field(default=2.0)
    max_radius: float = eqx.field(default=1e10)

    def step(
        self,
        x0: Primal,
        direction: Primal,
        solver_state: TrustRegionSolverState | None = None,
    ) -> StepResult[Primal, TrustRegionSolverState]:
        """Accept or reject ``direction`` and update the trust-region radius.

        Parameters
        ----------
        x0
            Current primal iterate.
        direction
            Proposed composite trust-region step.
        solver_state
            Must be a
            :class:`~slsqp_jax.sqpdax.subproblem.solver.trust_region.TrustRegionSolverState`
            carrying ``predicted_reduction``, ``radius``, and ``on_boundary``.

        Returns
        -------
        StepResult
            New (or retained) iterate, acceptance flag, merit at that
            iterate, and updated solver state (new ``radius`` /
            ``success``).

        Raises
        ------
        TypeError
            If ``solver_state`` is not a ``TrustRegionSolverState``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from slsqp_jax.sqpdax.merit import NormMerit
        >>> from slsqp_jax.sqpdax.primal import Primal
        >>> from slsqp_jax.sqpdax.problem import build_problem
        >>> from slsqp_jax.sqpdax.step_controller import TrustRegionManager
        >>> from slsqp_jax.sqpdax.subproblem.solver import RESULTS, TrustRegionSolverState
        >>> problem = build_problem(
        ...     n=2, meq=None, mineq=None,
        ...     fn=lambda x: jnp.sum(x**2), grad=lambda x: 2 * x, hvp=None,
        ...     eq_fn=None, ineq_fn=None, eq_fn_jac=None, ineq_fn_jac=None,
        ...     eq_fn_hvp=None, ineq_fn_hvp=None, lb=None, ub=None,
        ... )
        >>> mgr = TrustRegionManager(merit=NormMerit(problem=problem))
        >>> state = TrustRegionSolverState(
        ...     n_iter=jnp.asarray(0, jnp.int32),
        ...     success=jnp.asarray(False),
        ...     status=RESULTS.successful,
        ...     radius=jnp.asarray(1.0),
        ...     predicted_reduction=jnp.asarray(0.75),
        ...     merit_penalty=jnp.asarray(1.0),
        ...     n_cg_iter=jnp.asarray(0, jnp.int32),
        ...     on_boundary=jnp.asarray(False),
        ... )
        >>> result = mgr.step(
        ...     Primal(jnp.array([1.0, 0.0])),
        ...     Primal(jnp.array([-0.5, 0.0])),
        ...     state,
        ... )
        >>> bool(result.accepted)
        True
        >>> float(result.solver_state.radius)
        1.0
        """
        if not isinstance(solver_state, TrustRegionSolverState):
            raise TypeError(
                "TrustRegionManager.step requires a TrustRegionSolverState "
                "(carrying predicted_reduction and radius). Got "
                f"{type(solver_state)} instead."
            )
        pred = solver_state.predicted_reduction
        radius = solver_state.radius

        merit0 = self.merit(x0)
        x_trial = jax.tree.map(jnp.add, x0, direction)
        merit_trial = self.merit(x_trial)
        actual_reduction = merit0 - merit_trial

        # rho = actual / predicted reduction.  A non-positive predicted reduction
        # means the model failed; treat rho as -inf so the step is rejected and the
        # radius shrinks.
        tiny = jnp.asarray(jnp.finfo(pred.dtype).tiny, pred.dtype)
        good_pred = pred > tiny
        rho = jnp.where(
            good_pred,
            actual_reduction / jnp.where(good_pred, pred, 1.0),
            -jnp.inf,
        )

        accepted = rho >= self.eta
        x_new = jax.tree.map(lambda a, b: jnp.where(accepted, b, a), x0, x_trial)
        merit_new = jnp.where(accepted, merit_trial, merit0)

        new_radius = jnp.where(
            rho < self.shrink_threshold,
            self.shrink_factor * radius,
            jnp.where(
                (rho > self.grow_threshold) & solver_state.on_boundary,
                jnp.minimum(self.grow_factor * radius, self.max_radius),
                radius,
            ),
        )
        new_state = eqx.tree_at(
            lambda s: (s.radius, s.success),
            solver_state,
            (new_radius, accepted),
        )
        return cast(
            StepResult[Primal, TrustRegionSolverState],
            StepResult(
                x=x_new,
                accepted=accepted,
                merit_val=merit_new,
                solver_state=new_state,
            ),
        )
