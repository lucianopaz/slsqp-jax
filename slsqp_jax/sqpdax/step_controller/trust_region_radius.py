"""Trust-region radius update and step acceptance."""

from collections.abc import Callable
from typing import cast

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Scalar

from ..primal import Primal
from ..subproblem.solver.trust_region import TrustRegionSolverState
from .base import MeritStepController, StepResult, at_rounding_level


class TrustRegionManager(MeritStepController[Primal, TrustRegionSolverState]):
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

    The ratio ``ρ`` is written back to the returned state (``rho``) so the
    outer loop can detect model failures (``ρ = -inf`` when the predicted
    reduction is non-positive) without recomputing the merit.

    With a :attr:`~slsqp_jax.sqpdax.step_controller.base.StepController.correction`
    a step with ``ρ < η`` is retried at the corrected point ``x0 + d + d̂``
    against the *same* predicted reduction (Byrd, Hribar & Nocedal 1999,
    p. 892; scipy ``trust-constr``); the corrected point is taken when its
    ratio reaches ``η``, and the radius update then uses that ratio.

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

    supports_correction = True

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
            ``success`` / ``rho``). ``rounding_level`` flags a predicted
            reduction below the rounding level of the merit (see
            :func:`~slsqp_jax.sqpdax.step_controller.base.at_rounding_level`).

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
        ...     n=2,
        ...     fn=lambda x: jnp.sum(x**2),
        ...     grad=lambda x: 2 * x,
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
        >>> float(result.solver_state.rho)
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

        # rho = actual / predicted reduction.  A non-positive predicted reduction
        # means the model failed; treat rho as -inf so the step is rejected and the
        # radius shrinks.
        tiny = jnp.asarray(jnp.finfo(pred.dtype).tiny, pred.dtype)
        good_pred = pred > tiny

        def ratio(merit: Scalar) -> Scalar:
            return jnp.where(
                good_pred,
                (merit0 - merit) / jnp.where(good_pred, pred, 1.0),
                -jnp.inf,
            )

        rho = ratio(merit_trial)
        corrected = jnp.asarray(False)
        if self.correction is not None:
            x_trial, merit_trial, rho, corrected = self._try_correction(
                x0, direction, x_trial, merit_trial, rho, ratio
            )
        actual_reduction = merit0 - merit_trial

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
        self.logger.info(
            "rho={rho:.3e} accepted={accepted} radius {radius:.3e} -> "
            "{new_radius:.3e} actual={actual:.3e} predicted={pred:.3e}",
            rho=rho,
            accepted=accepted,
            radius=radius,
            new_radius=new_radius,
            actual=actual_reduction,
            pred=pred,
        )
        self.logger.warning(
            "rejected step: rho={rho:.3e} radius shrunk to {new_radius:.3e}",
            when=~accepted,
            rho=rho,
            new_radius=new_radius,
        )
        new_state = eqx.tree_at(
            lambda s: (s.radius, s.success, s.rho),
            solver_state,
            (new_radius, accepted, jnp.asarray(rho, solver_state.rho.dtype)),
        )
        proposed_step_norm = jnp.linalg.norm(x_new.x - x0.x)
        return cast(
            StepResult[Primal, TrustRegionSolverState],
            StepResult(
                x=x_new,
                accepted=accepted,
                merit_val=merit_new,
                solver_state=new_state,
                step_size=jnp.where(accepted, jnp.asarray(1.0), jnp.asarray(0.0)),
                proposed_step_norm=proposed_step_norm,
                rounding_level=at_rounding_level(pred, merit0),
                corrected=accepted & corrected,
            ),
        )

    def _try_correction(
        self,
        x0: Primal,
        direction: Primal,
        x_trial: Primal,
        merit_trial: Scalar,
        rho: Scalar,
        ratio: Callable[[Scalar], Scalar],
    ) -> tuple[Primal, Scalar, Scalar, Bool[Array, ""]]:
        """Retry a step with ``ρ < η`` at its second-order corrected point.

        Parameters
        ----------
        x0, direction
            Current iterate and proposed step.
        x_trial, merit_trial, rho
            Uncorrected trial point, its merit and its ratio.
        ratio
            ``merit ↦ ρ`` against this step's predicted reduction.

        Returns
        -------
        tuple
            ``(x, merit, ρ, corrected)`` of the corrected point when its ratio
            reaches :attr:`eta`, otherwise of the uncorrected trial.
        """

        def attempt() -> tuple[Primal, Scalar, Scalar, Bool[Array, ""]]:
            x_soc, result = self._corrected_point(x0, direction)
            merit_soc = self.merit(x_soc)
            rho_soc = ratio(merit_soc)
            ok = result.valid & (rho_soc >= self.eta)
            self.logger.info(
                "second-order correction accepted: |c(x+d)|={res:.3e} "
                "fraction={t:.3e} rho {rho:.3e} -> {rho_soc:.3e}",
                when=ok,
                res=result.residual_norm,
                t=result.fraction,
                rho=rho,
                rho_soc=rho_soc,
            )
            x = jax.tree.map(lambda a, b: jnp.where(ok, a, b), x_soc, x_trial)
            return (
                cast(Primal, x),
                jnp.where(ok, merit_soc, merit_trial),
                jnp.where(ok, rho_soc, rho),
                ok,
            )

        def keep() -> tuple[Primal, Scalar, Scalar, Bool[Array, ""]]:
            return x_trial, merit_trial, rho, jnp.asarray(False)

        return jax.lax.cond(rho < self.eta, attempt, keep)
