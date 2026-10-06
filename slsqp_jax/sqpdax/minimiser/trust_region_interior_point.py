"""Trust-region interior-point outer loop (Nocedal & Wright §19.5)."""

from __future__ import annotations

from typing import Any, Generic, Self, cast

import equinox as eqx
import optimistix as optx
from jax import numpy as jnp
from jaxtyping import Array, Bool, Int

from ..barrier import Barrier
from ..dual import Dual
from ..lagrangian import InteriorPointEvaluatedLagrangian, InteriorPointLagrangian
from ..merit import NormMerit
from ..primal import InteriorPointPrimal
from ..problem import ProblemProtocol
from ..results import (
    MINIMISER_RESULTS,
    ResultAdapter,
)
from ..secant import SecantResetSignals
from ..step_controller import StepController, StepResult, TrustRegionManager
from ..subproblem import ScaledBarrierSubProblem
from ..subproblem.solver import (
    RESULTS as SUBPROBLEM_RESULTS,
)
from ..subproblem.solver import (
    SubproblemContext,
    SubProblemSolver,
    TrustRegionInteriorPointSolver,
    TrustRegionSolverState,
    TrustRegionStateType,
)
from ..types import Scalar, Vector_n
from .base import OptimisationContext
from .interior_point import InteriorPointMinimiser
from .termination import TerminationFlags, TerminationMetrics

__all__ = [
    "TrustRegionInteriorPointMinimiser",
    "TrustRegionInteriorPointTerminationMetrics",
    "TRUST_REGION_INTERIOR_POINT_RESULTS",
    "TrustRegionInteriorPointResultAdapter",
]


class TRUST_REGION_INTERIOR_POINT_RESULTS(
    MINIMISER_RESULTS  # ty: ignore[subclass-of-final-class]
):
    """Fine-grained outcomes for the trust-region interior-point minimiser."""

    subproblem_max_steps = "The trust-region subproblem exhausted its step budget."
    subproblem_singular = "The trust-region subproblem was singular."
    subproblem_breakdown = "The trust-region subproblem solver broke down."
    subproblem_stagnation = "The trust-region subproblem solver stagnated."
    subproblem_condition_limit = (
        "The trust-region subproblem exceeded its condition-number limit."
    )
    subproblem_nonfinite = "The trust-region subproblem received non-finite input."


class TrustRegionInteriorPointTerminationMetrics(
    TerminationMetrics[TRUST_REGION_INTERIOR_POINT_RESULTS]
):
    """Termination measurements for :class:`TrustRegionInteriorPointMinimiser`.

    Mirrors the two nested stopping tests of Nocedal & Wright Algorithm 19.4:
    the inner loop runs until the *barrier* KKT error satisfies
    ``E(x, s, y, z; μ) ≤ ε_μ``, and the outer loop until the *unperturbed*
    error satisfies ``E(x, s, y, z; 0) ≤ ε_TOL``. A single residual field
    cannot express both, which is why this schema is algorithm-specific.

    Attributes
    ----------
    barrier_updated
        ``True`` when the last
        :class:`~slsqp_jax.sqpdax.barrier.update.BarrierUpdate` judged the
        barrier subproblem solved and reduced ``μ``. This *is* the inner
        ``E(·; μ) ≤ ε_μ`` test: the policy owns both the residual and the
        tolerance (``kappa_eps * μ`` for the monotone schedule), so ``ε_μ``
        tightens automatically as ``μ`` shrinks.
    optimality_residual
        ``E(x, s, y, z; 0)`` from N&W eq. 19.10, compared against ``atol``
        for the outer test.
    has_min_steps
        ``True`` once ``step_count >= min_steps``.
    """

    barrier_updated: Bool[Array, ""]
    optimality_residual: Scalar
    has_min_steps: Bool[Array, ""]


class TrustRegionInteriorPointResultAdapter(
    ResultAdapter[TRUST_REGION_INTERIOR_POINT_RESULTS]
):
    """Optimistix conversion for trust-region interior-point outcomes."""

    @property
    def result_type(self) -> type[TRUST_REGION_INTERIOR_POINT_RESULTS]:
        return TRUST_REGION_INTERIOR_POINT_RESULTS

    def to_optimistix(
        self, result: TRUST_REGION_INTERIOR_POINT_RESULTS
    ) -> optx.RESULTS:
        coarse = optx.RESULTS.nonlinear_divergence
        mappings = (
            (
                TRUST_REGION_INTERIOR_POINT_RESULTS.subproblem_max_steps,
                optx.RESULTS.max_steps_reached,
            ),
            (
                TRUST_REGION_INTERIOR_POINT_RESULTS.subproblem_singular,
                optx.RESULTS.singular,
            ),
            (
                TRUST_REGION_INTERIOR_POINT_RESULTS.subproblem_breakdown,
                optx.RESULTS.breakdown,
            ),
            (
                TRUST_REGION_INTERIOR_POINT_RESULTS.subproblem_stagnation,
                optx.RESULTS.stagnation,
            ),
            (
                TRUST_REGION_INTERIOR_POINT_RESULTS.subproblem_condition_limit,
                optx.RESULTS.conlim,
            ),
            (
                TRUST_REGION_INTERIOR_POINT_RESULTS.subproblem_nonfinite,
                optx.RESULTS.nonfinite_input,
            ),
        )
        for native, optimistix in mappings:
            coarse = optx.RESULTS.where(result == native, optimistix, coarse)
        coarse = optx.RESULTS.where(
            result == self.max_steps_reached,
            optx.RESULTS.nonlinear_max_steps_reached,
            coarse,
        )
        coarse = optx.RESULTS.where(
            result == self.nonfinite, optx.RESULTS.nonfinite, coarse
        )
        return optx.RESULTS.where(
            (result == self.successful) | (result == self.running),
            optx.RESULTS.successful,
            coarse,
        )


class TrustRegionInteriorPointMinimiser(
    InteriorPointMinimiser[
        ScaledBarrierSubProblem,
        TrustRegionStateType,
        TrustRegionInteriorPointTerminationMetrics,
        TRUST_REGION_INTERIOR_POINT_RESULTS,
    ],
    Generic[TrustRegionStateType],
):
    """Nocedal & Wright (2006) Section 19.5 trust-region interior-point method.

    Each outer step (i) evaluates the primal-dual barrier Lagrangian at the
    current ``(x, s)``, (ii) computes a composite normal+tangential step with
    :class:`~slsqp_jax.sqpdax.subproblem.solver.trust_region.TrustRegionInteriorPointSolver`
    for the current radius, (iii) accepts or rejects it with
    :class:`~slsqp_jax.sqpdax.step_controller.trust_region_radius.TrustRegionManager`
    from the actual/predicted reduction of the barrier merit and updates the
    radius, and (iv) reduces the barrier parameter ``μ`` via a pluggable
    :class:`~slsqp_jax.sqpdax.barrier.update.BarrierUpdate`. Slack seeding,
    initial multipliers and the barrier Lagrangian are inherited from
    :class:`~slsqp_jax.sqpdax.minimiser.interior_point.InteriorPointMinimiser`.

    Termination follows the two nested tests of N&W Algorithm 19.4. The inner
    one — solve the current barrier subproblem to ``E(x, s, y, z; μ) ≤ ε_μ``
    — is owned by :attr:`barrier_update` and surfaced as
    :attr:`barrier_updated`; the outer one compares the unperturbed KKT error
    ``E(x, s, y, z; 0)`` against ``atol``. There is no relative tolerance, so
    passing ``rtol`` raises.

    When the subproblem model uses a secant, the minimiser feeds the
    ``model`` channel of :class:`~slsqp_jax.sqpdax.secant.reset.SecantResetSignals`
    from two model-quality stalls detected after every step:

    * **radius collapse** — the trust-region radius has fallen below
      ``radius_floor * max(1, ||x||)`` while the unperturbed KKT error is
      still above ``atol``;
    * **model failure** — the step was rejected with an actual / predicted
      reduction ratio below ``model_failure_rho`` (including ``-inf`` when
      the model predicted no decrease).

    Consecutive stalls build ``consecutive_model_failures``; a healthy
    rejection or an accepted step with a sound radius resets it, so ordinary
    shrink-and-retry never triggers a reset. Subproblem failures and plain
    rejections are *not* reported to the reset policy: the former terminate
    immediately and the latter only shrink the radius.

    Attributes
    ----------
    initial_radius
        Initial trust-region radius.
    initial_penalty
        Initial merit penalty ``ν``.
    eta, shrink_threshold, grow_threshold, shrink_factor, grow_factor, max_radius
        Forwarded to :class:`TrustRegionManager`.
    radius_floor
        Relative radius floor for the radius-collapse stall
        (``radius < radius_floor * max(1, ||x||)``).
    model_failure_rho
        A rejected step with ``rho`` below this value counts as a model
        failure stall.
    secant_reset
        Shared soft / diagonal / identity / fatal recovery policy. Model
        stalls advance its one global recovery episode.
    consecutive_model_failures
        Dynamic count of consecutive model-quality stalls, reported to the
        secant reset policy as ``model_streak``.
    """

    initial_radius: float = eqx.field(static=True, default=1.0)
    initial_penalty: float = eqx.field(static=True, default=1.0)
    # trust-region acceptance / radius policy (forwarded to TrustRegionManager)
    eta: float = eqx.field(static=True, default=1e-4)
    shrink_threshold: float = eqx.field(static=True, default=0.25)
    grow_threshold: float = eqx.field(static=True, default=0.75)
    shrink_factor: float = eqx.field(static=True, default=0.25)
    grow_factor: float = eqx.field(static=True, default=2.0)
    max_radius: float = eqx.field(static=True, default=1e10)
    # secant model-stall detection (feeds the ``model`` reset channel)
    radius_floor: float = eqx.field(static=True, default=1e-8)
    model_failure_rho: float = eqx.field(static=True, default=-1.0)
    consecutive_model_failures: Int[Array, ""] = eqx.field(
        default_factory=lambda: jnp.asarray(0, jnp.int32)
    )

    @property
    def result_adapter(self) -> TrustRegionInteriorPointResultAdapter:
        """Native trust-region interior-point result policy."""
        return cast(
            TrustRegionInteriorPointResultAdapter,
            TrustRegionInteriorPointResultAdapter(),
        )

    def _subproblem_solver_type(self) -> type[SubProblemSolver]:
        """Root solver class for ``options['subproblem']`` validation."""
        return TrustRegionInteriorPointSolver

    def _init_dynamics(
        self, problem: ProblemProtocol[InteriorPointPrimal], x0: Vector_n
    ) -> Self:
        """Seed the secant and barrier (via super) and a zero model-stall streak."""
        base = super()._init_dynamics(problem, x0)
        return eqx.tree_at(
            lambda m: m.consecutive_model_failures,
            base,
            jnp.asarray(0, jnp.int32),
        )

    def _init_solver_state(
        self,
        problem: ProblemProtocol[InteriorPointPrimal],
        primal: InteriorPointPrimal,
    ) -> TrustRegionStateType:
        """Cold trust-region carry with :attr:`initial_radius` / :attr:`initial_penalty`.

        Subclasses binding a richer ``TrustRegionStateType`` must override
        this to build their own carry.
        """
        return cast(
            TrustRegionStateType,
            TrustRegionSolverState(
                n_iter=jnp.asarray(0, jnp.int32),
                radius=jnp.asarray(self.initial_radius),
                predicted_reduction=jnp.asarray(0.0),
                merit_penalty=jnp.asarray(self.initial_penalty),
                n_cg_iter=jnp.asarray(0, jnp.int32),
                on_boundary=jnp.asarray(False),
                success=jnp.asarray(False),
                status=SUBPROBLEM_RESULTS.successful,
                # Overwritten by ``TrustRegionManager.step`` before it is read.
                rho=jnp.asarray(1.0),
            ),
        )

    def _make_subproblem_solver(
        self, problem: ProblemProtocol[InteriorPointPrimal]
    ) -> TrustRegionInteriorPointSolver[TrustRegionStateType]:
        """Construct the default composite-step solver before ``options['subproblem']`` is applied.

        Parameters
        ----------
        problem
            NLP being minimised.

        Returns
        -------
        TrustRegionInteriorPointSolver
            Default solver. Its state type matches ``TrustRegionStateType``;
            subclasses binding a richer state must return a solver that
            consumes / produces it.
        """
        return cast(
            TrustRegionInteriorPointSolver[TrustRegionStateType],
            TrustRegionInteriorPointSolver(),
        )

    def _make_subproblem(
        self, lagrangian: InteriorPointEvaluatedLagrangian
    ) -> ScaledBarrierSubProblem:
        """Scaled-barrier model at the current ``(x, s)``."""
        return cast(ScaledBarrierSubProblem, ScaledBarrierSubProblem(lagrangian))

    def _step_controller(
        self,
        ctx: SubproblemContext[
            InteriorPointPrimal, ScaledBarrierSubProblem, TrustRegionStateType
        ],
        step_dual: Dual,
        solver_state: TrustRegionStateType,
    ) -> StepController[InteriorPointPrimal, TrustRegionStateType]:
        """Trust-region manager whose merit penalty matches the solver's ``ν``."""
        nu = solver_state.merit_penalty
        merit = NormMerit(
            ctx.problem,
            barrier=self.barrier,
            problem_weight=jnp.asarray(1.0),
            barrier_weight=jnp.asarray(1.0),
            feasibility_weight=nu,
            norm=2,
        )
        return cast(
            StepController[InteriorPointPrimal, TrustRegionStateType],
            TrustRegionManager(
                merit=merit,
                logger=self.logger.child("step_controller"),
                eta=self.eta,
                shrink_threshold=self.shrink_threshold,
                grow_threshold=self.grow_threshold,
                shrink_factor=self.shrink_factor,
                grow_factor=self.grow_factor,
                max_radius=self.max_radius,
            ),
        )

    def _advance_dynamics(
        self,
        ctx: SubproblemContext[
            InteriorPointPrimal, ScaledBarrierSubProblem, TrustRegionStateType
        ],
        result: StepResult[InteriorPointPrimal, TrustRegionStateType],
        step_dual: Dual,
    ) -> Self:
        """Reduce ``μ`` and refresh the model-stall streak at the new iterate.

        Also records whether the policy judged the barrier subproblem solved,
        which :meth:`termination_metrics` reads back as the inner
        ``E(·; μ) ≤ ε_μ`` test of N&W Algorithm 19.4. That test is evaluated
        at the iterate, so it runs on rejected steps too — ``result.x`` is
        then the retained iterate.

        The model-stall streak counts consecutive steps at which either the
        radius collapsed while the unperturbed KKT error ``E(·; 0)`` was still
        above ``atol``, or the step was rejected with ``rho`` below
        :attr:`model_failure_rho`. The unperturbed residual is used (rather
        than the barrier policy's ``updated`` flag) because adaptive
        schedules report ``updated`` unconditionally.

        Parameters
        ----------
        ctx
            Per-step subproblem context.
        result
            Outcome of the trust-region controller.
        step_dual
            Multipliers committed with ``result.x``.

        Returns
        -------
        Self
            Minimiser with ``barrier``, ``barrier_updated`` and
            ``consecutive_model_failures`` refreshed.
        """
        # ``ctx.lagrangian`` is the module; re-evaluate at (x_new, step_dual)
        # exactly as ``_update_secant`` does. Complementarity is secant-independent,
        # so using the step's module (pre-update barrier) is correct.
        x_new = result.x
        lag_module = cast(InteriorPointLagrangian, ctx.lagrangian)
        lagrangian = lag_module(x_new, step_dual)
        new_barrier, updated = self.barrier_update.update(
            cast(Barrier, self.barrier), lagrangian
        )

        # The trust-region controller always writes the solver state back.
        state = cast(TrustRegionSolverState, result.solver_state)
        unconverged = (
            self.barrier_update.optimality_residual(lagrangian, jnp.asarray(0.0))
            > self.atol
        )
        x_scale = jnp.maximum(1.0, jnp.linalg.norm(x_new.x))
        radius_collapse = (state.radius < self.radius_floor * x_scale) & unconverged
        model_failure = ~result.accepted & (state.rho < self.model_failure_rho)
        stall = radius_collapse | model_failure
        streak = jnp.where(stall, self.consecutive_model_failures + 1, 0).astype(
            jnp.int32
        )
        self.logger.warning(
            "model stall: radius_collapse={collapse} model_failure={failure} "
            "rho={rho:.3e} radius={radius:.3e} consecutive={streak}",
            when=stall,
            collapse=radius_collapse,
            failure=model_failure,
            rho=state.rho,
            radius=state.radius,
            streak=streak,
        )
        self.logger.info(
            "barrier reduced: mu={mu:.3e}",
            when=updated,
            mu=new_barrier.weight,
        )
        return eqx.tree_at(
            lambda m: (m.barrier, m.barrier_updated, m.consecutive_model_failures),
            self,
            (new_barrier, updated, streak),
        )

    def _step_log_fields(
        self,
        ctx: SubproblemContext[
            InteriorPointPrimal, ScaledBarrierSubProblem, TrustRegionStateType
        ],
        result: StepResult[InteriorPointPrimal, TrustRegionStateType],
        step_dual: Dual,
        octx: OptimisationContext[InteriorPointPrimal, TrustRegionStateType],
        metrics: TrustRegionInteriorPointTerminationMetrics,
    ) -> dict[str, Any]:
        """Barrier weight, KKT error, trust-region radius / ratio and penalty."""
        state = cast(TrustRegionSolverState, result.solver_state)
        return {
            "kkt": (metrics.optimality_residual, ".2e"),
            "mu": (cast(Barrier, self.barrier).weight, ".2e"),
            "mu_updated": metrics.barrier_updated,
            "radius": (state.radius, ".2e"),
            "rho": (state.rho, ".2e"),
            "nu": (state.merit_penalty, ".2e"),
            "cg_total": state.n_cg_iter,
            "model_fail": self.consecutive_model_failures,
        }

    def _diagnostic_fields(
        self,
        ctx: SubproblemContext[
            InteriorPointPrimal, ScaledBarrierSubProblem, TrustRegionStateType
        ],
        result: StepResult[InteriorPointPrimal, TrustRegionStateType],
        step_dual: Dual,
        octx: OptimisationContext[InteriorPointPrimal, TrustRegionStateType],
        metrics: TrustRegionInteriorPointTerminationMetrics,
    ) -> dict[str, Any]:
        """Base payload plus barrier weight, trust-region state and slacks."""
        fields = super()._diagnostic_fields(ctx, result, step_dual, octx, metrics)
        state = cast(TrustRegionSolverState, result.solver_state)
        fields.update(
            {
                "barrier_weight": cast(Barrier, self.barrier).weight,
                "barrier_updated": metrics.barrier_updated,
                "radius": state.radius,
                "rho": state.rho,
                "nu": state.merit_penalty,
                "predicted_reduction": state.predicted_reduction,
                "cg_total": state.n_cg_iter,
                "consecutive_model_failures": self.consecutive_model_failures,
                "slack": cast(InteriorPointPrimal, self.iterate).slack,
            }
        )
        return fields

    def _secant_reset_signals(self) -> SecantResetSignals:
        """Report model-quality stalls on the ``model`` channel only.

        Subproblem failures terminate the run and plain rejections only
        shrink the radius, so both of those channels stay at zero.

        Returns
        -------
        SecantResetSignals
            Raw ``model_streak`` from :attr:`consecutive_model_failures`.
        """
        zero = jnp.asarray(0, jnp.int32)
        return cast(
            SecantResetSignals,
            SecantResetSignals(
                subproblem_streak=zero,
                step_streak=zero,
                model_streak=self.consecutive_model_failures,
            ),
        )

    def termination_metrics(
        self, ctx: OptimisationContext[InteriorPointPrimal, TrustRegionStateType]
    ) -> TrustRegionInteriorPointTerminationMetrics:
        """Measure the unperturbed KKT error and the inner-loop state.

        Parameters
        ----------
        ctx
            Termination context at the current iterate.

        Returns
        -------
        TrustRegionInteriorPointTerminationMetrics
            ``E(x, s, y, z; 0)`` plus the barrier / non-finite / subproblem
            state the convergence test needs.
        """
        primal = ctx.lagrangian.ref
        lag = cast(InteriorPointEvaluatedLagrangian, ctx.lagrangian)

        # The residual is computed under 0 barrier weight as in algorithm 19.4 of N&W,
        # and the residual formula is taken from equation 19.10 in N&W.
        optimality_residual = self.barrier_update.optimality_residual(
            lag, jnp.asarray(0.0)
        )
        nonfinite = self._any_nonfinite(
            ctx.lagrangian.value,
            ctx.lagrangian.x_grad,
            primal,
            self.dual,
            optimality_residual,
        )
        status = cast(TrustRegionStateType, ctx.solver_state).status
        fatal_result = TRUST_REGION_INTERIOR_POINT_RESULTS.subproblem_max_steps
        mappings = (
            (
                SUBPROBLEM_RESULTS.singular,
                TRUST_REGION_INTERIOR_POINT_RESULTS.subproblem_singular,
            ),
            (
                SUBPROBLEM_RESULTS.breakdown,
                TRUST_REGION_INTERIOR_POINT_RESULTS.subproblem_breakdown,
            ),
            (
                SUBPROBLEM_RESULTS.stagnation,
                TRUST_REGION_INTERIOR_POINT_RESULTS.subproblem_stagnation,
            ),
            (
                SUBPROBLEM_RESULTS.conlim,
                TRUST_REGION_INTERIOR_POINT_RESULTS.subproblem_condition_limit,
            ),
            (
                SUBPROBLEM_RESULTS.nonfinite_input,
                TRUST_REGION_INTERIOR_POINT_RESULTS.subproblem_nonfinite,
            ),
        )
        for inner, native in mappings:
            fatal_result = TRUST_REGION_INTERIOR_POINT_RESULTS.where(
                status == inner, native, fatal_result
            )
        return cast(
            TrustRegionInteriorPointTerminationMetrics,
            TrustRegionInteriorPointTerminationMetrics(
                barrier_updated=self.barrier_updated,
                optimality_residual=optimality_residual,
                nonfinite=nonfinite,
                fatal_result=fatal_result,
                has_min_steps=self.step_count >= self.min_steps,
            ),
        )

    def termination_flags(
        self,
        ctx: OptimisationContext[InteriorPointPrimal, TrustRegionStateType],
        metrics: TrustRegionInteriorPointTerminationMetrics,
    ) -> TerminationFlags[TRUST_REGION_INTERIOR_POINT_RESULTS]:
        """Require *both* Algorithm 19.4 stopping tests before declaring success.

        Parameters
        ----------
        ctx
            Termination context at the current iterate.
        metrics
            Output of :meth:`termination_metrics`.

        Returns
        -------
        TerminationFlags
            Shared termination decision.
        """
        barrier_updated = metrics.barrier_updated
        acceptable_residual = metrics.optimality_residual <= self.atol
        nonfinite = metrics.nonfinite
        recovery_fatal = self.secant_recovery_state.fatal
        subproblem_fatal = (
            cast(TrustRegionStateType, ctx.solver_state).status
            != SUBPROBLEM_RESULTS.successful
        )
        converged = (
            barrier_updated
            & acceptable_residual
            & metrics.has_min_steps
            & ~nonfinite
            & ~subproblem_fatal
            & ~recovery_fatal
        )
        fatal_result = TRUST_REGION_INTERIOR_POINT_RESULTS.where(
            recovery_fatal,
            TRUST_REGION_INTERIOR_POINT_RESULTS.secant_recovery_failure,
            metrics.fatal_result,
        )
        return cast(
            TerminationFlags,
            TerminationFlags(
                converged=converged,
                nonfinite=nonfinite,
                fatal=subproblem_fatal | recovery_fatal,
                fatal_result=fatal_result,
            ),
        )
