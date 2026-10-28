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
    trust_region_collapse = (
        "The trust region collapsed on a barrier subproblem whose parameter is "
        "already below the tolerance."
    )


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
            (
                TRUST_REGION_INTERIOR_POINT_RESULTS.trust_region_collapse,
                optx.RESULTS.stagnation,
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

    Termination follows N&W Algorithm 19.4: the inner test — solve the
    current barrier subproblem to ``E(x, s, y, z; μ) ≤ ε_μ`` — is owned by
    :attr:`barrier_update` and surfaced as :attr:`barrier_updated`; the
    outer test compares the unperturbed KKT error ``E(x, s, y, z; 0)``
    against ``atol`` and alone decides convergence. There is no relative
    tolerance, so passing ``rtol`` raises.

    The inner loop also has an ``xtol``-style exit (trust-constr's
    ``tr_radius < xtol``): when the trust-region radius has fallen below
    ``radius_floor * max(1, ||x||)``, or a step was rejected with a
    predicted reduction at the rounding level of the merit
    (``StepResult.rounding_level``), the current barrier subproblem is
    solved to working precision whatever its KKT error says. ``μ`` is then
    reduced through :attr:`barrier_update` (``force=True``) and the radius
    returns to :attr:`initial_radius`. The run stops with
    :attr:`TRUST_REGION_INTERIOR_POINT_RESULTS.trust_region_collapse` when
    the exit fires again with ``μ`` already at or below ``atol`` and no step
    was accepted since the previous exit — reducing ``μ`` did not unlock
    any progress — or when the schedule could not reduce ``μ`` any further.

    When the subproblem model uses a secant, a radius collapse on a
    *rejected* step whose predicted reduction was above rounding level (the
    model claimed a decrease that never materialised, down to the floor) is
    reported to the ``model`` channel of
    :class:`~slsqp_jax.sqpdax.secant.reset.SecantResetSignals` as a model
    stall (switch with :attr:`collapse_resets_secant`). Ordinary rejections
    only shrink the radius and subproblem failures terminate immediately, so
    neither is reported.

    Attributes
    ----------
    initial_radius
        Initial trust-region radius.
    initial_penalty
        Initial merit penalty ``ν``.
    eta, shrink_threshold, grow_threshold, shrink_factor, grow_factor, max_radius
        Forwarded to :class:`TrustRegionManager`.
    radius_floor
        Relative radius floor of the inner-loop exit
        (``radius < radius_floor * max(1, ||x||)``).
    collapse_resets_secant
        Report a radius collapse on a rejected, non-rounding-level step as a
        model stall to the secant reset policy.
    secant_reset
        Shared soft / diagonal / identity / fatal recovery policy. Model
        stalls advance its one global recovery episode.
    consecutive_model_failures
        Dynamic count of consecutive model stalls, reported to the secant
        reset policy as ``model_streak``.
    exit_without_progress
        Dynamic flag: the last inner-loop exit has not been followed by an
        accepted step yet.
    trust_region_collapsed
        Dynamic flag set when the inner-loop exit exhausted the barrier
        schedule (see above); read by :meth:`termination_flags`.
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
    # inner-loop exit and secant model-stall detection
    radius_floor: float = eqx.field(static=True, default=1e-8)
    collapse_resets_secant: bool = eqx.field(static=True, default=True)
    consecutive_model_failures: Int[Array, ""] = eqx.field(
        default_factory=lambda: jnp.asarray(0, jnp.int32)
    )
    exit_without_progress: Bool[Array, ""] = eqx.field(
        default_factory=lambda: jnp.asarray(False)
    )
    trust_region_collapsed: Bool[Array, ""] = eqx.field(
        default_factory=lambda: jnp.asarray(False)
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
        """Seed the secant and barrier (via super) and clear the stall carry."""
        base = super()._init_dynamics(problem, x0)
        return eqx.tree_at(
            lambda m: (
                m.consecutive_model_failures,
                m.exit_without_progress,
                m.trust_region_collapsed,
            ),
            base,
            (jnp.asarray(0, jnp.int32), jnp.asarray(False), jnp.asarray(False)),
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
                correction=self._bind_step_correction(ctx, solver_state),
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
        """Reduce ``μ``, apply the inner-loop exit and refresh the stall carry.

        Records whether the policy judged the barrier subproblem solved,
        which :meth:`termination_metrics` reads back as the inner
        ``E(·; μ) ≤ ε_μ`` test of N&W Algorithm 19.4. That test is evaluated
        at the iterate, so it runs on rejected steps too — ``result.x`` is
        then the retained iterate.

        The inner-loop exit declares the subproblem finished when the radius
        has collapsed below ``radius_floor * max(1, ||x||)`` or the step was
        rejected at rounding level, provided the unperturbed KKT error
        ``E(·; 0)`` is still above ``atol`` (otherwise the outer test ends
        the run). ``μ`` is then reduced regardless of the barrier test and
        the radius returns to :attr:`initial_radius`. The
        :attr:`trust_region_collapsed` flag is raised when the exit fires
        with ``μ`` already at or below ``atol`` while no step was accepted
        since the previous exit, or when ``μ`` could not be reduced. The
        model-stall streak counts consecutive collapses on rejected steps
        whose predicted reduction was above rounding level.

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
            Minimiser with ``barrier``, ``barrier_updated``, ``solver_state``,
            ``consecutive_model_failures``, ``exit_without_progress`` and
            ``trust_region_collapsed`` refreshed.
        """
        # ``ctx.lagrangian`` is the module; re-evaluate at (x_new, step_dual)
        # exactly as ``_update_secant`` does. Complementarity is secant-independent,
        # so using the step's module (pre-update barrier) is correct.
        x_new = result.x
        lag_module = cast(InteriorPointLagrangian, ctx.lagrangian)
        lagrangian = lag_module(x_new, step_dual)
        barrier = cast(Barrier, self.barrier)
        mu_old = barrier.weight

        # The trust-region controller always writes the solver state back.
        raw_state = cast(TrustRegionSolverState, result.solver_state)
        unconverged = (
            self.barrier_update.optimality_residual(lagrangian, jnp.asarray(0.0))
            > self.atol
        )
        x_scale = jnp.maximum(1.0, jnp.linalg.norm(x_new.x))
        collapsed = raw_state.radius < self.radius_floor * x_scale
        rejected = ~result.accepted
        # xtol-style exit: the barrier subproblem is solved to working precision.
        finished = (collapsed | (rejected & result.rounding_level)) & unconverged
        new_barrier, updated = self.barrier_update.update(
            barrier, lagrangian, force=finished
        )
        # The previous exit is still "without progress" if this step was not
        # accepted either; a second exit then shows that reducing ``μ`` did
        # not unlock anything. A ``μ`` the schedule cannot reduce is final.
        no_progress = self.exit_without_progress & rejected
        exhausted = finished & (
            ((mu_old <= self.atol) & no_progress) | (new_barrier.weight >= mu_old)
        )
        state = eqx.tree_at(
            lambda s: s.radius,
            raw_state,
            jnp.where(
                finished,
                jnp.asarray(self.initial_radius, raw_state.radius.dtype),
                raw_state.radius,
            ),
        )
        # A collapse on a rejected step whose model still claimed a decrease is
        # evidence against the model; one at rounding level is not.
        stall = collapsed & rejected & ~result.rounding_level & unconverged
        if not self.collapse_resets_secant:
            stall = jnp.asarray(False)
        streak = jnp.where(stall, self.consecutive_model_failures + 1, 0).astype(
            jnp.int32
        )
        self.logger.warning(
            "model stall: trust region collapsed on a rejected step "
            "rho={rho:.3e} radius={radius:.3e} consecutive={streak}",
            when=stall,
            rho=raw_state.rho,
            radius=raw_state.radius,
            streak=streak,
        )
        self.logger.info(
            "inner-loop exit: collapsed={collapsed} rounding_level={rounding} "
            "mu={mu:.3e}",
            when=finished,
            collapsed=collapsed,
            rounding=result.rounding_level,
            mu=mu_old,
        )
        self.logger.info(
            "barrier reduced: mu={mu:.3e}",
            when=updated,
            mu=new_barrier.weight,
        )
        return eqx.tree_at(
            lambda m: (
                m.barrier,
                m.barrier_updated,
                m.solver_state,
                m.consecutive_model_failures,
                m.exit_without_progress,
                m.trust_region_collapsed,
            ),
            self,
            (
                new_barrier,
                updated,
                state,
                streak,
                finished | no_progress,
                self.trust_region_collapsed | exhausted,
            ),
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
                "trust_region_collapsed": self.trust_region_collapsed,
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
        """Converge on the unperturbed KKT error ``E(x, s, y, z; 0) <= atol``.

        This is the outer stopping test of N&W Algorithm 19.4; the inner one
        (``E(·; μ) <= ε_μ``) only drives the barrier update and is not
        required to hold on the same step. The test is evaluated at the
        current iterate before the next step is attempted and outranks the
        fatal flags of the last step (subproblem status, secant recovery
        exhaustion, trust-region collapse): a point with an acceptable KKT
        error is a solution whether or not the step that led there was
        rejected.

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
        acceptable_residual = metrics.optimality_residual <= self.atol
        nonfinite = metrics.nonfinite
        recovery_fatal = self.secant_recovery_state.fatal
        subproblem_fatal = (
            cast(TrustRegionStateType, ctx.solver_state).status
            != SUBPROBLEM_RESULTS.successful
        )
        converged = acceptable_residual & metrics.has_min_steps & ~nonfinite
        collapsed = self.trust_region_collapsed
        fatal_result = TRUST_REGION_INTERIOR_POINT_RESULTS.where(
            recovery_fatal,
            TRUST_REGION_INTERIOR_POINT_RESULTS.secant_recovery_failure,
            metrics.fatal_result,
        )
        fatal_result = TRUST_REGION_INTERIOR_POINT_RESULTS.where(
            collapsed,
            TRUST_REGION_INTERIOR_POINT_RESULTS.trust_region_collapse,
            fatal_result,
        )
        return cast(
            TerminationFlags,
            TerminationFlags(
                converged=converged,
                nonfinite=nonfinite,
                fatal=subproblem_fatal | recovery_fatal | collapsed,
                fatal_result=fatal_result,
            ),
        )
