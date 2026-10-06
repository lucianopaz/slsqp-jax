"""Trust-funnel interior-point outer loop (Curtis, Gould, Robinson & Toint 2017)."""

from __future__ import annotations

from typing import Any, Self, cast

import equinox as eqx
import jax
import optimistix as optx
from jax import numpy as jnp
from jaxtyping import Array, Bool, Int

from ..barrier import Barrier, FunnelBarrierUpdate
from ..dual import Dual
from ..lagrangian import InteriorPointEvaluatedLagrangian, InteriorPointLagrangian
from ..merit import ConstraintViolation, NormMerit
from ..primal import InteriorPointPrimal, Slack
from ..problem import ProblemProtocol
from ..results import MINIMISER_RESULTS, ResultAdapter
from ..secant import SecantResetSignals
from ..step_controller import StepController, StepResult, TrustFunnelManager
from ..subproblem import FunnelBarrierSubProblem
from ..subproblem.solver import (
    RESULTS as SUBPROBLEM_RESULTS,
)
from ..subproblem.solver import (
    IterationType,
    SubproblemContext,
    SubProblemSolver,
    TrustFunnelSolver,
    TrustFunnelSolverState,
)
from ..types import Scalar, Vector_n
from .base import OptimisationContext
from .interior_point import InteriorPointMinimiser
from .termination import TerminationFlags, TerminationMetrics

__all__ = [
    "TrustFunnelInteriorPointMinimiser",
    "TrustFunnelTerminationMetrics",
    "TRUST_FUNNEL_INTERIOR_POINT_RESULTS",
    "TrustFunnelInteriorPointResultAdapter",
]


class TRUST_FUNNEL_INTERIOR_POINT_RESULTS(
    MINIMISER_RESULTS  # ty: ignore[subclass-of-final-class]
):
    """Fine-grained outcomes for the trust-funnel interior-point minimiser.

    ``infeasible_stationary_point``, ``stationarity_stall`` and
    ``subproblem_nonfinite`` are raised by the minimiser's termination test.
    The remaining codes name the invariant violations the funnel diagnostics
    can promote to a fatal outcome (``v > v_max`` after an accepted step, a
    sub-step below its Cauchy decrease, a failed multiplier solve).
    """

    infeasible_stationary_point = (
        "The iterate is an infeasible stationary point of the constraint violation."
    )
    stationarity_stall = (
        "Consecutive y-iterations left the (nearly feasible) iterate unchanged "
        "without meeting the barrier stationarity tolerance."
    )
    subproblem_nonfinite = "The trust-funnel step was non-finite."
    funnel_invariant_violation = (
        "An accepted step left the funnel (v > v_max) — model bookkeeping error."
    )
    cauchy_decrease_violation = (
        "A normal or tangential step failed its Cauchy decrease condition."
    )
    multiplier_solve_failure = "The least-squares multiplier estimate failed."


class TrustFunnelTerminationMetrics(
    TerminationMetrics[TRUST_FUNNEL_INTERIOR_POINT_RESULTS]
):
    """Termination measurements for :class:`TrustFunnelInteriorPointMinimiser`.

    The trust-funnel loop has three tests: the barrier subproblem test
    (3.15a) ``πᶠ ≤ ε_π(μ)``, ``v ≤ ε_v(μ)`` that lets Algorithm 3 reduce
    ``μ``; the convergence test on the unperturbed KKT error; and the
    infeasible-stationary test of Algorithm 2 Step 8 (``v > 0`` with
    ``χᵛ = 0``). Because ``χᵛ`` only vanishes in the limit of zero slacks,
    the Step 8 test is complemented by a *fixed-point* detector: Algorithm 2
    is deterministic, so a run of y-iterations (zero primal step) at fixed
    ``μ`` means no further progress is possible.

    Attributes
    ----------
    barrier_updated
        ``True`` when the last
        :class:`~slsqp_jax.sqpdax.barrier.update.FunnelBarrierUpdate` call
        found (3.15a) satisfied and reduced ``μ`` (informational: it drives
        the ``μ`` schedule, not the convergence decision).
    optimality_residual
        Unperturbed KKT error ``E(x, s, y, z; 0)`` (N&W eq. 19.10), compared
        against ``atol`` for the convergence test.
    pi_f
        Scaled stationarity ``πᶠ`` (Definition 1.2 with ``n = 0``) at the
        committed ``(x, s, y)``.
    violation
        Constraint violation ``v = ‖c(x, s)‖₂``.
    chi_v
        Criticality measure ``χᵛ = ‖Âᵀĉ‖ / v`` of the violation.
    stalled
        ``consecutive_y_iterations >= stall_steps``: the loop sits at a
        fixed point of Algorithm 2 for the current ``μ``.
    infeasible_stationary
        ``v > atol`` with ``χᵛ ≤ infeasibility_tol``, or stalled while
        ``v > ε_v(μ)``.
    stationarity_stall
        Stalled while ``v ≤ ε_v(μ)`` (so ``πᶠ > ε_π(μ)`` is what blocks the
        ``μ`` reduction).
    has_min_steps
        ``True`` once ``step_count >= min_steps``.
    """

    barrier_updated: Bool[Array, ""]
    optimality_residual: Scalar
    pi_f: Scalar
    violation: Scalar
    chi_v: Scalar
    stalled: Bool[Array, ""]
    infeasible_stationary: Bool[Array, ""]
    stationarity_stall: Bool[Array, ""]
    has_min_steps: Bool[Array, ""]


class TrustFunnelInteriorPointResultAdapter(
    ResultAdapter[TRUST_FUNNEL_INTERIOR_POINT_RESULTS]
):
    """Optimistix conversion for trust-funnel interior-point outcomes."""

    @property
    def result_type(self) -> type[TRUST_FUNNEL_INTERIOR_POINT_RESULTS]:
        return TRUST_FUNNEL_INTERIOR_POINT_RESULTS

    def to_optimistix(
        self, result: TRUST_FUNNEL_INTERIOR_POINT_RESULTS
    ) -> optx.RESULTS:
        coarse = optx.RESULTS.nonlinear_divergence
        mappings = (
            (
                TRUST_FUNNEL_INTERIOR_POINT_RESULTS.subproblem_nonfinite,
                optx.RESULTS.nonfinite_input,
            ),
            (
                TRUST_FUNNEL_INTERIOR_POINT_RESULTS.multiplier_solve_failure,
                optx.RESULTS.singular,
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


class TrustFunnelInteriorPointMinimiser(
    InteriorPointMinimiser[
        FunnelBarrierSubProblem,
        TrustFunnelSolverState,
        TrustFunnelTerminationMetrics,
        TRUST_FUNNEL_INTERIOR_POINT_RESULTS,
    ]
):
    """Interior-point trust-funnel method of Curtis, Gould, Robinson & Toint (2017).

    Algorithm 3 of the paper: for a decreasing sequence of barrier weights
    ``μ_j`` the barrier subproblem ``BSP(μ_j)``

    ```
    min  f(x) − μ Σ log sᵢ   subject to   c(x, s) = 0
    ```

    is solved approximately by the trust-funnel Algorithm 2, whose iterations
    are the outer steps of this minimiser. Each step

    1. evaluates the barrier Lagrangian at ``(x, s, y)`` and freezes it into a
       :class:`~slsqp_jax.sqpdax.subproblem.funnel_barrier.FunnelBarrierSubProblem`
       with the ``μ``-dependent fraction-to-boundary constants
       ``κ_fbn(μ)``, ``κ_fbt(μ)`` of :attr:`barrier_update`;
    2. computes the trial step ``d = n + t`` and the multiplier estimate ``y``
       with :class:`~slsqp_jax.sqpdax.subproblem.solver.trust_funnel.TrustFunnelSolver`
       (Steps 7–37 of Algorithm 2);
    3. classifies it as a y-, f- or v-iteration and updates ``δᵛ``, ``δᶠ``,
       ``v_max`` with :class:`~slsqp_jax.sqpdax.step_controller.trust_funnel.TrustFunnelManager`
       (Steps 38–50);
    4. applies the slack reset (3.26)/(3.33) to the committed iterate, tests
       (3.15a) through :attr:`barrier_update` and, when ``μ`` is reduced,
       restarts Algorithm 2 for the new subproblem (fresh radii, funnel
       radius ``v_max = max{κ_ca, κ_cr v}`` and tolerances).

    The multiplier estimate is a property of the current iterate, so it is
    committed whether or not the primal step is accepted (y-iterations
    change only ``y``).

    Termination: the run converges once the unperturbed KKT error
    ``E(x, s, y, z; 0)`` (N&W eq. 19.10: stationarity, feasibility and
    complementarity) is below ``atol``. The (3.15a) test only drives the
    ``μ`` schedule — it is a property of the current barrier subproblem, and
    insisting on it as well would stall runs whose ``ε_π(μ)`` has dropped
    below the noise floor of the inexact projections and multiplier solves
    even though the iterate already is an ``atol``-KKT point. The loop
    stops with ``infeasible_stationary_point`` when
    ``v > atol`` while ``χᵛ ≤ infeasibility_tol`` (Step 8). Since ``χᵛ``
    reaches zero only as the slacks vanish, a run of :attr:`stall_steps`
    consecutive y-iterations at fixed ``μ`` — a fixed point of the
    deterministic Algorithm 2 — is treated the same way when ``v > ε_v(μ)``
    and reported as ``stationarity_stall`` otherwise. There is no relative
    tolerance, so passing ``rtol`` raises.

    When the model uses a secant, the ``model`` channel of
    :class:`~slsqp_jax.sqpdax.secant.reset.SecantResetSignals` is fed by a
    streak of steps at which both trust-region radii collapsed below
    ``radius_floor * max(1, ‖x‖)`` while unconverged, or the step was
    rejected with a ratio below :attr:`model_failure_rho`.

    Attributes
    ----------
    barrier_update
        :class:`~slsqp_jax.sqpdax.barrier.update.FunnelBarrierUpdate` owning
        the ``μ`` schedule and the ``μ``-dependent tolerances / constants of
        Table 1. A different kind of barrier update is rejected.
    initial_radius_v, initial_radius_f
        Initial trust-region radii ``δᵛ₀``, ``δᶠ₀`` (re-seeded whenever ``μ``
        is reduced).
    kappa_ca, kappa_cr
        Funnel seeding ``v_max₀ = max{κ_ca, κ_cr v₀}``.
    eta1, eta2, gamma1, gamma2, grow_factor, max_radius, kappa_t1, kappa_t2
        Forwarded to :class:`~slsqp_jax.sqpdax.step_controller.trust_funnel.TrustFunnelManager`.
    infeasibility_tol
        Threshold on ``χᵛ`` for the infeasible-stationary test.
    stall_steps
        Number of consecutive y-iterations at fixed ``μ`` after which the
        loop is declared stalled.
    radius_floor, model_failure_rho
        Model-stall detection constants (see above).
    consecutive_model_failures
        Dynamic count of consecutive model stalls reported to the secant
        reset policy as ``model_streak``.
    consecutive_y_iterations
        Dynamic count of consecutive y-iterations since the last primal
        move or ``μ`` reduction.
    """

    barrier_update: FunnelBarrierUpdate = eqx.field(default_factory=FunnelBarrierUpdate)
    initial_radius_v: float = eqx.field(static=True, default=1.0)
    initial_radius_f: float = eqx.field(static=True, default=1.0)
    kappa_ca: float = eqx.field(static=True, default=1e3)
    kappa_cr: float = eqx.field(static=True, default=2.0)
    # acceptance / radius / funnel policy (forwarded to TrustFunnelManager)
    eta1: float = eqx.field(static=True, default=0.1)
    eta2: float = eqx.field(static=True, default=0.75)
    gamma1: float = eqx.field(static=True, default=0.25)
    gamma2: float = eqx.field(static=True, default=0.5)
    grow_factor: float = eqx.field(static=True, default=2.0)
    max_radius: float = eqx.field(static=True, default=1e10)
    kappa_t1: float = eqx.field(static=True, default=0.5)
    kappa_t2: float = eqx.field(static=True, default=0.5)
    infeasibility_tol: float = eqx.field(static=True, default=1e-8)
    stall_steps: int = eqx.field(static=True, default=5)
    # secant model-stall detection (feeds the ``model`` reset channel)
    radius_floor: float = eqx.field(static=True, default=1e-8)
    model_failure_rho: float = eqx.field(static=True, default=-1.0)
    consecutive_model_failures: Int[Array, ""] = eqx.field(
        default_factory=lambda: jnp.asarray(0, jnp.int32)
    )
    consecutive_y_iterations: Int[Array, ""] = eqx.field(
        default_factory=lambda: jnp.asarray(0, jnp.int32)
    )

    def __check_init__(self) -> None:
        if not isinstance(self.barrier_update, FunnelBarrierUpdate):
            raise TypeError(
                "TrustFunnelInteriorPointMinimiser requires a FunnelBarrierUpdate; "
                f"got {type(self.barrier_update).__name__}"
            )
        if self.kappa_ca <= 0.0 or self.kappa_cr <= 0.0:
            raise ValueError(
                "kappa_ca and kappa_cr must be positive; got "
                f"kappa_ca={self.kappa_ca}, kappa_cr={self.kappa_cr}"
            )
        if self.initial_radius_v <= 0.0 or self.initial_radius_f <= 0.0:
            raise ValueError(
                "initial radii must be positive; got "
                f"initial_radius_v={self.initial_radius_v}, "
                f"initial_radius_f={self.initial_radius_f}"
            )
        if self.infeasibility_tol <= 0.0:
            raise ValueError(
                f"infeasibility_tol must be positive; got {self.infeasibility_tol}"
            )
        if self.stall_steps < 1:
            raise ValueError(f"stall_steps must be >= 1; got {self.stall_steps}")

    @property
    def result_adapter(self) -> TrustFunnelInteriorPointResultAdapter:
        """Native trust-funnel interior-point result policy."""
        return cast(
            TrustFunnelInteriorPointResultAdapter,
            TrustFunnelInteriorPointResultAdapter(),
        )

    # ================================ init =================================

    def _subproblem_solver_type(self) -> type[SubProblemSolver]:
        """Root solver class for ``options['subproblem']`` validation."""
        return TrustFunnelSolver

    def _init_dynamics(
        self, problem: ProblemProtocol[InteriorPointPrimal], x0: Vector_n
    ) -> Self:
        """Seed the secant and barrier (via super) and zero both streaks."""
        base = super()._init_dynamics(problem, x0)
        zero = jnp.asarray(0, jnp.int32)
        return eqx.tree_at(
            lambda m: (m.consecutive_model_failures, m.consecutive_y_iterations),
            base,
            (zero, zero),
        )

    def _funnel_radius(self, violation: Scalar) -> Scalar:
        """``v_max₀ = max{κ_ca, κ_cr v}`` for a fresh run of Algorithm 2."""
        return jnp.maximum(self.kappa_ca, self.kappa_cr * violation)

    def _init_solver_state(
        self,
        problem: ProblemProtocol[InteriorPointPrimal],
        primal: InteriorPointPrimal,
    ) -> TrustFunnelSolverState:
        """Cold funnel carry: initial radii, ``v_max₀`` and the ``μ₀`` tolerances."""
        mu = cast(Barrier, self.barrier).weight
        violation = cast(
            ConstraintViolation, ConstraintViolation(problem=problem, norm=2)
        )
        v0 = violation(primal)
        return TrustFunnelSolverState.cold(
            self.initial_radius_v,
            self.initial_radius_f,
            self._funnel_radius(v0),
            eps_pi=self.barrier_update.eps_pi(mu),
            eps_v=self.barrier_update.eps_v(mu),
            dtype=primal.x.dtype,
        )

    # ================================ step =================================

    def _make_subproblem_solver(
        self, problem: ProblemProtocol[InteriorPointPrimal]
    ) -> TrustFunnelSolver:
        """Default :class:`TrustFunnelSolver` before ``options['subproblem']`` is applied."""
        return cast(TrustFunnelSolver, TrustFunnelSolver())

    def _make_subproblem(
        self, lagrangian: InteriorPointEvaluatedLagrangian
    ) -> FunnelBarrierSubProblem:
        """Funnel model with the ``μ``-dependent fraction-to-boundary constants."""
        mu = cast(Barrier, self.barrier).weight
        return cast(
            FunnelBarrierSubProblem,
            FunnelBarrierSubProblem(
                lagrangian,
                kappa_fbn=self.barrier_update.kappa_fbn(mu),
                kappa_fbt=self.barrier_update.kappa_fbt(mu),
            ),
        )

    def _step_controller(
        self,
        ctx: SubproblemContext[
            InteriorPointPrimal, FunnelBarrierSubProblem, TrustFunnelSolverState
        ],
        step_dual: Dual,
        solver_state: TrustFunnelSolverState,
    ) -> StepController[InteriorPointPrimal, TrustFunnelSolverState]:
        """Funnel manager scoring the barrier function and ``‖c(x, s)‖₂``."""
        barrier_merit = NormMerit(
            ctx.problem,
            barrier=self.barrier,
            problem_weight=jnp.asarray(1.0),
            barrier_weight=jnp.asarray(1.0),
            feasibility_weight=jnp.asarray(0.0),
            norm=2,
        )
        return cast(
            StepController[InteriorPointPrimal, TrustFunnelSolverState],
            TrustFunnelManager(
                barrier_merit=barrier_merit,
                violation=ConstraintViolation(problem=ctx.problem, norm=2),
                logger=self.logger.child("step_controller"),
                eta1=self.eta1,
                eta2=self.eta2,
                gamma1=self.gamma1,
                gamma2=self.gamma2,
                grow_factor=self.grow_factor,
                max_radius=self.max_radius,
                kappa_t1=self.kappa_t1,
                kappa_t2=self.kappa_t2,
            ),
        )

    @staticmethod
    def _reset_slacks(
        iterate: InteriorPointPrimal, lagrangian: InteriorPointEvaluatedLagrangian
    ) -> tuple[InteriorPointPrimal, Bool[Array, ""]]:
        """Slack reset (3.26)/(3.33): ``sᵢ ← −cᵢ(x)`` wherever ``[c(x, s)]ᵢ < 0``.

        With ``c(x, s) = c(x) + s`` this is ``s ← s + max{0, −c(x, s)}``: slacks
        only increase, so the barrier term and the violation both decrease
        and the funnel invariant ``v ≤ v_max`` is preserved (Lemma 3.4).
        Dead (null-bound) slacks have a zero residual and are untouched.

        Parameters
        ----------
        iterate
            Committed primal.
        lagrangian
            Barrier Lagrangian evaluated at ``iterate``.

        Returns
        -------
        iterate
            Primal with the reset slacks.
        reset
            Whether any slack moved.
        """
        residual = lagrangian.dual_grad
        bump = cast(
            Slack,
            Slack(
                s=jnp.maximum(0.0, -residual.ineq_multipliers),
                s_lb=jnp.maximum(0.0, -residual.lb_multipliers),
                s_ub=jnp.maximum(0.0, -residual.ub_multipliers),
            ),
        )
        reset = jnp.any(bump.flatten() > 0.0)
        new_slack = cast(
            Slack, jax.tree.map(lambda old, up: old + up, iterate.slack, bump)
        )
        return (
            cast(
                InteriorPointPrimal, InteriorPointPrimal(x=iterate.x, slack=new_slack)
            ),
            reset,
        )

    def _advance_dynamics(
        self,
        ctx: SubproblemContext[
            InteriorPointPrimal, FunnelBarrierSubProblem, TrustFunnelSolverState
        ],
        result: StepResult[InteriorPointPrimal, TrustFunnelSolverState],
        step_dual: Dual,
    ) -> Self:
        """Slack reset, (3.15a) test / ``μ`` reduction and model-stall streak.

        Runs on the minimiser *after* ``iterate``, ``dual`` and
        ``solver_state`` were committed. The slack reset rewrites the
        committed iterate (``x`` is untouched, so the secant pair just
        appended is unaffected). The barrier test is then evaluated at the
        reset iterate; when it passes, ``μ`` is reduced and Algorithm 2 is
        restarted: both radii return to their initial values, ``v_max`` is
        re-seeded from the current violation, the ``S_f`` flag and
        ``πᶠ_{k−1}`` are cleared, and the state's tolerances follow the new
        ``μ``.

        Parameters
        ----------
        ctx
            Per-step subproblem context.
        result
            Outcome of the funnel controller.
        step_dual
            Multipliers committed with ``result.x``.

        Returns
        -------
        Self
            Minimiser with ``iterate``, ``solver_state``, ``barrier``,
            ``barrier_updated``, ``consecutive_model_failures`` and
            ``consecutive_y_iterations`` refreshed.
        """
        lag_module = cast(InteriorPointLagrangian, ctx.lagrangian)
        iterate = cast(InteriorPointPrimal, self.iterate)
        iterate, reset = self._reset_slacks(iterate, lag_module(iterate, step_dual))
        lagrangian = lag_module(iterate, step_dual)
        sub = cast(FunnelBarrierSubProblem, FunnelBarrierSubProblem(lagrangian))
        pi_f = sub.pi_f(sub._zero_primal(), step_dual)
        v = sub.violation()
        new_barrier, solved = self.barrier_update.update(
            cast(Barrier, self.barrier), lagrangian, pi_f=pi_f, v=v
        )
        new_mu = new_barrier.weight

        state = cast(TrustFunnelSolverState, self.solver_state)
        dtype = state.radius_v.dtype

        def reseed(fresh, current):
            return jnp.where(solved, jnp.asarray(fresh, dtype), current)

        state = cast(
            TrustFunnelSolverState,
            eqx.tree_at(
                lambda s: (
                    s.radius_v,
                    s.radius_f,
                    s.v_max,
                    s.sf_flag,
                    s.pi_f_prev,
                    s.eps_pi,
                    s.eps_v,
                ),
                state,
                (
                    reseed(self.initial_radius_v, state.radius_v),
                    reseed(self.initial_radius_f, state.radius_f),
                    reseed(self._funnel_radius(v), state.v_max),
                    state.sf_flag & ~solved,
                    reseed(0.0, state.pi_f_prev),
                    self.barrier_update.eps_pi(new_mu).astype(dtype),
                    self.barrier_update.eps_v(new_mu).astype(dtype),
                ),
            ),
        )

        unconverged = (
            self.barrier_update.optimality_residual(lagrangian, jnp.asarray(0.0))
            > self.atol
        )
        x_scale = jnp.maximum(1.0, jnp.linalg.norm(iterate.x))
        radius = jnp.maximum(state.radius_v, state.radius_f)
        radius_collapse = (radius < self.radius_floor * x_scale) & unconverged
        model_failure = ~result.accepted & (state.rho < self.model_failure_rho)
        stall = radius_collapse | model_failure
        streak = jnp.where(stall, self.consecutive_model_failures + 1, 0).astype(
            jnp.int32
        )
        is_y = state.iteration_type == IterationType.y_iteration
        y_streak = jnp.where(
            is_y & ~solved, self.consecutive_y_iterations + 1, 0
        ).astype(jnp.int32)
        self.logger.info(
            "slack reset applied: v={v:.3e}",
            when=reset,
            v=v,
        )
        self.logger.warning(
            "model stall: radius_collapse={collapse} model_failure={failure} "
            "rho={rho:.3e} radius_v={radius_v:.3e} radius_f={radius_f:.3e} "
            "consecutive={streak}",
            when=stall,
            collapse=radius_collapse,
            failure=model_failure,
            rho=state.rho,
            radius_v=state.radius_v,
            radius_f=state.radius_f,
            streak=streak,
        )
        self.logger.info(
            "barrier subproblem solved: pi_f={pi_f:.3e} v={v:.3e}; mu={mu:.3e} "
            "v_max={v_max:.3e}",
            when=solved,
            pi_f=pi_f,
            v=v,
            mu=new_mu,
            v_max=state.v_max,
        )
        self.logger.warning(
            "{n} consecutive y-iterations at mu={mu:.3e}: pi_f={pi_f:.3e} v={v:.3e}",
            when=y_streak >= self.stall_steps,
            n=y_streak,
            mu=new_mu,
            pi_f=pi_f,
            v=v,
        )
        return eqx.tree_at(
            lambda m: (
                m.iterate,
                m.solver_state,
                m.barrier,
                m.barrier_updated,
                m.consecutive_model_failures,
                m.consecutive_y_iterations,
            ),
            self,
            (iterate, state, new_barrier, solved, streak, y_streak),
        )

    def _step_log_fields(
        self,
        ctx: SubproblemContext[
            InteriorPointPrimal, FunnelBarrierSubProblem, TrustFunnelSolverState
        ],
        result: StepResult[InteriorPointPrimal, TrustFunnelSolverState],
        step_dual: Dual,
        octx: OptimisationContext[InteriorPointPrimal, TrustFunnelSolverState],
        metrics: TrustFunnelTerminationMetrics,
    ) -> dict[str, Any]:
        """Barrier weight, KKT error, funnel measures, radii and iteration type."""
        state = cast(TrustFunnelSolverState, self.solver_state)
        return {
            "kkt": (metrics.optimality_residual, ".2e"),
            "mu": (cast(Barrier, self.barrier).weight, ".2e"),
            "mu_updated": metrics.barrier_updated,
            "pi_f": (metrics.pi_f, ".2e"),
            "v": (metrics.violation, ".2e"),
            "v_max": (state.v_max, ".2e"),
            "radius_v": (state.radius_v, ".2e"),
            "radius_f": (state.radius_f, ".2e"),
            "rho": (state.rho, ".2e"),
            "type": state.iteration_type,
            "cg_total": state.n_cg_iter,
            "model_fail": self.consecutive_model_failures,
            "y_streak": self.consecutive_y_iterations,
        }

    def _diagnostic_fields(
        self,
        ctx: SubproblemContext[
            InteriorPointPrimal, FunnelBarrierSubProblem, TrustFunnelSolverState
        ],
        result: StepResult[InteriorPointPrimal, TrustFunnelSolverState],
        step_dual: Dual,
        octx: OptimisationContext[InteriorPointPrimal, TrustFunnelSolverState],
        metrics: TrustFunnelTerminationMetrics,
    ) -> dict[str, Any]:
        """Base payload plus barrier weight, funnel state and slacks."""
        fields = super()._diagnostic_fields(ctx, result, step_dual, octx, metrics)
        state = cast(TrustFunnelSolverState, self.solver_state)
        fields.update(
            {
                "barrier_weight": cast(Barrier, self.barrier).weight,
                "barrier_updated": metrics.barrier_updated,
                "radius_v": state.radius_v,
                "radius_f": state.radius_f,
                "v_max": state.v_max,
                "rho": state.rho,
                "iteration_type": state.iteration_type,
                "cg_total": state.n_cg_iter,
                "consecutive_model_failures": self.consecutive_model_failures,
                "consecutive_y_iterations": self.consecutive_y_iterations,
                "slack": cast(InteriorPointPrimal, self.iterate).slack,
            }
        )
        return fields

    def _secant_reset_signals(self) -> SecantResetSignals:
        """Report model-quality stalls on the ``model`` channel only."""
        zero = jnp.asarray(0, jnp.int32)
        return cast(
            SecantResetSignals,
            SecantResetSignals(
                subproblem_streak=zero,
                step_streak=zero,
                model_streak=self.consecutive_model_failures,
            ),
        )

    # ============================= termination =============================

    def termination_metrics(
        self, ctx: OptimisationContext[InteriorPointPrimal, TrustFunnelSolverState]
    ) -> TrustFunnelTerminationMetrics:
        """Measure ``E(·; 0)``, ``πᶠ``, ``v``, ``χᵛ`` and the inner-loop state.

        Parameters
        ----------
        ctx
            Termination context at the current iterate.

        Returns
        -------
        TrustFunnelTerminationMetrics
            Populated metrics for :meth:`termination_flags`.
        """
        lag = cast(InteriorPointEvaluatedLagrangian, ctx.lagrangian)
        sub = cast(FunnelBarrierSubProblem, FunnelBarrierSubProblem(lag))
        violation = sub.violation()
        chi_v = sub.chi_v()
        pi_f = sub.pi_f(sub._zero_primal(), lag.dual)
        optimality_residual = self.barrier_update.optimality_residual(
            lag, jnp.asarray(0.0)
        )
        nonfinite = self._any_nonfinite(
            lag.value,
            lag.x_grad,
            lag.ref,
            self.dual,
            optimality_residual,
            pi_f,
            violation,
        )
        stalled = self.consecutive_y_iterations >= self.stall_steps
        eps_v = self.barrier_update.eps_v(cast(Barrier, self.barrier).weight)
        infeasible_stationary = (violation > self.atol) & (
            (chi_v <= self.infeasibility_tol) | (stalled & (violation > eps_v))
        )
        stationarity_stall = stalled & ~infeasible_stationary
        status = cast(TrustFunnelSolverState, ctx.solver_state).status
        fatal_result = TRUST_FUNNEL_INTERIOR_POINT_RESULTS.where(
            status != SUBPROBLEM_RESULTS.successful,
            TRUST_FUNNEL_INTERIOR_POINT_RESULTS.subproblem_nonfinite,
            TRUST_FUNNEL_INTERIOR_POINT_RESULTS.where(
                infeasible_stationary,
                TRUST_FUNNEL_INTERIOR_POINT_RESULTS.infeasible_stationary_point,
                TRUST_FUNNEL_INTERIOR_POINT_RESULTS.stationarity_stall,
            ),
        )
        return cast(
            TrustFunnelTerminationMetrics,
            TrustFunnelTerminationMetrics(
                barrier_updated=self.barrier_updated,
                optimality_residual=optimality_residual,
                pi_f=pi_f,
                violation=violation,
                chi_v=chi_v,
                stalled=stalled,
                infeasible_stationary=infeasible_stationary,
                stationarity_stall=stationarity_stall,
                nonfinite=nonfinite,
                fatal_result=fatal_result,
                has_min_steps=self.step_count >= self.min_steps,
            ),
        )

    def termination_flags(
        self,
        ctx: OptimisationContext[InteriorPointPrimal, TrustFunnelSolverState],
        metrics: TrustFunnelTerminationMetrics,
    ) -> TerminationFlags[TRUST_FUNNEL_INTERIOR_POINT_RESULTS]:
        """Converge on ``E(·; 0) ≤ atol``; stop on Step 8, stalls and failures.

        The infeasible-stationary and stall stops are gated by ``min_steps``
        like the convergence test, so the loop always takes at least
        ``min_steps`` steps before giving up.

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
        nonfinite = metrics.nonfinite
        recovery_fatal = self.secant_recovery_state.fatal
        subproblem_fatal = (
            cast(TrustFunnelSolverState, ctx.solver_state).status
            != SUBPROBLEM_RESULTS.successful
        )
        stopped = (
            metrics.infeasible_stationary | metrics.stationarity_stall
        ) & metrics.has_min_steps
        fatal = subproblem_fatal | recovery_fatal | stopped
        converged = (
            (metrics.optimality_residual <= self.atol)
            & metrics.has_min_steps
            & ~nonfinite
            & ~fatal
        )
        fatal_result = TRUST_FUNNEL_INTERIOR_POINT_RESULTS.where(
            recovery_fatal,
            TRUST_FUNNEL_INTERIOR_POINT_RESULTS.secant_recovery_failure,
            metrics.fatal_result,
        )
        return cast(
            TerminationFlags,
            TerminationFlags(
                converged=converged,
                nonfinite=nonfinite,
                fatal=fatal,
                fatal_result=fatal_result,
            ),
        )
