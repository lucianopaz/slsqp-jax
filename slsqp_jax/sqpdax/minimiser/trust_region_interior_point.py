"""Trust-region interior-point outer loop (Nocedal & Wright §19.5)."""

from __future__ import annotations

from dataclasses import replace
from typing import Generic, Self, cast

import equinox as eqx
import jax
import optimistix as optx
from jax import numpy as jnp
from jaxtyping import Array, Bool

from ..barrier import Barrier, BarrierUpdate, LogBarrier, MonotoneBarrierUpdate
from ..dual import Dual
from ..lagrangian import (
    EvaluatedLagrangian,
    InteriorPointEvaluatedLagrangian,
    InteriorPointLagrangian,
    Lagrangian,
)
from ..merit import NormMerit
from ..primal import InteriorPointPrimal, Slack
from ..problem import ProblemProtocol
from ..results import (
    MINIMISER_RESULTS,
    ResultAdapter,
)
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
from .base import CommonMinimiser, OptimisationContext
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
    CommonMinimiser[
        InteriorPointPrimal,
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
    :class:`~slsqp_jax.sqpdax.barrier.update.BarrierUpdate`. Inequality / bound
    slacks are virtual variables absent from the user's ``x0``, so ``init``
    builds the
    :class:`~slsqp_jax.sqpdax.primal.InteriorPointPrimal` and picks
    strictly-interior default slack values.

    Termination follows the two nested tests of N&W Algorithm 19.4. The inner
    one — solve the current barrier subproblem to ``E(x, s, y, z; μ) ≤ ε_μ``
    — is owned by :attr:`barrier_update` and surfaced as
    :attr:`barrier_updated`; the outer one compares the unperturbed KKT error
    ``E(x, s, y, z; 0)`` against ``atol``. There is no relative tolerance, so
    passing ``rtol`` raises.

    Attributes
    ----------
    initial_mu
        Initial barrier weight ``μ``.
    initial_slack
        Floor used when seeding inequality / bound slacks.
    initial_radius
        Initial trust-region radius.
    initial_penalty
        Initial merit penalty ``ν``.
    primal_dual
        If ``True``, use the primal-dual slack-slack KKT block.
    barrier_update
        Policy that reduces ``μ``. Consulted after every outer step, accepted
        or not: the test it applies is a property of the iterate, not of the
        step, and on a rejected step the iterate is unchanged.
    rtol
        Unused; must be left at its ``NaN`` default. Present only to reject
        the inherited relative-tolerance knob, which this algorithm's
        absolute KKT-error test does not implement.
    eta, shrink_threshold, grow_threshold, shrink_factor, grow_factor, max_radius
        Forwarded to :class:`TrustRegionManager`.
    barrier
        Dynamic barrier whose ``weight`` *is* ``μ`` (seeded in
        :meth:`_init_dynamics`).
    barrier_updated
        Dynamic flag recording whether the last :attr:`barrier_update` call
        judged the barrier subproblem solved; read by
        :meth:`termination_metrics` as the inner stopping test.
    """

    initial_mu: float = eqx.field(static=True, default=1.0)
    initial_slack: float = eqx.field(static=True, default=1.0)
    initial_radius: float = eqx.field(static=True, default=1.0)
    initial_penalty: float = eqx.field(static=True, default=1.0)
    primal_dual: bool = eqx.field(static=True, default=True)
    barrier_update: BarrierUpdate = eqx.field(default_factory=MonotoneBarrierUpdate)
    barrier_updated: Bool[Array, ""] = eqx.field(
        default_factory=lambda: jnp.asarray(False)
    )
    rtol: float = eqx.field(static=True, default=jnp.nan)
    # trust-region acceptance / radius policy (forwarded to TrustRegionManager)
    eta: float = eqx.field(static=True, default=1e-4)
    shrink_threshold: float = eqx.field(static=True, default=0.25)
    grow_threshold: float = eqx.field(static=True, default=0.75)
    shrink_factor: float = eqx.field(static=True, default=0.25)
    grow_factor: float = eqx.field(static=True, default=2.0)
    max_radius: float = eqx.field(static=True, default=1e10)
    # The barrier is dynamic state: its ``weight`` *is* mu, updated each step.
    barrier: Barrier | None = None

    @property
    def result_adapter(self) -> TrustRegionInteriorPointResultAdapter:
        """Native trust-region interior-point result policy."""
        return cast(
            TrustRegionInteriorPointResultAdapter,
            TrustRegionInteriorPointResultAdapter(),
        )

    def __post_init__(self) -> None:
        if bool(~jnp.isnan(self.rtol)):
            raise ValueError(
                "TrustRegionInteriorPointMinimiser does not provide a relative "
                "tolerance for convergence. Use the absolute tolerance instead."
            )

    def _subproblem_solver_type(self) -> type[SubProblemSolver]:
        """Root solver class for ``options['subproblem']`` validation."""
        return TrustRegionInteriorPointSolver

    def _parse_options(self, options: dict | None) -> Self:
        """Freeze options and optionally rebuild ``barrier_update`` from a kind-spec."""
        base = super()._parse_options(options)
        spec = base.options.get("minimiser", {}).get("barrier_update")
        if spec is not None:
            base = replace(
                base,
                barrier_update=cast(BarrierUpdate, BarrierUpdate.from_spec(spec)),
            )
        return base

    def _make_barrier(self, problem: ProblemProtocol[InteriorPointPrimal]) -> Barrier:
        """Log barrier seeded at :attr:`initial_mu` with the problem's null masks."""
        return cast(
            Barrier,
            LogBarrier(
                weight=jnp.asarray(self.initial_mu),
                null_lb=problem.null_lb,
                null_ub=problem.null_ub,
            ),
        )

    def _init_dynamics(
        self, problem: ProblemProtocol[InteriorPointPrimal], x0: Vector_n
    ) -> Self:
        """Seed the secant (via super) then the barrier so Lagrangian / merit see ``μ``."""
        base = super()._init_dynamics(problem, x0)
        return eqx.tree_at(
            lambda m: m.barrier,
            base,
            self._make_barrier(problem),
            is_leaf=lambda z: z is None,
        )

    def _init_primal(
        self, problem: ProblemProtocol[InteriorPointPrimal], x0: Vector_n
    ) -> InteriorPointPrimal:
        """Strictly-interior default slacks for inequalities and finite bounds."""
        h0 = problem.ineq_fn(x0)
        s = jnp.maximum(-h0, self.initial_slack)
        s_lb = jnp.where(
            problem.null_lb,
            self.initial_slack,
            jnp.maximum(x0 - problem.lb, self.initial_slack),
        )
        s_ub = jnp.where(
            problem.null_ub,
            self.initial_slack,
            jnp.maximum(problem.ub - x0, self.initial_slack),
        )
        return cast(
            InteriorPointPrimal,
            InteriorPointPrimal(x=x0, slack=Slack(s=s, s_lb=s_lb, s_ub=s_ub)),
        )

    def _init_dual(self, problem: ProblemProtocol[InteriorPointPrimal]) -> Dual:
        """Positive inequality / bound multipliers for the primal-dual path."""
        return cast(
            Dual,
            Dual(
                eq_multipliers=jnp.zeros((problem.meq,)),
                ineq_multipliers=jnp.ones((problem.mineq,)),
                lb_multipliers=jnp.where(problem.null_lb, 0.0, 1.0),
                ub_multipliers=jnp.where(problem.null_ub, 0.0, 1.0),
            ),
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
            ),
        )

    def _lagrangian_module(
        self, problem: ProblemProtocol[InteriorPointPrimal]
    ) -> Lagrangian[InteriorPointPrimal, EvaluatedLagrangian[InteriorPointPrimal]]:
        """Barrier-augmented Lagrangian at the current secant / barrier."""
        return cast(
            Lagrangian[InteriorPointPrimal, EvaluatedLagrangian[InteriorPointPrimal]],
            InteriorPointLagrangian(
                problem,
                self.secant,
                cast(Barrier, self.barrier),
                primal_dual=self.primal_dual,
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

    def _init_subproblem(
        self, problem: ProblemProtocol[InteriorPointPrimal]
    ) -> SubproblemContext[
        InteriorPointPrimal, ScaledBarrierSubProblem, TrustRegionStateType
    ]:
        """Build a scaled-barrier trust-region context at the current ``(x, s)``."""
        iterate = cast(InteriorPointPrimal, self.iterate)
        dual = cast(Dual, self.dual)
        lag_module = cast(InteriorPointLagrangian, self._lagrangian_module(problem))
        lag = lag_module(iterate, dual)
        solver = self._make_subproblem_solver(problem)
        sub_opts = dict(self.options.get("subproblem", {}))
        if sub_opts:
            solver = solver.init(**sub_opts)
        zero_warm = cast(InteriorPointPrimal, jax.tree.map(jnp.zeros_like, iterate))
        return cast(
            SubproblemContext[
                InteriorPointPrimal, ScaledBarrierSubProblem, TrustRegionStateType
            ],
            SubproblemContext(
                problem=problem,
                lagrangian=lag_module,
                subproblem=ScaledBarrierSubProblem(lag),
                solver=solver,
                warm=(zero_warm, dual),
                state=cast(TrustRegionStateType, self.solver_state),
            ),
        )

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
        """Reduce ``μ`` from the KKT / complementarity state at the new iterate.

        Also records whether the policy judged the barrier subproblem solved,
        which :meth:`termination_metrics` reads back as the inner
        ``E(·; μ) ≤ ε_μ`` test of N&W Algorithm 19.4. That test is evaluated
        at the iterate, so it runs on rejected steps too — ``result.x`` is
        then the retained iterate.

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
            Minimiser with ``barrier`` and ``barrier_updated`` refreshed.
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
        return eqx.tree_at(
            lambda m: (m.barrier, m.barrier_updated),
            self,
            (new_barrier, updated),
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
        tracked = (
            ctx.lagrangian.value,
            ctx.lagrangian.x_grad,
            primal,
            self.dual,
            optimality_residual,
        )
        nonfinite = jnp.logical_not(
            jnp.all(
                jnp.stack(
                    [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(tracked)]
                )
            )
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
        fatal = (
            cast(TrustRegionStateType, ctx.solver_state).status
            != SUBPROBLEM_RESULTS.successful
        )
        converged = (
            barrier_updated
            & acceptable_residual
            & metrics.has_min_steps
            & ~nonfinite
            & ~fatal
        )
        return cast(
            TerminationFlags,
            TerminationFlags(
                converged=converged,
                nonfinite=nonfinite,
                fatal=fatal,
                fatal_result=metrics.fatal_result,
            ),
        )
