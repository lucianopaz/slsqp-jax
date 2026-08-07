"""Trust-region interior-point outer loop (Nocedal & Wright §19.5)."""

from __future__ import annotations

from dataclasses import replace
from typing import Self, cast

import equinox as eqx
import jax
import optimistix as optx
from jax import numpy as jnp
from jaxtyping import Array, Bool

from ..barrier import Barrier, BarrierUpdate, LogBarrier, MonotoneBarrierUpdate
from ..barrier.update import _inf_norm
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
from ..step_controller import StepController, TrustRegionManager
from ..subproblem import ScaledBarrierSubProblem
from ..subproblem.solver import (
    RESULTS,
    SubproblemContext,
    SubProblemSolver,
    TrustRegionInteriorPointSolver,
    TrustRegionSolverState,
)
from ..types import Scalar, Vector_n
from .base import CommonMinimiser, OptimisationContext

__all__ = [
    "TrustRegionInteriorPointMinimiser",
]


class TrustRegionInteriorPointMinimiser(
    CommonMinimiser[
        InteriorPointPrimal, ScaledBarrierSubProblem, TrustRegionSolverState
    ]
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
        Policy that reduces ``μ`` after each accepted iterate.
    eta, shrink_threshold, grow_threshold, shrink_factor, grow_factor, max_radius
        Forwarded to :class:`TrustRegionManager`.
    barrier
        Dynamic barrier whose ``weight`` *is* ``μ`` (seeded in
        :meth:`_init_dynamics`).
    """

    initial_mu: float = eqx.field(static=True, default=1.0)
    initial_slack: float = eqx.field(static=True, default=1.0)
    initial_radius: float = eqx.field(static=True, default=1.0)
    initial_penalty: float = eqx.field(static=True, default=1.0)
    primal_dual: bool = eqx.field(static=True, default=True)
    barrier_update: BarrierUpdate = eqx.field(default_factory=MonotoneBarrierUpdate)
    # trust-region acceptance / radius policy (forwarded to TrustRegionManager)
    eta: float = eqx.field(static=True, default=1e-4)
    shrink_threshold: float = eqx.field(static=True, default=0.25)
    grow_threshold: float = eqx.field(static=True, default=0.75)
    shrink_factor: float = eqx.field(static=True, default=0.25)
    grow_factor: float = eqx.field(static=True, default=2.0)
    max_radius: float = eqx.field(static=True, default=1e10)
    # The barrier is dynamic state: its ``weight`` *is* mu, updated each step.
    barrier: Barrier | None = None

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
    ) -> TrustRegionSolverState:
        """Cold trust-region carry with :attr:`initial_radius` / :attr:`initial_penalty`."""
        return cast(
            TrustRegionSolverState,
            TrustRegionSolverState(
                n_iter=jnp.asarray(0, jnp.int32),
                radius=jnp.asarray(self.initial_radius),
                predicted_reduction=jnp.asarray(0.0),
                merit_penalty=jnp.asarray(self.initial_penalty),
                n_cg_iter=jnp.asarray(0, jnp.int32),
                on_boundary=jnp.asarray(False),
                success=jnp.asarray(False),
                status=RESULTS.successful,
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

    def _init_subproblem(
        self, problem: ProblemProtocol[InteriorPointPrimal]
    ) -> SubproblemContext[
        InteriorPointPrimal, ScaledBarrierSubProblem, TrustRegionSolverState
    ]:
        """Build a scaled-barrier trust-region context at the current ``(x, s)``."""
        iterate = cast(InteriorPointPrimal, self.iterate)
        dual = cast(Dual, self.dual)
        lag_module = cast(InteriorPointLagrangian, self._lagrangian_module(problem))
        lag = lag_module(iterate, dual)
        solver = cast(TrustRegionInteriorPointSolver, TrustRegionInteriorPointSolver())
        sub_opts = dict(self.options.get("subproblem", {}))
        if sub_opts:
            solver = solver.init(**sub_opts)
        zero_warm = cast(InteriorPointPrimal, jax.tree.map(jnp.zeros_like, iterate))
        return cast(
            SubproblemContext[
                InteriorPointPrimal, ScaledBarrierSubProblem, TrustRegionSolverState
            ],
            SubproblemContext(
                problem=problem,
                lagrangian=lag_module,
                subproblem=ScaledBarrierSubProblem(lag),
                solver=solver,
                warm=(zero_warm, dual),
                state=cast(TrustRegionSolverState, self.solver_state),
            ),
        )

    def _step_controller(
        self,
        ctx: SubproblemContext[
            InteriorPointPrimal, ScaledBarrierSubProblem, TrustRegionSolverState
        ],
        step_dual: Dual,
        solver_state: TrustRegionSolverState,
    ) -> StepController[InteriorPointPrimal, TrustRegionSolverState]:
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
            StepController[InteriorPointPrimal, TrustRegionSolverState],
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

    def _feasibility_error(
        self,
        ctx: OptimisationContext[InteriorPointPrimal, TrustRegionSolverState],
    ) -> Scalar:
        """``∞``-norm of the slacked primal residual (``dual_grad`` blocks)."""
        residual = ctx.lagrangian.dual_grad
        return jnp.max(
            jnp.stack(
                [
                    _inf_norm(residual.eq_multipliers),
                    _inf_norm(residual.ineq_multipliers),
                    _inf_norm(residual.lb_multipliers),
                    _inf_norm(residual.ub_multipliers),
                ]
            )
        )

    def _extra_optimality(
        self,
        ctx: OptimisationContext[InteriorPointPrimal, TrustRegionSolverState],
    ) -> Scalar:
        """Complementarity / barrier-weight residual compared against ``atol``."""
        lag = cast(InteriorPointEvaluatedLagrangian, ctx.lagrangian)
        barrier = cast(Barrier, self.barrier)
        comp = self.barrier_update.complementarity(lag)
        return jnp.maximum(comp, barrier.weight)

    def _termination_diagnostics(
        self,
        ctx: OptimisationContext[InteriorPointPrimal, TrustRegionSolverState],
    ) -> tuple[tuple[Bool[Array, ""], optx.RESULTS], ...]:
        """Escalate a failed composite-step solve (non-finite → ``singular``)."""
        assert ctx.solver_state is not None
        status = optx.RESULTS.promote(ctx.solver_state.status)
        failed = status != optx.RESULTS.successful
        return super()._termination_diagnostics(ctx) + ((failed, status),)

    def _advance_dynamics(
        self,
        ctx: SubproblemContext[
            InteriorPointPrimal, ScaledBarrierSubProblem, TrustRegionSolverState
        ],
        x_new: InteriorPointPrimal,
        step_dual: Dual,
    ) -> Self:
        """Reduce ``μ`` from the KKT / complementarity state at the new iterate."""
        # ``ctx.lagrangian`` is the module; re-evaluate at (x_new, step_dual)
        # exactly as ``_update_secant`` does. Complementarity is secant-independent,
        # so using the step's module (pre-update barrier) is correct.
        lag_module = cast(InteriorPointLagrangian, ctx.lagrangian)
        lagrangian = lag_module(x_new, step_dual)
        new_barrier = self.barrier_update.update(
            cast(Barrier, self.barrier), lagrangian
        )
        return eqx.tree_at(lambda m: m.barrier, self, new_barrier)
