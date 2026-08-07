"""Active-set QP subproblem + Armijo line-search outer loop."""

from __future__ import annotations

from typing import Self, cast

import equinox as eqx
import optimistix as optx
from jax import numpy as jnp
from jaxtyping import Array, Bool

from ..active_set import ActiveSet
from ..barrier.update import _inf_norm
from ..dual import Dual
from ..lagrangian import EvaluatedLagrangian, Lagrangian
from ..merit import NormMerit
from ..primal import Primal
from ..problem import ProblemProtocol
from ..step_controller import ArmijoLineSearch, StepController
from ..subproblem import ActiveSetSubProblem
from ..subproblem.solver import (
    RESULTS,
    ActiveSetQPSolver,
    ActiveSetQPSolverState,
    ProjectedCGSubProblemSolver,
    SubproblemContext,
    SubProblemSolver,
)
from ..types import Scalar
from .base import CommonMinimiser, OptimisationContext

__all__ = [
    "ActiveSetLineSearchMinimiser",
]


class ActiveSetLineSearchMinimiser(
    CommonMinimiser[Primal, ActiveSetSubProblem, ActiveSetQPSolverState]
):
    """SLSQP-style active-set QP subproblem + L1-merit backtracking line search.

    Each outer step (i) freezes the Lagrangian Hessian (exact HVP or L-BFGS
    secant), (ii) solves the equality / inequality / bound QP with
    :class:`~slsqp_jax.sqpdax.subproblem.solver.active_set_loop.ActiveSetQPSolver`
    (a primal-dual active-set loop around projected CG), and (iii) globalises
    the QP direction with an Armijo backtracking search on the L1 merit

    ```
    φ(x) = f(x) + ρ (‖c_eq‖₁ + ‖[c_ineq]₊‖₁)
    ```

    whose penalty ``ρ`` is refreshed from the current QP multipliers. This is
    the line-search counterpart of
    :class:`~slsqp_jax.sqpdax.minimiser.trust_region_interior_point.TrustRegionInteriorPointMinimiser`
    and shares the entire ``init`` / ``step`` / ``terminate`` / ``postprocess``
    driver with it.

    Attributes
    ----------
    qp_tol
        Fixed add/drop threshold for the active-set QP working-set update.
    qp_max_iter
        Maximum outer working-set iterations of the QP solver.
    penalty_floor
        Lower bound on the L1 merit penalty ``ρ``.
    penalty_factor
        Multiplier on ``‖λ‖_∞`` used to form ``ρ``.
    armijo_c1
        Armijo sufficient-decrease constant.
    armijo_backtrack
        Geometric backtracking factor.
    line_search_max_steps
        Maximum Armijo trial evaluations per outer step.
    """

    # QP inner solve
    qp_tol: float = eqx.field(static=True, default=1e-8)
    qp_max_iter: int = eqx.field(static=True, default=20)
    # L1-merit penalty schedule: rho = max(penalty_floor, penalty_factor * ||lambda||_inf)
    penalty_floor: float = eqx.field(static=True, default=1.0)
    penalty_factor: float = eqx.field(static=True, default=2.0)
    # backtracking line search
    armijo_c1: float = eqx.field(static=True, default=1e-4)
    armijo_backtrack: float = eqx.field(static=True, default=0.5)
    line_search_max_steps: int = eqx.field(static=True, default=20)

    def _subproblem_solver_type(self) -> type[SubProblemSolver]:
        """Root solver class for ``options['subproblem']`` validation."""
        return ActiveSetQPSolver

    def _init_solver_state(
        self, problem: ProblemProtocol[Primal], primal: Primal
    ) -> ActiveSetQPSolverState:
        """Cold :class:`ActiveSetQPSolverState` for the first outer step."""
        return cast(
            ActiveSetQPSolverState,
            ActiveSetQPSolverState(
                n_iter=jnp.asarray(0, jnp.int32),
                n_cg_iter=jnp.asarray(0, jnp.int32),
                success=jnp.asarray(False),
                status=RESULTS.successful,
            ),
        )

    def _lagrangian_module(
        self, problem: ProblemProtocol[Primal]
    ) -> Lagrangian[Primal, EvaluatedLagrangian[Primal]]:
        """Unevaluated Lagrangian wrapping the current secant (if any)."""
        return cast(
            Lagrangian[Primal, EvaluatedLagrangian[Primal]],
            Lagrangian(problem, self.secant),
        )

    def _init_subproblem(
        self, problem: ProblemProtocol[Primal]
    ) -> SubproblemContext[Primal, ActiveSetSubProblem, ActiveSetQPSolverState]:
        """Build an active-set QP context at the current iterate.

        The Lagrangian is evaluated at a *zero* dual so the QP recovers the
        full multiplier ``λ_{k+1}``. The working set starts empty (cold).
        """
        iterate = cast(Primal, self.iterate)
        dual = cast(Dual, self.dual)
        n, meq, mineq = problem.n, problem.meq, problem.mineq
        dtype = iterate.x.dtype
        lag_module = self._lagrangian_module(problem)
        # Zero-dual eval => QP returns the *full* multiplier lambda_{k+1}.
        qp_lag = lag_module(iterate, self._init_dual(problem))

        solver = cast(
            ActiveSetQPSolver,
            ActiveSetQPSolver(
                subproblem_solver=ProjectedCGSubProblemSolver(),
                tol=self.qp_tol,
                max_iter=self.qp_max_iter,
            ),
        )
        sub_opts = dict(self.options.get("subproblem", {}))
        if sub_opts:
            solver = solver.init(**sub_opts)
        empty_active = ActiveSet(
            meq=meq,
            active_inequalities=jnp.zeros((mineq,), bool),
            active_lb=jnp.zeros((n,), bool),
            active_ub=jnp.zeros((n,), bool),
        )
        return cast(
            SubproblemContext[Primal, ActiveSetSubProblem, ActiveSetQPSolverState],
            SubproblemContext(
                problem=problem,
                lagrangian=lag_module,
                subproblem=ActiveSetSubProblem(qp_lag, empty_active),
                solver=solver,
                warm=(Primal(jnp.zeros((n,), dtype)), dual),
                state=cast(ActiveSetQPSolverState, self.solver_state),
            ),
        )

    def _step_controller(
        self,
        ctx: SubproblemContext[Primal, ActiveSetSubProblem, ActiveSetQPSolverState],
        step_dual: Dual,
        solver_state: ActiveSetQPSolverState,
    ) -> StepController[Primal, ActiveSetQPSolverState]:
        """Armijo line search on an L1 merit with multiplier-based penalty."""
        rho = jnp.maximum(
            self.penalty_floor, self.penalty_factor * _inf_norm(step_dual.flatten())
        )
        merit = NormMerit(
            ctx.problem,
            barrier=None,
            problem_weight=jnp.asarray(1.0),
            feasibility_weight=rho,
            norm=1,
        )
        return cast(
            StepController[Primal, ActiveSetQPSolverState],
            ArmijoLineSearch(
                merit=merit,
                max_steps=self.line_search_max_steps,
                c1=self.armijo_c1,
                backtrack=self.armijo_backtrack,
            ),
        )

    def _feasibility_error(
        self, ctx: OptimisationContext[Primal, ActiveSetQPSolverState]
    ) -> Scalar:
        """``∞``-norm of equality / inequality / bound violations."""
        lagrangian = ctx.lagrangian
        x = lagrangian.x_ref
        eq_v = _inf_norm(lagrangian.eq_fn_val)
        ineq_v = _inf_norm(jnp.maximum(0.0, lagrangian.ineq_fn_val))
        lb_v = _inf_norm(
            jnp.where(lagrangian.null_lb, 0.0, jnp.maximum(0.0, lagrangian.lb - x))
        )
        ub_v = _inf_norm(
            jnp.where(lagrangian.null_ub, 0.0, jnp.maximum(0.0, x - lagrangian.ub))
        )
        return jnp.max(jnp.stack([eq_v, ineq_v, lb_v, ub_v]))

    def _termination_diagnostics(
        self, ctx: OptimisationContext[Primal, ActiveSetQPSolverState]
    ) -> tuple[tuple[Bool[Array, ""], optx.RESULTS], ...]:
        """Escalate a failed QP subproblem solve (e.g. singular KKT).

        ``status`` is a lineax code, so it is promoted to the optimistix
        enumeration before it reaches ``RESULTS.where``. Init states carry
        ``successful`` so this never fires at step 0.
        """
        assert ctx.solver_state is not None
        status = optx.RESULTS.promote(ctx.solver_state.status)
        failed = status != optx.RESULTS.successful
        return super()._termination_diagnostics(ctx) + ((failed, status),)

    def _advance_dynamics(
        self,
        ctx: SubproblemContext[Primal, ActiveSetSubProblem, ActiveSetQPSolverState],
        x_new: Primal,
        step_dual: Dual,
    ) -> Self:
        """No post-iterate dynamics (secant is refreshed in ``_execute_step``)."""
        return self
