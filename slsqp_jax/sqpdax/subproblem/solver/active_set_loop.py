from typing import Generic, cast

import equinox as eqx
import jax
from jax import numpy as jnp
from typing_extensions import TypeVar

from ...active_set import ActiveSet
from ...dual import Dual
from ...primal import Primal
from ..active_set import ActiveSetSubProblem
from .base import (
    RESULTS,
    SubProblemSolver,
    SubProblemSolverState,
)
from .projected_cg import ProjectedCGState, ProjectedCGSubProblemSolver

# Inner KKT-solver state; defaults to projected CG so bare ``ActiveSetQPSolver``
# matches ``default_factory=ProjectedCGSubProblemSolver``.
KKTSolverStateType = TypeVar(
    "KKTSolverStateType", bound=SubProblemSolverState, default=ProjectedCGState
)


class ActiveSetQPSolverState(SubProblemSolverState):
    """Carry for the outer primal-dual active-set QP loop.

    Attributes
    ----------
    n_cg_iter
        Cumulative inner KKT (projected-CG) iterations across working-set
        refreshes. Seeded from the incoming state and updated from the
        inner solver's ``n_iter``.
    """

    n_cg_iter: int


class ActiveSetQPSolver(
    SubProblemSolver[Primal, ActiveSetSubProblem, ActiveSetQPSolverState],
    Generic[KKTSolverStateType],
):
    """Minimal primal-dual active-set QP solver.

    Each outer iteration builds a
    :class:`~slsqp_jax.sqpdax.subproblem.active_set.ActiveSetSubProblem` for
    the current working set, solves the equality-constrained KKT system with
    ``subproblem_solver`` (default
    :class:`~slsqp_jax.sqpdax.subproblem.solver.projected_cg.ProjectedCGSubProblemSolver`),
    then refreshes the *entire* working set with a plain fixed threshold (no
    EXPAND ramp):

    * **add** an inequality / bound whose linearised value is violated by more
      than ``tol`` at the computed step, and
    * **drop** an active inequality / bound whose recovered multiplier is
      below ``-tol`` (wrong-sign dual for the ``h(x) ≤ 0`` / bound convention).

    The loop stops when the working set stops changing (KKT-optimal for the
    QP) or the ``max_iter`` budget is exhausted. An EXPAND-style anti-cycling
    variant belongs in a subclass overriding the working-set update.

    Attributes
    ----------
    solver_state_class
        :class:`ActiveSetQPSolverState`.
    tol
        Fixed add/drop threshold for the working-set update.
    max_iter
        Maximum outer working-set iterations.
    subproblem_solver
        Inner KKT solver for each fixed working set.
    """

    # Outer QP-loop state (this solver); distinct from ``KKTSolverStateType``.
    solver_state_class: type[ActiveSetQPSolverState] = ActiveSetQPSolverState
    tol: float = 1e-8
    max_iter: int = 50
    subproblem_solver: SubProblemSolver[
        Primal, ActiveSetSubProblem, KKTSolverStateType
    ] = eqx.field(default_factory=ProjectedCGSubProblemSolver)

    def solve(
        self,
        subproblem: ActiveSetSubProblem,
        x0: tuple[Primal, Dual],
        initial_state: ActiveSetQPSolverState,
    ) -> tuple[tuple[Primal, Dual], ActiveSetQPSolverState]:
        """Solve the inequality / bound QP by an active-set loop.

        Parameters
        ----------
        subproblem
            Reference
            :class:`~slsqp_jax.sqpdax.subproblem.active_set.ActiveSetSubProblem`
            whose unmasked Lagrangian defines the QP. The incoming working
            set is ignored; a cold start is built from current violations.
        x0
            Warm-start ``(primal_step, dual)`` for the first KKT solve.
        initial_state
            Outer carry; ``n_cg_iter`` seeds the inner solver's ``n_iter``.

        Returns
        -------
        step
            Primal-dual QP solution.
        state
            Updated :class:`ActiveSetQPSolverState` (outer / CG counts,
            success, status).
        """
        lag = subproblem.lagrangian
        x = lag.ref.x
        tol = jnp.asarray(self.tol, x.dtype)
        meq = lag.evaluated.meq

        def working_set(step: tuple[Primal, Dual], current: ActiveSet) -> ActiveSet:
            dx = step[0].x
            lam_ineq = step[1].ineq_multipliers
            lam_lb = step[1].lb_multipliers
            lam_ub = step[1].ub_multipliers
            x_new = x + dx

            # ADD violated (linearised value > tol), DROP active-with-neg-dual.
            ineq_lin = lag.ineq_fn_val + lag.ineq_fn_jac_val @ dx
            ai = current.active_inequalities
            new_ai = (ai | (ineq_lin > tol)) & ~(ai & (lam_ineq < -tol))

            alb = current.active_lb
            lb_violated = (lag.lb - x_new) > tol
            new_alb = (~lag.null_lb) & ((alb | lb_violated) & ~(alb & (lam_lb < -tol)))

            aub = current.active_ub
            ub_violated = (x_new - lag.ub) > tol
            new_aub = (~lag.null_ub) & ((aub | ub_violated) & ~(aub & (lam_ub < -tol)))
            return cast(
                ActiveSet,
                ActiveSet(
                    meq=meq,
                    active_inequalities=new_ai,
                    active_lb=new_alb,
                    active_ub=new_aub,
                ),
            )

        def set_changed(a: ActiveSet, b: ActiveSet):
            return (
                jnp.any(a.active_inequalities != b.active_inequalities)
                | jnp.any(a.active_lb != b.active_lb)
                | jnp.any(a.active_ub != b.active_ub)
            )

        # Cold start: whatever is already active / violated at the current point.
        active0 = cast(
            ActiveSet,
            ActiveSet(
                meq=meq,
                active_inequalities=lag.ineq_fn_val >= -tol,
                active_lb=(~lag.null_lb) & ((x - lag.lb) <= tol),
                active_ub=(~lag.null_ub) & ((lag.ub - x) <= tol),
            ),
        )

        kkt_state0 = self.subproblem_solver.solver_state_class(
            n_iter=jnp.asarray(initial_state.n_cg_iter, jnp.int32),
            success=jnp.asarray(False),
            status=RESULTS.successful,
        )

        def run_kkt(
            active_set: ActiveSet,
            warm: tuple[Primal, Dual],
            kkt_state: KKTSolverStateType,
        ) -> tuple[tuple[Primal, Dual], KKTSolverStateType]:
            subproblem_k = cast(
                ActiveSetSubProblem,
                ActiveSetSubProblem(subproblem.lagrangian, active_set),
            )
            return self.subproblem_solver.solve(subproblem_k, warm, kkt_state)

        def cond_fn(carry):
            _active, _step, _kkt_state, n_iter, changed = carry
            return changed & (n_iter < self.max_iter)

        def body_fn(carry):
            active_set, step, kkt_state, n_iter, _ = carry
            step_new, kkt_state_new = run_kkt(active_set, step, kkt_state)
            next_set = working_set(step_new, active_set)
            changed = set_changed(next_set, active_set)
            return (next_set, step_new, kkt_state_new, n_iter + 1, changed)

        init_carry = (
            active0,
            x0,
            kkt_state0,
            jnp.asarray(initial_state.n_iter, jnp.int32),
            jnp.asarray(True),
        )
        _active_f, step_f, kkt_state_f, n_iter_f, changed_f = jax.lax.while_loop(
            cond_fn, body_fn, init_carry
        )
        step_f = cast(tuple[Primal, Dual], step_f)
        kkt_state_f = cast(KKTSolverStateType, kkt_state_f)

        # Converged QP == the working set stopped changing.  Propagate a KKT
        # failure (singular / unconverged) verbatim; otherwise it's success iff
        # the working set settled inside the iteration budget.
        qp_converged = ~changed_f
        success = qp_converged & kkt_state_f.success
        status = RESULTS.where(
            kkt_state_f.success,
            RESULTS.where(qp_converged, RESULTS.successful, RESULTS.max_steps_reached),
            kkt_state_f.status,
        )
        return step_f, cast(
            ActiveSetQPSolverState,
            ActiveSetQPSolverState(
                n_iter=n_iter_f,
                n_cg_iter=kkt_state_f.n_iter,
                success=success,
                status=status,
            ),
        )
