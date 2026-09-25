from typing import Generic, cast

import equinox as eqx
import jax
from equinox import Enumeration, tree_at
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

__all__ = [
    "ACTIVE_SET_QP_RESULTS",
    "ActiveSetQPSolverState",
    "ActiveSetStateType",
    "KKTSolverStateType",
    "ActiveSetQPSolver",
]

# Inner KKT-solver state; defaults to projected CG so bare ``ActiveSetQPSolver``
# matches ``default_factory=ProjectedCGSubProblemSolver``.
KKTSolverStateType = TypeVar(
    "KKTSolverStateType", bound=SubProblemSolverState, default=ProjectedCGState
)


class ACTIVE_SET_QP_RESULTS(Enumeration):
    """Why the active-set working-set loop stopped.

    Distinguishes the four outcomes that the shared
    :attr:`~slsqp_jax.sqpdax.subproblem.solver.base.SubProblemSolverState.status`
    code conflates: a clean working-set fixed point, a failure of the inner
    KKT solver, the anti-cycling guard, and budget exhaustion.
    """

    working_set_converged = "The working set stopped changing (QP KKT point)."
    kkt_solver_failure = "The inner KKT solver failed on the final working set."
    anti_cycling = "The working set cycled and the anti-cycling guard stopped the loop."
    max_iter_reached = "The working-set iteration budget was exhausted."


class ActiveSetQPSolverState(SubProblemSolverState):
    """Carry for the outer primal-dual active-set QP loop.

    The inherited ``n_iter`` counts working-set iterations accumulated over
    every solve that has consumed this carry (i.e. over the whole nonlinear
    solve); ``last_n_iter`` is the count of the most recent solve alone. The
    same split applies to the inner KKT iterations.

    Attributes
    ----------
    n_cg_iter
        Total inner KKT (projected-CG) iterations accumulated across solves.
    last_n_iter
        Working-set iterations of the most recent solve.
    last_n_cg_iter
        Inner KKT iterations of the most recent solve.
    qp_result
        :class:`ACTIVE_SET_QP_RESULTS` code explaining why the most recent
        working-set loop stopped.
    active_set
        Final working set of the most recent solve. Only consumed when the
        solver's ``warm_start`` flag is set.
    dual
        Multipliers returned by the most recent solve. Only consumed when the
        solver's ``warm_start`` flag is set.
    """

    n_cg_iter: int
    last_n_iter: int
    last_n_cg_iter: int
    qp_result: ACTIVE_SET_QP_RESULTS
    active_set: ActiveSet
    dual: Dual


# Outer QP-loop state; defaulted so bare ``ActiveSetQPSolver`` keeps meaning
# ``ActiveSetQPSolver[ProjectedCGState, ActiveSetQPSolverState]`` while
# subclasses can bind a richer carry and specialise ``solve`` without casts.
ActiveSetStateType = TypeVar(
    "ActiveSetStateType", bound=ActiveSetQPSolverState, default=ActiveSetQPSolverState
)


class ActiveSetQPSolver(
    SubProblemSolver[Primal, ActiveSetSubProblem, ActiveSetStateType],
    Generic[KKTSolverStateType, ActiveSetStateType],
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
    QP) or the ``max_iter`` budget is exhausted; the reason is recorded in
    :attr:`ActiveSetQPSolverState.qp_result`. An EXPAND-style anti-cycling
    variant belongs in a subclass overriding the working-set update.

    The initial working set is by default rebuilt from scratch at every
    solve (constraints active or violated at the reference point). With
    ``warm_start=True`` the working set and multipliers carried on the
    incoming state (the previous solve's answer) are merged into that cold
    start, which saves working-set iterations when the active set is stable
    across outer iterations but can cost iterations when it jumps around.

    Attributes
    ----------
    solver_state_class
        :class:`ActiveSetQPSolverState`.
    tol
        Fixed add/drop threshold for the working-set update.
    max_iter
        Maximum working-set iterations per solve.
    warm_start
        Merge the incoming state's ``active_set`` into the cold start and use
        its ``dual`` as the first KKT warm start. Defaults to ``False``.
    subproblem_solver
        Inner KKT solver for each fixed working set.
    """

    # Outer QP-loop state (this solver); distinct from ``KKTSolverStateType``.
    solver_state_class: type[ActiveSetQPSolverState] = ActiveSetQPSolverState
    tol: float = 1e-8
    max_iter: int = 50
    warm_start: bool = eqx.field(static=True, default=False)
    subproblem_solver: SubProblemSolver[
        Primal, ActiveSetSubProblem, KKTSolverStateType
    ] = eqx.field(default_factory=ProjectedCGSubProblemSolver)

    def _kkt_solver(
        self, subproblem: ActiveSetSubProblem
    ) -> SubProblemSolver[Primal, ActiveSetSubProblem, KKTSolverStateType]:
        """Inner KKT solver used for every working set of one :meth:`solve`.

        Called once per solve, outside the working-set loop, so subclasses
        may derive a per-solve variant of :attr:`subproblem_solver` (e.g.
        re-wrap its preconditioner around data carried by ``subproblem``)
        without rebuilding it for each working-set refresh.

        Parameters
        ----------
        subproblem
            The QP handed to :meth:`solve` (after any subclass preprocessing).

        Returns
        -------
        SubProblemSolver
            :attr:`subproblem_solver` unchanged by default.
        """
        return self.subproblem_solver

    def solve(
        self,
        subproblem: ActiveSetSubProblem,
        x0: tuple[Primal, Dual],
        initial_state: ActiveSetStateType,
    ) -> tuple[tuple[Primal, Dual], ActiveSetStateType]:
        """Solve the inequality / bound QP by an active-set loop.

        Parameters
        ----------
        subproblem
            Reference
            :class:`~slsqp_jax.sqpdax.subproblem.active_set.ActiveSetSubProblem`
            whose unmasked Lagrangian defines the QP. Its own working set is
            ignored; the initial set is the cold start built from current
            violations, merged with ``initial_state.active_set`` when
            :attr:`warm_start` is set.
        x0
            Warm-start ``(primal_step, dual)`` for the first KKT solve. The
            dual block is replaced by ``initial_state.dual`` when
            :attr:`warm_start` is set.
        initial_state
            Outer carry. Its cumulative counters are advanced by this solve;
            its ``active_set`` / ``dual`` are read only under
            :attr:`warm_start`.

        Returns
        -------
        step
            Primal-dual QP solution.
        state
            ``initial_state`` with the counters, ``success`` / ``status``,
            ``qp_result``, and the final ``active_set`` / ``dual`` refreshed
            (same type as the input).
        """
        inner = self._kkt_solver(subproblem)
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
        if self.warm_start:
            # Merge the previous solve's working set into the cold start and
            # reuse its multipliers for the first KKT solve.
            prev = initial_state.active_set
            active0 = cast(
                ActiveSet,
                ActiveSet(
                    meq=meq,
                    active_inequalities=active0.active_inequalities
                    | prev.active_inequalities,
                    active_lb=(~lag.null_lb) & (active0.active_lb | prev.active_lb),
                    active_ub=(~lag.null_ub) & (active0.active_ub | prev.active_ub),
                ),
            )
            x0 = (x0[0], initial_state.dual)

        # Per-solve budget: the inner counter starts from zero so the
        # cumulative ``n_cg_iter`` on the carry never starves a later solve.
        kkt_state0 = inner.solver_state_class(
            n_iter=0,
            success=jnp.asarray(False),
            status=RESULTS.successful,
        )

        def run_kkt(
            active_set: ActiveSet,
            warm: tuple[Primal, Dual],
            kkt_state: KKTSolverStateType,
        ) -> tuple[tuple[Primal, Dual], KKTSolverStateType]:
            subproblem_k = subproblem.with_active_set(active_set)
            return inner.solve(subproblem_k, warm, kkt_state)

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
            0,
            jnp.asarray(True),
        )
        active_f, step_f, kkt_state_f, n_iter_f, changed_f = jax.lax.while_loop(
            cond_fn, body_fn, init_carry
        )
        active_f = cast(ActiveSet, active_f)
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
        qp_result = ACTIVE_SET_QP_RESULTS.where(
            kkt_state_f.success,
            ACTIVE_SET_QP_RESULTS.where(
                qp_converged,
                ACTIVE_SET_QP_RESULTS.working_set_converged,
                ACTIVE_SET_QP_RESULTS.max_iter_reached,
            ),
            ACTIVE_SET_QP_RESULTS.kkt_solver_failure,
        )
        return step_f, cast(
            ActiveSetStateType,
            tree_at(
                lambda state: (
                    state.n_iter,
                    state.n_cg_iter,
                    state.last_n_iter,
                    state.last_n_cg_iter,
                    state.success,
                    state.status,
                    state.qp_result,
                    state.active_set,
                    state.dual,
                ),
                initial_state,
                (
                    initial_state.n_iter + n_iter_f,
                    initial_state.n_cg_iter + kkt_state_f.n_iter,
                    n_iter_f,
                    kkt_state_f.n_iter,
                    success,
                    status,
                    qp_result,
                    active_f,
                    step_f[1],
                ),
            ),
        )
