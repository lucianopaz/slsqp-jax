from typing import Generic, cast

import equinox as eqx
import jax
from equinox import Enumeration, tree_at
from jax import numpy as jnp
from typing_extensions import TypeVar

from ...active_set import ActiveSet
from ...dual import Dual
from ...primal import Primal
from ...types import Scalar
from ..active_set import ActiveSetSubProblem
from .base import (
    RESULTS,
    SubProblemSolver,
    SubProblemSolverState,
)
from .projected_cg import ProjectedCGState, ProjectedCGSubProblemSolver
from .working_set_policy import ThresholdWorkingSetPolicy, WorkingSetPolicy

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
    final_working_tol
        Working tolerance the policy had reached when the most recent loop
        stopped (equals ``tol`` unless an EXPAND ramp is active).
    n_anti_cycling
        Total number of solves stopped by the anti-cycling guard.
    """

    n_cg_iter: int
    last_n_iter: int
    last_n_cg_iter: int
    qp_result: ACTIVE_SET_QP_RESULTS
    active_set: ActiveSet
    dual: Dual
    final_working_tol: Scalar
    n_anti_cycling: int


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
    """Primal-dual active-set QP solver with a pluggable working-set policy.

    Each outer iteration builds a
    :class:`~slsqp_jax.sqpdax.subproblem.active_set.ActiveSetSubProblem` for
    the current working set, solves the equality-constrained KKT system with
    ``subproblem_solver`` (default
    :class:`~slsqp_jax.sqpdax.subproblem.solver.projected_cg.ProjectedCGSubProblemSolver`),
    then asks ``working_set_policy`` for the next working set. The default
    :class:`~slsqp_jax.sqpdax.subproblem.solver.working_set_policy.ThresholdWorkingSetPolicy`
    refreshes the *entire* set with a fixed threshold:

    * **add** an inequality / bound whose linearised value is violated by more
      than ``tol`` at the computed step, and
    * **drop** an active inequality / bound whose recovered multiplier is
      below ``-tol`` (wrong-sign dual for the ``h(x) ≤ 0`` / bound convention).

    Its EXPAND ramp, multiplier drop floor and set-level anti-cycling guard
    are opt-in through the policy's fields (or
    ``options['subproblem']['working_set_policy']`` from a minimiser).

    The loop stops when the working set stops changing (KKT-optimal for the
    QP), when the policy raises its anti-cycling flag, or when the policy's
    ``max_iter`` budget is exhausted; the reason is recorded in
    :attr:`ActiveSetQPSolverState.qp_result`. The base tolerance and the
    budget live on the policy (``working_set_policy.tol`` /
    ``working_set_policy.max_iter``); :attr:`tol` and :attr:`max_iter` are
    read-only views of them.

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
    warm_start
        Merge the incoming state's ``active_set`` into the cold start and use
        its ``dual`` as the first KKT warm start. Defaults to ``False``.
    working_set_policy
        :class:`~slsqp_jax.sqpdax.subproblem.solver.working_set_policy.WorkingSetPolicy`
        proposing the next working set after each KKT solve; also carries
        the base tolerance and the iteration budget.
    subproblem_solver
        Inner KKT solver for each fixed working set.
    """

    # Outer QP-loop state (this solver); distinct from ``KKTSolverStateType``.
    solver_state_class: type[ActiveSetQPSolverState] = ActiveSetQPSolverState
    warm_start: bool = eqx.field(static=True, default=False)
    working_set_policy: WorkingSetPolicy = eqx.field(
        default_factory=ThresholdWorkingSetPolicy
    )
    subproblem_solver: SubProblemSolver[
        Primal, ActiveSetSubProblem, KKTSolverStateType
    ] = eqx.field(default_factory=ProjectedCGSubProblemSolver)

    @property
    def tol(self) -> float:
        """Base add / drop tolerance, read from :attr:`working_set_policy`."""
        return self.working_set_policy.tol

    @property
    def max_iter(self) -> int:
        """Working-set iteration budget, read from :attr:`working_set_policy`."""
        return self.working_set_policy.max_iter

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
        policy = self.working_set_policy
        lag = subproblem.lagrangian
        x = lag.ref.x
        tol = jnp.asarray(self.tol, x.dtype)
        meq = lag.evaluated.meq

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

        # Carry layout: (proposed set, set the step was solved on, step, KKT
        # state, policy state, iteration, set changed?, anti-cycling fired?).
        def cond_fn(carry):
            _next, _solved, _step, _kkt, _policy, n_iter, changed, cycled = carry
            return changed & ~cycled & (n_iter < self.max_iter)

        def body_fn(carry):
            active_set, _solved, step, kkt_state, policy_state, n_iter, _, _ = carry
            step_new, kkt_state_new = run_kkt(active_set, step, kkt_state)
            next_set, policy_state_new, cycled = policy.update(
                lag, step_new, active_set, policy_state
            )
            changed = jnp.logical_not(eqx.tree_equal(next_set, active_set))
            return (
                next_set,
                active_set,
                step_new,
                kkt_state_new,
                policy_state_new,
                n_iter + 1,
                changed,
                cycled,
            )

        init_carry = (
            active0,
            active0,
            x0,
            kkt_state0,
            policy.init_state(active0),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(True),
            jnp.asarray(False),
        )
        (
            next_f,
            solved_f,
            step_f,
            kkt_state_f,
            policy_state_f,
            n_iter_f,
            changed_f,
            cycled_f,
        ) = jax.lax.while_loop(cond_fn, body_fn, init_carry)
        step_f = cast(tuple[Primal, Dual], step_f)
        kkt_state_f = cast(KKTSolverStateType, kkt_state_f)
        # Pin the counter dtypes: the outer minimiser threads this state
        # through its own ``while_loop``, whose carry must be dtype-stable
        # whether or not x64 is enabled.
        n_iter_f = jnp.asarray(n_iter_f, jnp.int32)
        n_cg_f = jnp.asarray(kkt_state_f.n_iter, jnp.int32)

        # Converged QP == the working set stopped changing.  Propagate a KKT
        # failure (singular / unconverged) verbatim; otherwise it's success iff
        # the working set settled inside the iteration budget without cycling.
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
                ACTIVE_SET_QP_RESULTS.where(
                    cycled_f,
                    ACTIVE_SET_QP_RESULTS.anti_cycling,
                    ACTIVE_SET_QP_RESULTS.max_iter_reached,
                ),
            ),
            ACTIVE_SET_QP_RESULTS.kkt_solver_failure,
        )
        # Carry the proposed set (it already holds the newly violated rows,
        # which is what a warm start wants) except when the guard fired: the
        # proposal is then a known cycle member, so keep the set the returned
        # step was actually solved on.
        active_f = cast(
            ActiveSet,
            jax.tree.map(
                lambda solved, proposed: jnp.where(cycled_f, solved, proposed),
                solved_f,
                next_f,
            ),
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
                    state.final_working_tol,
                    state.n_anti_cycling,
                ),
                initial_state,
                (
                    jnp.asarray(initial_state.n_iter, jnp.int32) + n_iter_f,
                    jnp.asarray(initial_state.n_cg_iter, jnp.int32) + n_cg_f,
                    n_iter_f,
                    n_cg_f,
                    success,
                    status,
                    qp_result,
                    active_f,
                    step_f[1],
                    jnp.asarray(policy_state_f.working_tol, x.dtype),
                    jnp.asarray(initial_state.n_anti_cycling, jnp.int32)
                    + cycled_f.astype(jnp.int32),
                ),
            ),
        )
