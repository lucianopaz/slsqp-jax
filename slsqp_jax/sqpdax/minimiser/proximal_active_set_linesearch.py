"""Proximal (stabilised-SQP) active-set QP + Armijo line-search outer loop."""

from __future__ import annotations

from typing import Self, cast

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Bool

from ..barrier.update import _inf_norm
from ..dual import Dual
from ..preconditioner import PreconditionerStrategy, SecantPreconditioner
from ..primal import Primal
from ..problem import ProblemProtocol
from ..secant import SecantResetSignals
from ..step_controller import StepResult
from ..subproblem import ActiveSetSubProblem
from ..subproblem.solver import (
    ACTIVE_SET_QP_RESULTS,
    KKT_SOLVER_RESULTS,
    RESULTS,
    ProjectedCGSubProblemSolver,
    ProximalActiveSetQPSolver,
    ProximalActiveSetQPSolverState,
    SubproblemContext,
    SubProblemSolver,
)
from .active_set_linesearch import (
    ACTIVE_SET_LINE_SEARCH_RESULTS,
    ActiveSetLineSearchMinimiser,
    ActiveSetLineSearchTerminationMetrics,
)

__all__ = [
    "ProximalActiveSetLineSearchMinimiser",
]


class ProximalActiveSetLineSearchMinimiser(
    ActiveSetLineSearchMinimiser[ProximalActiveSetQPSolverState]
):
    """Stabilised-SQP variant of :class:`ActiveSetLineSearchMinimiser`.

    Identical outer loop (L1 merit, Armijo backtracking, best-iterate
    rollback), but the QP direction comes from
    :class:`~slsqp_jax.sqpdax.subproblem.solver.proximal_active_set_loop.ProximalActiveSetQPSolver`,
    which eliminates the equality constraints through a proximal term with
    parameter ``μ = clip(res_k ** τ, μ_min, μ_max)``. The outer KKT residual

    ```
    res_k = max(‖∇_x L(x_k, λ_k)‖_∞, feasibility_∞(x_k))
    ```

    is read off the QP Lagrangian evaluated at the current multipliers (no
    extra problem evaluation) and written into the solver state before each
    solve. The multiplier centre ``λ_k`` lives on the
    solver state, is warm-started from the previous solve, and is re-synced to
    the committed dual after a best-iterate rollback.

    The default :attr:`preconditioner` is
    :class:`~slsqp_jax.sqpdax.preconditioner.strategy.SecantPreconditioner`
    with ``require_secant=False``: when the model uses a secant, the inner
    KKT solver is preconditioned with ``M = B`` (``M⁻¹ = H`` from the
    secant), and the proximal solver wraps it in the Woodbury update
    ``B + (1/μ) A_eqᵀ A_eq`` so the constraint preconditioner matches the
    stabilised Hessian. With exact curvature and no user-supplied
    preconditioner the solve runs unpreconditioned; pass
    ``{"minimiser": {"preconditioner": {"kind": "lbfgs"}}}`` to keep a secant
    for preconditioning anyway.

    Schedule parameters are set through ``options['subproblem']``:
    ``{"subproblem": {"tau": 0.5, "mu_min": 1e-6, "mu_max": 0.1}}``.

    Notes
    -----
    The proximal step satisfies ``A_eq d + c_eq = μ (λ − λ_k)`` rather than
    ``A_eq d = -c_eq``, so the L1-merit descent guarantee behind the Armijo
    search holds only up to ``O(μ ‖λ − λ_k‖)``. The residual-driven ``μ``
    makes that term vanish as the iterates approach a KKT point.

    For the same reason a stationary iterate need not be feasible: with a
    quadratic objective and linear equalities the very first proximal step
    is exactly stationary for the recovered multipliers while
    ``‖c_eq‖ = μ ‖λ − λ_k‖ ≠ 0``. The base minimiser's fatal
    "stationary but infeasible" test is therefore tightened here: it fires
    only once the QP step has also collapsed to zero for
    ``zero_step_patience`` consecutive iterations, which is the proximal
    signature of a genuine infeasible stationary point (``A_eqᵀ c_eq ≈ 0``
    with ``c_eq ≠ 0``, so the multipliers drift along ``null(A_eqᵀ)`` while
    ``x`` stays put).
    """

    preconditioner: PreconditionerStrategy = eqx.field(
        static=True,
        default_factory=lambda: SecantPreconditioner(require_secant=False),
    )

    def _subproblem_solver_type(self) -> type[SubProblemSolver]:
        """Root solver class for ``options['subproblem']`` validation."""
        return ProximalActiveSetQPSolver

    def _configured_qp_solver(
        self, problem: ProblemProtocol[Primal], dtype: jnp.dtype
    ) -> ProximalActiveSetQPSolver:
        """Default proximal solver with ``options['subproblem']`` applied."""
        return cast(
            ProximalActiveSetQPSolver, super()._configured_qp_solver(problem, dtype)
        )

    def _make_qp_solver(
        self, problem: ProblemProtocol[Primal], dtype: jnp.dtype
    ) -> ProximalActiveSetQPSolver:
        """Proximal active-set loop around a projected CG.

        The inner solver is left unpreconditioned here; the per-step
        :attr:`preconditioner` strategy fills it in after options are applied.
        """
        return cast(
            ProximalActiveSetQPSolver,
            ProximalActiveSetQPSolver(
                subproblem_solver=ProjectedCGSubProblemSolver(),
                working_set_policy=self._make_working_set_policy(),
                warm_start=self.qp_warm_start,
            ),
        )

    def _init_solver_state(
        self, problem: ProblemProtocol[Primal], primal: Primal
    ) -> ProximalActiveSetQPSolverState:
        """Cold state: infinite residual (``μ = μ_max``), zero centre."""
        dtype = primal.x.dtype
        solver = self._configured_qp_solver(problem, dtype)
        return cast(
            ProximalActiveSetQPSolverState,
            ProximalActiveSetQPSolverState(
                n_iter=jnp.asarray(0, jnp.int32),
                n_cg_iter=jnp.asarray(0, jnp.int32),
                last_n_iter=jnp.asarray(0, jnp.int32),
                last_n_cg_iter=jnp.asarray(0, jnp.int32),
                success=jnp.asarray(False),
                status=RESULTS.successful,
                qp_result=ACTIVE_SET_QP_RESULTS.working_set_converged,
                active_set=self._empty_active_set(problem),
                dual=self._init_dual(problem),
                final_working_tol=jnp.asarray(self.effective_qp_tol, dtype),
                n_anti_cycling=jnp.asarray(0, jnp.int32),
                last_kkt_feasibility_residual=jnp.asarray(0.0, dtype),
                last_kkt_n_refinements=jnp.asarray(0, jnp.int32),
                last_kkt_reason=KKT_SOLVER_RESULTS.converged,
                kkt_residual=jnp.asarray(jnp.inf, dtype),
                mu=jnp.asarray(solver.mu_max, dtype),
                eq_center=jnp.zeros((problem.meq,), dtype),
            ),
        )

    def _init_subproblem(
        self, problem: ProblemProtocol[Primal]
    ) -> SubproblemContext[Primal, ActiveSetSubProblem, ProximalActiveSetQPSolverState]:
        """Active-set context with the outer KKT residual written into the state."""
        ctx = super()._init_subproblem(problem)
        # The QP Lagrangian is already evaluated at the current multipliers,
        # so the NLP stationarity is read off it without re-evaluating.
        lag_k = ctx.subproblem.lagrangian
        residual = jnp.maximum(
            _inf_norm(lag_k.x_grad), self._feasibility_from_lagrangian(lag_k)
        )
        state = eqx.tree_at(
            lambda s: s.kkt_residual,
            ctx.state,
            residual.astype(ctx.state.kkt_residual.dtype),
        )
        return cast(
            SubproblemContext[
                Primal, ActiveSetSubProblem, ProximalActiveSetQPSolverState
            ],
            eqx.tree_at(lambda c: c.state, ctx, state),
        )

    def _advance_dynamics(
        self,
        ctx: SubproblemContext[
            Primal, ActiveSetSubProblem, ProximalActiveSetQPSolverState
        ],
        result: StepResult[Primal, ProximalActiveSetQPSolverState],
        step_dual: Dual,
    ) -> Self:
        """Run the base dynamics, then re-sync the proximal centre to the committed dual."""
        advanced = super()._advance_dynamics(ctx, result, step_dual)
        dual = cast(Dual, advanced.dual)
        state = cast(ProximalActiveSetQPSolverState, advanced.solver_state)
        state = eqx.tree_at(
            lambda s: s.eq_center,
            state,
            jnp.asarray(dual.eq_multipliers, state.eq_center.dtype),
        )
        return cast(Self, eqx.tree_at(lambda m: m.solver_state, advanced, state))

    def _secant_reset_signals(self) -> SecantResetSignals:
        """Report structural QP failures, but not productive proximal retries.

        A rejected proximal primal step still commits a multiplier-center
        update and reduces the residual-driven proximal parameter. It is
        therefore not evidence that the shared Hessian model needs recovery,
        unlike a structural QP failure.

        Returns
        -------
        SecantResetSignals
            The base raw QP streak with the step channel suppressed.
        """
        signals = super()._secant_reset_signals()
        return eqx.tree_at(
            lambda s: s.step_streak,
            signals,
            jnp.asarray(0, jnp.int32),
        )

    def _infeasible_stationary(
        self,
        metrics: ActiveSetLineSearchTerminationMetrics,
        stationary: Bool[Array, ""],
        feasible: Bool[Array, ""],
    ) -> Bool[Array, ""]:
        """Stationary, infeasible *and* stuck: the proximal QP step has vanished."""
        stuck = self.consecutive_zero_steps >= self.zero_step_patience
        return stationary & ~feasible & metrics.has_min_steps & stuck

    def _postprocess_stats(
        self,
        problem: ProblemProtocol[Primal],
        result: ACTIVE_SET_LINE_SEARCH_RESULTS,
    ) -> dict:
        """Base diagnostics plus the proximal parameter and residual."""
        stats = super()._postprocess_stats(problem, result)
        state = cast(ProximalActiveSetQPSolverState, self.solver_state)
        stats["proximal_mu"] = state.mu
        stats["proximal_kkt_residual"] = state.kkt_residual
        return stats
