"""Proximal (stabilised-SQP) active-set QP + Armijo line-search outer loop."""

from __future__ import annotations

from typing import Self, cast

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Bool

from ..barrier.update import _inf_norm
from ..dual import Dual
from ..preconditioner import preconditioner_from_secant
from ..primal import Primal
from ..problem import ProblemProtocol
from ..step_controller import StepResult
from ..subproblem import ActiveSetSubProblem
from ..subproblem.solver import (
    ACTIVE_SET_QP_RESULTS,
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

    is computed here from the cached zero-dual evaluation with the current
    multipliers swapped in (no extra problem evaluation) and written into the
    solver state before each solve. The multiplier centre ``λ_k`` lives on the
    solver state, is warm-started from the previous solve, and is re-synced to
    the committed dual after a best-iterate rollback.

    When a secant is active, the projected CG is preconditioned with ``M = B``
    (``M⁻¹ = H`` from the secant); the proximal solver wraps it in the
    Woodbury update ``B + (1/μ) A_eqᵀ A_eq`` so the constraint preconditioner
    matches the stabilised Hessian. With exact curvature and no user-supplied
    preconditioner the CG runs unpreconditioned.

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
        """Proximal active-set loop around a (secant-preconditioned) projected CG."""
        preconditioner = None
        if self.secant is not None:
            preconditioner = preconditioner_from_secant(
                self.secant, problem.n, dtype, inverse_as_forward=False
            )
        return cast(
            ProximalActiveSetQPSolver,
            ProximalActiveSetQPSolver(
                subproblem_solver=ProjectedCGSubProblemSolver(
                    preconditioner=preconditioner
                ),
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
        dual = cast(Dual, self.dual)
        # The QP Lagrangian is evaluated at zero dual; swap the current
        # multipliers in to measure the NLP stationarity without re-evaluating.
        lag_k = eqx.tree_at(lambda lag: lag.dual, ctx.subproblem.lagrangian, dual)
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
