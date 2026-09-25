"""Active-set QP subproblem + Armijo line-search outer loop."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Generic, Self, cast

import equinox as eqx
import jax
import optimistix as optx
from jax import numpy as jnp
from jaxtyping import Array, Bool

from ..active_set import ActiveSet
from ..active_set_prediction import LPECAPredictor
from ..barrier.update import _inf_norm
from ..dual import Dual
from ..lagrangian import EvaluatedLagrangian, Lagrangian
from ..merit import NormMerit
from ..primal import Primal
from ..problem import ProblemProtocol
from ..results import (
    MINIMISER_RESULTS,
    ResultAdapter,
)
from ..step_controller import ArmijoLineSearch, StepController, StepResult
from ..subproblem import ActiveSetSubProblem
from ..subproblem.solver import (
    ACTIVE_SET_QP_RESULTS,
    RESULTS,
    ActiveSetQPSolver,
    ActiveSetQPSolverState,
    ActiveSetStateType,
    ProjectedCGSubProblemSolver,
    SingleExchangeWorkingSetPolicy,
    SubproblemContext,
    SubProblemSolver,
    ThresholdWorkingSetPolicy,
)
from ..types import Scalar
from .base import CommonMinimiser, OptimisationContext
from .termination import TerminationFlags, TerminationMetrics, compute_mu_max

__all__ = [
    "ActiveSetLineSearchMinimiser",
    "ACTIVE_SET_LINE_SEARCH_RESULTS",
    "ActiveSetLineSearchTerminationMetrics",
    "ActiveSetLineSearchResultAdapter",
]


class ACTIVE_SET_LINE_SEARCH_RESULTS(
    MINIMISER_RESULTS  # ty: ignore[subclass-of-final-class]
):
    """Fine-grained outcomes for the active-set line-search minimiser."""

    merit_stagnation = "The merit function did not improve over the patience window."
    line_search_failure = "Consecutive line-search failures exceeded the fatal limit."
    qp_subproblem_failure = "Consecutive QP failures exceeded the fatal limit."
    iterate_blowup = "The merit repeatedly diverged; the best iterate was restored."
    infeasible = "The minimiser stopped at a primally infeasible iterate."
    infeasible_stationary = "The iterate is stationary but remains primally infeasible."


class ActiveSetLineSearchTerminationMetrics(
    TerminationMetrics[ACTIVE_SET_LINE_SEARCH_RESULTS]
):
    """Termination measurements for :class:`ActiveSetLineSearchMinimiser`.

    Convergence is a *relative* stationarity test plus an absolute
    feasibility test, so the scale used to relativise the gradient norm is
    carried alongside it rather than recomputed.

    Attributes
    ----------
    stationarity
        ``‖∇_x L‖_∞`` at the current iterate.
    stationarity_scale
        ``max(|L|, 1)`` by default, or ``max(μ_max, 1)`` when the filterSQP
        scale is enabled. The minimiser converges when
        ``stationarity <= rtol * stationarity_scale``.
    feasibility
        ``∞``-norm of the equality / inequality / bound violations, compared
        against ``atol``.
    kkt_ratio
        Dimensionless ``stationarity / stationarity_scale`` compared to
        ``rtol``.
    has_min_steps
        ``True`` once ``step_count >= min_steps``, gating convergence so a
        cold start cannot report success before doing any work.
    """

    stationarity: Scalar
    stationarity_scale: Scalar
    feasibility: Scalar
    kkt_ratio: Scalar
    has_min_steps: Bool[Array, ""]


class ActiveSetLineSearchResultAdapter(ResultAdapter[ACTIVE_SET_LINE_SEARCH_RESULTS]):
    """Optimistix conversion for active-set line-search outcomes."""

    @property
    def result_type(self) -> type[ACTIVE_SET_LINE_SEARCH_RESULTS]:
        return ACTIVE_SET_LINE_SEARCH_RESULTS

    def to_optimistix(self, result: ACTIVE_SET_LINE_SEARCH_RESULTS) -> optx.RESULTS:
        coarse = optx.RESULTS.nonlinear_divergence
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


class ActiveSetLineSearchMinimiser(
    CommonMinimiser[
        Primal,
        ActiveSetSubProblem,
        ActiveSetStateType,
        ActiveSetLineSearchTerminationMetrics,
        ACTIVE_SET_LINE_SEARCH_RESULTS,
    ],
    Generic[ActiveSetStateType],
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
        ``None`` (default) uses the outer feasibility tolerance ``atol`` so
        the QP declares a constraint active / violated on the same scale at
        which the outer loop declares the iterate feasible.
    qp_max_iter
        Maximum working-set iterations per QP solve.
    qp_warm_start
        Seed each QP with the previous QP's working set and multipliers
        (see :attr:`ActiveSetQPSolver.warm_start`). Off by default: a cold
        start is more robust when the active set changes a lot between
        outer iterations.
    qp_single_exchange
        Use the classical one-at-a-time exchange
        (:class:`~slsqp_jax.sqpdax.subproblem.solver.working_set_policy.SingleExchangeWorkingSetPolicy`)
        instead of the all-at-once threshold refresh. Slower per QP but
        immune to inconsistent working sets. Off by default.
    active_set_predictor
        :class:`~slsqp_jax.sqpdax.active_set_prediction.LPECAPredictor`
        seeding the QP working set from the LPEC-A identification test.
        Disabled by default (``method="expand"``); configure it through
        ``options['minimiser']['active_set_predictor']``, e.g.
        ``{"method": "lpeca", "warmup_steps": 2}``. In ``"lpeca"`` mode the
        working-set policy's EXPAND ramp is forced off.
    n_lpeca_bypassed, n_lpeca_capped, n_lpeca_bounds_prefixed
        Cumulative predictor diagnostics: steps whose prediction was
        discarded (trust gate or warm-up), steps where the rank cap
        truncated it, and bounds seeded into the working set.
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
    use_mu_max
        Use filterSQP's objective/Jacobian/multiplier scale for stationarity.
    zero_step_patience
        Consecutive converged zero QP steps required by guarded QP-KKT success.
    stagnation_tol, stagnation_patience
        Relative merit-improvement threshold and no-improvement window.
    divergence_factor, divergence_patience
        Excess-merit threshold and consecutive count before best-point rollback.
    qp_failure_patience, ls_failure_patience
        Failure streak scales; fatal termination occurs at twice each value.
    """

    # QP inner solve
    qp_tol: float | None = eqx.field(static=True, default=None)
    qp_max_iter: int = eqx.field(static=True, default=20)
    qp_warm_start: bool = eqx.field(static=True, default=False)
    qp_single_exchange: bool = eqx.field(static=True, default=False)
    # LPEC-A working-set prediction (all-static module; see _parse_options).
    active_set_predictor: LPECAPredictor = eqx.field(
        static=True, default_factory=LPECAPredictor
    )
    # L1-merit penalty schedule: rho = max(penalty_floor, penalty_factor * ||lambda||_inf)
    penalty_floor: float = eqx.field(static=True, default=1.0)
    penalty_factor: float = eqx.field(static=True, default=2.0)
    # backtracking line search
    armijo_c1: float = eqx.field(static=True, default=1e-4)
    armijo_backtrack: float = eqx.field(static=True, default=0.5)
    line_search_max_steps: int = eqx.field(static=True, default=20)
    # termination policy
    use_mu_max: bool = eqx.field(static=True, default=False)
    zero_step_patience: int = eqx.field(static=True, default=3)
    stagnation_tol: float = eqx.field(static=True, default=1e-12)
    stagnation_patience: int = eqx.field(static=True, default=10)
    divergence_factor: float = eqx.field(static=True, default=10.0)
    divergence_patience: int = eqx.field(static=True, default=3)
    qp_failure_patience: int = eqx.field(static=True, default=3)
    ls_failure_patience: int = eqx.field(static=True, default=3)
    # termination state
    best_merit: Scalar = eqx.field(default_factory=lambda: jnp.asarray(jnp.inf))
    best_iterate: Primal | None = None
    best_dual: Dual | None = None
    steps_without_improvement: Array = eqx.field(
        default_factory=lambda: jnp.asarray(0, jnp.int32)
    )
    blowup_count: Array = eqx.field(default_factory=lambda: jnp.asarray(0, jnp.int32))
    consecutive_zero_steps: Array = eqx.field(
        default_factory=lambda: jnp.asarray(0, jnp.int32)
    )
    consecutive_qp_failures: Array = eqx.field(
        default_factory=lambda: jnp.asarray(0, jnp.int32)
    )
    consecutive_ls_failures: Array = eqx.field(
        default_factory=lambda: jnp.asarray(0, jnp.int32)
    )
    qp_optimal: Bool[Array, ""] = eqx.field(default_factory=lambda: jnp.asarray(False))
    merit_stagnation: Bool[Array, ""] = eqx.field(
        default_factory=lambda: jnp.asarray(False)
    )
    iterate_blowup: Bool[Array, ""] = eqx.field(
        default_factory=lambda: jnp.asarray(False)
    )
    qp_fatal: Bool[Array, ""] = eqx.field(default_factory=lambda: jnp.asarray(False))
    ls_fatal: Bool[Array, ""] = eqx.field(default_factory=lambda: jnp.asarray(False))
    last_step_size: Scalar = eqx.field(default_factory=lambda: jnp.asarray(0.0))
    last_ls_success: Bool[Array, ""] = eqx.field(
        default_factory=lambda: jnp.asarray(False)
    )
    # LPEC-A diagnostics
    n_lpeca_bypassed: Array = eqx.field(
        default_factory=lambda: jnp.asarray(0, jnp.int32)
    )
    n_lpeca_capped: Array = eqx.field(default_factory=lambda: jnp.asarray(0, jnp.int32))
    n_lpeca_bounds_prefixed: Array = eqx.field(
        default_factory=lambda: jnp.asarray(0, jnp.int32)
    )

    def _parse_options(self, options: dict | None) -> Self:
        """Freeze options and configure ``active_set_predictor`` from a nested mapping.

        ``options['minimiser']['active_set_predictor']`` may be either an
        :class:`~slsqp_jax.sqpdax.active_set_prediction.LPECAPredictor`
        (installed as is) or a mapping of its fields, applied on top of the
        current predictor through
        :meth:`~slsqp_jax.sqpdax.types.InitializableModule.init`.
        """
        base = super()._parse_options(options)
        spec = base.options.get("minimiser", {}).get("active_set_predictor")
        if isinstance(spec, Mapping):
            base = replace(
                base, active_set_predictor=self.active_set_predictor.init(**spec)
            )
        return base

    @property
    def result_adapter(self) -> ActiveSetLineSearchResultAdapter:
        """Native active-set result policy."""
        return cast(
            ActiveSetLineSearchResultAdapter, ActiveSetLineSearchResultAdapter()
        )

    def _close_init(
        self,
        primal: Primal,
        dual: Dual,
        solver_state: ActiveSetStateType,
        problem: ProblemProtocol[Primal],
    ) -> Self:
        """Seed secant and best-iterate termination state."""
        base = super()._close_init(primal, dual, solver_state, problem)
        merit = cast(
            NormMerit,
            NormMerit(
                problem,
                barrier=None,
                problem_weight=jnp.asarray(1.0),
                feasibility_weight=jnp.asarray(base.penalty_floor),
                norm=1,
            ),
        )
        initial_merit = merit(primal)
        return eqx.tree_at(
            lambda m: (m.best_merit, m.best_iterate, m.best_dual),
            base,
            (initial_merit, primal, dual),
            is_leaf=lambda z: z is None,
        )

    def _subproblem_solver_type(self) -> type[SubProblemSolver]:
        """Root solver class for ``options['subproblem']`` validation."""
        return ActiveSetQPSolver

    @property
    def effective_qp_tol(self) -> float:
        """Working-set tolerance handed to the QP solver (``qp_tol`` or ``atol``)."""
        return self.atol if self.qp_tol is None else self.qp_tol

    def _empty_active_set(self, problem: ProblemProtocol[Primal]) -> ActiveSet:
        """All-inactive working set sized for ``problem``."""
        return cast(
            ActiveSet,
            ActiveSet(
                meq=problem.meq,
                active_inequalities=jnp.zeros((problem.mineq,), bool),
                active_lb=jnp.zeros((problem.n,), bool),
                active_ub=jnp.zeros((problem.n,), bool),
            ),
        )

    def _init_solver_state(
        self, problem: ProblemProtocol[Primal], primal: Primal
    ) -> ActiveSetStateType:
        """Cold :class:`ActiveSetQPSolverState` for the first outer step.

        Subclasses binding a richer ``ActiveSetStateType`` must override this
        to build their own carry.
        """
        return cast(
            ActiveSetStateType,
            ActiveSetQPSolverState(
                n_iter=jnp.asarray(0, jnp.int32),
                n_cg_iter=jnp.asarray(0, jnp.int32),
                last_n_iter=jnp.asarray(0, jnp.int32),
                last_n_cg_iter=jnp.asarray(0, jnp.int32),
                success=jnp.asarray(False),
                status=RESULTS.successful,
                qp_result=ACTIVE_SET_QP_RESULTS.working_set_converged,
                active_set=self._empty_active_set(problem),
                dual=self._init_dual(problem),
                final_working_tol=jnp.asarray(self.effective_qp_tol, primal.x.dtype),
                n_anti_cycling=jnp.asarray(0, jnp.int32),
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

    def _make_qp_solver(
        self, problem: ProblemProtocol[Primal], dtype: jnp.dtype
    ) -> ActiveSetQPSolver[Any, ActiveSetStateType]:
        """Construct the default QP solver before ``options['subproblem']`` is applied.

        Parameters
        ----------
        problem
            NLP being minimised (for sizes).
        dtype
            Floating dtype of the iterate.

        Returns
        -------
        ActiveSetQPSolver
            Active-set loop around an unpreconditioned projected CG. Its
            state type matches ``ActiveSetStateType``; subclasses binding a
            richer state must return a solver that consumes / produces it.
        """
        return cast(
            ActiveSetQPSolver[Any, ActiveSetStateType],
            ActiveSetQPSolver(
                subproblem_solver=ProjectedCGSubProblemSolver(),
                working_set_policy=self._make_working_set_policy(),
                warm_start=self.qp_warm_start,
            ),
        )

    def _make_working_set_policy(self) -> ThresholdWorkingSetPolicy:
        """Default working-set policy carrying ``qp_tol`` / ``qp_max_iter``.

        :class:`~slsqp_jax.sqpdax.subproblem.solver.working_set_policy.SingleExchangeWorkingSetPolicy`
        when ``qp_single_exchange`` is set, otherwise the all-at-once
        :class:`~slsqp_jax.sqpdax.subproblem.solver.working_set_policy.ThresholdWorkingSetPolicy`.
        Further knobs (EXPAND ramp, drop floor, anti-cycling) are applied on
        top through ``options['subproblem']['working_set_policy']``.
        """
        policy_cls = (
            SingleExchangeWorkingSetPolicy
            if self.qp_single_exchange
            else ThresholdWorkingSetPolicy
        )
        return cast(
            ThresholdWorkingSetPolicy,
            policy_cls(tol=self.effective_qp_tol, max_iter=self.qp_max_iter),
        )

    def _configured_qp_solver(
        self, problem: ProblemProtocol[Primal], dtype: jnp.dtype
    ) -> ActiveSetQPSolver[Any, ActiveSetStateType]:
        """Default QP solver with ``options['subproblem']`` applied.

        In ``"lpeca"`` prediction mode the EXPAND ramp of a
        :class:`~slsqp_jax.sqpdax.subproblem.solver.working_set_policy.ThresholdWorkingSetPolicy`
        is forced off afterwards (the predicted set replaces the tolerance
        ramp as the anti-zigzag device), regardless of user options.
        """
        solver = self._make_qp_solver(problem, dtype)
        sub_opts = dict(self.options.get("subproblem", {}))
        if sub_opts:
            solver = solver.init(**sub_opts)
        policy = solver.working_set_policy
        if self.active_set_predictor.disables_expand and isinstance(
            policy, ThresholdWorkingSetPolicy
        ):
            solver = eqx.tree_at(
                lambda s: s.working_set_policy,
                solver,
                replace(policy, expand_factor=0.0),
            )
        return solver

    def _seed_predicted_active_set(self, problem: ProblemProtocol[Primal]) -> Self:
        """Write the LPEC-A prediction into the carried QP state and count it.

        No-op when the predictor is disabled. Otherwise the Lagrangian is
        evaluated at the current ``(iterate, dual)``, the predicted set is
        OR-ed with the carried working set when ``qp_warm_start`` is set
        (otherwise it replaces it and the carried multipliers are re-synced
        to ``dual``, so the first KKT solve is warm-started exactly as a cold
        solve would be), and the ``n_lpeca_*`` counters are advanced.
        :meth:`_init_subproblem` then forces the solver's ``warm_start`` on so
        the seed is consumed.

        Parameters
        ----------
        problem
            NLP being minimised.

        Returns
        -------
        Self
            Minimiser with ``solver_state`` seeded and the counters updated.
        """
        predictor = self.active_set_predictor
        if not predictor.enabled:
            return self
        dual = cast(Dual, self.dual)
        state = cast(ActiveSetStateType, self.solver_state)
        prediction = predictor.predict(
            self._evaluated_lagrangian(problem), self.step_count
        )
        seed = prediction.active_set
        seed_dual = state.dual
        if self.qp_warm_start:
            seed = jax.tree.map(jnp.logical_or, seed, state.active_set)
        else:
            seed_dual = dual
        state = eqx.tree_at(lambda s: (s.active_set, s.dual), state, (seed, seed_dual))
        return cast(
            Self,
            eqx.tree_at(
                lambda m: (
                    m.solver_state,
                    m.n_lpeca_bypassed,
                    m.n_lpeca_capped,
                    m.n_lpeca_bounds_prefixed,
                ),
                self,
                (
                    state,
                    self.n_lpeca_bypassed + (~prediction.valid).astype(jnp.int32),
                    self.n_lpeca_capped + prediction.capped.astype(jnp.int32),
                    self.n_lpeca_bounds_prefixed + prediction.n_bounds_prefixed,
                ),
            ),
        )

    def step(self, problem: ProblemProtocol[Primal]) -> Self:
        """Seed the QP carry from the LPEC-A prediction, then run the shared step.

        Parameters
        ----------
        problem
            NLP being minimised.

        Returns
        -------
        Self
            Updated minimiser after the controlled step.
        """
        seeded = self._seed_predicted_active_set(problem)
        return CommonMinimiser.step(seeded, problem)

    def _init_subproblem(
        self, problem: ProblemProtocol[Primal]
    ) -> SubproblemContext[Primal, ActiveSetSubProblem, ActiveSetStateType]:
        """Build an active-set QP context at the current iterate.

        The Lagrangian is evaluated at a *zero* dual so the QP recovers the
        full multiplier ``λ_{k+1}``. The subproblem's own working set is
        empty; the QP solver builds its initial set from the current point
        and from the carried state when ``qp_warm_start`` is set or the
        LPEC-A predictor has seeded it (see :meth:`_seed_predicted_active_set`).
        """
        iterate = cast(Primal, self.iterate)
        dual = cast(Dual, self.dual)
        n = problem.n
        dtype = iterate.x.dtype
        lag_module = self._lagrangian_module(problem)
        # Zero-dual eval => QP returns the *full* multiplier lambda_{k+1}.
        qp_lag = lag_module(iterate, self._init_dual(problem))

        solver = self._configured_qp_solver(problem, dtype)
        if self.active_set_predictor.enabled:
            solver = solver.init(warm_start=True)
        return cast(
            SubproblemContext[Primal, ActiveSetSubProblem, ActiveSetStateType],
            SubproblemContext(
                problem=problem,
                lagrangian=lag_module,
                subproblem=ActiveSetSubProblem(qp_lag, self._empty_active_set(problem)),
                solver=solver,
                warm=(Primal(jnp.zeros((n,), dtype)), dual),
                state=cast(ActiveSetStateType, self.solver_state),
            ),
        )

    def _step_controller(
        self,
        ctx: SubproblemContext[Primal, ActiveSetSubProblem, ActiveSetStateType],
        step_dual: Dual,
        solver_state: ActiveSetStateType,
    ) -> StepController[Primal, ActiveSetStateType]:
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
            StepController[Primal, ActiveSetStateType],
            ArmijoLineSearch(
                merit=merit,
                max_steps=self.line_search_max_steps,
                c1=self.armijo_c1,
                backtrack=self.armijo_backtrack,
            ),
        )

    def _feasibility_error(
        self, ctx: OptimisationContext[Primal, ActiveSetStateType]
    ) -> Scalar:
        """``∞``-norm of equality / inequality / bound violations."""
        return self._feasibility_from_lagrangian(ctx.lagrangian)

    @staticmethod
    def _feasibility_from_lagrangian(
        lagrangian: EvaluatedLagrangian[Primal],
    ) -> Scalar:
        """Compute primal infeasibility from an evaluated Lagrangian."""
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

    def _advance_dynamics(
        self,
        ctx: SubproblemContext[Primal, ActiveSetSubProblem, ActiveSetStateType],
        result: StepResult[Primal, ActiveSetStateType],
        step_dual: Dual,
    ) -> Self:
        """Update progress/failure counters and restore the best point on blow-up."""
        lagrangian = ctx.lagrangian(result.x, step_dual)
        feasible = self._feasibility_from_lagrangian(lagrangian) <= self.atol
        solver_state = cast(ActiveSetStateType, result.solver_state)

        best_merit = self.best_merit
        merit_scale = jnp.maximum(jnp.abs(best_merit), 1.0)
        improved = (~jnp.isfinite(best_merit)) | (
            result.merit_val < best_merit - self.stagnation_tol * merit_scale
        )
        new_best_merit = jnp.where(improved, result.merit_val, best_merit)
        new_best_iterate = jax.tree.map(
            lambda new, old: jnp.where(improved, new, old),
            result.x,
            cast(Primal, self.best_iterate),
        )
        new_best_dual = jax.tree.map(
            lambda new, old: jnp.where(improved, new, old),
            step_dual,
            cast(Dual, self.best_dual),
        )
        steps_without = jnp.where(improved, 0, self.steps_without_improvement + 1)
        merit_stagnation = (self.step_count >= self.stagnation_patience) & (
            steps_without >= self.stagnation_patience
        )

        blowup_now = (
            result.merit_val - best_merit > self.divergence_factor * merit_scale
        ) | ~jnp.isfinite(result.merit_val)
        blowup_count = jnp.where(blowup_now, self.blowup_count + 1, 0)
        iterate_blowup = blowup_count >= self.divergence_patience

        zero_step = (
            solver_state.success
            & result.accepted
            & (result.step_size * result.proposed_step_norm < self.atol)
        )
        zero_steps = jnp.where(zero_step, self.consecutive_zero_steps + 1, 0)
        qp_optimal = zero_steps >= self.zero_step_patience

        # A non-finite QP direction is rejected by the line search without
        # moving; it can never become a useful step, so count it as a real QP
        # failure regardless of feasibility (the solver itself reports it as
        # ``max_iter_reached`` because a NaN residual never converges). Budget
        # exhaustion — of the working-set loop or of the inner CG — is *not*
        # a failure: the partial step is still usable and the next outer
        # iterate refreshes the set. Only a structural KKT failure (singular /
        # breakdown) counts.
        qp_nonfinite = ~jnp.isfinite(result.proposed_step_norm)
        qp_real_failure = (
            (solver_state.qp_result == ACTIVE_SET_QP_RESULTS.kkt_solver_failure)
            & (solver_state.status != RESULTS.max_steps_reached)
            & feasible
        ) | qp_nonfinite
        qp_failures = jnp.where(qp_real_failure, self.consecutive_qp_failures + 1, 0)
        ls_failures = jnp.where(
            result.accepted,
            0,
            jnp.where(feasible, self.consecutive_ls_failures + 1, 0),
        )
        qp_fatal = qp_failures >= 2 * self.qp_failure_patience
        ls_fatal = ls_failures >= 2 * self.ls_failure_patience

        updated = eqx.tree_at(
            lambda m: (
                m.best_merit,
                m.best_iterate,
                m.best_dual,
                m.steps_without_improvement,
                m.merit_stagnation,
                m.blowup_count,
                m.iterate_blowup,
                m.consecutive_zero_steps,
                m.qp_optimal,
                m.consecutive_qp_failures,
                m.consecutive_ls_failures,
                m.qp_fatal,
                m.ls_fatal,
                m.last_step_size,
                m.last_ls_success,
            ),
            self,
            (
                new_best_merit,
                new_best_iterate,
                new_best_dual,
                steps_without,
                merit_stagnation,
                blowup_count,
                iterate_blowup,
                zero_steps,
                qp_optimal,
                qp_failures,
                ls_failures,
                qp_fatal,
                ls_fatal,
                result.step_size,
                result.accepted,
            ),
            is_leaf=lambda z: z is None,
        )
        rolled_dual = cast(
            Dual,
            jax.tree.map(
                lambda best, current: jnp.where(iterate_blowup, best, current),
                new_best_dual,
                cast(Dual, updated.dual),
            ),
        )
        # After a rollback the carried QP working set / multipliers describe
        # the abandoned iterate: drop the set (the next solve falls back to a
        # cold start) and re-sync the multipliers to the restored dual.
        rolled_state = cast(ActiveSetStateType, updated.solver_state)
        rolled_state = eqx.tree_at(
            lambda s: (s.active_set, s.dual),
            rolled_state,
            (
                jax.tree.map(
                    lambda mask: jnp.where(iterate_blowup, False, mask),
                    rolled_state.active_set,
                ),
                jax.tree.map(
                    lambda best, current: jnp.where(iterate_blowup, best, current),
                    rolled_dual,
                    rolled_state.dual,
                ),
            ),
        )
        return cast(
            Self,
            eqx.tree_at(
                lambda m: (m.iterate, m.dual, m.solver_state),
                updated,
                (
                    jax.tree.map(
                        lambda best, current: jnp.where(iterate_blowup, best, current),
                        new_best_iterate,
                        cast(Primal, updated.iterate),
                    ),
                    rolled_dual,
                    rolled_state,
                ),
            ),
        )

    def termination_metrics(
        self, ctx: OptimisationContext[Primal, ActiveSetStateType]
    ) -> ActiveSetLineSearchTerminationMetrics:
        """Measure relative stationarity and absolute feasibility.

        Parameters
        ----------
        ctx
            Termination context at the current iterate.

        Returns
        -------
        ActiveSetLineSearchTerminationMetrics
            Stationarity, its scale, feasibility, and the shared non-finite /
            subproblem state.
        """
        lagrangian = ctx.lagrangian

        tracked = (ctx.lagrangian.value, ctx.lagrangian.x_grad, self.iterate, self.dual)
        nonfinite = jnp.logical_not(
            jnp.all(
                jnp.stack(
                    [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(tracked)]
                )
            )
        )

        stationarity = _inf_norm(lagrangian.x_grad)
        stationarity_scale = jnp.where(
            self.use_mu_max,
            jnp.maximum(compute_mu_max(lagrangian), 1.0),
            jnp.maximum(jnp.abs(lagrangian.value), 1.0),
        )
        return cast(
            ActiveSetLineSearchTerminationMetrics,
            ActiveSetLineSearchTerminationMetrics(
                stationarity=stationarity,
                stationarity_scale=stationarity_scale,
                feasibility=self._feasibility_error(ctx),
                kkt_ratio=stationarity / stationarity_scale,
                nonfinite=nonfinite,
                fatal_result=ACTIVE_SET_LINE_SEARCH_RESULTS.qp_subproblem_failure,
                has_min_steps=self.step_count >= self.min_steps,
            ),
        )

    def _infeasible_stationary(
        self,
        metrics: ActiveSetLineSearchTerminationMetrics,
        stationary: Bool[Array, ""],
        feasible: Bool[Array, ""],
    ) -> Bool[Array, ""]:
        """Detect a stationary but primally infeasible iterate (fatal).

        Parameters
        ----------
        metrics
            Output of :meth:`termination_metrics`.
        stationary
            Relative stationarity test outcome.
        feasible
            Absolute feasibility test outcome.

        Returns
        -------
        Bool[Array, ""]
            ``True`` when the iterate is stationary, infeasible and the
            minimum step count has been reached. Subclasses whose QP step
            does not enforce ``A d = -c`` (proximal variants) override this.
        """
        return stationary & ~feasible & metrics.has_min_steps

    def termination_flags(
        self,
        ctx: OptimisationContext[Primal, ActiveSetStateType],
        metrics: ActiveSetLineSearchTerminationMetrics,
    ) -> TerminationFlags[ACTIVE_SET_LINE_SEARCH_RESULTS]:
        """Converge on ``rtol``-relative stationarity plus ``atol`` feasibility.

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
        stationary = metrics.stationarity <= self.rtol * metrics.stationarity_scale
        feasible = metrics.feasibility <= self.atol
        classical = stationary & feasible & metrics.has_min_steps
        qp_kkt = (
            self.qp_optimal
            & feasible
            & self.last_ls_success
            & (self.last_step_size >= 1.0 - 1e-6)
            & metrics.has_min_steps
        )
        converged = classical | qp_kkt

        infeasible_stationary = self._infeasible_stationary(
            metrics, stationary, feasible
        )
        fatal = (
            self.merit_stagnation
            | self.ls_fatal
            | self.qp_fatal
            | self.iterate_blowup
            | infeasible_stationary
        )
        fatal_result = ACTIVE_SET_LINE_SEARCH_RESULTS.merit_stagnation
        fatal_result = ACTIVE_SET_LINE_SEARCH_RESULTS.where(
            self.qp_fatal,
            ACTIVE_SET_LINE_SEARCH_RESULTS.qp_subproblem_failure,
            fatal_result,
        )
        fatal_result = ACTIVE_SET_LINE_SEARCH_RESULTS.where(
            self.ls_fatal,
            ACTIVE_SET_LINE_SEARCH_RESULTS.line_search_failure,
            fatal_result,
        )
        fatal_result = ACTIVE_SET_LINE_SEARCH_RESULTS.where(
            self.iterate_blowup,
            ACTIVE_SET_LINE_SEARCH_RESULTS.iterate_blowup,
            fatal_result,
        )
        fatal_result = ACTIVE_SET_LINE_SEARCH_RESULTS.where(
            fatal & ~feasible,
            ACTIVE_SET_LINE_SEARCH_RESULTS.infeasible,
            fatal_result,
        )
        fatal_result = ACTIVE_SET_LINE_SEARCH_RESULTS.where(
            infeasible_stationary,
            ACTIVE_SET_LINE_SEARCH_RESULTS.infeasible_stationary,
            fatal_result,
        )

        return cast(
            TerminationFlags,
            TerminationFlags(
                converged=converged,
                nonfinite=metrics.nonfinite,
                fatal=fatal,
                fatal_result=fatal_result,
            ),
        )

    def _postprocess_stats(
        self,
        problem: ProblemProtocol[Primal],
        result: ACTIVE_SET_LINE_SEARCH_RESULTS,
    ) -> dict:
        """Return active-set termination and final-KKT diagnostics."""
        ctx = self._optimisation_context(problem)
        metrics = self.termination_metrics(ctx)
        lagrangian = ctx.lagrangian
        dual = cast(Dual, self.dual)
        solver_state = cast(ActiveSetStateType, self.solver_state)
        return {
            "num_steps": self.step_count,
            "final_objective": lagrangian.fn_val,
            "final_grad_norm": jnp.linalg.norm(lagrangian.grad_val),
            "final_lagrangian_grad_norm": jnp.linalg.norm(lagrangian.x_grad),
            "kkt_scale": metrics.stationarity_scale,
            "kkt_ratio": metrics.kkt_ratio,
            "multipliers_eq": dual.eq_multipliers,
            "multipliers_ineq": dual.ineq_multipliers,
            "multipliers_lb": dual.lb_multipliers,
            "multipliers_ub": dual.ub_multipliers,
            "qp_iterations": solver_state.last_n_iter,
            "qp_cg_iterations": solver_state.last_n_cg_iter,
            "total_qp_iterations": solver_state.n_iter,
            "total_qp_cg_iterations": solver_state.n_cg_iter,
            "last_qp_converged": solver_state.success,
            "qp_result": solver_state.qp_result,
            "qp_final_working_tol": solver_state.final_working_tol,
            "n_qp_anti_cycling": solver_state.n_anti_cycling,
            "n_lpeca_bypassed": self.n_lpeca_bypassed,
            "n_lpeca_capped": self.n_lpeca_capped,
            "n_lpeca_bounds_prefixed": self.n_lpeca_bounds_prefixed,
            "last_step_size": self.last_step_size,
            "steps_without_improvement": self.steps_without_improvement,
            "blowup_count": self.blowup_count,
            "consecutive_qp_failures": self.consecutive_qp_failures,
            "consecutive_ls_failures": self.consecutive_ls_failures,
            "merit_stagnation": self.merit_stagnation,
            "qp_fatal": self.qp_fatal,
            "ls_fatal": self.ls_fatal,
            "diverging": self.iterate_blowup,
            "sqpdax_result": result,
        }
