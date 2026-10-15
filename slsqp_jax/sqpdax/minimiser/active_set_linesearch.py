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
from ..secant import SecantResetSignals
from ..step_controller import ArmijoLineSearch, StepController, StepResult
from ..subproblem import ActiveSetSubProblem
from ..subproblem.solver import (
    ACTIVE_SET_QP_RESULTS,
    KKT_SOLVER_RESULTS,
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
from .diagnostics import ActiveSetLineSearchDiagnostics
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
        scale is enabled; ``max(|f|, 1)`` when the last QP solve did not
        converge, so that its multipliers cannot inflate the scale. The
        minimiser converges when ``stationarity <= rtol * stationarity_scale``.
    feasibility
        ``∞``-norm of the equality / inequality / bound violations, compared
        against ``atol``.
    kkt_ratio
        Dimensionless ``stationarity / stationarity_scale`` compared to
        ``rtol``.
    has_min_steps
        ``True`` once ``step_count >= min_steps``, gating convergence so a
        cold start cannot report success before doing any work.
    violation_stationarity
        Dimensionless stationarity of the squared violation measure
        ``ψ(x) = ½‖c_E‖² + ½‖max(c_I, 0)‖² + ½‖max(lb - x, 0)‖² + ½‖max(x - ub, 0)‖²``:
        the box-projected gradient ``∇ψ / ‖c⁺‖₂`` divided by the largest
        row norm of the violated constraint Jacobian. Small values certify
        that the iterate is a stationary point of the infeasibility measure,
        so no first-order method can restore feasibility from it.
    """

    stationarity: Scalar
    stationarity_scale: Scalar
    feasibility: Scalar
    kkt_ratio: Scalar
    has_min_steps: Bool[Array, ""]
    violation_stationarity: Scalar


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


class _ShiftedActiveSetSubProblem(ActiveSetSubProblem):
    """Active-set QP whose Hessian is shifted by ``delta I`` (Levenberg).

    Attributes
    ----------
    delta
        Non-negative scalar added to the Hessian block of the KKT system.
    """

    delta: Array

    def __init__(
        self,
        lagrangian: EvaluatedLagrangian[Primal],
        active_set: ActiveSet,
        delta: Array,
    ):
        super().__init__(lagrangian, active_set)
        self.delta = delta

    def with_active_set(self, active_set: ActiveSet) -> Self:
        """Rebuild on a new working set, keeping the shift."""
        return type(self)(self.lagrangian, active_set, self.delta)

    def kkt_mvp_primal(self, step: tuple[Primal, Dual]) -> Primal:
        """Shifted Hessian product ``(H + delta I) d``."""
        hd = super().kkt_mvp_primal(step)
        return cast(Primal, Primal(x=hd.x + self.delta * step[0].x))


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
    diagnostics
        :class:`~slsqp_jax.sqpdax.minimiser.diagnostics.active_set_linesearch.ActiveSetLineSearchDiagnostics`
        carrying the cumulative Armijo / fallback acceptance and LPEC-A
        predictor counters. Reporting only; nothing feeds back into the
        iteration.
    initial_penalty
        Initial (and therefore minimum) L1 merit penalty ``ρ``.
    penalty_multiplier_margin
        Han--Powell multiplier margin used in the monotone update
        ``ρ_next = max(ρ, margin * ‖λ_QP‖_∞)``.
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
        Relative merit-improvement threshold (relative to
        ``max(|merit|, atol)``) and no-improvement window.
    divergence_factor, divergence_patience
        Excess-merit threshold and consecutive count before best-point rollback.
    secant_reset
        Shared soft / diagonal / identity / fatal recovery policy. QP and
        line-search failures advance one global recovery episode.
    restoration
        Fall back to a feasibility-restoration step when the line search
        rejects the QP direction at a primally infeasible iterate (see
        :meth:`_restoration_direction`). The QP can produce a non-descent or
        non-finite direction there: an inconsistent / capped working set
        only linearises part of the violated rows, a failed KKT solve
        returns garbage. The restoration step is a truncated-CGLS
        Gauss--Newton normal step on ``½‖c⁺‖²``, bounded by
        ``restoration_radius * max(1, ‖x‖_∞)``, clipped to the box and
        globalised by an Armijo search on the L2 penalty ``f + ρ ‖c⁺‖₂``
        (see :meth:`_restoration_fallback`). On by default.
    restoration_radius
        Relative ``∞``-norm bound of the restoration step.
    restoration_cg_iter
        CGLS iterations of the normal-step solve (``1`` gives the Cauchy
        step).
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
    # Monotone Han--Powell L1-merit penalty schedule.
    initial_penalty: float = eqx.field(static=True, default=1.0)
    penalty_multiplier_margin: float = eqx.field(static=True, default=1.1)
    merit_penalty: Scalar = eqx.field(default_factory=lambda: jnp.asarray(1.0))
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
    # feasibility restoration fallback
    restoration: bool = eqx.field(static=True, default=True)
    restoration_radius: float = eqx.field(static=True, default=10.0)
    restoration_cg_iter: int = eqx.field(static=True, default=20)
    n_restoration_steps: Array = eqx.field(
        default_factory=lambda: jnp.asarray(0, jnp.int32)
    )
    last_step_restoration: Bool[Array, ""] = eqx.field(
        default_factory=lambda: jnp.asarray(False)
    )
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
    last_step_size: Scalar = eqx.field(default_factory=lambda: jnp.asarray(0.0))
    last_ls_success: Bool[Array, ""] = eqx.field(
        default_factory=lambda: jnp.asarray(False)
    )
    # Reporting-only line-search / LPEC-A counters.
    diagnostics: ActiveSetLineSearchDiagnostics = eqx.field(
        default_factory=ActiveSetLineSearchDiagnostics.zero
    )

    def __check_init__(self) -> None:
        """Validate the common minimiser and Han--Powell configuration."""
        super().__check_init__()
        if self.initial_penalty <= 0:
            raise ValueError(
                f"initial_penalty must be positive; got {self.initial_penalty}"
            )
        if self.penalty_multiplier_margin <= 1:
            raise ValueError(
                "penalty_multiplier_margin must be greater than 1; got "
                f"{self.penalty_multiplier_margin}"
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
        mopts = base.options.get("minimiser", {})
        spec = mopts.get("active_set_predictor")
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
        merit_penalty = jnp.asarray(base.initial_penalty, primal.x.dtype)
        merit = cast(
            NormMerit,
            NormMerit(
                problem,
                barrier=None,
                problem_weight=jnp.asarray(1.0),
                feasibility_weight=merit_penalty,
                norm=1,
            ),
        )
        initial_merit = merit(primal)
        return eqx.tree_at(
            lambda m: (m.merit_penalty, m.best_merit, m.best_iterate, m.best_dual),
            base,
            (merit_penalty, initial_merit, primal, dual),
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
                last_kkt_feasibility_residual=jnp.asarray(0.0, primal.x.dtype),
                last_kkt_n_refinements=jnp.asarray(0, jnp.int32),
                last_kkt_reason=KKT_SOLVER_RESULTS.converged,
            ),
        )

    def _lagrangian_module(
        self, problem: ProblemProtocol[Primal]
    ) -> Lagrangian[Primal, EvaluatedLagrangian[Primal]]:
        """Unevaluated Lagrangian wrapping the model secant (if the model uses one)."""
        return cast(
            Lagrangian[Primal, EvaluatedLagrangian[Primal]],
            Lagrangian(problem, self._model_secant(problem)),
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
        solver = solver.init(logger=self.logger.child("subproblem"))
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
                lambda m: (m.solver_state, m.diagnostics),
                self,
                (state, self.diagnostics.record_prediction(prediction)),
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
        ctx = seeded._init_subproblem(problem)
        step_primal, step_dual, solver_state = seeded._solve_direction(ctx)
        result = seeded._assess_direction(ctx, step_primal, step_dual, solver_state)
        result, restored, penalty = seeded._restoration_fallback(
            ctx, step_dual, solver_state, result
        )
        seeded = cast(
            Self,
            eqx.tree_at(
                lambda m: (
                    m.last_step_restoration,
                    m.n_restoration_steps,
                    m.merit_penalty,
                ),
                seeded,
                (
                    restored,
                    seeded.n_restoration_steps + restored.astype(jnp.int32),
                    penalty,
                ),
            ),
        )
        return seeded._execute_step(ctx, step_dual, result)

    def _restoration_fallback(
        self,
        ctx: SubproblemContext[Primal, ActiveSetSubProblem, ActiveSetStateType],
        step_dual: Dual,
        solver_state: ActiveSetStateType,
        result: StepResult[Primal, ActiveSetStateType],
    ) -> tuple[StepResult[Primal, ActiveSetStateType], Bool[Array, ""], Scalar]:
        """Retry the line search along the restoration direction when needed.

        Parameters
        ----------
        ctx
            Per-step subproblem context.
        step_dual
            Multipliers from the QP solve (sets the merit penalty).
        solver_state
            QP carry after the solve.
        result
            Outcome of the line search along the QP direction.

        Returns
        -------
        result
            ``result`` itself when it was accepted, the iterate is feasible
            or :attr:`restoration` is off; otherwise the outcome of the line
            search along :meth:`_restoration_direction`. The restoration
            search accepts on the L2 penalty ``f + ρ_R ‖c⁺‖₂``: the L1
            violation can be flat along a direction that reduces the
            violation (e.g. when the violated rows sum to a constant), while
            keeping ``f`` in the test stops the step from trading a small
            violation decrease for an exploding objective. ``merit_val`` is
            nevertheless reported in the L1 merit units of this step so the
            progress bookkeeping stays consistent.
        restored
            Whether a restoration step was tried *and* accepted.
        penalty
            Merit penalty to carry. ``ρ_R = max(ρ, 2 ∇fᵀd / Δv)`` where
            ``Δv`` is the decrease of the linearised violation along the
            restoration direction, so that the direction is a descent
            direction of the L2 penalty (Nocedal & Wright eq. 18.36 without
            the curvature term). It replaces the carried penalty only when
            the restoration step is accepted; otherwise the carried value is
            returned unchanged.
        """
        if not self.restoration:
            return result, jnp.asarray(False), self.merit_penalty
        lagrangian = ctx.subproblem.lagrangian
        infeasible = self._feasibility_from_lagrangian(lagrangian) > self.atol
        # An accepted null step (``alpha * ‖d‖ < atol``) leaves an infeasible
        # iterate where it is, so it must trigger restoration like a rejection,
        # unless the violation is stationary (restoration cannot reduce it and
        # the infeasibility certificate must be left to fire).
        reducible = self._violation_stationarity(lagrangian) > jnp.sqrt(self.rtol)
        null_step = (
            result.accepted
            & (result.step_size * result.proposed_step_norm < self.atol)
            & reducible
        )
        attempt = infeasible & (~result.accepted | null_step)
        direction, predicted_decrease = self._restoration_step(lagrangian)
        slope_f = jnp.dot(lagrangian.grad_val, direction)
        rho = self._next_merit_penalty(step_dual, solver_state)
        rho_r = jnp.where(
            predicted_decrease > 0.0,
            jnp.maximum(rho, 2.0 * jnp.maximum(slope_f, 0.0) / predicted_decrease),
            rho,
        )
        rho_r = jnp.where(jnp.isfinite(rho_r), rho_r, rho)
        self.logger.info(
            "QP direction rejected at an infeasible iterate: trying the "
            "restoration step (‖d‖_∞={norm:.3e}, predicted decrease={dv:.3e}, "
            "penalty {rho:.3e} -> {rho_r:.3e})",
            when=attempt,
            norm=_inf_norm(direction),
            dv=predicted_decrease,
            rho=rho,
            rho_r=rho_r,
        )
        l1_merit = replace(
            self._l1_merit(ctx.problem, step_dual, solver_state),
            feasibility_weight=rho_r,
        )
        search = cast(
            StepController[Primal, ActiveSetStateType],
            ArmijoLineSearch(
                merit=replace(l1_merit, norm=2),
                logger=self.logger.child("restoration"),
                max_steps=self.line_search_max_steps,
                c1=self.armijo_c1,
                backtrack=self.armijo_backtrack,
            ),
        )

        def restore() -> StepResult[Primal, ActiveSetStateType]:
            out = search.step(
                cast(Primal, self.iterate),
                cast(Primal, Primal(direction)),
                solver_state,
            )
            return replace(out, merit_val=l1_merit(out.x))

        fallback = jax.lax.cond(attempt, restore, lambda: result)
        restored = attempt & fallback.accepted
        penalty = jnp.where(restored, rho_r, self.merit_penalty)
        return fallback, restored, penalty

    def _restoration_direction(self, lagrangian: EvaluatedLagrangian[Primal]) -> Array:
        """Restoration direction only (see :meth:`_restoration_step`)."""
        return self._restoration_step(lagrangian)[0]

    def _restoration_step(
        self, lagrangian: EvaluatedLagrangian[Primal]
    ) -> tuple[Array, Scalar]:
        """Bounded, box-constrained Gauss--Newton normal step on ``½‖c⁺‖²``.

        With ``c⁺ = (c_E, max(c_I, 0))`` and ``J⁺`` the matching Jacobian
        rows, the step approximately minimises ``‖c⁺ + J⁺ d‖₂`` subject to
        ``lb <= x + d <= ub`` and ``‖d‖_∞ <= restoration_radius *
        max(1, ‖x‖_∞)``. Each of a few projection rounds runs
        :attr:`restoration_cg_iter` CGLS iterations from ``d = 0`` on the
        free variables (the first iterate is the Cauchy step, the residual
        decreases monotonically, and the truncation bounds the step on
        rank-deficient Jacobians without regularising anything), clips the
        result to the box, fixes the variables that hit a bound, and adds the
        inequality rows the clipped step would violate (``c_i + J_i d > atol``)
        as targets for the next round. Variables outside the box are moved
        back onto it in the first round regardless of the radius.

        Parameters
        ----------
        lagrangian
            Lagrangian evaluated at the current iterate.

        Returns
        -------
        direction
            Restoration direction ``d`` (``0`` at a feasible point).
        predicted_decrease
            ``‖c⁺‖₂ - ‖(c + J d)⁺‖₂`` (including the bound violations) for
            the returned direction.
        """
        x = lagrangian.x_ref
        j_eq = lagrangian.eq_fn_jac_val
        j_in = lagrangian.ineq_fn_jac_val
        c_eq = lagrangian.eq_fn_val
        c_raw = lagrangian.ineq_fn_val
        lb = jnp.where(lagrangian.null_lb, -jnp.inf, lagrangian.lb)
        ub = jnp.where(lagrangian.null_ub, jnp.inf, lagrangian.ub)
        tiny = jnp.asarray(jnp.finfo(x.dtype).tiny, x.dtype)
        radius = self.restoration_radius * jnp.maximum(1.0, _inf_norm(x))
        box_lo = jnp.maximum(lb - x, -radius)
        box_hi = jnp.minimum(ub - x, radius)
        box_lo = jnp.where(jnp.isfinite(box_lo), box_lo, -radius)
        box_hi = jnp.where(jnp.isfinite(box_hi), box_hi, radius)
        # Variables outside the box jump back onto it whatever the radius.
        box_hi = jnp.where(lb - x > 0.0, jnp.maximum(box_hi, lb - x), box_hi)
        box_lo = jnp.where(x - ub > 0.0, jnp.minimum(box_lo, ub - x), box_lo)

        def apply(v, rows):
            return (j_eq @ v, jnp.where(rows, j_in @ v, 0.0))

        def apply_t(u, rows):
            u_eq, u_in = u
            return j_eq.T @ u_eq + j_in.T @ jnp.where(rows, u_in, 0.0)

        def sq(u):
            return sum(jnp.sum(leaf**2) for leaf in u)

        def cgls(free, d_fixed, rows):
            """CGLS on the free variables for ``J (d_fixed + d) = -c`` on ``rows``."""
            target = (-c_eq, jnp.where(rows, -c_raw, 0.0))
            residual = jax.tree.map(lambda t, q: t - q, target, apply(d_fixed, rows))
            s0 = jnp.where(free, apply_t(residual, rows), 0.0)
            gamma0 = jnp.sum(s0**2)

            def body(_, carry):
                d, r, p, gamma = carry
                q = apply(p, rows)
                q_sq = sq(q)
                live = (gamma > 0.0) & (q_sq > 0.0)
                alpha = jnp.where(live, gamma / jnp.maximum(q_sq, tiny), 0.0)
                d = d + alpha * p
                r = jax.tree.map(lambda ri, qi: ri - alpha * qi, r, q)
                s = jnp.where(free, apply_t(r, rows), 0.0)
                gamma_new = jnp.sum(s**2)
                beta = jnp.where(live, gamma_new / jnp.maximum(gamma, tiny), 0.0)
                p = jnp.where(live, s + beta * p, p)
                return d, r, p, jnp.where(live, gamma_new, 0.0)

            d, _, _, _ = jax.lax.fori_loop(
                0,
                max(int(self.restoration_cg_iter), 1),
                body,
                (jnp.zeros_like(x), residual, s0, gamma0),
            )
            return d_fixed + d

        # Bound-constrained least squares by projection rounds: variables
        # outside the box (or on a bound they would leave) are fixed at the
        # bound, inequality rows the clipped step would violate join the
        # targets, and the remaining variables are re-solved. A few rounds
        # settle the typical handful of bound-hitting variables / rows.
        d_fixed = jnp.clip(jnp.zeros_like(x), box_lo, box_hi)
        free = (box_lo <= 0.0) & (box_hi >= 0.0)
        rows = c_raw > 0.0
        direction = d_fixed
        for _ in range(4):
            raw = cgls(free, d_fixed, rows)
            direction = jnp.clip(raw, box_lo, box_hi)
            hit = free & (raw != direction)
            free = free & ~hit
            d_fixed = jnp.where(free, 0.0, direction)
            rows = rows | (c_raw + j_in @ direction > self.atol)
        direction = jnp.where(jnp.isfinite(direction), direction, 0.0)

        def violation(dx):
            x_new = x + dx
            lb_v = jnp.where(jnp.isfinite(lb), jnp.maximum(0.0, lb - x_new), 0.0)
            ub_v = jnp.where(jnp.isfinite(ub), jnp.maximum(0.0, x_new - ub), 0.0)
            c_in = jnp.maximum(0.0, c_raw + j_in @ dx)
            return jnp.sqrt(sq((c_eq + j_eq @ dx, c_in, lb_v, ub_v)))

        return direction, violation(jnp.zeros_like(x)) - violation(direction)

    def _retry_subproblem(
        self,
        problem: ProblemProtocol[Primal],
        qp_lag: EvaluatedLagrangian[Primal],
        active_set: ActiveSet,
    ) -> ActiveSetSubProblem:
        """Build the QP subproblem, Levenberg-shifted after a rejection.

        With exact curvature a rejected direction is recomputed identically
        at the same iterate, so after ``j = consecutive_ls_failures >= 1``
        rejections the QP Hessian is shifted by
        ``delta_j = 1e-2 max(1, ||grad f||_inf) 10^(j-1)``. Secant models
        and ``j = 0`` build the plain subproblem.

        Parameters
        ----------
        problem
            NLP being minimised.
        qp_lag
            Lagrangian evaluated at the current ``(iterate, dual)``.
        active_set
            Initial working set.

        Returns
        -------
        ActiveSetSubProblem
            Plain or shifted subproblem.
        """
        if self._model_secant(problem) is not None:
            return cast(ActiveSetSubProblem, ActiveSetSubProblem(qp_lag, active_set))
        j = self.consecutive_ls_failures
        scale = jnp.maximum(1.0, _inf_norm(qp_lag.grad_val))
        exponent = jnp.maximum(j - 1, 0).astype(scale.dtype)
        delta = jnp.where(j >= 1, 1e-2 * scale * 10.0**exponent, 0.0)
        return cast(
            ActiveSetSubProblem,
            _ShiftedActiveSetSubProblem(qp_lag, active_set, delta),
        )

    def _init_subproblem(
        self, problem: ProblemProtocol[Primal]
    ) -> SubproblemContext[Primal, ActiveSetSubProblem, ActiveSetStateType]:
        """Build an active-set QP context at the current iterate.

        The Lagrangian is evaluated at the current ``(iterate, dual)`` so an
        exact-curvature problem contributes ``∇²ₓₓL(x_k, λ_k)`` to the QP
        Hessian; the QP still returns the full multiplier ``λ_{k+1}`` because
        :class:`~slsqp_jax.sqpdax.subproblem.active_set.ActiveSetSubProblem`
        uses the objective gradient as its right-hand side (SQP view). The
        subproblem's own working set is empty; the QP solver builds its
        initial set from the current point and from the carried state when
        ``qp_warm_start`` is set or the LPEC-A predictor has seeded it (see
        :meth:`_seed_predicted_active_set`). The configured solver then
        receives this step's
        :attr:`~slsqp_jax.sqpdax.minimiser.base.CommonMinimiser.preconditioner`
        (options are applied first, so a user-supplied inner solver is
        preconditioned too unless it carries its own preconditioner).
        """
        iterate = cast(Primal, self.iterate)
        dual = cast(Dual, self.dual)
        n = problem.n
        dtype = iterate.x.dtype
        lag_module = self._lagrangian_module(problem)
        qp_lag = lag_module(iterate, dual)

        solver = cast(
            ActiveSetQPSolver[Any, ActiveSetStateType],
            self._precondition(
                self._configured_qp_solver(problem, dtype), problem, qp_lag
            ),
        )
        if self.active_set_predictor.enabled:
            solver = solver.init(warm_start=True)
        return cast(
            SubproblemContext[Primal, ActiveSetSubProblem, ActiveSetStateType],
            SubproblemContext(
                problem=problem,
                lagrangian=lag_module,
                subproblem=self._retry_subproblem(
                    problem, qp_lag, self._empty_active_set(problem)
                ),
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
        """Armijo line search on the monotonically updated L1 merit."""
        return cast(
            StepController[Primal, ActiveSetStateType],
            ArmijoLineSearch(
                merit=self._l1_merit(ctx.problem, step_dual, solver_state),
                logger=self.logger.child("step_controller"),
                max_steps=self.line_search_max_steps,
                c1=self.armijo_c1,
                backtrack=self.armijo_backtrack,
            ),
        )

    def _l1_merit(
        self,
        problem: ProblemProtocol[Primal],
        step_dual: Dual,
        solver_state: ActiveSetStateType,
    ) -> NormMerit:
        """L1 merit ``f + ρ ‖c⁺‖₁`` at this step's Han--Powell penalty ``ρ``."""
        rho = self._next_merit_penalty(step_dual, solver_state)
        return cast(
            NormMerit,
            NormMerit(
                problem,
                barrier=None,
                problem_weight=jnp.asarray(1.0),
                feasibility_weight=rho,
                norm=1,
            ),
        )

    def _next_merit_penalty(
        self,
        step_dual: Dual,
        solver_state: ActiveSetStateType,
    ) -> Scalar:
        """Return the trusted monotone Han--Powell penalty for this QP.

        Equality multipliers enter by magnitude. Inequality and bound
        multipliers are clamped to their dual-feasible non-negative sign for
        this calculation only; the raw QP multipliers remain untouched.
        Failed QPs and non-finite multiplier estimates cannot ratchet the
        carried penalty.
        """
        penalty_multipliers = jnp.concatenate(
            [
                step_dual.eq_multipliers,
                jnp.maximum(step_dual.ineq_multipliers, 0.0),
                jnp.maximum(step_dual.lb_multipliers, 0.0),
                jnp.maximum(step_dual.ub_multipliers, 0.0),
            ]
        )
        finite = jnp.all(jnp.isfinite(penalty_multipliers))
        candidate = self.penalty_multiplier_margin * _inf_norm(penalty_multipliers)
        updated = jnp.maximum(self.merit_penalty, candidate)
        return jnp.where(solver_state.success & finite, updated, self.merit_penalty)

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

    @staticmethod
    def _violation_stationarity(
        lagrangian: EvaluatedLagrangian[Primal],
    ) -> Scalar:
        """Dimensionless box-projected gradient of the squared violation measure.

        With ``c⁺ = (c_E, max(c_I, 0), max(lb - x, 0), max(x - ub, 0))`` the
        measure ``ψ = ½‖c⁺‖²`` has gradient ``∇ψ = J⁺ᵀ c⁺`` where ``J⁺`` stacks
        the constraint Jacobian with ``-I`` / ``+I`` rows for the violated
        bounds. The returned quantity is ``‖P(∇ψ / ‖c⁺‖₂)‖_∞ / s`` where ``P``
        zeroes the components that point out of the box at an active bound
        and ``s`` is the largest ``∞``-norm of the violated rows of ``J⁺``.
        It is ``0`` at a feasible point and lies in ``[0, 1]`` otherwise.
        """
        x = lagrangian.x_ref
        c_eq = lagrangian.eq_fn_val
        c_in = jnp.maximum(0.0, lagrangian.ineq_fn_val)
        lb_v = jnp.where(lagrangian.null_lb, 0.0, jnp.maximum(0.0, lagrangian.lb - x))
        ub_v = jnp.where(lagrangian.null_ub, 0.0, jnp.maximum(0.0, x - lagrangian.ub))
        grad = (
            lagrangian.eq_fn_jac_val.T @ c_eq
            + lagrangian.ineq_fn_jac_val.T @ c_in
            - lb_v
            + ub_v
        )
        norm_c = jnp.sqrt(
            jnp.sum(c_eq**2) + jnp.sum(c_in**2) + jnp.sum(lb_v**2) + jnp.sum(ub_v**2)
        )
        tiny = jnp.asarray(jnp.finfo(x.dtype).tiny, x.dtype)
        unit_grad = grad / jnp.maximum(norm_c, tiny)
        # Project onto the box: a descent direction (-∇ψ) must not leave it.
        at_lb = ~lagrangian.null_lb & (x <= lagrangian.lb) & (unit_grad > 0.0)
        at_ub = ~lagrangian.null_ub & (x >= lagrangian.ub) & (unit_grad < 0.0)
        projected = jnp.where(at_lb | at_ub, 0.0, unit_grad)
        row_norm_eq = jnp.max(jnp.abs(lagrangian.eq_fn_jac_val), axis=1, initial=0.0)
        row_norm_in = jnp.max(jnp.abs(lagrangian.ineq_fn_jac_val), axis=1, initial=0.0)
        scale = jnp.max(
            jnp.concatenate(
                [
                    jnp.where(c_eq != 0.0, row_norm_eq, 0.0),
                    jnp.where(c_in > 0.0, row_norm_in, 0.0),
                    jnp.where((lb_v > 0.0) | (ub_v > 0.0), 1.0, 0.0),
                ]
            ),
            initial=0.0,
        )
        return jnp.where(
            norm_c > 0.0, _inf_norm(projected) / jnp.maximum(scale, tiny), 0.0
        )

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
        merit_penalty = self._next_merit_penalty(step_dual, solver_state)

        # The penalty may have increased since the stored best point was
        # selected. Re-evaluate that point in the current merit units before
        # comparing it with this step or using it for divergence rollback.
        merit = cast(
            NormMerit,
            NormMerit(
                ctx.problem,
                barrier=None,
                problem_weight=jnp.asarray(1.0),
                feasibility_weight=merit_penalty,
                norm=1,
            ),
        )
        best_merit = merit(cast(Primal, self.best_iterate))
        merit_scale = jnp.maximum(jnp.abs(best_merit), 1.0)
        # Progress is relative to the merit's own magnitude (floored at
        # ``atol``): with ``max(|merit|, 1)`` every step on a problem whose
        # optimum is ``f* = 0`` counts as stagnation as soon as the merit
        # drops below ``stagnation_tol``, however much it still decreases.
        improvement_scale = jnp.maximum(jnp.abs(best_merit), self.atol)
        improved = (~jnp.isfinite(best_merit)) | (
            result.merit_val < best_merit - self.stagnation_tol * improvement_scale
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
        # An accepted restoration step reduced the constraint violation even
        # when the L1 merit did not improve: it is progress, not stagnation.
        progress = improved | (self.last_step_restoration & result.accepted)
        steps_without = jnp.where(progress, 0, self.steps_without_improvement + 1)
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
            & ~self.last_step_restoration
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
        # iterate refreshes the set. Only a structural KKT failure counts:
        # ``singular`` / ``breakdown``, or ``stagnation`` — the inner solver's
        # feasibility projection bottomed out above its target (a
        # ``KKT_SOLVER_RESULTS.residual_floor`` from MINRES-QLP), so the step
        # does not satisfy the working set.
        qp_nonfinite = ~jnp.isfinite(result.proposed_step_norm)
        qp_real_failure = (
            (solver_state.qp_result == ACTIVE_SET_QP_RESULTS.kkt_solver_failure)
            & (solver_state.status != RESULTS.max_steps_reached)
            & feasible
        ) | qp_nonfinite
        qp_failures = jnp.where(qp_real_failure, self.consecutive_qp_failures + 1, 0)
        # A rejection whose predicted decrease was below the rounding level
        # of the merit says nothing about the model: the Armijo test compared
        # two merit values that differ by round-off. It is not a line-search
        # failure and clears the streak, exactly as the trust-region
        # minimisers do for their model-stall streak (the reset policy fires
        # on any positive streak, so merely holding the count would still
        # escalate the recovery).
        ls_failures = jnp.where(
            result.accepted | result.rounding_level,
            0,
            jnp.where(feasible, self.consecutive_ls_failures + 1, 0),
        )
        diagnostics = self.diagnostics.record_step(
            accepted=result.accepted, accepted_by_fallback=result.accepted_by_fallback
        )

        self.logger.warning(
            "qp failure: result={qp_result} status={status} nonfinite={nonfinite} "
            "consecutive={streak}",
            when=qp_real_failure,
            qp_result=solver_state.qp_result,
            status=solver_state.status,
            nonfinite=qp_nonfinite,
            streak=qp_failures,
        )
        self.logger.warning(
            "line search rejected the step: feasible={feasible} "
            "consecutive_failures={streak}",
            when=~result.accepted,
            feasible=feasible,
            streak=ls_failures,
        )
        self.logger.warning(
            "merit blow-up ({count} consecutive): rolling back to the best "
            "iterate (merit={best:.6e}, current={current:.6e})",
            when=iterate_blowup,
            count=blowup_count,
            best=new_best_merit,
            current=result.merit_val,
        )

        updated = eqx.tree_at(
            lambda m: (
                m.merit_penalty,
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
                m.last_step_size,
                m.last_ls_success,
                m.diagnostics,
            ),
            self,
            (
                merit_penalty,
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
                result.step_size,
                result.accepted,
                diagnostics,
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

    def _step_log_fields(
        self,
        ctx: SubproblemContext[Primal, ActiveSetSubProblem, ActiveSetStateType],
        result: StepResult[Primal, ActiveSetStateType],
        step_dual: Dual,
        octx: OptimisationContext[Primal, ActiveSetStateType],
        metrics: ActiveSetLineSearchTerminationMetrics,
    ) -> dict[str, Any]:
        """Stationarity / feasibility, penalty, QP outcome and failure streaks."""
        solver_state = cast(ActiveSetStateType, result.solver_state)
        return {
            "kkt": (metrics.stationarity, ".2e"),
            "kkt_ratio": (metrics.kkt_ratio, ".2e"),
            "feas": (metrics.feasibility, ".2e"),
            "penalty": (self.merit_penalty, ".2e"),
            "qp_result": solver_state.qp_result,
            "qp_iters": solver_state.last_n_iter,
            "cg_iters": solver_state.last_n_cg_iter,
            "ls_fallback": self.diagnostics.last_ls_fallback,
            "qp_fail": self.consecutive_qp_failures,
            "ls_fail": self.consecutive_ls_failures,
            "stall": self.steps_without_improvement,
        }

    def _diagnostic_fields(
        self,
        ctx: SubproblemContext[Primal, ActiveSetSubProblem, ActiveSetStateType],
        result: StepResult[Primal, ActiveSetStateType],
        step_dual: Dual,
        octx: OptimisationContext[Primal, ActiveSetStateType],
        metrics: ActiveSetLineSearchTerminationMetrics,
    ) -> dict[str, Any]:
        """Base payload plus penalty, failure streaks and the QP working set."""
        fields = super()._diagnostic_fields(ctx, result, step_dual, octx, metrics)
        solver_state = cast(ActiveSetStateType, result.solver_state)
        fields.update(
            {
                "merit_penalty": self.merit_penalty,
                "consecutive_qp_failures": self.consecutive_qp_failures,
                "consecutive_ls_failures": self.consecutive_ls_failures,
                "consecutive_zero_steps": self.consecutive_zero_steps,
                "steps_without_improvement": self.steps_without_improvement,
                "blowup_count": self.blowup_count,
                "last_ls_fallback": self.diagnostics.last_ls_fallback,
                "diagnostics": self.diagnostics,
                "qp_result": solver_state.qp_result,
                "qp_active_set": solver_state.active_set,
                "qp_dual": solver_state.dual,
            }
        )
        return fields

    def _secant_reset_signals(self) -> SecantResetSignals:
        """Report QP failures as subproblem and line-search failures as step streaks."""
        return cast(
            SecantResetSignals,
            SecantResetSignals(
                subproblem_streak=self.consecutive_qp_failures,
                step_streak=self.consecutive_ls_failures,
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
        # The multipliers of a QP solve that did not converge (working-set
        # budget exhausted, KKT solver failure) are not a KKT estimate: they
        # can be arbitrarily large and would inflate the relative scale
        # through ``|L| = |f + λᵀc|`` (or ``μ_max``) until any gradient
        # passes the test. The scale then falls back to the objective's.
        trusted_dual = cast(ActiveSetStateType, ctx.solver_state).success
        objective_scale = jnp.maximum(jnp.abs(lagrangian.fn_val), 1.0)
        stationarity_scale = jnp.where(
            trusted_dual,
            jnp.where(
                self.use_mu_max,
                jnp.maximum(compute_mu_max(lagrangian), 1.0),
                jnp.maximum(jnp.abs(lagrangian.value), 1.0),
            ),
            objective_scale,
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
                violation_stationarity=self._violation_stationarity(lagrangian),
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
            ``True`` when the iterate is infeasible, a stationary point of
            the violation measure (``violation_stationarity <= sqrt(rtol)``),
            the minimum step count has been reached and either the
            Lagrangian is stationary or the last step (QP direction and
            restoration fallback) was rejected. Lagrangian stationarity
            alone is not a certificate: with an over-determined working set
            the QP multipliers can zero ``∇L`` at any point, so the gradient
            of ``½‖c⁺‖²`` must vanish before infeasibility is declared.
            Subclasses whose QP step does not enforce ``A d = -c`` (proximal
            variants) override this.
        """
        violation_stationary = metrics.violation_stationarity <= jnp.sqrt(self.rtol)
        stuck = stationary | ~self.last_ls_success
        return stuck & ~feasible & violation_stationary & metrics.has_min_steps

    def termination_flags(
        self,
        ctx: OptimisationContext[Primal, ActiveSetStateType],
        metrics: ActiveSetLineSearchTerminationMetrics,
    ) -> TerminationFlags[ACTIVE_SET_LINE_SEARCH_RESULTS]:
        """Converge on ``rtol``-relative stationarity plus ``atol`` feasibility.

        Convergence is decided from the KKT metrics of the current iterate
        alone and outranks the fatal flags accumulated by the last step
        (merit stagnation, secant recovery exhaustion, ...): the outer loop
        evaluates this test before attempting the next step, so a rejected
        step at a solution still terminates successfully. The QP-optimality
        shortcut (QP reports optimal, full step accepted at a feasible
        point) is model-based and is suppressed once the secant recovery is
        exhausted.

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
        recovery_fatal = self.secant_recovery_state.fatal
        # The KKT test is evaluated at the current iterate before the next
        # step is attempted and takes precedence over the failure flags of
        # the last step: a point that passes the test is a solution whether
        # or not the step that led there was rejected or exhausted the secant
        # recovery (``classify_termination`` ranks ``converged`` above
        # ``fatal``). The QP-optimality shortcut trusts the *model* rather
        # than the measured KKT residual, so it keeps the recovery gate: an
        # exhausted secant is exactly the situation in which the model is
        # not to be trusted.
        converged = classical | (qp_kkt & ~recovery_fatal)

        infeasible_stationary = self._infeasible_stationary(
            metrics, stationary, feasible
        )
        fatal = (
            self.merit_stagnation
            | self.iterate_blowup
            | infeasible_stationary
            | recovery_fatal
        )
        fatal_result = ACTIVE_SET_LINE_SEARCH_RESULTS.merit_stagnation
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
        fatal_result = ACTIVE_SET_LINE_SEARCH_RESULTS.where(
            recovery_fatal,
            ACTIVE_SET_LINE_SEARCH_RESULTS.secant_recovery_failure,
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
        return super()._postprocess_stats(problem, result) | {
            "final_objective": lagrangian.fn_val,
            "final_grad_norm": jnp.linalg.norm(lagrangian.grad_val),
            "final_lagrangian_grad_norm": jnp.linalg.norm(lagrangian.x_grad),
            "kkt_scale": metrics.stationarity_scale,
            "kkt_ratio": metrics.kkt_ratio,
            "violation_stationarity": metrics.violation_stationarity,
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
            "kkt_feasibility_residual": solver_state.last_kkt_feasibility_residual,
            "kkt_n_refinements": solver_state.last_kkt_n_refinements,
            "kkt_reason": solver_state.last_kkt_reason,
            **self.diagnostics.stats(),
            "merit_penalty": self.merit_penalty,
            "last_step_size": self.last_step_size,
            "last_ls_success": self.last_ls_success,
            "steps_without_improvement": self.steps_without_improvement,
            "blowup_count": self.blowup_count,
            "consecutive_qp_failures": self.consecutive_qp_failures,
            "consecutive_ls_failures": self.consecutive_ls_failures,
            "merit_stagnation": self.merit_stagnation,
            "diverging": self.iterate_blowup,
            "n_restoration_steps": self.n_restoration_steps,
            "last_step_restoration": self.last_step_restoration,
            "sqpdax_result": result,
        }
