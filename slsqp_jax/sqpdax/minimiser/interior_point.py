"""Shared scaffolding for the interior-point outer loops.

Both the trust-region (Nocedal & Wright §19.5) and the trust-funnel
(Curtis, Gould, Robinson & Toint 2017) interior-point minimisers work on an
:class:`~slsqp_jax.sqpdax.primal.InteriorPointPrimal` with a dynamic
:class:`~slsqp_jax.sqpdax.barrier.base.Barrier` whose weight is ``μ``. The
slack seeding, positive initial multipliers, barrier-augmented Lagrangian
and the per-step subproblem-context assembly are identical, so they live on
:class:`InteriorPointMinimiser`; the concrete algorithms only choose the
subproblem, the solver, the step controller and the termination test.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import replace
from typing import Generic, Self, cast

import equinox as eqx
import jax
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
from ..primal import InteriorPointPrimal, Slack
from ..problem import ProblemProtocol
from ..results import ResultType
from ..subproblem.base import SubProblemType
from ..subproblem.solver import SubproblemContext, SubProblemSolver
from ..subproblem.solver.base import SubProblemSolverStateType
from ..types import Vector_n
from .base import CommonMinimiser
from .termination import TerminationMetricsType

__all__ = ["InteriorPointMinimiser"]


class InteriorPointMinimiser(
    CommonMinimiser[
        InteriorPointPrimal,
        SubProblemType,
        SubProblemSolverStateType,
        TerminationMetricsType,
        ResultType,
    ],
    Generic[
        SubProblemType, SubProblemSolverStateType, TerminationMetricsType, ResultType
    ],
):
    """Common state and hooks of the interior-point outer loops.

    Inequality / bound slacks are virtual variables absent from the user's
    ``x0``, so ``init`` builds the
    :class:`~slsqp_jax.sqpdax.primal.InteriorPointPrimal` with
    strictly-interior default slacks and positive multipliers, and seeds a
    :class:`~slsqp_jax.sqpdax.barrier.log.LogBarrier` at :attr:`initial_mu`.
    Every step evaluates the barrier-augmented Lagrangian at the current
    ``(x, s)`` and hands it to :meth:`_make_subproblem`, the only part of the
    subproblem-context assembly that differs between algorithms.

    Attributes
    ----------
    initial_mu
        Initial barrier weight ``μ``.
    initial_slack
        Floor used when seeding inequality / bound slacks.
    primal_dual
        If ``True``, use the primal-dual slack-slack KKT block.
    barrier_update
        Policy that reduces ``μ``. Consulted after every outer step, accepted
        or not: the test it applies is a property of the iterate, not of the
        step, and on a rejected step the iterate is unchanged.
    barrier_updated
        Dynamic flag recording whether the last :attr:`barrier_update` call
        judged the barrier subproblem solved; read by the termination test
        as the inner stopping criterion.
    rtol
        Unused; must be left at its ``NaN`` default. Present only to reject
        the inherited relative-tolerance knob, which the absolute KKT-error
        tests of the interior-point loops do not implement.
    barrier
        Dynamic barrier whose ``weight`` *is* ``μ`` (seeded in
        :meth:`_init_dynamics`).
    """

    initial_mu: float = eqx.field(static=True, default=1.0)
    initial_slack: float = eqx.field(static=True, default=1.0)
    primal_dual: bool = eqx.field(static=True, default=True)
    barrier_update: BarrierUpdate = eqx.field(default_factory=MonotoneBarrierUpdate)
    barrier_updated: Bool[Array, ""] = eqx.field(
        default_factory=lambda: jnp.asarray(False)
    )
    rtol: float = eqx.field(static=True, default=jnp.nan)
    # The barrier is dynamic state: its ``weight`` *is* mu, updated each step.
    barrier: Barrier | None = None

    def __post_init__(self) -> None:
        # ``math.isnan`` rather than ``jnp.isnan``: the module is rebuilt inside
        # traced code (e.g. the optimistix adapter), where a ``jnp`` test on a
        # static float would become a tracer.
        if not math.isnan(self.rtol):
            raise ValueError(
                f"{type(self).__name__} does not provide a relative tolerance "
                "for convergence. Use the absolute tolerance instead."
            )

    # ================================ init =================================

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
        """Seed the secant (via super) and the barrier."""
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
        """Strictly-interior default slacks for inequalities and finite bounds.

        The seeds satisfy ``c(x₀, s₀) ≥ 0`` componentwise (``s ≥ −h(x₀)``,
        ``s_lb ≥ x₀ − lb``, ``s_ub ≥ ub − x₀``), which is the funnel
        invariant (2.1) of Curtis et al. and harmless for the trust-region
        loop.
        """
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

    # ================================ step =================================

    def _lagrangian_module(
        self, problem: ProblemProtocol[InteriorPointPrimal]
    ) -> Lagrangian[InteriorPointPrimal, EvaluatedLagrangian[InteriorPointPrimal]]:
        """Barrier-augmented Lagrangian at the current secant / barrier."""
        return cast(
            Lagrangian[InteriorPointPrimal, EvaluatedLagrangian[InteriorPointPrimal]],
            InteriorPointLagrangian(
                problem,
                self._model_secant(problem),
                cast(Barrier, self.barrier),
                primal_dual=self.primal_dual,
            ),
        )

    @abstractmethod
    def _make_subproblem_solver(
        self, problem: ProblemProtocol[InteriorPointPrimal]
    ) -> SubProblemSolver:
        """Construct the default subproblem solver before ``options['subproblem']``.

        Parameters
        ----------
        problem
            NLP being minimised.

        Returns
        -------
        SubProblemSolver
            Default solver whose state type matches the minimiser's.
        """
        ...

    @abstractmethod
    def _make_subproblem(
        self, lagrangian: InteriorPointEvaluatedLagrangian
    ) -> SubProblemType:
        """Wrap the evaluated Lagrangian into this algorithm's subproblem model.

        Parameters
        ----------
        lagrangian
            Barrier-augmented Lagrangian evaluated at the current iterate.

        Returns
        -------
        SubProblemType
            Frozen model handed to the subproblem solver.
        """
        ...

    def _init_subproblem(
        self, problem: ProblemProtocol[InteriorPointPrimal]
    ) -> SubproblemContext[
        InteriorPointPrimal, SubProblemType, SubProblemSolverStateType
    ]:
        """Evaluate the Lagrangian, configure the solver and freeze the model."""
        iterate = cast(InteriorPointPrimal, self.iterate)
        dual = cast(Dual, self.dual)
        lag_module = cast(InteriorPointLagrangian, self._lagrangian_module(problem))
        lag = lag_module(iterate, dual)
        solver = self._make_subproblem_solver(problem)
        sub_opts = dict(self.options.get("subproblem", {}))
        if sub_opts:
            solver = solver.init(**sub_opts)
        solver = solver.init(logger=self.logger.child("subproblem"))
        solver = self._precondition(solver, problem, lag)
        zero_warm = cast(InteriorPointPrimal, jax.tree.map(jnp.zeros_like, iterate))
        return cast(
            SubproblemContext[
                InteriorPointPrimal, SubProblemType, SubProblemSolverStateType
            ],
            SubproblemContext(
                problem=problem,
                lagrangian=lag_module,
                subproblem=self._make_subproblem(lag),
                solver=solver,
                warm=(zero_warm, dual),
                state=cast(SubProblemSolverStateType, self.solver_state),
            ),
        )

    # ============================= termination =============================

    @staticmethod
    def _any_nonfinite(*trees) -> Bool[Array, ""]:
        """``True`` when any leaf of ``trees`` holds a non-finite entry."""
        leaves = jax.tree.leaves(trees)
        return jnp.logical_not(
            jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))
        )
