"""Fixtures shared by :mod:`slsqp_jax.sqpdax.minimiser` tests."""

from __future__ import annotations

from typing import Self, cast

import equinox as eqx
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, Bool

from slsqp_jax.sqpdax.active_set import ActiveSet
from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.lagrangian import EvaluatedLagrangian, Lagrangian
from slsqp_jax.sqpdax.merit import NormMerit
from slsqp_jax.sqpdax.minimiser import (
    CommonMinimiser,
    OptimisationContext,
)
from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.problem import Problem, ProblemProtocol, build_problem
from slsqp_jax.sqpdax.step_controller import ArmijoLineSearch, StepController
from slsqp_jax.sqpdax.subproblem import ActiveSetSubProblem
from slsqp_jax.sqpdax.subproblem.solver import (
    RESULTS,
    ActiveSetQPSolver,
    ActiveSetQPSolverState,
    ProjectedCGSubProblemSolver,
    SubproblemContext,
    SubProblemSolver,
)
from slsqp_jax.sqpdax.types import Scalar
from tests.sqpdax.lagrangian.conftest import make_problem
from tests.sqpdax.subproblem.solver.conftest import unbounded_box


def make_unconstrained_quadratic(*, n: int = 2) -> Problem:
    """``f(x) = ‖x‖²`` with exact HVP and no constraints / bounds."""
    lb, ub = unbounded_box(n)
    return make_problem(n=n, meq=0, mineq=0, lb=lb, ub=ub, with_curvature=True)


def make_build_unconstrained(*, n: int = 2) -> Problem:
    """Same quadratic via :func:`build_problem` (doctest-friendly path)."""
    return build_problem(
        n=n,
        meq=None,
        mineq=None,
        fn=lambda x: jnp.sum(x**2),
        grad=lambda x: 2 * x,
        hvp=lambda x, v: 2 * v,
        eq_fn=None,
        ineq_fn=None,
        eq_fn_jac=None,
        ineq_fn_jac=None,
        eq_fn_hvp=None,
        ineq_fn_hvp=None,
        lb=None,
        ub=None,
    )


class ActiveSetLineSearchStub(
    CommonMinimiser[Primal, ActiveSetSubProblem, ActiveSetQPSolverState]
):
    """Minimal active-set + Armijo concrete driver for unit tests.

    Mirrors the planned ``ActiveSetLineSearchMinimiser`` surface closely
    enough to exercise :class:`CommonMinimiser` end-to-end on small NLPs.
    """

    qp_tol: float = eqx.field(static=True, default=1e-8)
    qp_max_iter: int = eqx.field(static=True, default=20)
    penalty_floor: float = eqx.field(static=True, default=1.0)
    penalty_factor: float = eqx.field(static=True, default=2.0)
    armijo_c1: float = eqx.field(static=True, default=1e-4)
    armijo_backtrack: float = eqx.field(static=True, default=0.5)
    line_search_max_steps: int = eqx.field(static=True, default=20)

    def _subproblem_solver_type(self) -> type[SubProblemSolver]:
        return ActiveSetQPSolver

    def _init_solver_state(
        self, problem: ProblemProtocol[Primal], primal: Primal
    ) -> ActiveSetQPSolverState:
        return ActiveSetQPSolverState(
            n_iter=jnp.asarray(0, jnp.int32),
            n_cg_iter=jnp.asarray(0, jnp.int32),
            success=jnp.asarray(False),
            status=RESULTS.successful,
        )

    def _lagrangian_module(
        self, problem: ProblemProtocol[Primal]
    ) -> Lagrangian[Primal, EvaluatedLagrangian[Primal]]:
        return Lagrangian(problem, self.secant)

    def _init_subproblem(
        self, problem: ProblemProtocol[Primal]
    ) -> SubproblemContext[Primal, ActiveSetSubProblem, ActiveSetQPSolverState]:
        iterate = cast(Primal, self.iterate)
        dual = cast(Dual, self.dual)
        n, meq, mineq = problem.n, problem.meq, problem.mineq
        dtype = iterate.x.dtype
        lag_module = self._lagrangian_module(problem)
        qp_lag = lag_module(iterate, self._init_dual(problem))
        solver = ActiveSetQPSolver(
            subproblem_solver=ProjectedCGSubProblemSolver(),
            tol=self.qp_tol,
            max_iter=self.qp_max_iter,
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
        from slsqp_jax.sqpdax.barrier.update import _inf_norm

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
        from slsqp_jax.sqpdax.barrier.update import _inf_norm

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
        return self
