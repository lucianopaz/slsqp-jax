"""Limited-memory reduced-Hessian SQP + Armijo line-search outer loop."""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Self, cast

import equinox as eqx
from jax import numpy as jnp

from ..primal import Primal
from ..problem import ProblemProtocol
from ..registry import FrozenDict
from ..secant import LimitedMemoryReducedHessian, ReducedHessianSecant, Secant
from ..subproblem.solver import (
    LRHDirectQPSolver,
    LRHQPSolverState,
    SubProblemSolver,
)
from ..types import Vector_n
from .active_set_linesearch import ActiveSetLineSearchMinimiser

__all__ = [
    "LRHLineSearchMinimiser",
]


class LRHLineSearchMinimiser(ActiveSetLineSearchMinimiser[LRHQPSolverState]):
    """SLSQP-style line-search minimiser on a limited-memory reduced Hessian.

    Same outer loop as
    :class:`~slsqp_jax.sqpdax.minimiser.active_set_linesearch.ActiveSetLineSearchMinimiser`
    (L1 merit with Powell's decaying penalty, Armijo backtracking, the
    feasibility-restoration fallback, failure counters, best-iterate rollback,
    secant recovery and the violation-stationarity infeasibility certificate),
    but the Hessian model is a
    :class:`~slsqp_jax.sqpdax.secant.lrh.LimitedMemoryReducedHessian` and each
    QP is solved without any Krylov iteration by
    :class:`~slsqp_jax.sqpdax.subproblem.solver.lrh_direct_qp.LRHDirectQPSolver`
    (Householder / NNLS chain on the general constraints, a dual active-set
    loop on the variable bounds). Nothing of size ``n × n`` is formed.

    The model is always the quasi-Newton one: ``curvature="exact"`` is
    rejected, and problems that supply exact Hessian-vector products still run
    on the LRH model (the direct solver needs its factors). ``qp_max_iter``
    defaults to ``100`` (the bound loop's budget).

    Notes
    -----
    ``qp_single_exchange`` and ``options["subproblem"]["working_set_policy"]``
    have no meaning for the direct solver (it has no working-set policy) and
    are ignored with a warning. The LPEC-A predictor is allowed: it seeds the
    carried working set, which the solver consumes through ``warm_start``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.minimiser import LRHLineSearchMinimiser, minimise
    >>> from slsqp_jax.sqpdax.problem import build_problem
    >>> problem = build_problem(
    ...     lambda x: jnp.sum((x - jnp.array([1.0, 2.0])) ** 2),
    ...     n=2,
    ...     autodiff_mode="jax",
    ... )
    >>> sol = minimise(problem, LRHLineSearchMinimiser(), jnp.zeros(2), throw=False)
    >>> [round(float(v), 4) for v in sol.value]
    [1.0, 2.0]
    """

    qp_max_iter: int = eqx.field(static=True, default=100)

    def __check_init__(self) -> None:
        """Reject exact curvature, then validate the inherited configuration."""
        super().__check_init__()
        if self.curvature == "exact":
            raise ValueError(
                f"{type(self).__name__} always uses the limited-memory "
                "reduced-Hessian model; curvature='exact' is not supported "
                "(use 'auto' or 'secant')"
            )

    def validate_options(self) -> None:
        """Warn that ``qp_single_exchange`` is ignored, then validate as usual."""
        if self.qp_single_exchange:
            warnings.warn(
                f"{type(self).__name__}: 'qp_single_exchange' is ignored; the "
                "direct QP solver has no working-set policy"
            )
        super().validate_options()

    def _parse_options(self, options: dict | None) -> Self:
        """Parse as the parent does, dropping the ignored working-set policy.

        ``options['subproblem']['working_set_policy']`` is meaningless for the
        direct solver: it is reported with a warning and removed from the
        frozen bag, so it is neither applied nor reported again as an unknown
        key by :meth:`validate_options`.
        """
        base = super()._parse_options(options)
        sub = base.options.get("subproblem", {})
        if "working_set_policy" not in sub:
            return base
        warnings.warn(
            f"{type(self).__name__}: options['subproblem']['working_set_policy'] "
            "is ignored; the direct QP solver has no working-set policy"
        )
        stripped = {k: v for k, v in sub.items() if k != "working_set_policy"}
        return replace(
            base, options=FrozenDict({**dict(base.options), "subproblem": stripped})
        )

    def _uses_secant_model(self, problem: ProblemProtocol[Primal]) -> bool:
        """Always ``True``: the direct solver needs the reduced-Hessian factors."""
        return True

    def _make_secant(self, problem: ProblemProtocol[Primal]) -> Secant:
        """Default LRH secant; a ``minimiser.secant`` spec must be a reduced-Hessian one.

        Raises
        ------
        TypeError
            If the configured secant is not a
            :class:`~slsqp_jax.sqpdax.secant.reduced_hessian.ReducedHessianSecant`.
        """
        spec = self.options.get("minimiser", {}).get("secant")
        if spec is None:
            return cast(
                Secant,
                LimitedMemoryReducedHessian(n=problem.n, memory=self.secant_memory),
            )
        secant = super()._make_secant(problem)
        if not isinstance(secant, ReducedHessianSecant):
            raise TypeError(
                f"{type(self).__name__} needs a ReducedHessianSecant "
                f"(e.g. kind 'lrh'); got {type(secant).__name__}"
            )
        return secant

    def _init_primal(self, problem: ProblemProtocol[Primal], x0: Vector_n) -> Primal:
        """Project ``x0`` onto the variable box (``O(n)``)."""
        lb = jnp.where(problem.null_lb, -jnp.inf, problem.lb)
        ub = jnp.where(problem.null_ub, jnp.inf, problem.ub)
        return cast(Primal, Primal(x=jnp.clip(x0, lb, ub)))

    def _subproblem_solver_type(self) -> type[SubProblemSolver]:
        """Root solver class for ``options['subproblem']`` validation."""
        return LRHDirectQPSolver

    # The parent annotates its QP solvers as ``ActiveSetQPSolver``; the direct
    # solver is a sibling ``SubProblemSolver`` (it has no working-set policy and
    # no inner KKT solver), hence the override ignores below.
    def _make_qp_solver(  # ty: ignore[invalid-method-override]
        self, problem: ProblemProtocol[Primal], dtype: jnp.dtype
    ) -> LRHDirectQPSolver:
        """Direct solver carrying ``qp_max_iter`` / ``effective_qp_tol`` / warm start.

        There is no working-set policy to build.
        """
        return cast(
            LRHDirectQPSolver,
            LRHDirectQPSolver(
                max_iter=self.qp_max_iter,
                tol=self.effective_qp_tol,
                warm_start=self.qp_warm_start,
            ),
        )

    def _configured_qp_solver(  # ty: ignore[invalid-method-override]
        self, problem: ProblemProtocol[Primal], dtype: jnp.dtype
    ) -> LRHDirectQPSolver:
        """Default direct solver with ``options['subproblem']`` and the logger applied."""
        solver = self._make_qp_solver(problem, dtype)
        sub_opts = dict(self.options.get("subproblem", {}))
        if sub_opts:
            solver = solver.init(**sub_opts)
        return solver.init(logger=self.logger.child("subproblem"))

    def _init_solver_state(
        self, problem: ProblemProtocol[Primal], primal: Primal
    ) -> LRHQPSolverState:
        """Cold :class:`LRHQPSolverState` for the first outer step."""
        return LRHQPSolverState.cold(
            problem.n, problem.meq, problem.mineq, primal.x.dtype
        )
