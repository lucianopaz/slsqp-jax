"""Proximal (stabilised-SQP) wrapper around the active-set QP loop."""

from typing import Generic, cast

from equinox import tree_at
from jax import numpy as jnp

from ...dual import Dual
from ...preconditioner import IdentityPreconditioner, woodbury_preconditioner
from ...primal import Primal
from ...types import Scalar, Vector_meq
from ..active_set import ActiveSetSubProblem
from ..proximal import ProximalActiveSetSubProblem
from .active_set_loop import (
    ActiveSetQPSolver,
    ActiveSetQPSolverState,
    KKTSolverStateType,
)
from .base import SubProblemSolver

__all__ = [
    "ProximalActiveSetQPSolverState",
    "ProximalActiveSetQPSolver",
]


class ProximalActiveSetQPSolverState(ActiveSetQPSolverState):
    """Carry for :class:`ProximalActiveSetQPSolver`.

    Attributes
    ----------
    kkt_residual
        Outer KKT residual of the NLP at the current iterate. Written by the
        minimiser before each solve; drives the proximal parameter.
    mu
        Proximal parameter used by the most recent solve.
    eq_center
        Equality-multiplier centre ``λ_k`` of the proximal term. Warm-started
        from the previous solve's recovered multipliers.
    """

    kkt_residual: Scalar
    mu: Scalar
    eq_center: Vector_meq


class ProximalActiveSetQPSolver(
    ActiveSetQPSolver[KKTSolverStateType, ProximalActiveSetQPSolverState],
    Generic[KKTSolverStateType],
):
    """Stabilised-SQP active-set QP solver.

    Wraps the working-set loop of
    :class:`~slsqp_jax.sqpdax.subproblem.solver.active_set_loop.ActiveSetQPSolver`
    around a
    :class:`~slsqp_jax.sqpdax.subproblem.proximal.ProximalActiveSetSubProblem`:
    the equalities are eliminated through a proximal term with parameter

    ```
    μ = clip(kkt_residual ** τ, μ_min, μ_max),      τ ∈ [0, 1),
    ```

    so the stabilised system tends to the plain SQP system as the outer
    iterate approaches a KKT point, while ``μ_min > 0`` keeps ``1/μ`` finite
    however tight the outer tolerance. Inequalities and bounds are handled by
    the inherited working-set loop unchanged. After the loop the equality
    multipliers are recovered as ``λ_k + (A_eq d + c_eq) / μ`` and stored as
    the next centre.

    When the inner KKT solver carries a non-identity ``preconditioner`` and
    the problem has equalities, the preconditioner is replaced by its
    Woodbury update ``M + (1/μ) A_eqᵀ A_eq`` for the duration of the solve,
    so the constraint preconditioner stays consistent with the stabilised
    Hessian.

    Attributes
    ----------
    tau
        Exponent of the residual-driven schedule; ``τ = 0`` gives the
        constant-penalty variant ``μ = μ_max``. Proximal treatment is not
        switchable off here; use ``ActiveSetQPSolver`` for that.
    mu_min
        Floor on ``μ`` (must be positive).
    mu_max
        Ceiling on ``μ``.
    """

    solver_state_class: type[ProximalActiveSetQPSolverState] = (
        ProximalActiveSetQPSolverState
    )
    tau: float = 0.5
    mu_min: float = 1e-6
    mu_max: float = 0.1

    def __check_init__(self) -> None:
        """Validate the schedule parameters."""
        if not (0.0 <= self.tau < 1.0):
            raise ValueError(f"tau must lie in [0, 1); got {self.tau}")
        if not self.mu_min > 0.0:
            raise ValueError(f"mu_min must be positive; got {self.mu_min}")
        if not self.mu_max >= self.mu_min:
            raise ValueError(
                f"mu_max must be at least mu_min; got mu_max={self.mu_max}, "
                f"mu_min={self.mu_min}"
            )

    def proximal_mu(self, kkt_residual: Scalar) -> Scalar:
        """Proximal parameter ``clip(kkt_residual ** τ, μ_min, μ_max)``.

        Parameters
        ----------
        kkt_residual
            Outer KKT residual. ``+∞`` (the cold-start seed) maps to
            ``μ_max``; a NaN residual is not masked here — it is already
            caught upstream by the minimiser's non-finite termination check.

        Returns
        -------
        Scalar
            Proximal parameter in ``[μ_min, μ_max]``.
        """
        raw = jnp.power(jnp.asarray(kkt_residual), self.tau)
        return jnp.clip(raw, self.mu_min, self.mu_max)

    def _kkt_solver(
        self, subproblem: ActiveSetSubProblem
    ) -> SubProblemSolver[Primal, ActiveSetSubProblem, KKTSolverStateType]:
        """Inner solver with its preconditioner Woodbury-wrapped for ``μ``.

        Parameters
        ----------
        subproblem
            The :class:`~slsqp_jax.sqpdax.subproblem.proximal.ProximalActiveSetSubProblem`
            built by :meth:`solve`; supplies ``A_eq`` and ``μ``.

        Returns
        -------
        SubProblemSolver
            :attr:`subproblem_solver` unchanged when it has no preconditioner,
            an identity one, or the problem has no equalities; otherwise a
            copy whose preconditioner is
            :func:`~slsqp_jax.sqpdax.preconditioner.utils.woodbury_preconditioner`
            of the original.
        """
        inner = self.subproblem_solver
        pre = getattr(inner, "preconditioner", None)
        if (
            pre is None
            or isinstance(pre, IdentityPreconditioner)
            or subproblem.lagrangian.meq == 0
        ):
            return inner
        prox = cast(ProximalActiveSetSubProblem, subproblem)
        return tree_at(
            lambda s: s.preconditioner,
            inner,
            woodbury_preconditioner(pre, prox.eq_jac, prox.mu),
        )

    def solve(
        self,
        subproblem: ActiveSetSubProblem,
        x0: tuple[Primal, Dual],
        initial_state: ProximalActiveSetQPSolverState,
    ) -> tuple[tuple[Primal, Dual], ProximalActiveSetQPSolverState]:
        """Solve the stabilised QP by the inherited active-set loop.

        Parameters
        ----------
        subproblem
            Reference working-set QP; only its unmasked Lagrangian is used.
        x0
            Warm-start ``(primal_step, dual)`` for the first KKT solve.
        initial_state
            Carry providing ``kkt_residual`` (drives ``μ``) and ``eq_center``.

        Returns
        -------
        step
            Primal step and multipliers with the equality block recovered
            from the proximal update.
        state
            Updated carry with ``mu`` and the new ``eq_center``.
        """
        lag = subproblem.lagrangian
        dtype = lag.x_ref.dtype
        mu = self.proximal_mu(initial_state.kkt_residual).astype(dtype)
        eq_center = jnp.asarray(initial_state.eq_center, dtype)
        prox = cast(
            ProximalActiveSetSubProblem,
            ProximalActiveSetSubProblem(lag, subproblem.active_set, mu, eq_center),
        )

        step, state = super().solve(prox, x0, initial_state)
        lam_eq = prox.recover_eq_multipliers(step[0].x)
        step = (step[0], tree_at(lambda d: d.eq_multipliers, step[1], lam_eq))
        new_state = tree_at(lambda s: (s.mu, s.eq_center), state, (mu, lam_eq))
        return step, new_state
