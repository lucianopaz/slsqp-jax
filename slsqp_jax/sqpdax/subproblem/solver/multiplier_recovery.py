"""Multiplier recovery strategies for KKT subproblem solvers.

Given a primal step ``d`` the multipliers are the least-squares solution of the
stationarity row of the KKT system,

```
min_λ ‖Aᵀ λ − t‖,      t = −(∇f + H d)   (KKT)   or   t = −∇f   (LS),
```

where ``Aᵀ λ`` is the subproblem's
:meth:`~slsqp_jax.sqpdax.subproblem.base.SubProblem.kkt_mvp_upper_offdiag`
(Nocedal & Wright eq. 16.20 for the QP form, eq. 18.21 / 19.37 for the
Hessian-free form). A strategy answers "return *this* subproblem's dual in
*this* subproblem's convention"; it never adds ``λ_k`` itself.

Two linear-algebra paths share the same target and safeguard:

- with a :class:`~slsqp_jax.sqpdax.subproblem.solver.projector.ProjectionContext`
  the general block is solved exactly through the context's range-space
  solve on the free subspace and the bound block closes the fixed
  coordinates (today's projected-CG recovery);
- without one, LSMR runs matrix-free on the full off-diagonal operator, so
  any :class:`~slsqp_jax.sqpdax.subproblem.base.SubProblem` (including
  interior-point models whose bound rows live on slacks) is a valid input.

Safeguards interpret the sign of the dual block, so they refuse subproblems
that encode the Newton view
(:attr:`~slsqp_jax.sqpdax.subproblem.base.SubProblem.is_kkt_dual_increment`).
"""

from abc import abstractmethod
from typing import cast

import jax
import lineax as lx
from equinox import Module, field
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float
from lineax import AbstractLinearOperator, AbstractLinearSolver

from ...dual import Dual
from ...lagrangian.evaluated import InteriorPointEvaluatedLagrangian
from ...primal import PrimalType
from ...types import InitializableModule
from ..base import SubProblem
from .projector import ProjectionContext

__all__ = [
    "Safeguard",
    "ClampSafeguard",
    "BarrierSafeguard",
    "MultiplierRecovery",
    "KKTMultiplierRecovery",
    "LeastSquaresMultiplierRecovery",
]


def _require_sqp_view(subproblem: SubProblem, who: str) -> None:
    if subproblem.is_kkt_dual_increment:
        raise TypeError(
            f"{who} interprets the sign of the dual block and therefore requires "
            "SQP-view multipliers (λ_{k+1}); the subproblem declares the Newton "
            "view (a Δλ block)."
        )


class Safeguard(Module):
    """Post-hoc dual-feasibility repair applied to a recovered dual."""

    @abstractmethod
    def apply(self, subproblem: SubProblem, dual: Dual) -> Dual:
        """Repair ``dual`` in place of the least-squares estimate.

        Parameters
        ----------
        subproblem
            Model the multipliers were recovered on (SQP view required).
        dual
            Least-squares multipliers.

        Returns
        -------
        Dual
            Safeguarded multipliers.

        Raises
        ------
        TypeError
            If ``subproblem.is_kkt_dual_increment`` is set.
        """


class ClampSafeguard(Safeguard):
    """Clamp inequality and bound multipliers at ``max(0, ·)``.

    Equality multipliers are signed and left untouched. Rows outside the
    working set already carry exact zeros, so the clamp only acts on active
    rows whose unconstrained least-squares value came out negative.
    """

    def apply(self, subproblem: SubProblem, dual: Dual) -> Dual:
        _require_sqp_view(subproblem, "ClampSafeguard")
        clamp = lambda v: jnp.maximum(v, jnp.zeros_like(v))  # noqa: E731
        return cast(
            Dual,
            Dual(
                eq_multipliers=dual.eq_multipliers,
                ineq_multipliers=clamp(dual.ineq_multipliers),
                lb_multipliers=clamp(dual.lb_multipliers),
                ub_multipliers=clamp(dual.ub_multipliers),
            ),
        )


class BarrierSafeguard(Safeguard):
    """Interior-point positivity safeguard (Nocedal & Wright eq. 19.38).

    Non-positive inequality / bound multipliers are replaced by
    ``min(cap, μ / s_i)`` with the matching slack ``s_i``; null bounds get an
    exact zero.

    Attributes
    ----------
    cap
        Upper cap of the replacement value (``1e-3`` in N&W).
    """

    cap: float = field(static=True, default=1e-3)

    def apply(self, subproblem: SubProblem, dual: Dual) -> Dual:
        _require_sqp_view(subproblem, "BarrierSafeguard")
        lag = subproblem.lagrangian
        if not isinstance(lag, InteriorPointEvaluatedLagrangian):
            raise TypeError(
                "BarrierSafeguard needs an interior-point subproblem (barrier "
                f"weight and slacks); got {type(lag).__name__}."
            )
        mu = lag.barrier.weight
        s, s_lb, s_ub = lag.slack.s, lag.slack.s_lb, lag.slack.s_ub
        cap = jnp.asarray(self.cap, s.dtype)

        def fix(z: Float[Array, " k"], slack: Float[Array, " k"]) -> Float[Array, " k"]:
            return jnp.where(z > 0, z, jnp.minimum(cap, mu / slack))

        z_lb = jnp.where(
            lag.null_lb,
            0.0,
            fix(dual.lb_multipliers, jnp.where(lag.null_lb, 1.0, s_lb)),
        )
        z_ub = jnp.where(
            lag.null_ub,
            0.0,
            fix(dual.ub_multipliers, jnp.where(lag.null_ub, 1.0, s_ub)),
        )
        return cast(
            Dual,
            Dual(
                eq_multipliers=dual.eq_multipliers,
                ineq_multipliers=fix(dual.ineq_multipliers, s),
                lb_multipliers=z_lb,
                ub_multipliers=z_ub,
            ),
        )


class MultiplierRecovery(InitializableModule):
    """Strategy returning a subproblem's dual for a given primal step.

    Attributes
    ----------
    refinement_rounds
        Rounds of iterative refinement on the normal / least-squares solve
        (Nocedal & Wright §16.3 closing remark). Each round re-solves on the
        residual ``t − Aᵀλ`` and adds the correction.
    safeguard
        Optional :class:`Safeguard` applied to the final dual.
    rtol, atol, max_steps
        LSMR tolerances for the matrix-free path (used when
        :meth:`recover` is called without a projection context).
    """

    refinement_rounds: int = field(static=True, default=1)
    safeguard: Safeguard | None = None
    rtol: float = field(static=True, default=1e-8)
    atol: float = field(static=True, default=1e-8)
    max_steps: int | None = field(static=True, default=None)

    @abstractmethod
    def stationarity_target(
        self, subproblem: SubProblem[PrimalType], primal_step: PrimalType
    ) -> PrimalType:
        """Vector ``t`` such that the recovered dual solves ``min ‖Aᵀλ − t‖``.

        Parameters
        ----------
        subproblem
            KKT model at the current iterate.
        primal_step
            Primal step the multipliers are recovered for.

        Returns
        -------
        PrimalType
            Target in the primal space.
        """

    def recover(
        self,
        subproblem: SubProblem[PrimalType],
        projector: ProjectionContext | None,
        primal_step: PrimalType,
    ) -> Dual:
        """Recover the dual block for ``primal_step``.

        Parameters
        ----------
        subproblem
            KKT model at the current iterate.
        projector
            Projection context built on ``subproblem``'s working set. When
            given, the general block is solved exactly through its
            range-space solve and the bound block closes the fixed
            coordinates; requires the primal to be the ``x`` block only.
            ``None`` selects the matrix-free LSMR path.
        primal_step
            Primal step the multipliers are recovered for.

        Returns
        -------
        Dual
            Multipliers in the subproblem's own dual convention.

        Raises
        ------
        TypeError
            If ``projector`` is given but the subproblem's primal has more
            coordinates than the projector's Jacobian columns.
        """
        target, unravel_primal = ravel_pytree(
            self.stationarity_target(subproblem, primal_step)
        )
        zero_primal = unravel_primal(jnp.zeros_like(target))
        zero_dual = cast(Dual, jax.tree.map(jnp.zeros_like, subproblem.lagrangian.dual))
        zero_dual_flat, unravel_dual = ravel_pytree(zero_dual)

        def At(lam_flat: Float[Array, " m"]) -> Float[Array, " n_p"]:
            lam = cast(Dual, unravel_dual(lam_flat))
            return ravel_pytree(subproblem.kkt_mvp_upper_offdiag((zero_primal, lam)))[0]

        def A(v_flat: Float[Array, " n_p"]) -> Float[Array, " m"]:
            v = unravel_primal(v_flat)
            return ravel_pytree(subproblem.kkt_mvp_lower_offdiag((v, zero_dual)))[0]

        if projector is None:
            dual = self._recover_matrix_free(
                At, A, unravel_dual, zero_dual_flat.shape[0], target
            )
        else:
            if target.shape[0] != projector.A.shape[1]:
                raise TypeError(
                    "A projection context acts on the decision variables only; "
                    f"the subproblem primal has {target.shape[0]} coordinates but "
                    f"the projector Jacobian {projector.A.shape[1]} columns. Pass "
                    "projector=None to use the matrix-free path."
                )
            dual = self._recover_with_projector(projector, At, zero_dual, target)
        if self.safeguard is not None:
            dual = self.safeguard.apply(subproblem, dual)
        return dual

    def _recover_with_projector(
        self,
        ctx: ProjectionContext,
        At,
        zero_dual: Dual,
        target: Float[Array, " n"],
    ) -> Dual:
        free_f = ctx.free_f
        active = ctx.active_rows
        A, A_work = ctx.A, ctx.A_work
        lam_g = jnp.where(active, ctx.solve_normal(A_work @ (free_f * target)), 0.0)
        for _ in range(self.refinement_rounds):
            r_ref = free_f * (target - A.T @ lam_g)
            lam_g = jnp.where(active, lam_g + ctx.solve_normal(A_work @ r_ref), 0.0)
        meq = zero_dual.meq
        # Bound multipliers close the stationarity residual on the fixed
        # coordinates. The ``±1`` bound Jacobian entries of the working set
        # are read off the operator itself (lower bound wins a tie).
        sign_lb = At(ravel_pytree(_bound_probe(zero_dual, lower=True))[0])
        sign_ub = At(ravel_pytree(_bound_probe(zero_dual, lower=False))[0])
        gap = target - A.T @ lam_g
        lam_lb = jnp.where(sign_lb != 0, sign_lb * gap, 0.0)
        lam_ub = jnp.where((sign_ub != 0) & (sign_lb == 0), sign_ub * gap, 0.0)
        return cast(
            Dual,
            Dual(
                eq_multipliers=lam_g[:meq],
                ineq_multipliers=lam_g[meq:],
                lb_multipliers=lam_lb,
                ub_multipliers=lam_ub,
            ),
        )

    def _recover_matrix_free(
        self,
        At,
        A,
        unravel_dual,
        m_total: int,
        target: Float[Array, " n_p"],
    ) -> Dual:
        # ``min_λ ‖Aᵀλ − t‖`` on the full dual by LSMR (pseudo-inverse solution
        # for rank-deficient ``A``: masked rows have zero columns and come out
        # as exact zeros). Same construction as the trust-region interior-point
        # solver's eq. 19.37 multipliers.
        dtype = target.dtype
        At_op = cast(
            AbstractLinearOperator,
            lx.FunctionLinearOperator(At, jax.ShapeDtypeStruct((m_total,), dtype)),
        )
        lsmr = cast(
            AbstractLinearSolver,
            lx.LSMR(rtol=self.rtol, atol=self.atol, max_steps=self.max_steps),
        )

        def solve(rhs: Float[Array, " n_p"]) -> Float[Array, " m"]:
            # ``A rhs = 0`` means ``rhs ⟂ range(Aᵀ)``: the least-squares solution
            # is exactly zero, and LSMR's bidiagonalisation breaks down (0/0)
            # on it — e.g. an empty working set, or a refinement residual that
            # is already orthogonal to the range.
            in_range = jnp.any(A(rhs) != 0)
            val = lx.linear_solve(At_op, rhs, lsmr, throw=False).value
            return jnp.where(in_range, val, jnp.zeros_like(val))

        lam = solve(target)
        for _ in range(self.refinement_rounds):
            lam = lam + solve(target - At(lam))
        return cast(Dual, unravel_dual(lam))


class KKTMultiplierRecovery(MultiplierRecovery):
    """QP-consistent multipliers ``min ‖Aᵀλ + ∇f + H d‖`` (N&W eq. 16.20).

    The Hessian term makes the dual consistent with the *step* ``d``: this is
    what a working-set policy needs for its sign tests, and what the inner
    KKT solvers use by default.
    """

    def stationarity_target(
        self, subproblem: SubProblem[PrimalType], primal_step: PrimalType
    ) -> PrimalType:
        zero_dual = cast(Dual, jax.tree.map(jnp.zeros_like, subproblem.lagrangian.dual))
        Hd = subproblem.kkt_mvp_primal((primal_step, zero_dual))
        grad = subproblem.primal_grad()
        return cast(PrimalType, jax.tree.map(lambda g, h: -(g + h), grad, Hd))


class LeastSquaresMultiplierRecovery(MultiplierRecovery):
    """Hessian-free multipliers ``min ‖Aᵀλ + ∇f‖`` at the evaluation point.

    Independent of the QP Hessian (exact or secant) and of the step, so the
    estimate is the multiplier vector minimising the actual stationarity
    residual at the reference point (N&W eq. 18.21 / 19.37). Pair with a
    :class:`ClampSafeguard` (active-set) or :class:`BarrierSafeguard`
    (interior-point) to enforce dual feasibility.
    """

    def stationarity_target(
        self, subproblem: SubProblem[PrimalType], primal_step: PrimalType
    ) -> PrimalType:
        del primal_step
        return cast(PrimalType, jax.tree.map(jnp.negative, subproblem.primal_grad()))


def _bound_probe(zero_dual: Dual, *, lower: bool) -> Dual:
    """All-ones lower (or upper) bound multipliers, zeros elsewhere."""
    probe = jnp.ones_like(zero_dual.lb_multipliers)
    return cast(
        Dual,
        Dual(
            eq_multipliers=zero_dual.eq_multipliers,
            ineq_multipliers=zero_dual.ineq_multipliers,
            lb_multipliers=probe if lower else zero_dual.lb_multipliers,
            ub_multipliers=zero_dual.ub_multipliers if lower else probe,
        ),
    )
