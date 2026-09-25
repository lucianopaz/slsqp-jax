"""Proximal (stabilised-SQP) active-set QP subproblem."""

from typing import Self, cast

from equinox import tree_at
from jax import numpy as jnp

from ..active_set import ActiveSet
from ..dual import Dual
from ..lagrangian import EvaluatedLagrangian
from ..primal import Primal
from ..types import Scalar, Vector_meq, Vector_n
from .active_set import ActiveSetSubProblem

__all__ = [
    "ProximalActiveSetSubProblem",
]


class ProximalActiveSetSubProblem(ActiveSetSubProblem):
    """Working-set QP with the equalities eliminated through a proximal term.

    Stabilised SQP (Wright 1998; Gill & Robinson 2013) replaces the equality
    rows of the QP by a proximal-point term in the multipliers,

    ```
    min_d  max_λ  gᵀd + ½ dᵀBd + λᵀ(c_eq + A_eq d) − (μ/2) ‖λ − λ_k‖²
    ```

    Eliminating ``λ`` gives an unconstrained-in-``d`` model whose Hessian and
    gradient are

    ```
    B̃ = B + (1/μ) A_eqᵀ A_eq,        g̃ = g + A_eqᵀ (λ_k + c_eq / μ),
    ```

    and whose multipliers are recovered as ``λ = λ_k + (A_eq d + c_eq) / μ``.
    This class exposes exactly that model through the
    :class:`~slsqp_jax.sqpdax.subproblem.base.SubProblem` surface: the
    equality rows of the masked Lagrangian :attr:`L_k` are zeroed, so any
    inner KKT solver (e.g.
    :class:`~slsqp_jax.sqpdax.subproblem.solver.projected_cg.ProjectedCGSubProblemSolver`)
    sees ``A_eq = 0`` / ``c_eq = 0`` and only projects onto the active
    inequality / bound rows, while the stabilised Hessian and gradient enter
    through :meth:`kkt_mvp_primal` and :meth:`primal_grad`.

    Attributes
    ----------
    mu
        Proximal parameter ``μ > 0``.
    eq_center
        Multiplier centre ``λ_k`` of the proximal term (length ``meq``).

    Notes
    -----
    Sign convention follows the rest of ``sqpdax``:
    ``∇_x L = ∇f + A_eqᵀ λ_eq + …`` and ``dual_grad.eq = c_eq``, so
    ``kkt_rhs()[1].eq = -c_eq``.

    The equality multipliers of :attr:`L_k` are zeroed together with the
    equality rows. With exact curvature this drops the equality-constraint
    curvature ``Σ λ_i ∇² c_i`` from the QP Hessian; the line-search
    minimiser evaluates the QP Lagrangian at a zero dual anyway, so this
    matches its existing behaviour.
    """

    mu: Scalar
    eq_center: Vector_meq

    def __init__(
        self,
        lagrangian: EvaluatedLagrangian[Primal],
        active_set: ActiveSet,
        mu: Scalar,
        eq_center: Vector_meq,
    ):
        """Build the proximal working-set QP.

        Parameters
        ----------
        lagrangian
            Cached Lagrangian at the current iterate (unmasked).
        active_set
            Working-set masks applied to form :attr:`L_k`.
        mu
            Proximal parameter ``μ > 0``.
        eq_center
            Multiplier centre ``λ_k`` (length ``meq``).
        """
        L_k = active_set.mask_lagrangian(lagrangian)
        self.lagrangian = lagrangian
        self.active_set = active_set
        self.L_k = tree_at(
            lambda lag: (
                lag.evaluated.eq_fn_val,
                lag.evaluated.eq_fn_jac_val,
                lag.dual.eq_multipliers,
            ),
            L_k,
            (
                jnp.zeros_like(L_k.evaluated.eq_fn_val),
                jnp.zeros_like(L_k.evaluated.eq_fn_jac_val),
                jnp.zeros_like(L_k.dual.eq_multipliers),
            ),
        )
        self.mu = jnp.asarray(mu, lagrangian.x_ref.dtype)
        self.eq_center = jnp.asarray(eq_center, lagrangian.x_ref.dtype)

    def with_active_set(self, active_set: ActiveSet) -> Self:
        """Rebuild with a new working set, keeping ``mu`` and ``eq_center``."""
        return type(self)(self.lagrangian, active_set, self.mu, self.eq_center)

    @property
    def eq_jac(self):
        """Unmasked equality Jacobian ``A_eq`` (shape ``(meq, n)``)."""
        return self.lagrangian.eq_fn_jac_val

    @property
    def eq_val(self) -> Vector_meq:
        """Unmasked equality residual ``c_eq``."""
        return self.lagrangian.eq_fn_val

    def stabilisation_hvp(self, v: Vector_n) -> Vector_n:
        """Proximal curvature ``(1/μ) A_eqᵀ (A_eq v)``.

        Parameters
        ----------
        v
            Decision-variable tangent.

        Returns
        -------
        Vector_n
            ``A_eqᵀ A_eq v / μ``; zero when ``meq == 0``.
        """
        A = self.eq_jac
        return (A.T @ (A @ v)) / self.mu

    def kkt_mvp_primal(self, step: tuple[Primal, Dual]) -> Primal:
        """Stabilised Hessian product ``B v + (1/μ) A_eqᵀ A_eq v``."""
        base = self.L_k.kkt_mvp_primal(step)
        return cast(Primal, Primal(x=base.x + self.stabilisation_hvp(step[0].x)))

    def primal_grad(self) -> Primal:
        """Stabilised gradient ``g + A_eqᵀ (λ_k + c_eq / μ)``."""
        base = self.L_k.primal_grad
        shift = self.eq_jac.T @ (self.eq_center + self.eq_val / self.mu)
        return cast(Primal, Primal(x=base.x + shift))

    def kkt_mvp(self, step: tuple[Primal, Dual]) -> tuple[Primal, Dual]:
        """Full KKT product assembled from the stabilised blocks.

        Notes
        -----
        Routes through :meth:`~slsqp_jax.sqpdax.subproblem.base.SubProblem.kkt_operator`
        so :meth:`residual`, :meth:`model_value` and
        :meth:`predicted_reduction` all see the stabilised operator rather
        than the masked Lagrangian's own ``kkt_mvp``.
        """
        return self.kkt_operator(step)

    def recover_eq_multipliers(self, d: Vector_n) -> Vector_meq:
        """Proximal multiplier update ``λ_k + (A_eq d + c_eq) / μ``.

        Parameters
        ----------
        d
            Primal QP step.

        Returns
        -------
        Vector_meq
            Recovered equality multipliers.
        """
        return self.eq_center + (self.eq_jac @ d + self.eq_val) / self.mu
