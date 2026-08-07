"""Abstract matrix-free KKT subproblem shared by SQP step solvers."""

from abc import abstractmethod
from typing import Generic

import jax
from equinox import Module
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float

from ..dual import Dual
from ..lagrangian import EvaluatedLagrangian
from ..primal import PrimalType
from ..types import Scalar

__all__ = [
    "SubProblem",
]


class SubProblem(Module, Generic[PrimalType]):
    """Local KKT model of a Lagrangian that a solver turns into a step.

    Every concrete subclass implements the abstract blocks below; shared
    helpers (:meth:`kkt_operator`, :meth:`kkt_rhs`, :meth:`model_value`,
    :meth:`predicted_reduction`, bound geometry) are derived once here.

    Attributes
    ----------
    lagrangian
        Cached Lagrangian at the reference point about which the local
        model is built.

    Notes
    -----
    Encodes the quadratic program

    ```
    min_p  1/2 ⟨p, H p⟩ + ⟨g, p⟩    s.t.  A p = b
    ```

    in matrix-free form: ``H`` and ``A`` are exposed only through Hessian /
    Jacobian-vector products.
    """

    lagrangian: EvaluatedLagrangian[PrimalType]

    @abstractmethod
    def primal_grad(self) -> PrimalType:
        """Primal gradient ``g`` of the local model.

        Returns
        -------
        PrimalType
            Packed primal gradient (decision variables, and slacks when
            present).
        """

    @abstractmethod
    def dual_grad(self) -> Dual:
        """Dual residual ``b`` (linearized constraint / bound violation).

        Returns
        -------
        Dual
            Equality, inequality, and bound residual blocks.
        """

    @abstractmethod
    def kkt_mvp_primal(self, step: tuple[PrimalType, Dual]) -> PrimalType:
        """Primal-primal KKT block ``H p``.

        Parameters
        ----------
        step
            Primal-dual tangent ``(p, δλ)``.

        Returns
        -------
        PrimalType
            Hessian-vector product in the primal block.
        """

    @abstractmethod
    def kkt_mvp_upper_offdiag(self, step: tuple[PrimalType, Dual]) -> PrimalType:
        """Primal-dual upper off-diagonal ``Aᵀ δλ``.

        Parameters
        ----------
        step
            Primal-dual tangent ``(p, δλ)``.

        Returns
        -------
        PrimalType
            Constraint-Jacobian transpose times the dual tangent.
        """

    @abstractmethod
    def kkt_mvp_lower_offdiag(self, step: tuple[PrimalType, Dual]) -> Dual:
        """Dual-primal lower off-diagonal ``A p``.

        Parameters
        ----------
        step
            Primal-dual tangent ``(p, δλ)``.

        Returns
        -------
        Dual
            Constraint Jacobian times the primal tangent.
        """

    @abstractmethod
    def kkt_mvp_dual(self, step: tuple[PrimalType, Dual]) -> Dual:
        """Dual-dual KKT block (regularization, or zeros).

        Parameters
        ----------
        step
            Primal-dual tangent ``(p, δλ)``.

        Returns
        -------
        Dual
            Dual-dual product; typically zeros unless dual regularization
            is active.
        """

    @abstractmethod
    def residual(self, step: tuple[PrimalType, Dual]) -> tuple[PrimalType, Dual]:
        """Full KKT residual ``K z - rhs`` at ``step``.

        Parameters
        ----------
        step
            Candidate primal-dual step.

        Returns
        -------
        tuple of PrimalType and Dual
            Primal and dual residual blocks.
        """

    @property
    def is_kkt_dual_regularized(self) -> bool:
        """Whether :meth:`kkt_operator` includes the dual-dual block.

        Returns
        -------
        bool
            ``False`` by default; interior-point subclasses may override.
        """
        return False

    @abstractmethod
    def nonbound_constraint_jac(self) -> Float[Array, " meq+mineq n"]:
        """Stacked equality and inequality Jacobians (no bound rows).

        Returns
        -------
        jax.Array
            Matrix of shape ``(meq + mineq, n)``.
        """

    def kkt_operator(self, step: tuple[PrimalType, Dual]) -> tuple[PrimalType, Dual]:
        """Assemble the saddle-point operator ``K z``.

        Parameters
        ----------
        step
            Primal-dual tangent ``(p, δλ)``.

        Returns
        -------
        tuple of PrimalType and Dual
            Action of the KKT operator on ``step``.

        Notes
        -----
        The primal row is ``H p + Aᵀ δλ``. The dual row is ``A p``, plus the
        dual-dual block when :attr:`is_kkt_dual_regularized` is true.
        """
        return (
            jax.tree.map(
                jnp.add, self.kkt_mvp_primal(step), self.kkt_mvp_upper_offdiag(step)
            ),
            jax.lax.cond(
                self.is_kkt_dual_regularized,
                lambda: jax.tree.map(
                    jnp.add, self.kkt_mvp_lower_offdiag(step), self.kkt_mvp_dual(step)
                ),
                lambda: self.kkt_mvp_lower_offdiag(step),
            ),
        )

    def kkt_rhs(self) -> tuple[PrimalType, Dual]:
        """Right-hand side ``-g, -b`` of the Newton-KKT system.

        Returns
        -------
        tuple of PrimalType and Dual
            Negated primal and dual gradients.
        """
        return jax.tree.map(lambda x: -x, (self.primal_grad(), self.dual_grad()))

    def model_value(self, step: tuple[PrimalType, Dual]) -> Scalar:
        """Quadratic model of the objective change along the primal step.

        Parameters
        ----------
        step
            Primal-dual step; only ``step[0]`` enters the model.

        Returns
        -------
        Scalar
            Predicted objective change.

        Notes
        -----
        Computes ``⟨g, p⟩ + 1/2 ⟨p, H p⟩``. The dual tangent is zeroed before
        the Hessian product so only the primal-primal block contributes.
        """
        primal = step[0]
        primal_step = (primal, jax.tree.map(lambda x: jnp.zeros_like(x), step[1]))
        return jnp.inner(self.primal_grad().x, primal.x) + 0.5 * jnp.inner(
            primal.x, self.kkt_mvp_primal(primal_step).x
        )

    def linearized_infeasibility(self, step: tuple[PrimalType, Dual]) -> Scalar:
        """Norm of the dual residual block (linearized constraint violation).

        Parameters
        ----------
        step
            Candidate primal-dual step.

        Returns
        -------
        Scalar
            Euclidean norm of the flattened dual residual.

        Notes
        -----
        Equals ``‖A p - b‖``. :meth:`residual` already subtracts
        :meth:`kkt_rhs`, so the dual residual must not be subtracted again;
        ``Dual.flatten()`` turns the dual pytree into a vector for the norm.
        """
        return jnp.linalg.norm(self.residual(step)[1].flatten())

    def predicted_reduction(
        self, step: tuple[PrimalType, Dual], penalty: Scalar
    ) -> Scalar:
        """Merit-function predicted reduction (Nocedal & Wright eq. 19.41).

        Parameters
        ----------
        step
            Candidate primal-dual step.
        penalty
            Merit penalty parameter ``μ`` (or equivalent).

        Returns
        -------
        Scalar
            Predicted reduction in the exact-penalty merit function.

        Notes
        -----
        ``pred = -m(p) + μ (‖c‖ - ‖c + A p‖)``, with ``m`` from
        :meth:`model_value` and the infeasibility norms from
        :meth:`kkt_rhs` / :meth:`linearized_infeasibility`.
        """
        m0 = jnp.linalg.norm(self.kkt_rhs()[1].flatten())
        return -self.model_value(step) + penalty * (
            m0 - self.linearized_infeasibility(step)
        )

    def step_norm(self, step: tuple[PrimalType, Dual]) -> Scalar:
        """Trust-region norm of the primal step.

        Parameters
        ----------
        step
            Primal-dual step.

        Returns
        -------
        Scalar
            ``‖flatten(p)‖₂``. Subclasses with non-Euclidean geometry
            should override.
        """
        return jnp.linalg.norm(step[0].flatten())

    def to_native_step(self, step: tuple[PrimalType, Dual]) -> tuple[PrimalType, Dual]:
        """Map a solver step into native (unscaled) primal-dual coordinates.

        Parameters
        ----------
        step
            Step in the geometry used by the trust-region / line-search
            solver.

        Returns
        -------
        tuple of PrimalType and Dual
            Identity by default; scaled subclasses unscale here.
        """
        return step

    def primal_box(self) -> tuple[Float[Array, " n_p"], Float[Array, " n_p"]]:
        """Componentwise ``(lower, upper)`` for the primal step in ``flatten()`` order.

        Returns
        -------
        lower, upper
            Flat bound vectors matching
            :meth:`~slsqp_jax.sqpdax.primal.Primal.flatten`.

        Notes
        -----
        Default geometry is classical variable bounds in *step* units:
        ``ℓ = lb - x_k``, ``u = ub - x_k``, with ``±inf`` on null bounds.
        Consumed by gradient-projection solvers (Nocedal & Wright §16.7).
        """
        lag = self.lagrangian
        dtype = lag.ref.x.dtype
        inf = jnp.asarray(jnp.inf, dtype)
        lo = jnp.where(lag.null_lb, -inf, lag.lb - lag.ref.x)
        hi = jnp.where(lag.null_ub, inf, lag.ub - lag.ref.x)
        return lo, hi

    def active_bounds(
        self, primal: PrimalType, tol: Scalar | float = 0.0
    ) -> tuple[Bool[Array, " n_p"], Bool[Array, " n_p"]]:
        """Variable-bound faces active at ``primal`` (Nocedal & Wright ``A(xᶜ)``).

        Parameters
        ----------
        primal
            Primal step (or Cauchy point) in the same layout as
            :meth:`primal_box`.
        tol
            Absolute tolerance for declaring a face active.

        Returns
        -------
        active_lb, active_ub
            Boolean masks of length ``n``.

        Notes
        -----
        A coordinate of the plain :class:`~slsqp_jax.sqpdax.primal.Primal`
        step is active when it sits on the corresponding face of
        :meth:`primal_box`. Subclasses whose primal is not a plain ``x``-step
        (interior-point slacks, scalings, …) override this and
        :meth:`primal_box` so the solver never inspects the concrete type.
        """
        lag = self.lagrangian
        lo, hi = self.primal_box()
        flat = primal.flatten()
        active_lb = (~lag.null_lb) & (flat <= lo + tol)
        active_ub = (~lag.null_ub) & (flat >= hi - tol)
        return active_lb, active_ub

    def unflatten_primal(self, flat: Float[Array, " n_p"]) -> PrimalType:
        """Rebuild a primal pytree from a flat vector.

        Parameters
        ----------
        flat
            Contiguous primal step in ``flatten()`` order.

        Returns
        -------
        PrimalType
            Rebuilt primal pytree.

        Notes
        -----
        The template is ``lagrangian.ref`` (a plain
        :class:`~slsqp_jax.sqpdax.primal.Primal` for bound-constrained QPs,
        an :class:`~slsqp_jax.sqpdax.primal.InteriorPointPrimal` for the
        barrier subproblem) rather than :meth:`primal_grad` — cheaper (no
        gradient / barrier recompute) and does not assume the gradient is
        materialised.
        """
        template = self.lagrangian.ref
        return type(template).from_flat(flat, *template.sizes)
