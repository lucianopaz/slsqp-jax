"""Null-space projectors for the working constraints of a KKT subproblem.

A :class:`ProjectionContext` is the reusable, data-only bundle that every
null-space (projected) KKT solver needs for one working set: the masked
constraint matrix, the free-subspace geometry inherited from the active
bounds, and the two range-space solves that define the projector, the
particular solution and the multiplier recovery. A :class:`Projector` is the
static configuration that knows how to *build* one from a
:class:`~slsqp_jax.sqpdax.subproblem.base.SubProblem`.

Notation follows Nocedal & Wright (2006) §16.3: the working constraints
``A_work d = b`` are the equality rows plus the active inequality rows of
:meth:`~slsqp_jax.sqpdax.subproblem.base.SubProblem.nonbound_constraint_jac`,
restricted to the free columns given by
:meth:`~slsqp_jax.sqpdax.subproblem.base.SubProblem.free_subspace`; variables
fixed by an active bound are eliminated rather than kept as rows.
"""

from abc import abstractmethod
from typing import cast

import jax
from equinox import Module, field
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from ...preconditioner import IdentityPreconditioner, Preconditioner
from ...types import InitializableModule, Vector_n
from ..base import SubProblem

__all__ = [
    "ProjectionContext",
    "Projector",
    "SVDProjectionContext",
    "SVDProjector",
]


class ProjectionContext(Module):
    """Projector, particular solution and range-space solves for one working set.

    Concrete backends only implement the two range-space solves
    (:meth:`solve_normal` and :meth:`solve_preconditioned_normal`); the
    projector, the particular solution and the feasibility correction are
    derived here in the Nocedal & Wright forms

    ```
    P v   = M⁻¹ v − M⁻¹ Aᵀ (A M⁻¹ Aᵀ)⁺ A M⁻¹ v        (eq. 16.33; eq. 16.30 for M = I)
    d_p   = M⁻¹ Aᵀ (A M⁻¹ Aᵀ)⁺ b + d_fixed            (§16.3 particular solution)
    ```

    with ``A = A_work``. ``M`` is the optional SPD reduced-Hessian
    preconditioner; with ``M = I`` the projector is orthogonal.

    Attributes
    ----------
    A
        Row-masked constraint matrix ``(m, n)``: inactive rows are zero, all
        columns are present (needed to shift the right-hand side by the
        fixed variables).
    A_work
        ``A`` with the fixed columns zeroed: the matrix the projector acts on.
    active_rows
        Boolean row mask (equalities and active inequalities).
    free_mask
        Boolean column mask of the free variables.
    d_fixed
        Prescribed step on the fixed variables (zero elsewhere).
    preconditioner
        Optional ``M`` supplying ``M⁻¹`` through ``invert``; ``None`` is the
        identity.
    converged
        Whether the backend's own linear solves met their tolerance while
        building the context (always ``True`` for direct backends).
    n_iter
        Iterations the backend spent building the context (zero for direct
        backends).
    """

    A: Float[Array, "m n"]
    A_work: Float[Array, "m n"]
    active_rows: Bool[Array, " m"]
    free_mask: Bool[Array, " n"]
    d_fixed: Float[Array, " n"]
    preconditioner: Preconditioner | None
    converged: Bool[Array, ""]
    n_iter: Int[Array, ""]

    @property
    def free_f(self) -> Float[Array, " n"]:
        """Free-variable mask as a float multiplier."""
        return self.free_mask.astype(self.A_work.dtype)

    @property
    def is_preconditioned(self) -> bool:
        """Whether a non-identity ``M`` is attached."""
        pre = self.preconditioner
        return pre is not None and not isinstance(pre, IdentityPreconditioner)

    @abstractmethod
    def solve_normal(self, rhs: Float[Array, " m"]) -> Float[Array, " m"]:
        """Apply ``(A_work A_workᵀ)⁺`` (2-norm range-space solve).

        Parameters
        ----------
        rhs
            Right-hand side of length ``m``.

        Returns
        -------
        jax.Array
            Pseudo-inverse solution; zero on inactive rows.
        """

    @abstractmethod
    def solve_preconditioned_normal(
        self, rhs: Float[Array, " m"]
    ) -> Float[Array, " m"]:
        """Apply ``(A_work M⁻¹ A_workᵀ)⁺`` (``M``-metric range-space solve).

        Equals :meth:`solve_normal` when no preconditioner is attached.

        Parameters
        ----------
        rhs
            Right-hand side of length ``m``.

        Returns
        -------
        jax.Array
            Pseudo-inverse solution; zero on inactive rows.
        """

    def apply_Minv(self, v: Vector_n) -> Vector_n:
        """``M⁻¹`` restricted to the free subspace (identity when ``M = I``).

        Parameters
        ----------
        v
            Decision-variable vector.

        Returns
        -------
        Vector_n
            ``free ⊙ M⁻¹ (free ⊙ v)``.
        """
        free = self.free_f
        if not self.is_preconditioned:
            return free * v
        pre = self.preconditioner
        assert pre is not None
        return free * pre.invert(free * v)

    def project(self, v: Vector_n) -> Vector_n:
        """Project ``v`` onto ``null(A_work)`` (``M``-orthogonally when preconditioned).

        Parameters
        ----------
        v
            Decision-variable vector.

        Returns
        -------
        Vector_n
            ``P v`` (Nocedal & Wright eq. 16.33; eq. 16.30 for ``M = I``).
        """
        mv = self.apply_Minv(v)
        return mv - self.apply_Minv(
            self.A_work.T @ self.solve_preconditioned_normal(self.A_work @ mv)
        )

    def effective_rhs(self, b: Float[Array, " m"]) -> Float[Array, " m"]:
        """Constraint right-hand side after eliminating the fixed variables.

        Parameters
        ----------
        b
            Full right-hand side ``b`` of ``A d = b``.

        Returns
        -------
        jax.Array
            ``b − A d_fixed`` on active rows, zero elsewhere.
        """
        return jnp.where(self.active_rows, b - self.A @ self.d_fixed, 0.0)

    def particular_solution(self, b: Float[Array, " m"]) -> Vector_n:
        """Minimum-``M``-norm solution of ``A_work d = b_eff`` plus the fixed block.

        Parameters
        ----------
        b
            Full right-hand side ``b`` of ``A d = b``.

        Returns
        -------
        Vector_n
            ``M⁻¹ Aᵀ (A M⁻¹ Aᵀ)⁺ b_eff + d_fixed`` (Nocedal & Wright §16.3).
        """
        b_eff = self.effective_rhs(b)
        return (
            self.apply_Minv(self.A_work.T @ self.solve_preconditioned_normal(b_eff))
            + self.d_fixed
        )

    def feasibility_residual(
        self, d: Vector_n, b: Float[Array, " m"]
    ) -> Float[Array, ""]:
        """Euclidean norm of ``A_work d − b_eff`` over the active rows.

        Parameters
        ----------
        d
            Candidate primal step (full length ``n``).
        b
            Full right-hand side ``b`` of ``A d = b``.

        Returns
        -------
        jax.Array
            Scalar residual norm.
        """
        r = jnp.where(self.active_rows, self.A_work @ d - self.effective_rhs(b), 0.0)
        return jnp.linalg.norm(r)

    def feasibility_correction(self, d: Vector_n, b: Float[Array, " m"]) -> Vector_n:
        """Pull ``d`` back onto ``A_work d = b_eff`` with one 2-norm range-space solve.

        Parameters
        ----------
        d
            Candidate primal step (full length ``n``).
        b
            Full right-hand side ``b`` of ``A d = b``.

        Returns
        -------
        Vector_n
            ``d − A_workᵀ (A_work A_workᵀ)⁺ (A_work d − b_eff)``.
        """
        r = self.A_work @ d - self.effective_rhs(b)
        return d - self.A_work.T @ self.solve_normal(r)


class SVDProjectionContext(ProjectionContext):
    """Direct :class:`ProjectionContext` from a thin SVD of ``A_work``.

    With ``A_work = U diag(s) Vᵀ`` the 2-norm range-space solve is
    ``(A_work A_workᵀ)⁺ = U diag(pinv(s²)) Uᵀ``; singular values below
    ``rcond · max(s)`` are dropped (rank-revealing, so LICQ violations and
    the zero rows of inactive inequalities fall out on their own). The
    preconditioned solve uses an explicit pseudo-inverse of the small
    ``A_work M⁻¹ A_workᵀ`` matrix when a preconditioner is attached.

    Attributes
    ----------
    U
        Left singular vectors ``(m, k)``.
    inv_s2
        ``1 / s²`` on the retained singular values, zero on the dropped ones.
    AMAt_pinv
        ``(A_work M⁻¹ A_workᵀ)⁺`` when preconditioned, else ``None``.
    """

    U: Float[Array, "m k"]
    inv_s2: Float[Array, " k"]
    AMAt_pinv: Float[Array, "m m"] | None

    def solve_normal(self, rhs: Float[Array, " m"]) -> Float[Array, " m"]:
        """``U diag(1/s²) Uᵀ rhs``."""
        return self.U @ (self.inv_s2 * (self.U.T @ rhs))

    def solve_preconditioned_normal(
        self, rhs: Float[Array, " m"]
    ) -> Float[Array, " m"]:
        """``(A_work M⁻¹ A_workᵀ)⁺ rhs`` (falls back to :meth:`solve_normal`)."""
        if self.AMAt_pinv is None:
            return self.solve_normal(rhs)
        return self.AMAt_pinv @ rhs


class Projector(InitializableModule):
    """Static configuration that builds a :class:`ProjectionContext`.

    Subclasses choose the linear-algebra backend (direct SVD, iterative
    CRAIG, …); the geometry is always read off the subproblem through
    :meth:`~slsqp_jax.sqpdax.subproblem.base.SubProblem.nonbound_constraint_jac`,
    :meth:`~slsqp_jax.sqpdax.subproblem.base.SubProblem.active_constraint_rows`
    and :meth:`~slsqp_jax.sqpdax.subproblem.base.SubProblem.free_subspace`.
    """

    @staticmethod
    def working_geometry(
        subproblem: SubProblem,
    ) -> tuple[
        Float[Array, "m n"],
        Float[Array, "m n"],
        Bool[Array, " m"],
        Bool[Array, " n"],
        Float[Array, " n"],
    ]:
        """Read ``(A, A_work, active_rows, free_mask, d_fixed)`` off a subproblem.

        Parameters
        ----------
        subproblem
            KKT model exposing the general-constraint Jacobian and the
            free-subspace geometry.

        Returns
        -------
        tuple
            Row-masked Jacobian, its column-masked copy, the active-row mask,
            the free-column mask and the fixed step values.
        """
        A = subproblem.nonbound_constraint_jac()
        active_rows = subproblem.active_constraint_rows()
        free_mask, d_fixed = subproblem.free_subspace()
        A = jnp.where(active_rows[:, None], A, 0.0)
        A_work = A * free_mask.astype(A.dtype)[None, :]
        return A, A_work, active_rows, free_mask, d_fixed

    @abstractmethod
    def build(
        self, subproblem: SubProblem, preconditioner: Preconditioner | None = None
    ) -> ProjectionContext:
        """Build the projection context for ``subproblem``'s working set.

        Parameters
        ----------
        subproblem
            KKT model at the current iterate.
        preconditioner
            Optional SPD reduced-Hessian preconditioner ``M``.

        Returns
        -------
        ProjectionContext
            Backend-specific context.
        """


class SVDProjector(Projector):
    """Direct projector: thin SVD of ``A_work`` (Nocedal & Wright §16.3, §16.8).

    Attributes
    ----------
    rcond
        Relative singular-value floor for the pseudo-inverse rank cut.
        ``None`` uses ``eps * max(A_work.shape)`` (numpy ``pinv`` convention).
    """

    rcond: float | None = field(static=True, default=None)

    def build(
        self, subproblem: SubProblem, preconditioner: Preconditioner | None = None
    ) -> SVDProjectionContext:
        """See :meth:`Projector.build`."""
        A, A_work, active_rows, free_mask, d_fixed = self.working_geometry(subproblem)
        dtype = A_work.dtype
        U, s, _ = jnp.linalg.svd(A_work, full_matrices=False)
        eps = jnp.finfo(dtype).eps
        rcond = self.rcond if self.rcond is not None else eps * max(A_work.shape)
        # ``initial=0`` keeps ``m == 0`` (bound-only QPs) well-defined.
        keep = s > rcond * jnp.max(s, initial=jnp.asarray(0.0, dtype))
        # Guard the reciprocal so the dropped singular values never form a
        # ``1/0`` intermediate that could poison later AD.
        inv_s2 = jnp.where(keep, 1.0 / jnp.square(jnp.where(keep, s, 1.0)), 0.0)

        AMAt_pinv = None
        pre = preconditioner
        if pre is not None and not isinstance(pre, IdentityPreconditioner):
            free_f = free_mask.astype(dtype)

            def apply_Minv(v: Vector_n) -> Vector_n:
                return free_f * pre.invert(free_f * v)

            # ``A M⁻¹ Aᵀ`` by applying ``M⁻¹`` to each row of ``A_work``. Zero
            # rows (inactive inequalities) stay zero and fall under the same
            # rank cut as the plain path.
            AMi = jax.vmap(apply_Minv)(A_work)
            AMAt_pinv = jnp.linalg.pinv(A_work @ AMi.T, rcond=rcond)

        return cast(
            SVDProjectionContext,
            SVDProjectionContext(
                A=A,
                A_work=A_work,
                active_rows=active_rows,
                free_mask=free_mask,
                d_fixed=d_fixed,
                preconditioner=preconditioner,
                converged=jnp.asarray(True),
                n_iter=jnp.asarray(0, jnp.int32),
                U=U,
                inv_s2=inv_s2,
                AMAt_pinv=AMAt_pinv,
            ),
        )
