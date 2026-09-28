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
from collections.abc import Callable
from typing import cast

import jax
from equinox import Module, field, tree_at
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
    "CraigProjectionContext",
    "CraigProjector",
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

    def preconditioned_feasibility_correction(
        self, d: Vector_n, b: Float[Array, " m"]
    ) -> Vector_n:
        """Pull ``d`` back onto ``A_work d = b_eff`` along the ``M``-metric range space.

        The correction ``δd`` is the minimum-``M``-norm change satisfying
        ``A_work (d − δd) = b_eff``. Equals :meth:`feasibility_correction`
        when no preconditioner is attached.

        Parameters
        ----------
        d
            Candidate primal step (full length ``n``).
        b
            Full right-hand side ``b`` of ``A d = b``.

        Returns
        -------
        Vector_n
            ``d − M⁻¹ A_workᵀ (A_work M⁻¹ A_workᵀ)⁺ (A_work d − b_eff)``.
        """
        r = self.A_work @ d - self.effective_rhs(b)
        return d - self.apply_Minv(self.A_work.T @ self.solve_preconditioned_normal(r))


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


# ---------------------------------------------------------------------------
# CRAIG (Golub–Kahan bidiagonalisation) backend
# ---------------------------------------------------------------------------


def _craig_solve(
    A: Float[Array, "m n"],
    rhs: Float[Array, " m"],
    *,
    rtol: float,
    atol: float,
    max_iter: int,
    breakdown_tol: float,
) -> tuple[Float[Array, " n"], Bool[Array, ""], Int[Array, ""]]:
    """Minimum-norm solution of ``A x = rhs`` by CRAIG (Paige & Saunders 1982).

    Golub–Kahan bidiagonalisation needs only ``A v`` / ``Aᵀ u`` products and
    never forms ``A Aᵀ``. Convergence is ``‖A x − rhs‖ < max(atol, rtol ‖rhs‖)``;
    a bidiagonal coefficient below ``breakdown_tol`` signals rank deficiency
    (or a right-hand side outside ``range(A)``) and stops the recurrence at
    the last safe iterate.

    Parameters
    ----------
    A
        Matrix ``(m, n)``.
    rhs
        Right-hand side of length ``m``.
    rtol, atol
        Relative / absolute residual tolerances.
    max_iter
        Total bidiagonalisation budget (``>= 1``; the first step is always
        taken).
    breakdown_tol
        Absolute floor on ``α_k`` / ``β_k``.

    Returns
    -------
    x
        Best iterate (min-norm solution when converged; zero when ``rhs``
        is orthogonal to ``range(A)``).
    converged
        ``True`` iff the residual met the tolerance without breakdown.
    n_iter
        Bidiagonalisation steps taken.
    """
    dtype = rhs.dtype
    tiny = jnp.asarray(1e-30, dtype)
    beta1 = jnp.linalg.norm(rhs)
    beta1_safe = jnp.maximum(beta1, tiny)
    threshold = jnp.maximum(jnp.asarray(atol, dtype), rtol * beta1)
    u1 = rhs / beta1_safe

    Atu1 = A.T @ u1
    alpha1 = jnp.linalg.norm(Atu1)
    # The bidiagonal coefficients are ``O(‖A‖)``; a coefficient at roundoff
    # level relative to that scale is a breakdown whatever the absolute
    # ``breakdown_tol`` says (which matters in float32).
    a_scale = jnp.linalg.norm(A, ord="fro")
    breakdown_floor = jnp.maximum(
        jnp.asarray(breakdown_tol, dtype), 10 * jnp.finfo(dtype).eps * a_scale
    )
    alpha_breakdown1 = alpha1 < breakdown_floor
    v1 = Atu1 / jnp.maximum(alpha1, tiny)
    s1 = beta1 / jnp.maximum(alpha1, tiny)
    # ``α₁ ≈ 0`` with ``β₁ ≠ 0`` means ``rhs ⟂ range(A)``: the least-squares
    # solution is zero.
    x1 = jnp.where(alpha_breakdown1, jnp.zeros_like(v1), s1 * v1)

    u_hat = A @ v1 - alpha1 * u1
    beta2 = jnp.linalg.norm(u_hat)
    u2 = u_hat / jnp.maximum(beta2, tiny)

    trivial = beta1 < threshold  # zero right-hand side: x = 0 is exact
    residual1 = jnp.abs(beta2 * s1)
    converged1 = trivial | (residual1 < threshold)
    breakdown1 = alpha_breakdown1 & ~trivial

    init = (
        x1,
        s1,
        u2,
        v1,
        beta2,
        residual1,
        converged1 | breakdown1,
        breakdown1,
        jnp.asarray(1, jnp.int32),
    )

    def body(_, carry):
        x, s, u, v, beta, residual, done, breakdown, k = carry

        def step(c):
            x, s, u, v, beta, _residual, _done, breakdown, k = c
            v_hat = A.T @ u - beta * v
            v_hat = v_hat - jnp.dot(v, v_hat) * v  # local reorthogonalisation
            alpha_new = jnp.linalg.norm(v_hat)
            alpha_bad = alpha_new < breakdown_floor
            v_new = v_hat / jnp.maximum(alpha_new, tiny)
            s_new = -beta * s / jnp.maximum(alpha_new, tiny)
            x_new = jnp.where(alpha_bad, x, x + s_new * v_new)

            u_hat = A @ v_new - alpha_new * u
            u_hat = u_hat - jnp.dot(u, u_hat) * u
            beta_new = jnp.linalg.norm(u_hat)
            beta_bad = beta_new < breakdown_floor
            u_new = u_hat / jnp.maximum(beta_new, tiny)

            residual_new = jnp.abs(beta_new * s_new)
            conv = residual_new < threshold
            broke = alpha_bad | beta_bad
            return (
                x_new,
                s_new,
                u_new,
                v_new,
                beta_new,
                residual_new,
                conv | broke,
                breakdown | (broke & ~conv),
                k + 1,
            )

        return jax.lax.cond(jnp.reshape(done, ()), lambda c: c, step, carry)

    # The initialisation above is bidiagonalisation step 1.
    x, _, _, _, _, _, _, breakdown, n_iter = jax.lax.fori_loop(
        0, max(max_iter - 1, 0), body, init
    )
    # Judge convergence on the true residual: the recurrence estimate
    # ``|β_{k+1} s_k|`` also vanishes when the Krylov space is exhausted on an
    # inconsistent system (rank-deficient ``A``), which is not a solution.
    residual = jnp.linalg.norm(A @ x - rhs)
    converged = (residual < threshold) & ~breakdown
    return x, converged, n_iter


def _cg_spd_solve(
    apply: Callable[[Float[Array, " m"]], Float[Array, " m"]],
    rhs: Float[Array, " m"],
    *,
    rtol: float,
    atol: float,
    max_iter: int,
) -> Float[Array, " m"]:
    """Plain CG for an SPD operator, stopping at ``‖r‖ < max(atol, rtol ‖rhs‖)``.

    Parameters
    ----------
    apply
        SPD operator ``v ↦ N v``.
    rhs
        Right-hand side.
    rtol, atol
        Residual tolerances.
    max_iter
        Iteration budget.

    Returns
    -------
    jax.Array
        Approximate solution of ``N x = rhs`` (zero when ``rhs = 0``).
    """
    dtype = rhs.dtype
    tol_sq = jnp.square(
        jnp.maximum(jnp.asarray(atol, dtype), rtol * jnp.linalg.norm(rhs))
    )
    r0 = rhs
    rr0 = jnp.dot(r0, r0)
    init = (jnp.zeros_like(rhs), r0, r0, rr0, jnp.reshape(rr0 < tol_sq, ()))

    eps = jnp.finfo(dtype).eps

    def body(_, carry):
        def step(c):
            x, r, p, rr, _ = c
            Np = apply(p)
            pNp = jnp.dot(p, Np)
            # A singular (PSD) operator with a right-hand side partly outside
            # its range makes ``pᵀNp → 0`` once the range is exhausted; freeze
            # on the last iterate instead of taking the spurious huge step.
            freeze = pNp <= eps * jnp.dot(p, p)
            alpha = rr / jnp.maximum(pNp, jnp.asarray(1e-30, dtype))
            x_new = x + alpha * p
            r_new = r - alpha * Np
            rr_new = jnp.dot(r_new, r_new)
            beta = rr_new / jnp.maximum(rr, jnp.asarray(1e-30, dtype))
            return (
                jnp.where(freeze, x, x_new),
                jnp.where(freeze, r, r_new),
                jnp.where(freeze, p, r_new + beta * p),
                jnp.where(freeze, rr, rr_new),
                freeze | (rr_new < tol_sq),
            )

        return jax.lax.cond(carry[-1], lambda c: c, step, carry)

    x, _, _, _, _ = jax.lax.fori_loop(0, max_iter, body, init)
    return x


class CraigProjectionContext(ProjectionContext):
    """Matrix-free :class:`ProjectionContext` (Golub–Kahan CRAIG + CG).

    Without a preconditioner, :meth:`project` and :meth:`particular_solution`
    run CRAIG on ``A_work`` (min-norm solves, no ``A Aᵀ`` formed). The
    range-space solves :meth:`solve_normal` /
    :meth:`solve_preconditioned_normal` run CG on ``A_work M⁻¹ A_workᵀ``
    regularised to the identity on inactive rows so the operator stays SPD
    at fixed shape; with a preconditioner the inherited ``M``-metric
    projection formulas use those solves.

    Non-finite values (e.g. a breakdown on a rank-deficient working set
    that CRAIG could not step over) are propagated, never replaced by an
    identity projection.

    Attributes
    ----------
    rtol, atol
        CRAIG residual tolerances.
    max_iter
        CRAIG bidiagonalisation budget per solve.
    breakdown_tol
        Absolute floor on the bidiagonal coefficients.
    normal_rtol, normal_atol, normal_max_iter
        CG tolerances / budget for the range-space solves.
    """

    rtol: float = field(static=True)
    atol: float = field(static=True)
    max_iter: int = field(static=True)
    breakdown_tol: float = field(static=True)
    normal_rtol: float = field(static=True)
    normal_atol: float = field(static=True)
    normal_max_iter: int = field(static=True)

    def _craig(self, rhs: Float[Array, " m"]) -> Float[Array, " n"]:
        x, _, _ = _craig_solve(
            self.A_work,
            rhs,
            rtol=self.rtol,
            atol=self.atol,
            max_iter=self.max_iter,
            breakdown_tol=self.breakdown_tol,
        )
        return x

    def _normal_operator(
        self, preconditioned: bool
    ) -> Callable[[Float[Array, " m"]], Float[Array, " m"]]:
        A_work = self.A_work
        reg = jnp.where(self.active_rows, 0.0, 1.0).astype(A_work.dtype)
        if preconditioned:

            def apply(v):
                return A_work @ self.apply_Minv(A_work.T @ v) + reg * v

        else:

            def apply(v):
                return A_work @ (A_work.T @ v) + reg * v

        return apply

    def _cg_normal(self, rhs: Float[Array, " m"], preconditioned: bool):
        sol = _cg_spd_solve(
            self._normal_operator(preconditioned),
            rhs,
            rtol=self.normal_rtol,
            atol=self.normal_atol,
            max_iter=self.normal_max_iter,
        )
        return jnp.where(self.active_rows, sol, 0.0)

    def solve_normal(self, rhs: Float[Array, " m"]) -> Float[Array, " m"]:
        """CG on ``A_work A_workᵀ`` (identity on inactive rows)."""
        return self._cg_normal(rhs, preconditioned=False)

    def solve_preconditioned_normal(
        self, rhs: Float[Array, " m"]
    ) -> Float[Array, " m"]:
        """CG on ``A_work M⁻¹ A_workᵀ`` (falls back to :meth:`solve_normal`)."""
        if not self.is_preconditioned:
            return self.solve_normal(rhs)
        return self._cg_normal(rhs, preconditioned=True)

    def project(self, v: Vector_n) -> Vector_n:
        """CRAIG null-space projection ``v − A_workᵀ (A_work A_workᵀ)⁺ A_work v``.

        Falls back to the inherited ``M``-metric formula when preconditioned.
        """
        if self.is_preconditioned:
            return super().project(v)
        v_work = self.free_f * v
        return v_work - self._craig(self.A_work @ v_work)

    def particular_solution(self, b: Float[Array, " m"]) -> Vector_n:
        """Min-norm ``A_work d = b_eff`` by CRAIG, plus the fixed step."""
        if self.is_preconditioned:
            return super().particular_solution(b)
        return self._craig(self.effective_rhs(b)) + self.d_fixed


class CraigProjector(Projector):
    """Iterative projector: CRAIG on ``A_work`` and CG on the normal equations.

    Suited to large ``n`` where the thin SVD of :class:`SVDProjector` is too
    expensive; every operation is a sequence of ``A_work`` / ``A_workᵀ``
    products. :meth:`build` runs the particular-solution CRAIG once and
    reports its convergence flag / iteration count on the context (and so
    on the inner :class:`~slsqp_jax.sqpdax.subproblem.solver.base.KKTSolverState`).

    Attributes
    ----------
    rtol, atol
        CRAIG residual tolerances ``‖A x − b‖ < max(atol, rtol ‖b‖)``. The
        absolute floor keeps near-KKT iterates (``‖b‖ ~ eps``) from chasing
        a relative target below machine precision.
    max_iter
        CRAIG bidiagonalisation budget.
    breakdown_tol
        Absolute floor on the bidiagonal coefficients.
    normal_rtol, normal_atol, normal_max_iter
        CG tolerances / budget for the range-space solves.
    """

    rtol: float = field(static=True, default=1e-10)
    atol: float = field(static=True, default=1e-12)
    max_iter: int = field(static=True, default=200)
    breakdown_tol: float = field(static=True, default=1e-14)
    normal_rtol: float = field(static=True, default=1e-12)
    normal_atol: float = field(static=True, default=1e-12)
    normal_max_iter: int = field(static=True, default=200)

    def build(
        self, subproblem: SubProblem, preconditioner: Preconditioner | None = None
    ) -> CraigProjectionContext:
        """See :meth:`Projector.build`."""
        A, A_work, active_rows, free_mask, d_fixed = self.working_geometry(subproblem)
        ctx = cast(
            CraigProjectionContext,
            CraigProjectionContext(
                A=A,
                A_work=A_work,
                active_rows=active_rows,
                free_mask=free_mask,
                d_fixed=d_fixed,
                preconditioner=preconditioner,
                converged=jnp.asarray(True),
                n_iter=jnp.asarray(0, jnp.int32),
                rtol=self.rtol,
                atol=self.atol,
                max_iter=self.max_iter,
                breakdown_tol=self.breakdown_tol,
                normal_rtol=self.normal_rtol,
                normal_atol=self.normal_atol,
                normal_max_iter=self.normal_max_iter,
            ),
        )
        # Probe the particular solution on the subproblem's own right-hand
        # side so a breakdown on this working set is reported to the caller.
        _, dual_rhs = subproblem.kkt_rhs()
        b = jnp.concatenate([dual_rhs.eq_multipliers, dual_rhs.ineq_multipliers])
        _, converged, n_iter = _craig_solve(
            A_work,
            ctx.effective_rhs(b),
            rtol=self.rtol,
            atol=self.atol,
            max_iter=self.max_iter,
            breakdown_tol=self.breakdown_tol,
        )
        return cast(
            CraigProjectionContext,
            tree_at(lambda c: (c.converged, c.n_iter), ctx, (converged, n_iter)),
        )
