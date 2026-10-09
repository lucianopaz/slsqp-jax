"""Scaled primal-dual interior-point trust-region subproblem."""

from collections.abc import Callable
from typing import Self, cast

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float

from ..dual import Dual
from ..lagrangian import InteriorPointEvaluatedLagrangian
from ..linalg.scaled_normal_equations import (
    ResolvedNormalEquationsStrategy,
    SchurNormalEquations,
    slack_eliminated_normal_solver,
)
from ..primal import InteriorPointPrimal, Slack
from ..types import Scalar
from .base import SubProblem

__all__ = [
    "ScaledBarrierSubProblem",
]


class ScaledBarrierSubProblem(SubProblem[InteriorPointPrimal]):
    """Scaled interior-point trust-region subproblem as a :class:`SubProblem`.

    Wraps a primal-dual
    :class:`~slsqp_jax.sqpdax.lagrangian.evaluated.InteriorPointEvaluatedLagrangian`
    and exposes its Newton-KKT saddle system in scaled slack coordinates.

    Attributes
    ----------
    lagrangian
        Primal-dual interior-point Lagrangian at the reference point.
    tau
        Fraction-to-boundary parameter for :meth:`primal_box`.
    schur_cache
        Optional precomputed
        :class:`~slsqp_jax.sqpdax.linalg.scaled_normal_equations.SchurNormalEquations`
        of the unmasked operator ``Â``, attached through
        :meth:`with_schur_normal_equations` so that every solver working on
        this subproblem instance shares one factorisation.

    Notes
    -----
    The scaled slack is ``p̃_s = S⁻¹ p_s`` (Nocedal & Wright eq. 19.32), so
    the trust region is a Euclidean ball. Incoming steps carry the slack
    block in this ball scale; operators convert to the original slack scale
    (``* S``), apply the Lagrangian blocks, and scale the slack rows back
    (``* S``) so the saddle system stays symmetric. ``τ`` (default ``0.995``)
    defines the fraction-to-boundary faces of the primal box (Nocedal &
    Wright eqs. 19.33e / 19.34c).
    """

    lagrangian: InteriorPointEvaluatedLagrangian
    tau: float = 0.995
    schur_cache: SchurNormalEquations | None = None

    def __init__(
        self,
        lagrangian: InteriorPointEvaluatedLagrangian,
        tau: float = 0.995,
    ):
        """Attach a primal-dual interior-point Lagrangian.

        Parameters
        ----------
        lagrangian
            Cached IP Lagrangian; must have ``primal_dual=True``.
        tau
            Fraction-to-boundary parameter for :meth:`primal_box`.

        Raises
        ------
        ValueError
            If ``lagrangian.primal_dual`` is false.
        """
        if not lagrangian.primal_dual:
            raise ValueError("SubProblem must be used with a primal-dual lagrangian")
        self.lagrangian = lagrangian
        self.tau = tau
        self.schur_cache = None

    # ------------------------------------------------------------------
    # matrix-free scaled operators
    # ------------------------------------------------------------------
    def _zero_dual(self) -> Dual:
        return cast(Dual, jax.tree.map(jnp.zeros_like, self.lagrangian.dual))

    def _zero_primal(self) -> InteriorPointPrimal:
        return cast(
            InteriorPointPrimal, jax.tree.map(jnp.zeros_like, self.lagrangian.ref)
        )

    def hess_mvp(self, w: InteriorPointPrimal) -> InteriorPointPrimal:
        """Scaled Hessian product ``Ĥ w = P G P w``.

        Parameters
        ----------
        w
            Scaled primal step.

        Returns
        -------
        InteriorPointPrimal
            ``Ĥ w`` in the same layout as ``w``.
        """
        return self.kkt_mvp_primal((w, self._zero_dual()))

    def jac_mvp(self, w: InteriorPointPrimal) -> Dual:
        """Scaled Jacobian product ``Â w = J(x, s) P w``.

        Parameters
        ----------
        w
            Scaled primal step.

        Returns
        -------
        Dual
            Linearised constraint change, rows ordered as :meth:`dual_grad`.
        """
        return self.kkt_mvp_lower_offdiag((w, self._zero_dual()))

    def jac_t_mvp(self, y: Dual) -> InteriorPointPrimal:
        """Scaled transposed Jacobian product ``Âᵀ y = P J(x, s)ᵀ y``.

        Parameters
        ----------
        y
            Dual vector (multipliers or constraint residual).

        Returns
        -------
        InteriorPointPrimal
            ``Âᵀ y`` in scaled primal layout.
        """
        return self.kkt_mvp_upper_offdiag((self._zero_primal(), y))

    # ------------------------------------------------------------------
    # structured solves of the scaled normal equations Â Âᵀ
    # ------------------------------------------------------------------
    def schur_normal_equations(
        self,
        *,
        free_x: Bool[Array, " n"] | None = None,
        live_lb: Bool[Array, " n"] | None = None,
        live_ub: Bool[Array, " n"] | None = None,
        rcond: float | None = None,
    ) -> SchurNormalEquations:
        """Explicit pseudo-inverse Schur complement of ``Â Âᵀ``.

        Parameters
        ----------
        free_x
            Optional mask of free ``x`` columns (frozen columns are removed
            from ``Â``); ``None`` keeps every column.
        live_lb, live_ub
            Optional masks of bound rows whose slack column is live;
            ``None`` uses the non-null bounds.
        rcond
            Relative eigenvalue cutoff; ``None`` uses the factor default.

        Returns
        -------
        SchurNormalEquations
            The cached :attr:`schur_cache` when no mask is given and the
            cutoff matches, otherwise a freshly built factor.
        """
        unmasked = free_x is None and live_lb is None and live_ub is None
        cache = self.schur_cache
        if unmasked and cache is not None and (rcond is None or cache.rcond == rcond):
            return cache
        lag = self.lagrangian
        return SchurNormalEquations.build(
            lag.eq_fn_jac_val,
            lag.ineq_fn_jac_val,
            lag.slack.s,
            lag.slack.s_lb,
            lag.slack.s_ub,
            live_lb=~lag.null_lb if live_lb is None else live_lb,
            live_ub=~lag.null_ub if live_ub is None else live_ub,
            free_x=free_x,
            rcond=rcond,
        )

    def with_schur_normal_equations(self, *, rcond: float | None = None) -> Self:
        """Return a copy carrying the unmasked Schur factor in :attr:`schur_cache`.

        Parameters
        ----------
        rcond
            Relative eigenvalue cutoff forwarded to
            :meth:`~slsqp_jax.sqpdax.linalg.scaled_normal_equations.SchurNormalEquations.build`.
        """
        factor = self.schur_normal_equations(rcond=rcond)
        return cast(
            Self,
            eqx.tree_at(
                lambda s: s.schur_cache, self, factor, is_leaf=lambda z: z is None
            ),
        )

    def normal_equations_solver(
        self,
        strategy: ResolvedNormalEquationsStrategy,
        *,
        free_x: Bool[Array, " n"] | None = None,
        live_lb: Bool[Array, " n"] | None = None,
        live_ub: Bool[Array, " n"] | None = None,
        rcond: float | None = None,
        tol: float = 1e-10,
        max_iter: int = 100,
    ) -> Callable[[Float[Array, " m"]], Float[Array, " m"]]:
        """Structured solve ``b ↦ (Â Âᵀ)⁺ b`` on the flat dual ``[eq | ineq | lb | ub]``.

        This is the only place that knows how the row / column layout of
        ``Â`` maps onto the Lagrangian's Jacobians, slacks and null masks.

        Parameters
        ----------
        strategy
            ``"schur"`` for the explicit pseudo-inverse Schur complement,
            ``"matrix-free"`` for the nested PCG elimination.
        free_x, live_lb, live_ub
            Column / row masks, see :meth:`schur_normal_equations`.
        rcond
            Eigenvalue cutoff of the ``"schur"`` path.
        tol, max_iter
            Krylov controls of the ``"matrix-free"`` path.

        Returns
        -------
        Callable
            ``solve(rhs) -> y`` returning the minimum-norm solution.

        Raises
        ------
        ValueError
            If ``strategy`` is ``"generic"`` (the caller owns that path) or
            unknown.
        """
        if strategy == "schur":
            return self.schur_normal_equations(
                free_x=free_x, live_lb=live_lb, live_ub=live_ub, rcond=rcond
            ).solve
        if strategy != "matrix-free":
            raise ValueError(
                "normal_equations_solver handles 'schur' and 'matrix-free'; got "
                f"{strategy!r}"
            )
        lag = self.lagrangian
        jac_eq, jac_ineq = lag.eq_fn_jac_val, lag.ineq_fn_jac_val
        solve = slack_eliminated_normal_solver(
            lambda v: jac_eq @ v,
            lambda y: jac_eq.T @ y,
            lambda v: jac_ineq @ v,
            lambda y: jac_ineq.T @ y,
            lag.slack.s,
            lag.slack.s_lb,
            lag.slack.s_ub,
            live_lb=~lag.null_lb if live_lb is None else live_lb,
            live_ub=~lag.null_ub if live_ub is None else live_ub,
            free_x=free_x,
            meq=lag.meq,
            tol=tol,
            max_iter=max_iter,
        )
        # ``solve(rhs, with_info=True)`` remains available to callers that
        # want the iteration count; the default call returns the vector.
        return cast(Callable[[Float[Array, " m"]], Float[Array, " m"]], solve)

    def primal_box(self) -> tuple[Float[Array, " n_p"], Float[Array, " n_p"]]:
        """Box for the scaled primal step ``[x | s̃ | s̃_lb | s̃_ub]``.

        Returns
        -------
        lower, upper
            Flat bound vectors matching
            :meth:`~slsqp_jax.sqpdax.primal.InteriorPointPrimal.flatten`.

        Notes
        -----
        ``x`` is unbounded (bounds are equality-constrained via slacks). Each
        slack coordinate has a fraction-to-boundary lower face ``-τ`` and no
        upper bound; null bound-slacks are left free (``±inf``).
        """
        lag = self.lagrangian
        n, mineq = lag.n, lag.mineq
        dtype = lag.ref.x.dtype
        inf = jnp.asarray(jnp.inf, dtype)
        lo_x = jnp.full((n,), -inf, dtype)
        hi_x = jnp.full((n,), inf, dtype)
        lo_s = jnp.full((mineq,), -self.tau, dtype)
        hi_s = jnp.full((mineq,), inf, dtype)
        lo_lb = jnp.where(lag.null_lb, -inf, -self.tau)
        hi_lb = jnp.full((n,), inf, dtype)
        lo_ub = jnp.where(lag.null_ub, -inf, -self.tau)
        hi_ub = jnp.full((n,), inf, dtype)
        lo = jnp.concatenate([lo_x, lo_s, lo_lb, lo_ub])
        hi = jnp.concatenate([hi_x, hi_s, hi_lb, hi_ub])
        return lo, hi

    def active_bounds(
        self, primal: InteriorPointPrimal, tol: Scalar | float = 0.0
    ) -> tuple[Bool[Array, " n_p"], Bool[Array, " n_p"]]:
        """Bound faces active when the corresponding bound-slack sits on its FTB face.

        Parameters
        ----------
        primal
            Scaled primal step in ``[x | s̃ | s̃_lb | s̃_ub]`` layout.
        tol
            Absolute tolerance for declaring an FTB face active.

        Returns
        -------
        active_lb, active_ub
            Boolean masks of length ``n`` for lower- and upper-bound faces.

        Notes
        -----
        Only the lower/upper *bound* slack coordinates are inspected; general
        inequality slacks and the ``x`` block do not contribute to these masks.
        """
        lag = self.lagrangian
        n, mineq = lag.n, lag.mineq
        flat = primal.flatten()
        lo, _hi = self.primal_box()
        s_lb = flat[n + mineq : n + mineq + n]
        s_ub = flat[n + mineq + n :]
        lo_lb = lo[n + mineq : n + mineq + n]
        lo_ub = lo[n + mineq + n :]
        active_lb = (~lag.null_lb) & (s_lb <= lo_lb + tol)
        active_ub = (~lag.null_ub) & (s_ub <= lo_ub + tol)
        return active_lb, active_ub

    @property
    def is_kkt_dual_regularized(self) -> bool:
        """Whether the dual-dual KKT block is active on the Lagrangian.

        Notes
        -----
        Delegates to the Lagrangian so the base :meth:`kkt_operator` folds in
        the dual-dual regularization block whenever it is active.
        """
        return self.lagrangian.is_kkt_dual_regularized

    def primal_grad(self) -> InteriorPointPrimal:
        """Objective gradient of the scaled barrier QP.

        Notes
        -----
        This is Nocedal & Wright eq. 19.33a, *not* the full Lagrangian
        gradient: multiplier terms enter through the QP dual block, so the
        step's dual is the full multiplier estimate. The ``x``-block is
        ``∇f``; the slack block is the barrier gradient scaled by ``S``
        (chain rule for ``p_s = S p̃_s``), i.e. ``-μ e`` for the log barrier.
        Null bound slacks are zeroed by :meth:`_slack_to_orig_scale`.
        """
        return cast(
            InteriorPointPrimal,
            InteriorPointPrimal(
                x=self.lagrangian.grad_val,
                slack=self._slack_to_orig_scale(self.lagrangian.barrier.grad_val),
            ),
        )

    def dual_grad(self) -> Dual:
        """Dual residual of the interior-point Lagrangian."""
        return self.lagrangian.dual_grad

    def kkt_rhs(self) -> tuple[InteriorPointPrimal, Dual]:
        """Right-hand side ``(-g, -c)`` with the dual regularisation centred at ``λ_k``.

        Returns
        -------
        tuple of InteriorPointPrimal and Dual
            Negated primal gradient and negated dual gradient; when
            :attr:`is_kkt_dual_regularized` is set the equality rows carry
            the extra shift ``-δ λ_k``.

        Notes
        -----
        The step's dual block is the *full* multiplier estimate, so with the
        dual-dual block ``-δ I`` alone the regularised row would read
        ``Â p - δ λ = -c`` and shrink ``λ`` toward zero. Shifting the
        right-hand side by ``-δ λ_k`` (``λ_k`` the multipliers the Lagrangian
        was evaluated at) turns it into ``Â p - δ (λ - λ_k) = -c``, the
        proximal / stabilised-SQP form (Ipopt's ``δ_c`` acts on the
        increment in the same way). :meth:`dual_grad` is left untouched so
        the normal-step solvers still see the plain residual ``c``.
        """
        primal_rhs, dual_rhs = super().kkt_rhs()
        if not self.is_kkt_dual_regularized:
            return primal_rhs, dual_rhs
        lag = self.lagrangian
        shift = lag.dual_kkt_regularization * lag.dual.eq_multipliers
        return primal_rhs, eqx.tree_at(
            lambda d: d.eq_multipliers,
            dual_rhs,
            dual_rhs.eq_multipliers - shift,
        )

    def _to_ball_scale(
        self, step: tuple[InteriorPointPrimal, Dual]
    ) -> tuple[InteriorPointPrimal, Dual]:
        """Map original-scale slacks in ``step`` to ball coordinates ``S⁻¹ p_s``."""
        return eqx.tree_at(
            lambda p: p[0].slack,
            step,
            self._slack_to_ball_scale(step[0].slack),
        )

    def _to_orig_scale(
        self, step: tuple[InteriorPointPrimal, Dual]
    ) -> tuple[InteriorPointPrimal, Dual]:
        """Map ball-scale slacks in ``step`` to original coordinates ``S p̃_s``."""
        return eqx.tree_at(
            lambda p: p[0].slack,
            step,
            self._slack_to_orig_scale(step[0].slack),
        )

    def _bound_scale(self):
        """Slack scale ``S`` with null bounds neutralised.

        Notes
        -----
        General-inequality slacks are always real. Null lower/upper bounds
        carry no slack, so their scale is replaced by ``1`` (avoiding ``0/0``
        or ``inf`` under AD) and :meth:`_slack_to_ball_scale` /
        :meth:`_slack_to_orig_scale` mask those entries back to ``0``.
        """
        S = self.lagrangian.slack
        null_lb, null_ub = self.lagrangian.null_lb, self.lagrangian.null_ub
        safe_lb = jnp.where(null_lb, 1.0, S.s_lb)
        safe_ub = jnp.where(null_ub, 1.0, S.s_ub)
        return S.s, safe_lb, safe_ub, null_lb, null_ub

    def _slack_to_ball_scale(self, slack: Slack) -> Slack:
        """Map slack tangents to ball scale ``p̃_s = S⁻¹ p_s``.

        Notes
        -----
        Nocedal & Wright eq. 19.32. Null bound entries stay exactly ``0``.
        """
        s_scale, lb_scale, ub_scale, null_lb, null_ub = self._bound_scale()
        return cast(
            Slack,
            Slack(
                s=slack.s / s_scale,
                s_lb=jnp.where(null_lb, 0.0, slack.s_lb / lb_scale),
                s_ub=jnp.where(null_ub, 0.0, slack.s_ub / ub_scale),
            ),
        )

    def _slack_to_orig_scale(self, slack: Slack) -> Slack:
        """Map slack tangents to original scale ``p_s = S p̃_s``.

        Notes
        -----
        Null bound entries contribute nothing (masked to ``0``).
        """
        s_scale, lb_scale, ub_scale, null_lb, null_ub = self._bound_scale()
        return cast(
            Slack,
            Slack(
                s=slack.s * s_scale,
                s_lb=jnp.where(null_lb, 0.0, slack.s_lb * lb_scale),
                s_ub=jnp.where(null_ub, 0.0, slack.s_ub * ub_scale),
            ),
        )

    def kkt_mvp_primal(
        self, step: tuple[InteriorPointPrimal, Dual]
    ) -> InteriorPointPrimal:
        """Primal-primal KKT product in scaled slack coordinates.

        Notes
        -----
        Unscales the slack tangent, applies the Lagrangian ``x``/slack
        Hessian blocks, then rescales the slack row by ``S``.
        """
        scaled_tangent = self._to_orig_scale(step)
        kkt_x = jax.tree.map(
            jnp.add,
            self.lagrangian.kkt_mvp_x_x(scaled_tangent),
            self.lagrangian.kkt_mvp_x_slack_upper_offdiag(scaled_tangent),
        )
        kkt_slack = self._slack_to_orig_scale(
            jax.tree.map(
                jnp.add,
                self.lagrangian.kkt_mvp_slack_x_lower_offdiag(scaled_tangent),
                self.lagrangian.kkt_mvp_slack_slack(scaled_tangent),
            )
        )
        return cast(
            InteriorPointPrimal,
            InteriorPointPrimal(
                x=kkt_x,
                slack=kkt_slack,
            ),
        )

    def kkt_mvp_upper_offdiag(
        self, step: tuple[InteriorPointPrimal, Dual]
    ) -> InteriorPointPrimal:
        """Primal-dual upper off-diagonal ``Aᵀ δλ`` with scaled slack row.

        Notes
        -----
        The dual block is not scaled, so the Lagrangian operator is applied
        directly and only the slack row is multiplied by ``S`` (the slack
        equation was multiplied by ``S`` in the change of variables).
        """
        primal = self.lagrangian.kkt_mvp_upper_offdiag(step)
        return cast(
            InteriorPointPrimal,
            InteriorPointPrimal(
                x=primal.x,
                slack=self._slack_to_orig_scale(primal.slack),
            ),
        )

    def kkt_mvp_lower_offdiag(self, step: tuple[InteriorPointPrimal, Dual]) -> Dual:
        """Dual-primal lower off-diagonal after unscaling the slack tangent."""
        return self.lagrangian.kkt_mvp_lower_offdiag(self._to_orig_scale(step))

    def kkt_mvp_dual(self, step: tuple[InteriorPointPrimal, Dual]) -> Dual:
        """Dual-dual KKT regularization (``-reg`` on equality rows).

        Notes
        -----
        Acts on the unscaled dual, so no slack rescaling is needed. Calls
        :meth:`~slsqp_jax.sqpdax.lagrangian.evaluated.InteriorPointEvaluatedLagrangian.kkt_mvp_dual_dual`
        because the inherited ``kkt_mvp_dual`` returns zeros and would drop
        the term.
        """
        return self.lagrangian.kkt_mvp_dual_dual(step)

    def residual(
        self, step: tuple[InteriorPointPrimal, Dual]
    ) -> tuple[InteriorPointPrimal, Dual]:
        """KKT residual ``K z - rhs`` of the scaled saddle system.

        Notes
        -----
        The primal block is the (scaled) Lagrangian stationarity; the dual
        block is the linearized infeasibility. :meth:`kkt_operator` already
        folds in dual-dual regularization when
        :attr:`is_kkt_dual_regularized` is set.
        """
        return jax.tree.map(
            jnp.subtract,
            self.kkt_operator(step),
            self.kkt_rhs(),
        )

    def nonbound_constraint_jac(self) -> Float[Array, " meq+mineq n"]:
        """``x``-only Jacobian of the general (equality + inequality) constraints.

        Returns
        -------
        jax.Array
            Matrix of shape ``(meq + mineq, n)``.

        Notes
        -----
        Satisfies the abstract :class:`SubProblem` contract. The
        interior-point normal step couples ``x`` and slacks, so a
        trust-region normal-step solver that needs the augmented
        ``[A_x | S]`` operator must build it separately.
        """
        return self.lagrangian.nonbound_constraint_jac
