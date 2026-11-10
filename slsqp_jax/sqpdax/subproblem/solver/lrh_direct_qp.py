"""Direct QP solver for the limited-memory reduced-Hessian (LRH) model.

:class:`LRHDirectQPSolver` solves the QP of an SQP step

```text
min  gᵀd + ½ dᵀ B d    s.t.  A_E d = b_E,   A_I d ≤ b_I,   lo ≤ d ≤ hi
```

(``lo = lb - x``, ``hi = ub - x``) whose Hessian ``B`` is the model of a
:class:`~slsqp_jax.sqpdax.secant.reduced_hessian.ReducedHessianSecant`. The
general constraints are handled by the direct chain of
:func:`~slsqp_jax.sqpdax.subproblem.solver.lrh_fixed_bounds.solve_fixed_bounds_qp`
(Householder QR, ``m_c × m_c`` factorisations and Lawson-Hanson NNLS; no
Krylov iteration), which solves the QP exactly for a *fixed* set ``A`` of
variables pinned at a bound. This module adds the outer **bound working-set
loop** that chooses ``A``. The ~``2n`` bounds are never stacked into a
constraint matrix: the working set is a pair of boolean masks and every step
of the loop is ``O(n)``.

**The loop is a dual active-set method on the bounds** (Goldfarb-Idnani
style). Let ``θ(μ)`` be the dual function of the bounds alone (``μ ≥ 0`` the
bound multipliers; the general constraints stay in the inner problem). The
QP value is ``max_{μ ≥ 0} θ(μ)``. The oracle on a working set ``A`` returns
the maximiser of ``θ`` over multipliers supported on ``A`` *of any sign*, or,
when the pinned problem is infeasible (``θ`` unbounded on that span), a
Farkas vector that gives an ascent ray. The loop is the primal active-set
method on ``max θ s.t. μ ≥ 0``, whose feasible start ``μ = 0`` is trivial:

* *full step* (the oracle solves with all droppable multipliers
  non-negative): every bound the solution violates is pinned at once
  (``μ = 0`` there). If there is none, the solution is the QP's KKT point;
* *partial dual step* (some droppable multiplier is negative): move ``μ``
  towards the oracle's multipliers until the first one reaches zero, then
  release it;
* *ray step* (the pinned problem is infeasible): move ``μ`` along the
  certificate ray until the first droppable multiplier reaches zero, then
  release it. If no droppable multiplier decreases, the certificate proves
  the QP infeasible.

``θ(μ)`` never decreases, the values at full steps strictly increase, and
between full steps every iteration releases a pin, so the loop is finite in
exact arithmetic. It needs no primal point that satisfies the general
constraints, ``x`` need not lie in the box, and a pinned problem that is
infeasible only means too many pins, never a failure. Pinned variables are
batched (the iteration count does not grow with the number of active
bounds); a safeguard keeps the most violated pin of the last addition through
a degenerate step, which is what prevents batch additions from cycling.
"""

from __future__ import annotations

from typing import cast

import equinox as eqx
import jax
from equinox import tree_at
from jax import numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Int

from ...active_set import ActiveSet
from ...dual import Dual
from ...linalg.lawson_hanson import LDP_RESULTS
from ...primal import Primal
from ...secant.reduced_hessian import ReducedHessianSecant
from ..active_set import ActiveSetSubProblem
from .active_set_loop import ACTIVE_SET_QP_RESULTS, ActiveSetQPSolverState
from .base import KKT_SOLVER_RESULTS, RESULTS, SubProblemSolver
from .lrh_fixed_bounds import solve_fixed_bounds_qp

__all__ = [
    "LRHQPSolverState",
    "LRHDirectQPSolver",
]

# Outcome codes of the bound loop (carry entry, ``int32``).
_RUNNING, _CONVERGED, _INFEASIBLE, _ORACLE_FAILURE, _STALLED = 0, 1, 2, 3, 4


class LRHQPSolverState(ActiveSetQPSolverState):
    """Carry of :class:`LRHDirectQPSolver`.

    The inherited
    :class:`~slsqp_jax.sqpdax.subproblem.solver.active_set_loop.ActiveSetQPSolverState`
    fields keep the meaning the parent minimiser relies on, with these
    solver-specific readings:

    * ``n_iter`` / ``last_n_iter``: bound working-set iterations (each one a
      fixed-bounds solve), cumulative / of the most recent solve;
    * ``n_cg_iter`` / ``last_n_cg_iter``: NNLS iterations (a documented
      re-use of the Krylov-iteration counters), cumulative / of the most
      recent solve;
    * ``qp_result``: ``working_set_converged`` (QP KKT point),
      ``kkt_solver_failure`` (the QP is certified infeasible, or the
      least-distance reduction failed without a certificate),
      ``max_iter_reached`` (budget) or ``anti_cycling`` (a numerical stall of
      the bound loop, see :class:`LRHDirectQPSolver`);
    * ``success``: ``True`` exactly when ``qp_result`` is
      ``working_set_converged``. It is ``False`` on every other exit; the
      minimiser distrusts the multipliers then;
    * ``active_set``: final bound working set (pinned lower / upper
      variables) and ``active_inequalities = λ_I > 0``;
    * ``dual``: the returned QP multipliers (SQP view);
    * ``final_working_tol``: the solver's ``tol``;
    * ``n_anti_cycling``: ``+1`` for every stalled solve;
    * ``last_kkt_feasibility_residual``:
      ``‖A_E d − b_E‖_∞ + ‖max(A_I d − b_I, 0)‖_∞`` at the returned step;
    * ``last_kkt_n_refinements``: always ``0``;
    * ``last_kkt_reason``: the least-distance status of the last fixed-bounds
      solve as a
      :class:`~slsqp_jax.sqpdax.subproblem.solver.base.KKT_SOLVER_RESULTS`
      code (``converged``, ``max_iter_reached``, ``nonfinite``, and
      ``projector_failure`` for an infeasible or inconsistent reduction).
      ``KKT_SOLVER_RESULTS`` has no "infeasible" member (adding one would
      edit the shared base enumeration), so ``projector_failure`` stands for
      "the direct reduction found no solution"; ``last_ldp_status`` carries
      the exact reason.

    Attributes
    ----------
    n_nnls_iter
        Cumulative NNLS iterations over every solve that consumed this carry.
    last_ldp_status
        :class:`~slsqp_jax.sqpdax.linalg.lawson_hanson.LDP_RESULTS` code of
        the last fixed-bounds solve of the most recent solve.
    """

    n_nnls_iter: Int[Array, ""]
    last_ldp_status: LDP_RESULTS

    @classmethod
    def cold(cls, n: int, meq: int, mineq: int, dtype: DTypeLike) -> LRHQPSolverState:
        """Zero-iteration carry sized for ``(n, meq, mineq)``.

        Parameters
        ----------
        n
            Number of decision variables.
        meq
            Number of equality constraints.
        mineq
            Number of general inequality constraints.
        dtype
            Floating dtype of the iterate (fixes the carry dtype).

        Returns
        -------
        LRHQPSolverState
            Counters ``0``, ``success = False``, ``status = successful``,
            ``qp_result = working_set_converged``, empty working set and zero
            multipliers.
        """
        zero = jnp.asarray(0, jnp.int32)
        return cls(
            n_iter=zero,
            success=jnp.asarray(False),
            status=RESULTS.successful,
            n_cg_iter=zero,
            last_n_iter=zero,
            last_n_cg_iter=zero,
            qp_result=ACTIVE_SET_QP_RESULTS.working_set_converged,
            active_set=cast(
                ActiveSet,
                ActiveSet(
                    meq=meq,
                    active_inequalities=jnp.zeros((mineq,), bool),
                    active_lb=jnp.zeros((n,), bool),
                    active_ub=jnp.zeros((n,), bool),
                ),
            ),
            dual=Dual(
                eq_multipliers=jnp.zeros((meq,), dtype),
                ineq_multipliers=jnp.zeros((mineq,), dtype),
                lb_multipliers=jnp.zeros((n,), dtype),
                ub_multipliers=jnp.zeros((n,), dtype),
            ),
            final_working_tol=jnp.asarray(0.0, dtype),
            n_anti_cycling=zero,
            last_kkt_feasibility_residual=jnp.asarray(0.0, dtype),
            last_kkt_n_refinements=zero,
            last_kkt_reason=KKT_SOLVER_RESULTS.converged,
            n_nnls_iter=zero,
            last_ldp_status=LDP_RESULTS.feasible,
        )


class LRHDirectQPSolver(
    SubProblemSolver[Primal, ActiveSetSubProblem, LRHQPSolverState]
):
    """Direct QP solver for an LRH quasi-Newton model with variable bounds.

    Each fixed-bounds QP is solved by
    :func:`~slsqp_jax.sqpdax.subproblem.solver.lrh_fixed_bounds.solve_fixed_bounds_qp`
    (direct linear algebra on ``n × m``, ``m_c × n`` and ``m_c × m_c``
    objects); the bound working set is chosen by the dual active-set loop
    described in the module docstring, with only ``O(n)`` masks and vectors.
    The loop is one ``lax.while_loop`` with a fixed-shape carry. A variable
    with ``lb == ub`` is pinned from the start and never released; a variable
    with an infinite bound is never pinned at it. The cold start pins nothing
    else (not even variables at or beyond a bound): they are pinned by the
    first full step if the step pushes them out. ``x`` need not lie in the
    box.

    With ``τ = max(tol, 100 eps)`` (``eps`` of the working dtype), ``D`` the
    droppable pins (every pin but ``lb == ub``) and ``μ̂`` the oracle's bound
    multipliers, a multiplier is *negative* when
    ``μ̂ < -τ max(1, ‖μ̂_D‖∞)``, a bound is *violated* when
    ``lo - d > τ max(1, |lo|)`` (or the upper analogue), and a ray entry is
    *decreasing* when ``r < -τ ‖r_D‖∞``.

    **Outcomes, returned step and multipliers.** The returned step is
    ``clip(d_full, lo, hi)`` on every path, with ``d_full`` the solution of
    the last full step (zero before the first one): it is always inside the
    box, and on convergence it moves the solution by at most ``τ``.

    | outcome | ``qp_result`` | ``status`` | ``success`` | dual |
    |---|---|---|---|---|
    | converged | ``working_set_converged`` | ``successful`` | ``True`` | last full step |
    | budget exhausted | ``max_iter_reached`` | ``max_steps_reached`` | ``False`` | last full step |
    | stalled (rounding) | ``anti_cycling`` | ``max_steps_reached`` | ``False`` | last full step |
    | QP certified infeasible | ``kkt_solver_failure`` | ``singular`` | ``False`` | zeros |
    | oracle failure, NNLS budget | ``kkt_solver_failure`` | ``max_steps_reached`` | ``False`` | zeros |
    | oracle failure, non-finite / no certificate | ``kkt_solver_failure`` | ``singular`` | ``False`` | zeros |

    A stall (a degenerate step with nothing to release) is impossible in exact
    arithmetic; it is reported instead of looped on, and ``n_anti_cycling``
    counts it. ``success == (qp_result == working_set_converged)`` always.

    **Multipliers of ``lb == ub`` variables.** The oracle's net multiplier
    ``ρ`` is returned split as ``λ_lb = max(ρ, 0)`` and
    ``λ_ub = max(-ρ, 0)``: both faces are active, so complementarity holds,
    and every bound multiplier is non-negative.

    Attributes
    ----------
    solver_state_class
        :class:`LRHQPSolverState`.
    max_iter
        Budget of bound working-set iterations (oracle calls).
    tol
        Tolerance of the bound violation and multiplier sign tests (floored
        at ``100 eps`` of the working dtype, see above).
    nnls_max_iter
        NNLS iteration budget per oracle call (``None``: ``3`` times the
        number of general constraints).
    rank_tol
        Eigenvalue threshold of the free-set factor (``None``: its default).
    eq_tol
        Equality-consistency tolerance of the least-distance reduction
        (``None``: ``√eps``).

    Examples
    --------
    Minimise ``½‖d‖² + gᵀd`` with ``d_0 ≥ -0.5`` from ``x = 0``; the bound
    is active at the solution with multiplier ``0.5``:

    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.active_set import ActiveSet
    >>> from slsqp_jax.sqpdax.dual import Dual
    >>> from slsqp_jax.sqpdax.lagrangian.evaluated import EvaluatedLagrangian
    >>> from slsqp_jax.sqpdax.primal import Primal
    >>> from slsqp_jax.sqpdax.problem.basic import EvaluatedProblem
    >>> from slsqp_jax.sqpdax.secant import LimitedMemoryReducedHessian
    >>> from slsqp_jax.sqpdax.subproblem import ActiveSetSubProblem
    >>> from slsqp_jax.sqpdax.subproblem.solver import (
    ...     LRHDirectQPSolver,
    ...     LRHQPSolverState,
    ... )
    >>> n = 2
    >>> ev = EvaluatedProblem(
    ...     ref=Primal(x=jnp.zeros(n)),
    ...     fn_val=jnp.zeros(()),
    ...     grad_val=jnp.array([1.0, -2.0]),
    ...     fn_qvp=None,
    ...     eq_fn_val=jnp.zeros((0,)),
    ...     eq_fn_jac_val=jnp.zeros((0, n)),
    ...     eq_fn_qvp=None,
    ...     ineq_fn_val=jnp.zeros((0,)),
    ...     ineq_fn_jac_val=jnp.zeros((0, n)),
    ...     ineq_fn_qvp=None,
    ...     lb=jnp.array([-0.5, -jnp.inf]),
    ...     ub=jnp.full(n, jnp.inf),
    ...     null_lb=jnp.array([False, True]),
    ...     null_ub=jnp.array([True, True]),
    ...     aux_val=None,
    ... )
    >>> dual = Dual(jnp.zeros((0,)), jnp.zeros((0,)), jnp.zeros(n), jnp.zeros(n))
    >>> secant = LimitedMemoryReducedHessian(n=n, memory=2)  # empty: B = I
    >>> lag = EvaluatedLagrangian(evaluated=ev, secant=secant, dual=dual)
    >>> active = ActiveSet(
    ...     meq=0,
    ...     active_inequalities=jnp.zeros((0,), bool),
    ...     active_lb=jnp.zeros(n, bool),
    ...     active_ub=jnp.zeros(n, bool),
    ... )
    >>> sub = ActiveSetSubProblem(lag, active)
    >>> state = LRHQPSolverState.cold(n, 0, 0, jnp.zeros(()).dtype)
    >>> (step, qp_dual), new_state = LRHDirectQPSolver().solve(
    ...     sub, (Primal(jnp.zeros(n)), dual), state
    ... )
    >>> [round(float(v), 6) for v in step.x]
    [-0.5, 2.0]
    >>> round(float(qp_dual.lb_multipliers[0]), 6)
    0.5
    >>> bool(new_state.success)
    True
    """

    solver_state_class: type[LRHQPSolverState] = eqx.field(
        static=True, default=LRHQPSolverState
    )
    max_iter: int = eqx.field(static=True, default=100)
    tol: float = eqx.field(static=True, default=1e-8)
    nnls_max_iter: int | None = eqx.field(static=True, default=None)
    rank_tol: float | None = eqx.field(static=True, default=None)
    eq_tol: float | None = eqx.field(static=True, default=None)

    def requires_secant(self) -> bool:
        """``True``: the model ``B`` comes from a reduced-Hessian secant."""
        return True

    def solve(
        self,
        subproblem: ActiveSetSubProblem,
        x0: tuple[Primal, Dual],
        initial_state: LRHQPSolverState,
    ) -> tuple[tuple[Primal, Dual], LRHQPSolverState]:
        """Solve the bound / equality / inequality QP by the dual bound loop.

        Parameters
        ----------
        subproblem
            SQP-view
            :class:`~slsqp_jax.sqpdax.subproblem.active_set.ActiveSetSubProblem`
            whose unmasked Lagrangian carries a
            :class:`~slsqp_jax.sqpdax.secant.reduced_hessian.ReducedHessianSecant`.
            Its own working set is ignored.
        x0
            Ignored (cold start); present for the solver interface.
        initial_state
            Carry whose cumulative counters are advanced by this solve.

        Returns
        -------
        step
            ``(Primal(d), Dual)``: the QP step (inside the box on every path)
            and its multipliers (SQP view; see the class docstring for what
            each outcome returns).
        state
            ``initial_state`` with the counters, ``success`` / ``status``,
            ``qp_result``, working set, multipliers and residual fields
            refreshed (see :class:`LRHQPSolverState`).

        Raises
        ------
        TypeError
            If ``subproblem.is_kkt_dual_increment`` is true (the bound sign
            tests are on ``λ_{k+1}``), or if
            ``subproblem.lagrangian.secant`` is not a
            :class:`~slsqp_jax.sqpdax.secant.reduced_hessian.ReducedHessianSecant`.
        """
        if subproblem.is_kkt_dual_increment:
            raise TypeError(
                "LRHDirectQPSolver requires SQP-view multipliers (the dual block "
                "must be λ_{k+1}, not an increment): the bound sign tests are "
                "tests on the QP multipliers."
            )
        lag = subproblem.lagrangian
        secant = lag.secant
        if not isinstance(secant, ReducedHessianSecant):
            raise TypeError(
                "LRHDirectQPSolver requires the Lagrangian's secant to be a "
                f"ReducedHessianSecant; got {type(secant).__name__}."
            )
        factors = secant.factors()
        ev = lag.evaluated
        n = lag.n
        dtype = lag.x_ref.dtype
        inf = jnp.asarray(jnp.inf, dtype)
        tau = jnp.asarray(max(self.tol, 100.0 * float(jnp.finfo(dtype).eps)), dtype)
        iota = jnp.arange(n)

        # Box of the step, ``lo ≤ d ≤ hi`` (±inf on null bounds); x may be outside.
        lo, hi = subproblem.primal_box()
        lo, hi = lo.astype(dtype), hi.astype(dtype)
        fixed = ~lag.null_lb & ~lag.null_ub & (lag.lb == lag.ub)
        zeros_n = jnp.zeros((n,), dtype)
        false_n = jnp.zeros((n,), bool)

        def vmax(a):
            return jnp.max(jnp.abs(a), initial=0.0)

        def clip_mu(mu_lb, mu_ub, a_lb, a_ub):
            """Zero off the working set; non-negative off the ``lb == ub`` pins."""
            mu_lb = jnp.where(
                a_lb, jnp.where(fixed, mu_lb, jnp.maximum(mu_lb, 0.0)), 0.0
            )
            mu_ub = jnp.where(a_ub, jnp.maximum(mu_ub, 0.0), 0.0)
            return mu_lb, mu_ub

        dual_zero = Dual(
            eq_multipliers=jnp.zeros((lag.meq,), dtype),
            ineq_multipliers=jnp.zeros((lag.mineq,), dtype),
            lb_multipliers=zeros_n,
            ub_multipliers=zeros_n,
        )
        zero_i = jnp.asarray(0, jnp.int32)
        # Carry: working set (A_lb, A_ub), multipliers (mu_lb, mu_ub), pins added
        # by the last full step whose multiplier is still 0 (fresh_*), the most
        # violated one of them (p, p_ub; p = -1: none), the last full step and
        # its dual, the last oracle status, NNLS count, iteration count, outcome.
        init = (
            fixed,
            false_n,
            zeros_n,
            zeros_n,
            false_n,
            false_n,
            jnp.asarray(-1, jnp.int32),
            jnp.asarray(False),
            zeros_n,
            dual_zero,
            LDP_RESULTS.feasible,
            zero_i,
            zero_i,
            jnp.asarray(_RUNNING, jnp.int32),
        )

        def cond_fn(carry):
            return (carry[-1] == _RUNNING) & (carry[-2] < self.max_iter)

        # Equivalent mutants (a mutation run need not chase them): (1) the
        # partial step with ``dir = mu_hat`` instead of ``mu_hat - mu`` (every
        # drop decision is invariant to a positive rescaling of ``mu``);
        # (2) ``fresh`` not cleared after a positive step (``p = -1`` then, so
        # the safeguard never reads it before the next batch resets it);
        # (3) ``mu`` not clipped / zeroed off the working set (stale entries
        # are never read, and a full step resets ``mu``).
        def body_fn(carry):
            (
                a_lb,
                a_ub,
                mu_lb,
                mu_ub,
                fresh_lb,
                fresh_ub,
                p,
                p_ub,
                d_full,
                dual_full,
                _status,
                n_nnls,
                n_it,
                _outcome,
            ) = carry
            res = solve_fixed_bounds_qp(
                lag,
                factors,
                a_lb,
                a_ub,
                rank_tol=self.rank_tol,
                eq_tol=self.eq_tol,
                nnls_max_iter=self.nnls_max_iter,
            )
            ok = res.ldp_status == LDP_RESULTS.feasible
            ray = ~ok & res.has_ray
            d_hat = res.step.astype(dtype)
            drop_lb = a_lb & ~fixed  # droppable pins
            drop_ub = a_ub

            # (F) the oracle solved P(A): negative droppable multipliers?
            mh_lb = res.dual.lb_multipliers.astype(dtype)
            mh_ub = res.dual.ub_multipliers.astype(dtype)
            mu_scale = jnp.maximum(
                1.0,
                jnp.maximum(
                    vmax(jnp.where(drop_lb, mh_lb, 0.0)),
                    vmax(jnp.where(drop_ub, mh_ub, 0.0)),
                ),
            )
            neg_f_lb = drop_lb & (mh_lb < -tau * mu_scale)
            neg_f_ub = drop_ub & (mh_ub < -tau * mu_scale)
            full = ok & ~jnp.any(neg_f_lb | neg_f_ub)
            partial = ok & ~full

            # Full step: violated bounds of d_hat (batch add).
            free = ~a_lb & ~a_ub
            viol_lb = jnp.where(free & jnp.isfinite(lo), lo - d_hat, -inf)
            viol_ub = jnp.where(free & jnp.isfinite(hi), d_hat - hi, -inf)
            v_lb = viol_lb > tau * jnp.maximum(1.0, jnp.abs(lo))
            v_ub = (viol_ub > tau * jnp.maximum(1.0, jnp.abs(hi))) & ~v_lb
            any_v = jnp.any(v_lb | v_ub)
            converged = full & ~any_v
            batch = full & any_v
            max_lb = jnp.max(jnp.where(v_lb, viol_lb, -inf), initial=-jnp.inf)
            max_ub = jnp.max(jnp.where(v_ub, viol_ub, -inf), initial=-jnp.inf)
            p_ub_new = max_ub > max_lb
            p_new = jnp.where(
                p_ub_new,
                jnp.argmax(jnp.where(v_ub, viol_ub, -inf)),
                jnp.argmax(jnp.where(v_lb, viol_lb, -inf)),
            ).astype(jnp.int32)

            # (R) the pinned problem is infeasible: ray of the certificate.
            r_lb = res.ray_lb.astype(dtype)
            r_ub = res.ray_ub.astype(dtype)
            r_scale = jnp.maximum(
                vmax(jnp.where(drop_lb, r_lb, 0.0)), vmax(jnp.where(drop_ub, r_ub, 0.0))
            )
            neg_r_lb = drop_lb & (r_lb < -tau * r_scale)
            neg_r_ub = drop_ub & (r_ub < -tau * r_scale)
            infeasible = ray & ~jnp.any(neg_r_lb | neg_r_ub)
            oracle_failure = ~ok & ~res.has_ray
            ray_step = ray & ~infeasible

            # Dual step (partial towards mu_hat, or along the ray).
            step = partial | ray_step
            dir_lb = jnp.where(ray, r_lb, mh_lb - mu_lb)
            dir_ub = jnp.where(ray, r_ub, mh_ub - mu_ub)
            neg_lb = jnp.where(ray, neg_r_lb, neg_f_lb) & step
            neg_ub = jnp.where(ray, neg_r_ub, neg_f_ub) & step
            safe_lb = jnp.where(neg_lb, -dir_lb, 1.0)
            safe_ub = jnp.where(neg_ub, -dir_ub, 1.0)
            ratio_lb = jnp.where(neg_lb, mu_lb / safe_lb, inf)
            ratio_ub = jnp.where(neg_ub, mu_ub / safe_ub, inf)
            t = jnp.minimum(
                jnp.min(ratio_lb, initial=jnp.inf), jnp.min(ratio_ub, initial=jnp.inf)
            )
            blk_lb = neg_lb & (ratio_lb <= t * (1.0 + tau))
            blk_ub = neg_ub & (ratio_ub <= t * (1.0 + tau))
            positive = t > 0.0

            # Degenerate step (t = 0): keep the most violated pin of the last
            # addition; release the other fresh pins instead.
            p_safe = jnp.maximum(p, 0)
            p_in_blk = (p >= 0) & jnp.where(p_ub, blk_ub[p_safe], blk_lb[p_safe])
            is_p = iota == p
            deg_lb = fresh_lb & ~(is_p & ~p_ub)
            deg_ub = fresh_ub & ~(is_p & p_ub)
            use_deg = ~positive & p_in_blk
            blk_lb = jnp.where(use_deg, deg_lb, blk_lb)
            blk_ub = jnp.where(use_deg, deg_ub, blk_ub)
            stalled = step & ~positive & ~jnp.any(blk_lb | blk_ub)
            release = step & ~stalled

            # Next working set and multipliers.
            a_lb_next = jnp.where(
                batch, a_lb | v_lb, jnp.where(release, a_lb & ~blk_lb, a_lb)
            )
            a_ub_next = jnp.where(
                batch, a_ub | v_ub, jnp.where(release, a_ub & ~blk_ub, a_ub)
            )
            mu_lb_full = jnp.where(a_lb, mh_lb, 0.0)
            mu_ub_full = jnp.where(a_ub, mh_ub, 0.0)
            move = release & positive
            mu_lb_step = jnp.where(a_lb, mu_lb + jnp.where(move, t, 0.0) * dir_lb, 0.0)
            mu_ub_step = jnp.where(a_ub, mu_ub + jnp.where(move, t, 0.0) * dir_ub, 0.0)
            mu_lb_new = jnp.where(full, mu_lb_full, jnp.where(step, mu_lb_step, mu_lb))
            mu_ub_new = jnp.where(full, mu_ub_full, jnp.where(step, mu_ub_step, mu_ub))
            mu_lb_next, mu_ub_next = clip_mu(mu_lb_new, mu_ub_new, a_lb_next, a_ub_next)

            fresh_lb_next = (
                jnp.where(batch, v_lb, jnp.where(move, False, fresh_lb)) & a_lb_next
            )
            fresh_ub_next = (
                jnp.where(batch, v_ub, jnp.where(move, False, fresh_ub)) & a_ub_next
            )
            p_next = jnp.where(batch, p_new, jnp.where(move, -1, p)).astype(jnp.int32)
            p_ub_next = jnp.where(batch, p_ub_new, p_ub)

            d_full_next = jnp.where(full, d_hat, d_full)
            dual_full_next = jax.tree.map(
                lambda new, old: jnp.where(full, new.astype(dtype), old),
                res.dual,
                dual_full,
            )
            outcome = jnp.where(
                converged,
                _CONVERGED,
                jnp.where(
                    infeasible,
                    _INFEASIBLE,
                    jnp.where(
                        oracle_failure,
                        _ORACLE_FAILURE,
                        jnp.where(stalled, _STALLED, _RUNNING),
                    ),
                ),
            ).astype(jnp.int32)
            return (
                a_lb_next,
                a_ub_next,
                mu_lb_next,
                mu_ub_next,
                fresh_lb_next,
                fresh_ub_next,
                p_next,
                p_ub_next,
                d_full_next,
                dual_full_next,
                res.ldp_status,
                n_nnls + res.n_nnls_iter.astype(jnp.int32),
                n_it + 1,
                outcome,
            )

        (
            a_lb_f,
            a_ub_f,
            _mu_lb_f,
            _mu_ub_f,
            _fresh_lb_f,
            _fresh_ub_f,
            _p_f,
            _p_ub_f,
            d_full_f,
            dual_full_f,
            ldp_f,
            n_nnls_f,
            n_it_f,
            outcome_f,
        ) = jax.lax.while_loop(cond_fn, body_fn, init)
        dual_full_f = cast(Dual, dual_full_f)

        converged = outcome_f == _CONVERGED
        stalled = outcome_f == _STALLED
        infeasible = outcome_f == _INFEASIBLE
        failed = infeasible | (outcome_f == _ORACLE_FAILURE)
        budget = outcome_f == _RUNNING
        qp_result = ACTIVE_SET_QP_RESULTS.where(
            converged,
            ACTIVE_SET_QP_RESULTS.working_set_converged,
            ACTIVE_SET_QP_RESULTS.where(
                stalled,
                ACTIVE_SET_QP_RESULTS.anti_cycling,
                ACTIVE_SET_QP_RESULTS.where(
                    budget,
                    ACTIVE_SET_QP_RESULTS.max_iter_reached,
                    ACTIVE_SET_QP_RESULTS.kkt_solver_failure,
                ),
            ),
        )
        status = RESULTS.where(
            converged,
            RESULTS.successful,
            RESULTS.where(
                failed & ~(ldp_f == LDP_RESULTS.max_iter_reached) & ~budget & ~stalled,
                RESULTS.singular,
                RESULTS.max_steps_reached,
            ),
        )
        kkt_reason = KKT_SOLVER_RESULTS.where(
            ldp_f == LDP_RESULTS.feasible,
            KKT_SOLVER_RESULTS.converged,
            KKT_SOLVER_RESULTS.where(
                ldp_f == LDP_RESULTS.max_iter_reached,
                KKT_SOLVER_RESULTS.max_iter_reached,
                KKT_SOLVER_RESULTS.where(
                    ldp_f == LDP_RESULTS.nonfinite,
                    KKT_SOLVER_RESULTS.nonfinite,
                    KKT_SOLVER_RESULTS.projector_failure,
                ),
            ),
        )

        # Returned step and dual. Split the net multiplier of lb == ub pins; on
        # convergence the bound multipliers may be negative by at most the
        # sign tolerance, so clip them at 0 (every bound multiplier is then
        # non-negative, as the minimiser assumes).
        step_f = jnp.clip(d_full_f, lo, hi)
        rho = dual_full_f.lb_multipliers
        mu_ub_full = dual_full_f.ub_multipliers
        dual_out = Dual(
            eq_multipliers=dual_full_f.eq_multipliers,
            ineq_multipliers=dual_full_f.ineq_multipliers,
            lb_multipliers=jnp.where(fixed | converged, jnp.maximum(rho, 0.0), rho),
            ub_multipliers=jnp.where(
                fixed,
                jnp.maximum(-rho, 0.0),
                jnp.where(converged, jnp.maximum(mu_ub_full, 0.0), mu_ub_full),
            ),
        )
        dual_out = jax.tree.map(
            lambda a: jnp.where(failed, jnp.zeros_like(a), a), dual_out
        )

        # Residual of the general constraints at the returned step.
        resid_eq = ev.eq_fn_jac_val.astype(dtype) @ step_f + ev.eq_fn_val.astype(dtype)
        resid_ineq = ev.ineq_fn_jac_val.astype(dtype) @ step_f + ev.ineq_fn_val.astype(
            dtype
        )
        residual = jnp.max(jnp.abs(resid_eq), initial=0.0) + jnp.max(
            jnp.maximum(resid_ineq, 0.0), initial=0.0
        )

        active_set = cast(
            ActiveSet,
            ActiveSet(
                meq=lag.meq,
                active_inequalities=dual_out.ineq_multipliers > 0.0,
                active_lb=a_lb_f,
                active_ub=a_ub_f,
            ),
        )
        n_it_f = jnp.asarray(n_it_f, jnp.int32)
        n_nnls_f = jnp.asarray(n_nnls_f, jnp.int32)
        new_state: LRHQPSolverState = tree_at(
            lambda s: (
                s.n_iter,
                s.n_cg_iter,
                s.last_n_iter,
                s.last_n_cg_iter,
                s.success,
                s.status,
                s.qp_result,
                s.active_set,
                s.dual,
                s.final_working_tol,
                s.n_anti_cycling,
                s.last_kkt_feasibility_residual,
                s.last_kkt_n_refinements,
                s.last_kkt_reason,
                s.n_nnls_iter,
                s.last_ldp_status,
            ),
            initial_state,
            (
                jnp.asarray(initial_state.n_iter, jnp.int32) + n_it_f,
                jnp.asarray(initial_state.n_cg_iter, jnp.int32) + n_nnls_f,
                n_it_f,
                n_nnls_f,
                converged,
                status,
                qp_result,
                active_set,
                dual_out,
                jnp.asarray(self.tol, dtype),
                jnp.asarray(initial_state.n_anti_cycling, jnp.int32)
                + stalled.astype(jnp.int32),
                residual.astype(dtype),
                jnp.asarray(0, jnp.int32),
                kkt_reason,
                jnp.asarray(initial_state.n_nnls_iter, jnp.int32) + n_nnls_f,
                ldp_f,
            ),
        )
        step = cast(Primal, Primal(x=step_f))
        return (step, dual_out), new_state
