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

**Elastic mode** (Kraft's device for inconsistent linearisations; on by
default). When the loop above ends with the QP *certified infeasible* or with
an oracle failure that has no certificate, the whole loop is re-run on the
relaxed QP ``A_E d + δ b_E = b_E``, ``A_I,j d + δ b_I,j ≤ b_I,j`` (rows with
``b_I,j < 0``), ``0 ≤ δ ≤ 1``, objective ``+ ½ ρ_e δ²``
(:func:`~slsqp_jax.sqpdax.subproblem.solver.lrh_fixed_bounds.solve_fixed_bounds_qp`
with ``elastic_rho``). The relaxation is defined on the *unshifted* ``b``, so
it is one fixed QP for every bound working set, and
``ρ_e = elastic_weight · τ²`` is computed once from the unpinned data
(:func:`~slsqp_jax.sqpdax.subproblem.solver.lrh_fixed_bounds.elastic_penalty`)
and frozen. An infeasible pinned elastic LDP is a ray step like any other. A
QP the plain loop solves never enters elastic mode: the result is then
bit-identical to ``elastic=False``.
"""

from __future__ import annotations

from typing import cast

import equinox as eqx
import jax
from equinox import tree_at
from jax import numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Bool, Int

from ...active_set import ActiveSet
from ...dual import Dual
from ...linalg.lawson_hanson import LDP_RESULTS
from ...primal import Primal
from ...secant.reduced_hessian import ReducedHessianSecant
from ...types import Scalar
from ..active_set import ActiveSetSubProblem
from .active_set_loop import ACTIVE_SET_QP_RESULTS, ActiveSetQPSolverState
from .base import KKT_SOLVER_RESULTS, RESULTS, SubProblemSolver
from .lrh_fixed_bounds import elastic_penalty, solve_fixed_bounds_qp

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

    After an **elastic solve** (see :class:`LRHDirectQPSolver`) the same
    fields describe the elastic bound loop: counters are cumulative over the
    plain and the elastic loop, and ``success`` is true exactly when the
    elastic loop converged, i.e. the multipliers are the KKT multipliers of the
    *relaxed* QP (those of the general rows and bounds; the two rows of ``δ``
    never appear). ``last_kkt_feasibility_residual`` measures the *unrelaxed*
    constraints at the step, so it is about ``δ ‖b‖`` there.

    Attributes
    ----------
    n_nnls_iter
        Cumulative NNLS iterations over every solve that consumed this carry.
    last_ldp_status
        :class:`~slsqp_jax.sqpdax.linalg.lawson_hanson.LDP_RESULTS` code of
        the last fixed-bounds solve of the most recent solve.
    last_elastic_delta
        The relaxation ``δ ∈ [0, 1]`` of the most recent solve: ``0`` when
        elastic mode was not used, and also when it was used but the solve did
        not return a step (the failure exits, like the dual).
    n_elastic
        Cumulative number of solves that entered elastic mode (``int32``),
        whatever their outcome.
    """

    n_nnls_iter: Int[Array, ""]
    last_ldp_status: LDP_RESULTS
    last_elastic_delta: Scalar
    n_elastic: Int[Array, ""]

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
            last_elastic_delta=jnp.asarray(0.0, dtype),
            n_elastic=zero,
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

    With ``elastic=True`` the rows "QP certified infeasible" and "oracle
    failure" are the exits of the plain loop, which hands over to the elastic
    loop (see below); the row finally reported is the one the elastic loop
    takes. The other rows are unchanged.

    **Elastic mode.** With ``elastic=True`` (default) and at least one general
    constraint, a plain loop that ends *certified infeasible* or with an
    *oracle failure* (the two failure rows above) is followed by the same
    loop on the elastic QP of the module docstring (same start, warm trial
    included; its own budget of ``max_iter`` oracle calls; the counters are
    cumulative, ``n_elastic`` is incremented). The returned step is then the
    step of the relaxed QP, inside the box, and ``last_elastic_delta`` is
    ``δ``. The relaxed QP is always feasible when ``x`` lies in the box, so the
    elastic loop converges in the generic case, and ``δ ≤ δ_min + O(1 /
    elastic_weight)`` with ``δ_min`` the smallest feasible relaxation (the
    default weight ``1e3`` gives ``δ_min ≤ δ ≤ δ_min + 2 / elastic_weight`` on
    the SQP-like data of the tests). The step is a descent direction of the L1
    merit whenever ``δ < 1`` and the merit penalty exceeds ``‖λ‖∞``.

    **What ``success`` means after an elastic solve.** The exit table applies
    to the elastic loop, so a converged elastic solve reports
    ``qp_result = working_set_converged``, ``status = successful`` and
    ``success = True``: the multipliers are a KKT estimate of the *relaxed*
    QP (finite, ``λ_I ≥ 0`` and bound multipliers ``≥ 0``), which is what the
    merit-penalty update needs (its descent bound uses ``‖λ‖∞``, plan §2.5).
    The relaxed step does *not* satisfy the linearised constraints when
    ``δ > 0``: consumers that read ``success`` as "the linearisation is
    consistent" must also read ``last_elastic_delta`` (or ``n_elastic``). If
    the elastic loop itself fails (budget, stall, certified infeasible, oracle
    failure; e.g. ``x`` outside the box with no relaxed feasible point),
    ``success`` is ``False`` and the exit is that loop's row of the table.

    A stall (a degenerate step with nothing to release) is impossible in exact
    arithmetic; it is reported instead of looped on, and ``n_anti_cycling``
    counts it. ``success == (qp_result == working_set_converged)`` always.

    **Multipliers of ``lb == ub`` variables.** The oracle's net multiplier
    ``ρ`` is returned split as ``λ_lb = max(ρ, 0)`` and
    ``λ_ub = max(-ρ, 0)``: both faces are active, so complementarity holds,
    and every bound multiplier is non-negative.

    **Logging.** The solver uses the inherited ``logger`` (minimisers inject
    their ``minimiser.subproblem`` child through ``init``). Per solve it emits
    an ``INFO`` line on convergence, a ``WARNING`` line on every other exit, a
    further ``WARNING`` on a stall, and a ``"qp"`` diagnostic record with the
    keys of the active-set solver's record plus ``n_nnls_iter``,
    ``last_ldp_status``, ``warm_trial_accepted`` / ``warm_trial_rejected``,
    ``n_ray_steps``, ``n_safeguard``, ``stalled``, ``n_pinned``, ``elastic``
    (elastic mode was entered in this solve), ``last_elastic_delta`` and
    ``n_elastic``. An ``INFO`` line reports an elastic solve. Logging never
    changes the returned values.

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
    elastic
        Enter elastic mode on a certified-infeasible or uncertified-failure
        exit of the plain loop (default ``True``). ``False`` reproduces the
        plain behaviour bit-identically (infeasible QPs are failures).
    elastic_weight
        Dimensionless weight of the elastic penalty, ``ρ_e = elastic_weight ·
        τ²`` (default ``1e3``; see
        :func:`~slsqp_jax.sqpdax.linalg.lsei.elastic_weight_scale`). A larger
        weight gives ``δ`` closer to ``δ_min`` at the price of accuracy
        (``∝ √elastic_weight``).
    warm_start
        Seed the bound working set from ``initial_state.active_set`` (default
        ``False``: cold start, the state's working set is ignored). The warm
        set is the *bound* part of that working set (``active_lb`` /
        ``active_ub``); the general-constraint part is ignored, since the
        oracle chooses it. Infinite bounds are never pinned, the lower face
        wins when both faces are marked, and ``lb == ub`` variables are always
        pinned. The first iteration is a *trial*: one oracle call on
        ``fixed ∪ warm set`` with ``μ = 0``. It is accepted only if it is a
        dual-feasible full step (the same test as the loop: the oracle solved
        the pinned problem and no droppable multiplier is negative); the loop
        then continues from it (the warm pins keep the trial's multipliers,
        and an optimal warm set converges in this one iteration). Otherwise
        the trial is discarded and the loop continues exactly as the cold
        start. The trial counts as an iteration (``n_iter`` /
        ``last_n_iter``, and its NNLS iterations in ``n_cg_iter`` /
        ``n_nnls_iter``), whether accepted or not, and consumes one unit of
        ``max_iter``.

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
    warm_start: bool = eqx.field(static=True, default=False)
    elastic: bool = eqx.field(static=True, default=True)
    elastic_weight: float = eqx.field(static=True, default=1e3)

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
        zero_s = jnp.asarray(0.0, dtype)

        # Carry: working set (A_lb, A_ub), multipliers (mu_lb, mu_ub), pins added
        # by the last full step whose multiplier is still 0 (fresh_*), the most
        # violated one of them (p, p_ub; p = -1: none), the last full step and
        # its dual, the monitoring counters (ray steps, safeguard firings, warm
        # trial rejected; they never feed back into the loop), the last oracle
        # status, NNLS count, iteration count, outcome.
        def run(elastic_rho):
            """One bound working-set loop: on the plain QP, or on the elastic QP."""
            if self.warm_start:
                aset = initial_state.active_set
                warm_lb = jnp.asarray(aset.active_lb, bool) & jnp.isfinite(lo) & ~fixed
                warm_ub = (
                    jnp.asarray(aset.active_ub, bool)
                    & jnp.isfinite(hi)
                    & ~warm_lb
                    & ~fixed
                )
                start_lb, start_ub = fixed | warm_lb, warm_ub
            else:
                start_lb, start_ub = fixed, false_n
            init = (
                start_lb,
                start_ub,
                zeros_n,
                zeros_n,
                false_n,
                false_n,
                jnp.asarray(-1, jnp.int32),
                jnp.asarray(False),
                zeros_n,
                dual_zero,
                zero_s,
                zero_i,
                zero_i,
                zero_i,
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
                    delta_full,
                    n_ray,
                    n_safeguard,
                    n_reject,
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
                    elastic_rho=elastic_rho,
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
                    vmax(jnp.where(drop_lb, r_lb, 0.0)),
                    vmax(jnp.where(drop_ub, r_ub, 0.0)),
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
                    jnp.min(ratio_lb, initial=jnp.inf),
                    jnp.min(ratio_ub, initial=jnp.inf),
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
                mu_lb_step = jnp.where(
                    a_lb, mu_lb + jnp.where(move, t, 0.0) * dir_lb, 0.0
                )
                mu_ub_step = jnp.where(
                    a_ub, mu_ub + jnp.where(move, t, 0.0) * dir_ub, 0.0
                )
                mu_lb_new = jnp.where(
                    full, mu_lb_full, jnp.where(step, mu_lb_step, mu_lb)
                )
                mu_ub_new = jnp.where(
                    full, mu_ub_full, jnp.where(step, mu_ub_step, mu_ub)
                )
                mu_lb_next, mu_ub_next = clip_mu(
                    mu_lb_new, mu_ub_new, a_lb_next, a_ub_next
                )

                fresh_lb_next = (
                    jnp.where(batch, v_lb, jnp.where(move, False, fresh_lb)) & a_lb_next
                )
                fresh_ub_next = (
                    jnp.where(batch, v_ub, jnp.where(move, False, fresh_ub)) & a_ub_next
                )
                p_next = jnp.where(batch, p_new, jnp.where(move, -1, p)).astype(
                    jnp.int32
                )
                p_ub_next = jnp.where(batch, p_ub_new, p_ub)

                d_full_next = jnp.where(full, d_hat, d_full)
                dual_full_next = jax.tree.map(
                    lambda new, old: jnp.where(full, new.astype(dtype), old),
                    res.dual,
                    dual_full,
                )
                delta_full_next = jnp.where(full, res.delta.astype(dtype), delta_full)
                reject = jnp.asarray(False)
                if self.warm_start:
                    # First iteration = warm trial: unless it is a dual-feasible
                    # full step, discard it and restart from the cold carry (the
                    # iteration and NNLS counts below still include the trial).
                    # Only the resets of ``A``, ``infeasible`` and
                    # ``oracle_failure`` are live. Equivalent mutants: the resets
                    # of ``mu`` (it is 0 throughout a rejected trial, so every
                    # ratio is 0 and nothing moves), ``fresh`` and ``p`` (still
                    # at their cold values, since only a batch or a positive step
                    # changes them), ``converged`` (``converged`` implies
                    # ``full``) and ``stalled`` (with ``p = -1`` a negative
                    # multiplier always has a blocker, so no stall).
                    reject = (n_it == 0) & ~full
                    a_lb_next = jnp.where(reject, fixed, a_lb_next)
                    a_ub_next = jnp.where(reject, false_n, a_ub_next)
                    mu_lb_next = jnp.where(reject, 0.0, mu_lb_next)
                    mu_ub_next = jnp.where(reject, 0.0, mu_ub_next)
                    fresh_lb_next = fresh_lb_next & ~reject
                    fresh_ub_next = fresh_ub_next & ~reject
                    p_next = jnp.where(reject, -1, p_next).astype(jnp.int32)
                    p_ub_next = p_ub_next & ~reject
                    converged = converged & ~reject
                    infeasible = infeasible & ~reject
                    oracle_failure = oracle_failure & ~reject
                    stalled = stalled & ~reject
                # Monitoring counters (outside the numerics above).
                n_ray_next = n_ray + (ray_step & ~stalled & ~reject).astype(jnp.int32)
                n_safeguard_next = n_safeguard + (use_deg & ~stalled & ~reject).astype(
                    jnp.int32
                )
                n_reject_next = n_reject + reject.astype(jnp.int32)
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
                    delta_full_next,
                    n_ray_next,
                    n_safeguard_next,
                    n_reject_next,
                    res.ldp_status,
                    n_nnls + res.n_nnls_iter.astype(jnp.int32),
                    n_it + 1,
                    outcome,
                )

            return jax.lax.while_loop(cond_fn, body_fn, init)

        carry_p = run(None)

        # Elastic mode (plan 2.5, rev 11): entered only when the plain loop
        # certifies the QP infeasible or the oracle fails without a
        # certificate; the WHOLE bound loop is then re-run, from the same start
        # (warm trial included), on the elastic QP, which is one fixed QP for
        # every working set (unshifted relaxation, rho_e frozen). The two loops
        # have separate iteration budgets; the counters are cumulative, the
        # trial-rejected flag is that of the final loop.
        use_elastic = self.elastic and lag.meq + lag.mineq > 0
        if use_elastic:
            out_p = carry_p[-1]
            entered = (out_p == _INFEASIBLE) | (out_p == _ORACLE_FAILURE)

            def enter(c_p):
                rho = elastic_penalty(
                    lag, factors, self.elastic_weight, rank_tol=self.rank_tol
                ).astype(dtype)
                c_e = run(rho)
                return (
                    *c_e[:11],
                    c_p[11] + c_e[11],  # ray steps
                    c_p[12] + c_e[12],  # safeguard firings
                    c_e[13],  # warm trial rejected (final loop)
                    c_e[14],  # last oracle status
                    c_p[15] + c_e[15],  # NNLS iterations
                    c_p[16] + c_e[16],  # oracle calls
                    c_e[17],  # outcome
                )

            carry_f = jax.lax.cond(entered, enter, lambda c: c, carry_p)
        else:
            entered = jnp.asarray(False)
            carry_f = carry_p
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
            delta_full_f,
            n_ray_f,
            n_safeguard_f,
            n_reject_f,
            ldp_f,
            n_nnls_f,
            n_it_f,
            outcome_f,
        ) = carry_f

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
        # δ of the last full step (0 unless that step solved an elastic QP; also
        # 0 when the solve failed, like the dual).
        delta_out = jnp.where(failed, 0.0, delta_full_f).astype(dtype)

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
                s.last_elastic_delta,
                s.n_elastic,
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
                delta_out,
                jnp.asarray(initial_state.n_elastic, jnp.int32)
                + entered.astype(jnp.int32),
            ),
        )
        step = cast(Primal, Primal(x=step_f))
        self._log_solve(
            step,
            dual_out,
            new_state,
            n_pinned=jnp.sum(a_lb_f, dtype=jnp.int32)
            + jnp.sum(a_ub_f, dtype=jnp.int32),
            n_ray=jnp.asarray(n_ray_f, jnp.int32),
            n_safeguard=jnp.asarray(n_safeguard_f, jnp.int32),
            n_nnls=n_nnls_f,
            trial_rejected=jnp.asarray(n_reject_f, jnp.int32) > 0,
            elastic_entered=entered,
        )
        return (step, dual_out), new_state

    def _log_solve(
        self,
        step: Primal,
        dual: Dual,
        state: LRHQPSolverState,
        *,
        n_pinned: Int[Array, ""],
        n_ray: Int[Array, ""],
        n_safeguard: Int[Array, ""],
        n_nnls: Int[Array, ""],
        trial_rejected: Bool[Array, ""],
        elastic_entered: Bool[Array, ""],
    ) -> None:
        """Logger lines and ``"qp"`` diagnostic record of one solve.

        Pure monitoring: nothing here feeds back into the returned values.
        The record carries the keys of
        :class:`~slsqp_jax.sqpdax.subproblem.solver.active_set_loop.ActiveSetQPSolver`
        (``iters`` = bound working-set iterations, ``cg_iters`` = NNLS
        iterations of this solve) plus ``n_nnls_iter`` (same value),
        ``last_ldp_status``, ``warm_trial_accepted`` / ``warm_trial_rejected``
        (both ``False`` without :attr:`warm_start`), ``n_ray_steps``,
        ``n_safeguard`` (degenerate steps resolved by the most-violated-pin
        safeguard), ``stalled`` and ``n_pinned`` (final bound working set).
        ``n_ray_steps`` counts the iterations that move along a certificate
        ray; neither counter includes the iteration that stalls (it takes no
        step), the terminal certified-infeasible iteration, or a discarded
        (rejected) warm trial. ``warm_trial_accepted`` is ``False`` when no
        trial ran (``max_iter = 0``). The elastic fields are ``elastic``
        (elastic mode was entered in this solve), ``last_elastic_delta`` and
        ``n_elastic`` (cumulative).
        """
        converged = state.success
        stalled = state.qp_result == ACTIVE_SET_QP_RESULTS.anti_cycling
        self.logger.info(
            "qp solved: iters={iters} nnls_iters={nnls_iters} pinned={n_pinned} "
            "rays={n_ray} safeguards={n_safeguard}",
            when=converged,
            iters=state.last_n_iter,
            nnls_iters=n_nnls,
            n_pinned=n_pinned,
            n_ray=n_ray,
            n_safeguard=n_safeguard,
        )
        self.logger.warning(
            "qp failed: result={qp_result} status={status} iters={iters} "
            "nnls_iters={nnls_iters} ldp_status={ldp_status} kkt_reason={kkt_reason}",
            when=~converged,
            qp_result=state.qp_result,
            status=state.status,
            iters=state.last_n_iter,
            nnls_iters=n_nnls,
            ldp_status=state.last_ldp_status,
            kkt_reason=state.last_kkt_reason,
        )
        self.logger.warning(
            "qp bound loop stalled: iters={iters} pinned={n_pinned}",
            when=stalled,
            iters=state.last_n_iter,
            n_pinned=n_pinned,
        )
        self.logger.info(
            "qp elastic mode: delta={delta} solved={solved}",
            when=elastic_entered,
            delta=state.last_elastic_delta,
            solved=converged,
        )
        warm = jnp.asarray(self.warm_start)
        self.logger.diagnostic(
            "qp",
            lambda: {
                "qp_result": state.qp_result,
                "status": state.status,
                "success": state.success,
                "iters": state.last_n_iter,
                "cg_iters": state.last_n_cg_iter,
                "step": step,
                "dual": dual,
                "active_set": state.active_set,
                "qp_dual": state.dual,
                "final_working_tol": state.final_working_tol,
                "n_anti_cycling": state.n_anti_cycling,
                "kkt_feasibility_residual": state.last_kkt_feasibility_residual,
                "kkt_n_refinements": state.last_kkt_n_refinements,
                "kkt_reason": state.last_kkt_reason,
                "n_nnls_iter": n_nnls,
                "last_ldp_status": state.last_ldp_status,
                "warm_trial_accepted": warm & (state.last_n_iter > 0) & ~trial_rejected,
                "warm_trial_rejected": warm & trial_rejected,
                "n_ray_steps": n_ray,
                "n_safeguard": n_safeguard,
                "stalled": stalled,
                "n_pinned": n_pinned,
                "elastic": elastic_entered,
                "last_elastic_delta": state.last_elastic_delta,
                "n_elastic": state.n_elastic,
            },
        )
