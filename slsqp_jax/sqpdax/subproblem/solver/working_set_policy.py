"""Working-set update policies for the active-set QP loop.

The :class:`~slsqp_jax.sqpdax.subproblem.solver.active_set_loop.ActiveSetQPSolver`
solves an equality-constrained KKT system for a fixed working set, then asks
a :class:`WorkingSetPolicy` which constraints to add / drop for the next
iteration. Separating the policy from the loop lets the proximal and plain
solvers share one anti-cycling implementation and lets callers tune it
through ``options['subproblem']['working_set_policy']``.
"""

from abc import abstractmethod
from typing import cast

import equinox as eqx
import jax
from equinox import Module
from jax import numpy as jnp
from jaxtyping import Array, Bool, Int

from ...active_set import ActiveSet
from ...dual import Dual
from ...lagrangian import EvaluatedLagrangian
from ...primal import Primal
from ...types import InitializableModule, Scalar

__all__ = [
    "WorkingSetPolicyState",
    "WorkingSetPolicy",
    "ThresholdWorkingSetPolicy",
    "SingleExchangeWorkingSetPolicy",
    "cap_rows",
    "independent_rows",
]


def cap_rows(
    keep: Bool[Array, " m"],
    candidates: Bool[Array, " m"],
    scores: Array,
    cap: int,
) -> Bool[Array, " m"]:
    """Select all of ``keep``, then the best ``candidates`` up to ``cap`` rows in total.

    Parameters
    ----------
    keep
        Rows that are always selected (the rows already in the working set,
        or the rows active at the reference point): they are consistent
        with the current point, so any number of them leaves the linearised
        system solvable. They count against ``cap`` but are never dropped.
    candidates
        Rows eligible to fill the remaining ``cap - |keep|`` slots.
    scores
        Ranking of the candidates (larger first), e.g. their violation.
    cap
        Maximum number of selected rows (a static integer), binding on the
        candidates only.

    Returns
    -------
    Bool[Array, "m"]
        ``keep`` plus at most ``max(cap - |keep|, 0)`` candidates.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.subproblem.solver.working_set_policy import cap_rows
    >>> keep = jnp.array([False, True, False, False])
    >>> cand = jnp.array([True, False, True, True])
    >>> cap_rows(keep, cand, jnp.array([0.5, 0.0, 2.0, 1.0]), 2).tolist()
    [False, True, True, False]
    """
    m = keep.shape[0]
    priority = jnp.where(
        keep, jnp.inf, jnp.where(candidates, jnp.asarray(scores), -jnp.inf)
    )
    order = jnp.argsort(-priority, stable=True)
    rank = jnp.zeros((m,), jnp.int32).at[order].set(jnp.arange(m, dtype=jnp.int32))
    return keep | (candidates & (rank < cap))


def independent_rows(
    keep: Bool[Array, " m"],
    candidates: Bool[Array, " m"],
    scores: Array,
    cap: int,
    lag: EvaluatedLagrangian[Primal],
    *,
    tol: float = 1e-4,
) -> Bool[Array, " m"]:
    """Select ``keep`` plus mutually independent ``candidates`` up to ``cap`` rows.

    Rows are visited as in :func:`cap_rows` (``keep`` first, then the
    ``candidates`` by decreasing score). ``keep`` rows are always selected
    (they are consistent with the current point, so dependence among them
    is harmless) and seed the orthonormal basis together with the equality
    Jacobian; a candidate is accepted greedily when it is linearly
    independent of that basis and of the candidates accepted so far, i.e.
    rejected when the residual of its Gram--Schmidt projection onto the
    accepted span is below ``tol * ‖a‖₂``. Only the first ``4 * cap`` rows
    in priority order are examined, so the cost is ``O(cap * n²)`` per call.

    Parameters
    ----------
    keep, candidates, scores, cap
        As in :func:`cap_rows`; ``cap`` counts ``keep`` rows as well, so
        no candidate enters once ``|keep| >= cap``.
    lag
        Lagrangian evaluation providing the equality Jacobian, the
        inequality Jacobian and the sizes. The stacked row order is
        ``[inequalities | lower bounds | upper bounds]`` and the bound rows
        are ``∓eᵢ``.
    tol
        Relative independence threshold.

    Returns
    -------
    Bool[Array, "m"]
        Mask of the accepted rows.

    Examples
    --------
    Two parallel violated rows and one orthogonal one: the parallel
    duplicate is skipped even though it is more violated than the third row.

    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.lagrangian import Lagrangian
    >>> from slsqp_jax.sqpdax.primal import Primal
    >>> from slsqp_jax.sqpdax.problem import build_problem
    >>> from slsqp_jax.sqpdax.subproblem.solver.working_set_policy import (
    ...     independent_rows,
    ... )
    >>> A = jnp.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    >>> problem = build_problem(
    ...     n=2,
    ...     fn=lambda x: jnp.sum(x**2),
    ...     grad=lambda x: 2 * x,
    ...     hvp=lambda x, v: 2 * v,
    ...     ineq_fn=lambda x: A @ x + 1.0,
    ...     ineq_fn_jac=lambda x: A,
    ...     ineq_fn_hvp=lambda x, v: jnp.zeros((3, 2)),
    ... )
    >>> lag = Lagrangian(problem)(Primal(jnp.zeros(2)), None)
    >>> keep = jnp.zeros(7, bool)
    >>> cand = jnp.array([True, True, True, False, False, False, False])
    >>> scores = jnp.array([3.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    >>> independent_rows(keep, cand, scores, 2, lag).tolist()[:3]
    [False, True, True]
    """
    m = keep.shape[0]
    n, meq, mineq = int(lag.n), int(lag.meq), int(lag.mineq)
    eq_jac = lag.eq_fn_jac_val
    ineq_jac = lag.ineq_fn_jac_val
    dtype = eq_jac.dtype
    eligible = keep | candidates
    priority = jnp.where(
        keep, jnp.inf, jnp.where(candidates, jnp.asarray(scores), -jnp.inf)
    )
    order = jnp.argsort(-priority, stable=True)
    visits = min(m, 4 * cap)
    tol_arr = jnp.asarray(tol, dtype)

    def stacked_row(j: Int[Array, ""]) -> Array:
        if mineq:
            a_ineq = ineq_jac[jnp.minimum(j, mineq - 1)]
        else:
            a_ineq = jnp.zeros((n,), dtype)
        b = j - mineq
        is_lb = b < n
        idx = jnp.clip(jnp.where(is_lb, b, b - n), 0, n - 1)
        e = jax.nn.one_hot(idx, n, dtype=dtype) * jnp.where(is_lb, -1.0, 1.0)
        return jnp.where(j < mineq, a_ineq, e)

    def insert(carry, a, allow):
        """Append ``a`` to the orthonormal basis when allowed and independent."""
        basis, k = carry
        r = a - basis @ (basis.T @ a)
        r = r - basis @ (basis.T @ r)  # re-orthogonalise for stability
        norm_r = jnp.linalg.norm(r)
        independent = norm_r > tol_arr * jnp.linalg.norm(a)
        accept = allow & independent & (k < n)
        slot = jnp.minimum(k, n - 1)
        q = jnp.where(accept, r / jnp.where(norm_r > 0.0, norm_r, 1.0), basis[:, slot])
        return (basis.at[:, slot].set(q), k + accept.astype(k.dtype)), accept

    carry = (jnp.zeros((n, n), dtype), jnp.asarray(0, jnp.int32))
    if meq:
        carry = jax.lax.fori_loop(
            0, meq, lambda i, c: insert(c, eq_jac[i], jnp.asarray(True))[0], carry
        )

    def body(i, loop_carry):
        carry, count, selected = loop_carry
        j = order[i]
        is_keep = keep[j]
        carry, independent = insert(carry, stacked_row(j), eligible[j])
        # ``keep`` rows are selected unconditionally (and extend the basis
        # when independent); candidates need independence and a free slot.
        accept = is_keep | (candidates[j] & independent & (count < cap))
        selected = selected.at[j].set(accept)
        return carry, count + accept.astype(count.dtype), selected

    selected = jnp.zeros((m,), bool)
    _, _, selected = jax.lax.fori_loop(
        0, visits, body, (carry, jnp.asarray(0, jnp.int32), selected)
    )
    # ``keep`` rows beyond the visited prefix are selected all the same.
    return selected | keep


def _split_active(
    stacked: Bool[Array, " m"], meq: int, mineq: int, n: int
) -> ActiveSet:
    """Rebuild an :class:`ActiveSet` from a stacked ``[ineq | lb | ub]`` mask."""
    return cast(
        ActiveSet,
        ActiveSet(
            meq=meq,
            active_inequalities=stacked[:mineq],
            active_lb=stacked[mineq : mineq + n],
            active_ub=stacked[mineq + n :],
        ),
    )


class WorkingSetPolicyState(Module):
    """Per-solve carry of a :class:`WorkingSetPolicy`.

    Attributes
    ----------
    working_tol
        Add / drop threshold the *next* :meth:`WorkingSetPolicy.update` will
        use. Constant for a fixed-tolerance policy; ramped by EXPAND.
    ramp_increment
        Amount added to ``working_tol`` after every update (``0`` when the
        ramp is off).
    prev_set
        Working set the previous KKT solve used (one behind ``current``).
    prev_prev_set
        Working set two KKT solves back.
    cycle_count
        Consecutive updates whose proposal reproduced ``prev_set`` or
        ``prev_prev_set`` (a 2- or 3-cycle).
    """

    working_tol: Scalar
    ramp_increment: Scalar
    prev_set: ActiveSet
    prev_prev_set: ActiveSet
    cycle_count: Int[Array, ""]


class WorkingSetPolicy(InitializableModule):
    """Strategy deciding the next working set of the active-set QP loop.

    Subclasses implement :meth:`init_state` (once per solve) and
    :meth:`update` (once per working-set iteration). The policy owns the
    base tolerance and the iteration budget of the working-set loop; the
    loop reads both, runs the convergence test (``next == current``) and
    stops on the policy's anti-cycling flag.

    Attributes
    ----------
    tol
        Base add / drop tolerance. Also used by the solver to build the
        cold-start working set.
    max_iter
        Maximum working-set iterations per solve.
    cap_working_set
        Never let an *inconsistent* working set hold more inequality / bound
        rows than the problem has degrees of freedom, ``n - meq``. Violated
        rows demand ``aᵢ d = -cᵢ``; more than ``n - meq`` of them (an
        infeasible start with many violated constraints) make the KKT system
        inconsistent, which the Craig / MINRES-QLP KKT solvers cannot solve
        and the projected-CG solver answers with a least-squares particular
        solution and no null-space step. The cold-start set is therefore
        cut down to its ``n - meq`` most violated rows whenever the
        reference point violates a constraint (inequality, bound or
        equality) by more than :attr:`tol` (see :meth:`initial_set`), and
        during the refresh the rows staying in the set are never dropped
        while the most violated of the new rows fill the slots left below
        the cap (see :func:`cap_rows`). At a feasible reference point the
        cold-start set is kept whole: all of its rows are consistent with
        ``d = 0`` (a degenerate vertex with more active bounds than free
        variables, HS119 / LSNNODOC). The constraints left out by the cap
        are handled by the merit function of the outer loop and re-examined
        at the next solve. On by default.
    independence_tol
        When the capped set would be full (at least ``n - meq`` rows
        requested), the candidate rows are additionally screened for linear
        independence against the equality Jacobian, the rows staying in the
        set and the candidates already accepted (see
        :func:`independent_rows`): a row whose Gram--Schmidt residual is
        below ``independence_tol`` times its norm is skipped. This keeps
        near-duplicate violated rows (discretised semi-infinite constraints,
        bounds that fix the variables of an equality) from producing a
        singular KKT system. Rows staying in the set are never screened,
        and neither is an all-active cold-start set (two nearly parallel
        active constraints at a degenerate solution, as in HS13, are both
        needed). ``0`` disables the screening.
    """

    tol: float = 1e-8
    max_iter: int = 50
    cap_working_set: bool = eqx.field(static=True, default=True)
    independence_tol: float = eqx.field(static=True, default=1e-4)

    @staticmethod
    def _row_cap(lag: EvaluatedLagrangian[Primal]) -> int:
        """Degrees of freedom left by the equalities, at least one."""
        return max(int(lag.n) - int(lag.meq), 1)

    def _select_rows(
        self,
        lag: EvaluatedLagrangian[Primal],
        keep: Bool[Array, " m"],
        candidates: Bool[Array, " m"],
        scores: Array,
    ) -> Bool[Array, " m"]:
        """Capped selection, with the independence screen when the set is full."""
        cap = self._row_cap(lag)
        capped = cap_rows(keep, candidates, scores, cap)
        if self.independence_tol <= 0.0:
            return capped
        full = jnp.sum(keep | candidates) >= cap
        return jax.lax.cond(
            full,
            lambda: independent_rows(
                keep, candidates, scores, cap, lag, tol=self.independence_tol
            ),
            lambda: capped,
        )

    def initial_set(
        self, lag: EvaluatedLagrangian[Primal], active0: ActiveSet
    ) -> ActiveSet:
        """Apply the row cap to a cold / warm-started initial working set.

        Parameters
        ----------
        lag
            Unmasked Lagrangian evaluation defining the QP.
        active0
            Candidate initial set (rows active or violated at the reference
            point, possibly merged with a warm start).

        Returns
        -------
        ActiveSet
            ``active0`` when :attr:`cap_working_set` is off, when the set
            fits, or when no row of ``active0`` is violated by more than
            :attr:`tol` (all rows consistent with ``d = 0``); otherwise the
            ``n - meq`` most violated rows of ``active0``.
        """
        if not self.cap_working_set:
            return active0
        values, infeasible = self._reference_violation(lag)
        active = jnp.concatenate(
            [active0.active_inequalities, active0.active_lb, active0.active_ub]
        )
        # At a feasible reference point every row is consistent with
        # ``d = 0`` and the whole set is kept (degenerate vertex). Otherwise
        # the set is capped by decreasing violation: the violated rows carry
        # the infeasibility information, the active rows fill what is left,
        # and the QP's own drop / add exchange sorts out the rest (a start on
        # all of its bounds with a violated equality, DUALC1, must not be
        # pinned to ``d = 0``).
        keep = jnp.where(infeasible, False, active)
        kept = self._select_rows(lag, keep, active & ~keep, values)
        return _split_active(kept, active0.meq, lag.mineq, lag.n)

    def _reference_violation(
        self, lag: EvaluatedLagrangian[Primal]
    ) -> tuple[Array, Bool[Array, ""]]:
        """Stacked row violations at the reference point and whether any
        row (including the equalities) is violated by more than :attr:`tol`.
        """
        x = lag.ref.x
        values = jnp.concatenate(
            [
                lag.ineq_fn_val,
                jnp.where(lag.null_lb, -jnp.inf, lag.lb - x),
                jnp.where(lag.null_ub, -jnp.inf, x - lag.ub),
            ]
        )
        tol = jnp.asarray(self.tol, values.dtype)
        # The equality rows are always in the set and demand ``A_eq d = -c_eq``;
        # a non-zero residual makes them inconsistent with ``d = 0`` too.
        infeasible = jnp.any(values > tol) | jnp.any(jnp.abs(lag.eq_fn_val) > tol)
        return values, infeasible

    @abstractmethod
    def init_state(self, active0: ActiveSet) -> WorkingSetPolicyState:
        """Build the per-solve carry from ``active0`` and the policy's fields.

        Parameters
        ----------
        active0
            Initial working set of the loop.

        Returns
        -------
        WorkingSetPolicyState
            Cold policy carry.
        """
        ...

    @abstractmethod
    def update(
        self,
        lag: EvaluatedLagrangian[Primal],
        step: tuple[Primal, Dual],
        current: ActiveSet,
        state: WorkingSetPolicyState,
    ) -> tuple[ActiveSet, WorkingSetPolicyState, Bool[Array, ""]]:
        """Propose the next working set from the KKT solution on ``current``.

        Parameters
        ----------
        lag
            Unmasked Lagrangian evaluation defining the QP.
        step
            Primal step and multipliers solved on ``current``.
        current
            Working set the step was solved on.
        state
            Policy carry from the previous update.

        Returns
        -------
        next_set
            Proposed working set for the next KKT solve.
        state
            Refreshed policy carry.
        cycled
            ``True`` when the anti-cycling guard fired; the loop then stops
            and reports ``anti_cycling``.
        """
        ...


class ThresholdWorkingSetPolicy(WorkingSetPolicy):
    """Vectorised add / drop update with optional EXPAND ramp and cycle guard.

    With every knob at its default the update is the classic fixed-threshold
    refresh of the whole set:

    * **add** an inequality / bound whose linearised value exceeds
      ``working_tol`` at the computed step,
    * **drop** an active inequality / bound whose multiplier is below
      ``-max(working_tol, drop_floor)``.

    ``expand_factor > 0`` turns on an EXPAND-style ramp (Gill, Murray,
    Saunders & Wright, 1989): the working tolerance grows linearly from
    ``tol`` at the first update to ``tol * (1 + expand_factor)`` at the
    iteration budget,

    ```
    working_tol_k = tol * (1 + expand_factor * k / max_iter),
    ```

    so a constraint that hovers at the threshold is not added and dropped
    forever. ``drop_floor`` decouples the drop test from the (possibly small)
    add tolerance so multiplier-recovery noise does not evict a genuinely
    active row. ``ping_pong_threshold`` stops the loop once the proposed set
    has reproduced one of the two previous sets that many times in a row.

    Attributes
    ----------
    tol, max_iter
        Inherited base tolerance and iteration budget.
    expand_factor
        Total relative growth of the working tolerance over ``max_iter``
        iterations; ``0.0`` (default) keeps it fixed.
    drop_floor
        Absolute floor on the negative-multiplier drop test; ``0.0``
        (default) uses ``working_tol`` alone.
    ping_pong_threshold
        Consecutive cycle detections before the loop is stopped; ``None``
        (default) disables the guard.

    Examples
    --------
    >>> from slsqp_jax.sqpdax.subproblem.solver import (
    ...     ActiveSetQPSolver,
    ...     ThresholdWorkingSetPolicy,
    ... )
    >>> solver = ActiveSetQPSolver(
    ...     working_set_policy=ThresholdWorkingSetPolicy(
    ...         tol=1e-6, max_iter=20, expand_factor=1.0, ping_pong_threshold=3
    ...     )
    ... )
    >>> solver.working_set_policy.expand_factor
    1.0
    >>> solver.max_iter  # the loop reads its budget from the policy
    20

    The same fields can be set through the nested option path of an
    :class:`~slsqp_jax.sqpdax.types.InitializableModule`:

    >>> solver = ActiveSetQPSolver().init(
    ...     working_set_policy={"tol": 1e-6, "expand_factor": 0.5}
    ... )
    >>> solver.tol
    1e-06
    """

    expand_factor: float = 0.0
    drop_floor: float = 0.0
    ping_pong_threshold: int | None = eqx.field(static=True, default=None)

    def __check_init__(self) -> None:
        """Validate the knobs."""
        if self.tol < 0.0:
            raise ValueError(f"tol must be non-negative; got {self.tol}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be at least 1; got {self.max_iter}")
        if self.expand_factor < 0.0:
            raise ValueError(
                f"expand_factor must be non-negative; got {self.expand_factor}"
            )
        if self.drop_floor < 0.0:
            raise ValueError(f"drop_floor must be non-negative; got {self.drop_floor}")
        if self.ping_pong_threshold is not None and self.ping_pong_threshold < 1:
            raise ValueError(
                "ping_pong_threshold must be None or at least 1; "
                f"got {self.ping_pong_threshold}"
            )

    def init_state(self, active0: ActiveSet) -> WorkingSetPolicyState:
        """Cold carry: ``working_tol = tol``, history seeded with ``active0``.

        Parameters
        ----------
        active0
            Initial working set of the loop.

        Returns
        -------
        WorkingSetPolicyState
            Cold policy carry.
        """
        tol = jnp.asarray(self.tol)
        return cast(
            WorkingSetPolicyState,
            WorkingSetPolicyState(
                working_tol=tol,
                # Linear ramp from ``tol`` to ``tol * (1 + expand_factor)`` over
                # the iteration budget.
                ramp_increment=tol * (self.expand_factor / max(self.max_iter, 1)),
                prev_set=active0,
                prev_prev_set=active0,
                cycle_count=jnp.asarray(0, jnp.int32),
            ),
        )

    def propose(
        self,
        lag: EvaluatedLagrangian[Primal],
        step: tuple[Primal, Dual],
        current: ActiveSet,
        working_tol: Scalar,
    ) -> ActiveSet:
        """Vectorised add / drop refresh of ``current`` at ``working_tol``.

        Parameters
        ----------
        lag
            Unmasked Lagrangian evaluation defining the QP.
        step
            Primal step and multipliers solved on ``current``.
        current
            Working set the step was solved on.
        working_tol
            Add tolerance; the drop tolerance is ``max(working_tol, drop_floor)``.

        Returns
        -------
        ActiveSet
            Proposed working set (bounds masked by ``null_lb`` / ``null_ub``).
        """
        drop_tol = jnp.maximum(
            working_tol, jnp.asarray(self.drop_floor, working_tol.dtype)
        )

        dx = step[0].x
        lam_ineq = step[1].ineq_multipliers
        lam_lb = step[1].lb_multipliers
        lam_ub = step[1].ub_multipliers
        x_new = lag.ref.x + dx

        ineq_lin = lag.ineq_fn_val + lag.ineq_fn_jac_val @ dx
        ai = current.active_inequalities
        keep_ai = ai & ~(lam_ineq < -drop_tol)
        add_ai = ~ai & (ineq_lin > working_tol)

        alb = current.active_lb
        keep_alb = alb & ~(lam_lb < -drop_tol)
        add_alb = (~lag.null_lb) & ~alb & ((lag.lb - x_new) > working_tol)

        aub = current.active_ub
        keep_aub = aub & ~(lam_ub < -drop_tol)
        add_aub = (~lag.null_ub) & ~aub & ((x_new - lag.ub) > working_tol)

        refreshed = cast(
            ActiveSet,
            ActiveSet(
                meq=current.meq,
                active_inequalities=keep_ai | add_ai,
                active_lb=(~lag.null_lb) & (keep_alb | add_alb),
                active_ub=(~lag.null_ub) & (keep_aub | add_aub),
            ),
        )
        if not self.cap_working_set:
            return refreshed
        # Rows staying in the set are never dropped by the cap; the most
        # violated of the new rows fill the slots left below ``n - meq``.
        keep = jnp.concatenate([keep_ai, keep_alb, keep_aub])
        add = jnp.concatenate([add_ai, add_alb, add_aub])
        values = jnp.concatenate([ineq_lin, lag.lb - x_new, x_new - lag.ub])
        kept = self._select_rows(lag, keep, add, values)
        return _split_active(kept, current.meq, lag.mineq, lag.n)

    def update(
        self,
        lag: EvaluatedLagrangian[Primal],
        step: tuple[Primal, Dual],
        current: ActiveSet,
        state: WorkingSetPolicyState,
    ) -> tuple[ActiveSet, WorkingSetPolicyState, Bool[Array, ""]]:
        """:meth:`propose` at ``state.working_tol``, detect cycles, advance the ramp.

        Parameters
        ----------
        lag
            Unmasked Lagrangian evaluation defining the QP.
        step
            Primal step and multipliers solved on ``current``.
        current
            Working set the step was solved on.
        state
            Policy carry from the previous update.

        Returns
        -------
        next_set
            Proposed working set.
        state
            Carry with the ramped tolerance, shifted history and cycle count.
        cycled
            Anti-cycling flag (always ``False`` when
            :attr:`ping_pong_threshold` is ``None``).
        """
        working_tol = state.working_tol
        next_set = self.propose(lag, step, current, working_tol)

        # Set-level cycle detection: the proposal reproduces a set the loop
        # already solved on (A -> B -> A or A -> B -> C -> A) without being a
        # fixed point of the update.
        revisits = eqx.tree_equal(next_set, state.prev_set) | eqx.tree_equal(
            next_set, state.prev_prev_set
        )
        is_cycle = revisits & jnp.logical_not(eqx.tree_equal(next_set, current))
        cycle_count = jnp.where(is_cycle, state.cycle_count + 1, 0).astype(jnp.int32)
        if self.ping_pong_threshold is None:
            cycled = jnp.asarray(False)
        else:
            cycled = cycle_count >= self.ping_pong_threshold

        new_state = cast(
            WorkingSetPolicyState,
            WorkingSetPolicyState(
                working_tol=working_tol + state.ramp_increment,
                ramp_increment=state.ramp_increment,
                prev_set=current,
                prev_prev_set=state.prev_set,
                cycle_count=cycle_count,
            ),
        )
        return next_set, new_state, cycled


class SingleExchangeWorkingSetPolicy(ThresholdWorkingSetPolicy):
    """Classical one-at-a-time exchange: add *or* drop a single row per iteration.

    Same tolerances, EXPAND ramp, drop floor and cycle guard as
    :class:`ThresholdWorkingSetPolicy`, but the proposal changes the working
    set by at most one row:

    1. if any inactive inequality / bound is violated by more than
       ``working_tol`` at the computed step, **add the most violated one**;
    2. otherwise, if any active row has a multiplier below
       ``-max(working_tol, drop_floor)``, **drop the most negative one**;
    3. otherwise return ``current`` (the loop's convergence test fires).

    Adding one row at a time keeps the working set consistent: the
    all-at-once refresh can add several mutually incompatible rows in one go
    (e.g. two bounds and an inequality that cannot hold simultaneously),
    after which the KKT system has no solution. The price is one KKT solve
    per exchanged row.

    Examples
    --------
    >>> from slsqp_jax.sqpdax.subproblem.solver import (
    ...     ActiveSetQPSolver,
    ...     SingleExchangeWorkingSetPolicy,
    ... )
    >>> solver = ActiveSetQPSolver(
    ...     working_set_policy=SingleExchangeWorkingSetPolicy(max_iter=30)
    ... )
    >>> solver.max_iter
    30
    """

    def propose(
        self,
        lag: EvaluatedLagrangian[Primal],
        step: tuple[Primal, Dual],
        current: ActiveSet,
        working_tol: Scalar,
    ) -> ActiveSet:
        """Exchange at most one row of ``current`` at ``working_tol``.

        Parameters
        ----------
        lag
            Unmasked Lagrangian evaluation defining the QP.
        step
            Primal step and multipliers solved on ``current``.
        current
            Working set the step was solved on.
        working_tol
            Add tolerance; the drop tolerance is ``max(working_tol, drop_floor)``.

        Returns
        -------
        ActiveSet
            ``current`` with one row added, one row dropped, or unchanged.
        """
        drop_tol = jnp.maximum(
            working_tol, jnp.asarray(self.drop_floor, working_tol.dtype)
        )
        n, mineq = lag.n, lag.mineq
        dx = step[0].x
        x_new = lag.ref.x + dx

        # Stack general inequalities, lower and upper bounds into one vector.
        values = jnp.concatenate(
            [
                lag.ineq_fn_val + lag.ineq_fn_jac_val @ dx,
                jnp.where(lag.null_lb, -jnp.inf, lag.lb - x_new),
                jnp.where(lag.null_ub, -jnp.inf, x_new - lag.ub),
            ]
        )
        multipliers = jnp.concatenate(
            [
                step[1].ineq_multipliers,
                step[1].lb_multipliers,
                step[1].ub_multipliers,
            ]
        )
        active = jnp.concatenate(
            [current.active_inequalities, current.active_lb, current.active_ub]
        )

        violated = ~active & (values > working_tol)
        add_idx = jnp.argmax(jnp.where(violated, values, -jnp.inf))
        negative = active & (multipliers < -drop_tol)
        drop_idx = jnp.argmin(jnp.where(negative, multipliers, jnp.inf))

        can_add = jnp.any(violated)
        if self.cap_working_set:
            # A full set must drop a row before it can take another one.
            can_add = can_add & (jnp.sum(active) < self._row_cap(lag))
        next_active = jnp.where(
            can_add,
            active.at[add_idx].set(True),
            jnp.where(jnp.any(negative), active.at[drop_idx].set(False), active),
        )
        return _split_active(next_active, current.meq, mineq, n)
