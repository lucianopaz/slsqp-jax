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
]


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
    """

    tol: float = 1e-8
    max_iter: int = 50

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
        new_ai = (ai | (ineq_lin > working_tol)) & ~(ai & (lam_ineq < -drop_tol))

        alb = current.active_lb
        lb_violated = (lag.lb - x_new) > working_tol
        new_alb = (~lag.null_lb) & ((alb | lb_violated) & ~(alb & (lam_lb < -drop_tol)))

        aub = current.active_ub
        ub_violated = (x_new - lag.ub) > working_tol
        new_aub = (~lag.null_ub) & ((aub | ub_violated) & ~(aub & (lam_ub < -drop_tol)))
        return cast(
            ActiveSet,
            ActiveSet(
                meq=current.meq,
                active_inequalities=new_ai,
                active_lb=new_alb,
                active_ub=new_aub,
            ),
        )

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
