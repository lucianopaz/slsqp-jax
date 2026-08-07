"""Powell dogleg solver for the normal (feasibility) trust-region subproblem."""

from typing import Any, cast

import jax
from equinox import field
from jax import numpy as jnp
from jaxtyping import Array, Bool, Scalar

from ...dual import Dual
from ...primal import Primal
from ...types import Vector_n
from ..base import SubProblem
from .base import RESULTS, SubProblemSolver, SubProblemSolverState
from .gradient_projection import GradientProjection


class DogLegSolverState(SubProblemSolverState):
    """Carry for a dogleg normal-step solve.

    Attributes
    ----------
    n_cg_iter
        Reserved CG-iteration counter (dogleg itself does no CG; the field
        lets an orchestrator accumulate work across normal / tangential
        solves).
    on_boundary
        ``True`` when the accepted step saturates the trust-region radius.
    radius
        Trust-region radius used for this solve.
    active_bounds
        Optional ``(active_lb, active_ub)`` masks of length ``n``. ``None``
        means the solver identifies them via
        :class:`~slsqp_jax.sqpdax.subproblem.solver.gradient_projection.GradientProjection`.
    """

    n_cg_iter: int
    on_boundary: Bool[Array, ""]
    radius: Scalar
    active_bounds: tuple[Bool[Array, " n"], Bool[Array, " n"]] | None = None


class DogLegSolver(SubProblemSolver[Primal, SubProblem[Any], DogLegSolverState]):
    """Powell's dogleg (N&W Section 4.1) for the *normal* (feasibility) subproblem.

    Solves

    ```
    min_p  ½ ‖A p + c‖²    s.t.  ‖p‖ ≤ radius
    ```

    where ``A = subproblem.nonbound_constraint_jac()`` is the equality +
    inequality constraint Jacobian and ``c`` the matching constraint values
    (``subproblem.dual_grad()``). ``radius`` is read from ``initial_state``.

    Active-bound coordinates supplied in ``initial_state.active_bounds`` are
    pinned at zero step (their columns of ``A`` are zeroed). When the masks
    are ``None``, :class:`GradientProjection` identifies N&W ``A(x^c)``. The
    returned dual block is zeros — the trust-region orchestrator recovers
    multipliers by least squares (N&W eq. 19.37). ``x0`` is accepted for
    interface compatibility but unused.

    Attributes
    ----------
    solver_state_class
        :class:`DogLegSolverState`.
    rcond
        Relative singular-value floor for the Gauss–Newton pseudoinverse.
        ``None`` → ``eps * max(A.shape)``.
    gradient_projection
        Bound-face identifier used when ``active_bounds`` is ``None``.
    """

    solver_state_class: type[DogLegSolverState] = DogLegSolverState

    rcond: float | None = None
    gradient_projection: GradientProjection = field(default_factory=GradientProjection)

    def solve(
        self,
        subproblem: SubProblem[Any],
        x0: tuple[Primal, Dual],
        initial_state: DogLegSolverState,
    ) -> tuple[tuple[Primal, Dual], DogLegSolverState]:
        """Compute the dogleg normal step inside the trust region.

        Parameters
        ----------
        subproblem
            Local model exposing ``nonbound_constraint_jac`` / ``dual_grad``.
        x0
            Unused warm-start (kept for the
            :class:`~slsqp_jax.sqpdax.subproblem.solver.base.SubProblemSolver`
            interface).
        initial_state
            Must carry ``radius``; optional ``active_bounds``.

        Returns
        -------
        step
            ``(primal_step, zero_dual)``.
        state
            Updated :class:`DogLegSolverState` with boundary / success flags.
        """
        del x0  # interface compatibility; dogleg is computed from the model alone
        radius = initial_state.radius

        # Active bounds: use the caller-supplied set, else identify it via GP.
        if initial_state.active_bounds is None:
            active_lb, active_ub = self.gradient_projection.find_active_bounds(
                subproblem
            )
        else:
            active_lb, active_ub = initial_state.active_bounds

        constraint_jac = subproblem.nonbound_constraint_jac()  # A: (m_gen, n)
        constraint_dual = subproblem.dual_grad()
        constraint_values = jnp.concatenate(  # c: (m_gen,)
            [constraint_dual.eq_multipliers, constraint_dual.ineq_multipliers]
        )
        dtype = constraint_jac.dtype

        # Freeze active-bound variables: zero their columns of A so the normal step
        # leaves them at their bound (dogleg_step is then 0 on those coordinates).
        free_x = (~(active_lb | active_ub)).astype(dtype)
        constraint_jac = constraint_jac * free_x[None, :]

        # --- Cauchy point of 1/2 ||A p + c||^2: min along steepest descent. ---
        # The gradient at p = 0 is A^T c; the exact 1-D minimiser along -A^T c
        # has length ||A^T c||^2 / ||A A^T c||^2 (normal-equation curvature).
        feasibility_grad = constraint_jac.T @ constraint_values  # A^T c: (n,)
        jac_feasibility_grad = constraint_jac @ feasibility_grad  # A A^T c: (m_gen,)
        grad_sq = jnp.dot(feasibility_grad, feasibility_grad)
        curvature_sq = jnp.dot(jac_feasibility_grad, jac_feasibility_grad)
        cauchy_step_length = jnp.where(
            curvature_sq > 1e-30, grad_sq / jnp.maximum(curvature_sq, 1e-30), 0.0
        )
        cauchy_step = -cauchy_step_length * feasibility_grad  # p_C: (n,)

        # --- Gauss-Newton point: least-norm solution of A p = -c. ---
        # p_GN = -A^T (A A^T)^+ c via a thin, rank-revealing SVD of A.
        left_sv, sv, _ = jnp.linalg.svd(constraint_jac, full_matrices=False)
        eps = jnp.finfo(dtype).eps
        rcond = (
            self.rcond if self.rcond is not None else eps * max(constraint_jac.shape)
        )
        max_sv = jnp.max(sv, initial=0.0)  # 0.0 keeps m_gen == 0 well-defined
        keep = sv > rcond * max_sv
        inv_sv_sq = jnp.where(keep, 1.0 / jnp.square(jnp.where(keep, sv, 1.0)), 0.0)
        normal_eq_solve = left_sv @ (inv_sv_sq * (left_sv.T @ constraint_values))
        gauss_newton_step = -(constraint_jac.T @ normal_eq_solve)  # p_GN: (n,)

        # --- Powell's dogleg between the two, inside the trust region. ---
        dogleg_step = self._dogleg(cauchy_step, gauss_newton_step, radius)

        step = (
            cast(Primal, Primal(dogleg_step)),
            cast(Dual, jax.tree.map(jnp.zeros_like, constraint_dual)),
        )

        step_length = jnp.linalg.norm(dogleg_step)
        on_boundary = step_length >= radius * (1.0 - 1e-10)
        finite = jnp.all(jnp.isfinite(dogleg_step))
        status = RESULTS.where(finite, RESULTS.successful, RESULTS.singular)
        new_state = cast(
            DogLegSolverState,
            DogLegSolverState(
                n_iter=initial_state.n_iter + 1,
                n_cg_iter=initial_state.n_cg_iter,
                on_boundary=on_boundary,
                success=finite,
                status=status,
                radius=radius,
                active_bounds=(active_lb, active_ub),
            ),
        )
        return step, new_state

    @staticmethod
    def _dogleg(v_C: Vector_n, v_GN: Vector_n, radius: Scalar) -> Vector_n:
        """Powell dogleg point between Cauchy ``v_C`` and Gauss–Newton ``v_GN``.

        Parameters
        ----------
        v_C
            Cauchy point of the linear least-squares model.
        v_GN
            Unconstrained Gauss–Newton (least-norm) step.
        radius
            Trust-region radius.

        Returns
        -------
        Vector_n
            Point on the dogleg path with length at most ``radius``.
        """
        n_GN = jnp.linalg.norm(v_GN)
        n_C = jnp.linalg.norm(v_C)
        diff = v_GN - v_C
        a = jnp.dot(diff, diff)
        b = 2.0 * jnp.dot(v_C, diff)
        c = jnp.dot(v_C, v_C) - radius**2
        disc = jnp.sqrt(jnp.maximum(b * b - 4.0 * a * c, 0.0))
        theta = jnp.where(a > 1e-30, (-b + disc) / (2.0 * jnp.maximum(a, 1e-30)), 0.0)
        v_interp = v_C + theta * diff
        v_cauchy_bnd = v_C * (radius / jnp.maximum(n_C, 1e-30))
        return jnp.where(
            n_GN <= radius, v_GN, jnp.where(n_C >= radius, v_cauchy_bnd, v_interp)
        )
