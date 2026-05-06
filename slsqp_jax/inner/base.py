"""Abstract base class for pluggable inner equality-constrained QP solvers."""

from __future__ import annotations

import abc
from collections.abc import Callable

import equinox as eqx
from jaxtyping import Array, Bool, Float

from slsqp_jax.state import InnerSolveResult, ProjectionContext
from slsqp_jax.types import Scalar, Vector


class AbstractInnerSolver(eqx.Module):
    """Strategy for solving the equality-constrained QP subproblem.

    Subclasses implement ``solve`` to compute the search direction ``d``
    and Lagrange multipliers for the active constraints.
    """

    @abc.abstractmethod
    def solve(
        self,
        hvp_fn: Callable[[Vector], Vector],
        g: Vector,
        A: Float[Array, "m n"],
        b: Float[Array, " m"],
        active_mask: Bool[Array, " m"],
        precond_fn: Callable[[Vector], Vector] | None = None,
        free_mask: Bool[Array, " n"] | None = None,
        d_fixed: Vector | None = None,
        adaptive_tol: Scalar | float | None = None,
    ) -> InnerSolveResult:
        """Solve the equality-constrained QP subproblem.

        Solves::

            minimize    (1/2) d^T B d + g^T d
            subject to  A[active] d = b[active]
                        d[i] = d_fixed[i]  for i where free_mask[i] is False

        where B is given implicitly via ``hvp_fn(v) = B @ v``.

        Args:
            hvp_fn: Hessian-vector product function v -> B @ v.
            g: Linear term (gradient of objective).
            A: Combined constraint matrix (m x n).
            b: Combined RHS vector (m,).
            active_mask: Boolean mask (m,) indicating active constraints.
            precond_fn: Optional preconditioner v -> M @ v where M ~ B^{-1}.
            free_mask: Optional boolean mask (n,).  When provided, only
                variables with ``free_mask[i] = True`` are optimized.
            d_fixed: Values for fixed variables (n,).  Required when
                ``free_mask`` is provided.
            adaptive_tol: Optional Eisenstat-Walker tolerance override.
                When provided, overrides the solver's default convergence
                tolerance for this call only.

        Returns:
            ``InnerSolveResult`` with the direction, multipliers, and
            convergence flag.
        """
        ...  # pragma: no cover

    def build_projection_context(
        self,
        hvp_fn: Callable[[Vector], Vector],
        g: Vector,
        A: Float[Array, "m n"],
        b: Float[Array, " m"],
        active_mask: Bool[Array, " m"],
        precond_fn: Callable[[Vector], Vector] | None = None,
        free_mask: Bool[Array, " n"] | None = None,
        d_fixed: Vector | None = None,
    ) -> ProjectionContext:
        """Build a reusable projector + multiplier-recovery context.

        Composed strategies (e.g. ``HRInexactSTCG``) call this on the
        underlying inner solver to obtain its null-space projector,
        particular solution and multiplier-recovery closure without
        running the projector's own CG loop.

        The default implementation raises ``NotImplementedError`` so
        full-KKT solvers (``MinresQLPSolver``) cleanly opt out — they
        have no separate projection step and therefore cannot supply
        the inexact-projector ``W̃_k`` that HR Algorithm 4.5 needs.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a projection context; "
            "it cannot be used as the inner projector for HRInexactSTCG."
        )


__all__ = ["AbstractInnerSolver"]
