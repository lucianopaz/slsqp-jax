"""Abstract subproblem-solver interface and per-step context carrier."""

from abc import abstractmethod
from typing import Generic, Self, TypeVar

from equinox import Enumeration, Module, field, tree_at
from jax import numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Bool, Int
from lineax._solution import RESULTS

from ...dual import Dual
from ...lagrangian.basic import Lagrangian
from ...lagrangian.evaluated import EvaluatedLagrangian
from ...preconditioner import Preconditioner
from ...primal import PrimalType
from ...problem import ProblemProtocol
from ...types import InitializableModule, Scalar
from ..base import SubProblemType

__all__ = [
    "RESULTS",
    "KKT_SOLVER_RESULTS",
    "KKTSolverState",
    "SubProblemSolverState",
    "SubProblemSolverStateType",
    "SubProblemSolver",
    "SubproblemContext",
    "install_default_preconditioner",
]


class SubProblemSolverState(Module):
    """Base carry threaded across a subproblem solve.

    Concrete solvers may add fields. Outer loops (e.g. the active-set QP)
    read the shared triad below.

    Attributes
    ----------
    n_iter
        Cumulative iteration count owned by this solver (CG steps, outer
        QP iterations, …). Semantics are solver-specific.
    success
        ``True`` when the most recent solve met its convergence criteria.
    status
        Lineax :class:`~lineax.RESULTS` code for the most recent solve.
    """

    n_iter: int
    success: bool
    status: RESULTS


SubProblemSolverStateType = TypeVar(
    "SubProblemSolverStateType", bound=SubProblemSolverState
)


class KKT_SOLVER_RESULTS(Enumeration):
    """Why an inner (fixed working set) KKT solve stopped.

    Finer than the shared :class:`~lineax.RESULTS` code: a null-space CG
    that froze on its roundoff floor and one that met its tolerance are
    both usable steps but different diagnostics, and a projector that failed
    to build is different from a KKT iteration that ran out of budget.
    """

    converged = "The KKT solve met its tolerance."
    residual_floor = (
        "The residual stopped decreasing before the tolerance was met "
        "(roundoff floor or negative-curvature freeze); the best iterate is "
        "returned."
    )
    max_iter_reached = "The KKT iteration budget was exhausted."
    projector_failure = "The projector's own linear solves did not converge."
    nonfinite = "The solve produced non-finite values."


class KKTSolverState(SubProblemSolverState):
    """Standardised carry for the inner KKT solvers of an active-set loop.

    Outer loops seed it with :meth:`cold` so the carry dtype matches the
    iterate whatever the inner solver is.

    Attributes
    ----------
    feasibility_residual
        ``‖A_work d − b‖`` of the returned step over the working constraints.
        Structurally ``0`` (up to roundoff) for null-space solvers; the floor
        of the post-solve feasibility projection for full-KKT solvers.
    n_refinements
        Iterative-refinement rounds actually applied to the step
        (``0`` for null-space solvers).
    projected_grad_norm
        Norm of the projected gradient the solver stopped at (the inexact
        stationarity proxy of Heinkenschloss & Ridzal 2014); ``inf`` when the
        solver does not produce it.
    reason
        :class:`KKT_SOLVER_RESULTS` code for the most recent solve.
    nonfinite
        ``True`` when the returned step or multipliers contain non-finite
        values.
    """

    feasibility_residual: Scalar
    n_refinements: Int[Array, ""]
    projected_grad_norm: Scalar
    reason: KKT_SOLVER_RESULTS
    nonfinite: Bool[Array, ""]

    @classmethod
    def cold(cls, dtype: DTypeLike) -> Self:
        """Zero-iteration carry with residual fields in ``dtype``.

        Parameters
        ----------
        dtype
            Floating dtype of the iterate (fixes the carry dtype for
            ``while_loop`` stability).

        Returns
        -------
        Self
            Cold state: ``n_iter = 0``, ``success = False``,
            ``status = successful``, zero residual, ``inf`` projected
            gradient, ``reason = converged``, ``nonfinite = False``.
        """
        return cls(
            n_iter=jnp.asarray(0, jnp.int32),
            success=jnp.asarray(False),
            status=RESULTS.successful,
            feasibility_residual=jnp.asarray(0.0, dtype),
            n_refinements=jnp.asarray(0, jnp.int32),
            projected_grad_norm=jnp.asarray(jnp.inf, dtype),
            reason=KKT_SOLVER_RESULTS.converged,
            nonfinite=jnp.asarray(False),
        )


class SubProblemSolver(
    InitializableModule, Generic[PrimalType, SubProblemType, SubProblemSolverStateType]
):
    """Matrix-free solver that turns a :class:`~slsqp_jax.sqpdax.subproblem.base.SubProblem`
    into a primal-dual step.

    Concrete subclasses (projected CG, dogleg, Steihaug–Toint, …) specialise
    the three type parameters to their primal / subproblem / state types and
    implement :meth:`solve`.

    Attributes
    ----------
    solver_state_class
        Concrete :class:`SubProblemSolverState` subclass used to seed and
        refresh carry. Marked static so Equinox treats it as configuration.
    """

    solver_state_class: type[SubProblemSolverStateType] = field(static=True)

    @abstractmethod
    def solve(
        self,
        subproblem: SubProblemType,
        x0: tuple[PrimalType, Dual],
        initial_state: SubProblemSolverStateType,
    ) -> tuple[tuple[PrimalType, Dual], SubProblemSolverStateType]:
        """Compute a primal-dual step for ``subproblem``.

        The dual block is returned *in the subproblem's own convention*: the
        multiplier ``λ_{k+1}`` when the subproblem encodes the SQP view
        (Nocedal & Wright eq. 18.9) or the increment ``Δλ`` when it encodes
        the Newton view (eq. 18.6), as declared by
        :attr:`~slsqp_jax.sqpdax.subproblem.base.SubProblem.is_kkt_dual_increment`.
        Solvers must not add ``λ_k`` themselves; the minimiser resolves the
        convention once through
        :meth:`~slsqp_jax.sqpdax.subproblem.base.SubProblem.to_native_step`.
        Solvers whose logic interprets the dual block (sign tests, clamps)
        should reject increment-view subproblems with a ``TypeError`` at trace
        time.

        Parameters
        ----------
        subproblem
            Local KKT / trust-region model at the current iterate.
        x0
            Warm-start ``(primal_step, dual)`` guess. Some solvers ignore the
            primal block (e.g. dogleg) or the dual block (e.g. gradient
            projection).
        initial_state
            Solver carry (radius, iteration counters, optional active set).

        Returns
        -------
        step
            Updated ``(primal_step, dual)`` in the subproblem's convention.
        state
            Refreshed solver carry.
        """

    def requires_secant(self) -> bool:
        """Whether this solver needs a secant Hessian approximation.

        Returns
        -------
        bool
            ``False`` by default. Override when the solver cannot work from
            exact / problem-supplied HVPs alone.
        """
        return False

    def accepts_preconditioner(self) -> bool:
        """Whether :meth:`with_default_preconditioner` can install anything.

        Returns
        -------
        bool
            ``False`` by default. Solvers with a ``preconditioner`` field (or
            wrapping such a solver) override this.
        """
        return False

    def with_default_preconditioner(
        self, preconditioner: Preconditioner | None
    ) -> Self:
        """Install ``preconditioner`` unless one is already configured.

        Minimisers call this every outer step with a preconditioner rebuilt
        from the current curvature information. A preconditioner the user
        configured explicitly always wins.

        Parameters
        ----------
        preconditioner
            Candidate preconditioner, or ``None`` for no change.

        Returns
        -------
        Self
            ``self`` unchanged by default.
        """
        return self


SolverT = TypeVar("SolverT", bound=SubProblemSolver)


def install_default_preconditioner(
    solver: SolverT, preconditioner: Preconditioner | None
) -> SolverT:
    """Set ``solver.preconditioner`` when it is unset and a candidate exists.

    Shared implementation of
    :meth:`SubProblemSolver.with_default_preconditioner` for solvers that
    declare a ``preconditioner: Preconditioner | None`` field.

    Parameters
    ----------
    solver
        Solver with a ``preconditioner`` field.
    preconditioner
        Candidate preconditioner, or ``None`` for no change.

    Returns
    -------
    SolverT
        ``solver`` with the candidate installed, or unchanged when it
        already has a preconditioner or the candidate is ``None``.
    """
    if preconditioner is None or getattr(solver, "preconditioner") is not None:
        return solver
    return tree_at(
        lambda s: s.preconditioner,
        solver,
        preconditioner,
        is_leaf=lambda z: z is None,
    )


class SubproblemContext(
    Module, Generic[PrimalType, SubProblemType, SubProblemSolverStateType]
):
    """Everything the high-level minimiser needs to compute a direction.

    Built once per outer step. Holds the unevaluated Lagrangian module (so a
    secant update can re-evaluate it), the frozen local model, a configured
    solver, and the warm-start / carry that :meth:`SubProblemSolver.solve`
    consumes.

    Attributes
    ----------
    problem
        NLP being minimised (needed by the step controller to build its merit).
    lagrangian
        Unevaluated :class:`~slsqp_jax.sqpdax.lagrangian.basic.Lagrangian`
        module at the current secant / barrier configuration.
    subproblem
        Local model wrapping the Lagrangian evaluation at the current iterate.
    solver
        Fully configured (including freshened preconditioner)
        :class:`SubProblemSolver`.
    warm
        Initial ``(Primal, Dual)`` guess for :meth:`SubProblemSolver.solve`.
    state
        Solver carry to thread in (radius / penalty / warm-started active set).
    """

    problem: ProblemProtocol[PrimalType]
    lagrangian: Lagrangian[PrimalType, EvaluatedLagrangian[PrimalType]]
    subproblem: SubProblemType
    solver: SubProblemSolver[PrimalType, SubProblemType, SubProblemSolverStateType]
    warm: tuple[PrimalType, Dual]
    state: SubProblemSolverStateType
