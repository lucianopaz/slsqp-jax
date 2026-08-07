"""Abstract subproblem-solver interface and per-step context carrier."""

from abc import abstractmethod
from typing import Generic, TypeVar

from equinox import Module, field
from lineax._solution import RESULTS

from ...dual import Dual
from ...lagrangian.basic import Lagrangian
from ...lagrangian.evaluated import EvaluatedLagrangian
from ...primal import PrimalType
from ...problem import ProblemProtocol
from ...types import InitializableModule
from ..base import SubProblemType

__all__ = [
    "RESULTS",
    "SubProblemSolverState",
    "SubProblemSolverStateType",
    "SubProblemSolver",
    "SubproblemContext",
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
            Updated ``(primal_step, dual)``.
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
