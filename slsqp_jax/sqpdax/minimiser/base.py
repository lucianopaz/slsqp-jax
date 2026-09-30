"""Problem-centric constrained minimiser base classes and termination context."""

from __future__ import annotations

import warnings
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import fields, replace
from typing import Any, Generic, Literal, Self, cast, get_origin, get_type_hints

import equinox as eqx
import jax
import optimistix as optx
from equinox import Module
from jax import numpy as jnp
from jaxtyping import Array, Bool, Int

from ..dual import Dual
from ..lagrangian import EvaluatedLagrangian, Lagrangian
from ..preconditioner import (
    NoPreconditioner,
    PreconditionerContext,
    PreconditionerStrategy,
)
from ..primal import Primal, PrimalType
from ..problem import ProblemProtocol
from ..registry import FrozenDict, static_field_names
from ..results import ResultAdapter, ResultType
from ..secant import (
    LBFGS,
    Secant,
    SecantRecoveryState,
    SecantResetPolicy,
    SecantResetSignals,
    SecantStatistics,
)
from ..step_controller import StepController, StepResult
from ..subproblem.base import SubProblemType
from ..subproblem.solver import SubProblemSolver, SubProblemSolverStateType
from ..subproblem.solver.base import SubproblemContext
from ..types import Vector_n
from .termination import (
    TerminationFlags,
    TerminationMetricsType,
    classify_termination,
)
from .utils import minimiser_option_keys, solver_option_keys

__all__ = [
    "OptimisationContext",
    "AbstractConstrainedMinimiser",
    "CommonMinimiser",
]


class OptimisationContext(Module, Generic[PrimalType, SubProblemSolverStateType]):
    """Bundle consumed by convergence / termination hooks.

    Built once per :meth:`~CommonMinimiser.terminate` call from the *current*
    iterate. Carries the NLP, the Lagrangian evaluated at that iterate, and
    the subproblem-solver carry left by the last :meth:`~CommonMinimiser.step`
    (so termination can escalate on a subproblem failure it did not itself
    observe).

    Unlike :class:`~slsqp_jax.sqpdax.subproblem.solver.base.SubproblemContext`
    there is no ``subproblem`` / ``solver`` here: those are phase-1 step
    artifacts that do not exist when ``terminate`` runs in the driver's loop
    condition. Algorithms that need extra state at convergence-check time
    override :meth:`~CommonMinimiser._optimisation_context` to add fields.

    Attributes
    ----------
    problem
        NLP being solved.
    lagrangian
        Lagrangian evaluated at the current iterate (feasibility /
        optimality residuals read ``dual_grad``, constraint values,
        complementarity, ``value``, ``x_grad`` from this).
    solver_state
        Subproblem-solver carry from the last step, or ``None`` before the
        first step.
    """

    problem: ProblemProtocol[PrimalType]
    lagrangian: EvaluatedLagrangian[PrimalType]
    solver_state: SubProblemSolverStateType | None


class AbstractConstrainedMinimiser(
    Module,
    Generic[
        PrimalType,
        SubProblemType,
        SubProblemSolverStateType,
        TerminationMetricsType,
        ResultType,
    ],
):
    """Problem-centric constrained minimiser: fused configuration + running state.

    Unlike ``optimistix.AbstractMinimiser`` (which takes ``fn(y, args) -> scalar``),
    this base ingests a whole
    :class:`~slsqp_jax.sqpdax.problem.basic.ProblemProtocol` so the objective,
    constraints, and their Jacobians travel together. The instance is *both*
    the configured solver and the running state: static ``eqx.field``s hold
    algorithm parameters, while the dynamic fields (``iterate`` / ``dual`` /
    ``secant`` / ``solver_state`` / ``step_count``) are seeded by :meth:`init`
    and advanced by :meth:`step`. Being an :class:`~equinox.Module` it is
    immutable, so every method returns a *new* instance rather than mutating
    in place.

    ``x0`` is a plain :data:`~slsqp_jax.sqpdax.types.Vector_n`; the concrete
    :class:`~slsqp_jax.sqpdax.primal.Primal` (with interior-point slacks when
    applicable) is built inside :meth:`init` via ``_init_primal`` — slacks are
    virtual variables detached from the user's problem, so their default
    values are the solver's responsibility, not the caller's.

    Attributes
    ----------
    iterate
        Current primal iterate, or ``None`` before :meth:`init`.
    dual
        Current multipliers, or ``None`` before :meth:`init`.
    secant
        Maintained curvature approximation, or ``None`` when neither the
        subproblem model nor the preconditioner needs one.
    solver_state
        Subproblem-solver carry threaded across outer steps.
    step_count
        Number of completed outer steps.
    options
        Frozen option bag (``minimiser`` / ``subproblem`` sections).
    """

    # --- dynamic state (None until seeded by ``init``) ---
    iterate: PrimalType | None = None
    dual: Dual | None = None
    secant: Secant | None = None
    solver_state: SubProblemSolverStateType | None = None
    step_count: Int[Array, ""] = eqx.field(
        default_factory=lambda: jnp.asarray(0, jnp.int32)
    )
    options: Mapping[str, Any] = eqx.field(static=True, default_factory=dict)

    @property
    @abstractmethod
    def result_adapter(self) -> ResultAdapter[ResultType]:
        """Native-result construction and Optimistix conversion policy."""
        ...

    def validate_options(self) -> None:
        """Validate ``self.options``.

        No-op on the bare base;
        :class:`~slsqp_jax.sqpdax.minimiser.base.CommonMinimiser` implements
        eager, recursive Schema-B validation against auto-derived option keys.
        """
        return None

    @abstractmethod
    def init(
        self,
        problem: ProblemProtocol[PrimalType],
        x0: Vector_n,
        options: dict | None = None,
    ) -> Self:
        """Seed dynamic state from ``problem`` / ``x0`` / ``options``.

        Parameters
        ----------
        problem
            NLP to minimise.
        x0
            Decision-variable starting point (no slacks).
        options
            Optional ``{"minimiser": {...}, "subproblem": {...}}`` bag.

        Returns
        -------
        Self
            Initialised minimiser instance.
        """
        ...

    @abstractmethod
    def step(self, problem: ProblemProtocol[PrimalType]) -> Self:
        """Take one outer constrained-optimisation step.

        Parameters
        ----------
        problem
            NLP being minimised.

        Returns
        -------
        Self
            Updated minimiser instance.
        """
        ...

    @abstractmethod
    def terminate(
        self, problem: ProblemProtocol[PrimalType]
    ) -> tuple[Bool[Array, ""], ResultType]:
        """Whether the outer loop should stop, and why.

        Parameters
        ----------
        problem
            NLP being minimised.

        Returns
        -------
        done
            ``True`` on convergence or a fatal diagnostic.
        result
            Fine-grained native status code selected by the minimiser.
        """
        ...

    @abstractmethod
    def postprocess(
        self, problem: ProblemProtocol[PrimalType], result: ResultType
    ) -> optx.Solution:
        """Pack the final iterate into an :class:`optimistix.Solution`.

        Parameters
        ----------
        problem
            NLP being minimised.
        result
            Termination status from :meth:`terminate` (possibly remapped by
            the driver on max-steps exhaustion).

        Returns
        -------
        optimistix.Solution
            ``value`` is the decision vector ``iterate.x``.
        """
        ...


class CommonMinimiser(
    AbstractConstrainedMinimiser[
        PrimalType,
        SubProblemType,
        SubProblemSolverStateType,
        TerminationMetricsType,
        ResultType,
    ],
    Generic[
        PrimalType,
        SubProblemType,
        SubProblemSolverStateType,
        TerminationMetricsType,
        ResultType,
    ],
):
    """Shared ``init`` / ``step`` / ``terminate`` / ``postprocess`` driver.

    ``init`` and ``step`` are concrete orchestrators; everything
    algorithm-specific is isolated in a small set of ``_``-prefixed hooks.

    ``init`` (three stages):

    * build the iterate (``_init_primal`` / ``_init_dual``);
    * initialise subproblem-solver-related state (``_make_secant``,
      ``_init_solver_state``, plus ``_init_dynamics`` for extras such as a
      barrier weight);
    * carry subproblem configuration options as static fields.

    ``step`` (four phases):

    1. ``_init_subproblem`` — build the
       :class:`~slsqp_jax.sqpdax.subproblem.solver.base.SubproblemContext`;
    2. ``_solve_direction`` — call ``solver.solve``;
    3. ``_assess_direction`` — run the
       :class:`~slsqp_jax.sqpdax.step_controller.base.StepController`;
    4. ``_execute_step`` — commit iterate / dual / solver state, refresh the
       secant, apply post-iterate updates via ``_advance_dynamics``, and
       finally let :attr:`secant_reset` react to the updated failure
       counters (``_reset_secant``).

    Curvature is configured by three independent components:
    :attr:`curvature` chooses what the subproblem model uses,
    :attr:`preconditioner` builds a per-step preconditioner for iterative
    subproblem solvers, and :attr:`secant_reset` decides when the maintained
    secant is reset. A secant is maintained whenever the model uses it *or*
    the preconditioner requires one, so exact HVPs can drive the QP while
    L-BFGS only preconditions it.

    Attributes
    ----------
    rtol
        Relative stationarity tolerance (scaled by ``max(|L|, 1)``).
    atol
        Absolute feasibility / extra-optimality tolerance.
    min_steps
        Minimum outer steps before convergence may fire.
    secant_memory
        Default L-BFGS memory when no ``minimiser.secant`` kind-spec is given.
    curvature
        Hessian used by the subproblem model: ``"exact"`` Lagrangian HVPs
        (the problem must supply them), the ``"secant"`` approximation, or
        ``"auto"`` (default) for exact when available and secant otherwise.
    preconditioner
        :class:`~slsqp_jax.sqpdax.preconditioner.strategy.PreconditionerStrategy`
        rebuilt every step; set through
        ``options['minimiser']['preconditioner']`` as a ``{"kind": ...}``
        spec, a mapping of the current strategy's fields, or an instance.
        Default ``"none"``.
    secant_reset
        :class:`~slsqp_jax.sqpdax.secant.reset.SecantResetPolicy` applied
        after every step; configure with a mapping of its fields. Each
        minimiser feeds the policy through
        :meth:`_secant_reset_signals`, mapping its own counters onto the
        ``subproblem`` / ``step`` / ``model`` channels of
        :class:`~slsqp_jax.sqpdax.secant.reset.SecantResetSignals` (the
        active-set loops use the first two, the trust-region interior-point
        loop the last).
    secant_stats
        :class:`~slsqp_jax.sqpdax.secant.statistics.SecantStatistics` of the
        maintained secant, or ``None`` when no secant is kept.
    secant_recovery_state
        Dynamic global recovery episode shared by all failure channels.
    """

    solver_state: SubProblemSolverStateType | None = None
    secant_stats: SecantStatistics | None = None
    secant_recovery_state: SecantRecoveryState = eqx.field(
        default_factory=SecantRecoveryState
    )
    # --- convergence / secant configuration ---
    rtol: float = eqx.field(static=True, default=1e-6)
    atol: float = eqx.field(static=True, default=1e-6)
    min_steps: int = eqx.field(static=True, default=1)
    secant_memory: int = eqx.field(static=True, default=10)
    curvature: Literal["auto", "exact", "secant"] = eqx.field(
        static=True, default="auto"
    )
    preconditioner: PreconditionerStrategy = eqx.field(
        static=True, default_factory=NoPreconditioner
    )
    secant_reset: SecantResetPolicy = eqx.field(
        static=True, default_factory=SecantResetPolicy
    )

    def __check_init__(self) -> None:
        if self.curvature not in ("auto", "exact", "secant"):
            raise ValueError(
                f"curvature must be 'auto', 'exact' or 'secant'; got {self.curvature!r}"
            )

    # ================================ init =================================

    @abstractmethod
    def _subproblem_solver_type(self) -> type[SubProblemSolver]:
        """Root subproblem-solver class whose fields define option keys.

        Returns
        -------
        type
            :class:`~slsqp_jax.sqpdax.subproblem.solver.base.SubProblemSolver`
            subclass used for ``options['subproblem']`` validation
            (auto-derived as ``fields - lagrangian``).
        """
        ...

    def validate_options(self) -> None:
        """Warn on any option key this solver would not consume.

        Recognised keys are derived from class structure: the ``minimiser``
        section accepts static ``eqx.field`` names plus any ``kind``-family
        fields (``secant`` / ``barrier_update``); the ``subproblem`` section
        accepts the subproblem-solver's fields minus ``lagrangian``,
        recursing into nested ``SubProblemSolver`` fields.
        """
        opts = self.options
        for k in sorted(set(opts) - {"minimiser", "subproblem"}):
            warnings.warn(
                f"{type(self).__name__}: ignoring unknown option section '{k}'"
            )
        m_allowed = minimiser_option_keys(type(self))
        for k in opts.get("minimiser", {}):
            if k not in m_allowed:
                warnings.warn(
                    f"{type(self).__name__}: ignoring unknown minimiser option '{k}'"
                )
        self._validate_solver_options(
            self._subproblem_solver_type(), opts.get("subproblem", {})
        )

    def _validate_solver_options(
        self, solver_type: type[SubProblemSolver], section: Mapping
    ) -> None:
        """Recursively warn on unknown keys in a subproblem option section.

        Parameters
        ----------
        solver_type
            Solver class whose fields define the allowed keys.
        section
            Mapping of option keys to values (nested mappings recurse into
            nested ``SubProblemSolver`` fields).
        """
        allowed = solver_option_keys(solver_type)
        hints = get_type_hints(solver_type)
        field_by_name = {f.name: f for f in fields(cast(Any, solver_type))}
        for k, v in section.items():
            if k not in allowed:
                warnings.warn(
                    f"{solver_type.__name__}: ignoring unknown subproblem option '{k}'"
                )
                continue
            hint = hints.get(k)
            origin = get_origin(hint) if hint is not None else None
            base = origin if origin is not None else hint
            if not (
                isinstance(base, type)
                and issubclass(base, SubProblemSolver)
                and isinstance(v, Mapping)
            ):
                continue
            # Prefer a concrete ``default_factory`` (e.g. projected CG) so
            # nested keys are checked against the configured solver class.
            nested_cls: type[SubProblemSolver] = base
            field = field_by_name.get(k)
            factory = (
                getattr(field, "default_factory", None) if field is not None else None
            )
            if isinstance(factory, type) and issubclass(factory, SubProblemSolver):
                nested_cls = factory
            self._validate_solver_options(nested_cls, v)

    def _parse_options(self, options: dict | None) -> Self:
        """Freeze ``options`` and map recognised minimiser static fields.

        The option bag is stored as a hashable :class:`~slsqp_jax.sqpdax.registry.FrozenDict`
        so the module stays a valid static-aux ``while_loop`` carry. Recognised
        ``minimiser`` static-field keys are applied via
        :func:`dataclasses.replace`. Concrete solvers may override to also
        build ``kind``-family fields (e.g. ``barrier_update``).

        Parameters
        ----------
        options
            Raw option dict, or ``None`` for defaults.

        Returns
        -------
        Self
            Copy with frozen ``options`` and any static-field updates applied.
        """
        frozen = FrozenDict({} if options is None else options)
        base = replace(self, options=frozen)
        mopts = frozen.get("minimiser", {})
        static_names = static_field_names(type(self)) - {
            "preconditioner",
            "secant_reset",
        }
        updates = {k: mopts[k] for k in mopts if k in static_names}
        if "preconditioner" in mopts:
            updates["preconditioner"] = self._parse_preconditioner(
                mopts["preconditioner"]
            )
        if "secant_reset" in mopts:
            spec = mopts["secant_reset"]
            updates["secant_reset"] = (
                self.secant_reset.init(**spec) if isinstance(spec, Mapping) else spec
            )
        if updates:
            base = replace(base, **updates)
        return base

    def _parse_preconditioner(self, spec: Any) -> PreconditionerStrategy:
        """Resolve ``options['minimiser']['preconditioner']`` into a strategy.

        Parameters
        ----------
        spec
            A :class:`~slsqp_jax.sqpdax.preconditioner.strategy.PreconditionerStrategy`
            (used as is), a ``{"kind": ..., **params}`` mapping (built from
            the registry), or a mapping without ``kind`` (applied to the
            current strategy's fields).

        Returns
        -------
        PreconditionerStrategy
            Configured strategy.

        Raises
        ------
        TypeError
            If ``spec`` is neither a strategy nor a mapping.
        """
        if isinstance(spec, PreconditionerStrategy):
            return spec
        if isinstance(spec, Mapping):
            if "kind" in spec:
                return cast(
                    PreconditionerStrategy, PreconditionerStrategy.from_spec(spec)
                )
            return self.preconditioner.init(**spec)
        raise TypeError(
            "options['minimiser']['preconditioner'] must be a PreconditionerStrategy "
            f"or a mapping; got {type(spec).__name__}"
        )

    def _init_primal(
        self, problem: ProblemProtocol[PrimalType], x0: Vector_n
    ) -> PrimalType:
        """Build the initial primal (interior-point solvers add slacks).

        Parameters
        ----------
        problem
            NLP being minimised.
        x0
            Decision-variable starting point.

        Returns
        -------
        PrimalType
            Initial primal pytree.
        """
        return cast(PrimalType, Primal(x=x0))

    def _init_dual(self, problem: ProblemProtocol[PrimalType]) -> Dual:
        """Default zero multipliers; override for primal-dual positivity.

        Parameters
        ----------
        problem
            NLP providing ``n`` / ``meq`` / ``mineq``.

        Returns
        -------
        Dual
            Zero multiplier vector.
        """
        return cast(
            Dual,
            Dual(
                eq_multipliers=jnp.zeros(problem.meq),
                ineq_multipliers=jnp.zeros(problem.mineq),
                lb_multipliers=jnp.zeros(problem.n),
                ub_multipliers=jnp.zeros(problem.n),
            ),
        )

    @abstractmethod
    def _init_solver_state(
        self, problem: ProblemProtocol[PrimalType], primal: PrimalType
    ) -> SubProblemSolverStateType:
        """Seed the subproblem-solver state (radius / penalty / warm set).

        Parameters
        ----------
        problem
            NLP being minimised.
        primal
            Initial primal from :meth:`_init_primal`.

        Returns
        -------
        SubProblemSolverStateType
            Cold solver carry.
        """
        ...

    def _make_secant(self, problem: ProblemProtocol[PrimalType]) -> Secant:
        """Curvature approximation seeded at init (diagonal L-BFGS by default).

        A ``minimiser.secant`` ``kind``-spec selects/parameterises an
        alternative registered :class:`~slsqp_jax.sqpdax.secant.base.Secant`.
        Only ``n`` is force-injected; :attr:`secant_memory` supplies
        ``memory`` unless the spec overrides it.

        Parameters
        ----------
        problem
            NLP providing dimension ``n``.

        Returns
        -------
        Secant
            Initialised curvature approximation.
        """
        spec = self.options.get("minimiser", {}).get("secant")
        if spec is not None:
            spec = dict(spec)
            spec.setdefault("memory", self.secant_memory)
            return cast(Secant, Secant.from_spec(spec, n=problem.n))
        return cast(Secant, LBFGS(n=problem.n, memory=self.secant_memory))

    def _init_dynamics(
        self, problem: ProblemProtocol[PrimalType], x0: Vector_n
    ) -> Self:
        """Seed dynamic state that ``step`` needs before the iterate is built.

        Owns the secant: a curvature approximation is created via
        :meth:`_make_secant` when :meth:`_maintains_secant` holds, together
        with its :attr:`secant_stats`. Overrides should call
        ``super()._init_dynamics(...)`` first, then seed any extra state
        (e.g. barrier weight ``μ``).

        Parameters
        ----------
        problem
            NLP being minimised.
        x0
            Decision-variable starting point.

        Returns
        -------
        Self
            Copy with ``secant`` / ``secant_stats`` (and any subclass extras)
            seeded.

        Raises
        ------
        ValueError
            If the configuration needs exact HVPs the problem lacks.
        """
        self._validate_curvature(problem)
        secant = self._make_secant(problem) if self._maintains_secant(problem) else None
        stats = (
            None
            if secant is None
            else SecantStatistics.initial(secant, jnp.result_type(x0.dtype, float))
        )
        return eqx.tree_at(
            lambda m: (m.secant, m.secant_stats, m.secant_recovery_state),
            self,
            (secant, stats, SecantRecoveryState()),
            is_leaf=lambda z: z is None,
        )

    def _validate_curvature(self, problem: ProblemProtocol[PrimalType]) -> None:
        """Reject configurations that need exact HVPs the problem lacks.

        Parameters
        ----------
        problem
            NLP being minimised.

        Raises
        ------
        ValueError
            If ``curvature="exact"`` or the preconditioner strategy requires
            exact HVPs while ``problem.has_exact_curvature`` is false.
        """
        if problem.has_exact_curvature:
            return
        if self.curvature == "exact":
            raise ValueError(
                f"{type(self).__name__}: curvature='exact' requires a problem "
                "with exact Hessian-vector products"
            )
        if self.preconditioner.requires_exact_hvp:
            raise ValueError(
                f"{type(self).__name__}: preconditioner "
                f"'{self.preconditioner.kind}' requires a problem with exact "
                "Hessian-vector products"
            )

    def _uses_secant_model(self, problem: ProblemProtocol[PrimalType]) -> bool:
        """Whether the subproblem model uses the secant instead of exact HVPs.

        Parameters
        ----------
        problem
            NLP being minimised.

        Returns
        -------
        bool
            Resolution of :attr:`curvature` for ``problem``.
        """
        if self.curvature == "auto":
            return not problem.has_exact_curvature
        return self.curvature == "secant"

    def _maintains_secant(self, problem: ProblemProtocol[PrimalType]) -> bool:
        """Whether a secant is kept (for the model or for preconditioning).

        Parameters
        ----------
        problem
            NLP being minimised.

        Returns
        -------
        bool
            ``True`` when the model uses the secant or the preconditioner
            strategy requires one.
        """
        return self._uses_secant_model(problem) or self.preconditioner.requires_secant

    def _model_secant(self, problem: ProblemProtocol[PrimalType]) -> Secant | None:
        """Secant handed to the subproblem Lagrangian, or ``None`` for exact HVPs.

        Parameters
        ----------
        problem
            NLP being minimised.

        Returns
        -------
        Secant or None
            :attr:`secant` when the model uses it, otherwise ``None`` (even
            if a secant is maintained for preconditioning).
        """
        return self.secant if self._uses_secant_model(problem) else None

    def _precondition(
        self,
        solver: SubProblemSolver,
        problem: ProblemProtocol[PrimalType],
        lagrangian: EvaluatedLagrangian[PrimalType],
    ) -> SubProblemSolver:
        """Install this step's :attr:`preconditioner` into ``solver``.

        Parameters
        ----------
        solver
            Fully configured subproblem solver (options already applied).
        problem
            NLP being minimised.
        lagrangian
            Lagrangian evaluated at the current iterate (with or without the
            model secant; the exact view is derived from it).

        Returns
        -------
        SubProblemSolver
            ``solver`` with the freshly built preconditioner installed where
            it accepts one (a user-configured preconditioner is kept), or
            ``solver`` unchanged when the strategy is inactive or the solver
            accepts no preconditioner (which warns, since the configured
            strategy would be ignored).
        """
        strategy = self.preconditioner
        if not strategy.is_active:
            return solver
        if not solver.accepts_preconditioner():
            warnings.warn(
                f"{type(self).__name__}: preconditioner '{strategy.kind}' is "
                f"ignored because {type(solver).__name__} accepts no "
                "preconditioner"
            )
            return solver
        ctx = cast(
            PreconditionerContext,
            PreconditionerContext(
                x_ref=lagrangian.x_ref,
                secant=self.secant,
                lagrangian=(
                    lagrangian.without_secant() if problem.has_exact_curvature else None
                ),
                step_count=self.step_count,
            ),
        )
        return solver.with_default_preconditioner(strategy.build(ctx))

    def init(
        self,
        problem: ProblemProtocol[PrimalType],
        x0: Vector_n,
        options: dict | None = None,
    ) -> Self:
        """Parse options, seed dynamics, and install the initial iterate.

        Parameters
        ----------
        problem
            NLP to minimise.
        x0
            Decision-variable starting point.
        options
            Optional configuration bag.

        Returns
        -------
        Self
            Fully initialised minimiser.
        """
        x0 = jnp.asarray(x0)
        base = self._parse_options(options)
        base.validate_options()
        base = base._init_dynamics(problem, x0)

        primal = base._init_primal(problem, x0)
        dual = base._init_dual(problem)
        solver_state = base._init_solver_state(problem, primal)
        return base._close_init(primal, dual, solver_state, problem)

    def _close_init(
        self,
        primal: PrimalType,
        dual: Dual,
        solver_state: SubProblemSolverStateType,
        problem: ProblemProtocol[PrimalType],
    ) -> Self:
        """Close the initialisation phase.

        Parameters
        ----------
        primal
            Initial primal.
        dual
            Initial dual.
        solver_state
            Initial solver state.
        problem
            NLP being minimised.

        Returns
        -------
        Self
            Copy with ``iterate``, ``dual``, and ``solver_state`` seeded.
        """
        return eqx.tree_at(
            lambda m: (m.iterate, m.dual, m.solver_state, m.step_count),
            self,
            (primal, dual, solver_state, jnp.asarray(0, jnp.int32)),
            is_leaf=lambda z: z is None,
        )

    # ================================ step ================================

    def step(self, problem: ProblemProtocol[PrimalType]) -> Self:
        """Run one outer step (setup → solve → assess → execute).

        Parameters
        ----------
        problem
            NLP being minimised.

        Returns
        -------
        Self
            Updated minimiser after the controlled step.
        """
        ctx = self._init_subproblem(problem)  # 1 setup
        step_primal, step_dual, solver_state = self._solve_direction(ctx)  # 2 solve
        result = self._assess_direction(  # 3 assess
            ctx, step_primal, step_dual, solver_state
        )
        return self._execute_step(ctx, step_dual, result)  # 4 execute

    @abstractmethod
    def _init_subproblem(
        self, problem: ProblemProtocol[PrimalType]
    ) -> SubproblemContext[PrimalType, SubProblemType, SubProblemSolverStateType]:
        """Build the per-step :class:`SubproblemContext`.

        Parameters
        ----------
        problem
            NLP being minimised.

        Returns
        -------
        SubproblemContext
            Frozen model, configured solver, warm start, and carry.
        """
        ...

    @abstractmethod
    def _lagrangian_module(
        self, problem: ProblemProtocol[PrimalType]
    ) -> Lagrangian[PrimalType, EvaluatedLagrangian[PrimalType]]:
        """Unevaluated Lagrangian module at the current secant / barrier.

        Parameters
        ----------
        problem
            NLP being minimised.

        Returns
        -------
        Lagrangian
            Callable that evaluates at ``(primal, dual)``.
        """
        ...

    def _solve_direction(
        self,
        ctx: SubproblemContext[PrimalType, SubProblemType, SubProblemSolverStateType],
    ) -> tuple[PrimalType, Dual, SubProblemSolverStateType]:
        """Call the subproblem solver for a primal-dual direction.

        The solver's answer is passed through
        :meth:`~slsqp_jax.sqpdax.subproblem.base.SubProblem.to_native_step`,
        which resolves the subproblem's dual convention
        (:attr:`~slsqp_jax.sqpdax.subproblem.base.SubProblem.is_kkt_dual_increment`)
        so ``step_dual`` is always the absolute multiplier ``λ_{k+1}`` to commit.

        Parameters
        ----------
        ctx
            Per-step subproblem context.

        Returns
        -------
        step_primal
            Proposed primal step.
        step_dual
            Absolute multipliers ``λ_{k+1}`` (updated or recovered).
        state
            Refreshed solver carry.
        """
        step, state = ctx.solver.solve(
            subproblem=ctx.subproblem, x0=ctx.warm, initial_state=ctx.state
        )
        step_primal, step_dual = ctx.subproblem.to_native_step(step)
        return step_primal, step_dual, state

    def _assess_direction(
        self,
        ctx: SubproblemContext[PrimalType, SubProblemType, SubProblemSolverStateType],
        step_primal: PrimalType,
        step_dual: Dual,
        solver_state: SubProblemSolverStateType,
    ) -> StepResult[PrimalType, SubProblemSolverStateType]:
        """Run the step controller (accept / reject + control updates).

        Parameters
        ----------
        ctx
            Per-step subproblem context.
        step_primal
            Proposed primal step.
        step_dual
            Multipliers associated with the proposal.
        solver_state
            Solver carry after the subproblem solve.

        Returns
        -------
        StepResult
            Accepted (or retained) iterate and updated solver carry.
        """
        controller = self._step_controller(ctx, step_dual, solver_state)
        iterate = cast(PrimalType, self.iterate)
        return controller.step(iterate, step_primal, solver_state)

    @abstractmethod
    def _step_controller(
        self,
        ctx: SubproblemContext[PrimalType, SubProblemType, SubProblemSolverStateType],
        step_dual: Dual,
        solver_state: SubProblemSolverStateType,
    ) -> StepController[PrimalType, SubProblemSolverStateType]:
        """Build the step controller for this outer iteration.

        Parameters
        ----------
        ctx
            Per-step subproblem context.
        step_dual
            Multipliers from the subproblem solve (e.g. for merit penalty).
        solver_state
            Solver carry (e.g. trust-region radius / predicted reduction).

        Returns
        -------
        StepController
            Line search or trust-region manager for this step.
        """
        ...

    def _execute_step(
        self,
        ctx: SubproblemContext[PrimalType, SubProblemType, SubProblemSolverStateType],
        step_dual: Dual,
        result: StepResult[PrimalType, SubProblemSolverStateType],
    ) -> Self:
        """Commit the controlled step and refresh secant / dynamics.

        The multiplier estimate ``step_dual`` is a function of the *current*
        iterate (a KKT / least-squares / proximal-point estimate), so it is
        committed independently of whether the primal step was accepted —
        this is what lets the interior-point and proximal loops correct a
        wrong ``λ`` at a primal-stationary point through rejected steps. The
        only exception is a non-finite estimate (e.g. from a NaN subproblem
        direction, which the step controller refuses): committing it would
        make the next termination check report ``nonfinite`` even though the
        iterate never moved, so the previous multipliers are kept instead and
        the failure is left to the subproblem-failure counters.

        Parameters
        ----------
        ctx
            Per-step subproblem context (provides the unevaluated Lagrangian).
        step_dual
            Multipliers proposed by the subproblem; committed when finite.
        result
            Outcome of :meth:`_assess_direction`.

        Returns
        -------
        Self
            Minimiser after :meth:`_advance_dynamics`.
        """
        x_new = result.x
        dual_finite = jnp.all(
            jnp.stack(
                [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(step_dual)]
            )
        )
        committed_dual = cast(
            Dual,
            jax.tree.map(
                lambda new, old: jnp.where(dual_finite, new, old),
                step_dual,
                cast(Dual, self.dual),
            ),
        )
        new_secant, new_stats = self._update_secant(
            ctx.lagrangian, result, committed_dual
        )
        advanced = cast(
            Self,
            eqx.tree_at(
                lambda m: (
                    m.iterate,
                    m.dual,
                    m.secant,
                    m.secant_stats,
                    m.solver_state,
                    m.step_count,
                ),
                self,
                (
                    x_new,
                    committed_dual,
                    new_secant,
                    new_stats,
                    result.solver_state,
                    self.step_count + 1,
                ),
                is_leaf=lambda z: z is None,
            ),
        )
        advanced = advanced._advance_dynamics(ctx, result, committed_dual)
        return advanced._reset_secant()

    def _secant_reset_signals(self) -> SecantResetSignals:
        """Failure streaks reported to :attr:`secant_reset` after a step.

        Called on the minimiser *after* :meth:`_advance_dynamics`, so
        overrides read the counters of the step just taken. The default
        reports no failures on any channel, leaving only the conditioning
        trigger; it is only consulted when a secant is maintained.

        Returns
        -------
        SecantResetSignals
            Current raw failure streaks.
        """
        return SecantResetSignals.none()  # pragma: no cover

    def _reset_secant(self) -> Self:
        """Apply :attr:`secant_reset` to the maintained secant and record it.

        Returns
        -------
        Self
            Minimiser with a possibly reset ``secant`` and updated
            ``secant_stats``; unchanged when no secant is kept.
        """
        if self.secant is None:
            return self
        secant, recovery, severity = self.secant_reset.apply(
            self.secant,
            self._secant_reset_signals(),
            self.secant_recovery_state,
        )
        stats = cast(SecantStatistics, self.secant_stats).record_reset(severity, secant)
        return eqx.tree_at(
            lambda m: (m.secant, m.secant_stats, m.secant_recovery_state),
            self,
            (secant, stats, recovery),
        )

    def _update_secant(
        self,
        lagrangian: Lagrangian[PrimalType, EvaluatedLagrangian[PrimalType]],
        result: StepResult[PrimalType, SubProblemSolverStateType],
        step_dual: Dual,
    ) -> tuple[Secant | None, SecantStatistics | None]:
        """Append a usable accepted curvature pair, or preserve the secant.

        Parameters
        ----------
        lagrangian
            Unevaluated Lagrangian module (uses current ``secant``).
        result
            Controlled-step result. Rejected, zero-displacement, and
            non-finite-curvature steps leave both secant and append
            statistics unchanged.
        step_dual
            Multipliers shared at both secant endpoints (N&W §18.3); the
            committed dual (the previous one if the proposal was non-finite).

        Returns
        -------
        secant
            Updated approximation, or ``None`` when no secant is active.
        stats
            :attr:`secant_stats` with the attempted append recorded, or
            ``None`` when no secant is active.
        """
        if self.secant is None:
            return None, None
        iterate = cast(PrimalType, self.iterate)
        x_new = result.x
        s = x_new.x - iterate.x
        displacement_usable = (
            result.accepted & jnp.all(jnp.isfinite(s)) & jnp.any(s != 0)
        )
        secant0 = self.secant
        stats0 = cast(SecantStatistics, self.secant_stats)

        def maybe_append(_: None) -> tuple[Secant, SecantStatistics]:
            # Share one freshly solved multiplier vector at both endpoints
            # (N&W 18.3); constant bound Jacobians then cancel from y.
            prev = lagrangian(iterate, step_dual)
            y = lagrangian.curvature_estimate(x_new, prev)
            finite_curvature = jnp.all(jnp.isfinite(y))

            def append(_: None) -> tuple[Secant, SecantStatistics]:
                diagnostics = secant0.diagnostics(s, y)
                secant = secant0.append(s, y)
                stats = stats0.record_append(diagnostics, secant)
                return secant, stats

            return jax.lax.cond(
                finite_curvature,
                append,
                lambda _: (secant0, stats0),
                operand=None,
            )

        return jax.lax.cond(
            displacement_usable,
            maybe_append,
            lambda _: (secant0, stats0),
            operand=None,
        )

    @abstractmethod
    def _advance_dynamics(
        self,
        ctx: SubproblemContext[PrimalType, SubProblemType, SubProblemSolverStateType],
        result: StepResult[PrimalType, SubProblemSolverStateType],
        step_dual: Dual,
    ) -> Self:
        """Phase-4 post-iterate parameter update.

        Default implementations are no-ops (return ``self``). Interior-point
        solvers override to reduce the barrier weight.

        Runs on rejected steps too: ``result.x`` is the retained iterate in
        that case, and tests evaluated at the iterate (rather than at the
        step) are still meaningful there. Overrides that must distinguish the
        two cases should branch on ``result.accepted``.

        Parameters
        ----------
        ctx
            Per-step subproblem context.
        result
            Outcome of :meth:`_assess_direction`, carrying the committed
            iterate ``result.x``, the acceptance flag, and the refreshed
            solver carry.
        step_dual
            Committed multipliers: the subproblem's estimate when finite, the
            previous ones otherwise.

        Returns
        -------
        Self
            Minimiser with any post-iterate parameters updated.
        """
        ...

    def _evaluated_lagrangian(
        self, problem: ProblemProtocol[PrimalType]
    ) -> EvaluatedLagrangian[PrimalType]:
        """Evaluate the Lagrangian at the current ``(iterate, dual)``.

        Parameters
        ----------
        problem
            NLP being minimised.

        Returns
        -------
        EvaluatedLagrangian
            Pointwise Lagrangian quantities at the current iterate.
        """
        iterate = cast(PrimalType, self.iterate)
        dual = cast(Dual, self.dual)
        return self._lagrangian_module(problem)(iterate, dual)

    def _optimisation_context(
        self, problem: ProblemProtocol[PrimalType]
    ) -> OptimisationContext[PrimalType, SubProblemSolverStateType]:
        """Build the context the convergence / termination test consumes.

        Default bundles ``problem``, the evaluated Lagrangian, and the
        ``solver_state`` left behind by the last ``step``; override to attach
        extra state (e.g. a subproblem projector).

        Parameters
        ----------
        problem
            NLP being minimised.

        Returns
        -------
        OptimisationContext
            Bundle for :meth:`terminate` hooks.
        """
        return cast(
            OptimisationContext[PrimalType, SubProblemSolverStateType],
            OptimisationContext(
                problem=problem,
                lagrangian=self._evaluated_lagrangian(problem),
                solver_state=self.solver_state,
            ),
        )

    def terminate(
        self, problem: ProblemProtocol[PrimalType]
    ) -> tuple[Bool[Array, ""], ResultType]:
        """Test convergence and failure diagnostics at the current iterate.

        Pure orchestration: the algorithm-specific work lives in
        :meth:`termination_metrics` and :meth:`termination_flags`, and the
        mapping from flags to a status code lives in
        :func:`~slsqp_jax.sqpdax.minimiser.termination.classify_termination`
        so it stays identical across minimisers.

        Parameters
        ----------
        problem
            NLP being minimised.

        Returns
        -------
        done
            ``True`` on convergence or a fired failure diagnostic.
        result
            Status code (``successful`` while still running or on clean
            convergence; a failure code when a diagnostic fires).
        """
        ctx = self._optimisation_context(problem)
        metrics = self.termination_metrics(ctx)
        flags = self.termination_flags(ctx, metrics)
        return classify_termination(flags, self.result_adapter)

    @abstractmethod
    def termination_metrics(
        self, ctx: OptimisationContext[PrimalType, SubProblemSolverStateType]
    ) -> TerminationMetricsType:
        """Measure the quantities this algorithm's convergence test consumes.

        The returned type is the minimiser's own
        :class:`~slsqp_jax.sqpdax.minimiser.termination.TerminationMetrics`
        subclass, so an algorithm needing several residuals or tolerances
        (inexact SQP, N&W Algorithm 19.4) is not forced into a shared schema.

        Parameters
        ----------
        ctx
            Termination context at the current iterate.

        Returns
        -------
        TerminationMetricsType
            Populated metrics for :meth:`termination_flags`.
        """
        ...

    @abstractmethod
    def termination_flags(
        self,
        ctx: OptimisationContext[PrimalType, SubProblemSolverStateType],
        metrics: TerminationMetricsType,
    ) -> TerminationFlags[ResultType]:
        """Reduce this algorithm's metrics to the shared termination booleans.

        This is where algorithm-specific diagnostics are reduced to the
        shared flags:
        :func:`~slsqp_jax.sqpdax.minimiser.termination.classify_termination`
        consumes only :class:`~slsqp_jax.sqpdax.minimiser.termination.TerminationFlags`
        and the concrete native-result adapter.

        Parameters
        ----------
        ctx
            Termination context at the current iterate.
        metrics
            Output of :meth:`termination_metrics` at the same iterate.

        Returns
        -------
        TerminationFlags
            Convergence / non-finite / fatal decision, plus the native result
            to report if the fatal branch fires.
        """
        ...

    def _postprocess_stats(
        self, problem: ProblemProtocol[PrimalType], result: ResultType
    ) -> dict[str, Any]:
        """Build algorithm-specific solution statistics.

        Always reports ``num_steps`` and the global recovery state; adds the
        remaining ``secant_*`` statistics when a secant is maintained.
        """
        stats: dict[str, Any] = {
            "num_steps": self.step_count,
            "secant_recovery_streak": self.secant_recovery_state.failure_streak,
            "secant_recovery_stage": self.secant_recovery_state.stage,
            "secant_recovery_fatal": self.secant_recovery_state.fatal,
        }
        if self.secant_stats is not None:
            stats.update(self.secant_stats.as_stats())
        return stats

    def postprocess(
        self, problem: ProblemProtocol[PrimalType], result: ResultType
    ) -> optx.Solution:
        """Pack ``iterate.x`` into an :class:`optimistix.Solution`.

        Parameters
        ----------
        problem
            NLP being minimised; its objective is evaluated at the final
            iterate to populate :attr:`optimistix.Solution.aux`.
        result
            Final status code.

        Returns
        -------
        optimistix.Solution
            ``value`` is the decision vector; ``aux`` and
            ``stats["final_objective"]`` come from the objective at that
            vector.

        Raises
        ------
        ValueError
            If :meth:`init` has not been called (``iterate is None``).
        """
        if self.iterate is None:
            raise ValueError("iterate is not set")
        objective_value, aux = problem.fn(self.iterate.x)
        stats = self._postprocess_stats(problem, result)
        stats.setdefault("final_objective", objective_value)
        return cast(
            optx.Solution,
            optx.Solution(
                value=self.iterate.x,
                result=result,
                aux=aux,
                stats=stats,
                state=self,
            ),
        )
