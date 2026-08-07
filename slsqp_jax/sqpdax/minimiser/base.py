"""Problem-centric constrained minimiser base classes and termination context."""

from __future__ import annotations

import warnings
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import fields, replace
from typing import Any, Generic, Self, cast, get_origin, get_type_hints

import equinox as eqx
import jax
import optimistix as optx
from equinox import Module
from jax import numpy as jnp
from jaxtyping import Array, Bool, Int

from ..barrier.update import _inf_norm
from ..dual import Dual
from ..lagrangian import EvaluatedLagrangian, Lagrangian
from ..primal import Primal, PrimalType
from ..problem import ProblemProtocol
from ..registry import FrozenDict, static_field_names
from ..secant import LBFGS, Secant
from ..step_controller import StepController, StepResult
from ..subproblem.base import SubProblemType
from ..subproblem.solver import SubProblemSolver, SubProblemSolverStateType
from ..subproblem.solver.base import SubproblemContext
from ..types import Scalar, Vector_n
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
    Module, Generic[PrimalType, SubProblemType, SubProblemSolverStateType]
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
        Optional curvature approximation, or ``None`` when the problem
        supplies exact HVPs.
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
    ) -> tuple[Bool[Array, ""], optx.RESULTS]:
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
            :class:`optimistix.RESULTS` status code.
        """
        ...

    @abstractmethod
    def postprocess(
        self, problem: ProblemProtocol[PrimalType], result: optx.RESULTS
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
    AbstractConstrainedMinimiser[PrimalType, SubProblemType, SubProblemSolverStateType],
    Generic[PrimalType, SubProblemType, SubProblemSolverStateType],
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
       secant, and apply post-iterate updates via ``_advance_dynamics``.

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
    """

    solver_state: SubProblemSolverStateType | None = None
    # --- convergence / secant configuration ---
    rtol: float = eqx.field(static=True, default=1e-6)
    atol: float = eqx.field(static=True, default=1e-6)
    min_steps: int = eqx.field(static=True, default=1)
    secant_memory: int = eqx.field(static=True, default=10)

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
        static_names = static_field_names(type(self))
        updates = {k: mopts[k] for k in mopts if k in static_names}
        if updates:
            base = replace(base, **updates)
        return base

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
        :meth:`_make_secant` unless the problem supplies exact curvature.
        Overrides should call ``super()._init_dynamics(...)`` first, then
        seed any extra state (e.g. barrier weight ``μ``).

        Parameters
        ----------
        problem
            NLP being minimised.
        x0
            Decision-variable starting point.

        Returns
        -------
        Self
            Copy with ``secant`` (and any subclass extras) seeded.
        """
        secant = None if problem.has_exact_curvature else self._make_secant(problem)
        return eqx.tree_at(
            lambda m: m.secant, self, secant, is_leaf=lambda z: z is None
        )

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
        return eqx.tree_at(
            lambda m: (m.iterate, m.dual, m.solver_state, m.step_count),
            base,
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

        Parameters
        ----------
        ctx
            Per-step subproblem context.

        Returns
        -------
        step_primal
            Proposed primal step.
        step_dual
            Updated (or recovered) multipliers.
        state
            Refreshed solver carry.
        """
        (step_primal, step_dual), state = ctx.solver.solve(
            subproblem=ctx.subproblem, x0=ctx.warm, initial_state=ctx.state
        )
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

        Parameters
        ----------
        ctx
            Per-step subproblem context (provides the unevaluated Lagrangian).
        step_dual
            Multipliers committed with the new iterate.
        result
            Outcome of :meth:`_assess_direction`.

        Returns
        -------
        Self
            Minimiser after :meth:`_advance_dynamics`.
        """
        x_new = result.x
        new_secant = self._update_secant(ctx.lagrangian, x_new, step_dual)
        advanced = eqx.tree_at(
            lambda m: (m.iterate, m.dual, m.secant, m.solver_state, m.step_count),
            self,
            (x_new, step_dual, new_secant, result.solver_state, self.step_count + 1),
            is_leaf=lambda z: z is None,
        )
        return advanced._advance_dynamics(ctx, x_new, step_dual)

    def _update_secant(
        self,
        lagrangian: Lagrangian[PrimalType, EvaluatedLagrangian[PrimalType]],
        x_new: PrimalType,
        step_dual: Dual,
    ) -> Secant | None:
        """Append a curvature pair at shared multipliers, or keep ``None``.

        Parameters
        ----------
        lagrangian
            Unevaluated Lagrangian module (uses current ``secant``).
        x_new
            Accepted (or retained) primal after the controlled step.
        step_dual
            Multipliers shared at both secant endpoints (N&W §18.3).

        Returns
        -------
        Secant or None
            Updated approximation, or ``None`` when no secant is active.
        """
        if self.secant is None:
            return None
        iterate = cast(PrimalType, self.iterate)
        # Share a single multiplier vector (the freshly solved ``step_dual``) at
        # both endpoints so the secant pair satisfies the Lagrangian secant
        # condition (N&W 18.3); the constant +-1 bound Jacobian then cancels.
        prev = lagrangian(iterate, step_dual)
        s = x_new.x - iterate.x
        y = lagrangian.curvature_estimate(x_new, prev)
        return self.secant.append(s, y)

    @abstractmethod
    def _advance_dynamics(
        self,
        ctx: SubproblemContext[PrimalType, SubProblemType, SubProblemSolverStateType],
        x_new: PrimalType,
        step_dual: Dual,
    ) -> Self:
        """Phase-4 post-iterate parameter update.

        Default implementations are no-ops (return ``self``). Interior-point
        solvers override to reduce the barrier weight.

        Parameters
        ----------
        ctx
            Per-step subproblem context.
        x_new
            New primal iterate.
        step_dual
            New multipliers.

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

    @abstractmethod
    def _feasibility_error(
        self, ctx: OptimisationContext[PrimalType, SubProblemSolverStateType]
    ) -> Scalar:
        """Constraint / bound violation measure compared against ``atol``.

        Parameters
        ----------
        ctx
            Termination context at the current iterate.

        Returns
        -------
        Scalar
            Non-negative feasibility residual.
        """
        ...

    def _extra_optimality(
        self, ctx: OptimisationContext[PrimalType, SubProblemSolverStateType]
    ) -> Scalar:
        """Extra optimality residual (e.g. complementarity) vs ``atol``.

        Default ``0`` — solvers with no barrier need nothing extra.

        Parameters
        ----------
        ctx
            Termination context at the current iterate.

        Returns
        -------
        Scalar
            Non-negative extra optimality residual.
        """
        return jnp.asarray(0.0)

    def _nonfinite_detected(
        self, ctx: OptimisationContext[PrimalType, SubProblemSolverStateType]
    ) -> Bool[Array, ""]:
        """Whether any tracked quantity is non-finite (NaN / ±Inf).

        Tracks Lagrangian value / gradient and every leaf of the iterate /
        dual. Once these blow up no further iteration can recover, so
        :meth:`terminate` should exit immediately.

        Parameters
        ----------
        ctx
            Termination context at the current iterate.

        Returns
        -------
        Bool[Array, ""]
            ``True`` when a non-finite value is detected.
        """
        tracked = (ctx.lagrangian.value, ctx.lagrangian.x_grad, self.iterate, self.dual)
        finite = jnp.all(
            jnp.stack(
                [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(tracked)]
            )
        )
        return jnp.logical_not(finite)

    def _termination_diagnostics(
        self, ctx: OptimisationContext[PrimalType, SubProblemSolverStateType]
    ) -> tuple[tuple[Bool[Array, ""], optx.RESULTS], ...]:
        """Ordered ``(fired, result)`` failure diagnostics (highest priority first).

        :meth:`terminate` returns the earliest-fired code. The base supplies
        the generic non-finite guard; solvers override to prepend/append
        checks driven by ``ctx.solver_state`` (call ``super()`` and extend).

        Parameters
        ----------
        ctx
            Termination context at the current iterate.

        Returns
        -------
        tuple of (Bool, optimistix.RESULTS)
            Failure predicates with their status codes.
        """
        return ((self._nonfinite_detected(ctx), optx.RESULTS.nonfinite),)

    def terminate(
        self, problem: ProblemProtocol[PrimalType]
    ) -> tuple[Bool[Array, ""], optx.RESULTS]:
        """Test convergence and failure diagnostics at the current iterate.

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
        lagrangian = ctx.lagrangian

        # Convergence: relative stationarity + feasibility + extra optimality.
        grad_norm = _inf_norm(lagrangian.x_grad)
        scale = jnp.maximum(jnp.abs(lagrangian.value), 1.0)
        stationary = grad_norm <= self.rtol * scale
        feasible = self._feasibility_error(ctx) <= self.atol
        optimal = self._extra_optimality(ctx) <= self.atol
        has_min_steps = self.step_count >= self.min_steps
        converged = stationary & feasible & optimal & has_min_steps

        # Failure diagnostics: fold the ordered list so the earliest-listed fired
        # code wins (apply in reverse -> first entry overwrites last).  ``result``
        # stays ``successful`` when nothing fires, so a converged run reports
        # success and a still-running one reports success too (the driver then
        # maps ``done == False`` to ``nonlinear_max_steps_reached``).
        result = optx.RESULTS.successful
        failed = jnp.asarray(False)
        for fired, code in reversed(self._termination_diagnostics(ctx)):
            result = optx.RESULTS.where(fired, code, result)
            failed = failed | fired

        done = converged | failed
        return done, result

    def postprocess(
        self, problem: ProblemProtocol[PrimalType], result: optx.RESULTS
    ) -> optx.Solution:
        """Pack ``iterate.x`` into an :class:`optimistix.Solution`.

        Parameters
        ----------
        problem
            NLP being minimised (unused; kept for the abstract interface).
        result
            Final status code.

        Returns
        -------
        optimistix.Solution
            ``value`` is the decision vector; ``stats`` carries ``num_steps``.

        Raises
        ------
        ValueError
            If :meth:`init` has not been called (``iterate is None``).
        """
        if self.iterate is None:
            raise ValueError("iterate is not set")
        return cast(
            optx.Solution,
            optx.Solution(
                value=self.iterate.x,
                result=result,
                aux=None,
                stats={"num_steps": self.step_count},
                state=self,
            ),
        )
