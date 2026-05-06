"""SLSQP outer-loop minimiser.

This module replaces the legacy monolithic ``slsqp_jax/solver.py``.  The
high-level :class:`SLSQP` class accepts a single :class:`SLSQPConfig`
instance grouping the previous 40+ flat keyword arguments, plus the
constraint structure (functions, counts, bounds), optional
user-supplied derivatives, an optional pluggable inner solver, and the
verbose printer.

The class implements the four ``optimistix.AbstractMinimiser``
methods (``init``, ``step``, ``terminate``, ``postprocess``) and
delegates as much logic as possible to:

* :mod:`slsqp_jax.slsqp.bounds` — NLP-level bound machinery.
* :mod:`slsqp_jax.slsqp.derivatives` — gradient / Jacobian / HVP
  closure factories.
* :mod:`slsqp_jax.slsqp.preconditioner` — preconditioner factories.
* :mod:`slsqp_jax.slsqp.hvp` — Lagrangian HVP factories.
* :mod:`slsqp_jax.slsqp.termination` — single source of truth for
  the termination classification.
* :mod:`slsqp_jax.slsqp.verbose` — verbose printer callbacks.
* :mod:`slsqp_jax.qp.bound_fixing` — reduced-space bound-fixing pass.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional, cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import optimistix._misc as optx_misc
from jaxtyping import Array, Bool, Float

from slsqp_jax.config import SLSQPConfig
from slsqp_jax.hessian import (
    compute_lagrangian_gradient,
    lbfgs_init,
)
from slsqp_jax.inner.base import AbstractInnerSolver
from slsqp_jax.inner.cholesky import ProjectedCGCholesky
from slsqp_jax.lpeca import compute_lpeca_active_set
from slsqp_jax.merit import compute_merit
from slsqp_jax.qp.api import solve_qp
from slsqp_jax.qp.bound_fixing import package_qp_result_no_bounds, run_bound_fixing
from slsqp_jax.results import RESULTS
from slsqp_jax.slsqp.bounds import (
    build_bound_jacobian,
    clip_to_bounds,
    compute_bound_constraint_values,
    recover_bound_multipliers,
)
from slsqp_jax.slsqp.derivatives import (
    build_grad_impl,
    build_hvp_contrib_impl,
    build_jacobian_impl,
    build_obj_hvp_impl,
)
from slsqp_jax.slsqp.hvp import (
    build_exact_lagrangian_hvp,
    build_lbfgs_lagrangian_hvp,
)  # noqa: F401  (build_exact_lagrangian_hvp re-exported for tests)
from slsqp_jax.slsqp.preconditioner import (
    build_diagonal_preconditioner,
    build_lbfgs_preconditioner,
)
from slsqp_jax.slsqp.verbose import no_verbose, slsqp_verbose
from slsqp_jax.state import (
    QPResult,
    SLSQPState,
    _init_diagnostics,
)
from slsqp_jax.types import (
    ConstraintFn,
    ConstraintHVPFn,
    GradFn,
    HVPFn,
    JacobianFn,
    Vector,
)
from slsqp_jax.utils import to_scalar

STAGNATION_MESSAGE = (
    "The solver stagnated: the L1 merit function did not improve over "
    "the patience window (max_steps / 10 consecutive iterations). This "
    "may indicate cycling in the QP subproblem or an infeasible/degenerate "
    "problem."
)


class SLSQP(optx.AbstractMinimiser):
    """SLSQP minimiser using Sequential Quadratic Programming.

    See ``README.md`` and ``docs/source/index.md`` for the full
    algorithmic description.  The user-facing API collapses the
    legacy 40+ flat keyword arguments into a single
    :class:`SLSQPConfig` instance grouping the parameters by purpose;
    consult :mod:`slsqp_jax.config` for the sub-config dataclasses.

    Attributes:
        config: Aggregate configuration.  Defaults to
            :class:`SLSQPConfig` with all sub-config defaults.
        eq_constraint_fn: Function ``(x, args) -> c_eq(x)`` evaluated
            for equality-constraint feasibility ``c_eq(x) = 0``.
        ineq_constraint_fn: Function ``(x, args) -> c_ineq(x)``
            evaluated for inequality-constraint feasibility
            ``c_ineq(x) >= 0``.
        n_eq_constraints: Number of equality constraints (static).
        n_ineq_constraints: Number of inequality constraints (static).
        bounds: Optional ``(n, 2)`` array of ``[lower, upper]`` per
            variable; iterates are projected onto this box after every
            step.  Use ``-inf`` / ``+inf`` for unbounded dimensions.
        obj_grad_fn / eq_jac_fn / ineq_jac_fn / obj_hvp_fn /
        eq_hvp_fn / ineq_hvp_fn: Optional user-supplied derivative
            callables; the AD fallbacks (``jax.grad`` / ``jax.jacrev``
            / forward-over-reverse ``jvp(grad(.))``) are used when
            these are ``None``.
        inner_solver: Optional pluggable inner equality-constrained
            QP solver.  ``None`` constructs a default
            :class:`ProjectedCGCholesky` derived from ``config``.
        verbose: ``True``/``False`` or a custom ``(**kwargs) -> None``
            callable for per-step diagnostics.
    """

    norm: Callable = eqx.field(static=True, default=optx_misc.max_norm)

    config: SLSQPConfig = eqx.field(default_factory=SLSQPConfig)

    eq_constraint_fn: Optional[ConstraintFn] = eqx.field(static=True, default=None)
    ineq_constraint_fn: Optional[ConstraintFn] = eqx.field(static=True, default=None)

    n_eq_constraints: int = eqx.field(static=True, default=0)
    n_ineq_constraints: int = eqx.field(static=True, default=0)

    bounds: Optional[Float[Array, "n 2"]] = None

    _lower_bound_mask: Optional[tuple[bool, ...]] = eqx.field(static=True, default=None)
    _upper_bound_mask: Optional[tuple[bool, ...]] = eqx.field(static=True, default=None)
    _n_lower_bounds: int = eqx.field(static=True, default=0)
    _n_upper_bounds: int = eqx.field(static=True, default=0)
    _lower_indices: Optional[tuple[int, ...]] = eqx.field(static=True, default=None)
    _upper_indices: Optional[tuple[int, ...]] = eqx.field(static=True, default=None)

    obj_grad_fn: Optional[GradFn] = eqx.field(static=True, default=None)
    eq_jac_fn: Optional[JacobianFn] = eqx.field(static=True, default=None)
    ineq_jac_fn: Optional[JacobianFn] = eqx.field(static=True, default=None)
    obj_hvp_fn: Optional[HVPFn] = eqx.field(static=True, default=None)
    eq_hvp_fn: Optional[ConstraintHVPFn] = eqx.field(static=True, default=None)
    ineq_hvp_fn: Optional[ConstraintHVPFn] = eqx.field(static=True, default=None)

    _grad_impl: Callable = eqx.field(static=True, default=None)
    _eq_jac_impl: Callable = eqx.field(static=True, default=None)
    _ineq_jac_impl: Callable = eqx.field(static=True, default=None)
    _eq_hvp_contrib_impl: Callable = eqx.field(static=True, default=None)
    _ineq_hvp_contrib_impl: Callable = eqx.field(static=True, default=None)
    _obj_hvp_impl: Optional[Callable] = eqx.field(static=True, default=None)

    inner_solver: Optional[AbstractInnerSolver] = eqx.field(static=True, default=None)

    _stagnation_window: int = eqx.field(static=True, default=10)
    _proximal_mu_min: float = eqx.field(static=True, default=1e-6)
    _proximal_mu_max: float = eqx.field(static=True, default=0.1)

    verbose: Callable = eqx.field(static=True, default=False)

    # ------------------------------------------------------------------
    # Convenience accessors that surface the most-frequently used
    # config fields directly on the SLSQP instance.  These keep the
    # method bodies readable (``self.rtol`` instead of
    # ``self.config.tolerance.rtol``) without re-introducing the flat
    # constructor surface.
    # ------------------------------------------------------------------

    @property
    def rtol(self) -> float:
        return self.config.tolerance.rtol

    @property
    def atol(self) -> float:
        return self.config.tolerance.atol

    @property
    def max_steps(self) -> int:
        return self.config.tolerance.max_steps

    @property
    def min_steps(self) -> int:
        return self.config.tolerance.min_steps

    @property
    def stagnation_tol(self) -> float:
        return self.config.tolerance.stagnation_tol

    @property
    def divergence_factor(self) -> float:
        return self.config.tolerance.divergence_factor

    @property
    def divergence_patience(self) -> int:
        return self.config.tolerance.divergence_patience

    @property
    def lbfgs_memory(self) -> int:
        return self.config.lbfgs.memory

    @property
    def damping_threshold(self) -> float:
        return self.config.lbfgs.damping_threshold

    @property
    def lbfgs_diag_floor(self) -> float:
        return self.config.lbfgs.diag_floor

    @property
    def lbfgs_diag_ceil(self) -> float:
        return self.config.lbfgs.diag_ceil

    @property
    def line_search_max_steps(self) -> int:
        return self.config.line_search.max_steps

    @property
    def armijo_c1(self) -> float:
        return self.config.line_search.armijo_c1

    @property
    def ls_failure_patience(self) -> int:
        return self.config.line_search.failure_patience

    @property
    def qp_max_iter(self) -> int:
        return self.config.qp.max_iter

    @property
    def qp_max_cg_iter(self) -> int:
        return self.config.qp.max_cg_iter

    @property
    def qp_failure_patience(self) -> int:
        return self.config.qp.failure_patience

    @property
    def zero_step_patience(self) -> int:
        return self.config.qp.zero_step_patience

    @property
    def qp_ping_pong_threshold(self) -> int:
        return self.config.qp.ping_pong_threshold

    @property
    def mult_drop_floor(self) -> float:
        return self.config.qp.mult_drop_floor

    @property
    def cg_regularization(self) -> float:
        return self.config.qp.cg_regularization

    @property
    def use_exact_hvp_in_qp(self) -> bool:
        return self.config.qp.use_exact_hvp

    @property
    def proximal_tau(self) -> float:
        return self.config.proximal.tau

    @property
    def proximal_mu_min(self) -> Optional[float]:
        return self.config.proximal.mu_min

    @property
    def proximal_mu_max(self) -> float:
        return self.config.proximal.mu_max

    @property
    def use_preconditioner(self) -> bool:
        return self.config.preconditioner.enabled

    @property
    def preconditioner_type(self) -> str:
        return self.config.preconditioner.type

    @property
    def diagonal_n_probes(self) -> int:
        return self.config.preconditioner.diagonal_n_probes

    @property
    def active_set_method(self) -> str:
        return self.config.lpeca.method

    @property
    def lpeca_sigma(self) -> float:
        return self.config.lpeca.sigma

    @property
    def lpeca_beta(self) -> Optional[float]:
        return self.config.lpeca.beta

    @property
    def lpeca_use_lp(self) -> bool:
        return self.config.lpeca.use_lp

    @property
    def lpeca_trust_threshold(self) -> float:
        return self.config.lpeca.trust_threshold

    @property
    def lpeca_warmup_steps(self) -> int:
        return self.config.lpeca.warmup_steps

    @property
    def lpeca_predict_bounds(self) -> bool:
        return self.config.lpeca.predict_bounds

    @property
    def adaptive_cg_tol(self) -> bool:
        return self.config.adaptive_cg.enabled

    @property
    def use_inexact_stationarity(self) -> bool:
        return self.config.adaptive_cg.use_inexact_stationarity

    # ------------------------------------------------------------------
    # __check_init__: validate config + precompute derivative closures
    # and bound metadata.
    # ------------------------------------------------------------------

    def __check_init__(self):
        config = self.config
        object.__setattr__(
            self, "_stagnation_window", max(1, config.tolerance.max_steps // 10)
        )

        if config.proximal.mu_min is not None:
            object.__setattr__(self, "_proximal_mu_min", config.proximal.mu_min)
        else:
            object.__setattr__(self, "_proximal_mu_min", config.tolerance.atol)
        object.__setattr__(self, "_proximal_mu_max", config.proximal.mu_max)

        if not (0 <= config.proximal.tau < 1):
            raise ValueError(
                "proximal.tau must be in the half-open interval [0, 1), "
                f"got {config.proximal.tau}"
            )

        if config.preconditioner.type not in ("lbfgs", "diagonal"):
            raise ValueError(
                "preconditioner.type must be 'lbfgs' or 'diagonal', "
                f"got {config.preconditioner.type!r}"
            )
        if config.preconditioner.type == "diagonal" and not (
            self.obj_hvp_fn is not None or config.qp.use_exact_hvp
        ):
            raise ValueError(
                "preconditioner.type='diagonal' requires an exact HVP: "
                "set qp.use_exact_hvp=True or provide obj_hvp_fn"
            )

        if config.lpeca.method not in ("expand", "lpeca_init", "lpeca"):
            raise ValueError(
                "lpeca.method must be 'expand', 'lpeca_init', or 'lpeca', "
                f"got {config.lpeca.method!r}"
            )
        if not (0 < config.lpeca.sigma < 1):
            raise ValueError(
                "lpeca.sigma must be in the open interval (0, 1), "
                f"got {config.lpeca.sigma}"
            )

        # Verbose callable.
        if self.verbose is True:
            object.__setattr__(self, "verbose", slsqp_verbose)
        elif self.verbose is False:
            object.__setattr__(self, "verbose", no_verbose)
        elif callable(self.verbose):  # pragma: no cover
            user_fn = self.verbose

            def _strip_fmt(**kwargs: tuple) -> None:
                user_fn(**{k: v[:2] for k, v in kwargs.items()})

            object.__setattr__(self, "verbose", _strip_fmt)
        else:  # pragma: no cover
            raise ValueError(
                f"Unrecognized `verbose` of type {type(self.verbose)}. "
                "Expected True, False, or a callable."
            )

        # Bound metadata.
        if self.bounds is not None:
            bounds_np = np.asarray(self.bounds)
            if np.any(np.isnan(bounds_np)):
                raise ValueError("bounds must not contain NaN values")
            if np.any(bounds_np[:, 0] > bounds_np[:, 1]):
                raise ValueError(
                    "Lower bounds must be strictly less or equal to upper bounds."
                )
            if np.any(np.isinf(bounds_np[:, 0]) & (bounds_np[:, 0] > 0)) or np.any(
                np.isinf(bounds_np[:, 1]) & (bounds_np[:, 1] < 0)
            ):
                raise ValueError(
                    "Lower bounds cannot be set to +inf and upper bounds cannot be "
                    "set to -inf."
                )
            lower_mask = np.isfinite(bounds_np[:, 0])
            upper_mask = np.isfinite(bounds_np[:, 1])
            object.__setattr__(self, "_lower_bound_mask", tuple(lower_mask.tolist()))
            object.__setattr__(self, "_upper_bound_mask", tuple(upper_mask.tolist()))
            object.__setattr__(self, "_n_lower_bounds", int(np.sum(lower_mask)))
            object.__setattr__(self, "_n_upper_bounds", int(np.sum(upper_mask)))
            object.__setattr__(
                self,
                "_lower_indices",
                tuple(int(i) for i in np.where(lower_mask)[0]),
            )
            object.__setattr__(
                self,
                "_upper_indices",
                tuple(int(i) for i in np.where(upper_mask)[0]),
            )

        # Derivative closures.
        m_eq = self.n_eq_constraints
        m_ineq = self.n_ineq_constraints
        object.__setattr__(self, "_grad_impl", build_grad_impl(self.obj_grad_fn))
        object.__setattr__(
            self,
            "_eq_jac_impl",
            build_jacobian_impl(
                user_jac=self.eq_jac_fn,
                constraint_fn=self.eq_constraint_fn,
                n_constraints=m_eq,
            ),
        )
        object.__setattr__(
            self,
            "_ineq_jac_impl",
            build_jacobian_impl(
                user_jac=self.ineq_jac_fn,
                constraint_fn=self.ineq_constraint_fn,
                n_constraints=m_ineq,
            ),
        )
        object.__setattr__(
            self,
            "_eq_hvp_contrib_impl",
            build_hvp_contrib_impl(
                user_hvp=self.eq_hvp_fn,
                constraint_fn=self.eq_constraint_fn,
                n_constraints=m_eq,
            ),
        )
        object.__setattr__(
            self,
            "_ineq_hvp_contrib_impl",
            build_hvp_contrib_impl(
                user_hvp=self.ineq_hvp_fn,
                constraint_fn=self.ineq_constraint_fn,
                n_constraints=m_ineq,
            ),
        )
        object.__setattr__(
            self,
            "_obj_hvp_impl",
            build_obj_hvp_impl(
                user_obj_hvp=self.obj_hvp_fn,
                use_exact_hvp_in_qp=config.qp.use_exact_hvp,
            ),
        )

    # ------------------------------------------------------------------
    # NLP-level bound helpers (thin wrappers over slsqp_jax.slsqp.bounds).
    # ------------------------------------------------------------------

    def _clip_to_bounds(self, y: Vector) -> Vector:
        return clip_to_bounds(y, self.bounds)

    def _compute_bound_constraint_values(self, y: Vector) -> Float[Array, " m_bounds"]:
        return compute_bound_constraint_values(
            y, self.bounds, self._lower_indices or (), self._upper_indices or ()
        )

    def _build_bound_jacobian(self, n: int) -> Float[Array, "m_bounds n"]:
        return build_bound_jacobian(
            n, self.bounds, self._lower_indices or (), self._upper_indices or ()
        )

    def _recover_bound_multipliers(
        self,
        *,
        y_new: Vector,
        grad_new: Vector,
        eq_jac_new: Float[Array, "m_eq n"],
        ineq_jac_new: Float[Array, "m_ineq n"],
        mult_eq: Float[Array, " m_eq"],
        mult_ineq_general: Float[Array, " m_general"],
    ) -> tuple[Float[Array, " n_lower"], Float[Array, " n_upper"]]:
        return recover_bound_multipliers(
            y_new=y_new,
            grad_new=grad_new,
            eq_jac_new=eq_jac_new,
            ineq_jac_new=ineq_jac_new,
            mult_eq=mult_eq,
            mult_ineq_general=mult_ineq_general,
            bounds=self.bounds,
            lower_indices=self._lower_indices or (),
            upper_indices=self._upper_indices or (),
            m_general=self.n_ineq_constraints,
        )

    # ------------------------------------------------------------------
    # HVP / preconditioner factories (thin wrappers).
    # ------------------------------------------------------------------

    def _build_lagrangian_hvp(
        self,
        fn: Callable,
        y: Vector,
        args: Any,
        state: SLSQPState,
    ) -> Callable[[Vector], Vector]:
        if self.use_exact_hvp_in_qp and self._obj_hvp_impl is not None:
            return build_exact_lagrangian_hvp(
                fn=fn,
                y=y,
                args=args,
                multipliers_eq=state.multipliers_eq,
                multipliers_ineq=state.multipliers_ineq,
                obj_hvp_impl=self._obj_hvp_impl,
                eq_hvp_contrib_impl=self._eq_hvp_contrib_impl,
                ineq_hvp_contrib_impl=self._ineq_hvp_contrib_impl,
                n_ineq_general=self.n_ineq_constraints,
            )
        return build_lbfgs_lagrangian_hvp(state.lbfgs_history)

    def _build_preconditioner(
        self,
        state: SLSQPState,
        proximal_mu: float | jnp.ndarray = 0.0,
        lagrangian_hvp_fn: Optional[Callable[[Vector], Vector]] = None,
    ) -> Optional[Callable[[Vector], Vector]]:
        if not self.use_preconditioner:
            return None
        proximal_active = self.n_eq_constraints > 0 and self.proximal_tau > 0
        if self.preconditioner_type == "diagonal":
            assert lagrangian_hvp_fn is not None, (
                "diagonal preconditioner requires an exact Lagrangian HVP"
            )
            return build_diagonal_preconditioner(
                lagrangian_hvp_fn=lagrangian_hvp_fn,
                n=state.grad.shape[0],
                step_count=state.step_count,
                n_probes=self.diagonal_n_probes,
                eq_jac=state.eq_jac if self.n_eq_constraints > 0 else None,
                proximal_active=proximal_active,
                proximal_mu=proximal_mu,
            )
        return build_lbfgs_preconditioner(
            lbfgs_history=state.lbfgs_history,
            eq_jac=state.eq_jac if self.n_eq_constraints > 0 else None,
            proximal_active=proximal_active,
            proximal_mu=proximal_mu,
        )

    # ------------------------------------------------------------------
    # init / step / terminate / postprocess.
    # ------------------------------------------------------------------

    def init(
        self,
        fn: Callable,
        y: Vector,
        args: Any,
        options: dict[str, Any],
        f_struct: Any,
        aux_struct: Any,
        tags: frozenset[object],
    ) -> SLSQPState:
        n = y.shape[0]
        y = self._clip_to_bounds(y)
        m_eq = self.n_eq_constraints
        m_ineq_general = self.n_ineq_constraints
        m_bounds = self._n_lower_bounds + self._n_upper_bounds
        m_ineq_total = m_ineq_general + m_bounds

        f_val, _aux = fn(y, args)
        f_val = to_scalar(f_val)
        grad = self._grad_impl(fn, y, args)

        if self.eq_constraint_fn is not None and m_eq > 0:
            eq_val = self.eq_constraint_fn(y, args)
        else:
            eq_val = jnp.zeros((m_eq,))

        if self.ineq_constraint_fn is not None and m_ineq_general > 0:
            ineq_val_general = self.ineq_constraint_fn(y, args)
        else:
            ineq_val_general = jnp.zeros((m_ineq_general,))

        bound_vals = self._compute_bound_constraint_values(y)
        ineq_val = jnp.concatenate([ineq_val_general, bound_vals])

        eq_jac = self._eq_jac_impl(y, args)
        ineq_jac_general = self._ineq_jac_impl(y, args)
        bound_jac = self._build_bound_jacobian(n)
        ineq_jac = jnp.concatenate([ineq_jac_general, bound_jac], axis=0)

        lbfgs_history = lbfgs_init(n, self.lbfgs_memory)

        if m_eq > 0:
            multipliers_eq, _, _, _ = jnp.linalg.lstsq(eq_jac.T, grad)
        else:
            multipliers_eq = jnp.zeros((m_eq,))
        multipliers_ineq = jnp.zeros((m_ineq_total,))

        prev_grad_lagrangian = compute_lagrangian_gradient(
            grad, eq_jac, ineq_jac, multipliers_eq, multipliers_ineq
        )

        merit_penalty = jnp.array(1.0)
        initial_merit = compute_merit(f_val, eq_val, ineq_val, merit_penalty)

        return SLSQPState(  # ty: ignore[invalid-return-type]
            step_count=jnp.array(0),
            f_val=f_val,
            grad=grad,
            eq_val=eq_val,
            ineq_val=ineq_val,
            eq_jac=eq_jac,
            ineq_jac=ineq_jac,
            lbfgs_history=lbfgs_history,
            multipliers_eq=multipliers_eq,
            multipliers_ineq=multipliers_ineq,
            prev_grad_lagrangian=prev_grad_lagrangian,
            grad_lagrangian=prev_grad_lagrangian,
            merit_penalty=merit_penalty,
            bound_jac=bound_jac,
            qp_iterations=jnp.array(0),
            qp_converged=jnp.array(True),
            prev_active_set=jnp.zeros((m_ineq_total,), dtype=bool),
            consecutive_qp_failures=jnp.array(0),
            consecutive_ls_failures=jnp.array(0),
            consecutive_zero_steps=jnp.array(0),
            qp_optimal=jnp.array(False),
            best_merit=initial_merit,
            steps_without_improvement=jnp.array(0),
            stagnation=jnp.array(False),
            last_alpha=jnp.array(1.0),
            last_projected_grad_norm=jnp.asarray(jnp.inf),
            ls_success=jnp.array(True),
            ls_fatal=jnp.array(False),
            qp_fatal=jnp.array(False),
            termination_code=RESULTS.successful,
            best_x=y,
            blowup_count=jnp.array(0),
            diverging=jnp.array(False),
            diagnostics=_init_diagnostics(),
        )

    def step(
        self,
        fn: Callable,
        y: Vector,
        args: Any,
        options: dict[str, Any],
        state: SLSQPState,
        tags: frozenset[object],
    ) -> tuple[Vector, SLSQPState, Any]:
        from slsqp_jax.slsqp._step_body import _step_impl

        return _step_impl(self, fn, y, args, options, state, tags)

    def terminate(
        self,
        fn: Callable,
        y: Vector,
        args: Any,
        options: dict[str, Any],
        state: SLSQPState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], Any]:
        from slsqp_jax.slsqp._step_body import _terminate_impl

        return _terminate_impl(self, fn, y, args, options, state, tags)

    def postprocess(
        self,
        fn: Callable,
        y: Vector,
        aux: Any,
        args: Any,
        options: dict[str, Any],
        state: SLSQPState,
        tags: frozenset[object],
        result: Any,
    ) -> tuple[Vector, Any, dict[str, Any]]:
        y = self._clip_to_bounds(y)
        stats = {
            "num_steps": state.step_count,
            "final_objective": state.f_val,
            "final_grad_norm": jnp.linalg.norm(state.grad),
            "merit_penalty": state.merit_penalty,
            "total_qp_iterations": state.qp_iterations,
            "last_qp_converged": state.qp_converged,
            "qp_tolerance": 1e-8,
            "multipliers_eq": state.multipliers_eq,
            "multipliers_ineq": state.multipliers_ineq,
            "stagnation": state.stagnation,
            "ls_fatal": state.ls_fatal,
            "qp_fatal": state.qp_fatal,
            "diverging": state.diverging,
            "last_step_size": state.last_alpha,
            "consecutive_ls_failures": state.consecutive_ls_failures,
            "consecutive_qp_failures": state.consecutive_qp_failures,
            "slsqp_result": state.termination_code,
        }
        return y, aux, stats

    # ------------------------------------------------------------------
    # _solve_qp_subproblem: orchestrate LPEC-A + solve_qp + bound-fixing.
    # ------------------------------------------------------------------

    def _solve_qp_subproblem(
        self,
        state: SLSQPState,
        hvp_fn: Callable[[Vector], Vector],
        y: Vector,
    ) -> QPResult:
        g = state.grad
        A_eq = state.eq_jac
        b_eq = -state.eq_val

        m_ineq_general = self.n_ineq_constraints
        m_bounds = self._n_lower_bounds + self._n_upper_bounds

        A_ineq = state.ineq_jac[:m_ineq_general]
        b_ineq = -state.ineq_val[:m_ineq_general]

        kkt_residual = jnp.linalg.norm(state.prev_grad_lagrangian)
        initial_active_set = (
            state.prev_active_set[:m_ineq_general] if m_ineq_general > 0 else None
        )

        # LPEC-A predicted active set + bound warm-start masks.
        predicted_active_set = None
        lpeca_bypassed = jnp.array(False)
        lpeca_capped = jnp.array(False)
        n_vars_static = g.shape[0]
        lpeca_bound_lower = jnp.zeros(n_vars_static, dtype=bool)
        lpeca_bound_upper = jnp.zeros(n_vars_static, dtype=bool)
        lpeca_bounds_prefixed_count = jnp.array(0)
        if self.active_set_method in ("lpeca_init", "lpeca"):
            m_ineq_total = m_ineq_general + m_bounds
            if m_ineq_total > 0:
                lpeca_result = compute_lpeca_active_set(
                    c_ineq=state.ineq_val,
                    c_eq=state.eq_val,
                    grad=state.grad,
                    A_ineq=state.ineq_jac,
                    A_eq=state.eq_jac,
                    lambda_ineq=state.multipliers_ineq,
                    mu_eq=state.multipliers_eq,
                    sigma=self.lpeca_sigma,
                    beta=self.lpeca_beta,
                    trust_threshold=self.lpeca_trust_threshold,
                    use_lp=self.lpeca_use_lp,
                )
                in_warmup = state.step_count < self.lpeca_warmup_steps
                bypassed = in_warmup | ~lpeca_result.valid
                full_predicted = jnp.where(
                    bypassed,
                    jnp.zeros_like(lpeca_result.predicted),
                    lpeca_result.predicted,
                )
                lpeca_bypassed = bypassed
                lpeca_capped = lpeca_result.capped & ~bypassed
                if m_ineq_general > 0:
                    predicted_active_set = full_predicted[:m_ineq_general]
                if (
                    self.lpeca_predict_bounds
                    and m_bounds > 0
                    and m_ineq_general <= full_predicted.shape[0]
                ):
                    n_lower = self._n_lower_bounds
                    n_upper = self._n_upper_bounds
                    if n_lower > 0:
                        pred_lower = full_predicted[
                            m_ineq_general : m_ineq_general + n_lower
                        ]
                        lower_idx = jnp.asarray(self._lower_indices, dtype=jnp.int32)
                        lpeca_bound_lower = lpeca_bound_lower.at[lower_idx].set(
                            pred_lower
                        )
                    if n_upper > 0:
                        pred_upper = full_predicted[m_ineq_general + n_lower :]
                        upper_idx = jnp.asarray(self._upper_indices, dtype=jnp.int32)
                        lpeca_bound_upper_raw = jnp.zeros(n_vars_static, dtype=bool)
                        lpeca_bound_upper_raw = lpeca_bound_upper_raw.at[upper_idx].set(
                            pred_upper
                        )
                        lpeca_bound_upper = lpeca_bound_upper_raw & ~lpeca_bound_lower
                    lpeca_bounds_prefixed_count = jnp.sum(
                        lpeca_bound_lower.astype(jnp.int32)
                    ) + jnp.sum(lpeca_bound_upper.astype(jnp.int32))

        use_proximal = self.proximal_tau > 0
        if use_proximal:
            mu = jnp.reshape(
                jnp.clip(
                    kkt_residual**self.proximal_tau,
                    self._proximal_mu_min,
                    self._proximal_mu_max,
                ),
                (),
            )
        else:
            mu = 0.0

        precond_fn = self._build_preconditioner(
            state,
            proximal_mu=mu,
            lagrangian_hvp_fn=(
                hvp_fn if self.preconditioner_type == "diagonal" else None
            ),
        )

        adaptive_tol = (
            jnp.minimum(0.1, jnp.maximum(self.atol, kkt_residual))
            if self.adaptive_cg_tol
            else None
        )
        inner_cg_tol = adaptive_tol if adaptive_tol is not None else self.atol

        if self.inner_solver is not None:
            inner_solver = self.inner_solver  # ty: ignore[invalid-assignment]
        else:
            inner_solver = cast(
                AbstractInnerSolver,
                ProjectedCGCholesky(
                    max_cg_iter=self.qp_max_cg_iter,
                    cg_tol=inner_cg_tol,
                    cg_regularization=self.cg_regularization,
                    use_constraint_preconditioner=self.use_exact_hvp_in_qp,
                ),
            )

        qp_result = solve_qp(
            hvp_fn=hvp_fn,
            g=g,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ineq=A_ineq,
            b_ineq=b_ineq,
            max_iter=self.qp_max_iter,
            max_cg_iter=self.qp_max_cg_iter,
            initial_active_set=initial_active_set,
            kkt_residual=kkt_residual,
            proximal_mu=mu,
            prev_multipliers_eq=state.multipliers_eq,
            precond_fn=precond_fn,
            cg_tol=adaptive_tol,
            cg_regularization=self.cg_regularization,
            use_proximal=use_proximal,
            predicted_active_set=predicted_active_set,
            active_set_method=self.active_set_method,
            use_constraint_preconditioner=self.use_exact_hvp_in_qp,
            inner_solver=inner_solver,
            mult_drop_floor=self.mult_drop_floor,
            ping_pong_threshold=self.qp_ping_pong_threshold,
        )

        if m_bounds > 0:
            return run_bound_fixing(
                qp_result,
                inner_solver=inner_solver,
                hvp_fn=hvp_fn,
                g=g,
                A_eq=A_eq,
                b_eq=b_eq,
                A_ineq_general=A_ineq,
                b_ineq_general=b_ineq,
                n_eq_constraints=self.n_eq_constraints,
                m_ineq_general=m_ineq_general,
                bounds=self.bounds,
                y=y,
                n_lower_bounds=self._n_lower_bounds,
                n_upper_bounds=self._n_upper_bounds,
                lower_indices=self._lower_indices,
                upper_indices=self._upper_indices,
                precond_fn=precond_fn,
                adaptive_tol=adaptive_tol,
                lpeca_bound_lower=lpeca_bound_lower,
                lpeca_bound_upper=lpeca_bound_upper,
                lpeca_bypassed=lpeca_bypassed,
                lpeca_capped=lpeca_capped,
                lpeca_bounds_prefixed_count=lpeca_bounds_prefixed_count,
            )

        # Override the lpeca_bypassed / lpeca_capped flags from the
        # default :func:`package_qp_result_no_bounds` (which assumes no
        # LPEC-A) so the diagnostics on the no-bounds path still reflect
        # whether LPEC-A fired on this step.
        wrapped = package_qp_result_no_bounds(qp_result)
        return QPResult(  # ty: ignore[invalid-return-type]
            direction=wrapped.direction,
            multipliers_eq=wrapped.multipliers_eq,
            multipliers_ineq=wrapped.multipliers_ineq,
            active_set=wrapped.active_set,
            converged=wrapped.converged,
            iterations=wrapped.iterations,
            bound_fix_solves=wrapped.bound_fix_solves,
            n_bound_fixed=wrapped.n_bound_fixed,
            ping_ponged=wrapped.ping_ponged,
            reached_max_iter=wrapped.reached_max_iter,
            lpeca_bypassed=lpeca_bypassed,
            lpeca_capped=lpeca_capped,
            n_lpeca_bounds_prefixed=lpeca_bounds_prefixed_count,
            proj_residual=wrapped.proj_residual,
            n_proj_refinements=wrapped.n_proj_refinements,
            projected_grad_norm=wrapped.projected_grad_norm,
        )


__all__ = ["SLSQP", "STAGNATION_MESSAGE"]
