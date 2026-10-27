"""Second-order corrections of a proposed step (Maratos effect).

A full SQP step ``d`` satisfies the linearised constraints, but the true
constraints miss by ``O(‖d‖²)`` at ``x + d``. With an exact-penalty merit
``φ = f + ρ ‖c‖`` that second-order error is weighted by ``ρ`` and can make
every full step fail the acceptance test even next to a solution (the
Maratos effect). A second-order correction ``d̂`` pulls the trial point back
towards the constraints, ``c(x + d + d̂) = O(‖d‖³)``, and the step controller
tries ``x + d + d̂`` before it shortens the step.

The correction is written against the
:class:`~slsqp_jax.sqpdax.subproblem.base.SubProblem` interface only (rows,
coordinates, box and least-norm solve of the subproblem), so the same module
serves active-set and interior-point subproblems; the step controllers decide
when to try it and how to accept it.
"""

from abc import abstractmethod
from typing import Any, Generic, Self, cast

import equinox as eqx
import jax
from equinox import Module
from jax import numpy as jnp
from jaxtyping import Array, Bool, Scalar

from ..dual import Dual
from ..linalg.box import box_segment_fraction
from ..primal import PrimalType
from ..subproblem.base import SubProblem
from ..types import InitializableModule

__all__ = [
    "CorrectionResult",
    "StepCorrection",
    "LeastNormCorrection",
]


def _inf_norm(v: Array) -> Scalar:
    return jnp.max(jnp.abs(v), initial=jnp.asarray(0.0, v.dtype))


def _all_finite(tree: Any) -> Bool[Array, ""]:
    leaves = [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(tree)]
    return jnp.all(jnp.stack(leaves)) if leaves else jnp.asarray(True)


class CorrectionResult(Module, Generic[PrimalType]):
    """Outcome of one :class:`StepCorrection` evaluation.

    Attributes
    ----------
    step
        Correction ``d̂`` in native coordinates; all zeros when ``valid`` is
        false, so ``x + d + step`` is always a well-defined point.
    valid
        Whether the correction may be tried.
    residual_norm
        ``‖c(x + d)‖_∞`` over the rows the correction targets.
    fraction
        Box scaling ``t ∈ [0, 1]`` applied to the least-norm step.
    """

    step: PrimalType
    valid: Bool[Array, ""]
    residual_norm: Scalar
    fraction: Scalar


class StepCorrection(InitializableModule, Generic[PrimalType]):
    """Abstract second-order correction of a proposed primal step.

    Instances hold two kinds of fields: static configuration, set once (for
    example from ``options['minimiser']['step_correction']``), and the
    per-step data that :meth:`bind` fills in before the step controller
    calls the correction. An unbound instance is hashable and can sit on a
    minimiser as a static field.
    """

    @abstractmethod
    def bind(
        self,
        *,
        subproblem: SubProblem[PrimalType],
        lagrangian: Any,
        dual: Dual,
        enabled: Bool[Array, ""],
        default_tol: float,
    ) -> Self:
        """Return a copy carrying this step's data.

        Parameters
        ----------
        subproblem
            Subproblem the direction was computed from (its rows, coordinates
            and box define the correction).
        lagrangian
            Unevaluated Lagrangian module, called as ``lagrangian(x, dual)``
            to evaluate the constraints at the trial point.
        dual
            Multipliers used for that evaluation.
        enabled
            Traced switch; ``False`` makes every call invalid (e.g. a failed
            subproblem solve).
        default_tol
            Residual threshold used when the correction has none configured.

        Returns
        -------
        Self
            Bound copy.
        """

    @abstractmethod
    def __call__(
        self, x0: PrimalType, direction: PrimalType
    ) -> CorrectionResult[PrimalType]:
        """Correction of the trial point ``x0 + direction``.

        Parameters
        ----------
        x0
            Current iterate (the subproblem's reference point).
        direction
            Proposed full step in native coordinates.

        Returns
        -------
        CorrectionResult
            Correction ``d̂`` and whether it may be tried.
        """


class LeastNormCorrection(StepCorrection[PrimalType]):
    """Minimum-norm second-order correction (Nocedal & Wright Alg. 18.3).

    With ``A`` the constraint rows of the subproblem at ``x`` (model
    coordinates) and ``r = c(x + d)`` the residual of the same rows at the
    trial point, the correction is

    ```
    w = −Aᵀ (A Aᵀ)⁺ r,        d̂ = t · w   (back in native coordinates)
    ```

    with ``t ∈ [0, 1]`` the largest fraction that keeps ``d + t w`` inside
    the subproblem's :meth:`~slsqp_jax.sqpdax.subproblem.base.SubProblem.primal_box`
    (variable bounds for working-set subproblems, fraction-to-the-boundary
    faces for interior-point ones). As in scipy's ``trust-constr`` it is only
    tried near feasibility: the range-space part ``d_n = −Aᵀ(A Aᵀ)⁺ c(x)``
    of the step must be small next to its tangential part,
    ``‖d_n‖ ≤ soc_threshold ‖d − d_n‖``.

    Attributes
    ----------
    soc_threshold
        Near-feasibility gate on ``‖d_n‖ / ‖d_t‖`` (scipy: ``0.1``).
    tol
        The correction is invalid when ``‖r‖_∞ ≤ tol`` (nothing to correct).
        ``None`` takes the ``default_tol`` passed to :meth:`bind`.
    cg_tol, cg_max_iter
        Controls of the range-space solves
        (:meth:`~slsqp_jax.sqpdax.subproblem.base.SubProblem.least_norm_step`).
    subproblem, lagrangian, dual, enabled
        Per-step data filled in by :meth:`bind`; ``None`` until then.

    Examples
    --------
    On the circle ``x₁² + x₂² = 1`` a tangent step ``d`` of length ``0.1``
    leaves the circle by ``‖d‖² = 0.01``; the corrected point misses it by
    ``‖d‖⁴ / 4``:

    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.active_set import ActiveSet
    >>> from slsqp_jax.sqpdax.dual import Dual
    >>> from slsqp_jax.sqpdax.lagrangian import Lagrangian
    >>> from slsqp_jax.sqpdax.primal import Primal
    >>> from slsqp_jax.sqpdax.problem import build_problem
    >>> from slsqp_jax.sqpdax.step_controller import LeastNormCorrection
    >>> from slsqp_jax.sqpdax.subproblem.active_set import ActiveSetSubProblem
    >>> problem = build_problem(lambda x: -x[0], n=2, meq=1,
    ...                         eq_fn=lambda x: jnp.array([x @ x - 1.0]),
    ...                         autodiff_mode="jax", force_hvp_in_jax_mode=True)
    >>> x0, d = Primal(jnp.array([1.0, 0.0])), Primal(jnp.array([0.0, 0.1]))
    >>> dual = Dual(jnp.zeros(1), jnp.zeros(0), jnp.zeros(2), jnp.zeros(2))
    >>> lagrangian = Lagrangian(problem)
    >>> working_set = ActiveSet(meq=1, active_inequalities=jnp.zeros(0, bool),
    ...                         active_lb=jnp.zeros(2, bool),
    ...                         active_ub=jnp.zeros(2, bool))
    >>> subproblem = ActiveSetSubProblem(lagrangian(x0, dual), working_set)
    >>> correction = LeastNormCorrection().bind(
    ...     subproblem=subproblem, lagrangian=lagrangian, dual=dual,
    ...     enabled=jnp.asarray(True), default_tol=1e-8)
    >>> out = correction(x0, d)
    >>> bool(out.valid), round(float(out.residual_norm), 6)
    (True, 0.01)
    >>> x = x0.x + d.x + out.step.x
    >>> abs(float(x @ x - 1.0) - 0.1**4 / 4) < 1e-6
    True
    """

    soc_threshold: float = eqx.field(static=True, default=0.1)
    tol: float | None = eqx.field(static=True, default=None)
    cg_tol: float = eqx.field(static=True, default=1e-10)
    cg_max_iter: int | None = eqx.field(static=True, default=None)
    subproblem: SubProblem[PrimalType] | None = None
    lagrangian: Any = None
    dual: Dual | None = None
    enabled: Bool[Array, ""] | None = None

    def __check_init__(self) -> None:
        """Validate the static configuration."""
        if self.soc_threshold <= 0.0:
            raise ValueError(
                f"soc_threshold must be positive; got {self.soc_threshold}"
            )

    def bind(
        self,
        *,
        subproblem: SubProblem[PrimalType],
        lagrangian: Any,
        dual: Dual,
        enabled: Bool[Array, ""],
        default_tol: float,
    ) -> Self:
        """Attach this step's subproblem, Lagrangian and multipliers.

        See :meth:`StepCorrection.bind` for the parameters.
        """
        return self.init(
            subproblem=subproblem,
            lagrangian=lagrangian,
            dual=dual,
            enabled=jnp.asarray(enabled),
            tol=default_tol if self.tol is None else self.tol,
        )

    def __call__(
        self, x0: PrimalType, direction: PrimalType
    ) -> CorrectionResult[PrimalType]:
        """Least-norm correction of ``x0 + direction``.

        See :meth:`StepCorrection.__call__` for the parameters.

        Raises
        ------
        ValueError
            If the correction has not been bound (:meth:`bind`).
        """
        if self.subproblem is None or self.lagrangian is None or self.dual is None:
            raise ValueError("LeastNormCorrection must be bound before it is called")
        sub = self.subproblem
        enabled = jnp.asarray(True) if self.enabled is None else self.enabled
        tol = 0.0 if self.tol is None else self.tol

        def negate(tree):
            return jax.tree.map(jnp.negative, tree)

        trial = jax.tree.map(jnp.add, x0, direction)
        residual = sub.constraint_residual(self.lagrangian(trial, self.dual))
        residual_norm = _inf_norm(residual.flatten())

        # Near-feasibility gate in model coordinates: range-space part of d.
        d_model = sub.to_model_primal(direction).flatten()
        d_normal = sub.least_norm_step(
            negate(sub.dual_grad()), tol=self.cg_tol, max_iter=self.cg_max_iter
        ).flatten()
        tangential = jnp.linalg.norm(d_model - d_normal)
        near_feasible = jnp.linalg.norm(d_normal) <= self.soc_threshold * tangential

        w = sub.least_norm_step(
            negate(residual), tol=self.cg_tol, max_iter=self.cg_max_iter
        )
        lo, hi = sub.primal_box()
        # Round-off on the faces must not block the correction: a start that
        # violates the box by at most ``tol`` is moved onto it first.
        outside = jnp.maximum(jnp.max(lo - d_model, initial=0.0), 0.0)
        outside = jnp.maximum(outside, jnp.max(d_model - hi, initial=0.0))
        start = jnp.where(outside <= tol, jnp.clip(d_model, lo, hi), d_model)
        fraction = box_segment_fraction(start, w.flatten(), lo, hi)
        step = sub.to_native_primal(jax.tree.map(lambda v: fraction * v, w))

        valid = (
            enabled
            & near_feasible
            & (residual_norm > tol)
            & (fraction > 0.0)
            & _all_finite(step)
        )
        step = cast(
            PrimalType,
            jax.tree.map(lambda v: jnp.where(valid, v, jnp.zeros_like(v)), step),
        )
        return cast(
            CorrectionResult[PrimalType],
            CorrectionResult(
                step=step,
                valid=valid,
                residual_norm=residual_norm,
                fraction=fraction,
            ),
        )
