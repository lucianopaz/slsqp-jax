"""Per-step preconditioner construction policies for iterative subproblem solves.

A :class:`PreconditionerStrategy` is static configuration held by a
minimiser. Every outer step the minimiser packs what is currently known
about curvature into a :class:`PreconditionerContext` and asks the strategy
to :meth:`~PreconditionerStrategy.build` a fresh
:class:`~slsqp_jax.sqpdax.preconditioner.base.Preconditioner`, which it then
hands to the subproblem solver. Strategies never own solver state and never
decide where a preconditioner is plugged in.
"""

from abc import abstractmethod
from typing import ClassVar, cast

import jax
from equinox import Module, field
from jax import numpy as jnp
from jaxtyping import Array, Int

from ..lagrangian.evaluated import EvaluatedLagrangian
from ..registry import KindRegistryMixin
from ..secant.base import Secant
from ..types import InitializableModule, Vector_n
from .base import DiagonalPreconditioner, Preconditioner
from .utils import preconditioner_from_secant, stochastic_diagonal

__all__ = [
    "PreconditionerContext",
    "PreconditionerStrategy",
    "NoPreconditioner",
    "SecantPreconditioner",
    "StochasticDiagonalPreconditioner",
]


class PreconditionerContext(Module):
    """Curvature information available when a step's preconditioner is built.

    Attributes
    ----------
    x_ref
        Current decision vector (fixes ``n`` and the dtype).
    secant
        Maintained curvature approximation, or ``None`` when none is kept.
    lagrangian
        Lagrangian at the current iterate *without* any secant, so its
        :meth:`~slsqp_jax.sqpdax.lagrangian.evaluated.EvaluatedLagrangian.hvp`
        is the exact HVP; ``None`` when the problem lacks exact curvature.
    step_count
        Outer step index (seeds stochastic strategies deterministically).
    """

    x_ref: Vector_n
    secant: Secant | None
    lagrangian: EvaluatedLagrangian | None
    step_count: Int[Array, ""]


class PreconditionerStrategy(KindRegistryMixin, InitializableModule):
    """Abstract policy building a subproblem preconditioner each outer step.

    Concrete strategies register under a ``kind`` and can be selected with
    ``options['minimiser']['preconditioner'] = {"kind": ..., **params}``.
    All fields are static so a strategy can live on a static minimiser field.
    """

    _registry: ClassVar[dict] = {}

    @property
    def requires_secant(self) -> bool:
        """Whether the minimiser must maintain a secant for this strategy."""
        return False

    @property
    def requires_exact_hvp(self) -> bool:
        """Whether the strategy needs exact Lagrangian HVPs."""
        return False

    @property
    def is_active(self) -> bool:
        """Whether the strategy can ever produce a preconditioner."""
        return True

    @abstractmethod
    def build(self, ctx: PreconditionerContext) -> Preconditioner | None:
        """Build this step's preconditioner.

        Parameters
        ----------
        ctx
            Curvature information at the current iterate.

        Returns
        -------
        Preconditioner or None
            SPD preconditioner ``M`` (its ``invert`` is ``M⁻¹``), or
            ``None`` to leave the solver unpreconditioned.
        """
        ...  # pragma: no cover


class NoPreconditioner(PreconditionerStrategy):
    """Never precondition (the solver's own configuration is left as is).

    Examples
    --------
    >>> from slsqp_jax.sqpdax.preconditioner import PreconditionerStrategy
    >>> PreconditionerStrategy.from_spec({"kind": "none"}).is_active
    False
    """

    kind: ClassVar[str] = "none"

    @property
    def is_active(self) -> bool:
        """Always ``False``."""
        return False

    def build(self, ctx: PreconditionerContext) -> Preconditioner | None:
        """Return ``None``."""
        return None


class SecantPreconditioner(PreconditionerStrategy):
    """L-BFGS inverse preconditioner ``M = B``, ``M⁻¹ = H`` from the current secant.

    Rebuilt every step from the secant carried by the minimiser, so it
    tracks appends and resets automatically. Wraps
    :func:`~slsqp_jax.sqpdax.preconditioner.utils.preconditioner_from_secant`
    with ``inverse_as_forward=False``.

    Attributes
    ----------
    require_secant
        If ``True`` (default), the minimiser maintains a secant for this
        preconditioner even when the QP model uses exact HVPs. If ``False``,
        it only preconditions when the model already carries a secant.

    Examples
    --------
    >>> from slsqp_jax.sqpdax.preconditioner import PreconditionerStrategy
    >>> strategy = PreconditionerStrategy.from_spec({"kind": "lbfgs"})
    >>> strategy.requires_secant
    True
    """

    kind: ClassVar[str] = "lbfgs"

    require_secant: bool = field(static=True, default=True)

    @property
    def requires_secant(self) -> bool:
        """:attr:`require_secant`."""
        return self.require_secant

    def build(self, ctx: PreconditionerContext) -> Preconditioner | None:
        """Wrap ``ctx.secant``, or return ``None`` when no secant is kept."""
        if ctx.secant is None:
            return None
        return preconditioner_from_secant(
            ctx.secant,
            ctx.x_ref.shape[0],
            ctx.x_ref.dtype,
            inverse_as_forward=False,
        )


class StochasticDiagonalPreconditioner(PreconditionerStrategy):
    """Diagonal preconditioner from a stochastic estimate of ``diag(∇²ₓₓL)``.

    Every step, ``n_probes`` Rademacher probes of the *exact* Lagrangian HVP
    estimate the Hessian diagonal
    (:func:`~slsqp_jax.sqpdax.preconditioner.utils.stochastic_diagonal`).
    The Lagrangian Hessian may be indefinite, so ``M = diag(max(|d|, floor))``
    with ``floor = max(absolute_floor, relative_floor · median|d|)`` keeps
    ``M`` SPD. Costs ``n_probes`` HVPs per outer step and is only accurate
    for diagonally dominant Hessians; it is opt-in for that reason.

    Attributes
    ----------
    n_probes
        Rademacher probes per step.
    seed
        Base PRNG seed; the per-step key is ``fold_in(key(seed), step_count)``.
    relative_floor
        Floor relative to the median absolute diagonal estimate.
    absolute_floor
        Absolute floor on the diagonal.

    Examples
    --------
    >>> from slsqp_jax.sqpdax.preconditioner import PreconditionerStrategy
    >>> spec = {"kind": "stochastic_diagonal", "n_probes": 8}
    >>> PreconditionerStrategy.from_spec(spec).n_probes
    8
    """

    kind: ClassVar[str] = "stochastic_diagonal"

    n_probes: int = field(static=True, default=20)
    seed: int = field(static=True, default=0)
    relative_floor: float = field(static=True, default=1e-6)
    absolute_floor: float = field(static=True, default=1e-8)

    def __check_init__(self) -> None:
        if self.n_probes < 1:
            raise ValueError(f"n_probes must be at least 1; got {self.n_probes}")

    @property
    def requires_exact_hvp(self) -> bool:
        """Always ``True``."""
        return True

    def build(self, ctx: PreconditionerContext) -> Preconditioner | None:
        """Probe the exact Lagrangian HVP and return a floored diagonal.

        Raises
        ------
        ValueError
            If ``ctx.lagrangian`` is ``None`` (no exact curvature).
        """
        if ctx.lagrangian is None:
            raise ValueError(
                "StochasticDiagonalPreconditioner requires exact Lagrangian HVPs"
            )
        x = ctx.x_ref
        key = jax.random.fold_in(jax.random.key(self.seed), ctx.step_count)
        estimate = stochastic_diagonal(
            ctx.lagrangian.hvp, x.shape[0], key, self.n_probes, x.dtype
        )
        magnitude = jnp.abs(estimate)
        floor = jnp.maximum(
            self.absolute_floor, self.relative_floor * jnp.median(magnitude)
        )
        return cast(
            Preconditioner, DiagonalPreconditioner(jnp.maximum(magnitude, floor))
        )
