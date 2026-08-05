"""Abstract secant / quasi-Newton curvature interface for sqpdax."""

from abc import abstractmethod
from typing import ClassVar, Self

from equinox import Module
from jaxtyping import Array, Bool

from ..registry import KindRegistryMixin
from ..types import InitializableModule, Scalar, Vector_n

__all__ = [
    "CurvatureDiagnostics",
    "Secant",
]


class CurvatureDiagnostics(Module):
    """Scalar diagnostics for a candidate curvature pair ``(s, y)``.

    Produced by :meth:`Secant.diagnostics` so callers can observe skip /
    quality signals without re-implementing the acceptance predicate.

    Attributes
    ----------
    raw_curvature
        Inner product ``sᵀ y``.
    relative_curvature
        Normalised curvature ``|sᵀ y| / (‖s‖ ‖y‖)`` (cosine of the angle
        between ``s`` and ``y``).
    skipped
        Whether the pair would be rejected by the secant's skip logic.
    """

    raw_curvature: Scalar  # s dot product y
    relative_curvature: Scalar  # |s.y| / (||s|| * ||y||)
    skipped: Bool[Array, ""]  # |s.y| < skip_threshold


class Secant(KindRegistryMixin, InitializableModule):
    """Abstract matrix-free Hessian / inverse-Hessian approximation.

    Concrete members (e.g. :class:`~slsqp_jax.sqpdax.secant.lbfgs.LBFGS`)
    register themselves under a ``kind`` ClassVar via
    :class:`~slsqp_jax.sqpdax.registry.KindRegistryMixin` and can be built
    with :meth:`from_spec`.

    The interface is deliberately operator-only: store curvature pairs,
    apply ``B`` and ``B⁻¹`` to vectors, report diagnostics, and escalate
    resets when the approximation becomes ill-conditioned.
    """

    _registry: ClassVar[dict] = {}

    @abstractmethod
    def hvp(self, v: Vector_n) -> Vector_n:
        """Apply the Hessian approximation: ``B v``.

        Parameters
        ----------
        v
            Vector of length ``n``.

        Returns
        -------
        Vector_n
            ``B v``.
        """
        ...  # pragma: no cover

    @abstractmethod
    def inverse_hvp(self, v: Vector_n) -> Vector_n:
        """Apply the inverse Hessian approximation: ``H v = B⁻¹ v``.

        Parameters
        ----------
        v
            Vector of length ``n``.

        Returns
        -------
        Vector_n
            ``H v``.
        """
        ...  # pragma: no cover

    @abstractmethod
    def append(self, s: Vector_n, y: Vector_n) -> Self:
        """Ingest a new curvature pair ``(s, y)``, returning an updated copy.

        Implementations may damp, clip, or skip the pair. The original
        module is left unchanged.

        Parameters
        ----------
        s
            Step ``x₊ - x``.
        y
            Gradient difference (typically of the Lagrangian).

        Returns
        -------
        Self
            Updated secant, or ``self`` if the pair was skipped.
        """
        ...  # pragma: no cover

    @abstractmethod
    def diagnostics(self, s: Vector_n, y: Vector_n) -> CurvatureDiagnostics:
        """Compute curvature diagnostics for ``(s, y)`` without mutating state.

        Parameters
        ----------
        s
            Step vector.
        y
            Gradient difference.

        Returns
        -------
        CurvatureDiagnostics
            Raw / relative curvature and skip flag.
        """
        ...  # pragma: no cover

    @abstractmethod
    def reset(self, severity: int = 0) -> Self:
        """Return a copy with history partially or fully discarded.

        Severity is escalating: lower values keep more curvature
        information; higher values are more aggressive.

        Parameters
        ----------
        severity
            Non-negative integer selecting the reset strength. Concrete
            subclasses define the mapping (e.g. soft / diagonal /
            identity).

        Returns
        -------
        Self
            Reset secant approximation.
        """
        ...  # pragma: no cover
