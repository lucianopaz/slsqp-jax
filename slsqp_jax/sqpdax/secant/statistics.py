"""Cumulative curvature-update statistics carried by the outer minimiser."""

from dataclasses import fields, replace
from typing import Any, Self

from equinox import Module
from jax import numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Int

from ..types import Scalar
from .base import CurvatureDiagnostics, Secant

__all__ = [
    "SecantStatistics",
]

N_RESET_SEVERITIES = 3


def _condition(lower: Scalar, upper: Scalar) -> Scalar:
    return upper / jnp.maximum(lower, 1e-30)


class SecantStatistics(Module):
    """Running record of what a :class:`~slsqp_jax.sqpdax.secant.base.Secant` did.

    Updated by the minimiser after every attempted append and every reset,
    so the carry exposes skip / damping / conditioning behaviour that the
    secant itself never reports (``append`` returns only the new secant).

    The conditioning fields come from
    :meth:`~slsqp_jax.sqpdax.secant.base.Secant.curvature_bounds`, i.e. the
    scaling matrix of the approximation (``B₀ = diag(d)`` for
    :class:`~slsqp_jax.sqpdax.secant.lbfgs.LBFGS`), not the full ``B``.

    Attributes
    ----------
    n_appends
        Attempted appends that stored a pair.
    n_skips
        Attempted appends rejected by the skip predicate.
    n_damped
        Stored pairs whose ``y`` was damped (``θ < 1``).
    n_resets
        Reset counts per severity: ``[soft, diagonal, identity]``.
    last_sty, last_relative_curvature, last_damping_theta
        Diagnostics of the most recent attempted append.
    min_damping_theta
        Smallest ``θ`` applied to a stored pair (``1`` when none was damped).
    min_diagonal, max_diagonal
        Running extrema of the curvature bounds over the run.
    last_condition, max_condition
        Current and largest condition estimate ``upper / lower``.
    """

    n_appends: Int[Array, ""]
    n_skips: Int[Array, ""]
    n_damped: Int[Array, ""]
    n_resets: Int[Array, " 3"]
    last_sty: Scalar
    last_relative_curvature: Scalar
    last_damping_theta: Scalar
    min_damping_theta: Scalar
    min_diagonal: Scalar
    max_diagonal: Scalar
    last_condition: Scalar
    max_condition: Scalar

    @classmethod
    def initial(cls, secant: Secant, dtype: DTypeLike) -> Self:
        """Zero counters with conditioning seeded from ``secant``.

        Parameters
        ----------
        secant
            Freshly initialised curvature approximation.
        dtype
            Floating dtype of the iterate (fixes the carry dtype).

        Returns
        -------
        Self
            Statistics before any append.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from slsqp_jax.sqpdax.secant import LBFGS, SecantStatistics
        >>> stats = SecantStatistics.initial(LBFGS(n=2, memory=3), jnp.float64)
        >>> int(stats.n_appends), float(stats.last_condition)
        (0, 1.0)
        """
        lower, upper = (jnp.asarray(b, dtype) for b in secant.curvature_bounds())
        cond = _condition(lower, upper)
        zero = jnp.asarray(0, jnp.int32)
        return cls(
            n_appends=zero,
            n_skips=zero,
            n_damped=zero,
            n_resets=jnp.zeros((N_RESET_SEVERITIES,), jnp.int32),
            last_sty=jnp.asarray(0.0, dtype),
            last_relative_curvature=jnp.asarray(0.0, dtype),
            last_damping_theta=jnp.asarray(1.0, dtype),
            min_damping_theta=jnp.asarray(1.0, dtype),
            min_diagonal=lower,
            max_diagonal=upper,
            last_condition=cond,
            max_condition=cond,
        )

    def _with_conditioning(self, secant: Secant, **updates: Any) -> Self:
        lower, upper = secant.curvature_bounds()
        dtype = self.min_diagonal.dtype
        lower = jnp.asarray(lower, dtype)
        upper = jnp.asarray(upper, dtype)
        cond = _condition(lower, upper)
        return replace(
            self,
            **updates,
            min_diagonal=jnp.minimum(self.min_diagonal, lower),
            max_diagonal=jnp.maximum(self.max_diagonal, upper),
            last_condition=cond,
            max_condition=jnp.maximum(self.max_condition, cond),
        )

    def record_append(self, diagnostics: CurvatureDiagnostics, secant: Secant) -> Self:
        """Account for one attempted append.

        Parameters
        ----------
        diagnostics
            :meth:`~slsqp_jax.sqpdax.secant.base.Secant.diagnostics` of the
            candidate pair, computed on the secant *before* the append.
        secant
            Secant *after* the append (unchanged when the pair was skipped).

        Returns
        -------
        Self
            Updated statistics.
        """
        dtype = self.last_sty.dtype
        stored = ~diagnostics.skipped
        theta = jnp.asarray(diagnostics.damping_theta, dtype)
        damped = stored & (theta < 1.0)
        return self._with_conditioning(
            secant,
            n_appends=self.n_appends + stored.astype(jnp.int32),
            n_skips=self.n_skips + diagnostics.skipped.astype(jnp.int32),
            n_damped=self.n_damped + damped.astype(jnp.int32),
            last_sty=jnp.asarray(diagnostics.raw_curvature, dtype),
            last_relative_curvature=jnp.asarray(diagnostics.relative_curvature, dtype),
            last_damping_theta=theta,
            min_damping_theta=jnp.where(
                stored,
                jnp.minimum(self.min_damping_theta, theta),
                self.min_damping_theta,
            ),
        )

    def record_reset(self, severity: Int[Array, ""], secant: Secant) -> Self:
        """Account for a (possibly absent) reset.

        Parameters
        ----------
        severity
            Applied reset severity, or a negative value when no reset fired.
            Severities above the identity level count as identity resets.
        secant
            Secant after the (possible) reset.

        Returns
        -------
        Self
            Updated statistics.
        """
        fired = severity >= 0
        idx = jnp.clip(severity, 0, N_RESET_SEVERITIES - 1)
        return self._with_conditioning(
            secant, n_resets=self.n_resets.at[idx].add(fired.astype(jnp.int32))
        )

    def as_stats(self, prefix: str = "secant_") -> dict[str, Array]:
        """Flatten into a ``postprocess`` statistics mapping.

        Parameters
        ----------
        prefix
            Prefix prepended to every field name.

        Returns
        -------
        dict
            ``{prefix + field: value}`` for every field.
        """
        return {f"{prefix}{f.name}": getattr(self, f.name) for f in fields(self)}
