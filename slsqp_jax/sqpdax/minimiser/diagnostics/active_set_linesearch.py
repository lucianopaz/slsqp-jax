"""Cumulative line-search and LPEC-A predictor counters of the active-set minimiser."""

from __future__ import annotations

from typing import Self

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Bool, Int

from ...active_set_prediction import LPECAPrediction

__all__ = ["ActiveSetLineSearchDiagnostics"]


class ActiveSetLineSearchDiagnostics(eqx.Module):
    """Run-level counters of the L1-merit line search and the LPEC-A predictor.

    Carried by
    :class:`~slsqp_jax.sqpdax.minimiser.active_set_linesearch.ActiveSetLineSearchMinimiser`
    and refreshed once per outer step. None of these quantities feeds back
    into the algorithm: they only describe how the globalisation and the
    working-set prediction behaved, and are surfaced through the per-step
    diagnostic record and the final ``stats`` mapping.

    Attributes
    ----------
    last_ls_fallback
        Whether the last accepted step came from the non-Armijo fallback
        (unit step accepted after exhausting the backtracking budget).
    n_armijo_accepts, n_fallback_accepts
        Cumulative number of steps accepted by the Armijo test and by the
        fallback rule respectively.
    n_lpeca_bypassed, n_lpeca_capped, n_lpeca_bounds_prefixed
        Cumulative predictor diagnostics: steps whose prediction was
        discarded (trust gate or warm-up), steps where the rank cap
        truncated it, and bounds seeded into the working set.

    Examples
    --------
    ```python
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.minimiser.diagnostics import ActiveSetLineSearchDiagnostics
    >>> diag = ActiveSetLineSearchDiagnostics.zero()
    >>> diag = diag.record_step(
    ...     accepted=jnp.asarray(True), accepted_by_fallback=jnp.asarray(False)
    ... )
    >>> int(diag.n_armijo_accepts), int(diag.n_fallback_accepts)
    (1, 0)

    ```
    """

    last_ls_fallback: Bool[Array, ""]
    n_armijo_accepts: Int[Array, ""]
    n_fallback_accepts: Int[Array, ""]
    n_lpeca_bypassed: Int[Array, ""]
    n_lpeca_capped: Int[Array, ""]
    n_lpeca_bounds_prefixed: Int[Array, ""]

    @classmethod
    def zero(cls) -> Self:
        """Return the all-zero carry used at the start of a run."""
        zero = jnp.asarray(0, jnp.int32)
        return cls(
            last_ls_fallback=jnp.asarray(False),
            n_armijo_accepts=zero,
            n_fallback_accepts=zero,
            n_lpeca_bypassed=zero,
            n_lpeca_capped=zero,
            n_lpeca_bounds_prefixed=zero,
        )

    def record_prediction(self, prediction: LPECAPrediction) -> Self:
        """Accumulate the outcome of one LPEC-A working-set prediction.

        Parameters
        ----------
        prediction
            Prediction returned by
            :meth:`~slsqp_jax.sqpdax.active_set_prediction.LPECAPredictor.predict`.

        Returns
        -------
        Self
            Carry with the ``n_lpeca_*`` counters advanced.
        """
        return eqx.tree_at(
            lambda d: (d.n_lpeca_bypassed, d.n_lpeca_capped, d.n_lpeca_bounds_prefixed),
            self,
            (
                self.n_lpeca_bypassed + (~prediction.valid).astype(jnp.int32),
                self.n_lpeca_capped + prediction.capped.astype(jnp.int32),
                self.n_lpeca_bounds_prefixed + prediction.n_bounds_prefixed,
            ),
        )

    def record_step(
        self, *, accepted: Bool[Array, ""], accepted_by_fallback: Bool[Array, ""]
    ) -> Self:
        """Accumulate the acceptance outcome of one line-search step.

        Parameters
        ----------
        accepted
            Whether the step was accepted at all.
        accepted_by_fallback
            Whether the acceptance came from the fallback rule rather than
            the Armijo test (only meaningful when ``accepted`` is true).

        Returns
        -------
        Self
            Carry with ``last_ls_fallback`` refreshed and the acceptance
            counters advanced.
        """
        fallback_accept = accepted & accepted_by_fallback
        armijo_accept = accepted & ~accepted_by_fallback
        return eqx.tree_at(
            lambda d: (d.last_ls_fallback, d.n_armijo_accepts, d.n_fallback_accepts),
            self,
            (
                fallback_accept,
                self.n_armijo_accepts + armijo_accept.astype(jnp.int32),
                self.n_fallback_accepts + fallback_accept.astype(jnp.int32),
            ),
        )

    def stats(self) -> dict[str, Array]:
        """Flatten the carry into the keys reported in the final ``stats`` mapping."""
        return {
            "n_lpeca_bypassed": self.n_lpeca_bypassed,
            "n_lpeca_capped": self.n_lpeca_capped,
            "n_lpeca_bounds_prefixed": self.n_lpeca_bounds_prefixed,
            "last_ls_fallback": self.last_ls_fallback,
            "n_armijo_accepts": self.n_armijo_accepts,
            "n_fallback_accepts": self.n_fallback_accepts,
        }
