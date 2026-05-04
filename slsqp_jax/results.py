"""SLSQP-specific termination codes.

This module defines :class:`RESULTS`, a subclass of
``optimistix.RESULTS`` that adds finer-grained classification for the
SLSQP solver's failure modes.  The base class' members
(``successful``, ``nonlinear_max_steps_reached``,
``nonlinear_divergence``, ``nonfinite``, ...) are inherited; the
SLSQP-specific failure cases below replace what previously all mapped
to ``optx.RESULTS.nonlinear_divergence``.

.. note::

    Cross-class equality between members of ``optx.RESULTS`` and
    ``slsqp_jax.RESULTS`` raises ``ValueError`` per equinox's
    ``Enumeration`` semantics::

        sol.result == optx.RESULTS.successful   # raises ValueError
        sol.result == slsqp_jax.RESULTS.successful   # works

    Use :func:`is_successful` if you want a comparison that is robust
    to the parent/subclass distinction, or call
    ``slsqp_jax.RESULTS.promote(parent_result)`` to lift an
    ``optx.RESULTS`` value into this enumeration.
"""

from __future__ import annotations

from typing import Any

import optimistix as optx


class RESULTS(optx.RESULTS):  # type: ignore[misc]  # ty: ignore[subclass-of-final-class]
    """SLSQP-specific termination codes.

    Subclasses :class:`optimistix.RESULTS` and adds finer
    classifications for the failure modes that the upstream solver
    lumps into :attr:`optimistix.RESULTS.nonlinear_divergence`.

    .. note::

        :class:`equinox.Enumeration` reports its concrete subclasses
        as ``@final`` via the metaclass, but optimistix itself does
        the same trick (``optx.RESULTS`` subclasses ``lx.RESULTS``)
        and the docstring example for ``Enumeration.promote`` shows
        the pattern is supported.  ``ty: ignore[subclass-of-final-class]``
        suppresses the false positive.
    """

    merit_stagnation = (
        "The L1 merit function did not improve over the patience "
        "window (max_steps // 10 iterations). The iterate is "
        "primally feasible but stationarity could not be driven below "
        "rtol; this typically signals L-BFGS multiplier-recovery "
        "noise, a degenerate vertex, or an ill-conditioned problem. "
        "Try loosening rtol, switching to use_exact_hvp_in_qp=True, "
        "or checking constraint LICQ."
    )
    line_search_failure = (
        "Consecutive line-search failures exceeded "
        "2 * ls_failure_patience. The QP direction is not a descent "
        "direction for the L1 merit even after escalating L-BFGS "
        "soft-then-identity resets. Try MinresQLPSolver, exact HVPs, "
        "or a larger initial penalty parameter."
    )
    iterate_blowup = (
        "The L1 merit grew by more than divergence_factor times the "
        "best-seen merit (or returned NaN/Inf) for divergence_patience "
        "consecutive steps. The returned iterate is the best-merit "
        "iterate seen so far; subsequent steps were diverging."
    )
    infeasible = (
        "Termination at a primally infeasible iterate. Either the "
        "constraints are infeasible or the active-set machinery could "
        "not satisfy them within atol. Inspect c_eq / c_ineq at the "
        "returned iterate and consider relaxing constraints or "
        "providing a feasible initial point."
    )
    qp_subproblem_failure = (
        "Consecutive QP-subproblem failures exceeded "
        "2 * qp_failure_patience. Even after L-BFGS soft-then-identity "
        "resets, the inner QP solver could not produce a usable "
        "descent direction. Likely a rank-deficient equality "
        "Jacobian or extreme conditioning."
    )


def is_successful(result: Any) -> bool:
    """Return ``True`` if ``result`` represents successful convergence.

    Robust to both :class:`optimistix.RESULTS` and
    :class:`slsqp_jax.RESULTS` inputs by promoting the value before
    comparison.  Useful for downstream code that may receive results
    from either enumeration::

        from slsqp_jax import is_successful
        if is_successful(sol.result):
            ...

    Args:
        result: A ``RESULTS`` member from either ``optx.RESULTS`` or
            its :class:`RESULTS` subclass.

    Returns:
        ``True`` iff the result equals :attr:`RESULTS.successful`.
    """
    if isinstance(result, RESULTS):
        promoted = result
    else:
        promoted = RESULTS.promote(result)
    return bool(promoted == RESULTS.successful)
