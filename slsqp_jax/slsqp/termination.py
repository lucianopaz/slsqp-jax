"""Single source of truth for SLSQP termination classification.

Both ``SLSQP.step`` and ``SLSQP.terminate`` need to decide whether
the run should stop and *why*.  Historically these were two
near-identical bodies (the ``step`` mirror existed because Optimistix's
``iterate_solve`` driver requires the coarse ``optx.RESULTS`` enum
while we want to expose the granular :class:`slsqp_jax.RESULTS`).
This module collapses both into a pair of small pure helpers that
share their inputs by structure.

* :func:`classify_outcome` — given primal-feasibility / stationarity /
  failure flags, return the granular :class:`slsqp_jax.RESULTS` code.
  Used by ``step`` to populate ``state.termination_code``.
* :func:`coarse_outcome` — collapse the same flags onto the
  ``optx.RESULTS`` enum that the Optimistix driver requires.  Used
  by ``terminate``.
* :func:`compute_mu_max` — filterSQP's normalisation denominator
  (Fletcher & Leyffer, *User manual for filterSQP*, eq. 5) used as
  the reference scale for the relative-stationarity convergence
  test.  See the docstring for the exact formula.

Both classification helpers are thin pattern-matchers over the same
flag set; refactoring either one in the future should preserve their
structural symmetry.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, Bool, Float

from slsqp_jax.results import RESULTS
from slsqp_jax.types import Scalar, Vector


class TerminationFlags(NamedTuple):
    """Flag bundle classify_outcome / coarse_outcome both consume."""

    converged: Bool[Array, ""]
    nonfinite: Bool[Array, ""]
    diverging: Bool[Array, ""]
    ls_fatal: Bool[Array, ""]
    qp_fatal: Bool[Array, ""]
    merit_stagnation: Bool[Array, ""]
    max_iters_reached: Bool[Array, ""]
    primal_feasible: Bool[Array, ""]


def classify_outcome(flags: TerminationFlags) -> RESULTS:
    """Granular :class:`slsqp_jax.RESULTS` classification.

    Priority order (highest wins):

    1. ``successful``
    2. ``nonfinite``
    3. ``infeasible`` (override on any non-success termination at
       an infeasible iterate)
    4. ``iterate_blowup`` (best-iterate divergence rollback)
    5. ``line_search_failure``
    6. ``qp_subproblem_failure``
    7. ``merit_stagnation``
    8. ``nonlinear_max_steps_reached``
    """
    code = RESULTS.successful
    code = RESULTS.where(
        jnp.reshape(flags.max_iters_reached, ()),
        RESULTS.nonlinear_max_steps_reached,
        code,
    )
    code = RESULTS.where(
        jnp.reshape(flags.merit_stagnation, ()),
        RESULTS.merit_stagnation,
        code,
    )
    code = RESULTS.where(
        jnp.reshape(flags.qp_fatal, ()),
        RESULTS.qp_subproblem_failure,
        code,
    )
    code = RESULTS.where(
        jnp.reshape(flags.ls_fatal, ()),
        RESULTS.line_search_failure,
        code,
    )
    code = RESULTS.where(
        jnp.reshape(flags.diverging, ()),
        RESULTS.iterate_blowup,
        code,
    )
    non_success_done = (
        flags.merit_stagnation | flags.qp_fatal | flags.ls_fatal | flags.diverging
    )
    code = RESULTS.where(
        jnp.reshape(non_success_done & ~flags.primal_feasible, ()),
        RESULTS.infeasible,
        code,
    )
    code = RESULTS.where(
        jnp.reshape(flags.nonfinite, ()),
        RESULTS.nonfinite,
        code,
    )
    code = RESULTS.where(flags.converged, RESULTS.successful, code)
    return code


def coarse_outcome(flags: TerminationFlags) -> tuple[Bool[Array, ""], optx.RESULTS]:
    """Coarse ``(done, result)`` for the Optimistix driver.

    The driver requires the parent ``optx.RESULTS`` enum class
    (subclass instances would crash ``RESULTS.where``), so we only
    expose the four codes it knows about: ``successful``,
    ``nonfinite``, ``nonlinear_divergence`` (catch-all for stagnation /
    LS / QP / blowup), and ``nonlinear_max_steps_reached``.
    """
    done = (
        flags.converged
        | flags.max_iters_reached
        | flags.merit_stagnation
        | flags.ls_fatal
        | flags.qp_fatal
        | flags.diverging
        | flags.nonfinite
    )
    non_success_done = (
        flags.merit_stagnation | flags.ls_fatal | flags.qp_fatal | flags.diverging
    )
    result = jax.lax.cond(
        flags.converged,
        lambda: optx.RESULTS.successful,
        lambda: jax.lax.cond(
            flags.nonfinite,
            lambda: optx.RESULTS.nonfinite,
            lambda: jax.lax.cond(
                non_success_done,
                lambda: optx.RESULTS.nonlinear_divergence,
                lambda: jax.lax.cond(
                    flags.max_iters_reached,
                    lambda: optx.RESULTS.nonlinear_max_steps_reached,
                    lambda: optx.RESULTS.successful,
                ),
            ),
        ),
    )
    return done, result


def compute_mu_max(
    grad_f: Vector,
    eq_jac: Float[Array, "m_eq n"],
    ineq_jac_general: Float[Array, "m_ineq_general n"],
    mult_eq: Float[Array, " m_eq"],
    mult_ineq_general: Float[Array, " m_ineq_general"],
    mult_bound: Float[Array, " m_bound"],
) -> Scalar:
    """filterSQP normalisation denominator (eq. 5 of the manual).

    Computes::

        μ_max = max_i { ‖∇f‖₂, |ν_i|, ‖a_i‖₂ · |λ_i| }

    where the max is over the objective-gradient norm (one term),
    every bound multiplier ``ν_i`` (one term per active bound), and
    every general-constraint contribution ``‖a_i‖₂ · |λ_i|`` (one
    term per equality and per general inequality, ``a_i`` being the
    Jacobian row).  The result is the *largest single contributor*
    to the residual ``‖∇f − Jᵀλ − νᵀI_b‖`` and is exactly the
    reference scale filterSQP uses in eq. (6).

    Bound rows are not passed via a Jacobian: by construction the
    bound block of :func:`build_bound_jacobian` has rows ``±e_i`` of
    L2 norm ``1``, so ``‖a_i‖₂·|ν_i| = |ν_i|`` and we can feed the
    bound multipliers in directly.  This also matches the
    Fletcher–Leyffer split of ``λ`` (general) and ``ν`` (bound)
    multipliers in the manual.

    The empty-constraint case reduces cleanly to ``‖∇f‖₂``.

    Args:
        grad_f: Objective gradient ``∇f(x)``, shape ``(n,)``.
        eq_jac: Equality-constraint Jacobian, shape ``(m_eq, n)``.
            May be empty (``m_eq == 0``).
        ineq_jac_general: General-inequality-constraint Jacobian,
            shape ``(m_ineq_general, n)``.  *Excludes* the bound
            block — bound contributions are passed via
            ``mult_bound``.  May be empty.
        mult_eq: Equality multipliers ``λ_eq``, shape ``(m_eq,)``.
        mult_ineq_general: General-inequality multipliers
            ``λ_ineq``, shape ``(m_ineq_general,)``.
        mult_bound: Bound multipliers ``ν`` (lower bounds stacked on
            upper bounds, matching the layout of the bound block of
            ``state.multipliers_ineq_ls``), shape ``(m_bound,)``.

    Returns:
        Scalar ``μ_max``.  Has the dtype of ``grad_f``.
    """
    grad_norm = jnp.linalg.norm(grad_f)
    eq_terms = jnp.linalg.norm(eq_jac, axis=1) * jnp.abs(mult_eq)
    ineq_terms = jnp.linalg.norm(ineq_jac_general, axis=1) * jnp.abs(mult_ineq_general)
    bound_terms = jnp.abs(mult_bound)
    buf = jnp.concatenate(
        [
            jnp.reshape(grad_norm, (1,)),
            eq_terms,
            ineq_terms,
            bound_terms,
        ]
    )
    return jnp.reshape(jnp.max(buf), ())


__all__ = [
    "TerminationFlags",
    "classify_outcome",
    "coarse_outcome",
    "compute_mu_max",
]
