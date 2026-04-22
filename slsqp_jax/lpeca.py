"""LPEC-A Active Set Identification for SLSQP.

Implements the LPEC-A (Linear Program with Equilibrium Constraints
Approximation) method from Oberlin & Wright (2005, Section 3.3) for
identifying active inequality constraints in nonlinear programming.

The method computes a proximity measure ``rho_bar`` from primal
constraint values and dual multiplier estimates, then applies a
threshold test to predict the active set.  Under Mangasarian-Fromovitz
constraint qualification (MFCQ) and second-order sufficiency conditions,
the prediction is asymptotically exact as the iterate converges to
the solution.

An optional LP refinement step (via ``mpax.r2HPDHG``) can tighten
the multiplier estimates used in the threshold computation, but is
not required for asymptotic correctness.

**Far-from-solution trust gate.**  Theorem 5 of Oberlin & Wright only
guarantees asymptotic exactness when the iterate is *close* to a
solution (so ``rho_bar`` is small).  Far from the solution, the raw
``threshold = (beta * rho_bar) ** sigma`` can grow large enough that
the test ``c_ineq_i <= threshold`` includes nearly every constraint,
producing an over-saturated working set that destroys QP convergence.
The implementation guards against this with a configurable
``trust_threshold`` on ``rho_bar``: when ``rho_bar > trust_threshold``,
LPEC-A returns an *empty* prediction so the QP active-set loop falls
back to its warm-start / cold-start path.

**Rank-aware size cap.**  As a secondary safety net, even when the
trust gate passes, the prediction is truncated so that at most
``n - m_eq - 1`` constraints are predicted active.  This preserves a
LICQ-like rank margin in the working-set Jacobian
``[A_eq; A_active]`` (which has at most ``n`` rows).  Selection
prioritises the most-violated / most-confident constraints
(smallest ``c_ineq_i``).

Sign convention
---------------
This module uses the ``slsqp-jax`` convention where
``c_ineq(x) >= 0`` means feasible.  The Oberlin & Wright paper uses
``c(x) <= 0`` for feasible constraints, so all formulas are adapted
by mapping ``c_paper_i = -c_ineq_i``.

References
----------
Oberlin, C. & Wright, S. J. (2005). "An accelerated Newton method
for equations with semismooth Jacobians and nonlinear complementarity
problems." *Mathematical Programming*, 117(1-2), 355-386.
"""

from typing import NamedTuple

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Float, jaxtyped

from slsqp_jax.types import Scalar, Vector


class LPECAResult(NamedTuple):
    """LPEC-A active-set identification result.

    Attributes:
        predicted: Boolean mask of shape ``(m_ineq,)``.  ``True`` means
            the constraint is predicted active.  When ``valid=False``
            this is the all-``False`` mask (the trust gate fired).
        valid: ``True`` when ``rho_bar`` is below the trust threshold,
            so the prediction is theoretically meaningful.  ``False``
            indicates the iterate is too far from the solution for
            LPEC-A to be reliable.
        capped: ``True`` when the rank-aware size cap truncated the
            prediction (the raw threshold predicted more than
            ``n - m_eq - 1`` constraints active, so the most confident
            entries were kept).
        rho_bar: The scalar proximity measure used for the trust gate.
    """

    predicted: Bool[Array, " m_ineq"]
    valid: Bool[Array, ""]
    capped: Bool[Array, ""]
    rho_bar: Scalar


@jaxtyped(typechecker=beartype)
def compute_rho_bar(
    c_ineq: Float[Array, " m_ineq"],
    c_eq: Float[Array, " m_eq"],
    grad: Vector,
    A_ineq: Float[Array, "m_ineq n"],
    A_eq: Float[Array, "m_eq n"],
    lambda_ineq: Float[Array, " m_ineq"],
    mu_eq: Float[Array, " m_eq"],
) -> Scalar:
    """Compute the LPEC-A proximity measure rho_bar.

    This implements Eq. 36 of Oberlin & Wright (2005), adapted to the
    ``c_ineq >= 0`` (feasible) sign convention.

    For feasible inequality constraints (``c_ineq_i > 0``), the
    contribution is ``sqrt(c_ineq_i * lambda_ineq_i)``; for violated
    constraints (``c_ineq_i <= 0``), it is ``-c_ineq_i`` (the
    violation magnitude).

    Args:
        c_ineq: Inequality constraint values at current point.
        c_eq: Equality constraint values at current point.
        grad: Objective gradient at current point.
        A_ineq: Inequality constraint Jacobian.
        A_eq: Equality constraint Jacobian.
        lambda_ineq: Multiplier estimates for inequality constraints.
        mu_eq: Multiplier estimates for equality constraints.

    Returns:
        The scalar proximity measure rho_bar >= 0.
    """
    feasible = c_ineq > 0
    ineq_contribution = jnp.sum(
        jnp.where(
            feasible,
            jnp.sqrt(jnp.maximum(c_ineq * lambda_ineq, 0.0)),
            -c_ineq,
        )
    )

    eq_contribution = jnp.sum(jnp.abs(c_eq))

    stationarity_residual = grad
    if A_ineq.shape[0] > 0:
        stationarity_residual = stationarity_residual - A_ineq.T @ lambda_ineq
    if A_eq.shape[0] > 0:
        stationarity_residual = stationarity_residual - A_eq.T @ mu_eq
    stationarity_contribution = jnp.sum(jnp.abs(stationarity_residual))

    return ineq_contribution + eq_contribution + stationarity_contribution


@jaxtyped(typechecker=beartype)
def identify_active_set_lpeca(
    c_ineq: Float[Array, " m_ineq"],
    c_eq: Float[Array, " m_eq"],
    grad: Vector,
    A_ineq: Float[Array, "m_ineq n"],
    A_eq: Float[Array, "m_eq n"],
    lambda_ineq: Float[Array, " m_ineq"],
    mu_eq: Float[Array, " m_eq"],
    sigma: float = 0.9,
    beta: float | None = None,
    trust_threshold: float = 1.0,
) -> LPECAResult:
    """Predict the active inequality set using the LPEC-A threshold test.

    Applies Eq. 43 of Oberlin & Wright (2005), adapted to our sign
    convention.  An inequality constraint ``i`` is predicted active when
    ``c_ineq_i <= (beta * rho_bar) ** sigma`` *and* ``rho_bar`` is
    below the trust threshold (otherwise the prediction is empty).

    The result is wrapped in an :class:`LPECAResult` so the caller can
    distinguish "no constraints predicted active" (a valid prediction)
    from "LPEC-A bypassed" (``valid=False``) or "size cap fired"
    (``capped=True``).

    Args:
        c_ineq: Inequality constraint values at current point.
        c_eq: Equality constraint values at current point.
        grad: Objective gradient at current point.
        A_ineq: Inequality constraint Jacobian.
        A_eq: Equality constraint Jacobian.
        lambda_ineq: Multiplier estimates for inequality constraints.
        mu_eq: Multiplier estimates for equality constraints.
        sigma: Threshold exponent (``sigma_bar`` in the paper).
            Must be in (0, 1).  Default 0.9 per paper recommendation.
        beta: Threshold scaling factor.  Default ``None`` uses the
            paper's recommendation ``1 / (m_ineq + n + m_eq)``.
        trust_threshold: Maximum ``rho_bar`` for which the LPEC-A
            prediction is trusted.  When ``rho_bar > trust_threshold``,
            the predicted set is empty (the iterate is considered too
            far from the solution for the asymptotic guarantees to
            apply).  Default ``1.0``.

    Returns:
        :class:`LPECAResult` containing the boolean prediction mask
        and the diagnostic flags ``valid`` / ``capped`` / ``rho_bar``.
    """
    m_ineq = c_ineq.shape[0]
    n = grad.shape[0]
    m_eq = c_eq.shape[0]

    if beta is None:
        beta = 1.0 / max(m_ineq + n + m_eq, 1)

    rho_bar = compute_rho_bar(c_ineq, c_eq, grad, A_ineq, A_eq, lambda_ineq, mu_eq)

    # LPEC-A threshold (Oberlin & Wright 2005, Eq. 43).  Far from the
    # solution ``rho_bar`` can be large, which inflates the threshold
    # so much that nearly every inequality is predicted active; the
    # resulting over-saturated working set typically causes the QP
    # equality solve to fail.  Two layers of protection:
    #
    # 1. Trust gate: when ``rho_bar > trust_threshold`` (default 1.0)
    #    the asymptotic correctness conditions of Theorem 5 are not
    #    even approximately satisfied, so we return an empty prediction
    #    and let the QP active-set loop fall back to warm-start.  This
    #    replaces the previous ``min(threshold, max|c_ineq|)`` clamp
    #    which trivially evaluated to "all constraints active" because
    #    every ``c_ineq_i <= max|c_ineq_i|`` by construction.
    # 2. Size cap: even when the trust gate passes, the prediction is
    #    truncated so at most ``n - m_eq - 1`` constraints are
    #    predicted active.  This guarantees the working-set Jacobian
    #    ``[A_eq; A_active]`` retains a LICQ-like rank margin (at most
    #    ``n`` rows out of ``n + 1`` columns of slack).  Truncation
    #    keeps the most-violated constraints (smallest ``c_ineq_i``).
    threshold_raw = (beta * rho_bar) ** sigma
    valid = rho_bar <= trust_threshold
    threshold = jnp.where(valid, threshold_raw, 0.0)
    raw_predicted = (c_ineq <= threshold) & valid

    # Rank-aware size cap.  ``n_dof`` is static (computed from shape
    # constants), so the cap can be expressed with a static mask
    # ``rank < n_dof`` which is jit-friendly.
    n_dof_static = max(n - m_eq - 1, 1)
    if m_ineq == 0:
        return LPECAResult(
            predicted=raw_predicted,
            valid=valid,
            capped=jnp.array(False),
            rho_bar=rho_bar,
        )

    n_dof = min(n_dof_static, m_ineq)
    # Score: most violated (smallest c_ineq_i) wins.  Non-predicted
    # entries get -inf so they are never selected.
    scores = jnp.where(raw_predicted, -c_ineq, -jnp.inf)
    # ``argsort(argsort(-scores))`` produces 0-indexed descending
    # rank: 0 = largest score = most violated predicted entry.
    ranks = jnp.argsort(jnp.argsort(-scores))
    capped_predicted = raw_predicted & (ranks < n_dof)

    predicted_count = jnp.sum(raw_predicted.astype(jnp.int32))
    capped_flag = predicted_count > n_dof

    return LPECAResult(
        predicted=capped_predicted,
        valid=valid,
        capped=capped_flag,
        rho_bar=rho_bar,
    )


@jaxtyped(typechecker=beartype)
def solve_lpeca_lp(
    c_ineq: Float[Array, " m_ineq"],
    c_eq: Float[Array, " m_eq"],
    grad: Vector,
    A_ineq: Float[Array, "m_ineq n"],
    A_eq: Float[Array, "m_eq n"],
    lambda_bound: float = 1e6,
    eps_abs: float = 1e-6,
    eps_rel: float = 1e-6,
    max_iter: int = 1000,
) -> tuple[Float[Array, " m_ineq"], Float[Array, " m_eq"]]:
    """Solve the LPEC-A LP to obtain tighter multiplier estimates.

    Solves the LP from Eq. 42 of Oberlin & Wright (2005), adapted to
    the ``c_ineq >= 0`` sign convention::

        min_{lambda, mu, u, v}  sum(c_ineq_i * lambda_i for feasible i)
                                + e^T u + e^T v
        s.t.  grad - A_ineq^T lambda - A_eq^T mu = u - v
              0 <= lambda <= K_1
              u, v >= 0

    The LP is solved using ``mpax.r2HPDHG``, the reflected restarted
    Halpern PDHG algorithm (Lu & Yang, 2024), which achieves
    accelerated linear convergence on LP.

    Args:
        c_ineq: Inequality constraint values at current point.
        c_eq: Equality constraint values at current point.
        grad: Objective gradient at current point.
        A_ineq: Inequality constraint Jacobian.
        A_eq: Equality constraint Jacobian.
        lambda_bound: Upper bound ``K_1`` on lambda.  Default 1e6.
        eps_abs: Absolute tolerance for the LP solver.
        eps_rel: Relative tolerance for the LP solver.
        max_iter: Maximum LP solver iterations.

    Returns:
        Tuple of (lambda_ineq, mu_eq) — the LP-optimal multiplier
        estimates for inequality and equality constraints.

    Raises:
        ImportError: If ``mpax`` is not installed.
    """
    try:
        from mpax import create_lp, r2HPDHG
    except ImportError:
        raise ImportError(
            "The LPEC-A LP solve requires the 'mpax' package. "
            "Install it with:\n"
            "  pip install slsqp-jax[extras]\n"
            "or\n"
            "  uv sync --group extras"
        ) from None

    m_ineq = c_ineq.shape[0]
    m_eq = c_eq.shape[0]
    n = grad.shape[0]

    # Decision variables: z = [lambda (m_ineq), mu (m_eq), u (n), v (n)]
    n_vars = m_ineq + m_eq + 2 * n

    # Objective: sum(c_ineq_i * lambda_i for feasible i) + e^T u + e^T v
    # For feasible constraints (c_ineq > 0), the cost on lambda_i is c_ineq_i.
    # For violated constraints, the cost is 0 (they're not penalized in the LP
    # objective, only through the stationarity constraint).
    feasible = c_ineq > 0
    obj_lambda = jnp.where(feasible, c_ineq, 0.0)
    obj_mu = jnp.zeros(m_eq)
    obj_u = jnp.ones(n)
    obj_v = jnp.ones(n)
    obj_vec = jnp.concatenate([obj_lambda, obj_mu, obj_u, obj_v])

    # Equality constraint: grad - A_ineq^T lambda - A_eq^T mu - u + v = 0
    # Rewritten as: [-A_ineq^T, -A_eq^T, -I, I] z = -grad
    lp_A_eq = jnp.concatenate(
        [
            -A_ineq.T if m_ineq > 0 else jnp.zeros((n, 0)),
            -A_eq.T if m_eq > 0 else jnp.zeros((n, 0)),
            -jnp.eye(n),
            jnp.eye(n),
        ],
        axis=1,
    )
    lp_b_eq = -grad

    # Bounds: 0 <= lambda <= K_1, -inf <= mu <= inf, u >= 0, v >= 0
    lb = jnp.concatenate(
        [
            jnp.zeros(m_ineq),
            jnp.full(m_eq, -jnp.inf),
            jnp.zeros(n),
            jnp.zeros(n),
        ]
    )
    ub = jnp.concatenate(
        [
            jnp.full(m_ineq, lambda_bound),
            jnp.full(m_eq, jnp.inf),
            jnp.full(n, jnp.inf),
            jnp.full(n, jnp.inf),
        ]
    )

    lp = create_lp(
        c=obj_vec,
        A=lp_A_eq,
        b=lp_b_eq,
        G=jnp.zeros((0, n_vars)),
        h=jnp.zeros(0),
        l=lb,
        u=ub,
    )

    solver = r2HPDHG(
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        iteration_limit=max_iter,
        jit=True,
        verbose=False,
    )
    result = solver.optimize(lp)

    # Extract multiplier estimates from primal solution
    lambda_opt = result.primal_solution[:m_ineq]
    mu_opt = result.primal_solution[m_ineq : m_ineq + m_eq]

    return lambda_opt, mu_opt


def compute_lpeca_active_set(
    c_ineq: Float[Array, " m_ineq"],
    c_eq: Float[Array, " m_eq"],
    grad: Vector,
    A_ineq: Float[Array, "m_ineq n"],
    A_eq: Float[Array, "m_eq n"],
    lambda_ineq: Float[Array, " m_ineq"],
    mu_eq: Float[Array, " m_eq"],
    sigma: float = 0.9,
    beta: float | None = None,
    trust_threshold: float = 1.0,
    use_lp: bool = False,
    lp_lambda_bound: float = 1e6,
    lp_eps: float = 1e-6,
    lp_max_iter: int = 1000,
) -> LPECAResult:
    """Compute the LPEC-A predicted active set, optionally refining multipliers.

    This is the main entry point for LPEC-A active set identification.
    It optionally solves the LPEC-A LP to obtain tighter multiplier
    estimates, then applies the threshold test (with the trust gate
    and rank-aware size cap from :func:`identify_active_set_lpeca`).

    Args:
        c_ineq: Inequality constraint values at current point.
        c_eq: Equality constraint values at current point.
        grad: Objective gradient at current point.
        A_ineq: Inequality constraint Jacobian.
        A_eq: Equality constraint Jacobian.
        lambda_ineq: Current multiplier estimates for inequalities.
        mu_eq: Current multiplier estimates for equalities.
        sigma: Threshold exponent (default 0.9).
        beta: Threshold scaling factor (default: paper recommendation).
        trust_threshold: Maximum ``rho_bar`` for which the prediction
            is trusted (see :func:`identify_active_set_lpeca`).
        use_lp: If True, solve the LPEC-A LP to refine multiplier
            estimates before the threshold test.  Requires ``mpax``.
        lp_lambda_bound: Upper bound K_1 on lambda in the LP.
        lp_eps: Tolerance for the LP solver.
        lp_max_iter: Maximum LP solver iterations.

    Returns:
        :class:`LPECAResult` with the predicted active mask and
        diagnostic flags.
    """
    if use_lp:
        lambda_ineq, mu_eq = solve_lpeca_lp(
            c_ineq=c_ineq,
            c_eq=c_eq,
            grad=grad,
            A_ineq=A_ineq,
            A_eq=A_eq,
            lambda_bound=lp_lambda_bound,
            eps_abs=lp_eps,
            eps_rel=lp_eps,
            max_iter=lp_max_iter,
        )

    return identify_active_set_lpeca(
        c_ineq=c_ineq,
        c_eq=c_eq,
        grad=grad,
        A_ineq=A_ineq,
        A_eq=A_eq,
        lambda_ineq=lambda_ineq,
        mu_eq=mu_eq,
        sigma=sigma,
        beta=beta,
        trust_threshold=trust_threshold,
    )
