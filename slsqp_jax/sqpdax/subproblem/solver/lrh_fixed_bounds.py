"""QP of the limited-memory reduced-Hessian model on a fixed bound working set.

Given the quadratic model ``B`` of a reduced-Hessian secant (as
:class:`~slsqp_jax.sqpdax.secant.reduced_hessian.ReducedHessianFactors`), the
QP of an SQP step is

```text
min  gᵀd + ½ dᵀ B d    s.t.  A_E d = b_E,   A_I d ≤ b_I,   d_W = d_fix
```

with ``g = ∇f(x)``, ``A_E = J_g``, ``b_E = -g(x)``, ``A_I = J_h``,
``b_I = -h(x)`` (constraints ``g = 0``, ``h ≤ 0``) and a set ``W`` of
variables pinned at a bound (``d_fix = lb - x`` or ``ub - x`` there). This is
the *SQP view*: the returned multipliers are those of the new iterate, not
increments (see
:class:`~slsqp_jax.sqpdax.subproblem.active_set.ActiveSetSubProblem`).

:func:`solve_fixed_bounds_qp` solves it for one given ``W`` with direct linear
algebra (no iteration other than the NNLS inside the least-distance solver):
the pinned block is eliminated, the free block is transformed with the factor
``C_F`` of ``B_FF`` (:class:`~slsqp_jax.sqpdax.linalg.free_set_factor.FreeSetFactor`)
into a least-distance problem on ``m_c`` unknowns
(:func:`~slsqp_jax.sqpdax.linalg.lsei.least_distance_lsei`), and the step and
multipliers are mapped back. The bound multipliers on ``W`` follow from
stationarity. Bounds are never stacked into the constraint matrix: ``W`` is a
pair of boolean masks, and every dense object is ``n × m`` (LRH memory),
``n × m_c``, ``m_c × n`` or smaller.

**Elastic variant.** With ``elastic_rho`` the same QP is relaxed by one scalar
``δ ∈ [0, 1]`` (Kraft's device for inconsistent linearisations, see
:func:`~slsqp_jax.sqpdax.linalg.lsei.elastic_least_distance_lsei`):
``A_E d + δ b_E = b_E``, ``A_I,j d + δ b_I,j ≤ b_I,j`` on the rows violated at
``d = 0``, objective ``+ ½ ρ_e δ²``. The column of ``δ`` is the **unshifted**
``b``, so the elastic QP is the same for every working set (only the
right-hand side carries ``d_fix``); ``ρ_e`` is an absolute weight computed
once per QP by :func:`elastic_penalty` and frozen across the working-set loop.
``δ`` is a general variable: it never enters the returned multipliers.
"""

from __future__ import annotations

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from ...dual import Dual
from ...lagrangian.evaluated import EvaluatedLagrangian
from ...linalg.free_set_factor import FreeSetFactor
from ...linalg.lawson_hanson import LDP_RESULTS
from ...linalg.lsei import (
    elastic_delta_column,
    elastic_least_distance_lsei,
    elastic_weight_scale,
    least_distance_lsei,
)
from ...primal import Primal
from ...secant.reduced_hessian import ReducedHessianFactors
from ...types import Scalar, Vector_n

# Weight factor of the certificate retry of the elastic variant (see the
# comment at its use).
_CERT_RETRY_SCALE = 1e-2

__all__ = ["FixedBoundsQPResult", "elastic_penalty", "solve_fixed_bounds_qp"]


class FixedBoundsQPResult(eqx.Module):
    """Solution of the QP on a fixed bound working set.

    Attributes
    ----------
    step
        The step ``d``, shape ``(n,)``. On the pinned variables it equals
        ``d_fix`` exactly (``lb - x`` on the lower face, ``ub - x`` on the
        upper face). **When ``ldp_status`` is not ``feasible`` it is the
        fallback ``d_fix`` (zero on the free variables)**, not a solution of
        the QP: callers must branch on the status.
    dual
        Multipliers of the QP (SQP view): ``eq_multipliers`` (free sign,
        minimum-norm when equality rows are dependent), ``ineq_multipliers``
        (``≥ 0``), ``lb_multipliers`` and ``ub_multipliers`` (zero off the
        working set; on it, the stationarity residual with the sign of the
        face, ``λ_lb = ρ_s`` / ``λ_ub = -ρ_s``, which is **not** clipped to be
        non-negative: deciding which bounds to release is the job of the
        caller). **All zero when ``ldp_status`` is not ``feasible``.**
    ldp_status
        :class:`~slsqp_jax.sqpdax.linalg.lawson_hanson.LDP_RESULTS` code of
        the least-distance reduction: ``feasible``, ``infeasible``,
        ``inconsistent_equalities``, ``max_iter_reached`` or ``nonfinite``.
    n_nnls_iter
        Number of NNLS steps of the least-distance problem (``int32``).
    feasibility_residual
        ``max(‖A_E d - b_E‖_∞, max(A_I d - b_I, 0))`` at the returned
        ``step`` (zero without general constraints). It is computed for the
        fallback step too, where it measures how far ``d_fix`` is from
        satisfying the general constraints. The bound constraints are met by
        construction on ``W`` and are not part of it.
    certificate_eq
        Equality part ``y_E`` (free sign) of the Farkas certificate of the
        pinned QP, shape ``(m_E,)``. Zero unless ``has_ray``.
    certificate_ineq
        Inequality part ``y_I ≥ 0`` of the certificate, shape ``(m_I,)``.
        Zero unless ``has_ray``. With ``y = [y_E; y_I]``, ``A_Fᵀ y ≈ 0`` on
        the free variables and ``b̃ᵀ y < 0`` (``b̃ = b - A d_fix``), scaled to
        ``‖y‖_∞ = 1``: it proves the pinned QP infeasible. For the elastic
        variant the unit sup-norm is that of the augmented certificate
        ``[y; y_lo; y_hi]`` (the entries of the rows ``δ ≥ 0``, ``δ ≤ 1``), so
        ``‖y‖_∞ ≤ 1`` there, and the proof is ``b̃ᵀ y + max(-cᵀy, 0) < 0``
        (``c`` the column of ``δ``) instead of ``b̃ᵀ y < 0``.
    ray_lb, ray_ub
        Ray of the bound multipliers of the pinned QP, shape ``(n,)``.
        With ``ρ_y = A_Eᵀ y_E + A_Iᵀ y_I``, ``ray_lb = ρ_y`` on the lower
        pins and ``ray_ub = -ρ_y`` on the upper pins, zero elsewhere (the
        stationarity formula without ``B d + g``), so
        ``A_Eᵀ y_E + A_Iᵀ y_I - ray_lb + ray_ub = 0`` on all ``n``
        coordinates (zero on the free ones: ``ρ_y`` vanishes there). Along
        the ray the dual objective grows at rate ``-b̃ᵀ y > 0``. On an
        ``lb == ub`` pin (carried by the lower face) the entry is
        free-signed. Zero unless ``has_ray``.
    has_ray
        Boolean scalar: true iff ``ldp_status`` is ``infeasible`` or
        ``inconsistent_equalities`` and a finite certificate was formed
        (see :func:`~slsqp_jax.sqpdax.linalg.lsei.least_distance_lsei`).
        False, with all four fields above zero, when the status is
        ``feasible``, ``max_iter_reached`` or ``nonfinite``, and on the
        primal-check ``infeasible`` of the least-distance solver.
    ray_rate
        Growth rate of the dual objective along the ray, ``-b̃ᵀ y > 0`` (for
        the elastic variant ``-(b̃ᵀ y_gen + y_hi)``, with ``y_gen`` the
        certificate of the general rows and ``y_hi`` the certificate entry of
        the row ``δ ≤ 1``: the ``δ`` rows of the certificate satisfy
        ``cᵀ y_gen - y_lo + y_hi = 0``). Zero unless ``has_ray``.
    delta
        The relaxation ``δ ∈ [0, 1]`` of the elastic variant (``0`` for the
        plain variant, and whenever ``ldp_status`` is not ``feasible``).
    """

    step: Vector_n
    dual: Dual
    ldp_status: LDP_RESULTS
    n_nnls_iter: Int[Array, ""]
    feasibility_residual: Scalar
    certificate_eq: Float[Array, " m_eq"]
    certificate_ineq: Float[Array, " m_ineq"]
    ray_lb: Vector_n
    ray_ub: Vector_n
    has_ray: Bool[Array, ""]
    ray_rate: Scalar
    delta: Scalar


def _general_rows(ev, dtype):
    """General constraint rows ``A d = / ≤ b`` (``[E, I]``) of the SQP-view QP."""
    a_mat = jnp.concatenate([ev.eq_fn_jac_val, ev.ineq_fn_jac_val], axis=0).astype(
        dtype
    )
    b_vec = -jnp.concatenate([ev.eq_fn_val, ev.ineq_fn_val], axis=0).astype(dtype)
    return a_mat, b_vec


def elastic_penalty(
    lagrangian: EvaluatedLagrangian[Primal],
    factors: ReducedHessianFactors,
    elastic_weight: float,
    *,
    rank_tol: float | None = None,
) -> Scalar:
    """Absolute elastic weight ``ρ_e = elastic_weight · τ²`` of one QP.

    ``τ²`` is :func:`~slsqp_jax.sqpdax.linalg.lsei.elastic_weight_scale` of the
    **unpinned** least-distance data ``G₀ = A C⁻¹``, ``h₀ = b + A B⁻¹ g`` (no
    variable pinned, ``C`` the factor of the full model ``B``) and the column
    ``c = b`` of ``δ``. The elastic QP must be one fixed QP for every bound
    working set, so this scale is computed once per QP and the same ``ρ_e`` is
    passed (as ``elastic_rho``) to every call of
    :func:`solve_fixed_bounds_qp`. Cost ``O(n r m_c)``.

    Parameters
    ----------
    lagrangian
        Evaluated Lagrangian at the reference point (as for
        :func:`solve_fixed_bounds_qp`).
    factors
        Reduced-Hessian factors of the model ``B``.
    elastic_weight
        Dimensionless weight (default of the solver ``1e3``).
    rank_tol
        As in :func:`solve_fixed_bounds_qp`.

    Returns
    -------
    Array
        The scalar ``ρ_e`` (``nan`` on non-finite data).
    """
    ev = lagrangian.evaluated
    dtype = jnp.result_type(ev.grad_val, factors.basis, jnp.float32)
    g = jnp.asarray(ev.grad_val, dtype=dtype)
    m_e = ev.meq
    a_mat, b_vec = _general_rows(ev, dtype)
    free = jnp.ones(g.shape, dtype=bool)
    factor = FreeSetFactor.build(factors, free, rank_tol=rank_tol)
    g_t0 = factor.apply_c_inv_t_matrix(a_mat.T)
    h0 = b_vec + a_mat @ factor.solve(g)
    delta_col = elastic_delta_column(b_vec, m_e)
    return elastic_weight * elastic_weight_scale(g_t0, h0, delta_col, m_e)


def solve_fixed_bounds_qp(
    lagrangian: EvaluatedLagrangian[Primal],
    factors: ReducedHessianFactors,
    active_lb: Bool[Array, " n"],
    active_ub: Bool[Array, " n"],
    *,
    rank_tol: float | None = None,
    eq_tol: float | None = None,
    nnls_max_iter: int | None = None,
    elastic_rho: float | Float[Array, ""] | None = None,
) -> FixedBoundsQPResult:
    """Solve the LRH QP with the variables of a working set pinned at a bound.

    The working set is ``W = (active_lb & ~null_lb) | (active_ub & ~null_ub)``:
    a flag on an infinite bound is ignored. A variable flagged on both faces
    is pinned at the **lower** bound (so ``lb == ub`` is handled, with the
    multiplier carried by the lower face). The free set is ``F = ¬W``.

    The pinned block is eliminated, ``g̃ = I_F (g + B d_fix)`` (with the full
    ``B`` applied to the full ``d_fix``), ``b̃ = b - A d_fix``,
    ``A_F = A I_F``. The substitution ``w = C_F d_F + C_F⁻ᵀ g̃`` turns the QP
    into ``min ½‖w‖²  s.t.  G w ≤ h`` (equalities as ``=``) with
    ``G = A_F C_F⁻¹`` and ``h = b̃ + A_F B_FF⁻¹ g̃``, solved by
    :func:`~slsqp_jax.sqpdax.linalg.lsei.least_distance_lsei`. The step is
    ``d = C_F⁻¹ w - B_FF⁻¹ g̃ + d_fix``. The multipliers ``λ_E``, ``λ_I`` of
    the least-distance problem are those of the QP, and the bound
    multipliers come from the stationarity residual
    ``ρ_s = B d + g + A_Eᵀ λ_E + A_Iᵀ λ_I``. With the sign conventions
    ``L = f + λ_Eᵀ g + λ_Iᵀ h + λ_lbᵀ(lb - x) + λ_ubᵀ(x - ub)``,

    ```text
    B d + g + A_Eᵀ λ_E + A_Iᵀ λ_I - λ_lb + λ_ub = 0
    ```

    holds on the free variables up to the accuracy of the solve and on ``W``
    by construction.

    The ``ldp_status`` is the one of the least-distance reduction. Unless it
    is ``feasible`` the step is ``d_fix`` (zero on the free variables) and the
    dual is zero: consumers (the working-set loop) must branch on it, never
    read those outputs as a QP solution (a zero ``w`` would silently be the
    unconstrained step ``-B_FF⁻¹ g̃``, which is why it is not returned). No
    elastic relaxation is attempted unless ``elastic_rho`` is given.

    **Elastic variant** (``elastic_rho`` not ``None``, a static choice; the
    value may be traced). The least-distance problem is replaced by
    :func:`~slsqp_jax.sqpdax.linalg.lsei.elastic_least_distance_lsei` with the
    column ``c = b`` of ``δ`` (the **unshifted** right-hand side, built by
    :func:`~slsqp_jax.sqpdax.linalg.lsei.elastic_delta_column`; it is not
    transformed by ``C_F``) and the absolute weight ``ρ_e = elastic_rho``
    (use :func:`elastic_penalty`). The relaxation reads
    ``A_F,E d_F = b̃_E - δ b_E`` and ``A_F,I d_F ≤ b̃_I - δ b_I,j`` on the rows
    with ``b_I,j < 0``. The result carries ``delta``; ``dual`` holds the
    multipliers of the general rows only (the rows of ``δ`` never enter it).
    The step, the bound multipliers (stationarity, ``δ`` does not enter
    ``B d + g + Aᵀλ``) and the certificate are formed as in the plain variant;
    the growth rate of the dual objective along the ray is
    ``-(b̃ᵀ y_gen + y_hi)`` (``ray_rate``). With no pinned variable the
    elastic problem is feasible whenever ``d = 0`` meets the relaxed rows at
    ``δ = 1``, and an infeasible *pinned* elastic problem is a ray step. The
    augmented least-distance
    problem is ill conditioned at the default weight (the ``δ`` coordinate is
    scaled by ``√ρ_e``), and on an infeasible pinned problem its NNLS residual
    can land just above the noise threshold of the infeasibility test, which
    gives ``ldp_status = infeasible`` *without* a certificate (about 6 % of the
    infeasible pinned solves of the near-infeasible battery at weight ``1e3``).
    The infeasibility of the pinned problem in ``(w, δ)`` and its Farkas
    certificate (a statement about the rows) do not depend on ``ρ_e``, so in
    that case the certificate is taken from a re-solve at ``ρ_e / 100``; the
    step, multipliers and status are not changed.

    **Farkas certificate.** When the pinned QP is infeasible (``ldp_status``
    ``infeasible`` or ``inconsistent_equalities``) and the least-distance
    solver formed a certificate, ``has_ray`` is true and the result carries
    ``y = [certificate_eq; certificate_ineq]`` with ``y_I ≥ 0``,
    ``A_Fᵀ y ≈ 0`` (``Gᵀ y = 0`` with ``Gᵀ = C_F⁻ᵀ A_Fᵀ``) and
    ``b̃ᵀ y < 0`` (``hᵀ y = b̃ᵀ y + (Gᵀ y)ᵀ C_F⁻ᵀ g̃ = b̃ᵀ y``), plus the ray
    ``ρ_y = A_Eᵀ y_E + A_Iᵀ y_I`` of the bound multipliers: ``ray_lb = ρ_y``
    on the lower pins, ``ray_ub = -ρ_y`` on the upper pins. It costs
    ``O(n m_c)`` and no extra solve.

    Parameters
    ----------
    lagrangian
        Evaluated Lagrangian at the reference point; only the objective
        gradient, the equality / inequality values and Jacobians and the
        bounds with their null masks are read (the Hessian comes from
        ``factors``). ``x`` may lie outside the bounds.
    factors
        Reduced-Hessian factors of the model ``B``.
    active_lb, active_ub
        Boolean masks, shape ``(n,)``, of the variables pinned at their lower
        / upper bound. They may be traced.
    rank_tol
        Eigenvalue threshold passed to
        :meth:`~slsqp_jax.sqpdax.linalg.free_set_factor.FreeSetFactor.build`.
        ``None`` selects its default.
    eq_tol
        Consistency tolerance of the equalities passed to
        :func:`~slsqp_jax.sqpdax.linalg.lsei.least_distance_lsei` (in the
        units of the equilibrated problem). ``None`` selects ``√eps`` of the
        working dtype.
    nnls_max_iter
        Iteration budget of the NNLS inside the least-distance solver
        (``None``: its default).
    elastic_rho
        Absolute elastic weight ``ρ_e > 0`` (see :func:`elastic_penalty`);
        ``None`` (default) solves the plain QP.

    Returns
    -------
    FixedBoundsQPResult
        ``step``, ``dual``, ``ldp_status``, ``n_nnls_iter``,
        ``feasibility_residual``, the certificate fields (``certificate_eq``,
        ``certificate_ineq``, ``ray_lb``, ``ray_ub``, ``has_ray``,
        ``ray_rate``) and ``delta``.

    Examples
    --------
    Minimise ``½‖d‖² + gᵀd`` with ``d_0`` pinned at its lower bound ``-0.5``
    (``x = 0``), no general constraint:

    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.dual import Dual
    >>> from slsqp_jax.sqpdax.lagrangian.evaluated import EvaluatedLagrangian
    >>> from slsqp_jax.sqpdax.primal import Primal
    >>> from slsqp_jax.sqpdax.problem.basic import EvaluatedProblem
    >>> from slsqp_jax.sqpdax.secant import ReducedHessianFactors
    >>> from slsqp_jax.sqpdax.subproblem.solver.lrh_fixed_bounds import (
    ...     solve_fixed_bounds_qp,
    ... )
    >>> n = 2
    >>> ev = EvaluatedProblem(
    ...     ref=Primal(x=jnp.zeros(n)),
    ...     fn_val=jnp.zeros(()),
    ...     grad_val=jnp.array([1.0, -2.0]),
    ...     fn_qvp=None,
    ...     eq_fn_val=jnp.zeros((0,)),
    ...     eq_fn_jac_val=jnp.zeros((0, n)),
    ...     eq_fn_qvp=None,
    ...     ineq_fn_val=jnp.zeros((0,)),
    ...     ineq_fn_jac_val=jnp.zeros((0, n)),
    ...     ineq_fn_qvp=None,
    ...     lb=jnp.array([-0.5, -jnp.inf]),
    ...     ub=jnp.full(n, jnp.inf),
    ...     null_lb=jnp.array([False, True]),
    ...     null_ub=jnp.array([True, True]),
    ...     aux_val=None,
    ... )
    >>> dual = Dual(jnp.zeros((0,)), jnp.zeros((0,)), jnp.zeros(n), jnp.zeros(n))
    >>> lag = EvaluatedLagrangian(evaluated=ev, secant=None, dual=dual)
    >>> factors = ReducedHessianFactors(  # rank 0: B = I
    ...     basis=jnp.zeros((n, 1)),
    ...     cholesky=jnp.eye(1),
    ...     sigma=jnp.asarray(1.0),
    ...     rank=jnp.asarray(0, dtype=jnp.int32),
    ... )
    >>> res = solve_fixed_bounds_qp(
    ...     lag, factors, jnp.array([True, False]), jnp.array([False, False])
    ... )
    >>> [round(float(v), 6) for v in res.step]
    [-0.5, 2.0]
    >>> round(float(res.dual.lb_multipliers[0]), 6)  # B d + g = -0.5 + 1
    0.5
    """
    ev = lagrangian.evaluated
    dtype = jnp.result_type(ev.grad_val, factors.basis, jnp.float32)
    x = jnp.asarray(lagrangian.x_ref, dtype=dtype)
    g = jnp.asarray(ev.grad_val, dtype=dtype)
    m_e, m_i = ev.meq, ev.mineq
    m_c = m_e + m_i
    eps = float(jnp.finfo(dtype).eps)
    eq_tol = eps**0.5 if eq_tol is None else eq_tol

    # Working set: infinite bounds never enter W; the lower face wins a tie.
    pin_lb = jnp.asarray(active_lb, dtype=bool) & ~lagrangian.null_lb
    pin_ub = jnp.asarray(active_ub, dtype=bool) & ~lagrangian.null_ub & ~pin_lb
    free = ~(pin_lb | pin_ub)
    # Target of every variable: its pinned bound, else x itself (d_fix = 0).
    target = jnp.where(
        pin_lb, lagrangian.lb, jnp.where(pin_ub, lagrangian.ub, x)
    ).astype(dtype)
    d_fix = target - x

    # General constraints (m_c x n at most): A d = / <= b.
    a_mat, b_vec = _general_rows(ev, dtype)

    # Eliminate the pinned block (full B applied to the full d_fix, then masked).
    factor = FreeSetFactor.build(factors, free, rank_tol=rank_tol)
    g_tilde = jnp.where(free, g + factors.hvp(d_fix), 0.0)
    b_tilde = b_vec - a_mat @ d_fix
    a_free = jnp.where(free[None, :], a_mat, 0.0)

    # Least-distance problem on w = C_F d_F + C_F^{-T} g~.
    g_t = factor.apply_c_inv_t_matrix(a_free.T)  # (n, m_c): G^T = C_F^{-T} A_F^T
    b_inv_g = factor.solve(g_tilde)
    h = b_tilde + a_free @ b_inv_g
    if elastic_rho is None:
        lsei = least_distance_lsei(g_t, h, m_e, eq_tol=eq_tol, max_iter=nnls_max_iter)
        w, delta_raw, y_hi = lsei.w, jnp.zeros((), dtype), jnp.zeros((), dtype)
    else:
        # Relaxation A_F d_F + δ b = b~: the column of δ is the UNSHIFTED b,
        # untransformed by C_F, and ρ_e is the frozen absolute weight.
        delta_col = elastic_delta_column(b_vec, m_e)
        rho_e = jnp.asarray(elastic_rho, dtype)

        def elastic_solve(rho):
            return elastic_least_distance_lsei(
                g_t,
                h,
                m_e,
                delta_col,
                rho,
                eq_tol=eq_tol,
                max_iter=nnls_max_iter,
            )

        lsei = elastic_solve(rho_e)
        # Certificate retry. The pinned elastic problem is often infeasible (a
        # ray step of the loop), and the least-distance solver must then hand
        # back a Farkas certificate. At the default weight the augmented
        # problem is ill conditioned (the δ coordinate is scaled by √ρ_e) and
        # the NNLS residual of an infeasible problem can land just above the
        # noise threshold of the infeasibility test: the status is then
        # ``infeasible`` (by the primal check) but there is no certificate.
        # Infeasibility of the pinned problem in (w, δ) does not depend on
        # ρ_e (it only rescales the δ coordinate) and a certificate is a
        # statement about the rows, so re-solving with a smaller weight, which
        # is well conditioned, yields a valid certificate of the same problem.
        # The retry replaces only the certificate and only when it forms one.
        need_cert = (lsei.status == LDP_RESULTS.infeasible) & ~lsei.has_certificate
        alt = jax.lax.cond(
            need_cert,
            lambda: elastic_solve(rho_e * _CERT_RETRY_SCALE),
            lambda: lsei,
        )
        take = need_cert & alt.has_certificate & (alt.status == LDP_RESULTS.infeasible)
        lsei = eqx.tree_at(
            lambda r: (
                r.cert_eq,
                r.cert_ineq,
                r.cert_delta,
                r.has_certificate,
                r.n_iter,
            ),
            lsei,
            (
                jnp.where(take, alt.cert_eq, lsei.cert_eq),
                jnp.where(take, alt.cert_ineq, lsei.cert_ineq),
                jnp.where(take, alt.cert_delta, lsei.cert_delta),
                take | lsei.has_certificate,
                lsei.n_iter + jnp.where(need_cert, alt.n_iter, 0),
            ),
        )
        w, delta_raw, y_hi = lsei.w, lsei.delta.astype(dtype), lsei.cert_delta[1]

    # Step and bound multipliers (garbage unless ``ok``; selected below).
    d_free = factor.apply_c_inv(w) - b_inv_g
    step_ok = d_free + d_fix
    lam_eq, lam_ineq = lsei.lam_eq, lsei.lam_ineq
    rho_s = (
        factors.hvp(step_ok)
        + g
        + lam_eq @ ev.eq_fn_jac_val.astype(dtype)
        + lam_ineq @ ev.ineq_fn_jac_val.astype(dtype)
    )
    lam_lb = jnp.where(pin_lb, rho_s, 0.0)
    lam_ub = jnp.where(pin_ub, -rho_s, 0.0)

    # The reduction only sees ``g`` through the free block: a non-finite
    # entry on a pinned variable would otherwise surface in ``ρ_s`` alone.
    finite = (
        jnp.all(jnp.isfinite(g))
        & jnp.all(jnp.isfinite(step_ok))
        & jnp.all(jnp.isfinite(lam_lb))
        & jnp.all(jnp.isfinite(lam_ub))
    )
    status = LDP_RESULTS.where(finite, lsei.status, LDP_RESULTS.nonfinite)
    ok = status == LDP_RESULTS.feasible

    step = jnp.where(ok, step_ok, d_fix)
    dual = Dual(
        eq_multipliers=jnp.where(ok, lam_eq, 0.0),
        ineq_multipliers=jnp.where(ok, lam_ineq, 0.0),
        lb_multipliers=jnp.where(ok, lam_lb, 0.0),
        ub_multipliers=jnp.where(ok, lam_ub, 0.0),
    )

    if m_c > 0:
        resid = a_mat @ step - b_vec
        eq_viol = jnp.max(jnp.abs(resid[:m_e]), initial=0.0)
        ineq_viol = jnp.max(jnp.maximum(resid[m_e:], 0.0), initial=0.0)
        feasibility_residual = jnp.maximum(eq_viol, ineq_viol)
    else:
        feasibility_residual = jnp.zeros((), dtype)

    # Farkas ray (the stationarity formula without ``B d + g``): zero unless
    # the least-distance solver formed a certificate.
    has_ray = (
        lsei.has_certificate
        & finite
        & (
            (status == LDP_RESULTS.infeasible)
            | (status == LDP_RESULTS.inconsistent_equalities)
        )
    )
    cert_eq = jnp.where(has_ray, lsei.cert_eq, 0.0)
    cert_ineq = jnp.where(has_ray, lsei.cert_ineq, 0.0)
    rho_ray = cert_eq @ ev.eq_fn_jac_val.astype(dtype) + cert_ineq @ (
        ev.ineq_fn_jac_val.astype(dtype)
    )
    ray_lb = jnp.where(has_ray & pin_lb, rho_ray, 0.0)
    ray_ub = jnp.where(has_ray & pin_ub, -rho_ray, 0.0)
    # Growth rate of the dual objective along the ray: -(b~'y_gen + y_hi); the
    # general rows of the elastic certificate satisfy G'y_gen = 0, so
    # h'y_gen = b~'y_gen, and y_hi = 0 for the plain variant.
    ray_rate = jnp.where(
        has_ray, -(b_tilde @ jnp.concatenate([cert_eq, cert_ineq]) + y_hi), 0.0
    )

    return FixedBoundsQPResult(  # ty: ignore[invalid-return-type]
        step=step,
        dual=dual,
        ldp_status=status,
        n_nnls_iter=lsei.n_iter,
        feasibility_residual=feasibility_residual,
        certificate_eq=cert_eq,
        certificate_ineq=cert_ineq,
        ray_lb=ray_lb,
        ray_ub=ray_ub,
        has_ray=has_ray,
        ray_rate=ray_rate,
        delta=jnp.where(ok, delta_raw, 0.0),
    )
