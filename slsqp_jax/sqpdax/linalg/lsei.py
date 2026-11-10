"""Least-distance problem with equality and inequality constraints (LSEI).

Solves

    min ½‖w‖²   s.t.   G_E w = h_E,   G_I w ≤ h_I

for a *small* constraint set (``m = m_E + m_I`` general constraints, assumed
much smaller than the dimension ``n`` of ``w``). This is the problem the
limited-memory reduced-Hessian QP solver reduces its quadratic program to
(``w`` is the step in the coordinates in which the model Hessian is the
identity). Everything is expressed through the ``n × m`` matrix ``Gᵀ``; the
only dense objects are ``n × m`` and ``m × m`` ones, so the memory is
``O(n m + m²)``.

The reduction follows the SLSQP chain, with direct linear algebra:

0. equilibrate: every column of ``Gᵀ`` (constraint row) and its right-hand
   side entry are divided by the column norm, so the rank cut and the
   consistency test below do not depend on how the rows are scaled (``w`` is
   unchanged, the multipliers are divided by the same norms afterwards);
1. thin Householder QR ``Gᵀ = Q R`` with columns ordered ``[E, I]``; the
   minimum-norm ``w`` lies in ``range(Gᵀ)``, ``w = Q z``, ``G w = Rᵀ z``;
2. split ``R = [[R_EE, R_EI], [0, R_II]]``. The equalities read
   ``R_EEᵀ z_E = h_E``. With the SVD ``R_EEᵀ = U Σ Vᵀ`` and a rank cut the
   general solution is ``z_E = z_E0 + V_⊥ t`` (dependent equality rows give
   free directions ``V_⊥``) and the equalities are consistent iff the
   component of ``h_E`` outside ``range(U_k)`` vanishes;
3. the inequalities become a least-distance problem (LDP) in
   ``ζ = [t; z_I]`` (fixed size ``m_E + m_I``; determined directions are
   zero columns, so the minimum-norm solution sets them to zero), solved by
   :func:`~slsqp_jax.sqpdax.linalg.lawson_hanson.ldp`. An inequality row in
   the span of the equality rows (or, without equalities, an exactly zero
   column of ``Gᵀ``) has a zero row in this system (``O(eps)`` noise in
   floating point, which ``ldp`` would normalise into a random constraint):
   such rows are zeroed, and their right-hand side is clamped to ``≥ 0``
   unless it is a genuine violation;
4. the multipliers follow by back-substitution.
"""

from __future__ import annotations

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from .lawson_hanson import LDP_RESULTS, ldp

__all__ = ["LSEIResult", "least_distance_lsei"]


class LSEIResult(eqx.Module):
    """Solution of a least-distance problem with equalities and inequalities.

    Attributes
    ----------
    w
        Minimum-norm point of ``{w : G_E w = h_E, G_I w ≤ h_I}``, shape
        ``(n,)``. Zero unless ``status`` is ``feasible``.
    lam_eq
        Multipliers of the equalities, shape ``(m_E,)`` (free sign). When the
        equality rows are linearly dependent they are not unique; the
        minimum-norm choice is returned. Zero unless ``status`` is
        ``feasible``.
    lam_ineq
        Multipliers of the inequalities, shape ``(m_I,)``, ``≥ 0``,
        complementary to ``G_I w - h_I``. Zero unless ``status`` is
        ``feasible``.
    status
        :class:`~slsqp_jax.sqpdax.linalg.lawson_hanson.LDP_RESULTS` code:
        ``feasible``, ``infeasible`` (no ``w`` satisfies the inequalities),
        ``inconsistent_equalities``, ``max_iter_reached`` or ``nonfinite``.
    n_iter
        Number of NNLS steps of the LDP (``int32`` scalar; zero without
        inequalities).
    cert_eq
        Equality part of the Farkas certificate, shape ``(m_E,)`` (free
        sign). See ``has_certificate``.
    cert_ineq
        Inequality part of the Farkas certificate, shape ``(m_I,)``, ``≥ 0``.
        See ``has_certificate``.
    has_certificate
        Boolean scalar. True iff ``status`` is ``infeasible`` or
        ``inconsistent_equalities``, a certificate was formed and it is
        finite. Then ``y = [cert_eq; cert_ineq]`` satisfies
        ``y_I ≥ 0``, ``Gᵀ y ≈ 0`` and ``hᵀ y < 0`` (in the original,
        un-equilibrated units), which proves ``G_E w = h_E, G_I w ≤ h_I``
        infeasible, and is scaled to ``‖y‖_∞ = 1``. Otherwise ``cert_eq`` and
        ``cert_ineq`` are zero.
    """

    w: Float[Array, " n"]
    lam_eq: Float[Array, " m_eq"]
    lam_ineq: Float[Array, " m_ineq"]
    status: LDP_RESULTS
    n_iter: Int[Array, ""]
    cert_eq: Float[Array, " m_eq"]
    cert_ineq: Float[Array, " m_ineq"]
    has_certificate: Bool[Array, ""]


def least_distance_lsei(
    g_t: Float[Array, "n m"],
    h: Float[Array, " m"],
    m_eq: int,
    eq_tol: float,
    max_iter: int | None = None,
    tol: float | None = None,
    ldp_tol: float | None = None,
    rank_tol: float | None = None,
) -> LSEIResult:
    """Solve ``min ½‖w‖²  s.t.  G_E w = h_E,  G_I w ≤ h_I``.

    The first ``m_eq`` columns of ``g_t`` (``= Gᵀ``) are the equality rows
    ``G_E``, the remaining ones the inequality rows ``G_I``; ``h`` is ordered
    the same way. The KKT conditions of the returned solution are

    * stationarity ``w + Gᵀ λ = 0``, i.e. ``w = -(G_Eᵀ λ_E + G_Iᵀ λ_I)``
      (Lagrangian ``½‖w‖² + λᵀ(G w - h)``);
    * ``G_E w = h_E``, ``G_I w ≤ h_I``;
    * ``λ_I ≥ 0`` and ``λ_I,i (G_I w - h_I)_i = 0``.

    The dense objects are ``n × m`` and ``m × m``; ``m_E = 0`` or ``m_I = 0``
    are handled by static branches.

    The constraints are first equilibrated: each column of ``g_t`` and the
    matching entry of ``h`` are divided by the column norm (a zero column
    keeps norm 1), and the multipliers are divided by the same norms on
    return. ``w`` is unchanged. This is separate from the normalisation that
    :func:`~slsqp_jax.sqpdax.linalg.lawson_hanson.ldp` performs on the
    projected inequality system (which stays); it makes the equality
    decisions (rank cut and consistency test) independent of the row scales.
    Dependent equality rows have non-unique multipliers; the minimum-norm
    ones *in the equilibrated units* are returned.

    The statuses ``infeasible``, ``max_iter_reached`` and ``nonfinite`` are
    those of :func:`~slsqp_jax.sqpdax.linalg.lawson_hanson.ldp`;
    ``inconsistent_equalities`` is reported when
    ``‖U_⊥ᵀ ĥ_E‖ > eq_tol · max(1, ‖ĥ_E‖)``, where ``ĥ`` is the equilibrated
    right-hand side and ``U_⊥`` spans the left null space of the equality
    rows. It takes precedence over the status of the LDP. Inequality rows
    whose gradient lies in the span of the equality gradients (an exactly zero
    gradient, e.g. a row supported only on pinned variables, included, also
    without equalities) are neutralised (zero row, right-hand side clamped to non-negative when within the
    ``eq_tol`` floor of feasible) so that rounding noise is not amplified
    into a spurious constraint; a genuine violation stays ``infeasible``.
    Unless the status is ``feasible``, ``w`` and both multiplier vectors are
    zero, and callers must branch on the status and never use them.

    **Farkas certificate.** When the status is ``infeasible`` (by the LDP's
    residual test) or ``inconsistent_equalities``, ``cert_eq`` and
    ``cert_ineq`` hold a vector ``y`` with ``y_I ≥ 0``, ``Gᵀ y ≈ 0`` and
    ``hᵀ y < 0``, scaled to ``‖y‖_∞ = 1`` (``has_certificate`` is then true).
    For inconsistent equalities ``ŷ_E = -U_⊥ U_⊥ᵀ ĥ_E`` and ``ŷ_I = 0``.
    For an infeasible LDP, the LDP certificate ``u`` gives ``ŷ_I = u`` and
    ``ŷ_E = -R_EE⁺ R_EI u`` (the multiplier formula with ``z = 0``). It is
    computed in the equilibrated units and divided by the column norms, as
    the multipliers are. The ``infeasible`` outcome of the LDP's primal
    check carries no certificate (``has_certificate`` false, both parts
    zero).

    Parameters
    ----------
    g_t
        Transposed constraint matrix ``Gᵀ``, shape ``(n, m)``, columns
        ordered ``[E, I]``. ``m < n`` is not required.
    h
        Right-hand side, shape ``(m,)``, ordered ``[E, I]``.
    m_eq
        Number of equality constraints ``m_E`` (static Python ``int``).
    eq_tol
        Tolerance of the consistency test of the equalities, in the units of
        ``w`` (the equilibrated constraints have unit rows): relative to
        ``‖ĥ_E‖`` above 1, absolute below (an absolute floor of 1). It also
        sets the slack below which an inequality that is determined by the
        equalities is accepted as active.
    max_iter, tol, ldp_tol
        Passed to :func:`~slsqp_jax.sqpdax.linalg.lawson_hanson.ldp`.
    rank_tol
        Singular values of the equilibrated ``R_EE`` below
        ``rank_tol · σ_max`` count as zero (dependent equality rows), and
        projected inequality rows of norm at most ``rank_tol`` count as noise.
        ``None`` selects ``10 · eps · max(n, m)``.

    Returns
    -------
    LSEIResult
        ``w``, ``lam_eq``, ``lam_ineq``, ``status``, ``n_iter`` and the
        Farkas certificate ``cert_eq``, ``cert_ineq``, ``has_certificate``.
        The float dtype follows the inputs (at least ``float32``).

    Examples
    --------
    Distance from the origin to the line ``w_1 + w_2 = 2`` intersected with
    ``w_1 ≤ 0.5``:

    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.linalg import LDP_RESULTS, least_distance_lsei
    >>> g_t = jnp.array([[1.0, 1.0], [1.0, 0.0]])  # columns: [equality, inequality]
    >>> res = least_distance_lsei(g_t, jnp.array([2.0, 0.5]), m_eq=1, eq_tol=1e-8)
    >>> [round(float(v), 6) for v in res.w]
    [0.5, 1.5]
    >>> bool(res.status == LDP_RESULTS.feasible)
    True
    """
    dtype = jnp.result_type(g_t, h, jnp.float32)
    g_t = jnp.asarray(g_t, dtype=dtype)
    h = jnp.asarray(h, dtype=dtype)
    n, m = g_t.shape
    if h.shape != (m,):
        raise ValueError(f"h must have shape ({m},), got {h.shape}")
    if not 0 <= m_eq <= m:
        raise ValueError(f"m_eq must lie in [0, {m}], got {m_eq}")
    m_e, m_i = m_eq, m - m_eq

    if m == 0:  # static shape: no constraint, the origin is optimal
        return LSEIResult(  # ty: ignore[invalid-return-type]
            w=jnp.zeros((n,), dtype),
            lam_eq=jnp.zeros((0,), dtype),
            lam_ineq=jnp.zeros((0,), dtype),
            status=LDP_RESULTS.feasible,
            n_iter=jnp.zeros((), jnp.int32),
            cert_eq=jnp.zeros((0,), dtype),
            cert_ineq=jnp.zeros((0,), dtype),
            has_certificate=jnp.zeros((), bool),
        )

    eps = float(jnp.finfo(dtype).eps)
    rank_tol = 10.0 * eps * max(n, m) if rank_tol is None else rank_tol

    # Equilibrate (O(n m)): unit columns of Gᵀ, h scaled along. w is
    # unchanged and λ_j = λ̂_j / ‖g_j‖. Without it the rank cut and the
    # consistency test are dominated by the largest row (a small dependent or
    # near-dependent equality is judged in the big row's units). The
    # non-finite check below uses the original data.
    col_norm = jnp.linalg.norm(g_t, axis=0)
    col_norm = jnp.where(col_norm > 0.0, col_norm, 1.0)
    g_hat = g_t / col_norm
    h_hat = h / col_norm

    # Thin QR needs at least m rows. For n < m pad with zero rows: the
    # minimum-norm w stays in range(Gᵀ), which has no component in the
    # padding, so truncating w to n entries is exact.
    n_eff = max(n, m)
    g_pad = (
        g_hat
        if n_eff == n
        else jnp.concatenate([g_hat, jnp.zeros((n_eff - n, m), dtype)])
    )
    q, r = jnp.linalg.qr(g_pad, mode="reduced")  # (n_eff, m), (m, m)
    h_e, h_i = h_hat[:m_e], h_hat[m_e:]
    r_ei = r[:m_e, m_e:]
    r_ii = r[m_e:, m_e:]

    # Equalities: R_EEᵀ z_E = h_E through the SVD, never an inverse (R_EE has
    # a zero diagonal on dependent rows).
    if m_e > 0:
        u, s, vt = jnp.linalg.svd(r[:m_e, :m_e].T)
        v = vt.T
        keep = s > rank_tol * s[0]  # s is sorted decreasingly; s[0] == 0: none
        inv_s = jnp.where(keep, 1.0 / jnp.where(keep, s, 1.0), 0.0)
        u_h = u.T @ h_e
        z_e0 = v @ (inv_s * u_h)
        inconsistent = jnp.linalg.norm(
            jnp.where(keep, 0.0, u_h)
        ) > eq_tol * jnp.maximum(1.0, jnp.linalg.norm(h_e))
        v_null = v * (~keep)[None, :].astype(dtype)  # zero columns: determined
        # Certificate of inconsistent equalities: ŷ_E = -U_⊥ U_⊥ᵀ ĥ_E has
        # R_EE ŷ_E = 0 and ĥ_Eᵀ ŷ_E = -‖U_⊥ᵀ ĥ_E‖² < 0.
        cert_inc_e = -(u @ jnp.where(keep, 0.0, u_h))
    else:
        z_e0 = jnp.zeros((0,), dtype)
        inconsistent = jnp.zeros((), bool)

    # Inequalities: R_EIᵀ z_E + R_IIᵀ z_I ≤ h_I in ζ = [t; z_I]. ldp wants
    # "≥", hence the sign flips.
    if m_i > 0:
        if m_e > 0:
            c = jnp.concatenate([r_ei.T @ v_null, r_ii.T], axis=1)
            rhs = h_i - r_ei.T @ z_e0
            z_e0_norm = jnp.linalg.norm(z_e0)
        else:
            c = r_ii.T
            rhs = h_i
            z_e0_norm = jnp.zeros((), dtype)
        # An inequality gradient in the span of the equality gradients (or,
        # with ``m_E = 0``, an exactly zero ``Gᵀ`` column: a row supported
        # only on pinned variables) gives a zero row of c (exact arithmetic)
        # and rhs = its slack at the point the equalities fix. In floating
        # point the row is noise that ldp would normalise into an O(1) random
        # constraint. Zero such rows; a slack >= -eq_tol (floor 1, units of
        # w) counts as satisfied (rhs -> max(rhs, 0)), a larger violation
        # stays negative so that ldp reports ``infeasible``.
        noise = jnp.linalg.norm(c, axis=1) <= rank_tol
        near_ok = rhs >= -eq_tol * jnp.maximum(1.0, jnp.abs(h_i) + z_e0_norm)
        c = jnp.where(noise[:, None], 0.0, c)
        rhs = jnp.where(noise & near_ok, jnp.maximum(rhs, 0.0), rhs)
        res = ldp(-c, -rhs, max_iter=max_iter, tol=tol, ldp_tol=ldp_tol)
        zeta = res.x
        lam_i = res.multipliers
        ldp_status = res.status
        n_iter = res.n_iter
        cert_i = res.certificate  # u ≥ 0, cᵀu = 0, rhsᵀu < 0 (ĥ-units)
        t, z_i = zeta[:m_e], zeta[m_e:]
    else:
        t = jnp.zeros((m_e,), dtype)
        z_i = jnp.zeros((0,), dtype)
        lam_i = jnp.zeros((0,), dtype)
        ldp_status = LDP_RESULTS.feasible
        n_iter = jnp.zeros((), jnp.int32)
        cert_i = jnp.zeros((0,), dtype)

    # Primal point and the equality multipliers: stationarity z = -R λ gives
    # R_EE λ_E = -(z_E + R_EI λ_I), solved in the least-squares sense with
    # the same SVD (R_EE = V Σ Uᵀ).
    if m_e > 0:
        z_e = z_e0 + v_null @ t
        lam_e = u @ (inv_s * (v.T @ -(z_e + r_ei @ lam_i)))
    else:
        z_e = z_e0
        lam_e = jnp.zeros((0,), dtype)
    w = (q @ jnp.concatenate([z_e, z_i]))[:n]
    lam_e = lam_e / col_norm[:m_e]
    lam_i = lam_i / col_norm[m_e:]

    # Farkas certificate. LDP infeasible: ŷ_I = u and ŷ_E = -R_EE⁺ R_EI u
    # (the multiplier formula with z = 0); then R_G ŷ = 0, i.e. Ĝᵀŷ = 0, and
    # ĥᵀŷ = rhsᵀu < 0. Inconsistent equalities take precedence (as the
    # status does), with ŷ_I = 0. Divide by the column norms as the
    # multipliers are (so that Gᵀy = 0, hᵀy = ĥᵀŷ), then scale to ‖y‖_∞ = 1.
    if m_e > 0:
        cert_e_ldp = u @ (inv_s * (v.T @ -(r_ei @ cert_i)))
        cert_e = jnp.where(inconsistent, cert_inc_e, cert_e_ldp)
    else:
        cert_e = jnp.zeros((0,), dtype)
    cert_i = jnp.where(inconsistent, 0.0, cert_i)
    cert_e = cert_e / col_norm[:m_e]
    cert_i = cert_i / col_norm[m_e:]

    finite = (
        jnp.all(jnp.isfinite(g_t))
        & jnp.all(jnp.isfinite(h))
        & jnp.all(jnp.isfinite(w))
        & jnp.all(jnp.isfinite(lam_e))
        & jnp.all(jnp.isfinite(lam_i))
    )
    status = LDP_RESULTS.where(
        finite,
        LDP_RESULTS.where(
            inconsistent, LDP_RESULTS.inconsistent_equalities, ldp_status
        ),
        LDP_RESULTS.nonfinite,
    )
    ok = status == LDP_RESULTS.feasible
    cert_scale = jnp.maximum(
        jnp.max(jnp.abs(cert_e), initial=0.0), jnp.max(jnp.abs(cert_i), initial=0.0)
    )
    has_cert = (
        finite
        & (inconsistent | (jnp.max(jnp.abs(cert_i), initial=0.0) > 0.0))
        & (
            (status == LDP_RESULTS.infeasible)
            | (status == LDP_RESULTS.inconsistent_equalities)
        )
        & jnp.all(jnp.isfinite(cert_e))
        & jnp.all(jnp.isfinite(cert_i))
        & (cert_scale > 0.0)
    )
    cert_div = jnp.where(has_cert, cert_scale, 1.0)
    return LSEIResult(  # ty: ignore[invalid-return-type]
        w=jnp.where(ok, w, 0.0),
        lam_eq=jnp.where(ok, lam_e, 0.0),
        lam_ineq=jnp.where(ok, lam_i, 0.0),
        status=status,
        n_iter=n_iter,
        cert_eq=jnp.where(has_cert, cert_e / cert_div, 0.0),
        cert_ineq=jnp.where(has_cert, cert_i / cert_div, 0.0),
        has_certificate=has_cert,
    )
