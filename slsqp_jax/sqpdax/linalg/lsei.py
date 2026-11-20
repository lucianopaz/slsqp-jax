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
   unless it is a genuine violation. The rows are first rank-revealed (thin SVD,
   singular values below ``eps^(2/3) σ_max`` dropped: the unpivoted QR leaves
   rounding noise as *directions* too) and the noise threshold of a row grows
   with the condition number of ``R_EE`` (a projection onto the equality span
   rounds like ``eps · cond(R_EE)``);
4. the multipliers follow by back-substitution.

:func:`elastic_least_distance_lsei` solves the same problem relaxed by one extra
scalar ``δ ∈ [0, 1]`` (Kraft's elastic mode, plan §2.5): it appends ``δ`` as
one more coordinate of ``w`` and two more general inequality rows, and calls
:func:`least_distance_lsei` unchanged, so that function is not affected. Its
weight ``ρ_e`` must be on the scale of ``‖w‖²`` (``w`` is accurate relative
to ``‖(w, √ρ_e δ)‖``): :func:`elastic_weight_scale` gives that scale.
"""

from __future__ import annotations

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from .lawson_hanson import LDP_RESULTS, ldp

__all__ = [
    "ElasticLSEIResult",
    "LSEIResult",
    "elastic_delta_column",
    "elastic_least_distance_lsei",
    "elastic_weight_scale",
    "least_distance_lsei",
]


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


def _primal_point(q, z_e0, v_null, zeta, m_e, n):
    """``w = Q [z_E0 + V_null t; z_I]`` (first ``n`` entries) from ``ζ = [t; z_I]``."""
    z_e = z_e0 + v_null @ zeta[:m_e]
    return (q @ jnp.concatenate([z_e, zeta[m_e:]]))[:n]


def _row_violation(g_hat, h_hat, w, m_e):
    """Largest violation of the equilibrated rows ``[E, I]`` at ``w`` (∞-norm)."""
    resid = g_hat.T @ w - h_hat
    return jnp.maximum(
        jnp.max(jnp.abs(resid[:m_e]), initial=0.0),
        jnp.max(jnp.maximum(resid[m_e:], 0.0), initial=0.0),
    )


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

    **Verification.** The projection noise of an inequality row lying in the
    span of the equalities is ``eps · cond(R_EE)``, above the ``√eps`` cap of
    the noise threshold once ``cond(R_EE) ≳ 1e8``; the LDP can then exploit
    it with a ``‖w‖ ~ 1/eps`` step that violates the rows. Whenever the cap
    binds (``rank_tol · cond(R_EE) > max(rank_tol, √eps)``), ``feasible`` is
    only reported for a point that meets the equilibrated rows to
    ``max(10 √eps, 2 eq_tol) · max(1, ‖ĥ‖ + ‖z_E0‖)`` (``z_E0`` the point the
    equalities fix in the reduced coordinates): an absolute test in the data
    scale, never relative to ``‖w‖``. When the check fails, the inequality
    part is solved once more with the uncapped threshold
    ``rank_tol · cond(R_EE)`` (only on that path); an answer that still fails
    it is ``infeasible``, without a certificate unless the LDP produced one.
    Where the cap does not bind, the answer is accurate relative to ``‖w‖``,
    like any LDP solution (a far-away vertex of near-parallel rows included).

    **Farkas certificate.** When the status is ``infeasible`` (by the LDP's
    residual test, or by its primal check with a verified certificate) or
    ``inconsistent_equalities``, ``cert_eq`` and
    ``cert_ineq`` hold a vector ``y`` with ``y_I ≥ 0``, ``Gᵀ y ≈ 0`` and
    ``hᵀ y < 0``, scaled to ``‖y‖_∞ = 1`` (``has_certificate`` is then true).
    For inconsistent equalities ``ŷ_E = -U_⊥ U_⊥ᵀ ĥ_E`` and ``ŷ_I = 0``.
    For an infeasible LDP, the LDP certificate ``u`` gives ``ŷ_I = u`` and
    ``ŷ_E = -R_EE⁺ R_EI u`` (the multiplier formula with ``z = 0``). It is
    computed in the equilibrated units and divided by the column norms, as
    the multipliers are. The ``infeasible`` outcome of the LDP's primal
    check carries a certificate only when the LDP's NNLS solution verifies
    as one (see :func:`~slsqp_jax.sqpdax.linalg.lawson_hanson.ldp`);
    otherwise ``has_certificate`` is false and both parts are zero.

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
        projected inequality rows of norm at most
        ``min(rank_tol · max(1, σ_max / σ_min,kept), max(rank_tol, √eps))``
        count as noise (``σ_min,kept`` the smallest singular value of ``R_EE``
        kept by the rank cut; plain ``rank_tol`` without equalities), a
        threshold raised further to ``eps^(2/3) σ_max(c)`` of the projected
        system ``c`` (see the module docstring). ``None`` selects
        ``min(10 · eps · max(n, m), max(10 · eps · m, √eps))``: the cut grows
        linearly in ``n`` and is capped at ``√eps``, the cap of the noise
        threshold. The columns of ``Gᵀ = C_F⁻ᵀ A_Fᵀ`` that are dependent in ``A``
        carry the rounding of the constraint data amplified by the metric
        (about ``eps · κ_j`` per column, ``κ_j`` growing with the stiffness of
        the model along the row), not just the few ``eps`` of the QR, so the
        cut must keep the linear ``n`` growth that covers that amplification.
        The cap stops it from reaching ``O(1)`` in float32 at large ``n``
        (``0.12`` at ``n = 1e5`` without it), where it would reject
        well-conditioned equalities as dependent. It equals the plain
        ``10 · eps · max(n, m)`` in float64 for ``n`` below about ``6.7e6`` and
        in float32 below about ``2900``; equality sets whose conditioning
        exceeds about ``1/√eps`` count as dependent and go to the ``eq_tol``
        consistency test.

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
    # Default rank cut min(10·eps·max(n, m), max(10·eps·m, √eps)). The noise
    # it must cover on the dependent part of R_EE is the rounding of the
    # constraint data amplified by the metric C_F⁻ᵀ (≈ eps·κ_j per column of
    # Gᵀ = C_F⁻ᵀ A_Fᵀ), not the QR's own few eps; the linear n growth covers
    # metric amplification up to κ ≈ n. The √eps cap (the cap of ``noise_tol``)
    # keeps float32 at large n from rejecting well-conditioned rows (the
    # uncapped cut is 0.12 at n = 1e5), and ``10·eps·m`` keeps the cut above
    # the QR noise of very wide systems.
    if rank_tol is None:
        rank_tol = min(10.0 * eps * max(n, m), max(10.0 * eps * m, eps**0.5))

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
        # Row-noise tolerance. Projecting a row onto range(G_Eᵀ) rounds like
        # eps·cond(R_EE) (the unpivoted QR and the solve with R_EE), so a row
        # lying in that span survives a fixed ``rank_tol`` as noise once R_EE
        # is ill-conditioned. Scale the tolerance with the condition of the
        # kept part of R_EE, capped at max(rank_tol, √eps) so that genuine
        # (if small) rows are never discarded wholesale.
        s_min_kept = jnp.min(jnp.where(keep, s, jnp.inf), initial=jnp.inf)
        has_kept = jnp.isfinite(s_min_kept)
        cond_e = jnp.where(has_kept, s[0] / jnp.where(has_kept, s_min_kept, 1.0), 1.0)
        noise_tol = jnp.minimum(
            rank_tol * jnp.maximum(1.0, cond_e), max(rank_tol, eps**0.5)
        )
    else:
        z_e0 = jnp.zeros((0,), dtype)
        inconsistent = jnp.zeros((), bool)
        noise_tol = rank_tol

    # Verification of a ``feasible`` answer when the √eps cap of ``noise_tol``
    # binds (ill-conditioned R_EE): the projection noise of a row in the span
    # of the equalities is eps·cond(R_EE), above the cap once cond(R_EE) ≳
    # 1e8, and ldp exploits such noise rows with a ‖w‖ ~ 1/eps step whose
    # violations sit at the rounding level of that ‖w‖. The check is therefore
    # absolute in the data scale, never relative to ‖w‖, on the equilibrated
    # (unit) rows, covering the reduction's own acceptance rules (the
    # ``eq_tol`` consistency test and the ``near_ok`` slack of noise rows) with
    # a √eps floor. It is not applied where the cap does not bind: a genuine
    # far-away vertex (‖w‖ ~ 1e8 for near-parallel rows) is only accurate
    # relative to ‖w‖, like any ldp solution.
    check_tol = jnp.maximum(10.0 * eps**0.5, 2.0 * eq_tol) * jnp.maximum(
        1.0, jnp.linalg.norm(h_hat) + jnp.linalg.norm(z_e0)
    )
    cap_binds = jnp.zeros((), bool)

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
        #
        # Rank-reveal the rows first. The unpivoted QR leaves rounding noise
        # (eps/pivot) as *directions* inside genuine rows too, and ldp would
        # exploit them (a huge ``feasible`` step that violates the rows by
        # O(1)). Drop the singular directions of c below eps^(2/3)·σ_max,
        # rebuild c, and raise the row tolerance to that floor so rows lying
        # wholly in the dropped directions are zeroed by the rule below. ζ's
        # coordinates are unchanged; the cost is O(m_I² m_c).
        u_c, s_c, vt_c = jnp.linalg.svd(c, full_matrices=False)
        s_floor = eps ** (2.0 / 3.0) * jnp.max(s_c, initial=0.0)
        c = (u_c * jnp.where(s_c > s_floor, s_c, 0.0)) @ vt_c
        c_norm = jnp.linalg.norm(c, axis=1)
        near_ok = rhs >= -eq_tol * jnp.maximum(1.0, jnp.abs(h_i) + z_e0_norm)

        def solve_ineq(row_tol):
            noise = c_norm <= jnp.maximum(row_tol, s_floor)
            c_kept = jnp.where(noise[:, None], 0.0, c)
            rhs_kept = jnp.where(noise & near_ok, jnp.maximum(rhs, 0.0), rhs)
            return ldp(-c_kept, -rhs_kept, max_iter=max_iter, tol=tol, ldp_tol=ldp_tol)

        res = solve_ineq(noise_tol)
        if m_e > 0:
            # Where the cap binds, verify the answer on the original rows; a
            # ``feasible`` point that violates them let noise through.
            # Re-solve once with the uncapped tolerance rank_tol·cond(R_EE)
            # (only on that path: lax.cond).
            noise_hi = jnp.minimum(rank_tol * jnp.maximum(1.0, cond_e), 1.0)
            cap_binds = noise_hi > noise_tol
            w_first = _primal_point(q, z_e0, v_null, res.x, m_e, n)
            retry = (
                cap_binds
                & (res.status == LDP_RESULTS.feasible)
                & (_row_violation(g_hat, h_hat, w_first, m_e) > check_tol)
            )
            first_iter = res.n_iter
            res = jax.lax.cond(retry, lambda: solve_ineq(noise_hi), lambda: res)
            res = eqx.tree_at(
                lambda r: r.n_iter,
                res,
                jnp.where(retry, first_iter + res.n_iter, res.n_iter),
            )
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
    # Where the cap binds, ``feasible`` is only reported for a verified point:
    # a ``w`` that still violates the rows after the retry is ``infeasible``,
    # with a certificate only if the LDP produced one (here it did not: its
    # status was ``feasible``), as at the primal check of ``ldp``.
    unverified = (
        cap_binds
        & (ldp_status == LDP_RESULTS.feasible)
        & (_row_violation(g_hat, h_hat, w, m_e) > check_tol)
    )
    ldp_status = LDP_RESULTS.where(unverified, LDP_RESULTS.infeasible, ldp_status)

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


class ElasticLSEIResult(eqx.Module):
    """Solution of the elastic (relaxed) least-distance problem.

    Attributes
    ----------
    w
        The ``w`` part of the minimiser, shape ``(n,)``. Zero unless
        ``status`` is ``feasible``.
    delta
        The relaxation ``δ ∈ [0, 1]`` (scalar; zero unless ``status`` is
        ``feasible``). ``δ = 0`` means the plain problem is solved.
    lam_eq
        Multipliers of the ``m_E`` general equalities, shape ``(m_E,)``. The
        extra variable and its two rows never enter this vector.
    lam_ineq
        Multipliers of the ``m_I`` general inequalities, shape ``(m_I,)``,
        ``≥ 0``. Does not contain the multipliers of ``δ``'s own rows.
    lam_delta
        Multipliers of ``δ ≥ 0`` and ``δ ≤ 1``, shape ``(2,)``, ``≥ 0``
        (``[lower, upper]``).
    status
        :class:`~slsqp_jax.sqpdax.linalg.lawson_hanson.LDP_RESULTS` code of
        the augmented problem (see
        :class:`~slsqp_jax.sqpdax.linalg.lsei.LSEIResult`).
    n_iter
        Number of NNLS steps (``int32`` scalar).
    cert_eq, cert_ineq, cert_delta
        Farkas certificate of the augmented problem split into the general
        equalities, the general inequalities and the two rows of ``δ``
        (shape ``(2,)``, ``≥ 0``), scaled to unit sup-norm. See
        ``has_certificate``.
    has_certificate
        As in :class:`~slsqp_jax.sqpdax.linalg.lsei.LSEIResult`: the stacked
        ``y = [cert_eq; cert_ineq; cert_delta]`` has ``y_I ≥ 0``,
        ``G_augᵀ y ≈ 0`` and ``h_augᵀ y < 0``. With ``w = 0, δ = 1`` the
        relaxed rows hold, so the augmented problem can be infeasible only
        when that point is excluded, e.g. by pinned variables in the caller's
        reduction (or when no row is relaxed).
    """

    w: Float[Array, " n"]
    delta: Float[Array, ""]
    lam_eq: Float[Array, " m_eq"]
    lam_ineq: Float[Array, " m_ineq"]
    lam_delta: Float[Array, " 2"]
    status: LDP_RESULTS
    n_iter: Int[Array, ""]
    cert_eq: Float[Array, " m_eq"]
    cert_ineq: Float[Array, " m_ineq"]
    cert_delta: Float[Array, " 2"]
    has_certificate: Bool[Array, ""]


def elastic_delta_column(b: Float[Array, " m"], m_eq: int) -> Float[Array, " m"]:
    """Column of ``δ`` for the relaxation of ``A d = b_E``, ``A_I d ≤ b_I``.

    Kraft's relaxation (plan §2.5) is ``A_E d + δ b_E = b_E`` and
    ``A_I,j d + δ b_I,j ≤ b_I,j`` for the inequality rows violated at
    ``d = 0`` (``b_I,j < 0``); the other rows are not relaxed. This returns
    the coefficients of ``δ``: ``b_E`` for the equalities, ``b_I,j`` where it
    is negative and ``0`` elsewhere. ``b`` is the **unshifted** right-hand
    side, so the column does not depend on the bound working set.

    Parameters
    ----------
    b
        Unshifted right-hand side, shape ``(m,)``, ordered ``[E, I]``.
    m_eq
        Number of equalities (static ``int``).

    Returns
    -------
    Array
        Shape ``(m,)``, to be passed as ``delta_col`` of
        :func:`elastic_least_distance_lsei`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.linalg import elastic_delta_column
    >>> elastic_delta_column(jnp.array([2.0, -1.0, 3.0]), m_eq=1).tolist()
    [2.0, -1.0, 0.0]
    """
    b = jnp.asarray(b, dtype=jnp.result_type(b, jnp.float32))
    return jnp.concatenate([b[:m_eq], jnp.where(b[m_eq:] < 0.0, b[m_eq:], 0.0)])


def elastic_weight_scale(
    g_t: Float[Array, "n m"],
    h: Float[Array, " m"],
    delta_col: Float[Array, " m"],
    m_eq: int,
) -> Float[Array, ""]:
    """Scale ``τ²`` of the elastic weight, ``ρ_e = elastic_weight · τ²``.

    :func:`elastic_least_distance_lsei` solves one least-distance problem in
    ``(w, √ρ_e δ)``, so ``w`` is accurate only relative to
    ``‖(w, √ρ_e δ)‖``: ``ρ_e`` must have the units of ``‖w‖²``. ``τ`` is the
    largest distance, in the coordinates of ``w``, that the relaxation has to
    cover. Over the rows with ``‖G_j‖ > 0`` (``G_j`` is column ``j`` of
    ``g_t``) it is the maximum of

    * ``|h_j| / ‖G_j‖`` for the equalities,
    * ``max(-h_j, 0) / ‖G_j‖`` for the inequalities (slack rows are left out
      on purpose: one far slack row would inflate ``ρ_e`` and spoil the
      accuracy),
    * ``|c_j| / ‖G_j‖`` for all rows (``c = delta_col``),

    and ``τ = 1`` when that maximum is zero (the problem is then homogeneous
    and any ``ρ_e > 0`` gives the same solution). Scaling ``w`` by ``κ``
    scales ``τ²`` by ``κ²``; rescaling constraint rows leaves it unchanged.

    The cost is ``O(n m)``; the function is jit-able with ``m_eq`` static.
    Non-finite inputs give a ``nan`` result. For the fixed-bounds QP compute
    it once from the unpinned data and use the same ``ρ_e`` for every
    working set.

    Parameters
    ----------
    g_t, h
        As in :func:`least_distance_lsei`.
    delta_col
        Column of ``δ``, as in :func:`elastic_least_distance_lsei`.
    m_eq
        Number of equalities (static ``int``).

    Returns
    -------
    Array
        The scalar ``τ²``, in the dtype of the inputs (at least ``float32``).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.linalg import elastic_weight_scale
    >>> g_t = jnp.array([[2.0, 1.0]])  # one equality, one violated inequality
    >>> float(elastic_weight_scale(g_t, jnp.array([4.0, -3.0]), jnp.zeros(2), 1))
    9.0
    """
    dtype = jnp.result_type(g_t, h, delta_col, jnp.float32)
    g_t = jnp.asarray(g_t, dtype=dtype)
    h = jnp.asarray(h, dtype=dtype)
    delta_col = jnp.asarray(delta_col, dtype=dtype)
    m = g_t.shape[1]
    if h.shape != (m,) or delta_col.shape != (m,):
        raise ValueError(
            f"h and delta_col must have shape ({m},), got {h.shape} and "
            f"{delta_col.shape}"
        )
    if not 0 <= m_eq <= m:
        raise ValueError(f"m_eq must lie in [0, {m}], got {m_eq}")
    norm = jnp.linalg.norm(g_t, axis=0)
    live = norm > 0.0
    h_term = jnp.concatenate([jnp.abs(h[:m_eq]), jnp.maximum(-h[m_eq:], 0.0)])
    dist = jnp.maximum(h_term, jnp.abs(delta_col)) / jnp.where(live, norm, 1.0)
    tau = jnp.max(jnp.where(live, dist, 0.0), initial=0.0)
    tau = jnp.where(tau > 0.0, tau, 1.0)
    finite = jnp.all(jnp.isfinite(g_t)) & jnp.all(jnp.isfinite(h))
    finite &= jnp.all(jnp.isfinite(delta_col))
    return jnp.where(finite, tau * tau, jnp.nan)


def elastic_least_distance_lsei(
    g_t: Float[Array, "n m"],
    h: Float[Array, " m"],
    m_eq: int,
    delta_col: Float[Array, " m"],
    elastic_weight: float | Float[Array, ""],
    eq_tol: float,
    max_iter: int | None = None,
    tol: float | None = None,
    ldp_tol: float | None = None,
    rank_tol: float | None = None,
) -> ElasticLSEIResult:
    """Solve the relaxed problem ``min ½‖w‖² + ½ρ_e δ²``.

    The constraints are the general rows of
    :func:`least_distance_lsei`, relaxed by one extra scalar,

    ``G_E w + δ c_E = h_E``,  ``G_I w + δ c_I ≤ h_I``,  ``0 ≤ δ ≤ 1``,

    where ``c = delta_col`` is the column of ``δ`` (zero for rows that are
    not relaxed; see :func:`elastic_delta_column`) and ``ρ_e =
    elastic_weight``. It is the problem of plan §2.5 in the coordinates of
    ``w``. For the fixed-bounds QP with ``w = C_F d_F + C_F⁻ᵀ g̃``, ``G =
    A_F C_F⁻¹`` and ``h = b̃ + A_F B_FF⁻¹ g̃``, the relaxation
    ``A_F d_F + δ b = b̃`` has ``c = b``, the *unshifted* right-hand side
    (only ``h`` carries ``d_fix``). The column is **not** transformed by
    ``C_F``: the coordinate of ``δ`` is not part of the model, its own factor
    is ``√ρ_e`` and it has no linear term.

    **Reduction.** ``δ`` becomes the extra coordinate ``w_δ = √ρ_e δ`` and
    the ``n × m`` matrix is extended to ``(n + 1) × (m + 2)``: row ``n`` of
    the old columns holds ``c_j / √ρ_e``, and two new inequality columns
    ``∓e_n / √ρ_e`` (right-hand sides ``0`` and ``1``) encode ``δ ≥ 0`` and
    ``δ ≤ 1``. These two are *general* rows (columns of ``Gᵀ``), not bounds.
    The result is passed to :func:`least_distance_lsei` unchanged, which is
    therefore bit-identical to before when this function is not used. The
    stationarity condition of the elastic problem is
    ``w + Gᵀλ = 0`` and ``ρ_e δ + cᵀλ - μ_lo + μ_hi = 0``. Multipliers are
    those of the original problem (the extra rows' multipliers are returned
    separately as ``lam_delta``).

    **Weight.** The reduction solves one least-distance problem in
    ``(w, √ρ_e δ)``, so ``w`` is accurate only relative to
    ``‖(w, √ρ_e δ)‖``: ``ρ_e`` must be on the scale of ``‖w‖²``. Use
    ``ρ_e = elastic_weight · elastic_weight_scale(g_t, h, delta_col, m_eq)``
    with a dimensionless ``elastic_weight`` (about ``1e3``). A fixed ``ρ_e``
    returns wrong ``feasible`` answers when the data are very small or large.

    **When it is used.** The elastic problem is feasible whenever ``w = 0,
    δ = 1`` meets the relaxed rows. Callers enter it only after the plain
    problem is infeasible. On a *feasible* plain problem ``δ`` is not
    necessarily zero: ``δ > 0`` as soon as a relaxed row is active with a
    non-zero multiplier (first-order gain ``-cᵀλ``), but it is ``O(1/ρ_e)``;
    with no active relaxed row the solution is exactly the plain one and
    ``δ = 0``.

    Parameters
    ----------
    g_t, h, m_eq, eq_tol, max_iter, tol, ldp_tol, rank_tol
        As in :func:`least_distance_lsei`.
    delta_col
        Coefficients of ``δ`` in the rows, shape ``(m,)``, ordered ``[E, I]``.
    elastic_weight
        ``ρ_e > 0``, the weight of ``½δ²`` relative to ``½‖w‖²``. May be a
        traced scalar; it is not validated.

    Returns
    -------
    ElasticLSEIResult
        ``w`` (``n`` entries), ``delta ∈ [0, 1]``, the multipliers of the
        general rows and, separately, those of ``δ``'s rows, the status and
        the certificate. Unless the status is ``feasible`` the outputs are
        zero, as for :func:`least_distance_lsei`.

    Examples
    --------
    ``w ≤ -1`` and ``-w ≤ 0.9`` are contradictory with a margin of
    ``0.1``: relaxing the violated row (``c = b``) needs ``δ ≥ 0.1``.

    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.linalg import (
    ...     LDP_RESULTS,
    ...     elastic_delta_column,
    ...     elastic_least_distance_lsei,
    ...     elastic_weight_scale,
    ... )
    >>> g_t = jnp.array([[1.0, -1.0]])
    >>> h = jnp.array([-1.0, 0.9])
    >>> c = elastic_delta_column(h, 0)
    >>> rho = 1e3 * elastic_weight_scale(g_t, h, c, 0)
    >>> res = elastic_least_distance_lsei(g_t, h, 0, c, rho, eq_tol=1e-8)
    >>> bool(res.status == LDP_RESULTS.feasible), round(float(res.delta), 4)
    (True, 0.1)
    """
    dtype = jnp.result_type(g_t, h, delta_col, jnp.float32)
    g_t = jnp.asarray(g_t, dtype=dtype)
    h = jnp.asarray(h, dtype=dtype)
    delta_col = jnp.asarray(delta_col, dtype=dtype)
    n, m = g_t.shape
    if delta_col.shape != (m,):
        raise ValueError(f"delta_col must have shape ({m},), got {delta_col.shape}")
    if h.shape != (m,):
        raise ValueError(f"h must have shape ({m},), got {h.shape}")
    if not 0 <= m_eq <= m:
        raise ValueError(f"m_eq must lie in [0, {m}], got {m_eq}")
    root = jnp.sqrt(jnp.asarray(elastic_weight, dtype=dtype))

    # (n + 1) x (m + 2): the last row is the coordinate w_δ = √ρ_e δ; the new
    # columns (general inequality rows) are  -w_δ/√ρ_e ≤ 0  and  w_δ/√ρ_e ≤ 1.
    delta_row = jnp.concatenate([delta_col, jnp.array([-1.0, 1.0], dtype)]) / root
    g_aug = jnp.concatenate([g_t, jnp.zeros((n, 2), dtype)], axis=1)
    g_aug = jnp.concatenate([g_aug, delta_row[None, :]], axis=0)
    h_aug = jnp.concatenate([h, jnp.array([0.0, 1.0], dtype)])
    res = least_distance_lsei(
        g_aug,
        h_aug,
        m_eq,
        eq_tol,
        max_iter=max_iter,
        tol=tol,
        ldp_tol=ldp_tol,
        rank_tol=rank_tol,
    )
    m_i = m - m_eq
    ok = res.status == LDP_RESULTS.feasible
    delta = jnp.where(ok, jnp.clip(res.w[n] / root, 0.0, 1.0), 0.0)
    return ElasticLSEIResult(  # ty: ignore[invalid-return-type]
        w=res.w[:n],
        delta=delta,
        lam_eq=res.lam_eq,
        lam_ineq=res.lam_ineq[:m_i],
        lam_delta=res.lam_ineq[m_i:],
        status=res.status,
        n_iter=res.n_iter,
        cert_eq=res.cert_eq,
        cert_ineq=res.cert_ineq[:m_i],
        cert_delta=res.cert_ineq[m_i:],
        has_certificate=res.has_certificate,
    )
