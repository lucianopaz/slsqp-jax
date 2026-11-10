"""Non-negative least squares by the Lawson–Hanson active-set method.

Solves ``min ‖E x - f‖₂  s.t.  x ≥ 0`` for a small dense ``E`` (``m × k``,
at most a few hundred columns: the number of general constraints of the
LRH direct QP solver). This is the work-horse of the least-distance
problem (LDP) of the SLSQP chain (Lawson & Hanson, 1974, ch. 23).

The implementation keeps every shape fixed, so it runs under ``jit``. The
passive (free) set is a boolean mask of length ``k``; the least-squares
problem on the passive columns is solved with ``jnp.linalg.lstsq`` on the
*column-masked* matrix ``E ⊙ passive`` (an SVD based minimum-norm solve),
which returns exactly zero on the masked columns and stays well defined
for rank-deficient passive sets. The whole algorithm is one flat
``lax.while_loop`` in which each iteration is either an *add* step (test
optimality and move the index of the largest positive dual into the passive
set) or a *solve* step (passive-set least squares followed by the
Lawson–Hanson interpolation back into the feasible region).

:func:`ldp` builds on it: the least-distance problem
``min ‖x‖  s.t.  g x ≥ h`` is turned into one NNLS problem (Lawson & Hanson,
ch. 23), whose residual decides feasibility and yields the solution and the
multipliers.
"""

from __future__ import annotations

import equinox as eqx
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

__all__ = ["NNLSResult", "nnls", "LDP_RESULTS", "LDPResult", "ldp"]


class NNLSResult(eqx.Module):
    """Solution of a non-negative least-squares problem.

    Attributes
    ----------
    x
        Minimiser, shape ``(k,)``, with ``x ≥ 0``.
    residual
        Residual *vector* ``E x - f``, shape ``(m,)`` (the sign convention of
        Lawson & Hanson, ch. 23). Its norm is the optimal objective value.
    passive
        Boolean mask, shape ``(k,)``, of the passive (free) set at
        termination. Components of ``x`` off the passive set are exactly
        zero; when ``converged`` the passive components are positive up to
        roundoff.
    n_iter
        Number of steps performed (``int32`` scalar): one per addition to
        the passive set, per skipped addition and per interpolation step
        (an interpolation step may release several indices).
    converged
        ``True`` when the Karush–Kuhn–Tucker conditions hold to the
        tolerance, ``False`` when the iteration budget ``max_iter`` was
        exhausted first (``x`` is then feasible but not optimal).
    """

    x: Float[Array, " k"]
    residual: Float[Array, " m"]
    passive: Bool[Array, " k"]
    n_iter: Int[Array, ""]
    converged: Bool[Array, ""]


def nnls(
    e: Float[Array, "m k"],
    f: Float[Array, " m"],
    max_iter: int | None = None,
    tol: float | None = None,
    dep_tol: float = 0.0,
) -> NNLSResult:
    """Solve ``min ‖E x - f‖₂`` subject to ``x ≥ 0``.

    Classic Lawson–Hanson. With ``w = Eᵀ(f - E x)`` the dual (negative
    gradient), the outer step moves the index of the largest positive
    ``w_j`` (outside the passive set ``P``) into ``P``; the inner step
    solves the least-squares problem on the columns in ``P`` and, whenever
    a component of that solution is non-positive, interpolates back towards
    the previous iterate and releases the components that reach zero. If the
    newly added index cannot enter (its least-squares coefficient is not
    positive) it is skipped until the iterate next moves, as in the original
    algorithm. Passive-set membership is tracked in a mask, never inferred
    from ``x == 0``.

    With ``dep_tol > 0`` a candidate column is also tested for linear
    dependence on the passive columns before it enters (Kraft's SLSQP does
    the same): if its relative distance ``‖e_j - E_P E_P⁺ e_j‖ / ‖e_j‖`` from
    the span of the passive columns is at most ``dep_tol`` it is frozen
    instead of added, and counts as one step. Frozen indices are released
    like the ones frozen by the skipping rule above, i.e. when the iterate
    next moves. Admitting a column that is dependent up to rounding makes
    the passive least-squares solve explode on degenerate polytopes (see
    :func:`ldp`). The test costs one least-squares solve per addition.

    Parameters
    ----------
    e
        Matrix ``E``, shape ``(m, k)``. ``k == 0`` is handled (static shape).
    f
        Right-hand side, shape ``(m,)``.
    max_iter
        Budget of steps, counted as in :attr:`NNLSResult.n_iter`
        (additions, skipped additions and interpolation steps). ``None``
        selects ``3 k``.
    tol
        Optimality tolerance on the duals: the iteration stops when
        ``w_j ≤ tol`` for every ``j`` outside the passive set. The duals
        are bilinear in ``E`` and ``f``, so ``tol`` is *absolute* with
        respect to ``‖f‖`` (the default does not scale with ``f``; rescale
        ``f`` or pass ``tol`` for data of very small or very large norm).
        A negative value is allowed: the method then also enters indices
        with a negative dual, whose least-squares coefficient is not
        positive, which exercises the skipping rule (used by the tests).
        ``None`` selects ``10 · eps · ‖E‖_F · max(m, k)``.
    dep_tol
        Relative linear-dependence threshold described above. The default
        ``0.0`` disables the test (plain Lawson–Hanson). Never applied when
        the passive set is empty.

    Returns
    -------
    NNLSResult
        ``x``, the residual vector ``E x - f``, the passive mask, the
        iteration count and the convergence flag. The float dtype follows
        the inputs (at least ``float32``).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.linalg import nnls
    >>> e = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    >>> f = jnp.array([1.0, -1.0, 0.0])
    >>> res = nnls(e, f)
    >>> [round(float(v), 6) for v in res.x]  # the unconstrained x_2 < 0 is clipped
    [0.5, 0.0]
    >>> bool(res.converged)
    True
    """
    dtype = jnp.result_type(e, f, jnp.float32)
    e = jnp.asarray(e, dtype=dtype)
    f = jnp.asarray(f, dtype=dtype)
    m, k = e.shape

    if k == 0:  # static shape: nothing to optimise
        return NNLSResult(  # ty: ignore[invalid-return-type]
            x=jnp.zeros((0,), dtype),
            residual=-f,
            passive=jnp.zeros((0,), bool),
            n_iter=jnp.zeros((), jnp.int32),
            converged=jnp.ones((), bool),
        )

    eps = float(jnp.finfo(dtype).eps)
    max_iter = 3 * k if max_iter is None else max_iter
    max_iter_arr = jnp.asarray(max_iter, jnp.int32)
    tol_arr = (
        10.0 * eps * jnp.linalg.norm(e) * max(m, k)
        if tol is None
        else jnp.asarray(tol, dtype)
    )

    def passive_solution(passive: Bool[Array, " k"]) -> Float[Array, " k"]:
        s = jnp.linalg.lstsq(jnp.where(passive[None, :], e, 0.0), f)[0]
        return jnp.where(passive, s, 0.0)

    def cond(state):
        _, _, _, new_idx, n_iter, feasible, done, _ = state
        # Past the budget only the solve that follows the last addition and
        # the optimality test of a feasible state may still run, so that an
        # optimal final iterate is reported as converged.
        return ~done & ((n_iter < max_iter_arr) | feasible | (new_idx >= 0))

    def add_step(state):
        x, passive, frozen, _, n_iter, _, _, _ = state
        w = e.T @ (f - e @ x)
        candidate = ~passive & ~frozen
        w_masked = jnp.where(candidate, w, -jnp.inf)
        j = jnp.argmax(w_masked).astype(jnp.int32)
        optimal = w_masked[j] <= tol_arr  # also true when no candidate is left
        stop = optimal | (n_iter >= max_iter_arr)
        if dep_tol > 0.0:
            # Relative distance of the candidate from span(E_P); a column
            # that is dependent up to ``dep_tol`` is frozen, not added.
            e_p = jnp.where(passive[None, :], e, 0.0)
            col = e[:, j]
            coef = jnp.linalg.lstsq(e_p, col)[0]
            dist = jnp.linalg.norm(col - e_p @ coef)
            dependent = (
                (dist <= dep_tol * jnp.linalg.norm(col)) & ~stop & jnp.any(passive)
            )
        else:
            dependent = jnp.zeros((), bool)
        add = ~stop & ~dependent
        passive = jnp.where(add, passive.at[j].set(True), passive)
        frozen = jnp.where(dependent, frozen.at[j].set(True), frozen)
        new_idx = jnp.where(add, j, jnp.int32(-1))
        n_iter = n_iter + jnp.where(stop, 0, 1).astype(jnp.int32)
        # A frozen column leaves the iterate unchanged: go straight back to
        # the next addition (the state is "feasible").
        return x, passive, frozen, new_idx, n_iter, stop | dependent, stop, optimal

    def solve_step(state):
        x, passive, frozen, new_idx, n_iter, _, _, _ = state
        s = passive_solution(passive)

        # The index that just entered cannot be added: skip it for now.
        has_new = new_idx >= 0
        safe_new = jnp.maximum(new_idx, 0)
        freeze = has_new & (s[safe_new] <= 0.0)
        accept = jnp.all(jnp.where(passive, s > 0.0, True)) & ~freeze

        # Interpolation back into the feasible region (non-positive part of
        # ``s``); a vanishing denominator means ``x_i = s_i = 0``: step 0.
        blocking = passive & (s <= 0.0)
        denom = x - s
        ratio = jnp.where(
            blocking,
            jnp.where(denom > 0.0, x / jnp.where(denom > 0.0, denom, 1.0), 0.0),
            jnp.inf,
        )
        alpha = jnp.clip(jnp.min(ratio), 0.0, 1.0)
        x_int = x + alpha * (s - x)
        tiny = 10.0 * eps * jnp.maximum(1.0, jnp.max(x_int))
        released = passive & ((x_int <= tiny) | (jnp.arange(k) == jnp.argmin(ratio)))
        x_int = jnp.where(released, 0.0, jnp.maximum(x_int, 0.0))
        passive_int = passive & ~released

        freeze_mask = jnp.arange(k) == safe_new
        x_new = jnp.where(accept, s, jnp.where(freeze, x, x_int))
        passive_new = jnp.where(
            accept,
            passive,
            jnp.where(freeze, passive & ~freeze_mask, passive_int),
        )
        frozen_new = jnp.where(
            accept,
            jnp.zeros_like(frozen),
            jnp.where(freeze, frozen | freeze_mask, frozen),
        )
        n_iter = n_iter + jnp.where(accept, 0, 1).astype(jnp.int32)
        feasible = accept | freeze
        return (
            x_new,
            passive_new,
            frozen_new,
            jnp.int32(-1),
            n_iter,
            feasible,
            jnp.zeros((), bool),
            jnp.zeros((), bool),
        )

    def body(state):
        return lax.cond(state[5], add_step, solve_step, state)

    init = (
        jnp.zeros((k,), dtype),
        jnp.zeros((k,), bool),
        jnp.zeros((k,), bool),
        jnp.int32(-1),
        jnp.int32(0),
        jnp.ones((), bool),
        jnp.zeros((), bool),
        jnp.zeros((), bool),
    )
    x, passive, _, _, n_iter, _, _, converged = lax.while_loop(cond, body, init)
    return NNLSResult(  # ty: ignore[invalid-return-type]
        x=x,
        residual=e @ x - f,
        passive=passive,
        n_iter=n_iter,
        converged=converged,
    )


class LDP_RESULTS(eqx.Enumeration):
    """Outcomes of :func:`ldp`.

    ``inconsistent_equalities`` is defined here for the benefit of the
    equality-constrained reduction built on top of :func:`ldp` (the LSEI step
    of the direct QP solver); :func:`ldp` itself never produces it.
    """

    feasible = "The constraints are feasible; x is the least-norm point."
    infeasible = "The constraints g x >= h have no solution."
    inconsistent_equalities = (
        "The equality constraints are inconsistent "
        "(only produced by the LSEI reduction, never by ldp)."
    )
    max_iter_reached = "The NNLS iteration budget was exhausted."
    nonfinite = "The data or the computed solution contain non-finite values."


class LDPResult(eqx.Module):
    """Solution of a least-distance problem.

    Attributes
    ----------
    x
        Minimum-norm point of ``{x : g x ≥ h}``, shape ``(n,)``. Zero unless
        ``status`` is ``feasible``.
    multipliers
        Multipliers ``μ ≥ 0`` of the constraints, shape ``(k,)``, with
        ``x = gᵀ μ`` and ``μ_i (g x - h)_i = 0``. Zero unless ``status`` is
        ``feasible``.
    status
        :class:`LDP_RESULTS` code.
    n_iter
        Number of NNLS steps performed (``int32`` scalar).
    """

    x: Float[Array, " n"]
    multipliers: Float[Array, " k"]
    status: LDP_RESULTS
    n_iter: Int[Array, ""]


def ldp(
    g: Float[Array, "k n"],
    h: Float[Array, " k"],
    max_iter: int | None = None,
    tol: float | None = None,
    ldp_tol: float | None = None,
    dep_tol: float | None = None,
    check_tol: float | None = None,
) -> LDPResult:
    """Least-distance problem ``min ‖x‖₂  s.t.  g x ≥ h`` by NNLS.

    Lawson & Hanson (1974), ch. 23. With ``E = [gᵀ; hᵀ]`` (shape
    ``(n + 1, k)``) and ``f = e_{n+1}``, solve ``min ‖E u - f‖, u ≥ 0`` with
    :func:`nnls` and let ``r = E u - f``. Optimality of the NNLS problem gives
    ``r_{n+1} = -‖r‖²``, so

    * if ``‖r‖ ≤ ldp_tol`` the constraints are infeasible;
    * otherwise ``x = r_{:n} / ‖r‖² = gᵀ μ`` and the multipliers are
      ``μ = u / ‖r‖² ≥ 0`` (complementarity holds because ``u`` vanishes
      off the NNLS passive set). At the optimum these equal
      ``-r_{:n} / r_{n+1}`` and ``u / (1 - hᵀu)``, but ``r_{n+1} = hᵀu - 1``
      is computed by cancellation (relative error of ``x`` about
      ``eps ‖x‖²``, garbage for ``‖x‖ ≳ 1e8``), whereas ``‖r‖²`` is
      determined by the accurately computed ``r_{:n}`` (relative error about
      ``eps ‖x‖``). For ``‖r‖² ≤ 1/4`` it is evaluated from ``‖r_{:n}‖²``
      alone, through ``‖r‖² = ‖r_{:n}‖² + ‖r‖⁴``, so that the rounding noise
      of ``r_{n+1}`` does not enter even for ``‖x‖ ~ 1e12``.

    ``ldp`` normalises the data internally, in ``O(k n)``: rows of ``g`` to
    unit norm, then ``h`` to unit sup-norm (``y = x / s`` with
    ``s = max_i |h_i| / ‖g_i‖``; zero rows and a zero ``h`` keep scale 1),
    and rescales ``x = s y`` and ``μ_i = s μ̂_i / ‖g_i‖`` afterwards.
    Without it the NNLS dual tolerance, which grows with ``‖h‖``, would
    exceed the true duals (``∝ 1/‖x‖``) and stop NNLS early. Callers must
    not normalise themselves, and ``tol`` and ``ldp_tol`` apply to the
    normalised problem. Non-finite checks use the original data.

    **Degenerate polytopes.** If the feasible set has an empty interior (some
    ``c ≥ 0`` with ``gᵀc = 0`` and ``hᵀc = 0``: e.g. a singleton vertex where
    more constraints are active than there are free dimensions) the NNLS
    columns ``E`` are dependent up to rounding. Plain NNLS then admits such a
    column, its passive solve explodes, and ``ldp`` reports a false
    ``infeasible`` or, worse, a garbage ``x`` as ``feasible``. Two guards
    address this. ``dep_tol`` is passed to :func:`nnls` (a candidate column
    within relative distance ``dep_tol`` of the passive span is frozen, as in
    Kraft's SLSQP; the frozen constraint is implied by the passive ones up to
    ``dep_tol``). After forming the solution ``y`` of the normalised problem
    (``ĝ y ≥ ĥ/s``) its primal violation ``max(ĥ/s - ĝ y, 0)_∞`` is
    checked: above ``check_tol · max(1, ‖y‖)`` the status is ``infeasible``,
    so that ``feasible`` never comes with violated constraints.

    Status codes, in order of precedence: ``nonfinite`` when ``g``, ``h``,
    the NNLS residual or the solution contain a non-finite value (NNLS runs
    to its budget on such data); ``infeasible`` when ``‖r‖ ≤ ldp_tol``
    (also if NNLS ran out of budget: ``u ≥ 0`` with ``gᵀu ≈ 0``,
    ``hᵀu ≈ 1`` already certifies infeasibility) or when the primal check
    above fails; ``max_iter_reached`` when
    NNLS did not converge; else ``feasible``. ``feasible`` always means that
    ``x`` and ``multipliers`` were formed; otherwise both are zero. Zero
    constraints (``k == 0``) are handled by a static branch: ``x = 0``,
    feasible.

    Parameters
    ----------
    g
        Constraint matrix, shape ``(k, n)``.
    h
        Right-hand side, shape ``(k,)``.
    max_iter
        NNLS step budget (see :func:`nnls`); ``None`` selects ``3 k``.
    tol
        NNLS dual tolerance (see :func:`nnls`), applied to the normalised
        problem; ``None`` selects its default.
    ldp_tol
        Infeasibility threshold on ``‖r‖`` of the *normalised* problem (see
        above). The right-hand side ``f`` has unit norm, so this is a
        relative quantity. ``None`` selects the
        rounding-error level of the residual,
        ``10 · eps · max(n + 1, k) · (‖E‖_F ‖u‖ + 1)``: data that is
        infeasible by less than that is reported feasible, with a (very)
        large ``x``.
    dep_tol
        Linear-dependence threshold of the NNLS (see :func:`nnls`); ``None``
        selects ``√eps`` of the working dtype, ``0.0`` disables the test.
    check_tol
        Tolerance of the primal feasibility check of the solution, relative
        to ``max(1, ‖y‖)`` in the normalised units (see above); ``None``
        selects ``100 · √eps``, ``inf`` disables the check.

    Returns
    -------
    LDPResult
        ``x``, ``multipliers``, ``status`` and ``n_iter``. The float dtype
        follows the inputs (at least ``float32``).

    Examples
    --------
    Distance from the origin to the half-space ``x_1 + x_2 ≥ 2``:

    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.linalg import LDP_RESULTS, ldp
    >>> res = ldp(jnp.array([[1.0, 1.0]]), jnp.array([2.0]))
    >>> [round(float(v), 6) for v in res.x]
    [1.0, 1.0]
    >>> [round(float(v), 6) for v in res.multipliers]
    [1.0]
    >>> bool(res.status == LDP_RESULTS.feasible)
    True
    """
    dtype = jnp.result_type(g, h, jnp.float32)
    g = jnp.asarray(g, dtype=dtype)
    h = jnp.asarray(h, dtype=dtype)
    k, n = g.shape

    if k == 0:  # static shape: no constraint, the origin is optimal
        return LDPResult(  # ty: ignore[invalid-return-type]
            x=jnp.zeros((n,), dtype),
            multipliers=jnp.zeros((0,), dtype),
            status=LDP_RESULTS.feasible,
            n_iter=jnp.zeros((), jnp.int32),
        )

    # Normalise before the NNLS (O(k n)): unit rows, then h of unit sup-norm.
    # The NNLS dual tolerance scales with ‖E‖_F (hence with ‖h‖) whereas the
    # true duals shrink like 1/‖x‖, so without this NNLS stops early for
    # ‖h‖ ≳ 1e6 and a wrong ``feasible`` results. With y = x / s the problem
    # is ĝ y ≥ ĥ / s, and x = s y, μ_i = s μ̂_i / ‖g_i‖.
    row_norm = jnp.linalg.norm(g, axis=1)
    row_norm = jnp.where(row_norm > 0.0, row_norm, 1.0)
    h_hat = h / row_norm
    h_scale = jnp.max(jnp.abs(h_hat))
    h_scale = jnp.where(h_scale > 0.0, h_scale, 1.0)
    g_n = g / row_norm[:, None]
    h_n = h_hat / h_scale

    e = jnp.concatenate([g_n.T, h_n[None, :]], axis=0)  # (n + 1, k)
    f = jnp.zeros((n + 1,), dtype).at[n].set(1.0)
    eps = float(jnp.finfo(dtype).eps)
    dep_tol = eps**0.5 if dep_tol is None else dep_tol
    check_tol = 100.0 * eps**0.5 if check_tol is None else check_tol
    res = nnls(e, f, max_iter=max_iter, tol=tol, dep_tol=dep_tol)
    u, r = res.x, res.residual

    if ldp_tol is None:
        infeasible_tol = (
            10.0 * eps * max(n + 1, k) * (jnp.linalg.norm(e) * jnp.linalg.norm(u) + 1.0)
        )
    else:
        infeasible_tol = jnp.asarray(ldp_tol, dtype)
    infeasible = jnp.linalg.norm(r) <= infeasible_tol

    # ρ = ‖r‖² (= -r_n at the NNLS optimum). Summing all squares is accurate
    # unless ρ is tiny: then the cancelled entry r_n = hᵀu - 1 carries an
    # absolute error of order eps and its square pollutes ρ at the relative
    # level (eps/‖r‖)². Since ρ = a + ρ² with a = ‖r_{:n}‖², the small root
    # ρ = 2a / (1 + √(1 - 4a)) uses only the accurate part r_{:n}; it is used
    # for ρ ≤ 1/4, where the square root is well conditioned.
    usable = ~infeasible & res.converged
    a = jnp.sum(r[:n] ** 2)
    rho_full = a + r[n] ** 2
    rho_small = 2.0 * a / (1.0 + jnp.sqrt(jnp.maximum(1.0 - 4.0 * a, 0.0)))
    rho = jnp.where(rho_full <= 0.25, rho_small, rho_full)
    denom = jnp.where(usable, jnp.maximum(rho, jnp.finfo(dtype).tiny), 1.0)
    x = jnp.where(usable, r[:n] / denom, 0.0)
    multipliers = jnp.where(usable, u / denom, 0.0)
    # Primal check in the normalised units: a "feasible" answer must never
    # violate the constraints (garbage x from a near-singular passive solve
    # on a degenerate polytope). Frozen dependent columns only violate at the
    # ``dep_tol`` level, far below ``check_tol``.
    violation = jnp.max(jnp.maximum(h_n - g_n @ x, 0.0))
    violated = usable & (violation > check_tol * jnp.maximum(1.0, jnp.linalg.norm(x)))
    infeasible = infeasible | violated
    usable = usable & ~violated
    x = jnp.where(violated, 0.0, x)
    multipliers = jnp.where(violated, 0.0, multipliers)
    # Undo the normalisation (before the finiteness check, so an overflow of
    # the rescaled solution is reported as ``nonfinite``).
    x = h_scale * x
    multipliers = h_scale * multipliers / row_norm

    finite = (
        jnp.all(jnp.isfinite(g))
        & jnp.all(jnp.isfinite(h))
        & jnp.all(jnp.isfinite(r))
        & jnp.all(jnp.isfinite(x))
        & jnp.all(jnp.isfinite(multipliers))
    )
    # ``feasible`` is derived from ``usable``: it implies x, μ were formed.
    status = LDP_RESULTS.where(
        finite,
        LDP_RESULTS.where(
            infeasible,
            LDP_RESULTS.infeasible,
            LDP_RESULTS.where(
                usable, LDP_RESULTS.feasible, LDP_RESULTS.max_iter_reached
            ),
        ),
        LDP_RESULTS.nonfinite,
    )
    ok = finite & usable
    return LDPResult(  # ty: ignore[invalid-return-type]
        x=jnp.where(ok, x, 0.0),
        multipliers=jnp.where(ok, multipliers, 0.0),
        status=status,
        n_iter=res.n_iter,
    )
