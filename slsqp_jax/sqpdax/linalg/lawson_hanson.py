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
"""

from __future__ import annotations

import equinox as eqx
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

__all__ = ["NNLSResult", "nnls"]


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
        passive = jnp.where(stop, passive, passive.at[j].set(True))
        new_idx = jnp.where(stop, jnp.int32(-1), j)
        n_iter = n_iter + jnp.where(stop, 0, 1).astype(jnp.int32)
        return x, passive, frozen, new_idx, n_iter, stop, stop, optimal

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
