"""Bound/slack-row elimination for the scaled interior-point normal equations.

The scaled constraint operator of an interior-point subproblem,
``Â = J(x, s) P`` with column blocks ``(x | s | s_lb | s_ub)``, has the rows

```
E:  [ J_E    0    0     0   ]      lb:  [ -M_l   0    S_l   0   ]
I:  [ J_I    S    0     0   ]      ub:  [  M_u   0    0     S_u ]
```

where ``M_l`` / ``M_u`` mask the live bounds and null-bound rows vanish
identically. Every bound row touches a single ``x_i`` and a single slack, so
the ``2n × 2n`` bound block of ``Â Âᵀ`` is diagonal per variable and can be
eliminated exactly. Writing ``r_x = (Âᵀ y)_x`` and

```
a_l = 1 / s_lb²  (live rows)      κ_l = free_x ∧ live_lb      c_l = κ_l a_l
a_u = 1 / s_ub²  (live rows)      κ_u = free_x ∧ live_ub      c_u = κ_u a_u
```

the bound rows give ``y_lb = a_l (b_lb + κ_l r_x)`` and
``y_ub = a_u (b_ub − κ_u r_x)``, and substituting back yields

```
r_x = D (Jᵀ y_g + h),     D = free_x / (1 + c_l + c_u),     h = −c_l b_lb + c_u b_ub,
S_g y_g = b_g − J D h,    S_g = J D Jᵀ + blockdiag(0, S²),   J = [J_E; J_I].
```

``D`` is diagonal with ``d_i = 1`` on free variables, ``d_i → 0`` as
``x_i`` approaches a bound and ``d_i = 0`` on frozen coordinates. Once
``y_g`` is known the bound multipliers are recovered in the
roundoff-stable form

```
y_lb = a_l D [b_lb + c_u σ + (Jᵀ y_g)_x]       (κ_l = 1)
y_ub = a_u D [b_ub + c_l σ − (Jᵀ y_g)_x]       (κ_u = 1)       σ = b_lb + b_ub
```

(and ``y = a b`` on bound rows decoupled from ``x``), which avoids the
catastrophic cancellation of ``a_l (b_lb + r_x)`` when a slack is tiny:
``a_l D ≤ 1`` there, so no ``1/s²`` factor ever multiplies a rounding
error. ``S_g`` is a Schur complement of ``Â Âᵀ`` and hence no worse
conditioned than ``Â Âᵀ`` itself. Two realisations of the reduction are
provided:

* :class:`SchurNormalEquations` forms the ``m × m`` matrix ``S_g`` once,
  takes its symmetric eigendecomposition and applies the Moore–Penrose
  pseudo-inverse. Each solve then costs ``O(mn + m²)`` and is exact to
  roundoff. No regularisation is ever added: rank deficiency is handled by
  clipping eigenvalues at ``rcond · λ_max``.
* :func:`slack_eliminated_normal_solver` keeps everything matrix-free and
  runs CG on ``S_g`` through the Jacobian products only: each iteration
  costs one ``J`` and one ``Jᵀ`` product, at most ``m`` iterations are
  needed in exact arithmetic (``O(m² n)`` per solve, independent of the
  number of bounds), and CG from zero on a consistent singular system
  stays in the range, so this path also returns the minimum-norm solution
  without a shift.

Both operate on the flat dual layout ``[eq | ineq | lb | ub]`` and return
exact zeros on rows whose slack column is dead (null or frozen bound).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Self

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from .projection import pcg

__all__ = [
    "NormalEquationsStrategy",
    "ResolvedNormalEquationsStrategy",
    "SchurNormalEquations",
    "resolve_normal_equations_strategy",
    "slack_eliminated_normal_solver",
    "strategy_code",
]

NormalEquationsStrategy = Literal["auto", "schur", "matrix-free", "generic"]
ResolvedNormalEquationsStrategy = Literal["schur", "matrix-free", "generic"]

_STRATEGY_CODES: dict[str, int] = {"generic": 0, "schur": 1, "matrix-free": 2}

Operator = Callable[[Float[Array, " a"]], Float[Array, " b"]]


def resolve_normal_equations_strategy(
    strategy: NormalEquationsStrategy, m: int, schur_max_rows: int
) -> ResolvedNormalEquationsStrategy:
    """Resolve ``"auto"`` against the static number of general constraint rows.

    Parameters
    ----------
    strategy
        Requested strategy.
    m
        Number of general (equality + inequality) rows, ``m_E + m_I``.
    schur_max_rows
        ``"auto"`` resolves to ``"schur"`` when ``m < schur_max_rows`` and
        to ``"matrix-free"`` otherwise.

    Returns
    -------
    str
        One of ``"schur"``, ``"matrix-free"``, ``"generic"``.

    Raises
    ------
    ValueError
        On an unknown strategy name.

    Examples
    --------
    >>> from slsqp_jax.sqpdax.linalg import resolve_normal_equations_strategy
    >>> resolve_normal_equations_strategy("auto", 3, 100)
    'schur'
    >>> resolve_normal_equations_strategy("auto", 100, 100)
    'matrix-free'
    """
    if strategy == "auto":
        return "schur" if m < schur_max_rows else "matrix-free"
    if strategy in ("schur", "matrix-free", "generic"):
        return strategy
    raise ValueError(
        f"unknown normal-equations strategy {strategy!r}; expected one of "
        "'auto', 'schur', 'matrix-free', 'generic'"
    )


def strategy_code(strategy: ResolvedNormalEquationsStrategy) -> Int[Array, ""]:
    """Integer code of a resolved strategy for diagnostic payloads.

    ``0`` = generic, ``1`` = schur, ``2`` = matrix-free.
    """
    return jnp.asarray(_STRATEGY_CODES[strategy], jnp.int32)


def _bound_weights(
    s_lb: Float[Array, " n"],
    s_ub: Float[Array, " n"],
    live_lb: Bool[Array, " n"],
    live_ub: Bool[Array, " n"],
    free_x: Bool[Array, " n"] | None,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Return ``(a_l, a_u, κ_l, κ_u, free, d)`` of the module docstring."""
    dtype = s_lb.dtype
    one = jnp.ones_like(s_lb)
    free = one if free_x is None else free_x.astype(dtype)
    a_l = jnp.where(live_lb, 1.0 / jnp.where(live_lb, s_lb, 1.0) ** 2, 0.0)
    a_u = jnp.where(live_ub, 1.0 / jnp.where(live_ub, s_ub, 1.0) ** 2, 0.0)
    kappa_l = free * live_lb.astype(dtype)
    kappa_u = free * live_ub.astype(dtype)
    d = free / (1.0 + kappa_l * a_l + kappa_u * a_u)
    return a_l, a_u, kappa_l, kappa_u, free, d


def _bound_multipliers(
    b_lb: Float[Array, " n"],
    b_ub: Float[Array, " n"],
    g: Float[Array, " n"],
    a_l: Float[Array, " n"],
    a_u: Float[Array, " n"],
    kappa_l: Float[Array, " n"],
    kappa_u: Float[Array, " n"],
    d: Float[Array, " n"],
) -> tuple[Float[Array, " n"], Float[Array, " n"]]:
    """Roundoff-stable bound multipliers given ``g = (Jᵀ y_g)_x``.

    Implements the stable form of the module docstring: coupled rows use
    ``a D [...]`` with ``a D ≤ 1``, decoupled rows use ``a b``.
    """
    c_l = kappa_l * a_l
    c_u = kappa_u * a_u
    # ``σ = b_lb + b_ub`` is evaluated once so that its rounding error enters
    # both multipliers with the same weight ``a_l a_u d`` and therefore lies
    # along the near-null direction of the two-bound block (``s_lb, s_ub → 0``)
    # instead of producing a residual.
    sigma = b_lb + b_ub
    coupled_lb = d * (b_lb + c_u * sigma + g)
    coupled_ub = d * (b_ub + c_l * sigma - g)
    y_lb = a_l * ((1.0 - kappa_l) * b_lb + kappa_l * coupled_lb)
    y_ub = a_u * ((1.0 - kappa_u) * b_ub + kappa_u * coupled_ub)
    return y_lb, y_ub


class SchurNormalEquations(eqx.Module):
    """Explicit Schur-complement pseudo-inverse of the scaled normal equations.

    Built once per subproblem (``Â`` is fixed within an outer iteration) and
    applied through :meth:`solve`. See the module docstring for the
    reduction; the general block is handled by the symmetric
    eigendecomposition ``S_g = V Λ Vᵀ`` and ``S_g⁺ = V Λ⁺ Vᵀ`` with
    ``λ⁺ = 1/λ`` for ``λ > rcond · λ_max`` and ``0`` otherwise.

    The returned ``y`` is the **minimum-norm** solution of ``Â Âᵀ y = b``
    (the least-squares solution when ``b ∉ range(Â)``): the null space of
    ``Â Âᵀ`` lives entirely in the equality rows (``Âᵀ y = 0`` forces every
    slack-weighted component to vanish), it coincides with ``null(S_g)``
    there, and ``S_g⁺`` returns a vector orthogonal to it.

    Attributes
    ----------
    n, meq, mineq
        Static sizes.
    rcond
        Relative eigenvalue cutoff used to build the pseudo-inverse.
    jac
        Stacked general Jacobian ``J = [J_E; J_I]`` (``m × n``).
    d
        Diagonal ``D`` (length ``n``).
    a_lb, a_ub
        ``1 / s²`` on live bound rows, ``0`` on dead rows.
    kappa_lb, kappa_ub
        ``0/1`` coupling indicators between bound rows and ``x``.
    eigvecs, inv_eigvals
        Eigenvectors of ``S_g`` and the clipped reciprocal eigenvalues.
    rank
        Number of eigenvalues kept.

    Examples
    --------
    ```python
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.linalg import SchurNormalEquations
    >>> J_E = jnp.array([[1.0, 1.0]])
    >>> J_I = jnp.zeros((0, 2))
    >>> s = jnp.zeros((0,))
    >>> live = jnp.array([True, False])
    >>> ne = SchurNormalEquations.build(
    ...     J_E, J_I, s, jnp.array([0.5, 1.0]), jnp.array([1.0, 1.0]),
    ...     live_lb=live, live_ub=jnp.array([False, False]),
    ... )
    >>> int(ne.rank)
    1
    >>> y = ne.solve(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    >>> y.shape
    (5,)

    ```
    """

    n: int = eqx.field(static=True)
    meq: int = eqx.field(static=True)
    mineq: int = eqx.field(static=True)
    rcond: float = eqx.field(static=True)
    jac: Float[Array, "m n"]
    d: Float[Array, " n"]
    a_lb: Float[Array, " n"]
    a_ub: Float[Array, " n"]
    kappa_lb: Float[Array, " n"]
    kappa_ub: Float[Array, " n"]
    s_sq: Float[Array, " mineq"]
    eigvecs: Float[Array, "m m"]
    inv_eigvals: Float[Array, " m"]
    rank: Int[Array, ""]

    @classmethod
    def build(
        cls,
        jac_eq: Float[Array, "meq n"],
        jac_ineq: Float[Array, "mineq n"],
        s: Float[Array, " mineq"],
        s_lb: Float[Array, " n"],
        s_ub: Float[Array, " n"],
        *,
        live_lb: Bool[Array, " n"],
        live_ub: Bool[Array, " n"],
        free_x: Bool[Array, " n"] | None = None,
        rcond: float | None = None,
    ) -> Self:
        """Assemble ``S_g`` and its pseudo-inverse.

        Parameters
        ----------
        jac_eq, jac_ineq
            Dense equality / inequality Jacobians with respect to ``x``.
        s
            General inequality slacks (scale of the ``s`` columns).
        s_lb, s_ub
            Bound slacks; ignored on dead rows.
        live_lb, live_ub
            Rows whose slack column is live (not null and not frozen).
        free_x
            Optional mask of free ``x`` columns; frozen columns are removed
            from ``Â`` (their bound rows then decouple from ``x``).
        rcond
            Relative eigenvalue cutoff; ``None`` uses ``m · eps(dtype)``.

        Returns
        -------
        SchurNormalEquations
            Factor ready for :meth:`solve`.
        """
        meq, n = jac_eq.shape
        mineq = jac_ineq.shape[0]
        m = meq + mineq
        dtype = jac_eq.dtype
        jac = jnp.concatenate([jac_eq, jac_ineq], axis=0)
        a_l, a_u, kappa_l, kappa_u, _free, d = _bound_weights(
            s_lb, s_ub, live_lb, live_ub, free_x
        )
        s_sq = s * s
        if rcond is None:
            rcond = float(max(m, 1) * jnp.finfo(dtype).eps)
        if m == 0:
            eigvecs = jnp.zeros((0, 0), dtype)
            inv_eigvals = jnp.zeros((0,), dtype)
            rank = jnp.asarray(0, jnp.int32)
        else:
            schur = (jac * d) @ jac.T
            diag = jnp.concatenate([jnp.zeros((meq,), dtype), s_sq])
            schur = schur + jnp.diag(diag)
            eigvals, eigvecs = jnp.linalg.eigh(schur)
            lam_max = jnp.maximum(jnp.max(eigvals), 0.0)
            keep = eigvals > jnp.asarray(rcond, dtype) * lam_max
            inv_eigvals = jnp.where(keep, 1.0 / jnp.where(keep, eigvals, 1.0), 0.0)
            rank = jnp.sum(keep).astype(jnp.int32)
        return cls(
            n=n,
            meq=meq,
            mineq=mineq,
            rcond=rcond,
            jac=jac,
            d=d,
            a_lb=a_l,
            a_ub=a_u,
            kappa_lb=kappa_l,
            kappa_ub=kappa_u,
            s_sq=s_sq,
            eigvecs=eigvecs,
            inv_eigvals=inv_eigvals,
            rank=rank,
        )

    @property
    def m(self) -> int:
        """Number of general rows ``m_E + m_I``."""
        return self.meq + self.mineq

    def pinv_general(self, r: Float[Array, " m"]) -> Float[Array, " m"]:
        """Apply ``S_g⁺`` to a vector on the general rows."""
        return self.eigvecs @ (self.inv_eigvals * (self.eigvecs.T @ r))

    def split(self, flat: Float[Array, " md"]) -> tuple[Array, Array, Array]:
        """Split a flat dual ``[eq | ineq | lb | ub]`` into general / lb / ub."""
        m, n = self.m, self.n
        return flat[:m], flat[m : m + n], flat[m + n :]

    def solve(self, rhs: Float[Array, " md"]) -> Float[Array, " md"]:
        """Minimum-norm solution ``y`` of ``Â Âᵀ y = rhs``.

        Parameters
        ----------
        rhs
            Flat right-hand side in the ``[eq | ineq | lb | ub]`` layout.

        Returns
        -------
        jax.Array
            Flat ``y`` in the same layout; exact zeros on dead bound rows.
        """
        b_g, b_lb, b_ub = self.split(rhs)
        h = -self.kappa_lb * self.a_lb * b_lb + self.kappa_ub * self.a_ub * b_ub
        y_g = self.pinv_general(b_g - self.jac @ (self.d * h))
        y_lb, y_ub = _bound_multipliers(
            b_lb,
            b_ub,
            self.jac.T @ y_g,
            self.a_lb,
            self.a_ub,
            self.kappa_lb,
            self.kappa_ub,
            self.d,
        )
        return jnp.concatenate([y_g, y_lb, y_ub])


def slack_eliminated_normal_solver(
    jac_eq_mvp: Operator,
    jac_eq_rmvp: Operator,
    jac_ineq_mvp: Operator,
    jac_ineq_rmvp: Operator,
    s: Float[Array, " mineq"],
    s_lb: Float[Array, " n"],
    s_ub: Float[Array, " n"],
    *,
    live_lb: Bool[Array, " n"],
    live_ub: Bool[Array, " n"],
    free_x: Bool[Array, " n"] | None = None,
    meq: int,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> Callable[..., Array | tuple[Array, Int[Array, ""]]]:
    """Matrix-free CG solve of ``Â Âᵀ y = b`` with the bound rows eliminated.

    Applies the reduction of the module docstring without forming anything:
    the ``m × m`` operator ``S_g = J D Jᵀ + blockdiag(0, S²)`` is applied
    through one ``J`` and one ``Jᵀ`` product per CG iteration, so a solve
    costs ``O(m² n)`` regardless of how many bounds are present, and the
    bound multipliers follow from the stable closed form. CG starts from
    zero, hence returns the minimum-norm solution on consistent singular
    systems.

    Parameters
    ----------
    jac_eq_mvp, jac_eq_rmvp
        ``v ↦ J_E v`` and ``y ↦ J_Eᵀ y``.
    jac_ineq_mvp, jac_ineq_rmvp
        ``v ↦ J_I v`` and ``y ↦ J_Iᵀ y``.
    s, s_lb, s_ub
        Slacks (general, lower-bound, upper-bound).
    live_lb, live_ub
        Rows whose slack column is live.
    free_x
        Optional mask of free ``x`` columns.
    meq
        Static number of equality rows.
    tol, max_iter
        Relative residual tolerance and iteration cap of the CG on ``S_g``.

    Returns
    -------
    Callable
        ``solve(rhs, *, with_info=False)``; with ``with_info=True`` it
        returns ``(y, iterations)``.
    """
    n = s_lb.shape[0]
    mineq = s.shape[0]
    m = meq + mineq
    dtype = s_lb.dtype
    a_l, a_u, kappa_l, kappa_u, _free, d = _bound_weights(
        s_lb, s_ub, live_lb, live_ub, free_x
    )
    s_sq = s * s
    zeros_eq = jnp.zeros((meq,), dtype)

    def jac_mvp(v: Float[Array, " n"]) -> Float[Array, " m"]:
        return jnp.concatenate([jac_eq_mvp(v), jac_ineq_mvp(v)])

    def jac_rmvp(y: Float[Array, " m"]) -> Float[Array, " n"]:
        return jac_eq_rmvp(y[:meq]) + jac_ineq_rmvp(y[meq:])

    def schur_op(y: Float[Array, " m"]) -> Float[Array, " m"]:
        return jac_mvp(d * jac_rmvp(y)) + jnp.concatenate([zeros_eq, s_sq * y[meq:]])

    def solve(rhs: Float[Array, " md"], *, with_info: bool = False):
        b_g, b_lb, b_ub = rhs[:m], rhs[m : m + n], rhs[m + n :]
        h = -kappa_l * a_l * b_lb + kappa_u * a_u * b_ub
        if m == 0:
            y_g = b_g
            n_iter = jnp.asarray(0, jnp.int32)
        else:
            y_g, n_iter = pcg(
                schur_op, b_g - jac_mvp(d * h), tol=tol, max_iter=max_iter
            )
        y_lb, y_ub = _bound_multipliers(
            b_lb, b_ub, jac_rmvp(y_g), a_l, a_u, kappa_l, kappa_u, d
        )
        y = jnp.concatenate([y_g, y_lb, y_ub])
        if with_info:
            return y, n_iter
        return y

    return solve
