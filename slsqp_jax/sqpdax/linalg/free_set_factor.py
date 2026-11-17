"""Factor of the free-variable block of a reduced-Hessian model.

For a limited-memory reduced-Hessian (LRH) model
``B = σ I + Z (Rᵀ R - σ I) Zᵀ`` and a set ``F`` of free variables (the
complement ``W`` is pinned at a bound), the QP on the free variables uses
the block ``B_FF = I_F B I_F`` where ``I_F = diag(F)``. Rather than rotating
the basis each time a bound enters or leaves the working set, the basis
``Z`` is kept fixed and the restricted model is re-derived from small
matrices (Gill & Leonard, 2003; Ferry, Gill, Wong & Zhang, 2021, for the
bound-constrained setting):

```text
Z_F   = I_F Z                        (n × m, rows of fixed variables zeroed)
Z_Fᵀ Z_F = V Λ Vᵀ                    (m × m eigen-decomposition)
Q     = Z_F V_k Λ_k^{-1/2}           (supported on F; orthonormal to O(eps/Λ_min))
K     = σ I + Λ_k^{1/2} V_kᵀ (RᵀR - σ I) V_k Λ_k^{1/2} = L Lᵀ
B_FF  = Q K Qᵀ + σ (I_F - Q Qᵀ)
C_F   = Q Lᵀ Qᵀ + √σ (I_F - Q Qᵀ)     C_Fᵀ C_F = B_FF
```

Every object is ``n × m``, ``m × m`` or an ``n``-vector (``m`` = LRH
memory): the mask is applied elementwise and no ``n × n`` array is formed.
Building costs ``O(n m²)`` plus one ``m × m`` eigendecomposition and one
``m × m`` Cholesky factorisation; every application costs ``O(n m)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
from jax import numpy as jnp
from jax.scipy.linalg import cho_solve, solve_triangular
from jaxtyping import Array, Bool, Float, Int

from ..types import Scalar

if TYPE_CHECKING:  # avoid an import cycle with the secant package
    from ..secant.reduced_hessian import ReducedHessianFactors

__all__ = ["FreeSetFactor"]


def _eps_rank_tol(dtype, m: int) -> float:
    """Default eigenvalue threshold ``1e3 · eps · m`` for dropping directions."""
    return 1e3 * float(jnp.finfo(dtype).eps) * m


class FreeSetFactor(eqx.Module):
    """Factor ``C_F`` of the free-variable block ``B_FF`` of an LRH model.

    Build it with :meth:`build`. All ``apply_*`` and ``hvp`` / ``solve``
    methods take an ``n``-vector, mask it to the free set before applying
    the operator and mask the result, so every output is exactly zero on the
    fixed variables. The ``*_matrix`` variants do the same column by column
    for an ``n × k`` block. Eigen-directions of ``Z_Fᵀ Z_F`` with
    ``Λ ≤ rank_tol`` (including the padded columns of ``Z``) are dropped:
    they contribute zero columns to ``Q`` (never ``NaN``) and the operator
    acts there as the complement curvature ``σ``.

    Attributes
    ----------
    free_mask
        Boolean mask of the free variables, shape ``(n,)``.
    basis_q
        ``Q``, shape ``(n, m)``: columns supported on the free variables,
        with zero columns for dropped directions. Orthonormal only to
        ``O(eps / Λ_min)``; use the ``apply_*`` methods rather than
        ``basis_q`` as an exact orthonormal basis.
    k_matrix
        ``K = Qᵀ B Q``, shape ``(m, m)`` (``σ`` on the dropped block).
    k_chol
        Lower-triangular Cholesky factor ``L`` of ``K``, shape ``(m, m)``.
    sigma
        Complement curvature ``σ > 0``.
    free_rank
        Number of kept directions, i.e. the numerical rank of ``Z_F``
        (``int32`` scalar array).
    """

    free_mask: Bool[Array, " n"]
    basis_q: Float[Array, "n m"]
    k_matrix: Float[Array, "m m"]
    k_chol: Float[Array, "m m"]
    sigma: Scalar
    free_rank: Int[Array, ""]

    @classmethod
    def build(
        cls,
        factors: ReducedHessianFactors,
        free_mask: Bool[Array, " n"],
        rank_tol: float | None = None,
    ) -> FreeSetFactor:
        """Factor the free-variable block of ``factors`` for a free mask.

        Parameters
        ----------
        factors
            Reduced-Hessian factors ``(Z, R, σ, r)`` of the model. Its
            :attr:`~slsqp_jax.sqpdax.secant.reduced_hessian.ReducedHessianFactors.active_cholesky`
            (identity-padded ``R``) is used, so the padded block of the
            stored ``cholesky`` is ignored.
        free_mask
            Boolean mask, shape ``(n,)``, ``True`` for free variables. It
            may be traced.
        rank_tol
            Eigenvalues ``Λ`` of ``Z_Fᵀ Z_F`` (which lie in ``[0, 1]``)
            at or below this threshold are dropped. ``None`` selects
            ``1e3 · eps · m``.

        Returns
        -------
        FreeSetFactor
            The factor. Building costs one masked ``n × m`` Gram product,
            one ``m × m`` ``eigh`` and one ``m × m`` Cholesky.
        """
        basis = factors.basis
        m = basis.shape[1]
        dtype = basis.dtype
        sigma = jnp.asarray(factors.sigma, dtype=dtype)
        free = jnp.asarray(free_mask, dtype=bool)
        tol = _eps_rank_tol(dtype, m) if rank_tol is None else rank_tol

        z_f = jnp.where(free[:, None], basis, 0.0)
        lam, vec = jnp.linalg.eigh(z_f.T @ z_f)
        keep = lam > tol
        lam_k = jnp.where(keep, jnp.clip(lam, 0.0, 1.0), 0.0)
        sqrt_lam = jnp.sqrt(lam_k)
        inv_sqrt_lam = jnp.where(keep, 1.0 / jnp.sqrt(jnp.where(keep, lam, 1.0)), 0.0)

        basis_q = (z_f @ vec) * inv_sqrt_lam[None, :]

        # K = Qᵀ B Q = Wᵀ W + σ (I - Λ_k) with W = R V Λ_k^{1/2}; `1 - Λ` is
        # clipped at zero through `lam_k`, so K is SPD even when roundoff
        # pushes Λ above 1, and equals σ I on the dropped directions.
        w = factors.active_cholesky @ (vec * sqrt_lam[None, :])
        k_matrix = w.T @ w + sigma * jnp.diag(1.0 - lam_k)
        k_chol = jnp.linalg.cholesky(k_matrix)
        return cls(
            free_mask=free,
            basis_q=basis_q,
            k_matrix=k_matrix,
            k_chol=k_chol,
            sigma=sigma,
            free_rank=jnp.sum(keep).astype(jnp.int32),
        )

    # -- internals ---------------------------------------------------------

    def _mask(self, x: Array) -> Array:
        """Zero the rows of a vector / ``n × k`` block on fixed variables."""
        mask = self.free_mask.reshape((-1,) + (1,) * (x.ndim - 1))
        return jnp.where(mask, x, 0.0)

    def _apply(self, x: Array, op, scale: Array) -> Array:
        """``scale · x_F + Q (op(Qᵀ x_F) - scale · Qᵀ x_F)``, masked.

        ``op`` acts on the ``m``-dimensional coordinates; the orthogonal
        complement of ``range(Q)`` inside the free set is scaled by
        ``scale``.
        """
        xf = self._mask(x)
        u = self.basis_q.T @ xf
        return self._mask(scale * xf + self.basis_q @ (op(u) - scale * u))

    # -- vector applies ----------------------------------------------------

    def hvp(self, v: Float[Array, " n"]) -> Float[Array, " n"]:
        """Apply ``B_FF = Q K Qᵀ + σ (I_F - Q Qᵀ)`` (input and output masked).

        Parameters
        ----------
        v
            Vector of length ``n``.

        Returns
        -------
        Array
            ``I_F B I_F v``, zero on the fixed variables.
        """
        return self._apply(v, lambda u: self.k_matrix @ u, self.sigma)

    def solve(self, v: Float[Array, " n"]) -> Float[Array, " n"]:
        """Apply ``B_FF⁻¹ = Q K⁻¹ Qᵀ + σ⁻¹ (I_F - Q Qᵀ)`` on the free set.

        Parameters
        ----------
        v
            Vector of length ``n``.

        Returns
        -------
        Array
            The solution of ``B_FF x = I_F v`` supported on the free set.
        """
        return self._apply(
            v, lambda u: cho_solve((self.k_chol, True), u), 1.0 / self.sigma
        )

    def apply_c(self, v: Float[Array, " n"]) -> Float[Array, " n"]:
        """Apply ``C_F = Q Lᵀ Qᵀ + √σ (I_F - Q Qᵀ)`` (``C_Fᵀ C_F = B_FF``).

        Parameters
        ----------
        v
            Vector of length ``n``.

        Returns
        -------
        Array
            ``C_F v``, zero on the fixed variables.
        """
        return self._apply(v, lambda u: self.k_chol.T @ u, jnp.sqrt(self.sigma))

    def apply_c_t(self, v: Float[Array, " n"]) -> Float[Array, " n"]:
        """Apply ``C_Fᵀ = Q L Qᵀ + √σ (I_F - Q Qᵀ)``.

        Parameters
        ----------
        v
            Vector of length ``n``.

        Returns
        -------
        Array
            ``C_Fᵀ v``, zero on the fixed variables.
        """
        return self._apply(v, lambda u: self.k_chol @ u, jnp.sqrt(self.sigma))

    def apply_c_inv(self, v: Float[Array, " n"]) -> Float[Array, " n"]:
        """Apply ``C_F⁻¹ = Q L⁻ᵀ Qᵀ + σ^{-1/2} (I_F - Q Qᵀ)`` on the free set.

        Parameters
        ----------
        v
            Vector of length ``n``.

        Returns
        -------
        Array
            The inverse of ``C_F`` restricted to the free set applied to
            ``I_F v``, zero on the fixed variables.
        """
        return self._apply(
            v,
            lambda u: solve_triangular(self.k_chol, u, trans=1, lower=True),
            1.0 / jnp.sqrt(self.sigma),
        )

    def apply_c_inv_t(self, v: Float[Array, " n"]) -> Float[Array, " n"]:
        """Apply ``C_F⁻ᵀ = Q L⁻¹ Qᵀ + σ^{-1/2} (I_F - Q Qᵀ)`` on the free set.

        Parameters
        ----------
        v
            Vector of length ``n``.

        Returns
        -------
        Array
            The inverse of ``C_Fᵀ`` restricted to the free set applied to
            ``I_F v``, zero on the fixed variables.
        """
        return self._apply(
            v,
            lambda u: solve_triangular(self.k_chol, u, trans=0, lower=True),
            1.0 / jnp.sqrt(self.sigma),
        )

    # -- n × k block applies -----------------------------------------------

    def hvp_matrix(self, x: Float[Array, "n k"]) -> Float[Array, "n k"]:
        """Column-wise :meth:`hvp` of an ``n × k`` block (``O(n m k)``).

        Parameters
        ----------
        x
            Block of ``k`` columns, shape ``(n, k)``.

        Returns
        -------
        Array
            Shape ``(n, k)``, rows of fixed variables zero.
        """
        return self.hvp(x)

    def solve_matrix(self, x: Float[Array, "n k"]) -> Float[Array, "n k"]:
        """Column-wise :meth:`solve` of an ``n × k`` block (``O(n m k)``).

        Parameters
        ----------
        x
            Block of ``k`` columns, shape ``(n, k)``.

        Returns
        -------
        Array
            Shape ``(n, k)``, rows of fixed variables zero.
        """
        return self.solve(x)

    def apply_c_matrix(self, x: Float[Array, "n k"]) -> Float[Array, "n k"]:
        """Column-wise :meth:`apply_c` of an ``n × k`` block.

        Parameters
        ----------
        x
            Block of ``k`` columns, shape ``(n, k)``.

        Returns
        -------
        Array
            Shape ``(n, k)``, rows of fixed variables zero.
        """
        return self.apply_c(x)

    def apply_c_t_matrix(self, x: Float[Array, "n k"]) -> Float[Array, "n k"]:
        """Column-wise :meth:`apply_c_t` of an ``n × k`` block.

        Parameters
        ----------
        x
            Block of ``k`` columns, shape ``(n, k)``.

        Returns
        -------
        Array
            Shape ``(n, k)``, rows of fixed variables zero.
        """
        return self.apply_c_t(x)

    def apply_c_inv_matrix(self, x: Float[Array, "n k"]) -> Float[Array, "n k"]:
        """Column-wise :meth:`apply_c_inv` of an ``n × k`` block.

        Parameters
        ----------
        x
            Block of ``k`` columns, shape ``(n, k)``.

        Returns
        -------
        Array
            Shape ``(n, k)``, rows of fixed variables zero.
        """
        return self.apply_c_inv(x)

    def apply_c_inv_t_matrix(self, x: Float[Array, "n k"]) -> Float[Array, "n k"]:
        """Column-wise :meth:`apply_c_inv_t` of an ``n × k`` block.

        Used to form ``C_F⁻ᵀ A_Fᵀ`` for the general constraints ``A``
        (``x = Aᵀ``, ``k = m_c`` columns).

        Parameters
        ----------
        x
            Block of ``k`` columns, shape ``(n, k)``.

        Returns
        -------
        Array
            Shape ``(n, k)``, rows of fixed variables zero.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from slsqp_jax.sqpdax.linalg import FreeSetFactor
        >>> from slsqp_jax.sqpdax.secant import ReducedHessianFactors
        >>> factors = ReducedHessianFactors(
        ...     basis=jnp.zeros((4, 2)),
        ...     cholesky=jnp.eye(2),
        ...     sigma=jnp.asarray(4.0),
        ...     rank=jnp.asarray(0, dtype=jnp.int32),
        ... )
        >>> free = jnp.array([True, True, False, True])
        >>> fs = FreeSetFactor.build(factors, free)
        >>> fs.apply_c_inv_t_matrix(jnp.ones((4, 1))).ravel().tolist()
        [0.5, 0.5, 0.0, 0.5]
        """
        return self.apply_c_inv_t(x)
