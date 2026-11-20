"""Structure access for secants stored as a limited-memory reduced Hessian.

A reduced-Hessian (RH) approximation (Gill & Leonard, 2003, *SIAM J. Optim.*
14) represents the model as

```text
B = σ I + Z (Rᵀ R - σ I) Zᵀ
```

with ``Z`` an ``n × m`` basis whose first ``r`` columns are orthonormal and
whose remaining columns are zero, ``R`` an ``m × m`` upper-triangular
Cholesky factor of the reduced Hessian (identity on the padded block) and
``σ > 0`` the curvature on the orthogonal complement of ``range(Z)``.

:class:`ReducedHessianFactors` is the plain pytree of these quantities, with
the ``O(n r)`` products ``B v`` and ``B⁻¹ v``. :class:`ReducedHessianSecant`
is the abstract capability a direct (factor-based) subproblem solver
requires from a secant; it is not registered under any ``kind``.
"""

from abc import abstractmethod

import equinox as eqx
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array, Bool, Float, Int

from ..types import Scalar, Vector_n
from .base import Secant

__all__ = [
    "ReducedHessianFactors",
    "ReducedHessianSecant",
]


class ReducedHessianFactors(eqx.Module):
    """Factors ``(Z, R, σ, r)`` of a reduced-Hessian model.

    The model is ``B = σ I + Z (Rᵀ R - σ I) Zᵀ``. Fixed shapes are kept by
    zero padding: only the first ``rank`` columns of ``basis`` are used, and
    only the leading ``rank × rank`` block of ``cholesky`` is read. No
    object of size ``n × n`` is ever formed.

    Attributes
    ----------
    basis
        ``Z``, shape ``(n, m)``; first ``rank`` columns orthonormal, the
        rest zero.
    cholesky
        ``R``, shape ``(m, m)``, upper triangular; its padded block is
        ignored (the products below replace it by the identity).
    sigma
        Complement curvature ``σ > 0``.
    rank
        Current rank ``r`` (``int32`` scalar array), ``0 ≤ r ≤ m``.
    """

    basis: Float[Array, "n m"]
    cholesky: Float[Array, "m m"]
    sigma: Scalar
    rank: Int[Array, ""]

    @property
    def column_mask(self) -> Bool[Array, " m"]:
        """Boolean mask of the active (non-padded) columns, ``index < rank``.

        Returns
        -------
        Bool[Array, " m"]
            ``True`` for the first ``rank`` entries.
        """
        return jnp.arange(self.basis.shape[1]) < self.rank

    @property
    def active_cholesky(self) -> Float[Array, "m m"]:
        """``R`` with its padded block replaced by the identity.

        The leading ``rank × rank`` block equals the stored ``cholesky``;
        every other entry of the ``m × m`` result is that of the identity, so
        the result is invertible whatever is stored in the padded block.

        Returns
        -------
        Float[Array, "m m"]
            Upper-triangular ``m × m`` matrix (``O(m²)`` memory).
        """
        mask = self.column_mask
        active = mask[:, None] & mask[None, :]
        eye = jnp.eye(mask.shape[0], dtype=self.cholesky.dtype)
        return jnp.where(active, self.cholesky, eye)

    def hvp(self, v: Vector_n) -> Vector_n:
        """Apply the model: ``B v = σ v + Z (RᵀR - σ I) Zᵀ v``.

        Parameters
        ----------
        v
            Vector of length ``n``.

        Returns
        -------
        Vector_n
            ``B v``, computed in ``O(n r)``.
        """
        r_mat = self.active_cholesky
        u = self.basis.T @ v
        w = r_mat.T @ (r_mat @ u) - self.sigma * u
        return self.sigma * v + self.basis @ w

    def inverse_hvp(self, v: Vector_n) -> Vector_n:
        """Apply the inverse: ``B⁻¹ v = σ⁻¹ v + Z (R⁻¹R⁻ᵀ - σ⁻¹ I) Zᵀ v``.

        Parameters
        ----------
        v
            Vector of length ``n``.

        Returns
        -------
        Vector_n
            ``B⁻¹ v``, computed in ``O(n r)`` with two ``m × m`` triangular
            solves.
        """
        r_mat = self.active_cholesky
        u = self.basis.T @ v
        t = solve_triangular(r_mat, u, trans=1, lower=False)
        t = solve_triangular(r_mat, t, trans=0, lower=False)
        w = t - u / self.sigma
        return v / self.sigma + self.basis @ w


class ReducedHessianSecant(Secant):
    """Abstract secant that exposes its reduced-Hessian factors.

    Direct subproblem solvers need ``Z``, ``R`` and ``σ`` (not only
    ``hvp``) to build factors of the free-variable block. They depend on
    this capability (``isinstance(secant, ReducedHessianSecant)``) rather
    than on a concrete class. This class defines no ``kind`` and is
    therefore not registered in :class:`~slsqp_jax.sqpdax.secant.base.Secant`'s
    registry; it cannot be instantiated.
    """

    @abstractmethod
    def factors(self) -> ReducedHessianFactors:
        """Return the current reduced-Hessian factors.

        Returns
        -------
        ReducedHessianFactors
            ``(Z, R, σ, r)`` of the current model. None of the returned
            arrays is of size ``n × n``.
        """
        ...  # pragma: no cover
