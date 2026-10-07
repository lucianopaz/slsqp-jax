"""Matrix-free spectral-norm estimates by power iteration."""

from collections.abc import Callable

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Scalar

__all__ = ["power_iteration_norm", "spectral_norm_estimate"]


def power_iteration_norm(
    matvec: Callable[[Float[Array, " n"]], Float[Array, " n"]],
    x0: Float[Array, " n"],
    *,
    n_iter: int = 10,
) -> Scalar:
    """Estimate the dominant eigenvalue magnitude of a symmetric operator.

    Runs ``n_iter`` normalised power iterations ``x ← M x / ‖M x‖`` from
    ``x0`` and returns the last ``‖M x‖`` (the Rayleigh quotient magnitude of
    the final unit vector). For a symmetric ``M`` this is ``max |λ(M)| =
    ‖M‖₂`` in the limit and a **lower bound** on it at every finite
    ``n_iter`` (``‖M x‖ ≤ ‖M‖ ‖x‖``). Dead coordinates (zero rows and
    columns) are handled naturally; an identically zero operator returns
    ``0``.

    Parameters
    ----------
    matvec
        Symmetric linear operator ``x ↦ M x``.
    x0
        Non-zero start vector; a vector of ones works well for the
        diagonally dominant operators met in interior-point methods.
    n_iter
        Number of power iterations (static).

    Returns
    -------
    Scalar
        Estimate of ``‖M‖₂`` from below.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.linalg import power_iteration_norm
    >>> M = jnp.diag(jnp.array([1.0, -3.0, 2.0]))
    >>> float(power_iteration_norm(lambda v: M @ v, jnp.ones(3), n_iter=50))
    3.0
    """
    dtype = x0.dtype
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype)

    def normalise(v: Float[Array, " n"]) -> tuple[Float[Array, " n"], Scalar]:
        norm = jnp.linalg.norm(v)
        return jnp.where(norm > tiny, v / jnp.maximum(norm, tiny), v), norm

    def body(_, carry):
        x, _ = carry
        mx = matvec(x)
        return normalise(mx)

    x0_unit, _ = normalise(x0)
    _, estimate = jax.lax.fori_loop(0, n_iter, body, (x0_unit, jnp.asarray(0.0, dtype)))
    return estimate


def spectral_norm_estimate(
    matvec: Callable[[Float[Array, " n"]], Float[Array, " m"]],
    rmatvec: Callable[[Float[Array, " m"]], Float[Array, " n"]],
    x0: Float[Array, " n"],
    *,
    n_iter: int = 10,
) -> Scalar:
    """Estimate ``‖A‖₂`` of a rectangular operator from ``AᵀA``.

    Power iteration on the symmetric positive semidefinite ``AᵀA`` from
    ``x0`` gives a lower bound on ``‖A‖₂² = λ_max(AᵀA)``; the square root is
    returned.

    Parameters
    ----------
    matvec
        ``x ↦ A x``.
    rmatvec
        ``y ↦ Aᵀ y``.
    x0
        Non-zero start vector in the domain of ``A``.
    n_iter
        Number of power iterations (static).

    Returns
    -------
    Scalar
        Estimate of ``‖A‖₂`` from below.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.linalg import spectral_norm_estimate
    >>> A = jnp.array([[3.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    >>> est = spectral_norm_estimate(lambda v: A @ v, lambda y: A.T @ y, jnp.ones(2), n_iter=30)
    >>> round(float(est), 6)
    3.0
    """
    sq = power_iteration_norm(lambda v: rmatvec(matvec(v)), x0, n_iter=n_iter)
    return jnp.sqrt(sq)
