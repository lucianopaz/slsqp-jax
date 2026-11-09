"""Limited-memory reduced-Hessian (LRH) secant with Powell damping.

The model is the reduced-Hessian quasi-Newton approximation of Gill & Leonard
(2003, *SIAM J. Optim.* 14),

```text
B = Z RᵀR Zᵀ + σ (I - Z Zᵀ)
```

with ``Z`` an ``n × m`` basis (orthonormal first ``r`` columns, zero padding),
``R`` an ``m × m`` upper-triangular Cholesky factor of the reduced Hessian
and ``σ > 0`` the curvature on the complement of ``range(Z)``. The basis is
built from the *steps* ``s`` only; the curvature vector ``y`` is never stored,
it only drives the BFGS update of ``R`` and the reinitialisation of ``σ``.
Storage is ``n × m`` plus three ``m × m`` blocks, and no ``n × n`` object is
ever formed.
"""

import math
from typing import ClassVar, NamedTuple, Self, cast

import jax
from equinox import field, tree_at
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from ..linalg.reduced_hessian import (
    bfgs_factor_update,
    drop_oldest_basis_vector,
    orthogonalize_against,
)
from ..types import Scalar, Vector_n
from .base import CurvatureDiagnostics
from .reduced_hessian import ReducedHessianFactors, ReducedHessianSecant

__all__ = [
    "LimitedMemoryReducedHessian",
]

SIGMA_RULES = ("yy_over_sy", "sy_over_ss")


class _Analysis(NamedTuple):
    """Quantities shared by :meth:`LimitedMemoryReducedHessian.append` and
    :meth:`LimitedMemoryReducedHessian.diagnostics` (steps 2-4b of the update)."""

    u: Float[Array, " m"]  # Zᵀ s (both Gram-Schmidt passes)
    w: Vector_n  # component of s orthogonal to range(Z)
    rho: Scalar  # ‖w‖
    accepted: Bool[Array, ""]  # the new direction enters the basis
    s_tilde: Vector_n  # projected step Z' s_r
    theta: Scalar  # Powell weight toward B s̃
    late_skip: Bool[Array, ""]  # step-4b curvature guard fires


class LimitedMemoryReducedHessian(ReducedHessianSecant):
    """Limited-memory reduced-Hessian secant (L-RHR, Gill & Leonard 2003).

    The model is ``B = σ I + Z (RᵀR - σ I) Zᵀ``; products with ``B`` and
    ``B⁻¹`` cost ``O(n r)``. Each :meth:`append` performs, for the pair
    ``(s, y)``:

    1. a skip test on the norms and ratio of ``s`` and ``y``;
    2. classical Gram–Schmidt (twice) of ``s`` against ``Z``; the new
       direction enters the basis iff ``ρ ≥ accept_tol ‖s‖``;
    3. Powell damping toward the model's own ``B s̃`` (``s̃`` the projected
       step), threshold ``damping_threshold``, in ``O(r²)`` for ``s̃ᵀB s̃``;
    4. a curvature guard that skips the pair when the damped curvature
       ``s̃ᵀ y_d`` is not above half the damping threshold times ``s̃ᵀB s̃``
       (only reachable when ``s̃ᵀB s̃ < 1e-12``, i.e. when the floor of the
       damping formula is active, e.g. for tiny steps near convergence);
    5. reinitialisation of the complement curvature,
       ``σ = clip(s̃ᵀ y_d / s̃ᵀ s̃, σ_floor, σ_ceil)`` (the default,
       ``sigma_rule="sy_over_ss"``: the Rayleigh quotient along the step) or
       ``σ = clip(y_dᵀ y_d / s̃ᵀ y_d, …)`` (the paper's rule of Gill &
       Leonard, ``sigma_rule="yy_over_sy"``, never smaller than the first),
       keeping the old ``σ`` when the value is non-finite or
       ``s̃ᵀ y_d ≤ 0``, and expansion of ``R`` with ``√σ``. The paper's rule
       estimates the *largest* curvature of the pair, which then acts on
       every soft direction outside ``range(Z)``; in a narrow valley with
       steps leaving the basis it stiffened the soft curvature ``tᵀB t`` by
       up to four orders of magnitude (3-10x stiffer than L-BFGS at equal
       memory), which is why the Rayleigh quotient is the default;
    6. a BFGS update of ``R`` by a small dense QR;
    7. removal of the oldest step when the expanded rank is ``memory + 1``.

    All shapes are fixed (``memory + 1`` working slots), so the update is
    jit-able and can be used inside ``lax.scan``.

    Attributes
    ----------
    kind
        Registry name ``"lrh"``.
    n
        Decision-variable dimension.
    memory
        LRH memory ``m``: maximum number of basis directions.
    skip_threshold
        Norm floor used by :meth:`should_skip`.
    accept_tol
        A new direction enters the basis iff ``ρ ≥ accept_tol ‖s‖``.
    damping_threshold
        Powell threshold ``τ`` (default ``0.2``).
    sigma_floor, sigma_ceil
        Clip bracket for the complement curvature ``σ``.
    sigma_rule
        Reinitialisation rule for ``σ``: ``"sy_over_ss"`` (default,
        ``s̃ᵀy_d / s̃ᵀs̃``) or ``"yy_over_sy"`` (the paper's
        ``y_dᵀy_d / s̃ᵀy_d``); see the class description of step 5.
    initial_sigma
        Complement curvature of the empty model and the value a severity-2
        :meth:`reset` returns to (not clipped).
    basis
        ``Z``, shape ``(n, memory)``; first ``rank`` columns orthonormal,
        the rest exactly zero.
    triangular
        ``T``, shape ``(memory, memory)``, upper triangular, with
        ``Z T`` the newest ``rank`` steps (oldest first); zero padding.
    cholesky
        ``R``, shape ``(memory, memory)``, upper triangular, with the
        padded block exactly the identity.
    sigma
        Complement curvature ``σ``.
    rank
        Current rank ``r`` (``int32``).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.secant import LimitedMemoryReducedHessian
    >>> secant = LimitedMemoryReducedHessian(n=3, memory=2)
    >>> s = jnp.array([1.0, 0.0, 0.0])
    >>> secant = secant.append(s, 4.0 * s)
    >>> int(secant.num_pairs), round(float(secant.hvp(s)[0]), 4)
    (1, 4.0)
    """

    kind: ClassVar[str] = "lrh"

    n: int = field(static=True)
    memory: int = field(static=True)
    skip_threshold: float = field(static=True)
    accept_tol: float = field(static=True)
    damping_threshold: float = field(static=True)
    sigma_floor: float = field(static=True)
    sigma_ceil: float = field(static=True)
    sigma_rule: str = field(static=True)
    initial_sigma: float = field(static=True)
    basis: Float[Array, "n memory"]
    triangular: Float[Array, "memory memory"]
    cholesky: Float[Array, "memory memory"]
    sigma: Scalar
    rank: Int[Array, ""]

    def __init__(
        self,
        n: int,
        memory: int,
        skip_threshold: float = 1e-8,
        accept_tol: float = 1e-4,
        damping_threshold: float = 0.2,
        sigma_floor: float = 1e-4,
        sigma_ceil: float = 1e6,
        sigma_rule: str = "sy_over_ss",
        initial_sigma: float = 1.0,
    ):
        """Construct an empty model ``B = initial_sigma · I``.

        Parameters
        ----------
        n
            Dimension of the parameter space.
        memory
            LRH memory ``m`` (maximum rank of the basis).
        skip_threshold
            Minimum ``‖s‖`` / ``‖y‖`` accepted by :meth:`should_skip`.
        accept_tol
            Relative threshold ``ρ / ‖s‖`` for admitting a new direction.
        damping_threshold
            Powell damping threshold ``τ`` toward the model's own ``B s̃``.
        sigma_floor, sigma_ceil
            Clip limits for the complement curvature ``σ``.
        sigma_rule
            ``"sy_over_ss"`` (default) or ``"yy_over_sy"`` (the paper's
            rule); see the class docstring.
        initial_sigma
            Initial complement curvature ``σ`` (also the severity-2 reset
            value).

        Raises
        ------
        ValueError
            If ``sigma_rule`` is not one of the two rules above, or if
            ``initial_sigma`` is not finite and positive.
        """
        if sigma_rule not in SIGMA_RULES:
            raise ValueError(
                f"unknown sigma_rule {sigma_rule!r}; expected one of {SIGMA_RULES}"
            )
        if not (math.isfinite(initial_sigma) and initial_sigma > 0.0):
            raise ValueError(
                f"initial_sigma must be finite and positive, got {initial_sigma!r}"
            )
        self.n = n
        self.memory = memory
        self.skip_threshold = skip_threshold
        self.accept_tol = accept_tol
        self.damping_threshold = damping_threshold
        self.sigma_floor = sigma_floor
        self.sigma_ceil = sigma_ceil
        self.sigma_rule = sigma_rule
        self.initial_sigma = initial_sigma
        self.basis = jnp.zeros((n, memory))
        self.triangular = jnp.zeros((memory, memory))
        self.cholesky = jnp.eye(memory)
        self.sigma = jnp.full((), initial_sigma, dtype=self.basis.dtype)
        self.rank = jnp.zeros((), dtype=jnp.int32)

    # ------------------------------------------------------------------
    # Structure and products
    # ------------------------------------------------------------------

    def factors(self) -> ReducedHessianFactors:
        """Return the current factors ``(Z, R, σ, r)``.

        Returns
        -------
        ReducedHessianFactors
            Views of the stored arrays; none is of size ``n × n``.
        """
        return cast(
            ReducedHessianFactors,
            ReducedHessianFactors(
                basis=self.basis,
                cholesky=self.cholesky,
                sigma=self.sigma,
                rank=self.rank,
            ),
        )

    def hvp(self, v: Vector_n) -> Vector_n:
        """Apply the model ``B v = σ v + Z (RᵀR - σ I) Zᵀ v`` in ``O(n r)``.

        Parameters
        ----------
        v
            Vector of length ``n``.

        Returns
        -------
        Vector_n
            ``B v``.
        """
        return self.factors().hvp(v)

    def inverse_hvp(self, v: Vector_n) -> Vector_n:
        """Apply the inverse ``B⁻¹ v`` in ``O(n r)``.

        Parameters
        ----------
        v
            Vector of length ``n``.

        Returns
        -------
        Vector_n
            ``B⁻¹ v``.
        """
        return self.factors().inverse_hvp(v)

    @property
    def num_pairs(self) -> Int[Array, ""]:
        """Current rank ``r`` (number of basis directions stored)."""
        return self.rank

    def curvature_bounds(self) -> tuple[Scalar, Scalar]:
        """Bounds of the scaling matrix ``σ I``: ``(σ, σ)``.

        As documented on the ABC these describe the initial / scaling matrix,
        not the spectrum of ``B``; the conditioning reset trigger is
        therefore inert for this secant.

        Returns
        -------
        tuple[Scalar, Scalar]
            ``(σ, σ)``.
        """
        return self.sigma, self.sigma

    def spectrum_bounds(self) -> tuple[Scalar, Scalar]:
        """Exact extreme eigenvalues of ``B`` (diagnostics only).

        The spectrum of ``B`` is that of the active ``r × r`` block
        ``RᵀR`` plus ``σ`` (multiplicity ``n - r``) when ``r < n``. Costs one
        ``m × m`` eigendecomposition, ``O(m³)``; no ``n × n`` object is formed.

        Returns
        -------
        tuple[Scalar, Scalar]
            ``(λ_min, λ_max)`` of ``B``.
        """
        factors = self.factors()
        r_mat = factors.active_cholesky
        gram = r_mat.T @ r_mat
        mask = factors.column_mask
        active = mask[:, None] & mask[None, :]
        # Replace the padded identity by a value above every active
        # eigenvalue (trace bound) so the active ones sort first.
        big = 1.0 + jnp.trace(gram)
        eigs = jnp.linalg.eigvalsh(
            jnp.where(active, gram, jnp.where(jnp.eye(self.memory) > 0, big, 0.0))
        )
        has_reduced = self.rank > 0
        lower = jnp.where(has_reduced, eigs[0], self.sigma)
        upper = jnp.where(has_reduced, eigs[jnp.maximum(self.rank - 1, 0)], self.sigma)
        has_complement = self.rank < self.n
        lower = jnp.where(has_complement, jnp.minimum(lower, self.sigma), lower)
        upper = jnp.where(has_complement, jnp.maximum(upper, self.sigma), upper)
        return lower, upper

    # ------------------------------------------------------------------
    # Skip / damping / diagnostics
    # ------------------------------------------------------------------

    def should_skip(self, s: Vector_n, y: Vector_n) -> Bool[Array, ""]:
        """Step-1 skip predicate of :meth:`append`.

        Skips when ``‖s‖`` or ``‖y‖`` is below :attr:`skip_threshold`, when
        ``‖y‖ / ‖s‖`` lies outside ``[1e-8, 1e8]`` or when a norm or
        ``sᵀy`` is non-finite. (Unlike the L-BFGS predicate there is no
        relative-curvature test: damping makes it moot.) The later
        curvature guard of :meth:`append` is *not* part of this predicate;
        :meth:`diagnostics` reports both.

        Parameters
        ----------
        s
            Step vector.
        y
            Gradient difference.

        Returns
        -------
        Bool[Array, ""]
            ``True`` if the pair is rejected at step 1.
        """
        s_norm = jnp.linalg.norm(s)
        y_norm = jnp.linalg.norm(y)
        curvature_ratio = y_norm / jnp.maximum(s_norm, 1e-30)
        has_bad_values = ~(
            jnp.isfinite(s_norm) & jnp.isfinite(y_norm) & jnp.isfinite(jnp.dot(s, y))
        )
        return (
            (s_norm < self.skip_threshold)
            | (y_norm < self.skip_threshold)
            | (curvature_ratio > 1e8)
            | (curvature_ratio < 1e-8)
            | has_bad_values
        )

    def _damping_theta(self, sbs: Scalar, sty: Scalar) -> Scalar:
        """Powell weight, written exactly as ``LBFGS._damping_theta``.

        With ``τ = damping_threshold`` and ``sᵀB s`` floored at ``1e-12``,
        ``θ = 1`` if ``sᵀy ≥ τ sᵀB s`` and
        ``θ = (1 - τ) sᵀB s / (sᵀB s - sᵀy + 1e-12)`` otherwise, clipped to
        ``[0, 1]``.
        """
        tau = self.damping_threshold
        sbs_safe = jnp.maximum(sbs, 1e-12)
        theta = jnp.where(
            sty < tau * sbs_safe,
            (1.0 - tau) * sbs_safe / (sbs_safe - sty + 1e-12),
            1.0,
        )
        return jnp.clip(theta, 0.0, 1.0)

    def _analyse(self, s: Vector_n, y: Vector_n) -> _Analysis:
        """Steps 2-4b: orthogonalise, damp against the current model, guard.

        Costs ``O(n r)`` (the two Gram-Schmidt passes) plus ``O(r²)``; the
        value of ``s̃ᵀB s̃`` is ``‖R_act u‖² + σ ρ²`` (accepted direction) so
        no ``hvp`` is needed. Values are meaningless (possibly non-finite)
        for a pair that fails the step-1 predicate.
        """
        dtype = self.basis.dtype
        s = jnp.asarray(s, dtype)
        y = jnp.asarray(y, dtype)
        u, w, rho = orthogonalize_against(self.basis, s)
        # ``rho > 0`` is implied by the first test for every pair that passed
        # ``should_skip`` (``accept_tol * ||s|| > 0``); it only guards the
        # division by ``rho`` for pairs whose results are discarded.
        accepted = (rho >= self.accept_tol * jnp.linalg.norm(s)) & (rho > 0.0)
        w_used = jnp.where(accepted, w, jnp.zeros_like(w))
        s_tilde = self.basis @ u + w_used
        ru = self.factors().active_cholesky @ u
        sbs = jnp.dot(ru, ru) + jnp.where(accepted, self.sigma * rho**2, 0.0)
        sty = jnp.dot(s_tilde, y)
        theta = self._damping_theta(sbs, sty)
        # s̃ᵀ y_d for y_d = θ y + (1 - θ) B s̃, without forming y_d.
        curvature = theta * sty + (1.0 - theta) * sbs
        guard_ok = (
            jnp.isfinite(curvature)
            & jnp.isfinite(theta)
            & (curvature > 0.5 * self.damping_threshold * sbs)
        )
        return _Analysis(u, w, rho, accepted, s_tilde, theta, ~guard_ok)

    def diagnostics(self, s: Vector_n, y: Vector_n) -> CurvatureDiagnostics:
        """Return ``(sᵀ y, relative curvature, skipped, θ)`` for ``(s, y)``.

        ``skipped`` is ``True`` iff :meth:`append` would leave the model
        unchanged: the step-1 predicate :meth:`should_skip` or the
        curvature guard of step 4b. ``θ`` is the Powell weight toward
        ``B s̃`` that :meth:`append` applies, and ``1`` for a skipped pair.
        Costs ``O(n r)``.

        Parameters
        ----------
        s
            Step vector ``x₊ - x``.
        y
            Gradient difference.

        Returns
        -------
        CurvatureDiagnostics
            Raw inner product, normalised curvature, skip flag and damping
            weight.
        """
        s_norm = jnp.linalg.norm(s)
        y_norm = jnp.linalg.norm(y)
        sty = jnp.dot(s, y)
        relative_curvature = jnp.abs(sty) / jnp.maximum(s_norm * y_norm, 1e-30)
        analysis = self._analyse(s, y)
        skipped = self.should_skip(s, y) | analysis.late_skip
        theta = jnp.where(skipped, 1.0, analysis.theta)
        return cast(
            CurvatureDiagnostics,
            CurvatureDiagnostics(sty, relative_curvature, skipped, theta),
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def _rebuilt(
        self,
        basis: Float[Array, "n memory"],
        triangular: Float[Array, "memory memory"],
        cholesky: Float[Array, "memory memory"],
        rank: Int[Array, ""],
        sigma: Scalar | None = None,
    ) -> Self:
        """New instance with ``T`` zero- and ``R`` identity-padded beyond ``rank``.

        Enforces the padding invariants (``T`` zero and ``R`` the identity
        outside the leading ``rank × rank`` block) and the stored dtypes.
        ``sigma`` defaults to the current one.
        """
        dtype = self.basis.dtype
        mask = jnp.arange(self.memory) < rank
        active = mask[:, None] & mask[None, :]
        triangular = jnp.where(active, triangular, 0.0)
        cholesky = jnp.where(active, cholesky, jnp.eye(self.memory, dtype=dtype))
        sigma = self.sigma if sigma is None else sigma
        return cast(
            Self,
            tree_at(
                lambda mod: (
                    mod.basis,
                    mod.triangular,
                    mod.cholesky,
                    mod.sigma,
                    mod.rank,
                ),
                self,
                (
                    basis.astype(dtype),
                    triangular.astype(dtype),
                    cholesky.astype(dtype),
                    jnp.asarray(sigma, dtype),
                    rank.astype(jnp.int32),
                ),
            ),
        )

    def _apply(self, s: Vector_n, y: Vector_n, a: _Analysis) -> Self:
        """Steps 5-7 for a pair that passed the skip tests."""
        m = self.memory
        dtype = self.basis.dtype
        s = jnp.asarray(s, dtype)
        y = jnp.asarray(y, dtype)
        r = self.rank
        accepted = a.accepted

        # Damped pair y_d = θ y + (1 - θ) B s̃ (current model, old sigma).
        y_d = a.theta * y + (1.0 - a.theta) * self.hvp(a.s_tilde)
        sty_d = jnp.dot(a.s_tilde, y_d)

        # 5. reinitialisation scalar with fallback to the old one
        numerator = {
            "yy_over_sy": jnp.dot(y_d, y_d),
            "sy_over_ss": sty_d,
        }[self.sigma_rule]
        denominator = {
            "yy_over_sy": sty_d,
            "sy_over_ss": jnp.dot(a.s_tilde, a.s_tilde),
        }[self.sigma_rule]
        raw = numerator / denominator
        sigma_new = jnp.where(
            jnp.isfinite(raw) & (sty_d > 0.0),
            jnp.clip(raw, self.sigma_floor, self.sigma_ceil),
            self.sigma,
        ).astype(dtype)

        # Expanded quantities in m + 1 slots; the new direction goes to slot r.
        w_hat = jnp.where(
            accepted, a.w / jnp.where(accepted, a.rho, 1.0), jnp.zeros_like(a.w)
        )
        basis_x = jnp.concatenate([self.basis, jnp.zeros((self.n, 1), dtype)], axis=1)
        basis_x = basis_x.at[:, r].set(w_hat)

        chol_x = (
            jnp.eye(m + 1, dtype=dtype).at[:m, :m].set(self.factors().active_cholesky)
        )
        chol_x = chol_x.at[r, r].set(
            jnp.where(accepted, jnp.sqrt(sigma_new), 1.0).astype(dtype)
        )

        u_x = jnp.concatenate([a.u, jnp.zeros((1,), dtype)])
        col = u_x.at[r].set(jnp.where(accepted, a.rho, 0.0).astype(dtype))
        tri_x = jnp.zeros((m + 1, m + 1), dtype=dtype).at[:m, :m].set(self.triangular)
        tri_x = jnp.where(accepted, tri_x.at[:, r].set(col), tri_x)

        # 6. BFGS update of the factor in the reduced coordinates
        y_r = basis_x.T @ y_d
        chol_bfgs = bfgs_factor_update(chol_x, col, y_r)

        # 7. drop the oldest step iff the expanded rank is m + 1
        expanded_rank = r + accepted.astype(jnp.int32)

        def drop(args):
            return drop_oldest_basis_vector(*args)

        def truncate(args):
            basis_a, tri_a, chol_a = args
            return basis_a[:, :m], tri_a[:m, :m], chol_a[:m, :m]

        basis_new, tri_new, chol_new = jax.lax.cond(
            expanded_rank == m + 1, drop, truncate, (basis_x, tri_x, chol_bfgs)
        )
        return self._rebuilt(
            basis_new, tri_new, chol_new, jnp.minimum(expanded_rank, m), sigma_new
        )

    def append(self, s: Vector_n, y: Vector_n) -> Self:
        """Ingest ``(s, y)`` with Powell damping (steps 1-7 of the class docs).

        The basis is built from ``s`` only; ``y`` is never stored. A pair that
        fails :meth:`should_skip` or the step-4b curvature guard returns
        ``self`` unchanged.

        Parameters
        ----------
        s
            Step ``x₊ - x``.
        y
            Gradient difference ``∇L₊ - ∇L``.

        Returns
        -------
        Self
            Updated model, or ``self`` if the pair was skipped.
        """

        def do_append() -> Self:
            analysis = self._analyse(s, y)
            return jax.lax.cond(
                analysis.late_skip,
                lambda: self,
                lambda: self._apply(s, y, analysis),
            )

        return jax.lax.cond(self.should_skip(s, y), lambda: self, do_append)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _empty(self, sigma: Scalar) -> Self:
        """Empty basis with complement curvature ``sigma``."""
        return self._rebuilt(
            jnp.zeros_like(self.basis),
            jnp.zeros_like(self.triangular),
            jnp.eye(self.memory, dtype=self.basis.dtype),
            jnp.zeros((), jnp.int32),
            sigma,
        )

    def _soft_reset(self) -> Self:
        """Keep only the newest basis direction with its model curvature."""
        dtype = self.basis.dtype
        has_pairs = self.rank > 0
        idx = jnp.maximum(self.rank - 1, 0)
        t = self.triangular[:, idx]
        t_norm = jnp.linalg.norm(t)
        t_safe = jnp.where(has_pairs, t_norm, 1.0)
        z = self.basis @ t / t_safe
        r_scalar = jnp.linalg.norm(self.factors().active_cholesky @ t) / t_safe
        basis = jnp.zeros_like(self.basis).at[:, 0].set(z)
        triangular = jnp.zeros_like(self.triangular).at[0, 0].set(t_norm)
        cholesky = jnp.eye(self.memory, dtype=dtype).at[0, 0].set(r_scalar)
        kept = self._rebuilt(basis, triangular, cholesky, jnp.ones((), jnp.int32))
        return cast(
            Self,
            jax.tree.map(lambda new, old: jnp.where(has_pairs, new, old), kept, self),
        )

    def reset(self, severity: int = 0) -> Self:
        """Escalating reset: ``0`` newest direction, ``1`` empty, ``2+`` identity.

        ``severity`` may be a traced integer; it is clipped to ``[0, 2]`` and
        dispatched with ``lax.switch``.

        * ``0`` keeps only the newest basis direction ``b = Z T[:, r-1]`` with
          its model curvature (``Z = b / ‖b‖``,
          ``R = [‖R T[:, r-1]‖ / ‖T[:, r-1]‖]``, ``T = [‖b‖]``) and ``σ``;
          a no-op on an empty model.
        * ``1`` drops the whole basis and keeps ``σ``.
        * ``2`` drops the whole basis and sets ``σ = initial_sigma`` (``B = I``
          for the default ``initial_sigma = 1``).

        Parameters
        ----------
        severity
            Reset strength.

        Returns
        -------
        Self
            Reset model.
        """
        index = jnp.clip(jnp.asarray(severity, jnp.int32), 0, 2)
        return jax.lax.switch(
            index,
            [
                lambda: self._soft_reset(),
                lambda: self._empty(self.sigma),
                lambda: self._empty(jnp.full_like(self.sigma, self.initial_sigma)),
            ],
        )
