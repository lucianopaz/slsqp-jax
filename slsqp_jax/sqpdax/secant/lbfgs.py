"""Diagonal-initial-Hessian L-BFGS secant with VARCHEN damping."""

from typing import ClassVar, Self

import jax
from equinox import field, tree_at
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from ..types import Scalar, Vector_n
from .base import CurvatureDiagnostics, Secant

__all__ = [
    "LBFGS",
]


class LBFGS(Secant):
    """Limited-memory BFGS with diagonal ``B₀`` and VARCHEN-style damping.

    Stores the last ``memory`` curvature pairs ``(s, y)`` in a circular
    buffer and applies the compact representation
    (Byrd, Nocedal & Schnabel 1994):

    ```text
    B = B₀ - [B₀ S, Y] M⁻¹ [Sᵀ B₀; Yᵀ]
    ```

    with ``B₀ = diag(diagonal)``. Damping is computed against ``B₀``
    (Lotfi et al., 2020), not the full ``B``, so it stays ``O(n)`` and
    well-conditioned.

    ``inv_eig_lower`` / ``inv_eig_upper`` bound the eigenvalues of the
    *inverse* Hessian ``H₀ = diag(1 / diagonal)``:

    ```text
    inv_eig_lower = 1 / max(diagonal)
    inv_eig_upper = 1 / min(diagonal)
    ```

    Their ratio equals ``κ(B₀) = max(diagonal) / min(diagonal)`` and
    drives soft-reset triggers in the outer solver.

    Attributes
    ----------
    kind
        Registry name ``"lbfgs"``.
    n
        Decision-variable dimension.
    memory
        Maximum number of stored ``(s, y)`` pairs.
    skip_threshold
        Norm floor used by :meth:`should_skip`.
    regularization
        Optional ridge on the compact-form ``M``. Defaults to ``0``
        (exact solve); nonzero only when explicitly requested.
    damping_threshold
        Powell / VARCHEN damping threshold (default ``0.2``).
    diag_floor, diag_ceil
        Absolute clip bracket for the per-variable diagonal.
    s_history, y_history
        Circular buffers of step and (damped) gradient-difference vectors.
    diagonal
        Per-variable initial Hessian scaling ``B₀ = diag(d)``.
    count
        Number of valid pairs currently stored (``0 … memory``).
    next_idx
        Next write index in the circular buffer.
    inv_eig_lower, inv_eig_upper
        Estimated eigenvalue bounds of ``H₀``.
    """

    kind: ClassVar[str] = "lbfgs"

    n: int = field(static=True)
    memory: int = field(static=True)
    skip_threshold: float = field(static=True)
    regularization: float = field(static=True)
    damping_threshold: float = field(static=True)
    diag_floor: float = field(static=True)
    diag_ceil: float = field(static=True)
    s_history: Float[Array, "memory n"]
    y_history: Float[Array, "memory n"]
    diagonal: Float[Array, " n"]
    count: Int[Array, ""]
    next_idx: Int[Array, ""]
    inv_eig_lower: Scalar
    inv_eig_upper: Scalar

    def __init__(
        self,
        n: int,
        memory: int,
        skip_threshold: float = 1e-8,
        # Opt-in only, and 0 (off) on purpose. This deliberately deviates from the
        # core package policy: 0 keeps the compact-form solve exact. Do NOT change
        # the default to nonzero to "match the core"; see the backstop note at the
        # ``M`` assembly in ``hvp``.
        regularization: float = 0.0,
        damping_threshold: float = 0.2,
        diag_floor: float = 1e-4,
        diag_ceil: float = 1e6,
    ):
        """Construct an empty L-BFGS history with identity ``B₀``.

        Parameters
        ----------
        n
            Dimension of the parameter space.
        memory
            Maximum number of ``(s, y)`` pairs (typically 5–20).
        skip_threshold
            Minimum ``‖s‖`` / ``‖y‖`` accepted by :meth:`should_skip`.
        regularization
            Ridge added to the ``2k × 2k`` compact matrix ``M`` inside
            :meth:`hvp`. Default ``0`` keeps the solve exact.
        damping_threshold
            VARCHEN / Powell damping threshold.
        diag_floor, diag_ceil
            Absolute clip limits for the per-variable diagonal.
        """
        self.n = n
        self.memory = memory
        self.skip_threshold = skip_threshold
        self.regularization = regularization
        self.damping_threshold = damping_threshold
        self.diag_floor = diag_floor
        self.diag_ceil = diag_ceil
        self.s_history = jnp.zeros((memory, n))
        self.y_history = jnp.zeros((memory, n))
        self.diagonal = jnp.ones(n)
        self.count = jnp.array(0)
        self.next_idx = jnp.array(0)
        self.inv_eig_lower = jnp.array(1.0)
        self.inv_eig_upper = jnp.array(1.0)

    def diagnostics(self, s: Vector_n, y: Vector_n) -> CurvatureDiagnostics:
        """Return ``(sᵀ y, relative curvature, skipped)`` for ``(s, y)``.

        Parameters
        ----------
        s
            Step vector ``x₊ - x``.
        y
            Gradient difference.

        Returns
        -------
        CurvatureDiagnostics
            Raw inner product, normalised curvature, and skip flag from
            :meth:`should_skip`.
        """
        s_norm = jnp.linalg.norm(s)
        y_norm = jnp.linalg.norm(y)
        sty = jnp.dot(s, y)
        relative_curvature = jnp.abs(sty) / jnp.maximum(s_norm * y_norm, 1e-30)
        skipped = self.should_skip(s, y)
        return CurvatureDiagnostics(  # ty: ignore[invalid-return-type]
            sty, relative_curvature, skipped
        )

    def should_skip(self, s: Vector_n, y: Vector_n) -> Bool[Array, ""]:
        """Return whether ``append`` would reject the pair ``(s, y)``.

        Skips when the step or gradient difference is below
        :attr:`skip_threshold`, the curvature ratio ``‖y‖ / ‖s‖`` is
        extreme, relative curvature is below machine-noise for a
        positive ``sᵀ y``, or any of the norms / inner product is
        non-finite.

        Parameters
        ----------
        s
            Step vector.
        y
            Gradient difference.

        Returns
        -------
        Bool[Array, ""]
            ``True`` if the pair should be skipped.
        """
        s_norm = jnp.linalg.norm(s)
        y_norm = jnp.linalg.norm(y)

        step_too_small = s_norm < self.skip_threshold
        grad_diff_too_small = y_norm < self.skip_threshold

        curvature_ratio = y_norm / jnp.maximum(s_norm, 1e-30)
        curvature_too_extreme = (curvature_ratio > 1e8) | (curvature_ratio < 1e-8)

        sTy_raw = jnp.dot(s, y)
        relative_curvature = jnp.abs(sTy_raw) / jnp.maximum(s_norm * y_norm, 1e-30)
        # Floor lowered from ``1e-8`` to ``1e-12`` (machine-precision floor
        # for ``float64``).  After an identity reset on a near-KKT iterate
        # ``||s|| ~ rtol`` and ``||y|| = ||B s|| = ||s||`` (since ``B = I``),
        # so ``s.y / (||s|| ||y||)`` is dominated by floating-point
        # cancellation in the dot product and can dip below ``1e-8`` purely
        # from rounding even when the pair carries genuine curvature
        # information.  ``1e-12`` is the smallest threshold where the pair
        # is truly *noise* rather than under-resolved curvature.
        curvature_too_small = (relative_curvature < 1e-12) & (sTy_raw > 0)

        has_bad_values = ~(
            jnp.isfinite(s_norm) & jnp.isfinite(y_norm) & jnp.isfinite(sTy_raw)
        )

        return (
            step_too_small
            | grad_diff_too_small
            | curvature_too_extreme
            | curvature_too_small
            | has_bad_values
        )

    def hvp(self, v: Vector_n) -> Vector_n:
        """Compute ``B v`` via the L-BFGS compact representation.

        Uses diagonal initial Hessian ``B₀ = diag(d)``
        (Byrd, Nocedal & Schnabel, 1994, Theorem 2.2):

        ```text
        B = B₀ - [B₀ S, Y] M⁻¹ [Sᵀ B₀; Yᵀ]
        ```

        with

        ```text
        M = [[Sᵀ B₀ S, L], [Lᵀ, -D_sy]]   (2k × 2k)
        ```

        When ``count == 0`` this reduces to ``B v = d ⊙ v``.

        Complexity: ``O(k² n)`` where ``k`` is the memory size.

        Parameters
        ----------
        v
            Vector to multiply by the Hessian approximation.

        Returns
        -------
        Vector_n
            ``B v``. Falls back to ``d ⊙ v`` if the compact solve is
            non-finite or unreasonably large.
        """
        k = self.s_history.shape[0]
        d = self.diagonal
        count = self.count

        # Reorder to chronological order from the circular buffer
        start = (self.next_idx - count + k) % k
        indices = (start + jnp.arange(k)) % k
        S = self.s_history[indices]  # (k, n)
        Y = self.y_history[indices]  # (k, n)

        # Zero out invalid entries (positions >= count are not valid)
        valid_mask = (jnp.arange(k) < count)[:, None]  # (k, 1)
        S = S * valid_mask
        Y = Y * valid_mask

        # Compute the diagonal part of the Hessian approximation
        # DS[i, :] = d * s_i  (B_0 applied row-wise)
        DS = S * d[None, :]  # (k, n)

        # Build compact form inner matrices
        SY = S @ Y.T  # (k, k)
        SDSS = DS @ S.T  # (k, k): S^T B_0 S

        L = jnp.tril(SY, k=-1)
        D_diag = jnp.diag(SY)

        invalid_diag = jnp.where(jnp.arange(k) < count, 0.0, 1.0)

        top_left = SDSS + jnp.diag(invalid_diag)
        top_right = L
        bottom_left = L.T
        bottom_right = -jnp.diag(D_diag) + jnp.diag(invalid_diag)

        top = jnp.concatenate([top_left, top_right], axis=1)
        bottom = jnp.concatenate([bottom_left, bottom_right], axis=1)
        M = jnp.concatenate([top, bottom], axis=0)
        # BACKSTOP (deliberate deviation from the core package policy — do not
        # "fix" this back). Regularization of the 2k×2k compact-form ``M`` is
        # opt-in and defaults to 0. A nonzero ridge here corrupts the L-BFGS
        # arithmetic and hides rank/curvature failure modes instead of surfacing
        # them (cf. the AAᵀ projection, where a Cholesky ridge caused convergence
        # errors on large-meq/large-n problems and was replaced by an exact SVD
        # that attacks the root cause — rank deficiency — directly). Keep this
        # guard: ill-conditioning must be handled at the root, not by jittering M.
        if self.regularization > 0:
            M = M + self.regularization * jnp.eye(2 * k)

        # p = [S^T B_0 v; Y^T v] = [DS @ v; Y @ v]  but DS rows are d*s_i
        # so DS @ v would be wrong shape.  We need S @ (d * v).
        dv = d * v
        p = jnp.concatenate([S @ dv, Y @ v])

        # Least-squares (Moore–Penrose) solve of ``M q = p`` rather than a plain
        # ``solve`` (deliberate deviation from the core package policy — do not
        # "fix" this back). On a rank-deficient compact-form ``M`` this returns the
        # min-norm solution via the SVD rank cut (``eps * 2k``, numpy ``pinv``
        # convention) instead of crashing / emitting NaNs, keeping the arithmetic
        # exact and surfacing — not masking — rank deficiency. Same policy as
        # ``_compute_diagonal``.
        eps = jnp.finfo(M.dtype).eps
        rcond = float(eps * (2 * k))
        q = jnp.linalg.lstsq(M, p, rcond=rcond)[0]

        # B v = B_0 v - [B_0 S, Y]^T @ q = d*v - DS^T @ q[:k] - Y^T @ q[k:]
        result = dv - DS.T @ q[:k] - Y.T @ q[k:]

        # Guard against NaN/inf from numerical blowup in the compact form solve.
        # The magnitude threshold scales with the diagonal so that it is never
        # triggered by legitimate Hessian eigenvalues.  B₀ = diag(d), so the
        # expected scale of ||Bv|| is O(max|d| · ||v||); using a 1000× margin
        # on top of that catches only genuine numerical failures.
        v_norm = jnp.linalg.norm(v)
        result_norm = jnp.linalg.norm(result)
        max_diag = jnp.max(jnp.abs(d))
        threshold = 1000.0 * jnp.maximum(max_diag, 1.0)

        is_bad = jnp.any(~jnp.isfinite(result)) | (
            result_norm > threshold * jnp.maximum(v_norm, 1e-10)
        )

        result = jnp.where(is_bad, dv, result)

        return result

    def inverse_hvp(self, v: Vector_n) -> Vector_n:
        """Compute ``H v = B⁻¹ v`` via the L-BFGS two-loop recursion.

        Implements Nocedal & Wright Algorithm 7.4 with diagonal initial
        scaling ``H₀ = diag(1 / diagonal)``.

        Complexity: ``O(k n)``.

        Parameters
        ----------
        v
            Vector to multiply by the inverse Hessian approximation.

        Returns
        -------
        Vector_n
            ``H v``. Falls back to ``v / d`` if the two-loop result is
            non-finite or unreasonably large.
        """
        k = self.s_history.shape[0]
        d = self.diagonal
        count = self.count

        start = (self.next_idx - count + k) % k
        indices = (start + jnp.arange(k)) % k
        S = self.s_history[indices]  # (k, n) chronological
        Y = self.y_history[indices]  # (k, n) chronological

        valid_mask = (jnp.arange(k) < count)[:, None]
        S = S * valid_mask
        Y = Y * valid_mask

        sTy = jnp.sum(S * Y, axis=1)  # (k,)
        pair_ok = (jnp.arange(k) < count) & (sTy > 1e-12)
        rho = jnp.where(pair_ok, 1.0 / sTy, 0.0)

        # Backward loop: q = v; for i = k-1,...,0: alpha_i = rho_i s_i^T q; q -= alpha_i y_i
        alphas_init = jnp.zeros(k)

        def backward_step(carry, idx):
            q, alphas = carry
            rev_idx = k - 1 - idx
            s_i = S[rev_idx]
            y_i = Y[rev_idx]
            rho_i = rho[rev_idx]
            is_valid = rev_idx < count
            alpha_i = jnp.where(is_valid, rho_i * jnp.dot(s_i, q), 0.0)
            q = q - alpha_i * y_i
            alphas = alphas.at[rev_idx].set(alpha_i)
            return (q, alphas), None

        (q, alphas), _ = jax.lax.scan(backward_step, (v, alphas_init), jnp.arange(k))

        # Apply initial inverse Hessian: H_0 = diag(1/d)
        d_safe = jnp.maximum(d, 1e-30)
        r = q / d_safe

        # Forward loop: for i = 0,...,k-1: beta = rho_i y_i^T r; r += s_i (alpha_i - beta)
        def forward_step(r, idx):
            s_i = S[idx]
            y_i = Y[idx]
            rho_i = rho[idx]
            alpha_i = alphas[idx]
            is_valid = idx < count
            beta = jnp.where(is_valid, rho_i * jnp.dot(y_i, r), 0.0)
            r = r + s_i * (alpha_i - beta)
            return r, None

        r, _ = jax.lax.scan(forward_step, r, jnp.arange(k))

        v_norm = jnp.linalg.norm(v)
        r_norm = jnp.linalg.norm(r)

        # Fall back to the diagonal initial inverse Hessian H₀ = diag(1/d)
        # if the two-loop result is non-finite or unreasonably large.
        # Expected scale of ||H v|| is O(max(1/d) · ||v||).
        max_inv_diag = 1.0 / jnp.min(d_safe)
        threshold = 1000.0 * jnp.maximum(max_inv_diag, 1.0)
        is_bad = jnp.any(~jnp.isfinite(r)) | (
            r_norm > threshold * jnp.maximum(v_norm, 1e-10)
        )
        r = jnp.where(is_bad, v / d_safe, r)

        return r

    def append(self, s: Vector_n, y: Vector_n) -> Self:
        """Append a curvature pair with VARCHEN damping toward ``B₀``.

        Damping is computed against ``B₀ = diag(diagonal)``
        (Lotfi et al., 2020, eqs. 7–8):

        ```text
        y_damped = θ y + (1 - θ) B₀ s
        ```

        with ``θ ∈ [0, 1]`` chosen so
        ``sᵀ y_damped ≥ damping_threshold · sᵀ B₀ s``.

        The per-variable diagonal is then updated with a component-wise
        secant estimate clipped relative to the scalar curvature scale
        ``yᵀy / (yᵀs)`` and the absolute ``[diag_floor, diag_ceil]``
        bracket. Pairs rejected by :meth:`should_skip` leave ``self``
        unchanged.

        Parameters
        ----------
        s
            Step ``x₊ - x``.
        y
            Gradient difference ``∇L₊ - ∇L``.

        Returns
        -------
        Self
            Updated history, or ``self`` if the pair was skipped.
        """
        should_skip = self.should_skip(s, y)

        def do_append() -> Self:
            # VARCHEN-style damping toward B0 = diag(diagonal).
            # O(n) and always well-conditioned, unlike the full lbfgs_hvp.
            B0s = self.diagonal * s
            sTB0s = jnp.dot(s, B0s)
            sTy = jnp.dot(s, y)
            sTB0s_safe = jnp.maximum(sTB0s, 1e-12)

            use_damping = sTy < self.damping_threshold * sTB0s_safe
            theta = jax.lax.cond(
                use_damping,
                lambda: (1.0 - self.damping_threshold)
                * sTB0s_safe
                / (sTB0s_safe - sTy + 1e-12),
                lambda: jnp.array(1.0),
            )
            theta = jnp.clip(theta, 0.0, 1.0)
            y_damped = theta * y + (1.0 - theta) * B0s

            yTy = jnp.dot(y_damped, y_damped)
            yTs = jnp.dot(y_damped, s)
            # Scalar curvature scale (Byrd, Nocedal & Schnabel 1994, eq 3.6),
            # used only to set the *relative* clip bracket for the per-variable
            # diagonal below.  It is NOT persisted: this variant uses the full
            # ``diagonal`` (not a scalar gamma) as B_0.  Falls back to the
            # current diagonal's median when the curvature is non-positive /
            # non-finite.
            curvature_candidate = yTy / jnp.maximum(yTs, 1e-12)
            curvature_scale = jax.lax.cond(
                (yTs > 1e-12) & jnp.isfinite(curvature_candidate),
                lambda: jnp.clip(curvature_candidate, self.diag_floor, self.diag_ceil),
                lambda: jnp.median(self.diagonal),
            )

            s_sq = s**2
            per_var_estimate = jnp.abs(y_damped * s) / jnp.maximum(s_sq, 1e-12)
            clip_lo = jnp.maximum(curvature_scale * 1e-2, self.diag_floor)
            clip_hi = jnp.minimum(curvature_scale * 1e2, self.diag_ceil)
            per_var_clipped = jnp.clip(per_var_estimate, clip_lo, clip_hi)
            has_signal = s_sq > 1e-20
            new_diagonal = jnp.where(
                has_signal & jnp.isfinite(per_var_estimate),
                per_var_clipped,
                self.diagonal,
            )

            k = self.s_history.shape[0]
            idx = self.next_idx
            new_s_history = self.s_history.at[idx].set(s)
            new_y_history = self.y_history.at[idx].set(y_damped)
            new_count = jnp.minimum(self.count + 1, jnp.array(k))
            new_idx = (idx + 1) % k

            history_fields = (
                "s_history",
                "y_history",
                "diagonal",
                "count",
                "next_idx",
            )
            tmp = tree_at(
                lambda x: tuple(getattr(x, field) for field in history_fields),
                self,
                (
                    new_s_history,
                    new_y_history,
                    new_diagonal,
                    new_count,
                    new_idx,
                ),
            )
            inv_eig_lo, inv_eig_hi = tmp.estimate_condition()
            return tree_at(
                lambda x: (x.inv_eig_lower, x.inv_eig_upper),
                tmp,
                (inv_eig_lo, inv_eig_hi),
            )

        return jax.lax.cond(should_skip, lambda: self, do_append)

    def estimate_condition(self) -> tuple[Scalar, Scalar]:
        """Eigenvalue bounds of the inverse Hessian ``H₀ = diag(1 / diagonal)``.

        Returns ``(1 / max(diagonal), 1 / min(diagonal))``; the ratio is
        ``κ(B₀) = max(diagonal) / min(diagonal)``.

        Returns
        -------
        tuple[Scalar, Scalar]
            ``(inv_eig_lower, inv_eig_upper)``.
        """
        d = self.diagonal
        d_safe = jnp.maximum(d, 1e-30)
        inv_eig_min = jnp.min(1.0 / d_safe)
        inv_eig_max = jnp.max(1.0 / d_safe)
        return inv_eig_min, inv_eig_max

    def _soft_reset(self) -> Self:
        """VARCHEN soft reset: keep only the most recent ``(s, y)`` pair.

        Less aggressive than :meth:`_reset` (diagonal extraction) or
        :meth:`_identity_reset`. Based on VARCHEN Algorithm 1, Step 7
        (Lotfi et al., 2020).

        Returns
        -------
        Self
            History with at most one retained pair and refreshed
            ``inv_eig_*`` bounds.
        """
        k, n = self.s_history.shape

        newest_idx = (self.next_idx - 1 + k) % k
        newest_s = self.s_history[newest_idx]
        newest_y = self.y_history[newest_idx]

        new_s_history = jnp.zeros((k, n)).at[0].set(newest_s)
        new_y_history = jnp.zeros((k, n)).at[0].set(newest_y)

        has_pairs = self.count > 0
        new_count = jnp.where(has_pairs, jnp.array(1), jnp.array(0))

        d_safe = jnp.maximum(self.diagonal, 1e-30)
        inv_eig_lo = jnp.min(1.0 / d_safe)
        inv_eig_hi = jnp.max(1.0 / d_safe)
        affected_fields = (
            "s_history",
            "y_history",
            "diagonal",
            "count",
            "next_idx",
            "inv_eig_lower",
            "inv_eig_upper",
        )
        return tree_at(
            lambda x: tuple(getattr(x, field) for field in affected_fields),
            self,
            (
                new_s_history,
                new_y_history,
                self.diagonal,
                new_count,
                jnp.where(has_pairs, jnp.array(1), jnp.array(0)),
                inv_eig_lo,
                inv_eig_hi,
            ),
        )

    def _reset(self) -> Self:
        """SNOPT-style diagonal reset: extract ``diag(Bₖ)``, discard pairs.

        Restarts with ``B₀ = diag(clip(diag(Bₖ)))``, preserving
        per-variable curvature (Gill, Murray & Saunders, 2005, §3.3).

        Returns
        -------
        Self
            Empty history whose diagonal is the clipped compact-form
            diagonal of the previous approximation.
        """
        diag_B = self._compute_diagonal()
        diag_clipped = jnp.clip(diag_B, self.diag_floor, self.diag_ceil)

        # Ensure all values are finite; fall back to 1.0 otherwise
        diag_safe = jnp.where(jnp.isfinite(diag_clipped), diag_clipped, 1.0)

        d_safe = jnp.maximum(diag_safe, 1e-30)
        inv_eig_lo = jnp.min(1.0 / d_safe)
        inv_eig_hi = jnp.max(1.0 / d_safe)

        k, n = self.s_history.shape
        affected_fields = (
            "s_history",
            "y_history",
            "diagonal",
            "count",
            "next_idx",
            "inv_eig_lower",
            "inv_eig_upper",
        )
        return tree_at(
            lambda x: tuple(getattr(x, field) for field in affected_fields),
            self,
            (
                jnp.zeros((k, n)),
                jnp.zeros((k, n)),
                diag_safe,
                jnp.array(0),
                jnp.array(0),
                inv_eig_lo,
                inv_eig_hi,
            ),
        )

    def _identity_reset(self) -> Self:
        """Hard reset to the identity Hessian ``B₀ = I``.

        Unlike :meth:`_reset`, discards all per-variable curvature.
        Used as an escalation when repeated diagonal resets fail to
        break an ill-conditioning cycle.

        Returns
        -------
        Self
            Fresh :class:`LBFGS` with the same static configuration.
        """
        return LBFGS(  # ty: ignore[invalid-return-type]
            n=self.n,
            memory=self.memory,
            skip_threshold=self.skip_threshold,
            regularization=self.regularization,
            damping_threshold=self.damping_threshold,
            diag_floor=self.diag_floor,
            diag_ceil=self.diag_ceil,
        )

    def _compute_diagonal(self) -> Float[Array, " n"]:
        """Extract ``diag(Bₖ)`` from the L-BFGS compact representation.

        From ``B = B₀ - W M⁺ Wᵀ``,

        ```text
        diag(Bₖ) = diagonal - diag(W M⁺ Wᵀ)
        ```

        where ``W = [B₀ S, Y]`` is ``(n, 2k)``. The correction is formed
        in ``O(k² n)`` by a least-squares solve (no ridge on ``M``).

        Returns
        -------
        Float[Array, " n"]
            Diagonal of the current compact-form Hessian approximation.
        """
        k = self.s_history.shape[0]
        d = self.diagonal
        count = self.count

        start = (self.next_idx - count + k) % k
        indices = (start + jnp.arange(k)) % k
        S = self.s_history[indices]
        Y = self.y_history[indices]

        valid_mask = (jnp.arange(k) < count)[:, None]
        S = S * valid_mask
        Y = Y * valid_mask

        DS = S * d[None, :]  # (k, n): B_0 applied row-wise

        SY = S @ Y.T
        SDSS = DS @ S.T

        L_mat = jnp.tril(SY, k=-1)
        D_diag = jnp.diag(SY)
        invalid_diag = jnp.where(jnp.arange(k) < count, 0.0, 1.0)

        top_left = SDSS + jnp.diag(invalid_diag)
        top_right = L_mat
        bottom_left = L_mat.T
        bottom_right = -jnp.diag(D_diag) + jnp.diag(invalid_diag)

        top = jnp.concatenate([top_left, top_right], axis=1)
        bottom = jnp.concatenate([bottom_left, bottom_right], axis=1)
        M = jnp.concatenate([top, bottom], axis=0)
        # No regularization ridge here on purpose (deliberate deviation from the
        # core package policy — do not "fix" this back). Rather than forming an
        # explicit inverse of the possibly rank-deficient 2k×2k compact-form ``M``
        # (which is why the old ``+ 1e-10 I`` jitter existed), solve the system in
        # the least-squares / Moore–Penrose sense so rank deficiency is handled
        # exactly by the SVD rank cut instead of being masked by a spurious ridge.
        #
        # We need ``diag(W M⁺ Wᵀ)``.  With ``Q = W M⁺`` this is the row-wise sum
        # of ``Q * W``.  ``Q`` solves ``Q M = W``; transposing (``M`` symmetric)
        # gives ``M Qᵀ = Wᵀ`` — a min-norm least-squares solve, no inverse.
        W = jnp.concatenate([DS.T, Y.T], axis=1)  # (n, 2k)
        Wt = W.T  # (2k, n)

        # Rank cut at the numpy ``pinv`` convention ``eps * max(M.shape)`` (same
        # policy as the AAᵀ projection solver); ``M`` is (2k, 2k) so it is
        # ``eps * 2k``.  Invalid buffer slots carry a unit diagonal from
        # ``invalid_diag`` and zero ``W`` columns, so they contribute nothing.
        eps = jnp.finfo(M.dtype).eps
        rcond = float(eps * (2 * k))
        Qt = jnp.linalg.lstsq(M, Wt, rcond=rcond)[0]  # (2k, n): M⁺ Wᵀ

        # diag(W M⁺ Wᵀ) = column-wise sum of (Qt * Wt) (== row-wise sum of Q * W).
        diag_correction = jnp.sum(Qt * Wt, axis=0)  # (n,)

        return d - diag_correction

    def reset(self, severity: int = 0) -> Self:
        """Escalating reset: ``0`` soft, ``1`` SNOPT diagonal, ``2+`` identity.

        Parameters
        ----------
        severity
            Reset strength. ``≤ 0`` keeps the newest pair
            (:meth:`_soft_reset`); ``1`` extracts ``diag(Bₖ)`` and clears
            pairs (:meth:`_reset`); ``≥ 2`` restarts at ``B₀ = I``
            (:meth:`_identity_reset`).

        Returns
        -------
        Self
            Reset approximation.
        """
        return jax.lax.cond(
            severity <= 0,
            lambda _: self._soft_reset(),
            lambda sev: jax.lax.cond(
                sev <= 1,
                lambda _: self._reset(),
                lambda _: self._identity_reset(),
                sev,
            ),
            severity,
        )
