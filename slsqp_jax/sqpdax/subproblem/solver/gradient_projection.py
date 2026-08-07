from typing import Any, Callable, cast

import jax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Scalar

from ...dual import Dual
from ...primal import Primal
from ..base import SubProblem
from .base import RESULTS, SubProblemSolver, SubProblemSolverState

__all__ = [
    "GradientProjectionState",
    "GradientProjection",
]


class GradientProjectionState(SubProblemSolverState):
    """State returned by :class:`GradientProjection`.

    Attributes
    ----------
    active_lb
        Lower-bound faces active at the accepted step (N&W ``A(x^c)``),
        length ``n``.
    active_ub
        Upper-bound faces active at the accepted step, length ``n``.
    direction
        Accepted primal step (Cauchy point, or a further free-space
        improvement when ``cauchy_only=False``).
    """

    active_lb: Bool[Array, " n"]
    active_ub: Bool[Array, " n"]
    direction: Primal


class GradientProjection(
    SubProblemSolver[Primal, SubProblem[Any], GradientProjectionState]
):
    """Gradient-projection solver for the *bound-constrained* primal QP (N&W §16.7).

    Operates only on the primal model exposed by ``SubProblem``:

    ```
    min_p  gᵀ p + ½ pᵀ H p    s.t.  ℓ ≤ p ≤ u
    ```

    where ``g = subproblem.primal_grad()``, ``H`` is applied through
    ``kkt_mvp_primal``, and the box ``(ℓ, u)`` comes from
    ``subproblem.primal_box()``. Equality / general inequality rows are
    ignored — the dual block of ``x0`` is returned unchanged.

    Bound *geometry* is entirely the subproblem's job:

    * ``primal_box()`` — componentwise bounds in ``flatten()`` order;
    * ``active_bounds(p, tol)`` — which of the ``n`` variable lower/upper
      faces are active at a primal step;
    * ``unflatten_primal(flat)`` — rebuild the primal pytree.

    Interior-point / scaled layouts override those hooks; this class never
    branches on the concrete primal type. The returned ``active_lb`` /
    ``active_ub`` let a subsequent KKT or trust-region solve fix those
    variables.

    Algorithm 16.5:

    1. **Cauchy point** — project the steepest-descent path
       ``p(t) = P(p_0 - t g, ℓ, u)`` onto the box and take the first local
       minimizer of the piecewise-quadratic ``q(p(t))`` (eqs. 16.69–16.73).
    2. **Subspace step** (optional) — with ``cauchy_only=False``, run a short
       free-space CG on the face ``A(x^c)``, truncated at the first new bound.

    Attributes
    ----------
    solver_state_class
        :class:`GradientProjectionState`.
    tol
        Absolute tolerance for breakpoints, free-face detection, and CG.
    cauchy_only
        If ``True`` (default), stop after the Cauchy point.
    max_segments
        Cap on Cauchy piecewise-quadratic segments.
    max_cg_iter
        Cap on free-space CG iterations when ``cauchy_only=False``.
    """

    solver_state_class: type[GradientProjectionState] = GradientProjectionState

    tol: float = 1e-10
    cauchy_only: bool = True
    max_segments: int = 256
    max_cg_iter: int = 20

    def solve(
        self,
        subproblem: SubProblem[Any],
        x0: tuple[Primal, Dual],
        initial_state: GradientProjectionState,
    ) -> tuple[tuple[Primal, Dual], GradientProjectionState]:
        """Compute a bound-constrained Cauchy / subspace step.

        Parameters
        ----------
        subproblem
            Local model exposing ``primal_grad``, ``kkt_mvp_primal``,
            ``primal_box``, ``active_bounds``, and ``unflatten_primal``.
        x0
            Warm-start ``(primal_step, dual)``. The primal is projected into
            the box; the dual is returned unchanged.
        initial_state
            Carry whose ``n_iter`` is accumulated into the returned state.

        Returns
        -------
        step
            ``(direction, initial_dual)``.
        state
            Updated :class:`GradientProjectionState` with active faces.
        """
        initial_primal, initial_dual = x0
        dtype = initial_primal.flatten().dtype
        tol = jnp.asarray(self.tol, dtype)

        lo, hi = subproblem.primal_box()
        # Feasible start (N&W requires a feasible x0; clip warm-starts into the box).
        p0 = self.project(initial_primal.flatten(), lo, hi)

        zero_dual = cast(Dual, jax.tree.map(jnp.zeros_like, initial_dual))

        def H_flat(v: Float[Array, " n_p"]) -> Float[Array, " n_p"]:
            return subproblem.kkt_mvp_primal(
                (subproblem.unflatten_primal(v), zero_dual)
            ).flatten()

        g0 = subproblem.primal_grad().flatten()
        # Gradient of q at the (projected) warm-start: ∇q(p0) = g + H p0.
        g = g0 + H_flat(p0)

        p_c, n_seg = self._cauchy_point(p0, g, H_flat, lo, hi)

        if self.cauchy_only:
            p_plus = p_c
            n_cg = jnp.zeros((), jnp.int32)
        else:
            p_plus, n_cg = self._subspace_cg(p_c, g0, H_flat, lo, hi)

        direction = subproblem.unflatten_primal(p_plus)
        active_lb, active_ub = subproblem.active_bounds(direction, tol)

        finite = jnp.all(jnp.isfinite(p_plus))
        status = RESULTS.where(finite, RESULTS.successful, RESULTS.singular)
        state = cast(
            GradientProjectionState,
            GradientProjectionState(
                n_iter=jnp.asarray(initial_state.n_iter, jnp.int32) + n_seg + n_cg,
                success=finite,
                status=status,
                active_lb=active_lb,
                active_ub=active_ub,
                direction=direction,
            ),
        )
        return (direction, initial_dual), state

    # ------------------------------------------------------ active-bound API --

    def init_state(
        self, subproblem: SubProblem[Any], x0: tuple[Primal, Dual]
    ) -> GradientProjectionState:
        """Zero state seeded to the shapes of ``subproblem``.

        Parameters
        ----------
        subproblem
            Model whose ``lagrangian.n`` / ``ref`` define mask and direction
            shapes.
        x0
            Unused; kept so callers can pass the same warm-start as
            :meth:`solve`.

        Returns
        -------
        GradientProjectionState
            Carry with zero iteration count, inactive masks, and a zero
            direction.
        """
        del x0
        n = subproblem.lagrangian.n
        template = subproblem.lagrangian.ref
        return cast(
            GradientProjectionState,
            GradientProjectionState(
                n_iter=jnp.zeros((), jnp.int32),
                success=jnp.asarray(False),
                status=RESULTS.successful,
                active_lb=jnp.zeros((n,), bool),
                active_ub=jnp.zeros((n,), bool),
                direction=jax.tree.map(jnp.zeros_like, template),
            ),
        )

    def find_active_bounds(
        self, subproblem: SubProblem[Any]
    ) -> tuple[Bool[Array, " n"], Bool[Array, " n"]]:
        """Active variable-bound faces at the Cauchy point of the current iterate.

        Runs the gradient-projection Cauchy step from the *zero* primal step
        (i.e. from the current point ``x_k``) and returns
        ``(active_lb, active_ub)`` — N&W ``A(x^c)``. Dogleg / Steihaug–Toint
        call this to freeze bound variables when their state does not already
        carry an active set.

        Parameters
        ----------
        subproblem
            Local model at the current iterate.

        Returns
        -------
        active_lb, active_ub
            Boolean masks of length ``n``.
        """
        lag = subproblem.lagrangian
        x0 = (
            jax.tree.map(jnp.zeros_like, lag.ref),
            jax.tree.map(jnp.zeros_like, lag.dual),
        )
        (_, _), state = self.solve(subproblem, x0, self.init_state(subproblem, x0))
        return state.active_lb, state.active_ub

    # ------------------------------------------------------------------ box --

    @staticmethod
    def project(
        x: Float[Array, " n_p"],
        lo: Float[Array, " n_p"],
        hi: Float[Array, " n_p"],
    ) -> Float[Array, " n_p"]:
        """Projection onto the box (N&W eq. 16.69).

        Parameters
        ----------
        x
            Unconstrained vector.
        lo, hi
            Componentwise lower / upper bounds (may be infinite).

        Returns
        -------
        jax.Array
            ``clip(x, lo, hi)``.
        """
        return jnp.clip(x, lo, hi)

    @staticmethod
    def _breakpoints(
        x: Float[Array, " n_p"],
        g: Float[Array, " n_p"],
        lo: Float[Array, " n_p"],
        hi: Float[Array, " n_p"],
    ) -> Float[Array, " n_p"]:
        """Per-coordinate hit times ``t̄_i`` along ``x - t g`` (N&W eq. 16.71)."""
        t_ub = jnp.where(
            (g < 0) & jnp.isfinite(hi),
            (x - hi) / g,
            jnp.inf,
        )
        t_lb = jnp.where(
            (g > 0) & jnp.isfinite(lo),
            (x - lo) / g,
            jnp.inf,
        )
        return jnp.minimum(t_ub, t_lb)

    # ---------------------------------------------------------- Cauchy point --

    def _cauchy_point(
        self,
        x0: Float[Array, " n_p"],
        g: Float[Array, " n_p"],
        H: Callable[[Float[Array, " n_p"]], Float[Array, " n_p"]],
        lo: Float[Array, " n_p"],
        hi: Float[Array, " n_p"],
    ) -> tuple[Float[Array, " n_p"], Int[Array, ""]]:
        """First local minimizer of ``q`` along ``P(x0 - t g, ℓ, u)`` (eq. 16.70)."""
        n_p = x0.shape[0]
        dtype = x0.dtype
        tbar = self._breakpoints(x0, g, lo, hi)
        # Sort finite positive breakpoints to the front; invalids become +inf.
        t_valid = jnp.where((tbar > 0) & jnp.isfinite(tbar), tbar, jnp.inf)
        t_sorted = jnp.sort(t_valid)
        n_seg_cap = int(min(self.max_segments, n_p))

        def segment_direction(t_left: Scalar) -> Float[Array, " n_p"]:
            # p^{j-1}: −g on free coords (those with t̄_i > t_left), else 0 (16.72).
            return jnp.where(tbar > t_left, -g, jnp.zeros_like(g))

        def _segment_min(t_prev, x_curr, width):
            # First stationary point of q on [t_prev, t_prev + width] along
            # ``pdir = segment_direction(t_prev)`` (N&W eq. 16.73).  ``width`` may
            # be ``+inf`` for the trailing ray.  Returns ``(found, t_new, x_new)``.
            pdir = segment_direction(t_prev)
            Hp = H(pdir)
            # Directional derivative of q along pdir at x_curr, with
            # g = ∇q(x0):  f' = ⟨g, pdir⟩ + ⟨x_curr - x0, H pdir⟩.
            # Equivalent to N&W's absolute-coordinate form
            # cᵀ pdir + x_currᵀ G pdir after p = x - x_ref.
            f1 = jnp.dot(g, pdir) + jnp.dot(x_curr - x0, Hp)
            f2 = jnp.dot(pdir, Hp)
            # Case (i): ascending → minimizer at the left endpoint.
            stop_left = f1 >= 0
            # Case (ii): interior critical point (guard the division so the
            # unused branch of the ``where`` never evaluates a NaN).
            safe_f2 = jnp.where(f2 > 0, f2, 1.0)
            dt_star = jnp.where(f2 > 0, jnp.clip(-f1 / safe_f2, 0.0, width), width)
            interior = (~stop_left) & (f2 > 0) & (dt_star < width)
            # Case (iii): otherwise advance to the right endpoint t_prev + width.
            t_new = jnp.where(
                stop_left,
                t_prev,
                jnp.where(interior, t_prev + dt_star, t_prev + width),
            )
            x_new = self.project(x0 - t_new * g, lo, hi)
            return stop_left | interior, t_new, x_new

        def body(j, carry):
            done, t_prev, x_curr, n_used = carry
            t_j = t_sorted[j]
            # Skip duplicate / invalid breakpoints; once done, freeze the carry.
            usable = (~done) & jnp.isfinite(t_j) & (t_j > t_prev)

            def examine(_):
                found, _t_new, x_new = _segment_min(t_prev, x_curr, t_j - t_prev)
                # Non-terminal segment: advance the left endpoint to t_j.
                t_next = jnp.where(found, t_prev, t_j)
                x_next = jnp.where(found, x_new, self.project(x0 - t_j * g, lo, hi))
                return found, t_next, x_next, n_used + 1

            def skip(_):
                return done, t_prev, x_curr, n_used

            return jax.lax.cond(usable, examine, skip, operand=None)

        init = (
            jnp.asarray(False),
            jnp.asarray(0.0, dtype),
            x0,
            jnp.zeros((), jnp.int32),
        )
        done, t_prev, x_curr, n_used = jax.lax.fori_loop(0, n_seg_cap, body, init)

        # Trailing ray [t_prev, +∞): the coordinates still free at t_prev keep
        # descending along -g.  Skipping it would stop at the last breakpoint
        # instead of the true Cauchy point — and, when the box is unbounded,
        # return no step at all.  Non-positive curvature with f1 < 0 means q is
        # unbounded below on the ray; ``dt = +inf`` then drives the free
        # coordinates to their (possibly infinite) bounds, which ``solve`` flags
        # as a non-finite direction.
        _found, _t_tail, x_tail = _segment_min(
            t_prev, x_curr, jnp.asarray(jnp.inf, dtype)
        )
        x_c = jnp.where(done, x_curr, x_tail)
        n_seg = n_used + jnp.where(done, 0, 1)
        return x_c, n_seg

    # ------------------------------------------------------ subspace stage --

    def _subspace_cg(
        self,
        x_c: Float[Array, " n_p"],
        g0: Float[Array, " n_p"],
        H: Callable[[Float[Array, " n_p"]], Float[Array, " n_p"]],
        lo: Float[Array, " n_p"],
        hi: Float[Array, " n_p"],
    ) -> tuple[Float[Array, " n_p"], Int[Array, ""]]:
        """Approximate solve of (16.74) by free-space CG, truncating at a bound."""
        tol = jnp.asarray(self.tol, x_c.dtype)
        free = (x_c > lo + tol) & (x_c < hi - tol)
        free_f = free.astype(x_c.dtype)

        def H_free(v):
            return free_f * H(free_f * v)

        # Residual of ∇q at x_c on the free face: r = -(g0 + H x_c) ⊙ free.
        grad = g0 + H(x_c)
        r = -free_f * grad
        p = r
        rz = jnp.dot(r, r)
        tol_sq = tol * tol * jnp.maximum(rz, 1.0)

        def body(i, carry):
            x, r, p, rz, done, n_it = carry

            def step(c):
                x, r, p, rz, _, n_it = c
                Hp = H_free(p)
                pHp = jnp.dot(p, Hp)
                # Negative curvature or vanishing curvature → stop (keep x).
                neg = pHp <= tol * jnp.dot(p, p)
                alpha = jnp.where(pHp > 0, rz / jnp.maximum(pHp, 1e-30), 0.0)
                x_trial = x + alpha * p
                # Truncate to the box; if any free coord hits a bound, accept
                # the truncated step and stop (N&W intermediate strategy).
                x_proj = self.project(x_trial, lo, hi)
                hit = jnp.any(free & (jnp.abs(x_proj - x_trial) > tol))
                step_to_bound = jnp.where(
                    p > 0,
                    (hi - x) / jnp.where(p > 0, p, 1.0),
                    jnp.where(p < 0, (lo - x) / jnp.where(p < 0, p, -1.0), jnp.inf),
                )
                alpha_max = jnp.min(
                    jnp.where(
                        free & jnp.isfinite(step_to_bound), step_to_bound, jnp.inf
                    )
                )
                alpha_use = jnp.where(hit, jnp.minimum(alpha, alpha_max), alpha)
                x_new = self.project(x + alpha_use * p, lo, hi)
                r_new = r - alpha_use * Hp
                rz_new = jnp.dot(r_new, r_new)
                beta = jnp.where(rz > 1e-30, rz_new / jnp.maximum(rz, 1e-30), 0.0)
                p_new = r_new + beta * p
                stop = neg | hit | (rz_new < tol_sq)
                x_out = jnp.where(neg, x, x_new)
                return x_out, r_new, p_new, rz_new, stop, n_it + 1

            return jax.lax.cond(done, lambda c: c, step, carry)

        init = (x_c, r, p, rz, rz < tol_sq, jnp.zeros((), jnp.int32))
        x_f, _, _, _, _, n_it = jax.lax.fori_loop(0, self.max_cg_iter, body, init)

        # Feasibility + non-ascent safeguard required by N&W for global convergence.
        def q(z: Float[Array, " n_p"]) -> Scalar:
            return jnp.dot(g0, z) + 0.5 * jnp.dot(z, H(z))

        x_out = jnp.where(q(x_f) <= q(x_c) + tol, x_f, x_c)
        return x_out, n_it
