"""Unit tests for :mod:`slsqp_jax.sqpdax.linalg.steihaug`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.linalg import (
    boundary_step_length,
    null_space_projector,
    steihaug_cg,
    steihaug_step,
)

from .conftest import random_ball_point, random_direction

RADIUS = 0.8


@pytest.mark.parametrize("fill", [0.0, 0.5, 1.0], ids=["centre", "interior", "on-ball"])
@pytest.mark.parametrize("seed", [0, 1])
def test_boundary_step_lands_on_the_ball(dim: int, fill: float, seed: int):
    """``‖w + β p‖ = radius`` with ``β ≥ 0`` for any interior ``w`` and direction ``p``."""
    w = random_ball_point(seed, dim, RADIUS, fill=fill)
    p = 3.0 * random_direction(seed + 10, dim)
    beta = boundary_step_length(w, p, jnp.asarray(RADIUS))
    assert beta >= 0.0
    assert jnp.allclose(jnp.linalg.norm(w + beta * p), RADIUS, rtol=1e-5)


def test_boundary_step_is_zero_for_zero_direction(dim: int):
    """A zero search direction yields ``β = 0`` instead of ``0/0``."""
    w = random_ball_point(0, dim, RADIUS, fill=0.5)
    beta = boundary_step_length(w, jnp.zeros(dim), jnp.asarray(RADIUS))
    assert beta == 0.0


def test_boundary_step_is_zero_when_leaving_from_the_boundary_outward(dim: int):
    """On the ball with ``p`` pointing outward the only root is ``β = 0``."""
    w = random_ball_point(2, dim, RADIUS, fill=1.0)
    p = 2.0 * w  # outward normal direction
    beta = boundary_step_length(w, p, jnp.asarray(RADIUS))
    assert jnp.allclose(beta, 0.0, atol=1e-6)


@pytest.mark.parametrize(
    ("alpha", "expect_hit"),
    [(0.1, False), (5.0, True)],
    ids=["interior-step", "crossing-step"],
)
def test_steihaug_step_takes_cg_step_or_boundary(
    dim: int, alpha: float, expect_hit: bool
):
    """Interior CG steps pass through; crossing steps are replaced by the boundary point."""
    w = random_ball_point(3, dim, RADIUS, fill=0.3)
    p = random_direction(4, dim)
    w_new, hit = steihaug_step(w, p, jnp.asarray(alpha), jnp.asarray(RADIUS))
    assert bool(hit) is expect_hit
    if expect_hit:
        assert jnp.allclose(jnp.linalg.norm(w_new), RADIUS, rtol=1e-5)
        # Boundary point lies on the ray from ``w`` along ``p`` (β ≥ 0).
        beta = jnp.dot(w_new - w, p) / jnp.dot(p, p)
        assert beta >= 0.0
        assert jnp.allclose(w_new, w + beta * p, atol=1e-6)
    else:
        assert jnp.allclose(w_new, w + alpha * p)


def test_steihaug_step_force_boundary_overrides_interior_step(dim: int):
    """``force_boundary`` (negative curvature) exits on the ball even if ``α p`` is tiny."""
    w = random_ball_point(5, dim, RADIUS, fill=0.3)
    p = random_direction(6, dim)
    w_new, hit = steihaug_step(
        w, p, jnp.asarray(0.0), jnp.asarray(RADIUS), force_boundary=jnp.asarray(True)
    )
    assert bool(hit)
    assert jnp.allclose(jnp.linalg.norm(w_new), RADIUS, rtol=1e-5)


def test_steihaug_step_is_jittable_and_vmappable(dim: int):
    """The rule traces under ``jit`` and batches over iterates under ``vmap``."""
    ws = jnp.stack([random_ball_point(s, dim, RADIUS, fill=0.3) for s in range(3)])
    p = random_direction(7, dim)
    alphas = jnp.array([0.1, 5.0, 0.2])

    step = jax.jit(jax.vmap(lambda w, a: steihaug_step(w, p, a, jnp.asarray(RADIUS))))
    w_new, hit = step(ws, alphas)
    assert hit.tolist() == [False, True, False]
    assert jnp.all(jnp.linalg.norm(w_new, axis=1) <= RADIUS * (1 + 1e-5))


# ---------------------------------------------------------------------------
# steihaug_cg kernel
# ---------------------------------------------------------------------------


def spd_quadratic(seed: int, n: int):
    """``(H, b)`` with ``H ≻ 0``; the unconstrained minimiser is ``H⁻¹ b``."""
    B = jax.random.normal(jax.random.key(seed), (n, n))
    H = B.T @ B + jnp.eye(n)
    b = jax.random.normal(jax.random.key(seed + 1), (n,))
    return H, b


@pytest.mark.parametrize("recompute", [False, True], ids=["recursive", "recomputed"])
def test_cg_large_radius_reaches_the_newton_point(dim: int, recompute: bool):
    """Inside a huge ball the kernel converges to ``H⁻¹ b`` within ``n`` steps."""
    H, b = spd_quadratic(0, dim)
    residual = (lambda w: b - H @ w) if recompute else None
    out = steihaug_cg(
        lambda p: H @ p,
        b,
        jnp.zeros(dim),
        jnp.asarray(1e3),
        tol_sq=1e-10,
        max_iter=2 * dim,
        residual=residual,
    )
    assert jnp.allclose(out.w, jnp.linalg.solve(H, b), atol=1e-3)
    assert out.converged
    assert not out.on_boundary
    assert out.n_iter <= dim + 1


def test_cg_small_radius_stops_on_the_ball(dim: int):
    """When the Newton point is outside, the kernel exits on the boundary."""
    H, b = spd_quadratic(1, dim)
    radius = 0.1 * jnp.linalg.norm(jnp.linalg.solve(H, b))
    out = steihaug_cg(
        lambda p: H @ p, b, jnp.zeros(dim), radius, tol_sq=1e-12, max_iter=20
    )
    assert out.on_boundary
    assert jnp.allclose(jnp.linalg.norm(out.w), radius, rtol=1e-5)
    # Model decrease is at least that of the Cauchy point along ``b``.
    q = lambda w: 0.5 * w @ H @ w - b @ w  # noqa: E731
    alpha_c = jnp.minimum(b @ b / (b @ H @ b), radius / jnp.linalg.norm(b))
    assert q(out.w) <= q(alpha_c * b) + 1e-6


def test_cg_negative_curvature_exits_on_the_ball(dim: int):
    """An indefinite ``H`` with ``bᵀHb < 0`` triggers the boundary exit at once."""
    H = -jnp.eye(dim)
    b = jnp.ones(dim)
    out = steihaug_cg(
        lambda p: H @ p,
        b,
        jnp.zeros(dim),
        jnp.asarray(RADIUS),
        tol_sq=1e-12,
        max_iter=10,
        curvature_floor=1e-10,
    )
    assert out.on_boundary
    assert out.n_iter == 1
    assert jnp.allclose(out.w, RADIUS * b / jnp.linalg.norm(b), rtol=1e-5)


def test_cg_projected_variant_stays_in_the_null_space():
    """With ``project`` / a projected ``residual`` the step keeps ``A w = A w0``."""
    n, m = 6, 2
    H, b = spd_quadratic(4, n)
    A = jax.random.normal(jax.random.key(11), (m, n))
    proj = null_space_projector(lambda v: A @ v, lambda y: A.T @ y)
    w0 = jax.random.normal(jax.random.key(12), (n,)) * 0.1
    r0 = proj(b - H @ w0)
    kwargs = dict(tol_sq=1e-10, max_iter=2 * n)
    recursive = steihaug_cg(
        lambda p: H @ p, r0, w0, jnp.asarray(1e3), project=proj, **kwargs
    )
    recomputed = steihaug_cg(
        lambda p: H @ p,
        r0,
        w0,
        jnp.asarray(1e3),
        residual=lambda w: proj(b - H @ w),
        **kwargs,
    )
    for out in (recursive, recomputed):
        assert jnp.allclose(A @ out.w, A @ w0, atol=1e-4)
        assert out.converged
    assert jnp.allclose(recursive.w, recomputed.w, atol=1e-3)
    # Reduced-space optimality: the projected gradient vanishes.
    assert jnp.linalg.norm(proj(b - H @ recursive.w)) <= 1e-3


def test_cg_done_flag_returns_start_point_untouched(dim: int):
    """``done=True`` performs no iteration and reports neither convergence nor boundary."""
    H, b = spd_quadratic(2, dim)
    w0 = jnp.full((dim,), 0.1)
    out = steihaug_cg(
        lambda p: H @ p,
        b,
        w0,
        jnp.asarray(RADIUS),
        tol_sq=1e-12,
        max_iter=10,
        done=jnp.asarray(True),
    )
    assert jnp.array_equal(out.w, w0)
    assert out.n_iter == 0
    assert not out.on_boundary
    assert not out.converged


def test_cg_zero_residual_converges_immediately(dim: int):
    out = steihaug_cg(
        lambda p: p,
        jnp.zeros(dim),
        jnp.zeros(dim),
        jnp.asarray(1.0),
        tol_sq=1e-12,
        max_iter=5,
    )
    assert out.n_iter == 0
    assert out.converged


def test_cg_is_jittable(dim: int):
    H, b = spd_quadratic(3, dim)

    @jax.jit
    def run(radius):
        return steihaug_cg(
            lambda p: H @ p, b, jnp.zeros(dim), radius, tol_sq=1e-10, max_iter=10
        )

    out = run(jnp.asarray(1e3))
    assert jnp.allclose(out.w, jnp.linalg.solve(H, b), atol=1e-3)
