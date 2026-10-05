"""Unit tests for :mod:`slsqp_jax.sqpdax.linalg.steihaug`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.linalg import boundary_step_length, steihaug_step

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
