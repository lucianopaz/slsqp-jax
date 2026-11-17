"""Unit tests for :mod:`slsqp_jax.sqpdax.secant.lrh`.

The secant is compared, after every append, with the dense NumPy oracle
``DenseLRHReference`` (PLAN.md section 2.2). The oracle forms dense ``n x n``
matrices on purpose and is used here only as a reference on tiny ``n``.
"""

from __future__ import annotations

import copy
import functools
import warnings
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax.sqpdax.secant import (
    LBFGS,
    LimitedMemoryReducedHessian,
    ReducedHessianSecant,
    Secant,
    SecantRecoveryState,
    SecantResetPolicy,
    SecantResetSignals,
    SecantStatistics,
)

from .conftest import (
    PAIR_SCENARIOS,
    DenseLRHReference,
    LRHAppendRecord,
    PairScenario,
    append_jit,
    build_lrh_with_pairs,
    dense_hessian,
    pair_with_curvature_ratio,
)

jax.config.update("jax_enable_x64", True)

TOL = 1e-9

SIZES = pytest.mark.parametrize("n", [5, 12])
MEMORIES = pytest.mark.parametrize("memory", [1, 2, 4])
COUNTS = pytest.mark.parametrize(
    "count_rule",
    [
        lambda m: 0,
        lambda m: 1,
        lambda m: m,
        lambda m: m + 1,
        lambda m: 2 * m + 1,
    ],
    ids=["none", "one", "full", "overfull", "twice"],
)
SEEDS = pytest.mark.parametrize("seed", [0, 1])

diagnostics_jit = eqx.filter_jit(lambda hist, s, y: hist.diagnostics(s, y))
reset_jit = eqx.filter_jit(lambda hist, severity: hist.reset(severity))


@dataclass(frozen=True)
class Step:
    """State around one append: LRH before/after, oracle snapshot and record."""

    before: LimitedMemoryReducedHessian
    after: LimitedMemoryReducedHessian
    ref: DenseLRHReference
    record: LRHAppendRecord | None
    diag: object | None
    pair: tuple | None


@functools.lru_cache(maxsize=None)
def _replay(
    scenario: PairScenario,
    n: int,
    memory: int,
    count: int,
    seed: int,
    sigma_rule: str = "sy_over_ss",
    initial_sigma: float = 1.0,
):
    """Replay a scenario on the secant and the oracle, once per parameter set."""
    options = {"sigma_rule": sigma_rule, "initial_sigma": initial_sigma}
    ref = DenseLRHReference(n, memory, **options)
    secant = LimitedMemoryReducedHessian(n=n, memory=memory, **options)
    steps = [Step(secant, secant, copy.deepcopy(ref), None, None, None)]
    for s, y in scenario.make(seed, n, count):
        s, y = jnp.asarray(s), jnp.asarray(y)
        diag = diagnostics_jit(secant, s, y)
        record = ref.append(np.asarray(s), np.asarray(y))
        new = append_jit(secant, s, y)
        steps.append(Step(secant, new, copy.deepcopy(ref), record, diag, (s, y)))
        secant = new
    return steps


def _history(pair_scenario, n, memory, count_rule, seed, **options) -> list[Step]:
    return _replay(pair_scenario, n, memory, count_rule(memory), seed, **options)


def _assert_storage_invariants(secant: LimitedMemoryReducedHessian) -> None:
    """Orthonormal ``Z``, zero/identity padding of ``Z``/``T``/``R``, triangularity.

    The *stored* padded block of ``R`` is checked (products mask it, so they
    could not reveal a broken invariant).
    """
    r = int(secant.rank)
    m = secant.memory
    z = np.asarray(secant.basis)
    t = np.asarray(secant.triangular)
    chol = np.asarray(secant.cholesky)
    assert secant.rank.dtype == jnp.int32
    assert 0 <= r <= m
    np.testing.assert_array_equal(z[:, r:], 0.0)
    np.testing.assert_allclose(z[:, :r].T @ z[:, :r], np.eye(r), atol=1e-12)
    # padded block of R is exactly the identity, with no coupling
    np.testing.assert_array_equal(chol[r:, r:], np.eye(m - r))
    np.testing.assert_array_equal(chol[:r, r:], 0.0)
    np.testing.assert_array_equal(chol[r:, :r], 0.0)
    np.testing.assert_array_equal(np.tril(chol, -1), 0.0)
    assert np.all(np.diag(chol)[:r] > 0.0)
    # T: upper triangular, zero padded
    np.testing.assert_array_equal(np.tril(t, -1), 0.0)
    np.testing.assert_array_equal(t[r:, :], 0.0)
    np.testing.assert_array_equal(t[:, r:], 0.0)
    assert np.all(np.diag(t)[:r] > 0.0)


# ---------------------------------------------------------------------------
# Construction, registry, from_spec
# ---------------------------------------------------------------------------


def test_initial_state_is_identity_model():
    secant = LimitedMemoryReducedHessian(n=6, memory=3)
    v = jnp.arange(1.0, 7.0)
    np.testing.assert_allclose(secant.hvp(v), v)
    np.testing.assert_allclose(secant.inverse_hvp(v), v)
    assert int(secant.num_pairs) == 0
    assert float(secant.sigma) == 1.0
    _assert_storage_invariants(secant)
    assert isinstance(secant, ReducedHessianSecant)
    assert secant.kind == "lrh"
    assert Secant._registry["lrh"] is LimitedMemoryReducedHessian


@pytest.mark.parametrize(
    "spec",
    [
        {"kind": "lrh", "memory": 3},
        {"kind": "lrh", "memory": 2, "accept_tol": 1e-3, "damping_threshold": 0.3},
    ],
)
def test_from_spec_injects_n_and_memory(spec):
    secant = Secant.from_spec(spec, n=5)
    assert type(secant) is LimitedMemoryReducedHessian
    assert (secant.n, secant.memory) == (5, spec["memory"])
    for key, value in spec.items():
        if key not in ("kind", "memory"):
            assert getattr(secant, key) == value
    assert secant.basis.shape == (5, spec["memory"])


@pytest.mark.parametrize("sigma_rule", ["yy_over_sy", "sy_over_ss"])
@pytest.mark.parametrize("initial_sigma", [1.0, 7.5])
def test_sigma_options_through_constructor_and_from_spec(sigma_rule, initial_sigma):
    spec = {
        "kind": "lrh",
        "memory": 2,
        "sigma_rule": sigma_rule,
        "initial_sigma": initial_sigma,
    }
    for secant in (
        LimitedMemoryReducedHessian(
            n=4, memory=2, sigma_rule=sigma_rule, initial_sigma=initial_sigma
        ),
        Secant.from_spec(spec, n=4),
    ):
        assert (secant.sigma_rule, secant.initial_sigma) == (sigma_rule, initial_sigma)
        v = jnp.arange(1.0, 5.0)
        np.testing.assert_allclose(secant.hvp(v), initial_sigma * v)
        np.testing.assert_allclose(secant.inverse_hvp(v), v / initial_sigma)
        assert float(secant.sigma) == initial_sigma
        _assert_storage_invariants(secant)
    assert LimitedMemoryReducedHessian(n=4, memory=2).sigma_rule == "sy_over_ss"
    assert LimitedMemoryReducedHessian(n=4, memory=2).initial_sigma == 1.0


@pytest.mark.parametrize("initial_sigma", [0.0, -1.0, np.inf, np.nan])
def test_invalid_initial_sigma_is_rejected(initial_sigma):
    with pytest.raises(ValueError, match="initial_sigma"):
        LimitedMemoryReducedHessian(n=3, memory=2, initial_sigma=initial_sigma)
    with pytest.raises(ValueError, match="initial_sigma"):
        Secant.from_spec(
            {"kind": "lrh", "memory": 2, "initial_sigma": initial_sigma}, n=3
        )


def test_unknown_sigma_rule_is_rejected():
    with pytest.raises(ValueError, match="sigma_rule"):
        LimitedMemoryReducedHessian(n=3, memory=2, sigma_rule="bogus")
    with pytest.raises(ValueError, match="sigma_rule"):
        Secant.from_spec({"kind": "lrh", "memory": 2, "sigma_rule": "bogus"}, n=3)


@pytest.mark.parametrize("initial_sigma", [0.5, 40.0])
def test_sy_over_ss_never_exceeds_yy_over_sy(initial_sigma):
    """Cauchy-Schwarz: the Rayleigh-quotient rule gives the softer complement."""
    s, y = (jnp.asarray(v) for v in PAIR_SCENARIOS["spd"].make(1, 6, 1)[0])
    sigmas = {
        rule: float(
            LimitedMemoryReducedHessian(
                n=6, memory=3, sigma_rule=rule, initial_sigma=initial_sigma
            )
            .append(s, y)
            .sigma
        )
        for rule in ("yy_over_sy", "sy_over_ss")
    }
    assert sigmas["sy_over_ss"] <= sigmas["yy_over_sy"] + 1e-12
    assert sigmas["sy_over_ss"] != sigmas["yy_over_sy"]


def test_from_spec_warns_on_unknown_parameter():
    with pytest.warns(UserWarning, match="ignoring unknown params"):
        secant = Secant.from_spec({"kind": "lrh", "memory": 2, "bogus": 1}, n=4)
    assert isinstance(secant, LimitedMemoryReducedHessian)


# ---------------------------------------------------------------------------
# Agreement with the dense reference
# ---------------------------------------------------------------------------


SIGMA_OPTIONS = pytest.mark.parametrize(
    ("sigma_rule", "initial_sigma"),
    [
        ("yy_over_sy", 1.0),
        ("sy_over_ss", 1.0),
        ("yy_over_sy", 25.0),
        ("sy_over_ss", 0.3),
    ],
)


@SIZES
@MEMORIES
@COUNTS
@SEEDS
@SIGMA_OPTIONS
def test_agrees_with_dense_reference_after_every_append(
    pair_scenario, n, memory, count_rule, seed, sigma_rule, initial_sigma
):
    """``B``, ``B⁻¹``, ``span Z``, ``R``, ``σ``, rank, θ and skip flag match the oracle."""
    rng = np.random.default_rng(7)
    probe = rng.normal(size=n)
    options = {"sigma_rule": sigma_rule, "initial_sigma": initial_sigma}
    for step in _history(pair_scenario, n, memory, count_rule, seed, **options):
        after, ref = step.after, step.ref
        r = int(after.rank)
        assert r == ref.rank
        np.testing.assert_allclose(
            dense_hessian(after), ref.B, atol=TOL, rtol=TOL, err_msg="B"
        )
        # inverse solves are only as accurate as the conditioning allows
        cond = np.linalg.cond(ref.B)
        np.testing.assert_allclose(
            after.inverse_hvp(probe),
            np.linalg.solve(ref.B, probe),
            atol=TOL,
            rtol=1e-7 + 1e-14 * cond,
        )
        np.testing.assert_allclose(
            after.inverse_hvp(after.hvp(probe)),
            probe,
            atol=TOL + 1e-13 * cond,
            rtol=1e-7,
        )
        np.testing.assert_allclose(float(after.sigma), ref.sigma, rtol=1e-10)
        z = np.asarray(after.basis)[:, :r]
        np.testing.assert_allclose(z @ z.T, ref.projector(), atol=TOL)
        # R is the Cholesky factor of the reduced Hessian Zᵀ B Z
        chol = np.asarray(after.cholesky)[:r, :r]
        np.testing.assert_allclose(chol.T @ chol, z.T @ ref.B @ z, atol=TOL, rtol=TOL)
        # factors(), num_pairs and curvature_bounds expose the same state
        factors = after.factors()
        np.testing.assert_array_equal(factors.basis, after.basis)
        np.testing.assert_array_equal(factors.cholesky, after.cholesky)
        assert float(factors.sigma) == float(after.sigma)
        assert int(factors.rank) == int(after.num_pairs) == r
        lower, upper = after.curvature_bounds()
        assert float(lower) == float(upper) == float(after.sigma)
        if step.record is None:
            continue
        # diagnostics report what the append applied
        diag = step.diag
        assert bool(diag.skipped) == step.record.skipped
        np.testing.assert_allclose(
            float(diag.damping_theta), step.record.theta, atol=1e-10
        )
        np.testing.assert_allclose(
            float(diag.raw_curvature), float(step.pair[0] @ step.pair[1])
        )
        assert (float(diag.damping_theta) == 1.0) >= step.record.skipped


@SIZES
@MEMORIES
@COUNTS
@SEEDS
def test_basis_orthonormality_padding_and_step_span(
    pair_scenario, n, memory, count_rule, seed
):
    """Storage invariants after every append, and ``Z T`` = the newest accepted steps."""
    for step in _history(pair_scenario, n, memory, count_rule, seed):
        after = step.after
        _assert_storage_invariants(after)
        r = int(after.rank)
        z_t = np.asarray(after.basis) @ np.asarray(after.triangular)
        steps = np.column_stack([np.zeros((n, 0)), *step.ref.steps])
        np.testing.assert_allclose(z_t[:, :r], steps, atol=TOL, rtol=TOL)


@SIZES
@MEMORIES
@COUNTS
@SEEDS
def test_secant_condition_right_after_append(
    pair_scenario, n, memory, count_rule, seed
):
    """``B_new s̃ = Z Zᵀ ŷ`` (``Z`` the new basis), and ``s̃ᵀ B_new s̃ = s̃ᵀ y_d``."""
    for step in _history(pair_scenario, n, memory, count_rule, seed)[1:]:
        r = step.record
        assert r is not None and not r.skipped
        z = np.asarray(step.after.basis)[:, : int(step.after.rank)]
        b_s = np.asarray(step.after.hvp(r.s_tilde))
        np.testing.assert_allclose(b_s, z @ (z.T @ r.y_hat), atol=TOL, rtol=TOL)
        # without a drop the full secant condition holds (the check is
        # vacuous after a drop, which discards the oldest direction)
        holds = np.allclose(b_s, r.y_hat, atol=TOL, rtol=TOL) and np.allclose(
            r.s_tilde @ b_s, r.s_tilde @ r.y_damped, atol=TOL, rtol=TOL
        )
        assert holds or r.dropped


def test_rank_and_dropping_behaviour_match_the_scenarios(pair_scenario):
    """Independent steps fill the memory then keep rank ``m``; rejection never drops."""
    memory = 2
    history = _replay(pair_scenario, 6, memory, 2 * memory + 1, 0)
    ranks = [int(s.after.rank) for s in history]
    assert max(ranks) <= memory
    assert ranks == sorted(ranks)  # rank never decreases through appends
    for step in history[1:]:
        assert not (step.record.dropped and not step.record.accepted)
    assert (ranks == [0, 1, 2, 2, 2, 2]) >= pair_scenario.independent
    rejected = any(not s.record.accepted for s in history[1:])
    assert rejected >= (pair_scenario.rejects_by <= 2 * memory + 1)


def test_memory_one_keeps_the_newest_step_only():
    """``memory = 1``: the basis is always the newest step direction."""
    n = 4
    pairs = PAIR_SCENARIOS["spd"].make(5, n, 4)
    secant = LimitedMemoryReducedHessian(n=n, memory=1)
    for s, y in pairs:
        secant = append_jit(secant, jnp.asarray(s), jnp.asarray(y))
        assert int(secant.rank) == 1
        _assert_storage_invariants(secant)
        newest = s / np.linalg.norm(s)
        z = np.asarray(secant.basis)[:, 0]
        np.testing.assert_allclose(np.abs(z @ newest), 1.0, atol=1e-12)
        np.testing.assert_allclose(float(secant.triangular[0, 0]), np.linalg.norm(s))


# ---------------------------------------------------------------------------
# Damping
# ---------------------------------------------------------------------------


@pytest.fixture
def warm_secant() -> LimitedMemoryReducedHessian:
    """Secant holding three accepted pairs of a random SPD quadratic (n=6, m=4)."""
    return build_lrh_with_pairs(6, PAIR_SCENARIOS["spd"].make(3, 6, 3), memory=4)


@pytest.mark.parametrize("ratio", [-1.0, 0.0, 0.1, 0.19, 0.2, 0.5, 1.0])
@pytest.mark.parametrize("seed", [0, 1])
def test_damping_ratio_sweep(warm_secant, ratio, seed):
    """``θ = 1`` iff ``sᵀy ≥ τ sᵀB s``; else ``sᵀy_d = τ sᵀB s``; same θ as LBFGS."""
    hist = warm_secant
    tau = hist.damping_threshold
    s, y = pair_with_curvature_ratio(hist, ratio, seed)
    b_s = hist.hvp(s)
    diag = hist.diagnostics(s, y)
    theta = float(diag.damping_theta)
    assert not bool(diag.skipped)
    assert 0.0 <= theta <= 1.0
    assert (theta < 1.0 - 1e-9) == (ratio < tau)

    y_d = theta * y + (1.0 - theta) * b_s
    target = (s @ y) if ratio >= tau else tau * (s @ b_s)
    np.testing.assert_allclose(s @ y_d, target, rtol=1e-10, atol=1e-10)

    lbfgs = LBFGS(n=hist.n, memory=hist.memory, damping_threshold=tau)
    np.testing.assert_allclose(
        theta, float(lbfgs._damping_theta(s, y, b_s)), atol=1e-12
    )

    # the applied update: the new model has curvature s̃ᵀ y_d along s
    updated = hist.append(s, y)
    np.testing.assert_allclose(s @ updated.hvp(s), s @ y_d, rtol=1e-9, atol=1e-10)


# ---------------------------------------------------------------------------
# Skipped pairs
# ---------------------------------------------------------------------------


def _identical(a: LimitedMemoryReducedHessian, b: LimitedMemoryReducedHessian) -> bool:
    return all(
        np.array_equal(x, y)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)
    )


def _assert_identical(a: LimitedMemoryReducedHessian, b: LimitedMemoryReducedHessian):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize(
    ("s_factor", "y_factor"),
    [
        (1e-10, 1.0),
        (1.0, 1e-10),
        (1.0, 1e9),
        (1e3, 1e-6),
        (np.nan, 1.0),
        (1.0, np.inf),
        (0.0, 1.0),
    ],
    ids=["tiny_s", "tiny_y", "ratio_high", "ratio_low", "nan_s", "inf_y", "zero_s"],
)
@pytest.mark.parametrize("memory", [1, 4])
def test_step_one_skips_leave_the_state_unchanged(memory, s_factor, y_factor):
    hist = build_lrh_with_pairs(
        6, PAIR_SCENARIOS["negative_definite"].make(2, 6, 2), memory=memory
    )
    rng = np.random.default_rng(0)
    with np.errstate(invalid="ignore"):
        s = jnp.asarray(s_factor * np.ones(6) * rng.uniform(0.5, 1.0, size=6))
        y = jnp.asarray(y_factor * rng.uniform(0.5, 1.0, size=6))
    assert bool(hist.should_skip(s, y))
    diag = hist.diagnostics(s, y)
    assert bool(diag.skipped)
    assert float(diag.damping_theta) == 1.0
    _assert_identical(hist.append(s, y), hist)
    _assert_identical(append_jit(hist, s, y), hist)


def test_a_regular_pair_is_not_skipped(warm_secant):
    s, y = pair_with_curvature_ratio(warm_secant, 0.5, 0)
    assert not bool(warm_secant.should_skip(s, y))
    assert not bool(warm_secant.diagnostics(s, y).skipped)
    assert int(warm_secant.append(s, y).rank) == int(warm_secant.rank) + 1


@pytest.mark.parametrize(
    ("multiplier", "expect_skip"),
    [(-2.0, True), (-1.6, True), (-1.0, False)],
    ids=["y=-2Bs", "y=-1.6Bs", "y=-Bs"],
)
def test_curvature_guard_late_skip(warm_secant, multiplier, expect_skip):
    """Step 4b: ``‖s‖ = 1e-7`` passes step 1 but the guard skips ``y = k B s``, ``k < -1.5``."""
    hist = warm_secant
    ref = DenseLRHReference(hist.n, hist.memory)
    for s, y in PAIR_SCENARIOS["spd"].make(3, 6, 3):
        ref.append(s, y)
    rng = np.random.default_rng(0)
    s = rng.normal(size=hist.n)
    s = jnp.asarray(1e-7 * s / np.linalg.norm(s))
    y = multiplier * hist.hvp(s)
    assert not bool(hist.should_skip(s, y))

    diag = hist.diagnostics(s, y)
    updated = hist.append(s, y)
    record = ref.append(np.asarray(s), np.asarray(y))

    assert bool(diag.skipped) == expect_skip == record.skipped
    assert (float(diag.damping_theta) == 1.0) >= expect_skip
    assert (int(updated.rank) == int(hist.rank)) == expect_skip
    assert _identical(updated, hist) == expect_skip
    np.testing.assert_allclose(float(diag.damping_theta), record.theta, atol=1e-10)
    np.testing.assert_allclose(dense_hessian(updated), ref.B, atol=TOL, rtol=TOL)
    assert np.linalg.eigvalsh(np.asarray(dense_hessian(updated))).min() > 0.0
    _assert_storage_invariants(updated)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def _expected_soft(ref: DenseLRHReference) -> np.ndarray:
    if ref.rank == 0:
        return ref.B
    b = ref.steps[-1] / np.linalg.norm(ref.steps[-1])
    curvature = b @ ref.B @ b
    return ref.sigma * np.eye(ref.n) + (curvature - ref.sigma) * np.outer(b, b)


# severity -> (dense B, sigma, rank) after the reset; the second argument of
# each lambda is the secant's ``initial_sigma``.
RESETS = {
    0: (
        lambda ref, init: _expected_soft(ref),
        lambda ref, init: ref.sigma,
        lambda ref: min(ref.rank, 1),
    ),
    1: (
        lambda ref, init: ref.sigma * np.eye(ref.n),
        lambda ref, init: ref.sigma,
        lambda ref: 0,
    ),
    2: (
        lambda ref, init: init * np.eye(ref.n),
        lambda ref, init: init,
        lambda ref: 0,
    ),
}


@pytest.mark.parametrize("count", [0, 1, 3, 6])
@pytest.mark.parametrize("severity", [-3, 0, 1, 2, 7])
@pytest.mark.parametrize("initial_sigma", [1.0, 5.0])
def test_reset_with_traced_severity_under_jit(count, severity, initial_sigma):
    """Severity is traced and clipped to ``[0, 2]``; padding invariants hold after.

    Severity 2 returns to ``initial_sigma``.
    """
    n, memory = 7, 4
    pairs = PAIR_SCENARIOS["spd"].make(11, n, count)
    hist = build_lrh_with_pairs(n, pairs, memory=memory, initial_sigma=initial_sigma)
    ref = DenseLRHReference(n, memory, initial_sigma=initial_sigma)
    for s, y in pairs:
        ref.append(s, y)

    out = reset_jit(hist, jnp.asarray(severity))
    expected_b, expected_sigma, expected_rank = RESETS[int(np.clip(severity, 0, 2))]
    np.testing.assert_allclose(
        dense_hessian(out), expected_b(ref, initial_sigma), atol=TOL, rtol=TOL
    )
    np.testing.assert_allclose(
        float(out.sigma), expected_sigma(ref, initial_sigma), rtol=1e-12
    )
    assert int(out.rank) == int(out.num_pairs) == expected_rank(ref)
    _assert_storage_invariants(out)
    assert jax.tree.structure(out) == jax.tree.structure(hist)
    for a, b in zip(jax.tree.leaves(out), jax.tree.leaves(hist), strict=True):
        assert (a.shape, a.dtype) == (b.shape, b.dtype)

    # the reset model keeps working: a new pair gives the same result as a
    # fresh oracle started from the reset model's complement curvature (a
    # reset that left a basis behind is allowed to differ)
    s, y = (jnp.asarray(v) for v in PAIR_SCENARIOS["spd"].make(12, n, 1)[0])
    appended = append_jit(out, s, y)
    _assert_storage_invariants(appended)
    assert int(appended.rank) == int(out.rank) + 1 - (int(out.rank) == memory)
    fresh = DenseLRHReference(
        n, memory, initial_sigma=expected_sigma(ref, initial_sigma)
    )
    fresh.append(np.asarray(s), np.asarray(y))
    matches_fresh = np.allclose(dense_hessian(appended), fresh.B, atol=TOL, rtol=TOL)
    assert matches_fresh or expected_rank(ref) != 0


def test_reset_python_int_severity_matches_traced():
    hist = build_lrh_with_pairs(5, PAIR_SCENARIOS["spd"].make(1, 5, 3), memory=3)
    for severity in (0, 1, 2):
        _assert_identical(hist.reset(severity), reset_jit(hist, jnp.asarray(severity)))


def test_secant_reset_policy_apply_under_jit():
    """The policy drives soft, diagonal and identity resets through ``lax.cond``."""
    n = 5
    hist = build_lrh_with_pairs(n, PAIR_SCENARIOS["spd"].make(2, n, 4), memory=4)
    policy = SecantResetPolicy()
    state = SecantRecoveryState()
    signals = SecantResetSignals(
        subproblem_streak=jnp.asarray(1),
        step_streak=jnp.asarray(0),
        model_streak=jnp.asarray(0),
    )
    apply = jax.jit(policy.apply)

    # healthy iteration: nothing happens
    same, _, severity = apply(hist, SecantResetSignals.none(), state)
    assert int(severity) == -1
    _assert_identical(same, hist)

    ranks = []
    for expected_severity in (0, 1, 2):
        hist, state, severity = apply(hist, signals, state)
        assert int(severity) == expected_severity
        ranks.append(int(hist.rank))
        _assert_storage_invariants(hist)
    assert ranks == [1, 0, 0]
    np.testing.assert_allclose(dense_hessian(hist), np.eye(n), atol=1e-12)


def test_secant_statistics_under_jit():
    """``SecantStatistics`` counts appends, damped pairs, skips and resets."""
    n = 3
    e = np.eye(n)
    pairs = [
        (e[0], 2.0 * e[0]),  # accepted
        (e[1], -1.0 * e[1]),  # damped (negative curvature)
        (np.zeros(n), np.ones(n)),  # skipped
        (e[2], 3.0 * e[2]),  # accepted
    ]
    secant = LimitedMemoryReducedHessian(n=n, memory=4)
    stats = SecantStatistics.initial(secant, jnp.float64)

    @eqx.filter_jit
    def step(secant, stats, s, y):
        diag = secant.diagnostics(s, y)
        new = secant.append(s, y)
        return new, stats.record_append(diag, new)

    for s, y in pairs:
        secant, stats = step(secant, stats, jnp.asarray(s), jnp.asarray(y))
    assert int(stats.n_appends) == 3
    assert int(stats.n_skips) == 1
    assert int(stats.n_damped) == 1
    assert int(stats.n_appends) == int(secant.num_pairs)
    assert float(stats.min_damping_theta) < 1.0
    np.testing.assert_allclose(float(stats.last_condition), 1.0)

    reset = reset_jit(secant, jnp.asarray(0))
    stats = jax.jit(lambda st, sev, sec: st.record_reset(sev, sec))(
        stats, jnp.asarray(0), reset
    )
    np.testing.assert_array_equal(stats.n_resets, [1, 0, 0])
    assert set(stats.as_stats()) >= {"secant_n_appends", "secant_n_resets"}


# ---------------------------------------------------------------------------
# jit / scan / memory budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("memory", [2, 4])
def test_append_inside_lax_scan(pair_scenario, memory):
    """``append`` is a valid ``lax.scan`` body (fixed pytree structure and dtypes)."""
    n, count = 6, 2 * memory + 3
    pairs = pair_scenario.make(4, n, count)
    s_all = jnp.stack([jnp.asarray(s) for s, _ in pairs])
    y_all = jnp.stack([jnp.asarray(y) for _, y in pairs])
    init = LimitedMemoryReducedHessian(n=n, memory=memory)

    @jax.jit
    def run(init, s_all, y_all):
        def body(hist, sy):
            return hist.append(*sy), hist.diagnostics(*sy).damping_theta

        return jax.lax.scan(body, init, (s_all, y_all))

    final, thetas = run(init, s_all, y_all)
    ref = DenseLRHReference(n, memory)
    for s, y in pairs:
        ref.append(s, y)
    np.testing.assert_allclose(dense_hessian(final), ref.B, atol=TOL, rtol=TOL)
    assert int(final.rank) == ref.rank
    assert thetas.shape == (count,)
    _assert_storage_invariants(final)


def _all_avals(jaxpr):
    """Every abstract value in a jaxpr, recursing into sub-jaxprs."""
    for var in (*jaxpr.invars, *jaxpr.outvars, *jaxpr.constvars):
        yield var.aval
    for eqn in jaxpr.eqns:
        for var in eqn.outvars:
            yield var.aval
        for param in eqn.params.values():
            for item in param if isinstance(param, (tuple, list)) else (param,):
                inner = getattr(item, "jaxpr", item)
                if hasattr(inner, "eqns"):
                    yield from _all_avals(inner)


@pytest.mark.parametrize("operation", ["append", "diagnostics", "reset"])
def test_no_quadratic_in_n_arrays(operation):
    """No array of the update, diagnostics or reset has two dimensions equal to ``n``."""
    n, memory = 31, 3  # n differs from memory, memory + 1 and every product of them
    hist = build_lrh_with_pairs(n, PAIR_SCENARIOS["spd"].make(0, n, 2), memory=memory)
    s, y = (jnp.asarray(v) for v in PAIR_SCENARIOS["spd"].make(1, n, 1)[0])
    calls = {
        "append": lambda h: h.append(s, y),
        "diagnostics": lambda h: h.diagnostics(s, y),
        "reset": lambda h: h.reset(jnp.asarray(0)),
    }
    closed = jax.make_jaxpr(calls[operation])(hist)
    for aval in _all_avals(closed.jaxpr):
        assert sum(dim >= n for dim in getattr(aval, "shape", ())) <= 1, aval


def test_append_does_not_warn_or_emit_nans_on_a_long_run():
    """A long run with all branches (damping, rejection, drop) stays finite."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        history = _replay(PAIR_SCENARIOS["dependent_indefinite"], 8, 3, 30, 5)
    final = history[-1].after
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(final))
    np.testing.assert_allclose(
        dense_hessian(final), history[-1].ref.B, atol=TOL, rtol=TOL
    )


# ---------------------------------------------------------------------------
# spectrum_bounds
# ---------------------------------------------------------------------------


@SIZES
@MEMORIES
@COUNTS
@SEEDS
def test_spectrum_bounds_match_dense_eigenvalues(
    pair_scenario, n, memory, count_rule, seed
):
    """``spectrum_bounds`` equals the extreme eigenvalues of the dense ``B``."""
    for step in _history(pair_scenario, n, memory, count_rule, seed):
        eigs = np.linalg.eigvalsh(np.asarray(dense_hessian(step.after)))
        lower, upper = step.after.spectrum_bounds()
        np.testing.assert_allclose(float(upper), eigs[-1], rtol=1e-8)
        np.testing.assert_allclose(float(lower), eigs[0], rtol=1e-6, atol=1e-8)
        assert 0.0 < float(lower) <= float(upper)


@pytest.mark.parametrize("count", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("memory", [3, 4])
def test_spectrum_bounds_full_rank_excludes_the_complement(count, memory):
    """With ``rank = n`` there is no complement: ``σ`` must not enter the bounds."""
    n = 3
    hist = build_lrh_with_pairs(
        n,
        PAIR_SCENARIOS["spd"].make(2, n, count),
        memory=memory,
        initial_sigma=1e-3,
    )
    assert int(hist.rank) == min(count, n)
    eigs = np.linalg.eigvalsh(np.asarray(dense_hessian(hist)))
    lower, upper = hist.spectrum_bounds()
    np.testing.assert_allclose([float(lower), float(upper)], eigs[[0, -1]], rtol=1e-8)
    assert jax.jit(lambda h: h.spectrum_bounds())(hist)[0] == lower


# ---------------------------------------------------------------------------
# Narrow valley (the LOGROS failure that removed VARCHEN damping)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sigma_rule", ["yy_over_sy", "sy_over_ss"])
def test_narrow_valley_soft_curvature_is_not_stiffened_by_sigma(
    narrow_valley, sigma_rule
):
    """Powell damping toward ``B s̃`` keeps the soft-direction curvature small.

    Mirrors ``test_narrow_valley_soft_curvature_is_not_stiffened_by_the_diagonal``
    in ``test_lbfgs.py``: ``H = 0.1 t tᵀ + 1.5e5 n nᵀ``, a stiff
    ``initial_sigma = 1e4`` and 30 pairs crawling along ``t``. Measured
    ``tᵀB t`` (final value, maximum over the last ten pairs) over the three
    fixture angles: ``yy_over_sy`` 5.6-6.4 and 8.0-10.3, ``sy_over_ss``
    0.85-1.9 and 0.88-2.1, against the L-BFGS bound ``100``.
    """
    tangent, pairs = narrow_valley
    hist = LimitedMemoryReducedHessian(
        n=2, memory=2, initial_sigma=1e4, sigma_rule=sigma_rule
    )
    soft_curvature = []
    for s, y in pairs:
        hist = append_jit(hist, s, y)
        soft_curvature.append(float(tangent @ hist.hvp(tangent)))

    assert min(soft_curvature) > 0.0
    assert soft_curvature[-1] < 100.0
    assert max(soft_curvature[-10:]) < 100.0


def _embedded_valley(
    angle_deg: float, seed: int, c_rel: float, n: int = 8, count: int = 30
) -> tuple[np.ndarray, list[tuple]]:
    """Narrow valley embedded in ``R^n`` with steps leaving its plane.

    ``H = 0.1 t tᵀ + 1.5e5 n nᵀ + I_⊥`` where ``t, n`` span a random 2-plane
    (``t`` at ``angle_deg`` inside it) and ``I_⊥`` is the identity on the
    orthogonal ``n - 2`` directions. Steps are ``s = a t + b n + c`` with
    ``a`` and ``b`` as in the ``narrow_valley`` fixture and ``c`` a random
    vector in the orthogonal space with ``‖c‖ = c_rel · a``; ``y = H s``.
    Returns the tangent ``t`` and the pairs.
    """
    rng = np.random.default_rng(seed)
    theta = np.deg2rad(angle_deg)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    tangent = np.cos(theta) * q[:, 0] + np.sin(theta) * q[:, 1]
    normal = -np.sin(theta) * q[:, 0] + np.cos(theta) * q[:, 1]
    others = q[:, 2:]
    hessian = (
        0.1 * np.outer(tangent, tangent)
        + 1.5e5 * np.outer(normal, normal)
        + others @ others.T
    )
    pairs = []
    for _ in range(count):
        a = 10.0 ** rng.uniform(-1.0, 0.0)
        b = 0.05 * a * rng.uniform(0.5, 1.5) * rng.choice([-1.0, 1.0])
        c = others @ rng.normal(size=n - 2)
        c *= c_rel * a / np.linalg.norm(c)
        s = a * tangent + b * normal + c
        pairs.append((jnp.asarray(s), jnp.asarray(hessian @ s)))
    return tangent, pairs


def _soft_curvature_trace(secant, tangent, pairs) -> list[float]:
    """``tᵀB t`` after each appended pair."""
    trace = []
    for s, y in pairs:
        secant = append_jit(secant, s, y)
        trace.append(float(tangent @ secant.hvp(jnp.asarray(tangent))))
    return trace


@pytest.mark.parametrize("c_rel", [0.05, 0.3])
@pytest.mark.parametrize(("angle", "seed"), [(15.0, 0), (45.0, 1), (75.0, 2)])
def test_embedded_narrow_valley_soft_curvature(angle, seed, c_rel):
    """Soft curvature in a valley embedded in ``R^8`` with off-plane steps.

    With ``n = memory = 2`` the basis becomes the whole space and ``σ`` stops
    acting; here ``n = 8``, ``memory = 2``, so steps keep a component outside
    ``range(Z)`` and a stiff ``σ`` (``initial_sigma = 1e4``) meets soft
    directions that are new to the basis (PLAN.md section 2.2, step 4).
    The true ``tᵀH t`` is ``0.1``. This is a measurement and regression test,
    not the L-BFGS bound of 100 (neither rule nor L-BFGS at ``memory = 2``
    meets it): it asserts positivity, no stiffening up to the stiff
    curvature itself (``tᵀB t < 3e4 = 0.2 λ_n`` over the last ten pairs), that
    the Rayleigh-quotient rule (the default) stays below ``3e3`` over the last
    ten pairs (measured maximum ``873``; the paper's rule reaches ``5.6e3`` to
    ``1.1e4`` at ``c_rel = 0.3``), and that it is no stiffer at the end than
    the paper's rule.

    Measured ``tᵀB t`` (final value / maximum over the last ten pairs), with
    L-BFGS at ``memory = 2`` and ``diag = 1e4`` on the same pairs for
    comparison:

    ```text
    c_rel  angle  yy_over_sy       sy_over_ss    L-BFGS m=2
    0.05   15     263 / 1.2e3      1.5 / 478     47 / 76
    0.05   45     1.8e3 / 1.8e3    793 / 793     91 / 91
    0.05   75     261 / 1.0e3      31 / 425      34 / 1.5e3
    0.3    15     1.9e3 / 5.6e3    23 / 613      508 / 1.0e3
    0.3    45     3.2e3 / 1.1e4    873 / 873     687 / 3.8e3
    0.3    75     4.5e3 / 6.6e3    98 / 477      1.0e3 / 1.4e3
    ```

    The paper's rule (``yy_over_sy``) stiffens the soft direction by up to
    five orders of magnitude over the true ``0.1`` and is 3-10x stiffer than
    L-BFGS at equal memory, while ``sy_over_ss`` is comparable to or softer
    than L-BFGS. That is why ``sy_over_ss`` is the default ``sigma_rule``
    (PLAN.md Decision 8); ``yy_over_sy`` stays available as an option.
    """
    tangent, pairs = _embedded_valley(angle, seed, c_rel)
    traces = {
        rule: _soft_curvature_trace(
            LimitedMemoryReducedHessian(
                n=8, memory=2, initial_sigma=1e4, sigma_rule=rule
            ),
            tangent,
            pairs,
        )
        for rule in ("yy_over_sy", "sy_over_ss")
    }
    for trace in traces.values():
        assert np.all(np.isfinite(trace))
        assert min(trace) > 0.0
        assert max(trace[-10:]) < 3e4
    assert max(traces["sy_over_ss"][-10:]) < 3e3
    assert traces["sy_over_ss"][-1] <= traces["yy_over_sy"][-1]
