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
def _replay(scenario: PairScenario, n: int, memory: int, count: int, seed: int):
    """Replay a scenario on the secant and the oracle, once per parameter set."""
    ref = DenseLRHReference(n, memory)
    secant = LimitedMemoryReducedHessian(n=n, memory=memory)
    steps = [Step(secant, secant, copy.deepcopy(ref), None, None, None)]
    for s, y in scenario.make(seed, n, count):
        s, y = jnp.asarray(s), jnp.asarray(y)
        diag = diagnostics_jit(secant, s, y)
        record = ref.append(np.asarray(s), np.asarray(y))
        new = append_jit(secant, s, y)
        steps.append(Step(secant, new, copy.deepcopy(ref), record, diag, (s, y)))
        secant = new
    return steps


def _history(pair_scenario, n, memory, count_rule, seed) -> list[Step]:
    return _replay(pair_scenario, n, memory, count_rule(memory), seed)


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


def test_from_spec_warns_on_unknown_parameter():
    with pytest.warns(UserWarning, match="ignoring unknown params"):
        secant = Secant.from_spec({"kind": "lrh", "memory": 2, "bogus": 1}, n=4)
    assert isinstance(secant, LimitedMemoryReducedHessian)


# ---------------------------------------------------------------------------
# Agreement with the dense reference
# ---------------------------------------------------------------------------


@SIZES
@MEMORIES
@COUNTS
@SEEDS
def test_agrees_with_dense_reference_after_every_append(
    pair_scenario, n, memory, count_rule, seed
):
    """``B``, ``B⁻¹``, ``span Z``, ``R``, ``σ``, rank, θ and skip flag match the oracle."""
    rng = np.random.default_rng(7)
    probe = rng.normal(size=n)
    for step in _history(pair_scenario, n, memory, count_rule, seed):
        after, ref = step.after, step.ref
        r = int(after.rank)
        assert r == ref.rank
        np.testing.assert_allclose(
            dense_hessian(after), ref.B, atol=TOL, rtol=TOL, err_msg="B"
        )
        np.testing.assert_allclose(
            after.inverse_hvp(probe), np.linalg.solve(ref.B, probe), atol=TOL, rtol=1e-7
        )
        # the round trip is only as accurate as the conditioning allows
        round_trip_tol = TOL + 1e-13 * np.linalg.cond(ref.B)
        np.testing.assert_allclose(
            after.inverse_hvp(after.hvp(probe)), probe, atol=round_trip_tol, rtol=1e-7
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
        if step.record.skipped:
            assert float(diag.damping_theta) == 1.0


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
        if r:
            z_t = np.asarray(after.basis) @ np.asarray(after.triangular)
            np.testing.assert_allclose(
                z_t[:, :r], np.column_stack(step.ref.steps), atol=TOL, rtol=TOL
            )


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
        if not r.dropped:
            np.testing.assert_allclose(b_s, r.y_hat, atol=TOL, rtol=TOL)
            np.testing.assert_allclose(
                r.s_tilde @ b_s, r.s_tilde @ r.y_damped, atol=TOL, rtol=TOL
            )


def test_rank_and_dropping_behaviour_match_the_scenarios(pair_scenario):
    """Independent steps fill the memory then keep rank ``m``; rejection never drops."""
    memory = 2
    history = _replay(pair_scenario, 6, memory, 2 * memory + 1, 0)
    ranks = [int(s.after.rank) for s in history]
    assert max(ranks) <= memory
    assert ranks == sorted(ranks)  # rank never decreases through appends
    for step in history[1:]:
        assert not (step.record.dropped and not step.record.accepted)
    if pair_scenario.independent:
        assert ranks == [0, 1, 2, 2, 2, 2]
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
    if expect_skip:
        _assert_identical(updated, hist)
    else:
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


RESETS = {
    0: (_expected_soft, lambda ref: ref.sigma, lambda ref: min(ref.rank, 1)),
    1: (lambda ref: ref.sigma * np.eye(ref.n), lambda ref: ref.sigma, lambda ref: 0),
    2: (lambda ref: np.eye(ref.n), lambda ref: 1.0, lambda ref: 0),
}


@pytest.mark.parametrize("count", [0, 1, 3, 6])
@pytest.mark.parametrize("severity", [-3, 0, 1, 2, 7])
def test_reset_with_traced_severity_under_jit(count, severity):
    """Severity is traced and clipped to ``[0, 2]``; padding invariants hold after."""
    n, memory = 7, 4
    pairs = PAIR_SCENARIOS["spd"].make(11, n, count)
    hist = build_lrh_with_pairs(n, pairs, memory=memory)
    ref = DenseLRHReference(n, memory)
    for s, y in pairs:
        ref.append(s, y)

    out = reset_jit(hist, jnp.asarray(severity))
    expected_b, expected_sigma, expected_rank = RESETS[int(np.clip(severity, 0, 2))]
    np.testing.assert_allclose(dense_hessian(out), expected_b(ref), atol=TOL, rtol=TOL)
    np.testing.assert_allclose(float(out.sigma), expected_sigma(ref), rtol=1e-12)
    assert int(out.rank) == int(out.num_pairs) == expected_rank(ref)
    _assert_storage_invariants(out)
    assert jax.tree.structure(out) == jax.tree.structure(hist)
    for a, b in zip(jax.tree.leaves(out), jax.tree.leaves(hist), strict=True):
        assert (a.shape, a.dtype) == (b.shape, b.dtype)

    # the reset model keeps working: a new pair gives the same result as a
    # fresh oracle started from the reset model's complement curvature
    # (only checkable when the reset left no basis)
    s, y = (jnp.asarray(v) for v in PAIR_SCENARIOS["spd"].make(12, n, 1)[0])
    appended = append_jit(out, s, y)
    _assert_storage_invariants(appended)
    assert int(appended.rank) == int(out.rank) + 1 - (int(out.rank) == memory)
    if expected_rank(ref) == 0:
        fresh = DenseLRHReference(n, memory, initial_sigma=expected_sigma(ref))
        fresh.append(np.asarray(s), np.asarray(y))
        np.testing.assert_allclose(dense_hessian(appended), fresh.B, atol=TOL, rtol=TOL)


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
