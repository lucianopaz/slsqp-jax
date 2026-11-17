"""Self-tests of :class:`DenseLRHReference` (PLAN.md section 2.2).

The reference is the oracle the LRH secant is compared against, so its own
properties are pinned here against independent formulas: the secant
condition, the drop identity, the damping property, the plain dense BFGS
update, and the structure ``B = Z M Zᵀ + σ (I - Z Zᵀ)``. Everything is dense
NumPy on tiny ``n``, as a reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import (
    PAIR_SCENARIOS,
    DenseLRHReference,
    PairScenario,
    numpy_pair_with_curvature_ratio,
    plain_bfgs_update,
    replay_pairs,
)

TOL = 1e-9

# Pairwise design over (n, memory, seed). The histories of shorter counts are
# prefixes of the longest one (``2 m + 1`` pairs), so only that one is
# replayed. The oracle needs only a small grid.
GRID = pytest.mark.parametrize(
    ("n", "memory", "seed"),
    [(5, 1, 0), (5, 2, 1), (5, 4, 0), (12, 1, 1), (12, 2, 0), (12, 4, 1)],
)


def _replay(pair_scenario: PairScenario, n, memory, seed, **kwargs):
    """Fresh reference replayed over scenario pairs; returns (reference, history)."""
    pairs = pair_scenario.make(seed, n, 2 * memory + 1)
    ref = DenseLRHReference(n, memory, **kwargs)
    return ref, replay_pairs(ref, pairs)


def _orthonormal_span_projector(vectors: list[np.ndarray], n: int) -> np.ndarray:
    if not vectors:
        return np.zeros((n, n))
    q, _ = np.linalg.qr(np.column_stack(vectors))
    return q @ q.T


# ---------------------------------------------------------------------------
# State invariants
# ---------------------------------------------------------------------------


@GRID
def test_state_invariants_after_every_append(pair_scenario, n, memory, seed):
    """Symmetric PD ``B``, ``ZᵀZ = I``, ``span Z = span(newest steps)``, structure."""
    ref, history = _replay(pair_scenario, n, memory, seed)
    eye = np.eye(n)
    for state, record in [(DenseLRHReference(n, memory), None), *history]:
        Z = state.Z
        rank = state.rank
        assert rank <= memory
        assert rank == len(state.steps)
        np.testing.assert_allclose(Z.T @ Z, np.eye(rank), atol=1e-12)
        np.testing.assert_allclose(state.B, state.B.T, atol=TOL)
        assert np.linalg.eigvalsh((state.B + state.B.T) / 2).min() > 0.0
        np.testing.assert_allclose(
            state.projector(),
            _orthonormal_span_projector(state.steps, n),
            atol=TOL,
        )
        # B acts as sigma on the orthogonal complement of range(Z), and the
        # model is block diagonal with respect to range(Z).
        complement = eye - state.projector()
        np.testing.assert_allclose(
            state.B @ complement, state.sigma * complement, atol=TOL
        )
    # rank is the number of accepted steps, capped by the memory
    accepted = sum(bool(r.accepted) for _, r in history)
    assert ref.rank == min(accepted, memory)


@GRID
def test_scenarios_exercise_their_branches(pair_scenario, n, memory, seed):
    """Each scenario really reaches the damping / rejection / drop branches."""
    _, history = _replay(pair_scenario, n, memory, seed)
    records = [r for _, r in history]
    assert not any(r.skipped for r in records)
    # the thresholds hold for every prefix of the longest history
    for count in range(len(records) + 1):
        prefix = records[:count]
        damped = any(r.theta < 1.0 - 1e-9 for r in prefix)
        rejected = any(not r.accepted for r in prefix)
        dropped = any(r.dropped for r in prefix)
        damped_rejection = any(not r.accepted and r.theta < 1.0 - 1e-9 for r in prefix)
        assert damped_rejection >= (count >= pair_scenario.damped_rejection_by)
        assert damped >= (count >= pair_scenario.damps_by)
        assert rejected >= (count >= pair_scenario.rejects_by)
        assert dropped >= (pair_scenario.independent and count > memory)
        # a rank-full rejected step never drops
        assert not any(r.dropped and not r.accepted for r in prefix)


# ---------------------------------------------------------------------------
# Secant condition, BFGS formula, drop identity
# ---------------------------------------------------------------------------


@GRID
def test_secant_condition_and_bfgs_formula(pair_scenario, n, memory, seed):
    """``B'' s̃ = ŷ`` and ``B''`` is the plain BFGS update of the expanded ``B'``."""
    _, history = _replay(pair_scenario, n, memory, seed)
    previous = [DenseLRHReference(n, memory)] + [state for state, _ in history[:-1]]
    eye = np.eye(n)
    for before, (_, r) in zip(previous, history):
        np.testing.assert_allclose(r.B_old, before.B, atol=TOL)
        np.testing.assert_allclose(r.B_bfgs @ r.s_tilde, r.y_hat, atol=TOL, rtol=TOL)
        np.testing.assert_allclose(
            r.B_bfgs,
            plain_bfgs_update(r.B_expanded, r.s_tilde, r.y_hat),
            atol=TOL,
            rtol=TOL,
        )
        # the expansion only moves the complement curvature from the old to
        # the new sigma (it is the old model off range(Z), unchanged on it)
        np.testing.assert_allclose(
            r.B_expanded,
            r.B_old + (r.sigma_new - r.sigma_old) * (eye - before.projector()),
            atol=TOL,
            rtol=TOL,
        )


@GRID
def test_drop_identity_and_secant_after_drop(pair_scenario, n, memory, seed):
    """``B_new = P̄ B'' P̄ + σ_new (I - P̄)`` and ``B_new s̃ = P̄ ŷ``.

    ``P̄`` is the projector onto the span of the retained steps; it equals the
    expanded projector when nothing is dropped, so the identity is generic.
    """
    _, history = _replay(pair_scenario, n, memory, seed)
    eye = np.eye(n)
    for state, r in history:
        P = state.projector()
        np.testing.assert_allclose(
            state.B, P @ r.B_bfgs @ P + r.sigma_new * (eye - P), atol=TOL, rtol=TOL
        )
        np.testing.assert_allclose(state.sigma, r.sigma_new)
        np.testing.assert_allclose(state.B @ r.s_tilde, P @ r.y_hat, atol=TOL, rtol=TOL)


# ---------------------------------------------------------------------------
# Damping
# ---------------------------------------------------------------------------


@pytest.fixture
def warm_reference() -> DenseLRHReference:
    """A reference holding three accepted pairs of a random SPD quadratic."""
    ref = DenseLRHReference(6, 4)
    replay_pairs(ref, PAIR_SCENARIOS["spd"].make(3, 6, 3))
    return ref


@pytest.mark.parametrize("ratio", [-1.0, 0.0, 0.1, 0.19, 0.2, 0.5, 1.0])
@pytest.mark.parametrize("seed", [0, 1])
def test_damping_property(warm_reference, ratio, seed):
    """``θ = 1`` iff ``sᵀy ≥ τ sᵀB s``, else ``s̃ᵀy_d = τ s̃ᵀB s̃`` with ``θ ∈ [0, 1]``."""
    ref = warm_reference
    tau = ref.damping_threshold
    s, y = numpy_pair_with_curvature_ratio(ref.B, ratio, seed)
    sbs = s @ ref.B @ s
    record = ref.append(s, y)
    assert record.accepted
    assert 0.0 <= record.theta <= 1.0
    assert (record.theta < 1.0 - 1e-9) == (ratio < tau)
    s_t = record.s_tilde
    np.testing.assert_allclose(s_t, s, atol=1e-12)
    # undamped pairs keep y, damped pairs land exactly on the threshold
    target = np.where(ratio >= tau, s @ y, tau * sbs)
    np.testing.assert_allclose(s_t @ record.y_damped, target, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        record.y_damped,
        record.theta * y + (1.0 - record.theta) * (record.B_old @ s_t),
        atol=1e-12,
    )
    assert ref.theta == record.theta


# ---------------------------------------------------------------------------
# Skipped pairs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("s_factor", "y_factor"),
    [
        (1e-10, 1.0),  # tiny step
        (1.0, 1e-10),  # tiny gradient difference
        (1.0, 1e9),  # curvature ratio above 1e8
        (1e3, 1e-6),  # curvature ratio below 1e-8
        (np.nan, 1.0),  # non-finite step
        (1.0, np.inf),  # non-finite gradient difference
        (0.0, 1.0),  # zero step
    ],
    ids=["tiny_s", "tiny_y", "ratio_high", "ratio_low", "nan_s", "inf_y", "zero_s"],
)
@pytest.mark.parametrize("memory", [1, 4])
def test_skipped_pairs_leave_state_unchanged(memory, s_factor, y_factor):
    """Step 1: a skipped pair changes nothing and reports ``θ = 1``."""
    ref = DenseLRHReference(6, memory)
    replay_pairs(ref, PAIR_SCENARIOS["negative_definite"].make(2, 6, 2))
    before = (ref.B.copy(), ref.Z.copy(), ref.sigma, ref.rank, ref.theta)
    n_steps = len(ref.steps)
    rng = np.random.default_rng(0)
    with np.errstate(invalid="ignore"):
        s = s_factor * np.ones(6) * rng.uniform(0.5, 1.0, size=6)
        y = y_factor * rng.uniform(0.5, 1.0, size=6)
    assert ref.should_skip(s, y)
    record = ref.append(s, y)
    assert record.skipped
    assert record.theta == 1.0
    np.testing.assert_array_equal(ref.B, before[0])
    np.testing.assert_array_equal(ref.Z, before[1])
    assert (ref.sigma, ref.rank, ref.theta) == before[2:]
    assert len(ref.steps) == n_steps


@pytest.mark.parametrize(
    ("multiplier", "expect_skip"),
    [(-2.0, True), (-1.6, True), (-1.0, False)],
    ids=["y=-2Bs", "y=-1.6Bs", "y=-Bs"],
)
def test_curvature_guard_late_skip(warm_reference, multiplier, expect_skip):
    """Step 4b: with the 1e-12 floor active, ``s̃ᵀy_d <= τ/2 · sᵀBs`` skips the pair.

    ``‖s‖ = 1e-7`` passes the step-1 predicate but gives ``sᵀB s < 1e-12``;
    ``y = k B s`` with ``k < 0`` then leaves ``s̃ᵀy_d`` non-positive or below
    the guard for ``k ∈ {-2, -1.6}`` and positive for ``k = -1``.
    """
    ref = warm_reference
    rng = np.random.default_rng(0)
    s = rng.normal(size=ref.n)
    s = 1e-7 * s / np.linalg.norm(s)
    y = multiplier * (ref.B @ s)
    assert not ref.should_skip(s, y)
    before = (ref.B.copy(), ref.Z.copy(), ref.sigma, ref.rank, ref.theta)
    n_steps = len(ref.steps)

    record = ref.append(s, y)

    assert record.skipped == expect_skip
    assert record.late_skip == expect_skip
    assert (record.theta == 1.0) >= expect_skip
    assert np.array_equal(ref.B, before[0]) == expect_skip
    assert np.array_equal(ref.Z, before[1]) == expect_skip
    assert ((ref.sigma, ref.rank, ref.theta) == before[2:]) == expect_skip
    assert len(ref.steps) == n_steps + (not expect_skip)
    # always an SPD model; after an update (no drop: rank 3 < memory 4) it
    # satisfies the secant condition B s = ŷ (trivially B s = B s if skipped)
    assert np.linalg.eigvalsh((ref.B + ref.B.T) / 2).min() > 0.0
    expected = (record.y_hat, ref.B @ s)[expect_skip]
    np.testing.assert_allclose(ref.B @ s, expected, atol=TOL, rtol=TOL)


# ---------------------------------------------------------------------------
# Sigma reinitialisation
# ---------------------------------------------------------------------------

SIGMA_RULES = {
    "yy_over_sy": lambda s, y: (y @ y) / (s @ y),
    "sy_over_ss": lambda s, y: (s @ y) / (s @ s),
}


@pytest.mark.parametrize("sigma_rule", list(SIGMA_RULES))
@pytest.mark.parametrize("initial_sigma", [0.5, 1.0, 40.0])
@pytest.mark.parametrize("curvature", [1e-7, 1e-2, 3.0, 1e4, 1e7])
def test_sigma_rule_clip_and_initial_sigma(sigma_rule, initial_sigma, curvature):
    """``σ_new = clip(rule(s̃, y_d))`` for both rules; empty model is ``σ₀ I``."""
    n = 5
    ref = DenseLRHReference(n, 3, sigma_rule=sigma_rule, initial_sigma=initial_sigma)
    np.testing.assert_array_equal(ref.B, initial_sigma * np.eye(n))
    assert ref.sigma == initial_sigma
    assert ref.rank == 0
    rng = np.random.default_rng(1)
    s = rng.normal(size=n)
    y = curvature * s + 1e-3 * curvature * rng.normal(size=n)
    assert not ref.should_skip(s, y)
    record = ref.append(s, y)
    expected = np.clip(
        SIGMA_RULES[sigma_rule](record.s_tilde, record.y_damped),
        ref.sigma_floor,
        ref.sigma_ceil,
    )
    np.testing.assert_allclose(record.sigma_new, expected, rtol=1e-12)
    assert ref.sigma == record.sigma_new
    assert record.sigma_old == initial_sigma
    # the complement curvature after the append is sigma_new
    complement = np.eye(n) - ref.projector()
    np.testing.assert_allclose(
        ref.B @ complement, record.sigma_new * complement, atol=TOL
    )
    assert ref.sigma_floor <= record.sigma_new <= ref.sigma_ceil


# ---------------------------------------------------------------------------
# Agreement with plain dense BFGS when nothing is dropped/damped/reinitialised
# ---------------------------------------------------------------------------


def _sigma_preserving_pairs(
    seed: int, n: int, count: int, sigma: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pairs with ``yᵀy / sᵀy = σ`` and ``y ∈ span(s_1..s_k)``.

    ``y = σ |s| cos φ (cos φ ŝ + sin φ ê)`` with ``ê ⊥ s`` a unit vector in
    the span of the earlier steps (``ê`` absent for the first pair, where
    ``y = σ s``) satisfies ``yᵀ(y - σ s) = 0``, i.e. ``yᵀy = σ sᵀy``. The
    angle ``φ ≤ 45°`` keeps ``sᵀy ≥ σ|s|² / 2``.
    """
    rng = np.random.default_rng(seed)
    pairs = []
    previous: list[np.ndarray] = []
    for _ in range(count):
        s = rng.normal(size=n)
        s_hat = s / np.linalg.norm(s)
        if previous:
            v = np.column_stack(previous) @ rng.normal(size=len(previous))
            v = v - (v @ s_hat) * s_hat
            e_hat = v / np.linalg.norm(v)
            phi = rng.uniform(0.0, np.pi / 4)
        else:
            e_hat, phi = np.zeros(n), 0.0
        y = (
            sigma
            * np.linalg.norm(s)
            * np.cos(phi)
            * (np.cos(phi) * s_hat + np.sin(phi) * e_hat)
        )
        pairs.append((s, y))
        previous.append(s)
    return pairs


@pytest.mark.parametrize(
    ("n", "memory", "sigma", "seed"),
    [
        (5, 1, 0.7, 0),
        (5, 2, 3.0, 1),
        (5, 4, 0.7, 1),
        (12, 1, 3.0, 0),
        (12, 2, 0.7, 0),
        (12, 4, 3.0, 1),
    ],
)
def test_matches_plain_dense_bfgs_when_nothing_is_dropped_or_damped(
    n, memory, sigma, seed
):
    """With no drop, damping, rejection or sigma change, ``B`` is plain BFGS.

    The pairs satisfy ``yᵀy / sᵀy = σ``, so they preserve ``σ`` only under the
    paper's rule, which is therefore passed explicitly (it is not the default).
    """
    pairs = _sigma_preserving_pairs(seed, n, memory, sigma)
    ref = DenseLRHReference(n, memory, initial_sigma=sigma, sigma_rule="yy_over_sy")
    history = replay_pairs(ref, pairs)

    expected = sigma * np.eye(n)
    for (state, record), (s, y) in zip(history, pairs):
        # the construction really avoids every special branch
        assert record.accepted
        assert not record.dropped
        np.testing.assert_allclose(record.theta, 1.0, atol=1e-12)
        np.testing.assert_allclose(record.sigma_new, sigma, rtol=1e-10)
        np.testing.assert_allclose(record.y_hat, y, atol=1e-10)
        expected = plain_bfgs_update(expected, s, y)
        np.testing.assert_allclose(state.B, expected, atol=TOL, rtol=TOL)
    assert ref.rank == memory
