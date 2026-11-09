"""Fixtures for :mod:`slsqp_jax.sqpdax.secant` tests."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax.sqpdax.secant import LBFGS, Secant

# Compiled once per static configuration; eager ``append`` retraces its
# ``lax.cond`` on every call and dominates the runtime of history-building tests.
# Works for any :class:`Secant` (only ``hist.append`` is used, so no ``hist.n``).
append_jit = eqx.filter_jit(lambda hist, s, y: hist.append(s, y))


@pytest.fixture
def n() -> int:
    return 5


@pytest.fixture
def memory() -> int:
    return 4


@pytest.fixture
def empty_lbfgs(n: int, memory: int) -> LBFGS:
    """Fresh L-BFGS history with identity ``B₀``."""
    return LBFGS(n=n, memory=memory)


def build_lbfgs_with_pairs(
    n: int,
    pairs: list[tuple],
    *,
    memory: int = 10,
    **kwargs,
) -> LBFGS:
    """Append ``(s, y)`` pairs sequentially to a new :class:`LBFGS`."""
    hist = LBFGS(n=n, memory=memory, **kwargs)
    for s, y in pairs:
        hist = append_jit(hist, jnp.asarray(s), jnp.asarray(y))
    return hist


def explicit_diagonal(hist: LBFGS) -> jnp.ndarray:
    """Reference ``diag(B)`` by probing unit vectors."""
    dim = hist.diagonal.shape[0]
    diag = jnp.zeros(dim)
    for i in range(dim):
        e_i = jnp.zeros(dim).at[i].set(1.0)
        diag = diag.at[i].set(jnp.dot(hist.hvp(e_i), e_i))
    return diag


def dense_hessian(hist: Secant) -> jnp.ndarray:
    """Dense ``B`` assembled from ``hvp`` applied to the identity columns."""
    # ``Secant`` does not declare ``n``, but every concrete secant has it.
    return jax.vmap(hist.hvp)(jnp.eye(hist.n)).T


def random_pairs(
    seed: int, n: int, count: int, *, negative_every: int = 3
) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
    """Curvature pairs of a random SPD quadratic, every k-th one with ``sᵀy < 0``.

    ``y = H s`` for a fixed random SPD ``H`` plus a little noise; each
    ``negative_every``-th pair has its sign flipped (negative curvature).
    Pass ``negative_every=0`` for positive curvature only.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    hessian = a @ a.T / n + 0.5 * np.eye(n)
    pairs = []
    for k in range(count):
        s = rng.normal(size=n) * 10.0 ** rng.uniform(-1.0, 0.0)
        y = hessian @ s
        if negative_every and k % negative_every == negative_every - 1:
            y = -y
        y = y + 0.01 * rng.normal(size=n) * np.linalg.norm(y)
        pairs.append((jnp.asarray(s), jnp.asarray(y)))
    return pairs


def pair_with_curvature_ratio(
    hist: Secant, ratio: float, seed: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pair ``(s, y)`` with ``sᵀy = ratio · sᵀ B s`` for the model ``B = hist``.

    ``y = ratio · B s + w`` with ``w ⊥ s``, so ``ratio`` is exactly the quantity
    Powell damping compares with the threshold.
    """
    rng = np.random.default_rng(seed)
    # ``Secant`` does not declare ``n``, but every concrete secant has it.
    s = jnp.asarray(rng.normal(size=hist.n))
    Bs = hist.hvp(s)
    w = jnp.asarray(rng.normal(size=hist.n))
    w = w - jnp.dot(w, s) / jnp.dot(s, s) * s
    return s, ratio * Bs + 0.3 * jnp.linalg.norm(Bs) * w / jnp.linalg.norm(w)


# Narrow-valley quadratic (the LOGROS crawl): soft curvature along ``t``, stiff
# curvature along the normal ``n``; ``H = λ_t t tᵀ + λ_n n nᵀ``.
VALLEY_LAMBDA_T = 0.1
VALLEY_LAMBDA_N = 1.5e5


@pytest.fixture(
    params=[(15.0, 0), (45.0, 1), (75.0, 2)], ids=["15deg", "45deg", "75deg"]
)
def narrow_valley(request) -> tuple[jnp.ndarray, list[tuple]]:
    """Tangent ``t`` of a rotated narrow valley and 30 pairs crawling along it.

    Each step is ``s = a t + b n`` with ``a ∈ [0.1, 1]`` and a normal
    component ``|b| ≈ 0.05 |a|`` of random sign; ``y = H s`` exactly.
    """
    angle_deg, seed = request.param
    rng = np.random.default_rng(seed)
    theta = np.deg2rad(angle_deg)
    t = np.array([np.cos(theta), np.sin(theta)])
    nrm = np.array([-np.sin(theta), np.cos(theta)])
    hessian = VALLEY_LAMBDA_T * np.outer(t, t) + VALLEY_LAMBDA_N * np.outer(nrm, nrm)
    pairs = []
    for _ in range(30):
        a = 10.0 ** rng.uniform(-1.0, 0.0)
        b = 0.05 * a * rng.uniform(0.5, 1.5) * rng.choice([-1.0, 1.0])
        s = a * t + b * nrm
        pairs.append((jnp.asarray(s), jnp.asarray(hessian @ s)))
    return jnp.asarray(t), pairs


# ---------------------------------------------------------------------------
# Dense NumPy reference of the limited-memory reduced-Hessian (LRH) model
# ---------------------------------------------------------------------------
#
# REFERENCE CODE ONLY: it forms dense ``n x n`` matrices on purpose, so it is
# valid for tiny ``n`` and must never be used inside the code under test
# (see ``.claude/rules/memory-budget.md``). It implements PLAN.md section 2.2
# steps 1-7 in the stated order, in float64 NumPy (independent of
# ``jax_enable_x64``).


@dataclass(frozen=True)
class LRHAppendRecord:
    """Intermediate quantities of one :meth:`DenseLRHReference.append` call.

    For a skipped pair only ``skipped`` (``True``), ``theta`` (``1.0``) and
    ``late_skip`` are meaningful; every other field is ``None``.

    Attributes
    ----------
    skipped
        The pair was skipped, by step 1 or by the step-4b curvature guard.
    late_skip
        The skip came from the step-4b guard (``False`` for a step-1 skip and
        for an applied pair).
    accepted
        Step 2 accepted the new direction ``w`` (rank grew before the drop).
    dropped
        Step 7 dropped the oldest step (expanded rank was ``memory + 1``).
    theta
        Powell damping weight of step 4.
    s_tilde, y_damped, y_hat
        ``s̃``, ``y_d`` and ``ŷ = Z' Z'ᵀ y_d``.
    sigma_old, sigma_new
        Complement curvature before and after step 5.
    B_old
        ``B'_old``: the model before the append.
    B_expanded
        ``B'``: step 5 expansion with ``σ_new``.
    B_bfgs
        ``B''``: after the BFGS update of step 6, before any drop.
    """

    skipped: bool
    theta: float = 1.0
    late_skip: bool = False
    accepted: bool | None = None
    dropped: bool | None = None
    s_tilde: np.ndarray | None = None
    y_damped: np.ndarray | None = None
    y_hat: np.ndarray | None = None
    sigma_old: float | None = None
    sigma_new: float | None = None
    B_old: np.ndarray | None = None
    B_expanded: np.ndarray | None = None
    B_bfgs: np.ndarray | None = None


class DenseLRHReference:
    """Dense NumPy model of the LRH secant, implementing PLAN.md section 2.2.

    The state is the dense matrix ``B = Z M Zᵀ + σ (I - Z Zᵀ)`` plus the
    orthonormal basis ``Z`` (``n x r``) and the accepted steps whose span
    ``Z`` represents. :meth:`append` mutates the object and returns the
    :class:`LRHAppendRecord` of the call. The object exposes the same
    ``n`` / ``hvp`` as a :class:`Secant`, so it can feed helpers such as
    ``dense_hessian``-style probes.

    Parameters
    ----------
    n
        Number of variables.
    memory
        LRH memory ``m``: at most ``m`` basis directions are kept.
    skip_threshold, accept_tol, damping_threshold, sigma_floor, sigma_ceil
        As in PLAN.md section 3.3 (``1e-8``, ``1e-4``, ``0.2``, ``1e-4``,
        ``1e6``).
    sigma_rule
        ``"yy_over_sy"`` (paper, ``y_dᵀy_d / s̃ᵀy_d``) or ``"sy_over_ss"``
        (``s̃ᵀy_d / s̃ᵀs̃``).
    initial_sigma
        Initial complement curvature; the empty model is ``σ I``.

    Attributes
    ----------
    B
        Dense model matrix.
    Z
        Orthonormal basis, shape ``(n, rank)``.
    sigma
        Current complement curvature.
    rank
        Current rank ``r = Z.shape[1]``.
    theta
        Damping weight of the last non-skipped append (``1.0`` initially).
    steps
        The newest accepted steps (at most ``memory``), oldest first;
        ``span(Z) = span(steps)``.
    last
        :class:`LRHAppendRecord` of the last append (``None`` initially).
    """

    def __init__(
        self,
        n: int,
        memory: int,
        *,
        skip_threshold: float = 1e-8,
        accept_tol: float = 1e-4,
        damping_threshold: float = 0.2,
        sigma_floor: float = 1e-4,
        sigma_ceil: float = 1e6,
        sigma_rule: str = "yy_over_sy",
        initial_sigma: float = 1.0,
    ):
        if sigma_rule not in ("yy_over_sy", "sy_over_ss"):
            raise ValueError(f"unknown sigma_rule {sigma_rule!r}")
        self.n = n
        self.memory = memory
        self.skip_threshold = skip_threshold
        self.accept_tol = accept_tol
        self.damping_threshold = damping_threshold
        self.sigma_floor = sigma_floor
        self.sigma_ceil = sigma_ceil
        self.sigma_rule = sigma_rule
        self.sigma = float(initial_sigma)
        self.B = self.sigma * np.eye(n)
        self.Z = np.zeros((n, 0))
        self.steps: list[np.ndarray] = []
        self.theta = 1.0
        self.last: LRHAppendRecord | None = None

    @property
    def rank(self) -> int:
        return self.Z.shape[1]

    def hvp(self, v):
        """``B v`` (dense)."""
        return self.B @ np.asarray(v)

    def projector(self) -> np.ndarray:
        """``Z Zᵀ`` (dense)."""
        return self.Z @ self.Z.T

    def should_skip(self, s: np.ndarray, y: np.ndarray) -> bool:
        """Step 1: the skip predicate (``LBFGS.should_skip`` minus the
        relative-curvature test)."""
        with np.errstate(all="ignore"):
            s_norm = np.linalg.norm(s)
            y_norm = np.linalg.norm(y)
            sty = np.dot(s, y)
            ratio = y_norm / max(s_norm, 1e-30)
            return bool(
                (s_norm < self.skip_threshold)
                | (y_norm < self.skip_threshold)
                | (ratio > 1e8)
                | (ratio < 1e-8)
                | ~(np.isfinite(s_norm) & np.isfinite(y_norm) & np.isfinite(sty))
            )

    def _damping_theta(self, s_t: np.ndarray, y: np.ndarray, sbs: float) -> float:
        """Step 4 weight, written exactly as ``LBFGS._damping_theta``."""
        tau = self.damping_threshold
        sbs_safe = max(sbs, 1e-12)
        sty = float(s_t @ y)
        theta = (
            (1.0 - tau) * sbs_safe / (sbs_safe - sty + 1e-12)
            if sty < tau * sbs_safe
            else 1.0
        )
        return float(np.clip(theta, 0.0, 1.0))

    def _sigma_estimate(self, s_t: np.ndarray, y_d: np.ndarray) -> float:
        """Step 5: clipped reinitialisation scalar, with the fallback."""
        sty = float(s_t @ y_d)
        with np.errstate(all="ignore"):
            raw = {
                "yy_over_sy": lambda: float(y_d @ y_d) / sty,
                "sy_over_ss": lambda: sty / float(s_t @ s_t),
            }[self.sigma_rule]()
        if not np.isfinite(raw) or sty <= 0.0:
            return self.sigma
        return float(np.clip(raw, self.sigma_floor, self.sigma_ceil))

    def append(self, s, y) -> LRHAppendRecord:
        """Apply steps 1-7 of PLAN.md section 2.2 for the pair ``(s, y)``."""
        s = np.asarray(s, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # 1. skip
        if self.should_skip(s, y):
            self.last = LRHAppendRecord(skipped=True, theta=1.0)
            return self.last

        # 2. orthogonalise (classical Gram-Schmidt twice)
        Z = self.Z
        u = Z.T @ s
        w = s - Z @ u
        delta = Z.T @ w
        u = u + delta
        w = w - Z @ delta
        rho = float(np.linalg.norm(w))
        accepted = rho >= self.accept_tol * float(np.linalg.norm(s))

        # 3. projected step
        if accepted:
            Zp = np.column_stack([Z, w / rho])
            s_r = np.append(u, rho)
        else:
            Zp = Z
            s_r = u
        s_t = Zp @ s_r
        B_old = self.B.copy()  # B'_old is the current model in both cases
        sigma_old = self.sigma

        # 4. Powell damping against the current model
        sbs = float(s_t @ B_old @ s_t)
        theta = self._damping_theta(s_t, y, sbs)
        y_d = theta * y + (1.0 - theta) * (B_old @ s_t)

        # 4b. curvature guard ("late skip"): the damped pair must carry
        # positive curvature, else it is skipped like a step-1 skip. Only
        # reachable when the 1e-12 floor of step 4 is active.
        c = float(s_t @ y_d)
        if not (
            np.isfinite(c)
            and np.isfinite(theta)
            and c > 0.5 * self.damping_threshold * sbs
        ):
            self.last = LRHAppendRecord(skipped=True, theta=1.0, late_skip=True)
            return self.last

        # 5. reinitialise sigma, expand with sqrt(sigma_new)
        sigma_new = self._sigma_estimate(s_t, y_d)
        reduced = Z.T @ B_old @ Z  # R_rᵀ R_r
        if accepted:
            k = reduced.shape[0]
            expanded = np.zeros((k + 1, k + 1))
            expanded[:k, :k] = reduced
            expanded[k, k] = sigma_new
        else:
            expanded = reduced
        eye = np.eye(self.n)
        B_exp = Zp @ expanded @ Zp.T + sigma_new * (eye - Zp @ Zp.T)
        y_hat = Zp @ (Zp.T @ y_d)

        # 6. BFGS update of the model (dense form); s̃ᵀŷ = s̃ᵀy_d > 0 by 4b
        sty_hat = float(s_t @ y_hat)
        Bs = B_exp @ s_t
        B_bfgs = B_exp - np.outer(Bs, Bs) / float(s_t @ Bs)
        B_bfgs = B_bfgs + np.outer(y_hat, y_hat) / sty_hat

        # 7. drop the oldest step when the expanded rank is memory + 1
        if accepted:
            self.steps.append(s.copy())
        dropped = Zp.shape[1] == self.memory + 1
        if dropped:
            self.steps = self.steps[1:]
            Zbar, _ = np.linalg.qr(np.column_stack(self.steps))
            Pbar = Zbar @ Zbar.T
            self.B = Pbar @ B_bfgs @ Pbar + sigma_new * (eye - Pbar)
            self.Z = Zbar
        else:
            self.B = B_bfgs
            self.Z = Zp
        self.sigma = sigma_new
        self.theta = theta
        self.last = LRHAppendRecord(
            skipped=False,
            theta=theta,
            accepted=bool(accepted),
            dropped=bool(dropped),
            s_tilde=s_t,
            y_damped=y_d,
            y_hat=y_hat,
            sigma_old=sigma_old,
            sigma_new=sigma_new,
            B_old=B_old,
            B_expanded=B_exp,
            B_bfgs=B_bfgs,
        )
        return self.last


def replay_pairs(
    ref: DenseLRHReference, pairs: list[tuple]
) -> list[tuple[DenseLRHReference, LRHAppendRecord]]:
    """Append ``pairs`` to ``ref``; return a snapshot and record per append.

    Each entry is ``(state_after_the_append, record)``; the snapshot is a deep
    copy, so per-step state can be inspected after the replay.
    """
    out = []
    for s, y in pairs:
        record = ref.append(s, y)
        out.append((copy.deepcopy(ref), record))
    return out


def plain_bfgs_update(B: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Textbook dense BFGS update ``B - B s sᵀ B / sᵀB s + y yᵀ / yᵀs``."""
    Bs = B @ s
    return B - np.outer(Bs, Bs) / (s @ Bs) + np.outer(y, y) / (y @ s)


def numpy_pair_with_curvature_ratio(
    B: np.ndarray, ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy twin of :func:`pair_with_curvature_ratio` for a dense ``B``."""
    rng = np.random.default_rng(seed)
    s = rng.normal(size=B.shape[0])
    Bs = B @ s
    w = rng.normal(size=B.shape[0])
    w = w - (w @ s) / (s @ s) * s
    return s, ratio * Bs + 0.3 * np.linalg.norm(Bs) * w / np.linalg.norm(w)


def _symmetric_with_spectrum(rng: np.random.Generator, eigs: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(eigs.size, eigs.size)))
    return (q * eigs) @ q.T


def _random_step(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.normal(size=n) * 10.0 ** rng.uniform(-1.0, 0.0)


def _spd_pairs(seed: int, n: int, count: int) -> list[tuple]:
    rng = np.random.default_rng(seed)
    a = _symmetric_with_spectrum(rng, rng.uniform(0.5, 5.0, size=n))
    steps = [_random_step(rng, n) for _ in range(count)]
    return [(s, a @ s) for s in steps]


def _negative_definite_pairs(seed: int, n: int, count: int) -> list[tuple]:
    rng = np.random.default_rng(seed)
    a = _symmetric_with_spectrum(rng, -rng.uniform(0.5, 3.0, size=n))
    steps = [_random_step(rng, n) for _ in range(count)]
    return [(s, a @ s) for s in steps]


def _dependent_pairs(seed: int, n: int, count: int) -> list[tuple]:
    """SPD pairs where every third step is (almost) parallel to the previous one."""
    rng = np.random.default_rng(seed)
    a = _symmetric_with_spectrum(rng, rng.uniform(0.5, 5.0, size=n))
    steps: list[np.ndarray] = []
    for k in range(count):
        if k % 3 == 2:
            prev = steps[-1]
            c = rng.uniform(0.5, 2.0) * rng.choice([-1.0, 1.0])
            steps.append(c * prev + 1e-7 * np.linalg.norm(prev) * rng.normal(size=n))
        else:
            steps.append(_random_step(rng, n))
    return [(s, a @ s) for s in steps]


def _dependent_indefinite_pairs(seed: int, n: int, count: int) -> list[tuple]:
    """Dependent steps (as in ``_dependent_pairs``) on an indefinite quadratic.

    The step before each near-dependent one lies in the negative-curvature
    eigenspace, so the rejected direction (``s̃ = Z u ≠ s``) has ``s̃ᵀy < 0``
    and Powell damping acts on it.
    """
    rng = np.random.default_rng(seed)
    signs = np.where(np.arange(n) % 2 == 0, -1.0, 1.0)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigs = signs * rng.uniform(0.5, 3.0, size=n)
    a = (q * eigs) @ q.T
    negative = q[:, signs < 0]
    steps: list[np.ndarray] = []
    for k in range(count):
        if k % 3 == 1:
            steps.append(negative @ rng.normal(size=negative.shape[1]))
        elif k % 3 == 2:
            prev = steps[-1]
            c = rng.uniform(0.5, 2.0) * rng.choice([-1.0, 1.0])
            steps.append(c * prev + 1e-7 * np.linalg.norm(prev) * rng.normal(size=n))
        else:
            steps.append(_random_step(rng, n))
    return [(s, a @ s) for s in steps]


@dataclass(frozen=True)
class PairScenario:
    """A family of curvature pairs and the branches it is built to exercise.

    Attributes
    ----------
    make
        ``make(seed, n, count) -> [(s, y), ...]``.
    damps_by
        After this many pairs at least one Powell-damped append (``θ < 1``)
        must have happened (``10**9``: not guaranteed).
    rejects_by
        After this many pairs at least one direction must have been rejected
        by the orthogonalisation (``10**9``: not guaranteed).
    damped_rejection_by
        After this many pairs at least one append must have both rejected its
        direction and been damped (``s̃ ≠ s`` damping path; ``10**9``: not
        guaranteed).
    independent
        Steps are generic, so every pair is accepted and the oldest step is
        dropped as soon as there are more than ``memory`` pairs.
    """

    make: Callable[[int, int, int], list[tuple]]
    damps_by: int = 10**9
    rejects_by: int = 10**9
    damped_rejection_by: int = 10**9
    independent: bool = False


PAIR_SCENARIOS: dict[str, PairScenario] = {
    "spd": PairScenario(_spd_pairs, independent=True),
    "negative_definite": PairScenario(
        _negative_definite_pairs, damps_by=1, independent=True
    ),
    "dependent": PairScenario(_dependent_pairs, rejects_by=3),
    "dependent_indefinite": PairScenario(
        _dependent_indefinite_pairs, rejects_by=3, damps_by=3, damped_rejection_by=3
    ),
}


@pytest.fixture(params=list(PAIR_SCENARIOS))
def pair_scenario(request) -> PairScenario:
    """Each :data:`PAIR_SCENARIOS` entry in turn."""
    return PAIR_SCENARIOS[request.param]
