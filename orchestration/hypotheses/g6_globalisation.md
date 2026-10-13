# G6 hypotheses: globalisation / scaling (cycling, merit_stagnation, step_collapse, slow_progress, local_minimum, secant_recovery_stuck)

Data: `diagnostics/v4/classified.jsonl`, `diagnostics/v4/traj/*.json`, digests `/tmp/digest_<cause>.md`.
Ledger: no entry for any G6 cause (`ledger.py list -v`); nothing below repeats a rejected attempt. Round c
(`c-rounding-level`) only changed which rejections count as secant failures; it did not touch the penalty, the QP
Hessian or the proximal parameter.

Scope here: line-search configs only. Of 283 G6 tasks, 116 are asls/pasls (`trip`/`tfip` are covered elsewhere). Rows that scipy-slsqp solves, per cause (asls-pcg-exact / asls-pcg / craig / minresqlp / pasls):
merit_stagnation 12/0/0/0/23, cycling 0/6/6/5/6, step_collapse 3/2/2/3/0, slow_progress 0/2/2/4/3,
secant_recovery_stuck 0/2/3/3/0, local_minimum 3/1/1/1/1. Set used below: 89 distinct (problem, config) pairs, y0 index 0, n <= 200.

## What the traces show (D2)

Scratch probes ran one step of the QP at the stalled iterate, using in-process monkeypatches only. No repo file was changed.
Scripts are in the session scratchpad: `probe2.py`, `exp.py`.

1. **merit_stagnation is a deterministic repeat, not a slow decrease.** Every scipy-solved merit_stagnation row
   ends with `ls_fail` 7..10 and `alpha = 0` at a feasible point (`feas` <= 8e-7). After a rejection, x, λ, H and the
   warm working set are all unchanged, so the next QP returns the same `d`. Examples: PALMER2 asls-pcg-exact gives
   `dnorm = 166` in 10 consecutive steps; HS23 pasls gives `dnorm = 0.501` in 10 steps. The patience of 10 then ends the run.
   pasls suppresses the secant step channel (`_secant_reset_signals`), and asls-pcg-exact has no secant, so
   nothing changes between the attempts.
2. The rejected directions are not descent directions for the L1 merit. There are three types (probe at the stall):
   - indefinite exact Hessian, `KKT reason = residual_floor` (negative-curvature freeze):
     `∇fᵀd > 0` on PALMER2 (+1.03e5, |d| = 166), PALMER8A (+1.43e4, |d| = 48), PALMER7E (+0.16, |d| = 46), all asls-pcg-exact;
   - huge Han–Powell penalty: the merit is dominated by `ρ‖c⁺‖₁`. ρ is far above the current multipliers
     (`ρ = max_k 1.1‖λ_k‖∞`, monotone): HS56 8.8e16, BT9 1.7e255, HAIFAM 4.9e11, BT7 4.6e5,
     HS106 2.7e4, CSFI2 2.5e3, BT1 5.5e3. In step_collapse, Armijo accepts only α ≈ 1e-3..1e-6 for 500 steps.
   - pasls proximal equality residual `A_E d + c_E = μ(λ − λ_k)` times a large ρ: HS69 ‖A d + c‖ = 0.75, ρ = 890, ∇fᵀd = +43.6;
     HS62 0.049 x 1.25e4 = 611 > |∇fᵀd| = 331; BT7 0.144 x 1580.
3. Not addressed below, recorded as leads:
   - The QP returns a step that violates its own linearisation and still reports `working_set_converged` /
     success when the round-b cap (`|W| ≤ n − meq`) is full and no exchange is done (`ThresholdWorkingSetPolicy.propose`, `cap_rows` never
     drops a kept row). Cases: HS23 pasls `max(c+Jd) = +1.0` with W = {rows 1,2} (c = −1.0, −1.05) while row 4 (c = −3e-9) is excluded;
     HS114 exact 0.38 with |W| = 7 = n − meq; also budget-exhausted solves (HS106 pasls 1.1e5, DEGENQP 16, HS98 box 0.93).
     None of H1–H3 fixes HS23, HS97, HS98, HS106 or DEGENQP.
   - 9 tasks end at f = f* and are feasible, but stop with `merit_stagnation` / `secant_recovery_failure`: OPTCNTRL x4,
     HS87/HS102/GOULDQP1/DUAL4 pasls, HS119 asls-pcg-exact. This is a G3 termination issue: the multipliers are bad, x is correct.
   - cycling/slow_progress (LOGROS, PALMER1A/3A/6A/8E, HS268): α = 1 is accepted for 500 steps. H1–H3 solve none of
     them. Secant-scaling territory (see G3 H1).

## Measured effect (monkeypatch, 500 steps, solved = success status & rel f-gap ≤ 1e-4 & feas ≤ 1e-6)

| Variant | asls set (58) | pasls set (31) | Gained |
|---|---|---|---|
| base (HEAD) | 0 | 0 | (matches the benchmark: all 89 unsolved) |
| H1 Powell penalty | 5 | 3 | BT1, BT7 (exact); CSFI2 x3 (pcg/craig/minresqlp); BT13, HS102, GOULDQP1 (pasls) |
| H2 Hessian shift after rejection | 5 | 0 | PALMER2, PALMER7E, PALMER8A, HS114, HS119 (all exact) |
| H1 + H2 | **12** | n/a | adds HS101 minresqlp and HS56 exact to the union |
| H3 proximal μ back-off | n/a | 3 | HS62, HS119, GOULDQP1 |
| H1 + H3 | n/a | **6** | HS69, BT13, HS87, HS102, DUALC5, HS119 |

There were no losses inside the set. The tiny-tier regression is not measured (see the risks).

---

## H1 (rank 1): monotone Han–Powell penalty → Powell's decaying penalty (scipy SLSQP's rule)

**Claim.** `ActiveSetLineSearchMinimiser._next_merit_penalty` (`minimiser/active_set_linesearch.py:882-906`) only
ratchets up: `ρ_{k+1} = max(ρ_k, 1.1‖λ_{k+1}‖∞)`. A transient multiplier spike (inexact or early QP) then fixes ρ
orders of magnitude above `‖λ*‖`. On curved constraints the second-order violation term dominates the Armijo test:

    φ(x+αd) − φ(x) ≈ α D(φ;d) + ρ α² ‖½ dᵀ∇²c d‖₁   ⇒   accepted α ≲ |D| / (ρ ‖dᵀ∇²c d‖₁),

so α shrinks as 1/ρ (step_collapse). At a feasible point the merit cannot see objective progress (merit_stagnation).
scipy SLSQP (Kraft 1988, after Powell 1978) uses `ρ_i ← max(|λ_i|, ½(ρ_i + |λ_i|))`, which decays.

**Must hold if true.** At the stall, `ρ_k / (1.1‖λ_k‖∞) ≫ 1` (≥ 10), and after the change ρ falls geometrically
(×½ per step) towards `1.1‖λ‖∞` with α → 1 within about `log2(ρ/‖λ‖)` steps.

**Evidence.** ρ at the end: HS56 8.8e16, BT9 1.7e255, HAIFAM 4.9e11, BT7 exact 4.6e5, HS106 2.7e4, BT1 5.5e3, CSFI2 2.5e3 (α 9.8e-4 x 500 steps).
Monkeypatch: CSFI2 asls-pcg 500 → 19 steps (f = 55.017606 = f*), BT7 exact 500 → 41, BT1 exact 500 → 265, BT13 pasls stagnation → success in 39.

**Change.** `slsqp_jax/sqpdax/minimiser/active_set_linesearch.py`, about 3 lines in `_next_merit_penalty` (plus the
"Monotone" wording in the class docstring and the comment at line 246):

    candidate = self.penalty_multiplier_margin * _inf_norm(penalty_multipliers)
    updated = jnp.maximum(
        jnp.maximum(candidate, 0.5 * (self.merit_penalty + candidate)),
        self.initial_penalty,
    )

`_advance_dynamics` already re-evaluates `best_merit` at the current ρ, so a decrease is handled. Test:
`tests/sqpdax/minimiser/test_active_set_linesearch.py:116` asserts monotonicity (5.5 → expect
`max(0.55, ½(5.5+0.55), 2.0) = 3.025`); rename the test and adjust it. Total about 3 source lines plus 3 test lines, 2 files.

**Cheap falsifier.** CSFI2 asls-pcg and BT7 asls-pcg-exact with the change: if ρ does not fall below 1e2 within 20
steps, or the runs still reach 500 steps with α ≤ 1e-3, H1 is false.

**Expected.** 8 of 89 measured (asls-pcg-exact 2, asls-pcg 1, craig 1, minresqlp 1, pasls 3). Group G1 gate: pass.

**Risk.** It touches every asls/pasls step. A non-monotone penalty can cycle (Chamberlain's example). The tiny-tier regression
is the decisive gate. The restoration fallback uses `ρ_R = max(ρ, …)` (G1), and a lower ρ makes the restoration Armijo
weaker on infeasible starts, so watch `restoration_failure`/`infeasible` rows. Overlaps with G3-H3 (SOC for Maratos):
measure H1 first, because it is the 3-line alternative on the same step_collapse rows. G1-H1 edits the same file (merge conflict only).

---

## H2 (rank 3 by benefit per line): after a line-search rejection, shift the QP Hessian by δ_j I (Levenberg)

**Claim.** A rejected direction at a feasible point is recomputed identically (same x, λ, H, working set), so the run
spends its 10-step patience on one bad `d`. With exact curvature the bad `d` comes from an indefinite reduced
Hessian (projected CG freeze, `residual_floor`) and is an ascent direction. Raising δ until `Zᵀ(H+δI)Z ≻ 0` makes
the QP step a descent direction. As δ → ∞ it tends to the projected-gradient step `−Z Zᵀ∇f / δ`.

**Must hold if true.** For the shifted step, `∇fᵀd = −dᵀ(H+δI)d − λ_Wᵀ A_W d < 0` once `δ > −λ_min(ZᵀHZ)`. Observed now:
∇fᵀd = +1.03e5 (PALMER2), +1.43e4 (PALMER8A), +0.16 (PALMER7E), each with |d| ≥ 46.

**Change.** `slsqp_jax/sqpdax/minimiser/active_set_linesearch.py`, about 30 lines: a private
`_ShiftedActiveSetSubProblem(ActiveSetSubProblem)` with a `delta` field that overrides `kkt_mvp_primal`
(`+ δ d`) and `with_active_set` (keeps δ). `_init_subproblem` wraps the subproblem with

    j = consecutive_ls_failures;  δ_j = 1e-2 · max(1, ‖∇f‖∞) · 10^(j−1) if j ≥ 1 else 0.

This is the tested variant. No public API changes. `ProximalActiveSetSubProblem` is rebuilt inside the proximal solver, so δ is
dropped for pasls (no pasls gains were seen; leave it so). Add one test in `tests/sqpdax/minimiser/test_active_set_linesearch.py`
(an indefinite quadratic with a rejected step gives a descent step on the retry).

**Cheap falsifier.** PALMER2 asls-pcg-exact: on the first retry after the rejection, ∇fᵀd must be < 0 and the step
accepted within 3 retries (monkeypatch: success in 7 steps, f = 3651.0975 = f*). If ∇fᵀd stays > 0 for δ up to 1e3·‖∇f‖∞, H2 is false.

**Expected.** 5 alone (all asls-pcg-exact), 12 together with H1 on the asls set (adds HS56, HS101).

**Risk.** It is active only after a rejection at a feasible iterate (`ls_failures` counts only when feasible), so runs without
rejections are unchanged. In secant configs it interacts with the secant reset (`step_streak`): no gains or losses were seen there. If the tiny tier
regresses on secant configs, gate δ to exact curvature. G3-H3 (SOC) also edits this file.

---

## H3 (rank 2): pasls backs off the proximal parameter after each rejection

**Claim.** The pasls step satisfies `A_E d + c_E = μ(λ − λ_k)` with `μ = clip(res^τ, 1e-6, 0.1)`, where `res` includes the stationarity
at the stale centre `λ_k`, so μ stays at `μ_max = 0.1` while `res` is large because of λ_k (kkt 300+). The first-order L1 merit change
is bounded above by

    D(φ;d) ≤ ∇fᵀd + ρ‖μ(λ − λ_k)‖₁,

and with ρ ≈ 1e3..1e4 this is positive. A rejection moves the centre only a little (HS69: kkt 356 → 177 in 10 rejections),
so the patience runs out. If μ is cut on each rejection, the pasls step tends to the plain SQP step. asls-pcg solves HS23,
HS62, HS69, BT13, HS87, HS97, HS119, HS114, HS102, DUALC5, DEGENQP, DUAL4, MATRIX2, and pasls fails all 13 of them.

**Must hold if true.** At the stall, `ρ μ ‖λ − λ_k‖₁ > |∇fᵀd|`: HS62 611 > 331; HS69 668 vs ∇fᵀd = +43.6.

**Change.** `slsqp_jax/sqpdax/minimiser/proximal_active_set_linesearch.py::_init_subproblem`, about 3 lines:
`residual ← residual · 1e-2^j` with `j = self.consecutive_ls_failures`, so `μ ≈ μ_prev · 10^(−j)` for τ = 0.5,
floored at `mu_min`. This is the tested variant. Add one test in `tests/sqpdax/minimiser/` (a rejected step lowers the
next `prox_mu`).

**Cheap falsifier.** HS62 pasls: `prox_mu` ≤ 1e-3 within 2 rejections and an accepted step by the third. Monkeypatch:
merit_stagnation at step 17 → success in 45 steps. If the step is still rejected with μ = 1e-6, H3 is false.

**Expected.** 3 alone; 6 together with H1 (HS69, BT13, HS87, HS102, DUALC5, HS119). HS62 was solved by H3 alone but not by H1+H3,
so check the order of the two.

**Risk.** pasls only, and only after a rejection. It weakens the stabilisation on degenerate equality problems (the reason pasls exists),
so watch for pasls tiny-tier losses on DEGENQP-like / rank-deficient `A_E` rows. A smaller μ after a rejection could also
bring back KKT solver failures on degenerate equalities.

---

## Order of measurement
H1 first (3 lines, largest measured gain, affects 5 configs). Then H3 on top of H1 (pasls). Then H2 on top of H1 (asls-pcg-exact).
Upper bound if all three are kept: +18 of 89 scipy-solved G6 asls/pasls tasks
(asls-pcg-exact +8, which would bring its −9 gap to scipy-slsqp to about −1; pasls +6; asls-pcg +1; craig +1; minresqlp +2).
