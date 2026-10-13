# G3 hypotheses: termination / secant (termination_missed, secant_recovery_premature, merit_stagnation_near_opt, loose_tolerance)

Data: `diagnostics/v4/classified.jsonl`, `diagnostics/v4/traj/*.json`, `results/new_run/results.jsonl`.
Group: 69 + 17 + 13 + 13 = 112 tasks (scipy-slsqp solves 54 + 8 + 13 + 0 = 75).

## Why round c is not repeated

Round c (`c-kkt-before-fatal`, `c-rounding-level`, both ACCEPTED, 10/115) changed *when* the KKT test and the
secant streak counters fire. The v4 trajectories show that the test is not missed. The iteration stops
converging: in the line-search family, 61/61 `max_steps_reached` rows have `accept_rate` ~1, `rejected_tail` = 0,
`last_qp_converged`, `kkt_scale` = 1, and `kkt_last` flat for 300 or more steps. The decisive control is the same algorithm (same QP, L1
merit, Armijo, termination test) with the exact Hessian. `asls-pcg-exact` solves every problem that L-BFGS
`asls-*`/`pasls` miss in this group: AIRCRFTB 10 steps, BQPGABIM 1, DIAGPQE 3, DUAL3 1, HS26 18, HS47 17,
PALMER2B 15 (vs 500 / 500 / 222 / 281 / 500 / 500 / 78 with `asls-pcg`). It appears in only one G3 row
(HS268 merit_stagnation_near_opt). So the cause is the secant model, not the termination test, the merit or the
line search. No tolerance changes are proposed below.

`loose_tolerance` (13): all `successful`, scipy-slsqp and trust-constr both fail them too and reach the same f as
sqpdax (HS88/89/90 1.36266 vs catalogue 1.3601, HS57 0.03065 vs 0.02846, DGOSPEC matches trust-constr). This is a reference /
local-minimum artefact (G7, benchmark frozen). No sqpdax change proposed, 0 scipy-solved tasks at stake.

---

## H1 (rank 1): the per-variable L-BFGS diagonal update has an absolute floor `max(s_i², 1e-12)` that crushes `B₀` once steps are about 1e-6. The model then over-steps, Armijo returns α ≈ 2⁻¹⁵..2⁻⁸, and `‖s‖ < 1e-8` locks the corrupted diagonal in place.

**Claim.** `slsqp_jax/sqpdax/secant/lbfgs.py::LBFGS.append` sets

    d_i ← clip(|y_i s_i| / max(s_i², 1e-12), 1e-2·γ, 1e2·γ),   γ = yᵀy / yᵀs,

and keeps the old `d_i` only when `s_i² ≤ 1e-20`. For `1e-10 < |s_i| < 1e-6` the estimate is the true curvature
times `s_i²/1e-12 ≪ 1`, so `d_i` falls to the clip floor `1e-2·γ`, up to 100x too small. The QP step `d = −B⁻¹∇L` then grows by up to 100x.
Armijo accepts only `α ≲ 2·|∇φᵀd| / (dᵀ∇²φ d)`, so the next `‖s‖ = α‖d‖` is < 1e-8. `should_skip` (absolute
`skip_threshold = 1e-8`) then drops every later pair and the corrupted `B₀` is never repaired. The accepted steps
never trigger secant recovery (`secant_recovery_stage = −1`, accept rate 1), so no reset happens either.

**Must hold if true.** At the collapse step `k`:
`‖s_{k-1}‖ ≲ 1e-6` and `‖d_k‖ / ‖d_{k-1}‖ ∈ [10, 100]` (up to the 1e2 clip ratio), then
`α_j‖d_j‖ < 1e-8` for most `j > k`.

**Evidence** (trajectory scan, scratch script; `s_pre = α‖d‖` of the step before collapse):

| task | s_pre | ‖d‖ before → after | tail share with ‖s‖<1e-8 |
|---|---|---|---|
| AIRCRFTB asls-pcg/craig | 2.3e-8 | 1.2e-5 → 1.9e-4 (16x) | 1.00 |
| BQPGABIM asls-pcg/craig | 3.4e-8 | 2.2e-6 → 2.5e-4 (110x) | 1.00 |
| BQPGABIM pasls / minresqlp(MSNO) | 4.6e-8 / 4.2e-8 | 3e-6 → 1.5e-4 / 1.3e-6 → 1.1e-4 | 1.00 / 0.52 |
| BQPGASIM minresqlp / pasls | 4.5e-8 / 3.6e-8 | 2.9e-6 → 2.7e-4 / 1.2e-6 → 9.2e-5 | 1.00 |
| HS38 pasls | 9.3e-8 (step 112 had kkt 1.15e-6, f 9.6e-15) | 2.2e-7 → 8.6e-5 | 1.00 |
| PALMER2B asls-pcg/craig (secant_recovery_failure) | 1.8e-7 | 1.8e-7 → 8.8e-6 (49x) | — |
| DIAGPQE x4 (MSNO, n=1000) | 6.7e-8..2.5e-7 | 4e-6..1.6e-5 → 7e-4..1.5e-3 (~100x) | — |
| DUAL3 asls-craig/pcg (MSNO) | 7.6e-8 / 5.5e-8 | 2.4e-6 → 3.8e-4 / 1.7e-6 → 6.7e-5 | — |
| OPTPRLOC asls-pcg, HAIFAM minresqlp (SRP) | 2.8e-8 / 8.2e-7 | → 7.2 / → 3e26 | — |

Isolated reproduction (no solver; `LBFGS(n=4, memory=10)`, 10 pairs from `H = diag(1, 10, 1e2, 1e3)` at
‖s‖ ≈ 1e-2, then 3 pairs at the stated scale, exact `y = Hs`):

| ‖s‖ of last pairs | diagonal | ‖H_LBFGS g‖ / ‖H⁻¹g‖ |
|---|---|---|
| 1e-4 | [4.9, 10, 100, 1000] | 0.455 |
| 1e-6 | [4.9, 4.9, 9.3, 8.2] | 6.9 |
| 1e-7 | [0.09, 0.09, 0.09, 0.09] | 371 |
| 1e-9 | unchanged (skipped) | — |

Exact pairs of a quadratic are enough to corrupt the model 15x to 800x, so this is not noise in `y`.

**Change.** Edit one file, `slsqp_jax/sqpdax/secant/lbfgs.py`, about 10 to 15 lines. Use module-private constants and add no public API.
1. In `append`, mark the components that carry signal relative to the step,
   `has_signal = s_i² > r²·‖s‖²` with `r = 1e-3` (or `sqrt(eps)`). Divide by `s_i²` only on those components
   (`jnp.where(has_signal, s_sq, 1.0)`) and keep the old `d_i` elsewhere. Drop the absolute `1e-12` / `1e-20`.
2. The same scale bug appears in three other places. In `_damping_theta`, `sTB0s_safe = max(sᵀB₀s, 1e-12)` and the `+1e-12` in the denominator bias θ once
   `‖s‖² · d ≲ 1e-12`. Replace them with `finfo.tiny` or the relative floor `eps·‖s‖·‖B₀s‖`. Make the `yTs > 1e-12` test in
   `append` (curvature_scale) relative, `yᵀs > eps·‖y‖‖s‖`. Make `pair_ok = sTy > 1e-12` in
   `inverse_hvp` relative per pair as well; `hvp` already keeps those pairs, so the two products disagree today.
3. Leave `skip_threshold = 1e-8` unchanged in H1. That limits the blast radius (see H2).
`tests/sqpdax/secant/test_lbfgs.py` needs one new parametrised case: a pair with ‖s‖ = 1e-7 from a diagonal
quadratic must not reduce `diagonal` below the true curvature divided by 10. The existing `s = e₁` tests keep their meaning,
because zero components still keep the old `d_i`.

**Cheap falsifier** (before any benchmark run). (a) The reproduction script above, run with the patch: the diagonal for
‖s‖ = 1e-6 and 1e-7 must stay within 10x of [1, 10, 100, 1000]. (b) Rerun 5 tasks: AIRCRFTB asls-pcg,
BQPGABIM asls-pcg, HS38 pasls, DIAGPQE pasls, DUAL3 asls-craig. If ‖d‖ still jumps ≥ 10x right after a step with
‖s‖ < 1e-6, or the runs still end at 500 steps with α < 1e-3 in the tail, H1 is falsified.

**Expected.** Strong signature: termination_missed 10 (AIRCRFTB 2, BQPGABIM 3, BQPGASIM 2, HS38 pasls 1,
PALMER2B 2) and merit_stagnation_near_opt 8 (BQPGABIM 1, DIAGPQE 4, DUAL3 2, PALMER2B 1), all solved by scipy-slsqp.
Weaker or partial signature: PFIT1LS x4 and DEVGLA1B x3 (tail share 0.96 to 1.0 with ‖s‖ < 1e-8, onset differs), HS47 x4
and HS26 x4 (‖s‖ ≈ 2e-5 to 4e-5 over n = 3 to 5, so single components below 1e-6 are likely; same exact-Hessian control),
OPTPRLOC and HAIFAM (secant_recovery_premature), PALMER2B/3B/4B tfip (final ‖d‖ ≈ 2.5e-8).
**Estimate: 12 to 18 of 112 solved** (about 70% of the 18 strong tasks plus a few weak ones).

**Risk to other causes.** Every L-BFGS config is affected (all except asls-pcg-exact), including trip and tfip. The change is
monotone towards the correct curvature, and the falsifier table shows no other effect on pairs with ‖s‖ ≥ 1e-4.
Possible losses: tasks that currently solve *because* the crushed diagonal gives long steps. This is most likely in G6
`step_collapse` / `slow_progress`, which probably share this signature and may gain. Run the tiny-tier gate (PROTOCOL I6 and I7) for all 6 L-BFGS configs.

---

## H2 (rank 2, only if H1 leaves the tails locked): the absolute skip `‖s‖ < 1e-8 or ‖y‖ < 1e-8` freezes a stale model near x*

**Claim.** `LBFGS.should_skip` uses `skip_threshold = 1e-8` on ‖s‖ and ‖y‖. Near a solution with ‖x*‖ = O(1),
any step shorter than 1e-8 is skipped, even though its `y` is well above round-off:
`‖y‖ ≈ λ‖s‖ ≫ eps·‖∇L‖`. After any transient mis-scaling (H1, or a damped pair) the model cannot recover by
small steps, and secant recovery never fires because the steps are accepted.

**Must hold if true.** In the tail, `‖s_j‖ < 1e-8` but `‖y_j‖ / (eps · max(1, ‖∇L(x_j)‖)) ≫ 1`, i.e. at least 1e3. The pairs carry
information and are discarded.

**Evidence.** The tail share with ‖s‖ < 1e-8 is 1.00 on AIRCRFTB x2, BQPGABIM x3, BQPGASIM x2, HS38 pasls, PFIT1LS x4 and DEVGLA1B x3,
for hundreds of steps with f still decreasing slowly (AIRCRFTB 3.21e-12 → 2.98e-12 over 380 steps, so the
direction is still a descent direction and `y` is not noise). PFIT1LS and DEVGLA1B have no H1 onset (s_pre about 1e-3 to 1)
yet lock in the same way.

**Change.** In `slsqp_jax/sqpdax/secant/lbfgs.py` (about 5 lines), plus the caller in `minimiser/base.py::_update_secant` if
the scale must be passed (about 3 lines, private argument). Replace the absolute test with a noise test:
skip if `‖s‖ ≤ eps·max(1, ‖x‖)` or `‖y‖ ≤ c·eps·max(1, ‖∇L‖)` with `c` about 1e2. The existing relative-curvature
and extreme-ratio tests stay. `skip_threshold` stays as a parameter; only its default use changes, or it becomes a
floor of 0.

**Cheap falsifier.** Count skipped appends (`SecantStatistics` already records `skipped`) on PFIT1LS asls-pcg and
DEVGLA1B asls-pcg. If the tail is not skip-dominated (fewer than 50% of appends skipped), H2 is falsified.

**Expected.** PFIT1LS x4 and DEVGLA1B x3 (all solved by scipy-slsqp), plus a backstop for the H1 set: **3 to 7 additional tasks**.

**Risk.** Noisier pairs can enter on problems where y is at round-off, such as f* = 0 least squares with
‖∇L‖ ≈ 1e-10. Damping and the relative-curvature test remain the guards. This is higher risk than H1 because it
touches every secant run's history length, so it needs the full tiny-tier gate. Do not combine it with H1 in one measurement.

---

## H3 (rank 3, lowest confidence): Maratos effect on equality-constrained degenerate problems needs a second-order correction (SOC)

**Claim.** On HS47 and HS26 (both have meq ≥ 1 and quartic objectives, so f* = 0 is a degenerate minimum), the full step's
constraint error `½ dᵀ∇²c d` multiplied by ρ exceeds the L1-merit decrease. Armijo then accepts only α ≈ 2e-3 to 6e-2 for
400 or more steps.

**Must hold if true.** `α_accepted ≲ |∇φᵀd| / (ρ · ½ |dᵀ∇²c d|)`, and `φ(x + d + d_soc) ≤ φ(x) + c₁∇φᵀd` with
`d_soc = −J_W⁺ c_W(x + d)`.

**Evidence.** HS47 asls-pcg: ρ = 5.64, merit − f = 7e-8 ≈ ρ·feas (feas 1.25e-8 to 2.2e-8, flat), ‖d‖ ≈ 5e-3,
kkt ≈ 1e-4, so the bound gives α ≲ 1.7e-3. Observed α = 3.9e-3 for 450 steps. HS26 asls-pcg: ρ = 1.09, feas between 1e-8 and 3.5e-7,
α between 5e-4 and 6e-2. **Contrary evidence:** asls-pcg-exact uses the same merit and line search and solves both in
17 and 18 steps. Maratos does not depend on the Hessian, so the L-BFGS step (H1: per-component floor with
‖s‖ ≈ 2e-5 over n = 3 to 5) is the more likely cause. Test H1 on these 8 tasks first.

**Change.** In `slsqp_jax/sqpdax/minimiser/active_set_linesearch.py`, about 40 to 60 lines. After `_assess_direction`, if
α < 1 and the working set or the equalities are non-empty, try `x + d + d_soc` once against the same Armijo bound. Use the
`lstsq` min-norm correction from `ctx.lagrangian` Jacobians. Accept the point with α = 1 if it passes. No public
API (private method).

**Cheap falsifier.** If H1 alone makes HS47 and HS26 converge on asls-pcg, drop H3. Otherwise evaluate the SOC test
offline at 3 stalled iterates of HS47 asls-pcg: if `φ(x + d + d_soc)` also fails Armijo, Maratos is falsified.

**Expected.** HS47 x4 and HS26 x4 (scipy-solved), possibly HS68 (MSNO). **0 to 8 tasks; most likely 0 after H1.**

**Risk.** One extra merit evaluation per backtracked step. SOC can move bound-active variables outside their bounds unless it is clipped.
Affects only the line-search family.

---

## Not covered

- IP rows where complementarity dominates (MAKELA3 trip/tfip, DUAL2/DUAL4 tfip, OET5 tfip, HS71 trip:
  `max sᵢzᵢ` ≈ kkt_last ≈ 30 to 100 times μ after the forced μ reductions of round d). This is the end game in G2/G8, not secant.
- BIGGS5 x6, HS57 x4, POLAK2 x3, HS105, HS25: scipy-slsqp fails as well (not deficit tasks).
- secant_recovery_premature HS98 x3, OET5 x3, TENBARS3/4: kkt 0.03 to 3.5, far from optimal, probably G6.

## Summary
Ranking: H1 (L-BFGS per-component floor `max(s_i², 1e-12)`, 1 file, about 12 lines, 12 to 18 tasks) > H2 (relative skip test, 3 to 7) > H3 (SOC, 0 to 8, probably absorbed by H1).
Most informative falsifier: rerun AIRCRFTB asls-pcg, BQPGABIM asls-pcg, HS38 pasls, DIAGPQE pasls and DUAL3 asls-craig with H1. ‖d‖ must no longer jump ≥ 10x after a step with ‖s‖ < 1e-6, and the runs must end `successful`.
Not verified: per-step secant diagonal and skip counts inside real runs (the trajectories do not record them; the mechanism is inferred from ‖d‖ jumps and reproduced in isolation); whether trip and tfip hit the same path; HS47/HS26 per-component step sizes.
