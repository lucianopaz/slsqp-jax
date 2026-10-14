# G3 H1b: refine the L-BFGS per-variable diagonal (sign gate on sub-floor components)

Follows `g3_termination.md` H1 and ledger `g3-lbfgs-relfloor` (INCONCLUSIVE: branch
`hyp/termination_missed-lbfgs-relfloor` 62b99f5, tiny +19/-8, group 24 solved rows).
All measurements below are in-process: the worktree of 62b99f5 with `LBFGS.append` monkeypatched through a
`sitecustomize.py` on `PYTHONPATH` (scratch, not in the repo). The patched "base" variant (`max(s_i^2, 1e-12)`)
reproduces the attribution exactly: the tiny tier joined against the branch gives the same +19/-8 as
`regress_join.json`, so the tiny tier is deterministic and differences are due to `append` only.
Tiny tier = 273 instances x {asls-craig, asls-pcg, asls-minresqlp, pasls, trip, tfip}, `--repeats 1`.

## Why each of the 8 lost tasks loses under H1

The outcome that matters: **all 8 end at x\*** (benchmark row of the branch run: f_gap <= 1e-7, feas <= 2e-11). In 7 of 8 the
benchmark-scaled stationarity is <= 5.6e-6. The losses come from termination tests that do not fire at x\*, not from a wrong model.
In base, the crushed diagonal (`|y_i s_i| / 1e-12`) produced a different end game that happened to satisfy them.

| task | branch end (f_gap / feas / stat_scaled) | mechanism (trace of every `append`) |
|---|---|---|
| HS17 asls-craig | merit_stagnation, 3.6e-15 / 1.7e-19 / 5.4e-7 | Same iterates as base to the last digit (17 appends; appends 9-16 skipped, ‖s‖ < 1e-8). Only the last stored pair differs: s = (2.2e-5, -6.4e-9). Base clamps d_2 to clip_lo 0.02, H1 keeps 180 (true ∂²f/∂x2² = 200). Stationarity uses the QP multipliers, ∇L(x, λ_QP) = -B d on the free part, so it scales with B_22: 1.1e-6 (H1) vs 3.1e-10 (base). Base passes the KKT test only because the model is wrong. |
| HS36 trip | trust_region_collapse, 2e-9 / 0 / 3e-16 | Steps ~1e-6. H1 converges to round-off (‖s‖ 1e-13 for 35 steps) but trip's E(x,s,y,z;0) <= atol test does not pass before the TR collapses. Base reaches it at step 18 by a different path. |
| HS37 trip | trust_region_collapse, 6e-10 / 0 / 1.7e-9 | f = -x1x2x3 has a zero Hessian diagonal, so every |y_i/s_i| is cross-coupling. Components with s_i² < 1e-12 and **s_i y_i < 0** get d = |(Hv)_i/v_i| = [6, 24, 24]. The iteration then locks into a 2-cycle (s_{k+1} = -s_k, 16 pairs) and the TR collapses. |
| DIPIGRI trip | trust_region_collapse, 8.7e-8 / 0 / 2e-8 | Pair 14 (‖s‖ = 6e-7): base crushes d to 0.37 (all 7 components); H1 keeps 4..184. Then the same ± 2-cycle (s_17 ≈ -s_16). |
| HS119 trip | trust_region_collapse, 2.6e-8 / 2e-15 / 2e-10 | Same class as HS36 (converged, trip end game not declared). |
| HS86 tfip | multiplier_solve_failure, 1e-7 / 0 / 5e-9 | Converged, then the fatal multiplier flag fires (tfip end game). |
| HS49 pasls | max_steps_reached, 7.9e-9 / 9e-16 / 5.6e-6 | Degenerate (quartic/sextic) objective, sublinear tail. Base needed 478 of 500 steps, so this is budget noise. |
| HS102 asls-minresqlp | merit_stagnation, 4.6e-10 / 1.6e-11 / 5e-2 | Badly scaled (d up to 6e5). At pair 26 (‖s‖ = 2e-5, s_1 = 3e-7, y_1 = -0.07 from coupling to x7): est = 2.3e5 (H1) vs 2.2e4 (base). f is at f\*, but the multipliers are poor. |

Common thread for the model-side part (HS37, DIPIGRI, HS102): for a component below the old floor, the ratio `|y_i s_i| / s_i²` is
dominated by the coupling `Σ_{j≠i} H_ij s_j` and often has **s_i y_i < 0**. `abs()` turns that into a large positive "curvature".
The old floor hid this by attenuating such components. H1 removed the attenuation and exposed it.

## H1b claim

Keep H1 (no absolute floor in the denominator). Accept the per-component secant estimate on a component below the old floor
(`s_i² < 1e-12`) **only if its component curvature is positive**, `s_i·y_i > 0`. Otherwise keep the old `d_i`, as for `s_i² <= 1e-20`.
Components with `s_i² >= 1e-12` are untouched, so they behave exactly as in base and in H1.

**Must hold if true:** on HS37 trip the 2-cycle disappears (`d` no longer converges to `|(Hv)_i/v_i|` on sign-negative components).
On the tiny tier, losses vs base drop below H1's 8 and the H1 gains are kept.

## Change (slsqp_jax/sqpdax/secant/lbfgs.py, `LBFGS.append`, +1 line / 1 changed vs 62b99f5, no public API)

```python
s_sq = s**2
# Below the old 1e-12 floor a component ratio is dominated by cross-coupling;
# accept it only when it is a positive curvature (s_i y_i > 0).
has_signal = (s_sq > 1e-20) & ((s_sq >= 1e-12) | (y_damped * s > 0))
safe_s_sq = jnp.where(has_signal, s_sq, 1.0)
per_var_estimate = jnp.abs(y_damped * s) / safe_s_sq
```
(Use a module-private constant for 1e-12 if preferred.) The test added on the branch
(`test_append_small_steps_do_not_collapse_diagonal`, exact diagonal quadratic, so s_i y_i > 0) is unaffected. Add one case:
a pair with s_i² < 1e-12 and s_i y_i < 0 leaves `diagonal[i]` unchanged.

## Measured evidence (in-process, monkeypatched)

Tiny tier, solved vs patched base (6 L-BFGS configs, 1638 tasks):

| variant | gained | lost | net | keeps H1's 19 gains | recovers H1's 8 losses | losses far from x\* |
|---|---|---|---|---|---|---|
| H1 (branch 62b99f5) | 19 | 8 | +11 | 19/19 | 0/8 | 0 |
| **H1b (sign gate below 1e-12)** | **22** | **6** | **+16** | 16/19 | 4/8 (HS36, HS37, HS119 trip, HS49 pasls) | **0** |
| sign gate on all components | 44 | 23 | +21 | 16/19 | 6/8 | 7 (BT4 x4 pasls infeasible, MINMAXBD x3 infeasible, KOEBHELB trip, HS96 tfip, HS116 x3) |
| floor min(1e-12, 1e-4‖s‖²) | 16 | 6 | +10 | 14/19 | 5/8 | 0 |
| keep d_i where |s_i| <= 0.1‖s‖ | 50 | 44 | +6 | 15/19 | 7/8 | many (HS3 x4, PALMER2 x4, PALMER3 x4 wrong f, TRIGON1B x4) |

H1b tiny losses (all at x\*, f_gap <= 1e-7, feas <= 1e-9): HS17 asls-craig, HS102 asls-minresqlp, DIPIGRI trip, HS86 tfip
(4 of the original 8), plus HS1 tfip (secant_recovery_failure, stat 2.5e-6) and HS37 tfip (multiplier_solve_failure, stat 1.8e-11).
H1b drops 3 of H1's gains: PALMER2 trip, PALMER3B tfip, HS91 trip. It adds 6 new ones: PALMER3 trip, CSFI1 trip, HS91 tfip,
OET5 asls-minresqlp, HS116 trip, HS119 tfip.

Group (the 24 rows the branch solves in `compare.json`), solved / 24: base 0, H1 24, **H1b 23** (loses PALMER3B tfip, which ends
trust_region_collapse), sign gate on all 24, floor-min 20 (DIAGPQE x4 fail: the ‖s‖-relative floor still attenuates at n = 1000).
Lead tasks with H1b: AIRCRFTB asls-pcg 185 steps, BQPGABIM asls-pcg 21, HS38 pasls 109, DIAGPQE pasls 6, DUAL3 asls-craig 243 (all successful).

Screens on the 13 tasks (8 losses + 5 leads) were **not predictive**. Neighbouring thresholds flip HS102, DIPIGRI and HS37 (e.g. the
0.1‖s‖ mask scored 12/13 on the screen and then -44 on the tier). Judge only on the full tier.

## Cheap falsifier

`tests/sqpdax` unit case above, then the tiny tier gate on the 6 L-BFGS configs. If the gate shows more than 6 losses or any
loss with f_gap > 1e-6, the sign gate is not what removes the H1 losses: reject H1b and fall back to H1 with the waiver.

## Expected

Tiny tier +22/-6 vs base (H1: +19/-8). Group 23 of the 24 H1 rows. Every loss is a converged-but-undeclared end game.

## What H1b cannot fix inside secant/lbfgs.py (route elsewhere)

- HS17 asls-craig: the iterate is identical to base; only the QP-multiplier stationarity ‖B d‖ differs. Recovering it means
  re-crushing a correct curvature. Owner: active-set termination (stationarity measured with QP duals), stream A, not secant.
- trip/tfip at x\* with trust_region_collapse / multiplier_solve_failure (DIPIGRI trip, HS86/HS37/HS1 tfip; also H1's
  HS36/HS119 trip): complementarity end game, stream B (G2/G8). These tasks flip with any change of the end-game path.
- HS102 asls-minresqlp: f = f\* but stat_scaled 5e-2 (multipliers). Coupling-dominated scaling; no diagonal rule tested keeps it
  stable (it flips between OK and FAIL across thresholds).

## Risk to other causes

Smaller than H1: the change applies only to components with s_i² < 1e-12 and s_i y_i <= 0. Those keep the previous d_i, as base
and H1 already do for s_i² <= 1e-20. Every tiny-tier loss is NEAR x\*. Medium-tier IP rows (trip/tfip) were not run beyond the
2 tfip group rows.
