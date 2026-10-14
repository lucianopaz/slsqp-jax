# Decisions waiting for the user (unattended mode never merges a branch with a real regression)

Merged so far (no real regressions): G1 H1 null-step restoration, G6 H3 pasls mu back-off (stream A); G8 H1 v3 double projection (float64), funnel end game / noise accept, analytic Cauchy decrease, trip penalty cap (stream B). Tip of `local_bench_cleanup`: see `git log`.

| Branch (all from the merged tip or earlier) | Group gain | Tiny-tier (vs `new_run` rows) | Real losses | Notes |
|---|---|---|---|---|
| `hyp/termination_missed-lbfgs-signgate` (L-BFGS: no absolute floor + sign gate) | 25/112 | +28 / -6 (HS116 pasls is noise) | HS17 asls-craig, HS102 asls-minresqlp, HS1/HS37/HS86 tfip | all end at the solution; supersedes `hyp/termination_missed-lbfgs-relfloor` |
| `hyp/merit_stagnation-exact-hessian-shift` (Levenberg shift, exact curvature) | 10/44 | +11 / -1 | HS111LNP asls-pcg-exact (chaotic path change in a slow region) | asls-pcg-exact 229 -> 237 (+1 vs scipy-slsqp) |
| `hyp/merit_stagnation-decaying-penalty` (scipy decaying merit penalty) | 10/283 | +20 / -6 | HS23 on four asls configs, HS81 asls-pcg-exact, MINMAXBD pasls | |
| `hyp/ip_barrier_stall-funnel-projection-refinement` | 3/111 | +4 / -0 | none | only fails the G1 count (+1 vs threshold 2); suite clean |
| `hyp/unclassified-rejected-secant-pair` (stream B: rejected trial pair fed to L-BFGS in trip/tfip, asls bit-identical) | 9 (unclassified 8/46, ip_barrier_stall 1/44) | trip/tfip +14 / -7 vs the tip rows | AIRCRFTB, BT13, HS116, KOEBHELB, MINMAXBD trip; HS86, HS91 tfip | trip 163 -> 167, tfip 169 -> 172; suite 1724 passed |

Combined (`hyp/combo-waiver2` = shift + decaying penalty + sign gate on the merged tip `9473abc`):
- Tiny tier, seven configs vs `new_run`: +68 gained / -14 lost; losses: BIGGS3/HS1/HS37/HS86/POLAK5 tfip, HS102 asls-minresqlp, HS23 x4, HS68 asls-pcg, HS81 asls-pcg-exact, MINMAXBD pasls, HS116 pasls (noise). Suite 1735 passed.
- Medium union, over the previously unsolved tasks only (`diagnostics/v5_combo`; losses among tasks `new_run` solved are NOT included): asls-craig 223, asls-pcg 225, asls-pcg-exact 243 (+7 vs scipy-slsqp), asls-minresqlp 224, pasls 212, trip 172, tfip 175; sqpdax union 257 vs scipy union 254; deficit 10.
- Without the combo (merged tip only, `diagnostics/v5`): craig 212, pcg 212, pcg-exact 229, minresqlp 209, pasls 199, trip 169, tfip 174; union 250.

Decision requested: waive the listed regressions (attended rule: gain >= 3x losses holds for each branch) for some or all branches. To merge one: `orchestration/hyp.sh merge <branch-name-without-hyp/>`.

Other open decision: a scaled (relative) stationarity test for the interior-point end game (the sign gate makes E(.;0) stall at ~1e-6 for |f| ~ 30, and forced mu reductions then end in `trust_region_collapse`); this is a tolerance decision.
