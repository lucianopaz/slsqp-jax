# Work queue

Status: `open` | `in-progress` | `done` | `blocked`. Edit by hand or at the end of a session (step R3).
Objective: every sqpdax config solves more instances than the scipy baselines (frozen). Scope: `slsqp_jax/sqpdax/**` only (see PROTOCOL.md). Current gap: `python orchestration/gap.py results/new_run/results.jsonl` (sqpdax union 248 vs scipy union 254; best config asls-pcg-exact 227 vs scipy-slsqp 236).

Baseline for all items: the diagnosis of `results/new_run/results.jsonl` (to be created by Q1 in
`diagnostics/v4`, `--previous diagnostics/v2/classified.jsonl`). Until Q1 exists, group sizes
below come from `diagnostics/v2` and are stale.

| ID | Item | Status | Notes |
|---|---|---|---|
| Q0 | Housekeeping: track the tooling files | done | Already tracked at `1e2ef9d`. |
| Q1 | Diagnose `results/new_run` | done | `diagnostics/v4/{classified,diag}.jsonl`, `report.html` (740 tasks, 33 min). Narrative counts refreshed, `PRIORITIES` re-sorted by scipy-solved count: G6 127, G3 76, G1 39, G2 33, G4 11, G5 8. New group G8 below. 451 fixed, 33 regressed vs v2 (different machines). |
| Q2 | Settle round d (`ip_barrier_stall`) | done | User decision Oct 13: keep round d. Losses attributed (ledger `d-attribution-20261013`): at `mu` ~ 1e-7 the end game fails (kkt just above atol, multiplier solve, radius collapse). Open lead for G2: HS55 tfip false success (f=6.667 vs f*=6.333, 16 steps), HS30/PRICE4B/HS119 near-solved. |
| Q3 | Tiny-tier baseline | dropped | Replaced by the tiny-tier rows of `results/new_run/results.jsonl` (user decision, Oct 13). G2 of every `G*` joins against those rows; `regress_join.py` may need a `--tier tiny` filter or a filtered copy, check in Q2. |
| G1 | Infeasible starts, over-determined working sets | partial | Accepted: null-step restoration trigger (asls, 9/99). Rejected: joint merit (trip), NaN rho, CG budget for rho = -inf (correct but solve nothing). Open: DUALC family (infeasible start, row scaling of the normal step: design only, see `hypotheses/g8_dualc_design.md` if written), `kkt_solver_failure` Craig/MINRES-QLP on rank-deficient equality Jacobians (MSS1, FLT, TABLE7, DIAGPQB/T), `near_feasible`. |
| G2 | Interior-point stall remainder | partial | Accepted (stream B): penalty cap/decay for trip (d984abc), funnel end game. Open: OET2/4/6/7 end game, `ip_bound_violation`, scaled stationarity test (tolerance decision for the user: the L-BFGS sign gate makes E(.;0) stall at ~1e-6 on |f|~30 so forced mu reductions end in `trust_region_collapse`). |
| G3 | Termination / secant | partial | Waiver pending: `hyp/termination_missed-lbfgs-signgate` (H1b: 25/112 group, tiny +28/-6 of which 1 noise; suite clean) supersedes `hyp/termination_missed-lbfgs-relfloor`. Remaining losses end at the solution (HS17: stationarity measured with QP multipliers; tfip end game). |
| G4 | Funnel y-iteration deadlock | done (stream B) | funnel Cauchy slope = ||proj g||^2, noise-accepted f-iterations, analytic dm_cg (float64): multiplier_failure 5/13, tfip +7/-0 on tiny. HS20 is a local minimum. |
| G5 | `qp_kkt` success guard | skipped | Not started by user decision (12 tasks, low yield). |
| G6 | Globalisation and scaling | partial | Accepted: pasls mu back-off (+4, pasls 193 -> 197). Waiver pending: decaying merit penalty (`hyp/merit_stagnation-decaying-penalty`, 10/283, tiny +20/-6) and Levenberg shift for exact curvature (`hyp/merit_stagnation-exact-hessian-shift`, asls-pcg-exact 229 -> 237, +1 vs scipy-slsqp, tiny +11/-1 HS111LNP chaotic path change). QP cap without exchange: Goldfarb-Idnani exchange measured (`hyp/merit_stagnation-working-set-exchange`, waiver list; HS97 false success to investigate). Open leads: HS106 (QP budget exhausted, not a blocked set), cycling/slow_progress (PALMER*A/E, LOGROS, HS268: secant scaling). |
| G7 | Artefacts and performance | closed (out of scope) | `unbounded`, `reference_mismatch`, `local_minimum_consensus`, `problem_infeasible`, `evaluation_error`, `stationarity_scale_f`; `compile_timeout`, `run_timeout`. Benchmark side is frozen; reopen a label only if the digest shows a sqpdax defect behind it. `compile_timeout`/`run_timeout` are sqpdax performance and stay open. |
| G8 | `trust_region_collapse` (was `unclassified`) | partial | Accepted (stream B): double projection in Steihaug-Toint for float64 when the leak is above 1e-6 (+3 tiny, 0 lost). Rejected: rounding-level acceptance. Remaining: B-class collapses (L-BFGS curvature far below the true curvature) are G6/G3 stream A. |

Streams (see `STREAMS.md`): A = G3, G5, G6, G1 asls leftovers; B = G8, G2, G4, G1 trip items, HS55 false success. Pending waivers (stream A): `hyp/merit_stagnation-decaying-penalty` (G6 H1); the two L-BFGS branches were superseded by the Powell damping merge.

Order of `G*` follows the report's `PRIORITIES`; re-sort after Q1 by the per-config deficit to scipy-slsqp (`gap.py`, `digest.py` scipy columns), largest first.

Decisions waiting for the user: `WAIVERS.md`. State of play: `CONTEXT.md` sections 5-7. Stream B stopped after batch 8 by user decision; both streams idle.

Secant audit (after the damping probes settle): `secant/lbfgs.py` and `secant/reset.py` docstrings still say "VARCHEN damping" / "Powell damping"; the damping is toward the rebuilt diagonal B0, not toward the full B (Powell) and not scalar-B0 (L-BFGS); the per-pair diagonal rebuild is replaced, not blended; stored pairs hold the damped y. Decide what remains (cautious skip vs damping, diagonal rebuild rule, reset ladder: soft / SNOPT diagonal / identity), then rename `damping_threshold`, rewrite the module/class/`append`/`_damping_theta`/`_soft_reset` docstrings and the test names accordingly.

Oct 15: `hyp/secant-powell-damping-model` merged (7fec079), user waiver; L-BFGS branches relfloor/signgate dropped. Secant audit done for `secant/lbfgs.py` and `secant/base.py`; remaining: `docs/source/index.md` and legacy files still say VARCHEN damping (out of scope here). Next: fresh diagnosis of the merged tip (`diagnostics/v6`), then open leads (HS116/CSFI1/MINMAXBD regressions, tfip losses, interior-point end game, DUALC).

Oct 15: fresh diagnosis of the merged tip in `diagnostics/v6` (see CONTEXT section 6). asls-craig/pcg are within 4-5 of scipy-slsqp; asls-pcg-exact stalled at 229 (the Levenberg shift branch, waiver-pending, is its lever); trip/tfip are 58-59 behind and now hold 125 unsolved rows each. Biggest scipy-solved groups: trust_region_collapse (34), termination_missed (29), merit_stagnation (28), cycling (27), merit_stagnation_near_opt (18).

Oct 15 (measured): full tip run in `results/tip_run` + diagnosis `diagnostics/v7` supersede the v6 estimates (CONTEXT section 6). Only asls-craig (-4) and asls-pcg (-6) are near scipy-slsqp; minresqlp -16, pasls -24, trip -60, tfip -59. Candidates: the Levenberg-shift waiver (asls-pcg-exact), the minresqlp/pasls regressions (HS116, DUAL3, MINMAXBD recur), then the interior-point groups.

Oct 15: Levenberg shift merged (8ba5df8); asls-pcg-exact 238 (+2 vs scipy-slsqp); suite 1806 passed. `diagnostics/v7` and `results/tip_run` predate this merge for asls-pcg-exact only; `results/shift_on_tip` has the matching run. Remaining waiver: `hyp/merit_stagnation-decaying-penalty`.

Oct 15: decaying penalty merged (feebd24), no waiver-pending branches left; suite 1809 passed. asls-craig 235 (-1), pcg 232 (-4), pcg-exact 241 (+5), minresqlp 224 (-12), pasls 218 (-18), trip 176, tfip 177. `diagnostics/v8` predates this merge for the five active-set configs (stale for them); a refresh would re-run them via `results/decay_on_tip`. Next: interior-point gap (trip/tfip -59/-60), minresqlp/pasls regressions, PALMER7E-type degenerate plateaus (termination), strict-stationarity view of the deficit.

Oct 15: `orchestration/strict.py` re-scores any results file with stationarity (`python orchestration/strict.py results/decay_on_tip/results_with_scipy_decay.jsonl`). Under solved-and-stationary (tau 1e-4) scipy-slsqp drops 236 -> 167, trust-constr 174 -> 173; sqpdax configs lose only 0-10 each; trip/tfip (176/177) are then ahead of both scipy solvers; strict deficit 9 (DIAGPQB DIAGPQT HAIFAM HS106 HS25 HS98 MSS1 OPTPRLOC TRO4X4). The report and benchmark tools are unchanged, so `report.html` still uses the flag.
