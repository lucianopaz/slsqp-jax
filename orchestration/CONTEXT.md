# sqpdax diagnostic cycle: state of play

Reconciled 2026-10-13 from the Cursor session "CUTEr integration test review"
(Oct 9 – Oct 12), `diagnostics/`, `scripts/diagnose_benchmark.py` and the git working tree.
**Read this file first; do not re-read the Cursor transcript** (2.7k messages, 135 KB of text).
Items marked (?) are unverified inferences.

## 1. Goal and loop

Raise the number of benchmark tasks that the seven sqpdax configs solve, using scipy
(`scipy-slsqp`, `scipy-trust-constr`) as baseline. One cause group at a time:
diagnose, hypothesise, implement the minimal change, re-run only the affected tasks,
`compare`, keep or revert, record in `FIX_ATTEMPTS`, run the tiny-tier regression check.

How-to for each step: `.cursor/skills/benchmark-diagnostics/SKILL.md`.
Repo rules: `AGENTS.md`. Code state is committed on branch `local_bench_cleanup` (user commit, Oct 13); `main` is still at `4a31657` (Oct 9). Commit `be49ae8` "Dirty state of benchmark fixes".

## 2. User constraints (standing)

- Run only commands on the allowed list. If one needs extra permission, do not run it.
- Never `git stash` (the user keeps stashes). Bisects use `git worktree add /tmp/x <sha>`, symlink `.venv`, set `PYTHONPATH`.
- One bug at a time; if a fix does not work, record what was tried in the report instead of silently dropping it.
- Do not remove bound/inequality slacks from the interior-point methods (infeasible starts must keep working).
- Radius collapse should sometimes trigger a secant reset; the user wants this tested, not assumed.
- Objective (Oct 13): every sqpdax model should solve more instances than the scipy baselines. Scipy rows and the benchmark problems are frozen; never re-run them.
- Scope (Oct 13): the code is in a refactor. Read and change only `slsqp_jax/sqpdax/**` (tests: `tests/sqpdax/**`). The rest of `slsqp_jax/` is legacy: do not read, change or gate on it.
- Prefer short, math-heavy communication. Budget is tight.
- `_cursor_export/state.vscdb` (1 GB) holds the Cursor chats, possibly with consulting-org code. Delete the folder when done.

## 3. Data map

| What | Path | Notes |
|---|---|---|
| Medium-tier run, first diagnosis | `diagnostics/classified.jsonl`, `diagnostics/diag.jsonl`, `diagnostics/report.html` | Oct 9 07:14. 1429 unsolved sqpdax tasks. The root `results.jsonl` it was built from is **no longer in the repo** (?) |
| Second diagnosis (run at `4a31657`) | `diagnostics/v2/{classified,diag}.jsonl`, `v2/report.html`, `v2/traj/` | 1281 unsolved tasks. `report.html` regenerated Oct 9 16:52 with `WORK_LOG` of rounds a/b/c |
| Fix measurements, round 1 (pre-v3) | `diagnostics/fix1_pcg_freeze`, `fix2a_slack_seed`, `fix2_slack_normal` | each has `compare.json` |
| Fix measurements, v3 | `diagnostics/v3/fix_{a_catalog,b1_cap,b_restoration,b_final,c_termination}`, `fix_d_barrier_{before,v1,v2}` | each has `compare.json` + `.log` |
| Tiny-tier regression runs | `diagnostics/v3/regress_tiny`, `regress_tiny2`, `regress_tiny_ip_v1`, `regress_tiny_ip_v2` | 273 instances; `regress_tiny*` use 7 configs, `_ip_*` only trip/tfip |
| Newest full benchmark | `results/new_run/results.jsonl` | run `20261012T130618Z`, tier medium, 303 instances x 9 configs = 2727 tasks. Run locally on the dirty working tree, which is the state now committed on `local_bench_cleanup` (user-confirmed). The recorded sha `4a31657` is just the base commit |
| Report generator | `scripts/diagnose_benchmark.py` | constants `CAUSES`, `RECOMMENDATIONS`, `PRIORITIES`, `FIX_ATTEMPTS`, `WORK_LOG` |
| Tests | `tests/`, `tests/benchmarks/`, `tests/sqpdax/**` | |

Baseline pairing for `compare`: `compare <baseline_dir> <fix_dir> --results <results.jsonl>`.
Rounds a–d were measured against `diagnostics/classified.jsonl` (the first diagnosis), **not** v2.

## 4. Fix rounds already done

Until Oct 13 these changes were uncommitted (33 modified files, +3.6k/−1k lines on top of `4a31657`); the user has since committed them on `local_bench_cleanup`. `scripts/`, `benchmarks/validate.py`, `tests/benchmarks/test_validate.py`, `.cursor/` and `orchestration/` were still untracked after that commit (see section 8).

| Round | Target causes | Change | Result | Decision |
|---|---|---|---|---|
| 1 | `false_convergence`, `stationarity_scale_f` | projected-CG freeze rule fix | 98/300 now solve | kept (partial) |
| 2a | `ip_bound_slack_residual` | IPOPT-style slack seeding | 17/173 | kept, insufficient alone |
| 2 | `ip_bound_slack_residual` | scaled normal step moves slacks (N&W 19.34) | 52/173 (trip) | kept, bug fixed, group partly solved |
| a | `unbounded`, `reference_mismatch`, `local_minimum_consensus`, `problem_infeasible` | L-type sign fix; `python -m benchmarks validate` checks f*/x* vs scipy consensus | 191/333 solve on corrected catalogue | kept |
| b | `kkt_solver_failure`, `false_infeasible_certificate`, `restoration_failure`, `ip_bound_slack_residual` | working-set cap at `n - meq` independent rows (`working_set_policy.cap_rows/independent_rows`), `infeasible_stationary` certified with the violation gradient, bounded projected-CGLS restoration step (asls family) | first pass 151/339, final rule 142/382 | kept; trip has no restoration phase (0/36 then 13/122) |
| c | `termination_missed`, `merit_stagnation_near_opt`, `secant_recovery_premature` | KKT test before fatal flags; `StepResult.rounding_level` keeps rounding rejections out of secant streaks; asls stagnation relative to `max(|phi_best|, atol)` | only 10/115 | kept with two guards (MATRIX2 had become a false success) |
| tiny regression after a–c | all | `regress_tiny` vs baseline | +370 newly solved, 23 lost: 11 catalogue effects, 9 platform noise, 3 real (HS119 asls-pcg-exact/pasls, DEGENQP pasls) | accepted |

Tiny-tier solved per config, before → after a–c: asls-craig 137→199, asls-minresqlp 147→198, asls-pcg 147→203, asls-pcg-exact 127→211, pasls 144→185, tfip 103→130, trip 123→149.

## 5. Round d settled, round e (Oct 13-14, two streams)

Round d is kept (user, Oct 13). Its six tiny-tier losses (HS119/HS30/HS55 tfip, HS30/LSNNODOC/PRICE4B trip) were attributed: at `mu` ~ 1e-7 the end game fails. HS55 is a path-dependent second KKT point, not a false success. Round e = the work of two parallel streams (`STREAMS.md`; stream A on `local_bench_cleanup`, stream B on `stream/b`, merged back and forth). Every attempt is in `ledger.jsonl` (32 entries since Oct 13: 6 accepted, 9 rejected, 17 inconclusive, most of them probes or superseded versions).

Accepted and merged (zero real losses, suite clean): G1 null-step restoration trigger (asls); G6 pasls mu back-off; G8 double projection in Steihaug-Toint (float64, leak > 1e-6); funnel end game (Cauchy slope, noise accept, radius_f collapse); analytic Cauchy decrease (float64); trip penalty cap (monotone to nu = 50, then decay 0.2).

Waiting for a user waiver (real losses but gains >= 3x; all listed with numbers in `WAIVERS.md`): L-BFGS sign gate (`hyp/termination_missed-lbfgs-signgate`), Levenberg shift for exact curvature (asls-pcg-exact 229 -> 237), decaying merit penalty, rejected-pair secant update (trip/tfip), funnel projection refinement (+1, clean), working-set exchange (new false success on HS97: investigate first). Combination of the first three on the merged tip (`diagnostics/v5_combo`): asls-pcg-exact 243 (+7 vs scipy-slsqp), union 257 vs scipy 254 (counts previously unsolved tasks only).

Rejected (do not repeat): trip joint-norm merit, NaN rho / CG budget for rho = -inf, `always_normal` default, rounding-level acceptance, trip shrink/grow thresholds, DUALC row scaling (premise falsified), trip slack reset after every step, penalty floor 50.

## 6. Where the code stands today

**Measured** state of the merged tip (`local_bench_cleanup`, after the Powell-damping merge): full run of the seven sqpdax configs on all 303 instances, `results/tip_run/results.jsonl` (2121 tasks, 4 jobs, 3913 s, timeout 300 s); scipy rows concatenated from `new_run` in `results/tip_run/results_with_scipy.jsonl` (2727 rows, same layout as `new_run`). Diagnosis of the 637 unsolved sqpdax tasks: `diagnostics/v7/{classified,diag}.jsonl`, `report.html` (the report is now built against tip results, so its overview is correct; its `loose_tolerance` label still lacks an f_tol gate).

| Config | `new_run` | **tip** | gained / lost vs `new_run` | vs scipy-slsqp (236) |
|---|---|---|---|---|
| asls-craig | 209 | **235** (decaying penalty merged, feebd24; was 232 in `tip_run`) | see ledger | -1 |
| asls-pcg | 211 | **232** (was 230) | | -4 |
| asls-pcg-exact | 227 | **241** (shift 8ba5df8 + decaying penalty feebd24; was 229 in `tip_run`) | | +5 |
| asls-minresqlp | 208 | **224** (was 220) | | -12 |
| pasls | 193 | **218** (was 212) | | -18 |
| trip | 163 | **176** | +15 / -2 | -60 (+2 vs trust-constr) |
| tfip | 166 | **177** | +11 / -0 | -59 (+3 vs trust-constr) |

Union: sqpdax 258 vs scipy union 254 (+4) with the shift and the decaying penalty (255 in `tip_run`). Deficit (scipy solves, no sqpdax config) 9: DECONVC, HAIFAM, HS106, HS25, HS98, LEVYMONT9, MSS1 (lost by asls-craig to the decaying penalty), OPTPRLOC, TRO4X4; sqpdax-only wins 11. asls-pcg-exact is the first config above scipy-slsqp (241 vs 236); asls-craig is 1 below. Measured with `results/decay_on_tip/results_with_scipy_decay.jsonl` (the five active-set configs re-run on the merged code; trip/tfip from `tip_run`). `python orchestration/gap.py results/tip_run/results_with_scipy.jsonl` reproduces it.

Tasks lost vs `new_run` (all from the Powell damping and the earlier merges): minresqlp BYRDSPHR DUAL3 DUAL4 DUALC1 HS102 HS116 HS13 OET4; pasls CSFI1 DALLASS DUAL3 HS116 MINMAXBD TRIGON1B; pcg MINMAXBD; trip KOEBHELB POLAK6. On two configs: DUAL3, HS116, MINMAXBD. Timeouts at the tip (300 s, 4 jobs): DEGENQPC on 5 configs, HS105 pcg-exact, CLEUVEN7 and TRO21X5 craig.

Still unsolved at the tip by cause (n, solved by scipy-slsqp): local_minimum 68 (16), reference_mismatch 63 (15), cycling 62 (29), unclassified/trust_region_collapse 54 (34), merit_stagnation 47 (31), termination_missed 36 (29), local_minimum_consensus 32 (0), ip_bound_slack_residual 31 (13), merit_stagnation_near_opt 23 (21), slow_progress 22 (7), step_collapse 20 (11), ip_bound_violation 20 (5), secant_recovery_premature 19 (8), ip_barrier_stall 19 (14). By config: trip 127, tfip 126, pasls 91, minresqlp 83, pcg-exact 74, pcg 73, craig 71 (274 of 645 rows solved by scipy-slsqp).

## 7. Next steps

1. User decides the waivers in `WAIVERS.md` (cheapest, largest effect: Levenberg shift, one loss, asls-pcg-exact above scipy-slsqp). Then re-run the diagnosis and a full medium benchmark (human decision) to confirm.
2. Open leads, by size: trip/tfip class-B end-game collapses (~30 scipy-solved: L-BFGS curvature under-estimate; the rejected-pair update is the candidate); DUALC infeasible start on trip (normal step moves x by 1e-3 per step: grow the normal radius independently of rho); OET crawl (rho = 0.25 constant: second-order correction); scaled relative stationarity test for the interior-point end game (tolerance decision for the user); QP exchange investigation (HS97 false success); cycling/slow_progress (PALMER*A/E, LOGROS, HS268: secant scaling); G5 (`qp_kkt` guard, 12 tasks) not started.
3. Report text: `FIX_ATTEMPTS` entries per ledger entry are not written yet; one summary bullet is in `WORK_LOG`.

## 8. Orchestration

Checklist-driven session per cause group: `orchestration/README.md` (overview), `PROTOCOL.md` (checklist and gates),
`queue.md` (work items), `ledger.jsonl` (every hypothesis tried, accepted or not; seeded with rounds 1–d above),
`hyp.sh` (worktree + branch `hyp/<name>` per hypothesis, guarded `--no-ff` merge into `local_bench_cleanup`).
The user has not yet agreed to this as the final way of working; treat it as a proposal and follow explicit instructions first.
The first draft is in `orchestration/_superseded_draft/`.

Prerequisite: `scripts/`, `benchmarks/validate.py`, `tests/benchmarks/test_validate.py`, `.cursor/skills/` and `orchestration/`
were untracked when `be49ae8` was made, so worktrees from that commit lack them. Track them (not the `diagnostics/` data) first.
