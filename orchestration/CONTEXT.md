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

**Flag changed (Oct 15, user decision):** `solved` now requires stationarity: successful, finite, feas <= 1e-6, f_gap <= 1e-4 and `stat_scaled` <= 1e-4 (NaN passes). Implemented in `benchmarks/criteria.py`, `worker.py`, `analysis.py`, `__main__.py` (`--stat-tol`, `rescore`) and `scripts/diagnose_benchmark.py` (`compare`, header); merged as `4df81ab`. Old results files are re-scored with `python -m benchmarks rescore IN OUT` (no reruns). `orchestration/strict.py` gives the same numbers plus sensitivity to tau and a complementarity-aware variant.

**Measured state of the merged tip** (all 303 instances, `results/final/results_rescored.jsonl` = the five active-set configs run at the final code, trip/tfip from `tip_run`, scipy rows from `new_run`; diagnosis `diagnostics/v9`, `report.html` built against it, so its overview matches):

| Config | solved (strict flag) | flagged-solved before | vs scipy-slsqp (167) | vs trust-constr (173) |
|---|---|---|---|---|
| asls-pcg-exact | **239** | 241 | +72 | +66 |
| asls-craig | **228** | 235 | +61 | +55 |
| asls-pcg | **226** | 232 | +59 | +53 |
| asls-minresqlp | **218** | 224 | +51 | +45 |
| pasls | **209** | 218 | +42 | +36 |
| tfip | **177** | 177 | +10 | +4 |
| trip | **176** | 176 | +9 | +3 |
| scipy-slsqp | 167 | 236 | | |
| scipy-trust-constr | 173 | 174 | | |

Union: sqpdax 255 vs scipy union 236 (+19). Deficit (scipy solves, no sqpdax config) 9: DIAGPQB, DIAGPQT, HAIFAM, HS106, HS25, HS98, MSS1, OPTPRLOC, TRO4X4; sqpdax-only wins 28. (`python orchestration/gap.py results/final/results_rescored.jsonl`.) scipy-slsqp loses 69 flagged solves because it stops on the change in f at non-stationary points (e.g. PALMER7E, stat 0.53); sqpdax loses 30 rows (asls-craig 7, pasls 9, pcg 6, minresqlp 6, pcg-exact 2; BT5 DIAGPQB DIAGPQT HS13 HS40 HS81 HS87 ODFITS PALMER1/2/3): false successes where the solver stops at stat_scaled 2e-4..8e-3 (classified `stationarity_scale_f` 27, `false_convergence` 2).

Still unsolved under the strict flag by cause (n, solved by scipy-slsqp): local_minimum 70 (6), cycling 62 (14), reference_mismatch 62 (10), unclassified/trust_region_collapse 54 (13), stationarity_scale_f 45 (10), termination_missed 35 (15), local_minimum_consensus 33 (0), ip_bound_slack_residual 31 (10), merit_stagnation 27 (12), slow_progress 22 (1), ip_bound_violation 20 (4), merit_stagnation_near_opt 20 (2), step_collapse 19 (8), ip_barrier_stall 19 (9). By config: trip 127, tfip 126, pasls 94, minresqlp 85, craig 75, pcg 77, pcg-exact 64.

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
