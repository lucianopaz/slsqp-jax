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

## 5. Round d: mu-barrier stall (`ip_barrier_stall`, trip/tfip) — **left halfway**

User's request (Oct 10 06:11): (1) decide the mu update on the scaled barrier KKT error against `kappa_eps * mu` (Fiacco–McCormick; trust-constr uses `BARRIER_DECAY_RATIO * mu`) instead of the final atol; (2) terminate only once mu is below the barrier tolerance and the radius is still collapsed, with a dedicated result code (xtol-style exit); (3) test whether radius collapse must reset the secant memory.

Implemented in the working tree (all in the diff, none committed):
- `barrier/update.py`: `scale_max` (default 100), `scaled_optimality_residual`, `residual_components`, `force` argument on all `update`s; `FunnelBarrierUpdate.kappa_eps` (static, default 10.0); `MonotoneBarrierUpdate` test `E(x,s;mu) <= kappa_eps*mu`.
- `trust_region_interior_point.py` and `trust_funnel_interior_point.py`: `collapse_resets_secant` (static, default `True`), carries `exit_without_progress`, `trust_region_collapsed`, `stalled_since_exit`; new result code `trust_region_collapse`; xtol-style exit forces the mu reduction (`force=True`); funnel restricts rounding-level exits to f-iterations.
- Tests added/rewritten for the stall streaks and xtol exit in `tests/sqpdax/barrier/test_update.py`, `tests/sqpdax/minimiser/test_trust_{region,funnel}_interior_point.py`, `.../diagnostics/test_trust_funnel.py`.

Measurements (66 `ip_barrier_stall` tasks: tfip 56, trip 10):

| Run | Code | Solved | Still `ip_barrier_stall` |
|---|---|---|---|
| `fix_d_barrier_before` | HEAD behaviour | 3/66 (tfip 3/56, trip 0/10) | 50 |
| `fix_d_barrier_v2` | working tree, `collapse_resets_secant=True` | 8/66 (tfip 7/56, trip 1/10) | 23 |
| `fix_d_barrier_v1` | worktree `/tmp/wt_v1`, `collapse_resets_secant=False` | 8/66 (same) | 23 |

- v1 and v2 gave identical results on both the 66-task run and on the trip/tfip tiny regression (+46 / −6 against `regress_tiny2`; trip 149→157, tfip 130→162). So the secant-reset hypothesis is **untested in effect**: the flag changed nothing here. Either it never triggers on these tasks or the v1 build did not differ as intended (?). Check before concluding that the user's worry was unwarranted.
- New diagnoses of the 43 that left `ip_barrier_stall`: `termination_missed` 6, `unclassified` 5, `unbounded` 5, `cycling` 4, `step_collapse` 3, `multiplier_failure` 3, plus smaller groups. CANTILVR, CSFI2, ELATVIDUB, PRICE4B, HS1 now solve on tfip.
- Tiny-tier losses of the round, **not yet attributed**: HS119 tfip, HS30 tfip, HS30 trip, HS55 tfip, LSNNODOC trip, PRICE4B trip. Attribution protocol is in the skill (rerun at base commit from a worktree, check catalogue `f*`, platform noise).
- No `FIX_ATTEMPTS` entry, no `WORK_LOG` bullet, no report regeneration for round d (`grep fix_d scripts/diagnose_benchmark.py` is empty).

Tests (last known, Oct 10 07:18, full non-slow suite, 22 min): **1 failed, 1892 passed**.
Failing: `tests/sqpdax/minimiser/diagnostics/test_trust_funnel.py::test_infeasible_run_reports_multiplier_bound_hits`
(the infeasible float32 run now ends with `infeasible_stationary_point` and `n_multiplier_bound = 0`; the test asserts bound hits). An earlier failure of `test_run_emits_documented_diagnostic_records[box]` seems to have been fixed (?). The rerun started at 07:20 never reported back.

Open investigation when the session stopped: HS100 on trip (and tfip) stalls with mu ≈ 1e-6 and predicted reduction ≈ 1e-6..1e-4 at radius 2; the agent was probing `probe_pred.py` and a throw-away patch of `slsqp_jax/sqpdax/linalg/projection.py` (free_mask handling) in `/tmp/wt_exp`. **That patch is not in the repo.** The scratch scripts (`/tmp/trace_tfip.py`, `/tmp/probe_pred.py`, `/tmp/diff_runs.py`, `/tmp/trace_infeas.py`) are in `/tmp` on the user's Mac and may be gone.

## 6. Where the code stands today

`results/new_run` (Oct 12, 303 instances, medium tier) was produced from the dirty working tree that is now committed on `local_bench_cleanup` (user-confirmed), so it includes rounds a–d. Unsolved sqpdax tasks: 1281 (v2) → 744.

| Config | Solved /303 | Unsolved before (v2) |
|---|---|---|
| asls-pcg-exact | 227 | 180 |
| asls-pcg | 211 | 167 |
| asls-craig | 209 | 179 |
| asls-minresqlp | 208 | 167 |
| pasls | 193 | 174 |
| tfip | 166 | 218 |
| trip | 163 | 196 |
| scipy-slsqp | 236 | |
| scipy-trust-constr | 174 | |

Gap to scipy (`orchestration/gap.py`): no config beats scipy-slsqp yet (asls-pcg-exact −9, asls-craig −27, asls-pcg −25, asls-minresqlp −28, pasls −43, trip −73, tfip −70; all but trip/tfip beat trust-constr). Deficit set (scipy solves, no sqpdax config does): 16 instances; sqpdax-only wins: 10.

Instance level: sqpdax union 248, scipy-slsqp 236, trust-constr 174, scipy union 254.
Unsolved sqpdax rows by status: `max_steps_reached` 306, `successful` 182 (solver says success, reference disagrees: false convergence or mismatched f*), `merit_stagnation` 74, `trust_region_collapse` 56, `secant_recovery_failure` 46, `infeasible` 38, `infeasible_stationary` 17, `multiplier_solve_failure` 13.
Catalogue validation labels in that run: ok 1395, fstar_from_consensus 513, unverified 333, fstar_mismatch 261, sign_flipped 207, error 9.

## 7. Next steps, in order (cheap first)

1. Baseline is settled: `results/new_run` = committed state of `local_bench_cleanup`.
2. Settle the branch: fix or mark `test_infeasible_run_reports_multiplier_bound_hits`; rerun `uv run pytest -m "not slow"`; `uv run prek run`.
3. Attribute the six round-d tiny-tier losses; decide keep/revert for round d; resolve the v1 = v2 puzzle.
4. Run the diagnostics for `results/new_run/results.jsonl` into `diagnostics/v4` (skill steps 1–3, `--previous diagnostics/v2/classified.jsonl`; ~744 tasks, roughly 20 min on 4–7 jobs).
5. Update the narrative constants (`FIX_ATTEMPTS` for round d, `WORK_LOG`, `PRIORITIES`, counts quoted in `CAUSES`/`RECOMMENDATIONS`), regenerate `report.html`.
6. Next causes: order by the per-config deficit to scipy-slsqp after step 4 (see `orchestration/queue.md`). Benchmark-artefact causes are out of scope (benchmark side is frozen). The IP configs (trip, tfip) are furthest behind (−73, −70).

## 8. Orchestration

Checklist-driven session per cause group: `orchestration/README.md` (overview), `PROTOCOL.md` (checklist and gates),
`queue.md` (work items), `ledger.jsonl` (every hypothesis tried, accepted or not; seeded with rounds 1–d above),
`hyp.sh` (worktree + branch `hyp/<name>` per hypothesis, guarded `--no-ff` merge into `local_bench_cleanup`).
The user has not yet agreed to this as the final way of working; treat it as a proposal and follow explicit instructions first.
The first draft is in `orchestration/_superseded_draft/`.

Prerequisite: `scripts/`, `benchmarks/validate.py`, `tests/benchmarks/test_validate.py`, `.cursor/skills/` and `orchestration/`
were untracked when `be49ae8` was made, so worktrees from that commit lack them. Track them (not the `diagnostics/` data) first.
