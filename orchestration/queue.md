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
| Q2 | Settle round d (`ip_barrier_stall`) | open | Attribute the six tiny-tier losses (`ledger.py list --cause ip_barrier_stall -v`), fix or classify `test_infeasible_run_reports_multiplier_bound_hits`, note that HS55 tfip is now `successful` but unsolved (a false success, the protocol's stop-and-ask case), resolve the v1 = v2 puzzle (does `collapse_resets_secant` ever trigger?), then decide keep/revert. Needs Q0. |
| Q3 | Tiny-tier baseline | dropped | Replaced by the tiny-tier rows of `results/new_run/results.jsonl` (user decision, Oct 13). G2 of every `G*` joins against those rows; `regress_join.py` may need a `--tier tiny` filter or a filtered copy, check in Q2. |
| G1 | Infeasible starts, over-determined working sets | open | `kkt_solver_failure`, `false_infeasible_certificate`, `restoration_failure`, `ip_bound_slack_residual`, `near_feasible`, `nonfinite`. Largest scipy-solved deficit. Open leads: trip has no restoration phase. |
| G2 | Interior-point stall remainder | blocked on Q2 | `ip_barrier_stall`, `ip_bound_violation`, `ip_infeasible_stationary`. |
| G3 | Termination / secant | open | `termination_missed`, `secant_recovery_premature`, `merit_stagnation_near_opt`, `loose_tolerance`. Low yield last time (10/115). |
| G4 | Funnel y-iteration deadlock | open | `multiplier_failure`: `always_normal` + shrink `v_max` on y-iterations (`subproblem/solver/trust_funnel.py` Step 10). |
| G5 | `qp_kkt` success guard | open | `false_convergence`, `qp_inactive_multipliers`: require a converged QP and check complementarity. |
| G6 | Globalisation and scaling | open | `cycling`, `slow_progress`, `step_collapse`, `secant_recovery_stuck`, `merit_stagnation`, `local_minimum`. |
| G7 | Artefacts and performance | closed (out of scope) | `unbounded`, `reference_mismatch`, `local_minimum_consensus`, `problem_infeasible`, `evaluation_error`, `stationarity_scale_f`; `compile_timeout`, `run_timeout`. Benchmark side is frozen; reopen a label only if the digest shows a sqpdax defect behind it. `compile_timeout`/`run_timeout` are sqpdax performance and stay open. |
| G8 | `trust_region_collapse` (was `unclassified`) | open | trip 36, tfip 6; 34 of 42 solved by scipy SLSQP, all end `trust_region_collapse`. Classifier has no rule for it (`scripts/` is a tool; narrative entry added in `PRIORITIES`). |

Order of `G*` follows the report's `PRIORITIES`; re-sort after Q1 by the per-config deficit to scipy-slsqp (`gap.py`, `digest.py` scipy columns), largest first.
