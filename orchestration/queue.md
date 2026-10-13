# Work queue

Status: `open` | `in-progress` | `done` | `blocked`. Edit by hand or at the end of a session (step R3).
Objective: every sqpdax config solves more instances than the scipy baselines (frozen). Scope: `slsqp_jax/sqpdax/**` only (see PROTOCOL.md). Current gap: `python orchestration/gap.py results/new_run/results.jsonl` (sqpdax union 248 vs scipy union 254; best config asls-pcg-exact 227 vs scipy-slsqp 236).

Baseline for all items: the diagnosis of `results/new_run/results.jsonl` (to be created by Q1 in
`diagnostics/v4`, `--previous diagnostics/v2/classified.jsonl`). Until Q1 exists, group sizes
below come from `diagnostics/v2` and are stale.

| ID | Item | Status | Notes |
|---|---|---|---|
| Q0 | Housekeeping: track the tooling files on `local_bench_cleanup` | open | `scripts/`, `benchmarks/validate.py`, `tests/benchmarks/test_validate.py`, `.cursor/skills/`, `orchestration/` are untracked. `hyp.sh new` refuses to start without them. Do not commit `diagnostics/` data. |
| Q1 | Diagnose `results/new_run` | open | Skill `benchmark-diagnostics` steps 1–3 into `diagnostics/v4` (about 744 tasks, 20–30 min), then refresh the narrative constants. Not a cause-group session; one session, mostly waiting. |
| Q2 | Settle round d (`ip_barrier_stall`) | open | Attribute the six tiny-tier losses (`ledger.py list --cause ip_barrier_stall -v`), fix or classify `test_infeasible_run_reports_multiplier_bound_hits`, note that HS55 tfip is now `successful` but unsolved (a false success, the protocol's stop-and-ask case), resolve the v1 = v2 puzzle (does `collapse_resets_secant` ever trigger?), then decide keep/revert. Needs Q0. |
| Q3 | Tiny-tier baseline at the `local_bench_cleanup` tip | open | `diagnostics/baselines/tiny_<sha7>/` for all seven configs (about 35 min on 4 jobs). Created once, reused by every G2. Needs Q0. |
| G1 | Infeasible starts, over-determined working sets | open | `kkt_solver_failure`, `false_infeasible_certificate`, `restoration_failure`, `ip_bound_slack_residual`, `near_feasible`, `nonfinite`. Largest scipy-solved deficit. Open leads: trip has no restoration phase. |
| G2 | Interior-point stall remainder | blocked on Q2 | `ip_barrier_stall`, `ip_bound_violation`, `ip_infeasible_stationary`. |
| G3 | Termination / secant | open | `termination_missed`, `secant_recovery_premature`, `merit_stagnation_near_opt`, `loose_tolerance`. Low yield last time (10/115). |
| G4 | Funnel y-iteration deadlock | open | `multiplier_failure`: `always_normal` + shrink `v_max` on y-iterations (`subproblem/solver/trust_funnel.py` Step 10). |
| G5 | `qp_kkt` success guard | open | `false_convergence`, `qp_inactive_multipliers`: require a converged QP and check complementarity. |
| G6 | Globalisation and scaling | open | `cycling`, `slow_progress`, `step_collapse`, `secant_recovery_stuck`, `merit_stagnation`, `local_minimum`. |
| G7 | Artefacts and performance | closed (out of scope) | `unbounded`, `reference_mismatch`, `local_minimum_consensus`, `problem_infeasible`, `evaluation_error`, `stationarity_scale_f`; `compile_timeout`, `run_timeout`. Benchmark side is frozen; reopen a label only if the digest shows a sqpdax defect behind it. `compile_timeout`/`run_timeout` are sqpdax performance and stay open. |

Order of `G*` follows the report's `PRIORITIES`; re-sort after Q1 by the per-config deficit to scipy-slsqp (`gap.py`, `digest.py` scipy columns), largest first.
