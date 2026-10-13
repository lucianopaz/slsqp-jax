# Two parallel streams

Both streams follow `PROTOCOL.md`; only the checkout, branch and file ownership differ.

| | Stream A | Stream B |
|---|---|---|
| Checkout (merges) | `/Users/lucianopaz/repos/slsqp-jax` | `/Users/lucianopaz/repos/slsqp-jax-streamB` |
| Integration branch | `local_bench_cleanup` | `stream/b` |
| Worktrees of hypotheses | `slsqp-jax/.worktrees/` | `slsqp-jax-streamB/.worktrees/` |
| Wrapper | `orchestration/hyp.sh` | `orchestration/hyp_b.sh` (same arguments) |
| Queue items | G3, G5, G6, G1 active-set leftovers (`kkt_solver_failure`, `false_infeasible_certificate`, `near_feasible`) and the waiver-pending branches | G8, G2, G4, G1 trip items (NaN rho, rho = -inf), HS55 tfip false success |

## File ownership (never edit a file owned by the other stream)

- A owns: `minimiser/active_set_linesearch.py`, `minimiser/proximal_active_set_linesearch.py`, `secant/**` (L-BFGS feeds trip/tfip too: after an A merge, B merges `local_bench_cleanup` into `stream/b` and re-checks trip/tfip), `subproblem/active_set*`, `subproblem/solver/{craig,minres_qlp,pcg*}.py`, `linalg/**`, `minimiser/base.py`.
- B owns: `minimiser/trust_region_interior_point.py`, `minimiser/trust_funnel*.py`, `barrier/**`, `step_controller/**`, `subproblem/solver/{steihaug_toint_cg,trust_funnel,trust_region,funnel_*}.py`.
- Anything else: the stream that needs it asks the other (message in `queue.md` notes) before touching it.

## Shared resources

- Ledger and queue: ALWAYS written through the main checkout (`python /Users/lucianopaz/repos/slsqp-jax/orchestration/ledger.py add ...`, edits of `/Users/lucianopaz/repos/slsqp-jax/orchestration/queue.md`); stream B never commits them on `stream/b`. Stream A commits them. Do not use `ledger.py lock` (single global lock); the file ownership above replaces it.
- Data: outputs go to `/Users/lucianopaz/repos/slsqp-jax/diagnostics/v4/<cause>/<slug>` (distinct slug per hypothesis, so no overlap).
- CPU: at most 5 benchmark jobs per stream (`--jobs 5`).
- Merges: each stream merges only into its own branch. Stream A merges `stream/b` into `local_bench_cleanup` (fast, files are disjoint) when stream B reports an accepted hypothesis; B pulls A with `git -C ../slsqp-jax-streamB merge local_bench_cleanup`.
- Baselines: tiny-tier regression joins against the `results/new_run/results.jsonl` rows; attribute every loss at a base worktree of the stream's own tip.
