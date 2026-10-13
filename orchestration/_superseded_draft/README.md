# sqpdax convergence loop

Layout (copy into the repo root):
- `.claude/agents/triage.md`       Sonnet, read-only: builds `failure_table.json`
- `.claude/agents/implementer.md`  Sonnet: applies ONE hypothesis, minimal diff
- `kkt_check.py`                   independent judge used for every solver (wire it into your benchmark)
- `orchestrate.py`                 the loop; only the diagnose step calls Fable (1 turn, no tools)

## results.json (written by your benchmark, one record per problem x solver)
{ problem, solver: "sqpdax"|"scipy_slsqp"|"scipy_trust_constr", n, m_eq, m_ineq, iterations,
  termination_reason, final_step_size, qp_status,
  kkt_stationarity, max_violation, complementarity, converged_independent }
`converged_independent` MUST come from kkt_check.kkt_report, never from a solver's own flag.

## Run
git switch -c auto/sqpdax && python orchestrate.py --rounds 3 --hyps-per-round 2 --budget-usd 15

## Caveats
- Verify CLI flags (`--allowedTools`, `--max-turns`, `--permission-mode`) with `claude --help`.
- `total_cost_usd` is an API-equivalent figure; on a subscription it is a proxy for usage, not your bill.
  Fable on Pro bills usage credits: set a low credit cap in Claude settings as the real backstop.
- Test that `model: claude-fable-5-1` is accepted wherever you use it before a full run.
- The accept rule (gain >= 1, zero regressions) is strict by design; relax it if good changes get rejected.
- Hypotheses are tested sequentially on the same branch; failed ones are reset with `git reset --hard`.
