# G6 lead: QP cap without exchange (Stream A, active-set files)

Base: worktree `.worktrees/base-9473abc` (tip 9473abc). Classification: `diagnostics/v5/classified.jsonl`.
Ledger (`ledger.py list -v`): round b kept the cap (`b1-cap`, `b-final`); "all active rows always kept" was discarded then.
No exchange rule for a full set has been tried, and no single-drop rule either. Nothing below repeats a rejected attempt.
Scratch (in-process monkeypatches, no repo edits): `$SP/cap/{probe.py,capx.py,patchdir/sitecustomize.py,tiny.sh,cmp.py}`,
`SP=/private/tmp/claude-501/-Users-lucianopaz-repos-slsqp-jax/e5d414fe-d379-4c1f-aef2-fa13378cf106/scratchpad`.
Tiny runs: `$SP/cap/tiny/{base,gi}/results.jsonl` (5 active-set configs x 273 instances = 1365 tasks, both at 9473abc).

## What the QP does when the set is full (trace, `probe.py base`)

`ThresholdWorkingSetPolicy.propose` (`subproblem/solver/working_set_policy.py:557-622`) keeps every row whose
multiplier is ≥ -tol, then `_select_rows` adds violated rows only into the free slots below `cap = n - meq`
(`cap_rows`), or skips them when they depend linearly on W (`independent_rows`). When W is full, or the violated
row depends on W, and no multiplier in W is negative, the proposal equals `current`. The loop then reads
`changed = False` and reports `working_set_converged` / success, so the returned d violates `a_jᵀd + c_j ≤ 0`.
Over a whole run, "blocked" means a QP that ended converged with an excluded row violated by > 1e-6 at the step:

| task | QPs blocked / QPs ending converged | max excluded violation | outcome |
|---|---|---|---|
| HS23 pasls (n=2, cap 2) | 17/18 | 6.9 | merit_stagnation, f = 2.0071 |
| HS23 asls-pcg | 18/21 | 6.9 | solves, but needs 21 steps |
| HS114 asls-pcg-exact (cap 7) | 33 full + 9 dependent / 50 | 6.7e3 | merit_stagnation |
| DEGENQP pasls (cap 5, \|W\| = 26 kept at a degenerate feasible vertex) | 25/26 | 17 | merit_stagnation |
| HS98 asls-pcg-exact (cap 6) | 6/10 | 1.1 | "successful" at the local point f = 4.0712 |
| HS97 asls-pcg-exact | 9/10 | 0.41 | merit_stagnation |
| HS106 pasls / asls-pcg | 0 | - | **not this cause**: QP budget runs out (all-at-once add/drop wanders, 50 iters); see H2 |

## H1 (rank 1): Goldfarb–Idnani exchange when the refresh is blocked

**Claim.** A full (or dependent) working set needs an exchange, not a stop. Pick the most violated excluded row j.
Write a_j in the basis of the working rows, `A_Eᵀ r_E + A_Wᵀ r_W = a_j` (least squares). Adding j at multiplier t with
d fixed keeps stationarity only if `λ_W(t) = λ_W − t r_W`. The first row to reach zero is
`i* = argmin_{i∈W, r_i>0} λ_i / r_i`, the dual ratio test of Goldfarb & Idnani (1983, step 2(c)). Swap i* out and j in.
If no r_i > 0 exists, the linearisation with W is inconsistent in j. Keep today's behaviour for that case.

**Must hold if true.** (a) After the change no QP ends converged with `max_{j∉W}(a_jᵀd + c_j) > tol`, except in the
no-r_i>0 case. (b) Dropping by smallest multiplier instead of the ratio test must do worse. Both hold:
- blocked QPs fall to 0 on HS23, HS114-exact, DEGENQP, HS98-exact and HS97 (`probe.py gi`);
- the `minmult` variant (drop argmin λ_W) loses HS23 pasls (f 2.31), HS98 exact (4.44), HS97 exact (ends `infeasible_stationary`, f = −2.9e7) and HS98 pasls (infeasible).
  So the ratio test, not the exchange itself, is what carries the gain.

**Evidence (tiny tier, 1365 tasks, base → gi, both at 9473abc).** craig 201→203, minresqlp 199→201, pcg 204→205,
pcg-exact 213→214, pasls 190→194. Total **+15 / −5**.
- Gained: DEGENQP pasls, HS114 exact, HS23 pasls, HS97 pasls, HS98 exact, HS64 x4 (craig/minresqlp/pcg/pasls),
  OET1 exact, OET4 craig/pcg/pasls, OET6 minresqlp, OET7 minresqlp.
- Lost: HS41 exact, MADSEN exact, DUALC2 pasls, HS102 pcg, HS102 minresqlp.
- HS41 exact was traced. The exchange takes a different path, and the run stops after 2 steps as "successful" at a degenerate
  vertex (|W| = 4 > cap 3, d = 0, multipliers ≈ 0, f = 2 vs f* = 1.926). This is a false-termination weakness at a
  degenerate vertex, exposed by the path change and not caused by the exchange step itself (?). The other 4 losses are not attributed.
- Steps also fall where both runs solve: HS23 asls-pcg 21 → 6, HS114 exact 57 → 12.

**Change.** `slsqp_jax/sqpdax/subproblem/solver/working_set_policy.py`, about 35 lines.
- Add a private helper `_ratio_exchange(lag, current, step, values, excluded)`.
- Call it at the end of `ThresholdWorkingSetPolicy.propose` under `jax.lax.cond(blocked, ...)`, where
  `blocked = all(next == current) & any(excluded)` and `excluded = ~next & (values > working_tol)`.
  Gating on `blocked` means the change is inert whenever the loop would have continued anyway.
- The stacked rows are `[ineq | −e_i (lb) | +e_i (ub)]`, as in `independent_rows`.
  The tested variant uses `pinv([A_E; mask_W·A]ᵀ, rtol=1e-10) @ a_j` and treats `r_i > 1e-10·max(1, ‖r_W‖∞)` as positive.
- For production, solve the least squares only on free coordinates plus the inequality rows of W (bound rows give r_i
  directly as ±residual) to keep the cost at O(n²·|W_I|) under the cond. This is the same order as the existing independence screen.
- Optionally apply the same exchange in `SingleExchangeWorkingSetPolicy.propose` (`can_add & full` currently returns
  `current` too). No benchmark config uses it, so expect 0 tasks.
- One parametrised test in `tests/sqpdax/subproblem/solver/` (e.g. HS23-like 2-D QP with W full and a violated third row):
  the QP returns a step satisfying all linearised rows.
- No public API change. No tolerance change.

**Cheap falsifier.** HS23 pasls and HS114 asls-pcg-exact with the patch (about 30 s). If either still logs a converged QP
with an excluded row violated by > 1e-6 and r_W has a positive entry, or does not solve, H1 is false.
Monkeypatch result: 6 and 12 steps, both solved.

**Expected.** About 10 net on the tiny tier (+15 / −5). Group-level gain on the named lead tasks: 5 of 8 (HS23 pasls, HS114 exact,
DEGENQP pasls, HS98 exact, HS97 pasls). Not gained: HS114 pasls, HS98 pasls, HS97 exact (local point f = 4.0712), HS106 (all).

**Risk.** It touches every asls/pasls QP that ends blocked: 18/21 QPs even on solved HS23 asls-pcg, so the change is broad.
- Path changes expose the degenerate-vertex false termination (HS41). The 5 tiny losses need the attribution protocol (rerun at base, noise).
- The exchange can cycle in theory. The 50-iteration budget bounds it, and no `anti_cycling` or budget increase was seen on the targets.
- `working_set_policy.py` is not listed in STREAMS.md ownership: it is not in `subproblem/active_set*`. Ask B, or have the
  caller confirm A owns it. B's IP configs do not use this policy.
- Interaction with the waiver-pending G6 penalty decay: HS23 was the regression of `hyp/merit_stagnation-decaying-penalty`.
  With blocked QPs gone, HS23 asls-* should no longer cycle under the decay. This is not measured.
- Small/medium tiers are not run: 99 scipy-solved unsolved active-set rows in v5 are outside tiny.

## H2 (rank 2): drop one negative multiplier at a time when the step is primal-feasible

**Claim.** The all-at-once refresh drops every row with λ_i < −tol in one go. When the step already satisfies all
linearised rows, this throws away rows that are still needed, and the loop wanders until `max_iter = 50`.
HS106 pasls repeats the same 20-set cycle every QP: |W| 8→5→7→5→8…, |W|=4 → 1 with a primal-feasible step and
λ_min = −8.6e3. Standard primal active-set practice (N&W Alg. 16.3) drops one row, `argmin λ_i`, when `p` is feasible.

**Must hold if true.** QPs that end on `max_iter_reached` fall on HS106/HS98.

**Evidence (in process, `none+d1` and `gi+d1`).**
- HS106 asls-pcg: f 13532 → 7179 (relgap 0.92 → 0.018, feas 1.9e-7). Still max_steps, outer step_collapse remains (G6 H1 territory).
- HS106 pasls: 14086 → 9981. Not solved.
- HS98 pasls: solved with `gi+d1` (10 steps), not with gi alone and not with d1 alone.
- HS114 exact: d1 alone solves it but needs 98 steps (gi: 12).
- No tiny-tier run for d1.

**Change.** Same file, about 6 lines in `ThresholdWorkingSetPolicy.propose`: if no row is violated at the step and more
than one row has λ < −drop_tol, drop only the most negative one.

**Cheap falsifier.** Count `max_iter_reached` QPs on HS106 asls-pcg with d1. If it does not fall, or HS106 does not move
towards f* within 500 steps, H2 has no value alone.

**Expected.** 1 confirmed (HS98 pasls, only stacked on H1). HS106 needs H2 plus the merit-penalty fix (not tested together).

**Risk.** More QP iterations per solve (one drop per iteration). Budget exhaustion can rise on problems with many
wrong-signed rows at the cold start. Measure tiny on `gi+d1` vs `gi` before keeping.

## Not a hypothesis: HS106 and the remaining lead tasks
HS106 (scipy-only win) is not blocked by the cap on any config. Its trajectory is a QP that runs out of budget (H2) plus an
outer crawl with ρ = 2.6e4 (G6 H1, decaying penalty). HS97/HS98 asls-pcg-exact end at the local point f = 4.0712 with gi.
HS114 pasls keeps 2-3 blocked QPs with no r_i > 0 (ratio test empty: inconsistent proximal linearisation). Leave that to the
proximal μ work.

## Re-measurement (2026-10-28, branch `working-set-exchange`)

Found again from the HS64 bound crossing while tracing the per-row merit penalty (branch `vector_rho`, independent of this change): at step 11 of HS64
asls-minresqlp / pasls the set {h0, l0, l2} is full (cap 3), l1 is violated by the step (x₂ 220 → −668) and cannot
enter; the QP reports `working_set_converged`. With the cap off, or with H1, HS64 solves in 19 steps (also under the per-row
penalty rule of `vector_rho`). This note was missed before the re-implementation.

Implementation `2083ba2` (`exchange_row` + `ThresholdWorkingSetPolicy.exchange_when_full`, default off, 9 tests): a
Gram--Schmidt QR of `[eq rows | selected rows]` gives α; leaving row `argmin_{αⱼ>0} λⱼ/αⱼ` (λ clipped at 0); unchanged when
a_p is independent of the span (rank-deficient set) or no αⱼ > 0. Gated on `full & any(rejected)` after `_select_rows`.

**Tiny tier** (flag on, scalar merit rule, measured on a tree whose solver code equals `main` plus this change; `diagnostics/exchange/`): craig 216→215, minresqlp 210→213, pcg 218→220,
pcg-exact 224→226, pasls 203→204. **+14 / −7, all 21 attributable** (re-run with the flag off and on).
- Gained: HS98 ×5 (all configs; 191 → 9 steps on craig/pcg/minresqlp), HS23 pcg-exact + pasls, HS116 minresqlp, HS81
  pcg-exact, DEGENQP / LSNNODOC pasls, OET1 pcg-exact, OET4 minresqlp, TENBARS1 pcg.
- Lost: CSFI2 pcg-exact (infeasible start; the first QP exchanges on clipped-zero multipliers and x₅ runs to 4.5e4),
  HS41 pcg-exact (infeasible start → degenerate vertex x = 0, 5 rows in 4-D, ∇f = 0: a first-order KKT point, f = 2 vs
  1.926), HS102 / HS69 / DUALC5 pasls, HS116 / TENBARS2 craig (not traced).

**Leaving-row variants** (the 21 tasks only): exchange only at feasible references 3 gains / 1 loss (HS98 and most gains
need exchanges at infeasible references); never remove a row violated at x_k + max-α ties 7 / 4 (HS98 lost: it needs a
violated row to leave). Plain ratio test is best.

Open: the rank-deficient case (a_p independent of a full, dependent set) is left unchanged; not counted yet. HS97 false
success of the 10-14 prototype not seen in this run (HS97 not among the changed tasks).

**G1** (all 388 unsolved asls / pasls tasks of v9, flag on, `diagnostics/wse/g1`): **21/388 solved, all attributable**
(the 7 not already in the tiny attribution re-run with the flag off: none solves). HS98 ×5, OPTCNTRL craig / minresqlp /
pcg / pasls (3 steps; the cap diagnosis of OPTCNTRL), GOULDQP1 minresqlp / pcg / pcg-exact, HS23 pcg-exact / pasls,
DEGENQP / LSNNODOC pasls, HS116 minresqlp, OET1 / HS81 pcg-exact, OET4 minresqlp, TENBARS1 pcg. Gap: sqpdax union
255 → 257 / 303, scipy deficit 9 → 8 (HS98 leaves). With the 7 tiny losses, per config approximately craig 228 → 228,
pcg 226 → 230, pcg-exact 239 → 242, minresqlp 218 → 223, pasls 209 → 211 (+21 / −7).

**Decision:** default `exchange_when_full=True` (`SingleExchangeWorkingSetPolicy` keeps False). Non-slow sqpdax suite
with the new default: 1819 passed, 9 skipped.
