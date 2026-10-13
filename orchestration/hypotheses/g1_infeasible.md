# G1 hypotheses: infeasible starts, over-determined working sets

Inputs: `/tmp/digest_{kkt_solver_failure,false_infeasible_certificate,restoration_failure,ip_bound_slack_residual}.md`,
`digest.py --cause near_feasible|nonfinite` (v4), trajectories `diagnostics/v4/traj/*`, ledger (r2a, r2, b1-cap, b-restoration, b-final).
Group size N = 99 (kkt_solver_failure 20, false_infeasible_certificate 13, restoration_failure 23,
ip_bound_slack_residual 31 (all trip), near_feasible 11, nonfinite 1). Gate G1 needs S >= max(2, ceil(0.03 N)) = 3.

What is different from round b: round b only touched the active-set family (row cap, independence screen,
violation-gradient certificate, CGLS restoration). Its restoration fires only on a *rejected* line search.
trip was untouched (0/36). The v4 trip trajectories show that the trip failures are not caused by a missing restoration
phase. trip's acceptance test is inconsistent with its own model to first order (H2), and it has a NaN fixed point (H3).
A Byrd-Omojokun trust region with a consistent merit already drives `||c||` to a stationary point, so a restoration
phase on top of an inconsistent merit would hit the same rejection. None of the hypotheses removes slacks or loosens a tolerance.

Ranking is by expected benefit per changed line, with one exception: H3 is cheaper per line than H2,
but it cannot pass Gate G1 alone (about 1-2 tasks), so it is ranked third.

---

## H1 `restoration_failure-null-step-trigger`: restoration never fires on an accepted null QP step

**Claim.** In `ActiveSetLineSearchMinimiser._restoration_fallback` (`minimiser/active_set_linesearch.py:633`),
`attempt = infeasible & ~result.accepted`. When the QP returns `d ~ 0` at an infeasible point, the Armijo search
*accepts* the null step: `phi(x + a d) = phi(x)` and the slope is about 0. So restoration is never tried, the
stagnation counter reaches its patience and the run ends `infeasible`. The violation is not stationary there,
so the existing CGLS restoration step would reduce it.

**Inequality that must hold.** At the stuck iterates:
`||d_QP|| * alpha < atol`, `v(x) = ||c+(x)||_inf > atol`, and the violation stationarity
`||P_box(J_v^T c_v)|| / ||c_v||` (`stats_violation_stationarity`) is O(1). The restoration step then predicts
`dv = m_v(0) - m_v(d_R) > 0`.

**Evidence** (v4 rows with status infeasible / infeasible_stationary / merit_stagnation, `dnorm_last < 1e-8`, `feas > 1e-6`; 13 tasks):
- HS83 on all 5 active-set configs: feas 0.19, `dnorm` 1.4e-14 from step 2 to step 11, accepted (alpha 0.25),
  stall 1..10, `n_restoration_steps = 0`, `violation_stationarity = 1.0`. scipy-slsqp and trust-constr both solve it.
- HS116 asls-pcg-exact: feas 3.3e-3, dnorm 0 for the last 6 steps, stall 5..10, viol_stat 0.80 (scipy solves).
- HS81 pasls: feas 0.01, dnorm 0, nrest 0, viol_stat 0.49 (scipy solves).
- HS68 pasls: feas 7.6e-5, dnorm 0, viol_stat 1.0 (scipy solves; nrest 62 already, so less likely to flip).
- No scipy reference: DEGENLPA asls-minresqlp (feas 6.9e-5), FLT/MSS1 asls-craig, HS88/HS90 pasls.

**Change.** `slsqp_jax/sqpdax/minimiser/active_set_linesearch.py`, about 3 lines:
`null_step = result.accepted & (result.step_size * result.proposed_step_norm < self.atol)` (same test as `zero_step`, l. 1040-1045),
`attempt = infeasible & (~result.accepted | null_step)`. Add one parametrised case to the existing restoration test in
`tests/sqpdax/minimiser/` (QP stub that returns `d = 0` at an infeasible point leads to `n_restoration_steps == 1`). No public API.

**Cheap falsifier.** Run HS83 on asls-pcg alone with the `restoration` logger at info. H1 is false if the
"trying the restoration step" line shows `predicted decrease <= 0`, or if the restoration line search is rejected on
every one of steps 2-11.

**Expected solved in G1.** 4-7 (HS83 x5 behave alike; HS116, HS81; HS68 less likely). All 8 scipy-solved ones are in the scipy deficit tally.

**Risk to other causes.** The change applies only at infeasible iterates with a null accepted step. Feasible-point termination
(`qp_optimal`, `zero_step_patience`, G3/G5) is unchanged. Problems that really are infeasible (`problem_infeasible`, 4)
can loop longer before the certificate. A successful restoration counts as progress (l. 1028), so they could end at
`max_steps` instead of `infeasible`; both outcomes are unsolved. pasls zero-step logic (`proximal_active_set_linesearch.py:262`)
reads `consecutive_zero_steps`, which still excludes restoration steps.

---

## H2 `ip_bound_slack_residual-joint-merit`: trip's merit uses a split norm, its model a joint norm

**Claim.** trip's predicted reduction (`subproblem/solver/trust_region.py:338-353`) uses N&W (19.41) with a single joint norm,
`m(p) = ||c_hat + A_hat w||_2` over `[eq; ineq+s; lb rows; ub rows]`. The merit that measures the actual reduction
(`TrustRegionInteriorPointMinimiser._step_controller`, `NormMerit(norm=2)`) uses
`||g(x)||_2 + ||[h+s; lb-x+s_lb; x-u+s_u]||_2`, a sum of two norms. N&W (19.40) prescribes the same joint norm as (19.41).
When both blocks are non-zero, the two directional derivatives differ:
`D_split(p) = g^T J_g p / ||g|| + b^T J_b p / ||b||` but `D_joint(p) = c^T J p / ||c||`.
So rho does not tend to 1 as Delta goes to 0. When the normal step trades a little equality residual for bound/inequality
residual (`D_joint < 0 < D_split`), rho tends to a negative constant, every step is rejected down to `radius_floor`,
nu is pumped up (eq. 19.42), and the run ends `trust_region_collapse`. tfip already scores violation with the joint
`ConstraintViolation` and shows 1 collapse in 49 eq+ineq tasks.

**Inequality that must hold.** At the collapse iterates, `lim_{Delta->0} rho = D_split(p_hat) / D_joint(p_hat) < 0`
(nu dominant), while with the joint merit `lim_{Delta->0} rho = 1`.

**Evidence.**
- All 14 trip G1 tasks ending `trust_region_collapse` have `meq > 0` and a non-zero bound or slack-row residual. Median rho on steps
  with radius < 1e-3: FLETCHER -61, CSFI2 -1.0, HS114 -4.4, TENBARS1-4 -43/-179/-206/-82, DEGENLPA -21, DEGENLPB -0.66,
  TRO3X3 -9.8, TRO4X4 -4e-4, TRO5X5 -2.9, OPTCNTRL -34.5, TABLE7 -1220. The meq = 0 trip tasks never show this
  (HS97 +0.20, HS98 +0.54).
- OPTCNTRL trip (eq 0.020, lb 0.0197 both non-zero): rho = -169, -57, -58, -54, -28, -25 while Delta goes 4.9e-4 -> 4.8e-7,
  pred > 0 throughout (finite rho), nu 519 -> 6.6e4. HS114 trip (eq 0.028, ineq 0.107): rho -1.19 / -4.6 at every Delta from 128 to 6e-5.
- Population (results/new_run): trip eq+ineq/bounds 20 collapses / 58 (34%), no-eq 27/180 (15%), eq-only 3/64 (5%).
  For eq-only the two norms coincide.
- DUALC1/2/5/8 accepted steps show rho 3-19 at small Delta (actual > pred): the same mismatch with the opposite sign.

**Change.** `slsqp_jax/sqpdax/minimiser/trust_region_interior_point.py`, about 15-20 lines: a module-private
`_JointViolationMerit(Merit)` that returns `NormMerit(problem, barrier, feasibility_weight=0, norm=2)(x) + nu * ConstraintViolation(problem, norm=2)(x)`
(the same pair tfip uses in `trust_funnel_interior_point.py:524-536`), used in `_step_controller`. Add a test in
`tests/sqpdax/minimiser/test_trust_region_interior_point.py`: on a 2-var problem with one equality and one violated bound,
rho tends to 1 for a scaled-down step. No public API; `NormMerit` itself (used by asls/pasls) is untouched.

**Cheap falsifier.** At the stored OPTCNTRL (or HS114) trip iterate, after one step with Delta = 1e-6, compute rho with both merits for
the solver's step. H2 is false if the split-merit rho is about 1, or if the joint-merit rho also stays < 0.
Without code changes: if `feas_eq` or the bound/slack block were exactly 0 at these points, the mechanism would be the
`||.||` kink instead. The v4 rows show both blocks non-zero for all 14.

**Expected solved in G1.** 3-5 (CSFI2, HS114, OPTCNTRL scipy-solved; TRO4X4 trust-constr-solved; TENBARS/DEGENLP/TRO*/TABLE7 have no scipy-slsqp reference).
Spillover to G8 (the other 6 trip eq+ineq collapses) and to G2 (trip `ip_barrier_stall`).

**Risk.** trip only. 23 currently solved trip eq+ineq tasks change trajectory. joint <= split, so the merit penalises
infeasibility a little less; check the trip tiny-tier rows. No effect on tfip/asls/pasls.

---

## H3 `ip_bound_slack_residual-nan-rho`: NaN rho freezes trip forever

**Claim.** `TrustRegionManager.step` (`step_controller/trust_region_radius.py:159-177`): when the trial merit is NaN
(trial `x` outside the domain of `f`, e.g. fractional powers), `rho = NaN`. Both `rho >= eta` and `rho < shrink_threshold`
are then False, so `x_{k+1} = x_k` and `Delta_{k+1} = Delta_k`. The outer map has a fixed point and every later iteration
is identical. The funnel controller does shrink on NaN, so tfip is not affected.

**Inequality.** For a fix to be needed: `rho_k = NaN` for all `k >= k0` with `Delta_k` constant. After the fix, `Delta_{k+1} = 0.25 Delta_k` until the trial point is in the domain.

**Evidence.** HS101 trip: NaN rho on 487/500 steps, Delta = 2.0 and kkt = 9.12 constant from step 14 to 500.
HS102 trip: 488/500. DALLASS trip: 499/500 from step 1. No other trip run in v4 has a NaN rho.

**Change.** `slsqp_jax/sqpdax/step_controller/trust_region_radius.py`, 1 line: `rho = jnp.where(jnp.isnan(rho), -jnp.inf, rho)`.
Add one parametrised NaN-merit case in `tests/sqpdax/step_controller/` (rejected, radius shrunk). No public API.

**Cheap falsifier.** Re-run HS101 trip. H3 is false if the radius still does not change after step 14.
If it shrinks but the run then ends `trust_region_collapse` at kkt ~ 9, the fix works but solves nothing.

**Expected solved in G1.** 1-2 (HS101, HS102 are scipy-slsqp-solved; DALLASS has no reference). Below the G1 gate alone.

**Risk.** None for runs without a NaN trial merit. `TrustRegionManager` is used only by trip.

---

## Open lead, not proposed: rho = -inf on trip (v_pred <= 0)
DUALC1/2/5/8, CLEUVEN7, TRO21X5 and HS116 have 127-169 steps with rho = -inf. In `trust_region.py` this happens only when
`m(0) - m(w) <= 1e-30`: the composite step loses the normal step's Cauchy decrease. nu then explodes (DUALC2 1.6e7).
6 of the 7 have `meq + mineq > schur_max_rows = 100`, so the projector is matrix-free with an inner CG (100 iterations).
Two candidate mechanisms:
- (i) `steihaug_toint_cg.py:279` warm-starts from `free_w * n`, which zeroes the normal step on GP-active bound columns,
  while `ScaledNormalStepSolver` does not freeze them.
- (ii) inexact null-space projection.

A safeguard `m(w) <= m(w_n)`, otherwise fall back to the normal step, would cover both (about 10 lines in `trust_region.py`).
Falsifier first: log `m(w_n)` (normal `dm_v_n`), `m(w)` and `any(active_bounds)` on DUALC5 trip at the -inf steps.

Not covered: Craig/MINRES-QLP `kkt_solver_failure` on rank-deficient *equality* Jacobians (MSS1, FLT, TABLE7, DIAGPQB/T).
Those need a regularised projector.
