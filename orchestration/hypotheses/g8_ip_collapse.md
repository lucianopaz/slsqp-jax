# G8 + G2 hypotheses: trip/tfip trust-region collapse and barrier stall

Inputs: `/tmp/digest_{unclassified,ip_barrier_stall,ip_bound_violation,ip_infeasible_stationary}.md`, v4 trajectories
(`diagnostics/v4/traj`), ledger (d-barrier-v1/v2, d-attribution, g1-joint-merit REJECTED), `g1_infeasible.md`.
In-process traces at HEAD (`local_bench_cleanup`), `TrustRegionInteriorPointMinimiser(atol=1e-6, curvature="secant")`, float64.
These match the v4 trip rows step for step until roundoff makes them diverge (HS1: identical to step 21).
Scratch scripts are in the session scratchpad (`run7.py` paired A/B with a monkeypatched projector, `run2.py` tr_step diagnostics).
Problems were re-typed by hand (HS1, HS15, HS86, HS100, BT7, an OET-like minimax). HS62, LOGROS and QINGB/PRICE4B bounds are guesses and are only used as controls.

## Tail taxonomy of the 42 `trust_region_collapse` tasks (last 4 steps of each v4 trajectory)

| class | n | signature | tasks |
|---|---|---|---|
| A | 13 (all trip, all constrained, all scipy-solved) | `rho = -inf` (pred <= 0) for 25-140 steps, radius driven to the floor, kkt 1e-6..1e-4 | HS15, HS19, SIPOW1/2, POLAK4, BT7 x2, CSFI1, HS83, HS86, OET5, HS107, HAIFAM |
| B | 22 (mostly bound-only) | finite rho < 0, \|rho\| shrinks roughly like Delta (second-order model error), floor reached | PALMER1 x2, PALMER2B/3B/4B, PALMER2 tfip, LEVYMONT/8/9 x2/10, DGOSPEC x2, PFIT3LS, DEVGLA1B, QINGB, PRICE4B, DIAGPQE, HS69 tfip, HS100 tfip, HS87, LSNNODOC |
| C | 7 | two rounding-level exits in a row (radius back to 0.25), no radius collapse | HS1, HS62, HS69, LEVYMONT7, PALMER2, PALMER3, MSS1 (trip) |

G2 `ip_barrier_stall` (49): at least 30 tasks (PALMER*A/E/B x2, LOGROS x2, DIAGPQB/T x2, PFIT4LS, DEVGLA1B) also fail on **every**
active-set secant config (`cycling` / `slow_progress`), so the cause is shared L-BFGS progress, not the barrier schedule.
In the tail, 70-100 % of steps are accepted with median rho 1.0-1.5. Trip-specific crawl: HS89/90/91/96 and OET2/4/6/7 (8 tasks).
There 100 % of steps are accepted, rho is constant in (0.27, 0.6), so the radius never grows (it needs rho > 0.75).
nu is 668-6e5 and the iterate is infeasible (merit - f = nu ||c||). See the open leads.

---

## H1 `ip-collapse-projection-refinement` (trip): the tangential step leaks out of null(Â) near a KKT point

**Claim.** `SteihaugTointCGTangentialStepSolver` (`subproblem/solver/steihaug_toint_cg.py:257`) projects with one pass of
`P v = v - Âᵀ (ÂÂᵀ)⁺ Â v` (`linalg/projection.py:201`). Near a KKT point the gradient is almost all in range(Âᵀ),
so the projected vector is tiny: `||P ĝ|| / ||ĝ|| ~ 1e-8..1e-10`. The absolute projection error `eps * kappa * ||ĝ||` is then a large
*relative* error in the tangential direction. The Steihaug step `w ~ Delta * Pĝ/||Pĝ||` therefore has `||Â w|| ~ (eps kappa ||ĝ|| / ||Pĝ||) Delta`.
That leak is linear in Delta, so it gives `v_pred = ||ĉ|| - ||ĉ + Â w|| < 0`. Once `nu` times the leak exceeds the objective decrease,
`pred = -q(w) + nu v_pred <= 0`. That gives rho = -inf at every radius, so the run ends in radius collapse, and kkt stalls just above atol.
The leak is real: the actual merit agrees with pred (HS15: actual -4.714e-3 vs pred -4.715e-3), so a different merit cannot fix it.
This is the mechanism behind the open lead "rho = -inf" in `g1_infeasible.md`.
One step of iterative refinement (Gould, Hribar & Nocedal 2001; N&W §16.3), `P(P v)`, removes it.

**Inequality that must hold.** At class-A iterates `pred < 0` exactly when `||Pĝ|| * Delta < nu * ||Â w||`, i.e. `||Pĝ||^2 < nu * eps_proj * ||ĝ||`
(eps_proj = absolute relative projector error). The resulting kkt floor is about `sqrt(nu * eps_proj * ||ĝ||)`.
After refinement `||Â P(P v)|| / ||P v|| = O(eps)`, and `v_pred >= 0` for a zero normal step.

**Evidence.**
- HS86 trip (collapse at HEAD): over the last 12 steps `v_pred/Delta = -5.66e-7`, `obj/Delta = -4.5e-6`, `nu = 25.3`. So `nu*5.66e-7 = 1.4e-5 > 4.5e-6`
  and rho = -inf from Delta = 1 down to 6e-8. kkt is stuck at 2.56e-6. Measured projector: `||Â Pv||/||v|| = 4e-14..8e-14`, `||Pv||/||v|| = 1e-8..1e-10`
  (relative leak 1e-5..1e-6). After a second projection `||Â P(Pv)||/||v|| = 1e-23..1e-26`.
- HS15 trip: `v_pred = -1.33e-7` at Delta = 1, scales with Delta, `nu = 3.5e4`, pred = actual = -4.7e-3; kkt stuck at 2.9e-6.
- Paired A/B over 12 starts (x0 and 11 perturbations, same starts in both arms), trip, `P` -> `P∘P` in the Steihaug solver only:
  HS86 3/12 -> 12/12 successful (steps 30-35 instead of 58-194), HS100 1/12 -> 12/12, HS15 2/12 -> 12/12,
  BT7 7/12 -> 8/12. Controls: HS1 (bounds only) 9/12 -> 9/12, QINGB/PRICE4B unchanged.
- v4 class A: 13 trip tasks, 13 scipy-solved, all with `meq + mineq > 0`, 25-140 rho = -inf steps each, kkt_last 1.35e-6 (CSFI1) .. 6.3e-6 (HS86).

**Change.** `slsqp_jax/sqpdax/subproblem/solver/steihaug_toint_cg.py`, about 3 lines. Rename the built projector `proj1` and set
`proj = lambda v: proj1(proj1(v))`, used both for the initial residual and in `residual=`. Add one parametrised case in
`tests/sqpdax/subproblem/solver/` for the Steihaug solver: a constraint matrix with gradient `ĝ = Âᵀλ + δ t`, `δ = 1e-9`.
The returned step must satisfy `||Â w|| <= 1e-12 ||w||`. That case fails with a single projection. No public API.
(Alternative with the same effect in one file: refine inside `null_space_projector`'s `proj`. That also changes tfip, see H2.)

**Cheap falsifier.** Run CSFI1 and HS83 trip (v4 class A, scipy-solved) with the change.
H1 is false if `rho = -inf` still appears in the last 10 steps, or if `tr_step.v_pred < 0` with `||normal_step|| < 1e-12`.
Cheaper, without code: with `run7.py` on any class-A problem, check that the `tr_step` diagnostic `v_pred` is negative and proportional to `radius` at the stuck iterate.

**Expected solved in the group.** 4-7 of 13 class A (CSFI1, HS83, HS86, OET5, HS107, HS19 likely). Not BT7 x2 or HS15: their f_gap against the catalogue f* is 0.15-0.18 (another local solution), whatever the status.
SIPOW1/2, POLAK4, HAIFAM have kkt 3e-3..0.8, so they are less likely. Spillover:
G1 `ip_bound_slack_residual` and the rho = -inf lead (DUALC*, CLEUVEN7, TRO21X5, HS116). These use the matrix-free projector, whose inner-CG
error (tol 1e-10) is amplified the same way. Also tiny-tier HS30 trip (kkt 2.7e-6..9.7e-6 > atol in the d-attribution).

**Risk.** trip only. The cost of a tangential CG iteration doubles (one more `(ÂÂᵀ)⁺` solve). On the matrix-free path (`m > schur_max_rows = 100`:
OET*, SIPOW*, ELATTAR, DUALC*) that is up to 100 more inner CG iterations per projection, so watch `run_timeout`.
Steps that used to be rejected now pass, which changes trajectories of solved trip tasks. Run the tiny-tier trip regression.

---

## H2 `ip-collapse-projection-refinement-tfip`: the same refinement in the funnel tangential step

**Claim.** `FunnelTangentialStepSolver` (`subproblem/solver/funnel_tangential_step.py:284`) uses the same single-pass projector, so tfip's
f-iterations show the same leak. Their predicted decrease `Δm_f` includes the leak, and the funnel rejects or shrinks.

**Inequality.** Same as H1, with `nu * v_pred` replaced by the funnel's `v`-model change of the tangential step.

**Evidence.** Paired A/B, tfip, 12 starts, with `P∘P` in both tangential solvers: HS100 3/12 -> 8/12, HS86 4/12 -> 11/12.
**HS15 4/12 -> 3/12: 9 runs go from `secant_recovery`-type failures to max_steps (500).** So refinement on tfip unmasks another stall.
G8 has only 6 tfip tasks, mostly class B (bound-only, where H1 showed no effect). The constrained ones are HS100 and HS69.

**Change.** `slsqp_jax/sqpdax/subproblem/solver/funnel_tangential_step.py`, about 3 lines (as H1). Test: the H1 test parametrised over both tangential solvers.

**Cheap falsifier.** HS100 and HS69 tfip from the v4 x0. H2 is false if either still ends `trust_region_collapse` with kkt < 1e-4.
Run HS15 tfip too: if it ends `max_steps_reached`, the regression above is real.

**Expected solved.** 1-2 of G8 tfip (HS100, maybe HS69). 0-3 in `ip_bound_violation` (OPTCNTRL, DEGENLPA/B, TRO*: eq-constrained, kkt 1e-5..1e-2).

**Risk.** Measured: HS15 tfip gets worse. Do only after H1, as a separate branch.

---

## H3 `ip-collapse-rounding-acceptance` (trip): rounding-level steps are rejected instead of taken

**Claim.** Near a solution with large `|f|` or large `nu`, the predicted reduction falls below `100 eps max(1,|merit|)`. It is inflated by
`nu * v_pred`, where `v_pred` is at roundoff (BT7: `pred = 3.6e-12`, `nu = 4035`, step 4.5e-9). The trial merit is then bit-identical (`actual = 0`), so rho = 0 and the step is rejected.
The xtol exit reduces mu, the next step is rejected the same way, and the run ends `trust_region_collapse` with kkt 2e-6. The model step itself is a Newton step
(interior, cg = 2) that would cut kkt. Accepting a rounding-level step whose actual change is also at rounding level keeps Newton going. The KKT test is unchanged, so no tolerance is loosened.

**Inequality.** At the exit: `|pred| <= 100 eps max(1,|merit0|)` and `|actual| <= 100 eps max(1,|merit0|)`, and E(.;0) at the Newton trial point is below E(.;0) at x.

**Evidence (weak).** Paired A/B, trip, 12 starts: BT7 7/12 -> 9/12, HS1 9/12 -> 9/12 (HS1 collapses come from genuine negative-rho model failures).
v4 class C: HS69 trip (kkt 3.4e-6), LEVYMONT7 (1.3e-6), PALMER2/3 (2e-5), HS62 (2.4e-6). Not measured on those.

**Change.** `slsqp_jax/sqpdax/step_controller/trust_region_radius.py`, about 6 lines: `accepted = (rho >= eta) | (rounding_level & (|actual| <= thr))`, keep the radius,
and report `rounding_level` False on such steps so that `_advance_dynamics` does not fire the xtol exit. Test: one parametrised case in `tests/sqpdax/step_controller/`.

**Cheap falsifier.** HS69 and LEVYMONT7 trip. H3 is false if they still end `trust_region_collapse`, or if kkt does not decrease after the accepted rounding-level steps.

**Expected solved.** 1-3. **Risk.** It could loop on noise-level steps up to max_steps when kkt cannot get below atol (PALMER2/3 with f ~ 3e3). That replaces `trust_region_collapse` with `max_steps`; both are unsolved, but it costs time.

---

## Measured and dropped (do not retry without a new argument)

- Relative rounding floor (`|merit0|` instead of `max(1,|merit0|)` in `at_rounding_level` for the TR manager): paired 12 starts, HS1/PRICE4B/QINGB unchanged (0 effect).
- Shrink to `0.25*min(Delta, ||w||)` on rejection: identical repeat rejections are 74 / 12500 steps in G2 trip and 266 / 4509 in G8. Saves steps only, so it cannot fix a collapse.
- Scalar `B0 = (yᵀy/sᵀy) I` instead of the per-variable diagonal: LOGROS still at 500 steps, HS38 worse.

## Open leads (not proposed)

- **Class B (22)**: |rho| shrinks like Delta, so the true curvature along p is about 1e3-1e4 times the L-BFGS curvature (PALMER1: K ~ 6e3 at Delta = 2.4e-7, kkt 3.4e-4).
  For f ~ 1e3-1e4 the merit cannot resolve the decrease needed for kkt <= 1e-6 (`kkt^2/H < eps |f|`). This is a precision floor plus a secant-quality problem,
  shared with asls. A candidate is a secant update on *rejected* trust-region trials (N&W Alg. 6.2), which needs the trial point in `StepResult`.
- **G2 trip crawl (8)**: constant rho in (0.27, 0.6) with nu in [668, 6e5] on infeasible iterates. Candidate: Maratos effect (second-order constraint term times nu), remedy a second-order correction.
  Falsifier first: on HS89 trip log `||c(x+p)|| - ||c + A p||` against `(1-rho) * v_pred`.
- OET-like minimax (m = 102, matrix-free): `v_pred < 0` comes from the **normal** step (|n| >> |p|). Double projection does not help (0/12 -> 0/12).
  Check `m(w_n) < m(0)` in `ScaledNormalStepSolver` (g1 open lead (i)).
- G2 (>= 30 of 49) is shared L-BFGS slow progress (PALMER/LOGROS/DIAGPQ fail on all secant configs). Belongs to G6/G3, not to the barrier schedule.
