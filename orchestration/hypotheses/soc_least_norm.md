# Second-order correction (SOC) against the Maratos effect: least-norm correction, active-set line search

Status: **rejected as a default** (2026-10-27). The code is kept, switched off (`step_correction=None`), on branch `soc`
(15 commits `6f8a5e7..b426c25`, default turned off in `abd7df3`). Ledger id `soc-least-norm-default`.

## Hypothesis

`scripts/trace_solver.py` traces of two v9 failures solved by a scipy baseline and by no active-set config:

- **HS98, asls-craig.** The QP direction satisfies the linearised constraints, but the bilinear term `−169·x1·x3` makes `c0(x+d)`
  miss by 1.96. With the L1 penalty ρ ≈ 640, Armijo accepts only α ≈ 1e-3 for 183 steps, and then the iterate falls into the
  x3 vertex (f = 4.0712 vs f* = 3.1358).
- **TRO4X4, asls-craig.** The objective is linear and the 24 equalities are bilinear. The true violation grows as `κt²` along d
  (1.23 / 0.0123 / 1.2e-4 at t = 1 / 0.1 / 0.01). With ρ = 200–1500, α ≈ |gᵀd|/(ρκ) ≈ 2⁻⁹…2⁻⁶ for 325 steps, and stationarity
  stays at 0.41.

Claim: a least-norm second-order correction `d̂ = −Aᵀ(AAᵀ)⁺ c(x+d)` on the QP working set (Nocedal & Wright Alg. 18.3), tried
before backtracking, removes the `κt²` error and lets full steps through, so these and similar `merit_stagnation` /
`step_collapse` / `cycling` / `slow_progress` tasks would solve.

## Alternatives considered (user discussion, before implementing)

| Strategy | Verdict |
|---|---|
| Filter instead of merit (Fletcher–Leyffer, Wächter–Biegler) | Later. Large change to counters and restoration; IPOPT and filterSQP still add an SOC. |
| Quadratic constraints in the subproblem (SQCQP) | Rejected: nonconvex for bilinear rows; secant configs have no constraint Hessians. |
| SOC (this file) | Tried first: the ingredient the other approaches also ship. |
| Non-monotone acceptance (GLL / watchdog) | Second choice. A GLL window can't absorb the +333 merit jumps seen on TRO4X4 at ρ = 1500. |

scipy trust-constr's SOC (`_trustregion_constr/equality_constrained_sqp.py:170-189`) was taken as the reference:
`y = −Y c(x+d)` with `Y = Aᵀ(AAᵀ)⁻¹`, tried only when the ratio test fails and `‖d_n‖ ≤ 0.1‖d_t‖`, scaled by the largest
`t ∈ [0, 1]` that stays in the box, accepted against the original predicted reduction.

## What was implemented (branch `soc`)

- `SubProblem`: `constraint_residual`, `violated_part`, `range_space_solve` (matrix-free CG; Schur override for the scaled
  barrier), `least_norm_step`, `to_model_primal` / `to_native_primal`. Proximal subproblems drop their eliminated equality rows.
- `linalg.box_segment_fraction`.
- `step_controller.correction`: `StepCorrection`, `LeastNormCorrection` (gate on the violated residual, round-off snapping,
  box scaling). Written only against the `SubProblem` API, so it serves active-set and interior-point subproblems alike.
- `StepController.correction` with a `supports_correction` capability flag. Correction stages in `LineSearch` (Armijo with the
  slope of d) and `TrustRegionManager` (ratio against the original predicted reduction). `TrustFunnelManager` refuses one.
- `CommonMinimiser.step_correction` (option `options["minimiser"]["step_correction"]`: an instance, `None`, or a mapping of
  `LeastNormCorrection` fields), `_bind_step_correction`, `_correction_subproblem` (asls: the QP's final working set),
  `n_soc_accepts` / `soc=` diagnostics. Re-enable with `{"minimiser": {"step_correction": {}}}`.

## Measurements (all float64, this machine)

**Spot checks** (`scripts/trace_solver.py`, `diagnostics/soc` scratch logs):

- HS98 asls-craig: unchanged. The correction is never valid. The iterate violates the x6 upper bound by 0.00865 from step 1
  (the working-set cap leaves no room to add `u5`). With h0 in the working set, restoring that bound needs x3 += 0.063, so
  `‖d_n‖ = 0.064 > 0.1 ‖d_t‖ = 0.031`. The near-feasibility gate (scipy's rule) correctly says "not near feasible".
- TRO4X4 asls-craig: 387 attempts, 4 accepted, 294 valid but rejected by Armijo, 89 invalid. The box fraction t is mostly
  0.001–0.19: the correction pushes variables (truss areas near their lower bound 0) out of the box. Componentwise clipping
  instead of scaling was tried in a scratch run: 5 accepts, worse end point. The QP steps are 1–14 long, far outside the
  region where an O(‖d‖³) correction is accurate, so the corrected point still misses by `O(‖d‖·‖w‖·‖Q‖)` with large bilinear
  coefficients.
- Path changes on TRO4X4 (both still unsolved): asls-craig `max_steps_reached` f = 5.63 → zero-step-guard `successful`
  f = 5.58 (stat 0.377); asls-pcg f = 0.0087 → 6.78.
- Maratos test problem (N&W Ex. 15.4, float64): asls secant 7 → 4 steps; asls exact 7 → 8 steps.

**G1** (`diagnostics/soc/g1`, `compare.json`; the 129 active-set rows of the v9 groups merit_stagnation, step_collapse,
cycling, slow_progress, local_minimum): 6/129 newly solved (HS68, HS97, GOULDQP1, DUAL2 pasls; HS101 asls-minresqlp;
OPTPRLOC asls-pcg-exact).

**G2** tiny tier, 7 sqpdax configs (`diagnostics/soc/regress_tiny`, `regress_join.json`, baseline
`results/final/results_rescored.jsonl`): gained 9, lost 6. asls-craig 216 → 216, asls-minresqlp 210 → 210, asls-pcg 218 → 217,
asls-pcg-exact 224 → 225, pasls 203 → 206, trip 170 → 170, tfip 172 → 172. Union 239 → 240 (TENBARS1), deficit vs scipy 3 → 3.

**Attribution** (each task re-run on the `soc` tip with `--set step_correction=None` and with the correction on): all 6 G1
gains and all 15 tiny-tier differences give **identical** results with and without the correction. Every measured change is
run-to-run / baseline noise, not the correction.

## Conclusion

The correction changed no solved task on G1 or the tiny tier, and it made the TRO4X4 asls-pcg path worse. It does not address
the failures that motivated it:

1. On HS98 the real defect is the step-1 bound overshoot (working-set cap at `n − meq` rows); the iterate is never
   near-feasible, so a scipy-style SOC never fires.
2. On TRO4X4 the steps are far too long for a local correction; the crawl is a step-length / model-validity problem plus a
   bound-blocked correction, not the near-solution Maratos effect.

Default turned off; the SOC machinery is kept because it is tested, generic (active-set and interior-point, line search and
trust region), and costs nothing when `step_correction=None`.

## Do not repeat / next leads

- Do not re-run a scipy-style least-norm SOC on the active-set line search as a default without a new mechanism; this exact
  configuration measured zero attributable effect.
- If revisited: a bound-aware correction (fix the coordinates the correction would push out of the box and re-solve, as in the
  restoration step's projection rounds, or re-solve the QP with the shifted right-hand side, SNOPT/filterSQP style). Only worth
  it near a solution; TRO4X4 needs shorter, better-modelled steps first.
- Fix the HS98 working-set-cap overshoot (a violated bound that cannot be added to a full working set) before any end-game work
  on that cluster.
- trip: the `TrustRegionManager` stage exists but is untested on the benchmark; trace the trip end-game collapses for `κt²`
  growth before switching it on.
- Watchdog / filter acceptance remain untried.
- The float32 test on the Maratos problem showed an asls exact-curvature run ending in `merit_stagnation` at a rounding-level end
  game (f = −1 to all float32 digits). That's a float32 precision artefact, not an SOC defect, but worth knowing for float32 users.
