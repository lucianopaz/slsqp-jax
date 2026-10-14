# G8/G1 design notes (stream B, Oct 14): DUALC normal step, class-B collapses, OET crawl

## DUALC1/2/5/8 (trip, infeasible start): scaled normal step (design only, not implemented)

Evidence (DUALC5 trip, n=8, 1 eq, 277 ineq): `ScaledNormalStepSolver` hits its CG cap (n_cg = 50) every call; the Cauchy
step length is alpha* = pi_v^2 / ||A d||^2 = 4e-9..1e-5, i.e. ||A|| ~ 1e4 from the inequality rows, and v (= 0.92, all of it
in the equality row) drops by 1e-4..2e-4 per step. Raising the CG cap to 500 changes nothing (feas 0.84-0.92).
Exact projection (CG budget >= m) removes the leak |A w| > |A w_n| but solves nothing (ledger g1-projector-cg-budget).

Why it is not a 50-line change: the normal step minimises ||c + A w||_2 over the ball, and the same norm is the
merit's feasibility term and the `v_pred` in the penalty rule (`trust_region.py`: nu >= obj/((1-rho) v_pred)). Scaling rows
by D = diag(1/||row_i||) in the step only changes the minimiser but not the merit, so the Cauchy/dogleg decrease of
the *unscaled* model can fall (the comparison `use_cauchy` mixes them); scaling the merit consistently touches
`merit.py` (`NormMerit`, shared with the active-set line search) and the funnel's `v`. Row norms of A are not available
matrix-free (only products); a probe with `A A^T`-diagonal estimates (Hutchinson) costs m products per step.
Proposed design: (i) estimate row scale r_i = ||A^T e_i|| once per `minimise` call from the Jacobian at x0 (dense J is
available in the benchmarks' `Problem`; the scaled slack rows are S-weighted and need the current slack), (ii) build
`D` as part of `ScaledBarrierSubProblem` and use ||D(c + A w)|| in BOTH `model_v` (normal step, `m0`, `mp` in
`trust_region.py`) and `NormMerit(norm='scaled')`; (iii) measure on DUALC1/2/5/8 + tiny trip. Estimated 80-120 lines over
`scaled_barrier.py`, `scaled_normal_step.py`, `trust_region.py`, `merit.py` (merit.py is shared with stream A: ask first).

## Class B collapses (trip/tfip `unclassified`, 46 tasks in v5: HS1, QINGB, PRICE4B, PALMER*, LEVYMONT*, DGOSPEC, ...)

HS1 trip trace (rad, pred, actual, rho, |d|): the unconstrained (interior) Newton step has |d| = 1.3e-5 < radius, rho = -2.7
and repeating it with radius 1e-3 .. 1.5e-5 gives the *same* rejected step four times (the radius is not binding) before
it finally binds; the last exit has pred 6e-15 vs actual -1.6e-14 at |d| = 3e-8: the L-BFGS model under-estimates the
curvature along the Rosenbrock valley by a large factor, and no accepted step supplies a corrective (s, y) pair.
Candidate (needs `secant/**`, `minimiser/base.py`, owned by stream A): feed the *rejected* trial pair (s = d, y = grad L(x+d) - grad L(x))
to the L-BFGS update when rho < 0 (N&W alg. 6.2 variant), at the price of one extra gradient evaluation per rejection;
and shrink to `min(radius, 0.5 |d|)` on a non-binding rejection (measured earlier to save steps only).

## OET2/4/6/7 trip crawl (n=3..4, 1002 ineq rows)

With the cap-50 penalty (merged) the crawl persists at nu = 50 (the cap), rho = 0.249..0.252 on every step, radius 3.1e-2, |d| 5e-2, merit
falling 1.3e-3 per step from 2.0. Lowering the shrink/grow thresholds (0.1/0.2, 0.1/0.5, 0.0/0.3) leaves OET2/4/6/7 at max_steps (HS89/HS90/HS96 get faster: 132/54/58 steps instead of 360/86/404 at 0.1/0.2).
rho constant at 0.25 means actual = pred/4 at every radius: a systematic second-order effect of the 1002 semi-infinite
constraints (pred counts the linearised infeasibility decrease with weight nu = 50), i.e. Maratos-type; a second-order correction (project the trial point back with
one more normal step) is the standard remedy and needs the constraint values at the trial point (extra evaluation).

## Update (Oct 14, batch 7): the scaled normal step is falsified for DUALC

Cheap test of the premise before implementing it: wrap the benchmark problem so that every inequality row is divided by
max(1, ||grad c_i(x0)||) (scratch worktree hyp/scratch-dualc-rowscale, `benchmarks/problems.py`, env ROWSCALE=1; not committed
anywhere). DUALC1/2/5/8 trip, tiny tier: unscaled: max_steps, feas 0.92-0.98; row-scaled: trust_region_collapse at step ~140, feas 0.91-0.93 (unchanged).
So the row scaling of the least squares does not unlock the infeasible start; do not implement the scaled normal step/merit.
Remaining explanation: the 277 inequality rows with their seeded slacks (and the ``x >= 0`` bounds at ``x0``) cap the feasible
normal step through the fraction-to-boundary on the slacks; the equality row ``sum x = 1`` is only reachable by moving
``x`` by O(1), which the slack rows forbid at the current slack values. A restoration phase that frees the slacks (as asls does)
or re-seeds slacks from the violated rows is the next idea.
