# slsqp-jax

[![Build](https://github.com/lucianopaz/slsqp-jax/actions/workflows/test.yml/badge.svg)](https://github.com/lucianopaz/slsqp-jax/actions/workflows/test.yml)
[![Documentation](https://readthedocs.org/projects/slsqp-jax/badge/?version=latest)](https://slsqp-jax.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/lucianopaz/slsqp-jax/graph/badge.svg?token=K6Y9JBL6F2)](https://codecov.io/gh/lucianopaz/slsqp-jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI - Version](https://img.shields.io/pypi/v/slsqp-jax)](https://pypi.org/project/slsqp-jax/)
[![Open In Colab: CPU vs GPU](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lucianopaz/slsqp-jax/blob/main/benchmark_cpu_gpu.ipynb)
[![Open In Colab: Solver Configs](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lucianopaz/slsqp-jax/blob/main/benchmark_solver_configs.ipynb)

A pure-JAX implementation of the **SLSQP** (Sequential Least Squares Quadratic Programming) algorithm for constrained nonlinear optimization, designed for **moderate to large decision spaces** (5,000–50,000 variables). All linear algebra is performed through JAX, so the solver runs natively on CPU, GPU, and TPU, and is fully compatible with `jax.jit`, `jax.vmap`, and `jax.grad`.

The `SLSQP` solver is built on top of [Optimistix](https://github.com/patrick-kidger/optimistix), a JAX library for nonlinear solvers. It implements the {class}`optimistix.AbstractMinimiser` interface, so you run it through the standard {func}`optimistix.minimise` entry point — no manual iteration loop required.

> **Breaking API change.** The previous flat `SLSQP(rtol=…, atol=…, max_steps=…, …)` constructor has been replaced with a single nested `SLSQPConfig` argument. The basic example below uses the new layout; users on `slsqp_jax.compat.minimize_like_scipy` need no changes. See the project's `REFACTOR_NOTES.md` (in the repo root) for the full migration mapping (every removed flat kwarg listed alongside its new sub-config home) and a side-by-side migration example.

> **Behaviour change: automatic problem scaling is on by default.** As of this release, {func}`slsqp_jax.minimize_like_scipy` enables gradient-based automatic scaling at the initial point (`auto_scale=True`, now resolving to the **uniform mode** with `target_gradient=1.0, max_factor=1e3`). Uniform mode applies one shared scalar `s_c` to *every* constraint row (preserving inter-row magnitude ratios) and an independent scalar `s_f` to the objective, both clipped symmetrically by `max_factor`. Well-scaled problems are unaffected (every factor stays at `1.0`); badly-scaled problems where `||J||` and `||grad_f||` differ by orders of magnitude are now automatically reconciled, eliminating the `penalty_starvation -> merit_penalty_explosion -> divergence_rollback` cascade documented in the diagnostics notes — and the row ratios you encoded in your constraints are preserved. Multipliers and the Lagrangian gradient norm are unscaled before being returned in `sol.stats`; `atol_internal = s_c * atol_user` is propagated to the inner solver so the user-perceived feasibility test is preserved. Pass `auto_scale=False` to opt out, or `auto_scale="balanced"` to recover the previous per-row default. See the [Auto-scaling](#auto-scaling-default-on) section below for the full contract.

## Installation

You can install the package from PyPI using any standard method:

**pip**

```bash
pip install slsqp-jax
```

**uv**

```bash
uv add slsqp-jax
```

**pixi**

```bash
pixi add --pypi slsqp-jax
```

## Usage

### Basic: objective and constraints

Define an objective function with signature `(x, args) -> (scalar, aux)` and optional constraint functions with signature `(x, args) -> array`. Equality constraints must satisfy `c_eq(x) = 0` and inequality constraints must satisfy `c_ineq(x) >= 0`.

Then create an `SLSQP` solver and pass it to `optimistix.minimise`:

```python
import jax.numpy as jnp
import optimistix as optx
from slsqp_jax import SLSQP, SLSQPConfig, ToleranceConfig

# Objective: minimize x^2 + y^2
def objective(x, args):
    return jnp.sum(x**2), None

# Equality constraint: x + y = 1
def eq_constraint(x, args):
    return jnp.array([x[0] + x[1] - 1.0])

# Inequality constraint: x[0] <= 0.2
# (recall the convention c_ineq(x) >= 0, so we encode it as 0.2 - x[0] >= 0)
def ineq_constraint(x, args):
    return jnp.array([0.2 - x[0]])

solver = SLSQP(
    eq_constraint_fn=eq_constraint,
    n_eq_constraints=1,
    ineq_constraint_fn=ineq_constraint,
    n_ineq_constraints=1,
    config=SLSQPConfig(tolerance=ToleranceConfig(atol=1e-8)),
)

x0 = jnp.array([0.5, 0.5])
sol = optx.minimise(objective, solver, x0, has_aux=True, max_steps=100)

print(sol.value)   # [0.2, 0.8]
print(sol.result)  # RESULTS.successful
```

The returned `optimistix.Solution` object contains:

- `sol.value` — the optimal point.
- `sol.result` — the coarse `optimistix.RESULTS` status code (`successful`, `nonlinear_max_steps_reached`, `nonlinear_divergence`, `nonfinite`).
- `sol.stats["slsqp_result"]` — the **granular** `slsqp_jax.RESULTS` classification (see [Termination Codes](#termination-codes)).
- `sol.aux` — any auxiliary data returned by the objective.
- `sol.stats` — solver statistics (number of steps, line-search/QP failure counters, the granular result code, etc.).
- `sol.state` — the final internal solver state.

### Configuration

`SLSQP` takes a single `config: SLSQPConfig` argument that aggregates all the algorithmic knobs into eight semantically-grouped sub-configs. Pass only the sub-configs you want to override; the rest fall back to their defaults.

| Sub-config | Imported as | Controls |
| --- | --- | --- |
| `tolerance` | `ToleranceConfig` | `rtol`, `atol`, `max_steps`, `min_steps`, `stagnation_tol`, `divergence_factor`, `divergence_patience` |
| `lbfgs` | `LBFGSConfig` | L-BFGS `memory`, `damping_threshold`, per-variable diagonal `diag_floor` / `diag_ceil` |
| `line_search` | `LineSearchConfig` | Backtracking `max_steps`, `armijo_c1`, `failure_patience` |
| `qp` | `QPConfig` | QP active-set `max_iter`, inner `max_cg_iter`, `failure_patience`, `zero_step_patience`, `ping_pong_threshold`, `mult_drop_floor`, `cg_regularization`, Newton-CG `use_exact_hvp` |
| `proximal` | `ProximalConfig` | sSQP `tau`, `mu_min`, `mu_max` |
| `preconditioner` | `PreconditionerConfig` | `enabled`, `type` (`"lbfgs"` / `"diagonal"`), `diagonal_n_probes` |
| `lpeca` | `LPECAConfig` | LPEC-A `method`, `sigma`, `beta`, `use_lp`, `trust_threshold`, `warmup_steps`, `predict_bounds` |
| `adaptive_cg` | `AdaptiveCGConfig` | Eisenstat-Walker `enabled`, HR-STCG `use_inexact_stationarity` |

Constraint information (`eq_constraint_fn`, `ineq_constraint_fn`, `n_eq_constraints`, `n_ineq_constraints`, `bounds`), optional derivative overrides (`obj_grad_fn`, `eq_jac_fn`, `ineq_jac_fn`, `obj_hvp_fn`, `eq_hvp_fn`, `ineq_hvp_fn`), the optional pluggable `inner_solver`, and the `verbose` printer remain top-level keyword arguments on `SLSQP` itself; only the algorithmic numerical knobs moved into `SLSQPConfig`.

```python
from slsqp_jax import (
    SLSQP,
    SLSQPConfig,
    ToleranceConfig,
    LBFGSConfig,
    QPConfig,
    LPECAConfig,
)

solver = SLSQP(
    eq_constraint_fn=eq_constraint,
    n_eq_constraints=1,
    config=SLSQPConfig(
        tolerance=ToleranceConfig(rtol=1e-8, max_steps=200),
        lbfgs=LBFGSConfig(memory=20),
        qp=QPConfig(max_iter=200, max_cg_iter=120),
        lpeca=LPECAConfig(method="lpeca_init", warmup_steps=2),
    ),
)
```

The full migration table from the legacy flat keyword arguments is in the project's `REFACTOR_NOTES.md`. Other examples below in this page still occasionally use the legacy flat-kwarg style for brevity; the same parameters live in the sub-configs listed above.

#### Termination Codes

`slsqp_jax.RESULTS` is a subclass of `optimistix.RESULTS` that adds finer-grained failure classifications. The granular code is exposed via `Solution.stats["slsqp_result"]`; `Solution.result` itself carries only the coarse code accepted by optimistix's iterative driver.

| `slsqp_jax.RESULTS` member | Meaning | Recommended remediation |
|---|---|---|
| `successful` | Either classical `stationarity & primal_feasibility` or the guarded QP-KKT disjunct fired. | None — converged. |
| `nonlinear_max_steps_reached` | Iteration budget exhausted with no genuine failure flag latched. The disjunction inside `terminate()` is `non_success_done = stagnation_detected \| ls_stagnation \| qp_stagnation \| diverging` — `max_iters_reached` is intentionally *excluded* so a pure budget-exhaustion run lands here instead of `nonlinear_divergence`. | Increase `max_steps`, or loosen `rtol`. |
| `merit_stagnation` | The L1 merit did not improve over `max_steps // 10` consecutive steps. The iterate is feasible but stationarity could not be driven below `rtol`. | Loosen `rtol`, switch to `use_exact_hvp_in_qp=True`, or check constraint LICQ. |
| `line_search_failure` | `2 * ls_failure_patience` consecutive line searches failed to produce a descent direction even after escalating L-BFGS resets. | Try `MinresQLPSolver`, exact HVPs, or a larger initial penalty parameter. |
| `iterate_blowup` | The L1 merit grew by more than `divergence_factor * max(\|best_merit\|, 1)` (or returned NaN/Inf) for `divergence_patience` steps. The returned iterate is the best-merit iterate; the solver's later iterates were diverging. | Check problem scaling; consider supplying exact HVPs. |
| `qp_subproblem_failure` | `2 * qp_failure_patience` consecutive QP subproblem failures. The inner QP solver could not produce a usable descent direction even after L-BFGS escalations. | Check the equality Jacobian for rank deficiency; try a different `inner_solver`. |
| `infeasible` | Override applied on top of any algorithmic failure when the final iterate violates `atol` feasibility. The constraints are likely infeasible or the active-set machinery could not satisfy them. | Inspect `c_eq` / `c_ineq` at `sol.value`; consider a feasible warm-start. |
| `nonfinite` | NaN/Inf in the iterate or its objective value. | Check the objective for numerical breakdown near `sol.value`; clip / regularise as needed. |

```python
from slsqp_jax import RESULTS, is_successful

sol = optx.minimise(objective, solver, x0, has_aux=True, throw=False)

if is_successful(sol.result):
    ...
elif sol.stats["slsqp_result"] == RESULTS.merit_stagnation:
    # L-BFGS multiplier noise blocks the strict rtol; loosen rtol
    # or supply exact HVPs.
    ...
```

> **Migration note.** Equality between `optimistix.RESULTS` and `slsqp_jax.RESULTS` members raises `ValueError` (each item carries an `_enumeration` reference and equinox enforces strict same-class equality). If you previously wrote `sol.result == optx.RESULTS.successful`, switch to `slsqp_jax.is_successful(sol.result)` or compare via `slsqp_jax.RESULTS.promote(sol.result)`.

Since `SLSQP` is a standard Optimistix minimiser, it composes with all Optimistix features: `throw=False` for non-raising error handling, custom `adjoint` methods for differentiating through the solve, and so on. See the [Optimistix documentation](https://docs.kidger.site/optimistix/) for details.

### Supplying gradients and Jacobians

By default the solver computes the objective gradient via `jax.grad` and constraint Jacobians via `jax.jacrev`. You can supply your own functions to avoid redundant computation or to handle functions that are not reverse-mode differentiable:

```python
solver = SLSQP(
    eq_constraint_fn=eq_constraint,
    n_eq_constraints=1,
    # User-supplied gradient: (x, args) -> grad_f(x)
    obj_grad_fn=my_grad_fn,
    # User-supplied Jacobian: (x, args) -> J(x)  shape (m, n)
    eq_jac_fn=my_eq_jac_fn,
)
sol = optx.minimise(objective, solver, x0, has_aux=True, max_steps=100)
```

### Supplying Hessian-vector products

For problems where you have access to exact second-order information, you can supply Hessian-vector product (HVP) functions. The solver uses these to produce high-quality secant pairs for the L-BFGS Hessian approximation, which typically improves convergence compared to using gradient differences alone:

```python
# Objective HVP: (x, v, args) -> H_f(x) @ v
def obj_hvp(x, v, args):
    return 2.0 * v  # Hessian of sum(x^2) is 2*I

# Per-constraint HVP: (x, v, args) -> array of shape (m, n)
# Row i is H_{c_i}(x) @ v
def eq_hvp(x, v, args):
    return jnp.zeros((1, x.shape[0]))  # Linear constraint has zero Hessian

solver = SLSQP(
    eq_constraint_fn=eq_constraint,
    n_eq_constraints=1,
    obj_hvp_fn=obj_hvp,
    eq_hvp_fn=eq_hvp,  # Optional — AD fallback is used if omitted
)
sol = optx.minimise(objective, solver, x0, has_aux=True, max_steps=100)
```

Note that you supply HVPs for the **objective and constraint functions separately**, not for the Lagrangian. The solver composes the Lagrangian HVP internally using the current KKT multipliers:

$$
\nabla^2 L(x) v = \nabla^2 f(x) v - \sum_i \lambda_i^{\text{eq}} \nabla^2 c_i^{\text{eq}}(x) v - \sum_j \mu_j^{\text{ineq}} \nabla^2 c_j^{\text{ineq}}(x) v
$$

If you provide `obj_hvp_fn` but omit the constraint HVP functions, the solver automatically computes the missing constraint HVPs via forward-over-reverse AD on the scalar function $\lambda^T c(x)$, which costs one reverse pass plus one forward pass regardless of the number of constraints.

**Frozen Hessian in the QP subproblem**: By default the QP inner loop uses a frozen L-BFGS approximation to the Lagrangian Hessian, even when exact HVPs are available. The exact HVP is called only **once per main iteration** (to probe along the step direction and produce an exact secant pair for the L-BFGS update). This design ensures (1) the QP subproblem sees a truly constant quadratic model, and (2) expensive HVP evaluations are not repeated inside the projected CG solver.

**Newton-CG mode** (`use_exact_hvp_in_qp=True`): For ill-conditioned problems where L-BFGS cannot capture the curvature accurately, this option replaces the frozen L-BFGS HVP with the **exact Lagrangian HVP** inside the CG inner loop. Each CG step then costs one forward-over-reverse AD pass — the same as one gradient evaluation. L-BFGS is still updated and serves as the **preconditioner** (via two-loop recursion), so the number of CG iterations is typically small. When the user supplies `obj_hvp_fn`, that function is used; otherwise the solver auto-computes the objective HVP via `jax.jvp(jax.grad(f), ...)`. This is the standard Newton-CG approach used in production solvers such as KNITRO and Ipopt.

### Box constraints (bounds)

You can specify simple lower and upper bounds on decision variables using the `bounds` parameter:

```python
import jax.numpy as jnp

# bounds[i, 0] = lower bound for x_i (use -jnp.inf for unbounded)
# bounds[i, 1] = upper bound for x_i (use jnp.inf for unbounded)
bounds = jnp.array([
    [0.0, 1.0],      # 0 <= x_0 <= 1
    [-jnp.inf, 5.0], # x_1 <= 5 (no lower bound)
    [0.0, jnp.inf],  # x_2 >= 0 (no upper bound)
])

solver = SLSQP(
    bounds=bounds,
    rtol=1e-8,
)
```

Bounds play a **dual role** in the solver, following the projected-SQP methodology (Heinkenschloss & Ridzal, *Projected Sequential Quadratic Programming Methods*, SIAM J. Optim., 1996):

1. **QP inequality constraints** — inside the QP subproblem, bounds are linearised as ordinary inequality constraints so the search direction is aware of the feasible box. Their Jacobian is known to be $\pm 1$ and their Hessian is zero, so they add minimal computational overhead compared to general inequality constraints.
2. **Hard projection** — after every line search step (and at initialisation), the iterate is *projected* (clipped) onto the feasible box. This guarantees that the objective and constraint functions are **never evaluated outside the bounds**, which is critical when those functions are undefined or ill-conditioned outside the box (e.g. a log-likelihood with positivity constraints on its parameters).

## Algorithm

(auto-scaling-default-on)=

### Auto-scaling (default-on)

{func}`slsqp_jax.minimize_like_scipy` evaluates the gradient of the objective and the Jacobian of every constraint at the initial point `x0` and, by default, multiplies each component by a scalar so that the scaled gradients are in the same range as the scaled objective gradient (gradient-based scaling, IPOPT/KNITRO-style). The scaled problem is what the inner solver actually sees; the outputs are unscaled before being returned to the user. This is implemented in {mod}`slsqp_jax.scaling` and reachable through three entry points:

- {func}`slsqp_jax.minimize_like_scipy` with `auto_scale=True` — the recommended path. `auto_scale=True` is the default.
- {func}`slsqp_jax.auto_scaled_minimise` — a thin wrapper around {func}`optimistix.minimise` for users who construct an {class}`slsqp_jax.SLSQP` instance themselves.
- {func}`slsqp_jax.auto_scaled_problem` and {func}`slsqp_jax.unscale_solution` — the raw helpers for users who want to drive the {func}`optimistix.minimise` call directly.

The motivating failure mode is documented in the diagnostic notes for the feasible-start divergence run: a `||J_eq|| ~ 70` vs `||grad_f|| ~ 0.018` magnitude mismatch (~4000x) drove a `penalty_starvation -> merit_penalty_explosion -> divergence_rollback` cascade that no amount of solver tuning could fix. Manually rescaling the constraint solved it; the auto-scaler does the same thing automatically.

**Migration note (default-mode flip).** The default value of `auto_scale=True` now resolves to the new `"uniform"` mode rather than the legacy `"balanced"` per-row mode. Uniform mode applies a *single* shared scalar `s_c` across all constraint rows (preserving inter-row magnitude ratios) instead of flattening every row to the same target gradient. There is no signature change — existing code keeps working — but the search trajectory may differ on problems where constraint rows have intentionally heterogeneous magnitudes. To restore the previous behavior verbatim, pass `auto_scale="balanced"`. See the "When uniform vs balanced" note below.

**Modes (string or bool aliases for `auto_scale`).** Each mode is a `(target_gradient, max_factor, uniform)` triple:

| `auto_scale` | `target_gradient` | `max_factor` | `uniform` | Behaviour |
| --- | --- | --- | --- | --- |
| `True` (default) / `"uniform"` | `1.0` | `10^3` | yes | One shared `s_c` across all constraint rows + one independent `s_f` for the objective, both **symmetrically** clipped to `[1/max_factor, max_factor]`. `atol_internal = s_c * atol_user` (can exceed `atol_user` when `s_c > 1`). Preserves row ratios. |
| `"balanced"` | `1.0` | `10^3` | no | Legacy per-row default. Each constraint row gets its own factor driving `||grad c_i||_inf` to `1.0`. Shrinks (`||grad c_i|| > 1`) and amplifies (`||grad c_i|| < 1`) up to 1000x. Flattens inter-row magnitudes. |
| `"knitro"` | `1.0` | `1.0` | no | Strict shrink-only per-row. Opt-in for users who want zero amplification under any circumstances. |
| `"ipopt"` | `100.0` | `1.0` | no | IPOPT default. Very conservative per-row; may not fix all cascades because it never amplifies and only shrinks gradients above `100`. |
| `"aggressive"` | `1.0` | `10^6` | no | Per-row, pushes amplification to the noise-floor limit. Opt-in for problems where `||grad_f(x0)||` is genuinely small (and the user has audited their AD for cancellation). |
| `False` | — | — | — | No wrapping; the previous (pre-feature) behaviour. Use this if you have output-exactness assertions tied to the unscaled path. |

Explicit overrides via `auto_scale_target_gradient=` / `auto_scale_max_factor=` win over the mode preset. Under uniform mode the `target_gradient` value is consumed by both the `s_f` derivation (matched against `||grad_f||_inf`) and the `s_c` derivation (matched against the cross-row max `max_i ||grad c_i||_inf`); a future `auto_scale_target_constraint_gradient` knob would let the two be set independently.

**When uniform vs balanced.** Uniform mode preserves the *relative* magnitudes of constraint rows: a "10x bigger gradient" remains a "10x bigger gradient" post-scaling. This is the right default when row magnitudes encode meaning — e.g. a top-level budget in USD plus smaller sub-budgets where you want feasibility tolerated proportionally to budget size. Balanced (per-row) mode flattens every row to a common gradient magnitude, which is the right call when one constraint row has *vastly* different gradient magnitude from the rest and that's not a meaningful spread (the original `||J_eq|| ~ 70` vs `||grad_f|| ~ 0.018` cascade was exactly this case).

**Mathematics (uniform mode).** With `target = target_gradient`:

```
s_f          = clip(target / max(||grad_f||_inf, grad_floor), 1/max_factor, max_factor)
max_row_norm = max_i ||grad c_i(x0)||_inf            (over the union of eq + ineq)
s_c          = clip(target / max(max_row_norm, eps), 1/max_factor, max_factor)
s_eq         = jnp.full((m_eq,),   s_c)
s_ineq       = jnp.full((m_ineq,), s_c)
atol_internal = s_c * atol_user
```

`max_factor` must satisfy `>= 1.0`; values `< 1.0` raise `ValueError` (the symmetric interval would be empty). `max_factor == 1.0` is legal but emits a {class}`UserWarning` because the interval collapses to `[1, 1]` and no scaling is applied. When every constraint row's gradient inf-norm is below `grad_floor` (default `1e-12`), one {class}`UserWarning` is emitted and `s_c = 1.0`; uniform mode does *not* skip individual rows because the cross-row max dominates and per-row magnitudes are intentionally preserved.

**Mathematics (per-row modes: balanced / knitro / ipopt / aggressive).** For each component (objective + each constraint row) with gradient `g` evaluated at `x0`:

```
norm = max(||g||_inf, grad_floor)
s    = clip(target_gradient / norm, eps, max_factor)
```

Rows whose `||g||_inf < grad_floor` are *skipped*: `s = 1.0` and a per-row {class}`UserWarning` is emitted. The wrappers are `f_scaled = s_f * f`, `c_scaled = s * c` (element-wise), `J_scaled = s[:, None] * J`, and per-row scaling on the per-component constraint HVPs.

**`atol` compensation (per-row modes).** The user's feasibility tolerance is preserved by tightening:

```
s_min = min(min(s_eq), min(s_ineq), 1.0)
atol_internal = atol_user * s_min
```

so that `|c_scaled[i]| <= atol_internal` implies `|c_user[i]| <= atol_user` for the worst-scaled row. Under uniform mode the contract is the simpler exact equivalence `atol_internal = s_c * atol_user`. In both cases the compensated tolerance is what the solver actually uses; the user's `atol` is what the user sees. `rtol` is invariant to a uniform `s_f` rescaling and is left untouched.

**Output unscaling.** When scaling is applied, `sol.stats` carries:

- `scale_factors` — the {class}`slsqp_jax.ScaleFactors` instance with `s_f`, `s_eq`, `s_ineq`, the user/internal `atol` pair, and skipped-row counts.
- `multipliers_eq_user`, `multipliers_ineq_user` — multipliers in user units (`lambda_user = (s / s_f) * lambda_scaled`).
- `final_grad_norm_user` — the Lagrangian gradient norm in user units (`||grad_L||_user = ||grad_L||_scaled / s_f`).
- `merit_penalty_note = "scaled units"` — the merit penalty `rho` lives in scaled space; this label flags the unit.

**Verbose-printer behaviour.** When `verbose=True` is combined with `auto_scale=True`, the built-in printer is wrapped so that each per-step line reports user-unit values for the quantities that have a clean unscaled equivalent (`f`, `|c|`, `|grad_f|`, `|grad_L|`, `|grad_L|/|L|`, `|d|`), and keeps a `(s)` suffix on the column label for quantities that genuinely live in scaled space (`merit`, `Δmerit`, `rho`, `gamma`, `s·y`, `rel_curv`, `kappa_B`). A one-line preamble printed once at iteration 0 reports the active factors. Under uniform mode the preamble collapses the per-row min/max display to a single shared `s_c` value and additionally surfaces `atol_internal` next to the user-supplied `atol_user` (so the reader can immediately see whether the constraint feasibility tolerance has been tightened or loosened by the scaling).

User-supplied verbose callables (`verbose=callable`) are invoked with an additional `scale_factors` keyword argument and the *scaled* state — i.e. the wrapper does not silently rewrite the inputs. The exported {func}`slsqp_jax.wrap_verbose_for_scaling` helper is available for users who want the same unscaling pipeline applied to their own callback.

**Diagnostic-report integration.** The {class}`slsqp_jax.DebugReport` rendered by {func}`slsqp_jax.diagnose_minimize_like_scipy` (and the {func}`slsqp_jax.diagnostic_run` context manager) automatically picks up the active scale factors from the solver's verbose attribute and renders an `Auto-scaling` section. Under per-row modes the section lists `s_f`, `s_eq` / `s_ineq` min/max/median summaries, the `atol_user` / `atol_internal` pair, and the skipped-row counts; under uniform mode the per-row block collapses to a single `s_c (shared)` line tagged with the active `m_eq` / `m_ineq` counts, and the `atol_internal` line is annotated with `(uniform: atol_internal = s_c * atol_user; may exceed atol_user when s_c > 1)`. The `Final iterate metrics` block uses user-unit values where they are well-defined and flags the rest with `(scaled)`. Two of the existing signal evaluators (`merit_penalty_explosion`, `penalty_starvation`) carry updated bidirectional suggestion lists that name `auto_scale="uniform"` (preserves row magnitudes, the default) and `auto_scale="balanced"` (flattens row magnitudes, useful when a single row's gradient is *vastly* out of band) so the user can pick the mode that matches their problem.

**Asymmetric defaults across runners.** {func}`slsqp_jax.minimize_like_scipy` and the {func}`slsqp_jax.diagnostic_run` / {func}`slsqp_jax.diagnose_minimize_like_scipy` paths default to `auto_scale=True` (the user-facing recommended setting). The lower-level {func}`slsqp_jax.debug_run` does not run through the scaling layer at all — it is a debug-the-raw-solver tool by design — but users can route through {func}`slsqp_jax.auto_scaled_minimise` instead when they want both the diagnostics and the auto-scaling.

### Overview

Each SLSQP iteration performs four steps:

1. **QP subproblem**: Construct a quadratic approximation of the objective using the frozen L-BFGS Hessian and linearise the constraints around the current point. Solve the resulting QP to obtain a search direction.
2. **Line search**: Use a Han-Powell L1 merit function $\phi(x;\rho) = f(x) + \rho (\lVert c_{\text{eq}}\rVert_1 + \lVert\max(0, -c_{\text{ineq}})\rVert_1)$ with backtracking Armijo conditions to determine the step size.
3. **Accept step**: Update the iterate $x_{k+1} = x_k + \alpha d_k$.
4. **Hessian update**: Append the new curvature pair $(s, y)$ to the L-BFGS history, where $y$ is either an exact HVP probe $\nabla^2 L(x_k) s$ (if HVP functions are provided) or the gradient difference $\nabla_x L(x_{k+1}, \lambda_{k+1}) - \nabla_x L(x_k, \lambda_{k+1})$. The secant condition (Nocedal & Wright §18.3) requires both Lagrangian gradients to use the **same** multipliers $\lambda_{k+1}$; this code uses the *Hessian-free LS multiplier estimate* recovered at $x_{k+1}$ for both endpoints (see "Hessian-free LS multiplier recovery" below). The initial Hessian uses **component-wise secant diagonal scaling**: $B_0 = \text{diag}(d)$ where $d_i = |y_i \cdot s_i| / (s_i^2 + \epsilon)$, clipped to $[10^{-4}, 10^{6}]$. This gives $d_i \approx |H_{ii}|$ for diagonal Hessians regardless of the step direction. The classical Shanno-Phua formula $d_i = y_i^2 / (y^T s)$ uses a single scalar normalizer, producing $d \propto h_i^2$ for multi-component steps, which severely underestimates curvature and causes 10-100x slowdowns.

### Scaling considerations: why L-BFGS over BFGS

Classical SLSQP (e.g. SciPy's implementation) maintains a **dense** $n \times n$ BFGS approximation to the Hessian of the Lagrangian. This requires $O(n^2)$ memory and $O(n^2)$ work per iteration for the matrix update alone. For the target problem sizes:

| n | Dense Hessian memory | L-BFGS memory (k=10) |
|---|---|---|
| 1,000 | 8 MB | 160 KB |
| 10,000 | 800 MB | 1.6 MB |
| 50,000 | 20 GB | 8 MB |

L-BFGS stores only the last $k$ step/gradient-difference pairs $(s_i, y_i)$ and computes Hessian-vector products in $O(kn)$ time using the compact representation (Byrd, Nocedal & Schnabel, 1994):

$$
B_k = \gamma I - W N^{-1} W^T
$$

where $W = (\gamma S, Y)$ is the horizontal concatenation of matrices $\gamma S$ and $Y$, and $N$ is a small $2k \times 2k$ matrix built from inner products of the stored vectors. The $2k \times 2k$ system is solved directly — negligible cost for $k << n$.

**VARCHEN-style B0 damping** (Lotfi et al., 2020) is applied to each curvature pair before storage.  Instead of damping toward the full L-BFGS approximation $Bs$ (which costs $O(k^2 n)$ and can be unreliable when $B$ is ill-conditioned), the implementation damps toward $B_0 s = \text{diag}(d) \cdot s$:

$$y_{\text{damped}} = \theta \, y + (1 - \theta) \, B_0 s, \quad \text{where } \theta \text{ ensures } s^T y_{\text{damped}} \geq \eta \, s^T B_0 s$$

This is $O(n)$, always well-conditioned (since $B_0 = \text{diag}(d)$ is clipped to $[10^{-4}, 10^6]$), and avoids the circular dependency where a badly-conditioned $B$ poisons its own damping.  The damping threshold $\eta$ is configurable via `damping_threshold` (default 0.2, Powell's standard value). Setting `damping_threshold=0.0` disables damping entirely.

**Eigenvalue monitoring and soft reset.**  The solver tracks the condition number of $B_0$ via $\kappa = \max(d)/\min(d)$.  When $\kappa$ exceeds $10^6$, a **soft reset** drops all but the most recent $(s, y)$ pair (VARCHEN Algorithm 1, Step 7).  This is less aggressive than the full diagonal or identity reset, preserving the most relevant curvature information while shedding stale pairs that contribute to ill-conditioning.

### Scaling considerations: why projected CG over dense KKT

Classical SLSQP solves the QP subproblem by forming and factorising the $(n + m) \times (n + m)$ dense KKT system at $O(n^3)$ cost. This implementation instead uses **projected conjugate gradient** (CG) inside an active-set loop:

1. **Projection**: For active constraints with matrix $A$ ($m_{\text{active}} \times n$), define the null-space projector $P(v) = v - A^T (A A^T)^{-1} A v$. The $A A^T$ system is only $m_{\text{active}} \times m_{\text{active}}$ (tiny, since $m << n$) and is solved directly.
2. **CG in null space**: Run conjugate gradient on the projected system, where each iteration requires one HVP ($O(kn)$ for L-BFGS) and one projection ($O(mn)$).
3. **Active-set outer loop**: Add the most violated inequality or drop the most negative multiplier until KKT conditions are satisfied.

Total cost per QP solve: $O(n \cdot k \cdot t)$, where $t$ is the number of CG iterations (typically $t << n$). Compared to $O(n^3)$ for the dense approach, this is orders of magnitude faster for $n > 1000$.

**Multiplier iterative refinement.** The $A A^T$ factorization is regularised with $\varepsilon I$ ($\varepsilon = 10^{-8}$) for numerical stability, which introduces $O(\varepsilon \cdot \kappa(A A^T))$ error in the recovered multipliers. When many constraints are active and $\kappa$ is moderate (e.g. $10^3$), this error can reach $\sim 10^{-5}$ — large enough to prevent the outer convergence criterion from being satisfied even though the QP subproblem is solved to full accuracy. After the initial multiplier recovery $\hat\lambda = (A A^T + \varepsilon I)^{-1} A(Bd + g)$, one step of classical iterative refinement is applied: compute the primal KKT residual $r = Bd + g - A^T \hat\lambda$, then correct $\hat\lambda \leftarrow \hat\lambda + (A A^T + \varepsilon I)^{-1} A r$. This squares the relative error (e.g. from $10^{-5}$ to $10^{-10}$) at negligible cost since the Cholesky factor is already available.

### Pluggable inner QP solvers

The equality-constrained subproblem solved at each active-set iteration — minimise a quadratic subject to $A d = b$ for the currently active constraints — is delegated to a pluggable **inner solver**. All inner solvers implement the `AbstractInnerSolver` interface (an `equinox.Module`) and return an `InnerSolveResult` containing the search direction $d$, the Lagrange multipliers for the active constraints, and a convergence flag.

Four strategies are provided:

**`ProjectedCGCholesky`** (default). Null-space projected CG with Cholesky-based projection — the approach described in the previous section. Computes the Cholesky factorization of the regularised $A A^T + \varepsilon I$ once, then reuses it for all CG iterations and for multiplier recovery via iterative refinement. Cost: $O(m^3)$ factorization amortised over $t$ CG iterations, each costing $O(kn + m^2)$. Best when the number of active constraints $m$ is small relative to $n$.

**`ProjectedCGCraig`**. Null-space projected CG with CRAIG's method (Golub-Kahan bidiagonalization) replacing the Cholesky factorization. Each projection computes the minimum-norm solution $x = A^T (A A^T)^{-1} b$ iteratively without forming $A A^T$, eliminating both the $O(m^3)$ factorization and the $\varepsilon I$ diagonal regularisation. This typically yields better equality constraint satisfaction since there is no regularisation bias. Multiplier recovery uses CG on the normal equations $A A^T \lambda = A(Bd + g)$. Cost: $O(mn \cdot t_{\text{craig}})$ per projection, where $t_{\text{craig}}$ is the number of CRAIG iterations (controlled by `craig_max_iter` and `craig_tol`). Both CRAIG itself and the multiplier-recovery CG use a **hybrid absolute+relative convergence test** $\lVert A x - b \rVert < \max(10^{-12},\, \texttt{tol} \cdot \lVert b \rVert)$ with a hardcoded absolute floor $10^{-12}$ baked in (not user-tunable). The absolute floor protects against the pathology where $\lVert b \rVert$ is itself near machine epsilon — e.g. projecting $A v$ at a near-KKT iterate where the gradient itself is $O(\texttt{rtol})$ — in which case a pure relative target $\texttt{tol} \cdot \lVert b \rVert$ would drop below `eps` and convergence could never fire. Without the floor, near-KKT projections returned `converged=False`, the outer SQP treated each as an inner failure, and the L-BFGS reset chain escalated despite the iterate being healthy.

- Reference: Craig, *The Minimum Norm Solution of Certain Underdetermined Systems* (1981); Golub & Kahan, *Calculating the Singular Values and Pseudo-Inverse of a Matrix*, SIAM J. Numer. Anal. 2(2), 1965.

**`MinresQLPSolver`**. Solves the full $(n + m) \times (n + m)$ saddle-point KKT system directly using MINRES-QLP:

$$
\begin{bmatrix} B & A^T \\ A & 0 \end{bmatrix}
\begin{bmatrix} d \\ \lambda \end{bmatrix}
=
\begin{bmatrix} -g \\ b \end{bmatrix}
$$

This eliminates the need for null-space projection, particular solution computation, and separate multiplier recovery — the direction $d$ and multipliers $\lambda$ are obtained simultaneously. The implementation follows the full Preconditioned MINRES-QLP (PMINRES-QLP) algorithm from Table 3.5 of Choi (2006), including both left (QR) and right (QLP) orthogonalizations. The QLP extension yields minimum-length solutions for singular/near-singular systems and improved numerical stability over plain MINRES. All iterations use QLP mode (equivalent to `TranCond=1` in the reference implementation). Givens rotations use the numerically stable SymOrtho procedure (Table 2.9).

The preconditioner is a **block-diagonal SPD** matrix $M^{-1} = \text{diag}(B_{\text{diag}}^{-1}, S^{-1})$, where $B_{\text{diag}}^{-1}$ is the L-BFGS inverse Hessian diagonal and $S = A B_{\text{diag}}^{-1} A^T$ is the Schur complement. This satisfies the SPD requirement from Section 3.4 of Choi (2006), which is essential for the correctness of the preconditioned Lanczos process. When box-constraint handling fixes variables via `free_mask`, the primal preconditioner block is masked to the free subspace ($v \mapsto \mathbf{f} \odot B^{-1}(\mathbf{f} \odot v)$ where $\mathbf{f}$ is the free-variable indicator) to prevent L-BFGS cross-coupling from leaking into the zero rows/columns of the reduced KKT system.

**Posterior feasibility projection (M-metric range-space correction).** PMINRES-QLP minimises the *Euclidean* residual of the full KKT system, with no separate guarantee on the constraint subvector $\lVert A d - b \rVert$. When the Lanczos recurrence is truncated by `max_iter` or stalled by free/fixed coupling under the SPD preconditioner, the returned $d$ leaves the linearised feasible region, the SLSQP step takes a constraint-violating direction, and the active-set loop churns trying to re-fix everything (the "MINRES diverges" failure mode on bound-heavy problems such as `Portfolio(n = 5000)`). After the PMINRES-QLP iteration completes, a single **M-metric range-space projection** is applied:

$$\delta d = M A^T (A M A^T)^{-1} (A d - b), \qquad d \leftarrow d - \delta d$$

This is the closest correction to $d$ in the metric induced by the SPD preconditioner $M$ and the unique minimiser of $\lVert \delta d \rVert_{M^{-1}}$ subject to $A(d - \delta d) = b$. The factorisation of $A M A^T$ is **already computed for the Schur block of the preconditioner**, so the projection costs only one extra back-solve plus a matvec — essentially one additional Lanczos step. After a single shot, $\lVert A d - b \rVert$ is reduced to roughly $\varepsilon \cdot \kappa(A M A^T) \cdot \lVert A d_{\text{raw}} - b \rVert$ (the floor set by the $10^{-8} I$ regularisation on the Schur Cholesky), typically $10^4$–$10^7$ tighter than the un-projected MINRES iterate. To eliminate even that residual floor on ill-conditioned problems the projection is wrapped in a fixed-point **iterative refinement** loop, applying $d \leftarrow d - M A^T (A M A^T)^{-1} (A d - b)$ up to `proj_refine_max_iter` times (default `3`) with early termination once $\lVert A d - b \rVert$ drops below `proj_refine_atol + proj_refine_rtol * (1 + ||b||)`. Each refinement round costs only one additional matvec plus one Schur back-solve (no refactorisation), and on the test fixtures in `tests/test_inner_solver.py::TestMinresQLPProjectionRefinement` it drives the residual from $\sim 10^{-9}$ (single shot) to machine precision. The actual rounds used and final residual are surfaced through `SLSQPDiagnostics.n_proj_refinements` and `SLSQPDiagnostics.max_proj_residual`. When no SPD preconditioner is supplied, a small dedicated $A A^T + \varepsilon I$ Cholesky is built just for the 2-norm projection. Multipliers are left as PMINRES-QLP returned them; the residual stationarity inconsistency $O(\lVert B \rVert \cdot \lVert \delta d \rVert)$ is absorbed by the outer L1 merit and L-BFGS secant blending. References: Orban & Arioli, *Projected Krylov Methods for Saddle-Point Systems*, SIAM J. Matrix Anal. Appl. 36(3), 2014, Lemma 3.1; Benzi, Golub & Liesen, *Numerical Solution of Saddle Point Problems*, Acta Numerica, 2005, §5.

Cost: $O((n + m) \cdot t_{\text{minres}})$ per solve, where $t_{\text{minres}}$ is the number of Lanczos iterations (controlled by `max_iter` and `tol`); the posterior projection adds one $m \times m$ triangular solve.

- Reference: Choi, *Iterative Methods for Singular Linear Equations and Least-Squares Problems*, PhD thesis, Stanford University, 2006. Choi, Paige & Saunders, *MINRES-QLP: A Krylov Subspace Method for Indefinite or Singular Symmetric Systems*, SIAM J. Sci. Comput. 33(4), 2011. Orban & Arioli, *Projected Krylov Methods for Saddle-Point Systems*, SIAM J. Matrix Anal. Appl. 36(3), 2014. Heinkenschloss & Ridzal, *A Matrix-Free Trust-Region SQP Method for Equality Constrained Optimization*, SIAM J. Optim. 24(3), 2014, Algorithm 4.18.

**`HRInexactSTCG`** (composite). Heinkenschloss-Ridzal (2014) Algorithm 4.5 — a Steihaug-Toint conjugate gradient iteration designed to be robust to *inexact* null-space projectors. The class is a thin wrapper that **composes** another inner solver to obtain its projector, particular solution, and multiplier-recovery closure (packaged into an internal `ProjectionContext`), so the same Cholesky / CRAIG projection machinery is reused without code duplication. Two technical moves make the iteration noise-tolerant:

1. **Modified residual recurrence** ($r_{i+1} = r_i - \alpha_i H p_i$, *not* re-projected at each step — HR Remark 4.6.i). Re-projecting the residual at every CG iteration would re-inject projector noise on every step; the modified recurrence absorbs the projector exactly once, at initialisation, so under inexact $\widetilde{W}_k$ the iterates are the unique CG sequence for *some* fixed linear operator (HR Lemma 4.10).
2. **Full H-conjugacy reorthogonalisation** of every search direction $p_i$ against all stored $p_j$. The static-shape buffers are $(\texttt{max\_cg\_iter}, n)$; reorthogonalisation cost is $O(\texttt{max\_cg\_iter}^2 \cdot n)$ across the whole inner solve. This rebuilds the H-conjugacy that floating-point arithmetic erodes when the projector is not exact.

The convergence test uses the *projected* residual with a **hybrid absolute+relative** threshold: $\lVert \widetilde{W}_k r_i \rVert \le \max(10^{-14},\, \texttt{cg\_tol} \cdot \lVert \widetilde{W}_k r_0 \rVert)$. The relative target carries the noise-cancellation that HR Algorithm 4.5 was designed for (both numerator and denominator share the same projector noise, so the ratio reaches the inner solver's precision floor cleanly), and the hardcoded absolute floor $10^{-14}$ — set tighter than `_CRAIG_TOL_ABS` because the projected residual is one noise level cleaner than the raw $A x - b$ — protects the edge case where $\lVert \widetilde{W}_k r_0 \rVert$ is itself at or below machine epsilon (e.g. on a feasible QP whose effective gradient already lies in $\text{range}(A^T)$). Without the floor the relative target collapses below `eps` and the iteration would return a spurious `converged=False` flag despite the QP being at its KKT point. The initial value $\lVert \widetilde{W}_k r_0 \rVert$ is the *projected gradient norm* and is surfaced as `InnerSolveResult.projected_grad_norm`. This is the noise-aware stationarity proxy: $\widetilde{W}_k(g_k + B d_p)$ is independent of multiplier-recovery accuracy (the projector annihilates the multiplier-direction component of the gradient by construction), so it stays clean even when classical $\lVert \nabla_x L \rVert$ stagnates at the multiplier-recovery floor. The QP layer threads `projected_grad_norm` through `QPResult` and `_solve_qp_subproblem`'s bound-fixing loop with **latest-not-accumulated** semantics — only the most recent inner solve's value is forwarded — so the outer loop sees the projected gradient at the actually-used active set. The curvature guard `pHp ≤ max(cg_regularization·||p||², cg_regularization·||r₀||²)` combines a relative threshold (SNOPT-style scale-invariance) with an absolute floor anchored to the initial projected gradient: without the absolute floor, when both `||p||` and `pHp` decay together near convergence the relative test never fires, and a numerically-noise-dominated `pHp` in the denominator produces an `O(1/eps²)` step that contaminates the iterate. The composed inner solver must implement `build_projection_context` — `ProjectedCGCholesky` and `ProjectedCGCraig` do; `MinresQLPSolver` does not (the saddle-point KKT system has no separate null-space projector to reuse), and composing it raises `NotImplementedError`. The `precond_fn` argument is accepted for interface compatibility but silently ignored: HR Algorithm 4.5 has no inner preconditioner. Cost per solve: identical to the composed projector's per-iteration cost plus an additive $O(\texttt{max\_cg\_iter}^2 \cdot n)$ from full reorthogonalisation.

- Reference: Heinkenschloss & Ridzal, *A Matrix-Free Trust-Region SQP Method for Equality Constrained Optimization*, SIAM J. Optim. 24(3), 2014, Algorithm 4.5 and Lemma 4.10.

To use an alternative inner solver, construct the desired strategy and pass it to `SLSQP` (which forwards it to `solve_qp`):

```python
from slsqp_jax import SLSQP, ProjectedCGCraig, MinresQLPSolver

# CRAIG-based projection (matrix-free, no regularisation)
solver = SLSQP(
    eq_constraint_fn=eq_constraint,
    n_eq_constraints=1,
    inner_solver=ProjectedCGCraig(
        max_cg_iter=50,
        cg_tol=1e-8,
        craig_tol=1e-12,
        craig_max_iter=200,
    ),
)

# MINRES-QLP on the full KKT system
solver = SLSQP(
    eq_constraint_fn=eq_constraint,
    n_eq_constraints=1,
    inner_solver=MinresQLPSolver(max_iter=200, tol=1e-10),
)

# HR-STCG composing a CRAIG projector — noise-aware, matrix-free.
# Pair with ``use_inexact_stationarity=True`` so the outer loop can
# terminate on the projected-gradient floor when the classical
# stationarity test stalls on multiplier-recovery noise.
from slsqp_jax import HRInexactSTCG

solver = SLSQP(
    eq_constraint_fn=eq_constraint,
    n_eq_constraints=1,
    inner_solver=HRInexactSTCG(
        inner=ProjectedCGCraig(
            max_cg_iter=50, cg_tol=1e-8,
            craig_tol=1e-10, craig_max_iter=200,
        ),
        max_cg_iter=80,
        cg_tol=1e-8,
    ),
    use_inexact_stationarity=True,
)
```

### Reverse-mode AD and `while_loop`

By default, the solver computes:

- **Objective gradient** via `jax.grad` (reverse-mode).
- **Constraint Jacobians** via `jax.jacrev` (reverse-mode, $O(m)$ passes — faster than `jax.jacfwd`'s $O(n)$ passes when $m << n$).
- **HVP fallback** via `jax.jvp(jax.grad(f), ...)` (forward-over-reverse).

All of these require **reverse-mode differentiation** through the user's functions. JAX's reverse-mode AD does not support differentiating through `jax.lax.while_loop` or other variable-length control flow primitives. If your objective or constraint functions contain `while_loop`, `scan` with variable-length carries, or other non-reverse-differentiable operations, the AD fallback will fail.

**How to handle this**: supply your own derivative functions via the optional fields on `SLSQP`:

```python
solver = SLSQP(
    obj_grad_fn=my_custom_grad,        # bypass jax.grad
    eq_jac_fn=my_custom_eq_jac,        # bypass jax.jacrev
    ineq_jac_fn=my_custom_ineq_jac,    # bypass jax.jacrev
    obj_hvp_fn=my_custom_obj_hvp,      # bypass forward-over-reverse
    eq_hvp_fn=my_custom_eq_hvp,        # bypass forward-over-reverse
    ineq_hvp_fn=my_custom_ineq_hvp,    # bypass forward-over-reverse
    ...
)
```

When all derivative functions are supplied, the solver never calls `jax.grad`, `jax.jacrev`, or `jax.jvp` on your functions, so `while_loop` and other control flow work without issue. Alternatively, if only the HVP functions are problematic, you can supply `obj_grad_fn` and the Jacobian functions (which only require first-order derivatives) and let the solver fall back to L-BFGS mode (no HVPs needed) by omitting `obj_hvp_fn`.

### Convergence safeguards

The solver includes safeguards against premature termination, a known issue in SciPy's SLSQP where the optimizer can terminate after a single iteration when the initial point exactly satisfies equality constraints:

- **Minimum iterations** (`min_steps`, default 1): The solver will not declare convergence before completing at least `min_steps` iterations. This ensures that KKT multipliers have been computed by at least one QP solve before checking KKT optimality conditions.

- **Initial multiplier estimation**: When equality constraints are present, the initial Lagrange multipliers are estimated via least-squares rather than being set to zero. This prevents the Lagrangian gradient from collapsing to the objective gradient at the first convergence check.

```python
solver = SLSQP(
    eq_constraint_fn=eq_constraint,
    n_eq_constraints=1,
    min_steps=1,  # Default; set to 0 to allow convergence at step 0
)
```

### NLP-level bound multiplier recovery

When box constraints are present, the inequality multiplier vector
$\mu_{\text{ineq}} \in \mathbb{R}^{m_{\text{ineq}}}$ stored in
`state.multipliers_ineq` is laid out as
`[general; lower_bound; upper_bound]`. The bound block of this vector
appears in the Lagrangian gradient
$\nabla_x L = \nabla f - J_{\text{eq}}^T \lambda - J_{\text{ineq}}^T \mu_{\text{ineq}}$
and therefore directly affects the relative-stationarity convergence test
$\lVert \nabla_x L \rVert \le \texttt{rtol} \cdot \max(|L|, 1)$.

For each accepted SLSQP step the bound multipliers are **recomputed from
the partial Lagrangian gradient at $x_{k+1}$**, after the line search.
Concretely, for each variable $i$ at a bound at $x_{k+1}$:

$$
\mu_i^{\text{lower}} = \max\!\Bigl(0,\; \bigl[\nabla f(x_{k+1}) - J_{\text{eq}}^T \lambda - J_{\text{gen}}^T \mu_{\text{gen}}\bigr]_i\Bigr), \qquad
\mu_j^{\text{upper}} = \max\!\Bigl(0,\; -\bigl[\nabla f(x_{k+1}) - J_{\text{eq}}^T \lambda - J_{\text{gen}}^T \mu_{\text{gen}}\bigr]_j\Bigr).
$$

By construction this zeros $\nabla_x L$ exactly on every active-bound
component (only the residual on the equality / general-inequality
multipliers remains), and the clamp to $\geq 0$ enforces dual
feasibility. The active-bound test uses an absolute floor of $10^{-12}$
plus a relative safety term $\epsilon \cdot (1 + |y_{\text{new}}|)$;
variables clipped to a bound by `_clip_to_bounds` satisfy the test by
exact equality, the relative term only matters for variables the line
search drove to within floating-point precision of a bound without
explicit clipping.

**Why post-line-search.** The previous bound-multiplier recovery
happened *inside the QP* at $x_k$ via $B d + g - A^T \lambda$, which
inherits three sources of noise:

1. The L-BFGS HVP $B d \neq H d$ unless the L-BFGS approximation is
   exact (it isn't, for any $k > 0$ on a non-quadratic objective).
2. The recovery is at $x_k$, not at the iterate where the convergence
   test runs ($x_{k+1}$).
3. The active-bound mask is derived from the *QP direction* and not
   from $x_{k+1}$, so an $\alpha < 1$ line search can leave the recovered
   multiplier "for the wrong point".

On bound-heavy problems (e.g. `Portfolio(n=5000)` with `MinresQLPSolver`,
where the optimum has thousands of active bounds) this used to pin
$\lVert \nabla_x L \rVert / |L|$ above `rtol` even when the constraints
were satisfied and $\alpha = 1$. The post-line-search recovery removes
all three error sources for the bound block.

**Why the L-BFGS secant is unaffected.** The bound Jacobian
`state.bound_jac` is constant (identity-style rows for lower bounds and
their negatives for upper bounds), so the bound contribution to the
secant pair

$$
y_k = \nabla_x L(x_{k+1}, \lambda) - \nabla_x L(x_k, \lambda)
$$

vanishes identically: $[J_{\text{bound}} - J_{\text{bound}}]^T \mu_{\text{bound}} \equiv 0$
regardless of $\mu_{\text{bound}}$, as long as the same multiplier
vector is applied at both endpoints — which is the case in this
implementation. This matches the L-BFGS-B precedent (Byrd, Lu, Nocedal
& Zhu, *SIAM J. Sci. Comput.* 16(5), 1995; Zhu, Byrd, Lu & Nocedal,
*ACM TOMS* 23(4), 1997, Algorithm 778), which uses
$y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$ (i.e. $\mu_{\text{bound}} = 0$ at
both endpoints) for bound-constrained problems. The corrected bound
multipliers are also written back into `state.multipliers_ineq` so
LPEC-A's predictor, the verbose diagnostics, and the next QP's
warm-start all see the sharper values.

There is no user-facing parameter for this behaviour: the recovery is
always on whenever bounds are present, costs one extra
$J_{\text{eq}}^T \lambda$ / $J_{\text{gen}}^T \mu_{\text{gen}}$ matvec
that is amortised against the existing
`compute_lagrangian_gradient` call, and silently degrades to a no-op
when `bounds is None`.

### QP anti-cycling: the EXPAND procedure

The QP subproblem is solved by a primal active-set method that adds or removes inequality constraints one at a time. When the problem is **degenerate** — multiple constraints pass through the same vertex or have tied violation/multiplier values — the active-set loop can *cycle*: iteration $i$ activates a constraint, iteration $i+1$ drops it, iteration $i+2$ re-activates it, and so on. The QP then exhausts its iteration budget without converging, producing a poor search direction.

This implementation uses the **EXPAND** procedure (Gill, Murray, Saunders & Wright, *Mathematical Programming* 45, 1989) to break such cycles. The idea is simple: instead of a fixed feasibility tolerance, the active-set loop maintains a *working tolerance* that grows monotonically:

$$
\delta_k = \epsilon + k \tau, \qquad \tau = \frac{\epsilon \cdot c}{K}
$$

where $\epsilon$ is the `tol` parameter, $c$ is `expand_factor`, and $K$ is `max_iter`. With the default `expand_factor=1.0` the working tolerance grows from $\epsilon$ at $k=0$ to $2\epsilon$ at $k=K$ — enough to dislodge floating-point ties at degenerate vertices while staying tight enough that converged working sets remain consistent with the outer SLSQP stationarity tolerance.

At each active-set iteration $k$:

- A constraint is considered *violated* only if its residual is below $-\delta_k$ (progressively stricter threshold for activation).
- A multiplier is considered *negative* only if it is below $-\max(\delta_k, \mu_{\text{drop}})$, where $\mu_{\text{drop}}$ is the `mult_drop_floor` parameter.

Because $\delta_k$ increases at every step, marginally active or marginally infeasible constraints that cause cycling are gradually excluded, breaking the degeneracy.

EXPAND is the standard anti-cycling technique used in production solvers (MINOS, SNOPT, SQOPT) and is backed by a convergence guarantee: strict objective decrease within each expanding sequence. The `expand_factor` parameter on `solve_qp` controls the growth rate; set it to `0.0` to disable expansion entirely.

**Noise-aware multiplier drop.** Multipliers recovered from the projected-CG inner solver carry numerical noise (typically $O(\epsilon \cdot \kappa(AA^T))$, e.g. $10^{-5}$ for moderately conditioned problems). Combined with the EXPAND working tolerance ($O(10^{-8})$), a constraint that was just added with a true multiplier near zero can be misclassified as "negative" and dropped, only to be re-added on the next iteration. The `mult_drop_floor` parameter on `solve_qp` (default `1e-6`) sets a noise-aware lower bound on the drop test: a constraint is only dropped when its multiplier is below $-\max(\delta_k, \mu_{\text{drop}})$, where $\mu_{\text{drop}}$ is the `mult_drop_floor` parameter. The default conservatively matches typical CG multiplier-recovery error.

**Ping-pong detector (opt-in).** Even with EXPAND and the noise-aware drop test, pathological degeneracies can still cause add-then-drop oscillations on the same constraint. The active-set loop tracks the most recent add and drop indices: if the *same* constraint is added immediately after being dropped (or vice versa), an internal counter increments. When the counter reaches `qp_ping_pong_threshold` the QP loop short-circuits with `converged=True` and `ping_ponged=True`. The detector is **disabled by default** (`qp_ping_pong_threshold = 2**31 - 1`); production paths rely on the classical behaviour where an oscillating pair keeps iterating until `qp_max_iter` is reached, which keeps the outer SQP loop in sync with the multiplier stream produced by that working set. Setting `qp_ping_pong_threshold` to a small value (e.g. `3`–`8`) opts in to the short-circuit for problems with confirmed degenerate vertices where `qp_max_iter` was being exhausted regularly. The diagnostic counter `n_qp_ping_pong` records how often the counter reached the threshold across the SQP run; `qp_cyc` in the verbose output flags it on a per-step basis. Note that when `MinresQLPSolver` is used, the coupled multipliers returned by the solver on a degenerate working set can be ill-conditioned — the short-circuit is particularly unsafe in that configuration and should stay disabled unless each step's merit penalty is verified to remain bounded.

### LPEC-A active set identification

As an alternative to (or complement of) the EXPAND procedure, the solver supports **LPEC-A** (Oberlin & Wright, *Mathematical Programming*, 2005, Section 3.3), a method that predicts the active constraint set from the current NLP iterate before the QP active-set loop begins.

LPEC-A computes a proximity measure $\bar{\rho}$ from the primal constraint values, dual multiplier estimates, and stationarity residual, then predicts constraint $i$ as active when $c_{\text{ineq},i} \leq (\beta \cdot \bar{\rho})^{\bar{\sigma}}$. Under MFCQ and second-order sufficiency conditions, this prediction is asymptotically exact as the iterate converges to the solution — including weakly active constraints (where $c_i^* = 0$ but $\lambda_i^* = 0$), which are notoriously difficult for active-set methods.

The `active_set_method` parameter on `SLSQP` controls how LPEC-A interacts with the QP solver:

- `"expand"` (default): Standard EXPAND procedure. No LPEC-A prediction.
- `"lpeca_init"`: LPEC-A predicts the initial active set, then the EXPAND loop runs normally. This provides a better starting point (fewer active-set iterations) while retaining EXPAND's anti-cycling guarantee.
- `"lpeca"`: LPEC-A predicts the initial active set and the active-set loop runs with a fixed tolerance (no EXPAND growth). Relies on the LPEC-A prediction for anti-cycling; `max_iter` provides a hard stop.

The threshold parameters `lpeca_sigma` (default 0.9) and `lpeca_beta` (default $1/(m_{\text{ineq}} + n + m_{\text{eq}})$) control the sensitivity of the prediction.

**Trust gate.** Oberlin & Wright's Theorem 5 only guarantees correctness asymptotically — the prediction is reliable when the iterate is close enough to the solution that $\bar{\rho}$ is small. Far from the solution, $\bar{\rho}$ can be very large and the raw threshold $(\beta\bar{\rho})^{\bar{\sigma}}$ becomes wide enough to flag every inequality as active, producing a rank-deficient working set. The implementation gates this with `lpeca_trust_threshold` (default `1.0`): when $\bar{\rho}$ exceeds `lpeca_trust_threshold` the prediction is deemed untrustworthy and is replaced with an empty set, letting the QP fall back to its warm-start active set. The diagnostic `n_lpeca_bypassed` records the number of SQP steps where this gate (or the warm-up, below) fired. The earlier implementation used a `min(threshold, max|c_ineq|)` clamp instead, which is mathematically vacuous (every $c_i$ is bounded by $\max|c_j|$ by construction), so it left the over-prediction problem unaddressed.

**Rank-aware size cap.** Even when the trust gate passes, the predicted active set is truncated so that at most $n - m_{\text{eq}} - 1$ inequalities are predicted active. This ensures the working-set Jacobian $[A_{\text{eq}}; A_{\text{active}}]$ retains a rank margin (at most $n$ rows in $n$ columns of slack), preventing the LICQ-violating "everything active" prediction from poisoning the QP equality solve. When the cap fires, the most-violated constraints (smallest $c_{\text{ineq},i}$) are kept; the diagnostic `n_lpeca_capped` counts these events. The `lpeca_capped` field on `QPResult` flags it per step.

**SQP warm-up gate.** LPEC-A's asymptotic guarantees assume the multiplier estimates fed into $\bar{\rho}$ are themselves reasonable. During the first few SQP iterations the multipliers are still transient (especially when the initial guess is far from the solution), so the LPEC-A prediction is bypassed entirely for the first `lpeca_warmup_steps` outer iterations (default `3`). After warm-up the trust gate above takes over.

**Bound-constraint extension.** Box constraints are handled by a separate iterative bound-fixing loop after the QP solve (see the "Bound separation via variable fixing" discussion in the scaling section). By default that loop starts cold - with no variable fixed - and discovers the active bound set in one or two passes per SQP iteration. LPEC-A computes its prediction over *all* inequalities, including the lower- and upper-bound rows, so the bound portion of the prediction is an untapped source of warm-start information. With `lpeca_predict_bounds=True` (default) the bound rows of LPEC-A's predicted active set are scattered back onto the variable indices (using the precomputed lower/upper bound masks) and installed as the initial `free_mask` and `d_fixed` of the bound-fixing loop. When the prediction is accurate this typically halves the number of reduced-space inner solves required to converge the bound active set; when it is wrong the loop's existing add/drop logic refines it on the first pass, so there is no correctness risk. The pre-fixed count is accumulated in the diagnostic `n_lpeca_bounds_prefixed` and exposed per step on `QPResult.n_lpeca_bounds_prefixed`. The warm-up and trust gates apply to the bound extension as well: when LPEC-A is bypassed, the bound-fixing loop reverts to its cold start. Set `lpeca_predict_bounds=False` to restrict LPEC-A to general inequality constraints only.

**Optional LP refinement.** By default, LPEC-A uses the multiplier estimates from the previous QP solve. Setting `lpeca_use_lp=True` solves a small LP (via `mpax.r2HPDHG`) to obtain tighter multiplier estimates, producing a more accurate prediction. This requires the `mpax` package (`pip install slsqp-jax[extras]`).

```python
solver = SLSQP(
    active_set_method="lpeca_init",  # or "lpeca"
    lpeca_sigma=0.9,
    lpeca_trust_threshold=1.0,
    lpeca_warmup_steps=3,
    lpeca_predict_bounds=True,  # warm-start the bound-fixing loop
    lpeca_use_lp=False,  # True requires mpax
    ...
)
```

### Multiplier stability and convergence rate

Wright (*Properties of the Log-Barrier Function on Degenerate Nonlinear Programs*, SIAM J. Optim., 2002, Theorem 5.3) proved that the convergence rate of SQP methods depends critically on **multiplier stability**: the rate at which Lagrange multipliers change between successive iterations. Specifically, superlinear convergence requires $\lVert \lambda_k - \lambda_{k+1} \rVert = O(\delta)$ where $\delta$ is the step norm. When the QP subproblem is degenerate, the multiplier solution may not be unique, and arbitrary choices across iterations can destroy this stability — even though the primal iterates converge.

This implementation promotes multiplier stability through three mechanisms:

1. **Active-set warm-starting**: the final active set from each QP solve is passed as the initial guess for the next outer iteration's QP subproblem. Because the active set carries implicit information about which multipliers are nonzero, reusing it across iterations biases the QP toward consistent multiplier selections. This is the same strategy used by production SQP codes such as SNOPT (Gill, Murray & Saunders, 2005).

2. **Hessian-free LS multiplier recovery (post-line-search)**: the QP-recovered multipliers $\lambda_{\text{QP}} = (A_k A_k^T)^{-1} A_k (B d + g_k)$ inherit an $O(s_f / s_{\text{eq}} \cdot \kappa(B))$ error budget when the L-BFGS Hessian is poorly conditioned or when auto-scaling inflates the multiplier scale. Because both the L-BFGS secant pair and the convergence test consume the same multiplier vector at both endpoints, that error contaminates both subsystems symmetrically — producing the textbook "swing-up" pathology where $\lVert \nabla L \rVert / \lvert L \rvert$ jumps from $10^{-4}$ to $10^{+3}$ in one step while $f$ and $\lvert c \rvert$ barely move (the canonical bound-active-set / auto-scaling failure mode). The fix is a **least-squares multiplier estimate at the accepted iterate**:

   $$\lambda_{\text{LS}}(x_{k+1}) = \bigl(J(x_{k+1}) J(x_{k+1})^T\bigr)^{-1} J(x_{k+1}) \nabla f(x_{k+1})$$

   This is the multiplier vector that minimises the actual stationarity residual at the new iterate; it does not depend on $B$, on $d$, or on any QP machinery. It is consumed by the L-BFGS secant pair (so the same vector is used at both endpoints) and by the convergence test (so numerator and denominator share a common multiplier vector). The QP-recovered multipliers $\lambda_{\text{QP}}$ are still consumed by Han-Powell's penalty rule, by the LPEC-A predictor, and by the QP active-set warm-start — all subsystems that need the more conservative QP-side estimate. The general-inequality active set used by the LS recovery is *value-based at $x_{k+1}$* (`c_ineq_general(x_{k+1})[i] <= active_tol`) — the QP active set is intentionally not consumed because it is stale at $x_{k+1}$ after a partial line-search step. When bounds are present the LS Jacobian columns are masked to the free subspace so the LS fit does not absorb at-bound coordinates of $\nabla f$ into equality multipliers (those gradient components belong to the subsequent bound-multiplier recovery). Active-row inequality multipliers are clamped at $\max(0, \cdot)$ for dual feasibility.

3. **Active-set warm-starting + LS recovery for the secant**: the previous "alpha-scaled multiplier blending" mechanism ($\lambda_{k+1} = \lambda_k + \alpha (\lambda_{\text{QP}} - \lambda_k)$) is no longer used. The LS multipliers are multiplier-stable across iterations by construction (they depend only on `grad`/`J`/`c_ineq` at the new iterate, not on the previous iterate's QP estimate), so the Wright (2002) blend is unnecessary.

The two multiplier vectors are surfaced separately on `state`: `state.multipliers_eq_ls` / `state.multipliers_ineq_ls` are the LS estimate (also exposed via `Solution.stats["multipliers_eq"]` / `["multipliers_ineq"]`), and `state.multipliers_eq_qp` / `state.multipliers_ineq_qp` are the QP-side estimate (also exposed via `Solution.stats["multipliers_eq_qp"]` / `["multipliers_ineq_qp"]` for advanced diagnostics). On well-conditioned problems near a clean KKT point the two agree to floating-point precision; on ill-conditioned or noisy iterates the LS variant is the cleaner indicator of "is this iterate stationary?". The diagnostics layer's `StepSummary.qp_vs_ls_multiplier_ratio` field surfaces the per-step ratio so noise inflation can be inspected post-hoc.

For pathological cases where warm-starting is insufficient, the solver provides **proximal multiplier stabilization** (see next section).

### Proximal multiplier stabilization (sSQP)

When the QP subproblem is highly degenerate — for example, when SciPy's SLSQP reports "Inequality constraints incompatible" — the standard active-set method can cycle indefinitely, exhausting its iteration budget and producing a poor search direction. The solver then stagnates because the corrupted multipliers yield an inaccurate Lagrangian gradient.

The **stabilized SQP (sSQP)** formulation (Hager, *Computational Optimization and Applications*, 12(1–3), 1999; Wright, *Mathematics of Operations Research*, 27(3), 2002, Section 6) addresses this by absorbing equality constraints into the QP objective via an augmented-Lagrangian penalty. The standard QP subproblem

$$
\min_d \tfrac{1}{2} d^T B d + g^T d \quad \text{s.t. } A_{\text{eq}}\, d = b_{\text{eq}},\; A_{\text{ineq}}\, d \geq b_{\text{ineq}}
$$

is replaced by

$$
\min_d \tfrac{1}{2} d^T \widetilde{B}\, d + \widetilde{g}^T d \quad \text{s.t. } A_{\text{ineq}}\, d \geq b_{\text{ineq}}
$$

where

$$
\widetilde{B}(v) = B v + \frac{1}{\mu}\, A_{\text{eq}}^T (A_{\text{eq}}\, v), \qquad
\widetilde{g} = g - \frac{1}{\mu}\, A_{\text{eq}}^T b_{\text{eq}} - A_{\text{eq}}^T \lambda_k^{\text{eq}}
$$

Here $\mu > 0$ is the adaptive proximal parameter and $\lambda_k^{\text{eq}}$ are the equality multipliers from the previous outer iteration. The term $\tfrac{1}{\mu} A_{\text{eq}}^T A_{\text{eq}}$ regularizes the reduced Hessian in the equality-constraint normal directions, making the dual solution unique even at degenerate vertices. Equality multipliers are recovered from the penalty optimality condition:

$$
\lambda_{\text{eq}} = \lambda_k^{\text{eq}} - \frac{1}{\mu}\bigl(A_{\text{eq}}\, d - b_{\text{eq}}\bigr)
$$

Larger $\mu$ means more relaxation (softer equality enforcement); smaller $\mu$ tightens toward the standard hard-constraint QP. Near convergence, $b_{\text{eq}} = -c_{\text{eq}}(x_k) \to 0$ and $d \to 0$, so the penalty vanishes and constraint satisfaction is asymptotically exact.

Inequality constraints remain as hard constraints in the active-set method — the EXPAND procedure and warm-starting already handle inequality cycling. The equality absorption is the primary mechanism because it addresses QP infeasibility at degenerate vertices.

Proximal stabilization is **active by default** for equality-constrained problems (when `proximal_tau > 0`). The proximal parameter $\mu$ is computed adaptively at each iteration following Wright (2002, eq 6.6):

$$
\mu_k = \mathrm{clip}\!\bigl(\lVert \nabla_x L_k \rVert^{\tau},\;\mu_{\min},\;\mu_{\max}\bigr)
$$

where $\tau \in (0, 1)$ is the exponent (default 0.5), $\mu_{\min}$ is a floor (default `atol` — the feasibility tolerance, typically $10^{-6}$), and $\mu_{\max}$ is a ceiling (default 0.1). Wright's local convergence theory assumes the KKT residual is below 1; the ceiling handles the regime where the residual is large (far from the solution) by ensuring the proximal weight $1/\mu \geq 1/\mu_{\max}$ — preventing weak equality enforcement that would sabotage the L1 merit function descent. As the solver converges, $\mu$ shrinks below the ceiling, tightening equality enforcement while the floor prevents $1/\mu$ from exploding. Wright's Theorem 6.1 guarantees superlinear convergence when $\tau < 1$.

```python
solver = SLSQP(
    eq_constraint_fn=eq_constraint,
    n_eq_constraints=1,
    proximal_tau=0.5,       # Exponent for adaptive mu (must be in [0,1))
    proximal_mu_min=None,   # Floor on mu; None defaults to atol (feasibility tol)
    proximal_mu_max=0.1,    # Ceiling on mu; keeps 1/mu >= 10
)
```

Setting `proximal_tau=0` **disables sSQP entirely**. Equality constraints are then enforced via direct null-space projection in the QP subproblem (the same mechanism used for inequality constraints). This avoids the ill-conditioning from the $(1/\mu)\, A_{\text{eq}}^T A_{\text{eq}}$ term, which can be beneficial for well-conditioned problems or when cross-platform floating-point reproducibility is important.

```python
solver = SLSQP(
    eq_constraint_fn=eq_constraint,
    n_eq_constraints=1,
    proximal_tau=0,         # Disable sSQP; use direct null-space projection
)
```

For inequality-only problems (no equality constraints), the proximal path is not used and no overhead is incurred.

### Preconditioning the CG solver

When the L-BFGS scaling $\gamma$ is small (common after a few iterations) and proximal stabilization adds a $(1/\mu)\, A_{\text{eq}}^T A_{\text{eq}}$ term to the Hessian, the effective condition number of the QP subproblem can reach $O(10^5)$ or higher. Standard (unpreconditioned) CG requires $O(\sqrt{\kappa})$ iterations to converge — far exceeding the default CG budget of 50 iterations. The result is an inaccurate QP solution, which corrupts the search direction and causes the outer solver to stagnate.

This implementation uses **preconditioned conjugate gradient (PCG)** with the L-BFGS inverse Hessian as preconditioner. The key insight is that the L-BFGS pairs $(s_i, y_i)$ already stored in the history buffer can be used to compute $H_k v = B_k^{-1} v$ via the **two-loop recursion** (Nocedal & Wright, Algorithm 7.4) in $O(kn)$ time — the same cost as the forward HVP.

When the preconditioner $M = B^{-1}$ is used:

- **Without equality constraints**: $M B = I$, so PCG converges in 1 iteration (perfect preconditioning).
- **With equality constraints**: the preconditioner is automatically upgraded to $M = \widetilde{B}^{-1}$ via the **Woodbury identity**:

$$
\widetilde{B}^{-1} = B^{-1} - B^{-1} A_{\text{eq}}^T \bigl(\mu I + A_{\text{eq}} B^{-1} A_{\text{eq}}^T\bigr)^{-1} A_{\text{eq}} B^{-1}
$$

The inner matrix $(\mu I + A_{\text{eq}} B^{-1} A_{\text{eq}}^T)$ is only $m_{\text{eq}} \times m_{\text{eq}}$ (tiny) and is factored once per QP solve. Each preconditioner application costs $O(kn + mn)$. This ensures the preconditioner matches the actual QP system matrix and avoids the pathological case where a plain $B^{-1}$ preconditioner amplifies the proximal eigenvalues when $\gamma$ is small.

For projected (constrained) CG, two preconditioner application paths are available:

* The **simple null-space form** (default) applies $z = P(M(r))$, where $P$ is the null-space projector and $M \approx \tilde B^{-1}$.  Because the residual $r$ already lies in $\text{null}(A)$ after the active-set step, $P(M(P(r))) = P(M(r))$, so a single projection suffices.
* The **constraint preconditioner** of Gould, Hribar & Nocedal (SIAM J. Sci. Comput. 23(4), 2001) is enabled when `use_constraint_preconditioner=True` (which is auto-enabled in Newton-CG mode, `use_exact_hvp_in_qp=True`).  It applies $z = M r - M A^T (A M A^T)^{-1} A M r$, i.e. the projection is done in the $M^{-1}$ inner product rather than the Euclidean one.  This preserves the $M^{-1}$-orthogonality of the CG iterates inside $\text{null}(A)$ and is markedly more robust on ill-conditioned problems.  The extra cost is $m$ additional preconditioner applies to form $M A^T$, amortised over all CG iterations of the QP solve.

Preconditioning is enabled by default (`use_preconditioner=True`). Robust SPD guards ensure that if the preconditioner produces a non-positive-definite result for any residual, CG silently falls back to unpreconditioned mode for that step.

```python
solver = SLSQP(
    eq_constraint_fn=eq_constraint,
    n_eq_constraints=1,
    use_preconditioner=True,   # Default; set False to disable
    adaptive_cg_tol=False,     # Default; set True for Eisenstat-Walker tolerance
    cg_regularization=1e-6,    # Default; minimum eigenvalue threshold for CG
)
```

**CG regularization.** The inner CG solver uses a scale-invariant curvature guard inspired by SNOPT's reduced-Hessian CG solver (Gill, Murray & Saunders, *SIAM Review*, 2005, Section 4.5). At each CG iteration, the effective eigenvalue $p^T B p / \lVert p \rVert^2$ is checked against `cg_regularization` ($\delta^2$). If the effective eigenvalue falls below $\delta^2$, CG declares "bad curvature" and stops. This replaces the previous absolute threshold `pBp <= 1e-8`, which was not scale-invariant: when $\lVert p \rVert$ is small (e.g. because the projected gradient is small near convergence), even a legitimate direction with positive curvature could trigger the threshold.

The default value `cg_regularization=1e-6` ($\delta \approx 10^{-3}$) allows CG to continue through directions with eigenvalues as small as $10^{-6}$ while still stopping on numerical noise. The CG step length and residual update use the true (unregularized) curvature, so the solution is unbiased. Set to `0.0` to disable (CG only stops on truly non-positive curvature).

**Adaptive CG tolerance (Eisenstat-Walker).** When `adaptive_cg_tol=True`, the CG convergence tolerance is adapted based on the outer KKT residual: $\eta_k = \min(0.1,\, \max(\texttt{atol},\, \lVert \nabla_x L \rVert))$, where `atol` is the feasibility tolerance. This avoids over-solving early QPs (when the outer iterate is far from optimal) and tightens the tolerance as convergence proceeds (Eisenstat & Walker, *SIAM J. Sci. Comput.*, 17(1), 1996). This is off by default to preserve baseline convergence behavior. The adaptive tolerance is threaded through to user-provided inner solvers via the `adaptive_tol` parameter on `AbstractInnerSolver.solve()`. `ProjectedCGCholesky` and `ProjectedCGCraig` use it to override their instance CG tolerance; `MinresQLPSolver` ignores it because constraint satisfaction in the full KKT system requires tight tolerance (loosening the MINRES-QLP residual threshold would degrade primal feasibility $\lVert A d - b \rVert$).

**Why not rational CG?** We evaluated the rational conjugate gradient method (Kindermann & Zellinger, arXiv:2306.03670, 2023), which alternates CG steps with Tikhonov regularization steps in a mixed rational Krylov space. It is designed for ill-posed inverse problems with compact operators, not finite-dimensional positive definite QPs. Each rational step requires an inner solve of $(B + \alpha I)^{-1} v$, the regularization parameters need spectrum knowledge, and adapting the method to null-space projection is non-trivial. Standard PCG is simpler, cheaper, and directly addresses the ill-conditioning source.

### L-BFGS reset strategies

The L-BFGS initial Hessian is stored as a per-variable diagonal $B_0 = \text{diag}(d)$ rather than a scalar $\gamma I$. During normal operation the diagonal uses **component-wise secant scaling** ($d_i = |y_i \cdot s_i| / (s_i^2 + \epsilon)$). The diagonal is clipped to a user-configurable range `[lbfgs_diag_floor, lbfgs_diag_ceil]` (default `[1e-4, 1e6]`). The floor is deliberately higher than the machine-epsilon scale: values below $10^{-4}$ indicate that either the secant step is essentially zero (numerical noise) or the curvature on that coordinate is vanishing, both of which are better handled by re-estimating from the next pair than by extrapolating an inverse that is close to a division by zero. `lbfgs_append` and `lbfgs_reset` share these bounds so the clipping regime is consistent across cold starts, soft resets, and ordinary updates.

The reset chain follows a VARCHEN-inspired escalating strategy:

1. **Conditioning-triggered soft reset.**  After each `lbfgs_append`, the condition number $\kappa(B_0) = \max(d)/\min(d)$ is checked.  If $\kappa > 10^6$, all but the most recent $(s, y)$ pair are dropped.  This preserves the newest curvature information while shedding stale pairs that contribute to ill-conditioning.
2. **Failure-triggered soft reset (first failure of a streak only).**  When the QP solver or line search fails, the same soft reset is applied — but **only on the first failure of a streak** ($\texttt{new\_consecutive\_qp\_failures} == 1$, and the symmetric line-search version). Earlier revisions soft-reset on every failure of a streak *and* identity-reset at patience, double-discarding the buffer (3 soft resets + 1 identity reset by the time patience fired). The new chain keeps the most recent pair across iterations 2…patience−1 of the streak so the eventual identity reset has actual history to compare against.
3. **Real-vs-budget QP failure distinction.**  The QP-failure branch gates both the soft reset *and* the `consecutive_qp_failures` increment on `qp_real_failure = ~qp_result.converged & ~qp_result.reached_max_iter`. A QP that simply hits `qp_max_iter` returns a finite Newton direction the merit/line search has already vetted, so there is no reason to discard L-BFGS history; counting that as a failure used to soft-reset on every QP-budget exhaustion and identity-reset by the third one in a row, locking the buffer at $B_0 = I$ while the iterate was actually fine. Budget exhaustion is still surfaced through `diagnostics.n_qp_budget_exhausted` and the `QPcyc=2` verbose column.
4. **Escalating identity reset.**  After `qp_failure_patience` (default 3) consecutive *real* QP failures or `ls_failure_patience` (default 3) consecutive line search failures, the soft reset may be keeping the same problematic pair.  The solver performs a hard identity reset: $B_0 = I$, discarding all stored pairs and the diagonal, allowing L-BFGS to rebuild curvature from scratch.

**Lowered relative-curvature skip floor.** `lbfgs_should_skip(s, y)` skips a pair when `|s·y| / (||s|| ||y||) < 1e-12` (with `s·y > 0`). The floor was lowered from `1e-8` to `1e-12` (machine-precision floor for `float64`) because the previous threshold caused an identity-reset on a near-KKT iterate to lock the buffer in skip-forever mode: with $B_0 = I$ the secant pair satisfies $\lVert y \rVert = \lVert B s \rVert = \lVert s \rVert$, so `|s·y| / (||s|| ||y||)` collapses to the cosine of $(s, B s)$ which floating-point cancellation in the dot product can drive below `1e-8` even when the pair carries genuine information at the available precision. `1e-12` is the smallest threshold where the pair is *truly* noise rather than under-resolved curvature; together with the first-failure-only soft reset above, this breaks the post-identity-reset lock that previously trapped runs at `B = I` for hundreds of iterations.

```python
solver = SLSQP(
    eq_constraint_fn=eq_constraint,
    n_eq_constraints=1,
    qp_failure_patience=3,  # Identity reset after 3 consecutive QP failures
    ls_failure_patience=3,  # Identity reset after 3 consecutive LS failures
)
```

### QP failure recovery

When the QP subproblem fails to converge (exhausts its iteration budget), the solver applies two safeguards to prevent a cascade of failures:

1. **Penalty gating**: the penalty parameter $\rho$ is only updated from QP multipliers when the QP converged. Failed QPs can produce unreliable multipliers that would permanently inflate $\rho$ (since it is monotone non-decreasing), making future line searches harder.
2. **Steepest descent fallback**: the search direction falls back to the negative gradient $-\nabla f(x)$ instead of the unconverged QP direction.

**QP false convergence guard.** The EXPAND procedure's growing tolerance can cause the active-set loop to report convergence on its final iteration under relaxed tolerances, even though the QP did not truly converge at the base tolerance. To prevent this, the QP solver explicitly overrides the convergence flag to `False` whenever the iteration count reaches `max_iter`. This ensures the outer solver triggers L-BFGS resets and steepest descent fallback appropriately.

Combined with the L-BFGS soft reset (above), these mechanisms allow the solver to recover from QP failures rather than entering a permanent stagnation loop.

### Line search failure recovery

When the QP converges but the resulting direction is not a descent direction for the L1 merit function, the backtracking line search exhausts its iteration budget and returns a tiny step size ($\alpha \approx 0.5^{20}$). The solver tracks consecutive line search failures and applies the same escalating L-BFGS reset strategy as for QP failures: soft reset on each failure, escalating to identity reset after `ls_failure_patience` (default 3) consecutive failures. The counter resets to zero on any successful line search.

If `consecutive_ls_failures` reaches `2 * ls_failure_patience`, the solver flags the run as fatally stagnant (`state.ls_fatal = True`) and `terminate()` returns `optx.RESULTS.nonlinear_divergence` (with the granular code `slsqp_jax.RESULTS.line_search_failure` exposed via `Solution.stats["slsqp_result"]`; see [Termination Codes](#termination-codes)).

Convergence requires either (a) the strict pair `stationarity & primal_feasibility`, or (b) the **guarded QP-KKT disjunct** described below.

### Zero-step convergence detection (guarded QP-KKT success)

When the QP solver repeatedly converges with $\lVert d \rVert < \texttt{atol}$, the current point satisfies the QP's KKT conditions (within `cg_tol`) even though the outer stationarity criterion $\lVert \nabla_x L \rVert \leq \texttt{rtol} \cdot \max(|L|, 1)$ may not be met — typically because of residual multiplier imprecision from the regularised Cholesky projection. After `zero_step_patience` (default 3) consecutive zero-step iterations the `state.qp_optimal` flag latches, and the convergence test is augmented with the disjunct

$$
\text{qp\_kkt\_success} = \text{qp\_optimal} \land \text{primal\_feasible} \land \text{ls\_success} \land \bigl( \alpha \geq 1 - 10^{-6} \bigr) \land \text{has\_min\_steps}.
$$

The `ls_success` and `last_alpha == 1` guards are critical: without them, a line search that exhausts its budget without satisfying the Armijo / merit-descent condition (chronic line-search collapse) also produces consecutive zero-norm steps, which an unguarded `qp_optimal & primal_feasible` disjunct would silently classify as success. Requiring a *full* step rules that out: when $d \approx 0$ and the iterate is feasible, $x + 1.0 \cdot 0 = x$ is trivially accepted by the line search at $\alpha = 1$, so any $\alpha < 1$ signals that $d \neq 0$ and the LS had to backtrack — exactly the chronic-collapse symptom we want to keep classified as a failure.

The zero-step check has two stages: a **pre-line-search** check ($\lVert d \rVert < \texttt{atol}$) that catches null-space CG solvers returning $d = 0$ exactly, and a **post-line-search** check ($\alpha \lVert d \rVert < \texttt{atol}$, gated on `ls_success`) that catches full-KKT solvers like `MinresQLPSolver`. MINRES-QLP always returns $\lVert d \rVert > 0$ even when the projected gradient is negligible, because the coupled $(d, \lambda)$ KKT system always has a non-trivial solution. The line search then accepts only a tiny $\alpha$, making the effective step negligible. The post-line-search check detects this and correctly triggers zero-step convergence.

```python
solver = SLSQP(
    zero_step_patience=3,   # Default; declare convergence after 3 consecutive d≈0 steps
)
```

### Inexact stationarity (HR-STCG outer disjunct)

For runs that use `HRInexactSTCG` as the inner solver, the classical Lagrangian-gradient stationarity test $\lVert \nabla_x L \rVert \leq \texttt{rtol} \cdot \max(|L|, 1)$ can stagnate at a multiplier-recovery noise floor — the iterate is genuinely at a KKT point but $\lVert \nabla_x L \rVert$ never drops below the floor because the recovered $\lambda$ carries $O(\varepsilon \cdot \kappa(A A^T))$ noise that contaminates $\nabla_x L = \nabla f - A^T \lambda$. The Heinkenschloss-Ridzal *projected* gradient $\lVert \widetilde{W}_k (g + B d_p) \rVert$ does **not** suffer this floor: the null-space projector annihilates the multiplier-direction component of the gradient by construction, so it stays clean even when $\lambda$ does not. The latest projected gradient norm from each QP solve is forwarded into the outer state as `state.last_projected_grad_norm`, and the diagnostics block accumulates a low-water mark `min_projected_grad_norm`.

The opt-in flag `use_inexact_stationarity` (default `False`) augments the stationarity branch of `terminate()` with a disjunct on `state.last_projected_grad_norm`:

$$\text{stationarity} = \bigl( \lVert \nabla_x L \rVert \le \texttt{rtol} \cdot \max(|L|, 1) \bigr) \;\lor\; \bigl( \texttt{use\_inexact\_stationarity} \land \lVert \widetilde{W}_k g \rVert \le \texttt{rtol} \cdot \max(|L|, 1) \bigr).$$

The same disjunction is mirrored in `step()`'s convergence latch so both call sites agree. Primal feasibility (`max|c_eq| ≤ atol` and `max(0, -c_ineq) ≤ atol`) is still required regardless of which stationarity branch fires; the disjunct only relaxes the *gradient* test, not the constraint test. The toggle is independent of the inner solver — supplying `HRInexactSTCG` without setting the flag leaves the run on the classical test (the projected-gradient field is populated but ignored), while setting the flag with a non-HR inner solver leaves `last_projected_grad_norm = inf` so the disjunct cannot fire. Use this when a profile run shows `|∇L|/|L|` stalling above `rtol` while feasibility is reached and `min_projected_grad_norm / |L|` (visible in `get_diagnostics(state)`) is well below it; if the projected gradient is also stuck, the iterate is genuinely not stationary and the toggle will not change the outcome.

```python
solver = SLSQP(
    inner_solver=HRInexactSTCG(inner=ProjectedCGCraig(...), max_cg_iter=80, cg_tol=1e-8),
    use_inexact_stationarity=True,  # opt-in
    rtol=1e-6,
)
```

### Outer-loop stagnation detection

Even with anti-cycling in the QP, the outer SLSQP loop can fail to make progress — for example, when the problem is infeasible, highly degenerate, or the QP solution is of poor quality. The solver detects this using a **merit-based patience counter**.

At each step, the solver computes the L1 merit value $\varphi = f + \rho \cdot (\lVert c_\text{eq} \rVert_1 + \lVert \max(0, -c_\text{ineq}) \rVert_1)$ and checks whether:

$$
\varphi_\text{new} < \varphi_\text{best} - \eta \cdot \max(|\varphi_\text{best}|, 1)
$$

where $\eta$ is the `stagnation_tol` parameter. If yes, $\varphi_\text{best}$ is updated and the patience counter resets. Otherwise, `steps_without_improvement` increments. Stagnation fires when `steps_without_improvement >= W` (where $W = \lfloor N / 10 \rfloor$ with $N$ = `max_steps`) and at least $W$ steps have elapsed, terminating early with a `nonlinear_divergence` result code. This is O(1) state (vs O(Wn) for an x-history ring buffer) and directly measures optimization progress.

```python
solver = SLSQP(
    eq_constraint_fn=eq_constraint,
    n_eq_constraints=1,
    stagnation_tol=1e-12,    # Minimum relative merit improvement
)
```

The stagnation flag and last step size are included in `sol.stats` for diagnostics:

```python
sol = optx.minimise(objective, solver, x0, has_aux=True, max_steps=100, throw=False)
print(sol.stats["stagnation"])     # False if converged normally
print(sol.stats["last_step_size"]) # Step size from final iteration
```

### Best-iterate divergence rollback

Even with the inner self-correction in `MinresQLPSolver`, pathological QPs can occasionally produce SQP iterates that drift off the linearised feasible region or generate non-finite merit values. The solver carries a **best-iterate divergence detector** as a final guardrail:

- `state.best_x` tracks the iterate that achieved `state.best_merit`.
- At every step, if the new merit exceeds the best-seen merit by more than $\texttt{divergence\_factor} \cdot \max(\lvert\varphi_\text{best}\rvert, 1)$ — *or* if the merit is NaN/Inf — `state.blowup_count` increments. Otherwise it resets to zero.
- Once `blowup_count >= divergence_patience`, `state.diverging` latches `True`, the iterate returned by `step()` is **rolled back to** `state.best_x`, and `terminate()` routes to `RESULTS.nonlinear_divergence`.

Defaults are `divergence_factor=10.0` and `divergence_patience=3`, intentionally generous so the trigger only fires on genuine runaway growth and not on routine penalty inflation. The L-BFGS history, multipliers, and gradients still reflect the *attempted* step; only the iterate the user sees is rolled back. On a benign run the detector stays quiet — `divergence_triggered` is `False` and `n_divergence_blowups` is `0`. Inspired by Heinkenschloss & Ridzal (2014, §4.3) — applying the spirit of their normal-step over-correction control to a line-search SQP.

```python
solver = SLSQP(
    eq_constraint_fn=eq_constraint,
    n_eq_constraints=1,
    inner_solver=MinresQLPSolver(max_iter=200, tol=1e-10),
    divergence_factor=10.0,    # default
    divergence_patience=3,     # default
)
```

The diagnostic accumulator surfaces both the trigger and the cumulative count:

```python
diag = get_diagnostics(sol.state)
print(diag.divergence_triggered)    # True iff rollback fired at least once
print(diag.n_divergence_blowups)    # total blowup events (>=patience when triggered)
```

### Diagnostics

`SLSQPState` carries a `diagnostics` field of type `SLSQPDiagnostics` that accumulates lightweight counters and summary statistics on every call to `step()` (branch-free, so they work under JIT). Use `slsqp_jax.get_diagnostics(state)` on the final state to retrieve them:

```python
from slsqp_jax import SLSQP, get_diagnostics

sol = optx.minimise(objective, solver, x0, has_aux=True, max_steps=200, throw=False)
diag = get_diagnostics(sol.state)

print(diag.n_qp_inner_failures)  # inner-CG/MINRES non-convergence count
print(diag.n_ls_failures)        # line-search non-descent count
print(diag.n_lbfgs_skips)        # pairs rejected by the curvature filter
print(diag.n_nan_directions)     # non-finite QP directions observed
print(diag.max_gamma)            # largest scalar gamma seen during the run
print(diag.min_diag, diag.max_diag)  # per-variable diagonal extremes
print(diag.eq_jac_min_sv_est)    # lower bound on min singular value of J_eq
print(diag.ls_alpha_min)         # smallest line-search step accepted
print(diag.tail_ls_failures)     # consecutive LS failures at termination
print(diag.n_bound_fix_solves)   # non-trivial bound-fixing inner solves
print(diag.max_bound_fixed)      # peak number of variables pinned to bounds
print(diag.max_active_ineq)      # peak active general-ineq / bound count
print(diag.n_merit_regressions)  # steps where merit increased despite LS ok
print(diag.n_proj_refinements)   # total M-metric refinement rounds (MINRES-QLP)
print(diag.max_proj_residual)    # high-water ||A d - b|| post-refinement
print(diag.n_divergence_blowups) # cumulative blowup-count increments
print(diag.divergence_triggered) # True iff best-iterate rollback fired
print(diag.min_projected_grad_norm)         # low-water ||W̃ g|| (HRInexactSTCG only)
print(diag.n_steps_inexact_below_classical) # iterations with ||W̃ g|| < ||∇L||
```

Because a line search that can only take $\alpha \approx 2^{-20}$ no longer counts as convergence, these counters are the recommended way to assess the quality of a run whose result code is `RESULTS.successful` but whose objective / constraints look suspect. Large `n_qp_inner_failures` together with a small `eq_jac_min_sv_est` usually indicates a near-rank-deficient equality Jacobian; a non-zero `tail_ls_failures` at termination means the solver bailed out on stagnation and the returned iterate is a best-merit point, not a stationary point.

`n_bound_fix_solves` grows with the number of *non-trivial* bound-fixing passes (no-op passes where the free mask did not change are short-circuited and do not contribute). `max_bound_fixed` is the peak count of variables pinned to a box bound in any single iteration; when this tracks `n` closely you are essentially solving a degenerate LP and the outer SLSQP step tends to be very short. `n_merit_regressions` flags iterations whose line search reported success but whose L1 merit nonetheless went up - a sign that `rho` is undersized or the HVP approximation is stale.

#### L-BFGS skip counter fix

Prior to this release `n_lbfgs_skips` was computed as `new.count == state.count`, which misfires as soon as the ring buffer saturates (`count == memory`): every subsequent step was (incorrectly) flagged as a skipped pair. The counter is now based on the *same* `should_skip` predicate that `lbfgs_append` evaluates internally, exposed publicly as `slsqp_jax.lbfgs_should_skip(s, y)` so downstream code can reproduce the decision without diffing history fields.

#### Expanded verbose output

The built-in `verbose=True` callback now prints, in addition to the step summary:

- `|∇L|/|L|` - the relative stationarity ratio used by the convergence test.
- `|W̃g|` - the inner solver's projected-gradient norm (the noise-aware stationarity proxy used by `use_inexact_stationarity`); `+inf` for inner solvers that do not produce it.
- `Δmerit` - signed change of the L1 merit versus the best-so-far.
- `skip` - whether this step skipped the L-BFGS curvature update.
- `s·y` - raw curvature inner product `s . y` for the just-built pair.
- `rel_curv` - normalised curvature `|s·y| / (||s|| ||y||)`, the cosine used by the L-BFGS skip predicate.
- `#fix` - number of variables pinned at box bounds in the final QP direction.
- `fix#` - number of non-trivial bound-fixing inner solves used this step.
- `LS tail` - current consecutive line-search failure counter.

Together these let you distinguish "the solver is slow because of active-set churn" (`#fix` oscillating) from "the QP is producing bad directions" (`skip` or `LS ok: False` repeatedly) from "noise-floor stationarity stall rescuable by `use_inexact_stationarity`" (`|W̃g|/|L|` well below `|∇L|/|L|` and below `rtol` while `|∇L|/|L|` is stuck above) from "post-identity-reset L-BFGS skip lock" (`κ_B = 1.0e+00`, `skip = True`, `rel_curv` near `eps`) from "we are simply running out of iterations" (`|∇L|/|L|` decreasing monotonically but not below `rtol`).

#### Bound-fix inner-solve short-circuit

The bound-handling loop after the QP direction used to run exactly `5` reduced-space inner solves per outer iteration, even when the free-variable mask was already stable. On problems such as `Portfolio(n=5000)` with `MinresQLPSolver` this meant `5 * 500 = 2500` full `(n + m)` MINRES-QLP solves per run, the vast majority of which had their results masked out by `use_new=False`. The loop now wraps each inner solve in `jax.lax.cond(any_change & any_fixed, ...)` and records the number of non-trivial solves in `QPResult.bound_fix_solves` (surfaced as `diag.n_bound_fix_solves`). On the portfolio benchmark this cut the average from 5 to well under 1 solve per outer step with no change in the accepted direction.

## Diagnostics

The `slsqp_jax.diagnostics` sub-package layers post-mortem diagnostics on top of the production `SLSQP` API. It is intended for the case where a run failed (early termination, slow progress, suspected bad direction) and the user wants to narrow the search before deciding what to change.

> **Scope limitation.** The diagnostics layer reports *hypotheses with evidence*, not verdicts. It cannot solve the hard cases (wrong gradient, mis-formulated constraints, badly-scaled problem, ill-posed local geometry). What it does do is rule out solver-side causes faster than reading `AGENTS.md` by hand. Signal text reads "X looks suspicious because Y" and the report wording leans on "consistent with" / "candidate cause" rather than "your problem is X".

### Quick start

```python
import jax.numpy as jnp
from slsqp_jax import SLSQP, diagnose

def f(x, args):
    return jnp.sum((x - 1.0) ** 2)

def c_eq(x, args):
    return jnp.array([jnp.sum(x) - 5.0])

solver = SLSQP(eq_constraint_fn=c_eq, n_eq_constraints=1)
x0 = jnp.zeros(4)

report = diagnose(solver, f, x0)
report.print_summary()
```

The report includes:

1. The granular `slsqp_jax.RESULTS` termination code (with the exact message string) and the coarse `optx.RESULTS` it maps to.
2. Final-iterate metrics (`||grad_L||/max(|L|,1)`, primal feasibility, `rho`, L-BFGS condition number).
3. Fired signals — partitioned into "in scope for this termination code" and "less likely given the termination mode" (per `SCOPE_BY_TERMINATION`); within each group sorted by `confidence` (`high` → `medium` → `low`). Each signal carries a one-line summary, the evidence numbers, the artifact keys it built (e.g. `J_eq`, `JJT`, `singular_values`), and concrete suggestions.
4. Multi-signal diagnoses — pattern matches against fired-signal sets (e.g. `lbfgs_conditioning_extreme + line_search_collapse → stale_lbfgs_curvature`). Single-signal cases are surfaced directly without a rule.
5. A prose-annotated dump of every `SLSQPDiagnostics` counter (counters' docstrings drive the prose).
6. An ASCII trajectory chart of `(step, f, merit, rel_KKT, alpha, qp_iter, qp_ok, ls_ok)`.

### Quick start for `minimize_like_scipy` users

If you do not construct `SLSQP` yourself but go through `minimize_like_scipy` (or your own wrapper around `optimistix.minimise`), the diagnostics layer offers two zero-refactor opt-ins so you do not have to lift the call site into the lower-level API:

```python
import slsqp_jax

# (1) Context manager — wrap your existing wrapper code as-is.
with slsqp_jax.diagnostic_run() as ctx:
    sol = slsqp_jax.minimize_like_scipy(fun, x0, options={"maxiter": 100})
    # `sol` is a real optimistix.Solution; downstream code is unaffected.

# After the with-block, ctx.runs and ctx.reports hold one entry per
# intercepted call.  By default the on-exit print is *quiet on clean
# success* and only renders the report for any run that did NOT
# converge, plus a one-line trailer telling you how to inspect more.
print(ctx)  # DiagnosticContext(runs=1, failing=0, intercepted=1, ...)

# (2) Drop-in helper — same signature as minimize_like_scipy.
sol, report = slsqp_jax.diagnose_minimize_like_scipy(fun, x0, options={...})
report.print_summary()
```

`diagnostic_run(...)` works by monkey-patching `optimistix.minimise` for the duration of the `with` block. Calls whose `solver` is an `SLSQP` instance are re-routed through `debug_run` (and the resulting `DebugReport` is stashed on `ctx.reports`). Calls with any other solver pass through unchanged. The context manager is **not re-entrant and not thread-safe**: a process-wide lock rejects nested or concurrent enters with `RuntimeError`.

The `print_on_exit` argument controls the on-exit summary: `"auto"` (default) prints reports only for failing runs plus a trailer; `"always"` prints every report; `"never"` suppresses everything. Pass `output=` to redirect to a file-like sink instead of `sys.stdout`.

**Import-binding caveat.** The patch replaces `optimistix.minimise`-the-module-attribute. If your code does `from optimistix import minimise` and then calls `minimise(...)`, the name was bound at import time and the patch will *not* apply. The context manager detects this case (zero intercepts and zero passthroughs on exit) and emits a `UserWarning` pointing at `diagnose_minimize_like_scipy(...)` as the non-magic alternative.

### Public API

```python
from slsqp_jax import (
    diagnose,                # auto: run + classify + report (low-level: needs an SLSQP instance)
    diagnostic_run,          # context manager: intercepts optx.minimise inside the `with` block
    diagnose_minimize_like_scipy,  # drop-in replacement for minimize_like_scipy with diagnostics
    debug_run,               # manual: just produce a DebugRunResult
    capture_state_at_step,   # ad-hoc: re-run to a specific step (with reproducibility check)
    DebugReport,
    DebugRunResult,
    DiagnosticContext,
    Signal,
    Diagnosis,
    StepSummary,
)
```

`diagnose(...)` always emits a report, even on `RESULTS.successful`, because the user can request a diagnosis when "progress speed has degraded a lot, but not enough for an early exit to be fired". Confidence and the firing-signal set determine whether anything substantive prints; on a clean run the report is one paragraph plus the counter dump.

`capture_state_at_step(...)` re-runs the solver to a specific step and returns the live `SLSQPState`. When passed `expected_summary=run.summaries[k-1]`, it asserts the recovered state's `StepSummary` digest matches the original; on mismatch it raises `RuntimeError("debug-run trajectory is not reproducible: …")` with a list of diverging fields. This is the load-bearing reproducibility guard — without it the tool can silently lie about which iterate it is showing.

### Performance contract

The runner is a *debug* tool, not a production loop. It deliberately trades the on-device `jax.lax.while_loop` of `optimistix.iterative_solve` for a host-driven Python `for` loop calling `jit(step)`, paying one device→host sync per iteration. On GPU this can turn a 3 s production run into a 30-60 s diagnose run. That cost is acceptable because the alternative is the user reading verbose output by eye for hours.

To keep the cost bounded, every per-step evaluator splits into a cheap predicate (`cheap_predicate(summary) -> bool`, scalar-only, runs every step) and an expensive `build_artifacts(state) -> dict[str, np.ndarray]` (only invoked the first iteration the predicate fires; pulls the needed device arrays to host once and computes derived linalg). End-of-run evaluators only see the final state plus the trajectory of summaries and pay no per-step cost.

### Confidence

Each fired signal carries a single ranking field, `confidence ∈ {low, medium, high}`, derived from two underlying axes via the lookup table:

| `specificity \ magnitude` | `marginal` | `moderate` | `extreme` |
| ------------------------- | ---------- | ---------- | --------- |
| `specific`                | medium     | high       | high      |
| `ambiguous`               | low        | medium     | medium    |
| `generic`                 | low        | low        | low       |

`specificity` is a *static* editorial classification declared at evaluator registration (`specific` = pattern essentially diagnostic for one cause; `ambiguous` = consistent with two or three; `generic` = consistent with many). `magnitude` is *dynamic*, computed at fire time from the largest evidence-to-threshold ratio (`< 10×` → `marginal`; `< 100×` → `moderate`; otherwise → `extreme`). Use cases that need both axes can read `signal.specificity` / `signal.magnitude` directly.

### Starter signal set

| Signal | Flavour | Specificity | Threshold |
| ------ | ------- | ----------- | --------- |
| `eq_jacobian_rank_deficient` | per-step | specific | `eq_jac_min_sv_est < 1e-8` (cumulative low-water) |
| `lbfgs_conditioning_extreme` | per-step | ambiguous | `max_diag/min_diag > 1e6` for >= 3 consecutive steps OR a single-step burst above `1e8` |
| `multiplier_recovery_noise` | end-of-run | specific | textbook noise-floor signature (see `AGENTS.md`) |
| `line_search_collapse` | end-of-run | ambiguous | `tail_ls_failures > 0` AND `ls_alpha_min <= 10 * 2**-line_search.max_steps` (the LS floor) |
| `qp_budget_or_pingpong` | end-of-run | generic | `n_qp_budget_exhausted > 0` OR `n_qp_ping_pong > 0` |
| `merit_oscillation` | end-of-run | generic | `n_merit_regressions > step_count / 10` |
| `lpeca_overpredicting` | end-of-run | ambiguous | `n_lpeca_capped / step_count > 0.5` OR `n_lpeca_bypassed == step_count` |
| `infeasible_termination` | end-of-run | specific | termination code in `{infeasible, nonfinite}` OR feasibility never satisfied |
| `divergence_rollback_triggered` | end-of-run | specific | `divergence_triggered == True` OR `n_divergence_blowups > 0` (best-iterate rollback fired) |
| `merit_penalty_explosion` | end-of-run | specific | `max(rho)/min(rho) > 1e6` OR a single-step `rho` jump above `1e3` |
| `penalty_starvation` | end-of-run | specific | `rho` constant for the first `max(5, n_steps // 3)` steps while `max_eq_violation` (or `max_ineq_violation`) grew monotonically by at least `5x` |

The `lbfgs_conditioning_extreme` predicate carries two clauses: the original "streak" (`> 1e6` for 3 consecutive iterations, the same patience the L-BFGS reset chain uses) and a single-step "burst" above `1e8` that catches catastrophic one-shot blow-ups (e.g. `kappa(B0) = 1.1e+09` for a single step before the reset chain clamps back to identity). The streak gate alone misses these because the post-spike identity reset breaks the streak after one step. The fired signal carries `evidence["burst_clause"] == 1` when the burst clause fired, `0` when the streak fired.

The `line_search_collapse` predicate keys on the LS floor `10 * 2**-line_search.max_steps` (~ `9.5e-6` for the default `LineSearchConfig.max_steps == 20`) rather than a fixed `1e-10` literal: backtracking can never accept an `alpha` below `2**-max_steps` regardless of how badly the QP step misbehaves, so a hard `< 1e-10` threshold could never trip on the default solver. Solvers with a custom `LineSearchConfig.max_steps` get the floor recomputed automatically through `EvalContext`.

The three end-of-run signals `divergence_rollback_triggered`, `merit_penalty_explosion`, and `penalty_starvation` together cover the "feasible-start divergence cascade" failure mode: the run starts at or near feasibility, the Han-Powell penalty update stays starved while feasibility drifts (`penalty_starvation`), the merit-penalty update eventually over-corrects when the drift is noticed (`merit_penalty_explosion`), the resulting QP step poisons the L-BFGS history, the line search collapses, and the best-iterate divergence rollback fires (`divergence_rollback_triggered`). The multi-signal rule `penalty_starvation_cascade` (in `slsqp_jax/diagnostics/playbook.py`) names the chain when both `penalty_starvation` and `merit_penalty_explosion` are present.

The set is append-only; adding a new signal requires (a) one function in `slsqp_jax/diagnostics/signals.py` with a docstring citing the relevant `AGENTS.md` section + the threshold inline, (b) a `register_evaluator(...)` call, and (c) matching `test_signal_<name>_synthetic` + `test_signal_<name>_integration` in `tests/diagnostics/`. The cap-policy enforcement test (`tests/diagnostics/test_registry.py`) fails CI if either test is missing.

## License

MIT

```{toctree}
:maxdepth: 2
:caption: Contents
:hidden:

self
api/index
```
