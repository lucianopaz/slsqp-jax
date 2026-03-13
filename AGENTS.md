# JAX Optimization Engineer Agent

## Role
You are an expert Numerical Analyst and JAX Engineer. Your goal is to reimplement the SLSQP (Sequential Least Squares Programming) algorithm in pure JAX, designed to run on GPUs.

## Local Development

Local development uses a virtual environment managed by `uv`.

### Setup and Dependencies
```bash
# Install or update all dependencies (including dev extras)
uv sync --all-extras
```

### Running Tests
While iterating fixes, it's highly recommended to run tests excluding slow ones

```bash
# Run all tests
uv run pytest

# Run tests excluding slow ones
uv run pytest -m "not slow"

# Run specific test file or class
uv run pytest tests/test_slsqp.py::TestSLSQPBoxConstraints -v
```

### Linting and Formatting
The project uses `prek` to run pre-commit hooks for static type checks and formatting.

```bash
# Run all pre-commit hooks (linting, formatting, type checks)
prek run
```

## Architectural Guidelines

### Core Principles
1.  **Framework:** Use `optimistix` for the optimization loop structure (`AbstractMinimiser`). The solver implements the `optimistix.AbstractMinimiser` interface and runs through `optimistix.minimise`.
2.  **Differentiation:** Use `jax.grad`, `jax.jacrev`, and `jax.jvp` for derivatives. Do not approximate derivatives numerically unless strictly necessary.
3.  **Typing:** Strictly use `jaxtyping` with `beartype` decorators for all functions.
4.  **Functional:** All logic must be pure functions. No side effects. Use `jax.lax.scan` or `jax.lax.while_loop` for loops.

### Scaling Considerations: L-BFGS over BFGS

Classical SLSQP maintains a **dense** n×n BFGS approximation to the Hessian of the Lagrangian. This requires O(n²) memory and O(n²) work per iteration. For target problem sizes (5,000–50,000 variables), this is prohibitive:

| n | Dense Hessian memory | L-BFGS memory (k=10) |
|---|---|---|
| 1,000 | 8 MB | 160 KB |
| 10,000 | 800 MB | 1.6 MB |
| 50,000 | 20 GB | 8 MB |

L-BFGS stores only the last k step/gradient-difference pairs (s_i, y_i) and computes Hessian-vector products in O(kn) time using the compact representation:

```
B_k = γI - W N⁻¹ Wᵀ
```

where W = (γS, Y) and N is a small 2k×2k matrix. Powell's damping is applied to each curvature pair before storage to ensure positive definiteness.

### Scaling Considerations: Projected CG over Dense KKT

Classical SLSQP solves the QP subproblem by forming and factorising the (n+m)×(n+m) dense KKT system at O(n³) cost. This implementation uses **projected conjugate gradient** (CG) inside an active-set loop:

1. **Projection**: For active constraints with matrix A (m_active × n), define the null-space projector P(v) = v - Aᵀ(AAᵀ)⁻¹Av. The AAᵀ system is only m_active × m_active (tiny, since m << n).
2. **CG in null space**: Run conjugate gradient on the projected system. Each iteration requires one HVP (O(kn) for L-BFGS) and one projection (O(mn)).
3. **Active-set outer loop**: Add most violated inequality or drop most negative multiplier until KKT conditions are satisfied.

Total cost per QP solve: O(n·k·t), where t is the number of CG iterations (typically t << n).

**Preconditioning**: The CG solver supports an optional preconditioner M ≈ B̃⁻¹. By default, the L-BFGS inverse Hessian (two-loop recursion, Algorithm 7.4 of Nocedal & Wright) is used. When equality constraints are present *and* sSQP is active (`proximal_tau > 0`), the preconditioner is upgraded to B̃⁻¹ via the Woodbury identity so it matches the actual QP system matrix B̃ = B + (1/μ) A_eq^T A_eq, where μ is the adaptive proximal parameter. When sSQP is disabled (`proximal_tau = 0`), the plain B⁻¹ preconditioner is used (no Woodbury correction needed since there is no proximal term). For projected CG, the preconditioner is applied as z = P(M(P(r))) to keep the search direction in the constraint null space.

**CG regularization**: The CG curvature guard uses a scale-invariant relative threshold `pBp <= delta^2 * ||p||^2` instead of a hard absolute one. This prevents false negative-curvature detection when `||p||` is small (e.g. after an L-BFGS diagonal reset with high condition number). The step length and residual use the true curvature, so the CG solution is unbiased. Controlled by `cg_regularization` on `SLSQP` (default 1e-6). Based on SNOPT Section 4.5 (Gill, Murray & Saunders, 2005).

### Derivative Computation

By default, the solver computes:
- **Objective gradient** via `jax.grad` (reverse-mode)
- **Constraint Jacobians** via `jax.jacrev` (reverse-mode, O(m) passes — faster than jacfwd's O(n) passes when m << n)
- **HVP fallback** via `jax.jvp(jax.grad(f), ...)` (forward-over-reverse)

Users can supply custom derivative functions to:
- Bypass automatic differentiation when functions contain `while_loop` or other non-reverse-differentiable operations
- Avoid redundant computation
- Handle functions that are not reverse-mode differentiable

### Reverse-mode AD Limitations

JAX's reverse-mode AD does not support differentiating through `jax.lax.while_loop` or variable-length control flow. If user functions contain such operations, they must supply custom derivative functions via the optional fields on `SLSQP`:
- `obj_grad_fn` - bypasses `jax.grad`
- `eq_jac_fn`, `ineq_jac_fn` - bypasses `jax.jacrev`
- `obj_hvp_fn`, `eq_hvp_fn`, `ineq_hvp_fn` - bypasses forward-over-reverse

### Algorithm Structure

Each SLSQP iteration performs four steps:
1. **QP subproblem**: Construct quadratic approximation of objective and linearise constraints. Solve QP to obtain search direction.
2. **Line search**: Use Han-Powell L1 merit function with backtracking Armijo conditions.
3. **Accept step**: Update iterate x_{k+1} = x_k + α·d_k.
4. **Hessian update**: Append new curvature pair (s, y) to L-BFGS history (or skip if using exact HVPs).

### Robustness: Anti-cycling and Stagnation Detection

**QP anti-cycling (EXPAND procedure):** The active-set QP solver uses the EXPAND procedure (Gill, Murray, Saunders & Wright, 1989) to prevent cycling at degenerate vertices. A working tolerance `delta_k = tol + k * tau` grows monotonically each active-set iteration, ensuring strict progress and preventing the same constraint from being repeatedly activated and deactivated. Controlled by the `expand_factor` parameter on `solve_qp`.

**LPEC-A active set identification:** As an alternative to (or complement of) EXPAND, the solver supports LPEC-A (Oberlin & Wright, 2005, Section 3.3) for predicting the active constraint set before the QP active-set loop. LPEC-A computes a proximity measure `rho_bar` from primal constraint values, dual multipliers, and the stationarity residual, then applies a threshold test `c_ineq_i <= (beta * rho_bar)^sigma` to predict active constraints. Under MFCQ + second-order sufficiency, the prediction is asymptotically exact (including weakly active constraints). Controlled by `active_set_method` on `SLSQP`: `"expand"` (default, no LPEC-A), `"lpeca_init"` (LPEC-A warm-start + EXPAND fallback), or `"lpeca"` (LPEC-A + fixed tolerance, no EXPAND growth). Threshold parameters: `lpeca_sigma` (default 0.9), `lpeca_beta` (default `1/(m_ineq + n + m_eq)`). Optional LP refinement via `mpax.r2HPDHG` when `lpeca_use_lp=True` (requires `mpax` from the `extras` dependency group).

**Multiplier stability:** Wright (SIAM J. Optim., 2002, Theorem 5.3) proved that SQP superlinear convergence requires multiplier stability across iterations. The solver promotes this through: (1) warm-starting the QP active set from the previous iteration, (2) an adaptive EXPAND tolerance tied to the outer KKT residual, (3) adaptive proximal multiplier stabilization (sSQP, active for equality-constrained problems when `proximal_tau > 0`, with `mu = clip(kkt_residual^tau, mu_min, mu_max)` per Wright eq 6.6; controlled by `proximal_tau`, `proximal_mu_min`, and `proximal_mu_max` on `SLSQP`), and (4) alpha-scaled multiplier blending (`lambda_new = lambda_prev + alpha * (lambda_QP - lambda_prev)`) for L-BFGS secant pair consistency when the line search accepts a partial step. Raw QP multipliers are stored in the state for convergence checking. The ceiling `mu_max` (default 0.1) prevents weak equality enforcement when the KKT residual is large (Wright's local theory assumes eta < 1). Setting `proximal_tau = 0` disables sSQP entirely: equality constraints are enforced via direct null-space projection in the QP subproblem instead of the augmented Lagrangian penalty. This avoids the ill-conditioning introduced by the `(1/μ) A_eq^T A_eq` term, which can be beneficial for well-conditioned problems or when cross-platform reproducibility is important.

**L-BFGS diagonal reset (SNOPT-style):** The L-BFGS initial Hessian is stored as a per-variable diagonal `B_0 = diag(d)` instead of a scalar `gamma * I`. When the QP solver fails or the line search fails, `diag(B_k)` is extracted from the compact representation, all stored pairs are discarded, and the approximation restarts with the per-variable curvature preserved. This prevents the gamma-freeze loop where tiny steps prevent L-BFGS updates. Based on Gill, Murray & Saunders, SIAM Review, 47(1), 2005, Section 3.3. After `qp_failure_patience` (default 3) consecutive QP failures, or `ls_failure_patience` (default 3) consecutive line search failures, an escalating identity reset (`B_0 = I`) replaces the diagonal reset to break ill-conditioning cycles where the extracted diagonal perpetuates the same problematic scaling.

**QP failure recovery:** When the QP fails to converge, the solver (1) gates penalty parameter updates so unreliable multipliers cannot permanently inflate `rho`, and (2) falls back to a steepest descent direction (`-grad`) instead of using the unconverged QP direction. The QP solver also guards against false convergence: when the active-set loop reaches `max_iter`, the convergence flag is forced to `False` regardless of the EXPAND-relaxed tolerance state. Combined with the L-BFGS diagonal reset, this prevents the cascade where one QP failure leads to permanent stagnation.

**Line search failure recovery:** When the QP converges but the resulting direction is not a descent direction for the L1 merit function (typically due to a poorly conditioned L-BFGS Hessian), the backtracking line search exhausts its iteration budget and returns a tiny step size. The solver tracks consecutive line search failures and applies the same escalating L-BFGS reset strategy: SNOPT diagonal reset on each failure, escalating to identity reset after `ls_failure_patience` (default 3) consecutive failures. This closes the gap where the QP succeeds but the Hessian quality is too poor for the direction to be useful. Controlled by `ls_failure_patience` on `SLSQP`.

**Outer-loop stagnation detection:** The solver uses a sliding-window x-value comparison with window size `W = max_steps // 10`. At each step k >= W, it compares `||x_k - x_{k-W}|| / max(||x_k||, 1)` against `stagnation_tol`. If the relative change is below tolerance, the solver terminates early with `nonlinear_divergence`. This is more robust than merit-based consecutive-counter detection, which can miss cases where merit improvements are minuscule but above threshold. Controlled by `stagnation_tol` on `SLSQP`.

**Convergence criterion:** The solver declares convergence when both conditions hold (and at least `min_steps` iterations have elapsed): (1) **relative stationarity** `||∇_x L|| <= rtol * max(|L|, 1)` where `L = f − λ_eq^T c_eq − μ_ineq^T c_ineq` is the Lagrangian value, and (2) **primal feasibility** `max|c_eq| <= atol` and `max(0, −c_ineq) <= atol`. The relative stationarity criterion is motivated by floating-point arithmetic: when the gradient is negligible relative to `|L|`, a step of that magnitude cannot change `L` in finite precision. The `max(|L|, 1)` safeguard prevents the criterion from becoming vacuous when `L ≈ 0`. `rtol` controls stationarity (default 1e-6); `atol` controls feasibility (default 1e-6).

## The Task: SLSQP Implementation
You need to port the logic of SLSQP (Sequential Least Squares Programming) to JAX.
* **Core logic:** SLSQP solves a nonlinear optimization problem by solving a Quadratic Programming (QP) subproblem at each step to determine the search direction.
* **Subproblem:** Since JAX/Optimistix does not have a native QP solver for this specific task, you may need to implement an **Active Set method** or a **Projected Gradient method** to solve the QP subproblem at each iteration.
* **Merit Function:** Use an L1-merit function (Han-Powell) to accept/reject steps, similar to the reference implementation.

## Reference Material
You have access to two key references. You must study the logic of the C/Fortran source to understand the specific active-set updates used in SLSQP.

1.  **Design Patterns (Optimistix):** `https://github.com/patrick-kidger/optimistix`
    * Use this for the class structure (`AbstractMinimiser`, `AbstractDescent`).
2.  **Algorithm Logic (SciPy SLSQP Source):** `https://github.com/scipy/scipy/tree/main/scipy/optimize/`
    * Specifically, look at `src/slsqp.c` (C source) which SciPy wraps. This contains the LDLT factorization and active-set logic.

## Documentation Requirements

**Any change to the core optimization algorithm MUST be accompanied by corresponding updates to the user-facing documentation.** This includes:

1. **`README.md`**: Update the relevant Algorithm subsection (or add a new one) describing the change, its motivation, and any new parameters.
2. **`docs/source/index.md`**: Mirror the same changes (this file is the Sphinx/ReadTheDocs source and should stay in sync with the README).
3. **`AGENTS.md`**: Update the Architectural Guidelines if the change affects the high-level algorithm structure, scaling analysis, or robustness mechanisms.

This applies to all algorithmic changes — new robustness features, modified Hessian handling, preconditioning changes, QP solver modifications, line search adjustments, etc. Do not merge code-only changes without documentation.

## Verification
* Create a test suite using `pytest`.
* Compare your JAX implementation against `scipy.optimize.minimize(method='SLSQP')`.
* Verify gradients match `jax.grad` and constraints are satisfied to within `1e-6`.
