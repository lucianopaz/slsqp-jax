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

where W = (γS, Y) and N is a small 2k×2k matrix. VARCHEN-style B0 damping (Lotfi et al., 2020) is applied to each curvature pair before storage, damping toward `B0 = diag(diagonal)` instead of the full L-BFGS approximation B.  This is O(n) instead of O(k²n), always well-conditioned, and avoids the circular dependency where a badly-conditioned B poisons its own damping.  The damping threshold is configurable via `damping_threshold` on `SLSQP` (default 0.2); setting it to 0.0 disables damping entirely.  The solver tracks the condition number `kappa(B0) = max(d)/min(d)` and performs a soft reset (keep most recent pair, drop rest) when kappa exceeds 1e6.

### Scaling Considerations: Projected CG over Dense KKT

Classical SLSQP solves the QP subproblem by forming and factorising the (n+m)×(n+m) dense KKT system at O(n³) cost. This implementation uses **projected conjugate gradient** (CG) inside an active-set loop:

1. **Projection**: For active constraints with matrix A (m_active × n), define the null-space projector P(v) = v - Aᵀ(AAᵀ)⁻¹Av. The AAᵀ system is only m_active × m_active (tiny, since m << n).
2. **CG in null space**: Run conjugate gradient on the projected system. Each iteration requires one HVP (O(kn) for L-BFGS) and one projection (O(mn)).
3. **Active-set outer loop**: Add most violated inequality or drop most negative multiplier until KKT conditions are satisfied.

Total cost per QP solve: O(n·k·t), where t is the number of CG iterations (typically t << n).

**Preconditioning**: The CG solver supports an optional preconditioner M ≈ B̃⁻¹. Two preconditioner types are available, controlled by `preconditioner_type` on `SLSQP`:

- **`"lbfgs"` (default)**: L-BFGS inverse Hessian (two-loop recursion, Algorithm 7.4 of Nocedal & Wright). When equality constraints are present *and* sSQP is active (`proximal_tau > 0`), the preconditioner is upgraded to B̃⁻¹ via the Woodbury identity so it matches the actual QP system matrix B̃ = B + (1/μ) A_eq^T A_eq, where μ is the adaptive proximal parameter. When sSQP is disabled (`proximal_tau = 0`), the plain B⁻¹ preconditioner is used.

- **`"diagonal"`**: Stochastic Hessian diagonal estimate (Bekas, Kokiopoulou & Saad, 2007). Probes the exact Lagrangian HVP with `diagonal_n_probes` (default 20) Rademacher random vectors to estimate diag(H_L). Each probe costs one HVP evaluation. The preconditioner is M⁻¹ = diag(1/d̂) with small/negative entries clamped to a positive floor. Requires an exact HVP (`use_exact_hvp_in_qp=True` or `obj_hvp_fn` provided). When the proximal term is active, the Woodbury correction is applied with D = diag(d̂) instead of B. This preconditioner is independent of L-BFGS history quality, making it robust on ill-conditioned problems where L-BFGS resets degrade the preconditioner.

For projected CG, the **constraint preconditioner** (Gould, Hribar & Nocedal, SIAM J. Sci. Comput., 2001) is used: `z = M r - M A^T (A M A^T)^{-1} A M r`. This solves the augmented saddle-point system and preserves the M^{-1}-inner product in null(A), unlike the naive P(M(r)) which destroys preconditioning quality for ill-conditioned problems. The cost is m extra preconditioner applications to form M A^T, amortized over all CG iterations.

**Newton-CG mode** (`use_exact_hvp_in_qp=True`): Replaces the frozen L-BFGS HVP with the exact Lagrangian HVP (via AD) in the CG inner loop. Each CG step costs one forward-over-reverse pass. L-BFGS is still updated for preconditioning and as a fallback. When the user provides `obj_hvp_fn`, that is used; otherwise the solver auto-computes the objective HVP via `jax.jvp(jax.grad(f), ...)`. This is the standard Newton-CG approach (KNITRO, Ipopt) and dramatically improves convergence on ill-conditioned problems where L-BFGS cannot capture extreme eigenvalue spread.

**CG regularization**: The CG curvature guard uses a scale-invariant relative threshold `pBp <= delta^2 * ||p||^2` instead of a hard absolute one. This prevents false negative-curvature detection when `||p||` is small (e.g. after an L-BFGS diagonal reset with high condition number). The step length and residual use the true curvature, so the CG solution is unbiased. Controlled by `cg_regularization` on `SLSQP` (default 1e-6). Based on SNOPT Section 4.5 (Gill, Murray & Saunders, 2005).

**Cholesky-based projection**: The AAᵀ factorization used in the null-space projector is computed once via `jnp.linalg.cholesky` and reused for all CG iterations via `cho_solve`. This amortizes the O(m³) factorization cost: each subsequent projection solve is O(m²) instead of O(m³), giving a significant speedup when the CG loop runs many iterations.

**Bound separation via variable fixing**: Box constraints (x_lower ≤ x ≤ x_upper) are handled separately from general inequality constraints to avoid inflating the projection matrix. The QP solver operates on general equality and inequality constraints only (m_eq + m_gen rows), keeping the projection cost at O((m_eq + m_gen)³). After the QP solve, an iterative bound-fixing loop identifies variables whose QP direction violates bounds, fixes them at their bound values, and re-solves the CG in the reduced free-variable subspace. The loop also checks bound multiplier signs and releases variables with wrong-sign multipliers (indicating the variable wants to move away from its bound). This active-set iteration for bounds typically converges in 2-5 passes and is capped at 5 iterations. The `_solve_projected_cg` function supports `free_mask` and `d_fixed` parameters: fixed variables are held constant while the CG optimizes over the free subspace, with the effective gradient shifted to account for cross-coupling between fixed and free variables.

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
4. **Hessian update**: Append new curvature pair (s, y) to L-BFGS history (or skip if using exact HVPs). The gradient difference `y_k = ∇_x L(x_{k+1}, λ_{k+1}) − ∇_x L(x_k, λ_{k+1})` uses the **same** (blended) multipliers at both points to satisfy the secant condition (Nocedal & Wright §18.3). The initial Hessian uses **component-wise secant diagonal scaling**: `B₀ = diag(d)` where `d_i = |y_i · s_i| / (s_i² + ε)`, clipped to `[1e-2, 1e6]`. This gives `d_i ≈ |H_{ii}|` for diagonal Hessians regardless of the step direction. The classical Shanno-Phua formula `d_i = y_i² / (yᵀs)` uses a single scalar normalizer, producing `d ∝ h_i²` for multi-component steps, which severely underestimates curvature and causes 10-100x slowdowns on problems with moderate condition numbers.

### Robustness: Anti-cycling and Stagnation Detection

**QP anti-cycling (EXPAND procedure):** The active-set QP solver uses the EXPAND procedure (Gill, Murray, Saunders & Wright, 1989) to prevent cycling at degenerate vertices. A working tolerance `delta_k = tol + k * tau` grows monotonically each active-set iteration, ensuring strict progress and preventing the same constraint from being repeatedly activated and deactivated. Controlled by the `expand_factor` parameter on `solve_qp`.

**LPEC-A active set identification:** As an alternative to (or complement of) EXPAND, the solver supports LPEC-A (Oberlin & Wright, 2005, Section 3.3) for predicting the active constraint set before the QP active-set loop. LPEC-A computes a proximity measure `rho_bar` from primal constraint values, dual multipliers, and the stationarity residual, then applies a threshold test `c_ineq_i <= (beta * rho_bar)^sigma` to predict active constraints. Under MFCQ + second-order sufficiency, the prediction is asymptotically exact (including weakly active constraints). Controlled by `active_set_method` on `SLSQP`: `"expand"` (default, no LPEC-A), `"lpeca_init"` (LPEC-A warm-start + EXPAND fallback), or `"lpeca"` (LPEC-A + fixed tolerance, no EXPAND growth). Threshold parameters: `lpeca_sigma` (default 0.9), `lpeca_beta` (default `1/(m_ineq + n + m_eq)`). Optional LP refinement via `mpax.r2HPDHG` when `lpeca_use_lp=True` (requires `mpax` from the `extras` dependency group).

**Multiplier stability:** Wright (SIAM J. Optim., 2002, Theorem 5.3) proved that SQP superlinear convergence requires multiplier stability across iterations. The solver promotes this through: (1) warm-starting the QP active set from the previous iteration, (2) an adaptive EXPAND tolerance tied to the outer KKT residual, (3) adaptive proximal multiplier stabilization (sSQP, active for equality-constrained problems when `proximal_tau > 0`, with `mu = clip(kkt_residual^tau, mu_min, mu_max)` per Wright eq 6.6; controlled by `proximal_tau`, `proximal_mu_min`, and `proximal_mu_max` on `SLSQP`), and (4) alpha-scaled multiplier blending (`lambda_new = lambda_prev + alpha * (lambda_QP - lambda_prev)`) for L-BFGS secant pair consistency when the line search accepts a partial step. Raw QP multipliers are stored in the state for convergence checking. The ceiling `mu_max` (default 0.1) prevents weak equality enforcement when the KKT residual is large (Wright's local theory assumes eta < 1). Setting `proximal_tau = 0` disables sSQP entirely: equality constraints are enforced via direct null-space projection in the QP subproblem instead of the augmented Lagrangian penalty. This avoids the ill-conditioning introduced by the `(1/μ) A_eq^T A_eq` term, which can be beneficial for well-conditioned problems or when cross-platform reproducibility is important.

**L-BFGS reset strategies (VARCHEN-inspired):** The L-BFGS initial Hessian is stored as a per-variable diagonal `B_0 = diag(d)` using component-wise secant scaling (`d_i = |y_i · s_i| / (s_i² + ε)`). The reset chain follows an escalating strategy: (1) **soft reset** (keep most recent pair, drop rest) triggered by `kappa(B0) > 1e6` after each append, or on QP/line-search failure; (2) **identity reset** (`B_0 = I`) after `qp_failure_patience` (default 3) consecutive QP failures or `ls_failure_patience` (default 3) consecutive line search failures. The soft reset (VARCHEN Algorithm 1, Step 7) preserves the most relevant curvature pair, avoiding the catastrophic information loss from the old SNOPT diagonal/identity reset chain.

**QP failure recovery:** When the QP fails to converge, the solver (1) gates penalty parameter updates so unreliable multipliers cannot permanently inflate `rho`, and (2) falls back to a **projected steepest descent direction** `P(-grad_f)` where P projects onto `null(J_eq)`. This ensures the fallback direction satisfies `J_eq . d = 0`, preventing catastrophic constraint violation (the unprojected `-grad_f` can have `sum(d) = O(n)` for a simplex constraint, making the L1 merit DD massively positive and the line search unable to find a step). The QP solver also guards against false convergence: when the active-set loop reaches `max_iter`, the convergence flag is forced to `False` regardless of the EXPAND-relaxed tolerance state. Combined with the L-BFGS soft reset, this prevents the cascade where one QP failure leads to permanent stagnation.

**L1 merit directional derivative:** The backtracking line search uses the proper L1 merit directional derivative `D_phi = grad_f . d + rho * sum_i sign(c_eq_i) * (J_eq d)_i - rho * sum_{j: c_ineq_j < 0} (J_ineq d)_j` instead of the simpler `grad_f . d`. When the QP direction satisfies `J_eq d = -c_eq`, this simplifies to `grad_f . d - rho * ||c_eq||_1`, making the Armijo condition easier to satisfy. Constraint Jacobians are passed from the solver to the line search.

**Line search failure recovery:** When the QP converges but the resulting direction is not a descent direction for the L1 merit function (typically due to a poorly conditioned L-BFGS Hessian), the backtracking line search exhausts its iteration budget and returns a tiny step size. The solver tracks consecutive line search failures and applies the escalating L-BFGS reset strategy: soft reset on each failure, identity reset after `ls_failure_patience` (default 3) consecutive failures. Controlled by `ls_failure_patience` on `SLSQP`.

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
