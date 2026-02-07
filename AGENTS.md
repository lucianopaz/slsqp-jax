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

## Verification
* Create a test suite using `pytest`.
* Compare your JAX implementation against `scipy.optimize.minimize(method='SLSQP')`.
* Verify gradients match `jax.grad` and constraints are satisfied to within `1e-6`.
