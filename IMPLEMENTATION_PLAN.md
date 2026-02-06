# SLSQP JAX Implementation Specs

## 1. Mathematical Structure

We are implementing a Sequential Least Squares Programming (SLSQP) solver in pure JAX.

### Problem Formulation
```
minimize    f(x)
subject to  c_eq(x) = 0        (m_eq equality constraints)
            c_ineq(x) >= 0     (m_ineq inequality constraints)
            lb <= x <= ub      (box constraints)
```

### Algorithm Overview
* **Outer Loop:** At each iteration k, approximate the nonlinear problem with a Quadratic Program (QP).
* **Inner Loop (QP Solver):** Solve the QP subproblem to find search direction d:
  ```
  minimize    (1/2) d^T B_k d + g_k^T d
  subject to  A_eq d + c_eq(x_k) = 0
              A_ineq d + c_ineq(x_k) >= 0
  ```
  where:
  - B_k = approximate Hessian of the Lagrangian
  - g_k = gradient of objective at x_k
  - A_eq, A_ineq = Jacobians of constraints at x_k

* **Line Search:** Use L1-merit function (Han-Powell) to accept/reject steps.
* **Hessian Update:** Use damped BFGS to update B_k while maintaining positive definiteness.

---

## 2. Optimistix Architecture Map

We will extend `optimistix.AbstractMinimiser` following the library's patterns.

### File Structure
```
slsqp_jax/
├── __init__.py          # Public API exports
├── solver.py            # SLSQP solver (AbstractMinimiser subclass)
├── qp_solver.py         # QP subproblem solver (Active Set method)
├── merit.py             # L1-merit function and line search
├── hessian.py           # BFGS Hessian approximation updates
└── types.py             # Type definitions and custom types

tests/
├── __init__.py
├── test_comparison.py   # Compare against scipy.optimize.minimize(method='SLSQP')
├── test_qp_solver.py    # Unit tests for QP subproblem
└── test_gradients.py    # Verify gradients match jax.grad
```

---

## 3. Core Classes

### A. `SLSQPState` (eqx.Module)

The solver state must be a JAX PyTree for compatibility with JAX transformations.

```python
class SLSQPState(eqx.Module):
    """State for the SLSQP solver."""

    # Current iteration data
    step_count: Int[Array, ""]           # Current iteration number

    # Function values and gradients
    f_val: Float[Array, ""]              # f(x_k) - objective value
    grad: Float[Array, " n"]             # ∇f(x_k) - gradient of objective

    # Constraint information
    eq_val: Float[Array, " m_eq"]        # c_eq(x_k) - equality constraint values
    ineq_val: Float[Array, " m_ineq"]    # c_ineq(x_k) - inequality constraint values
    eq_jac: Float[Array, "m_eq n"]       # Jacobian of equality constraints
    ineq_jac: Float[Array, "m_ineq n"]   # Jacobian of inequality constraints

    # Hessian approximation
    hessian_approx: Float[Array, "n n"]  # B_k - approximate Hessian of Lagrangian

    # Lagrange multipliers (from QP solution)
    multipliers_eq: Float[Array, " m_eq"]      # λ - equality multipliers
    multipliers_ineq: Float[Array, " m_ineq"]  # μ - inequality multipliers

    # Previous step information (for BFGS update)
    prev_x: Float[Array, " n"]           # x_{k-1}
    prev_grad_lagrangian: Float[Array, " n"]  # ∇L(x_{k-1}, λ_{k-1}, μ_{k-1})

    # Convergence tracking
    first_step: Bool[Array, ""]          # True if this is the first iteration
```

### B. `QPState` (eqx.Module)

State for the QP subproblem solver (Active Set method).

```python
class QPState(eqx.Module):
    """State for the QP Active Set solver."""

    d: Float[Array, " n"]                # Current search direction
    active_set: Bool[Array, " m_ineq"]   # Which inequality constraints are active
    multipliers: Float[Array, " m"]      # KKT multipliers (m = m_eq + active inequalities)
    iteration: Int[Array, ""]            # Inner iteration count
    converged: Bool[Array, ""]           # Whether QP has converged
```

### C. `SLSQP` (AbstractMinimiser subclass)

```python
class SLSQP(optx.AbstractMinimiser[Float[Array, " n"], SLSQPState, Aux]):
    """SLSQP minimizer using Sequential Quadratic Programming."""

    rtol: float = 1e-6
    atol: float = 1e-6
    max_steps: int = 100

    # Constraint functions (static fields)
    eq_constraint_fn: Optional[Callable] = eqx.field(static=True, default=None)
    ineq_constraint_fn: Optional[Callable] = eqx.field(static=True, default=None)

    def init(self, fn, y, args, options, f_struct, aux_struct, tags) -> SLSQPState:
        """Initialize solver state with first function/gradient evaluation."""
        ...

    def step(self, fn, y, args, options, state, tags) -> tuple[Y, SLSQPState, Aux]:
        """Perform one SLSQP iteration:
        1. Solve QP subproblem for search direction
        2. Line search with merit function
        3. Update Hessian approximation (BFGS)
        """
        ...

    def terminate(self, fn, y, args, options, state, tags) -> tuple[Bool, RESULTS]:
        """Check KKT conditions for convergence."""
        ...
```

---

## 4. QP Subproblem Solver

### Active Set Method

For the QP subproblem, we implement a simplified Active Set method suitable for GPU:

```python
def solve_qp(
    H: Float[Array, "n n"],      # Hessian (positive definite)
    g: Float[Array, " n"],        # Gradient
    A_eq: Float[Array, "m_eq n"], # Equality constraint matrix
    b_eq: Float[Array, " m_eq"],  # Equality RHS
    A_ineq: Float[Array, "m_ineq n"],  # Inequality constraint matrix
    b_ineq: Float[Array, " m_ineq"],   # Inequality RHS
    max_iter: int = 100,
) -> tuple[Float[Array, " n"], Float[Array, " m"]]:
    """
    Solve: min (1/2) d^T H d + g^T d
           s.t. A_eq d = b_eq
                A_ineq d >= b_ineq

    Returns (d, multipliers).
    """
```

### Algorithm Steps
1. **Initialize:** Start with d=0 and estimate initial active set
2. **Solve EQP:** Solve equality-constrained QP with current active set
3. **Check Feasibility:** If d violates inactive constraints, add most violated to active set
4. **Check Multipliers:** If any active inequality has negative multiplier, drop it
5. **Iterate:** Use `jax.lax.while_loop` until convergence

---

## 5. Merit Function and Line Search

### Han-Powell L1 Merit Function

```python
def merit_function(
    x: Float[Array, " n"],
    f_val: Float[Array, ""],
    eq_val: Float[Array, " m_eq"],
    ineq_val: Float[Array, " m_ineq"],
    penalty: Float[Array, ""],
) -> Float[Array, ""]:
    """
    φ(x; ρ) = f(x) + ρ * (‖c_eq(x)‖_1 + ‖max(0, -c_ineq(x))‖_1)
    """
    eq_violation = jnp.sum(jnp.abs(eq_val))
    ineq_violation = jnp.sum(jnp.maximum(0.0, -ineq_val))
    return f_val + penalty * (eq_violation + ineq_violation)
```

### Backtracking Line Search

Use Armijo condition with the merit function to find step size α.

---

## 6. BFGS Hessian Update

### Damped BFGS Update

```python
def bfgs_update(
    B: Float[Array, "n n"],       # Current Hessian approx
    s: Float[Array, " n"],        # Step: x_{k+1} - x_k
    y: Float[Array, " n"],        # Gradient diff: ∇L_{k+1} - ∇L_k
    damping: float = 0.2,
) -> Float[Array, "n n"]:
    """
    Damped BFGS update to maintain positive definiteness.

    Uses Powell's damping when s^T y is too small.
    """
```

---

## 7. GPU Constraints & JAX Compatibility

### Requirements
* **No Python loops:** Use `jax.lax.while_loop`, `jax.lax.scan`, `jax.lax.cond`
* **Static Shapes:** Number of constraints must be known at compile time
* **Pure Functions:** All logic must be side-effect free
* **Batching:** Design to support `jax.vmap` for solving multiple problems in parallel

### Type Annotations
All functions must use `jaxtyping` with `beartype`:
```python
from jaxtyping import Array, Float, Int, Bool, jaxtyped
from beartype import beartype

@jaxtyped(typechecker=beartype)
def my_function(x: Float[Array, " n"]) -> Float[Array, ""]:
    ...
```

---

## 8. Testing Strategy

### Comparison Tests (`test_comparison.py`)
1. **Unconstrained Rosenbrock:** Compare final x with scipy.optimize.minimize
2. **Equality Constrained:** Sphere with linear equality constraint
3. **Inequality Constrained:** Sphere with linear inequality constraints
4. **Mixed Constraints:** Combination of equality and inequality

### Gradient Tests (`test_gradients.py`)
1. Verify `jax.grad` of objective matches internal gradient computation
2. Verify constraint Jacobians match `jax.jacfwd`

### QP Solver Tests (`test_qp_solver.py`)
1. Simple unconstrained QP
2. QP with equality constraints only
3. QP with inequality constraints only
4. QP with mixed constraints

---

## 9. Implementation Phases

### Phase 1: Core Infrastructure ✅
- [x] Define type annotations and base classes
- [x] Implement `SLSQPState` and `QPState`
- [x] Scaffold `SLSQP` class with init/step/terminate stubs

### Phase 2: QP Subproblem ✅
- [x] Implement equality-constrained QP solver (KKT method)
- [x] Implement active set method for inequality constraints
- [x] Add unit tests for QP solver (13 tests)

### Phase 3: Outer Loop ✅
- [x] Implement merit function and line search (`merit.py`)
- [x] Implement BFGS Hessian update (`hessian.py`)
- [x] Wire everything together in `SLSQP.step()`
- [x] Add `postprocess()` and `norm` attributes for Optimistix compatibility

### Phase 4: Testing & Validation ✅
- [x] Compare against SciPy on standard test problems (37 tests passing)
- [x] Test convergence properties
- [ ] Benchmark GPU performance (future work)
- [ ] Add vmap support for batched optimization (future work)
