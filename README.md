# slsqp-jax

[![Build](https://github.com/lucianopaz/slsqp-jax/actions/workflows/test.yml/badge.svg)](https://github.com/lucianopaz/slsqp-jax/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A pure-JAX implementation of the **SLSQP** (Sequential Least Squares Quadratic Programming) algorithm for constrained nonlinear optimization, designed for **moderate to large decision spaces** (5,000–50,000 variables). All linear algebra is performed through JAX, so the solver runs natively on CPU, GPU, and TPU, and is fully compatible with `jax.jit`, `jax.vmap`, and `jax.grad`.

The `SLSQP` solver is built on top of [Optimistix](https://github.com/patrick-kidger/optimistix), a JAX library for nonlinear solvers. It implements the `optimistix.AbstractMinimiser` interface, so you run it through the standard [`optimistix.minimise`](https://docs.kidger.site/optimistix/api/minimise/) entry point — no manual iteration loop required.

## Installation

The package is not yet published to PyPI. Install directly from GitHub:

**pip**

```bash
pip install "slsqp-jax @ git+https://github.com/lucianopaz/slsqp-jax.git"
```

**uv**

```bash
uv add "slsqp-jax @ git+https://github.com/lucianopaz/slsqp-jax.git"
```

**pixi**

```bash
pixi add --pypi "slsqp-jax @ git+https://github.com/lucianopaz/slsqp-jax.git"
```

## Usage

### Basic: objective and constraints

Define an objective function with signature `(x, args) -> (scalar, aux)` and optional constraint functions with signature `(x, args) -> array`. Equality constraints must satisfy `c_eq(x) = 0` and inequality constraints must satisfy `c_ineq(x) >= 0`.

Then create an `SLSQP` solver and pass it to `optimistix.minimise`:

```python
import jax.numpy as jnp
import optimistix as optx
from slsqp_jax import SLSQP

# Objective: minimize x^2 + y^2
def objective(x, args):
    return jnp.sum(x**2), None

# Equality constraint: x + y = 1
def eq_constraint(x, args):
    return jnp.array([x[0] + x[1] - 1.0])

# Inequality constraint: x >= 0.2
def ineq_constraint(x, args):
    return jnp.array([x[0] - 0.2])

solver = SLSQP(
    eq_constraint_fn=eq_constraint,
    n_eq_constraints=1,
    ineq_constraint_fn=ineq_constraint,
    n_ineq_constraints=1,
    rtol=1e-8,
    atol=1e-8,
)

x0 = jnp.array([0.5, 0.5])
sol = optx.minimise(objective, solver, x0, has_aux=True, max_steps=100)

print(sol.value)   # [0.2, 0.8]
print(sol.result)  # RESULTS.successful
```

The returned `optimistix.Solution` object contains:

- `sol.value` — the optimal point.
- `sol.result` — a status code (`RESULTS.successful` or an error).
- `sol.aux` — any auxiliary data returned by the objective.
- `sol.stats` — solver statistics (e.g. number of steps taken).
- `sol.state` — the final internal solver state.

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

For problems where you have access to exact second-order information, you can supply Hessian-vector product (HVP) functions. This switches the solver from the default L-BFGS approximation to **exact Hessian mode**, which typically converges in fewer iterations:

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

## Algorithm

### Overview

Each SLSQP iteration performs four steps:

1. **QP subproblem**: Construct a quadratic approximation of the objective and linearise the constraints around the current point. Solve the resulting QP to obtain a search direction.
2. **Line search**: Use a Han-Powell L1 merit function $\phi(x;\rho) = f(x) + \rho (\lVert c_{\text{eq}}\rVert_1 + \lVert\max(0, -c_{\text{ineq}})\rVert_1)$ with backtracking Armijo conditions to determine the step size.
3. **Accept step**: Update the iterate $x_{k+1} = x_k + \alpha d_k$.
4. **Hessian update**: Append the new curvature pair $(s, y)$ to the L-BFGS history (or skip if using exact HVPs).

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

Powell's damping is applied to each curvature pair before storage to ensure positive definiteness, which is essential for constrained problems where the standard curvature condition $s^T y > 0$ can fail.

### Scaling considerations: why projected CG over dense KKT

Classical SLSQP solves the QP subproblem by forming and factorising the $(n + m) \times (n + m)$ dense KKT system at $O(n^3)$ cost. This implementation instead uses **projected conjugate gradient** (CG) inside an active-set loop:

1. **Projection**: For active constraints with matrix $A$ ($m_{\text{active}} \times n$), define the null-space projector $P(v) = v - A^T (A A^T)^{-1} A v$. The $A A^T$ system is only $m_{\text{active}} \times m_{\text{active}}$ (tiny, since $m << n$) and is solved directly.
2. **CG in null space**: Run conjugate gradient on the projected system, where each iteration requires one HVP ($O(kn)$ for L-BFGS) and one projection ($O(mn)$).
3. **Active-set outer loop**: Add the most violated inequality or drop the most negative multiplier until KKT conditions are satisfied.

Total cost per QP solve: $O(n \cdot k \cdot t)$, where $t$ is the number of CG iterations (typically $t << n$). Compared to $O(n^3)$ for the dense approach, this is orders of magnitude faster for $n > 1000$.

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

## License

MIT
