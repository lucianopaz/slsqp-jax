# slsqp-jax

[![Build](https://github.com/lucianopaz/slsqp-jax/actions/workflows/test.yml/badge.svg)](https://github.com/lucianopaz/slsqp-jax/actions/workflows/test.yml)
[![Documentation](https://readthedocs.org/projects/slsqp-jax/badge/?version=latest)](https://slsqp-jax.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/lucianopaz/slsqp-jax/graph/badge.svg?token=K6Y9JBL6F2)](https://codecov.io/gh/lucianopaz/slsqp-jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI - Version](https://img.shields.io/pypi/v/slsqp-jax)](https://pypi.org/project/slsqp-jax/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lucianopaz/slsqp-jax/blob/main/benchmark_cpu_gpu.ipynb)

A pure-JAX implementation of the **SLSQP** (Sequential Least Squares Quadratic Programming) algorithm for constrained nonlinear optimization, designed for **moderate to large decision spaces** (5,000–50,000 variables). All linear algebra is performed through JAX, so the solver runs natively on CPU, GPU, and TPU, and is fully compatible with `jax.jit`, `jax.vmap`, and `jax.grad`.

The `SLSQP` solver is built on top of [Optimistix](https://github.com/patrick-kidger/optimistix), a JAX library for nonlinear solvers. It implements the `optimistix.AbstractMinimiser` interface, so you run it through the standard [`optimistix.minimise`](https://docs.kidger.site/optimistix/api/minimise/) entry point — no manual iteration loop required.

## Installation

You can install the package from PyPI using any standard method:

**pip**

```bash
pip install slsqp-jax
```

**uv**

```bash
uv add "slsqp-jax"
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

**Frozen Hessian in the QP subproblem**: The QP inner loop always uses a frozen L-BFGS approximation to the Lagrangian Hessian, even when exact HVPs are available. The exact HVP is called only **once per main iteration** (to probe along the step direction and produce an exact secant pair for the L-BFGS update). This design ensures (1) the QP subproblem sees a truly constant quadratic model, and (2) expensive HVP evaluations are not repeated thousands of times inside the projected CG solver.

### Box constraints (bounds)

You can specify simple lower and upper bounds on decision variables using the `bounds` parameter:

```python
import jax.numpy as jnp

bounds = jnp.array([
    [0.0, 1.0],      # 0 <= x_0 <= 1
    [-jnp.inf, 5.0], # x_1 <= 5 (no lower bound)
    [0.0, jnp.inf],  # x_2 >= 0 (no upper bound)
])

solver = SLSQP(bounds=bounds)
```

Bounds play a **dual role** in the solver, following the projected-SQP methodology (Heinkenschloss & Ridzal, *Projected Sequential Quadratic Programming Methods*, SIAM J. Optim., 1996):

1. **QP inequality constraints** — inside the QP subproblem, bounds are linearised as ordinary inequality constraints so the search direction is aware of the feasible box.
2. **Hard projection** — after every line search step (and at initialisation), the iterate is *projected* (clipped) onto the feasible box. This guarantees that the objective and constraint functions are **never evaluated outside the bounds**, which is critical when those functions are undefined or ill-conditioned outside the box (e.g. a log-likelihood with positivity constraints on its parameters).

## Algorithm

### Overview

Each SLSQP iteration performs four steps:

1. **QP subproblem**: Construct a quadratic approximation of the objective using the frozen L-BFGS Hessian and linearise the constraints around the current point. Solve the resulting QP to obtain a search direction.
2. **Line search**: Use a Han-Powell L1 merit function $\phi(x;\rho) = f(x) + \rho (\lVert c_{\text{eq}}\rVert_1 + \lVert\max(0, -c_{\text{ineq}})\rVert_1)$ with backtracking Armijo conditions to determine the step size.
3. **Accept step**: Update the iterate $x_{k+1} = x_k + \alpha d_k$.
4. **Hessian update**: Append the new curvature pair $(s, y)$ to the L-BFGS history, where $y$ is either an exact HVP probe $\nabla^2 L(x_k) s$ (if HVP functions are provided) or the gradient difference $\nabla L(x_{k+1}) - \nabla L(x_k)$.

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

### QP anti-cycling: the EXPAND procedure

The QP subproblem is solved by a primal active-set method that adds or removes inequality constraints one at a time. When the problem is **degenerate** — multiple constraints pass through the same vertex or have tied violation/multiplier values — the active-set loop can *cycle*: iteration $i$ activates a constraint, iteration $i+1$ drops it, iteration $i+2$ re-activates it, and so on. The QP then exhausts its iteration budget without converging, producing a poor search direction.

This implementation uses the **EXPAND** procedure (Gill, Murray, Saunders & Wright, *Mathematical Programming* 45, 1989) to break such cycles. The idea is simple: instead of a fixed feasibility tolerance, the active-set loop maintains a *working tolerance* that grows monotonically:

$$
\delta_k = \texttt{tol} + k \cdot \tau, \qquad \tau = \frac{\texttt{tol} \cdot \texttt{expand\_factor}}{\texttt{max\_iter}}
$$

At each active-set iteration $k$:

- A constraint is considered *violated* only if its residual is below $-\delta_k$ (progressively stricter threshold for activation).
- A multiplier is considered *negative* only if it is below $-\delta_k$ (progressively stricter threshold for deactivation).

Because $\delta_k$ increases at every step, marginally active or marginally infeasible constraints that cause cycling are gradually excluded, breaking the degeneracy. With the default settings (`expand_factor=1.0`, `tol=1e-8`, `max_iter=100`), the tolerance doubles from `tol` to `2·tol` over the full iteration budget — conservative enough to preserve solution quality while reliably eliminating cycles.

EXPAND is the standard anti-cycling technique used in production solvers (MINOS, SNOPT, SQOPT) and is backed by a convergence guarantee: strict objective decrease within each expanding sequence. The `expand_factor` parameter on `solve_qp` controls the growth rate; set it to `0.0` to disable expansion entirely.

### Multiplier stability and convergence rate

Wright (*Properties of the Log-Barrier Function on Degenerate Nonlinear Programs*, SIAM J. Optim., 2002, Theorem 5.3) proved that the convergence rate of SQP methods depends critically on **multiplier stability**: the rate at which Lagrange multipliers change between successive iterations. Specifically, superlinear convergence requires $\lVert \lambda_k - \lambda_{k+1} \rVert = O(\delta)$ where $\delta$ is the step norm. When the QP subproblem is degenerate, the multiplier solution may not be unique, and arbitrary choices across iterations can destroy this stability — even though the primal iterates converge.

This implementation promotes multiplier stability through three mechanisms:

1. **Active-set warm-starting**: the final active set from each QP solve is passed as the initial guess for the next outer iteration's QP subproblem. Because the active set carries implicit information about which multipliers are nonzero, reusing it across iterations biases the QP toward consistent multiplier selections. This is the same strategy used by production SQP codes such as SNOPT (Gill, Murray & Saunders, 2005).

2. **Alpha-scaled multiplier blending**: when the line search accepts a partial step ($\alpha < 1$), the QP multipliers correspond to the full-step linearization, not the actual accepted step. The solver blends the QP multipliers with the previous iteration's multipliers proportional to the step size: $\lambda_{k+1} = \lambda_k + \alpha (\lambda_{\text{QP}} - \lambda_k)$. The blended multipliers are used for the L-BFGS secant pair computation (ensuring consistency between the primal step and the curvature update), while the raw QP multipliers are stored for convergence checking (since they represent the best dual estimate at the current linearization point).

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

Proximal stabilization is **always active** for equality-constrained problems. The proximal parameter $\mu$ is computed adaptively at each iteration following Wright (2002, eq 6.6):

$$
\mu_k = \operatorname{clip}\!\bigl(\lVert \nabla_x L_k \rVert^{\tau},\;\mu_{\min},\;\mu_{\max}\bigr)
$$

where $\tau \in (0, 1)$ is the exponent (default 0.5), $\mu_{\min}$ is a floor (default `atol`, typically $10^{-6}$), and $\mu_{\max}$ is a ceiling (default 0.1). Wright's local convergence theory assumes the KKT residual is below 1; the ceiling handles the regime where the residual is large (far from the solution) by ensuring the proximal weight $1/\mu \geq 1/\mu_{\max}$ — preventing weak equality enforcement that would sabotage the L1 merit function descent. As the solver converges, $\mu$ shrinks below the ceiling, tightening equality enforcement while the floor prevents $1/\mu$ from exploding. Wright's Theorem 6.1 guarantees superlinear convergence when $\tau < 1$.

```python
solver = SLSQP(
    eq_constraint_fn=eq_constraint,
    n_eq_constraints=1,
    proximal_tau=0.5,       # Exponent for adaptive mu (must be in (0,1))
    proximal_mu_min=None,   # Floor on mu; None defaults to atol
    proximal_mu_max=0.1,    # Ceiling on mu; keeps 1/mu >= 10
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

For projected (constrained) CG, the preconditioner is applied as $z = P(M(P(r)))$, where $P$ is the null-space projector. The extra projection ensures the preconditioned direction stays in the feasible subspace (Nocedal & Wright, Chapter 16).

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

**Adaptive CG tolerance (Eisenstat-Walker).** When `adaptive_cg_tol=True`, the CG convergence tolerance is adapted based on the outer KKT residual: $\eta_k = \min(0.1,\, \max(\texttt{atol},\, \lVert \nabla_x L \rVert))$. This avoids over-solving early QPs (when the outer iterate is far from optimal) and tightens the tolerance as convergence proceeds (Eisenstat & Walker, *SIAM J. Sci. Comput.*, 17(1), 1996). This is off by default to preserve baseline convergence behavior.

**Why not rational CG?** We evaluated the rational conjugate gradient method (Kindermann & Zellinger, arXiv:2306.03670, 2023), which alternates CG steps with Tikhonov regularization steps in a mixed rational Krylov space. It is designed for ill-posed inverse problems with compact operators, not finite-dimensional positive definite QPs. Each rational step requires an inner solve of $(B + \alpha I)^{-1} v$, the regularization parameters need spectrum knowledge, and adapting the method to null-space projection is non-trivial. Standard PCG is simpler, cheaper, and directly addresses the ill-conditioning source.

### L-BFGS diagonal reset

The L-BFGS initial Hessian is stored as a per-variable diagonal $B_0 = \text{diag}(d)$ rather than a scalar $\gamma I$. During normal operation the diagonal is uniform ($d = \gamma \mathbf{1}$), but when the QP solver fails to converge, the solver performs an **SNOPT-style diagonal reset** (Gill, Murray & Saunders, *SIAM Review*, 47(1), 2005, Section 3.3):

1. Extract $\text{diag}(B_k)$ from the current L-BFGS compact representation in $O(k^2 n)$.
2. Discard all stored $(s, y)$ pairs.
3. Restart with $B_0 = \text{diag}(\text{clip}(\text{diag}(B_k),\, 10^{-3},\, 100))$.

This preserves per-variable curvature information across the reset, preventing the "everything is flat" effect that occurs when the scalar $\gamma$ becomes very small and gets frozen by tiny steps. After the reset, the first successful L-BFGS append returns the diagonal to uniform scaling.

**Escalating identity reset.** When the QP fails repeatedly, the SNOPT diagonal reset can re-extract the same problematic diagonal, perpetuating an ill-conditioning cycle. To break this deadlock, the solver tracks consecutive QP failures and, after `qp_failure_patience` consecutive failures (default 3), performs a hard identity reset: $B_0 = I$, discarding all stored pairs and the diagonal. This allows the L-BFGS to rebuild curvature from scratch.

```python
solver = SLSQP(
    eq_constraint_fn=eq_constraint,
    n_eq_constraints=1,
    qp_failure_patience=3,  # Identity reset after 3 consecutive QP failures
)
```

### QP failure recovery

When the QP subproblem fails to converge (exhausts its iteration budget), the solver applies two safeguards to prevent a cascade of failures:

1. **Penalty gating**: the penalty parameter $\rho$ is only updated from QP multipliers when the QP converged. Failed QPs can produce unreliable multipliers that would permanently inflate $\rho$ (since it is monotone non-decreasing), making future line searches harder.
2. **Steepest descent fallback**: the search direction falls back to the negative gradient $-\nabla f(x)$ instead of the unconverged QP direction.

**QP false convergence guard.** The EXPAND procedure's growing tolerance can cause the active-set loop to report convergence on its final iteration under relaxed tolerances, even though the QP did not truly converge at the base tolerance. To prevent this, the QP solver explicitly overrides the convergence flag to `False` whenever the iteration count reaches `max_iter`. This ensures the outer solver triggers L-BFGS resets and steepest descent fallback appropriately.

Combined with the L-BFGS diagonal reset (above), these mechanisms allow the solver to recover from QP failures rather than entering a permanent stagnation loop.

### Outer-loop stagnation detection

Even with anti-cycling in the QP, the outer SLSQP loop can fail to make progress — for example, when the problem is infeasible, highly degenerate, or the QP solution is of poor quality. The solver detects this using a **sliding-window x-value comparison**.

A circular buffer of size $W = \lfloor \texttt{max\_steps} / 10 \rfloor$ stores recent iterates. At each step $k \geq W$, the solver compares the current iterate with the one from $W$ steps ago:

$$
\frac{\lVert x_k - x_{k-W} \rVert}{\max(\lVert x_k \rVert,\, 1)} < \texttt{stagnation\_tol}
$$

If this relative change is below `stagnation_tol` (default $10^{-12}$), the solver terminates early with a `nonlinear_divergence` result code rather than running until `max_steps`. This avoids wasting computation on a problem the solver cannot solve.

This approach is more robust than merit-based consecutive-counter detection, which can miss cases where merit improvements are minuscule but technically above the threshold — causing the solver to run to `max_steps` without being flagged as stagnant.

```python
solver = SLSQP(
    eq_constraint_fn=eq_constraint,
    n_eq_constraints=1,
    stagnation_tol=1e-12,    # Minimum relative x-value change over window
)
```

The stagnation flag and last step size are included in `sol.stats` for diagnostics:

```python
sol = optx.minimise(objective, solver, x0, has_aux=True, max_steps=100, throw=False)
print(sol.stats["x_stagnation"])   # False if converged normally
print(sol.stats["last_step_size"]) # Step size from final iteration
```

## License

MIT
