# `slsqp_jax.inner`

Pluggable inner equality-constrained QP solvers and the Krylov primitives that
back them.  The active-set loop in [`slsqp_jax.qp`](../qp/index.md) is agnostic
to which strategy is used: every solver implements the
[`AbstractInnerSolver`](base.md) interface and returns an
`InnerSolveResult(d, multipliers, converged)`.

## Modules

```{toctree}
:maxdepth: 1

base
masking
krylov
projected_cg
cholesky
craig
minres_qlp
hr_stcg
```
