# `slsqp_jax.qp`

QP subproblem layer.  The [`api`](api.md) module routes to one of three
strategies based on the constraint geometry:

* [`proximal`](proximal.md) — equality + (any) inequality with sSQP
  proximal stabilisation.
* [`direct`](direct.md) — equality + (any) inequality with direct
  null-space projection (no proximal term).
* [`inequality`](inequality.md) — inequality only.

All three share the [`active_set`](active_set.md) loop body
(`run_active_set_loop`).  After the QP direction is computed,
[`bound_fixing`](bound_fixing.md) handles box-bound activation /
release in a reduced subspace.

## Modules

```{toctree}
:maxdepth: 1

api
active_set
proximal
direct
inequality
bound_fixing
```
