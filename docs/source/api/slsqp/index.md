# `slsqp_jax.slsqp`

Outer SLSQP loop.  [`solver`](solver.md) defines the `SLSQP` Optimistix
`AbstractMinimiser`; it delegates derivative wiring, bound handling,
preconditioning, Lagrangian HVPs, termination classification, and verbose
printing to the helper modules below.

## Modules

```{toctree}
:maxdepth: 1

solver
bounds
derivatives
hvp
preconditioner
termination
verbose
```
