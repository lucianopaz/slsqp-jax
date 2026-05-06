# API Reference

The `slsqp_jax` package is organised into one top-level facade module and three
subpackages.  Every page below is generated directly from the source
docstrings; this index just mirrors the package layout so you can navigate by
module.

```text
slsqp_jax/
├── compat            (SciPy-compatible entry point + constraint parser)
├── config            (nested *Config dataclasses passed to SLSQP)
├── hessian           (L-BFGS history, HVPs, Lagrangian gradient helpers)
├── lpeca             (LPEC-A active-set identification)
├── merit             (L1 merit + backtracking line search)
├── results           (granular RESULTS enumeration)
├── state             (SLSQPState / SLSQPDiagnostics / QPResult / …)
├── types             (Vector, Scalar, GradFn, JacobianFn, HVPFn, …)
├── utils             (small shared helpers)
├── inner/            (pluggable inner equality-constrained QP solvers)
│   ├── base, masking, krylov, projected_cg
│   └── cholesky, craig, minres_qlp, hr_stcg
├── qp/               (QP subproblem layer)
│   ├── api, active_set
│   └── proximal, direct, inequality, bound_fixing
└── slsqp/            (outer SLSQP loop)
    ├── solver
    └── bounds, derivatives, hvp, preconditioner, termination, verbose
```

## Top-level package

```{toctree}
:maxdepth: 1

slsqp_jax
config
state
results
hessian
merit
lpeca
compat
types
utils
```

## Subpackages

```{toctree}
:maxdepth: 1

inner/index
qp/index
slsqp/index
```
