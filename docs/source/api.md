# API Reference

This page contains the API reference for `slsqp-jax`.

## Solver

The main entry point is the {class}`~slsqp_jax.SLSQP` class, which implements the {class}`optimistix.AbstractMinimiser` interface.

```{eval-rst}
.. autoclass:: slsqp_jax.SLSQP
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

## Solver State

The internal state maintained by the solver during optimization.

```{eval-rst}
.. autoclass:: slsqp_jax.SLSQPState
   :members:
   :undoc-members:
   :show-inheritance:
```

## Hessian Approximation

L-BFGS history and Hessian-vector product utilities.

```{eval-rst}
.. autoclass:: slsqp_jax.LBFGSHistory
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: slsqp_jax.lbfgs_init

.. autofunction:: slsqp_jax.lbfgs_append

.. autofunction:: slsqp_jax.lbfgs_hvp
```

## QP Solver

The quadratic programming subproblem solver.

```{eval-rst}
.. autoclass:: slsqp_jax.QPResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: slsqp_jax.solve_qp
```

## Merit Function

Line search with the Han-Powell L1 merit function.

```{eval-rst}
.. autoclass:: slsqp_jax.LineSearchResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: slsqp_jax.backtracking_line_search

.. autofunction:: slsqp_jax.compute_merit

.. autofunction:: slsqp_jax.update_penalty_parameter
```

## SciPy Compatibility

Utilities to convert SciPy-style constraint specifications (dicts, `LinearConstraint`, `NonlinearConstraint`) and a convenience entry point that mirrors `scipy.optimize.minimize`.

```{eval-rst}
.. autofunction:: slsqp_jax.minimize_like_scipy

.. autofunction:: slsqp_jax.parse_constraints

.. autoclass:: slsqp_jax.ParsedConstraints
   :members:
   :undoc-members:
   :show-inheritance:
```

### Non-standard `NonlinearConstraint.hessp` extension

`scipy.optimize.NonlinearConstraint` does **not** ship a `hessp` attribute, does **not** accept one in `__init__`, and SciPy's own solvers never read one. The SLSQP-JAX compat layer nevertheless honours a user-attached `hessp` attribute on a `NonlinearConstraint`: if present and callable, it is used as the per-component constraint Hessian-vector product with **precedence over** `hess`.

This is a deliberate, unorthodox extension that exists so users can avoid materialising a dense `(n, n)` constraint Hessian (which SciPy's `hess(x, v)` convention forces) when all SLSQP-JAX actually needs is the HVP stack.

The expected signature is

```python
hessp(x, p) -> Array of shape (m, n)
```

where `x` is the current iterate, `p` is the direction vector, `m` is the number of components of the constraint, and row `i` of the returned array equals `(d^2 c_i / dx^2)(x) @ p`.

Attach it after construction:

```python
nlc = NonlinearConstraint(fun, lb, ub, jac=jac_fn)
nlc.hessp = my_hessp  # non-standard; ignored by SciPy, consumed here
```

Precedence rules:

1. If `hessp` is present and callable, it wins over `hess`.
2. If `hessp` is present but not callable (e.g. a sentinel string like `"2-point"`), it is ignored and `hess` is used if callable.
3. Validation is limited to positional-parameter arity via `inspect.signature`; shape/dtype mismatches surface as JAX errors on first use. Callables whose signature cannot be introspected (e.g. some C-level builtins) are accepted silently.

## Type Definitions

Type aliases used throughout the library.

```{eval-rst}
.. automodule:: slsqp_jax.types
   :members:
   :undoc-members:
   :show-inheritance:
```

## Utilities

Helper functions.

```{eval-rst}
.. automodule:: slsqp_jax.utils
   :members:
   :undoc-members:
   :show-inheritance:
```
