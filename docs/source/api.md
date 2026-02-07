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
