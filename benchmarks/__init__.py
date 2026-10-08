"""Benchmark suite running sqpdax minimisers on the sif2jax CUTEst problems.

The package is split into two halves with different import constraints:

- :mod:`benchmarks.problems`, :mod:`benchmarks.catalog`, :mod:`benchmarks.configs`,
  :mod:`benchmarks.metrics` and :mod:`benchmarks.worker` import ``jax``,
  ``sif2jax`` and ``slsqp_jax``; they run the solvers.
- :mod:`benchmarks.results` and :mod:`benchmarks.analysis` are pure
  ``pandas`` / ``numpy`` and are also imported by the marimo notebooks
  (``report.py`` and ``dashboard.py``), which must run under Pyodide.

Run ``python -m benchmarks --help`` for the command-line interface.
"""
