# sqpdax

[![Build](https://github.com/lucianopaz/slsqp-jax/actions/workflows/test.yml/badge.svg)](https://github.com/lucianopaz/slsqp-jax/actions/workflows/test.yml)
[![Documentation](https://readthedocs.org/projects/slsqp-jax/badge/?version=latest)](https://slsqp-jax.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/lucianopaz/slsqp-jax/graph/badge.svg?token=K6Y9JBL6F2)](https://codecov.io/gh/lucianopaz/slsqp-jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI - Version](https://img.shields.io/pypi/v/slsqp-jax)](https://pypi.org/project/slsqp-jax/)
[![Open In Colab: CPU vs GPU](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lucianopaz/slsqp-jax/blob/main/benchmark_cpu_gpu.ipynb)
[![Open In Colab: Solver Configs](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lucianopaz/slsqp-jax/blob/main/benchmark_solver_configs.ipynb)

Welcome to **SQPDAX** (formerly, `slsqp-jax`), a framework to write sequential quadratic programming (SQP) optimisation algorithms in [`jax`](https://docs.jax.dev/en/latest/index.html).

**A BIG REFACTOR IS TAKING PLACE RIGHT NOW**, the last `slsqp-jax` version will be v0.21.1. The refactored version aims to provide a framework to help write custom SQP optimizers that use jax as their backend. It will enable both active set and interior point based algorithms to work, and lay the ground work to use any kind of Lagrangian or augmented Lagrangian continuos optimiser.

The focus will be almost exclusively on matrix free methods (no full Hessian representation, not even sparse), but in the future, support for full Hessians, dense or not, might be included.

## Background

`slsqp-jax` came about as a pure-JAX implementation of **SLSQP**, but then evolved into a very different kind of beast, tuned to handle **moderate to large decision spaces** (5,000-50,000 variables). The way in which it did this was through a combination of low rank approximations to the Hessian and the use of matrix free methods. For details on why I chose to refactor `slsqp-jax`, have a look at the [refactor project](https://github.com/users/lucianopaz/projects/1).

## Installation and transition plan

During the refactor, `sqpdax` will be a module within `slsqp-jax`, so you can install the package as you used to do:

pip

```bash
pip install slsqp-jax
```

uv

```bash
uv add "slsqp-jax"
```

pixi

```bash
pixi add --pypi slsqp-jax
```

and you will then find all of the old `slsqp-jax` code from version v0.21.1, and the new `sqpdax` code under the `sqpdax` subpackage:

```python
from slsqp_jax import sqpdax
```

At the end of the refactor, when a new version will be released, the package will change its name to `sqpdax` and you can expect to import it and its contents as:

```python
import sqpdax
```

At that point, the old `slsqp-jax` code will be kept for a short period under a subpackage as

```python
from sqpdax import slsqp_jax
```

The retainment period will be a minimum of 6 months, and a maximum of 3 minor versions or 1 calendar year, whatever happens first. If more than 3 minor version releases are made before the 6 months have passed, the `slsqp-jax` subpackage will be removed at the end of the 6 months period.

## Usage

This section will be populated as `sqpdax` evolves. During the transition period, you can find `slsqp-jax` usage instructions in the [built documentation pages](https://slsqp-jax.readthedocs.io/en/latest/)

## License

MIT
