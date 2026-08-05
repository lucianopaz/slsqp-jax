# JAX Optimization Engineer Agent

## Role
You are an expert Numerical Analyst and JAX Engineer. Your goal is to reimplement the SLSQP (Sequential Least Squares Programming) algorithm in pure JAX, designed to run on GPUs.

## Local Development

Local development uses a virtual environment managed by `uv`.

### Setup and Dependencies
```bash
# Install or update all dependencies (including dev extras)
uv sync --all-extras
```

### Running Tests
While iterating fixes, it's highly recommended to run tests excluding slow ones

```bash
# Run all tests
uv run pytest

# Run tests excluding slow ones
uv run pytest -m "not slow"

# Run specific test file or class
uv run pytest tests/test_slsqp.py::TestSLSQPBoxConstraints -v
```

### Linting and Formatting
The project uses `prek` to run pre-commit hooks for static type checks and formatting.

```bash
# Run all pre-commit hooks (linting, formatting, type checks)
uv run prek run
```

### Writing tests

Common best practices that you should enforce:

- All modules should have unit tests written for their public functions and classes.
- All unit tests must be written using pytest.
- It's standard practice to use fixtures for common setup and teardown.
- Few parametrized tests are preferred to many tests that focus on a single aspect of the code.
- The unit test code should be generic enough to be applicable to many parametrizations, and not simply use if-else or match cases to select a section of code to run.
- Only when it is very hard to write a single generic-parametrized test should a separate unit test function be written.

#### Test folder convention

Tests for new sub-packages go under `tests/<subpackage>/` (e.g. `tests/diagnostics/` for `slsqp_jax/diagnostics/`) with their own `__init__.py` and `conftest.py` for sub-package-specific fixtures. The root `tests/conftest.py` still applies via fixture inheritance through the package hierarchy. Existing flat tests at `tests/test_*.py` are NOT migrated retroactively — only *new* sub-package code follows the nested layout.

### Documentation

- All public functions and methods should document their signature using numpy-style docstring
- The docs will be built using sphinx, so cross references and references to outside packages must follow the sphinx pattern
- Docstrings must be written in markdown, not in RST. This includes code and math blocks
- Example code in the docstrings will be tested along with the unit tests using pytest doctest. If a module or snippet of code is automatically needed, it can be made available through an autouse fixture.
