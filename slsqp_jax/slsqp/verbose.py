"""Verbose-output callbacks for the SLSQP outer loop.

Two callables are exposed:

* :func:`slsqp_verbose` — print one line per SLSQP step, with each
  field formatted by an optional ``fmt_spec`` provided as the third
  tuple element.  Wraps :func:`jax.debug.print`, so it is safe to call
  inside a JAX-traced ``step``.
* :func:`no_verbose` — no-op alternative used when ``verbose=False``
  is requested at construction time.

Both are kept tiny on purpose: the format specifiers are stripped at
class-construction time so that PyTree equality (and therefore
``optimistix`` JIT cache hits) is preserved across runs that differ
only in the verbose printer payload.
"""

from __future__ import annotations

from typing import Any

import jax


def slsqp_verbose(**kwargs: tuple) -> None:
    """Default verbose callback with per-field format specifiers.

    Each kwarg value is either ``(label, value)`` or
    ``(label, value, fmt_spec)``.  The ``fmt_spec`` string (e.g.
    ``".3e"``) is inserted into the ``jax.debug.print`` format
    placeholder.
    """
    string_pieces: list[str] = []
    arg_pieces: list[Any] = []
    for entry in kwargs.values():
        if len(entry) == 3:
            name, value, _fmt = entry
            string_pieces.append(f"{name}: {{:{_fmt}}}")
        else:
            name, value = entry
            string_pieces.append(f"{name}: {{}}")
        arg_pieces.append(value)
    if string_pieces:
        jax.debug.print(", ".join(string_pieces), *arg_pieces)


def no_verbose(**_kwargs: tuple) -> None:
    """No-op verbose callback."""


__all__ = ["no_verbose", "slsqp_verbose"]
