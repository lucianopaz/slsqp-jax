"""Importable entry point for spawn-based worker processes.

``multiprocessing`` with the ``spawn`` start method needs a target that can
be imported by name in the child. The worker is persistent: it pays the
``jax`` / ``sif2jax`` import once, then serves tasks from a queue until it
receives ``None``. Isolation against hangs and crashes is provided by the
parent, which kills and respawns a worker whose task overran its timeout.
"""

from __future__ import annotations

import traceback
from typing import Any


def worker_loop(task_queue: Any, result_queue: Any) -> None:
    """Serve benchmark tasks until ``None`` is received.

    Messages sent back on ``result_queue`` are ``(task_id, kind, payload)``
    with ``kind`` in ``{"phase", "ok", "error"}``.

    Parameters
    ----------
    task_queue
        Queue of ``(task_id, TaskSpec)`` items, terminated by ``None``.
    result_queue
        Queue receiving progress and completion messages.
    """
    import jax

    jax.config.update("jax_enable_x64", True)

    from .worker import run_task_inprocess

    result_queue.put((None, "ready", None))
    while True:
        item = task_queue.get()
        if item is None:
            break
        task_id, spec = item

        def report(kind: str, payload: Any, _task_id=task_id) -> None:
            result_queue.put((_task_id, kind, payload))

        try:
            result = run_task_inprocess(spec, report)
            result_queue.put((task_id, "ok", result))
        except BaseException:  # noqa: BLE001 - always report back to the parent
            result_queue.put((task_id, "error", traceback.format_exc()))
        finally:
            # Keep the executable cache from growing across tasks.
            jax.clear_caches()
