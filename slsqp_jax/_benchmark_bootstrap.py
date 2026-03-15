"""Minimal bootstrap for spawn-based multiprocessing with cloudpickle.

This module exists so that ``multiprocessing.Process(target=...)`` can
reference an importable function.  The actual benchmark logic lives in
closures serialised with cloudpickle by the notebook.
"""

import pickle


def run_cloudpickled_task(task_bytes, result_queue):
    """Deserialize a cloudpickle'd callable, execute it, send the result."""
    import jax

    jax.config.update("jax_enable_x64", True)

    try:
        task = pickle.loads(task_bytes)
        result = task()
        result_queue.put(("ok", result))
    except Exception:
        import traceback

        result_queue.put(("error", traceback.format_exc()))
