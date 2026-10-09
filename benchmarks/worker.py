"""Execute one ``(problem, y0_iD, config)`` benchmark task in isolation.

Phases inside the child process (each driven through the config's *runner*,
see :mod:`benchmarks.runners` and :mod:`benchmarks.baselines`):

1. **compile** - for sqpdax configs, ``jax.jit`` of ``minimise`` lowered and
   compiled once, with ``max_steps`` a traced ``int32`` so the same
   executable serves every later phase; for the SciPy baselines, one
   evaluation of every jitted problem callback. Timed as ``compile_s``;
   never part of the timing sample.
2. **warmup** - one call with ``max_steps=1`` (``maxiter=1`` for SciPy).
   Recorded as ``warmup_s`` and excluded from timing statistics and from
   the repeat computation.
3. **pilot** - first call with the full step budget. Timed, counted as
   repeat number one, and its solution provides the status, the solver
   statistics and the quality metrics.
4. **repeats** - ``repeats - 1`` further timed calls, where ``repeats`` is
   either the user-provided count or the budget rule in
   :func:`adaptive_repeats`. Skipped (``repeats_skipped`` records why) when
   the pilot did not solve the problem, unless ``repeat_unsolved`` is set.

The parent enforces a wall-clock timeout; the child also tracks the deadline
and stops adding repeats early so a slow-but-finite task still returns.

Worker processes are persistent (``import sif2jax`` alone costs ~10 s), so
:class:`WorkerPool` reuses them across tasks and only kills / respawns a
worker whose task overran its timeout or crashed.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import platform
import queue as queue_mod
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from typing import Any

__all__ = [
    "TaskSpec",
    "WorkerPool",
    "adaptive_repeats",
    "make_runner",
    "run_task",
    "run_task_inprocess",
    "run_tasks",
]

PHASES = ("build", "compile", "warmup", "pilot", "repeats", "done")
"""Phases a task passes through, in order; ``phase`` in a result row is the last one reached."""


@dataclass(frozen=True)
class TaskSpec:
    """Everything needed to run one benchmark task.

    Attributes
    ----------
    problem
        CUTEst problem name.
    y0_iD
        Starting-point selector.
    config
        Key into :data:`benchmarks.configs.CONFIGS`.
    max_steps
        Outer iteration budget for the full runs.
    repeats
        Exact number of timed runs (pilot included), or ``None`` for the
        adaptive rule.
    repeat_budget_s
        Target total time of the timing sample for the adaptive rule.
    max_repeats
        Cap on the adaptive repeat count.
    slow_floor
        Minimum repeats for pilots faster than ``slow_threshold_s``.
    slow_threshold_s
        Pilot duration above which only one timed run is required.
    timeout_s
        Wall-clock budget for the whole task (all phases).
    feas_tol, f_tol
        Feasibility and relative objective-gap tolerances used to decide
        whether the pilot solve counts as *solved*.
    repeat_unsolved
        Time repeats even when the pilot did not solve the problem. Off by
        default: a failed solve is reported from the pilot alone.
    """

    problem: str
    y0_iD: int = 0
    config: str = "asls-pcg"
    max_steps: int = 500
    repeats: int | None = None
    repeat_budget_s: float = 2.0
    max_repeats: int = 1000
    slow_floor: int = 3
    slow_threshold_s: float = 60.0
    timeout_s: float = 300.0
    feas_tol: float = 1e-6
    f_tol: float = 1e-4
    repeat_unsolved: bool = False


def adaptive_repeats(
    pilot_s: float,
    *,
    budget_s: float = 2.0,
    max_repeats: int = 1000,
    slow_floor: int = 3,
    slow_threshold_s: float = 60.0,
) -> int:
    """Number of timed runs (pilot included) for a pilot that took ``pilot_s``.

    ``clamp(ceil(budget_s / pilot_s), floor, max_repeats)`` where ``floor``
    is ``slow_floor`` below ``slow_threshold_s`` and ``1`` above it.

    Parameters
    ----------
    pilot_s
        Duration of the pilot run in seconds.
    budget_s
        Target total duration of the timing sample.
    max_repeats
        Upper bound on the count.
    slow_floor
        Lower bound for pilots faster than ``slow_threshold_s``.
    slow_threshold_s
        Pilot duration from which a single run suffices.

    Returns
    -------
    int
        Total number of timed runs, at least one.

    Examples
    --------
    >>> from benchmarks.worker import adaptive_repeats
    >>> [adaptive_repeats(t) for t in (1e-6, 1e-3, 1e-2, 0.1, 1.0, 30.0, 90.0)]
    [1000, 1000, 200, 20, 3, 3, 1]
    """
    if not math.isfinite(pilot_s) or pilot_s <= 0:
        return max(1, slow_floor)
    floor = slow_floor if pilot_s < slow_threshold_s else 1
    wanted = math.ceil(budget_s / pilot_s)
    return int(max(floor, min(max_repeats, wanted), 1))


def _device_info() -> dict[str, Any]:
    import jax

    dev = jax.devices()[0]
    return {
        "device": dev.platform,
        "device_kind": getattr(dev, "device_kind", dev.platform),
        "jax_version": jax.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
    }


def _scalar_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """Flatten the scalar entries of ``Solution.stats`` as ``stat_<name>`` columns."""
    import numpy as np

    out: dict[str, Any] = {}
    for key, value in stats.items():
        if key == "num_steps":
            continue
        try:
            arr = np.asarray(value)
        except Exception:  # noqa: BLE001 - non-array payloads are skipped
            continue
        if arr.ndim == 0 and arr.dtype.kind in "biuf":
            out[f"stats_{key}"] = arr.item()
    return out


def make_runner(cfg: Any, problem: Any, x0: Any) -> Any:
    """Instantiate the runner for a configuration.

    Parameters
    ----------
    cfg
        A :class:`benchmarks.configs.BenchConfig`.
    problem
        The sqpdax :class:`~slsqp_jax.sqpdax.problem.Problem`.
    x0
        Starting point.

    Returns
    -------
    Runner
        :class:`~benchmarks.runners.SqpdaxRunner` for ``backend="sqpdax"``,
        :class:`~benchmarks.baselines.ScipyRunner` for ``backend="scipy"``.

    Raises
    ------
    ValueError
        For an unknown backend.
    """
    minimiser = cfg.make_minimiser()
    options = cfg.make_options()
    if cfg.backend == "sqpdax":
        from .runners import SqpdaxRunner

        return SqpdaxRunner(problem, x0, minimiser, options)
    if cfg.backend == "scipy":
        from .baselines import ScipyRunner

        return ScipyRunner(problem, x0, minimiser, options)
    raise ValueError(f"unknown backend {cfg.backend!r} for config {cfg.name!r}")


def _solved(row: dict[str, Any], spec: TaskSpec) -> tuple[bool, str | None]:
    """Decide whether the pilot solve counts as solved.

    Returns
    -------
    tuple[bool, str | None]
        ``(solved, reason)`` where ``reason`` is ``None`` when solved and
        otherwise one of ``"unsuccessful"``, ``"infeasible"``,
        ``"wrong_objective"``, ``"non_finite"`` - the same classes the
        analysis layer uses, evaluated here with the task's tolerances so the
        worker can skip timing repeats on failures.
    """
    if not row.get("finite", True):
        return False, "non_finite"
    if not row["successful"]:
        return False, "unsuccessful"
    if not (row["feas"] <= spec.feas_tol):
        return False, "infeasible"
    f_gap = row.get("f_gap", float("nan"))
    if row.get("has_fstar") and not (math.isnan(f_gap) or f_gap <= spec.f_tol):
        return False, "wrong_objective"
    return True, None


def run_task_inprocess(
    spec: TaskSpec, report=None, *, start: float | None = None
) -> dict[str, Any]:
    """Run a task in the current process and return its result row.

    Parameters
    ----------
    spec
        Task description.
    report
        Optional ``report(kind, payload)`` callback used to announce the
        phase being entered (``("phase", name)``).
    start
        ``time.perf_counter()`` reference for the deadline; defaults to now.

    Returns
    -------
    dict[str, Any]
        Flat result row (see :mod:`benchmarks.results` for the schema).
    """
    import jax
    import numpy as np
    import sif2jax

    from .configs import CONFIGS
    from .metrics import quality_metrics
    from .problems import to_sqpdax

    jax.config.update("jax_enable_x64", True)
    start = time.perf_counter() if start is None else start
    deadline = start + spec.timeout_s

    def phase(name: str) -> None:
        if report is not None:
            report("phase", name)

    row: dict[str, Any] = {**asdict(spec), **_device_info(), "phase": "build"}
    phase("build")
    cfg = CONFIGS[spec.config]
    row.update(cfg.tags())
    raw = sif2jax.cutest.get_problem(spec.problem)
    problem, x0, meta = to_sqpdax(raw, y0_iD=spec.y0_iD)
    if spec.y0_iD != raw.y0_iD:
        raw = type(raw)(y0_iD=spec.y0_iD)
    row.update({k: v for k, v in meta.as_dict().items() if k != "name"})
    runner = make_runner(cfg, problem, x0)

    phase("compile")
    row["phase"] = "compile"
    row["compile_s"] = runner.compile()

    phase("warmup")
    row["phase"] = "warmup"
    t0 = time.perf_counter()
    runner.warmup()
    row["warmup_s"] = time.perf_counter() - t0

    phase("pilot")
    row["phase"] = "pilot"
    t0 = time.perf_counter()
    out = runner.solve(spec.max_steps)
    pilot_s = time.perf_counter() - t0
    times = [pilot_s]

    row["status"] = out.status
    row["successful"] = out.successful
    row["steps"] = out.steps
    row.update(_scalar_stats(out.stats))
    xstar = None
    try:
        xstar = raw.expected_result
    except NotImplementedError:
        pass
    row.update(
        quality_metrics(
            problem,
            out.x,
            out.dual,
            fstar=meta.fstar if meta.has_fstar else None,
            xstar=xstar,
        )
    )
    row["solved"], row["repeats_skipped"] = _solved(row, spec)

    if spec.repeats is not None:
        n_repeats = max(1, int(spec.repeats))
    else:
        n_repeats = adaptive_repeats(
            pilot_s,
            budget_s=spec.repeat_budget_s,
            max_repeats=spec.max_repeats,
            slow_floor=spec.slow_floor,
            slow_threshold_s=spec.slow_threshold_s,
        )
    if row["repeats_skipped"] is not None and not spec.repeat_unsolved:
        # Timing a failed solve is not informative; keep only the pilot.
        n_repeats = 1

    phase("repeats")
    row["phase"] = "repeats"
    truncated = False
    for _ in range(n_repeats - 1):
        if time.perf_counter() + pilot_s > deadline:
            truncated = True
            break
        t0 = time.perf_counter()
        runner.solve(spec.max_steps)
        times.append(time.perf_counter() - t0)

    arr = np.asarray(times, dtype=float)
    q10, q25, q75, q90 = np.quantile(arr, [0.10, 0.25, 0.75, 0.90])
    row.update(
        {
            "repeats_requested": n_repeats,
            "repeats": len(times),
            "repeats_truncated": truncated,
            "time_median_s": float(np.median(arr)),
            "time_min_s": float(arr.min()),
            "time_max_s": float(arr.max()),
            "time_mean_s": float(arr.mean()),
            "time_std_s": float(arr.std()) if len(times) > 1 else 0.0,
            "time_q10_s": float(q10),
            "time_q25_s": float(q25),
            "time_q75_s": float(q75),
            "time_q90_s": float(q90),
        }
    )
    row["error"] = None
    row["phase"] = "done"
    row["task_wall_s"] = time.perf_counter() - start
    return row


def _harness_row(
    spec: TaskSpec, *, status: str, phase: str, error: str, wall_s: float
) -> dict[str, Any]:
    """Result row for a task the worker could not complete."""
    from .configs import CONFIGS

    row: dict[str, Any] = {**asdict(spec), **CONFIGS[spec.config].tags()}
    row.update(
        {
            "phase": phase,
            "status": status,
            "successful": False,
            "error": error,
            "task_wall_s": wall_s,
        }
    )
    return row


class _Worker:
    """One persistent spawn worker and the task it is currently running."""

    def __init__(self, ctx: Any, result_queue: Any):
        from ._bootstrap import worker_loop

        self.task_queue: Any = ctx.Queue()
        self.process = ctx.Process(
            target=worker_loop, args=(self.task_queue, result_queue), daemon=True
        )
        self.process.start()
        self.current: tuple[int, TaskSpec] | None = None
        self.started_at = 0.0
        self.phase = "build"

    def submit(self, task_id: int, spec: TaskSpec) -> None:
        self.current = (task_id, spec)
        self.started_at = time.perf_counter()
        self.phase = "build"
        self.task_queue.put((task_id, spec))

    @property
    def deadline(self) -> float:
        assert self.current is not None
        return self.started_at + self.current[1].timeout_s

    def stop(self) -> None:
        if self.process.is_alive():
            try:
                self.task_queue.put(None)
                self.process.join(timeout=5.0)
            finally:
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join()
        self.task_queue.close()

    def kill(self) -> None:
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        self.task_queue.close()


class WorkerPool:
    """Run tasks on ``jobs`` persistent workers with per-task timeouts.

    Parameters
    ----------
    jobs
        Number of concurrent worker processes.
    poll_s
        Interval at which deadlines and worker liveness are checked.

    Examples
    --------
    ```python
    with WorkerPool(jobs=2) as pool:
        for row in pool.map(specs):
            write(row)
    ```
    """

    def __init__(self, jobs: int = 1, *, poll_s: float = 0.25):
        self.jobs = max(1, int(jobs))
        self.poll_s = poll_s
        self._ctx = mp.get_context("spawn")
        self._results: Any = self._ctx.Queue()
        self._workers: list[_Worker] = []

    def __enter__(self) -> WorkerPool:
        self._workers = [_Worker(self._ctx, self._results) for _ in range(self.jobs)]
        return self

    def __exit__(self, *exc_info: object) -> None:
        for worker in self._workers:
            worker.stop()
        self._results.close()

    def _respawn(self, worker: _Worker) -> _Worker:
        worker.kill()
        fresh = _Worker(self._ctx, self._results)
        self._workers[self._workers.index(worker)] = fresh
        return fresh

    def map(self, specs: Iterable[TaskSpec]) -> Iterator[dict[str, Any]]:
        """Yield one result row per spec, in completion order.

        Parameters
        ----------
        specs
            Tasks to run.

        Yields
        ------
        dict[str, Any]
            Result rows from the workers, or harness rows for tasks that
            timed out or crashed.
        """
        pending: dict[int, TaskSpec] = dict(enumerate(specs))
        todo = list(pending)
        by_worker: dict[int, _Worker] = {}

        while todo or by_worker:
            for worker in self._workers:
                if worker.current is None and todo:
                    task_id = todo.pop(0)
                    worker.submit(task_id, pending[task_id])
                    by_worker[task_id] = worker

            try:
                task_id, kind, payload = self._results.get(timeout=self.poll_s)
            except queue_mod.Empty:
                task_id = None
            else:
                if task_id is not None:
                    worker = by_worker[task_id]
                    if kind == "phase":
                        worker.phase = payload
                    else:
                        spec = pending.pop(task_id)
                        del by_worker[task_id]
                        wall = time.perf_counter() - worker.started_at
                        worker.current = None
                        if kind == "ok":
                            yield payload
                        else:
                            status = (
                                "compile_error"
                                if worker.phase in ("build", "compile")
                                else "runtime_error"
                            )
                            yield _harness_row(
                                spec,
                                status=status,
                                phase=worker.phase,
                                error=payload,
                                wall_s=wall,
                            )

            now = time.perf_counter()
            for worker in list(self._workers):
                if worker.current is None:
                    if not worker.process.is_alive():
                        self._respawn(worker)
                    continue
                task_id, spec = worker.current
                timed_out = now > worker.deadline
                died = not worker.process.is_alive()
                if timed_out or died:
                    pending.pop(task_id)
                    del by_worker[task_id]
                    wall = now - worker.started_at
                    if timed_out:
                        error = f"killed after {spec.timeout_s:.0f}s during phase {worker.phase!r}"
                        status = "timeout"
                    else:
                        error = (
                            f"worker exited with code {worker.process.exitcode} "
                            f"during phase {worker.phase!r}"
                        )
                        status = "runtime_error"
                    phase = worker.phase
                    self._respawn(worker)
                    yield _harness_row(
                        spec, status=status, phase=phase, error=error, wall_s=wall
                    )


def run_tasks(
    specs: Iterable[TaskSpec],
    *,
    jobs: int = 1,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run tasks on a :class:`WorkerPool` and collect the rows.

    Parameters
    ----------
    specs
        Tasks to run.
    jobs
        Number of concurrent workers.
    on_result
        Optional callback invoked with each row as soon as it is available
        (used by the CLI to flush JSONL incrementally).

    Returns
    -------
    list[dict[str, Any]]
        All result rows in completion order.
    """
    rows: list[dict[str, Any]] = []
    with WorkerPool(jobs=jobs) as pool:
        for row in pool.map(specs):
            rows.append(row)
            if on_result is not None:
                on_result(row)
    return rows


def run_task(spec: TaskSpec) -> dict[str, Any]:
    """Run a single task in its own worker process.

    Parameters
    ----------
    spec
        Task description; ``spec.timeout_s`` bounds the whole task.

    Returns
    -------
    dict[str, Any]
        The result row (see :meth:`WorkerPool.map`).
    """
    return run_tasks([spec], jobs=1)[0]
