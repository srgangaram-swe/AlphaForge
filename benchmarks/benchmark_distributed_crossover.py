"""Measure where distribution starts paying for itself.

SF-S5-MR8 evidence generator. Reproduces the crossover table in
``docs/adr/0018-bounded-distributed-research-execution.md``.

The number this produces is the one that decides whether to distribute at all:
below the crossover, a process pool is *slower* than a serial loop because
process startup and payload serialization cost more than the work itself.

Run::

    python benchmarks/benchmark_distributed_crossover.py

Results are wall-clock on one machine and are labelled as such. They are not a
benchmark of any cluster backend.
"""

from __future__ import annotations

import json
import platform
from typing import Any

from alphaforge.distributed import (
    ResourceRequest,
    TaskSpec,
    execute_local,
    execute_process_pool,
)

#: Work sizes chosen to bracket the crossover, not to flatter it.
ITERATION_COUNTS = (1_000, 50_000, 500_000, 2_000_000)
TASK_COUNT = 32
WORKERS = 8


def busy_work(payload: dict[str, Any]) -> float:
    """Deterministic CPU-bound work scaled by ``iters``.

    Module-scope and pure so the process pool can pickle it, and rounded so the
    result is identical across backends rather than merely close.
    """
    total = 0.0
    for index in range(int(payload["iters"])):
        total += (index % 7) ** 0.5
    return round(total, 6)


def _batch(count: int, iters: int) -> list[TaskSpec]:
    return [
        TaskSpec(
            name=f"probe-{index}",
            payload={"index": index, "iters": iters},
            seed=index,
            resources=ResourceRequest(
                cpus=1.0, memory_mb=256, gpus=0, scratch_mb=0, expected_seconds=60.0
            ),
            timeout_seconds=300.0,
        )
        for index in range(count)
    ]


def main() -> None:
    """Print the crossover table and a machine-readable summary."""
    rows: list[dict[str, Any]] = []
    print(f"{'per-task ms':>12} {'serial ms':>11} {'pool ms':>10} {'speedup':>9}  parity")
    for iters in ITERATION_COUNTS:
        tasks = _batch(TASK_COUNT, iters)
        serial = execute_local(busy_work, tasks)
        pooled = execute_process_pool(busy_work, tasks, workers=WORKERS)
        parity = serial.assembly_hash() == pooled.assembly_hash()
        per_task_ms = serial.wall_seconds / TASK_COUNT * 1_000
        speedup = serial.wall_seconds / pooled.wall_seconds
        rows.append(
            {
                "iterations": iters,
                "per_task_ms": per_task_ms,
                "serial_ms": serial.wall_seconds * 1_000,
                "pool_ms": pooled.wall_seconds * 1_000,
                "speedup": speedup,
                "parity": parity,
            }
        )
        print(
            f"{per_task_ms:12.3f} {serial.wall_seconds * 1_000:11.1f} "
            f"{pooled.wall_seconds * 1_000:10.1f} {speedup:8.2f}x  {parity}"
        )

    print()
    print(
        json.dumps(
            {
                "task_count": TASK_COUNT,
                "workers": WORKERS,
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                },
                "rows": rows,
                "note": (
                    "Wall-clock, single machine, not a cluster benchmark. Parity must "
                    "hold at every size: a backend that changes results is not an "
                    "accelerator."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
