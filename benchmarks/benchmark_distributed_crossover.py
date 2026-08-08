"""Measure where distributing work starts to pay for itself.

SF-S5-MR11 replaces the single-sample version this file used to contain. That
one took one unwarmed timing per work size and reported it as a finding; the
numbers it produced also disagreed with the ones quoted in ADR 0018, because
they came from a different run of the same unstable procedure.

This harness records **at least one warmup and at least seven measured
repetitions** per work size, keeps every raw nanosecond sample, and verifies
that the two backends produced identical output before reporting a ratio at all.
A speedup between backends that disagree is not a speedup.

Run::

    python benchmarks/benchmark_distributed_crossover.py --output raw.json

Single-machine wall clock on a synthetic workload. **Not a service-level
objective**, not a cluster benchmark, and not a claim about any real pipeline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from alphaforge.distributed import (
    ResourceRequest,
    TaskSpec,
    execute_local,
    execute_process_pool,
)
from alphaforge.evidence.measurements import (
    MIN_REPETITIONS,
    MIN_WARMUPS,
    SCHEMA_VERSION,
    BenchmarkEnvironment,
    CrossoverEvidence,
    WorkSizeMeasurement,
)

#: Work sizes chosen to bracket the crossover, not to flatter it.
ITERATION_COUNTS: tuple[int, ...] = (1_000, 50_000, 500_000, 2_000_000)
TASK_COUNT = 32
WORKERS = 8

LIMITATIONS: tuple[str, ...] = (
    "Single-machine wall clock on one synthetic CPU-bound workload.",
    "Not a service-level objective and not a performance guarantee.",
    "Not a cluster benchmark: only in-process and process-pool backends are measured.",
    "Absolute timings depend on machine load; the crossover interval is the durable finding.",
    "Process-pool startup cost dominates below the crossover and is platform-specific.",
)


def busy_work(payload: dict[str, Any]) -> float:
    """Deterministic CPU-bound work scaled by ``iters``.

    Module-scope and pure so the process pool can pickle it; rounded so both
    backends produce identical output rather than merely close output.
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
                cpus=1.0, memory_mb=256, gpus=0, scratch_mb=0, expected_seconds=300.0
            ),
            timeout_seconds=600.0,
        )
        for index in range(count)
    ]


def measure(iterations: int, *, repetitions: int, warmups: int) -> WorkSizeMeasurement:
    """Measure one work size with warmups and repeated samples."""
    tasks = _batch(TASK_COUNT, iterations)

    for _ in range(warmups):
        execute_local(busy_work, tasks)
        execute_process_pool(busy_work, tasks, workers=WORKERS)

    serial_samples: list[int] = []
    pool_samples: list[int] = []
    parity_identity = ""
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        serial = execute_local(busy_work, tasks)
        serial_samples.append(time.perf_counter_ns() - started)

        started = time.perf_counter_ns()
        pooled = execute_process_pool(busy_work, tasks, workers=WORKERS)
        pool_samples.append(time.perf_counter_ns() - started)

        if serial.assembly_hash() != pooled.assembly_hash():
            raise SystemExit(
                f"backend parity failed at {iterations} iterations: a speedup between "
                "backends that disagree is not a speedup"
            )
        parity_identity = serial.assembly_hash()

    return WorkSizeMeasurement(
        iterations=iterations,
        task_count=TASK_COUNT,
        workers=WORKERS,
        warmups=warmups,
        serial_ns=tuple(serial_samples),
        pool_ns=tuple(pool_samples),
        parity_identity=parity_identity,
    )


def main() -> None:
    """Run the harness and write raw evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="raw evidence JSON destination")
    parser.add_argument("--repetitions", type=int, default=MIN_REPETITIONS)
    parser.add_argument("--warmups", type=int, default=MIN_WARMUPS)
    arguments = parser.parse_args()

    measurements = tuple(
        measure(iterations, repetitions=arguments.repetitions, warmups=arguments.warmups)
        for iterations in ITERATION_COUNTS
    )
    evidence = CrossoverEvidence(
        schema_version=SCHEMA_VERSION,
        environment=BenchmarkEnvironment.capture(),
        measurements=measurements,
        limitations=LIMITATIONS,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(evidence.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    bounds = evidence.crossover_bounds_ms()
    print(f"{'per-task ms':>12} {'speedup':>9} {'range':>18}  reps")
    for item in evidence.measurements:
        low, high = item.speedup_range
        print(
            f"{item.per_task_ms:12.3f} {item.speedup:8.2f}x "
            f"{low:8.2f}-{high:.2f}x {len(item.serial_ns):5d}"
        )
    print()
    print(f"crossover interval (ms/task): {bounds}")
    print(f"evidence identity: {evidence.identity()[:16]}")


if __name__ == "__main__":
    main()
