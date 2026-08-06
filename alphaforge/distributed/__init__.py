"""Bounded distributed research execution (SF-S5-MR8).

- :mod:`~alphaforge.distributed.profiling` — measure the serial pipeline and the
  parallel fraction *before* distributing anything, with the Amdahl bound that
  caps achievable speedup regardless of worker count.
- :mod:`~alphaforge.distributed.tasks` — resource-declaring, content-addressed
  task specifications. Every task states CPU, RAM, GPU, scratch, duration, seed,
  input hash, timeout, retry budget, and cancellability.
- :mod:`~alphaforge.distributed.executor` — the local reference backend, a
  process-pool backend, and a parity check that refuses a backend which changes
  results.

**Cluster access is never required for reproducibility.** The local backend is
always available and defines the correct answer; any other backend is an optional
accelerator that must match it exactly.
"""

from alphaforge.distributed.executor import (
    MAX_TOTAL_SECONDS,
    MAX_WORKERS,
    BatchReport,
    ExecutionError,
    TaskOutcome,
    TaskResult,
    assert_backend_parity,
    execute_local,
    execute_process_pool,
)
from alphaforge.distributed.profiling import (
    MIN_USEFUL_PARALLEL_FRACTION,
    ProfilingError,
    SerialProfile,
    StageTiming,
    profile_stages,
)
from alphaforge.distributed.tasks import (
    MAX_RETRIES,
    MAX_TASKS_PER_BATCH,
    ResourceRequest,
    TaskContractError,
    TaskSpec,
    assert_unique_tasks,
    content_hash,
)

__all__ = [
    "MAX_RETRIES",
    "MAX_TASKS_PER_BATCH",
    "MAX_TOTAL_SECONDS",
    "MAX_WORKERS",
    "MIN_USEFUL_PARALLEL_FRACTION",
    "BatchReport",
    "ExecutionError",
    "ProfilingError",
    "ResourceRequest",
    "SerialProfile",
    "StageTiming",
    "TaskContractError",
    "TaskOutcome",
    "TaskResult",
    "TaskSpec",
    "assert_backend_parity",
    "assert_unique_tasks",
    "content_hash",
    "execute_local",
    "execute_process_pool",
    "profile_stages",
]
