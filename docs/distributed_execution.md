# Bounded distributed research execution (SF-S5-MR8)

Profile first, declare resources, assemble by identity, and never make a cluster
a prerequisite for reproducing a result.

> Simulation and research only. No worker touches the broker, credentials, the
> durable state store, or any network endpoint. The distributed layer computes;
> it does not act.

Framework selection and the measured adoption gate: [ADR 0018](adr/0018-bounded-distributed-research-execution.md).
Reproduce the measurements: `python benchmarks/benchmark_distributed_crossover.py`.

---

## 1. Profile before distributing

The work item's explicit non-goal is "distributing unprofiled code". Distribution
has a fixed cost — process startup, payload serialization, scheduling — that is
invisible until measured against the work it replaces.

`profile_stages` returns a `SerialProfile` carrying per-stage wall time, the
**parallel fraction**, and the Amdahl bound. The bound is the honest part: at a
parallel fraction `p`, speedup can never exceed `1 / (1 - p)` regardless of how
many machines are added.

Measured on the Sprint 4 robustness sweep (36 points, 3 repeats, minimum taken):

| Stage | Wall time | Parallelizable |
| --- | --- | --- |
| Grid construction | 0.06 ms | no |
| Per-point evaluation | 17.17 ms | **yes** |
| Aggregation | 0.42 ms | no |

Parallel fraction **0.9727** — 36.6× at infinite workers, 6.7× at eight.

## 2. But the parallel fraction is not the whole answer

A 97% parallel fraction says what *could* be gained. It says nothing about
whether the fixed cost of distributing is smaller than the work being
distributed. Measured with 32 tasks across 8 processes:

| Per-task cost | Serial | 8 workers | Speedup |
| --- | --- | --- | --- |
| 0.047 ms | 1.5 ms | 89.7 ms | **0.02×** |
| 1.92 ms | 61.5 ms | 93.0 ms | **0.66×** |
| 19.25 ms | 616.0 ms | 206.6 ms | **2.98×** |
| 78.7 ms | 2519.5 ms | 575.6 ms | **4.38×** |

**The crossover sits between 2 ms and 20 ms per task.** Below it, distribution is
up to fifty times *slower*.

The Sprint 4 sweep runs at **0.49 ms per point** — well below the crossover. A
97% parallel fraction with a sub-millisecond task cost is exactly the shape that
looks ideal on paper and loses to a `for` loop in practice. That is why ADR 0018
selects Dask but gates adoption on a per-task cost threshold rather than adopting
it now.

## 3. Tasks declare what they need

Every `TaskSpec` declares CPU, RAM, GPU, scratch, expected duration, seed,
timeout, retry budget, and cancellability. The safety-relevant fields have **no
permissive defaults** — an under-declared task is the one that gets a machine
killed, and a default would let it happen by accident.

Two refusals worth naming:

- **A timeout below the declared expected duration is refused.** Otherwise the
  task is cancelled while behaving exactly as specified.
- **A non-idempotent task may not request retries.** Retrying a task with side
  effects is how one logical unit of work becomes two.

`task_id` is content-addressed over payload, seed, and resources, so the same
logical work has the same identity on every machine and in every run. That is
what makes duplicate detection possible without a central coordinator.

## 4. Assembly is by identity, never completion order

This is the load-bearing property of the whole module.

A distributed run finishes tasks in whatever order workers happen to complete
them, and that order varies between runs on identical inputs. Any assembly that
depends on it — appending to a list, folding in arrival sequence — produces
results that differ run to run **while every individual task is perfectly
deterministic**. It is the hardest class of irreproducibility to diagnose,
because nothing looks wrong.

`BatchReport` sorts by `task_id` at construction. `assembly_hash()` covers every
task's identity and output hash, so two backends that agree on it agree on the
whole computation.

## 5. Backends

| Backend | Requires | Role |
| --- | --- | --- |
| `execute_local` | nothing | **Reference.** Defines the correct answer. |
| `execute_process_pool` | nothing | Real parallelism and real serialization, no external service. Where serialization bugs and order-dependence surface in CI. |
| Cluster (Dask) | a cluster | Not yet added — see the ADR 0018 adoption gate. |

**Cluster access is never required for reproducibility.** A result reproducible
only on a cluster is not reproducible. `assert_backend_parity` compares per-task
output hashes and names the divergent tasks, rather than reporting that
something, somewhere, differs — because a backend that changes results is not an
accelerator.

## 6. Failure model

Four named terminal outcomes, never a bare exception from an anonymous worker:

| Outcome | Meaning |
| --- | --- |
| `succeeded` | Produced a content-hashable value |
| `failed` | Raised, and the retry budget is exhausted |
| `timed_out` | Exceeded its declared per-task timeout |
| `cancelled` | The batch budget expired before it ran or finished |

Every non-success carries an attributed error naming the task. A task returning
an unhashable value **fails** rather than passing, because parity cannot be
checked on a result with no stable bytes.

## 7. Runbook

Profile first. If `distribution_is_justified` is false, stop — the parallel
fraction cannot repay the overhead at any worker count. If it is true, check the
per-task cost against the crossover before reaching for more than a process pool.

On a batch with failures: the report names each one. Failures do not abort the
batch, so one bad task does not discard the results of the others.

**Rollback:** the package is additive and nothing else imports it. Reverting the
commit removes it.

## 8. Residual limitations

- **No cluster backend yet.** The capability is local and process-pool only,
  which is what the measurement supports. Claiming distributed capability would
  be claiming an unmeasured benefit.
- **No network-partition testing**, because there is no network. That acceptance
  criterion cannot honestly be claimed until a cluster backend exists.
- **No GPU scheduling is exercised.** `gpus` is declared and validated, but no
  backend honours it.
- **Per-task timeouts are checked after the call returns.** A task that hangs
  forever in-process is bounded by the batch budget, not its own. Hard
  preemption needs the cluster backend.
- **Measurements are single-machine wall clock** on macOS/arm64 and are labelled
  as such throughout.
- **`ProcessPoolExecutor` requires module-scope functions**, the same constraint
  Dask and Ray impose.

## 9. Evidence

`tests/test_distributed_execution.py` — 50 tests: Amdahl bounds including the
mostly-serial case that reports distribution as unjustified; the nan-versus-zero
distinction for a zero-duration profile; duplicate stage names; content-addressed
identity across all four declared components; non-serializable payload refusal;
the non-idempotent-retry refusal; the timeout-below-duration refusal; resource
validation; duplicate-batch refusal; local execution; identity ordering
independent of submission order; assembly-hash stability; attributed failures;
retry budgets; the unhashable-result failure; batch cancellation; cross-backend
parity at 1, 2, and 4 workers; parity failure naming divergent tasks; the
different-task-set case; worker failure attribution; worker-count refusals;
duplicate rejection before any worker starts; deterministic assembly under
saturation; and the report contracts.

`benchmarks/benchmark_distributed_crossover.py` regenerates the crossover table
and asserts parity at every size.
