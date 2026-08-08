"""Raw benchmark samples and the summaries derived from them.

SF-S5-MR11. The Sprint 5 close-out originally carried its performance numbers as
hand-typed Python constants while its docstring claimed every panel was computed
from source. That claim was false, the transcribed values disagreed with ADR
0018, and each figure was a single unwarmed sample with no dispersion. This
module exists so none of those three things can recur.

**Summaries are computed, never asserted.** A :class:`WorkSizeMeasurement` holds
the raw nanosecond samples; median and range are properties over them. There is
no field a caller can set to a number the samples do not support.

**A single sample is refused.** ``MIN_REPETITIONS`` is 7, and warmups are
recorded separately and excluded from statistics. One timing is an anecdote: it
cannot show dispersion, and dispersion is what says whether a 2.98× speedup is a
finding or noise.

**The environment is part of the measurement.** A timing without the interpreter,
platform, machine, and CPU count that produced it cannot be compared against
another timing. Merging records from different environments is refused rather
than averaged.

Nothing here is a service-level objective. These are single-machine wall-clock
measurements of a synthetic workload, and every record says so.
"""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

#: A single timing is an anecdote. Seven is the smallest count that gives a
#: median which is not simply one of two middle values, and lets a range mean
#: something.
MIN_REPETITIONS: Final = 7

#: At least one untimed pass before measuring, so import, allocation, and
#: interpreter warm-up costs are not attributed to the work.
MIN_WARMUPS: Final = 1

#: Refusal thresholds, not tuning knobs.
MAX_REPETITIONS: Final = 1_000
MAX_WORK_SIZES: Final = 64
MAX_NANOSECONDS: Final = 24 * 60 * 60 * 1_000_000_000


class MeasurementError(ValueError):
    """Raised when a measurement record is malformed or uninterpretable."""


def _positive_int(value: object, *, field_name: str, maximum: int) -> int:
    """Return a bounded positive int, refusing bools."""
    if isinstance(value, bool):
        raise MeasurementError(f"{field_name} must be an int, not a bool")
    if not isinstance(value, int):
        raise MeasurementError(f"{field_name} must be an int, got {type(value).__name__}")
    if not 0 < value <= maximum:
        raise MeasurementError(f"{field_name} must lie in (0, {maximum}]")
    return value


def _nanoseconds(value: object, *, field_name: str) -> int:
    """Return a plausible nanosecond duration.

    Refuses zero as well as negatives: a measured duration of exactly zero means
    the clock did not advance, so the sample carries no information about the
    work rather than saying the work was free.
    """
    if isinstance(value, bool):
        raise MeasurementError(f"{field_name} must be an int, not a bool")
    if not isinstance(value, int):
        raise MeasurementError(f"{field_name} must be integer nanoseconds")
    if value <= 0:
        raise MeasurementError(
            f"{field_name} must be positive; a zero duration means the clock did not "
            "advance, which says nothing about the work"
        )
    if value > MAX_NANOSECONDS:
        raise MeasurementError(f"{field_name} exceeds the {MAX_NANOSECONDS}ns ceiling")
    return value


@dataclass(frozen=True)
class BenchmarkEnvironment:
    """The machine and runtime a measurement came from.

    Deliberately excludes anything identifying: no username, no home path, no
    hostname, no process environment. A CPU count and an interpreter version are
    what make two timings comparable; who ran them is not.
    """

    python_version: str
    platform_name: str
    machine: str
    processor: str
    cpu_count: int
    timing_clock: str

    def __post_init__(self) -> None:
        for field_name in ("python_version", "platform_name", "machine", "timing_clock"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise MeasurementError(f"{field_name} must be a non-empty string")
        if not isinstance(self.processor, str):
            raise MeasurementError("processor must be a string")
        parts = self.python_version.split(".")
        if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
            raise MeasurementError(
                f"python_version {self.python_version!r} is malformed; expected at least "
                "major.minor"
            )
        object.__setattr__(
            self, "cpu_count", _positive_int(self.cpu_count, field_name="cpu_count", maximum=4_096)
        )

    @classmethod
    def capture(cls) -> BenchmarkEnvironment:
        """Capture the current environment, excluding identifying details."""
        import os

        return cls(
            python_version=platform.python_version(),
            platform_name=platform.platform(),
            machine=platform.machine(),
            processor=platform.processor() or "unknown",
            cpu_count=os.cpu_count() or 1,
            timing_clock="time.perf_counter_ns",
        )

    def identity(self) -> str:
        """Content identity, so records from different machines cannot merge."""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "python_version": self.python_version,
            "platform": self.platform_name,
            "machine": self.machine,
            "processor": self.processor,
            "cpu_count": self.cpu_count,
            "timing_clock": self.timing_clock,
        }


@dataclass(frozen=True)
class WorkSizeMeasurement:
    """Raw samples for one work size under one backend pair.

    Every statistic is a property computed from ``serial_ns`` and ``pool_ns``.
    There is no field holding a summary, so a summary cannot disagree with the
    samples that produced it.
    """

    iterations: int
    task_count: int
    workers: int
    warmups: int
    serial_ns: tuple[int, ...]
    pool_ns: tuple[int, ...]
    parity_identity: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "iterations",
            _positive_int(self.iterations, field_name="iterations", maximum=1_000_000_000),
        )
        object.__setattr__(
            self,
            "task_count",
            _positive_int(self.task_count, field_name="task_count", maximum=100_000),
        )
        object.__setattr__(
            self, "workers", _positive_int(self.workers, field_name="workers", maximum=1_024)
        )
        if isinstance(self.warmups, bool) or not isinstance(self.warmups, int):
            raise MeasurementError("warmups must be an int")
        if self.warmups < MIN_WARMUPS:
            raise MeasurementError(
                f"at least {MIN_WARMUPS} warmup is required; without one, import and "
                "allocation costs are attributed to the work being measured"
            )
        for field_name in ("serial_ns", "pool_ns"):
            samples = tuple(getattr(self, field_name))
            if len(samples) < MIN_REPETITIONS:
                raise MeasurementError(
                    f"{field_name} has {len(samples)} sample(s); at least {MIN_REPETITIONS} are "
                    "required. A single timing is an anecdote — it cannot show dispersion, and "
                    "dispersion is what distinguishes a finding from noise."
                )
            if len(samples) > MAX_REPETITIONS:
                raise MeasurementError(f"{field_name} exceeds {MAX_REPETITIONS} samples")
            validated = tuple(
                _nanoseconds(sample, field_name=f"{field_name}[{index}]")
                for index, sample in enumerate(samples)
            )
            object.__setattr__(self, field_name, validated)
        if not isinstance(self.parity_identity, str) or len(self.parity_identity) != 64:
            raise MeasurementError(
                "parity_identity must be a full SHA-256 digest; without proof the two "
                "backends produced identical output, a speedup number is meaningless"
            )

    @property
    def serial_median_ns(self) -> float:
        """Median serial duration."""
        return statistics.median(self.serial_ns)

    @property
    def pool_median_ns(self) -> float:
        """Median pooled duration."""
        return statistics.median(self.pool_ns)

    @property
    def per_task_ms(self) -> float:
        """Median serial cost per task, in milliseconds."""
        return self.serial_median_ns / self.task_count / 1_000_000

    @property
    def speedup(self) -> float:
        """Median serial divided by median pooled duration."""
        return self.serial_median_ns / self.pool_median_ns

    @property
    def serial_range_ns(self) -> tuple[int, int]:
        """Min and max serial samples, so dispersion is visible."""
        return (min(self.serial_ns), max(self.serial_ns))

    @property
    def pool_range_ns(self) -> tuple[int, int]:
        """Min and max pooled samples."""
        return (min(self.pool_ns), max(self.pool_ns))

    @property
    def speedup_range(self) -> tuple[float, float]:
        """Speedup bounds from the extreme samples.

        Computed pessimistically and optimistically — slowest serial against
        fastest pool, and the reverse — so the interval brackets what the samples
        actually support rather than propagating only the medians.
        """
        low = min(self.serial_ns) / max(self.pool_ns)
        high = max(self.serial_ns) / min(self.pool_ns)
        return (low, high)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-friendly record, raw samples included."""
        return {
            "iterations": self.iterations,
            "task_count": self.task_count,
            "workers": self.workers,
            "warmups": self.warmups,
            "repetitions": len(self.serial_ns),
            "serial_ns": list(self.serial_ns),
            "pool_ns": list(self.pool_ns),
            "parity_identity": self.parity_identity,
            "derived": {
                "serial_median_ns": self.serial_median_ns,
                "pool_median_ns": self.pool_median_ns,
                "serial_range_ns": list(self.serial_range_ns),
                "pool_range_ns": list(self.pool_range_ns),
                "per_task_ms": self.per_task_ms,
                "speedup": self.speedup,
                "speedup_range": list(self.speedup_range),
            },
        }


@dataclass(frozen=True)
class CrossoverEvidence:
    """A complete measurement set from one environment.

    Raises:
        MeasurementError: On duplicate work sizes, an empty set, or records that
            did not come from the same environment.
    """

    schema_version: int
    environment: BenchmarkEnvironment
    measurements: tuple[WorkSizeMeasurement, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise MeasurementError(
                f"schema version {self.schema_version} does not match {SCHEMA_VERSION}"
            )
        measurements = tuple(self.measurements)
        if not measurements:
            raise MeasurementError("evidence must contain at least one measurement")
        if len(measurements) > MAX_WORK_SIZES:
            raise MeasurementError(f"exceeds the {MAX_WORK_SIZES}-work-size ceiling")
        sizes = [item.iterations for item in measurements]
        if len(set(sizes)) != len(sizes):
            raise MeasurementError(
                "duplicate work sizes; two records for one size would let a plot pick "
                "whichever is more flattering"
            )
        if not self.limitations:
            raise MeasurementError(
                "limitations must be stated; a measurement published without them invites "
                "reading it as a service-level objective"
            )
        object.__setattr__(
            self, "measurements", tuple(sorted(measurements, key=lambda item: item.iterations))
        )
        object.__setattr__(self, "limitations", tuple(self.limitations))

    def identity(self) -> str:
        """Content identity over the whole evidence set."""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def crossover_bounds_ms(self) -> tuple[float, float] | None:
        """Return the per-task interval bracketing break-even, or ``None``.

        Reports the interval between the largest measured size that lost and the
        smallest that won, rather than interpolating a single crossing point.
        The samples bound where break-even lies; they do not locate it.
        """
        losing = [m.per_task_ms for m in self.measurements if m.speedup < 1.0]
        winning = [m.per_task_ms for m in self.measurements if m.speedup >= 1.0]
        if not losing or not winning:
            return None
        return (max(losing), min(winning))

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-friendly evidence set."""
        return {
            "schema_version": self.schema_version,
            "environment": self.environment.to_dict(),
            "environment_identity": self.environment.identity(),
            "measurements": [item.to_dict() for item in self.measurements],
            "limitations": list(self.limitations),
        }


#: Bump only with a deliberate migration of committed reference records.
SCHEMA_VERSION: Final = 1


def evidence_from_payload(payload: Mapping[str, Any]) -> CrossoverEvidence:
    """Rebuild an evidence set from its stored JSON payload.

    Raises:
        MeasurementError: On missing fields or malformed records.
    """
    if not isinstance(payload, Mapping):
        raise MeasurementError("evidence payload is not an object")
    missing = {"schema_version", "environment", "measurements", "limitations"} - set(payload)
    if missing:
        raise MeasurementError(f"evidence payload is missing {sorted(missing)}")
    environment_payload = payload["environment"]
    environment = BenchmarkEnvironment(
        python_version=str(environment_payload["python_version"]),
        platform_name=str(environment_payload["platform"]),
        machine=str(environment_payload["machine"]),
        processor=str(environment_payload["processor"]),
        cpu_count=int(environment_payload["cpu_count"]),
        timing_clock=str(environment_payload["timing_clock"]),
    )
    measurements = tuple(
        WorkSizeMeasurement(
            iterations=int(item["iterations"]),
            task_count=int(item["task_count"]),
            workers=int(item["workers"]),
            warmups=int(item["warmups"]),
            serial_ns=tuple(int(sample) for sample in item["serial_ns"]),
            pool_ns=tuple(int(sample) for sample in item["pool_ns"]),
            parity_identity=str(item["parity_identity"]),
        )
        for item in payload["measurements"]
    )
    return CrossoverEvidence(
        schema_version=int(payload["schema_version"]),
        environment=environment,
        measurements=measurements,
        limitations=tuple(str(item) for item in payload["limitations"]),
    )


def summary_rows(evidence: CrossoverEvidence) -> list[dict[str, Any]]:
    """Return plot- and CSV-ready rows computed from the raw samples.

    Every value is derived here. Nothing in the published bundle restates a
    measurement as a literal.
    """
    return [
        {
            "iterations": item.iterations,
            "task_count": item.task_count,
            "workers": item.workers,
            "repetitions": len(item.serial_ns),
            "per_task_ms": item.per_task_ms,
            "serial_median_ms": item.serial_median_ns / 1_000_000,
            "pool_median_ms": item.pool_median_ns / 1_000_000,
            "speedup": item.speedup,
            "speedup_low": item.speedup_range[0],
            "speedup_high": item.speedup_range[1],
        }
        for item in evidence.measurements
    ]


def assert_same_environment(records: Sequence[CrossoverEvidence]) -> None:
    """Refuse to combine records captured on different machines.

    Raises:
        MeasurementError: If the environments differ.
    """
    identities = {item.environment.identity() for item in records}
    if len(identities) > 1:
        raise MeasurementError(
            f"records span {len(identities)} environments; timings from different machines "
            "cannot be combined, and averaging them would produce a number describing "
            "neither"
        )


__all__ = [
    "MAX_REPETITIONS",
    "MAX_WORK_SIZES",
    "MIN_REPETITIONS",
    "MIN_WARMUPS",
    "SCHEMA_VERSION",
    "BenchmarkEnvironment",
    "CrossoverEvidence",
    "MeasurementError",
    "WorkSizeMeasurement",
    "assert_same_environment",
    "evidence_from_payload",
    "summary_rows",
]
