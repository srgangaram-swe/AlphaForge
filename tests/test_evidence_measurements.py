"""Tests for raw benchmark records and the delivery ledger (SF-S5-MR11).

These exist because the original Sprint 5 close-out published hand-typed numbers
under a docstring claiming they were computed from source. The invariants that
carry the most weight:

* **A single sample is refused.** Dispersion is what distinguishes a finding
  from noise, and one timing cannot show it.
* **Summaries are computed, never stored**, so no published statistic can
  disagree with the samples behind it.
* **Records from different environments cannot merge.**
* **The ledger counts test functions and says so** — not collected cases, not
  passing tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from alphaforge.evidence import (
    COUNT_SEMANTICS,
    MIN_REPETITIONS,
    SCHEMA_VERSION,
    BenchmarkEnvironment,
    CrossoverEvidence,
    DeliveryLedger,
    InventoryError,
    MeasurementError,
    SliceRecord,
    WorkSizeMeasurement,
    assert_same_environment,
    count_test_functions,
    evidence_from_payload,
    ledger_from_payload,
    summary_rows,
)

PARITY = "a" * 64


def _environment(**kw: Any) -> BenchmarkEnvironment:
    base: dict[str, Any] = {
        "python_version": "3.13.11",
        "platform_name": "macOS-15.0-arm64",
        "machine": "arm64",
        "processor": "arm",
        "cpu_count": 10,
        "timing_clock": "time.perf_counter_ns",
    }
    base.update(kw)
    return BenchmarkEnvironment(**base)


def _measurement(**kw: Any) -> WorkSizeMeasurement:
    base: dict[str, Any] = {
        "iterations": 1_000,
        "task_count": 32,
        "workers": 8,
        "warmups": 1,
        "serial_ns": tuple(1_000_000 + index for index in range(MIN_REPETITIONS)),
        "pool_ns": tuple(2_000_000 + index for index in range(MIN_REPETITIONS)),
        "parity_identity": PARITY,
    }
    base.update(kw)
    return WorkSizeMeasurement(**base)


def _evidence(**kw: Any) -> CrossoverEvidence:
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "environment": _environment(),
        "measurements": (_measurement(),),
        "limitations": ("single machine",),
    }
    base.update(kw)
    return CrossoverEvidence(**base)


# ---------------------------------------------------------------------------
# A single sample is not evidence
# ---------------------------------------------------------------------------


def test_one_sample_is_refused() -> None:
    """The original defect: one unwarmed timing reported as a finding."""
    with pytest.raises(MeasurementError, match="anecdote"):
        _measurement(serial_ns=(1_000_000,), pool_ns=(2_000_000,))


def test_fewer_than_the_minimum_repetitions_is_refused() -> None:
    short = tuple(1_000_000 + i for i in range(MIN_REPETITIONS - 1))
    with pytest.raises(MeasurementError, match=f"at least {MIN_REPETITIONS}"):
        _measurement(serial_ns=short, pool_ns=short)


def test_a_missing_warmup_is_refused() -> None:
    """Without one, import and allocation costs are charged to the work."""
    with pytest.raises(MeasurementError, match="warmup"):
        _measurement(warmups=0)


def test_the_minimum_repetition_count_is_declared() -> None:
    assert MIN_REPETITIONS >= 7


# ---------------------------------------------------------------------------
# Summaries are computed, not stored
# ---------------------------------------------------------------------------


def test_the_median_comes_from_the_samples() -> None:
    measurement = _measurement(
        serial_ns=(10, 20, 30, 40, 50, 60, 70), pool_ns=(5, 5, 5, 5, 5, 5, 5)
    )
    assert measurement.serial_median_ns == 40
    assert measurement.pool_median_ns == 5
    assert measurement.speedup == pytest.approx(8.0)


def test_the_range_reflects_the_extremes() -> None:
    measurement = _measurement(
        serial_ns=(10, 20, 30, 40, 50, 60, 70), pool_ns=(5, 6, 7, 8, 9, 10, 11)
    )
    assert measurement.serial_range_ns == (10, 70)
    low, high = measurement.speedup_range
    assert low == pytest.approx(10 / 11)
    assert high == pytest.approx(70 / 5)
    assert low < measurement.speedup < high


def test_there_is_no_settable_summary_field() -> None:
    """A stored summary could disagree with its samples; a property cannot."""
    measurement = _measurement()
    for name in ("speedup", "serial_median_ns", "per_task_ms"):
        assert isinstance(getattr(type(measurement), name), property)


def test_per_task_cost_divides_by_the_task_count() -> None:
    measurement = _measurement(
        task_count=10, serial_ns=tuple(10_000_000 for _ in range(MIN_REPETITIONS))
    )
    assert measurement.per_task_ms == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------


def test_a_zero_duration_sample_is_refused() -> None:
    """A clock that did not advance says nothing about the work."""
    samples = (0, *(1_000 for _ in range(MIN_REPETITIONS - 1)))
    with pytest.raises(MeasurementError, match="clock did not advance"):
        _measurement(serial_ns=samples)


def test_a_negative_sample_is_refused() -> None:
    samples = (-5, *(1_000 for _ in range(MIN_REPETITIONS - 1)))
    with pytest.raises(MeasurementError, match="positive"):
        _measurement(serial_ns=samples)


def test_a_boolean_is_not_a_number() -> None:
    with pytest.raises(MeasurementError, match="not a bool"):
        _measurement(task_count=cast(int, True))


def test_a_float_sample_is_refused() -> None:
    samples = (1_000.5, *(1_000 for _ in range(MIN_REPETITIONS - 1)))
    with pytest.raises(MeasurementError, match="integer nanoseconds"):
        _measurement(serial_ns=cast(tuple[int, ...], samples))


def test_a_missing_parity_identity_is_refused() -> None:
    """A speedup between backends that disagree is not a speedup."""
    with pytest.raises(MeasurementError, match="parity_identity"):
        _measurement(parity_identity="")


def test_a_malformed_python_version_is_refused() -> None:
    with pytest.raises(MeasurementError, match="malformed"):
        _environment(python_version="banana")


def test_a_non_positive_cpu_count_is_refused() -> None:
    with pytest.raises(MeasurementError, match="cpu_count"):
        _environment(cpu_count=0)


def test_duplicate_work_sizes_are_refused() -> None:
    """Two records for one size let a plot pick whichever is more flattering."""
    with pytest.raises(MeasurementError, match="duplicate work sizes"):
        _evidence(measurements=(_measurement(), _measurement()))


def test_evidence_without_limitations_is_refused() -> None:
    with pytest.raises(MeasurementError, match="limitations must be stated"):
        _evidence(limitations=())


def test_an_empty_measurement_set_is_refused() -> None:
    with pytest.raises(MeasurementError, match="at least one measurement"):
        _evidence(measurements=())


def test_a_wrong_schema_version_is_refused() -> None:
    with pytest.raises(MeasurementError, match="schema version"):
        _evidence(schema_version=99)


# ---------------------------------------------------------------------------
# Environment consistency
# ---------------------------------------------------------------------------


def test_records_from_different_machines_cannot_merge() -> None:
    """Averaging them would produce a number describing neither."""
    with pytest.raises(MeasurementError, match="different machines"):
        assert_same_environment(
            [_evidence(), _evidence(environment=_environment(machine="x86_64"))]
        )


def test_records_from_one_machine_merge() -> None:
    assert_same_environment([_evidence(), _evidence()])


def test_the_environment_excludes_identifying_details() -> None:
    """No username, home path, hostname, or process environment."""
    payload = json.dumps(_environment().to_dict())
    for leak in ("/Users/", "/home/", "USER", "HOME", "hostname"):
        assert leak not in payload


def test_a_captured_environment_is_usable() -> None:
    captured = BenchmarkEnvironment.capture()
    assert captured.cpu_count >= 1
    assert captured.timing_clock == "time.perf_counter_ns"


# ---------------------------------------------------------------------------
# Crossover bounds
# ---------------------------------------------------------------------------


def test_the_crossover_is_an_interval_not_a_point() -> None:
    """The samples bound where break-even lies; they do not locate it."""
    losing = _measurement(
        iterations=1_000,
        serial_ns=tuple(1_000_000 for _ in range(MIN_REPETITIONS)),
        pool_ns=tuple(4_000_000 for _ in range(MIN_REPETITIONS)),
    )
    winning = _measurement(
        iterations=500_000,
        serial_ns=tuple(400_000_000 for _ in range(MIN_REPETITIONS)),
        pool_ns=tuple(100_000_000 for _ in range(MIN_REPETITIONS)),
    )
    bounds = _evidence(measurements=(losing, winning)).crossover_bounds_ms()
    assert bounds is not None
    assert bounds[0] < bounds[1]


def test_bounds_are_none_when_break_even_is_not_bracketed() -> None:
    losing = _measurement(
        serial_ns=tuple(1_000_000 for _ in range(MIN_REPETITIONS)),
        pool_ns=tuple(4_000_000 for _ in range(MIN_REPETITIONS)),
    )
    assert _evidence(measurements=(losing,)).crossover_bounds_ms() is None


def test_summary_rows_derive_every_value() -> None:
    rows = summary_rows(_evidence())
    assert rows[0]["repetitions"] == MIN_REPETITIONS
    assert rows[0]["speedup_low"] <= rows[0]["speedup"] <= rows[0]["speedup_high"]


def test_evidence_round_trips_through_its_payload() -> None:
    original = _evidence()
    assert evidence_from_payload(original.to_dict()).identity() == original.identity()


def test_a_payload_missing_fields_is_refused() -> None:
    payload = _evidence().to_dict()
    del payload["measurements"]
    with pytest.raises(MeasurementError, match="missing"):
        evidence_from_payload(payload)


def test_changed_samples_change_the_identity() -> None:
    other = _evidence(
        measurements=(_measurement(serial_ns=tuple(9_000_000 + i for i in range(MIN_REPETITIONS))),)
    )
    assert other.identity() != _evidence().identity()


# ---------------------------------------------------------------------------
# Delivery ledger
# ---------------------------------------------------------------------------


def test_test_functions_are_counted_by_parsing() -> None:
    source = """
def test_one() -> None: ...
def test_two() -> None: ...
def helper() -> None: ...
class Thing:
    def test_nested(self) -> None: ...
"""
    # Nested methods are not module-level functions and are not counted.
    assert count_test_functions(source) == 2


def test_a_parametrized_function_counts_once() -> None:
    """One function, many collected cases — the distinction the label states."""
    source = """
import pytest

@pytest.mark.parametrize("value", [1, 2, 3, 4, 5])
def test_many(value: int) -> None: ...
"""
    assert count_test_functions(source) == 1


def test_unparseable_source_is_refused() -> None:
    with pytest.raises(InventoryError, match="does not parse"):
        count_test_functions("def test_broken(:")


def test_the_count_semantics_are_stated_on_every_record() -> None:
    record = SliceRecord(
        label="slice",
        commit="c" * 40,
        source_paths=(),
        test_paths=(),
        doc_paths=(),
        test_function_count=3,
    )
    assert record.to_dict()["counts"] == COUNT_SEMANTICS
    assert "not collected parameter cases" in COUNT_SEMANTICS.lower()


def test_a_short_commit_id_is_refused() -> None:
    """The ledger pins full object ids so a slice cannot become ambiguous."""
    with pytest.raises(InventoryError, match="40-character"):
        SliceRecord(
            label="slice",
            commit="abc123",
            source_paths=(),
            test_paths=(),
            doc_paths=(),
            test_function_count=0,
        )


def test_two_slices_naming_one_commit_are_refused() -> None:
    def record(label: str) -> SliceRecord:
        return SliceRecord(
            label=label,
            commit="d" * 40,
            source_paths=(),
            test_paths=(),
            doc_paths=(),
            test_function_count=1,
        )

    with pytest.raises(InventoryError, match="same commit"):
        DeliveryLedger(sprint=5, slices=(record("a"), record("b")))


def test_the_ledger_total_is_the_sum_of_its_slices() -> None:
    ledger = DeliveryLedger(
        sprint=5,
        slices=(
            SliceRecord(
                label="a",
                commit="e" * 40,
                source_paths=(),
                test_paths=(),
                doc_paths=(),
                test_function_count=10,
            ),
            SliceRecord(
                label="b",
                commit="f" * 40,
                source_paths=(),
                test_paths=(),
                doc_paths=(),
                test_function_count=7,
            ),
        ),
    )
    assert ledger.total_test_functions == 17
    assert ledger.reconciles_to({"a": 10, "b": 7})
    assert not ledger.reconciles_to({"a": 10, "b": 8})


def test_a_ledger_round_trips(tmp_path: Path) -> None:
    ledger = DeliveryLedger(
        sprint=5,
        slices=(
            SliceRecord(
                label="a",
                commit="1" * 40,
                source_paths=("x.py",),
                test_paths=("t.py",),
                doc_paths=("d.md",),
                test_function_count=4,
            ),
        ),
    )
    assert ledger_from_payload(ledger.to_dict()).identity() == ledger.identity()


def test_an_empty_ledger_is_refused() -> None:
    with pytest.raises(InventoryError, match="at least one slice"):
        DeliveryLedger(sprint=5, slices=())


# ---------------------------------------------------------------------------
# Ledger derivation against a real repository
# ---------------------------------------------------------------------------


def _init_repository(root: Path) -> str:
    """Create a throwaway repository with one commit and return its id."""
    import subprocess

    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments], cwd=root, capture_output=True, text=True, check=True
        )
        return completed.stdout.strip()

    run("init", "--quiet")
    run("config", "user.email", "test@example.invalid")
    run("config", "user.name", "Test")
    (root / "alphaforge").mkdir()
    (root / "tests").mkdir()
    (root / "docs").mkdir()
    (root / "alphaforge" / "thing.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "docs" / "thing.md").write_text("# Thing\n", encoding="utf-8")
    (root / "tests" / "test_thing.py").write_text(
        "def test_a() -> None: ...\ndef test_b() -> None: ...\ndef helper() -> None: ...\n",
        encoding="utf-8",
    )
    run("add", "-A")
    run("commit", "--quiet", "-m", "initial")
    return run("rev-parse", "HEAD")


def _specification(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "label": "slice",
        "reference": "HEAD",
        "source_paths": ["alphaforge/thing.py"],
        "test_paths": ["tests/test_thing.py"],
        "doc_paths": ["docs/thing.md"],
    }
    base.update(kw)
    return base


def test_a_ledger_is_derived_from_a_real_repository(tmp_path: Path) -> None:
    """The counts come from parsing tracked files, not from a table."""
    from alphaforge.evidence.inventory import build_ledger

    commit = _init_repository(tmp_path)
    ledger = build_ledger(5, [_specification()], repository=tmp_path)
    assert ledger.slices[0].commit == commit
    assert ledger.slices[0].test_function_count == 2
    assert ledger.total_test_functions == 2


def test_an_unresolvable_reference_is_refused(tmp_path: Path) -> None:
    from alphaforge.evidence.inventory import build_ledger

    _init_repository(tmp_path)
    with pytest.raises(InventoryError, match="git rev-parse"):
        build_ledger(5, [_specification(reference="no-such-ref")], repository=tmp_path)


def test_a_missing_test_path_is_refused(tmp_path: Path) -> None:
    """The ledger must describe paths that exist, or it describes nothing."""
    from alphaforge.evidence.inventory import build_ledger

    _init_repository(tmp_path)
    with pytest.raises(InventoryError, match="is not a file"):
        build_ledger(5, [_specification(test_paths=["tests/absent.py"])], repository=tmp_path)


def test_a_missing_source_path_is_refused(tmp_path: Path) -> None:
    from alphaforge.evidence.inventory import build_ledger

    _init_repository(tmp_path)
    with pytest.raises(InventoryError, match="does not exist"):
        build_ledger(
            5, [_specification(source_paths=["alphaforge/absent.py"])], repository=tmp_path
        )


def test_an_incomplete_specification_is_refused(tmp_path: Path) -> None:
    from alphaforge.evidence.inventory import build_ledger

    _init_repository(tmp_path)
    incomplete = _specification()
    del incomplete["doc_paths"]
    with pytest.raises(InventoryError, match="missing"):
        build_ledger(5, [incomplete], repository=tmp_path)


def test_resolve_commit_returns_a_full_object_id(tmp_path: Path) -> None:
    from alphaforge.evidence.inventory import resolve_commit

    commit = _init_repository(tmp_path)
    assert resolve_commit("HEAD", repository=tmp_path) == commit
    assert len(resolve_commit(commit[:8], repository=tmp_path)) == 40


def test_an_empty_reference_is_refused(tmp_path: Path) -> None:
    from alphaforge.evidence.inventory import resolve_commit

    _init_repository(tmp_path)
    with pytest.raises(InventoryError, match="non-empty"):
        resolve_commit("  ", repository=tmp_path)


def test_a_directory_that_is_not_a_repository_is_refused(tmp_path: Path) -> None:
    from alphaforge.evidence.inventory import resolve_commit

    with pytest.raises(InventoryError):
        resolve_commit("HEAD", repository=tmp_path)
