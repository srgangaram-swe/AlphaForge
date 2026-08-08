"""Tests for the Sprint 5 close-out bundle (SF-S5-MR11 rewrite).

The property under test is the one the original module claimed and did not have:
**every published number is computed from a committed record**, not typed into
the generator.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from alphaforge.evidence.publication import PublicationError, verify_bundle
from alphaforge.readiness import Verdict, minimal_capital_checklist
from alphaforge.readiness import sprint_5_evidence as evidence_module
from alphaforge.readiness.sprint_5_evidence import (
    DELIVERY_LEDGER_PATH,
    RAW_MEASUREMENTS_PATH,
    load_delivery,
    load_measurements,
    publish_sprint_5_evidence,
    sprint_5_readiness,
)


def test_no_measurement_is_hard_coded_in_the_generator() -> None:
    """The original defect, pinned.

    Scans module-level assignments for numeric literals that could be a
    transcribed measurement. Small structural integers (versions, indices) are
    permitted; a float or a large int at module scope is how a measurement gets
    frozen into code and then drifts away from what it describes.
    """
    tree = ast.parse(inspect.getsource(evidence_module))
    offenders: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        for literal in ast.walk(node):
            if isinstance(literal, ast.Constant) and isinstance(literal.value, (int, float)):
                if isinstance(literal.value, bool):
                    continue
                if isinstance(literal.value, float) or literal.value > 2026:
                    offenders.append(repr(literal.value))
    assert offenders == [], f"module-level numeric literals could be transcribed data: {offenders}"


def test_the_committed_raw_records_exist_and_parse() -> None:
    assert RAW_MEASUREMENTS_PATH.is_file()
    assert DELIVERY_LEDGER_PATH.is_file()
    assert load_measurements().measurements
    assert load_delivery().slices


def test_a_missing_raw_record_refuses_rather_than_defaulting(tmp_path: Path) -> None:
    """Silently substituting defaults is how a plot stops describing anything."""
    with pytest.raises(FileNotFoundError, match="will not substitute defaults"):
        load_measurements(tmp_path / "absent.json")


def test_the_committed_measurements_meet_the_schema() -> None:
    evidence = load_measurements()
    for measurement in evidence.measurements:
        assert len(measurement.serial_ns) >= 7
        assert measurement.warmups >= 1
        assert len(measurement.parity_identity) == 64
    assert evidence.limitations


def test_the_committed_measurements_bracket_break_even() -> None:
    """The crossover claim depends on measuring both sides of it."""
    bounds = load_measurements().crossover_bounds_ms()
    assert bounds is not None
    assert bounds[0] < bounds[1]


def test_the_ledger_reconciles_with_its_own_total() -> None:
    ledger = load_delivery()
    assert ledger.total_test_functions == sum(item.test_function_count for item in ledger.slices)


def test_the_published_verdict_is_the_honest_one() -> None:
    decision = sprint_5_readiness()
    assert decision.verdict is Verdict.NOT_READY
    assert len(decision.unmet) == len(minimal_capital_checklist().items)


def test_the_bundle_publishes_and_verifies(tmp_path: Path) -> None:
    manifest = publish_sprint_5_evidence(tmp_path / "closeout", source_ref="test")
    assert manifest["verdict"] == "NOT_READY"
    assert manifest["capital_at_risk"] == "0"
    verify_bundle(tmp_path / "closeout")


def test_two_regenerations_are_byte_identical(tmp_path: Path) -> None:
    """Including the PNG: matplotlib's default metadata would otherwise differ."""
    publish_sprint_5_evidence(tmp_path / "a", source_ref="test")
    publish_sprint_5_evidence(tmp_path / "b", source_ref="test")
    for name in (
        "readiness_decision.json",
        "checklist.json",
        "crossover_summary.csv",
        "delivery_ledger.json",
        "sprint_5_closeout.png",
    ):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name


def test_changed_raw_inputs_change_the_bundle(tmp_path: Path) -> None:
    baseline = publish_sprint_5_evidence(tmp_path / "baseline", source_ref="test")
    payload = json.loads(RAW_MEASUREMENTS_PATH.read_text(encoding="utf-8"))
    payload["measurements"][0]["serial_ns"] = [
        sample + 1_000 for sample in payload["measurements"][0]["serial_ns"]
    ]
    altered_path = tmp_path / "altered.json"
    altered_path.write_text(json.dumps(payload), encoding="utf-8")
    altered = publish_sprint_5_evidence(
        tmp_path / "altered", measurements_path=altered_path, source_ref="test"
    )
    assert altered["input_identities"] != baseline["input_identities"]


def test_the_manifest_records_its_input_identities(tmp_path: Path) -> None:
    manifest = publish_sprint_5_evidence(tmp_path / "closeout", source_ref="test")
    assert set(manifest["input_identities"]) == {
        "crossover_measurements",
        "delivery_ledger",
        "readiness_checklist",
    }


def test_publication_refuses_an_existing_bundle(tmp_path: Path) -> None:
    publish_sprint_5_evidence(tmp_path / "closeout", source_ref="test")
    with pytest.raises(PublicationError, match="never overwritten"):
        publish_sprint_5_evidence(tmp_path / "closeout", source_ref="test")


def test_the_manifest_states_its_limitations(tmp_path: Path) -> None:
    manifest = publish_sprint_5_evidence(tmp_path / "closeout", source_ref="test")
    joined = " ".join(manifest["limitations"])
    assert "service-level objective" in joined
    assert "cannot verify" in joined
    assert "no paper or live trading is authorized" in joined
