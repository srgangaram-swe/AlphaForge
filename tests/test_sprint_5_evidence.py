"""Tests for the Sprint 5 close-out evidence bundle (SF-S5-MR10).

The bundle is the sprint's §10 visual evidence. These tests protect the property
that makes it trustworthy: **every number comes from the modules themselves**, so
a figure that disagrees with the code cannot be produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alphaforge.readiness import Verdict, minimal_capital_checklist
from alphaforge.readiness.sprint_5_evidence import (
    CROSSOVER_MEASUREMENTS,
    SPRINT_5_DELIVERY,
    publish_sprint_5_evidence,
    sprint_5_readiness,
)


def test_the_published_verdict_is_the_honest_one() -> None:
    """Nothing is qualified, no paper trading has run, no attestation exists."""
    decision = sprint_5_readiness()
    assert decision.verdict is Verdict.NOT_READY
    assert len(decision.unmet) == len(minimal_capital_checklist().items)


def test_the_bundle_writes_every_artifact(tmp_path: Path) -> None:
    manifest = publish_sprint_5_evidence(tmp_path / "closeout")
    expected = {
        "checklist.json",
        "distribution_crossover.csv",
        "manifest.json",
        "readiness_decision.json",
        "readiness_report.md",
        "sprint_5_closeout.png",
    }
    assert set(manifest["artifacts"]) == expected
    for name in expected:
        assert (tmp_path / "closeout" / name).stat().st_size > 0


def test_republishing_over_existing_evidence_is_refused(tmp_path: Path) -> None:
    """A republish must not silently overwrite evidence something already cites."""
    destination = tmp_path / "closeout"
    publish_sprint_5_evidence(destination)
    with pytest.raises(FileExistsError, match="not empty"):
        publish_sprint_5_evidence(destination)


def test_the_manifest_states_zero_capital_at_risk(tmp_path: Path) -> None:
    manifest = publish_sprint_5_evidence(tmp_path / "closeout")
    assert manifest["capital_at_risk"] == "0"
    assert manifest["verdict"] == "NOT_READY"
    assert manifest["unmet_count"] == manifest["total_items"]


def test_the_manifest_carries_its_limitations(tmp_path: Path) -> None:
    manifest = publish_sprint_5_evidence(tmp_path / "closeout")
    joined = " ".join(manifest["limitations"])
    assert "cannot verify policy or legal attestations" in joined
    assert "no broker connection" in joined
    assert "no paper or live trading is authorized" in joined


def test_the_bundle_is_deterministic(tmp_path: Path) -> None:
    first = publish_sprint_5_evidence(tmp_path / "a")
    second = publish_sprint_5_evidence(tmp_path / "b")
    first.pop("artifacts")
    second.pop("artifacts")
    assert first == second
    assert (tmp_path / "a" / "readiness_decision.json").read_text() == (
        tmp_path / "b" / "readiness_decision.json"
    ).read_text()


def test_the_delivery_record_matches_its_stated_total() -> None:
    """The figure's title claims 330; the data must actually sum to it."""
    assert sum(count for _, count in SPRINT_5_DELIVERY) == 330


def test_the_crossover_data_brackets_break_even() -> None:
    """The panel's claim depends on the measurements spanning 1.0x."""
    speedups = [speedup for _, speedup in CROSSOVER_MEASUREMENTS]
    assert min(speedups) < 1.0 < max(speedups)


def test_the_report_leads_with_unmet_items(tmp_path: Path) -> None:
    publish_sprint_5_evidence(tmp_path / "closeout")
    report = (tmp_path / "closeout" / "readiness_report.md").read_text(encoding="utf-8")
    assert "NOT_READY" in report
    assert "## Unmet" in report


def test_the_decision_json_is_machine_readable(tmp_path: Path) -> None:
    publish_sprint_5_evidence(tmp_path / "closeout")
    payload = json.loads(
        (tmp_path / "closeout" / "readiness_decision.json").read_text(encoding="utf-8")
    )
    assert payload["ready"] is False
    assert payload["unmet_by_category"]
