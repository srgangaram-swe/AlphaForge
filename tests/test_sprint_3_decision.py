"""Governed evidence synthesis tests for SF-S3-MR11."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from alphaforge.research.sprint_3_decision import (
    EVIDENCE_GATES,
    SPRINT_3_FAMILIES,
    EvidenceGateSupport,
    FamilyEvidence,
    SourceArtifact,
    Sprint3DecisionError,
    Sprint3EvaluationPlan,
    Sprint3GlobalEvidence,
    evaluate_sprint_3,
    family_evidence_frame,
    gate_matrix_frame,
    load_sprint_3_evaluation_plan,
    plot_gate_matrix,
    publish_sprint_3_decision,
    sha256_file,
)


def _source(tmp_path: Path, name: str = "evidence.json") -> SourceArtifact:
    path = tmp_path / name
    path.write_text('{"aggregate":true}\n', encoding="utf-8")
    return SourceArtifact(path=name, sha256=sha256_file(path))


def _family(source: SourceArtifact, **updates: Any) -> FamilyEvidence:
    values: dict[str, Any] = {
        "family": "deep_sequence",
        "context": "historical_engineering",
        "evaluated": True,
        "disposition": "reject",
        "reason": "the declared single-fold study did not establish a persistent net edge",
        "sources": (source,),
        "out_of_sample": True,
        "uncertainty": False,
        "net_economics": True,
        "selection_correction": False,
        "feature_ablation": False,
        "randomized_control": False,
        "regime_stability": False,
        "year_stability": False,
        "compute_accounting": True,
        "model_count": 6,
        "blocked_or_failed_count": 0,
    }
    values.update(updates)
    if "gate_support" not in updates:
        values["gate_support"] = tuple(
            EvidenceGateSupport(
                gate=gate,
                source_path=source.path,
                locator_kind="json_pointer",
                locator="/aggregate",
                claim=f"bounded fixture supports {gate}",
            )
            for gate in EVIDENCE_GATES
            if values[gate]
        )
    return FamilyEvidence(**values)


def _global(**updates: Any) -> Sprint3GlobalEvidence:
    values: dict[str, Any] = {
        "complete_point_in_time_data": False,
        "current_market_data": False,
        "complete_corporate_actions": False,
        "complete_delistings_and_symbol_history": False,
        "point_in_time_universe": False,
        "calibrated_execution_costs": False,
        "paper_shadow_period_complete": False,
        "broker_failure_rehearsal_complete": False,
    }
    values.update(updates)
    return Sprint3GlobalEvidence(**values)


def _complete_families(
    source: SourceArtifact,
    *overrides: FamilyEvidence,
) -> tuple[FamilyEvidence, ...]:
    by_name = {family.family: family for family in overrides}
    return tuple(by_name.get(name, _family(source, family=name)) for name in SPRINT_3_FAMILIES)


def _write_plan(
    tmp_path: Path,
    families: tuple[FamilyEvidence, ...],
    *,
    updates: dict[str, Any] | None = None,
) -> Sprint3EvaluationPlan:
    family_documents = json.loads(json.dumps([asdict(family) for family in families]))
    document = {
        "schema_version": "1.0.0",
        "study_id": "signal-foundry-sprint-3-final",
        "issue_number": 37,
        "frozen_at_utc": "2026-07-26T18:00:00Z",
        "protocol_dimensions": {
            name: {
                "status": "frozen",
                "reason": "the bounded fixture and synthesis plan fix this dimension",
                "source_paths": ["evidence.json", "sprint_3_decision.yaml"],
            }
            for name in (
                "configurations",
                "trial_family",
                "ablations",
                "compute_budgets",
                "folds",
                "costs",
                "decision_thresholds",
            )
        },
        "synthesis_policy": {
            "family_order": [family.family for family in families],
            "evidence_gates": list(EVIDENCE_GATES),
            "advance_requires_all_reported_gates": True,
            "no_cross_context_ranking": True,
            "final_holdout_reopened": False,
            "aggregate_only_publication": True,
            "maximum_families": 64,
        },
        "families": family_documents,
        "global_evidence": {
            **asdict(_global()),
            "executable_orders_emitted": False,
            "capital_deployed": False,
        },
    }
    document["protocol_dimensions"]["randomized_controls"] = {
        "status": "deferred",
        "reason": "the heterogeneous fixture has no valid randomized control",
        "source_paths": [],
    }
    if updates:
        document.update(updates)
    path = tmp_path / "sprint_3_decision.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return load_sprint_3_evaluation_plan(path, repository_root=tmp_path)


def test_source_artifact_detects_mutation_and_path_escape(tmp_path: Path) -> None:
    source = _source(tmp_path)

    assert source.verify(tmp_path) == tmp_path / "evidence.json"
    (tmp_path / "evidence.json").write_text('{"aggregate":false}\n', encoding="utf-8")
    with pytest.raises(Sprint3DecisionError, match="digest mismatch"):
        source.verify(tmp_path)
    with pytest.raises(Sprint3DecisionError, match="repository-relative"):
        SourceArtifact(path="../evidence.json", sha256="0" * 64)
    target = tmp_path / "target.json"
    target.write_text('{"aggregate":true}\n', encoding="utf-8")
    (tmp_path / "linked.json").symlink_to(target)
    linked = SourceArtifact(path="linked.json", sha256=sha256_file(target))
    with pytest.raises(Sprint3DecisionError, match="symlink"):
        linked.verify(tmp_path)


def test_gate_support_resolves_json_csv_and_markdown_locators(tmp_path: Path) -> None:
    json_source = _source(tmp_path)
    json_support = EvidenceGateSupport(
        gate="out_of_sample",
        source_path=json_source.path,
        locator_kind="json_pointer",
        locator="/aggregate",
        claim="the aggregate fixture exists",
    )
    json_support.verify(tmp_path, (json_source,))
    with pytest.raises(Sprint3DecisionError, match="does not resolve"):
        replace(json_support, locator="/missing").verify(tmp_path, (json_source,))

    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text("rank_ic,fold\n0.1,1\n", encoding="utf-8")
    csv_source = SourceArtifact(path="metrics.csv", sha256=sha256_file(csv_path))
    EvidenceGateSupport(
        gate="uncertainty",
        source_path=csv_source.path,
        locator_kind="csv_column",
        locator="rank_ic",
        claim="the aggregate table reports rank IC",
    ).verify(tmp_path, (csv_source,))

    markdown_path = tmp_path / "report.md"
    markdown_path.write_text("# Report\n\n## Limitations\n", encoding="utf-8")
    markdown_source = SourceArtifact(
        path="report.md",
        sha256=sha256_file(markdown_path),
    )
    EvidenceGateSupport(
        gate="compute_accounting",
        source_path=markdown_source.path,
        locator_kind="markdown_heading",
        locator="## Limitations",
        claim="the report contains a limitations section",
    ).verify(tmp_path, (markdown_source,))


def test_family_records_missing_gates_without_treating_them_as_not_applicable(
    tmp_path: Path,
) -> None:
    family = _family(_source(tmp_path))

    assert family.gate_score == 3
    assert family.missing_gates == (
        "uncertainty",
        "selection_correction",
        "feature_ablation",
        "randomized_control",
        "regime_stability",
        "year_stability",
    )
    frame = family_evidence_frame((family,))
    assert frame.loc[0, "gate_score"] == 3
    assert frame.loc[0, "context"] == "historical_engineering"
    assert "regime_stability" in frame.loc[0, "missing_gates"]
    with pytest.raises(Sprint3DecisionError, match="must report every"):
        replace(family, disposition="advance")


def test_unsupported_family_must_be_deferred_and_linked(tmp_path: Path) -> None:
    source = _source(tmp_path)
    deferred = _family(
        source,
        family="state_space",
        context="unsupported",
        evaluated=False,
        disposition="defer",
        reason="no bounded implementation or matched evidence exists",
        out_of_sample=False,
        net_economics=False,
        compute_accounting=False,
        model_count=0,
    )
    assert deferred.disposition == "defer"

    with pytest.raises(Sprint3DecisionError, match="cannot be marked evaluated"):
        replace(deferred, evaluated=True, disposition="reject")
    with pytest.raises(Sprint3DecisionError, match="must link"):
        replace(deferred, sources=())


def test_complete_family_names_are_exact_and_ordered(tmp_path: Path) -> None:
    source = _source(tmp_path)
    families = _complete_families(source)
    decision = evaluate_sprint_3(families, _global())

    assert decision.decision == "NOT_READY"
    assert decision.evaluated_families == len(SPRINT_3_FAMILIES)
    assert decision.rejected_families == len(SPRINT_3_FAMILIES)
    with pytest.raises(Sprint3DecisionError, match="exactly match"):
        evaluate_sprint_3(families[:-1], _global())


def test_synthesis_never_grants_paper_or_order_authority(
    tmp_path: Path,
) -> None:
    family = _family(
        _source(tmp_path),
        disposition="advance",
        reason="all pre-registered research evidence gates passed",
        uncertainty=True,
        selection_correction=True,
        feature_ablation=True,
        randomized_control=True,
        regime_stability=True,
        year_stability=True,
    )
    families = _complete_families(_source(tmp_path, "complete.json"), family)
    rejected = evaluate_sprint_3(families, _global())

    assert rejected.decision == "NOT_READY"
    assert "complete_point_in_time_data" in rejected.failed_readiness_gates
    still_rejected = evaluate_sprint_3(
        families,
        _global(
            complete_point_in_time_data=True,
            current_market_data=True,
            complete_corporate_actions=True,
            complete_delistings_and_symbol_history=True,
            point_in_time_universe=True,
            calibrated_execution_costs=True,
            paper_shadow_period_complete=True,
            broker_failure_rehearsal_complete=True,
        ),
    )
    assert still_rejected.decision == "NOT_READY"
    assert still_rejected.failed_readiness_gates == (
        "sprint_3_synthesis_has_no_paper_or_order_authority",
    )


def test_order_or_capital_claim_is_rejected() -> None:
    with pytest.raises(Sprint3DecisionError, match="must not emit"):
        _global(executable_orders_emitted=True)
    with pytest.raises(Sprint3DecisionError, match="must not emit"):
        _global(capital_deployed=True)


def test_gate_matrix_contains_every_family_gate(tmp_path: Path) -> None:
    source = _source(tmp_path)
    families = (
        _family(source),
        _family(
            source,
            family="latent",
            context="synthetic_engineering",
            disposition="reject",
            reason="mechanics passed but incremental evidence remained incomplete",
        ),
    )

    frame = gate_matrix_frame(families)

    assert len(frame) == len(families) * len(EVIDENCE_GATES)
    assert tuple(frame["family"].drop_duplicates()) == ("deep_sequence", "latent")
    assert set(frame["gate"]) == set(EVIDENCE_GATES)


def test_plot_uses_seaborn_heatmap_and_discloses_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    calls: list[str] = []
    from alphaforge.research import sprint_3_decision as module

    original = module.sns.heatmap

    def recording_heatmap(*args, **kwargs):
        calls.append("heatmap")
        return original(*args, **kwargs)

    monkeypatch.setattr(module.sns, "heatmap", recording_heatmap)
    output = tmp_path / "coverage.png"

    plot_gate_matrix((_family(source),), output)

    assert calls == ["heatmap"]
    assert output.stat().st_size > 10_000


def test_plan_loader_freezes_order_gates_and_content_hashes(tmp_path: Path) -> None:
    source = _source(tmp_path)
    plan = _write_plan(tmp_path, _complete_families(source))

    assert plan.family_order == SPRINT_3_FAMILIES
    assert plan.plan_id == sha256_file(tmp_path / "sprint_3_decision.yaml")
    assert plan.protocol_dimensions.deferred_dimensions == ("randomized_controls",)

    document = yaml.safe_load((tmp_path / "sprint_3_decision.yaml").read_text(encoding="utf-8"))
    document["synthesis_policy"]["family_order"] = ["changed_after_results"]
    (tmp_path / "sprint_3_decision.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(Sprint3DecisionError, match="family_order"):
        load_sprint_3_evaluation_plan(
            tmp_path / "sprint_3_decision.yaml",
            repository_root=tmp_path,
        )


def test_plan_loader_rejects_unknown_fields_and_unsupported_freeze_claims(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    families = _complete_families(source)
    with pytest.raises(Sprint3DecisionError, match="extra_forbidden"):
        _write_plan(tmp_path, families, updates={"undeclared": True})

    _write_plan(tmp_path, families)
    payload = yaml.safe_load((tmp_path / "sprint_3_decision.yaml").read_text(encoding="utf-8"))
    payload["protocol_dimensions"]["costs"]["source_paths"] = []
    (tmp_path / "sprint_3_decision.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(Sprint3DecisionError, match="requires source paths"):
        load_sprint_3_evaluation_plan(
            tmp_path / "sprint_3_decision.yaml",
            repository_root=tmp_path,
        )


def test_plan_loader_rejects_symlinked_plan(tmp_path: Path) -> None:
    source = _source(tmp_path)
    _write_plan(tmp_path, _complete_families(source))
    linked = tmp_path / "linked-plan.yaml"
    linked.symlink_to(tmp_path / "sprint_3_decision.yaml")

    with pytest.raises(Sprint3DecisionError, match="symlink"):
        load_sprint_3_evaluation_plan(linked, repository_root=tmp_path)


def test_plan_loader_bounds_source_references_and_total_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alphaforge.research import sprint_3_decision as module

    source = _source(tmp_path)
    families = _complete_families(source)
    monkeypatch.setattr(module, "MAX_SOURCE_REFERENCES", 9)
    with pytest.raises(Sprint3DecisionError, match="reference count"):
        _write_plan(tmp_path, families)

    monkeypatch.setattr(module, "MAX_SOURCE_REFERENCES", 128)
    monkeypatch.setattr(module, "MAX_TOTAL_SOURCE_BYTES", 1)
    with pytest.raises(Sprint3DecisionError, match="byte total"):
        _write_plan(tmp_path, families)


def test_plan_loader_hashes_each_unique_source_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alphaforge.research import sprint_3_decision as module

    source = _source(tmp_path)
    calls: list[str] = []
    original = module.SourceArtifact.verify

    def recording_verify(self: SourceArtifact, repository_root: str | Path) -> Path:
        calls.append(self.path)
        return original(self, repository_root)

    monkeypatch.setattr(module.SourceArtifact, "verify", recording_verify)
    _write_plan(tmp_path, _complete_families(source))

    assert calls == ["evidence.json"]


def test_publisher_is_atomic_content_addressed_and_aggregate_only(tmp_path: Path) -> None:
    source = _source(tmp_path)
    plan = _write_plan(tmp_path, _complete_families(source))
    output = tmp_path / "published"

    result = publish_sprint_3_decision(
        repository_root=tmp_path,
        plan=plan,
        output_dir=output,
    )

    assert result == output
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["decision"] == "NOT_READY"
    assert summary["publication_boundary"] == {
        "aggregate_only": True,
        "capital_deployed": False,
        "executable_orders_emitted": False,
        "licensed_observations_published": False,
        "model_weights_published": False,
        "row_predictions_published": False,
    }
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["source_artifacts"]) == len(SPRINT_3_FAMILIES)
    assert manifest["source_artifacts"][0] == {
        "family": "conventional_baselines",
        "path": "evidence.json",
        "sha256": source.sha256,
    }
    assert manifest["plan"]["sha256"] == plan.plan_id
    for relative, record in manifest["artifacts"].items():
        assert sha256_file(output / relative) == record["sha256"]
    assert tuple(pd.read_csv(output / "family_evidence.csv")["family"]) == SPRINT_3_FAMILIES
    replay = tmp_path / "replayed"
    publish_sprint_3_decision(
        repository_root=tmp_path,
        plan=plan,
        output_dir=replay,
    )
    first_files = {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    }
    replay_files = {
        path.relative_to(replay): path.read_bytes() for path in replay.rglob("*") if path.is_file()
    }
    assert replay_files == first_files
    with pytest.raises(FileExistsError):
        publish_sprint_3_decision(
            repository_root=tmp_path,
            plan=plan,
            output_dir=output,
        )


def test_publisher_cleans_staging_after_fault(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alphaforge.research import sprint_3_decision as module

    source = _source(tmp_path)
    plan = _write_plan(tmp_path, _complete_families(source))
    output = tmp_path / "failed-publication"
    original = module._write_json
    calls = 0

    def fail_second_json(path: Path, payload: dict[str, Any]) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected manifest write failure")
        original(path, payload)

    monkeypatch.setattr(module, "_write_json", fail_second_json)
    with pytest.raises(OSError, match="injected"):
        publish_sprint_3_decision(
            repository_root=tmp_path,
            plan=plan,
            output_dir=output,
        )

    assert not output.exists()
    assert not tuple(tmp_path.glob(".failed-publication.*"))
