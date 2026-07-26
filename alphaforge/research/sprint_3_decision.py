"""Governed Sprint 3 evidence synthesis and fail-closed release decision.

Sprint 3 studies do not all share a dataset or evaluation boundary. Historical
WIKI engineering evidence and deterministic synthetic references therefore
must not be collapsed into a model leaderboard. This module instead evaluates
each family against the same *evidence-quality* gates, preserves the study
context, and produces an explicit ``NOT_READY`` decision whenever a required
trading-readiness fact is absent.

The input boundary contains aggregate claims linked to content-hashed source
artifacts. Row-level observations, targets, predictions, model weights, and
credentials are neither accepted nor published.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

SPRINT_3_DECISION_SCHEMA_VERSION = "1.0.0"
MAX_SOURCE_BYTES = 32 * 1024 * 1024
MAX_FAMILIES = 64
MAX_PLAN_BYTES = 1024 * 1024
MAX_SOURCE_REFERENCES = 128
MAX_TOTAL_SOURCE_BYTES = 64 * 1024 * 1024
_HASH_CHUNK_BYTES = 1024 * 1024

EvidenceContext = Literal[
    "historical_engineering",
    "synthetic_engineering",
    "unsupported",
]
FamilyDisposition = Literal["advance", "reject", "defer"]
ReadinessDecision = Literal["NOT_READY"]
EvidenceLocatorKind = Literal["json_pointer", "csv_column", "markdown_heading"]

EVIDENCE_GATES = (
    "out_of_sample",
    "uncertainty",
    "net_economics",
    "selection_correction",
    "feature_ablation",
    "randomized_control",
    "regime_stability",
    "year_stability",
    "compute_accounting",
)
SPRINT_3_FAMILIES = (
    "conventional_baselines",
    "spectral_descriptors",
    "adaptive_decomposition",
    "regime_change_points",
    "state_space",
    "deep_sequence",
    "time_frequency_vision",
    "latent_representations",
    "governed_ensembles",
    "abstention_policy",
)


class Sprint3DecisionError(ValueError):
    """Raised when Sprint 3 evidence is incomplete, unsafe, or inconsistent."""


class _StrictPlanModel(BaseModel):
    """Immutable, non-coercing base for the frozen synthesis configuration."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _safe_identifier(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or value != value.strip()
        or not value.isascii()
        or not all(character.isalnum() or character in "._-" for character in value)
    ):
        raise Sprint3DecisionError(
            f"{field} must be a non-empty safe ASCII identifier of at most 128 characters"
        )
    return value


def sha256_file(path: str | Path, *, max_bytes: int = MAX_SOURCE_BYTES) -> str:
    """Hash one bounded regular file without following a symlink."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise Sprint3DecisionError("max_bytes must be a positive integer")
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise Sprint3DecisionError(f"evidence source must be a regular file: {source}")
    size = source.stat().st_size
    if not 0 < size <= max_bytes:
        raise Sprint3DecisionError(f"evidence source has invalid byte length {size}: {source}")
    digest = hashlib.sha256()
    total = 0
    with source.open("rb") as stream:
        while chunk := stream.read(min(_HASH_CHUNK_BYTES, max_bytes - total + 1)):
            total += len(chunk)
            if total > max_bytes:
                raise Sprint3DecisionError(
                    f"evidence source exceeds maximum byte length {max_bytes}: {source}"
                )
            digest.update(chunk)
    if total != size:
        raise Sprint3DecisionError(f"evidence source byte length changed during hashing: {source}")
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceArtifact:
    """Content-addressed aggregate source supporting one family claim."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, str)
            or not self.path
            or len(self.path) > 512
            or "\x00" in self.path
            or not self.path.isascii()
            or Path(self.path).is_absolute()
            or ".." in Path(self.path).parts
        ):
            raise Sprint3DecisionError(
                "source artifact path must be a safe repository-relative path"
            )
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise Sprint3DecisionError("source artifact sha256 must be lowercase hex")

    def verify(self, root: str | Path) -> Path:
        """Return the verified source path or fail on mutation."""

        root_path = Path(root)
        if root_path.is_symlink():
            raise Sprint3DecisionError("repository root must not be a symlink")
        repository = root_path.resolve()
        source = repository
        for component in Path(self.path).parts:
            source = source / component
            if source.is_symlink():
                raise Sprint3DecisionError(f"source artifact path traverses a symlink: {self.path}")
        source = source.resolve()
        try:
            source.relative_to(repository)
        except ValueError as exc:
            raise Sprint3DecisionError("source artifact escapes repository root") from exc
        observed = sha256_file(source)
        if observed != self.sha256:
            raise Sprint3DecisionError(
                f"source artifact digest mismatch for {self.path}: "
                f"expected {self.sha256}, observed {observed}"
            )
        return source


class _SourceArtifactSpec(_StrictPlanModel):
    """Strict YAML representation of one content-addressed source."""

    path: str
    sha256: str

    def to_domain(self) -> SourceArtifact:
        """Return the validated immutable domain object."""

        return SourceArtifact(path=self.path, sha256=self.sha256)


@dataclass(frozen=True)
class EvidenceGateSupport:
    """One verifiable locator supporting a positive evidence-gate claim."""

    gate: str
    source_path: str
    locator_kind: EvidenceLocatorKind
    locator: str
    claim: str

    def __post_init__(self) -> None:
        if self.gate not in EVIDENCE_GATES:
            raise Sprint3DecisionError(f"unsupported evidence gate {self.gate!r}")
        SourceArtifact(path=self.source_path, sha256="0" * 64)
        if self.locator_kind not in {
            "json_pointer",
            "csv_column",
            "markdown_heading",
        }:
            raise Sprint3DecisionError(f"unsupported evidence locator kind {self.locator_kind!r}")
        if not isinstance(self.locator, str) or not self.locator or len(self.locator) > 512:
            raise Sprint3DecisionError("evidence locator must contain at most 512 characters")
        if not isinstance(self.claim, str) or not self.claim.strip() or len(self.claim) > 1000:
            raise Sprint3DecisionError("evidence support requires a bounded explicit claim")
        if self.locator_kind == "json_pointer" and not self.locator.startswith("/"):
            raise Sprint3DecisionError("JSON evidence locator must be an absolute JSON pointer")
        if self.locator_kind == "markdown_heading" and not self.locator.startswith("#"):
            raise Sprint3DecisionError("Markdown evidence locator must be an exact heading")

    def verify(
        self,
        repository_root: str | Path,
        sources: tuple[SourceArtifact, ...],
        *,
        verified_sources: Mapping[str, Path] | None = None,
    ) -> None:
        """Verify that the declared source contains the addressed evidence."""

        by_path = {source.path: source for source in sources}
        if self.source_path not in by_path:
            raise Sprint3DecisionError(
                f"gate {self.gate!r} references undeclared source {self.source_path!r}"
            )
        if verified_sources is None:
            path = by_path[self.source_path].verify(repository_root)
        else:
            cached_path = verified_sources.get(self.source_path)
            if cached_path is None:
                raise Sprint3DecisionError(
                    f"gate {self.gate!r} source has not passed content verification"
                )
            path = cached_path
        if self.locator_kind == "json_pointer":
            try:
                value: Any = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise Sprint3DecisionError(f"gate {self.gate!r} source is not valid JSON") from exc
            for token in self.locator[1:].split("/"):
                key = token.replace("~1", "/").replace("~0", "~")
                if isinstance(value, dict) and key in value:
                    value = value[key]
                elif isinstance(value, list) and key.isdigit() and int(key) < len(value):
                    value = value[int(key)]
                else:
                    raise Sprint3DecisionError(f"gate {self.gate!r} JSON pointer does not resolve")
        elif self.locator_kind == "csv_column":
            try:
                with path.open("r", encoding="utf-8", newline="") as stream:
                    columns = next(csv.reader(stream))
            except (OSError, UnicodeError, StopIteration, csv.Error) as exc:
                raise Sprint3DecisionError(
                    f"gate {self.gate!r} source has no valid CSV header"
                ) from exc
            if self.locator not in columns:
                raise Sprint3DecisionError(f"gate {self.gate!r} CSV column does not exist")
        else:
            try:
                headings = {
                    line.strip()
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.startswith("#")
                }
            except (OSError, UnicodeError) as exc:
                raise Sprint3DecisionError(
                    f"gate {self.gate!r} source is not valid UTF-8 Markdown"
                ) from exc
            if self.locator not in headings:
                raise Sprint3DecisionError(f"gate {self.gate!r} Markdown heading does not exist")


class _EvidenceGateSupportSpec(_StrictPlanModel):
    """Strict YAML representation of one gate-to-source binding."""

    gate: str
    source_path: str
    locator_kind: EvidenceLocatorKind
    locator: str
    claim: str

    @model_validator(mode="after")
    def validate_domain_contract(self) -> _EvidenceGateSupportSpec:
        self.to_domain()
        return self

    def to_domain(self) -> EvidenceGateSupport:
        """Return the immutable evidence support record."""

        return EvidenceGateSupport(**self.model_dump())


@dataclass(frozen=True)
class FamilyEvidence:
    """Normalized aggregate evidence for one pre-registered family.

    Boolean evidence gates describe whether the linked source actually reports
    the named evidence. A false value is a visible missing gate, never silently
    treated as not applicable. Optional performance values remain contextual
    and are not ranked across studies.
    """

    family: str
    context: EvidenceContext
    evaluated: bool
    disposition: FamilyDisposition
    reason: str
    sources: tuple[SourceArtifact, ...]
    gate_support: tuple[EvidenceGateSupport, ...]
    out_of_sample: bool
    uncertainty: bool
    net_economics: bool
    selection_correction: bool
    feature_ablation: bool
    randomized_control: bool
    regime_stability: bool
    year_stability: bool
    compute_accounting: bool
    model_count: int = 0
    blocked_or_failed_count: int = 0

    def __post_init__(self) -> None:
        _safe_identifier(self.family, "family")
        if self.context not in {
            "historical_engineering",
            "synthetic_engineering",
            "unsupported",
        }:
            raise Sprint3DecisionError(f"unsupported evidence context {self.context!r}")
        if self.disposition not in {"advance", "reject", "defer"}:
            raise Sprint3DecisionError(f"unsupported disposition {self.disposition!r}")
        if not isinstance(self.reason, str) or not self.reason.strip() or len(self.reason) > 2000:
            raise Sprint3DecisionError(
                "family reason must be non-empty and contain at most 2000 characters"
            )
        if not self.sources:
            raise Sprint3DecisionError("every family must link at least one aggregate source")
        source_paths = {source.path for source in self.sources}
        support_gates = tuple(support.gate for support in self.gate_support)
        if len(support_gates) != len(set(support_gates)):
            raise Sprint3DecisionError("evidence gate support records must be unique")
        if any(support.source_path not in source_paths for support in self.gate_support):
            raise Sprint3DecisionError(
                "evidence gate support must reference a declared family source"
            )
        if self.context == "unsupported" and self.evaluated:
            raise Sprint3DecisionError("unsupported families cannot be marked evaluated")
        if not self.evaluated and self.disposition != "defer":
            raise Sprint3DecisionError("unevaluated families must be deferred")
        if self.disposition == "advance" and not self.evaluated:
            raise Sprint3DecisionError("only evaluated families may advance")
        for field in EVIDENCE_GATES:
            if not isinstance(getattr(self, field), bool):
                raise Sprint3DecisionError(f"{field} must be a boolean")
            if bool(getattr(self, field)) != (field in support_gates):
                raise Sprint3DecisionError(
                    f"{field} must be true exactly when a support locator is declared"
                )
        if self.disposition == "advance" and self.missing_gates:
            raise Sprint3DecisionError(
                "an advanced research family must report every governed evidence gate"
            )
        if (
            isinstance(self.model_count, bool)
            or not 0 <= self.model_count <= 100_000
            or isinstance(self.blocked_or_failed_count, bool)
            or not 0 <= self.blocked_or_failed_count <= self.model_count
        ):
            raise Sprint3DecisionError("family model counts are invalid")

    @property
    def gate_score(self) -> int:
        """Return the count of reported evidence gates, not a quality ranking."""

        return sum(bool(getattr(self, gate)) for gate in EVIDENCE_GATES)

    @property
    def missing_gates(self) -> tuple[str, ...]:
        """Return evidence gates absent from the linked source."""

        return tuple(gate for gate in EVIDENCE_GATES if not getattr(self, gate))


class _FamilyEvidenceSpec(_StrictPlanModel):
    """Strict YAML declaration for one complete representation family."""

    family: str
    context: EvidenceContext
    evaluated: bool
    disposition: FamilyDisposition
    reason: str
    sources: Annotated[list[_SourceArtifactSpec], Field(min_length=1, max_length=32)]
    gate_support: Annotated[list[_EvidenceGateSupportSpec], Field(max_length=32)]
    out_of_sample: bool
    uncertainty: bool
    net_economics: bool
    selection_correction: bool
    feature_ablation: bool
    randomized_control: bool
    regime_stability: bool
    year_stability: bool
    compute_accounting: bool
    model_count: Annotated[int, Field(ge=0, le=100_000)] = 0
    blocked_or_failed_count: Annotated[int, Field(ge=0, le=100_000)] = 0

    @model_validator(mode="after")
    def validate_domain_contract(self) -> _FamilyEvidenceSpec:
        self.to_domain()
        return self

    def to_domain(self) -> FamilyEvidence:
        """Return the fully validated immutable family evidence."""

        return FamilyEvidence(
            family=self.family,
            context=self.context,
            evaluated=self.evaluated,
            disposition=self.disposition,
            reason=self.reason,
            sources=tuple(source.to_domain() for source in self.sources),
            gate_support=tuple(support.to_domain() for support in self.gate_support),
            out_of_sample=self.out_of_sample,
            uncertainty=self.uncertainty,
            net_economics=self.net_economics,
            selection_correction=self.selection_correction,
            feature_ablation=self.feature_ablation,
            randomized_control=self.randomized_control,
            regime_stability=self.regime_stability,
            year_stability=self.year_stability,
            compute_accounting=self.compute_accounting,
            model_count=self.model_count,
            blocked_or_failed_count=self.blocked_or_failed_count,
        )


@dataclass(frozen=True)
class Sprint3GlobalEvidence:
    """Facts that apply to paper/live readiness across every family."""

    complete_point_in_time_data: bool
    current_market_data: bool
    complete_corporate_actions: bool
    complete_delistings_and_symbol_history: bool
    point_in_time_universe: bool
    calibrated_execution_costs: bool
    paper_shadow_period_complete: bool
    broker_failure_rehearsal_complete: bool
    executable_orders_emitted: bool = False
    capital_deployed: bool = False

    def __post_init__(self) -> None:
        for field, value in asdict(self).items():
            if not isinstance(value, bool):
                raise Sprint3DecisionError(f"global evidence {field} must be boolean")
        if self.executable_orders_emitted or self.capital_deployed:
            raise Sprint3DecisionError(
                "Sprint 3 evidence must not emit executable orders or deploy capital"
            )

    @property
    def failed_readiness_gates(self) -> tuple[str, ...]:
        """Return the paper-readiness facts that remain false."""

        required = (
            "complete_point_in_time_data",
            "current_market_data",
            "complete_corporate_actions",
            "complete_delistings_and_symbol_history",
            "point_in_time_universe",
            "calibrated_execution_costs",
            "paper_shadow_period_complete",
            "broker_failure_rehearsal_complete",
        )
        return tuple(field for field in required if not getattr(self, field))


class _GlobalEvidenceSpec(_StrictPlanModel):
    """Strict YAML declaration for cross-family paper/live readiness facts."""

    complete_point_in_time_data: bool
    current_market_data: bool
    complete_corporate_actions: bool
    complete_delistings_and_symbol_history: bool
    point_in_time_universe: bool
    calibrated_execution_costs: bool
    paper_shadow_period_complete: bool
    broker_failure_rehearsal_complete: bool
    executable_orders_emitted: Literal[False]
    capital_deployed: Literal[False]

    @model_validator(mode="after")
    def validate_domain_contract(self) -> _GlobalEvidenceSpec:
        self.to_domain()
        return self

    def to_domain(self) -> Sprint3GlobalEvidence:
        """Return the immutable fail-closed readiness facts."""

        return Sprint3GlobalEvidence(**self.model_dump())


@dataclass(frozen=True)
class ProtocolDimension:
    """Evidence-backed status of one required synthesis-protocol dimension."""

    status: Literal["frozen", "deferred"]
    reason: str
    source_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in {"frozen", "deferred"}:
            raise Sprint3DecisionError(f"unsupported protocol status {self.status!r}")
        if not isinstance(self.reason, str) or not self.reason.strip() or len(self.reason) > 2000:
            raise Sprint3DecisionError("protocol status requires an explicit bounded reason")
        for path in self.source_paths:
            SourceArtifact(path=path, sha256="0" * 64)
        if self.status == "frozen" and not self.source_paths:
            raise Sprint3DecisionError("a frozen protocol dimension requires source paths")


@dataclass(frozen=True)
class ProtocolDeclaration:
    """Complete, honest issue-#37 protocol status before final synthesis."""

    configurations: ProtocolDimension
    trial_family: ProtocolDimension
    ablations: ProtocolDimension
    randomized_controls: ProtocolDimension
    compute_budgets: ProtocolDimension
    folds: ProtocolDimension
    costs: ProtocolDimension
    decision_thresholds: ProtocolDimension

    @property
    def deferred_dimensions(self) -> tuple[str, ...]:
        """Return dimensions explicitly re-scoped instead of claimed complete."""

        return tuple(
            name for name, dimension in asdict(self).items() if dimension["status"] == "deferred"
        )


class _ProtocolDimensionSpec(_StrictPlanModel):
    """Strict YAML record for one frozen or explicitly deferred dimension."""

    status: Literal["frozen", "deferred"]
    reason: str
    source_paths: Annotated[list[str], Field(max_length=64)]

    @model_validator(mode="after")
    def validate_domain_contract(self) -> _ProtocolDimensionSpec:
        self.to_domain()
        return self

    def to_domain(self) -> ProtocolDimension:
        """Return the immutable protocol dimension."""

        return ProtocolDimension(
            status=self.status,
            reason=self.reason,
            source_paths=tuple(self.source_paths),
        )


class _ProtocolDeclarationSpec(_StrictPlanModel):
    """Machine-checkable protocol status for every issue-#37 dimension."""

    configurations: _ProtocolDimensionSpec
    trial_family: _ProtocolDimensionSpec
    ablations: _ProtocolDimensionSpec
    randomized_controls: _ProtocolDimensionSpec
    compute_budgets: _ProtocolDimensionSpec
    folds: _ProtocolDimensionSpec
    costs: _ProtocolDimensionSpec
    decision_thresholds: _ProtocolDimensionSpec

    def to_domain(self) -> ProtocolDeclaration:
        """Return the immutable protocol declaration."""

        return ProtocolDeclaration(
            configurations=self.configurations.to_domain(),
            trial_family=self.trial_family.to_domain(),
            ablations=self.ablations.to_domain(),
            randomized_controls=self.randomized_controls.to_domain(),
            compute_budgets=self.compute_budgets.to_domain(),
            folds=self.folds.to_domain(),
            costs=self.costs.to_domain(),
            decision_thresholds=self.decision_thresholds.to_domain(),
        )


class _SynthesisPolicy(_StrictPlanModel):
    """Fail-closed policy for comparing evidence without ranking contexts."""

    family_order: Annotated[list[str], Field(min_length=1, max_length=MAX_FAMILIES)]
    evidence_gates: Annotated[list[str], Field(min_length=1, max_length=64)]
    advance_requires_all_reported_gates: Literal[True]
    no_cross_context_ranking: Literal[True]
    final_holdout_reopened: Literal[False]
    aggregate_only_publication: Literal[True]
    maximum_families: Literal[64]


class _Sprint3PlanSpec(_StrictPlanModel):
    """Complete strict schema for the pre-publication decision plan."""

    schema_version: Literal["1.0.0"]
    study_id: str
    issue_number: Literal[37]
    frozen_at_utc: str
    protocol_dimensions: _ProtocolDeclarationSpec
    synthesis_policy: _SynthesisPolicy
    families: Annotated[list[_FamilyEvidenceSpec], Field(min_length=1, max_length=MAX_FAMILIES)]
    global_evidence: _GlobalEvidenceSpec

    @model_validator(mode="after")
    def validate_complete_plan(self) -> _Sprint3PlanSpec:
        _safe_identifier(self.study_id, "study_id")
        try:
            parsed = pd.Timestamp(self.frozen_at_utc)
        except ValueError as exc:
            raise ValueError("frozen_at_utc must be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None or not self.frozen_at_utc.endswith("Z"):
            raise ValueError("frozen_at_utc must be an explicit UTC timestamp ending in Z")
        if self.synthesis_policy.evidence_gates != list(EVIDENCE_GATES):
            raise ValueError("synthesis_policy.evidence_gates must match the governed gate order")
        names = [family.family for family in self.families]
        if names != list(SPRINT_3_FAMILIES):
            raise ValueError("families must exactly match the governed Sprint 3 family and order")
        if names != self.synthesis_policy.family_order:
            raise ValueError("family_order must exactly match the declared family sequence")
        if len(names) != len(set(names)):
            raise ValueError("family names must be unique")
        for family in self.families:
            paths = [source.path for source in family.sources]
            if len(paths) != len(set(paths)):
                raise ValueError(f"family {family.family!r} repeats a source artifact")
        return self


@dataclass(frozen=True)
class Sprint3EvaluationPlan:
    """Immutable, content-addressed plan used by the final publisher."""

    schema_version: str
    study_id: str
    issue_number: int
    frozen_at_utc: str
    config_source: SourceArtifact
    protocol_dimensions: ProtocolDeclaration
    family_order: tuple[str, ...]
    families: tuple[FamilyEvidence, ...]
    global_evidence: Sprint3GlobalEvidence

    @property
    def plan_id(self) -> str:
        """Return the SHA-256 identity of the complete frozen YAML document."""

        return self.config_source.sha256


@dataclass(frozen=True)
class Sprint3Decision:
    """Final evidence synthesis with no authority to route an order."""

    decision: ReadinessDecision
    failed_readiness_gates: tuple[str, ...]
    evaluated_families: int
    rejected_families: int
    deferred_families: int
    advanced_research_families: int


def load_sprint_3_evaluation_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> Sprint3EvaluationPlan:
    """Load and verify the bounded, strict Sprint 3 synthesis plan.

    The plan is itself content addressed. Every constituent source hash is
    checked during load so configuration validation fails immediately after a
    source report, aggregate table, or declared synthesis-protocol changes.
    """

    repository_input = Path(repository_root)
    if repository_input.is_symlink():
        raise Sprint3DecisionError("repository_root must not be a symlink")
    repository = repository_input.resolve()
    if not repository.is_dir():
        raise Sprint3DecisionError("repository_root must be a real directory")
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repository / candidate
    lexical_path = Path(os.path.abspath(candidate))
    try:
        relative_path = lexical_path.relative_to(repository)
    except ValueError as exc:
        raise Sprint3DecisionError("Sprint 3 plan must remain inside repository_root") from exc
    config_path = repository
    for component in relative_path.parts:
        config_path = config_path / component
        if config_path.is_symlink():
            raise Sprint3DecisionError("Sprint 3 plan path must not traverse a symlink")
    config_path = config_path.resolve()
    relative = relative_path.as_posix()
    if not config_path.is_file():
        raise Sprint3DecisionError("Sprint 3 plan must be a regular file")
    size = config_path.stat().st_size
    if not 0 < size <= MAX_PLAN_BYTES:
        raise Sprint3DecisionError(
            f"Sprint 3 plan has invalid byte length {size}; maximum is {MAX_PLAN_BYTES}"
        )
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise Sprint3DecisionError(f"unable to parse Sprint 3 plan: {exc}") from exc
    if not isinstance(document, Mapping):
        raise Sprint3DecisionError("Sprint 3 plan root must be a mapping")
    try:
        spec = _Sprint3PlanSpec.model_validate(document)
    except ValidationError as exc:
        raise Sprint3DecisionError(f"invalid Sprint 3 plan: {exc}") from exc
    plan = Sprint3EvaluationPlan(
        schema_version=spec.schema_version,
        study_id=spec.study_id,
        issue_number=spec.issue_number,
        frozen_at_utc=spec.frozen_at_utc,
        config_source=SourceArtifact(path=relative, sha256=sha256_file(config_path)),
        protocol_dimensions=spec.protocol_dimensions.to_domain(),
        family_order=tuple(spec.synthesis_policy.family_order),
        families=tuple(family.to_domain() for family in spec.families),
        global_evidence=spec.global_evidence.to_domain(),
    )
    references = [source for family in plan.families for source in family.sources]
    if len(references) > MAX_SOURCE_REFERENCES:
        raise Sprint3DecisionError(f"source reference count exceeds {MAX_SOURCE_REFERENCES}")
    unique_sources: dict[str, str] = {}
    counted_paths: set[str] = set()
    verified_sources: dict[str, Path] = {}
    total_source_bytes = 0
    for source in references:
        previous = unique_sources.setdefault(source.path, source.sha256)
        if previous != source.sha256:
            raise Sprint3DecisionError(f"source artifact {source.path!r} has conflicting digests")
        if source.path in counted_paths:
            continue
        counted_paths.add(source.path)
        verified = source.verify(repository)
        verified_sources[source.path] = verified
        total_source_bytes += verified.stat().st_size
        if total_source_bytes > MAX_TOTAL_SOURCE_BYTES:
            raise Sprint3DecisionError(f"source byte total exceeds {MAX_TOTAL_SOURCE_BYTES}")
    verified_non_plan_sources = set(verified_sources) - {plan.config_source.path}
    allowed_protocol_sources = set(unique_sources) | {plan.config_source.path}
    for name in (
        "configurations",
        "trial_family",
        "ablations",
        "randomized_controls",
        "compute_budgets",
        "folds",
        "costs",
        "decision_thresholds",
    ):
        dimension = getattr(plan.protocol_dimensions, name)
        undeclared = set(dimension.source_paths) - allowed_protocol_sources
        if undeclared:
            raise Sprint3DecisionError(
                f"protocol dimension {name!r} references undeclared sources: "
                f"{sorted(undeclared)}"
            )
        if dimension.status == "frozen" and not (
            set(dimension.source_paths) & verified_non_plan_sources
        ):
            raise Sprint3DecisionError(
                f"frozen protocol dimension {name!r} requires at least one "
                "verified non-plan source"
            )
    for family in plan.families:
        for support in family.gate_support:
            support.verify(
                repository,
                family.sources,
                verified_sources=verified_sources,
            )
    return plan


def evaluate_sprint_3(
    families: tuple[FamilyEvidence, ...],
    global_evidence: Sprint3GlobalEvidence,
    *,
    deferred_protocol_dimensions: tuple[str, ...] = (),
) -> Sprint3Decision:
    """Evaluate the complete frozen family without granting trade authority."""

    if not families or len(families) > MAX_FAMILIES:
        raise Sprint3DecisionError(f"family count must be in [1, {MAX_FAMILIES}]")
    names = tuple(family.family for family in families)
    if names != SPRINT_3_FAMILIES:
        raise Sprint3DecisionError(
            "families must exactly match the governed Sprint 3 family and order"
        )
    advanced = tuple(family for family in families if family.disposition == "advance")
    family_failures = tuple(
        f"family_evidence.{family.family}.{gate}"
        for family in advanced
        for gate in family.missing_gates
    )
    protocol_failures = tuple(
        f"protocol_dimension.{dimension}.deferred" for dimension in deferred_protocol_dimensions
    )
    failed = (
        global_evidence.failed_readiness_gates
        + (() if advanced else ("no_advanced_research_family",))
        + family_failures
        + protocol_failures
        + ("sprint_3_synthesis_has_no_paper_or_order_authority",)
    )
    return Sprint3Decision(
        decision="NOT_READY",
        failed_readiness_gates=failed,
        evaluated_families=sum(family.evaluated for family in families),
        rejected_families=sum(family.disposition == "reject" for family in families),
        deferred_families=sum(family.disposition == "defer" for family in families),
        advanced_research_families=len(advanced),
    )


def family_evidence_frame(families: tuple[FamilyEvidence, ...]) -> pd.DataFrame:
    """Return a deterministic machine-readable family decision table."""

    rows: list[dict[str, Any]] = []
    for family in families:
        support_by_gate = {support.gate: support for support in family.gate_support}
        rows.append(
            {
                "family": family.family,
                "context": family.context,
                "evaluated": family.evaluated,
                "disposition": family.disposition,
                "reason": family.reason,
                "gate_score": family.gate_score,
                "gate_count": len(EVIDENCE_GATES),
                "missing_gates": ";".join(family.missing_gates),
                "model_count": family.model_count,
                "blocked_or_failed_count": family.blocked_or_failed_count,
                "source_paths": ";".join(source.path for source in family.sources),
                "source_sha256": ";".join(source.sha256 for source in family.sources),
                "gate_support": json.dumps(
                    {
                        gate: {
                            "source_path": support_by_gate[gate].source_path,
                            "locator_kind": support_by_gate[gate].locator_kind,
                            "locator": support_by_gate[gate].locator,
                            "claim": support_by_gate[gate].claim,
                        }
                        for gate in EVIDENCE_GATES
                        if gate in support_by_gate
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                **{gate: bool(getattr(family, gate)) for gate in EVIDENCE_GATES},
            }
        )
    return pd.DataFrame(rows)


def gate_matrix_frame(families: tuple[FamilyEvidence, ...]) -> pd.DataFrame:
    """Return one long-form row per family/evidence gate."""

    rows: list[dict[str, Any]] = []
    for family in families:
        support_by_gate = {support.gate: support for support in family.gate_support}
        for gate in EVIDENCE_GATES:
            support = support_by_gate.get(gate)
            rows.append(
                {
                    "family": family.family,
                    "context": family.context,
                    "gate": gate,
                    "reported": support is not None,
                    "source_path": None if support is None else support.source_path,
                    "locator_kind": None if support is None else support.locator_kind,
                    "locator": None if support is None else support.locator,
                }
            )
    return pd.DataFrame(rows)


def plot_gate_matrix(families: tuple[FamilyEvidence, ...], output: str | Path) -> Path:
    """Render an accessible Seaborn heatmap of present and missing evidence."""

    frame = gate_matrix_frame(families)
    matrix = frame.pivot(index="family", columns="gate", values="reported").astype(int)
    matrix = matrix.loc[[family.family for family in families], list(EVIDENCE_GATES)]
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white", context="notebook")
    figure, axis = plt.subplots(figsize=(15, max(5, len(families) * 0.55)))
    sns.heatmap(
        matrix,
        cmap=sns.color_palette(["#b2182b", "#2166ac"], as_cmap=True),
        vmin=0,
        vmax=1,
        linewidths=0.6,
        linecolor="white",
        cbar=False,
        annot=True,
        fmt="d",
        ax=axis,
    )
    axis.set_title(
        "Signal Foundry Sprint 3 evidence coverage\n"
        "1 = reported evidence; 0 = missing gate (not a model-performance score)"
    )
    axis.set_xlabel("Governed synthesis evidence gate")
    axis.set_ylabel("Representation / decision family")
    axis.tick_params(axis="x", rotation=35)
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return destination


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def publish_sprint_3_decision(
    *,
    repository_root: str | Path,
    plan: Sprint3EvaluationPlan,
    output_dir: str | Path,
) -> Path:
    """Verify sources and atomically publish aggregate Sprint 3 evidence."""

    repository_input = Path(repository_root)
    if repository_input.is_symlink():
        raise Sprint3DecisionError("repository_root must not be a symlink")
    repository = repository_input.resolve()
    if not repository.is_dir():
        raise Sprint3DecisionError("repository_root must be a real directory")
    requested_output = Path(output_dir)
    if not requested_output.is_absolute():
        requested_output = repository / requested_output
    destination = Path(os.path.abspath(requested_output))
    try:
        output_relative = destination.relative_to(repository)
    except ValueError as exc:
        raise Sprint3DecisionError("decision evidence must remain inside repository_root") from exc
    output_component = repository
    for component in output_relative.parts:
        output_component = output_component / component
        if output_component.is_symlink():
            raise Sprint3DecisionError("decision evidence path must not traverse a symlink")
    if destination.exists():
        raise FileExistsError(f"decision evidence destination already exists: {destination}")
    loaded_plan = load_sprint_3_evaluation_plan(
        plan.config_source.path,
        repository_root=repository,
    )
    if loaded_plan != plan:
        raise Sprint3DecisionError(
            "in-memory Sprint 3 plan differs from its verified configuration"
        )
    plan = loaded_plan
    plan.config_source.verify(repository)
    families = plan.families
    global_evidence = plan.global_evidence
    verified_source_digests: dict[str, str] = {}
    reverified_source_paths: set[str] = set()
    for family in families:
        for source in family.sources:
            previous = verified_source_digests.setdefault(source.path, source.sha256)
            if previous != source.sha256:
                raise Sprint3DecisionError(
                    f"source artifact {source.path!r} has conflicting digests"
                )
            if source.path in reverified_source_paths:
                continue
            source.verify(repository)
            reverified_source_paths.add(source.path)
    decision = evaluate_sprint_3(
        families,
        global_evidence,
        deferred_protocol_dimensions=plan.protocol_dimensions.deferred_dimensions,
    )
    family_frame = family_evidence_frame(families)
    gate_frame = gate_matrix_frame(families)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        family_frame.to_csv(staging / "family_evidence.csv", index=False)
        gate_frame.to_csv(staging / "gate_matrix.csv", index=False)
        plot_gate_matrix(families, staging / "plots/evidence_coverage.png")
        _write_json(
            staging / "summary.json",
            {
                "schema_version": SPRINT_3_DECISION_SCHEMA_VERSION,
                "study": {
                    "study_id": plan.study_id,
                    "issue_number": plan.issue_number,
                    "plan_id": plan.plan_id,
                    "plan_path": plan.config_source.path,
                    "frozen_at_utc": plan.frozen_at_utc,
                    "protocol_dimensions": asdict(plan.protocol_dimensions),
                    "explicitly_deferred_protocol_dimensions": list(
                        plan.protocol_dimensions.deferred_dimensions
                    ),
                    "family_order": list(plan.family_order),
                    "evidence_gates": list(EVIDENCE_GATES),
                    "no_cross_context_ranking": True,
                    "final_holdout_reopened": False,
                },
                "decision": decision.decision,
                "failed_readiness_gates": list(decision.failed_readiness_gates),
                "family_counts": {
                    "evaluated": decision.evaluated_families,
                    "rejected": decision.rejected_families,
                    "deferred": decision.deferred_families,
                    "advanced_for_research": decision.advanced_research_families,
                },
                "global_evidence": asdict(global_evidence),
                "publication_boundary": {
                    "aggregate_only": True,
                    "licensed_observations_published": False,
                    "row_predictions_published": False,
                    "model_weights_published": False,
                    "executable_orders_emitted": False,
                    "capital_deployed": False,
                },
                "interpretation": (
                    "Evidence-quality synthesis across non-comparable historical and "
                    "synthetic engineering studies; not a model leaderboard, current "
                    "market forecast, profit claim, or paper/live authorization."
                ),
            },
        )
        (staging / "README.md").write_text(
            "# Signal Foundry Sprint 3 decision evidence\n\n"
            f"Decision: **{decision.decision}**.\n\n"
            f"Frozen plan: `{plan.config_source.path}` "
            f"(`{plan.plan_id}`).\n\n"
            "The heatmap reports whether each family produced the governed synthesis "
            "evidence category. It is not a performance score, and families from "
            "different data contexts are not ranked against one another.\n\n"
            "![Sprint 3 evidence coverage](plots/evidence_coverage.png)\n\n"
            "See `family_evidence.csv`, `gate_matrix.csv`, and `summary.json` for "
            "source hashes, explicit missing gates, dispositions, and residual "
            "readiness limitations.\n",
            encoding="utf-8",
        )
        artifacts = {
            path.relative_to(staging).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        _write_json(
            staging / "manifest.json",
            {
                "schema_version": SPRINT_3_DECISION_SCHEMA_VERSION,
                "artifacts": artifacts,
                "plan": asdict(plan.config_source),
                "source_artifacts": [
                    {"family": family.family, **asdict(source)}
                    for family in families
                    for source in family.sources
                ],
            },
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination
