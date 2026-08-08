"""Reproducible, content-addressed sprint evidence (SF-S5-MR11).

- :mod:`~alphaforge.evidence.measurements` — raw benchmark samples with computed
  summaries. A single sample is refused; medians and ranges are properties over
  the samples, so no published statistic can disagree with them.
- :mod:`~alphaforge.evidence.inventory` — delivery scope derived from frozen
  commit identities and tracked paths rather than counted by hand.
- :mod:`~alphaforge.evidence.publication` — all-or-nothing bundle publication
  with a manifest carrying a SHA-256 and byte size for every other artifact.

These modules exist because the original Sprint 5 close-out hand-transcribed its
performance and delivery numbers while claiming they were computed from source.
"""

from alphaforge.evidence.inventory import (
    COUNT_SEMANTICS,
    DeliveryLedger,
    InventoryError,
    SliceRecord,
    build_ledger,
    count_test_functions,
    ledger_from_payload,
    resolve_commit,
)
from alphaforge.evidence.measurements import (
    MIN_REPETITIONS,
    MIN_WARMUPS,
    SCHEMA_VERSION,
    BenchmarkEnvironment,
    CrossoverEvidence,
    MeasurementError,
    WorkSizeMeasurement,
    assert_same_environment,
    evidence_from_payload,
    summary_rows,
)
from alphaforge.evidence.publication import (
    MANIFEST_NAME,
    SELF_HASH_NOTE,
    ArtifactRecord,
    PublicationError,
    publish_bundle,
    verify_bundle,
)

__all__ = [
    "COUNT_SEMANTICS",
    "MANIFEST_NAME",
    "MIN_REPETITIONS",
    "MIN_WARMUPS",
    "SCHEMA_VERSION",
    "SELF_HASH_NOTE",
    "ArtifactRecord",
    "BenchmarkEnvironment",
    "CrossoverEvidence",
    "DeliveryLedger",
    "InventoryError",
    "MeasurementError",
    "PublicationError",
    "SliceRecord",
    "WorkSizeMeasurement",
    "assert_same_environment",
    "build_ledger",
    "count_test_functions",
    "evidence_from_payload",
    "ledger_from_payload",
    "publish_bundle",
    "resolve_commit",
    "summary_rows",
    "verify_bundle",
]
