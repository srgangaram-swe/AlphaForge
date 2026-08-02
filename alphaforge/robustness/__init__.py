"""Parameter, feature, and stable-region robustness analysis (SF-S4-MR6).

- :mod:`~alphaforge.robustness.grid` — the frozen study: parameter axes, feature
  families, ablation arms, negative controls, and named seed streams, published
  with a content identity so a mid-study edit is detectable.
- :mod:`~alphaforge.robustness.controls` — feature-permutation, randomized-label,
  and representation-placebo nulls under a leakage-safe temporal protocol.
- :mod:`~alphaforge.robustness.stability` — stable regions, sensitivity cliffs,
  and family redundancy, reported instead of a single optimum.

Simulation-only. Nothing here qualifies a strategy or claims a profit.
"""

from alphaforge.robustness.controls import (
    MIN_CONTROL_OBSERVATIONS,
    ControlOutcome,
    Fold,
    NegativeControlError,
    assert_streams_isolated,
    contiguous_folds,
    permute_within_folds,
    randomize_labels_within_folds,
    representation_placebo,
    run_control,
)
from alphaforge.robustness.grid import (
    CONTROL_KINDS,
    MAX_GRID_POINTS,
    ControlKind,
    FeatureFamily,
    GridPoint,
    NegativeControl,
    RobustnessGrid,
    RobustnessGridError,
    canonical_digest,
    verify_frozen,
)
from alphaforge.robustness.stability import (
    FamilyContribution,
    SensitivityCliff,
    StabilityError,
    StableRegion,
    family_redundancy,
    sensitivity_cliffs,
    stability_report,
    stable_regions,
    verify_rerun_determinism,
)

__all__ = [
    "CONTROL_KINDS",
    "MAX_GRID_POINTS",
    "MIN_CONTROL_OBSERVATIONS",
    "ControlKind",
    "ControlOutcome",
    "FamilyContribution",
    "FeatureFamily",
    "Fold",
    "GridPoint",
    "NegativeControl",
    "NegativeControlError",
    "RobustnessGrid",
    "RobustnessGridError",
    "SensitivityCliff",
    "StabilityError",
    "StableRegion",
    "assert_streams_isolated",
    "canonical_digest",
    "contiguous_folds",
    "family_redundancy",
    "permute_within_folds",
    "randomize_labels_within_folds",
    "representation_placebo",
    "run_control",
    "sensitivity_cliffs",
    "stability_report",
    "stable_regions",
    "verify_frozen",
    "verify_rerun_determinism",
]
