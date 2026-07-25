"""Governed quantitative-research workflows."""

from alphaforge.research.manifest import (
    ExperimentManifest,
    ManifestValidationError,
    capture_environment,
    capture_git_context,
    derive_seed_map,
    inventory_artifacts,
    redact_cli_arguments,
)
from alphaforge.research.signal_foundry import (
    GovernedResearchConfig,
    GovernedResearchResult,
    run_governed_signal_foundry_research,
)

__all__ = [
    "ExperimentManifest",
    "GovernedResearchConfig",
    "GovernedResearchResult",
    "ManifestValidationError",
    "capture_environment",
    "capture_git_context",
    "derive_seed_map",
    "inventory_artifacts",
    "redact_cli_arguments",
    "run_governed_signal_foundry_research",
]
