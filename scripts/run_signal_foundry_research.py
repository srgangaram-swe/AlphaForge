"""Run the pre-registered Signal Foundry final-holdout workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from alphaforge.data import load_signal_foundry_dataset
from alphaforge.evaluation import ReadinessThresholds
from alphaforge.research import GovernedResearchConfig, run_governed_signal_foundry_research
from alphaforge.utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one immutable Signal Foundry development/final-holdout evaluation."
    )
    parser.add_argument("bundle", help="Verified local Signal Foundry bundle directory.")
    parser.add_argument(
        "--config",
        default="configs/signal_foundry_research.yaml",
        help="Pre-registered research and readiness configuration.",
    )
    parser.add_argument("--output", default="runs/signal-foundry")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    expected = {
        "research",
        "readiness",
        "models",
        "features",
        "walk_forward",
        "backtest",
    }
    unknown = set(config) - expected
    missing = expected - set(config)
    if unknown or missing:
        raise ValueError(
            f"governed config fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    research_values = dict(config["research"])
    research_values["horizons"] = tuple(research_values.get("horizons", (1, 5, 20)))
    result = run_governed_signal_foundry_research(
        dataset=load_signal_foundry_dataset(Path(args.bundle)),
        model_specs=list(config["models"]),
        feature_config=dict(config["features"]),
        walk_forward_config=dict(config["walk_forward"]),
        backtest_config=dict(config["backtest"]),
        research_config=GovernedResearchConfig(**research_values),
        readiness_thresholds=ReadinessThresholds.from_mapping(dict(config["readiness"])),
        output_root=Path(args.output),
    )
    print(f"run_id={result.run_id}")
    print(f"candidate={result.candidate_model}")
    print(f"decision={result.dossier['decision']}")
    print(f"dossier={result.run_dir / 'dossier.md'}")


if __name__ == "__main__":
    main()
