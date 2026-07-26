"""Validate every committed AlphaForge YAML configuration before execution."""

from __future__ import annotations

from pathlib import Path

from alphaforge.config import SCHEMAS, load_config
from alphaforge.research.deep_sequence_study import load_deep_sequence_study_config
from alphaforge.research.ensemble_study import load_ensemble_study_config
from alphaforge.research.representation_study import load_representation_study_config
from alphaforge.research.time_frequency_study import load_time_frequency_study_config


def main() -> None:
    for kind in sorted(SCHEMAS):
        path = Path("configs") / f"{kind}.yaml"
        load_config(path, kind)
        print(f"validated {kind}: {path}")
    for profile in (
        "signal_foundry_wiki_bootstrap.yaml",
        "signal_foundry_sprint_2_study.yaml",
    ):
        path = Path("configs") / profile
        load_config(path, "signal_foundry_research")
        print(f"validated signal_foundry_research: {path}")
    deep_sequence_path = Path("configs/deep_sequence_benchmark.yaml")
    load_deep_sequence_study_config(deep_sequence_path)
    print(f"validated deep_sequence_study: {deep_sequence_path}")
    time_frequency_path = Path("configs/time_frequency_vision_benchmark.yaml")
    load_time_frequency_study_config(time_frequency_path)
    print(f"validated time_frequency_study: {time_frequency_path}")
    representation_path = Path("configs/latent_representation_benchmark.yaml")
    load_representation_study_config(representation_path)
    print(f"validated representation_study: {representation_path}")

    ensemble_path = Path("configs/ensemble_benchmark.yaml")
    load_ensemble_study_config(ensemble_path)
    print(f"validated ensemble_study: {ensemble_path}")


if __name__ == "__main__":
    main()
