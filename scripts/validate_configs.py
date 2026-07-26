"""Validate every committed AlphaForge YAML configuration before execution."""

from __future__ import annotations

from pathlib import Path

from alphaforge.config import SCHEMAS, load_config
from alphaforge.research.deep_sequence_study import load_deep_sequence_study_config


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


if __name__ == "__main__":
    main()
