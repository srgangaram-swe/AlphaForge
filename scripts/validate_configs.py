"""Validate every committed AlphaForge YAML configuration before execution."""

from __future__ import annotations

from pathlib import Path

from alphaforge.config import SCHEMAS, load_config


def main() -> None:
    for kind in sorted(SCHEMAS):
        path = Path("configs") / f"{kind}.yaml"
        load_config(path, kind)
        print(f"validated {kind}: {path}")
    bootstrap = Path("configs/signal_foundry_wiki_bootstrap.yaml")
    load_config(bootstrap, "signal_foundry_research")
    print(f"validated signal_foundry_research: {bootstrap}")


if __name__ == "__main__":
    main()
