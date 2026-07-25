"""Validate every committed AlphaForge YAML configuration before execution."""

from __future__ import annotations

from pathlib import Path

from alphaforge.config import SCHEMAS, load_config


def main() -> None:
    for kind in sorted(SCHEMAS):
        path = Path("configs") / f"{kind}.yaml"
        load_config(path, kind)
        print(f"validated {kind}: {path}")


if __name__ == "__main__":
    main()
