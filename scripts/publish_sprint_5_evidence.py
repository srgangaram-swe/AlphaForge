"""Publish the Sprint 5 close-out evidence bundle and figure.

Run::

    python scripts/publish_sprint_5_evidence.py \
        --output docs/evidence/signal_foundry_sprint_5/closeout

Deterministic and network-independent. Refuses a non-empty destination so a
republish cannot overwrite evidence that is already cited.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alphaforge.readiness.sprint_5_evidence import publish_sprint_5_evidence


def main() -> None:
    """Parse arguments and publish the bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evidence/signal_foundry_sprint_5/closeout"),
        help="destination directory; must not already contain files",
    )
    arguments = parser.parse_args()
    manifest = publish_sprint_5_evidence(arguments.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
