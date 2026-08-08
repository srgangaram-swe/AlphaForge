"""Derive the Sprint 5 delivery ledger from frozen commits and tracked paths.

SF-S5-MR11. Replaces the hand-typed per-slice counts the close-out figure used
to carry. Test functions are counted by parsing tracked files; the commit for
each slice is resolved from Git, so the ledger cannot drift from what merged.

Run::

    python scripts/build_delivery_ledger.py \
        --output docs/evidence/signal_foundry_sprint_5/raw/delivery_ledger.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alphaforge.evidence.inventory import build_ledger

#: Frozen squash-merge identities for the Sprint 5 slices, with the paths each
#: delivered. References are resolved to full object ids at build time.
#:
#: SF-S5-MR11 is deliberately absent. Its own squash commit does not exist until
#: it merges, so it has no frozen identity to record — and a slice recorded
#: without one would defeat the property this ledger exists to provide. The
#: ledger therefore describes *merged* Sprint 5 delivery, and the figure labels
#: it that way. MR11's scope is in ADR 0021 and its merge request.
SLICES = (
    {
        "label": "MR2 broker requirements",
        "reference": "975e015",
        "source_paths": [],
        "test_paths": [],
        "doc_paths": [
            "docs/broker_connectivity_requirements.md",
            "docs/adr/0015-broker-selection-for-paper-and-live-trading.md",
        ],
    },
    {
        "label": "MR3 broker contract\n+ paper adapter",
        "reference": "70fd5d7",
        "source_paths": [
            "alphaforge/broker/contracts.py",
            "alphaforge/broker/config.py",
            "alphaforge/broker/paper_adapter.py",
        ],
        "test_paths": [
            "tests/test_broker_contracts.py",
            "tests/test_broker_paper_adapter.py",
        ],
        "doc_paths": [
            "docs/broker_contract_and_paper_adapter.md",
            "docs/adr/0016-deny-by-default-broker-authorization.md",
        ],
    },
    {
        "label": "MR4 durable state\n+ reconciliation",
        "reference": "b10aaf2",
        "source_paths": [
            "alphaforge/broker/durable.py",
            "alphaforge/broker/reconciliation.py",
        ],
        "test_paths": ["tests/test_broker_durable_state.py"],
        "doc_paths": [
            "docs/durable_state_and_reconciliation.md",
            "docs/adr/0017-durable-session-state-and-halt-on-divergence.md",
        ],
    },
    {
        "label": "MR8 distributed\nexecution",
        "reference": "cb3a60e",
        "source_paths": [
            "alphaforge/distributed/profiling.py",
            "alphaforge/distributed/tasks.py",
            "alphaforge/distributed/executor.py",
        ],
        "test_paths": ["tests/test_distributed_execution.py"],
        "doc_paths": [
            "docs/distributed_execution.md",
            "docs/adr/0018-bounded-distributed-research-execution.md",
        ],
    },
    {
        "label": "MR9 checkpoints\n+ budgets",
        "reference": "cb2310f",
        "source_paths": [
            "alphaforge/distributed/checkpoints.py",
            "alphaforge/distributed/budgets.py",
        ],
        "test_paths": ["tests/test_checkpoints_and_budgets.py"],
        "doc_paths": [
            "docs/checkpointing_and_budgets.md",
            "docs/adr/0019-checkpoint-bindings-and-hard-budgets.md",
        ],
    },
    {
        "label": "MR10 live readiness\n+ capital gate",
        "reference": "60f572b",
        "source_paths": [
            "alphaforge/readiness/checklist.py",
            "alphaforge/readiness/capital.py",
        ],
        "test_paths": ["tests/test_live_readiness.py"],
        "doc_paths": [
            "docs/live_readiness.md",
            "docs/adr/0020-live-readiness-gate-and-inert-capital.md",
        ],
    },
)


def main() -> None:
    """Build the ledger and write it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    arguments = parser.parse_args()

    ledger = build_ledger(5, SLICES, repository=arguments.repository)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(ledger.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for record in ledger.slices:
        print(
            f"{record.label.replace(chr(10), ' '):40s} {record.test_function_count:4d}  {record.commit[:8]}"
        )
    print(f"{'TOTAL':40s} {ledger.total_test_functions:4d}")
    print(f"ledger identity: {ledger.identity()[:16]}")


if __name__ == "__main__":
    main()
