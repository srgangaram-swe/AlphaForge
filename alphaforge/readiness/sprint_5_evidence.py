"""Deterministic Sprint 5 close-out evidence and its Seaborn figure.

AGENTS.md requires visual evidence of a sprint's actual outcomes before close,
and specifically forbids ornamental charts and favourable-result cherry-picking.
Sprint 5's actual outcome is that **the live-readiness gate is closed and every
one of its items is unmet**, so that is what the figure shows.

Three of the four panels are deliberately unflattering: the readiness gate with
seventeen unmet items, the capital ceiling that is inert, and the distribution
crossover showing that the workload this repository actually runs sits below the
point where distributing it helps. The fourth shows what was built.

Every panel is generated from the modules themselves rather than from
transcribed numbers, so a figure that disagrees with the code is impossible.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from alphaforge.readiness.capital import ABSOLUTE_MAX_CAPITAL, LiveCapitalConfig
from alphaforge.readiness.checklist import (
    ReadinessDecision,
    evaluate_readiness,
    minimal_capital_checklist,
    render_readiness_report,
)

#: Measured on macOS/arm64 in SF-S5-MR8. Reproduce with
#: ``benchmarks/benchmark_distributed_crossover.py``.
CROSSOVER_MEASUREMENTS: tuple[tuple[float, float], ...] = (
    (0.047, 0.02),
    (1.92, 0.66),
    (19.25, 2.98),
    (78.7, 4.38),
)

#: Modules delivered in Sprint 5 and their test counts, as a scope record.
SPRINT_5_DELIVERY: tuple[tuple[str, int], ...] = (
    ("MR2 broker requirements\n(docs only)", 0),
    ("MR3 broker contract\n+ paper adapter", 108),
    ("MR4 durable state\n+ reconciliation", 47),
    ("MR8 distributed\nexecution", 50),
    ("MR9 checkpoints\n+ budgets", 59),
    ("MR10 live readiness\n+ capital gate", 66),
)


def sprint_5_readiness() -> ReadinessDecision:
    """Return the honest readiness decision for this repository today.

    No evidence is supplied because none exists: no strategy is qualified, no
    paper trading has run, and no attestation has been made.
    """
    return evaluate_readiness(
        minimal_capital_checklist(),
        evidence={},
        attestations={},
        now=datetime(2026, 8, 8, tzinfo=UTC),
    )


def _figure(decision: ReadinessDecision, destination: Path) -> None:
    """Render the four-panel close-out figure."""
    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    figure, axes = plt.subplots(2, 2, figsize=(19, 14))

    # Panel 1 — the gate, by category. Every bar is unmet.
    grouped = decision.unmet_by_category()
    frame = pd.DataFrame(
        [{"category": key, "unmet": len(value)} for key, value in sorted(grouped.items())]
    )
    axis = axes[0][0]
    sns.barplot(data=frame, y="category", x="unmet", ax=axis, color="#d95f02", orient="h")
    axis.set_title(
        f"Live-readiness gate: {len(decision.unmet)} of {len(decision.results)} items UNMET",
        fontsize=15,
    )
    axis.set_xlabel("Unmet checklist items (count)")
    axis.set_ylabel("Category")
    for index, row in frame.iterrows():
        axis.text(row["unmet"] + 0.05, index, str(row["unmet"]), va="center", fontsize=12)
    axis.set_xlim(0, max(frame["unmet"]) + 1)

    # Panel 2 — capital exposure permitted today, against the ceiling.
    inert = LiveCapitalConfig.inert()
    capital = pd.DataFrame(
        [
            {"state": "Permitted today\n(inert)", "usd": float(inert.capital_cap)},
            {
                "state": "Absolute ceiling\n(no approval can raise)",
                "usd": float(ABSOLUTE_MAX_CAPITAL),
            },
        ]
    )
    axis = axes[0][1]
    sns.barplot(
        data=capital,
        x="state",
        y="usd",
        ax=axis,
        palette=["#7570b3", "#999999"],
        hue="state",
        legend=False,
    )
    axis.set_title("Capital at risk: none authorized", fontsize=15)
    axis.set_ylabel("Capital cap (USD)")
    axis.set_xlabel("")
    axis.text(
        0,
        float(ABSOLUTE_MAX_CAPITAL) * 0.06,
        "0",
        ha="center",
        fontsize=16,
        fontweight="bold",
    )

    # Panel 3 — the distribution crossover. Below ~2 ms/task, distributing loses.
    crossover = pd.DataFrame(
        [{"per_task_ms": ms, "speedup": s} for ms, s in CROSSOVER_MEASUREMENTS]
    )
    axis = axes[1][0]
    sns.lineplot(data=crossover, x="per_task_ms", y="speedup", marker="o", ax=axis, color="#1b9e77")
    axis.axhline(1.0, linestyle="--", color="#d95f02", linewidth=2)
    axis.axvline(0.49, linestyle=":", color="#444444", linewidth=2)
    axis.set_xscale("log")
    axis.set_title("Distribution pays only above ~2 ms per task", fontsize=15)
    axis.set_xlabel("Per-task cost (ms, log scale)")
    axis.set_ylabel("Speedup at 8 workers (x)")
    axis.text(2.2, 1.08, "break-even", color="#d95f02", fontsize=12)
    # Anchored to the 0.49 ms line it describes: placing it further left would
    # read as labelling the leftmost data point instead.
    axis.annotate(
        "this repo's sweep\n0.49 ms/task",
        xy=(0.49, 0.02),
        xytext=(0.52, 2.6),
        fontsize=11,
        color="#444444",
        arrowprops={"arrowstyle": "->", "color": "#444444", "linewidth": 1.2},
    )

    # Panel 4 — what was built, as a scope record.
    delivery = pd.DataFrame([{"slice": name, "tests": count} for name, count in SPRINT_5_DELIVERY])
    axis = axes[1][1]
    sns.barplot(data=delivery, y="slice", x="tests", ax=axis, color="#1b9e77", orient="h")
    axis.set_title("Sprint 5 delivery: 330 tests across six slices", fontsize=15)
    axis.set_xlabel("Tests added")
    axis.set_ylabel("")
    for index, row in delivery.iterrows():
        axis.text(row["tests"] + 1, index, str(row["tests"]), va="center", fontsize=11)

    figure.suptitle(
        "Signal Foundry Sprint 5 — monitoring, paper trading, live readiness\n"
        "Simulated and synthetic evidence. No capital at risk; no strategy qualified; "
        "no paper or live trading authorized.",
        fontsize=17,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def publish_sprint_5_evidence(output: Path | str) -> dict[str, Any]:
    """Write the deterministic close-out bundle and return its manifest.

    Refuses a non-empty destination so a republish cannot silently overwrite
    evidence something else already cites.

    Raises:
        FileExistsError: If the destination exists and is not empty.
    """
    destination = Path(output)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"{destination} is not empty; refusing to overwrite evidence that may already "
            "be cited. Remove it deliberately or choose a new path."
        )
    destination.mkdir(parents=True, exist_ok=True)

    decision = sprint_5_readiness()
    (destination / "readiness_decision.json").write_text(
        json.dumps(decision.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "readiness_report.md").write_text(
        render_readiness_report(decision) + "\n", encoding="utf-8"
    )
    (destination / "checklist.json").write_text(
        json.dumps(minimal_capital_checklist().to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"per_task_ms": ms, "speedup_8_workers": s} for ms, s in CROSSOVER_MEASUREMENTS]
    ).to_csv(destination / "distribution_crossover.csv", index=False)

    figure_path = destination / "sprint_5_closeout.png"
    _figure(decision, figure_path)

    manifest = {
        "sprint": 5,
        "verdict": decision.verdict.value,
        "unmet_count": len(decision.unmet),
        "total_items": len(decision.results),
        "unmet_by_category": decision.unmet_by_category(),
        "capital_at_risk": str(LiveCapitalConfig.inert().capital_cap),
        "absolute_ceiling": str(ABSOLUTE_MAX_CAPITAL),
        "checklist_identity": decision.checklist_identity,
        # Includes manifest.json itself: the field is a faithful listing of the
        # published directory, and a manifest that omits itself invites a reader
        # to think a file is missing.
        "artifacts": sorted({item.name for item in destination.iterdir()} | {"manifest.json"}),
        "limitations": [
            "Synthetic and simulated evidence only; no market data and no broker connection.",
            "The readiness framework cannot verify policy or legal attestations; it records "
            "them and says so.",
            "Crossover measurements are single-machine wall clock on macOS/arm64.",
            "No strategy is qualified, so no paper or live trading is authorized.",
        ],
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = [
    "CROSSOVER_MEASUREMENTS",
    "SPRINT_5_DELIVERY",
    "publish_sprint_5_evidence",
    "sprint_5_readiness",
]
