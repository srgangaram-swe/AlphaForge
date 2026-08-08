"""Sprint 5 close-out evidence, derived from committed records rather than typed.

SF-S5-MR11 rewrote this module. The previous version carried its crossover
points and per-slice test counts as hand-typed Python constants while its own
docstring claimed *"Every panel is generated from the modules themselves rather
than from transcribed numbers"*. That claim was false: two of four panels were
transcription, the transcribed crossover values disagreed with ADR 0018, and each
was a single unwarmed sample.

The claim is now true, and it is true by construction rather than by assertion:

- Performance panels read :data:`RAW_MEASUREMENTS_PATH`, a committed record of
  raw nanosecond samples, and compute every statistic from it.
- The delivery panel reads a committed ledger derived from frozen commit
  identities and tracked paths.
- Readiness and capital panels call the modules directly, as before.

**No measurement appears as a numeric literal in this file.** A test asserts
that, because the previous failure was not a typo — it was a literal that drifted
away from the thing it described while the docstring kept insisting otherwise.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from alphaforge.evidence.inventory import DeliveryLedger, ledger_from_payload
from alphaforge.evidence.measurements import (
    CrossoverEvidence,
    evidence_from_payload,
    summary_rows,
)
from alphaforge.evidence.publication import publish_bundle
from alphaforge.readiness.capital import ABSOLUTE_MAX_CAPITAL, LiveCapitalConfig
from alphaforge.readiness.checklist import (
    ReadinessDecision,
    evaluate_readiness,
    minimal_capital_checklist,
    render_readiness_report,
)

#: Committed raw inputs. Every published number derives from one of these.
_EVIDENCE_ROOT: Final = (
    Path(__file__).resolve().parents[2] / "docs/evidence/signal_foundry_sprint_5/raw"
)
RAW_MEASUREMENTS_PATH: Final = _EVIDENCE_ROOT / "crossover_measurements.json"
DELIVERY_LEDGER_PATH: Final = _EVIDENCE_ROOT / "delivery_ledger.json"

#: Fixed decision date, so a regeneration is byte-identical rather than varying
#: with the wall clock. The readiness verdict does not depend on it.
DECISION_INSTANT: Final = datetime(2026, 8, 8, tzinfo=UTC)

BUNDLE_SCHEMA_VERSION: Final = 2


def load_measurements(path: Path | None = None) -> CrossoverEvidence:
    """Load the committed raw benchmark record.

    Raises:
        FileNotFoundError: If the record is absent — the bundle cannot be built
            from nothing, and silently substituting defaults is how a plot stops
            describing a measurement.
    """
    source = path or RAW_MEASUREMENTS_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"raw measurements are missing at {source}. Regenerate with "
            "`python benchmarks/benchmark_distributed_crossover.py --output <path>`; "
            "this module will not substitute defaults."
        )
    return evidence_from_payload(json.loads(source.read_text(encoding="utf-8")))


def load_delivery(path: Path | None = None) -> DeliveryLedger:
    """Load the committed delivery ledger.

    Raises:
        FileNotFoundError: If the ledger is absent.
    """
    source = path or DELIVERY_LEDGER_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"delivery ledger is missing at {source}. Regenerate with "
            "`python scripts/build_delivery_ledger.py`."
        )
    return ledger_from_payload(json.loads(source.read_text(encoding="utf-8")))


def sprint_5_readiness() -> ReadinessDecision:
    """Return the honest readiness decision for this repository.

    No evidence is supplied because none exists: no strategy is qualified, no
    paper trading has run, and no attestation has been made.
    """
    return evaluate_readiness(
        minimal_capital_checklist(), evidence={}, attestations={}, now=DECISION_INSTANT
    )


def _figure(
    decision: ReadinessDecision,
    measurements: CrossoverEvidence,
    delivery: DeliveryLedger,
    destination: Path,
) -> None:
    """Render the four-panel close-out figure from loaded records only."""
    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    figure, axes = plt.subplots(2, 2, figsize=(19, 14))

    # Panel 1 — the readiness gate, computed from the checklist.
    grouped = decision.unmet_by_category()
    gate = pd.DataFrame(
        [{"category": key, "unmet": len(value)} for key, value in sorted(grouped.items())]
    )
    axis = axes[0][0]
    sns.barplot(data=gate, y="category", x="unmet", ax=axis, color="#d95f02", orient="h")
    axis.set_title(
        f"Live-readiness gate: {len(decision.unmet)} of {len(decision.results)} items UNMET",
        fontsize=15,
    )
    axis.set_xlabel("Unmet checklist items (count)")
    axis.set_ylabel("Category")
    for index, row in gate.iterrows():
        axis.text(row["unmet"] + 0.05, index, str(row["unmet"]), va="center", fontsize=12)
    axis.set_xlim(0, int(gate["unmet"].max()) + 1)

    # Panel 2 — capital permitted, read from the configuration.
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
        hue="state",
        palette=["#7570b3", "#999999"],
        legend=False,
    )
    axis.set_title("Capital at risk: none authorized", fontsize=15)
    axis.set_ylabel("Capital cap (USD)")
    axis.set_xlabel("")
    axis.text(
        0, float(ABSOLUTE_MAX_CAPITAL) * 0.06, "0", ha="center", fontsize=16, fontweight="bold"
    )

    # Panel 3 — crossover with observed dispersion, from raw samples.
    rows = pd.DataFrame(summary_rows(measurements))
    axis = axes[1][0]
    axis.errorbar(
        rows["per_task_ms"],
        rows["speedup"],
        yerr=[rows["speedup"] - rows["speedup_low"], rows["speedup_high"] - rows["speedup"]],
        fmt="o-",
        color="#1b9e77",
        ecolor="#1b9e77",
        elinewidth=2,
        capsize=6,
        markersize=8,
    )
    axis.axhline(1.0, linestyle="--", color="#d95f02", linewidth=2)
    axis.set_xscale("log")
    bounds = measurements.crossover_bounds_ms()
    reps = int(rows["repetitions"].min())
    if bounds is not None:
        axis.axvspan(bounds[0], bounds[1], alpha=0.13, color="#444444")
        axis.set_title(
            f"Distribution pays above {bounds[0]:.1f}–{bounds[1]:.1f} ms per task", fontsize=15
        )
    else:
        axis.set_title("Distribution crossover not bracketed by these sizes", fontsize=15)
    axis.set_xlabel(
        f"Per-task cost (ms, log scale) — {rows['task_count'].iloc[0]} tasks, "
        f"{rows['workers'].iloc[0]} workers"
    )
    axis.set_ylabel(f"Speedup (x), median of {reps} reps; bars show min–max")
    axis.text(rows["per_task_ms"].iloc[0], 1.12, "break-even", color="#d95f02", fontsize=12)

    # Panel 4 — delivery, from the Git-derived ledger.
    scope = pd.DataFrame(
        [{"slice": item.label, "functions": item.test_function_count} for item in delivery.slices]
    )
    axis = axes[1][1]
    sns.barplot(data=scope, y="slice", x="functions", ax=axis, color="#1b9e77", orient="h")
    # "merged slices" is load-bearing: the ledger only records slices with a
    # frozen commit, so the corrective slice still in flight is not counted.
    axis.set_title(
        f"Sprint {delivery.sprint} merged slices: "
        f"{delivery.total_test_functions} test functions",
        fontsize=15,
    )
    axis.set_xlabel("Test functions defined (not collected cases, not passing tests)")
    axis.set_ylabel("")
    for index, row in scope.iterrows():
        axis.text(row["functions"] + 1, index, str(row["functions"]), va="center", fontsize=11)

    environment = measurements.environment
    figure.suptitle(
        "Signal Foundry Sprint 5 — monitoring, paper trading, live readiness\n"
        f"Synthetic and simulated evidence on {environment.machine} / "
        f"Python {environment.python_version}, {environment.cpu_count} CPUs. "
        "Not a service-level objective.\n"
        "No capital at risk; no strategy qualified; no paper or live trading authorized.",
        fontsize=16,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    # Deterministic metadata: matplotlib otherwise stamps its own version and a
    # creation date, which would make two identical regenerations differ.
    figure.savefig(
        destination,
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": None, "Creation Time": None},
    )
    plt.close(figure)


def publish_sprint_5_evidence(
    output: Path | str,
    *,
    measurements_path: Path | None = None,
    delivery_path: Path | None = None,
    source_ref: str = "unspecified",
) -> dict[str, Any]:
    """Publish the close-out bundle atomically and return its manifest.

    Every artifact is written into a staging sibling and renamed into place only
    after the whole bundle validates, so a failure leaves no partial bundle and
    an existing destination is never overwritten.

    Raises:
        PublicationError: On an existing or unsafe destination.
        FileNotFoundError: If a committed raw input is missing.
    """
    decision = sprint_5_readiness()
    measurements = load_measurements(measurements_path)
    delivery = load_delivery(delivery_path)

    def write(staging: Path) -> None:
        (staging / "readiness_decision.json").write_text(
            json.dumps(decision.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging / "readiness_report.md").write_text(
            render_readiness_report(decision) + "\n", encoding="utf-8"
        )
        (staging / "checklist.json").write_text(
            json.dumps(minimal_capital_checklist().to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "crossover_summary.csv").write_text(
            pd.DataFrame(summary_rows(measurements)).to_csv(index=False), encoding="utf-8"
        )
        (staging / "delivery_ledger.json").write_text(
            json.dumps(delivery.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _figure(decision, measurements, delivery, staging / "sprint_5_closeout.png")

    bounds = measurements.crossover_bounds_ms()
    return publish_bundle(
        output,
        writer=write,
        schema_version=BUNDLE_SCHEMA_VERSION,
        generator="alphaforge.readiness.sprint_5_evidence",
        source_ref=source_ref,
        environment=measurements.environment.to_dict(),
        input_identities={
            "crossover_measurements": measurements.identity(),
            "delivery_ledger": delivery.identity(),
            "readiness_checklist": decision.checklist_identity,
        },
        extra={
            "sprint": 5,
            "verdict": decision.verdict.value,
            "unmet_count": len(decision.unmet),
            "total_items": len(decision.results),
            "capital_at_risk": str(LiveCapitalConfig.inert().capital_cap),
            "crossover_interval_ms": None if bounds is None else list(bounds),
            "total_test_functions": delivery.total_test_functions,
            "limitations": [
                *measurements.limitations,
                "The readiness framework records policy and legal attestations; it cannot "
                "verify that a review occurred or reached its stated conclusion.",
                "No strategy is qualified, so no paper or live trading is authorized.",
            ],
        },
    )


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "DELIVERY_LEDGER_PATH",
    "RAW_MEASUREMENTS_PATH",
    "load_delivery",
    "load_measurements",
    "publish_sprint_5_evidence",
    "sprint_5_readiness",
]
