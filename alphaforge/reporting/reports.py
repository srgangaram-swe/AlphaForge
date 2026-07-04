"""Markdown report generation for AlphaForge runs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DISCLAIMER = (
    "AlphaForge is an educational quantitative research and ML engineering project. "
    "It is not financial advice, does not guarantee profitability, and should not be "
    "used to trade real money without professional review, additional validation, and "
    "appropriate risk controls. Backtests are not live results and may not predict "
    "future performance."
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _table(frame: pd.DataFrame, max_rows: int = 12) -> str:
    if frame.empty:
        return "_No rows available._"
    return "```\n" + frame.head(max_rows).to_string(index=False) + "\n```"


def _kv_lines(d: dict) -> list[str]:
    return [
        f"- **{k}**: {v:.6g}" if isinstance(v, (int, float)) else f"- **{k}**: {v}"
        for k, v in d.items()
    ]


def write_markdown_report(run_dir: str | Path, output_path: str | Path | None = None) -> Path:
    """Write a compact research report for a completed synthetic or real run."""
    run_dir = Path(run_dir)
    output_path = Path(output_path) if output_path else run_dir / "report.md"
    summary = _read_json(run_dir / "backtest_summary.json")
    overfit = _read_json(run_dir / "overfitting.json")
    model_metrics = _read_csv(run_dir / "model_metrics.csv")
    quantiles = _read_csv(run_dir / "quantile_returns.csv")
    ic_table = _read_csv(run_dir / "ic_summary.csv")
    decay = _read_csv(run_dir / "ic_decay.csv")
    regimes = _read_csv(run_dir / "regime_performance.csv")

    lines = ["# AlphaForge Research Report", "", DISCLAIMER, "", "## Backtest Summary", ""]
    lines.extend(_kv_lines(summary) if summary else ["_Backtest summary not found._"])

    lines += ["", "## Overfitting Diagnostics", ""]
    diag = {}
    if overfit:
        diag["probability_of_backtest_overfitting"] = overfit.get("pbo")
        diag["pbo_combinations"] = overfit.get("n_combinations")
        diag["pbo_strategies_compared"] = overfit.get("n_strategies")
    for key in ("deflated_sharpe_prob", "psr_vs_zero", "expected_max_sharpe", "n_trials"):
        if key in summary:
            diag[key] = summary[key]
    if diag:
        lines.extend(_kv_lines(diag))
        lines += [
            "",
            "_PBO is the probability the in-sample best model underperforms the median "
            "out-of-sample (CSCV). The deflated Sharpe probability is P(true Sharpe > 0) "
            "after correcting for multiple testing, sample length, skew, and fat tails._",
        ]
    else:
        lines.append("_No overfitting diagnostics available._")

    lines += ["", "## Information Coefficient by Model", "", _table(ic_table)]
    if not decay.empty:
        lines += ["", "## IC Decay (selected model)", "", _table(decay)]
    if not regimes.empty:
        lines += ["", "## Regime-Conditional Performance", "", _table(regimes)]
    lines += [
        "",
        "## Model Metrics (per walk-forward window)",
        "",
        _table(model_metrics),
        "",
        "## Prediction Quantiles",
        "",
        _table(quantiles),
        "",
    ]

    output_path.write_text("\n".join(lines))
    return output_path
