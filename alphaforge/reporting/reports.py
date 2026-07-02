"""Markdown report generation for AlphaForge runs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _table(frame: pd.DataFrame, max_rows: int = 12) -> str:
    if frame.empty:
        return "_No rows available._"
    return "```\n" + frame.head(max_rows).to_string(index=False) + "\n```"


def write_markdown_report(run_dir: str | Path, output_path: str | Path | None = None) -> Path:
    """Write a compact research report for a completed synthetic or real run."""
    run_dir = Path(run_dir)
    output_path = Path(output_path) if output_path else run_dir / "report.md"
    summary = _read_json(run_dir / "backtest_summary.json")
    model_metrics = (
        pd.read_csv(run_dir / "model_metrics.csv")
        if (run_dir / "model_metrics.csv").exists()
        else pd.DataFrame()
    )
    quantiles = (
        pd.read_csv(run_dir / "quantile_returns.csv")
        if (run_dir / "quantile_returns.csv").exists()
        else pd.DataFrame()
    )

    lines = [
        "# AlphaForge Research Report",
        "",
        "AlphaForge is an educational quantitative research and ML engineering project. It is not financial advice, does not guarantee profitability, and should not be used to trade real money without professional review, additional validation, and appropriate risk controls. Backtests are not live results and may not predict future performance.",
        "",
        "## Backtest Summary",
        "",
    ]
    if summary:
        lines.extend(
            f"- **{k}**: {v:.6g}" if isinstance(v, (int, float)) else f"- **{k}**: {v}"
            for k, v in summary.items()
        )
    else:
        lines.append("_Backtest summary not found._")
    lines.extend(
        [
            "",
            "## Model Metrics",
            "",
            _table(model_metrics),
            "",
            "## Prediction Quantiles",
            "",
            _table(quantiles),
            "",
        ]
    )

    output_path.write_text("\n".join(lines))
    return output_path
