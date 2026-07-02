from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from _common import latest_run_dir

from alphaforge.backtesting import run_backtest
from alphaforge.portfolio import construct_portfolio
from alphaforge.risk import monthly_returns, performance_summary, stress_test_summary
from alphaforge.signals import build_signals, select_model_predictions
from alphaforge.utils import load_yaml, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a realistic OOS backtest from predictions.")
    parser.add_argument("--config", default="configs/backtest.yaml")
    parser.add_argument("--risk-config", default="configs/risk.yaml")
    parser.add_argument("--run-dir")
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--model", help="Prediction model to backtest.")
    return parser.parse_args()


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return Path(args.run_dir)
    return latest_run_dir()


def main() -> None:
    args = parse_args()
    run_dir = _resolve_run_dir(args)
    cfg = load_yaml(args.config)
    risk_cfg = load_yaml(args.risk_config)
    meta = json.loads((run_dir / "run_meta.json").read_text())

    panel = pd.read_pickle(run_dir / "panel.pkl")
    features = pd.read_pickle(run_dir / "features.pkl")
    predictions = pd.read_pickle(run_dir / "predictions.pkl")
    selected = select_model_predictions(predictions, model=args.model)
    signals = build_signals(
        selected, strategy=cfg.get("strategy", "long_short"), params=cfg.get("strategy_params", {})
    )
    weights = construct_portfolio(signals, features=features, config=cfg.get("portfolio", {}))

    result = run_backtest(
        panel=panel,
        target_weights=weights,
        benchmark_symbol=meta.get("benchmark_symbol"),
        initial_capital=float(cfg.get("initial_capital", 1_000_000)),
        execution_lag=int(cfg.get("execution_lag", 1)),
        rebalance_frequency=int(cfg.get("rebalance_frequency", 1)),
        costs=cfg.get("costs", {}),
        risk=cfg.get("risk", {}),
    )
    summary = performance_summary(result.equity_curve)

    signals.to_csv(run_dir / "signals.csv", index=False)
    weights.to_csv(run_dir / "target_weights.csv", index=False)
    result.equity_curve.to_csv(run_dir / "equity_curve.csv", index=False)
    result.weights.to_csv(run_dir / "executed_weights.csv", index=False)
    result.trades.to_csv(run_dir / "trades.csv", index=False)
    monthly_returns(result.equity_curve).to_csv(run_dir / "monthly_returns.csv", index=False)
    stress_test_summary(result.weights, risk_cfg.get("stress_scenarios", [])).to_csv(
        run_dir / "stress_tests.csv", index=False
    )
    save_json(summary, run_dir / "backtest_summary.json")

    print(f"backtest run: {run_dir}")
    print(f"selected model: {selected['model'].iloc[0] if 'model' in selected else 'single'}")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
