from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from _common import latest_run_dir

from alphaforge.backtesting import run_backtest
from alphaforge.evaluation import deflated_sharpe_ratio
from alphaforge.portfolio import construct_portfolio
from alphaforge.risk import (
    exposure_summary,
    monthly_returns,
    performance_summary,
    regime_performance,
    stress_test_summary,
)
from alphaforge.signals import apply_regime_filter, build_signals, select_model_predictions
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
    if cfg.get("risk", {}).get("regime_filter"):
        signals = apply_regime_filter(signals, features)
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
    summary.update({f"latest_{k}": v for k, v in exposure_summary(result.weights).items()})

    # Deflated Sharpe: deflate by the number of model variants that competed
    # for selection in this run. Understating n_trials overstates the result.
    n_trials = int(predictions["model"].nunique()) if "model" in predictions else 1
    active_returns = (
        result.equity_curve.loc[result.equity_curve["active"], "return"]
        if "active" in result.equity_curve
        else result.equity_curve["return"]
    )
    summary.update(deflated_sharpe_ratio(active_returns, n_trials=n_trials))

    signals.to_csv(run_dir / "signals.csv", index=False)
    weights.to_csv(run_dir / "target_weights.csv", index=False)
    result.equity_curve.to_csv(run_dir / "equity_curve.csv", index=False)
    result.weights.to_csv(run_dir / "executed_weights.csv", index=False)
    result.trades.to_csv(run_dir / "trades.csv", index=False)
    monthly_returns(result.equity_curve).to_csv(run_dir / "monthly_returns.csv", index=False)

    # regime-conditional performance (causal HMM stress feature when present)
    regime_col = "hmm_stress_prob" if "hmm_stress_prob" in features.columns else "high_vol_regime"
    if regime_col in features.columns:
        regime = (features.groupby("date")[regime_col].first() > 0.5).map(
            {True: "stress", False: "calm"}
        )
        regime_performance(result.equity_curve, regime, regime_name="regime").to_csv(
            run_dir / "regime_performance.csv", index=False
        )

    betas = (
        features.sort_values("date").groupby("symbol")["beta"].last()
        if "beta" in features.columns
        else None
    )
    stress_test_summary(result.weights, risk_cfg.get("stress_scenarios", []), betas=betas).to_csv(
        run_dir / "stress_tests.csv", index=False
    )
    save_json(summary, run_dir / "backtest_summary.json")

    print(f"backtest run: {run_dir}")
    print(f"selected model: {selected['model'].iloc[0] if 'model' in selected else 'single'}")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
