from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from _common import latest_run_dir

from alphaforge.portfolio import construct_portfolio
from alphaforge.signals import build_signals, select_model_predictions
from alphaforge.utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct target portfolio weights.")
    parser.add_argument("--config", default="configs/portfolio.yaml")
    parser.add_argument("--backtest-config", default="configs/backtest.yaml")
    parser.add_argument("--run-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    portfolio_cfg = load_yaml(args.config)
    backtest_cfg = load_yaml(args.backtest_config)
    predictions = pd.read_pickle(run_dir / "predictions.pkl")
    features = pd.read_pickle(run_dir / "features.pkl")
    selected = select_model_predictions(predictions)
    signals = build_signals(
        selected,
        strategy=backtest_cfg.get("strategy", "long_short"),
        params=backtest_cfg.get("strategy_params", {}),
    )
    weights = construct_portfolio(signals, features=features, config=portfolio_cfg)
    weights.to_csv(run_dir / "target_weights.csv", index=False)
    print(f"target weights written: {run_dir / 'target_weights.csv'}")


if __name__ == "__main__":
    main()
