from __future__ import annotations

import argparse

from _common import configure_fast_demo, load_configs, make_run_dir, save_meta, write_latest

from alphaforge.data import data_quality_report, load_prices
from alphaforge.evaluation import quantile_return_table
from alphaforge.features import build_features
from alphaforge.labels.labels import build_labels
from alphaforge.signals import select_model_predictions
from alphaforge.training import run_walk_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-safe walk-forward validation.")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--feature-config", default="configs/features.yaml")
    parser.add_argument(
        "--synthetic", action="store_true", help="Force the no-network synthetic source."
    )
    parser.add_argument(
        "--fast", action="store_true", help="Use a small CI-friendly synthetic experiment."
    )
    parser.add_argument("--runs-dir", default="runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_cfg, data_cfg, feature_cfg = load_configs(
        args.config, args.data_config, args.feature_config
    )
    if args.synthetic:
        data_cfg["source"] = "synthetic"
    if args.fast:
        configure_fast_demo(model_cfg, data_cfg)

    run_dir = make_run_dir(model_cfg.get("runs_dir", args.runs_dir), prefix="wf")
    panel, benchmark = load_prices(data_cfg)
    quality = data_quality_report(panel)
    features = build_features(panel, benchmark_symbol=benchmark, config=feature_cfg)
    horizons = list(model_cfg.get("horizons", [1, 5, 20]))
    labels = build_labels(panel, benchmark_symbol=benchmark, horizons=horizons)

    result = run_walk_forward(
        features=features,
        labels=labels,
        model_specs=model_cfg.get("models", []),
        target=model_cfg.get("target", f"fwd_ret_{horizons[0]}"),
        config=model_cfg.get("walk_forward", {}),
        max_horizon=max(horizons),
    )

    panel.to_pickle(run_dir / "panel.pkl")
    features.to_pickle(run_dir / "features.pkl")
    labels.to_pickle(run_dir / "labels.pkl")
    result.predictions.to_pickle(run_dir / "predictions.pkl")
    quality.to_csv(run_dir / "data_quality.csv", index=False)
    result.metrics.to_csv(run_dir / "model_metrics.csv", index=False)
    result.windows.to_csv(run_dir / "walk_forward_windows.csv", index=False)
    if not result.feature_importance.empty:
        result.feature_importance.to_csv(run_dir / "feature_importance.csv", index=False)

    selected = select_model_predictions(result.predictions)
    quantiles = quantile_return_table(selected)
    quantiles.to_csv(run_dir / "quantile_returns.csv", index=False)

    save_meta(
        run_dir,
        benchmark_symbol=benchmark,
        target=model_cfg.get("target", f"fwd_ret_{horizons[0]}"),
        horizons=horizons,
        data_config=data_cfg,
        model_config=model_cfg,
        feature_config=feature_cfg,
    )
    write_latest(run_dir)
    print(f"walk-forward run: {run_dir}")
    print(f"oos predictions: {len(result.predictions):,}")
    print(f"models: {sorted(result.predictions['model'].unique())}")


if __name__ == "__main__":
    main()
