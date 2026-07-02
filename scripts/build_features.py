from __future__ import annotations

import argparse
from pathlib import Path

from alphaforge.data import load_prices
from alphaforge.features import build_features
from alphaforge.labels.labels import build_labels
from alphaforge.utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build feature and label panels.")
    parser.add_argument("--config", default="configs/features.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.models_config)
    panel, benchmark = load_prices(data_cfg)
    features = build_features(panel, benchmark, feature_cfg)
    labels = build_labels(panel, benchmark, horizons=model_cfg.get("horizons", [1, 5, 20]))
    output_dir = Path(feature_cfg.get("output_dir", "data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)
    panel.to_pickle(output_dir / "panel.pkl")
    features.to_pickle(output_dir / "features.pkl")
    labels.to_pickle(output_dir / "labels.pkl")
    print(f"features: {features.shape}; labels: {labels.shape}; output={output_dir}")


if __name__ == "__main__":
    main()
