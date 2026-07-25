from __future__ import annotations

import argparse
from pathlib import Path

from alphaforge.config import load_data_config, load_feature_config, load_models_config
from alphaforge.data import load_prices
from alphaforge.features import build_features
from alphaforge.labels.labels import build_labels
from alphaforge.research import write_frame_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build feature and label panels.")
    parser.add_argument("--config", default="configs/features.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_cfg = load_feature_config(args.config)
    data_cfg = load_data_config(args.data_config)
    model_cfg = load_models_config(args.models_config)
    panel, benchmark = load_prices(data_cfg)
    features = build_features(panel, benchmark, feature_cfg)
    labels = build_labels(panel, benchmark, horizons=model_cfg.get("horizons", [1, 5, 20]))
    output_dir = Path(feature_cfg.get("output_dir", "data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_frame_artifact(panel, output_dir / "panel.table.json")
    write_frame_artifact(features, output_dir / "features.table.json")
    write_frame_artifact(labels, output_dir / "labels.table.json")
    print(f"features: {features.shape}; labels: {labels.shape}; output={output_dir}")


if __name__ == "__main__":
    main()
