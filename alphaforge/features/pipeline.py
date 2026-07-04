"""Feature pipeline: panel in, feature matrix out — with train-only scaling.

Output layout: one row per (date, symbol) with feature columns. The
benchmark symbol contributes market-regime features but is excluded from
the tradable rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.features.technical import (
    compute_benchmark_relative,
    compute_market_regime,
    compute_symbol_features,
)
from alphaforge.models.regime import causal_stress_probability

ID_COLUMNS = ["date", "symbol"]


def build_features(
    panel: pd.DataFrame,
    benchmark_symbol: str,
    config: dict | None = None,
) -> pd.DataFrame:
    """Build the full leak-safe feature matrix from a canonical OHLCV panel."""
    cfg = config or {}
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)

    bench_bars = panel[panel["symbol"] == benchmark_symbol]
    if bench_bars.empty:
        raise ValueError(f"benchmark symbol {benchmark_symbol!r} not found in panel")
    regime = compute_market_regime(bench_bars, cfg)
    bench_ret = regime[["date", "bench_ret"]]

    frames = []
    for symbol, g in panel.groupby("symbol"):
        if symbol == benchmark_symbol:
            continue
        g = g.sort_values("date")
        feats = compute_symbol_features(g, cfg)
        rel = compute_benchmark_relative(g, bench_ret, cfg)
        block = pd.concat(
            [
                g[ID_COLUMNS].reset_index(drop=True),
                feats.reset_index(drop=True),
                rel.reset_index(drop=True),
            ],
            axis=1,
        )
        frames.append(block)

    features = pd.concat(frames, ignore_index=True)
    features = features.merge(regime.drop(columns=["bench_ret"]), on="date", how="left")

    if cfg.get("hmm_regime", True):
        # causal HMM stress probability: expanding refits + filtered inference
        stress = causal_stress_probability(
            regime["bench_ret"],
            refit_every=int(cfg.get("hmm_refit_every", 63)),
            min_train=int(cfg.get("hmm_min_train", 252)),
        )
        hmm_frame = pd.DataFrame(
            {"date": regime["date"].to_numpy(), "hmm_stress_prob": stress.to_numpy()}
        )
        features = features.merge(hmm_frame, on="date", how="left")

    if cfg.get("cross_sectional", True):
        features = _add_cross_sectional(features)

    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    return features.sort_values(ID_COLUMNS).reset_index(drop=True)


def _add_cross_sectional(features: pd.DataFrame) -> pd.DataFrame:
    """Per-date cross-sectional ranks in [0, 1]. Uses only same-date data."""
    for col in ["momentum_20", "momentum_60", "vol_20", "ret_5", "rolling_sharpe"]:
        if col in features.columns:
            features[f"cs_rank_{col}"] = features.groupby("date")[col].rank(pct=True)
    return features


def feature_columns(features: pd.DataFrame) -> list[str]:
    """All model input columns (everything except identifiers)."""
    return [c for c in features.columns if c not in ID_COLUMNS]


class FeatureScaler:
    """Z-score scaler whose statistics are fit on training rows only.

    Fitting on the full panel would leak test-period distribution information
    into training — a classic subtle leak. This class makes the train-only
    contract explicit and testable.
    """

    def __init__(self, clip: float = 5.0):
        self.clip = clip
        self.means_: pd.Series | None = None
        self.stds_: pd.Series | None = None
        self.columns_: list[str] | None = None

    def fit(self, X: pd.DataFrame) -> FeatureScaler:
        self.columns_ = list(X.columns)
        self.means_ = X.mean()
        self.stds_ = X.std().replace(0, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.means_ is None or self.stds_ is None:
            raise RuntimeError("FeatureScaler must be fit before transform")
        Z = (X[self.columns_] - self.means_) / self.stds_
        return Z.clip(-self.clip, self.clip)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
