"""Baselines every ML model must beat to justify its complexity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.models.base import AlphaModel


class ZeroBaseline(AlphaModel):
    """Predicts zero return everywhere — the efficient-market null."""

    name = "zero_baseline"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ZeroBaseline:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class HistoricalMeanBaseline(AlphaModel):
    """Predicts the training-set mean target for every row."""

    name = "historical_mean"

    def __init__(self):
        self.mean_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> HistoricalMeanBaseline:
        self.mean_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_)


class MomentumBaseline(AlphaModel):
    """Predicts a scaled momentum feature — the classic cross-sectional signal.

    No fitting: this is a rule, not a model, which is exactly why it is a
    useful baseline. If ML cannot beat `0.05 * momentum_20`, the ML is noise.
    """

    name = "momentum_baseline"

    def __init__(self, feature: str = "momentum_20", scale: float = 0.05):
        self.feature = feature
        self.scale = scale

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MomentumBaseline:
        if self.feature not in X.columns:
            raise ValueError(f"feature {self.feature!r} not in X")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (X[self.feature].fillna(0.0) * self.scale).to_numpy()
