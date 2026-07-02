"""Ensemble: average (optionally weighted) of member model predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.models.base import AlphaModel


class EnsembleModel(AlphaModel):
    """Fits every member on the same data and averages predictions.

    Cross-sectional signals are commonly combined this way; averaging
    decorrelated weak signals is usually worth more than tuning any single
    model harder.
    """

    name = "ensemble"

    def __init__(self, members: list[AlphaModel], weights: list[float] | None = None):
        if not members:
            raise ValueError("ensemble needs at least one member")
        self.members = members
        w = np.asarray(weights if weights is not None else [1.0] * len(members), dtype=float)
        self.weights = w / w.sum()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> EnsembleModel:
        for m in self.members:
            m.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.column_stack([m.predict(X) for m in self.members])
        return preds @ self.weights

    def feature_importance(self) -> pd.Series | None:
        imps = [m.feature_importance() for m in self.members]
        imps = [i / i.sum() for i in imps if i is not None and i.sum() > 0]
        if not imps:
            return None
        return pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
