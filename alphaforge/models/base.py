"""Common model interface for all alpha models (sklearn, torch, baselines)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class AlphaModel(ABC):
    """A model mapping a feature matrix to expected forward returns.

    X is a DataFrame of numeric features; rows may carry a (date, symbol)
    MultiIndex, which sequence models use to build causal windows. Tabular
    models simply ignore the index.
    """

    name: str = "alpha_model"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> AlphaModel: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def feature_importance(self) -> pd.Series | None:
        """Optional: per-feature importance (higher = more important)."""
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
