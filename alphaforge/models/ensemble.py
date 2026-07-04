"""Ensembles: equal-weight or IC-weighted combination of member models.

Cross-sectional alpha signals are usually combined, not selected: averaging
decorrelated weak signals adds more than tuning any single model harder. The
IC-weighted variant estimates each member's skill on a *time-ordered* inner
validation split (never a random one — that would leak) and weights members
by their positive rank IC, then refits every member on the full window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.models.base import AlphaModel


def _rank_ic(y: pd.Series, pred: np.ndarray) -> float:
    frame = pd.DataFrame({"y": y.to_numpy(), "p": pred}).dropna()
    if len(frame) < 10 or frame["p"].std() == 0 or frame["y"].std() == 0:
        return 0.0
    return float(frame["y"].rank().corr(frame["p"].rank()))


class EnsembleModel(AlphaModel):
    """Combine member predictions with equal or IC-based weights.

    weighting="ic": members are fit on the first (1 - val_fraction) of the
    training rows (rows arrive date-sorted from the walk-forward driver, so
    this is a chronological split), scored by rank IC on the held-out tail,
    weighted by max(IC, 0) + floor, then refit on the full training window.
    """

    name = "ensemble"

    def __init__(
        self,
        members: list[AlphaModel],
        weights: list[float] | None = None,
        weighting: str = "equal",
        val_fraction: float = 0.2,
        min_weight: float = 0.05,
    ):
        if not members:
            raise ValueError("ensemble needs at least one member")
        if weighting not in {"equal", "ic"}:
            raise ValueError("weighting must be 'equal' or 'ic'")
        self.members = members
        self.weighting = weighting
        self.val_fraction = val_fraction
        self.min_weight = min_weight
        w = np.asarray(weights if weights is not None else [1.0] * len(members), dtype=float)
        self.weights = w / w.sum()
        self.member_ics_: list[float] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> EnsembleModel:
        if self.weighting == "ic" and len(X) >= 50:
            self._fit_ic_weights(X, y)
        for m in self.members:
            m.fit(X, y)
        return self

    def _fit_ic_weights(self, X: pd.DataFrame, y: pd.Series) -> None:
        n_val = max(10, int(len(X) * self.val_fraction))
        X_tr, y_tr = X.iloc[:-n_val], y.iloc[:-n_val]
        X_val, y_val = X.iloc[-n_val:], y.iloc[-n_val:]
        ics = []
        for m in self.members:
            try:
                m.fit(X_tr, y_tr)
                ics.append(_rank_ic(y_val, np.asarray(m.predict(X_val))))
            except Exception:
                ics.append(0.0)
        self.member_ics_ = ics
        raw = np.clip(np.asarray(ics, dtype=float), 0.0, None) + self.min_weight
        self.weights = raw / raw.sum()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.column_stack([m.predict(X) for m in self.members])
        return preds @ self.weights

    def feature_importance(self) -> pd.Series | None:
        imps = [m.feature_importance() for m in self.members]
        imps = [i / i.sum() for i in imps if i is not None and i.sum() > 0]
        if not imps:
            return None
        return pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
