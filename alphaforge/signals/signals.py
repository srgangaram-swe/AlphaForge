"""Convert OOS predictions into tradable signal scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_COLUMNS = ["date", "symbol"]


def select_model_predictions(predictions: pd.DataFrame, model: str | None = None) -> pd.DataFrame:
    """Choose one model from a multi-model OOS prediction panel."""
    if "model" not in predictions.columns:
        return predictions.copy()
    models = list(pd.Series(predictions["model"].unique()).dropna())
    if model is not None:
        if model not in models:
            raise KeyError(f"model {model!r} not found in predictions; available={models}")
        chosen = model
    elif "ensemble" in models:
        chosen = "ensemble"
    elif "gradient_boosting" in models:
        chosen = "gradient_boosting"
    elif "ridge" in models:
        chosen = "ridge"
    else:
        chosen = sorted(models)[0]
    return predictions.loc[predictions["model"] == chosen].copy()


def _long_short(g: pd.DataFrame, prediction_col: str, quantile: float) -> pd.Series:
    n = len(g)
    if n == 0:
        return pd.Series(dtype=float)
    k = max(1, int(np.floor(n * quantile)))
    order = g[prediction_col].rank(method="first")
    signal = pd.Series(0.0, index=g.index)
    signal.loc[order <= k] = -1.0
    signal.loc[order > n - k] = 1.0
    if (signal > 0).any() and (signal < 0).any():
        return signal
    return pd.Series(0.0, index=g.index)


def _long_only_topk(g: pd.DataFrame, prediction_col: str, top_k: int) -> pd.Series:
    signal = pd.Series(0.0, index=g.index)
    if len(g) == 0:
        return signal
    winners = g[prediction_col].rank(method="first", ascending=False) <= top_k
    signal.loc[winners] = 1.0
    return signal


def _rank_weighted(g: pd.DataFrame, prediction_col: str) -> pd.Series:
    if g[prediction_col].nunique(dropna=True) < 2:
        return pd.Series(0.0, index=g.index)
    ranks = g[prediction_col].rank(pct=True) - 0.5
    return ranks / ranks.abs().sum()


def build_signals(
    predictions: pd.DataFrame,
    strategy: str = "long_short",
    params: dict | None = None,
    prediction_col: str = "prediction",
) -> pd.DataFrame:
    """Build a signal panel with values in [-1, 1]."""
    params = params or {}
    required = set(ID_COLUMNS + [prediction_col])
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"predictions missing columns: {sorted(missing)}")

    frame = predictions[ID_COLUMNS + [prediction_col]].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[prediction_col])
    out = []
    for date, g in frame.groupby("date", sort=True):
        if strategy == "long_short":
            signal = _long_short(g, prediction_col, float(params.get("quantile", 0.2)))
        elif strategy == "long_only_topk":
            signal = _long_only_topk(g, prediction_col, int(params.get("top_k", 5)))
        elif strategy == "rank_weighted":
            signal = _rank_weighted(g, prediction_col)
        elif strategy == "threshold":
            threshold = float(params.get("threshold", 0.0))
            allow_short = bool(params.get("allow_short", True))
            signal = pd.Series(0.0, index=g.index)
            signal.loc[g[prediction_col] > threshold] = 1.0
            if allow_short:
                signal.loc[g[prediction_col] < -threshold] = -1.0
        else:
            raise ValueError(f"unknown signal strategy: {strategy!r}")
        block = g[ID_COLUMNS + [prediction_col]].copy()
        block["signal"] = signal
        block["date"] = date
        out.append(block)
    if not out:
        return pd.DataFrame(columns=ID_COLUMNS + [prediction_col, "signal"])
    return pd.concat(out, ignore_index=True).sort_values(ID_COLUMNS).reset_index(drop=True)
