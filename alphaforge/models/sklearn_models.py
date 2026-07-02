"""Scikit-learn model wrappers.

All wrappers embed imputation + scaling in an sklearn Pipeline, so those
statistics are fit on the training window only — the walk-forward driver
never has to remember to scale separately.

Gradient boosting prefers LightGBM when installed and falls back to
sklearn's HistGradientBoostingRegressor, so the core install stays light.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from alphaforge.models.base import AlphaModel


class SklearnModel(AlphaModel):
    """Generic wrapper: impute -> scale -> estimator."""

    def __init__(self, estimator, name: str, scale: bool = True):
        steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
        if scale:
            steps.append(("scale", StandardScaler()))
        steps.append(("model", estimator))
        self.pipeline = Pipeline(steps)
        self.name = name
        self.columns_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SklearnModel:
        self.columns_ = list(X.columns)
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline.predict(X[self.columns_]))

    def feature_importance(self) -> pd.Series | None:
        est = self.pipeline.named_steps["model"]
        if hasattr(est, "feature_importances_"):
            vals = est.feature_importances_
        elif hasattr(est, "coef_"):
            vals = np.abs(np.ravel(est.coef_))
        else:
            return None
        return pd.Series(vals, index=self.columns_).sort_values(ascending=False)


def make_linear(**params) -> SklearnModel:
    return SklearnModel(LinearRegression(**params), "linear")


def make_ridge(alpha: float = 10.0, **params) -> SklearnModel:
    return SklearnModel(Ridge(alpha=alpha, **params), "ridge")


def make_lasso(alpha: float = 1e-4, **params) -> SklearnModel:
    return SklearnModel(Lasso(alpha=alpha, max_iter=10_000, **params), "lasso")


def make_elastic_net(alpha: float = 1e-3, l1_ratio: float = 0.5, **params) -> SklearnModel:
    return SklearnModel(
        ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10_000, **params), "elastic_net"
    )


def make_random_forest(random_state: int = 42, **params) -> SklearnModel:
    defaults = dict(n_estimators=200, max_depth=6, min_samples_leaf=50, n_jobs=-1)
    defaults.update(params)
    return SklearnModel(
        RandomForestRegressor(random_state=random_state, **defaults), "random_forest", scale=False
    )


def make_gradient_boosting(random_state: int = 42, **params) -> AlphaModel:
    try:
        import lightgbm as lgb

        defaults = dict(
            n_estimators=params.pop("max_iter", 300),
            max_depth=params.pop("max_depth", 4),
            learning_rate=params.pop("learning_rate", 0.05),
            verbose=-1,
        )
        defaults.update(params)
        return SklearnModel(
            lgb.LGBMRegressor(random_state=random_state, **defaults),
            "gradient_boosting",
            scale=False,
        )
    except ImportError:
        defaults = dict(max_iter=300, max_depth=4, learning_rate=0.05)
        defaults.update(params)
        return SklearnModel(
            HistGradientBoostingRegressor(random_state=random_state, **defaults),
            "gradient_boosting",
            scale=False,
        )
