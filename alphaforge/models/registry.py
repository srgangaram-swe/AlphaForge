"""Config-driven model registry.

configs/models.yaml lists models by registry name + params; the walk-forward
driver instantiates them here. Optional dependencies (lightgbm, torch) fail
with actionable messages only when actually requested.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from alphaforge.models.base import AlphaModel
from alphaforge.models.baselines import HistoricalMeanBaseline, MomentumBaseline, ZeroBaseline
from alphaforge.models.ensemble import EnsembleModel
from alphaforge.models.sklearn_models import (
    make_elastic_net,
    make_gradient_boosting,
    make_lasso,
    make_linear,
    make_random_forest,
    make_ridge,
)


def _make_torch(kind: str) -> Callable[..., AlphaModel]:
    def factory(**params: Any) -> AlphaModel:
        from alphaforge.models import torch_models as tm

        cls = {"mlp": tm.TorchMLP, "gru": tm.TorchGRU, "tcn": tm.TorchTemporalCNN}[kind]
        return cls(**params)

    return factory


def _make_ensemble(members: list[dict], weights: list[float] | None = None) -> EnsembleModel:
    built = [create_model(m["name"], **m.get("params", {})) for m in members]
    return EnsembleModel(built, weights=weights)


MODEL_REGISTRY: dict[str, Callable[..., AlphaModel]] = {
    "zero_baseline": ZeroBaseline,
    "historical_mean": HistoricalMeanBaseline,
    "momentum_baseline": MomentumBaseline,
    "linear": make_linear,
    "ridge": make_ridge,
    "lasso": make_lasso,
    "elastic_net": make_elastic_net,
    "random_forest": make_random_forest,
    "gradient_boosting": make_gradient_boosting,
    "torch_mlp": _make_torch("mlp"),
    "torch_gru": _make_torch("gru"),
    "torch_tcn": _make_torch("tcn"),
    "ensemble": _make_ensemble,
}


def create_model(name: str, **params: Any) -> AlphaModel:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[name](**params)
    model.name = name
    return model


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)
