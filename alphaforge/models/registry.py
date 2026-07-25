"""Config-driven model registry.

configs/models.yaml lists models by registry name + params; the walk-forward
driver instantiates them here. Optional dependencies (lightgbm, torch) fail
with actionable messages only when actually requested.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
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


def _make_temporal(**params: Any) -> AlphaModel:
    from alphaforge.models.temporal import TemporalAlphaModel

    return TemporalAlphaModel(**params)


def _make_ensemble(members: list[dict], **kwargs: Any) -> EnsembleModel:
    built = [create_model(m["name"], **m.get("params", {})) for m in members]
    return EnsembleModel(built, **kwargs)


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
    "temporal_alpha": _make_temporal,
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


def seed_model_specs(model_specs: list[dict[str, Any]], root_seed: int) -> list[dict[str, Any]]:
    """Inject the declared root seed into every applicable model backend.

    The transformation is independent of candidate order and never overwrites
    an explicitly pre-registered model seed.
    """

    if root_seed < 0:
        raise ValueError("root_seed must be non-negative")
    seeded = deepcopy(model_specs)
    for spec in seeded:
        name = str(spec.get("name", ""))
        params = spec.setdefault("params", {})
        if not isinstance(params, dict):
            raise TypeError(f"model {name!r} params must be a mapping")
        if name in {"random_forest", "gradient_boosting"}:
            params.setdefault("random_state", root_seed)
        elif name in {"torch_mlp", "torch_gru", "torch_tcn", "temporal_alpha"}:
            params.setdefault("seed", root_seed)
        elif name == "ensemble":
            members = params.get("members")
            if not isinstance(members, list):
                raise TypeError("ensemble members must be a list")
            params["members"] = seed_model_specs(members, root_seed)
    return seeded
