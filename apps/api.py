"""FastAPI service over the latest AlphaForge run artifacts.

Serves *research* outputs (out-of-sample walk-forward predictions, signals,
weights, risk analytics) — not live inference. Everything returned here is
simulated/backtested and carries the project's educational disclaimer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from alphaforge.research import read_frame_artifact

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install app extras with: pip install -e '.[app]'") from exc

from alphaforge.service import (
    BacktestRequest,
    BacktestServiceError,
    available_baselines,
    available_strategy_models,
    discover_bundles,
    run_backtest_service,
)

DISCLAIMER = "Educational research output. Simulated results only. Not financial advice."

app = FastAPI(
    title="AlphaForge API",
    version="0.2.1",
    description=DISCLAIMER,
)


def _latest_run() -> Path | None:
    pointer = Path("runs/latest_run.txt")
    return Path(pointer.read_text().strip()) if pointer.exists() else None


def _run_dir_or_404() -> Path:
    run_dir = _latest_run()
    if run_dir is None or not run_dir.exists():
        raise HTTPException(status_code=404, detail="no completed run found; run `make demo`")
    return run_dir


def _read_csv(name: str, tail: int = 200) -> list[dict[str, Any]]:
    path = _run_dir_or_404() / name
    if not path.exists():
        return []
    return pd.read_csv(path).tail(tail).to_dict(orient="records")


def _read_json(name: str) -> dict[str, Any]:
    path = _run_dir_or_404() / name
    return json.loads(path.read_text()) if path.exists() else {}


@app.get("/health")
def health() -> dict[str, Any]:
    run_dir = _latest_run()
    native = False
    try:
        from alphaforge.execution import NATIVE_AVAILABLE

        native = NATIVE_AVAILABLE
    except ImportError:
        pass
    return {
        "status": "ok",
        "latest_run": str(run_dir) if run_dir else None,
        "native_execution_core": native,
        "disclaimer": DISCLAIMER,
    }


@app.get("/predict")
def predict(symbol: str | None = None, model: str | None = None) -> dict[str, Any]:
    """Latest out-of-sample predictions from the saved walk-forward panel.

    These are research predictions generated strictly out-of-sample during
    walk-forward validation — not a live model endpoint.
    """
    path = _run_dir_or_404() / "predictions.table.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no prediction panel in latest run")
    preds = read_frame_artifact(path)
    if model is not None:
        if model not in set(preds["model"]):
            raise HTTPException(status_code=404, detail=f"model {model!r} not in run")
        preds = preds[preds["model"] == model]
    if symbol is not None:
        preds = preds[preds["symbol"] == symbol.upper()]
        if preds.empty:
            raise HTTPException(status_code=404, detail=f"symbol {symbol!r} not in run")
    latest_date = preds["date"].max()
    latest = preds[preds["date"] == latest_date]
    return {
        "as_of": str(latest_date),
        "disclaimer": DISCLAIMER,
        "predictions": latest[["symbol", "model", "prediction"]].to_dict(orient="records"),
    }


@app.get("/signals")
def signals() -> list[dict[str, Any]]:
    return _read_csv("signals.csv")


@app.get("/portfolio")
def portfolio() -> dict[str, Any]:
    return {
        "disclaimer": DISCLAIMER,
        "target_weights": _read_csv("target_weights.csv"),
        "executed_weights": _read_csv("executed_weights.csv"),
    }


@app.get("/backtest")
def backtest() -> dict[str, Any]:
    return {
        "summary": _read_json("backtest_summary.json"),
        "equity_curve_tail": _read_csv("equity_curve.csv"),
        "fills_tail": _read_csv("fills.csv"),
        "pnl_attribution_tail": _read_csv("pnl_attribution.csv"),
        "disclaimer": DISCLAIMER,
    }


@app.get("/risk")
def risk() -> dict[str, Any]:
    return {
        "summary": _read_json("backtest_summary.json"),
        "overfitting": _read_json("overfitting.json"),
        "stress_tests": _read_csv("stress_tests.csv"),
        "regime_performance": _read_csv("regime_performance.csv"),
        "capacity_curve": _read_csv("capacity_curve.csv"),
        "capacity_diagnostics": _read_json("capacity_diagnostics.json"),
        "disclaimer": DISCLAIMER,
    }


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return {
        "model_metrics": _read_csv("model_metrics.csv"),
        "ic_summary": _read_csv("ic_summary.csv"),
        "ic_decay": _read_csv("ic_decay.csv"),
        "quantile_returns": _read_csv("quantile_returns.csv"),
    }


# --- Interactive backtesting (SF-S2-MR10b): configure and run on demand -------


class BacktestSpec(BaseModel):
    """Request body for an on-demand backtest (mirrors the service contract)."""

    data_source: str = "synthetic"
    bundle_dir: str | None = None
    n_symbols: int = Field(default=8, ge=2, le=100)
    n_days: int = Field(default=600, ge=120, le=5000)
    benchmark_symbol: str = "BENCH"
    model: str = "random_forest"
    model_params: dict[str, Any] = Field(default_factory=dict)
    baselines: list[str] = Field(
        default_factory=lambda: ["zero_baseline", "historical_mean", "momentum_baseline"]
    )
    horizon: int = Field(default=1, ge=1, le=60)
    strategy: str = "long_short"
    cost_bps: float = Field(default=1.0, ge=0.0, le=100.0)
    seed: int = Field(default=42, ge=0)
    min_train_days: int = Field(default=252, ge=20)
    test_days: int = Field(default=63, ge=1)
    step_days: int = Field(default=63, ge=1)
    embargo_days: int = Field(default=10, ge=0)


@app.get("/catalog")
def catalog() -> dict[str, Any]:
    """Available models, baselines, strategies, and discoverable data bundles."""
    return {
        "models": available_strategy_models(),
        "baselines": available_baselines(),
        "strategies": ["long_short", "long_only_topk", "rank_weighted", "confidence"],
        "data_sources": ["synthetic", "signal_foundry"],
        "bundles": discover_bundles(),
        "disclaimer": DISCLAIMER,
    }


@app.post("/backtests")
def create_backtest(spec: BacktestSpec) -> dict[str, Any]:
    """Run a leakage-safe walk-forward backtest for a model and its baselines.

    Simulated research over the deterministic synthetic market (default) or a
    Signal Foundry bundle produced by Signalattice — not live or executable.
    """
    try:
        request = BacktestRequest(**spec.model_dump())
        result = run_backtest_service(request)
    except BacktestServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return result.to_dict()
