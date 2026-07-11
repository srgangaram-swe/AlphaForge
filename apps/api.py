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

try:
    from fastapi import FastAPI, HTTPException
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install app extras with: pip install -e '.[app]'") from exc

DISCLAIMER = "Educational research output. Simulated results only. Not financial advice."

app = FastAPI(
    title="AlphaForge API",
    version="0.2.0",
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
    path = _run_dir_or_404() / "predictions.pkl"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no prediction panel in latest run")
    preds = pd.read_pickle(path)
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
