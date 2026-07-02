from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

try:
    from fastapi import FastAPI
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install app extras with: pip install -e '.[app]'") from exc

app = FastAPI(title="AlphaForge API", version="0.1.0")


def _latest_run() -> Path | None:
    pointer = Path("runs/latest_run.txt")
    return Path(pointer.read_text().strip()) if pointer.exists() else None


def _read_csv(name: str) -> list[dict[str, Any]]:
    run_dir = _latest_run()
    if run_dir is None:
        return []
    path = run_dir / name
    if not path.exists():
        return []
    return pd.read_csv(path).tail(200).to_dict(orient="records")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "latest_run": str(_latest_run()) if _latest_run() else None}


@app.get("/metrics")
def metrics() -> list[dict[str, Any]]:
    return _read_csv("model_metrics.csv")


@app.get("/signals")
def signals() -> list[dict[str, Any]]:
    return _read_csv("signals.csv")


@app.get("/portfolio")
def portfolio() -> list[dict[str, Any]]:
    return _read_csv("target_weights.csv")


@app.get("/risk")
def risk() -> dict[str, Any]:
    run_dir = _latest_run()
    if run_dir is None or not (run_dir / "backtest_summary.json").exists():
        return {}
    return pd.read_json(run_dir / "backtest_summary.json", typ="series").to_dict()


@app.get("/backtest")
def backtest() -> list[dict[str, Any]]:
    return _read_csv("equity_curve.csv")


@app.post("/predict")
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "message": "Prediction serving requires a persisted model artifact. Use walk-forward outputs for research predictions.",
        "received_keys": sorted(payload.keys()),
    }
