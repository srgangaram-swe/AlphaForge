"""Tests for the backtest orchestration service (SF-S2-MR10a)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from alphaforge.service import (
    BacktestRequest,
    BacktestResult,
    BacktestServiceError,
    available_baselines,
    available_strategy_models,
    discover_bundles,
    run_backtest_service,
)

# A small, fast configuration using cheap baseline "models" (no heavy training).
FAST: dict[str, Any] = dict(
    model="momentum_baseline",
    baselines=("zero_baseline", "historical_mean"),
    n_symbols=5,
    n_days=440,
    min_train_days=180,
    test_days=60,
    step_days=60,
    seed=7,
)


@pytest.fixture(scope="module")
def result() -> BacktestResult:
    return run_backtest_service(BacktestRequest(**FAST))


# --- Catalog helpers ---------------------------------------------------------


def test_catalog_helpers() -> None:
    models = available_strategy_models()
    assert "random_forest" in models and "equal_probability" not in models
    assert set(available_baselines()) <= set(models) | set(available_baselines())
    assert "zero_baseline" in available_baselines()
    assert discover_bundles("does/not/exist") == []


# --- Request validation (no pipeline run) ------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": "nope"},
        {"baselines": ("zero_baseline", "nope")},
        {"data_source": "mars"},
        {"data_source": "signal_foundry"},  # missing bundle_dir
        {"strategy": "bogus"},
        {"horizon": 0},
        {"cost_bps": -1.0},
        {"seed": -1},
        {"n_symbols": 1},
        {"min_train_days": 0},
    ],
)
def test_invalid_requests_fail_closed(overrides: dict) -> None:
    with pytest.raises(BacktestServiceError):
        BacktestRequest(**{**FAST, **overrides})


def test_baselines_dedupe_and_drop_headline() -> None:
    request = BacktestRequest(
        model="zero_baseline",
        baselines=("zero_baseline", "momentum_baseline", "momentum_baseline"),
    )
    assert request.baselines == ("momentum_baseline",)  # headline removed, deduped


def test_config_hash_is_stable_and_sensitive() -> None:
    a = BacktestRequest(**FAST)
    b = BacktestRequest(**FAST)
    c = BacktestRequest(**{**FAST, "seed": 999})
    assert a.config_hash() == b.config_hash()
    assert a.config_hash() != c.config_hash()


# --- Result structure and correctness ----------------------------------------


def test_headline_is_the_chosen_model(result: BacktestResult) -> None:
    assert result.headline.name == FAST["model"]
    assert result.headline.is_headline is True


def test_comparison_covers_model_and_all_baselines(result: BacktestResult) -> None:
    names = {row["name"] for row in result.comparison}
    assert names == {"momentum_baseline", "zero_baseline", "historical_mean"}
    assert result.headline.name == "momentum_baseline"
    # Here the headline model is itself a baseline, so all rows are flagged.
    assert all(row["is_baseline"] for row in result.comparison)


def test_non_baseline_headline_is_not_flagged_baseline() -> None:
    # A real model (linear) as headline is not flagged a baseline; the requested
    # baselines are. Linear trains cheaply, so this stays fast.
    request = BacktestRequest(
        model="linear",
        baselines=("zero_baseline",),
        n_symbols=4,
        n_days=380,
        min_train_days=160,
        test_days=60,
        step_days=60,
    )
    result = run_backtest_service(request)
    assert result.headline.name == "linear" and result.headline.is_baseline is False
    flags = {row["name"]: row["is_baseline"] for row in result.comparison}
    assert flags == {"linear": False, "zero_baseline": True}


def test_result_has_evidence(result: BacktestResult) -> None:
    assert result.n_windows >= 1
    assert result.n_observations > 0
    assert len(result.headline.equity_curve) > 0
    point = result.headline.equity_curve[0]
    assert set(point) == {"date", "strategy_cum", "benchmark_cum", "drawdown"}
    assert "sharpe" in result.headline.metrics


def test_reproducibility_fields(result: BacktestResult) -> None:
    repro = result.reproducibility
    assert repro["seed"] == FAST["seed"]
    assert repro["data_id"] == result.data_id
    assert result.data_id.startswith("synthetic:")
    assert len(repro["config_hash"]) == 16


def test_result_is_json_serializable(result: BacktestResult) -> None:
    text = json.dumps(result.to_dict())
    assert "Not financial advice" in text
    # no NaN/Infinity tokens leaked into the JSON
    assert "NaN" not in text and "Infinity" not in text


def test_deterministic_replay() -> None:
    a = run_backtest_service(BacktestRequest(**FAST))
    b = run_backtest_service(BacktestRequest(**FAST))
    assert a.headline.metrics == b.headline.metrics
    assert a.comparison == b.comparison
    assert a.reproducibility["config_hash"] == b.reproducibility["config_hash"]
