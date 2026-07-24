"""End-to-end governed Signal Foundry research tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from alphaforge.data import SignalFoundryDataset, SyntheticMarketConfig, generate_synthetic_market
from alphaforge.evaluation import NOT_READY, ReadinessThresholds
from alphaforge.research import GovernedResearchConfig, run_governed_signal_foundry_research


def _dataset(tmp_path: Path) -> SignalFoundryDataset:
    panel = generate_synthetic_market(
        SyntheticMarketConfig(
            n_symbols=6,
            n_days=420,
            seed=7,
            benchmark_symbol="SPY",
        )
    )
    return SignalFoundryDataset(
        bundle_dir=tmp_path / "synthetic-bundle",
        manifest={
            "bundle_id": "a" * 64,
            "license": {
                "observations_redistributable": True,
                "bundle_must_remain_local": False,
                "public_evidence_must_be_aggregate_or_synthetic": False,
            },
            "point_in_time_limits": {
                "historical_revisions_complete": False,
                "universe_membership_point_in_time": False,
                "corporate_actions_complete": False,
            },
        },
        source_panel=pd.DataFrame(),
        panel=panel,
    )


def _run(tmp_path: Path):
    dataset = _dataset(tmp_path)
    dates = sorted(dataset.panel["date"].unique())
    return run_governed_signal_foundry_research(
        dataset=dataset,
        model_specs=[
            {"name": "momentum_baseline", "params": {"feature": "momentum_20", "scale": 0.05}},
            {"name": "ridge", "params": {"alpha": 10.0}},
        ],
        feature_config={
            "return_lags": [1, 5],
            "vol_windows": [5, 20],
            "ma_windows": [5, 20],
            "momentum_windows": [5, 20],
            "rsi_window": 5,
            "macd": {"fast": 3, "slow": 8, "signal": 3},
            "bollinger_window": 5,
            "mean_reversion_window": 3,
            "volume_window": 5,
            "beta_window": 20,
            "rolling_sharpe_window": 20,
            "drawdown_window": 20,
            "regime_vol_window": 5,
            "regime_trend_fast": 5,
            "regime_trend_slow": 20,
            "hmm_regime": False,
            "cross_sectional": True,
        },
        walk_forward_config={
            "scheme": "expanding",
            "min_train_days": 160,
            "test_days": 40,
            "step_days": 40,
            "embargo_days": 5,
        },
        backtest_config={
            "strategy": "long_short",
            "strategy_params": {"quantile": 0.25},
            "rebalance_frequency": 5,
            "execution_lag": 1,
            "liquidate_at_end": True,
            "initial_capital": 1_000_000,
            "execution": {
                "adv_lookback": 5,
                "volatility_lookback": 5,
                "max_participation_rate": 0.05,
                "impact_coefficient": 0.10,
                "missing_price_policy": "raise",
            },
            "costs": {
                "commission_bps": 1.0,
                "half_spread_bps": 2.5,
                "slippage_bps": 2.0,
            },
            "portfolio": {
                "max_weight": 0.25,
                "max_gross_exposure": 1.0,
                "inverse_vol_scaling": True,
                "turnover_cap": 0.50,
            },
            "risk": {},
            "capacity": {
                "aum_multiples": [0.5, 1.0, 2.0],
                "max_participation_rate": 0.05,
                "minimum_fill_ratio": 0.95,
            },
            "borrow_financing": {
                "short_borrow_bps_annual": 1000.0,
                "cash_financing_bps_annual": 500.0,
            },
        },
        research_config=GovernedResearchConfig(
            holdout_start=str(pd.Timestamp(dates[300]).date()),
            benchmark_symbol="SPY",
            target="fwd_ret_5",
            horizons=(1, 5),
            seed=17,
        ),
        readiness_thresholds=ReadinessThresholds(
            minimum_holdout_days=40,
            minimum_deflated_sharpe_probability=0.50,
            maximum_average_turnover=1.0,
        ),
        output_root=tmp_path / "runs",
        code_sha="b" * 40,
    )


def test_governed_run_is_transactional_auditable_and_not_ready_on_missing_pit(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path)

    assert result.run_dir.name == result.run_id
    assert result.dossier["decision"] == NOT_READY
    assert "point_in_time_evidence" in result.dossier["failed_gates"]
    assert (result.run_dir / "dossier.md").is_file()
    assert (result.run_dir / "final_holdout_predictions.csv").is_file()
    assert (result.run_dir / "capacity_curve.csv").is_file()
    manifest = json.loads((result.run_dir / "run_manifest.json").read_text())
    assert manifest["trial_ledger_head"]
    ledger = [
        json.loads(line)
        for line in (result.run_dir / "trial_ledger.jsonl").read_text().splitlines()
    ]
    assert ledger[0]["previous_hash"] == "0" * 64
    assert ledger[1]["previous_hash"] == ledger[0]["record_hash"]
    assert result.dossier["gates"]["missing_price_halt"]
    assert result.dossier["uncertainty"]["available"]
    assert result.dossier["year_stability"]
    assert result.dossier["regime_stability"]


def test_final_holdout_identity_cannot_be_overwritten_or_repeated(tmp_path: Path) -> None:
    first = _run(tmp_path)

    with pytest.raises(FileExistsError, match="cannot be repeated"):
        _run(tmp_path)

    assert first.run_dir.is_dir()
    assert not list((tmp_path / "runs").glob(".publishing-*"))


def test_clean_output_roots_produce_byte_identical_evidence(tmp_path: Path) -> None:
    first = _run(tmp_path / "first")
    second = _run(tmp_path / "second")

    assert first.run_id == second.run_id
    first_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in first.run_dir.iterdir()
    }
    second_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in second.run_dir.iterdir()
    }
    assert first_hashes == second_hashes
