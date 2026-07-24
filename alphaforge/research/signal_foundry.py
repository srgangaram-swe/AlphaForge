"""Governed development-selection and frozen final-holdout research.

The workflow consumes one verified Signal Foundry bundle, selects a candidate
using development-only walk-forward evidence, purges an embargo before the
pre-registered holdout, evaluates that holdout once per immutable run identity,
and emits an auditable paper-readiness dossier. It has no live-order path.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alphaforge.backtesting import BacktestResult, run_backtest
from alphaforge.data.signal_foundry import SignalFoundryDataset
from alphaforge.evaluation import (
    CapacityColumns,
    CapacityConfig,
    ReadinessThresholds,
    assess_paper_readiness,
    estimate_capacity,
    information_coefficient_by_date,
    probability_of_backtest_overfitting,
)
from alphaforge.features import build_features
from alphaforge.labels.labels import build_labels
from alphaforge.models.registry import create_model
from alphaforge.portfolio import construct_portfolio
from alphaforge.risk import drawdown_series, performance_summary, regime_performance
from alphaforge.signals import build_signals
from alphaforge.training import run_walk_forward
from alphaforge.training.walk_forward import supervised_frame


@dataclass(frozen=True)
class GovernedResearchConfig:
    """Pre-registered final-holdout and model-selection policy."""

    holdout_start: str
    benchmark_symbol: str = "SPY"
    target: str = "fwd_ret_5"
    horizons: tuple[int, ...] = (1, 5, 20)
    selection_metric: str = "rank_ic"
    seed: int = 42

    def __post_init__(self) -> None:
        holdout = pd.Timestamp(self.holdout_start)
        if holdout.tzinfo is not None:
            raise ValueError("holdout_start must be a timezone-naive market date")
        if not self.benchmark_symbol.strip():
            raise ValueError("benchmark_symbol must be non-empty")
        if not self.target.strip():
            raise ValueError("target must be non-empty")
        if not self.horizons or any(horizon < 1 for horizon in self.horizons):
            raise ValueError("horizons must contain positive integers")
        if self.selection_metric not in {"rank_ic", "ic"}:
            raise ValueError("selection_metric must be rank_ic or ic")


@dataclass(frozen=True)
class GovernedResearchResult:
    """Identity and artifacts for one immutable final-holdout evaluation."""

    run_id: str
    run_dir: Path
    candidate_model: str
    dossier: dict[str, Any]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        raise RuntimeError("cannot determine producer Git SHA for governed run")
    value = result.stdout.strip()
    if len(value) != 40:
        raise RuntimeError("governed run requires a full Git SHA")
    return value


def _dependency_versions() -> dict[str, str]:
    """Fingerprint the numerical stack that can affect governed evidence."""
    packages = (
        "alphaforge",
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "scipy",
    )
    resolved: dict[str, str] = {}
    for package in packages:
        try:
            resolved[package] = version(package)
        except PackageNotFoundError:
            resolved[package] = "not-installed"
    return resolved


def _model_matrix(frame: pd.DataFrame, columns: list[str], model: Any) -> pd.DataFrame:
    matrix = frame[columns].copy()
    if getattr(model, "needs_sequence_index", False):
        matrix.index = pd.MultiIndex.from_frame(frame[["date", "symbol"]])
    return matrix


def _validate_model_specs(model_specs: list[dict[str, Any]]) -> None:
    if not model_specs:
        raise ValueError("at least one pre-registered model is required")
    names: list[str] = []
    for spec in model_specs:
        if set(spec) - {"name", "params"}:
            raise ValueError(
                f"unknown model specification fields: {sorted(set(spec) - {'name', 'params'})}"
            )
        name = spec.get("name")
        params = spec.get("params", {})
        if not isinstance(name, str) or not name.strip() or not isinstance(params, dict):
            raise ValueError("each model specification requires a name and parameter mapping")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("model names must be unique within a governed trial set")


def _development_cutoff(
    dates: pd.Series,
    holdout_start: pd.Timestamp,
    embargo_sessions: int,
) -> pd.Timestamp:
    unique = pd.DatetimeIndex(pd.to_datetime(dates).drop_duplicates().sort_values())
    holdout_position = int(unique.searchsorted(holdout_start, side="left"))
    cutoff_position = holdout_position - embargo_sessions - 1
    if holdout_position >= len(unique) or cutoff_position < 0:
        raise ValueError("holdout_start leaves insufficient pre-holdout embargo history")
    return pd.Timestamp(unique[cutoff_position])


def _select_candidate(
    metrics: pd.DataFrame,
    *,
    selection_metric: str,
) -> tuple[str, pd.DataFrame]:
    if metrics.empty or selection_metric not in metrics:
        raise ValueError("development walk-forward produced no selection evidence")
    summary = (
        metrics.groupby("model", as_index=False)
        .agg(
            windows=("window_id", "nunique"),
            rank_ic=("rank_ic", "mean"),
            ic=("ic", "mean"),
            mae=("mae", "mean"),
        )
        .sort_values(
            [selection_metric, "mae", "model"],
            ascending=[False, True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    eligible = summary.loc[np.isfinite(summary[selection_metric])].copy()
    if eligible.empty:
        raise ValueError("no model produced finite development selection evidence")
    return str(eligible.iloc[0]["model"]), summary


def _trial_ledger(
    model_specs: list[dict[str, Any]],
    development_summary: pd.DataFrame,
) -> list[dict[str, Any]]:
    evidence = development_summary.set_index("model").to_dict(orient="index")
    previous_hash = "0" * 64
    records: list[dict[str, Any]] = []
    for sequence, spec in enumerate(model_specs, start=1):
        payload = {
            "sequence": sequence,
            "model": spec["name"],
            "params": spec.get("params", {}),
            "development_evidence": evidence.get(spec["name"], {}),
            "previous_hash": previous_hash,
        }
        record_hash = _sha256_bytes(_canonical_json(payload))
        records.append({**payload, "record_hash": record_hash})
        previous_hash = record_hash
    return records


def _daily_ic_matrix(predictions: pd.DataFrame) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for model_name, block in predictions.groupby("model", sort=True):
        daily = information_coefficient_by_date(block)
        columns[str(model_name)] = daily.set_index("date")["rank_ic"]
    return pd.DataFrame(columns).sort_index()


def _pbo(predictions: pd.DataFrame, seed: int) -> dict[str, Any]:
    matrix = _daily_ic_matrix(predictions)
    max_blocks = min(16, len(matrix) // 2)
    n_blocks = max_blocks - max_blocks % 2
    if n_blocks < 2:
        return {"pbo": np.nan, "n_combinations": 0, "n_obs": len(matrix)}
    result = probability_of_backtest_overfitting(
        matrix,
        n_blocks=n_blocks,
        seed=seed,
    )
    return {key: value for key, value in result.items() if key != "logits"}


def _backtest(
    *,
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    benchmark_symbol: str,
    backtest_config: dict[str, Any],
) -> BacktestResult:
    signal_config = backtest_config.get("strategy_params", {})
    signals = build_signals(
        predictions,
        strategy=str(backtest_config.get("strategy", "long_short")),
        params=signal_config,
    )
    weights = construct_portfolio(
        signals,
        features=features,
        config=backtest_config.get("portfolio", {}),
    )
    return run_backtest(
        panel=panel,
        target_weights=weights,
        benchmark_symbol=benchmark_symbol,
        initial_capital=float(backtest_config.get("initial_capital", 1_000_000.0)),
        execution_lag=int(backtest_config.get("execution_lag", 1)),
        rebalance_frequency=int(backtest_config.get("rebalance_frequency", 1)),
        costs=backtest_config.get("costs", {}),
        risk=backtest_config.get("risk", {}),
        execution=backtest_config.get("execution", {}),
        liquidate_at_end=bool(backtest_config.get("liquidate_at_end", True)),
    )


def _stress_scenarios(
    *,
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    benchmark_symbol: str,
    backtest_config: dict[str, Any],
    maximum_drawdown: float,
    seed: int,
) -> tuple[list[dict[str, Any]], bool]:
    scenarios: list[tuple[str, dict[str, Any]]] = []
    doubled_costs = dict(backtest_config)
    doubled_costs["costs"] = {
        key: float(value) * 2.0 for key, value in dict(backtest_config.get("costs", {})).items()
    }
    scenarios.append(("doubled_costs", doubled_costs))

    adverse_spread = dict(backtest_config)
    adverse_spread["costs"] = dict(backtest_config.get("costs", {}))
    for field in ("half_spread_bps", "slippage_bps"):
        adverse_spread["costs"][field] = float(adverse_spread["costs"].get(field, 0.0)) * 3.0
    scenarios.append(("tripled_spread_and_slippage", adverse_spread))

    delayed = dict(backtest_config)
    delayed["execution_lag"] = int(backtest_config.get("execution_lag", 1)) + 1
    scenarios.append(("additional_session_delay", delayed))

    lower_liquidity = dict(backtest_config)
    lower_liquidity["execution"] = dict(backtest_config.get("execution", {}))
    base_participation = float(lower_liquidity["execution"].get("max_participation_rate", 0.05))
    lower_liquidity["execution"]["max_participation_rate"] = base_participation / 2.0
    scenarios.append(("halved_participation", lower_liquidity))

    base_strategy = dict(backtest_config.get("strategy_params", {}))
    base_quantile = float(base_strategy.get("quantile", 0.20))
    for name, quantile in (
        ("narrower_selection", base_quantile * 0.75),
        ("wider_selection", min(base_quantile * 1.25, 0.49)),
    ):
        perturbed = dict(backtest_config)
        perturbed["strategy_params"] = {**base_strategy, "quantile": quantile}
        scenarios.append((name, perturbed))

    summaries: list[dict[str, Any]] = []
    passed = True
    for name, scenario_config in scenarios:
        result = _backtest(
            panel=panel,
            predictions=predictions,
            features=features,
            benchmark_symbol=benchmark_symbol,
            backtest_config=scenario_config,
        )
        summary = performance_summary(result.equity_curve)
        scenario_passed = bool(
            np.isfinite(summary["max_drawdown"]) and summary["max_drawdown"] >= -maximum_drawdown
        )
        passed = passed and scenario_passed
        summaries.append({"scenario": name, "passed": scenario_passed, **summary})

    placebo = predictions.copy()
    rng = np.random.default_rng(seed)
    placebo["prediction"] = placebo.groupby("date", sort=True)["prediction"].transform(
        lambda values: rng.permutation(values.to_numpy())
    )
    placebo_result = _backtest(
        panel=panel,
        predictions=placebo,
        features=features,
        benchmark_symbol=benchmark_symbol,
        backtest_config=backtest_config,
    )
    placebo_summary = performance_summary(placebo_result.equity_curve)
    summaries.append({"scenario": "permuted_signal_placebo", **placebo_summary})
    return summaries, passed


def _bootstrap_uncertainty(
    returns: pd.Series,
    *,
    seed: int,
    n_resamples: int = 500,
    block_size: int = 20,
) -> dict[str, Any]:
    """Circular moving-block intervals for serially dependent daily returns."""
    clean = returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(clean) < block_size * 2:
        return {
            "method": "circular_moving_block_bootstrap",
            "n_resamples": n_resamples,
            "block_size": block_size,
            "available": False,
        }
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(clean) / block_size))
    annual_returns: list[float] = []
    sharpes: list[float] = []
    for _ in range(n_resamples):
        starts = rng.integers(0, len(clean), size=n_blocks)
        sample = np.concatenate(
            [np.take(clean, np.arange(start, start + block_size), mode="wrap") for start in starts]
        )[: len(clean)]
        years = len(sample) / 252.0
        annual_returns.append(float(np.prod(1.0 + sample) ** (1.0 / years) - 1.0))
        volatility = float(np.std(sample, ddof=1))
        sharpes.append(
            np.nan if volatility == 0.0 else float(np.mean(sample) / volatility * np.sqrt(252.0))
        )
    return {
        "method": "circular_moving_block_bootstrap",
        "n_resamples": n_resamples,
        "block_size": block_size,
        "seed": seed,
        "available": True,
        "annual_return_ci_95": np.quantile(annual_returns, [0.025, 0.975]).tolist(),
        "sharpe_ci_95": np.nanquantile(sharpes, [0.025, 0.975]).tolist(),
    }


def _drawdown_diagnostics(equity_curve: pd.DataFrame) -> dict[str, Any]:
    drawdown = drawdown_series(equity_curve["equity"].astype(float))
    underwater = drawdown < 0.0
    longest = 0
    current = 0
    for value in underwater:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return {
        "time_under_drawdown_fraction": float(underwater.mean()),
        "maximum_drawdown_duration_sessions": longest,
    }


def _year_stability(equity_curve: pd.DataFrame) -> list[dict[str, Any]]:
    frame = equity_curve.copy()
    frame["year"] = pd.to_datetime(frame["date"]).dt.year
    records: list[dict[str, Any]] = []
    for year, block in frame.groupby("year", sort=True):
        records.append({"year": int(year), **performance_summary(block, trim_inactive=False)})
    return records


def _borrow_financing_sensitivity(
    equity_curve: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Apply explicit post-ledger borrow/funding drag as a conservative proxy."""
    expected = {"short_borrow_bps_annual", "cash_financing_bps_annual"}
    if set(config) != expected:
        raise ValueError(
            f"borrow_financing fields mismatch; missing={sorted(expected - set(config))}, "
            f"unknown={sorted(set(config) - expected)}"
        )
    borrow = float(config["short_borrow_bps_annual"])
    financing = float(config["cash_financing_bps_annual"])
    if not all(np.isfinite(value) and value >= 0.0 for value in (borrow, financing)):
        raise ValueError("borrow and financing assumptions must be finite and nonnegative")
    adjusted = equity_curve.copy()
    gross = adjusted["gross_exposure"].astype(float)
    net = adjusted["net_exposure"].astype(float)
    short_exposure = ((gross - net) / 2.0).clip(lower=0.0)
    financed_cash = (gross - 1.0).clip(lower=0.0)
    daily_drag = (short_exposure * borrow + financed_cash * financing) / 10_000.0 / 252.0
    adjusted["return"] = adjusted["return"].astype(float) - daily_drag
    initial = float(equity_curve["equity"].iloc[0]) / (1.0 + float(equity_curve["return"].iloc[0]))
    adjusted["equity"] = initial * (1.0 + adjusted["return"]).cumprod()
    return {
        "scenario": "stressed_borrow_and_financing_proxy",
        "short_borrow_bps_annual": borrow,
        "cash_financing_bps_annual": financing,
        "method": "post-ledger exposure-based sensitivity; not a locate or borrow-availability model",
        **performance_summary(adjusted),
    }


def _missing_price_halts(
    *,
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    benchmark_symbol: str,
    backtest_config: dict[str, Any],
) -> bool:
    corrupted = panel.copy()
    corrupted["open"] = np.nan
    try:
        _backtest(
            panel=corrupted,
            predictions=predictions,
            features=features,
            benchmark_symbol=benchmark_symbol,
            backtest_config=backtest_config,
        )
    except (ValueError, RuntimeError):
        return True
    return False


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical_json(value) + b"\n")


def _write_markdown_dossier(path: Path, dossier: dict[str, Any]) -> None:
    metrics = dossier["metrics"]
    lines = [
        "# Signal Foundry Final-Holdout Dossier",
        "",
        f"Decision: **{dossier['decision']}**",
        "",
        dossier["scope"],
        "",
        "## Gate results",
        "",
        *[
            f"- {name}: {'PASS' if passed else 'FAIL'}"
            for name, passed in sorted(dossier["gates"].items())
        ],
        "",
        "## Selected metrics",
        "",
        *[
            f"- {name}: {value}"
            for name, value in sorted(metrics.items())
            if isinstance(value, (bool, int, float))
        ],
        "",
        "## Interpretation",
        "",
        "A backtest is historical research evidence, not money and not a promise of future profit. "
        "READY_FOR_PAPER permits only a time-bounded, zero-capital shadow evaluation under the "
        "documented controls. NOT_READY is the mandatory outcome whenever any gate fails.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_governed_signal_foundry_research(
    *,
    dataset: SignalFoundryDataset,
    model_specs: list[dict[str, Any]],
    feature_config: dict[str, Any],
    walk_forward_config: dict[str, Any],
    backtest_config: dict[str, Any],
    research_config: GovernedResearchConfig,
    readiness_thresholds: ReadinessThresholds,
    output_root: str | Path = "runs/signal-foundry",
    code_sha: str | None = None,
) -> GovernedResearchResult:
    """Execute one immutable, governed development/final-holdout evaluation."""
    _validate_model_specs(model_specs)
    if research_config.benchmark_symbol not in set(dataset.panel["symbol"]):
        raise ValueError("pre-registered benchmark is absent from the verified bundle")
    if (
        not dataset.manifest["license"]["bundle_must_remain_local"]
        and not dataset.manifest["license"]["observations_redistributable"]
    ):
        raise ValueError("bundle license policy is inconsistent")

    resolved_sha = code_sha or _git_sha()
    identity = {
        "workflow_version": "1.0.0",
        "bundle_id": dataset.bundle_id,
        "code_sha": resolved_sha,
        "dependencies": _dependency_versions(),
        "research": asdict(research_config),
        "readiness": asdict(readiness_thresholds),
        "models": model_specs,
        "features": feature_config,
        "walk_forward": walk_forward_config,
        "backtest": backtest_config,
    }
    run_id = _sha256_bytes(_canonical_json(identity))
    root = Path(output_root)
    destination = root / run_id
    if destination.exists():
        raise FileExistsError(
            f"final holdout run {run_id} already exists; immutable runs cannot be repeated"
        )
    staging = root / f".publishing-{run_id}"
    if staging.exists():
        raise FileExistsError(f"stale governed-run staging exists: {staging}")
    staging.mkdir(parents=True)

    try:
        panel = dataset.panel
        decision_panel = dataset.decision_panel
        if decision_panel is None:
            if not dataset.source_panel.empty:
                raise ValueError("verified source data lacks a decision-eligible panel")
            decision_panel = panel
        features = build_features(
            decision_panel,
            benchmark_symbol=research_config.benchmark_symbol,
            config=feature_config,
        )
        labels = build_labels(
            panel,
            benchmark_symbol=research_config.benchmark_symbol,
            horizons=list(research_config.horizons),
        )
        max_horizon = max(research_config.horizons)
        holdout_start = pd.Timestamp(research_config.holdout_start)
        development_end = _development_cutoff(
            features["date"],
            holdout_start,
            embargo_sessions=max_horizon,
        )
        development_features = features.loc[features["date"].le(development_end)].copy()
        development_labels = labels.loc[labels["date"].le(development_end)].copy()
        development = run_walk_forward(
            features=development_features,
            labels=development_labels,
            model_specs=model_specs,
            target=research_config.target,
            config=walk_forward_config,
            max_horizon=max_horizon,
        )
        candidate_name, development_summary = _select_candidate(
            development.metrics,
            selection_metric=research_config.selection_metric,
        )
        selected_spec = next(spec for spec in model_specs if spec["name"] == candidate_name)
        ledger = _trial_ledger(model_specs, development_summary)

        supervised, columns = supervised_frame(features, labels, research_config.target)
        train = supervised.loc[supervised["date"].le(development_end)].dropna(
            subset=[research_config.target]
        )
        holdout = supervised.loc[supervised["date"].ge(holdout_start)].dropna(
            subset=[research_config.target]
        )
        if train.empty or holdout.empty:
            raise ValueError("pre-registered final holdout has no eligible train/test rows")
        model = create_model(candidate_name, **selected_spec.get("params", {}))
        model.fit(
            _model_matrix(train, columns, model),
            train[research_config.target].astype(float),
        )
        predictions = holdout[["date", "symbol", research_config.target]].rename(
            columns={research_config.target: "target"}
        )
        predictions["prediction"] = model.predict(_model_matrix(holdout, columns, model))
        predictions["model"] = candidate_name
        predictions["window_id"] = "final_holdout"

        primary = _backtest(
            panel=panel,
            predictions=predictions,
            features=features,
            benchmark_symbol=research_config.benchmark_symbol,
            backtest_config=backtest_config,
        )
        pbo = _pbo(development.predictions, research_config.seed)
        scenario_summaries, scenarios_passed = _stress_scenarios(
            panel=panel,
            predictions=predictions,
            features=features,
            benchmark_symbol=research_config.benchmark_symbol,
            backtest_config=backtest_config,
            maximum_drawdown=readiness_thresholds.maximum_drawdown,
            seed=research_config.seed,
        )
        borrow_sensitivity = _borrow_financing_sensitivity(
            primary.equity_curve,
            dict(backtest_config.get("borrow_financing", {})),
        )
        scenario_summaries.append(borrow_sensitivity)
        missing_price_halt = _missing_price_halts(
            panel=panel,
            predictions=predictions,
            features=features,
            benchmark_symbol=research_config.benchmark_symbol,
            backtest_config=backtest_config,
        )
        capacity_settings = dict(backtest_config.get("capacity", {}))
        expected_capacity = {"aum_multiples", "max_participation_rate", "minimum_fill_ratio"}
        if set(capacity_settings) != expected_capacity:
            raise ValueError(
                "capacity fields mismatch; "
                f"missing={sorted(expected_capacity - set(capacity_settings))}, "
                f"unknown={sorted(set(capacity_settings) - expected_capacity)}"
            )
        aum_values = tuple(
            float(backtest_config.get("initial_capital", 1_000_000.0)) * float(multiple)
            for multiple in capacity_settings["aum_multiples"]
        )
        capacity = estimate_capacity(
            primary.fills,
            CapacityConfig(
                reference_aum=float(backtest_config.get("initial_capital", 1_000_000.0)),
                aum_values=aum_values,
                max_participation_rate=float(capacity_settings["max_participation_rate"]),
                columns=CapacityColumns.for_fill_records(),
            ),
        )
        capacity_passed = bool(
            capacity.curve["fill_ratio"].min() >= float(capacity_settings["minimum_fill_ratio"])
        )
        primary_summary = performance_summary(primary.equity_curve)
        placebo_summary = next(
            item for item in scenario_summaries if item["scenario"] == "permuted_signal_placebo"
        )
        placebo_passed = bool(
            np.isfinite(primary_summary["annual_return"])
            and np.isfinite(placebo_summary["annual_return"])
            and primary_summary["annual_return"] >= placebo_summary["annual_return"]
        )
        dossier = assess_paper_readiness(
            equity_curve=primary.equity_curve,
            benchmark_returns=primary.equity_curve["benchmark_return"],
            n_trials=len(model_specs),
            probability_of_backtest_overfitting=float(pbo.get("pbo", np.nan)),
            point_in_time_limits=dataset.manifest["point_in_time_limits"],
            accounting_reconciled=True,
            stress_scenarios_passed=scenarios_passed and placebo_passed,
            thresholds=readiness_thresholds,
            additional_gates={
                "capacity_liquidity": capacity_passed,
                "missing_price_halt": missing_price_halt,
            },
        )
        dossier["candidate_model"] = candidate_name
        dossier["bundle_id"] = dataset.bundle_id
        dossier["run_id"] = run_id
        dossier["development_end"] = str(development_end.date())
        dossier["holdout_start"] = research_config.holdout_start
        dossier["overfitting"] = pbo
        dossier["scenarios"] = scenario_summaries
        dossier["placebo_outperformed"] = placebo_passed
        dossier["uncertainty"] = _bootstrap_uncertainty(
            primary.equity_curve["return"],
            seed=research_config.seed,
        )
        dossier["drawdown_diagnostics"] = _drawdown_diagnostics(primary.equity_curve)
        dossier["year_stability"] = _year_stability(primary.equity_curve)
        regime = (
            features[["date", "high_vol_regime"]]
            .drop_duplicates("date")
            .set_index("date")["high_vol_regime"]
        )
        dossier["regime_stability"] = regime_performance(
            primary.equity_curve,
            regime,
            regime_name="high_vol_regime",
        ).to_dict(orient="records")
        dossier["limitations"] = [
            "Daily bars do not establish intraday queue position or institutional execution quality.",
            "Borrow availability and locate failures are not modeled; borrow and financing are "
            "post-ledger sensitivity deductions.",
            "Point-in-time completeness is limited to the producer's explicit declarations.",
            "A readiness decision is not evidence that an edge will persist or be profitable.",
        ]

        development_summary.to_csv(staging / "development_model_selection.csv", index=False)
        development.metrics.to_csv(staging / "development_windows.csv", index=False)
        predictions.to_csv(staging / "final_holdout_predictions.csv", index=False)
        primary.equity_curve.to_csv(staging / "final_holdout_equity.csv", index=False)
        primary.orders.to_csv(staging / "orders.csv", index=False)
        primary.fills.to_csv(staging / "fills.csv", index=False)
        primary.pnl_attribution.to_csv(staging / "pnl_attribution.csv", index=False)
        capacity.curve.to_csv(staging / "capacity_curve.csv", index=False)
        capacity.scenario_trades.to_csv(staging / "capacity_scenario_trades.csv", index=False)
        _write_json(staging / "capacity_diagnostics.json", asdict(capacity.diagnostics))
        (staging / "trial_ledger.jsonl").write_text(
            "".join(_canonical_json(record).decode("utf-8") + "\n" for record in ledger),
            encoding="utf-8",
        )
        _write_json(staging / "dossier.json", dossier)
        _write_markdown_dossier(staging / "dossier.md", dossier)

        artifacts = [
            {
                "path": path.name,
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(staging.iterdir())
            if path.is_file()
        ]
        _write_json(
            staging / "run_manifest.json",
            {
                **identity,
                "run_id": run_id,
                "candidate_model": candidate_name,
                "development_end": str(development_end.date()),
                "holdout_start": research_config.holdout_start,
                "trial_ledger_head": ledger[-1]["record_hash"],
                "license": dataset.manifest["license"],
                "point_in_time_limits": dataset.manifest["point_in_time_limits"],
                "artifacts": artifacts,
            },
        )
        staging.replace(destination)
        return GovernedResearchResult(
            run_id=run_id,
            run_dir=destination,
            candidate_model=candidate_name,
            dossier=dossier,
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
