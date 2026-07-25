"""Typed backtest orchestration service (SF-S2-MR10a).

``run_backtest_service(BacktestRequest) -> BacktestResult`` composes the existing
AlphaForge stack — data, features, labels, leakage-safe walk-forward,
signals, portfolio construction, event-driven backtest, and performance
metrics — behind one narrow, JSON-serializable contract. The chosen model and
the requested naive baselines are run on **identical** data, splits, and costs,
so the comparison is apples-to-apples and included automatically.

Data can come from the deterministic synthetic market or from a **Signal Foundry
data bundle produced by Signalattice**, which is what makes the two repositories
complement one another: Signalattice supplies point-in-time data; AlphaForge
turns it into evidence.

Results are simulated research, not live inference or executable orders. The
service uses only stdlib + numpy/pandas types (no app-extra dependency) so it
stays importable in the core install; the API/dashboard layers wrap it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alphaforge.backtesting import run_backtest
from alphaforge.data.signal_foundry import load_signal_foundry_dataset
from alphaforge.data.synthetic import SyntheticMarketConfig, generate_synthetic_market
from alphaforge.features import build_features
from alphaforge.labels.labels import build_labels
from alphaforge.models.registry import available_models
from alphaforge.portfolio import construct_portfolio
from alphaforge.risk import performance_summary
from alphaforge.signals import build_signals, select_model_predictions
from alphaforge.training import WalkForwardConfig, run_walk_forward

SERVICE_VERSION = "1.0.0"

DISCLAIMER = "Educational research output. Simulated results only. Not financial advice."

#: Registry names that are naive baselines (not deployable strategies).
_BASELINE_NAMES = frozenset(
    {
        "zero_baseline",
        "historical_mean",
        "lag_baseline",
        "moving_average_baseline",
        "momentum_baseline",
        "equal_probability",
        "buy_and_hold",
        "equal_weight",
    }
)

_SIGNAL_STRATEGIES = frozenset({"long_short", "long_only_topk", "rank_weighted", "confidence"})

#: Curated metrics surfaced in the comparison table (subset of performance_summary).
_COMPARISON_METRICS = (
    "total_return",
    "annual_return",
    "annual_volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "hit_rate",
    "average_turnover",
)


class BacktestServiceError(ValueError):
    """Raised when a backtest request is invalid or cannot be fulfilled."""


def available_strategy_models() -> list[str]:
    """Registry names suitable as the headline strategy model (non-classifier)."""
    return [name for name in available_models() if name != "equal_probability"]


def available_baselines() -> list[str]:
    """Naive-baseline registry names available for comparison."""
    return sorted(name for name in available_models() if name in _BASELINE_NAMES)


def discover_bundles(bundles_dir: str | Path = "data/signal-foundry-bundles") -> list[str]:
    """Return Signal Foundry bundle directories under ``bundles_dir`` (by name)."""
    root = Path(bundles_dir)
    if not root.is_dir():
        return []
    return sorted(child.name for child in root.iterdir() if (child / "manifest.json").is_file())


@dataclass
class BacktestRequest:
    """A validated, JSON-serializable backtest specification."""

    data_source: str = "synthetic"
    bundle_dir: str | None = None
    n_symbols: int = 8
    n_days: int = 600
    benchmark_symbol: str = "BENCH"
    model: str = "random_forest"
    model_params: Mapping[str, Any] = field(default_factory=dict)
    baselines: Sequence[str] = ("zero_baseline", "historical_mean", "momentum_baseline")
    horizon: int = 1
    strategy: str = "long_short"
    cost_bps: float = 1.0
    seed: int = 42
    min_train_days: int = 252
    test_days: int = 63
    step_days: int = 63
    embargo_days: int = 10

    def __post_init__(self) -> None:
        if self.data_source not in {"synthetic", "signal_foundry"}:
            raise BacktestServiceError(f"unknown data_source {self.data_source!r}")
        if self.data_source == "signal_foundry" and not self.bundle_dir:
            raise BacktestServiceError("signal_foundry data_source requires bundle_dir")
        known = set(available_models())
        if self.model not in known:
            raise BacktestServiceError(f"unknown model {self.model!r}")
        unknown_baselines = [b for b in self.baselines if b not in known]
        if unknown_baselines:
            raise BacktestServiceError(f"unknown baselines {unknown_baselines}")
        if self.strategy not in _SIGNAL_STRATEGIES:
            raise BacktestServiceError(f"unknown strategy {self.strategy!r}")
        if self.horizon < 1:
            raise BacktestServiceError("horizon must be >= 1")
        if self.cost_bps < 0:
            raise BacktestServiceError("cost_bps must be >= 0")
        if self.seed < 0:
            raise BacktestServiceError("seed must be >= 0")
        if self.data_source == "synthetic" and self.n_symbols < 2:
            raise BacktestServiceError("need at least 2 symbols")
        for name, value in (
            ("min_train_days", self.min_train_days),
            ("test_days", self.test_days),
            ("step_days", self.step_days),
        ):
            if value < 1:
                raise BacktestServiceError(f"{name} must be >= 1")
        if self.embargo_days < 0:
            raise BacktestServiceError("embargo_days must be >= 0")
        # Deduplicate baselines, preserving order and dropping the headline model.
        object.__setattr__(
            self,
            "baselines",
            tuple(dict.fromkeys(b for b in self.baselines if b != self.model)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_source": self.data_source,
            "bundle_dir": self.bundle_dir,
            "n_symbols": self.n_symbols,
            "n_days": self.n_days,
            "benchmark_symbol": self.benchmark_symbol,
            "model": self.model,
            "model_params": dict(self.model_params),
            "baselines": list(self.baselines),
            "horizon": self.horizon,
            "strategy": self.strategy,
            "cost_bps": self.cost_bps,
            "seed": self.seed,
            "min_train_days": self.min_train_days,
            "test_days": self.test_days,
            "step_days": self.step_days,
            "embargo_days": self.embargo_days,
        }

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class StrategyOutcome:
    """One strategy's (model or baseline) backtest evidence."""

    name: str
    is_headline: bool
    is_baseline: bool
    metrics: dict[str, float | None]
    equity_curve: list[dict[str, Any]]  # date, strategy_cum, benchmark_cum, drawdown

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "is_headline": self.is_headline,
            "is_baseline": self.is_baseline,
            "metrics": self.metrics,
            "equity_curve": self.equity_curve,
        }


@dataclass(frozen=True)
class BacktestResult:
    """The complete, JSON-serializable result of one backtest run."""

    request: dict[str, Any]
    data_id: str
    benchmark_symbol: str
    n_observations: int
    n_windows: int
    headline: StrategyOutcome
    comparison: list[dict[str, Any]]
    trades_tail: list[dict[str, Any]]
    reproducibility: dict[str, Any]
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "data_id": self.data_id,
            "benchmark_symbol": self.benchmark_symbol,
            "n_observations": self.n_observations,
            "n_windows": self.n_windows,
            "headline": self.headline.to_dict(),
            "comparison": self.comparison,
            "trades_tail": self.trades_tail,
            "reproducibility": self.reproducibility,
            "disclaimer": self.disclaimer,
        }


def _finite(value: Any) -> float | None:
    """Coerce to a JSON-safe float, mapping NaN/inf to None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_scalar(value: Any) -> Any:
    """Convert a pandas/numpy scalar to a JSON-safe Python value."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.integer | int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.bool_ | bool):
        return bool(value)
    return _finite(value)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records (dates → strings, NaN → None)."""
    if frame.empty:
        return []
    safe = frame.copy()
    for column in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[column]):
            safe[column] = pd.to_datetime(safe[column]).dt.strftime("%Y-%m-%d")
    return [{key: _json_scalar(val) for key, val in row.items()} for row in safe.to_dict("records")]


def _load_panel(request: BacktestRequest) -> tuple[pd.DataFrame, str, str]:
    """Return (panel, benchmark_symbol, data_id) for the request's data source."""
    if request.data_source == "synthetic":
        config = SyntheticMarketConfig(
            n_symbols=request.n_symbols, n_days=request.n_days, seed=request.seed
        )
        panel = generate_synthetic_market(config)
        data_id = f"synthetic:symbols={request.n_symbols}:days={request.n_days}:seed={request.seed}"
        return panel, config.benchmark_symbol, data_id

    dataset = load_signal_foundry_dataset(request.bundle_dir or "")
    panel = dataset.panel
    benchmark = request.benchmark_symbol
    if benchmark not in set(panel["symbol"].unique()):
        raise BacktestServiceError(
            f"benchmark {benchmark!r} not in bundle symbols; "
            f"available: {sorted(panel['symbol'].unique())[:8]}"
        )
    return panel, benchmark, f"bundle:{dataset.bundle_id}"


def _equity_points(equity_curve: pd.DataFrame) -> list[dict[str, Any]]:
    """Cumulative strategy vs benchmark return and drawdown, per date."""
    curve = equity_curve.copy()
    equity = curve["equity"].to_numpy(dtype=float)
    strat_cum = equity / equity[0] - 1.0 if len(equity) and equity[0] != 0 else equity * 0.0
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    bench = (1.0 + curve.get("benchmark_return", pd.Series(0.0, index=curve.index))).cumprod() - 1.0
    dates = pd.to_datetime(curve["date"]).dt.strftime("%Y-%m-%d")
    return [
        {
            "date": dates.iloc[i],
            "strategy_cum": _finite(strat_cum[i]),
            "benchmark_cum": _finite(bench.iloc[i]),
            "drawdown": _finite(drawdown[i]),
        }
        for i in range(len(curve))
    ]


def _summary_metrics(summary: Mapping[str, Any]) -> dict[str, float | None]:
    metrics = {key: _finite(summary.get(key)) for key in _COMPARISON_METRICS}
    if metrics.get("total_return") is None:
        # Some summaries expose cumulative return under a different key.
        metrics["total_return"] = _finite(summary.get("cumulative_return"))
    return metrics


def _run_one_strategy(
    name: str,
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    panel: pd.DataFrame,
    benchmark: str,
    request: BacktestRequest,
) -> tuple[StrategyOutcome, pd.DataFrame]:
    selected = select_model_predictions(predictions, model=name)
    signals = build_signals(selected, strategy=request.strategy)
    weights = construct_portfolio(signals, features=features)
    result = run_backtest(
        panel=panel,
        target_weights=weights,
        benchmark_symbol=benchmark,
        costs={"commission_bps": request.cost_bps},
    )
    summary = performance_summary(result.equity_curve)
    outcome = StrategyOutcome(
        name=name,
        is_headline=(name == request.model),
        is_baseline=(name in _BASELINE_NAMES),
        metrics=_summary_metrics(summary),
        equity_curve=_equity_points(result.equity_curve),
    )
    return outcome, result.trades


def run_backtest_service(request: BacktestRequest) -> BacktestResult:
    """Run a leakage-safe walk-forward backtest for the model and its baselines.

    The chosen model and every requested baseline are trained and evaluated on
    identical data, splits, and costs; the headline strategy is the chosen model.
    """
    panel, benchmark, data_id = _load_panel(request)
    features = build_features(panel, benchmark_symbol=benchmark)
    labels = build_labels(panel, benchmark_symbol=benchmark, horizons=[request.horizon])
    target = f"fwd_ret_{request.horizon}"

    model_names = [request.model, *request.baselines]
    model_specs = [
        {"name": name, "params": dict(request.model_params) if name == request.model else {}}
        for name in model_names
    ]
    wf_config = WalkForwardConfig(
        scheme="expanding",
        min_train_days=request.min_train_days,
        test_days=request.test_days,
        step_days=request.step_days,
        embargo_days=request.embargo_days,
    )
    walk_forward = run_walk_forward(
        features, labels, model_specs, target=target, config=wf_config, max_horizon=request.horizon
    )
    predictions = walk_forward.predictions
    if predictions.empty:
        raise BacktestServiceError(
            "no out-of-sample predictions were produced; increase n_days or reduce min_train_days"
        )

    outcomes: list[StrategyOutcome] = []
    headline_trades = pd.DataFrame()
    for name in model_names:
        if name not in set(predictions["model"].unique()):
            continue
        outcome, trades = _run_one_strategy(name, predictions, features, panel, benchmark, request)
        outcomes.append(outcome)
        if outcome.is_headline:
            headline_trades = trades

    headline = next((o for o in outcomes if o.is_headline), None)
    if headline is None:
        raise BacktestServiceError(f"headline model {request.model!r} produced no predictions")

    comparison = [{"name": o.name, "is_baseline": o.is_baseline, **o.metrics} for o in outcomes]
    trades_tail = _records(headline_trades.tail(50))

    return BacktestResult(
        request=request.to_dict(),
        data_id=data_id,
        benchmark_symbol=benchmark,
        n_observations=int(len(panel)),
        n_windows=int(walk_forward.windows.shape[0]) if walk_forward.windows is not None else 0,
        headline=headline,
        comparison=comparison,
        trades_tail=trades_tail,
        reproducibility={
            "service_version": SERVICE_VERSION,
            "seed": request.seed,
            "config_hash": request.config_hash(),
            "data_id": data_id,
        },
    )
