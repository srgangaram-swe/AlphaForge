"""Vectorized close-to-close backtest with lagged execution and costs.

Execution model (deliberately conservative):
- ``execution_lag=1``: weights decided from date-t information earn date t+1
  close-to-close returns — no same-close fills.
- Transaction costs apply to *levered* traded notional, so volatility-target
  or drawdown-control adjustments generate turnover costs like any trade.
- Drawdown deleveraging uses the trailing drawdown of the unlevered strategy
  shifted by one day (causal, ignores the second-order feedback of the
  control on its own drawdown — documented in docs/backtesting.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaforge.data.schemas import to_wide
from alphaforge.execution import CostModel
from alphaforge.utils import ANNUALIZATION_DAYS


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame


def _weights_wide(target_weights: pd.DataFrame, dates: pd.Index) -> pd.DataFrame:
    wide = target_weights.pivot(index="date", columns="symbol", values="target_weight").sort_index()
    wide.index = pd.to_datetime(wide.index)
    return wide.reindex(dates)


def _apply_rebalance_frequency(weights: pd.DataFrame, frequency: int) -> pd.DataFrame:
    if frequency <= 1:
        return weights.ffill().fillna(0.0)
    scheduled = weights.copy()
    valid_dates = scheduled.dropna(how="all").index
    keep = set(valid_dates[::frequency])
    scheduled.loc[[d for d in valid_dates if d not in keep], :] = np.nan
    return scheduled.ffill().fillna(0.0)


def _causal_vol_leverage(
    gross_returns: pd.Series,
    vol_target: float | None,
    lookback: int,
    max_leverage: float,
) -> pd.Series:
    if vol_target is None:
        return pd.Series(1.0, index=gross_returns.index)
    realized = gross_returns.rolling(lookback).std().shift(1) * np.sqrt(ANNUALIZATION_DAYS)
    leverage = (vol_target / realized.replace(0, np.nan)).clip(upper=max_leverage)
    return leverage.replace([np.inf, -np.inf], np.nan).fillna(1.0)


def _causal_drawdown_scale(
    gross_returns: pd.Series, threshold: float | None, cut: float = 0.5
) -> pd.Series:
    """Exposure multiplier from the previous day's trailing drawdown."""
    if threshold is None:
        return pd.Series(1.0, index=gross_returns.index)
    wealth = (1.0 + gross_returns.fillna(0.0)).cumprod()
    dd = (wealth / wealth.cummax() - 1.0).shift(1)
    return pd.Series(np.where(dd < -abs(threshold), cut, 1.0), index=gross_returns.index)


def run_backtest(
    panel: pd.DataFrame,
    target_weights: pd.DataFrame,
    benchmark_symbol: str | None = None,
    initial_capital: float = 1_000_000.0,
    execution_lag: int = 1,
    rebalance_frequency: int = 1,
    costs: CostModel | dict | None = None,
    risk: dict | None = None,
) -> BacktestResult:
    """Run an OOS-only backtest from target weights."""
    if execution_lag < 1:
        raise ValueError("execution_lag must be >= 1 for next-bar execution")
    cost_model = costs if isinstance(costs, CostModel) else CostModel.from_config(costs)
    risk_cfg = risk or {}

    close = to_wide(panel, "close")
    close.index = pd.to_datetime(close.index)
    returns = close.pct_change().fillna(0.0)

    tradable_symbols = sorted(set(target_weights["symbol"]) & set(returns.columns))
    if not tradable_symbols:
        raise ValueError("target weights do not overlap panel symbols")

    raw_targets = _weights_wide(target_weights, returns.index)[tradable_symbols]
    scheduled = _apply_rebalance_frequency(raw_targets, int(rebalance_frequency))
    held_weights = scheduled.shift(execution_lag).fillna(0.0)
    asset_returns = returns[tradable_symbols]

    # risk overlays are sized from the *unlevered* strategy, then applied as a
    # causal exposure multiplier; the resulting weight changes are costed
    unlevered_returns = (held_weights * asset_returns).sum(axis=1)
    leverage = _causal_vol_leverage(
        unlevered_returns,
        risk_cfg.get("vol_target"),
        int(risk_cfg.get("vol_lookback", 20)),
        float(risk_cfg.get("max_leverage", 1.5)),
    ) * _causal_drawdown_scale(unlevered_returns, risk_cfg.get("drawdown_deleverage"))

    levered_weights = held_weights.mul(leverage, axis=0)
    gross_returns = (levered_weights * asset_returns).sum(axis=1)

    delta = levered_weights.diff()
    delta.iloc[0] = levered_weights.iloc[0]
    turnover = delta.abs().sum(axis=1)
    trading_cost = cost_model.costs(turnover)
    net_returns = gross_returns - trading_cost
    equity = initial_capital * (1.0 + net_returns).cumprod()

    benchmark_return = (
        returns[benchmark_symbol]
        if benchmark_symbol is not None and benchmark_symbol in returns
        else 0.0
    )
    gross_exposure = levered_weights.abs().sum(axis=1)
    curve = pd.DataFrame(
        {
            "date": returns.index,
            "gross_return": gross_returns.to_numpy(),
            "transaction_cost": trading_cost.to_numpy(),
            "return": net_returns.to_numpy(),
            "equity": equity.to_numpy(),
            "benchmark_return": np.asarray(benchmark_return),
            "turnover": turnover.to_numpy(),
            "gross_exposure": gross_exposure.to_numpy(),
            "net_exposure": levered_weights.sum(axis=1).to_numpy(),
            "leverage": leverage.to_numpy(),
            "active": (gross_exposure > 0).to_numpy(),
        }
    )

    weights_long = (
        levered_weights.reset_index()
        .melt(id_vars="date", var_name="symbol", value_name="weight")
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
    trades = (
        delta.reset_index()
        .melt(id_vars="date", var_name="symbol", value_name="trade_weight")
        .query("trade_weight != 0")
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
    return BacktestResult(curve, weights_long, trades)
