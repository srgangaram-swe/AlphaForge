"""Risk and performance analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.utils import ANNUALIZATION_DAYS


def drawdown_series(equity: pd.Series) -> pd.Series:
    wealth = equity.astype(float)
    peak = wealth.cummax()
    return wealth / peak - 1.0


def _annual_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    total = float((1.0 + returns).prod() - 1.0)
    years = len(returns) / ANNUALIZATION_DAYS
    return (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else np.nan


def performance_summary(equity_curve: pd.DataFrame) -> dict[str, float]:
    """Summarize backtest performance and risk."""
    r = equity_curve["return"].astype(float).fillna(0.0)
    equity = equity_curve["equity"].astype(float)
    dd = drawdown_series(equity)
    ann_vol = float(r.std() * np.sqrt(ANNUALIZATION_DAYS))
    downside = r[r < 0].std() * np.sqrt(ANNUALIZATION_DAYS)
    ann_ret = _annual_return(r)
    bench = equity_curve.get("benchmark_return", pd.Series(0.0, index=equity_curve.index)).astype(
        float
    )
    beta = np.nan
    alpha = np.nan
    if bench.std() > 0 and len(bench) == len(r):
        beta = float(np.cov(r, bench)[0, 1] / np.var(bench))
        alpha = float((r.mean() - beta * bench.mean()) * ANNUALIZATION_DAYS)

    var_95 = float(r.quantile(0.05))
    tail = r[r <= var_95]
    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else 0.0,
        "annual_return": float(ann_ret),
        "annual_volatility": ann_vol,
        "sharpe": np.nan if ann_vol == 0 else float(ann_ret / ann_vol),
        "sortino": np.nan if downside == 0 or np.isnan(downside) else float(ann_ret / downside),
        "max_drawdown": float(dd.min()),
        "calmar": np.nan if dd.min() == 0 else float(ann_ret / abs(dd.min())),
        "hit_rate": float((r > 0).mean()),
        "average_turnover": float(equity_curve.get("turnover", pd.Series(0.0)).mean()),
        "transaction_cost_impact": float(
            equity_curve.get("transaction_cost", pd.Series(0.0)).sum()
        ),
        "average_gross_exposure": float(equity_curve.get("gross_exposure", pd.Series(0.0)).mean()),
        "average_net_exposure": float(equity_curve.get("net_exposure", pd.Series(0.0)).mean()),
        "beta_to_benchmark": beta,
        "alpha_estimate": alpha,
        "var_95_daily": var_95,
        "expected_shortfall_95_daily": float(tail.mean()) if not tail.empty else np.nan,
    }


def monthly_returns(equity_curve: pd.DataFrame) -> pd.DataFrame:
    frame = equity_curve.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    monthly = (1.0 + frame.set_index("date")["return"]).resample("ME").prod() - 1.0
    return monthly.rename("return").reset_index()


def stress_test_summary(
    weights: pd.DataFrame,
    scenarios: list[dict] | None = None,
) -> pd.DataFrame:
    """Approximate instantaneous PnL from simple market shock scenarios."""
    scenarios = scenarios or [{"name": "market -10%", "market_shock": -0.10}]
    latest = weights.sort_values("date").groupby("symbol").tail(1)
    gross = float(latest["weight"].abs().sum()) if "weight" in latest else 0.0
    net = float(latest["weight"].sum()) if "weight" in latest else 0.0
    rows = []
    for scenario in scenarios:
        shock = float(scenario.get("market_shock", 0.0))
        rows.append(
            {
                "scenario": scenario.get("name", "scenario"),
                "market_shock": shock,
                "estimated_portfolio_return": net * shock,
                "gross_exposure": gross,
                "net_exposure": net,
            }
        )
    return pd.DataFrame(rows)
