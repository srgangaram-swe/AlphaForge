"""Paper-trading replay utilities.

This module never routes real orders. It converts target weights into
simulated orders and fills using historical prices.
"""

from __future__ import annotations

import pandas as pd

from alphaforge.data.schemas import to_wide


def simulate_paper_trading(
    panel: pd.DataFrame,
    target_weights: pd.DataFrame,
    capital: float = 1_000_000.0,
    lookback_days: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay the latest target-weight changes as simulated fills."""
    close = to_wide(panel, "close")
    weights = target_weights.pivot(
        index="date", columns="symbol", values="target_weight"
    ).sort_index()
    weights = weights.reindex(close.index).ffill().fillna(0.0).tail(lookback_days)
    prices = close.reindex(weights.index).ffill()
    deltas = weights.diff().fillna(weights)

    orders = []
    for date, row in deltas.iterrows():
        for symbol, delta_weight in row.items():
            if delta_weight == 0 or symbol not in prices.columns:
                continue
            price = float(prices.loc[date, symbol])
            notional = float(delta_weight * capital)
            orders.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "side": "BUY" if notional > 0 else "SELL",
                    "target_weight_change": float(delta_weight),
                    "simulated_notional": abs(notional),
                    "simulated_price": price,
                    "simulated_shares": abs(notional) / price if price > 0 else 0.0,
                    "status": "SIMULATED_FILL",
                }
            )
    latest = weights.tail(1).T.rename(columns={weights.index[-1]: "target_weight"}).reset_index()
    latest = latest.rename(columns={"index": "symbol"})
    latest["simulated_notional"] = latest["target_weight"] * capital
    return pd.DataFrame(orders), latest
