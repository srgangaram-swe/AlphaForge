"""Paper-trading replay utilities.

This module never routes real orders. It converts target weights into
simulated orders and fills using historical prices.

Fills are depth-aware: each order walks a synthetic limit order book (the C++
engine when built, the parity-tested Python reference otherwise), so larger
orders pay more slippage instead of a flat bps assumption. Book depth per
level is sized from the symbol's traded volume that day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.data.schemas import to_wide
from alphaforge.execution import BUY, SELL, simulate_fill

PRICE_TICKS_PER_DOLLAR = 100  # $0.01 tick


def _depth_aware_fill(
    side: int,
    shares: float,
    price: float,
    day_volume: float,
    half_spread_bps: float = 2.5,
    depth_fraction_per_level: float = 0.005,
    n_levels: int = 10,
) -> tuple[float, float, float]:
    """Fill ``shares`` against a synthetic book around ``price``.

    Returns (fill_price, filled_shares, slippage_bps vs mid).
    """
    mid_ticks = max(1, round(price * PRICE_TICKS_PER_DOLLAR))
    half_spread_ticks = max(1, round(mid_ticks * half_spread_bps / 1e4))
    qty_per_level = max(1, int(day_volume * depth_fraction_per_level))
    avg_ticks, filled = simulate_fill(
        side,
        int(max(1, round(shares))),
        mid=mid_ticks,
        half_spread=half_spread_ticks,
        tick=1,
        n_levels=n_levels,
        qty_per_level=qty_per_level,
    )
    if filled == 0:
        return (price, 0.0, 0.0)
    fill_price = avg_ticks / PRICE_TICKS_PER_DOLLAR
    sign = 1.0 if side == BUY else -1.0
    slippage_bps = sign * (fill_price - price) / price * 1e4
    return (fill_price, float(filled), float(slippage_bps))


def simulate_paper_trading(
    panel: pd.DataFrame,
    target_weights: pd.DataFrame,
    capital: float = 1_000_000.0,
    lookback_days: int = 20,
    half_spread_bps: float = 2.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay the latest target-weight changes as simulated, depth-aware fills."""
    close = to_wide(panel, "close")
    volume = to_wide(panel, "volume")
    weights = target_weights.pivot(
        index="date", columns="symbol", values="target_weight"
    ).sort_index()
    weights = weights.reindex(close.index).ffill().fillna(0.0).tail(lookback_days)
    prices = close.reindex(weights.index).ffill()
    volumes = volume.reindex(weights.index).ffill()
    deltas = weights.diff().fillna(weights)

    orders = []
    for date, row in deltas.iterrows():
        for symbol, delta_weight in row.items():
            if delta_weight == 0 or symbol not in prices.columns:
                continue
            price = float(prices.loc[date, symbol])
            if not np.isfinite(price) or price <= 0:
                continue
            notional = float(delta_weight * capital)
            side = BUY if notional > 0 else SELL
            shares = abs(notional) / price
            day_volume = float(volumes.loc[date, symbol]) if symbol in volumes.columns else 1e6
            fill_price, filled_shares, slippage_bps = _depth_aware_fill(
                side, shares, price, day_volume, half_spread_bps=half_spread_bps
            )
            orders.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "side": "BUY" if side == BUY else "SELL",
                    "target_weight_change": float(delta_weight),
                    "simulated_notional": abs(notional),
                    "reference_price": price,
                    "simulated_fill_price": fill_price,
                    "simulated_shares": filled_shares,
                    "slippage_bps": slippage_bps,
                    "status": (
                        "SIMULATED_FILL" if filled_shares >= shares - 0.5 else "SIMULATED_PARTIAL"
                    ),
                }
            )
    latest = weights.tail(1).T.rename(columns={weights.index[-1]: "target_weight"}).reset_index()
    latest = latest.rename(columns={"index": "symbol"})
    latest["simulated_notional"] = latest["target_weight"] * capital
    return pd.DataFrame(orders), latest
