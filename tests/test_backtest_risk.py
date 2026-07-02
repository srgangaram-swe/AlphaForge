from __future__ import annotations

import pandas as pd

from alphaforge.backtesting import run_backtest
from alphaforge.portfolio import construct_portfolio
from alphaforge.risk import performance_summary


def test_transaction_costs_reduce_equity(small_panel):
    dates = sorted(small_panel["date"].unique())[20:120]
    symbols = ["SYN000", "SYN001", "SYN002", "SYN003"]
    rows = []
    for i, date in enumerate(dates):
        for symbol in symbols:
            direction = 1.0 if (i + symbols.index(symbol)) % 2 == 0 else -1.0
            rows.append({"date": date, "symbol": symbol, "signal": direction})
    signals = pd.DataFrame(rows)
    weights = construct_portfolio(
        signals,
        config={"max_weight": 0.25, "max_gross_exposure": 1.0, "inverse_vol_scaling": False},
    )

    free = run_backtest(
        small_panel,
        weights,
        benchmark_symbol="BENCH",
        costs={"commission_bps": 0, "half_spread_bps": 0, "slippage_bps": 0},
    )
    costly = run_backtest(
        small_panel,
        weights,
        benchmark_symbol="BENCH",
        costs={"commission_bps": 20, "half_spread_bps": 20, "slippage_bps": 20},
    )
    assert costly.equity_curve["equity"].iloc[-1] < free.equity_curve["equity"].iloc[-1]
    summary = performance_summary(costly.equity_curve)
    assert "max_drawdown" in summary
    assert summary["transaction_cost_impact"] > 0
