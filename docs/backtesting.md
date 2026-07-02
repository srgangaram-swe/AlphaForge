# Backtesting Methodology

Backtests use saved out-of-sample predictions only. Signals are converted to target weights, then shifted by `execution_lag` before returns are applied. `execution_lag=1` is the minimum and prevents same-close execution.

Costs are charged on traded notional:

`cost = turnover * (commission_bps + half_spread_bps + slippage_bps) / 10000`

The engine reports gross returns, transaction costs, net returns, equity, turnover, gross exposure, net exposure, benchmark returns, and executed weights.
