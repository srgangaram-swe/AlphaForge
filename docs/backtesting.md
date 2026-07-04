# Backtesting Methodology

Backtests use saved out-of-sample predictions only. Signals are converted to
target weights, then shifted by `execution_lag` before returns are applied.
`execution_lag=1` is the minimum and prevents same-close execution.

## Costs

Costs are charged on *levered* traded notional:

`cost = turnover * (commission_bps + half_spread_bps + slippage_bps) / 10000`

Turnover is computed from the day-over-day change in levered weights, so
risk-overlay adjustments (volatility targeting, drawdown deleveraging) pay
transaction costs like any other trade. A vol-targeted strategy that "wins"
only because its exposure adjustments were free is a bug this design rules out.

## Risk overlays (causal by construction)

- **Volatility targeting**: leverage = vol_target / trailing realized vol of
  the unlevered strategy, lagged one day, capped at `max_leverage`.
- **Drawdown deleveraging**: exposure is cut (default 50%) when the previous
  day's trailing drawdown of the unlevered strategy breaches the threshold.
  Approximation: the control is sized from the unlevered path and ignores the
  second-order feedback of the control on its own drawdown.

## Reporting conventions

- Metrics are computed from the first day with non-zero exposure
  (`trim_inactive=True`); the dead ramp-up period before the first
  walk-forward test window would otherwise dilute every statistic.
- Sharpe/Sortino use arithmetic daily means annualized by sqrt(252);
  annual return is geometric.
- Every backtest summary includes the Probabilistic and Deflated Sharpe
  Ratios with `n_trials` = number of model variants that competed in the run,
  and walk-forward runs report the Probability of Backtest Overfitting across
  models (see docs/modeling.md).

The engine reports gross returns, transaction costs, net returns, equity,
turnover, exposures, leverage, an active flag, benchmark returns, executed
weights, and trades.

## Known simplifications

Close-to-close bars; weights are stepwise-constant between rebalances (no
intra-period drift); costs are linear in turnover. The C++ execution core
(docs/execution_engine.md) provides depth-aware fills for the paper simulator,
but the vectorized daily backtest intentionally stays simple and auditable.
