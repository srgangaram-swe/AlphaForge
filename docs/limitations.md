# Limitations

AlphaForge is an educational research platform, not a production trading system.

Important limitations:

- yfinance data can contain survivorship, adjustment, and availability issues.
- Synthetic data verifies engineering behavior but is not market evidence; its
  embedded edge is deliberately faint but still far cleaner than real markets,
  so demo Sharpe ratios overstate what identical code would earn live.
- Backtests are close-to-close approximations; weights do not drift between
  rebalances, and costs are linear in turnover.
- The C++ order book simulates fills against a *synthetic* book shape; it is
  an execution-modeling and systems-engineering module, not a market
  microstructure calibration.
- The drawdown-deleverage control ignores second-order feedback of the control
  on its own drawdown (documented in docs/backtesting.md).
- DSR/PBO correct for the trials the platform knows about; they cannot correct
  for ideas discarded before they were coded.
- No broker integration places live orders.
- Model results can overfit even with walk-forward validation, purging, and
  deflated statistics.
- Real deployment would require stronger data licensing, monitoring, capital
  controls, compliance review, and independent validation.
