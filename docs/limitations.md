# Limitations

AlphaForge is an educational research platform, not a production trading system.

Important limitations:

- yfinance data can contain survivorship, adjustment, and availability issues.
- Synthetic data verifies engineering behavior but is not market evidence; its
  embedded edge is deliberately faint but still far cleaner than real markets,
  so demo Sharpe ratios overstate what identical code would earn live.
- Backtests are daily-bar approximations. The close-decision/next-open ledger
  prevents pre-fill gap capture and lets holdings drift, but bars cannot reveal
  queue position, auction dynamics, intraday path, or order-book state.
- Spread, slippage, square-root impact, participation, and capacity settings
  are transparent sensitivities, not estimates calibrated to proprietary
  order-level execution data. Partial DAY-order residuals expire rather than
  following a production order-management lifecycle.
- The C++ order book simulates fills against a *synthetic* book shape; it is
  a systems-engineering and parity-testing module, not a historical market
  microstructure calibration or the paper replay's source of truth.
- DSR/PBO correct for the trials the platform knows about; they cannot correct
  for ideas discarded before they were coded.
- No broker integration places live orders.
- Model results can overfit even with walk-forward validation, purging, and
  deflated statistics.
- Real deployment would require stronger data licensing, monitoring, capital
  controls, compliance review, and independent validation.
