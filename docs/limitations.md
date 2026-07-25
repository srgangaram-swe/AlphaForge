# Limitations

AlphaForge is an educational research platform, not a production trading system.

Important limitations:

- yfinance data can contain survivorship, adjustment, and availability issues.
- The no-cost Nasdaq WIKI engineering bundle ends on 2018-03-27, is
  current-vintage, and lacks complete historical revisions, point-in-time
  universe membership, and corporate-action records. It can validate pipeline
  mechanics but must not qualify current paper or live readiness.
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
- Offline paper controls emit proposed state decisions only. They do not model
  broker acknowledgements, rejects, outages, recovery, exchange state, legal or
  tax obligations, or operational capital controls.
- Model results can overfit even with walk-forward validation, purging, and
  deflated statistics.
- Label diagnostics quantify overlap, autocorrelation, class balance, temporal
  drift, and parameter sensitivity but do not prove that a target is
  predictable. Daily-close triple barriers cannot establish intraday hit
  ordering, and synthetic label evidence is not market evidence.
- Real deployment would require stronger data licensing, monitoring, capital
  controls, compliance review, and independent validation.
