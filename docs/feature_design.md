# Feature Design

Feature families include lagged returns, log returns, rolling volatility, moving-average ratios, momentum, RSI, MACD, Bollinger z-scores, mean reversion, volume features, drawdown, rolling Sharpe-style ratios, rolling beta/correlation to a benchmark, market regime features, and cross-sectional ranks.

The feature contract is causal: a row at time `t` may only depend on data available at or before `t`. Tests mutate future market data and assert that past feature rows remain unchanged.

Model scaling is fit inside each training window through sklearn pipelines or train-only torch normalization.
