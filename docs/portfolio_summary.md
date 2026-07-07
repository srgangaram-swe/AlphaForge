# Portfolio Summary

## Suggested GitHub description

`Regime-aware quant ML research platform: leakage-safe walk-forward + purged CV, deflated Sharpe / PBO overfitting stats, HMM regime engine, costed OOS backtests, and a C++17 order-book execution core (pybind11).`

## Suggested topics

`quantitative-finance`, `machine-learning`, `backtesting`, `time-series`,
`alpha-research`, `portfolio-construction`, `risk-management`, `python`,
`cpp`, `pybind11`, `order-book`, `scikit-learn`, `hidden-markov-model`

## Portfolio website blurb

AlphaForge is a regime-aware quantitative ML research platform built to
demonstrate the engineering discipline behind credible trading research.
Every default in the ML toolbox silently cheats on financial time series —
random splits leak, overlapping labels leak, full-sample scalers leak, and
"best of N models" is selection bias. AlphaForge treats each failure mode as
an engineering requirement with a test: CI mutates future market data and
asserts past features are bit-identical; walk-forward and purged/combinatorial
CV splitters enforce embargoes; every run reports deflated Sharpe ratios and
the probability of backtest overfitting; and the backtest costs the turnover
of its own risk controls. A from-scratch Gaussian HMM provides causal regime
awareness, and a C++17 limit-order-book execution core (~6M ops/s, ~125 ns
median latency, parity-tested against a Python reference) powers depth-aware
fill simulation. Fully reproducible offline via a synthetic regime-switching
market generator; shipped with tests, CI, Docker, a Streamlit dashboard, and
a FastAPI service.

## One-liner

Research platform that shows *why most backtests are wrong* — and ships the
tests, statistics, and infrastructure that catch it.
