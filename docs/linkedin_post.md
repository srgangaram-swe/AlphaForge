# LinkedIn Post

I built AlphaForge, a quantitative ML research platform focused on the engineering details that make trading research credible.

The hard part in quant ML is not just fitting a model. It is preventing subtle research mistakes: lookahead leakage, random time-series splits, overlapping labels, in-sample backtests, same-close execution, and transaction costs that quietly disappear.

AlphaForge is built around those constraints. It uses a canonical long-format market data panel, causal technical and cross-sectional features, multi-horizon forward labels, embargoed walk-forward validation, and out-of-sample prediction panels. Backtests consume only OOS predictions and include next-bar execution, turnover, commissions, spread, slippage, portfolio caps, volatility targeting, and risk analytics.

The stack includes synthetic data for offline verification, yfinance/CSV loaders, sklearn baselines and ML models, optional PyTorch models, ensembles, a Streamlit dashboard, FastAPI endpoints, Docker, CI, and tests for leakage, label alignment, walk-forward split integrity, and cost impact.

This project was a way to connect quantitative research, ML engineering, and production-quality software design in one repo.
