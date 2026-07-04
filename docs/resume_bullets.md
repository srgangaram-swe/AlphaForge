# Resume Bullets

Pick 3–5 per application; lead with the ones matching the job description.

## Quant researcher / systematic trading flavor

- Built AlphaForge, an end-to-end quantitative ML research platform: public market data ingestion, causal feature engineering, multi-horizon return labeling, embargoed walk-forward validation, out-of-sample prediction panels, portfolio construction, and cost-aware backtesting.
- Implemented the modern anti-overfitting toolkit — Purged K-Fold and Combinatorial Purged CV, Deflated Sharpe Ratio, Probability of Backtest Overfitting (CSCV), and Newey-West IC inference — and wired it into every experiment report.
- Designed a regime-aware alpha stack: a from-scratch 2-state Gaussian HMM (Baum-Welch EM) applied strictly causally (expanding refits, filtered probabilities) for stress detection, regime-gated exposure, and regime-conditional performance attribution.
- Engineered leakage-safe time-series ML workflows with embargoed splits and CI tests that mutate future data and assert bit-identical past features.

## ML engineering / infrastructure flavor

- Built a config-driven model platform spanning baselines, regularized linear models, tree ensembles, optional PyTorch sequence models (GRU/TCN), and an IC-weighted ensemble that scores members on a chronological inner validation split.
- Designed a vectorized backtesting engine with next-bar execution, turnover-based commissions, spread/slippage, and volatility-targeting/drawdown controls whose own rebalancing trades are costed; added beta-aware stress testing and regime-conditional attribution.
- Shipped reproducible research infrastructure: YAML-configured CLI pipelines, deterministic seeds, versioned run artifacts, pytest suites (leakage, parity, split integrity, cost impact), Docker, GitHub Actions CI, a Streamlit dashboard, and FastAPI endpoints.

## Low-latency / systems flavor

- Implemented a C++17 price-time-priority limit order book (O(1) cancels via an iterator index, integer-tick determinism) with pybind11 bindings, benchmarked at ~6M ops/s with ~125 ns median and sub-microsecond p99 latency.
- Maintained a pure-Python reference implementation with identical semantics and CI parity tests requiring bit-identical fills, depth, and book state across both engines; used the native core for depth-aware fill simulation replacing flat-bps slippage assumptions.
