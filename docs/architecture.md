# Architecture

AlphaForge is a modular research pipeline:

```mermaid
flowchart TD
    raw[Raw market data] --> validate[Schema validation and quality report]
    validate --> features[Causal feature engineering]
    features --> labels[Forward labels]
    labels --> splits[Walk-forward splits with embargo]
    splits --> models[Model training]
    models --> preds[Out-of-sample prediction panel]
    preds --> signals[Signals]
    signals --> portfolio[Portfolio construction]
    portfolio --> backtest[Backtest engine]
    backtest --> risk[Risk analytics]
    risk --> outputs[Report, dashboard, API, paper sim]
```

The central contract is the canonical panel:

`date | symbol | open | high | low | close | volume`

Every downstream module either consumes this panel or a keyed derivative using `(date, symbol)`. Backtests never train models and never use in-sample predictions; they consume the saved OOS prediction panel from walk-forward validation.
