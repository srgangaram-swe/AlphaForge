# Model Card

## Intended Use

Educational research into return prediction, walk-forward validation, and realistic backtesting.

## Not Intended For

Live trading, financial advice, guaranteed return claims, or automated capital allocation.

## Inputs

Causal technical, cross-sectional, benchmark-relative, and market-regime features built from public or synthetic OHLCV bars.

## Outputs

Expected forward-return predictions by `(date, symbol)` for a configured horizon.

## Evaluation

Models are evaluated on out-of-sample walk-forward windows using MSE, MAE, R2, directional accuracy, Pearson IC, Spearman rank IC, and quantile return analysis.

## Risks

Data quality issues, leakage bugs, overfitting, transaction cost underestimation, and market regime instability.
