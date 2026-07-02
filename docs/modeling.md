# Modeling

AlphaForge includes:

- Zero, historical mean, and momentum baselines.
- Linear regression, ridge, lasso, and elastic net.
- Random forest and gradient boosting with a LightGBM fallback path when installed.
- Optional PyTorch MLP, GRU, and temporal CNN models.
- Simple weighted ensembles.

The registry is config-driven. Walk-forward validation instantiates a fresh model for each window, fits only on training rows, and emits predictions only for test rows.
