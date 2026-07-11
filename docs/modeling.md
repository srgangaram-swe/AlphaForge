# Modeling

## Models

- Zero, historical mean, and momentum baselines — every ML model must beat these.
- Linear regression, ridge, lasso, and elastic net (imputation + scaling
  embedded in the pipeline, so statistics are always train-window-only).
- Random forest and gradient boosting (LightGBM when installed, sklearn
  HistGradientBoosting otherwise).
- Optional PyTorch MLP, GRU, and temporal CNN with causal per-symbol sequence
  construction and time-ordered early-stopping splits.
- **TemporalAlphaNet** (`alphaforge/models/temporal.py`, ADR 0002): the
  flagship neural model — dilated causal TCN encoder with attention pooling,
  optional multi-task horizon heads, and a composite Huber + cross-sectional
  IC loss on date-batched mini-batches. Trained by a real loop (AdamW +
  cosine decay, gradient clipping, early stopping on validation rank IC,
  best-checkpoint restore, persisted per-epoch history) via
  `scripts/train_model.py` (`make train`), and available in walk-forward
  comparisons as `temporal_alpha`.
- **IC-weighted ensemble**: members are fit on the first 80% of the training
  window (chronological, never shuffled), scored by rank IC on the held-out
  tail, weighted by max(IC, 0) + floor, then refit on the full window.
- **Gaussian HMM regime model** (`alphaforge/models/regime.py`): 2-state
  Baum-Welch EM from scratch. Used causally — expanding parameter refits and
  filtered (never smoothed) state probabilities — as a feature
  (`hmm_stress_prob`) and for regime-gated exposure.

The registry is config-driven. Walk-forward validation instantiates a fresh
model per window, fits only on training rows, and emits predictions only for
test rows.

## Validation

- **Walk-forward** (primary): expanding or rolling windows with an embargo
  at least as long as the longest label horizon.
- **Purged K-Fold / CPCV** (`alphaforge/training/purged_cv.py`): purging
  removes train dates whose label intervals overlap a test block; the embargo
  kills serial-correlation leakage from trailing-window features. CPCV
  evaluates every C(n, k) test-group combination, producing many OOS paths.

## Overfitting statistics (alphaforge/evaluation/overfitting.py)

- **PSR** — P(true Sharpe > benchmark), adjusted for sample length, skew, kurtosis.
- **DSR** — PSR against the expected best-of-N unskilled Sharpe; `n_trials`
  counts every model variant that competed for selection.
- **PBO (CSCV)** — probability the in-sample winner underperforms the median
  out-of-sample, computed from per-date rank-IC panels across models.
- **Newey-West t-stats** — IC series are serially correlated under
  overlapping multi-day labels; HAC errors keep significance honest.
