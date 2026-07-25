# Unified model contract and naive baselines (SF-S2-MR4)

Every alpha model in AlphaForge — sklearn wrapper, torch sequence model,
ensemble, or naive baseline — implements one contract,
[`AlphaModel`](../alphaforge/models/base.py). This document describes the
contract, its fitted-state and serialization semantics, and the naive baselines
that bound what "adding value" means.

## The contract

| member | required | meaning |
|--------|----------|---------|
| `fit(X, y) -> self` | yes | fit on a feature frame `X` and target `y` |
| `predict(X) -> np.ndarray` | yes | expected forward return per row |
| `predict_proba(X) -> np.ndarray` | no | up-probability (classification); regressors raise `ProbabilityNotSupportedError` |
| `predict_uncertainty(X) -> np.ndarray \| None` | no | per-row predictive std (default `None`) |
| `feature_importance() -> pd.Series \| None` | no | per-feature importance (default `None`) |
| `metadata() -> ModelMetadata` | provided | versioned model + fitted-state description |
| `save(path)` / `load(path)` | provided | deterministic serialization (see below) |
| `get_params() -> dict` | no | JSON-safe constructor params used to rebuild the model |
| `required_features() -> tuple \| None` | no | features checked at predict (`None` = full training schema) |

`task` is `"regression"` (predict expected returns) or `"classification"`
(predict probabilities). `feature_agnostic = True` marks models whose prediction
ignores the feature columns (the constant baselines).

## Fitted-state and feature-schema semantics

The shared behaviour is installed once, in `AlphaModel.__init_subclass__`, which
wraps each subclass's own `fit` and `predict`:

* **fit** validates inputs (X is a numeric `DataFrame`, y is a finite `Series`,
  matching non-empty lengths — otherwise a typed `FeatureSchemaError` /
  `InvalidLabelError`), runs the model's fit, then records the feature schema and
  marks the model fitted.
* **predict** requires the model to be fitted (`NotFittedError` otherwise) and
  checks that the model's required features are present
  (`FeatureSchemaError` otherwise).
* A **re-entrancy guard** (`_fitting`) means a model that calls its own `predict`
  *during* `fit` (e.g. for early stopping) is never tripped by the not-fitted
  check.

Because the wrapping is in the base class, **all** models — including ones added
later — inherit these semantics without changing their fit/predict bodies. A
model restored via a custom `load` (e.g. `TemporalAlphaModel`) sets `_is_fitted`
so a loaded model reports fitted.

Errors are typed (`ModelError` and its subclasses `NotFittedError`,
`FeatureSchemaError`, `InvalidLabelError`, `ProbabilityNotSupportedError`), so
callers react to a specific failure rather than parsing a message. This is the
issue's non-goal made concrete: no behaviour hidden behind untyped dictionaries.

## Metadata

`metadata()` returns an immutable, versioned `ModelMetadata`
(`name`, `task`, `contract_version`, `fitted`, `n_features`, `feature_names`,
`params`). `contract_version` (currently `1.0.0`) is bumped when the interface
changes. `metadata().to_dict()` is JSON-friendly and deterministic under
`sort_keys`.

## Serialization

* Models with **JSON-safe fitted state** — every naive baseline — serialize to a
  deterministic, sorted-key JSON container. Two saves of an equally-fitted model
  are **byte-identical**, and a load reconstructs the model via the registry plus
  `_load_fitted_state`.
* Other models (sklearn pipelines, torch nets) fall back to **joblib**, which
  round-trips but is not byte-deterministic. `TemporalAlphaModel` keeps its own
  torch-checkpoint `save`/`load`.

`AlphaModel.load(path)` sniffs the format (JSON vs joblib) and raises `ModelError`
if a joblib payload is not an `AlphaModel`.

## Naive baselines

None is deployable; they are the floor every ML model must clear and are included
automatically in comparative evidence.

| registry name | task | signal |
|---------------|------|--------|
| `zero_baseline` | regression | 0 everywhere (efficient-market null) |
| `historical_mean` | regression | training-set mean; reports training std as uncertainty |
| `lag_baseline` | regression | most recent observed return (persistence) |
| `moving_average_baseline` | regression | sign of price-vs-moving-average (trend rule) |
| `momentum_baseline` | regression | scaled momentum feature |
| `equal_probability` | classification | constant up-probability (coin flip) |
| `buy_and_hold` | regression | constant long signal |
| `equal_weight` | regression | uniform score → equal weights |

`buy_and_hold` and `equal_weight` emit the same per-row constant signal; they are
distinct *portfolio* benchmarks (persistent long exposure vs equal cross-sectional
weighting), documented as such.

## Configuration and usage

```python
from alphaforge.models import create_model, available_models, AlphaModel

model = create_model("historical_mean").fit(X, y)
preds = model.predict(X)
model.save("hist.json")                 # deterministic JSON
restored = AlphaModel.load("hist.json")

print(available_models())               # registry names
print(model.metadata().to_dict())
```

Feature-selecting baselines take their column name as a parameter, e.g.
`create_model("lag_baseline", feature="ret_1d")`; the column must be present at
fit and predict.

## Reproducibility

Baseline predictions and serialized state are deterministic (no randomness; the
JSON container is byte-stable). Reproduce with
`pytest tests/test_model_contract.py -q`.

## Security, risks, rollback, and limitations

* No credential, licensed observation, dataset, or run artifact is committed;
  tests use in-process synthetic data.
* **Rollback.** The change is additive to the base class (new concrete methods
  and a fit/predict wrapper) plus new baselines; existing model behaviour is
  unchanged apart from `TemporalAlphaModel.load` now marking the model fitted.
  Reverting the commit restores the previous state.
* **Deterministic serialization is guaranteed only for JSON-state models** (the
  baselines); joblib payloads round-trip but are not byte-stable.
* **A baseline is not a strategy.** These bound comparison; they carry no
  predictive-edge claim and must never be treated as deployable.
* **Sprint 1 dependency.** The contract is built on the current `dev`; it consumes
  the Sprint 1 feature, label, configuration, and validation contracts once those
  are merged and verified.
