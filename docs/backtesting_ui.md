# User-facing backtesting UI (SF-S2-MR10)

An interactive **configure → run → evidence** workflow that lets a user pick data
and a model, run a leakage-safe walk-forward backtest, and compare it to naive
baselines — the point where **Signalattice and AlphaForge complement one
another**: Signalattice supplies point-in-time data (a Signal Foundry bundle),
AlphaForge turns it into simulated evidence.

Everything here is **simulated research, not live inference or executable orders**.

## Architecture

Three layers, thinnest on top:

1. **Service** — [`alphaforge/service/backtest_service.py`](../alphaforge/service/backtest_service.py).
   A typed, JSON-serializable `BacktestRequest → BacktestResult` that composes the
   existing stack (data → features → labels → walk-forward → signals → portfolio
   → event-driven backtest → metrics). Pure stdlib + numpy/pandas, so it stays
   importable in the core install; the chosen model and every requested baseline
   run on identical data, splits, and costs.
2. **API** — [`apps/api.py`](../apps/api.py). `GET /catalog` (models, baselines,
   strategies, discoverable bundles) and `POST /backtests` (a `BacktestSpec` →
   result). Pydantic-validated; invalid input `400`, missing bundle `404`.
3. **Dashboard** — [`apps/dashboard.py`](../apps/dashboard.py). A Streamlit
   "Run a backtest" view: a configuration sidebar, a Run button, and results —
   metric tiles, an equity-vs-benchmark curve, drawdown, a model-vs-baselines
   table, recent simulated fills, and a reproducibility caption.

The API and dashboard require the committed `app` extra
(`pip install -e '.[app]'`); the service does not.

## The request/result contract

`BacktestRequest` selects the data source (`synthetic` or `signal_foundry` with a
`bundle_dir`), universe/date size (synthetic), benchmark, model + params, the
baselines to compare, the signal strategy, transaction cost (bps), the seed, and
the walk-forward window sizes. It validates on construction and **fails closed**
with `BacktestServiceError` on unknown models/baselines/strategies, a missing
bundle, or out-of-range values.

`BacktestResult` carries the config echo, a `data_id` (synthetic descriptor or
`bundle:<bundle-id>`), the headline strategy's equity/benchmark/drawdown series
and metrics, a comparison row per strategy, a tail of simulated fills, a
reproducibility block (`seed`, `config_hash`, `data_id`, service version), and the
disclaimer. It is deterministic for a fixed request and JSON-safe (NaN/inf → null).

## Using it

```bash
pip install -e '.[app]'
uvicorn apps.api:app --reload            # API at http://127.0.0.1:8000/docs
streamlit run apps/dashboard.py          # dashboard
```

```python
from alphaforge.service import BacktestRequest, run_backtest_service

result = run_backtest_service(BacktestRequest(model="random_forest",
                                              baselines=("zero_baseline", "momentum_baseline")))
print(result.comparison)                 # model vs baselines, same data/costs
```

Signalattice bundles are discovered under `data/signal-foundry-bundles/`; select
one in the sidebar (or pass `data_source="signal_foundry", bundle_dir=...`) and a
benchmark symbol from the bundle universe.

## Design and accessibility

Following current backtesting-UI practice, the **equity curve vs benchmark** leads,
with **drawdown** below it and a compact metric row (total return, Sharpe, Sortino,
max drawdown, annualized vol, turnover); the honest model-vs-baselines comparison
sits alongside. Charts use a **colorblind-safe Wong palette** (strategy blue
`#0072B2`, benchmark orange `#E69F00`, drawdown vermillion `#D55E00`), percent
axes, and explicit legends. The pure presentation helpers (`format_metric_tiles`,
`build_comparison_table`, `build_equity_figure`, `build_drawdown_figure`) are
unit-tested independently of the Streamlit runtime.

## Reproducibility

No randomness beyond the recorded seed; a fixed request reproduces byte-identical
metrics and a stable `config_hash`. The result records the data identity and
config so any run can be reproduced. Tests use in-process synthetic data; no
credential, licensed observation, bundle, or run artifact is committed.

## Risks, rollback, and limitations

* **Rollback.** Additive: a new `alphaforge/service/` package and new endpoints/
  view; the existing latest-run API and dashboard are unchanged and remain
  reachable. Reverting the commit restores the previous state.
* **Simulated only.** No live data, order routing, or paper/live controls; the
  disclaimer is on every surface.
* **Synthetic default.** The default data source is the deterministic synthetic
  market; real evidence requires a Signalattice bundle, whose own limitations
  (stale/current-vintage engineering data) still apply.
* **Compute.** Each run trains the model and baselines across walk-forward
  windows; keep the synthetic universe/horizon modest for interactive latency.
* **Sprint 1 dependency.** Built on current `dev`; final integration with the
  Sprint 1 data/config/validation contracts follows as they land.
