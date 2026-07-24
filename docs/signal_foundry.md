# Signal Foundry governed research

AlphaForge consumes Signalattice output through a narrow, versioned data
contract. The repositories do not import each other and do not share runtime
state. This boundary makes provenance, licensing, temporal semantics, and
failure behavior independently reviewable.

This workflow supports research and zero-capital shadow evaluation. It has no
broker adapter, order-routing interface, credential path, or live-trading
authorization. A favorable backtest is not evidence of guaranteed profit.

## Trust boundary

`alphaforge.data.load_signal_foundry_dataset` accepts one content-addressed
bundle directory and fails closed unless all of these checks pass:

- contract name and exact schema version `1.0.0`;
- bundle directory name and semantic manifest SHA-256 identity;
- safe relative paths, unique partitions, partition SHA-256 hashes, row counts,
  aggregate date bounds, and ticker universe;
- exact column order, finite positive prices, nonnegative volume, valid OHLC
  bounds, unique `(date, ticker)` keys, and nonempty identifiers;
- timezone-normalized `effective_at`, `available_at`, and `observed_at`, with no
  observation or availability before effective time;
- explicit source adjustment state, license booleans, and point-in-time
  limitations for historical revisions, universe membership, and corporate
  actions.

The optional `as_of` argument enforces
`available_at <= decision timestamp`. Provider-adjusted close data is mapped
into AlphaForge's adjusted-OHLCV representation using the declared adjustment
state. Unknown adjustment semantics are rejected instead of guessed.

The loader preserves two distinct views. The market panel retains the actual
session dates used for labels, prices, and fills. The decision panel dates each
bar at the first represented session close at which its `available_at`
timestamp made it usable. Features use the decision panel; labels and execution
use the market panel. The generic one-panel loader therefore rejects Signal
Foundry input and directs operators to this governed workflow.

Licensed bundles must remain outside Git. The repository ignores
`data/signal-foundry/`, `data/raw/`, caches, processed panels, and all run
artifacts. Docker context exclusions provide a second boundary. Only synthetic
contract fixtures are committed.

## Pre-registered evaluation

The default policy lives in
`configs/signal_foundry_research.yaml`. Change it only before starting a new
immutable evaluation. The run identity hashes:

- the producer bundle;
- the AlphaForge Git commit;
- model candidates and hyperparameters;
- features and labels;
- walk-forward and embargo policy;
- backtest, cost, execution, portfolio, and risk settings; and
- the final paper-readiness rubric.

The workflow:

1. builds causal features and multi-horizon forward labels;
2. ends development before the final holdout and purges an embargo at least as
   long as the maximum label horizon;
3. compares every registered candidate using development-only walk-forward
   evidence;
4. records each candidate and result in an ordered SHA-256 hash chain;
5. fits the selected candidate on development data and touches the final
   holdout once for that immutable run identity;
6. backtests future-open execution with commissions, spread, slippage,
   square-root impact, participation limits, drifted positions, and reconciled
   cash/share accounting;
7. repeats the holdout under doubled costs, tripled spread/slippage, an extra
   session of latency, halved participation, perturbed selection breadth, and
   explicit borrow/funding drag, then compares against a deterministic
   permuted-signal placebo;
8. produces circular moving-block uncertainty intervals, year/regime
   stability, drawdown duration, missing-price halt evidence, and auditable AUM
   capacity sensitivities; and
9. publishes the run atomically with hashes and a machine-readable dossier.

Existing run identities cannot be overwritten or repeated. A failed run
removes its staging directory; a stale staging directory fails closed for
operator inspection.

## Run locally

Install the data dependencies, then provide the absolute path to an immutable
Signalattice bundle:

```bash
python -m pip install -e ".[dev,data]"
make signal-foundry \
  BUNDLE=/absolute/path/to/signalattice/data/processed/signal-foundry/<bundle-id>
```

The command prints the run identity, selected candidate, decision, and dossier
path. It never prints or needs the Nasdaq Data Link API key; acquisition occurs
in Signalattice.

## Readiness decision

The dossier reports exactly `READY_FOR_PAPER` or `NOT_READY`. Every configured
gate must pass:

- sufficient untouched holdout sessions;
- Deflated Sharpe probability after accounting for attempted candidates;
- bounded Probability of Backtest Overfitting;
- bounded drawdown and turnover;
- nonnegative annual return relative to the investable benchmark;
- complete declared point-in-time evidence;
- reconciled accounting; and
- cost, latency, liquidity, and placebo stress success.

`READY_FOR_PAPER` permits only a time-bounded, zero-capital offline or shadow
exercise under the documented controls. It does not authorize live trading,
capital deployment, broker connectivity, or a risk increase. Nasdaq source
bundles that honestly declare incomplete revision, universe-membership, or
corporate-action history must receive `NOT_READY` under the default rubric,
even when their return statistics look attractive.

## Paper-control boundary

`alphaforge.paper.PaperControlState` evaluates proposed shadow-state
transitions without creating executable orders. It enforces unique decision
identities, data freshness, gross/net/position/turnover/notional bounds,
drawdown and daily-loss limits, and a one-way manual kill switch. Any breach
produces a machine-readable halt reason. Invalid or non-finite input raises
before policy evaluation.

Moving beyond zero-capital shadow evaluation requires a separate milestone,
threat model, broker-specific contract, reconciliation and recovery design,
operational rehearsal, explicit capital-at-risk cap, legal and tax review, and
the owner's affirmative approval. None of those capabilities are present in
this release.
