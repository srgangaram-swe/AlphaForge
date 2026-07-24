# Changelog

Notable changes follow [Keep a Changelog](https://keepachangelog.com/) and
semantic versioning. AlphaForge is pre-1.0; research and artifact contracts may
still evolve between minor releases.

## [Unreleased]

## [0.2.1] - 2026-07-23

### Added

- Independent, fail-closed consumption of the Signal Foundry v1 market-data
  contract, including semantic identity, partition hash, schema, license,
  temporal, and point-in-time validation.
- Separate market and decision-eligible panels so after-close publication
  delays cannot leak a bar into a same-close feature or order decision.
- Pre-registered development selection with a purged pre-holdout embargo and
  one immutable final-holdout evaluation per bundle/code/config identity.
- Hash-chained trial ledger, cost/liquidity/latency/placebo stresses, and a
  machine-readable `READY_FOR_PAPER` or `NOT_READY` dossier.
- Offline paper-decision controls for idempotency, stale data, exposure,
  position, turnover, notional, drawdown, daily loss, and a one-way kill switch.
- Typed close-decision/future-open order and fill contracts.
- Self-financing signed-share/cash ledger with daily and symbol-level P&L reconciliation.
- Lagged-ADV participation caps, partial DAY fills, and square-root impact sensitivity.
- Auditable orders, fills, holdings, P&L attribution, and capacity-scenario artifacts.
- CI type checking, wheel smoke installation, Python 3.12–3.14 matrix, and branch coverage gate.

### Changed

- Backtesting and paper replay now share the same causal daily-bar execution policy.
- Target weights drift between explicit rebalances; restoring a target generates costed trades.
- Research CLI runs explicitly liquidate after the final OOS target stream.
- Package metadata and API version are aligned at `0.2.1` with an SPDX MIT license.
- CI installs every dependency needed by the supported test matrix instead of
  silently skipping temporal-model coverage.

### Fixed

- Prevented close-time decisions from receiving an overnight return that occurred before fill.
- Removed implicit free rebalancing from persisted target-weight matrices.
- Included the first active session in compounded total return.

## [0.1.0] - 2026-07-07

### Added

- Initial research pipeline, validation science, API/notebooks, and Python/C++ order-book parity.
