# Changelog

Notable changes follow [Keep a Changelog](https://keepachangelog.com/) and
semantic versioning. AlphaForge is pre-1.0; research and artifact contracts may
still evolve between minor releases.

## [Unreleased]

### Added

- Typed close-decision/future-open order and fill contracts.
- Self-financing signed-share/cash ledger with daily and symbol-level P&L reconciliation.
- Lagged-ADV participation caps, partial DAY fills, and square-root impact sensitivity.
- Auditable orders, fills, holdings, P&L attribution, and capacity-scenario artifacts.
- CI type checking, wheel smoke installation, Python 3.12–3.14 matrix, and branch coverage gate.

### Changed

- Backtesting and paper replay now share the same causal daily-bar execution policy.
- Target weights drift between explicit rebalances; restoring a target generates costed trades.
- Research CLI runs explicitly liquidate after the final OOS target stream.
- Package metadata and API version are aligned at `0.2.0` with an SPDX MIT license.

### Fixed

- Prevented close-time decisions from receiving an overnight return that occurred before fill.
- Removed implicit free rebalancing from persisted target-weight matrices.
- Included the first active session in compounded total return.

## [0.1.0] - 2026-07-07

### Added

- Initial research pipeline, validation science, API/notebooks, and Python/C++ order-book parity.
