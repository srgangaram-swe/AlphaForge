"""Transaction-cost inputs shared by historical and paper execution.

Spread and slippage are represented in the simulated fill price. Commission
is debited separately from cash.  Keeping those pieces separate makes the
portfolio ledger reconcile exactly while preserving the legacy ``rate`` and
``costs`` helpers used by callers that only need a linear cost estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 1.0
    half_spread_bps: float = 2.5
    slippage_bps: float = 2.0

    def __post_init__(self) -> None:
        values = (self.commission_bps, self.half_spread_bps, self.slippage_bps)
        if any(not np.isfinite(value) or value < 0 for value in values):
            raise ValueError("transaction-cost basis points must be finite and non-negative")

    @property
    def rate(self) -> float:
        return (self.commission_bps + self.half_spread_bps + self.slippage_bps) / 10_000.0

    @classmethod
    def from_config(cls, config: dict | None) -> CostModel:
        cfg = config or {}
        allowed = {"commission_bps", "half_spread_bps", "slippage_bps"}
        unknown = set(cfg) - allowed
        if unknown:
            raise ValueError(f"unknown transaction-cost settings: {sorted(unknown)}")
        return cls(
            commission_bps=float(cfg.get("commission_bps", 1.0)),
            half_spread_bps=float(cfg.get("half_spread_bps", 2.5)),
            slippage_bps=float(cfg.get("slippage_bps", 2.0)),
        )

    def costs(self, turnover: pd.Series) -> pd.Series:
        return turnover.astype(float) * self.rate
