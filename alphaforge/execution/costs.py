"""Transaction cost model used by the backtest engine."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 1.0
    half_spread_bps: float = 2.5
    slippage_bps: float = 2.0

    @property
    def rate(self) -> float:
        return (self.commission_bps + self.half_spread_bps + self.slippage_bps) / 10_000.0

    @classmethod
    def from_config(cls, config: dict | None) -> CostModel:
        cfg = config or {}
        return cls(
            commission_bps=float(cfg.get("commission_bps", 1.0)),
            half_spread_bps=float(cfg.get("half_spread_bps", 2.5)),
            slippage_bps=float(cfg.get("slippage_bps", 2.0)),
        )

    def costs(self, turnover: pd.Series) -> pd.Series:
        return turnover.astype(float) * self.rate
