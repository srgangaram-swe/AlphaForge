"""Typed execution contracts and a causal daily-bar fill model.

The daily-bar model deliberately does not pretend to reconstruct an order
book.  It fills at a configured bar price, applies explicit spread and impact
assumptions, and can cap quantity using *lagged* average daily volume.  The
caller is responsible for supplying only information known before the fill.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from alphaforge.execution.costs import CostModel

FillStatus = Literal["filled", "partial", "rejected"]
MissingPricePolicy = Literal["raise", "skip"]


@dataclass(frozen=True)
class ExecutionPolicy:
    """Assumptions used to turn a target rebalance into daily-bar fills."""

    price_field: Literal["open"] = "open"
    adv_lookback: int = 20
    volatility_lookback: int = 20
    max_participation_rate: float | None = None
    impact_coefficient: float = 0.0
    missing_price_policy: MissingPricePolicy = "raise"

    def __post_init__(self) -> None:
        if self.price_field != "open":
            raise ValueError("daily-bar execution currently supports next-open fills only")
        if self.adv_lookback < 1 or self.volatility_lookback < 2:
            raise ValueError("execution lookbacks must be positive")
        if self.max_participation_rate is not None and not (0 < self.max_participation_rate <= 1):
            raise ValueError("max_participation_rate must be in (0, 1]")
        if not np.isfinite(self.impact_coefficient) or self.impact_coefficient < 0:
            raise ValueError("impact_coefficient must be finite and non-negative")
        if self.missing_price_policy not in {"raise", "skip"}:
            raise ValueError("missing_price_policy must be 'raise' or 'skip'")

    @classmethod
    def from_config(cls, config: dict | None) -> ExecutionPolicy:
        cfg = config or {}
        allowed = {
            "price_field",
            "adv_lookback",
            "volatility_lookback",
            "max_participation_rate",
            "impact_coefficient",
            "missing_price_policy",
        }
        unknown = set(cfg) - allowed
        if unknown:
            raise ValueError(f"unknown execution settings: {sorted(unknown)}")
        return cls(
            price_field=str(cfg.get("price_field", "open")),  # type: ignore[arg-type]
            adv_lookback=int(cfg.get("adv_lookback", 20)),
            volatility_lookback=int(cfg.get("volatility_lookback", 20)),
            max_participation_rate=(
                None
                if cfg.get("max_participation_rate") is None
                else float(cfg["max_participation_rate"])
            ),
            impact_coefficient=float(cfg.get("impact_coefficient", 0.0)),
            missing_price_policy=str(  # type: ignore[arg-type]
                cfg.get("missing_price_policy", "raise")
            ),
        )


@dataclass(frozen=True)
class Order:
    """A day order generated from a close-time portfolio decision."""

    order_id: int
    symbol: str
    decision_date: pd.Timestamp
    fill_date: pd.Timestamp
    requested_shares: float
    target_weight: float
    pretrade_equity: float

    def __post_init__(self) -> None:
        if self.order_id < 1:
            raise ValueError("order_id must be positive")
        if not self.symbol:
            raise ValueError("symbol must be non-empty")
        if self.fill_date <= self.decision_date:
            raise ValueError("fill_date must be strictly after decision_date")
        values = (self.requested_shares, self.target_weight, self.pretrade_equity)
        if any(not np.isfinite(value) for value in values):
            raise ValueError("order values must be finite")
        if self.pretrade_equity <= 0:
            raise ValueError("pretrade_equity must be positive")

    def to_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Fill:
    """Execution result with every modeled source of implementation shortfall."""

    order_id: int
    symbol: str
    decision_date: pd.Timestamp
    fill_date: pd.Timestamp
    status: FillStatus
    requested_shares: float
    filled_shares: float
    residual_shares: float
    reference_price: float
    fill_price: float
    target_weight: float
    pretrade_equity: float
    lagged_adv_shares: float
    lagged_volatility: float
    participation_rate: float
    commission: float
    spread_cost: float
    fixed_slippage_cost: float
    impact_cost: float
    impact_bps: float

    @property
    def traded_notional(self) -> float:
        return abs(self.filled_shares) * self.reference_price

    @property
    def total_cost(self) -> float:
        return self.commission + self.spread_cost + self.fixed_slippage_cost + self.impact_cost

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        record["traded_notional"] = self.traded_notional
        record["total_cost"] = self.total_cost
        return record


class BarExecutionModel:
    """Deterministic next-open fill model for daily OHLCV research."""

    def __init__(self, costs: CostModel, policy: ExecutionPolicy) -> None:
        self.costs = costs
        self.policy = policy

    @classmethod
    def from_config(
        cls,
        costs: CostModel | dict | None = None,
        execution: ExecutionPolicy | dict | None = None,
    ) -> BarExecutionModel:
        cost_model = costs if isinstance(costs, CostModel) else CostModel.from_config(costs)
        policy = (
            execution
            if isinstance(execution, ExecutionPolicy)
            else ExecutionPolicy.from_config(execution)
        )
        return cls(cost_model, policy)

    def execute(
        self,
        order: Order,
        *,
        reference_price: float,
        lagged_adv_shares: float = np.nan,
        lagged_volatility: float = np.nan,
    ) -> Fill:
        """Execute one day order using inputs available before the open.

        ``lagged_adv_shares`` and ``lagged_volatility`` must already be lagged
        by the caller.  A participation limit with unavailable ADV rejects the
        order instead of silently assuming infinite liquidity.
        """
        if not np.isfinite(reference_price) or reference_price <= 0:
            if self.policy.missing_price_policy == "raise":
                raise ValueError(f"missing or invalid open price for {order.symbol}")
            return self._empty_fill(order, reference_price)

        requested_abs = abs(order.requested_shares)
        fill_abs = requested_abs
        adv_is_valid = np.isfinite(lagged_adv_shares) and lagged_adv_shares > 0
        if self.policy.max_participation_rate is not None:
            if not adv_is_valid:
                return self._empty_fill(
                    order,
                    reference_price,
                    lagged_adv_shares=lagged_adv_shares,
                    lagged_volatility=lagged_volatility,
                )
            fill_abs = min(
                fill_abs,
                float(lagged_adv_shares) * self.policy.max_participation_rate,
            )

        if fill_abs <= 0 or requested_abs == 0:
            return self._empty_fill(
                order,
                reference_price,
                lagged_adv_shares=lagged_adv_shares,
                lagged_volatility=lagged_volatility,
            )

        sign = float(np.sign(order.requested_shares))
        filled_shares = sign * fill_abs
        participation = fill_abs / float(lagged_adv_shares) if adv_is_valid else 0.0
        volatility = (
            float(lagged_volatility)
            if np.isfinite(lagged_volatility) and lagged_volatility >= 0
            else 0.0
        )
        impact_bps = (
            self.policy.impact_coefficient
            * volatility
            * 10_000.0
            * np.sqrt(max(participation, 0.0))
        )
        execution_bps = self.costs.half_spread_bps + self.costs.slippage_bps + impact_bps
        fill_price = reference_price * (1.0 + sign * execution_bps / 10_000.0)
        reference_notional = fill_abs * reference_price
        commission = reference_notional * self.costs.commission_bps / 10_000.0
        spread_cost = reference_notional * self.costs.half_spread_bps / 10_000.0
        fixed_slippage_cost = reference_notional * self.costs.slippage_bps / 10_000.0
        impact_cost = reference_notional * impact_bps / 10_000.0
        status: FillStatus = "filled" if np.isclose(fill_abs, requested_abs) else "partial"
        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            decision_date=order.decision_date,
            fill_date=order.fill_date,
            status=status,
            requested_shares=order.requested_shares,
            filled_shares=filled_shares,
            residual_shares=order.requested_shares - filled_shares,
            reference_price=float(reference_price),
            fill_price=float(fill_price),
            target_weight=order.target_weight,
            pretrade_equity=order.pretrade_equity,
            lagged_adv_shares=float(lagged_adv_shares),
            lagged_volatility=float(lagged_volatility),
            participation_rate=float(participation),
            commission=float(commission),
            spread_cost=float(spread_cost),
            fixed_slippage_cost=float(fixed_slippage_cost),
            impact_cost=float(impact_cost),
            impact_bps=float(impact_bps),
        )

    @staticmethod
    def _empty_fill(
        order: Order,
        reference_price: float,
        *,
        lagged_adv_shares: float = np.nan,
        lagged_volatility: float = np.nan,
    ) -> Fill:
        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            decision_date=order.decision_date,
            fill_date=order.fill_date,
            status="rejected",
            requested_shares=order.requested_shares,
            filled_shares=0.0,
            residual_shares=order.requested_shares,
            reference_price=float(reference_price),
            fill_price=float(reference_price),
            target_weight=order.target_weight,
            pretrade_equity=order.pretrade_equity,
            lagged_adv_shares=float(lagged_adv_shares),
            lagged_volatility=float(lagged_volatility),
            participation_rate=0.0,
            commission=0.0,
            spread_cost=0.0,
            fixed_slippage_cost=0.0,
            impact_cost=0.0,
            impact_bps=0.0,
        )
