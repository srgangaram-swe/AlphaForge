"""Self-financing portfolio accounting for event-driven backtests.

The ledger deliberately owns only accounting state: cash and signed share
quantities.  Execution models remain responsible for constructing all-in fill
prices (including spread and market impact) and for calculating commissions.
This separation keeps the cash equation explicit and makes fills easy to
reconcile.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

_POSITION_ZERO_TOLERANCE = 1e-12
_ACCOUNTING_REL_TOLERANCE = 1e-12
_ACCOUNTING_ABS_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class LedgerSnapshot:
    """An immutable mark-to-market view of a portfolio.

    Position quantities and market values are signed, so short positions have
    negative values and weights.  Cash is not included in ``weights``; its
    implicit portfolio weight is ``cash / equity``.
    """

    date: object
    cash: float
    equity: float
    positions: Mapping[str, float]
    market_values: Mapping[str, float]
    weights: Mapping[str, float]


class PortfolioLedger:
    """Track cash and signed shares under self-financing accounting.

    Positive fill quantities buy shares and negative quantities sell shares.
    For every fill the cash balance changes according to

    ``cash -= signed_quantity * fill_price + commission``.

    The fill price is assumed to be all-in: spread, slippage, and market impact
    must already be embedded in it.  Commission is the only separately charged
    execution cost.
    """

    def __init__(self, initial_cash: float = 1_000_000.0) -> None:
        cash = _finite_float(initial_cash, name="initial_cash")
        if cash <= 0.0:
            raise ValueError("initial_cash must be positive")
        self._cash = cash
        self._positions: dict[str, float] = {}

    @property
    def cash(self) -> float:
        """Current cash balance, which may be negative when the book is financed."""
        return self._cash

    @property
    def positions(self) -> Mapping[str, float]:
        """A read-only copy of current signed share quantities."""
        return MappingProxyType(dict(self._positions))

    def market_values(self, prices: Mapping[str, float]) -> dict[str, float]:
        """Mark all open positions at ``prices`` and return signed values."""
        validated_prices = _validate_prices(prices, required=set(self._positions))
        values = {
            symbol: quantity * validated_prices[symbol]
            for symbol, quantity in sorted(self._positions.items())
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("position market values must be finite")
        return values

    def equity(self, prices: Mapping[str, float]) -> float:
        """Return marked equity as cash plus signed position market values."""
        market_values = self.market_values(prices)
        return _finite_sum((self._cash, *market_values.values()), name="marked equity")

    def target_orders(
        self,
        target_weights: Mapping[str, float],
        reference_prices: Mapping[str, float],
    ) -> dict[str, float]:
        """Compute signed share trades needed to reach target weights.

        Every target is sized from the same pre-trade marked equity.  Existing
        positions omitted from ``target_weights`` receive a zero target and are
        therefore liquidated.  The method does not impose leverage, gross, or
        net exposure policy; it only requires finite weights.

        Returns only non-zero orders.  Positive quantities are buys and
        negative quantities are sells.
        """
        weights = _validate_weights(target_weights)
        symbols = set(self._positions) | set(weights)
        prices = _validate_prices(reference_prices, required=symbols)
        pretrade_equity = self.equity(prices)
        if pretrade_equity <= 0.0:
            raise ValueError("pre-trade equity must be positive to size target orders")

        orders: dict[str, float] = {}
        for symbol in sorted(symbols):
            target_quantity = weights.get(symbol, 0.0) * pretrade_equity / prices[symbol]
            signed_quantity = target_quantity - self._positions.get(symbol, 0.0)
            if not math.isfinite(signed_quantity):
                raise ValueError(f"target order quantity for {symbol!r} must be finite")
            if signed_quantity != 0.0:
                orders[symbol] = signed_quantity
        return orders

    def apply_fill(
        self,
        symbol: str,
        signed_quantity: float,
        fill_price: float,
        commission: float = 0.0,
    ) -> None:
        """Apply one fill atomically using the self-financing cash equation.

        No spread or impact adjustment is performed here; callers must supply
        the actual all-in execution price.  A commission must be finite and
        non-negative.
        """
        _validate_symbol(symbol)
        quantity = _finite_float(signed_quantity, name="signed_quantity")
        price = _positive_price(fill_price, name="fill_price")
        fee = _finite_float(commission, name="commission")
        if fee < 0.0:
            raise ValueError("commission must be non-negative")

        old_quantity = self._positions.get(symbol, 0.0)
        notional = quantity * price
        new_cash = self._cash - notional - fee
        new_quantity = old_quantity + quantity
        if not math.isfinite(notional):
            raise ValueError("fill notional must be finite")
        if not math.isfinite(new_cash):
            raise ValueError("cash balance after fill must be finite")
        if not math.isfinite(new_quantity):
            raise ValueError("position quantity after fill must be finite")

        old_value_at_fill = old_quantity * price
        if not math.isfinite(old_value_at_fill):
            raise ValueError("position market value at fill price must be finite")
        before_at_fill = _finite_sum(
            (self._cash, old_value_at_fill),
            name="pre-fill accounting value",
        )
        expected_after = _finite_float(before_at_fill - fee, name="post-fill accounting value")
        stored_quantity = new_quantity

        # Clean numerical residue only when its marked value is also negligible.
        # Otherwise retain the exact residual rather than hiding a potentially
        # material accounting difference in a large portfolio.
        if (
            math.isclose(
                new_quantity,
                0.0,
                rel_tol=0.0,
                abs_tol=_POSITION_ZERO_TOLERANCE,
            )
            and abs(new_quantity * price) <= _ACCOUNTING_ABS_TOLERANCE
        ):
            stored_quantity = 0.0

        stored_value_at_fill = stored_quantity * price
        if not math.isfinite(stored_value_at_fill):
            raise ValueError("position market value after fill must be finite")
        after_at_fill = _finite_sum(
            (new_cash, stored_value_at_fill),
            name="post-fill accounting value",
        )
        if not math.isclose(
            after_at_fill,
            expected_after,
            rel_tol=_ACCOUNTING_REL_TOLERANCE,
            abs_tol=_ACCOUNTING_ABS_TOLERANCE,
        ):
            raise RuntimeError("fill failed self-financing accounting reconciliation")

        self._cash = new_cash
        if stored_quantity == 0.0:
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = stored_quantity

    def snapshot(self, date: object, prices: Mapping[str, float]) -> LedgerSnapshot:
        """Return an immutable marked portfolio snapshot for ``date``."""
        position_values = self.market_values(prices)
        equity = _finite_sum((self._cash, *position_values.values()), name="marked equity")
        if equity <= 0.0:
            raise ValueError("marked equity must be positive to compute portfolio weights")

        weights = {symbol: value / equity for symbol, value in position_values.items()}
        if not all(math.isfinite(weight) for weight in weights.values()):
            raise ValueError("portfolio weights must be finite")

        return LedgerSnapshot(
            date=date,
            cash=self._cash,
            equity=equity,
            positions=MappingProxyType(dict(self._positions)),
            market_values=MappingProxyType(position_values),
            weights=MappingProxyType(weights),
        )


def _validate_symbol(symbol: object) -> None:
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbols must be non-empty strings")


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_price(value: object, *, name: str) -> float:
    price = _finite_float(value, name=name)
    if price <= 0.0:
        raise ValueError(f"{name} must be positive")
    return price


def _finite_sum(values: Iterable[float], *, name: str) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_prices(
    prices: Mapping[str, float],
    *,
    required: set[str],
) -> dict[str, float]:
    try:
        items = prices.items()
    except AttributeError as exc:
        raise ValueError("prices must be a symbol-to-price mapping") from exc

    validated: dict[str, float] = {}
    for symbol, value in items:
        _validate_symbol(symbol)
        validated[symbol] = _positive_price(value, name=f"price for {symbol!r}")

    missing = sorted(required - set(validated))
    if missing:
        raise ValueError(f"missing prices for symbols: {', '.join(missing)}")
    return validated


def _validate_weights(target_weights: Mapping[str, float]) -> dict[str, float]:
    try:
        items = target_weights.items()
    except AttributeError as exc:
        raise ValueError("target_weights must be a symbol-to-weight mapping") from exc

    validated: dict[str, float] = {}
    for symbol, value in items:
        _validate_symbol(symbol)
        validated[symbol] = _finite_float(value, name=f"target weight for {symbol!r}")

    try:
        gross_weight = math.fsum(abs(weight) for weight in validated.values())
        net_weight = math.fsum(validated.values())
    except OverflowError as exc:
        raise ValueError("target gross and net weights must be finite") from exc
    if not math.isfinite(gross_weight) or not math.isfinite(net_weight):
        raise ValueError("target gross and net weights must be finite")
    return validated
