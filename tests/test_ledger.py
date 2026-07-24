from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from alphaforge.backtesting.ledger import PortfolioLedger


def test_long_and_short_fills_update_cash_and_equity() -> None:
    ledger = PortfolioLedger(initial_cash=1_000.0)

    ledger.apply_fill("LONG", signed_quantity=5.0, fill_price=100.0, commission=1.0)
    assert ledger.cash == pytest.approx(499.0)
    assert ledger.positions == {"LONG": 5.0}
    assert ledger.equity({"LONG": 100.0}) == pytest.approx(999.0)

    ledger.apply_fill("SHORT", signed_quantity=-2.0, fill_price=50.0, commission=0.5)
    assert ledger.cash == pytest.approx(598.5)
    assert ledger.positions == {"LONG": 5.0, "SHORT": -2.0}
    assert ledger.equity({"LONG": 100.0, "SHORT": 50.0}) == pytest.approx(998.5)

    ledger.apply_fill("SHORT", signed_quantity=2.0, fill_price=45.0, commission=0.5)
    assert ledger.cash == pytest.approx(508.0)
    assert ledger.positions == {"LONG": 5.0}
    assert ledger.equity({"LONG": 100.0}) == pytest.approx(1_008.0)


def test_all_in_fill_price_and_commission_are_both_reflected_in_equity() -> None:
    ledger = PortfolioLedger(initial_cash=1_000.0)

    # The one-dollar difference between the reference and fill prices represents
    # spread/impact supplied by the execution model; the ledger adds no implicit cost.
    ledger.apply_fill("A", signed_quantity=10.0, fill_price=101.0, commission=1.0)

    assert ledger.cash == pytest.approx(-11.0)
    assert ledger.equity({"A": 100.0}) == pytest.approx(989.0)


def test_target_orders_use_one_pretrade_nav_and_include_liquidations() -> None:
    ledger = PortfolioLedger(initial_cash=1_000.0)
    ledger.apply_fill("A", signed_quantity=4.0, fill_price=100.0)

    # A has appreciated, so pre-trade NAV is 600 cash + 4 * 125 = 1,100.
    orders = ledger.target_orders(
        target_weights={"A": 0.5, "B": -0.25},
        reference_prices={"A": 125.0, "B": 50.0},
    )

    assert orders == pytest.approx({"A": 0.4, "B": -5.5})

    for symbol, quantity in orders.items():
        ledger.apply_fill(symbol, quantity, {"A": 125.0, "B": 50.0}[symbol])
    snapshot = ledger.snapshot("2026-01-02", {"A": 125.0, "B": 50.0})
    assert snapshot.equity == pytest.approx(1_100.0)
    assert snapshot.weights == pytest.approx({"A": 0.5, "B": -0.25})

    liquidation = ledger.target_orders(
        target_weights={"B": 0.0},
        reference_prices={"A": 125.0, "B": 50.0},
    )
    assert liquidation == pytest.approx({"A": -4.4, "B": 5.5})


def test_snapshot_captures_drift_and_is_detached_from_later_state() -> None:
    ledger = PortfolioLedger(initial_cash=1_000.0)
    ledger.apply_fill("A", signed_quantity=6.0, fill_price=100.0)
    ledger.apply_fill("B", signed_quantity=8.0, fill_price=50.0)

    snapshot = ledger.snapshot("2026-01-03", {"A": 110.0, "B": 40.0})

    assert snapshot.cash == pytest.approx(0.0)
    assert snapshot.positions == {"A": 6.0, "B": 8.0}
    assert snapshot.market_values == pytest.approx({"A": 660.0, "B": 320.0})
    assert snapshot.equity == pytest.approx(980.0)
    assert snapshot.weights == pytest.approx({"A": 660.0 / 980.0, "B": 320.0 / 980.0})
    assert snapshot.equity == pytest.approx(snapshot.cash + sum(snapshot.market_values.values()))

    ledger.apply_fill("A", signed_quantity=-1.0, fill_price=110.0)
    assert snapshot.positions == {"A": 6.0, "B": 8.0}
    with pytest.raises(TypeError):
        snapshot.positions["A"] = 99.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        snapshot.cash = 1.0  # type: ignore[misc]


def test_near_zero_position_cleanup_preserves_fill_reconciliation() -> None:
    ledger = PortfolioLedger(initial_cash=1_000.0)
    ledger.apply_fill("A", 0.3, 100.0)
    ledger.apply_fill("A", -(0.1 + 0.2), 100.0)

    assert ledger.positions == {}
    assert ledger.cash == pytest.approx(1_000.0)
    assert ledger.equity({}) == pytest.approx(1_000.0)

    large_ledger = PortfolioLedger(initial_cash=1e12)
    large_ledger.apply_fill("EXPENSIVE", 5e-13, 1e10)
    assert large_ledger.positions == {"EXPENSIVE": 5e-13}
    assert large_ledger.equity({"EXPENSIVE": 1e10}) == pytest.approx(1e12)


@pytest.mark.parametrize("price", [0.0, -1.0, float("nan"), float("inf")])
def test_prices_must_be_finite_and_positive(price: float) -> None:
    ledger = PortfolioLedger(initial_cash=1_000.0)

    with pytest.raises(ValueError):
        ledger.apply_fill("A", 1.0, price)

    assert ledger.cash == pytest.approx(1_000.0)
    assert ledger.positions == {}


def test_invalid_accounting_inputs_are_rejected_without_mutation() -> None:
    ledger = PortfolioLedger(initial_cash=1_000.0)
    ledger.apply_fill("A", 1.0, 100.0)

    with pytest.raises(ValueError, match="missing prices"):
        ledger.equity({})
    with pytest.raises(ValueError, match="target weight"):
        ledger.target_orders({"A": float("nan")}, {"A": 100.0})
    with pytest.raises(ValueError, match="gross and net"):
        ledger.target_orders({"A": 1e308, "B": 1e308}, {"A": 100.0, "B": 100.0})
    with pytest.raises(ValueError, match="commission"):
        ledger.apply_fill("A", 1.0, 100.0, commission=-1.0)

    assert ledger.cash == pytest.approx(900.0)
    assert ledger.positions == {"A": 1.0}

    oversized = PortfolioLedger(initial_cash=1e308)
    oversized.apply_fill("A", 1e308, 1.0)
    with pytest.raises(ValueError, match="market value"):
        oversized.apply_fill("A", 0.0, 2.0)
    assert oversized.cash == pytest.approx(0.0)
    assert oversized.positions == {"A": 1e308}


@pytest.mark.parametrize("initial_cash", [0.0, -1.0, float("nan"), float("inf")])
def test_initial_cash_must_be_finite_and_positive(initial_cash: float) -> None:
    with pytest.raises(ValueError):
        PortfolioLedger(initial_cash=initial_cash)
