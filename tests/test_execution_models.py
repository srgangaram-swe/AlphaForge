from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.execution.costs import CostModel
from alphaforge.execution.models import BarExecutionModel, ExecutionPolicy, Order


def _order(shares: float = 1_000.0) -> Order:
    return Order(
        order_id=1,
        symbol="AAA",
        decision_date=pd.Timestamp("2024-01-02"),
        fill_date=pd.Timestamp("2024-01-03"),
        requested_shares=shares,
        target_weight=0.5,
        pretrade_equity=1_000_000.0,
    )


def test_buy_fill_reconciles_explicit_cost_components() -> None:
    model = BarExecutionModel(
        CostModel(commission_bps=1.0, half_spread_bps=2.0, slippage_bps=3.0),
        ExecutionPolicy(impact_coefficient=0.10),
    )
    fill = model.execute(
        _order(),
        reference_price=100.0,
        lagged_adv_shares=100_000.0,
        lagged_volatility=0.02,
    )

    assert fill.status == "filled"
    assert fill.participation_rate == pytest.approx(0.01)
    assert fill.impact_bps == pytest.approx(2.0)
    assert fill.fill_price == pytest.approx(100.07)
    assert fill.total_cost == pytest.approx(80.0)
    assert fill.traded_notional == pytest.approx(100_000.0)


def test_sell_receives_a_price_below_the_reference() -> None:
    model = BarExecutionModel(
        CostModel(commission_bps=0.0, half_spread_bps=2.0, slippage_bps=3.0),
        ExecutionPolicy(),
    )
    fill = model.execute(_order(-100.0), reference_price=50.0)

    assert fill.filled_shares == -100.0
    assert fill.fill_price == pytest.approx(49.975)
    assert fill.total_cost == pytest.approx(2.5)


def test_lagged_adv_participation_cap_produces_a_partial_fill() -> None:
    model = BarExecutionModel(
        CostModel(commission_bps=0.0, half_spread_bps=0.0, slippage_bps=0.0),
        ExecutionPolicy(max_participation_rate=0.05),
    )
    fill = model.execute(
        _order(10_000.0),
        reference_price=20.0,
        lagged_adv_shares=100_000.0,
    )

    assert fill.status == "partial"
    assert fill.filled_shares == pytest.approx(5_000.0)
    assert fill.residual_shares == pytest.approx(5_000.0)
    assert fill.participation_rate == pytest.approx(0.05)


def test_missing_adv_rejects_when_participation_is_enforced() -> None:
    model = BarExecutionModel(
        CostModel(),
        ExecutionPolicy(max_participation_rate=0.05),
    )
    fill = model.execute(_order(), reference_price=20.0)
    assert fill.status == "rejected"
    assert fill.filled_shares == 0.0
    assert fill.total_cost == 0.0


def test_execution_config_rejects_unknown_or_unsafe_settings() -> None:
    with pytest.raises(ValueError, match="unknown execution settings"):
        ExecutionPolicy.from_config({"lookahead_volume": True})
    with pytest.raises(ValueError, match="max_participation_rate"):
        ExecutionPolicy(max_participation_rate=1.1)
    with pytest.raises(ValueError, match="non-negative"):
        CostModel(slippage_bps=-1.0)
    with pytest.raises(ValueError, match="unknown transaction-cost settings"):
        CostModel.from_config({"free_money_bps": 10.0})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"price_field": "close"}, "next-open"),
        ({"adv_lookback": 0}, "lookbacks"),
        ({"volatility_lookback": 1}, "lookbacks"),
        ({"impact_coefficient": -0.1}, "impact_coefficient"),
        ({"missing_price_policy": "invent"}, "missing_price_policy"),
    ],
)
def test_execution_policy_validation(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ExecutionPolicy(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"order_id": 0}, "order_id"),
        ({"symbol": ""}, "symbol"),
        ({"fill_date": pd.Timestamp("2024-01-02")}, "strictly after"),
        ({"requested_shares": float("nan")}, "finite"),
        ({"pretrade_equity": 0.0}, "positive"),
    ],
)
def test_order_contract_validation(overrides, message: str) -> None:
    values = {
        "order_id": 1,
        "symbol": "AAA",
        "decision_date": pd.Timestamp("2024-01-02"),
        "fill_date": pd.Timestamp("2024-01-03"),
        "requested_shares": 100.0,
        "target_weight": 0.5,
        "pretrade_equity": 1_000_000.0,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        Order(**values)


def test_missing_price_policy_is_explicit() -> None:
    strict = BarExecutionModel(CostModel(), ExecutionPolicy(missing_price_policy="raise"))
    with pytest.raises(ValueError, match="invalid open price"):
        strict.execute(_order(), reference_price=float("nan"))

    permissive = BarExecutionModel(CostModel(), ExecutionPolicy(missing_price_policy="skip"))
    rejected = permissive.execute(_order(), reference_price=float("nan"))
    assert rejected.status == "rejected"
    assert rejected.residual_shares == rejected.requested_shares


def test_factory_accepts_typed_configs_and_zero_order_is_rejected() -> None:
    costs = CostModel()
    policy = ExecutionPolicy()
    model = BarExecutionModel.from_config(costs=costs, execution=policy)
    fill = model.execute(_order(0.0), reference_price=100.0)
    assert fill.status == "rejected"
