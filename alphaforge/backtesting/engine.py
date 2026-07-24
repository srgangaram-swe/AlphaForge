"""Chronological, self-financing daily-bar backtest engine.

Timeline for ``execution_lag=1``:

1. positions held from the prior close own the overnight close-to-open move;
2. a target decided at close(t-1) becomes a day order at open(t);
3. fills change signed shares and cash through a self-financing ledger;
4. post-fill holdings own the open-to-close move; and
5. the portfolio is marked at close(t), when a new target may be decided.

This ordering prevents a close-time signal from capturing an overnight gap
that occurred before its fill.  Shares, rather than target weights, persist
between rebalances, so weights drift naturally and restoring a target creates
an observable, costed trade.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphaforge.backtesting.ledger import PortfolioLedger
from alphaforge.data.schemas import to_wide, validate_panel
from alphaforge.execution.costs import CostModel
from alphaforge.execution.models import (
    BarExecutionModel,
    ExecutionPolicy,
    Fill,
    Order,
)
from alphaforge.utils import ANNUALIZATION_DAYS


@dataclass
class BacktestResult:
    """Auditable backtest artifacts.

    The first three fields preserve the original public contract.  Additional
    tables expose the execution and accounting trail needed to reconcile P&L.
    """

    equity_curve: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame
    orders: pd.DataFrame = field(default_factory=pd.DataFrame)
    fills: pd.DataFrame = field(default_factory=pd.DataFrame)
    pnl_attribution: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class _ScheduledDecision:
    decision_date: pd.Timestamp
    target_weights: Mapping[str, float]
    risk_scale: float


def _decision_targets(
    target_weights: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    market_symbols: set[str],
    rebalance_frequency: int,
) -> dict[pd.Timestamp, dict[str, float]]:
    required = {"date", "symbol", "target_weight"}
    missing = required - set(target_weights)
    if missing:
        raise ValueError(f"target_weights missing columns: {sorted(missing)}")
    if rebalance_frequency < 1:
        raise ValueError("rebalance_frequency must be >= 1")

    frame = target_weights[["date", "symbol", "target_weight"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["symbol"] = frame["symbol"].astype(str)
    frame["target_weight"] = pd.to_numeric(frame["target_weight"], errors="coerce")
    if frame.empty:
        raise ValueError("target_weights must contain at least one row")
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("target_weights contains duplicate (date, symbol) rows")
    if not np.isfinite(frame["target_weight"]).all():
        raise ValueError("target weights must be finite")

    unknown_symbols = set(frame["symbol"]) - market_symbols
    if unknown_symbols:
        raise ValueError(f"target symbols missing from panel: {sorted(unknown_symbols)}")
    unknown_dates = set(frame["date"]) - set(calendar)
    if unknown_dates:
        formatted = sorted(pd.Timestamp(date).date().isoformat() for date in unknown_dates)
        raise ValueError(f"target dates are not trading sessions: {formatted[:5]}")

    all_symbols = sorted(frame["symbol"].unique())
    decision_dates = sorted(frame["date"].unique())[::rebalance_frequency]
    decisions: dict[pd.Timestamp, dict[str, float]] = {}
    for date in decision_dates:
        block = frame.loc[frame["date"] == date].set_index("symbol")["target_weight"]
        # A target row is a complete portfolio snapshot: an omitted symbol has
        # target weight zero.  This makes liquidations explicit and prevents
        # stale positions from surviving a sparse pivot/forward-fill.
        dense = block.reindex(all_symbols, fill_value=0.0).astype(float)
        decisions[pd.Timestamp(date)] = dense.to_dict()
    return decisions


def _lagged_execution_inputs(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    policy: ExecutionPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return open-time ADV and volatility using data through prior closes."""
    lagged_adv = (
        volume.rolling(policy.adv_lookback, min_periods=policy.adv_lookback).mean().shift(1)
    )
    daily_returns = close.pct_change(fill_method=None)
    lagged_volatility = (
        daily_returns.rolling(
            policy.volatility_lookback,
            min_periods=policy.volatility_lookback,
        )
        .std(ddof=1)
        .shift(1)
    )
    return lagged_adv, lagged_volatility


def _valid_price_map(
    prices: pd.Series,
    symbols: set[str],
    *,
    date: pd.Timestamp,
    field_name: str,
) -> dict[str, float]:
    out: dict[str, float] = {}
    invalid: list[str] = []
    for symbol in sorted(symbols):
        value = prices.get(symbol, np.nan)
        if not np.isfinite(value) or float(value) <= 0:
            invalid.append(symbol)
        else:
            out[symbol] = float(value)
    if invalid:
        raise ValueError(
            f"missing or invalid {field_name} prices on {date.date()}: {sorted(invalid)}"
        )
    return out


def _risk_scale(
    realized_returns: list[float],
    close_equities: list[float],
    risk: Mapping[str, object],
) -> float:
    """Causal exposure multiplier frozen when a close-time target is decided."""
    scale = 1.0
    vol_target_raw = risk.get("vol_target")
    if vol_target_raw is not None:
        vol_target = _finite_setting(vol_target_raw, "vol_target")
        lookback = _integer_setting(risk.get("vol_lookback", 20), "vol_lookback")
        max_leverage = _finite_setting(risk.get("max_leverage", 1.5), "max_leverage")
        if vol_target <= 0 or lookback < 2 or max_leverage <= 0:
            raise ValueError("volatility-target settings must be positive")
        sample = np.asarray(realized_returns[-lookback:], dtype=float)
        if sample.size >= 2:
            realized_vol = float(np.std(sample, ddof=1) * np.sqrt(ANNUALIZATION_DAYS))
            if np.isfinite(realized_vol) and realized_vol > 0:
                scale = min(vol_target / realized_vol, max_leverage)

    threshold_raw = risk.get("drawdown_deleverage")
    if threshold_raw is not None and close_equities:
        threshold = abs(_finite_setting(threshold_raw, "drawdown_deleverage"))
        cut = _finite_setting(risk.get("drawdown_cut", 0.5), "drawdown_cut")
        if threshold <= 0 or not 0 <= cut <= 1:
            raise ValueError("drawdown threshold must be positive and cut must be in [0, 1]")
        current = close_equities[-1]
        drawdown = current / max(close_equities) - 1.0
        if drawdown < -threshold:
            scale *= cut
    return float(scale)


def _finite_setting(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer_setting(value: object, name: str) -> int:
    numeric = _finite_setting(value, name)
    if not numeric.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(numeric)


def _orders_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "order_id",
        "symbol",
        "decision_date",
        "fill_date",
        "requested_shares",
        "requested_notional",
        "target_weight",
        "pretrade_equity",
    ]
    return pd.DataFrame(records, columns=columns)


def _fills_frame(fills: list[Fill]) -> pd.DataFrame:
    records = []
    for fill in fills:
        record = fill.to_record()
        record["requested_notional"] = abs(fill.requested_shares) * fill.reference_price
        record["lagged_adv_notional"] = (
            fill.lagged_adv_shares * fill.reference_price
            if np.isfinite(fill.lagged_adv_shares)
            else np.nan
        )
        records.append(record)
    return pd.DataFrame(records)


def _trades_frame(fills: pd.DataFrame) -> pd.DataFrame:
    if fills.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "trade_weight",
                "filled_shares",
                "reference_price",
                "fill_price",
                "traded_notional",
                "total_cost",
            ]
        )
    traded = fills.loc[fills["filled_shares"] != 0].copy()
    traded["date"] = pd.to_datetime(traded["fill_date"])
    traded["trade_weight"] = (
        traded["filled_shares"] * traded["reference_price"] / traded["pretrade_equity"]
    )
    columns = [
        "date",
        "symbol",
        "trade_weight",
        "filled_shares",
        "reference_price",
        "fill_price",
        "traded_notional",
        "total_cost",
        "decision_date",
        "status",
        "residual_shares",
        "participation_rate",
    ]
    return traded[columns].sort_values(["date", "symbol"]).reset_index(drop=True)


def run_backtest(
    panel: pd.DataFrame,
    target_weights: pd.DataFrame,
    benchmark_symbol: str | None = None,
    initial_capital: float = 1_000_000.0,
    execution_lag: int = 1,
    rebalance_frequency: int = 1,
    costs: CostModel | dict | None = None,
    risk: dict | None = None,
    execution: ExecutionPolicy | dict | None = None,
    liquidate_at_end: bool = False,
) -> BacktestResult:
    """Run a chronological OOS backtest from close-time target weights.

    ``target_weights.date`` is the decision session.  A lag of one fills at
    the next session open; larger lags count trading sessions, not wall-clock
    days.  Orders are DAY orders, so any residual from a participation-capped
    partial fill expires and is visible in the fill audit table.
    """
    if not isinstance(execution_lag, int) or execution_lag < 1:
        raise ValueError("execution_lag must be an integer >= 1")
    if not isinstance(rebalance_frequency, int) or rebalance_frequency < 1:
        raise ValueError("rebalance_frequency must be an integer >= 1")
    if not isinstance(liquidate_at_end, bool):
        raise ValueError("liquidate_at_end must be boolean")
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be finite and positive")

    clean_panel = validate_panel(panel, allow_na_volume=True)
    close = to_wide(clean_panel, "close")
    open_price = to_wide(clean_panel, "open").reindex(close.index)
    volume = to_wide(clean_panel, "volume").reindex(close.index)
    calendar = pd.DatetimeIndex(pd.to_datetime(close.index))
    close.index = open_price.index = volume.index = calendar

    decisions = _decision_targets(
        target_weights,
        calendar,
        set(close.columns),
        rebalance_frequency,
    )
    tradable_symbols = sorted({symbol for target in decisions.values() for symbol in target})
    if not tradable_symbols:
        raise ValueError("target weights do not overlap panel symbols")

    terminal_fill_date: pd.Timestamp | None = None
    if liquidate_at_end:
        last_decision = max(decisions)
        last_decision_idx = int(calendar.get_indexer([last_decision])[0])
        liquidation_decision_idx = last_decision_idx + rebalance_frequency
        terminal_fill_idx = liquidation_decision_idx + execution_lag
        if terminal_fill_idx >= len(calendar):
            raise ValueError(
                "panel needs enough sessions after the final target to execute terminal liquidation"
            )
        liquidation_date = calendar[liquidation_decision_idx]
        decisions[liquidation_date] = {symbol: 0.0 for symbol in tradable_symbols}
        terminal_fill_date = calendar[terminal_fill_idx]
    close = close.reindex(columns=tradable_symbols)
    open_price = open_price.reindex(columns=tradable_symbols)
    volume = volume.reindex(columns=tradable_symbols)

    execution_model = BarExecutionModel.from_config(costs=costs, execution=execution)
    if execution_model.policy.missing_price_policy != "raise":
        raise ValueError(
            "historical ledger requires missing_price_policy='raise' for auditable marking"
        )
    lagged_adv, lagged_volatility = _lagged_execution_inputs(
        close,
        volume,
        execution_model.policy,
    )
    risk_cfg: Mapping[str, object] = risk or {}
    ledger = PortfolioLedger(float(initial_capital))

    scheduled: dict[pd.Timestamp, _ScheduledDecision] = {}
    calendar_position = {date: i for i, date in enumerate(calendar)}
    curve_records: list[dict[str, object]] = []
    position_records: list[dict[str, object]] = []
    order_records: list[dict[str, object]] = []
    fills: list[Fill] = []
    attribution_records: list[dict[str, object]] = []
    realized_returns: list[float] = []
    close_equities: list[float] = []
    previous_close_prices: dict[str, float] = {}
    previous_equity = float(initial_capital)
    order_id = 0
    active_risk_scale = 1.0
    active_targets = {symbol: 0.0 for symbol in tradable_symbols}

    benchmark_close = (
        to_wide(clean_panel, "close")[benchmark_symbol].reindex(calendar)
        if benchmark_symbol is not None and benchmark_symbol in set(clean_panel["symbol"])
        else None
    )

    for date in calendar:
        old_positions = dict(ledger.positions)
        scheduled_decision = scheduled.get(date)
        required_at_open = set(old_positions)
        if scheduled_decision is not None:
            required_at_open.update(scheduled_decision.target_weights)
        open_prices = _valid_price_map(
            open_price.loc[date],
            required_at_open,
            date=date,
            field_name="open",
        )
        open_equity = ledger.equity(open_prices)
        overnight_pnl = open_equity - previous_equity

        day_fills: list[Fill] = []
        if scheduled_decision is not None:
            pretrade_equity = open_equity
            quantities = ledger.target_orders(
                scheduled_decision.target_weights,
                open_prices,
            )
            for symbol, requested_shares in quantities.items():
                order_id += 1
                order = Order(
                    order_id=order_id,
                    symbol=symbol,
                    decision_date=scheduled_decision.decision_date,
                    fill_date=date,
                    requested_shares=float(requested_shares),
                    target_weight=float(scheduled_decision.target_weights.get(symbol, 0.0)),
                    pretrade_equity=float(pretrade_equity),
                )
                order_record = order.to_record()
                order_record["requested_notional"] = (
                    abs(order.requested_shares) * open_prices[symbol]
                )
                order_records.append(order_record)
                fill = execution_model.execute(
                    order,
                    reference_price=open_prices[symbol],
                    lagged_adv_shares=float(lagged_adv.at[date, symbol]),
                    lagged_volatility=float(lagged_volatility.at[date, symbol]),
                )
                day_fills.append(fill)
                fills.append(fill)
                if fill.filled_shares != 0:
                    ledger.apply_fill(
                        symbol,
                        fill.filled_shares,
                        fill.fill_price,
                        fill.commission,
                    )
            active_risk_scale = scheduled_decision.risk_scale
            active_targets = dict(scheduled_decision.target_weights)

        post_trade_positions = dict(ledger.positions)
        post_trade_open_prices = _valid_price_map(
            open_price.loc[date],
            set(post_trade_positions),
            date=date,
            field_name="open",
        )
        post_trade_open_equity = ledger.equity(post_trade_open_prices)
        day_cost = float(sum(fill.total_cost for fill in day_fills))
        if not np.isclose(
            open_equity - post_trade_open_equity,
            day_cost,
            rtol=1e-10,
            atol=1e-6,
        ):
            raise RuntimeError(f"open-fill accounting did not reconcile on {date.date()}")

        close_prices = _valid_price_map(
            close.loc[date],
            set(post_trade_positions),
            date=date,
            field_name="close",
        )
        snapshot = ledger.snapshot(date, close_prices)
        close_equity = snapshot.equity
        intraday_pnl = close_equity - post_trade_open_equity
        market_pnl = overnight_pnl + intraday_pnl
        net_pnl = close_equity - previous_equity
        if not np.isclose(net_pnl, market_pnl - day_cost, rtol=1e-10, atol=1e-6):
            raise RuntimeError(f"daily P&L did not reconcile on {date.date()}")

        traded_notional = float(sum(fill.traded_notional for fill in day_fills))
        gross_return = market_pnl / previous_equity
        net_return = net_pnl / previous_equity
        transaction_cost = day_cost / previous_equity
        gross_exposure = float(sum(abs(value) for value in snapshot.market_values.values()))
        net_exposure = float(sum(snapshot.market_values.values()))
        gross_exposure_weight = gross_exposure / close_equity
        net_exposure_weight = net_exposure / close_equity
        turnover = traded_notional / open_equity if open_equity > 0 else 0.0
        benchmark_return = 0.0
        idx = calendar_position[date]
        if benchmark_close is not None and idx > 0:
            prior_benchmark = float(benchmark_close.iloc[idx - 1])
            current_benchmark = float(benchmark_close.iloc[idx])
            if (
                np.isfinite(prior_benchmark)
                and prior_benchmark > 0
                and np.isfinite(current_benchmark)
            ):
                benchmark_return = current_benchmark / prior_benchmark - 1.0

        curve_records.append(
            {
                "date": date,
                "gross_return": gross_return,
                "transaction_cost": transaction_cost,
                "return": net_return,
                "equity": close_equity,
                "cash": snapshot.cash,
                "market_pnl": market_pnl,
                "overnight_pnl": overnight_pnl,
                "intraday_pnl": intraday_pnl,
                "trading_cost": day_cost,
                "benchmark_return": benchmark_return,
                "turnover": turnover,
                "traded_notional": traded_notional,
                "gross_exposure": gross_exposure_weight,
                "net_exposure": net_exposure_weight,
                "leverage": active_risk_scale,
                "active": gross_exposure > 0,
            }
        )

        for symbol in tradable_symbols:
            quantity = float(snapshot.positions.get(symbol, 0.0))
            mark_price = float(close.at[date, symbol])
            market_value = quantity * mark_price if np.isfinite(mark_price) else 0.0
            position_records.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "shares": quantity,
                    "mark_price": mark_price,
                    "market_value": market_value,
                    "weight": float(snapshot.weights.get(symbol, 0.0)),
                    "target_weight": float(active_targets.get(symbol, 0.0)),
                }
            )

        costs_by_symbol: defaultdict[str, float] = defaultdict(float)
        for fill in day_fills:
            costs_by_symbol[fill.symbol] += fill.total_cost
        attribution_symbols = set(old_positions) | set(post_trade_positions) | set(costs_by_symbol)
        for symbol in sorted(attribution_symbols):
            prior_close = previous_close_prices.get(symbol, float(open_price.at[date, symbol]))
            open_value = float(open_price.at[date, symbol])
            close_value = float(close.at[date, symbol])
            symbol_overnight = old_positions.get(symbol, 0.0) * (open_value - prior_close)
            symbol_intraday = post_trade_positions.get(symbol, 0.0) * (close_value - open_value)
            symbol_cost = costs_by_symbol[symbol]
            attribution_records.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "overnight_pnl": symbol_overnight,
                    "intraday_pnl": symbol_intraday,
                    "market_pnl": symbol_overnight + symbol_intraday,
                    "trading_cost": symbol_cost,
                    "net_pnl": symbol_overnight + symbol_intraday - symbol_cost,
                }
            )

        realized_returns.append(float(net_return))
        close_equities.append(float(close_equity))
        previous_equity = float(close_equity)
        previous_close_prices = {
            symbol: float(close.at[date, symbol])
            for symbol in tradable_symbols
            if np.isfinite(close.at[date, symbol]) and float(close.at[date, symbol]) > 0
        }

        target = decisions.get(date)
        fill_idx = calendar_position[date] + execution_lag
        if target is not None and fill_idx < len(calendar):
            scale = _risk_scale(realized_returns, close_equities, risk_cfg)
            scaled_target = {symbol: float(weight) * scale for symbol, weight in target.items()}
            scheduled[calendar[fill_idx]] = _ScheduledDecision(date, scaled_target, scale)

        if terminal_fill_date is not None and date == terminal_fill_date:
            break

    curve = pd.DataFrame(curve_records)
    weights = pd.DataFrame(position_records)
    orders = _orders_frame(order_records)
    fill_frame = _fills_frame(fills)
    trades = _trades_frame(fill_frame)
    attribution = pd.DataFrame(attribution_records)

    if not attribution.empty:
        daily_attribution = attribution.groupby("date", sort=True)[
            ["market_pnl", "trading_cost", "net_pnl"]
        ].sum()
        curve_check = curve.set_index("date")[["market_pnl", "trading_cost"]]
        curve_check = curve_check.assign(
            net_pnl=curve.set_index("date")["return"]
            * curve.set_index("date")["equity"].shift(1).fillna(initial_capital)
        )
        common = daily_attribution.index.intersection(curve_check.index)
        if not np.allclose(
            daily_attribution.loc[common].to_numpy(),
            curve_check.loc[common].to_numpy(),
            rtol=1e-10,
            atol=1e-6,
        ):
            raise RuntimeError("symbol-level P&L attribution did not reconcile")

    return BacktestResult(
        equity_curve=curve,
        weights=weights,
        trades=trades,
        orders=orders,
        fills=fill_frame,
        pnl_attribution=attribution,
    )
