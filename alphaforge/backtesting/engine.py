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

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from alphaforge.backtesting.event_engine import DeterministicEventEngine
from alphaforge.backtesting.journal import Journal
from alphaforge.backtesting.ledger import LedgerSnapshot, reconciles
from alphaforge.data.schemas import to_wide, validate_panel
from alphaforge.execution.costs import CostModel
from alphaforge.execution.events import (
    MAX_TARGET_ASSETS,
    EngineHalted,
    EventCoordinate,
    EventPhase,
    ExecutionEvent,
    FeeCategory,
    FeeComponent,
    FillApplied,
    OrderAccepted,
    OrderCancelled,
    OrderRejected,
    OrderSubmitted,
    PortfolioMarked,
    SignalAvailable,
    TargetDecided,
)
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
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    accounting: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class _ScheduledDecision:
    decision_date: pd.Timestamp
    target_weights: Mapping[str, float]
    risk_scale: float
    target_event_id: str
    correlation_id: str


@dataclass(frozen=True, slots=True)
class _GeneratedOrder:
    order: Order
    fill: Fill
    submitted: ExecutionEvent


def _semantic_value(value: object) -> object:
    """Normalize supported configuration values for stable identity hashing."""

    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("identity inputs must not contain non-finite numbers")
        return {"float_hex": number.hex()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        timestamp = pd.Timestamp(value)
        return {"timestamp": timestamp.isoformat()}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("identity mappings must use string keys")
        return {key: _semantic_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_semantic_value(item) for item in value]
    raise TypeError(f"unsupported identity value: {type(value).__name__}")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        _semantic_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _stable_digest(value: object, *, domain: str) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\x00")
    digest.update(_canonical_bytes(value))
    return digest.hexdigest()


def _frame_digest(
    frame: pd.DataFrame,
    *,
    sort_by: tuple[str, ...],
    domain: str,
) -> str:
    """Hash a normalized frame without materializing one giant JSON document."""

    columns = tuple(sorted(str(column) for column in frame.columns))
    ordered = frame.sort_values(list(sort_by), kind="mergesort").loc[:, list(columns)]
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\x00")
    digest.update(_canonical_bytes(columns))
    for row in ordered.itertuples(index=False, name=None):
        normalized_row = tuple(
            {"missing": True} if bool(pd.isna(value)) else value for value in row
        )
        encoded = _canonical_bytes(normalized_row)
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _decisions_digest(decisions: Mapping[pd.Timestamp, Mapping[str, float]]) -> str:
    payload = [
        {
            "date": date,
            "weights": tuple(sorted(weights.items())),
        }
        for date, weights in sorted(decisions.items())
    ]
    return _stable_digest(payload, domain="alphaforge.backtest-decisions.v1")


def _execution_event(
    *,
    run_id: str,
    date: pd.Timestamp,
    bar_index: int,
    phase: EventPhase,
    ordinal: int,
    correlation_id: str,
    entity_id: str,
    payload: object,
    causation_id: str | None = None,
) -> ExecutionEvent:
    return ExecutionEvent(
        run_id=run_id,
        correlation_id=correlation_id,
        entity_id=entity_id,
        coordinate=EventCoordinate(
            session=date.date(),
            bar_index=bar_index,
            phase=phase,
            ordinal=ordinal,
        ),
        payload=payload,  # type: ignore[arg-type]
        causation_id=causation_id,
    )


def _mark_event(
    engine: DeterministicEventEngine,
    *,
    date: pd.Timestamp,
    bar_index: int,
    mark_type: str,
    prices: Mapping[str, float],
) -> tuple[ExecutionEvent, LedgerSnapshot]:
    snapshot = engine.value_portfolio(date.date(), prices)
    holdings = math.fsum(snapshot.market_values.values())
    mark_id = f"mark-{mark_type}-{bar_index:08d}"
    event = _execution_event(
        run_id=engine.run_id,
        date=date,
        bar_index=bar_index,
        phase=(EventPhase.OPEN_MARK if mark_type == "open" else EventPhase.CLOSE_MARK),
        ordinal=0,
        correlation_id=f"mark-{bar_index:08d}",
        entity_id=mark_id,
        payload=PortfolioMarked(
            mark_id=mark_id,
            mark_type=mark_type,  # type: ignore[arg-type]
            prices=tuple(sorted(prices.items())),
            cash=snapshot.cash,
            holdings_value=holdings,
            accrued_charges=snapshot.total_charges,
            equity=snapshot.equity,
        ),
    )
    engine.process(event)
    committed = engine.snapshot().portfolio
    if committed is None:
        raise RuntimeError("portfolio mark did not publish a reconciled snapshot")
    return event, committed


def _event_frame(events: tuple[ExecutionEvent, ...]) -> pd.DataFrame:
    columns = [
        "event_id",
        "run_id",
        "session",
        "bar_index",
        "phase",
        "ordinal",
        "event_type",
        "correlation_id",
        "entity_id",
        "causation_id",
    ]
    records = [
        {
            "event_id": event.event_id,
            "run_id": event.run_id,
            "session": pd.Timestamp(event.coordinate.session),
            "bar_index": event.coordinate.bar_index,
            "phase": event.coordinate.phase.name.lower(),
            "ordinal": event.coordinate.ordinal,
            "event_type": event.event_type,
            "correlation_id": event.correlation_id,
            "entity_id": event.entity_id,
            "causation_id": event.causation_id,
        }
        for event in events
    ]
    return pd.DataFrame(records, columns=columns)


def _record_control_halt(
    engine: DeterministicEventEngine,
    *,
    date: pd.Timestamp,
    bar_index: int,
    reason_code: str,
    detail: str,
    correlation_id: str | None = None,
    causation_id: str | None = None,
) -> ExecutionEvent:
    """Append one terminal, bounded control event and return it."""

    halt = _execution_event(
        run_id=engine.run_id,
        date=date,
        bar_index=bar_index,
        phase=EventPhase.CONTROL,
        ordinal=0,
        correlation_id=correlation_id or f"control-{bar_index:08d}",
        entity_id=f"engine-control-{bar_index:08d}",
        payload=EngineHalted(
            reason_code=reason_code,
            detail=detail,
        ),
        causation_id=causation_id,
    )
    engine.process(halt)
    return halt


def _halt_if_bankrupt(
    engine: DeterministicEventEngine,
    *,
    date: pd.Timestamp,
    bar_index: int,
    cause_event: ExecutionEvent,
    snapshot: LedgerSnapshot,
    source: str,
) -> None:
    """Record the required terminal control event for insolvent accounting."""

    if not snapshot.bankrupt:
        return
    _record_control_halt(
        engine,
        date=date,
        bar_index=bar_index,
        correlation_id=cause_event.correlation_id,
        reason_code="bankruptcy",
        detail=f"{source.capitalize()} produced non-positive portfolio equity.",
        causation_id=cause_event.event_id,
    )
    raise RuntimeError(f"backtest halted on non-positive {source} equity at {date.date()}")


def _accounting_record(date: pd.Timestamp, snapshot: LedgerSnapshot) -> dict[str, object]:
    return {
        "date": date,
        "cash": snapshot.cash,
        "equity": snapshot.equity,
        "gross_exposure": snapshot.gross_exposure,
        "net_exposure": snapshot.net_exposure,
        "realized_pnl": snapshot.realized_pnl,
        "unrealized_pnl": snapshot.unrealized_pnl,
        "fees": snapshot.charges["fees"],
        "financing": snapshot.charges["financing"],
        "borrow": snapshot.charges["borrow"],
        "other_charges": snapshot.charges["other"],
        "total_charges": snapshot.total_charges,
        "net_pnl": snapshot.net_pnl,
        "reconciliation_error": snapshot.reconciliation_error,
        "reconciliation_tolerance": snapshot.reconciliation_tolerance,
        "bankrupt": snapshot.bankrupt,
    }


def _require_reconciliation(
    observed: float,
    expected: float,
    *,
    operands: tuple[float, ...],
    message: str,
) -> None:
    if not reconciles(observed, expected, operands=operands):
        raise RuntimeError(message)


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
    event_journal: Journal | None = None,
    event_run_id: str | None = None,
) -> BacktestResult:
    """Run a chronological OOS backtest from close-time target weights.

    ``target_weights.date`` is the decision session.  A lag of one fills at
    the next session open; larger lags count trading sessions, not wall-clock
    days.  Orders are DAY orders, so any residual from a participation-capped
    partial fill expires and is visible in the fill audit table.

    ``event_journal`` must be empty; non-empty history is opened through
    :meth:`DeterministicEventEngine.replay`. The default in-memory journal has
    no filesystem or network side effect. A caller-supplied journal remains
    caller-owned and open. Canonical journal marks can contain licensed prices,
    while the returned ``events`` table contains metadata and causation only.
    ``event_run_id`` is optional; when omitted it is derived from canonical
    validated data, targets, and configuration without wall-clock state.
    """
    if not isinstance(execution_lag, int) or isinstance(execution_lag, bool) or execution_lag < 1:
        raise ValueError("execution_lag must be an integer >= 1")
    if (
        not isinstance(rebalance_frequency, int)
        or isinstance(rebalance_frequency, bool)
        or rebalance_frequency < 1
    ):
        raise ValueError("rebalance_frequency must be an integer >= 1")
    if not isinstance(liquidate_at_end, bool):
        raise ValueError("liquidate_at_end must be boolean")
    if (
        isinstance(initial_capital, bool)
        or not isinstance(initial_capital, (int, float, np.integer, np.floating))
        or not np.isfinite(initial_capital)
        or initial_capital <= 0
    ):
        raise ValueError("initial_capital must be finite and positive")

    clean_panel = validate_panel(panel, allow_na_volume=True)
    close = to_wide(clean_panel, "close")
    open_price = to_wide(clean_panel, "open").reindex(close.index)
    volume = to_wide(clean_panel, "volume").reindex(close.index)
    calendar = pd.DatetimeIndex(pd.to_datetime(close.index))
    if calendar.tz is not None:
        raise ValueError("backtest sessions must be timezone-naive daily timestamps")
    if not calendar.equals(calendar.normalize()):
        raise ValueError("backtest sessions must be normalized to midnight daily bars")
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
    if len(tradable_symbols) > MAX_TARGET_ASSETS:
        raise ValueError(
            f"event-backed backtests support at most {MAX_TARGET_ASSETS} target assets"
        )

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
    data_digest = _frame_digest(
        clean_panel,
        sort_by=("date", "symbol"),
        domain="alphaforge.backtest-panel.v1",
    )
    target_digest = _decisions_digest(decisions)
    configuration_digest = _stable_digest(
        {
            "benchmark_symbol": benchmark_symbol,
            "initial_capital": float(initial_capital),
            "execution_lag": execution_lag,
            "rebalance_frequency": rebalance_frequency,
            "liquidate_at_end": liquidate_at_end,
            "costs": asdict(execution_model.costs),
            "execution": asdict(execution_model.policy),
            "risk": risk_cfg,
        },
        domain="alphaforge.backtest-config.v1",
    )
    derived_identity = _stable_digest(
        {
            "data_digest": data_digest,
            "target_digest": target_digest,
            "configuration_digest": configuration_digest,
        },
        domain="alphaforge.event-run.v1",
    )
    resolved_run_id = event_run_id or f"backtest-{derived_identity[:32]}"
    event_engine = DeterministicEventEngine(
        resolved_run_id,
        calendar=tuple(timestamp.date() for timestamp in calendar),
        initial_cash=float(initial_capital),
        journal=event_journal,
    )

    scheduled: dict[pd.Timestamp, _ScheduledDecision] = {}
    calendar_position = {date: i for i, date in enumerate(calendar)}
    curve_records: list[dict[str, object]] = []
    position_records: list[dict[str, object]] = []
    order_records: list[dict[str, object]] = []
    fills: list[Fill] = []
    attribution_records: list[dict[str, object]] = []
    accounting_records: list[dict[str, object]] = []
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
        bar_index = calendar_position[date]
        old_positions = dict(event_engine.positions)
        scheduled_decision = scheduled.get(date)
        required_at_open = set(old_positions)
        if scheduled_decision is not None:
            required_at_open.update(scheduled_decision.target_weights)
        try:
            open_prices = _valid_price_map(
                open_price.loc[date],
                required_at_open,
                date=date,
                field_name="open",
            )
        except ValueError:
            _record_control_halt(
                event_engine,
                date=date,
                bar_index=bar_index,
                reason_code="missing_open_price",
                detail="A required open price was absent or invalid.",
            )
            raise
        open_event, open_snapshot = _mark_event(
            event_engine,
            date=date,
            bar_index=bar_index,
            mark_type="open",
            prices=open_prices,
        )
        _halt_if_bankrupt(
            event_engine,
            date=date,
            bar_index=bar_index,
            cause_event=open_event,
            snapshot=open_snapshot,
            source="open",
        )
        open_equity = open_snapshot.equity
        overnight_pnl = open_equity - previous_equity

        day_fills: list[Fill] = []
        if scheduled_decision is not None:
            pretrade_equity = open_equity
            quantities = event_engine.target_orders(
                scheduled_decision.target_weights,
                open_prices,
            )
            generated_orders: list[_GeneratedOrder] = []
            for submission_ordinal, (symbol, requested_shares) in enumerate(
                sorted(quantities.items())
            ):
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
                event_order_id = f"order-{order.order_id:08d}"
                submitted = _execution_event(
                    run_id=event_engine.run_id,
                    date=date,
                    bar_index=bar_index,
                    phase=EventPhase.ORDER_SUBMISSION,
                    ordinal=submission_ordinal,
                    correlation_id=scheduled_decision.correlation_id,
                    entity_id=event_order_id,
                    payload=OrderSubmitted(
                        order_id=event_order_id,
                        symbol=symbol,
                        side="buy" if requested_shares > 0 else "sell",
                        quantity=abs(float(requested_shares)),
                    ),
                    causation_id=scheduled_decision.target_event_id,
                )
                event_engine.process(submitted)
                generated_orders.append(_GeneratedOrder(order, fill, submitted))

            execution_ordinal = 0
            cancellations: list[tuple[_GeneratedOrder, ExecutionEvent]] = []
            for generated in generated_orders:
                order = generated.order
                fill = generated.fill
                event_order_id = f"order-{order.order_id:08d}"
                if fill.status == "rejected":
                    rejected = _execution_event(
                        run_id=event_engine.run_id,
                        date=date,
                        bar_index=bar_index,
                        phase=EventPhase.EXECUTION,
                        ordinal=execution_ordinal,
                        correlation_id=scheduled_decision.correlation_id,
                        entity_id=event_order_id,
                        payload=OrderRejected(
                            order_id=event_order_id,
                            reason_code="execution_model_rejected",
                        ),
                        causation_id=generated.submitted.event_id,
                    )
                    event_engine.process(rejected)
                    execution_ordinal += 1
                    continue

                accepted = _execution_event(
                    run_id=event_engine.run_id,
                    date=date,
                    bar_index=bar_index,
                    phase=EventPhase.EXECUTION,
                    ordinal=execution_ordinal,
                    correlation_id=scheduled_decision.correlation_id,
                    entity_id=event_order_id,
                    payload=OrderAccepted(
                        order_id=event_order_id,
                        accepted_quantity=abs(order.requested_shares),
                    ),
                    causation_id=generated.submitted.event_id,
                )
                event_engine.process(accepted)
                execution_ordinal += 1
                fee_components = (
                    (FeeComponent(FeeCategory.COMMISSION, fill.commission),)
                    if fill.commission > 0.0
                    else ()
                )
                fill_id = f"fill-{order.order_id:08d}-0001"
                fill_event = _execution_event(
                    run_id=event_engine.run_id,
                    date=date,
                    bar_index=bar_index,
                    phase=EventPhase.EXECUTION,
                    ordinal=execution_ordinal,
                    correlation_id=scheduled_decision.correlation_id,
                    entity_id=fill_id,
                    payload=FillApplied(
                        fill_id=fill_id,
                        order_id=event_order_id,
                        symbol=order.symbol,
                        side="buy" if fill.filled_shares > 0 else "sell",
                        quantity=abs(fill.filled_shares),
                        reference_price=fill.reference_price,
                        price=fill.fill_price,
                        fees=fee_components,
                    ),
                    causation_id=accepted.event_id,
                )
                event_engine.process(fill_event)
                fill_snapshot = event_engine.snapshot().portfolio
                if fill_snapshot is None:
                    raise RuntimeError("fill did not publish a reconciled portfolio snapshot")
                _halt_if_bankrupt(
                    event_engine,
                    date=date,
                    bar_index=bar_index,
                    cause_event=fill_event,
                    snapshot=fill_snapshot,
                    source="fill",
                )
                execution_ordinal += 1
                if fill.status == "partial":
                    cancellations.append((generated, fill_event))

            for cancellation_ordinal, (generated, fill_event) in enumerate(cancellations):
                cancelled = _execution_event(
                    run_id=event_engine.run_id,
                    date=date,
                    bar_index=bar_index,
                    phase=EventPhase.DAY_CANCEL,
                    ordinal=cancellation_ordinal,
                    correlation_id=scheduled_decision.correlation_id,
                    entity_id=f"order-{generated.order.order_id:08d}",
                    payload=OrderCancelled(
                        order_id=f"order-{generated.order.order_id:08d}",
                        reason_code="day_expired",
                        cancelled_quantity=abs(generated.fill.residual_shares),
                    ),
                    causation_id=fill_event.event_id,
                )
                event_engine.process(cancelled)
            active_risk_scale = scheduled_decision.risk_scale
            active_targets = dict(scheduled_decision.target_weights)

        post_trade_positions = dict(event_engine.positions)
        post_trade_open_prices = _valid_price_map(
            open_price.loc[date],
            set(post_trade_positions),
            date=date,
            field_name="open",
        )
        post_trade_open_equity = event_engine.value_portfolio(
            date.date(),
            post_trade_open_prices,
        ).equity
        day_cost = float(sum(fill.total_cost for fill in day_fills))
        _require_reconciliation(
            open_equity - post_trade_open_equity,
            day_cost,
            operands=(
                open_equity,
                post_trade_open_equity,
                day_cost,
                *(fill.total_cost for fill in day_fills),
            ),
            message=f"open-fill accounting did not reconcile on {date.date()}",
        )

        try:
            close_prices = _valid_price_map(
                close.loc[date],
                set(post_trade_positions),
                date=date,
                field_name="close",
            )
        except ValueError:
            _record_control_halt(
                event_engine,
                date=date,
                bar_index=bar_index,
                reason_code="missing_close_price",
                detail="A required close price was absent or invalid.",
            )
            raise
        close_event, snapshot = _mark_event(
            event_engine,
            date=date,
            bar_index=bar_index,
            mark_type="close",
            prices=close_prices,
        )
        _halt_if_bankrupt(
            event_engine,
            date=date,
            bar_index=bar_index,
            cause_event=close_event,
            snapshot=snapshot,
            source="close",
        )
        close_equity = snapshot.equity
        intraday_pnl = close_equity - post_trade_open_equity
        market_pnl = overnight_pnl + intraday_pnl
        net_pnl = close_equity - previous_equity
        _require_reconciliation(
            net_pnl,
            market_pnl - day_cost,
            operands=(
                close_equity,
                previous_equity,
                overnight_pnl,
                intraday_pnl,
                market_pnl,
                day_cost,
                net_pnl,
            ),
            message=f"daily P&L did not reconcile on {date.date()}",
        )
        accounting_records.append(_accounting_record(date, snapshot))

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
        if (
            target is not None
            and fill_idx < len(calendar)
            and (terminal_fill_date is None or date != terminal_fill_date)
        ):
            scale = _risk_scale(realized_returns, close_equities, risk_cfg)
            scaled_target = {symbol: float(weight) * scale for symbol, weight in target.items()}
            eligible_date = calendar[fill_idx]
            correlation_id = f"decision-{bar_index:08d}"
            signal_digest = _stable_digest(
                {
                    "decision_date": date,
                    "unscaled_target": tuple(sorted(target.items())),
                    "target_digest": target_digest,
                },
                domain="alphaforge.backtest-signal.v1",
            )
            signal_id = f"signal-{bar_index:08d}"
            signal_event = _execution_event(
                run_id=event_engine.run_id,
                date=date,
                bar_index=bar_index,
                phase=EventPhase.SIGNAL,
                ordinal=0,
                correlation_id=correlation_id,
                entity_id=signal_id,
                payload=SignalAvailable(
                    signal_id=signal_id,
                    model_id="caller-supplied-oos-targets",
                    signal_digest=signal_digest,
                ),
            )
            event_engine.process(signal_event)
            problem_digest = _stable_digest(
                {
                    "decision_date": date,
                    "eligible_date": eligible_date,
                    "scaled_target": tuple(sorted(scaled_target.items())),
                    "risk_scale": scale,
                },
                domain="alphaforge.backtest-target-problem.v1",
            )
            target_id = f"target-{bar_index:08d}"
            target_event = _execution_event(
                run_id=event_engine.run_id,
                date=date,
                bar_index=bar_index,
                phase=EventPhase.TARGET_DECISION,
                ordinal=0,
                correlation_id=correlation_id,
                entity_id=target_id,
                payload=TargetDecided(
                    target_id=target_id,
                    portfolio_id="caller-target-weights-v1",
                    solver_id="deterministic-risk-scaling-v1",
                    eligible_session=eligible_date.date(),
                    cash_weight=1.0 - math.fsum(scaled_target.values()),
                    weights=tuple(sorted(scaled_target.items())),
                    configuration_digest=configuration_digest,
                    data_digest=data_digest,
                    problem_digest=problem_digest,
                ),
                causation_id=signal_event.event_id,
            )
            event_engine.process(target_event)
            scheduled[eligible_date] = _ScheduledDecision(
                decision_date=date,
                target_weights=scaled_target,
                risk_scale=scale,
                target_event_id=target_event.event_id,
                correlation_id=correlation_id,
            )

        if terminal_fill_date is not None and date == terminal_fill_date:
            break

    curve = pd.DataFrame(curve_records)
    weights = pd.DataFrame(position_records)
    orders = _orders_frame(order_records)
    fill_frame = _fills_frame(fills)
    trades = _trades_frame(fill_frame)
    attribution = pd.DataFrame(attribution_records)
    accounting_columns = [
        "date",
        "cash",
        "equity",
        "gross_exposure",
        "net_exposure",
        "realized_pnl",
        "unrealized_pnl",
        "fees",
        "financing",
        "borrow",
        "other_charges",
        "total_charges",
        "net_pnl",
        "reconciliation_error",
        "reconciliation_tolerance",
        "bankrupt",
    ]
    accounting = pd.DataFrame(accounting_records, columns=accounting_columns)

    if not attribution.empty:
        daily_attribution = attribution.groupby("date", sort=True)[
            ["market_pnl", "trading_cost", "net_pnl"]
        ].sum()
        curve_check = curve.set_index("date")[["market_pnl", "trading_cost"]]
        curve_check = curve_check.assign(
            net_pnl=curve_check["market_pnl"] - curve_check["trading_cost"]
        )
        common = daily_attribution.index.intersection(curve_check.index)
        indexed_curve = curve.set_index("date")
        prior_equity = indexed_curve["equity"].shift(1).fillna(initial_capital)
        for common_date in common:
            day_rows = attribution.loc[attribution["date"] == common_date]
            current_equity = float(indexed_curve.at[common_date, "equity"])
            previous_day_equity = float(prior_equity.at[common_date])
            current_cash = float(indexed_curve.at[common_date, "cash"])
            for column in ("market_pnl", "trading_cost", "net_pnl"):
                observed = float(daily_attribution.at[common_date, column])
                expected = float(curve_check.at[common_date, column])
                _require_reconciliation(
                    observed,
                    expected,
                    operands=(
                        *(float(value) for value in day_rows[column]),
                        observed,
                        expected,
                        current_equity,
                        previous_day_equity,
                        current_cash,
                    ),
                    message="symbol-level P&L attribution did not reconcile",
                )

    events = _event_frame(event_engine.journal.events())
    event_engine.close()

    return BacktestResult(
        equity_curve=curve,
        weights=weights,
        trades=trades,
        orders=orders,
        fills=fill_frame,
        pnl_attribution=attribution,
        events=events,
        accounting=accounting,
    )
