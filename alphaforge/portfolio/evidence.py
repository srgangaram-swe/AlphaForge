"""Net-of-cost allocation comparison across folds, regimes, and capacity (SF-S4-MR1).

An allocation policy that looks best gross, on one window, at one capital level,
has demonstrated nothing. This harness runs every policy over the same
chronological folds, splits each fold by regime, and repeats the whole grid at
several capital levels — because a policy's ranking routinely inverts as capital
grows and liquidity caps start to bind.

Three disciplines, each of which exists to prevent a specific self-deception:

**No holdout selection.** :func:`compare_allocation_policies` runs on
development folds only and takes the final holdout as a *separate, later*
argument that is scored once. There is no code path that lets holdout results
influence a parameter, because the function that chooses never sees them.

**Costs before comparison.** Turnover is charged at a declared rate on every
rebalance. A policy that rebalances hard can look better gross and worse net,
and gross-only comparison systematically favours exactly the policies that will
not survive execution.

**Capacity is a dimension, not a footnote.** The same grid runs at each capital
level with liquidity caps recomputed, so the level at which a policy stops
scaling is visible rather than discovered later.

Nothing here selects a winner. It produces the comparison; the rejection rule
belongs to the frozen qualification decision (SF-S4-MR9).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from alphaforge.portfolio.allocation import (
    apply_uncertainty_sizing,
    inverse_volatility_portfolio,
    long_short_spread_portfolio,
    rank_weighted_portfolio,
    score_weighted_portfolio,
    top_k_portfolio,
)
from alphaforge.portfolio.contracts import (
    AllocationResult,
    InfeasibleConstraintsError,
    PortfolioConstraints,
    PortfolioError,
    liquidity_caps_from_adv,
)

#: Bars per year for annualization, matching the rest of the platform.
TRADING_DAYS = 252

#: Refusal thresholds, not tuning knobs.
MAX_FOLDS = 50
MAX_CAPITAL_LEVELS = 10

AllocationFn = Callable[..., AllocationResult]


@dataclass(frozen=True)
class BacktestPanel:
    """Aligned inputs for one allocation backtest.

    Attributes:
        scores: ``(date, symbol)`` predicted scores, wide.
        forward_returns: ``(date, symbol)`` return realized **after** the score
            date. Supplied pre-aligned and asserted, rather than shifted here,
            so the alignment is the caller's explicit statement.
        volatility: ``(date, symbol)`` trailing volatility for sizing.
        adv: ``(date, symbol)`` average daily volume in currency, for liquidity.
    """

    scores: pd.DataFrame
    forward_returns: pd.DataFrame
    volatility: pd.DataFrame
    adv: pd.DataFrame

    def __post_init__(self) -> None:
        frames = {
            "forward_returns": self.forward_returns,
            "volatility": self.volatility,
            "adv": self.adv,
        }
        for name, frame in frames.items():
            if not frame.index.equals(self.scores.index):
                raise PortfolioError(f"{name} index must match the score index")
            if list(frame.columns) != list(self.scores.columns):
                raise PortfolioError(f"{name} columns must match the score columns")
        if not self.scores.index.is_monotonic_increasing:
            raise PortfolioError("panel dates must be sorted ascending")


@dataclass(frozen=True)
class Fold:
    """One chronological evaluation window."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise PortfolioError(f"fold {self.name!r} ends before it starts")


def chronological_folds(index: pd.Index, *, n_folds: int) -> tuple[Fold, ...]:
    """Split an index into contiguous, non-overlapping chronological folds.

    Contiguous and ordered rather than shuffled: a random split of a time series
    lets a fold be trained around by its own future neighbours, which is the
    standard way a portfolio backtest acquires lookahead.
    """
    if not 1 <= n_folds <= MAX_FOLDS:
        raise PortfolioError(f"n_folds must be in [1, {MAX_FOLDS}]")
    if len(index) < n_folds:
        raise PortfolioError("index is shorter than the requested fold count")
    bounds = np.array_split(np.arange(len(index)), n_folds)
    return tuple(
        Fold(name=f"fold_{position + 1}", start=index[block[0]], end=index[block[-1]])
        for position, block in enumerate(bounds)
        if block.size
    )


def volatility_regimes(
    returns: pd.DataFrame, *, window: int = 63, quantile: float = 0.7
) -> pd.Series:
    """Label each date ``calm`` or ``stress`` by trailing cross-sectional volatility.

    The threshold is a quantile of the **trailing** series only, so a date's
    label never depends on volatility that had not happened yet. Warm-up dates
    are labelled ``unknown`` rather than guessed.
    """
    if window < 2:
        raise PortfolioError("regime window must be at least two observations")
    dispersion = returns.mean(axis=1).rolling(window, min_periods=window).std()
    labels = pd.Series("unknown", index=returns.index, dtype=object)
    expanding_threshold = dispersion.expanding(min_periods=window).quantile(quantile)
    observed = dispersion.notna() & expanding_threshold.notna()
    labels[observed] = np.where(
        dispersion[observed] > expanding_threshold[observed], "stress", "calm"
    )
    return labels.rename("regime")


def _default_policies() -> dict[str, AllocationFn]:
    """Return the policy grid, each pinned to its declared parameters."""
    return {
        "top_k_equal": lambda scores, vol, constraints, previous, caps: top_k_portfolio(
            scores,
            constraints,
            k=max(len(scores) // 5, 1),
            weighting="equal",
            previous=previous,
            liquidity_caps=caps,
        ),
        "top_k_rank": lambda scores, vol, constraints, previous, caps: top_k_portfolio(
            scores,
            constraints,
            k=max(len(scores) // 5, 1),
            weighting="rank",
            previous=previous,
            liquidity_caps=caps,
        ),
        "long_short_spread": lambda scores, vol, constraints, previous, caps: (
            long_short_spread_portfolio(
                scores,
                constraints,
                k=max(len(scores) // 5, 1),
                previous=previous,
                liquidity_caps=caps,
            )
        ),
        "score_weighted": lambda scores, vol, constraints, previous, caps: (
            score_weighted_portfolio(scores, constraints, previous=previous, liquidity_caps=caps)
        ),
        "rank_weighted": lambda scores, vol, constraints, previous, caps: (
            rank_weighted_portfolio(scores, constraints, previous=previous, liquidity_caps=caps)
        ),
        "inverse_volatility": lambda scores, vol, constraints, previous, caps: (
            inverse_volatility_portfolio(
                scores, vol, constraints, previous=previous, liquidity_caps=caps
            )
        ),
    }


def run_allocation_backtest(
    panel: BacktestPanel,
    policy: AllocationFn,
    constraints: PortfolioConstraints,
    *,
    capital: float,
    cost_bps: float,
    participation: float = 0.05,
    uncertainty: pd.DataFrame | None = None,
    uncertainty_strength: float = 0.0,
    dates: pd.Index | None = None,
) -> pd.DataFrame:
    """Run one policy over the panel and return its per-date net record.

    Returns a frame with gross and net return, turnover, cost, gross exposure,
    net exposure and position count per rebalance date. A date whose constraints
    are infeasible at this capital level is recorded with ``feasible=False``
    rather than dropped, so capacity exhaustion is visible in the record instead
    of quietly shrinking the sample.
    """
    if cost_bps < 0.0 or not np.isfinite(cost_bps):
        raise PortfolioError("cost_bps must be finite and non-negative")
    evaluation_dates = panel.scores.index if dates is None else dates
    previous: pd.Series | None = None
    rows: list[dict[str, Any]] = []

    for date in evaluation_dates:
        scores = panel.scores.loc[date].dropna()
        if scores.empty:
            continue
        volatility = panel.volatility.loc[date].reindex(scores.index)
        caps = liquidity_caps_from_adv(
            panel.adv.loc[date].reindex(scores.index).fillna(0.0),
            capital=capital,
            participation=participation,
        )
        try:
            result = policy(scores, volatility, constraints, previous, caps)
        except InfeasibleConstraintsError as exc:
            rows.append(
                {
                    "date": date,
                    "feasible": False,
                    "reason": str(exc)[:200],
                    "gross_return": 0.0,
                    "net_return": 0.0,
                    "turnover": 0.0,
                    "cost": 0.0,
                    "gross": 0.0,
                    "net_exposure": 0.0,
                    "n_positions": 0,
                }
            )
            continue

        weights = result.weights
        if uncertainty is not None and uncertainty_strength > 0.0:
            weights = apply_uncertainty_sizing(
                weights,
                uncertainty.loc[date].reindex(weights.index),
                strength=uncertainty_strength,
            )
        realized = panel.forward_returns.loc[date].reindex(weights.index).fillna(0.0)
        gross_return = float(weights @ realized)
        cost = result.turnover * cost_bps / 10_000.0
        rows.append(
            {
                "date": date,
                "feasible": True,
                "reason": "",
                "gross_return": gross_return,
                "net_return": gross_return - cost,
                "turnover": result.turnover,
                "cost": cost,
                "gross": result.gross,
                "net_exposure": result.net,
                "n_positions": result.n_positions,
            }
        )
        previous = weights

    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def summarize_record(record: pd.DataFrame) -> dict[str, Any]:
    """Reduce a per-date record to net-of-cost performance statistics."""
    if record.empty:
        return {
            "n_dates": 0,
            "feasible_fraction": 0.0,
            "net_return": float("nan"),
            "net_sharpe": float("nan"),
            "gross_return": float("nan"),
            "cost_drag": float("nan"),
            "max_drawdown": float("nan"),
            "mean_turnover": float("nan"),
        }
    net = record["net_return"].to_numpy(dtype=float)
    equity = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(equity)
    deviation = float(np.std(net, ddof=1)) if net.size > 1 else 0.0
    return {
        "n_dates": int(len(record)),
        "feasible_fraction": float(record["feasible"].mean()),
        "net_return": float(np.mean(net) * TRADING_DAYS),
        "net_sharpe": (
            float(np.mean(net) / deviation * np.sqrt(TRADING_DAYS)) if deviation > 0.0 else 0.0
        ),
        "gross_return": float(record["gross_return"].mean() * TRADING_DAYS),
        "cost_drag": float(record["cost"].mean() * TRADING_DAYS),
        "max_drawdown": float(np.min(equity / peak - 1.0)),
        "mean_turnover": float(record["turnover"].mean()),
    }


def compare_allocation_policies(
    panel: BacktestPanel,
    constraints: PortfolioConstraints,
    *,
    folds: tuple[Fold, ...],
    capital_levels: tuple[float, ...],
    cost_bps: float = 10.0,
    policies: dict[str, AllocationFn] | None = None,
    regimes: pd.Series | None = None,
    participation: float = 0.05,
) -> pd.DataFrame:
    """Compare every policy across folds, regimes, and capacity levels.

    **Development folds only.** The final holdout is deliberately not an
    argument: score it once with :func:`score_final_holdout` after every
    parameter is frozen. A function that cannot see the holdout cannot select on
    it.

    Rows are sorted by ``(policy, capital, fold, regime)`` — never by
    performance, because ranking a comparison table by its own metric invites
    reading the top row as a decision.
    """
    if not 1 <= len(capital_levels) <= MAX_CAPITAL_LEVELS:
        raise PortfolioError(f"capital_levels must hold 1..{MAX_CAPITAL_LEVELS} entries")
    if any(level <= 0.0 or not np.isfinite(level) for level in capital_levels):
        raise PortfolioError("every capital level must be finite and positive")
    if not folds:
        raise PortfolioError("at least one evaluation fold is required")
    policies = policies or _default_policies()

    rows: list[dict[str, Any]] = []
    for name in sorted(policies):
        for capital in sorted(capital_levels):
            record = run_allocation_backtest(
                panel,
                policies[name],
                constraints,
                capital=capital,
                cost_bps=cost_bps,
                participation=participation,
            )
            if record.empty:
                continue
            for fold in folds:
                window = record.loc[(record.index >= fold.start) & (record.index <= fold.end)]
                slices: list[tuple[str, pd.DataFrame]] = [("all", window)]
                if regimes is not None:
                    aligned = regimes.reindex(window.index)
                    for label in sorted(set(aligned.dropna()) - {"unknown"}):
                        slices.append((str(label), window.loc[aligned == label]))
                for regime_label, block in slices:
                    if block.empty:
                        continue
                    rows.append(
                        {
                            "policy": name,
                            "capital": capital,
                            "fold": fold.name,
                            "regime": regime_label,
                            **summarize_record(block),
                        }
                    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["policy", "capital", "fold", "regime"]).reset_index(drop=True)


def score_final_holdout(
    panel: BacktestPanel,
    policy: AllocationFn,
    constraints: PortfolioConstraints,
    *,
    holdout_dates: pd.Index,
    capital: float,
    cost_bps: float = 10.0,
    participation: float = 0.05,
) -> dict[str, Any]:
    """Score **one** frozen policy on the untouched holdout, once.

    Separate from the comparison by design. The holdout's only legitimate use is
    to report what a decision already made would have produced; running the
    comparison grid on it would convert it into another development fold and
    destroy the only unbiased estimate available.
    """
    record = run_allocation_backtest(
        panel,
        policy,
        constraints,
        capital=capital,
        cost_bps=cost_bps,
        participation=participation,
        dates=holdout_dates,
    )
    summary = summarize_record(record)
    summary["holdout"] = True
    summary["capital"] = capital
    summary["cost_bps"] = cost_bps
    return summary


def capacity_frontier(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return each policy's net performance as a function of capital.

    The number that decides whether a strategy is investable at size. A policy
    whose net Sharpe collapses between two capital levels has found its capacity
    ceiling, and reading only the smallest level would have hidden it.
    """
    if comparison.empty:
        return comparison
    overall = comparison.loc[comparison["regime"] == "all"]
    grouped = (
        overall.groupby(["policy", "capital"], as_index=False)
        .agg(
            net_sharpe=("net_sharpe", "mean"),
            net_return=("net_return", "mean"),
            cost_drag=("cost_drag", "mean"),
            feasible_fraction=("feasible_fraction", "mean"),
        )
        .sort_values(["policy", "capital"])
        .reset_index(drop=True)
    )
    return grouped
