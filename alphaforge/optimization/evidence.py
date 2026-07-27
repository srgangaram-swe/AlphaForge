"""Walk-forward evidence for the Markowitz optimizer (SF-S4-MR2).

Runs the optimizer chronologically against the SF-S4-MR1 baselines — equal
weight, inverse volatility, ranking portfolios — plus a **no-trade** baseline,
net of costs, across folds, regimes, turnover budgets, and capacity levels.

The no-trade arm matters more than it looks. An optimizer that rebalances hard
can beat a static book gross and lose to it net; without a do-nothing baseline
in the table there is no way to see that, and "our optimizer beat equal weight"
can be true while "our optimizer beat leaving it alone" is false.

**Causality.** At each date the covariance is estimated from returns strictly
before that date and the alpha is the score available at it. A mutation test
asserts that rewriting all later returns, later covariance observations, and
later universe membership cannot change an earlier allocation.

Nothing here selects a winner, and the final holdout is not an argument to the
comparison — see :mod:`alphaforge.portfolio.evidence` for the same discipline.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from alphaforge.optimization.mean_variance import (
    CostModel,
    Formulation,
    MeanVarianceProblem,
    OptimizerError,
    solve_mean_variance,
)
from alphaforge.optimization.risk_model import RiskModelError, shrinkage_covariance
from alphaforge.portfolio.contracts import (
    InfeasibleConstraintsError,
    PortfolioConstraints,
    PortfolioError,
    liquidity_caps_from_adv,
)
from alphaforge.portfolio.evidence import (
    BacktestPanel,
    Fold,
    summarize_record,
)

#: Refusal thresholds, not tuning knobs.
MAX_TURNOVER_BUDGETS = 8


def run_markowitz_backtest(
    panel: BacktestPanel,
    returns: pd.DataFrame,
    constraints: PortfolioConstraints,
    *,
    formulation: Formulation = "alpha_risk_cost",
    capital: float,
    cost_bps: float = 10.0,
    participation: float = 0.05,
    risk_aversion: float = 5.0,
    covariance_window: int = 252,
    budget: float | None = None,
    dates: pd.Index | None = None,
    max_iterations: int = 2_000,
) -> pd.DataFrame:
    """Run the optimizer walk-forward and return its per-date net record.

    The record shares its columns with
    :func:`alphaforge.portfolio.evidence.run_allocation_backtest`, so Markowitz
    variants and the MR1 baselines summarize identically and are directly
    comparable.

    A date whose risk model cannot be estimated, whose constraints are
    infeasible, or whose solve fails its audit is recorded with
    ``feasible=False`` and its reason — never dropped, because dropping the hard
    dates is how an optimizer acquires a survivorship-flattered record.
    """
    evaluation_dates = panel.scores.index if dates is None else dates
    previous: pd.Series | None = None
    rows: list[dict[str, Any]] = []

    for date in evaluation_dates:
        scores = panel.scores.loc[date].dropna()
        if scores.empty:
            continue
        reason = ""
        # A turnover cap limits *rebalancing*. The initial build is not a
        # rebalance — there is no prior book to trade away from — so applying the
        # cap to it makes every subsequent date unreachable and the whole arm
        # reads as infeasible. The first allocation is therefore uncapped, and
        # every later one is capped.
        active_constraints = (
            replace(constraints, max_turnover=None) if previous is None else constraints
        )
        try:
            risk_model = shrinkage_covariance(
                returns.loc[:, scores.index], as_of=date, window=covariance_window
            )
            usable = [asset for asset in risk_model.assets if asset in scores.index]
            alpha = scores.reindex(usable)
            caps = liquidity_caps_from_adv(
                panel.adv.loc[date].reindex(usable).fillna(0.0),
                capital=capital,
                participation=participation,
            )
            problem = MeanVarianceProblem(
                risk_model=risk_model,
                expected_returns=alpha if formulation != "minimum_variance" else None,
                constraints=active_constraints,
                previous_weights=previous,
                cost_model=CostModel(linear_bps=cost_bps),
                risk_aversion=risk_aversion,
                liquidity_caps=caps,
                budget=budget,
            )
            result = solve_mean_variance(
                problem, formulation=formulation, max_iterations=max_iterations
            )
            if not result.audit_passed:
                raise OptimizerError(f"audit failed: {result.audit_violations}")
        except (RiskModelError, OptimizerError, InfeasibleConstraintsError, PortfolioError) as exc:
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
        realized = panel.forward_returns.loc[date].reindex(weights.index).fillna(0.0)
        gross_return = float(weights @ realized)
        turnover = result.turnover
        cost = turnover * cost_bps / 10_000.0
        rows.append(
            {
                "date": date,
                "feasible": True,
                "reason": reason,
                "gross_return": gross_return,
                "net_return": gross_return - cost,
                "turnover": turnover,
                "cost": cost,
                "gross": result.gross,
                "net_exposure": result.net,
                "n_positions": int((weights.abs() > 1e-12).sum()),
                "ex_ante_volatility": result.ex_ante_volatility,
                "condition_number": result.risk_diagnostics["condition_number"],
                "solver_status": result.status,
            }
        )
        previous = weights

    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def run_no_trade_backtest(
    panel: BacktestPanel,
    initial_weights: pd.Series,
    *,
    cost_bps: float = 10.0,
    dates: pd.Index | None = None,
) -> pd.DataFrame:
    """Run the do-nothing baseline: buy once, then never rebalance.

    The arm that is easiest to omit and hardest to beat net of costs. It pays
    turnover exactly once, on the initial purchase, and nothing afterwards.
    """
    evaluation_dates = panel.scores.index if dates is None else dates
    rows: list[dict[str, Any]] = []
    for position, date in enumerate(evaluation_dates):
        realized = panel.forward_returns.loc[date].reindex(initial_weights.index).fillna(0.0)
        gross_return = float(initial_weights @ realized)
        turnover = float(initial_weights.abs().sum()) if position == 0 else 0.0
        cost = turnover * cost_bps / 10_000.0
        rows.append(
            {
                "date": date,
                "feasible": True,
                "reason": "",
                "gross_return": gross_return,
                "net_return": gross_return - cost,
                "turnover": turnover,
                "cost": cost,
                "gross": float(initial_weights.abs().sum()),
                "net_exposure": float(initial_weights.sum()),
                "n_positions": int((initial_weights.abs() > 1e-12).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def compare_markowitz_variants(
    panel: BacktestPanel,
    returns: pd.DataFrame,
    constraints: PortfolioConstraints,
    *,
    folds: tuple[Fold, ...],
    capital_levels: tuple[float, ...],
    turnover_budgets: tuple[float | None, ...] = (None,),
    formulations: tuple[Formulation, ...] = (
        "minimum_variance",
        "maximum_utility",
        "alpha_risk_cost",
    ),
    cost_bps: float = 10.0,
    risk_aversion: float = 5.0,
    budget: float | None = 1.0,
    regimes: pd.Series | None = None,
    include_no_trade: bool = True,
) -> pd.DataFrame:
    """Compare Markowitz formulations across folds, regimes, budgets, and capacity.

    Rows sort by ``(arm, capital, turnover_budget, fold, regime)`` — never by
    performance, because ranking a comparison by its own metric invites reading
    the top row as a decision. The rejection rule belongs to SF-S4-MR9.
    """
    if not folds:
        raise OptimizerError("at least one evaluation fold is required")
    if not 1 <= len(turnover_budgets) <= MAX_TURNOVER_BUDGETS:
        raise OptimizerError(f"turnover_budgets must hold 1..{MAX_TURNOVER_BUDGETS} entries")

    records: dict[tuple[str, float, str], pd.DataFrame] = {}
    # Budget appears in the key as its string form so `None` and a numeric cap
    # share one column type in the emitted comparison frame.
    for formulation in formulations:
        for capital in sorted(capital_levels):
            for cap in turnover_budgets:
                limited = replace(constraints, max_turnover=cap)
                records[(formulation, capital, str(cap))] = run_markowitz_backtest(
                    panel,
                    returns,
                    limited,
                    formulation=formulation,
                    capital=capital,
                    cost_bps=cost_bps,
                    risk_aversion=risk_aversion,
                    budget=budget,
                )
    if include_no_trade:
        first = panel.scores.iloc[0].dropna()
        initial = pd.Series(1.0 / len(first), index=first.index)
        for capital in sorted(capital_levels):
            records[("no_trade", capital, "None")] = run_no_trade_backtest(
                panel, initial, cost_bps=cost_bps
            )

    rows: list[dict[str, Any]] = []
    for (arm, capital, budget_label), record in records.items():
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
                summary = summarize_record(block)
                summary.update(
                    {
                        "arm": arm,
                        "capital": capital,
                        "turnover_budget": budget_label,
                        "fold": fold.name,
                        "regime": regime_label,
                        "mean_condition_number": (
                            float(block["condition_number"].mean())
                            if "condition_number" in block
                            else float("nan")
                        ),
                        "solver_failures": (
                            int((block["solver_status"] != "optimal").sum())
                            if "solver_status" in block
                            else 0
                        ),
                    }
                )
                rows.append(summary)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    ordered = ["arm", "capital", "turnover_budget", "fold", "regime"]
    return (
        frame.loc[:, ordered + [column for column in frame.columns if column not in ordered]]
        .sort_values(ordered)
        .reset_index(drop=True)
    )


def sensitivity_to_input_error(
    panel: BacktestPanel,
    returns: pd.DataFrame,
    constraints: PortfolioConstraints,
    *,
    capital: float,
    perturbations: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5),
    seed: int = 11,
    formulation: Formulation = "maximum_utility",
    budget: float | None = 1.0,
    risk_aversion: float = 5.0,
) -> pd.DataFrame:
    """Measure how much a given alpha error moves the optimizer's net result.

    The single most important diagnostic for mean-variance: it is famously more
    sensitive to expected-return error than to covariance error, and an
    optimizer whose result collapses under a 10% alpha perturbation is reporting
    the precision of its inputs, not skill.

    Perturbations are multiplicative Gaussian noise on the score, seeded so the
    sweep is reproducible.
    """
    rows: list[dict[str, Any]] = []
    for level in perturbations:
        generator = np.random.default_rng([seed, int(level * 1_000)])
        noisy = panel.scores.copy()
        if level > 0.0:
            noise = generator.normal(1.0, level, noisy.shape)
            noisy = pd.DataFrame(noisy.to_numpy() * noise, index=noisy.index, columns=noisy.columns)
        perturbed = BacktestPanel(
            scores=noisy,
            forward_returns=panel.forward_returns,
            volatility=panel.volatility,
            adv=panel.adv,
        )
        record = run_markowitz_backtest(
            perturbed,
            returns,
            constraints,
            formulation=formulation,
            capital=capital,
            budget=budget,
            risk_aversion=risk_aversion,
        )
        summary = summarize_record(record)
        summary["alpha_error"] = level
        rows.append(summary)
    return pd.DataFrame(rows).sort_values("alpha_error").reset_index(drop=True)
