from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _latest_run() -> Path | None:
    pointer = Path("runs/latest_run.txt")
    return Path(pointer.read_text().strip()) if pointer.exists() else None


def _csv(run_dir: Path, name: str) -> pd.DataFrame | None:
    path = run_dir / name
    return pd.read_csv(path) if path.exists() else None


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install app extras with: pip install -e '.[app]'") from exc

    st.set_page_config(page_title="AlphaForge", layout="wide")
    st.title("AlphaForge")
    st.caption("Educational research platform — simulated results only, not financial advice.")
    run_dir = _latest_run()
    if run_dir is None:
        st.info("No run found. Run `make demo` first.")
        return

    st.caption(f"Run: {run_dir}")
    summary = {}
    summary_path = run_dir / "backtest_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    overfit_path = run_dir / "overfitting.json"
    overfit = json.loads(overfit_path.read_text()) if overfit_path.exists() else {}

    cols = st.columns(6)
    cols[0].metric("Total return", f"{summary.get('total_return', 0):.2%}")
    cols[1].metric("Sharpe", f"{summary.get('sharpe', 0):.2f}")
    cols[2].metric("Max drawdown", f"{summary.get('max_drawdown', 0):.2%}")
    cols[3].metric("Deflated Sharpe P", f"{summary.get('deflated_sharpe_prob', float('nan')):.3f}")
    cols[4].metric("PBO", f"{overfit.get('pbo', float('nan')):.3f}")
    cols[5].metric("Avg turnover", f"{summary.get('average_turnover', 0):.2%}")

    tab_bt, tab_ic, tab_risk, tab_paper = st.tabs(
        ["Backtest", "Signal Quality", "Risk & Regimes", "Paper Trading"]
    )

    with tab_bt:
        curve_path = run_dir / "equity_curve.csv"
        if curve_path.exists():
            curve = pd.read_csv(curve_path, parse_dates=["date"])
            st.line_chart(curve.set_index("date")[["equity"]])
            st.area_chart(curve.set_index("date")[["gross_exposure"]])
        metrics = _csv(run_dir, "model_metrics.csv")
        if metrics is not None:
            st.subheader("Model metrics per walk-forward window")
            st.dataframe(metrics, use_container_width=True)
        fills = _csv(run_dir, "fills.csv")
        if fills is not None:
            st.subheader("Recent next-open fills")
            st.caption(
                "Decision and fill dates are explicit; liquidity inputs are lagged before the open."
            )
            st.dataframe(fills.tail(50), use_container_width=True)
        attribution = _csv(run_dir, "pnl_attribution.csv")
        if attribution is not None:
            by_symbol = (
                attribution.groupby("symbol")[["market_pnl", "trading_cost", "net_pnl"]]
                .sum()
                .sort_values("net_pnl")
            )
            st.subheader("P&L attribution by symbol")
            st.bar_chart(by_symbol["net_pnl"])

    with tab_ic:
        ic = _csv(run_dir, "ic_summary.csv")
        if ic is not None:
            st.subheader("Information coefficient by model (Newey-West t-stats)")
            st.dataframe(ic, use_container_width=True)
        decay = _csv(run_dir, "ic_decay.csv")
        if decay is not None:
            st.subheader("IC decay for the selected model")
            st.bar_chart(decay.set_index("horizon")["mean_rank_ic"])
        quantiles = _csv(run_dir, "quantile_returns.csv")
        if quantiles is not None:
            st.subheader("Forward return by prediction quantile")
            st.bar_chart(quantiles.set_index("quantile")["mean_return"])

    with tab_risk:
        regimes = _csv(run_dir, "regime_performance.csv")
        if regimes is not None:
            st.subheader("Regime-conditional performance")
            st.dataframe(regimes, use_container_width=True)
        stress = _csv(run_dir, "stress_tests.csv")
        if stress is not None:
            st.subheader("Beta-aware stress scenarios")
            st.dataframe(stress, use_container_width=True)
        capacity = _csv(run_dir, "capacity_curve.csv")
        if capacity is not None:
            st.subheader("Capacity sensitivity (not an AUM forecast)")
            st.line_chart(capacity.set_index("scenario_aum")[["fill_ratio"]])
            st.dataframe(capacity, use_container_width=True)
        weights = _csv(run_dir, "executed_weights.csv")
        if weights is not None:
            st.subheader("Latest holdings")
            weights["date"] = pd.to_datetime(weights["date"])
            latest = weights[weights["date"] == weights["date"].max()]
            st.dataframe(
                latest[latest["weight"] != 0].sort_values("weight"),
                use_container_width=True,
            )

    with tab_paper:
        st.warning("SIMULATED PAPER TRADING ONLY — no real orders are ever placed.")
        orders = _csv(run_dir, "paper_orders.csv")
        if orders is not None:
            st.subheader("Simulated causal next-open fills")
            st.dataframe(orders.tail(50), use_container_width=True)
        else:
            st.info("Run `make paper` to generate simulated orders.")


if __name__ == "__main__":
    main()
