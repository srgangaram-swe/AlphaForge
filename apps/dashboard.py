from __future__ import annotations

from pathlib import Path

import pandas as pd


def _latest_run() -> Path | None:
    pointer = Path("runs/latest_run.txt")
    return Path(pointer.read_text().strip()) if pointer.exists() else None


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install app extras with: pip install -e '.[app]'") from exc

    st.set_page_config(page_title="AlphaForge", layout="wide")
    st.title("AlphaForge")
    run_dir = _latest_run()
    if run_dir is None:
        st.info("No run found. Run `make demo` first.")
        return

    st.caption(f"Run: {run_dir}")
    cols = st.columns(3)
    summary_path = run_dir / "backtest_summary.json"
    if summary_path.exists():
        summary = pd.read_json(summary_path, typ="series")
        cols[0].metric("Total return", f"{summary.get('total_return', 0):.2%}")
        cols[1].metric("Sharpe", f"{summary.get('sharpe', 0):.2f}")
        cols[2].metric("Max drawdown", f"{summary.get('max_drawdown', 0):.2%}")

    curve_path = run_dir / "equity_curve.csv"
    if curve_path.exists():
        curve = pd.read_csv(curve_path, parse_dates=["date"])
        st.line_chart(curve.set_index("date")[["equity"]])
        st.dataframe(curve.tail(20), use_container_width=True)

    metrics_path = run_dir / "model_metrics.csv"
    if metrics_path.exists():
        st.subheader("Model Metrics")
        st.dataframe(pd.read_csv(metrics_path), use_container_width=True)

    weights_path = run_dir / "executed_weights.csv"
    if weights_path.exists():
        st.subheader("Latest Holdings")
        weights = pd.read_csv(weights_path, parse_dates=["date"])
        st.dataframe(weights.sort_values("date").tail(30), use_container_width=True)


if __name__ == "__main__":
    main()
