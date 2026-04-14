#!/usr/bin/env python3
"""
Streamlit dashboard for monitoring live trading activity.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import pandas as pd
import streamlit as st


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live trading dashboard")
    p.add_argument(
        "--events-file",
        default=os.path.join("results", "live_trading", "live_events.jsonl"),
        help="Path to JSONL events emitted by run_live_trader.py",
    )
    p.add_argument(
        "--refresh-seconds",
        type=int,
        default=5,
        help="Auto-refresh interval in seconds.",
    )
    return p.parse_args()


def _load_events(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows)
    if "ts_utc" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce")
    return df.sort_values("ts_utc", ascending=True).reset_index(drop=True)


def _kpi(label: str, value: str) -> None:
    st.metric(label=label, value=value)


def _fmt_num(v: Any, digits: int = 4) -> str:
    try:
        return f"{float(v):,.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="Live Trading Dashboard", layout="wide")
    st.title("Live Trading Dashboard")
    st.caption("Monitoring model decisions, orders, and runtime health")

    st.markdown(
        f"""
<script>
setTimeout(function() {{
  window.location.reload();
}}, {max(1, args.refresh_seconds) * 1000});
</script>
""",
        unsafe_allow_html=True,
    )

    st.write(f"Events file: `{args.events_file}`")
    df = _load_events(args.events_file)
    if df.empty:
        st.warning("No events yet. Start `run_live_trader.py` first.")
        return

    latest = df.iloc[-1].to_dict()
    cycle_df = df[df["event_type"] == "cycle_status"].copy()
    order_df = df[df["event_type"] == "order"].copy()
    error_df = df[df["event_type"] == "error"].copy()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi("Last Event", str(latest.get("event_type", "N/A")))
    with c2:
        _kpi("Symbol", str(latest.get("symbol", "N/A")))
    with c3:
        _kpi("Mode", str(latest.get("mode", "N/A")))
    with c4:
        _kpi("Model", str(latest.get("model", "N/A")))

    st.subheader("Runtime KPIs")
    k1, k2, k3, k4, k5 = st.columns(5)
    last_cycle = cycle_df.iloc[-1].to_dict() if not cycle_df.empty else {}
    with k1:
        _kpi("Last Price", _fmt_num(last_cycle.get("last_price")))
    with k2:
        _kpi("Target Weight", _fmt_num(last_cycle.get("target_weight"), 2))
    with k3:
        _kpi("Regime", str(last_cycle.get("regime_label", "N/A")))
    with k4:
        _kpi("Daily Drawdown", _fmt_num(100 * last_cycle.get("daily_drawdown", 0.0), 2) + "%")
    with k5:
        _kpi("Kill Switch", "ON" if bool(last_cycle.get("kill_switch", False)) else "OFF")

    if not cycle_df.empty:
        st.subheader("Equity and Price")
        plot_df = cycle_df[["ts_utc", "equity_before", "last_price", "target_weight"]].copy()
        plot_df = plot_df.set_index("ts_utc")
        st.line_chart(plot_df[["equity_before"]], height=220)
        st.line_chart(plot_df[["last_price"]], height=220)
        st.line_chart(plot_df[["target_weight"]], height=160)

        st.subheader("Regime timeline")
        regime_view = cycle_df[["ts_utc", "latest_regime", "regime_label", "target_weight"]].copy()
        st.dataframe(regime_view.tail(50), use_container_width=True)

    st.subheader("Recent Orders")
    if order_df.empty:
        st.info("No orders yet.")
    else:
        order_cols = [
            "ts_utc",
            "symbol",
            "regime_label",
            "target_weight",
            "order.status",
            "order.side",
            "order.qty",
            "order.price",
            "order.delta_notional",
            "equity_after",
        ]
        show_cols = [c for c in order_cols if c in order_df.columns]
        st.dataframe(order_df[show_cols].tail(100), use_container_width=True)

    st.subheader("Errors")
    if error_df.empty:
        st.success("No runtime errors logged.")
    else:
        show_cols = [c for c in ["ts_utc", "symbol", "mode", "model", "error"] if c in error_df.columns]
        st.dataframe(error_df[show_cols].tail(100), use_container_width=True)

    st.subheader("Raw Event Feed (latest 200)")
    st.dataframe(df.tail(200), use_container_width=True)


if __name__ == "__main__":
    main()
