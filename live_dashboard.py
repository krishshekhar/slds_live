#!/usr/bin/env python3
"""
Streamlit dashboard for monitoring live trading activity.

Local: reads ``results/live_trading/live_events.jsonl`` (or ``--events-file``).

Streamlit Community Cloud: set ``EVENTS_URL`` in **Settings → Secrets** to an https URL
that returns the same JSONL text (cloud cannot read your Mac filesystem).
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import urllib.error
import urllib.request
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

import altair as alt
import pandas as pd
import streamlit as st


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live trading dashboard")
    p.add_argument(
        "--events-file",
        default=os.path.join("results", "live_trading", "live_events.jsonl"),
        help="Path to JSONL events emitted by run_live_trader.py (ignored if --events-url is set).",
    )
    p.add_argument(
        "--events-url",
        default=None,
        help="HTTP(S) URL to the same JSONL log (for Streamlit Cloud). Overrides --events-file.",
    )
    p.add_argument(
        "--refresh-seconds",
        type=int,
        default=5,
        help="Auto-refresh interval in seconds.",
    )
    return p.parse_args()


def _parse_jsonl_lines(lines: Iterable[str]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Parse JSONL; return rows and the last cycle_status object that carries a full window."""
    rows: list[dict[str, Any]] = []
    last_window: dict[str, Any] | None = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(o)
        if o.get("event_type") == "cycle_status":
            w = o.get("window_close")
            if isinstance(w, list) and len(w) > 0:
                last_window = o
    return rows, last_window


def _dataframe_from_event_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows)
    if "ts_utc" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce")
    return df.sort_values("ts_utc", ascending=True).reset_index(drop=True)


def _load_events_from_path(path: str) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    if not os.path.exists(path):
        return pd.DataFrame(), None
    with open(path, encoding="utf-8") as f:
        rows, last_window = _parse_jsonl_lines(f)
    return _dataframe_from_event_rows(rows), last_window


def _normalize_events_url(url: str) -> str:
    """
    Clean secrets/paste issues and map GitHub *file* pages to raw.githubusercontent.com.
    """
    u = url.strip()
    if len(u) >= 2 and u[0] == u[-1] and u[0] in "\"'":
        u = u[1:-1].strip()
    parsed = urlsplit(u)
    u = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))

    # https://github.com/{owner}/{repo}/blob/{ref}/{path}
    m = re.match(
        r"^https?://(?:www\.)?github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)$",
        u,
        re.I,
    )
    if m:
        owner, repo, ref, path = m.groups()
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"

    # https://github.com/{owner}/{repo}/raw/{ref}/{path} (GitHub UI "Raw" sometimes)
    m2 = re.match(
        r"^https?://(?:www\.)?github\.com/([^/]+)/([^/]+)/raw/([^/]+)/(.+)$",
        u,
        re.I,
    )
    if m2:
        owner, repo, ref, path = m2.groups()
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"

    return u


def _load_events_from_url(
    url: str, *, timeout: int = 45
) -> tuple[pd.DataFrame, dict[str, Any] | None, str | None]:
    """
    Returns (dataframe, last_window_object, fetch_error).
    fetch_error is set when the HTTP request fails (not when the body is empty JSONL).
    """
    url = _normalize_events_url(url)
    if "..." in url:
        return (
            pd.DataFrame(),
            None,
            "URL still contains '...' — replace with your real GitHub username, repo name, branch, and file path.",
        )
    if not url.lower().startswith(("http://", "https://")):
        return pd.DataFrame(), None, "URL must start with https://"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; SLDS-Live-Dashboard/1.0; "
                "+https://github.com/krishshekhar/slds_live)"
            ),
            "Accept": "text/plain,text/*,*/*;q=0.01",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        hint = ""
        if e.code in (400, 404):
            hint = (
                " Check the path and branch name on raw.githubusercontent.com "
                "(open the same URL in a private browser tab). "
                "Use the **Raw** file link, not the repo browser page, unless it was auto-converted."
            )
        return pd.DataFrame(), None, f"HTTP {e.code}: {e.reason}.{hint}"
    except urllib.error.URLError as e:
        return pd.DataFrame(), None, str(e.reason if hasattr(e, "reason") else e)
    except (OSError, ValueError) as e:
        return pd.DataFrame(), None, str(e)
    rows, last_window = _parse_jsonl_lines(body.splitlines())
    return _dataframe_from_event_rows(rows), last_window, None


def _secret_events_url() -> str | None:
    try:
        s = st.secrets
        u = s.get("EVENTS_URL") or s.get("events_url")
        return str(u).strip() if u else None
    except (FileNotFoundError, RuntimeError, AttributeError, TypeError):
        return None


def _resolve_events_url(cli_url: str | None) -> str | None:
    if cli_url and str(cli_url).strip():
        return str(cli_url).strip()
    su = _secret_events_url()
    if su:
        return su
    env_u = os.environ.get("EVENTS_URL", "").strip()
    return env_u or None


def _regime_color_scale() -> dict[str, Any]:
    return {
        "domain": ["bull", "bear", "neutral"],
        "range": ["#2ecc71", "#e74c3c", "#bdc3c7"],
    }


def _kpi(label: str, value: str) -> None:
    st.metric(label=label, value=value)


def _fmt_num(v: Any, digits: int = 4) -> str:
    try:
        return f"{float(v):,.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def _visible_y_domain(
    series: pd.Series,
    *,
    pad_frac: float = 0.12,
    floor_pad: float = 0.0,
) -> tuple[float, float]:
    """
    Y-axis [lo, hi] so small moves are visible: pad is max(fraction of span, floor_pad).
    Avoids Streamlit/Altair defaults that zoom out to huge round numbers and flatten the line.
    """
    v = pd.to_numeric(series, errors="coerce").dropna()
    if v.empty:
        return (0.0, 1.0)
    lo, hi = float(v.min()), float(v.max())
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return (0.0, 1.0)
    span = hi - lo
    if span <= 0.0:
        pad = max(floor_pad, abs(lo) * 1e-6 if abs(lo) > 1e-12 else 1e-3)
        return (lo - pad, hi + pad)
    pad = max(span * pad_frac, floor_pad)
    return (lo - pad, hi + pad)


def _alt_line_tight_y(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    y_title: str,
    floor_pad: float,
    height: int = 220,
    color: str = "#1f77b4",
    tooltip_y_format: str = ",.4f",
) -> alt.Chart:
    d0, d1 = _visible_y_domain(data[y_col], pad_frac=0.12, floor_pad=floor_pad)
    return (
        alt.Chart(data)
        .mark_line(color=color, strokeWidth=2)
        .encode(
            x=alt.X(f"{x_col}:T", title="Time"),
            y=alt.Y(
                f"{y_col}:Q",
                title=y_title,
                scale=alt.Scale(domain=[d0, d1], nice=False, zero=False),
            ),
            tooltip=[
                alt.Tooltip(f"{x_col}:T", title="Time"),
                alt.Tooltip(f"{y_col}:Q", title=y_title, format=tooltip_y_format),
            ],
        )
        .properties(height=height, width="container")
    )


def _newest_first(df: pd.DataFrame, n: int | None = None) -> pd.DataFrame:
    """Sort by event time descending so the latest row appears at the top."""
    if df.empty or "ts_utc" not in df.columns:
        return df
    out = df.sort_values("ts_utc", ascending=False, na_position="last")
    return out.head(n) if n is not None else out


def _heading_with_info(title: str, help_text: str, level: int = 3) -> None:
    """Section title with a circled-i that shows `help_text` on hover (native tooltip)."""
    tag = f"h{level}"
    safe_title = html.escape(title)
    tip = html.escape(help_text.replace("\n", " "))
    st.markdown(
        f'<{tag} style="display:flex;align-items:center;gap:0.45rem;margin-bottom:0.35rem;margin-top:1rem;">'
        f"{safe_title}"
        f'<span title="{tip}" style="cursor:help;opacity:0.75;font-size:1.05rem;line-height:1;" '
        f'aria-label="Info about this chart">&#9432;</span>'
        f"</{tag}>",
        unsafe_allow_html=True,
    )


def _portfolio_snapshot(cycle_df: pd.DataFrame, order_df: pd.DataFrame) -> dict[str, Any]:
    """Latest cash, invested (market value of position), equity, session P/L."""
    out: dict[str, Any] = {
        "equity": None,
        "cash": None,
        "qty": None,
        "last_price": None,
        "invested": None,
        "session_start_equity": None,
        "pnl": None,
        "pnl_pct": None,
    }
    if cycle_df.empty:
        return out
    session_start = float(cycle_df["equity_before"].iloc[0])
    out["session_start_equity"] = session_start
    last_c = cycle_df.iloc[-1]
    out["last_price"] = last_c.get("last_price")

    if not order_df.empty and "equity_after" in order_df.columns:
        row = order_df.iloc[-1]
        out["equity"] = float(row["equity_after"])
        out["cash"] = float(row["cash_after"]) if "cash_after" in row and pd.notna(row["cash_after"]) else None
        out["qty"] = float(row["qty_after"]) if "qty_after" in row and pd.notna(row["qty_after"]) else 0.0
    else:
        out["equity"] = float(last_c["equity_before"])
        out["cash"] = float(last_c["cash_before"]) if "cash_before" in last_c else None
        out["qty"] = float(last_c["qty_before"]) if "qty_before" in last_c else 0.0

    lp = out["last_price"]
    if lp is not None and out["qty"] is not None:
        out["invested"] = float(out["qty"]) * float(lp)
    if out["equity"] is not None and out["invested"] is not None and out["cash"] is None:
        out["cash"] = float(out["equity"]) - float(out["invested"])

    if out["equity"] is not None and session_start is not None:
        out["pnl"] = float(out["equity"]) - session_start
        if abs(session_start) > 1e-9:
            out["pnl_pct"] = (out["pnl"] / session_start) * 100.0
    return out


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

    events_url = _resolve_events_url(args.events_url)
    url_fetch_error: str | None = None
    if events_url:
        fetch_url = _normalize_events_url(events_url)
        df, raw_last, url_fetch_error = _load_events_from_url(events_url)
        st.write("Events source: **remote URL** (refreshed each time this page reloads).")
        st.caption(f"Fetch URL: `{fetch_url}`")
    else:
        df, raw_last = _load_events_from_path(args.events_file)
        st.write(f"Events source: **local file** `{args.events_file}`")

    if df.empty:
        st.warning("No events loaded yet.")
        if events_url:
            if url_fetch_error:
                st.error(f"Fetch error: {url_fetch_error}")
                st.caption(
                    "Check **Settings → Secrets**: `EVENTS_URL` must be https and reachable from "
                    "Streamlit’s servers (not a URL on your laptop)."
                )
            else:
                st.info(
                    "The URL responded but contained no parseable JSONL lines (or an empty body). "
                    "Expected the same JSONL as `run_live_trader.py` writes (one JSON object per line). "
                    "In **Settings → Secrets**, `EVENTS_URL` must point to a **public** file (no login); "
                    "private buckets need a signed URL."
                )
        else:
            st.info(
                "On your laptop: run `run_live_trader.py` so it writes JSONL to the path above. "
                "On **Streamlit Community Cloud** there is no access to your Mac disk — set "
                "`EVENTS_URL` in the app’s **Settings → Secrets** to an HTTPS URL that serves "
                "the same `live_events.jsonl` text (see repo notes or deploy docs)."
            )
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

    if not cycle_df.empty:
        port = _portfolio_snapshot(cycle_df, order_df)
        _heading_with_info(
            "Portfolio (this session)",
            "Session P/L compares current total equity to equity at the first logged poll after startup. "
            "Cash and invested use the latest order event if present, else the latest cycle. "
            "Invested = shares × last price (market value of the stock leg). Paper mode starts near $10,000 cash.",
            level=3,
        )
        pc1, pc2, pc3, pc4, pc5 = st.columns(5)
        eq = port.get("equity")
        cash = port.get("cash")
        inv = port.get("invested")
        pnl = port.get("pnl")
        pnl_pct = port.get("pnl_pct")
        with pc1:
            st.metric(
                "Total equity",
                f"${eq:,.2f}" if eq is not None else "—",
                help="Cash plus value of shares at last quoted price.",
            )
        with pc2:
            st.metric(
                "Session P/L",
                f"${pnl:+,.2f}" if pnl is not None else "—",
                delta=f"{pnl_pct:+.2f}% vs start" if pnl_pct is not None else None,
                help="Current equity minus equity at first poll in this session.",
            )
        with pc3:
            st.metric("Cash", f"${cash:,.2f}" if cash is not None else "—")
        with pc4:
            st.metric(
                "Invested (position)",
                f"${inv:,.2f}" if inv is not None else "—",
                help="Qty × last price from latest event.",
            )
        with pc5:
            if pnl is None:
                st.metric("vs session start", "—")
            elif pnl > 1e-6:
                st.success("In profit")
            elif pnl < -1e-6:
                st.error("In loss")
            else:
                st.info("Flat vs session start")

    _heading_with_info(
        "Runtime KPIs",
        "Quick snapshot of the latest poll: last trade price, sizing (target weight), inferred regime label, drawdown since session equity baseline, and kill switch.",
        level=3,
    )
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
        plot_df = cycle_df[["ts_utc", "equity_before", "last_price", "target_weight"]].copy()
        plot_df = plot_df.set_index("ts_utc")

        _heading_with_info(
            "Session equity over time",
            "Simulated or broker-reported account equity at each poll, before the rebalance in that cycle.",
            level=4,
        )
        eq_chart_df = plot_df.reset_index()
        st.altair_chart(
            _alt_line_tight_y(
                eq_chart_df,
                "ts_utc",
                "equity_before",
                y_title="Equity ($)",
                floor_pad=2.0,
                height=220,
                color="#2e86ab",
                tooltip_y_format=",.2f",
            ),
            use_container_width=True,
        )

        _heading_with_info(
            "Session stock price over time",
            "Last Yahoo Finance close at each poll (same symbol you passed to run_live_trader.py).",
            level=4,
        )
        st.altair_chart(
            _alt_line_tight_y(
                eq_chart_df,
                "ts_utc",
                "last_price",
                y_title="Price",
                floor_pad=0.05,
                height=220,
                color="#c73e1d",
            ),
            use_container_width=True,
        )

        _heading_with_info(
            "Target position weight over time",
            "After regime label and risk caps: fraction of equity to hold long, before the order step.",
            level=4,
        )
        st.line_chart(plot_df[["target_weight"]], height=160)

        _heading_with_info(
            "Regime decisions (table)",
            "Each row is one trader poll: discrete state id, mapped bull/bear/neutral label, and resulting target weight. Newest poll at the top.",
            level=4,
        )
        regime_view = cycle_df[["ts_utc", "latest_regime", "regime_label", "target_weight"]].copy()
        st.dataframe(_newest_first(regime_view, 50), use_container_width=True)

    # Latest window: full path from raw JSONL (lists intact); from same load as df when using URL
    if raw_last:
        _heading_with_info(
            "Latest rolling window — price, colored by inferred regime (per bar)",
            "This chart uses only the **most recent** poll: the Gibbs path z over the window and the semantic map "
            "bear / neutral / bull from that poll (instant window mean ranks, or EMA-smoothed ranks if you run the "
            "trader with --label-mode smoothed_rank). When a new bar arrives, refit can change z on **older** bars "
            "and can re-rank state ids, so a bar that looked neutral on an earlier poll can appear bear here—this "
            "is retrospective consistency from the latest fit, not a frozen label history. Not a forecast.",
            level=3,
        )
        times = raw_last.get("window_bar_times", [])
        closes = raw_last.get("window_close", [])
        labels = raw_last.get("window_regime_label", [])
        if times and closes and labels and len(times) == len(closes) and len(closes) == len(labels):
            pw = pd.DataFrame(
                {
                    "bar_time": pd.to_datetime(times, utc=False),
                    "close": closes,
                    "regime_label": labels,
                }
            )
            y0, y1 = _visible_y_domain(pw["close"], pad_frac=0.12, floor_pad=0.05)
            close_y_scale = alt.Scale(domain=[y0, y1], nice=False, zero=False)
            line = (
                alt.Chart(pw)
                .mark_line(color="#1a5276", strokeWidth=2)
                .encode(
                    x="bar_time:T",
                    y=alt.Y("close:Q", title="Close", scale=close_y_scale),
                    tooltip=["bar_time", "close", "regime_label"],
                )
            )
            points = (
                alt.Chart(pw)
                .mark_point(filled=True, size=70)
                .encode(
                    x="bar_time:T",
                    y=alt.Y("close:Q", title="Close", scale=close_y_scale),
                    color=alt.Color(
                        "regime_label:N",
                        title="Regime (in-sample labels)",
                        scale=_regime_color_scale(),
                    ),
                    tooltip=["bar_time", "close", "regime_label"],
                )
            )
            c_regime = alt.layer(line, points).properties(height=320, width="container")
            st.altair_chart(c_regime, use_container_width=True)

            _heading_with_info(
                "Regime strip — same window",
                "Horizontal bands showing the mapped bull/bear/neutral label per bar; read with the chart above.",
                level=4,
            )
            pw2 = pw.copy()
            pw2["next_t"] = pw2["bar_time"].shift(-1)
            pw2.loc[pw2.index[-1], "next_t"] = pw2["bar_time"].iloc[-1] + pd.Timedelta(seconds=1)
            regime_strip = (
                alt.Chart(pw2)
                .mark_rect(height=24)
                .encode(
                    x=alt.X("bar_time:T", title="Bar time"),
                    x2=alt.X2("next_t:T"),
                    y=alt.Y("regime_label:N", sort=["bear", "neutral", "bull"], title="Regime"),
                    color=alt.Color("regime_label:N", scale=_regime_color_scale()),
                )
                .properties(height=100, width="container", title="Regime strip (per bar)")
            )
            st.altair_chart(regime_strip, use_container_width=True)

        pred = raw_last.get("predicted_regime_next_action")
        if isinstance(pred, dict):
            st.info(
                f"**Trading signal (latest bar):** id={pred.get('regime_id')} — **{pred.get('regime_label')}**  \n"
                f"{pred.get('note', '')}"
            )

        retro = raw_last.get("retrospective_same_bar")
        if isinstance(retro, dict):
            if retro.get("available") is False:
                st.warning(retro.get("reason", "Retrospective compare not available."))
            elif "regime_id_this_fit" in retro:
                m = retro.get("matches_prior_regime_id")
                _heading_with_info(
                    "Prior bar — last fit vs this refit",
                    "When a new bar arrives, Gibbs is rerun on a shifted window; the discrete state on an older timestamp can change. "
                    "Compares regime id for that bar from the previous cycle vs this cycle—not external ground truth.",
                    level=4,
                )
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Regime id (previous cycle)", str(retro.get("regime_id_last_cycle")))
                with c2:
                    st.metric("Regime id (this refit)", str(retro.get("regime_id_this_fit")))
                with c3:
                    st.metric("Same id after refit?", "Yes" if m else "No")

    _heading_with_info(
        "Recent orders",
        "One row per completed rebalance attempt: submitted market-style orders or skipped (tiny size / already at target). "
        "Shows side, quantity, price used, and simulated or broker equity after execution. Newest at the top.",
        level=3,
    )
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
        st.dataframe(_newest_first(order_df[show_cols], 100), use_container_width=True)

    _heading_with_info(
        "Runtime errors",
        "Exceptions caught inside a polling cycle (data download, model fit, broker). Newest first. Empty means no logged failures.",
        level=3,
    )
    if error_df.empty:
        st.success("No runtime errors logged.")
    else:
        show_cols = [c for c in ["ts_utc", "symbol", "mode", "model", "error"] if c in error_df.columns]
        st.dataframe(_newest_first(error_df[show_cols], 100), use_container_width=True)

    _heading_with_info(
        "Raw event feed (latest 200)",
        "Flattened JSONL rows for debugging: startup, cycle_status, order, error. Newest events first. Wide columns come from nested objects.",
        level=3,
    )
    st.dataframe(_newest_first(df, 200), use_container_width=True)


if __name__ == "__main__":
    main()
