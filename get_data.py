"""
Download NIFTY / India VIX (yfinance) and FRED macro series without pandas_datareader
(Python 3.12+ removed distutils; pandas_datareader still depends on it).

Macro block (``--macro``):
  - ``us`` — US Treasury 10Y–2Y (``T10Y2Y``) and US HY OAS (``BAMLH0A0HYM2``), daily.
  - ``india`` — India OECD 10Y gov yield minus OECD short rate (monthly series,
    forward-filled to equity dates) and ICE BofA *Asia* EM corporate OAS
    (``BAMLEMRACRPIASIAOAS``), daily (regional index; includes Indian issuers among others).
"""

from __future__ import annotations

import argparse
import datetime
import os

import numpy as np
import pandas as pd
import yfinance as yf


def _yf_adj_close(symbol: str, start, end) -> pd.Series:
    """yfinance often returns a MultiIndex columns index; normalize to a Series."""
    raw = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    if raw.empty:
        raise ValueError(f"No price data returned for {symbol!r}.")
    if isinstance(raw.columns, pd.MultiIndex):
        adj = raw.xs("Adj Close", axis=1, level=0)
        return adj.squeeze()
    return raw["Adj Close"].squeeze()


def _fred_series(series_id: str) -> pd.Series:
    """Load a FRED series from the public graph CSV endpoint."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    df.columns = [str(c).strip() for c in df.columns]
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    s = df.iloc[:, 0].replace(".", np.nan).astype(float)
    s.name = series_id
    return s


def _align_monthly_to_daily(
    monthly: pd.Series, daily_index: pd.DatetimeIndex
) -> pd.Series:
    """Forward-fill monthly observations onto ``daily_index`` (no back-fill)."""
    m = monthly.sort_index().astype(float)
    union = daily_index.union(m.index).sort_values()
    filled = m.reindex(union).ffill()
    return filled.reindex(daily_index)


def _strip_tz(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is not None:
        return idx.tz_localize(None)
    return idx


def main(*, macro: str = "india") -> None:
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime(2024, 1, 1)

    nifty = _yf_adj_close("^NSEI", start, end)
    india_vix = _yf_adj_close("^INDIAVIX", start, end)

    nifty_returns = np.log(nifty / nifty.shift(1))
    vix_change = india_vix.diff()

    base_df = pd.DataFrame({"nifty_log_return": nifty_returns, "vix_change": vix_change})
    base_df.index = _strip_tz(pd.DatetimeIndex(base_df.index))

    idx = base_df.index
    slice_start, slice_end = pd.Timestamp(start), pd.Timestamp(end)

    if macro == "us":
        ys = _fred_series("T10Y2Y")
        cr = _fred_series("BAMLH0A0HYM2")
        base_df["yield_spread"] = ys.reindex(idx).ffill()
        base_df["credit_spread"] = cr.reindex(idx).ffill()
    else:
        # India: OECD monthly 10Y G-Sec minus OECD India short rate; ffilled to trading days.
        y10 = _fred_series("INDIRLTLT01STM")
        y_short = _fred_series("INDLOCOSTORSTM")
        term_m = (y10 - y_short).dropna()
        base_df["india_term_spread"] = _align_monthly_to_daily(term_m, idx)
        # Regional EM Asia corporate OAS (not India-only); daily on FRED.
        cr_asia = _fred_series("BAMLEMRACRPIASIAOAS")
        base_df["asia_em_credit_oas"] = cr_asia.reindex(idx).ffill()

    df = base_df.loc[slice_start:slice_end].dropna()

    base = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base, "data", "market_features.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    out = df.reset_index()
    date_col = out.columns[0]
    if date_col != "Date":
        out = out.rename(columns={date_col: "Date"})
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows to {out_path} (macro={macro!r})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build data/market_features.csv")
    ap.add_argument(
        "--macro",
        choices=("india", "us"),
        default="india",
        help="FRED macro columns: india (default) or legacy US term/HY spreads.",
    )
    args = ap.parse_args()
    main(macro=args.macro)
