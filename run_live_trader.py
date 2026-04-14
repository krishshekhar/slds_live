#!/usr/bin/env python3
"""
Live / paper trading runner for regime models.

This script is intentionally conservative:
- Uses a rolling window of recent bars.
- Re-fits a selected model each cycle and uses only the latest inferred regime.
- Applies strict risk gates before any order is sent.
- Defaults to paper mode unless --mode live and --confirm-live are both set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.backtest import _classify_regimes_bull_bear_neutral, _regime_stats
from src.hdp_arhmm import StickyHDPARHMM
from src.hdp_slds import StickyHDPSLDS
from src.initialization import apply_rslds_initialization, initialize_rslds
from src.recurrent_slds import RecurrentSLDS


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class EventLogger:
    def __init__(self, file_path: str) -> None:
        self.path = Path(file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, payload: dict[str, Any]) -> None:
        row = {
            "ts_utc": _now_utc(),
            "event_type": event_type,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")


def _download_bars(symbol: str, interval: str, lookback_period: str) -> pd.DataFrame:
    df = yf.download(
        tickers=symbol,
        period=lookback_period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        prepost=False,
    )
    if df.empty:
        raise RuntimeError(f"No market data returned for {symbol}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for needed in ("Close", "Volume"):
        if needed not in df.columns:
            raise RuntimeError(f"Missing {needed} in market data for {symbol}.")
    out = df.copy().sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _build_live_features(bars: pd.DataFrame) -> pd.DataFrame:
    c = bars["Close"].astype(float)
    v = bars["Volume"].astype(float)
    lr = np.log(c / c.shift(1))
    rv_20 = lr.rolling(20).std()
    mom_10 = c.pct_change(10)
    vol_chg = np.log1p(v / (v.rolling(20).mean() + 1e-9))
    feat = pd.DataFrame(
        {
            "price_log_return": lr,
            "realized_vol_20": rv_20,
            "mom_10": mom_10,
            "volume_ratio_log": vol_chg,
        },
        index=bars.index,
    ).dropna()
    if feat.shape[0] < 80:
        raise RuntimeError(
            f"Need at least 80 feature rows, got {feat.shape[0]}. Increase lookback period."
        )
    return feat


def _safe_symbol(sym: str) -> str:
    return sym.replace("/", "_").replace("\\", "_")


def _align_warm_latent(
    old_times: list[str],
    old_z: np.ndarray,
    old_x: np.ndarray | None,
    new_times: list[str],
    z_cap_exclusive: int,
    state_dim: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Map prior z (and x) onto the new bar-time grid; random init for unseen bars."""
    T = len(new_times)
    z_out = np.empty(T, dtype=int)
    old_map_z = {old_times[i]: int(old_z[i]) for i in range(len(old_times))}
    for i, t in enumerate(new_times):
        if t in old_map_z:
            z_out[i] = int(np.clip(old_map_z[t], 0, z_cap_exclusive - 1))
        else:
            z_out[i] = int(rng.integers(0, z_cap_exclusive))

    x_out: np.ndarray | None = None
    if old_x is not None and old_x.size > 0 and old_x.ndim == 2:
        x_out = np.zeros((T, state_dim), dtype=float)
        old_map_x = {old_times[i]: old_x[i].copy() for i in range(len(old_times))}
        for i, t in enumerate(new_times):
            if t in old_map_x:
                x_out[i] = old_map_x[t]
            else:
                x_out[i] = rng.normal(size=(state_dim,))
    return z_out, x_out


def _fit_and_infer_last_regime(
    model_name: str,
    y: np.ndarray,
    n_iter: int,
    burn_in: int,
    random_state: int,
    *,
    warm_z: np.ndarray | None = None,
    warm_x: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Returns (last_z, persist) where persist holds last_x when applicable for checkpointing.
    """
    t_total, d_obs = y.shape
    if t_total < 80:
        raise RuntimeError(f"Need at least 80 samples, got {t_total}.")
    rng = np.random.default_rng(random_state)
    persist: dict[str, Any] = {"last_x": None}

    if model_name == "rslds":
        k = 20
        x_rs, z_rs, w_init, r_init = initialize_rslds(
            y,
            K=k,
            state_dim=d_obs,
            ar_order=1,
            n_gibbs_ar=max(25, n_iter // 2),
            random_state=random_state,
            standardize=False,
        )
        if warm_z is not None and warm_x is not None:
            x_init = np.asarray(warm_x, dtype=float)
            z_init = np.asarray(warm_z, dtype=int)
        else:
            x_init = x_rs
            z_init = z_rs
        rslds = RecurrentSLDS(
            K=k,
            state_dim=d_obs,
            obs_dim=d_obs,
            pg_trunc=200,
            random_state=random_state,
        )
        apply_rslds_initialization(rslds, w_init, r_init)
        hist = rslds.gibbs(
            y,
            n_iters=n_iter,
            burn_in=burn_in,
            x_init=x_init,
            z_init=z_init,
        )
        lz = np.asarray(hist["last_z"], dtype=int).ravel()
        persist["last_x"] = np.asarray(hist["last_x"], dtype=float)
        return lz, persist

    if model_name == "hdp_arhmm":
        L = min(12, max(6, d_obs + 4))
        model = StickyHDPARHMM(
            L=L,
            D=d_obs,
            ar_order=1,
            alpha=6.0,
            gamma=2.0,
            kappa=20.0,
            random_state=random_state,
        )
        kw: dict[str, Any] = {}
        if warm_z is not None:
            kw["z_init"] = np.asarray(warm_z, dtype=int)
        hist = model.gibbs(y, n_iters=n_iter, burn_in=burn_in, **kw)
        return np.asarray(hist["last_z"], dtype=int).ravel(), persist

    L = 20
    model = StickyHDPSLDS(
        L=L,
        state_dim=d_obs,
        obs_dim=d_obs,
        alpha=6.0,
        gamma=2.0,
        kappa=20.0,
        random_state=random_state,
        use_hdp_auxiliary_beta=True,
        canonicalize_labels=True,
        sample_alpha=True,
        sample_gamma_mh=False,
        sample_kappa_mh=True,
    )
    kw2: dict[str, Any] = {}
    if warm_z is not None and warm_x is not None:
        kw2["z_init"] = np.asarray(warm_z, dtype=int)
        kw2["x_init"] = np.asarray(warm_x, dtype=float)
    hist = model.gibbs(y, n_iters=n_iter, burn_in=burn_in, **kw2)
    lz = np.asarray(hist["last_z"], dtype=int).ravel()
    persist["last_x"] = np.asarray(hist["last_x"], dtype=float)
    return lz, persist


def _load_warmstart_npz(path: str) -> dict[str, Any] | None:
    if not os.path.isfile(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
        bar_times = [str(t) for t in data["bar_times"].tolist()]
        z = np.asarray(data["z"], dtype=int)
        xflat = data["x"]
        x = None
        if xflat.size > 0 and xflat.ndim == 2:
            x = np.asarray(xflat, dtype=float)
        return {
            "bar_times": bar_times,
            "z": z,
            "x": x,
            "model_name": str(data["model_name"]),
            "obs_dim": int(data["obs_dim"]),
            "window_bars": int(data["window_bars"]),
        }
    except (OSError, ValueError, KeyError):
        return None


def _save_warmstart_npz(
    path: str,
    *,
    bar_times: list[str],
    z: np.ndarray,
    x: np.ndarray | None,
    model_name: str,
    obs_dim: int,
    window_bars: int,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    x_save = x if x is not None else np.zeros((0,))
    np.savez_compressed(
        path,
        bar_times=np.asarray(bar_times, dtype=object),
        z=np.asarray(z, dtype=np.int32),
        x=np.asarray(x_save, dtype=np.float64),
        model_name=model_name,
        obs_dim=obs_dim,
        window_bars=window_bars,
    )


def _z_truncate_cap(model_name: str, d_obs: int) -> int:
    if model_name == "rslds":
        return 3
    if model_name == "hdp_arhmm":
        return min(12, max(6, d_obs + 4))
    return min(10, max(5, d_obs + 3))


def _label_mean_ema_path(state_dir: str, symbol: str, model: str) -> str:
    return os.path.join(
        state_dir, f"label_mean_ema_{_safe_symbol(symbol)}_{model}.json"
    )


def _target_weight_ema_path(state_dir: str, symbol: str, mode: str) -> str:
    return os.path.join(
        state_dir, f"target_weight_ema_{_safe_symbol(symbol)}_{mode}.json"
    )


def _classify_regimes_smoothed_mean_ema(
    z: np.ndarray,
    rets: np.ndarray,
    *,
    alpha: float,
    ema_path: str,
    symbol: str,
    model: str,
) -> tuple[dict[int, str], dict[int, dict[str, float]]]:
    """
    EMA each state's window mean log-return across refits, then rank states for bull/bear/neutral.
    alpha = weight on previous smoothed mean (higher = slower label churn).
    """
    raw_stats = _regime_stats(z, rets)
    prev: dict[str, float] = {}
    if os.path.isfile(ema_path):
        try:
            with open(ema_path, encoding="utf-8") as f:
                blob = json.load(f)
            if blob.get("symbol") == symbol and blob.get("model") == model:
                prev = {str(k): float(v) for k, v in blob.get("means", {}).items()}
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            prev = {}

    a = float(np.clip(alpha, 0.0, 0.999))
    means_out: dict[str, float] = dict(prev)
    merged: dict[int, dict[str, float]] = {}
    for u, st in raw_stats.items():
        cur = float(st["mean"])
        key = str(u)
        s_prev = prev.get(key, cur)
        s_new = a * s_prev + (1.0 - a) * cur
        means_out[key] = s_new
        merged[u] = {
            **st,
            "mean": s_new,
            "mean_raw_window": cur,
        }

    Path(ema_path).parent.mkdir(parents=True, exist_ok=True)
    with open(ema_path, "w", encoding="utf-8") as f:
        json.dump({"means": means_out, "symbol": symbol, "model": model}, f, indent=2)

    unique = sorted(merged.keys(), key=lambda u: (merged[u]["mean"], -merged[u]["var"]))
    n = len(unique)
    label_map: dict[int, str] = {}
    if n == 0:
        return label_map, merged
    if n == 1:
        label_map[unique[0]] = "neutral"
        return label_map, merged
    if n == 2:
        label_map[unique[0]] = "bear"
        label_map[unique[1]] = "bull"
        return label_map, merged
    label_map[unique[0]] = "bear"
    label_map[unique[-1]] = "bull"
    for u in unique[1:-1]:
        label_map[u] = "neutral"
    return label_map, merged


def _load_target_weight_ema_prev(path: str) -> float | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
        return float(blob["last_smoothed"])
    except (json.JSONDecodeError, OSError, TypeError, ValueError, KeyError):
        return None


def _save_target_weight_ema(path: str, value: float) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"last_smoothed": float(value)}, f, indent=2)


def _explain_weight_vs_regime_id(
    latest_regime: int,
    regime_label: str,
    label_map: dict[int, str],
    regime_stats: dict[int, dict[str, float]],
    raw_w: float,
    target_weight: float,
    max_position_weight: float,
    kill_switch: bool,
) -> dict[str, Any]:
    """
    Structured explanation: target weight can move even when z[-1] is unchanged, because
    labels and per-state means are recomputed on each rolling window.
    """
    bear_id = next((z for z, lb in label_map.items() if lb == "bear"), None)
    bull_id = next((z for z, lb in label_map.items() if lb == "bull"), None)
    st = regime_stats.get(latest_regime, {})
    m_used = st.get("mean")
    m_raw_only = st.get("mean_raw_window", m_used)
    return {
        "latest_regime_id": int(latest_regime),
        "semantic_label_for_that_id": regime_label,
        "mean_log_return_for_state_id_on_this_window": m_used,
        "mean_log_return_this_window_instantaneous": m_raw_only,
        "n_bars_assigned_to_that_state": st.get("n"),
        "bear_state_id": bear_id,
        "bull_state_id": bull_id,
        "mean_log_return_for_bear_state": regime_stats.get(bear_id, {}).get("mean") if bear_id is not None else None,
        "mean_log_return_for_bull_state": regime_stats.get(bull_id, {}).get("mean") if bull_id is not None else None,
        "raw_target_weight": float(raw_w),
        "target_weight_after_caps": float(target_weight),
        "max_position_weight": float(max_position_weight),
        "kill_switch_active": bool(kill_switch),
        "note": (
            "z[-1] is only the discrete state index at the last bar. Bull/bear/neutral labels "
            "are re-ranked from mean log returns each refit (or from EMA-smoothed means if "
            "--label-mode smoothed_rank); neutral blending uses the same means used for ranking. "
            "The same index can keep a different mean for its cluster or a different label when "
            "the window shifts or Gibbs assigns a different full path z_1:T."
        ),
    }


def _target_weight_from_regime(
    label: str,
    regime_id: int,
    label_map: dict[int, str],
    regime_stats: dict[int, dict[str, float]],
) -> float:
    """
    Bull -> 1, bear -> 0. For neutral states, interpolate in [0, 1] by where this state's
    mean return sits between the bear state's mean and the bull state's mean (same window).
    """
    if label == "bull":
        return 1.0
    if label == "bear":
        return 0.0

    bear_mean: float | None = None
    bull_mean: float | None = None
    for z, lb in label_map.items():
        if lb == "bear":
            bear_mean = regime_stats.get(z, {}).get("mean")
        elif lb == "bull":
            bull_mean = regime_stats.get(z, {}).get("mean")
    m_now = regime_stats.get(regime_id, {}).get("mean")

    if bear_mean is None or bull_mean is None or m_now is None:
        return 0.5

    lo, hi = (bear_mean, bull_mean) if bear_mean <= bull_mean else (bull_mean, bear_mean)
    span = hi - lo
    if span <= 1e-12:
        return 0.5

    w = (float(m_now) - lo) / span
    return float(max(0.0, min(1.0, w)))


@dataclass
class AccountState:
    equity: float
    cash: float
    qty: float
    last_price: float


class Broker:
    def get_account_state(self, symbol: str, last_price: float) -> AccountState:
        raise NotImplementedError

    def rebalance_to_weight(
        self, symbol: str, target_weight: float, last_price: float
    ) -> dict[str, Any]:
        raise NotImplementedError


class PaperBroker(Broker):
    def __init__(
        self,
        symbol: str,
        starting_equity: float = 10000.0,
        *,
        state_file: str | None = None,
        resume: bool = True,
    ) -> None:
        self.symbol = symbol
        self.state_file = state_file
        self.cash = float(starting_equity)
        self.qty = 0.0
        if not resume:
            print(
                f"[{_now_utc()}] Paper resume disabled (--no-resume); "
                f"starting cash={self.cash:.2f} qty=0"
            )
        elif not state_file:
            print(f"[{_now_utc()}] Paper state path unset; starting cash={self.cash:.2f} qty=0")
        elif not os.path.isfile(state_file):
            ap = os.path.abspath(state_file)
            print(
                f"[{_now_utc()}] No paper state file yet ({ap}); "
                f"starting cash={self.cash:.2f} qty=0 — will save after each cycle."
            )
        else:
            try:
                ap = os.path.abspath(state_file)
                with open(state_file, encoding="utf-8") as f:
                    d = json.load(f)
                if d.get("symbol") != symbol:
                    print(
                        f"[{_now_utc()}] Paper state file {ap} is for symbol {d.get('symbol')!r}, "
                        f"not {symbol!r}; starting fresh cash={self.cash:.2f} qty=0"
                    )
                else:
                    self.cash = float(d["cash"])
                    self.qty = float(d["qty"])
                    print(
                        f"[{_now_utc()}] Resumed paper account from {ap}: "
                        f"cash={self.cash:.2f} qty={self.qty:.6f}"
                    )
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                print(
                    f"[{_now_utc()}] Could not load paper state ({os.path.abspath(state_file)}): {e}; "
                    f"starting cash={self.cash:.2f} qty=0"
                )

    def save_state(self) -> None:
        if not self.state_file:
            return
        Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(
                {"symbol": self.symbol, "cash": self.cash, "qty": self.qty},
                f,
                indent=2,
            )

    def get_account_state(self, symbol: str, last_price: float) -> AccountState:
        equity = self.cash + self.qty * last_price
        return AccountState(equity=equity, cash=self.cash, qty=self.qty, last_price=last_price)

    def rebalance_to_weight(self, symbol: str, target_weight: float, last_price: float) -> dict[str, Any]:
        state = self.get_account_state(symbol, last_price)
        target_notional = max(0.0, min(1.0, target_weight)) * state.equity
        current_notional = state.qty * last_price
        delta_notional = target_notional - current_notional
        if abs(delta_notional) < 1e-6:
            return {"status": "skipped", "reason": "already_at_target"}
        delta_qty = delta_notional / max(last_price, 1e-9)
        self.qty += delta_qty
        self.cash -= delta_qty * last_price
        print(
            f"[{_now_utc()}] PAPER ORDER {symbol}: delta_qty={delta_qty:.6f}, "
            f"new_qty={self.qty:.6f}, cash={self.cash:.2f}"
        )
        return {
            "status": "submitted",
            "broker": "paper",
            "symbol": symbol,
            "side": "buy" if delta_qty > 0 else "sell",
            "qty": float(abs(delta_qty)),
            "delta_notional": float(delta_notional),
            "target_weight": float(target_weight),
            "price": float(last_price),
        }


class AlpacaBroker(Broker):
    def __init__(self, api_key: str, api_secret: str, paper: bool) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        url = f"{self.base}{path}"
        body = None
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
        }
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, method=method, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            details = e.read().decode("utf-8")
            raise RuntimeError(f"Broker HTTP {e.code} for {path}: {details}") from e

    def get_account_state(self, symbol: str, last_price: float) -> AccountState:
        acc = self._request("GET", "/v2/account")
        pos_qty = 0.0
        try:
            pos = self._request("GET", f"/v2/positions/{symbol}")
            pos_qty = _safe_float(pos.get("qty", 0.0), 0.0)
        except RuntimeError:
            pos_qty = 0.0
        return AccountState(
            equity=_safe_float(acc.get("equity"), 0.0),
            cash=_safe_float(acc.get("cash"), 0.0),
            qty=pos_qty,
            last_price=last_price,
        )

    def rebalance_to_weight(self, symbol: str, target_weight: float, last_price: float) -> dict[str, Any]:
        state = self.get_account_state(symbol, last_price)
        target_notional = max(0.0, min(1.0, target_weight)) * state.equity
        current_notional = state.qty * last_price
        delta_notional = target_notional - current_notional
        if abs(delta_notional) < 5.0:
            print(f"[{_now_utc()}] Skip tiny order ({delta_notional:.2f} notional).")
            return {"status": "skipped", "reason": "tiny_notional", "delta_notional": float(delta_notional)}
        side = "buy" if delta_notional > 0 else "sell"
        qty = abs(delta_notional) / max(last_price, 1e-9)
        payload = {
            "symbol": symbol,
            "qty": str(round(qty, 6)),
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }
        order_resp = self._request("POST", "/v2/orders", payload=payload)
        print(
            f"[{_now_utc()}] ALPACA ORDER {symbol}: side={side}, qty={qty:.6f}, "
            f"target_weight={target_weight:.2f}"
        )
        return {
            "status": "submitted",
            "broker": "alpaca",
            "symbol": symbol,
            "side": side,
            "qty": float(qty),
            "delta_notional": float(delta_notional),
            "target_weight": float(target_weight),
            "price": float(last_price),
            "order_id": order_resp.get("id"),
            "order_status": order_resp.get("status"),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run live/paper regime trading loop.")
    p.add_argument("--symbol", required=True, help="Ticker symbol, e.g. SPY or AAPL.")
    p.add_argument(
        "--model",
        choices=("rslds", "hdp_arhmm", "hdp_slds"),
        default="hdp_slds",
        help="Model used each cycle for regime inference.",
    )
    p.add_argument("--interval", default="1m", help="Bar interval supported by yfinance.")
    p.add_argument("--lookback", default="5d", help="yfinance lookback period.")
    p.add_argument(
        "--poll-seconds",
        type=int,
        default=180,
        help="Polling interval in seconds (larger = calmer refits).",
    )
    p.add_argument(
        "--window-bars",
        type=int,
        default=480,
        help="Trailing bars per fit cycle (larger = stabler statistics).",
    )
    p.add_argument(
        "--n-iter",
        type=int,
        default=120,
        help="Gibbs iterations per cycle (more = closer to stationary at higher cost).",
    )
    p.add_argument(
        "--burn-in",
        type=int,
        default=50,
        help="Burn-in iterations per cycle.",
    )
    p.add_argument("--random-state", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--mode",
        choices=("paper", "alpaca-paper", "live"),
        default="paper",
        help="paper=local simulator, alpaca-paper=broker paper account, live=real account",
    )
    p.add_argument(
        "--confirm-live",
        action="store_true",
        help="Required when --mode live to avoid accidental real trading.",
    )
    p.add_argument(
        "--max-position-weight",
        type=float,
        default=0.5,
        help="Hard cap on gross long exposure fraction of equity.",
    )
    p.add_argument(
        "--max-daily-loss-pct",
        type=float,
        default=0.03,
        help="Kill switch if intraday equity drawdown exceeds this threshold.",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Run one cycle and exit (recommended for smoke testing).",
    )
    p.add_argument(
        "--events-file",
        default=os.path.join(_ROOT, "results", "live_trading", "live_events.jsonl"),
        help="JSONL log file used by monitoring dashboard.",
    )
    p.add_argument(
        "--state-dir",
        default=os.path.join(_ROOT, "results", "live_trading", "session_state"),
        help="Directory for paper account JSON and model warm-start .npz.",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore saved paper account and model warm-start; start fresh.",
    )
    p.add_argument(
        "--paper-starting-equity",
        type=float,
        default=10000.0,
        help="Initial cash when no saved paper state exists.",
    )
    p.add_argument(
        "--label-mode",
        choices=("rank", "smoothed_rank"),
        default="smoothed_rank",
        help=(
            "rank: bull/bear from mean log-return on this window only (most reactive). "
            "smoothed_rank: EMA of each state's window mean across cycles, then rank "
            "(less label churn; persisted under --state-dir)."
        ),
    )
    p.add_argument(
        "--label-mean-ema-alpha",
        type=float,
        default=0.65,
        help=(
            "With --label-mode smoothed_rank: weight on previous smoothed mean per state id "
            "(0..0.999). Higher = slower-moving labels."
        ),
    )
    p.add_argument(
        "--target-weight-ema-alpha",
        type=float,
        default=0.0,
        help=(
            "If >0, blend capped target weight with previous cycle's smoothed weight "
            "(0..0.999). Higher = less trading churn. 0 disables. Persisted per symbol/mode."
        ),
    )
    return p.parse_args()


def _build_broker(args: argparse.Namespace) -> Broker:
    if args.mode == "paper":
        paper_path = os.path.join(
            args.state_dir, f"paper_{_safe_symbol(args.symbol)}.json"
        )
        return PaperBroker(
            args.symbol,
            starting_equity=args.paper_starting_equity,
            state_file=None if args.no_resume else paper_path,
            resume=not args.no_resume,
        )

    if args.mode == "live" and not args.confirm_live:
        raise SystemExit("Refusing live mode without --confirm-live.")

    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_API_SECRET")
    if not key or not secret:
        raise SystemExit("Set ALPACA_API_KEY and ALPACA_API_SECRET in env.")

    return AlpacaBroker(api_key=key, api_secret=secret, paper=(args.mode == "alpaca-paper"))


def _norm_ts(t: Any) -> pd.Timestamp:
    """Timezone-naive timestamp for stable equality in dashboards."""
    ts = pd.Timestamp(t)
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


def _log_cycle(
    symbol: str,
    last_price: float,
    regime_id: int,
    regime_label: str,
    target_weight: float,
    eq: float,
    cash: float,
    qty: float,
) -> None:
    print(
        f"[{_now_utc()}] {symbol} px={last_price:.4f} z={regime_id} label={regime_label} "
        f"target_w={target_weight:.2f} equity={eq:.2f} cash={cash:.2f} qty={qty:.6f}"
    )


def _clear_session_ema_files(args: argparse.Namespace) -> None:
    """Used with --no-resume so smoothed label / target-weight EMA does not reuse old runs."""
    for p in (
        _label_mean_ema_path(args.state_dir, args.symbol, args.model),
        _target_weight_ema_path(args.state_dir, args.symbol, args.mode),
    ):
        try:
            os.remove(p)
        except OSError:
            pass


def main() -> None:
    args = parse_args()
    if args.no_resume:
        _clear_session_ema_files(args)
    broker = _build_broker(args)
    event_logger = EventLogger(args.events_file)
    day_start_equity: float | None = None
    # Previous cycle: regime we used for the then-latest bar (for retrospective compare).
    prior_decision_ts: pd.Timestamp | None = None
    prior_decision_regime_id: int | None = None
    prior_decision_regime_label: str | None = None
    prev_poll_latest_regime: int | None = None
    prev_poll_raw_w: float | None = None

    print(f"[{_now_utc()}] Starting live loop: symbol={args.symbol}, mode={args.mode}")
    event_logger.log(
        "startup",
        {
            "symbol": args.symbol,
            "model": args.model,
            "mode": args.mode,
            "interval": args.interval,
            "lookback": args.lookback,
            "poll_seconds": args.poll_seconds,
            "window_bars": args.window_bars,
            "n_iter": args.n_iter,
            "burn_in": args.burn_in,
            "max_position_weight": args.max_position_weight,
            "max_daily_loss_pct": args.max_daily_loss_pct,
            "state_dir": args.state_dir,
            "resume": not args.no_resume,
            "label_mode": args.label_mode,
            "label_mean_ema_alpha": args.label_mean_ema_alpha,
            "target_weight_ema_alpha": args.target_weight_ema_alpha,
        },
    )

    while True:
        try:
            bars = _download_bars(args.symbol, args.interval, args.lookback)
            feat = _build_live_features(bars)
            feat = feat.tail(args.window_bars).copy()
            rets = feat["price_log_return"].to_numpy(dtype=float)
            y = feat.to_numpy(dtype=float)
            y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-8)
            d_obs = y.shape[1]
            bar_times = [_norm_ts(t).isoformat() for t in feat.index]

            warm_path = os.path.join(
                args.state_dir, f"warmstart_{args.model}_{_safe_symbol(args.symbol)}.npz"
            )
            warm_z = None
            warm_x = None
            if not args.no_resume:
                ck = _load_warmstart_npz(warm_path)
                if (
                    ck
                    and ck["model_name"] == args.model
                    and ck["obs_dim"] == d_obs
                    and ck["window_bars"] == args.window_bars
                ):
                    rng_ws = np.random.default_rng(args.random_state)
                    cap = _z_truncate_cap(args.model, d_obs)
                    warm_z, warm_x = _align_warm_latent(
                        ck["bar_times"],
                        ck["z"],
                        ck["x"],
                        bar_times,
                        cap,
                        d_obs,
                        rng_ws,
                    )
                    if args.model == "hdp_arhmm":
                        warm_x = None
                    elif args.model in ("hdp_slds", "rslds") and warm_x is None:
                        warm_z = None

            z, persist_fit = _fit_and_infer_last_regime(
                model_name=args.model,
                y=y,
                n_iter=args.n_iter,
                burn_in=args.burn_in,
                random_state=args.random_state,
                warm_z=warm_z,
                warm_x=warm_x,
            )
            rets_z = rets[-len(z) :]
            if args.label_mode == "smoothed_rank":
                label_map, regime_stats = _classify_regimes_smoothed_mean_ema(
                    z,
                    rets_z,
                    alpha=args.label_mean_ema_alpha,
                    ema_path=_label_mean_ema_path(
                        args.state_dir, args.symbol, args.model
                    ),
                    symbol=args.symbol,
                    model=args.model,
                )
            else:
                label_map, regime_stats = _classify_regimes_bull_bear_neutral(z, rets_z)
            if len(z) != len(feat):
                raise RuntimeError(
                    f"Inferred z length {len(z)} != feature rows {len(feat)}."
                )
            latest_regime = int(z[-1])
            regime_label = label_map.get(latest_regime, "neutral")

            # Aligned closes + per-bar semantic labels for dashboard plotting.
            closes = bars["Close"].reindex(feat.index).astype(float)
            if closes.isna().any():
                closes = closes.ffill().bfill()
            close_list = [float(x) for x in closes.to_numpy().tolist()]
            z_list = [int(x) for x in np.asarray(z, dtype=int).ravel().tolist()]
            win_labels = [label_map.get(zi, "neutral") for zi in z_list]
            label_map_json = {str(k): v for k, v in label_map.items()}

            # Retrospective: same calendar bar, new fit — does z at that bar change?
            norm_index = [_norm_ts(t) for t in feat.index]
            rev_match: bool | None = None
            rev_new_id: int | None = None
            rev_new_label: str | None = None
            if (
                prior_decision_ts is not None
                and prior_decision_regime_id is not None
            ):
                for i, t in enumerate(norm_index):
                    if t == prior_decision_ts:
                        rev_new_id = int(z_list[i])
                        rev_new_label = label_map.get(rev_new_id, "neutral")
                        rev_match = rev_new_id == prior_decision_regime_id
                        break

            last_price = float(bars["Close"].iloc[-1])
            state = broker.get_account_state(args.symbol, last_price)
            if day_start_equity is None:
                day_start_equity = state.equity
            dd = 0.0
            if day_start_equity > 1e-9:
                dd = (state.equity - day_start_equity) / day_start_equity
            tw_ema_path = _target_weight_ema_path(
                args.state_dir, args.symbol, args.mode
            )
            if dd <= -abs(args.max_daily_loss_pct):
                print(
                    f"[{_now_utc()}] KILL SWITCH: daily drawdown {dd:.2%} "
                    f"breached max {abs(args.max_daily_loss_pct):.2%}."
                )
                target_weight = 0.0
                raw_w = 0.0
                kill_switch = True
                if args.target_weight_ema_alpha > 0 and not args.no_resume:
                    _save_target_weight_ema(tw_ema_path, 0.0)
            else:
                raw_w = _target_weight_from_regime(
                    regime_label,
                    latest_regime,
                    label_map,
                    regime_stats,
                )
                tw_capped = min(args.max_position_weight, raw_w)
                if args.target_weight_ema_alpha > 0:
                    alpha_tw = float(
                        np.clip(args.target_weight_ema_alpha, 0.0, 0.999)
                    )
                    prev_tw = _load_target_weight_ema_prev(tw_ema_path)
                    if prev_tw is None:
                        target_weight = float(tw_capped)
                    else:
                        target_weight = float(
                            alpha_tw * prev_tw + (1.0 - alpha_tw) * tw_capped
                        )
                    target_weight = float(
                        max(0.0, min(args.max_position_weight, target_weight))
                    )
                    if not args.no_resume:
                        _save_target_weight_ema(tw_ema_path, target_weight)
                else:
                    target_weight = float(tw_capped)
                kill_switch = False

            weight_diag = _explain_weight_vs_regime_id(
                latest_regime,
                regime_label,
                label_map,
                regime_stats,
                raw_w,
                target_weight,
                args.max_position_weight,
                kill_switch,
            )
            weight_diag["label_mode"] = args.label_mode
            if args.label_mode == "smoothed_rank":
                weight_diag["label_mean_ema_alpha"] = float(args.label_mean_ema_alpha)
            if args.target_weight_ema_alpha > 0 and not kill_switch:
                weight_diag["target_weight_ema_alpha"] = float(
                    args.target_weight_ema_alpha
                )
                weight_diag["target_weight_pre_ema_cap"] = float(
                    min(args.max_position_weight, raw_w)
                )
            if (
                not kill_switch
                and prev_poll_latest_regime is not None
                and prev_poll_raw_w is not None
                and prev_poll_latest_regime == latest_regime
                and abs(prev_poll_raw_w - raw_w) > 1e-6
            ):
                print(
                    f"[{_now_utc()}] Weight moved with same z[-1]={latest_regime}: "
                    f"raw {prev_poll_raw_w:.4f} -> {raw_w:.4f} "
                    f"(semantic label={regime_label!r}; means/labels are recomputed each window)."
                )
            prev_poll_latest_regime = int(latest_regime)
            prev_poll_raw_w = float(raw_w)

            _log_cycle(
                args.symbol,
                last_price,
                latest_regime,
                regime_label,
                target_weight,
                state.equity,
                state.cash,
                state.qty,
            )
            cycle_payload: dict[str, Any] = {
                "symbol": args.symbol,
                "mode": args.mode,
                "model": args.model,
                "last_price": float(last_price),
                "latest_regime": int(latest_regime),
                "regime_label": regime_label,
                "target_weight": float(target_weight),
                "equity_before": float(state.equity),
                "cash_before": float(state.cash),
                "qty_before": float(state.qty),
                "daily_drawdown": float(dd),
                "kill_switch": bool(kill_switch),
                "bar_time": str(bars.index[-1]),
                "decision_bar_time": bar_times[-1],
                "raw_target_weight": float(raw_w),
                "neutral_weight_note": (
                    "Neutrals use mean-return interpolation between bear and bull means (capped by max-position-weight)."
                    if regime_label == "neutral"
                    else None
                ),
                "predicted_regime_next_action": {
                    "regime_id": int(latest_regime),
                    "regime_label": regime_label,
                    "note": "Regime for the latest bar in the window; used until next refit.",
                },
                "window_bar_times": bar_times,
                "window_close": close_list,
                "window_z": z_list,
                "window_regime_label": win_labels,
                "regime_label_map": label_map_json,
                "weight_vs_regime_explanation": weight_diag,
                "label_mode": args.label_mode,
                "label_mean_ema_alpha": float(args.label_mean_ema_alpha)
                if args.label_mode == "smoothed_rank"
                else None,
                "target_weight_ema_alpha": float(args.target_weight_ema_alpha)
                if args.target_weight_ema_alpha > 0
                else None,
            }
            if prior_decision_ts is not None:
                cycle_payload["prior_decision_bar_time"] = prior_decision_ts.isoformat()
                cycle_payload["prior_predicted_regime_id"] = prior_decision_regime_id
                cycle_payload["prior_predicted_regime_label"] = prior_decision_regime_label
            if rev_match is not None:
                cycle_payload["retrospective_same_bar"] = {
                    "bar_time": prior_decision_ts.isoformat() if prior_decision_ts else None,
                    "regime_id_last_cycle": prior_decision_regime_id,
                    "regime_id_this_fit": rev_new_id,
                    "regime_label_this_fit": rev_new_label,
                    "matches_prior_regime_id": rev_match,
                    "note": "After a new bar arrives, Gibbs refit can change z on older bars in the window.",
                }
            elif prior_decision_ts is not None:
                cycle_payload["retrospective_same_bar"] = {
                    "available": False,
                    "reason": "Prior decision bar no longer in this rolling window.",
                }

            event_logger.log("cycle_status", cycle_payload)

            prior_decision_ts = norm_index[-1]
            prior_decision_regime_id = int(latest_regime)
            prior_decision_regime_label = regime_label
            order_info = broker.rebalance_to_weight(args.symbol, target_weight, last_price)
            state_after = broker.get_account_state(args.symbol, last_price)
            event_logger.log(
                "order",
                {
                    "symbol": args.symbol,
                    "mode": args.mode,
                    "model": args.model,
                    "latest_regime": int(latest_regime),
                    "regime_label": regime_label,
                    "target_weight": float(target_weight),
                    "equity_after": float(state_after.equity),
                    "cash_after": float(state_after.cash),
                    "qty_after": float(state_after.qty),
                    "order": order_info,
                },
            )

            if not args.no_resume:
                lx = persist_fit.get("last_x")
                _save_warmstart_npz(
                    warm_path,
                    bar_times=bar_times,
                    z=np.asarray(z, dtype=int),
                    x=lx if args.model in ("hdp_slds", "rslds") else None,
                    model_name=args.model,
                    obs_dim=d_obs,
                    window_bars=args.window_bars,
                )
            if args.mode == "paper" and isinstance(broker, PaperBroker):
                broker.save_state()

        except Exception as exc:  # noqa: BLE001
            print(f"[{_now_utc()}] Cycle error: {exc}")
            event_logger.log(
                "error",
                {
                    "symbol": args.symbol,
                    "mode": args.mode,
                    "model": args.model,
                    "error": str(exc),
                },
            )

        if args.once:
            break
        time.sleep(max(1, args.poll_seconds))


if __name__ == "__main__":
    main()
