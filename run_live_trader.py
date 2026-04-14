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

from src.backtest import _classify_regimes_bull_bear_neutral
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


def _fit_and_infer_last_regime(
    model_name: str,
    y: np.ndarray,
    n_iter: int,
    burn_in: int,
    random_state: int,
) -> np.ndarray:
    t_total, d_obs = y.shape
    if t_total < 80:
        raise RuntimeError(f"Need at least 80 samples, got {t_total}.")

    if model_name == "rslds":
        k = 3
        x_init, z_init, w_init, r_init = initialize_rslds(
            y,
            K=k,
            state_dim=d_obs,
            ar_order=1,
            n_gibbs_ar=max(25, n_iter // 2),
            random_state=random_state,
            standardize=False,
        )
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
        return np.asarray(hist["last_z"], dtype=int).ravel()

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
        hist = model.gibbs(y, n_iters=n_iter, burn_in=burn_in)
        return np.asarray(hist["last_z"], dtype=int).ravel()

    L = min(10, max(5, d_obs + 3))
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
    hist = model.gibbs(y, n_iters=n_iter, burn_in=burn_in)
    return np.asarray(hist["last_z"], dtype=int).ravel()


def _label_to_target_weight(label: str) -> float:
    if label == "bull":
        return 1.0
    if label == "bear":
        return 0.0
    return 0.5


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
    def __init__(self, starting_equity: float = 10000.0) -> None:
        self.cash = float(starting_equity)
        self.qty = 0.0

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
    p.add_argument("--poll-seconds", type=int, default=60, help="Polling interval in seconds.")
    p.add_argument("--window-bars", type=int, default=240, help="Trailing bars per fit cycle.")
    p.add_argument("--n-iter", type=int, default=60, help="Gibbs iterations per cycle.")
    p.add_argument("--burn-in", type=int, default=25, help="Burn-in iterations per cycle.")
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
    return p.parse_args()


def _build_broker(args: argparse.Namespace) -> Broker:
    if args.mode == "paper":
        return PaperBroker()

    if args.mode == "live" and not args.confirm_live:
        raise SystemExit("Refusing live mode without --confirm-live.")

    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_API_SECRET")
    if not key or not secret:
        raise SystemExit("Set ALPACA_API_KEY and ALPACA_API_SECRET in env.")

    return AlpacaBroker(api_key=key, api_secret=secret, paper=(args.mode == "alpaca-paper"))


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


def main() -> None:
    args = parse_args()
    broker = _build_broker(args)
    event_logger = EventLogger(args.events_file)
    day_start_equity: float | None = None

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

            z = _fit_and_infer_last_regime(
                model_name=args.model,
                y=y,
                n_iter=args.n_iter,
                burn_in=args.burn_in,
                random_state=args.random_state,
            )
            label_map, _stats = _classify_regimes_bull_bear_neutral(z, rets[-len(z):])
            latest_regime = int(z[-1])
            regime_label = label_map.get(latest_regime, "neutral")

            last_price = float(bars["Close"].iloc[-1])
            state = broker.get_account_state(args.symbol, last_price)
            if day_start_equity is None:
                day_start_equity = state.equity
            dd = 0.0
            if day_start_equity > 1e-9:
                dd = (state.equity - day_start_equity) / day_start_equity
            if dd <= -abs(args.max_daily_loss_pct):
                print(
                    f"[{_now_utc()}] KILL SWITCH: daily drawdown {dd:.2%} "
                    f"breached max {abs(args.max_daily_loss_pct):.2%}."
                )
                target_weight = 0.0
                kill_switch = True
            else:
                target_weight = min(
                    args.max_position_weight,
                    _label_to_target_weight(regime_label),
                )
                kill_switch = False
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
            event_logger.log(
                "cycle_status",
                {
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
                },
            )
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
