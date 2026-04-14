"""
Regime-based strategy backtest vs. buy-and-hold baseline.

Maps inferred discrete regimes to Bull / Bear / Neutral from in-sample mean returns
(per regime), applies a one-day lag on positions to avoid lookahead, and reports
standard performance metrics plus an equity curve comparison plot.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from numpy.typing import ArrayLike

# Semantic regime colours (background shading)
REGIME_FILL: dict[str, str] = {
    "bull": "#b8e0b0",
    "bear": "#f5b5b5",
    "neutral": "#d9d9d9",
    "unknown": "#ececec",
}

# Distinct pastels for each discrete state labeled "neutral" (middle mean-return ranks)
NEUTRAL_PALETTE: list[str] = [
    "#d9d9d9",
    "#c9c9e8",
    "#d4c4eb",
    "#c4e0e8",
    "#e0d4c4",
    "#cde8d4",
    "#e8d4dc",
    "#dcd4e8",
    "#d8e8c4",
]


def _neutral_ids_sorted(label_map: dict[int, str]) -> list[int]:
    return sorted(z for z, lb in label_map.items() if lb == "neutral")


def facecolor_for_regime_id(z_id: int, label_map: dict[int, str]) -> str:
    """Background colour for discrete regime ``z_id`` given semantic ``label_map``."""
    lbl = label_map.get(int(z_id), "neutral")
    if lbl == "bull":
        return REGIME_FILL["bull"]
    if lbl == "bear":
        return REGIME_FILL["bear"]
    if lbl == "neutral":
        neutrals = _neutral_ids_sorted(label_map)
        if not neutrals:
            return REGIME_FILL["neutral"]
        try:
            idx = neutrals.index(int(z_id))
        except ValueError:
            return REGIME_FILL["neutral"]
        return NEUTRAL_PALETTE[idx % len(NEUTRAL_PALETTE)]
    return REGIME_FILL["unknown"]


def regime_shading_legend_patches(label_map: dict[int, str], alpha: float) -> list[Patch]:
    """One legend entry per discrete state (neutrals use distinct colours)."""
    patches: list[Patch] = []
    for z in sorted(label_map.keys()):
        lb = label_map[z]
        c = facecolor_for_regime_id(z, label_map)
        if lb == "bull":
            txt = f"Bull (z={z})"
        elif lb == "bear":
            txt = f"Bear (z={z})"
        elif lb == "neutral":
            txt = f"Neutral (z={z})"
        else:
            txt = f"z={z}"
        patches.append(Patch(facecolor=c, alpha=alpha, edgecolor="none", label=txt))
    return patches


import matplotlib.patches as mpatches

def add_regime_shading_to_axis(
    ax, dates, regimes, label_map, alpha=0.3
) -> list:
    """
    Shade the background of a matplotlib axis based on discrete regimes.
    Assigns distinct colors to Bull, Bear, and rotates through a palette for multiple Neutrals.
    """
    # 1. Define our distinct colors
    color_palette = {
        "bear": "#ff9999",  # Light Red
        "bull": "#99ff99",  # Light Green
    }
    

    # High-contrast, easily distinguishable colors for small areas
    neutral_colors = [
        "#3498db",  # Strong Blue
        "#f39c12",  # Vibrant Orange/Gold
        "#9b59b6",  # Amethyst Purple
        "#1abc9c",  # Turquoise / Teal
        "#e84393",  # Deep Pink / Magenta
        "#7f8c8d",  # Slate Gray
    ]
    
    # 2. Map every specific regime ID to a unique color
    regime_to_color = {}
    neutral_idx = 0
    
    for regime_id, label in label_map.items():
        if label == "neutral":
            # Assign the next available neutral color, wrapping around if we have many
            regime_to_color[regime_id] = neutral_colors[neutral_idx % len(neutral_colors)]
            neutral_idx += 1
        else:
            regime_to_color[regime_id] = color_palette.get(label, "#cccccc")

    # 3. Find contiguous blocks of regimes to draw the background spans
    patches = {}
    n = len(regimes)
    if n == 0:
        return []
        
    start_idx = 0
    current_regime = regimes[0]
    
    for i in range(1, n):
        if regimes[i] != current_regime:
            # Draw the block that just ended
            color = regime_to_color.get(current_regime, "#cccccc")
            ax.axvspan(dates[start_idx], dates[i], color=color, alpha=alpha, lw=0)
            
            # Keep track of patches for the legend
            if current_regime not in patches:
                lbl = f"Regime {current_regime} ({label_map.get(current_regime, 'unknown')})"
                patches[current_regime] = mpatches.Patch(color=color, alpha=alpha, label=lbl)
                
            start_idx = i
            current_regime = regimes[i]
            
    # Draw the final block
    color = regime_to_color.get(current_regime, "#cccccc")
    ax.axvspan(dates[start_idx], dates[-1], color=color, alpha=alpha, lw=0)
    if current_regime not in patches:
        lbl = f"Regime {current_regime} ({label_map.get(current_regime, 'unknown')})"
        patches[current_regime] = mpatches.Patch(color=color, alpha=alpha, label=lbl)

    # Return the patches sorted by regime ID so the legend looks organized
    return [patches[k] for k in sorted(patches.keys())]


def _regime_stats(regimes: np.ndarray, returns: np.ndarray) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for u in np.unique(regimes):
        mask = regimes == u
        r = returns[mask]
        out[int(u)] = {
            "mean": float(np.mean(r)) if r.size else 0.0,
            "var": float(np.var(r, ddof=0)) if r.size else 0.0,
            "std": float(np.std(r, ddof=0)) if r.size else 0.0,
            "n": int(r.size),
        }
    return out


def _classify_regimes_bull_bear_neutral(
    regimes: np.ndarray, returns: np.ndarray
) -> tuple[dict[int, str], dict[int, dict[str, float]]]:
    """
    Label each regime id as 'bull', 'bear', or 'neutral' using mean return rank.
    Ties in mean are broken by lower variance -> more bullish ordering (calmer bull).
    """
    stats = _regime_stats(regimes, returns)
    # Rank by mean return; for equal means, higher variance sorts earlier (bear / stress).
    unique = sorted(stats.keys(), key=lambda u: (stats[u]["mean"], -stats[u]["var"]))
    n = len(unique)
    label_map: dict[int, str] = {}

    if n == 0:
        return label_map, stats
    if n == 1:
        label_map[unique[0]] = "neutral"
        return label_map, stats
    if n == 2:
        label_map[unique[0]] = "bear"
        label_map[unique[1]] = "bull"
        return label_map, stats

    label_map[unique[0]] = "bear"
    label_map[unique[-1]] = "bull"
    for u in unique[1:-1]:
        label_map[u] = "neutral"
    return label_map, stats


def _position_from_label(label: str) -> float:
    if label == "bull":
        return 1.0
    if label == "bear":
        return 0.0
    return 0.5


def _compute_metrics(
    daily_returns: np.ndarray, risk_free_rate: float, trading_days: int = 252
) -> dict[str, float]:
    r = np.asarray(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {
            "cumulative_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }

    equity = np.cumprod(1.0 + r)
    cum_ret = float(equity[-1] - 1.0)

    daily_rf = risk_free_rate / trading_days
    excess = r - daily_rf
    std_d = float(np.std(excess, ddof=0))
    ann_vol = std_d * np.sqrt(trading_days)
    mean_d = float(np.mean(excess))
    sharpe = (mean_d / std_d * np.sqrt(trading_days)) if std_d > 1e-12 else 0.0

    peak = np.maximum.accumulate(equity)
    dd = np.where(peak > 0, (equity - peak) / peak, 0.0)
    max_dd = float(np.min(dd)) if dd.size else 0.0

    return {
        "cumulative_return": cum_ret,
        "annualized_volatility": float(ann_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_dd,
    }


def simulate_regime_strategy(
    dates: ArrayLike,
    asset_returns: ArrayLike,
    inferred_regimes: ArrayLike,
    risk_free_rate: float = 0.0,
    *,
    trading_days: int = 252,
    save_path: str | None = None,
    show_plot: bool = False,
) -> dict[str, Any]:
    """
    Backtest a lagged regime-position strategy vs. buy-and-hold.

    Regimes are labeled Bull / Bear / Neutral by ranking each regime's in-sample
    mean daily return (variance breaks ties: lower var ranks earlier within same mean).

    Positions use **yesterday's** regime (no lookahead): ``position[t]`` follows
    ``regime[t-1]``; ``position[0] = 0`` (cash until one observation exists).

    Parameters
    ----------
    dates : array-like, length T
        Calendar labels for plotting (e.g. datetime64 or strings).
    asset_returns : array-like, length T
        Simple or log daily returns of the traded asset; strategy uses simple
        compounding ``position * return`` (pass simple returns for standard interpretation).
    inferred_regimes : array-like, length T
        Discrete regime index per day (same length as returns).
    risk_free_rate : float
        Annualized risk-free rate for Sharpe (default 0).
    trading_days : int
        Days per year for annualization (default 252).
    save_path : str, optional
        If set, save the figure to this path (e.g. ``results/regime_backtest.png``).
    show_plot : bool
        If True, call ``plt.show()`` after saving (or if no save_path).

    Returns
    -------
    dict with keys:
        ``regime_label_map`` : id -> 'bull'|'bear'|'neutral'
        ``regime_stats`` : id -> {mean, var, std, n}
        ``positions`` : (T,) lagged positions
        ``strategy_returns`` : (T,) daily strategy returns
        ``strategy_metrics``, ``buy_hold_metrics`` : performance dicts
        ``equity_strategy``, ``equity_buy_hold`` : (T,) cumulative equity (start 1.0)
    """
    r = np.asarray(asset_returns, dtype=float).ravel()
    z = np.asarray(inferred_regimes, dtype=int).ravel()
    d = np.asarray(dates)

    if r.shape != z.shape:
        raise ValueError(
            f"asset_returns and inferred_regimes must match; got {r.shape} vs {z.shape}."
        )
    if d.shape[0] != r.shape[0]:
        raise ValueError(
            f"dates length {d.shape[0]} must equal returns length {r.shape[0]}."
        )

    label_map, stats = _classify_regimes_bull_bear_neutral(z, r)

    raw_pos = np.array([_position_from_label(label_map.get(int(zi), "neutral")) for zi in z])
    position = np.zeros_like(raw_pos, dtype=float)
    position[1:] = raw_pos[:-1]

    strat_r = position * r
    bh_r = r.copy()

    eq_s = np.cumprod(1.0 + strat_r)
    eq_bh = np.cumprod(1.0 + bh_r)

    strat_m = _compute_metrics(strat_r, risk_free_rate, trading_days)
    bh_m = _compute_metrics(bh_r, risk_free_rate, trading_days)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    x = np.arange(len(r), dtype=float)
    try:
        x_plot = np.asarray(d)
    except Exception:
        x_plot = x

    shade_alpha = 0.5
    regime_patches = add_regime_shading_to_axis(
        axes[0], x_plot, z, label_map, alpha=shade_alpha
    )
    add_regime_shading_to_axis(axes[1], x_plot, z, label_map, alpha=shade_alpha)

    axes[0].plot(x_plot, eq_s, color="#1a5276", lw=1.4, zorder=3, label="Regime strategy")
    axes[0].set_title("Regime strategy (lagged positions)")
    axes[0].set_ylabel("Cumulative equity ($1 start)")
    axes[0].grid(True, alpha=0.35, zorder=2)
    axes[0].legend(loc="upper left", framealpha=0.92)

    axes[1].plot(x_plot, eq_bh, color="#a04000", lw=1.4, zorder=3, label="Buy & hold")
    axes[1].set_title("Buy & hold (same regime shading = inferred state)")
    axes[1].grid(True, alpha=0.35, zorder=2)
    axes[1].legend(loc="upper left", framealpha=0.92)

    for ax in axes:
        dt = getattr(x_plot, "dtype", None)
        if dt is not None and str(dt).startswith("datetime64"):
            ax.tick_params(axis="x", rotation=30)
        ax.set_xlabel("Date")

    fig.suptitle(
        "Equity curves (background: green bull / red bear / distinct hues per neutral state z)",
        y=1.03,
    )
    n_leg = len(regime_patches)
    ncol_leg = min(4, max(1, n_leg))
    nrows_leg = (n_leg + ncol_leg - 1) // ncol_leg
    fig.legend(
        handles=regime_patches,
        loc="lower center",
        ncol=ncol_leg,
        frameon=True,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14 + 0.045 * max(0, nrows_leg - 1))
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        "regime_label_map": label_map,
        "regime_stats": stats,
        "positions": position,
        "strategy_returns": strat_r,
        "strategy_metrics": strat_m,
        "buy_hold_metrics": bh_m,
        "equity_strategy": eq_s,
        "equity_buy_hold": eq_bh,
    }
