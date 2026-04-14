#!/usr/bin/env python3
"""
AI61004 master pipeline: load data → fit a switching model → backtest + figures.

Models (``--model``):
  - ``rslds`` — Recurrent SLDS (+ Linderman-style init, W plot, latent plot)
  - ``hdp_arhmm`` — sticky HDP-AR-HMM (Fox et al.; truncation level ``--L``)
  - ``hdp_slds`` — sticky HDP-SLDS (latent plot; no W plot)

Examples:
  python run_master_backtest.py --model rslds --start 2020-01-01 --end 2022-12-31
  python run_master_backtest.py --model hdp_arhmm --full
  python run_master_backtest.py --model hdp_slds --n-iter 80 --burn-in 40
  python run_master_backtest.py --model hdp_slds --full --n-iter 500 --burn-in 200
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Project root on path when launched as python run_master_backtest.py
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.backtest import add_regime_shading_to_axis, simulate_regime_strategy
from src.hdp_arhmm import StickyHDPARHMM
from src.hdp_slds import StickyHDPSLDS
from src.initialization import apply_rslds_initialization, initialize_rslds
from src.recurrent_slds import RecurrentSLDS


def _load_and_slice(
    csv_path: str,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    if start is not None:
        df = df[df["Date"] >= start]
    if end is not None:
        df = df[df["Date"] <= end]
    return df.reset_index(drop=True)


def plot_recurrence_weights(
    W: np.ndarray,
    r: np.ndarray,
    feature_names: list[str],
    regime_label_map: dict[int, str],
    save_path: str,
) -> None:
    """
    Visualize stick-breaking recurrence weights W[j, k, d] (source state j, logit k, feature d).
    """
    K, k1, d = W.shape
    fig, axes = plt.subplots(K, 1, figsize=(max(10, d * 0.5), 2.8 * K), squeeze=False)
    vmax = float(np.nanmax(np.abs(W))) + 1e-8
    for j in range(K):
        ax = axes[j, 0]
        im = ax.imshow(W[j], aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_yticks(range(k1))
        ax.set_yticklabels([f"logit {k}" for k in range(k1)])
        ax.set_xticks(range(d))
        ax.set_xticklabels(feature_names, rotation=35, ha="right")
        label = regime_label_map.get(j, "?")
        ax.set_title(f"W[z_prev={j}] ({label}) — rows = stick logits, cols = features")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.suptitle("Recurrent SLDS: recurrence weights W (nu = W @ x + r)", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_latent_vs_returns(
    dates: np.ndarray,
    latent_dim0: np.ndarray,
    asset_returns: np.ndarray,
    save_path: str,
    *,
    inferred_regimes: np.ndarray | None = None,
    regime_label_map: dict[int, str] | None = None,
) -> None:
    """First latent dimension vs. raw asset returns (aligned time index)."""
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    d_arr = np.asarray(dates)
    z_arr = (
        np.asarray(inferred_regimes, dtype=int).ravel()
        if inferred_regimes is not None
        else None
    )
    shade_alpha = 0.35
    regime_patches = None
    if (
        z_arr is not None
        and regime_label_map is not None
        and z_arr.shape[0] == d_arr.shape[0]
    ):
        regime_patches = add_regime_shading_to_axis(
            ax1, d_arr, z_arr, regime_label_map, alpha=shade_alpha
        )
        add_regime_shading_to_axis(ax2, d_arr, z_arr, regime_label_map, alpha=shade_alpha)

    ax1.plot(
        dates,
        asset_returns,
        color="#1a5276",
        lw=0.75,
        alpha=0.95,
        zorder=3,
        label="Asset return",
    )
    ax1.set_ylabel("Return (as in CSV)")
    ax1.set_xlabel("Date")
    ax1.grid(True, alpha=0.35, zorder=2)
    ax2.plot(
        dates,
        latent_dim0,
        color="#a04000",
        lw=0.9,
        alpha=0.95,
        zorder=3,
        label=r"$x_t^{(0)}$",
    )
    ax2.set_ylabel(r"Latent $x_t$ (dim 0)")
    ax1.legend(loc="upper left", framealpha=0.92)
    ax2.legend(loc="upper right", framealpha=0.92)
    fig.suptitle(
        "Latent vs. returns (background: bull / bear / distinct hues per neutral state z)",
        y=1.02,
    )
    if regime_patches:
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
    if regime_patches:
        fig.subplots_adjust(bottom=0.12 + 0.045 * max(0, nrows_leg - 1))
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def summarize_w_bear_features(
    W: np.ndarray,
    feature_names: list[str],
    regime_label_map: dict[int, str],
) -> None:
    """Print features with largest |W| for source states labeled bear (heuristic narrative)."""
    for j in range(W.shape[0]):
        if regime_label_map.get(j) != "bear":
            continue
        mag = np.sum(np.abs(W[j]), axis=0)
        order = np.argsort(-mag)
        print(f"\nTop |W| features when z_prev={j} (bear source state):")
        for rank, d in enumerate(order[: min(5, len(order))], 1):
            print(f"  {rank}. {feature_names[d]}: sum_k |W[j,k,d]| = {mag[d]:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Master switching-model + backtest pipeline for AI61004."
    )
    p.add_argument(
        "--model",
        choices=("rslds", "hdp_arhmm", "hdp_slds"),
        default="rslds",
        help="Switching model: recurrent SLDS, sticky HDP-AR-HMM, or sticky HDP-SLDS.",
    )
    p.add_argument("--csv", default=os.path.join(_ROOT, "data", "market_features.csv"))
    p.add_argument(
        "--full",
        action="store_true",
        help="Use full CSV date range (ignore --start/--end).",
    )
    p.add_argument("--start", default="2020-01-01", help="Inclusive start date (YYYY-MM-DD).")
    p.add_argument("--end", default="2022-12-31", help="Inclusive end date (YYYY-MM-DD).")
    p.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of discrete states (RecurrentSLDS only).",
    )
    p.add_argument(
        "--L",
        type=int,
        default=None,
        help="HDP weak-limit truncation (hdp_arhmm / hdp_slds). Default: from D.",
    )
    p.add_argument(
        "--n-gibbs-ar",
        type=int,
        default=50,
        help="Gibbs iters for switching AR init (rslds only).",
    )
    p.add_argument("--n-iter", type=int, default=100, help="Gibbs iterations.")
    p.add_argument("--burn-in", type=int, default=50, help="Burn-in iterations.")
    p.add_argument("--risk-free", type=float, default=0.05, help="Annual risk-free rate for Sharpe.")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--show",
        action="store_true",
        help="Call plt.show() on the backtest figure (blocks in non-GUI environments).",
    )
    p.add_argument(
        "--asset-col",
        default=None,
        help="Column name for backtest returns (default: first numeric column after Date).",
    )
    p.add_argument(
        "--hdp-legacy",
        action="store_true",
        help=(
            "HDP-SLDS only: use old collapsed Dirichlet transitions only, fixed α/γ/κ, "
            "no CRT auxiliary for β, no label canonicalization."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.join(_ROOT, "results"), exist_ok=True)

    start = None if args.full else pd.Timestamp(args.start)
    end = None if args.full else pd.Timestamp(args.end)

    df = _load_and_slice(args.csv, start, end)
    if df.shape[0] < 80:
        raise SystemExit(
            f"Too few rows after slice ({df.shape[0]}). Widen the window or use --full."
        )

    dates = df["Date"].to_numpy()
    numeric_cols = [c for c in df.columns if c != "Date" and np.issubdtype(df[c].dtype, np.number)]
    if not numeric_cols:
        raise SystemExit("No numeric columns besides Date.")

    asset_col = args.asset_col or numeric_cols[0]
    if asset_col not in df.columns:
        raise SystemExit(f"Unknown --asset-col {asset_col!r}. Available: {numeric_cols}")

    asset_returns = df[asset_col].to_numpy(dtype=float)
    y = df[numeric_cols].to_numpy(dtype=float)
    y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-8)

    t_total, d_obs = y.shape
    k = args.k
    state_dim = d_obs
    tag = args.model

    print(f"Loaded {t_total} days ({df['Date'].min().date()} → {df['Date'].max().date()}).")
    print(f"Model: {args.model}")
    print(f"Features (D={d_obs}): {numeric_cols}")
    print(f"Backtest return column: {asset_col}")

    x_last: np.ndarray | None = None
    rslds: RecurrentSLDS | None = None

    if args.model == "rslds":
        print("Initializing rSLDS (PCA + switching AR + logistic W,r)...")
        x_init, z_init, w_init, r_init = initialize_rslds(
            y,
            K=k,
            state_dim=state_dim,
            ar_order=1,
            n_gibbs_ar=args.n_gibbs_ar,
            random_state=args.random_state,
            standardize=False,
        )
        rslds = RecurrentSLDS(
            K=k,
            state_dim=state_dim,
            obs_dim=d_obs,
            pg_trunc=200,
            random_state=args.random_state,
        )
        apply_rslds_initialization(rslds, w_init, r_init)
        print(f"Running RecurrentSLDS Gibbs ({args.n_iter} iters, burn_in={args.burn_in})...")
        hist = rslds.gibbs(
            y,
            n_iters=args.n_iter,
            burn_in=args.burn_in,
            x_init=x_init,
            z_init=z_init,
        )
        z_inferred = hist["last_z"]
        x_last = hist["last_x"]

    elif args.model == "hdp_arhmm":
        L = args.L if args.L is not None else min(12, max(6, d_obs + 4))
        print(f"Sticky HDP-AR-HMM with truncation L={L}...")
        model = StickyHDPARHMM(
            L=L,
            D=d_obs,
            ar_order=1,
            alpha=6.0,
            gamma=2.0,
            kappa=20.0,
            random_state=args.random_state,
        )
        hist = model.gibbs(y, n_iters=args.n_iter, burn_in=args.burn_in)
        z_inferred = hist["last_z"]

    else:
        L = args.L if args.L is not None else min(10, max(5, d_obs + 3))
        print(f"Sticky HDP-SLDS with truncation L={L}, state_dim={d_obs}...")
        legacy = args.hdp_legacy
        model = StickyHDPSLDS(
            L=L,
            state_dim=d_obs,
            obs_dim=d_obs,
            alpha=6.0,
            gamma=2.0,
            kappa=20.0,
            random_state=args.random_state,
            use_hdp_auxiliary_beta=not legacy,
            canonicalize_labels=not legacy,
            sample_alpha=not legacy,
            sample_gamma_mh=False,
            sample_kappa_mh=not legacy,
        )
        hist = model.gibbs(y, n_iters=args.n_iter, burn_in=args.burn_in)
        z_inferred = hist["last_z"]
        x_last = hist["last_x"]

    print("Running regime backtest...")
    back_path = os.path.join(_ROOT, "results", f"{tag}_strategy_backtest.png")
    results = simulate_regime_strategy(
        dates=dates,
        asset_returns=asset_returns,
        inferred_regimes=z_inferred,
        risk_free_rate=args.risk_free,
        save_path=back_path,
        show_plot=args.show,
    )

    print("\n=== Regime labels (from in-sample return stats) ===")
    print(results["regime_label_map"])
    print("\nStrategy metrics:", results["strategy_metrics"])
    print("Buy & hold metrics:", results["buy_hold_metrics"])

    if args.model == "rslds" and rslds is not None:
        w_plot = os.path.join(_ROOT, "results", "rslds_recurrence_weights.png")
        plot_recurrence_weights(
            rslds.W,
            rslds.r,
            numeric_cols,
            results["regime_label_map"],
            w_plot,
        )
        print(f"\nSaved recurrence weight figure: {w_plot}")
        summarize_w_bear_features(rslds.W, numeric_cols, results["regime_label_map"])

    if x_last is not None:
        x_plot = os.path.join(_ROOT, "results", f"{tag}_latent_vs_returns.png")
        plot_latent_vs_returns(
            dates,
            x_last[:, 0],
            asset_returns,
            x_plot,
            inferred_regimes=z_inferred,
            regime_label_map=results["regime_label_map"],
        )
        print(f"Saved latent vs returns: {x_plot}")
    else:
        print("(No latent state x_t for hdp_arhmm — AR-HMM observes dynamics on y only.)")

    print(f"\nBacktest figure: {back_path}")
    print("Done.")


if __name__ == "__main__":
    main()
