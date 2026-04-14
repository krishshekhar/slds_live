"""
Real-data benchmark for paper-aligned switching dynamical models.

Models:
- Sticky HDP-AR-HMM (Fox et al. 2008, AR-HMM branch)
- Sticky HDP-SLDS (Fox et al. 2008, SLDS branch)
- Recurrent AR-HMM with PG augmentation (Linderman et al. 2017 special case)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.hdp_arhmm import StickyHDPARHMM
from src.hdp_slds import StickyHDPSLDS
from src.rarhmm import RecurrentARHMMPG
from src.evaluation import empirical_runlength_stats


def _load_real_data(csv_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Missing dataset at {csv_path}. Follow instructions in data/README.md."
        )
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values("Date")
    numeric_cols = [c for c in df.columns if c != "Date" and np.issubdtype(df[c].dtype, np.number)]
    if not numeric_cols:
        raise ValueError("CSV must include at least one numeric feature column besides Date.")
    y = df[numeric_cols].to_numpy(dtype=float)
    y = y - y.mean(axis=0, keepdims=True)
    std = y.std(axis=0, keepdims=True) + 1e-8
    y = y / std
    return df["Date"].to_numpy(), y, numeric_cols


def _print_runlength(name: str, z: np.ndarray) -> None:
    print(f"{name:22s} {empirical_runlength_stats(z)}")


def main() -> None:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, "data", "market_features.csv")
    dates, y, cols = _load_real_data(data_path)
    T, D = y.shape
    print(f"Loaded {T} observations with {D} features: {cols}")

    hdp_arhmm = StickyHDPARHMM(
        L=min(10, max(5, D + 3)),
        D=D,
        ar_order=1,
        alpha=6.0,
        gamma=2.0,
        kappa=20.0,
        random_state=0,
    )
    hist_hdp_ar = hdp_arhmm.gibbs(y, n_iters=300, burn_in=150)
    z_hdp_ar = hist_hdp_ar["last_z"]

    rslds = RecurrentARHMMPG(
        K=min(8, max(4, D + 2)),
        D=D,
        ar_order=1,
        random_state=1,
    )
    hist_rar = rslds.gibbs(y, n_iters=250, burn_in=120)
    z_rar = hist_rar["last_z"]

    hdp_slds = StickyHDPSLDS(
        L=min(8, max(4, D + 2)),
        state_dim=D,
        obs_dim=D,
        alpha=6.0,
        gamma=2.0,
        kappa=20.0,
        random_state=2,
    )
    hist_slds = hdp_slds.gibbs(y, n_iters=220, burn_in=100)
    z_slds = hist_slds["last_z"]

    print("\nRun-length statistics:")
    _print_runlength("Sticky HDP-AR-HMM", z_hdp_ar)
    _print_runlength("Recurrent AR-HMM", z_rar)
    _print_runlength("Sticky HDP-SLDS", z_slds)

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    regime_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "z_sticky_hdp_arhmm": z_hdp_ar.astype(int),
            "z_recurrent_arhmm": z_rar.astype(int),
            "z_sticky_hdp_slds": z_slds.astype(int),
        }
    )
    regime_path = os.path.join(results_dir, "inferred_regimes.csv")
    regime_df.to_csv(regime_path, index=False)
    print(f"\nSaved inferred regimes to {regime_path}")

    fig_ll, axes_ll = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    it_hdp = np.arange(1, len(hist_hdp_ar["loglik"]) + 1)
    axes_ll[0].plot(it_hdp, hist_hdp_ar["loglik"], lw=0.8, color="C0")
    axes_ll[0].set_ylabel("log p(y)")
    axes_ll[0].set_title("Gibbs marginal log-likelihood (sticky HDP-AR-HMM)")

    it_rar = np.arange(1, len(hist_rar["loglik"]) + 1)
    axes_ll[1].plot(it_rar, hist_rar["loglik"], lw=0.8, color="C1")
    axes_ll[1].set_ylabel("log p(y)")
    axes_ll[1].set_title("Gibbs marginal log-likelihood (recurrent AR-HMM)")

    it_slds = np.arange(1, len(hist_slds["loglik"]) + 1)
    axes_ll[2].plot(it_slds, hist_slds["loglik"], lw=0.8, color="C2")
    axes_ll[2].set_ylabel("log joint")
    axes_ll[2].set_xlabel("Gibbs iteration")
    axes_ll[2].set_title(
        "SLDS complete-data log p(y, x | z, θ) — not on same scale as marginals above"
    )
    fig_ll.tight_layout()
    loglik_path = os.path.join(results_dir, "gibbs_loglikelihood_traces.png")
    fig_ll.savefig(loglik_path, dpi=220)
    plt.close(fig_ll)
    print(f"Saved log-likelihood traces to {loglik_path}")

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(dates, y[:, 0], lw=0.6)
    axes[0].set_ylabel(cols[0])
    axes[0].set_title("Real financial features and inferred regimes")
    axes[1].plot(dates, z_hdp_ar, lw=0.7)
    axes[1].set_ylabel("HDP-AR")
    axes[2].plot(dates, z_rar, lw=0.7)
    axes[2].set_ylabel("rAR-HMM")
    axes[3].plot(dates, z_slds, lw=0.7)
    axes[3].set_ylabel("HDP-SLDS")
    axes[3].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "real_regimes_all_models.png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()

