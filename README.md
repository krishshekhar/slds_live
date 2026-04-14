# Bayesian switching dynamical systems for market regimes

Production-oriented documentation for a **real-data-only** research codebase: **sticky HDP-SLDS**, **sticky HDP-AR-HMM**, and **recurrent SLDS (Polya–Gamma)** with optional **regime-based backtests** and publication-style figures.

**References:** Fox et al. (2008) switching LDS / HDP-AR-HMM; Linderman et al. (2017) recurrent SLDS.

---

## Table of contents

1. [Overview](#overview)
2. [Repository layout](#repository-layout)
3. [Requirements & installation](#requirements--installation)
4. [Data pipeline](#data-pipeline)
5. [Models (what each file implements)](#models-what-each-file-implements)
6. [Running the master backtest](#running-the-master-backtest)
7. [Outputs & result figures](#outputs--result-figures)
8. [Realtime trading runner](#realtime-trading-runner)
9. [Methodology notes](#methodology-notes)
10. [Limitations & honest caveats](#limitations--honest-caveats)
11. [Troubleshooting](#troubleshooting)
12. [Deploy dashboard (Streamlit Community Cloud)](#deploy-dashboard-streamlit-community-cloud)
13. [References](#references)

---

## Overview

| Capability | Description |
|------------|-------------|
| **Multivariate observations** | All numeric CSV columns (except `Date`) are stacked into \(y_t \in \mathbb{R}^D\) for model fitting. |
| **Sticky HDP (weak limit)** | Discrete modes \(z_t \in \{0,\ldots,L-1\}\) with global weights \(\beta\) and sticky transitions (\(\kappa\) self-bias). |
| **HDP-SLDS** | Latent \(x_t\), linear-Gaussian emissions \(y_t \approx C x_t + \mathcal{N}(0,R)\), mode-specific dynamics \(x_t = A_{z_t} x_{t-1} + \mathcal{N}(0,Q_{z_t})\), **Kalman FFBS** for \(x \mid z\). |
| **rSLDS** | Transitions via stick-breaking logits \(\nu = W[z_t] x_t + r[z_t]\), **Polya–Gamma** augmentation, FFBS for \(x\). |
| **Backtest** | Maps inferred \(z_t\) to bull/bear/neutral using **mean return ranks** on the traded column; **lagged** positions (no same-day lookahead in the rule). |

---

## Repository layout

| Path | Role |
|------|------|
| `src/hdp_slds.py` | `StickyHDPSLDS`: sticky HDP + SLDS Gibbs (CRT auxiliary for \(\beta\), optional \(\alpha\)/\(\kappa\) updates, \(\beta\)-sorted canonical labels). |
| `src/hdp_arhmm.py` | `StickyHDPARHMM`: sticky HDP + VAR emissions. |
| `src/recurrent_slds.py` | `RecurrentSLDS`: PG-augmented rSLDS. |
| `src/rarhmm.py` | Recurrent AR-HMM (PG). |
| `src/backtest.py` | Regime strategy vs buy-and-hold, bull/bear/neutral labeling, shaded regime plots. |
| `src/initialization.py` | PCA + switching AR + logistic init for rSLDS (`initialize_rslds`). |
| `src/hmm_baselines.py`, `src/switching_ar.py` | Baseline HMM / switching AR utilities. |
| `src/evaluation.py` | Hamming distance with permutation, run-length stats. |
| `run_master_backtest.py` | **Main CLI:** load CSV → scale features → fit model → backtest → save PNGs. |
| `get_data.py` | Download/build `data/market_features.csv` (India or US macro columns). |
| `experiments/run_realdata_example.py` | Train all three cores; CSV + trace + regime overview figures. |
| `data/README.md` | CSV schema and column expectations. |
| `report/` | LaTeX report drafts. |
| `results/` | Generated PNGs and experiment CSVs (create on first run). |

---

## Requirements & installation

- **Python 3.10+** recommended (tested in 3.12+ / 3.14 venvs).
- Dependencies: `requirements.txt` (`numpy`, `scipy`, `matplotlib`, `pandas`, `scikit-learn`, `yfinance`).

**macOS / Homebrew Python (PEP 668):** use a venv; do not `pip install` on the system interpreter.

```bash
cd slds_project
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run entrypoints with the venv active or via `.venv/bin/python …`.

---

## Data pipeline

### 1. Build `data/market_features.csv`

```bash
python get_data.py              # default: --macro india
python get_data.py --macro us   # US Treasury term spread + US HY OAS
```

**India default columns (typical):** `nifty_log_return`, `vix_change`, `india_term_spread`, `asia_em_credit_oas`.  
**US macro mode:** `yield_spread`, `credit_spread` instead of the India/Asia pair.

See `get_data.py` docstring and `data/README.md` for FRED series IDs and caveats (e.g. Asia EM OAS is **regional**, not India-only).

### 2. CSV contract

- **`Date`:** parsed as dates, sorted ascending.
- **Numeric columns:** any number of features; master script uses **all** of them for \(y_t\) (multivariate fit).
- **Missing values:** should be dropped or imputed before modeling (download script uses `dropna()`).

---

## Models (what each file implements)

### Sticky HDP-SLDS (`StickyHDPSLDS`)

- **Truncation:** at most **`L`** discrete modes (weak-limit HDP); fewer may be occupied.
- **Continuous state:** \(x_t \in \mathbb{R}^M\), \(M = D\) in this project (`state_dim = obs_dim`).
- **Observation:** \(y_t = C x_t + \eta_t\), \(C = I\), \(R \sim\) inverse-Wishart update.
- **Dynamics:** \(x_t = A_{z_t} x_{t-1} + \epsilon_t\), MN–IW updates for \((A_k, Q_k)\).
- **Discrete \(z\):** forward–backward / backward sampling using \(\pi\), \(\beta\), and Gaussian transition likelihoods from \(x\).
- **Extensions (see `src/hdp_slds.py` docstring):** CRT auxiliary counts for \(\beta\); optional Escobar–West-style \(\alpha\); MH on \(\kappa\); post-\(z\) **canonical sort by \(\beta\)**. Use `--hdp-legacy` in the master script to disable these for ablation.

### Sticky HDP-AR-HMM (`StickyHDPARHMM`)

- Observations are **directly** \(y_t\); VAR(\(r\)) emissions per mode with MNIW updates. Default `ar_order=1` in the master script.

### Recurrent SLDS (`RecurrentSLDS`)

- Finite **`K`** modes; transitions from **latent** \(x_t\) through stick-breaking with PG augmentation; initialization via `initialize_rslds`.

---

## Running the master backtest

```bash
python run_master_backtest.py --model <rslds|hdp_arhmm|hdp_slds> [options]
```

### Common arguments

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | `rslds` | Which sampler to run. |
| `--csv` | `data/market_features.csv` | Input table. |
| `--full` | off | Use entire CSV date range (ignores `--start` / `--end`). |
| `--start`, `--end` | `2020-01-01` … `2022-12-31` | Inclusive slice when not `--full`. |
| `--n-iter` | `100` | Gibbs iterations. |
| `--burn-in` | `50` | Post-burn-in stored samples (where implemented). |
| `--k` | `3` | Discrete states for **rSLDS** only. |
| `--L` | (data-driven) | HDP truncation for **hdp_*** models: SLDS default `min(10, max(5, D+3))`, AR-HMM default `min(12, max(6, D+4))`. |
| `--risk-free` | `0.05` | Annualized rate for Sharpe in backtest. |
| `--asset-col` | first numeric column | **Return series** for P&L and latent plot **left axis** (raw CSV units). |
| `--random-state` | `42` | RNG seed. |
| `--show` | off | Call `plt.show()` (blocks in headless environments). |
| `--n-gibbs-ar` | `50` | Switching-AR Gibbs iters during rSLDS init. |
| `--hdp-legacy` | off | **HDP-SLDS only:** legacy collapsed Dirichlet transitions, fixed hypers, no CRT / canonicalization. |

**Feature scaling (current `run_master_backtest.py`):** all numeric columns are **column-wise z-scored on the selected window** (full-sample mean/std over rows in that slice). The **backtest return** uses the **raw** `asset_col` series.

**Minimum rows:** 80 after date slice (script exits otherwise).

### Example commands

```bash
# Recurrent SLDS, default window
python run_master_backtest.py --model rslds

# HDP-SLDS, full history, longer chain
python run_master_backtest.py --model hdp_slds --full --n-iter 500 --burn-in 200

# HDP-AR-HMM with explicit truncation
python run_master_backtest.py --model hdp_arhmm --full --L 12

# Different traded column
python run_master_backtest.py --model hdp_slds --asset-col nifty_log_return
```

---

## Outputs & result figures

Figures are written under **`results/`** (created automatically). Typical filenames:

| File | When |
|------|------|
| `{model}_strategy_backtest.png` | Always (after backtest). |
| `{model}_latent_vs_returns.png` | **hdp_slds**, **rslds** (latent \(x_t\)). |
| `rslds_recurrence_weights.png` | **rslds** only. |
| `inferred_regimes.csv` | `experiments/run_realdata_example.py` |
| `gibbs_loglikelihood_traces.png` | experiment script |
| `real_regimes_all_models.png` | experiment script |

### Embedded examples (regenerate locally if missing)

**HDP-SLDS: regime strategy vs buy-and-hold (shaded regimes)**

![HDP-SLDS strategy backtest](results/hdp_slds_strategy_backtest.png)

**HDP-SLDS: first latent dimension vs raw asset return (dual axis, regime shading)**

![HDP-SLDS latent vs returns](results/hdp_slds_latent_vs_returns.png)

**HDP-AR-HMM: backtest figure**

![HDP-AR-HMM strategy backtest](results/hdp_arhmm_strategy_backtest.png)

**Recurrent SLDS: backtest + recurrence weights (when run with `--model rslds`)**

![rSLDS strategy backtest](results/rslds_strategy_backtest.png)

![rSLDS recurrence weights](results/rslds_recurrence_weights.png)

![rSLDS latent vs returns](results/rslds_latent_vs_returns.png)

---

## Realtime trading runner

Use `run_live_trader.py` to run a continuous infer-and-trade loop on fresh bars.

### What it does

- Downloads latest bars from `yfinance`.
- Builds online features (`price_log_return`, rolling volatility, momentum, volume ratio).
- Re-fits one of `rslds`, `hdp_arhmm`, `hdp_slds` on a rolling window.
- Infers the latest regime, maps it to bull/bear/neutral, then maps that to target weight.
- Rebalances through either:
  - local simulator (`--mode paper`), or
  - Alpaca paper/live account (`--mode alpaca-paper` / `--mode live`).
- Applies hard safety rails:
  - max position cap (`--max-position-weight`)
  - daily drawdown kill switch (`--max-daily-loss-pct`)
  - live-mode confirmation flag (`--confirm-live`)
- Writes structured runtime logs to `results/live_trading/live_events.jsonl` for dashboarding.

### Live monitoring dashboard

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run live_dashboard.py -- --events-file results/live_trading/live_events.jsonl --refresh-seconds 5
```

Keep `run_live_trader.py` running in another terminal. The dashboard auto-refreshes and shows:
- latest model decisions and risk state
- equity/price/target-weight time series
- recent orders (submitted/skipped)
- runtime errors and raw event feed

### Deploy dashboard (Streamlit Community Cloud)

[Streamlit Community Cloud](https://streamlit.io/cloud) deploys from a **GitHub** repository, not from your laptop alone. That is why you see: *code is not connected to a remote GitHub repository* until you push.

1. **Create a new empty repository** on GitHub (no README/license if you already have a local repo).
2. **Add the remote and push** your `main` branch (from this project folder):

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

3. In **Streamlit Community Cloud**, click **New app**, connect GitHub if asked, pick the repo and branch **`main`**, and set:

   - **Main file path:** `live_dashboard.py`
   - **Python version:** 3.10+ (match your local env if possible)

4. **Deploy.** The first build installs packages from `requirements.txt`.

**Note:** The hosted dashboard only sees files in the repo. It does **not** receive `results/live_trading/live_events.jsonl` from a trader running on your machine unless you add a backend that writes events somewhere the app can read (object storage, database, etc.). For a live remote view of local runs, keep using `streamlit run` on your computer, or extend the pipeline later.

### Quick start (safe)

```bash
python run_live_trader.py \
  --symbol SPY \
  --model hdp_slds \
  --mode paper \
  --once
```

Then run continuously:

```bash
python run_live_trader.py \
  --symbol SPY \
  --model hdp_slds \
  --mode paper \
  --poll-seconds 60 \
  --window-bars 240
```

### Alpaca paper trading

Set credentials:

```bash
export ALPACA_API_KEY="..."
export ALPACA_API_SECRET="..."
```

Run:

```bash
python run_live_trader.py \
  --symbol SPY \
  --model hdp_slds \
  --mode alpaca-paper \
  --poll-seconds 60 \
  --max-position-weight 0.3 \
  --max-daily-loss-pct 0.02
```

### Live capital (real orders)

Live trading requires `--mode live --confirm-live` together:

```bash
python run_live_trader.py \
  --symbol SPY \
  --model hdp_slds \
  --mode live \
  --confirm-live \
  --poll-seconds 60 \
  --max-position-weight 0.2 \
  --max-daily-loss-pct 0.01
```

Recommended rollout: run `--mode paper` first, then `--mode alpaca-paper`, and only then `--mode live`.

---

## Methodology notes

### What is fed into the SLDS / HDP models?

- **`y`:** matrix of shape `(T, D)` = all numeric feature columns after **per-column standardization** over the **selected date window**.
- **Trading / blue line in latent plot:** **`asset_col`** from the **original CSV scale** (not necessarily the same numeric scaling as column 0 of `y` if you override `--asset-col`).

### Bull / bear / neutral

Defined in **`src/backtest.py`**: for each discrete state that appears, compute the **mean** of the **traded return** on days with that state; **lowest** mean → bear, **highest** → bull, **middle** ranks → neutral (with a variance tie-break in the sort). This is a **post hoc economic label**, not part of the generative model.

### Position rule

`position[t]` uses **`z_{t-1}`** (one-day lag), so the strategy rule does not use **same-day** regime for trading.

### Gibbs “convergence”

The code runs a **fixed** iteration count. There is **no** built-in \(\hat R\), ESS, or automatic stopping. For reporting, inspect `history["loglik"]` (or experiment traces), run **longer** chains, or add external diagnostics.

### Maximum number of regimes

- **HDP models:** upper bound = **`L`** (weak limit). Default `L` is a function of **`D`** unless `--L` is set. Unused indices may never appear in `z`.
- **rSLDS:** exactly **`K`** labels (`--k`).

---

## Limitations & honest caveats

1. **In-sample evaluation:** Unless you maintain a separate train/holdout script, the master pipeline **fits and evaluates on the same window** (metrics are **not** out-of-sample forecasts).
2. **Global z-score on the window:** Using mean/std over **all** rows in the slice injects **look-ahead into the scaler** relative to a strictly causal deployment. (If your branch adds `src/preprocess.py` / `--feature-scale` / `--train-end`, document that variant here.)
3. **FFBS / smoothing:** Latent and discrete updates use **full-sequence** information within the fitted segment (standard for Gibbs SLDS, not an online filter).
4. **Bull/bear labels** are **mean-return ranks** on the evaluation window; they need not match intuitive “bear markets” from volatility or drawdowns.
5. **rSLDS numerics:** Very long chains can hit ill-conditioned covariances; the implementation uses stabilizing inverses / jitter where noted.

---

## Troubleshooting

| Issue | Suggestion |
|-------|------------|
| `externally-managed-environment` (pip) | Create and use `.venv` as above. |
| `Too few rows after slice` | Widen `--start`/`--end` or use `--full`. |
| sklearn `FutureWarning` on `LogisticRegression` | Cosmetic; can migrate to future sklearn defaults when convenient. |
| `LinAlgError` / overflow in rSLDS | Reduce `--n-iter`, shorten window, or increase `--random-state` retry; see `recurrent_slds.py` stabilizations. |

---

## Experiment bundle (all models)

```bash
python experiments/run_realdata_example.py
```

Produces regime CSVs and comparison figures under `results/` (see that script for exact filenames).

---

## References

1. E. B. Fox, E. B. Sudderth, M. I. Jordan, A. S. Willsky (2008). *Nonparametric Bayesian Learning of Switching Linear Dynamical Systems.* NeurIPS.
2. S. W. Linderman, M. J. Johnson, A. C. Miller, R. P. Adams, D. M. Blei, L. Paninski (2017). *Bayesian Learning and Inference in Recurrent Switching Linear Dynamical Systems.* AISTATS.
3. Y. W. Teh et al. (2006). *Hierarchical Dirichlet Processes.* JASA (HDP / weak-limit background).
4. M. Escobar, M. West (1995). *Bayesian Density Estimation and Inference Using Mixtures.* JASA (concentration sampling ideas).

