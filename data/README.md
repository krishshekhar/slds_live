## Data preparation (real market data only)

The project expects a single real-data CSV file:

- path: `data/market_features.csv`
- required column: `Date`
- additional columns: one or more numeric features (returns, realized volatility,
  macro covariates, spreads, volume signals, etc.)

### Minimum example

`market_features.csv`

- `Date`
- `spx_log_return`

### Multivariate example

- `Date`
- `nifty_log_return`
- `vix_change`
- Macro columns from `get_data.py` (choose with `--macro india` or `--macro us`):
  - **India (default):** `india_term_spread` (OECD India 10Y G-Sec minus OECD India short rate, monthly FRED series forward-filled to each trading day), `asia_em_credit_oas` (ICE BofA Asia EM corporate OAS on FRED — regional index, not India-only).
  - **US:** `yield_spread` (`T10Y2Y`), `credit_spread` (`BAMLH0A0HYM2`).

### Suggested preprocessing

1. Align all series by date.
2. Forward-fill sparse macro series where appropriate.
3. Drop remaining rows with missing values.
4. Export as `market_features.csv` with `Date` in ISO format.

The experiment script standardizes numeric features internally before fitting
the models.

