# Next steps with the new (big) dataset

The user ran `colab/MARS_RL_mega_dataset_downloader.ipynb`. Expected output folder `mars_data/`:
`raw/prices_ohlcv_adjusted_long.(parquet|csv.gz)` (cols date,open,high,low,close,volume,tic,asset_class), `raw/macro_fred_lagged.csv`, `raw/fama_french_5f_mom_daily.csv`, `raw/cboe_indices_daily.csv`, `raw/sp500_constituents_TODAY_survivorship_biased.csv`, optional earnings/fundamentals/FNSPID, `processed/features_stocks_daily.(parquet|csv.gz)` (~74 features), `reports/data_quality.json`.
Put it under `data/mega/`.

## Step 1 — Inspect (no modelling yet)
Read `data_quality.json`; check date coverage per ticker, NaN share, outliers (|ret|>50%), duplicate keys, volume zeros; list failed/dropped tickers.

## Step 2 — Alpha go/no-go (validation only)
Use `src/alpha.py` (fit on train range with embargo H) on the new features; report rank-IC and t-stat on validation for H = 5, 20 (and 60). Add FF/macro/CBOE/rank features. **Decision rule:** if validation IC is not clearly > 0 (e.g. mean IC > 0.01 and t > 2), the data does not contain an exploitable edge; say so and stop trying to "beat" the paper through selection.

## Step 3 — Universe and panel
Pick a fixed universe: e.g. top 50–100 stocks by 20d dollar volume with complete history over the chosen train/val/test span, plus optional ETFs (TLT, GLD, …) as hedge assets. Build the panel with `build_panel` (or extend `data.py` to take the new feature columns as extra channels; `Panel.extra` (T,N) and `mkt` (T,M) exist). Add macro/FF/CBOE/VIX term-structure to the **market-state** vector (also used by the retrieval gate). Keep point-in-time rules.

## Step 4 — Protocol redesign (document it as protocol v2)
With 2000–2024 history use walk-forward: e.g. train 2000–2012 → val 2013–2015 → test A; expanding windows for several test years including 2008 is only possible if the data covers it. Keep the original paper window (test 2020-05..2023-05, train ≤ 2018, val 2019-01..2020-04) as the headline comparison, and add the other windows as robustness.

## Step 5 — Models
Re-run v2_b-style residual PPO ensembles (3–5 seeds) on the new panel; test cheap ideas on validation only: rank-based alpha tilt as the base policy instead of equal weight, vol-targeted leverage, regime gate on macro state, turnover penalties. Keep `margin_env.py` costs unchanged for comparability.

## Step 6 — Locked test and stats
One test run per final config; multi-seed; paired bootstrap / JKM vs equal-weight and vs paper-style agents; cost stress 1×/2×/3×; twin crash stress; log n_trials.

## Step 7 — Only now regenerate deliverables
Tables/figures (`src/report.py`, `src/compare_base_paper.py`), executed notebook, README (EN + Hinglish), manuscript, data package. Do not write "beats the paper" unless Step 6 shows it.
