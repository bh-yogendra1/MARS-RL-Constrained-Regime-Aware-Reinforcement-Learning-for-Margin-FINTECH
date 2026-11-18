# Experiment log (everything tried; n_trials matters for deflated-Sharpe style honesty)

Data: 28 DJIA tickers (DOW, WBA absent), yfinance adjusted OHLCV 2009-01..2023-05 (3,627 days). Splits: train 2009-01..2018-12, val 2019-01..2020-04, test 2020-05..2023-05.

| Phase | What | Selection | Outcome |
|---|---|---|---|
| v1 | 8 roles × 3 seeds × 150k steps (finrl_long, mt_long, mt_split, mars_long/ls/neutral, plain_ls, mars_ls_notwin) + gate/uniform/shuffled | val Sharpe checkpoint | MARS-gate test Sharpe 0.10, ann. 0.4%, MDD −18.4%; EW 0.85. Long-short experts negative. Cost-fragile (turnover 0.3–0.8/day) |
| v1 note | Test metrics printed once on 40k-step stage-1 models during smoke test | – | disclosed; no design change from it |
| v2 | 6 variants (v2_a–f): residual-on-EW, smoothing 0.3, turnover penalty, guard/no guard, reward mars/mt/plain | highest mean val Sharpe → **v2_b** | v2_b test Sharpe 0.68; v2_c/d (no guard) had test 0.89/0.87 but MDD −32/−29% and were NOT val-selected (post-hoc only) |
| v3 | 4 variants: conformal budget 0.05/0.08, κ = 1.5/3 | val | all worse than v2_b on val |
| v4 | 2 variants: conformal budget 0.02/0.015 | val | v4_a val ensemble 1.00 vs v2_b 1.04 → v2_b stays |
| ensemble | mean of 3 seeds' actions | val | v2_b ens test Sharpe 0.80, ret 11.3% |
| refit | v2_b retrained on train+val, 5 seeds, last checkpoint, ensemble | fixed config | **test Sharpe 0.854, ret 11.3%, MDD −19.2%, Calmar 0.59; 2×/3× cost Sharpe 0.74/0.63; paper-style costs 0.878** |
| alpha | ridge / GBM cross-sectional alpha on same features (H = 5, 20), train-only fit with embargo | val IC | train IC 0.03–0.16 but **val IC −0.036..+0.014 ≈ 0** → no edge |
| Track A | authors' env + A2C/PPO, 50k steps, 3 seeds, our data | – | A2C test Sharpe 0.50, ann 11.1%; PPO 0.60, ann 11.1% (paper: 30.9% Long) |

Variants tried after v1: 12 (+ alpha experiment). Statistics so far: paired bootstrap / JKM vs EW for v1 (significantly worse); v2 not yet tested formally.

Key lessons
1. Turnover/costs killed v1; residual-around-EW + smoothing fixed most of it.
2. Tight conformal guard lowers drawdown but also exposure (avg gross ≈ 0.7) → return below EW.
3. Features of the 28-stock set carry no out-of-sample alpha; leverage on beta cannot raise Sharpe.
4. Seed variance is large; ensembles help.
