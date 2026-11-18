# MARS-RL: Constrained, Regime-Aware Reinforcement Learning for Margin Trading — A Replication Study with Negative Findings

*Manuscript draft (IEEE-Transactions format, Markdown). All numbers are read from `results/tables/*.csv`.*

**Abstract —** Gu et al. (IEEE TKDE 2026) show that deep-RL agents trained in a margin-trading simulator (leverage, short selling, maintenance-margin rules) can earn high returns on 28 Dow-Jones stocks. We study whether the base setting can be improved along five axes: (i) frictions (slippage, borrow and financing costs), (ii) learned instead of fixed leverage, (iii) an explicit constrained-MDP (CMDP) formulation with Lagrangian multipliers and a conformal exposure guard, (iv) training on regime-switching block-bootstrap "digital twin" paths, and (v) a retrieval-guided mixture-of-experts (MoE) gate. We evaluate under two separated protocols: *Track A* runs the authors' own simulator and agents on our data; *Track B* runs re-implemented baselines and our agents in a common-cost simulator, with a single locked test (2020-05 to 2023-05), three seeds, cost stress (1×–3×) and synthetic crash stress. **The proposed method did not improve risk-adjusted return**: test Sharpe 0.10 (MARS-RL) vs 0.85 (equal-weight) and 0.15 (paper-style 2× long-only PPO); the gap to equal-weight is statistically significant. It did reduce tail risk (test max drawdown −18.4% vs −54.0% for the paper-style 2× agent; zero margin calls), but a shuffled-memory gate performs the same, so the retrieval component is not shown to help. All learned agents are cost-fragile. We report the negative result, its likely causes, and the components that were not evaluated.

**Index Terms —** Deep reinforcement learning, portfolio management, margin trading, constrained MDP, conformal prediction, digital twin, mixture of experts, negative results.

## I. Introduction
Leverage and short selling are central to practical portfolio management but are rarely modelled with realistic margin rules in RL research. Gu et al. [1] introduce a margin-aware environment and report large gains. Their evaluation, however, (a) omits slippage, borrowing and financing costs, (b) uses fixed leverage and a heuristic margin-call penalty, (c) relies on a single historical path, and (d) does not test significance across seeds. We hypothesised that a CMDP formulation with learned leverage, an execution-time conformal guard, twin-path augmentation and a retrieval-gated expert mixture would improve risk-adjusted return and tail risk. **Contributions (as actually delivered):** (1) an auditable replication pipeline with data provenance, a point-in-time feature layer, unit-tested no-look-ahead and ledger invariants; (2) a common-cost margin simulator and re-implemented baselines; (3) the MARS-RL components above with ablations; (4) a locked, multi-seed evaluation with cost and crash stress and formal tests; (5) an honest negative result.

## II. Related Work
Deep RL for trading and portfolio management is surveyed in FinRL [2]; PPO [3] and A2C are standard policy-gradient learners. Safe RL uses CMDPs [4] and Lagrangian/constrained policy optimisation [5]. Conformal prediction [6] provides distribution-free coverage under exchangeability (not guaranteed for financial time series; our guard is therefore a heuristic, not a coverage guarantee). Block bootstraps [7] generate synthetic return paths; retrieval augmentation [8] conditions decisions on similar past cases. Sharpe-ratio comparison uses the Jobson–Korkie test with Memmel's correction [9] and the Deflated Sharpe Ratio [10].

## III. Problem Formulation
Daily decisions on N=28 stocks. State: 10 stock features per name (returns over 1/5/20/60 days, 20-day volatility, SMA-50 gap, MACD, RSI-30, CCI-30, DX-30), 9 market-state features, account variables. Action: N+1 numbers in [−1,1]; the last sets leverage L = L_max(a+1)/2 and the rest set direction/weights (long-only, long-short, market-neutral or 50/50 split modes). A decision at the close of day t executes at the open of t+1 and is marked at the close of t+1. Frictions (identical for every strategy): 0.1% fee + 2 bp slippage per side, 0.5%/yr borrow fee on shorts, 4%/yr interest on borrowed cash; margin level < 40% blocks risk-increasing trades; < 30% triggers forced liquidation to a 50% margin level plus a 0.1% fee; initial equity $100,000; episode ruin at 20%.

## IV. Method
**A. CMDP experts.** Reward r_t = 100·log-return − μ_m·1[margin call] − μ_d·100·max(0, DD_t − 10%) − η·(100·min(r_t,0))². Multipliers follow dual ascent after every rollout (clipped to [0,5]; budgets 0.001 margin calls/step and 0.01 mean drawdown excess/step; step sizes were not tuned). Policies are PPO (MLP 128×128; 150k steps; 4 envs).
**B. Conformal exposure guard.** Score s_t = |r_EW,t|/σ_EW,t−1 with EWMA (λ=0.94) volatility; q_t = rolling 95% quantile over 250 days. At execution, weights are scaled by min(1, B/(q_t·√(wᵀΣ_t w))), B=3%. The guard never increases exposure (unit test).
**C. Digital twin.** Three volatility regimes (terciles of past-only 20-day market volatility, train period only), a Markov chain estimated on train, geometric blocks (mean 10) of whole cross-sections, and synthetic OHLC rebuilt from resampled ratios and passed through the same feature code. Fidelity (Table I). Stress paths: demeaned log-returns ×1.3 with −0.15%/day drift, 250 days.
**D. Retrieval-guided MoE gate.** Embedding = standardised market-state features. Memory stores, for each past day s, each expert's 20-day forward Sharpe; for decision day t only s ≤ t−20 is retrievable (embargo, unit-tested by corrupting the future). The k=30 nearest neighbours' mean scores pass through softmax(β=10), shrunk 50% toward uniform. Mixed weights are executed through the conformal guard.
**E. Not evaluated here:** Mamba encoder, DAgger distillation, causal module and the LLM/news agent (code only).

## V. Experimental Setup
Data: 28 DJIA constituents (DOW, WBA absent), daily Yahoo-adjusted OHLCV 2009-01 to 2023-05 (user-downloaded; SHA-256 manifest), 3,627 trading days. Splits: train 2009-01–2018-12, validation 2019-01–2020-04 (checkpoint selection only), test 2020-05–2023-05. Validation and test are run as one continuous account and sliced by date. Seeds 0–2; risk-free rate 0. Baselines in Track B: equal-weight buy-and-hold, daily-rebalanced EW (1× and 2×), monthly min-variance (Ledoit–Wolf), monthly momentum top-7, PPO long-only 1×, PPO long-only 2× with paper-style reward (profit + 5-step Sharpe), PPO 50/50 long-short 2×. Track A: authors' unmodified environment and A2C/PPO scripts, 50k steps, 3 seeds each. Tests: paired stationary bootstrap (2,000 resamples, block 10), JKM, DSR.

**Table I — Digital-twin fidelity (train period vs twin).**

|                  |   real_train |   twin (mean of 8 paths) |
|:-----------------|-------------:|-------------------------:|
| ann_vol          |       0.225  |                   0.2181 |
| excess_kurtosis  |       6.1652 |                   5.8948 |
| abs_ret_autocorr |       0.1605 |                   0.1419 |
| avg_corr         |       0.4079 |                   0.4085 |
| ew_maxdd         |      -0.1593 |                  -0.1925 |

## VI. Results
**Table II — Test period (2020-05 to 2023-05), common costs, mean ± sd over 3 seeds.**

|                                       | Ann. return    | Ann. vol      | Sharpe       | Sortino      | Max DD         | Calmar       | CVaR5 (daily)    | Avg gross   | Avg turnover   | Margin calls   | Costs (% init)   |
|:--------------------------------------|:---------------|:--------------|:-------------|:-------------|:---------------|:-------------|:-----------------|:------------|:---------------|:---------------|:-----------------|
| EW buy&hold (1x)                      | 0.137 ± 0.000  | 0.169 ± 0.000 | 0.85 ± 0.00  | 1.20 ± 0.00  | -0.216 ± 0.000 | 0.64 ± 0.00  | -0.0248 ± 0.0000 | 1.00 ± 0.00 | 0.000 ± 0.000  | 0.0 ± 0.0      | 0.000 ± 0.000    |
| EW daily-rebal (1x)                   | 0.138 ± 0.000  | 0.169 ± 0.000 | 0.85 ± 0.00  | 1.22 ± 0.00  | -0.211 ± 0.000 | 0.65 ± 0.00  | -0.0244 ± 0.0000 | 1.00 ± 0.00 | 0.010 ± 0.000  | 0.0 ± 0.0      | 0.013 ± 0.000    |
| EW daily-rebal (2x lev)               | 0.204 ± 0.000  | 0.338 ± 0.000 | 0.72 ± 0.00  | 1.03 ± 0.00  | -0.414 ± 0.000 | 0.49 ± 0.00  | -0.0488 ± 0.0000 | 2.00 ± 0.00 | 0.027 ± 0.000  | 0.0 ± 0.0      | 0.042 ± 0.000    |
| Min-variance (monthly)                | 0.120 ± 0.000  | 0.159 ± 0.000 | 0.79 ± 0.00  | 1.13 ± 0.00  | -0.212 ± 0.000 | 0.56 ± 0.00  | -0.0229 ± 0.0000 | 1.00 ± 0.00 | 0.005 ± 0.000  | 0.0 ± 0.0      | 0.005 ± 0.000    |
| Momentum top-7 (monthly)              | 0.097 ± 0.000  | 0.182 ± 0.000 | 0.60 ± 0.00  | 0.84 ± 0.00  | -0.236 ± 0.000 | 0.41 ± 0.00  | -0.0262 ± 0.0000 | 1.00 ± 0.00 | 0.043 ± 0.000  | 0.0 ± 0.0      | 0.054 ± 0.000    |
| PPO long-only 1x (plain reward)       | 0.041 ± 0.010  | 0.182 ± 0.008 | 0.32 ± 0.06  | 0.45 ± 0.09  | -0.269 ± 0.055 | 0.16 ± 0.06  | -0.0260 ± 0.0019 | 1.00 ± 0.00 | 0.423 ± 0.023  | 0.0 ± 0.0      | 0.471 ± 0.030    |
| PPO long-only 2x (paper-style reward) | -0.012 ± 0.045 | 0.364 ± 0.020 | 0.15 ± 0.12  | 0.21 ± 0.17  | -0.540 ± 0.078 | -0.02 ± 0.09 | -0.0523 ± 0.0037 | 2.00 ± 0.00 | 0.790 ± 0.010  | 0.0 ± 0.0      | 0.976 ± 0.062    |
| PPO 50/50 long-short 2x (paper-style) | -0.151 ± 0.080 | 0.108 ± 0.006 | -1.53 ± 0.92 | -1.98 ± 1.14 | -0.444 ± 0.118 | -0.32 ± 0.11 | -0.0153 ± 0.0003 | 2.00 ± 0.00 | 0.806 ± 0.057  | 0.0 ± 0.0      | 0.592 ± 0.080    |
| MARS expert: long                     | 0.025 ± 0.022  | 0.161 ± 0.002 | 0.23 ± 0.14  | 0.33 ± 0.20  | -0.273 ± 0.013 | 0.09 ± 0.08  | -0.0231 ± 0.0006 | 0.94 ± 0.01 | 0.384 ± 0.020  | 0.0 ± 0.0      | 0.408 ± 0.032    |
| MARS expert: long-short               | -0.093 ± 0.017 | 0.063 ± 0.003 | -1.51 ± 0.30 | -2.00 ± 0.39 | -0.291 ± 0.025 | -0.32 ± 0.04 | -0.0093 ± 0.0003 | 0.95 ± 0.01 | 0.379 ± 0.010  | 0.0 ± 0.0      | 0.300 ± 0.012    |
| MARS expert: neutral                  | -0.060 ± 0.023 | 0.037 ± 0.001 | -1.67 ± 0.69 | -2.15 ± 0.82 | -0.181 ± 0.065 | -0.33 ± 0.01 | -0.0055 ± 0.0001 | 0.71 ± 0.01 | 0.290 ± 0.019  | 0.0 ± 0.0      | 0.243 ± 0.014    |
| MARS-RL (retrieval gate)              | 0.004 ± 0.019  | 0.099 ± 0.006 | 0.10 ± 0.20  | 0.14 ± 0.28  | -0.184 ± 0.020 | 0.03 ± 0.11  | -0.0143 ± 0.0008 | 0.78 ± 0.01 | 0.323 ± 0.012  | 0.0 ± 0.0      | 0.321 ± 0.017    |
| MARS uniform mix                      | -0.017 ± 0.014 | 0.067 ± 0.003 | -0.21 ± 0.21 | -0.29 ± 0.28 | -0.156 ± 0.029 | -0.10 ± 0.08 | -0.0097 ± 0.0007 | 0.72 ± 0.01 | 0.293 ± 0.010  | 0.0 ± 0.0      | 0.274 ± 0.007    |
| MARS gate, shuffled memory            | 0.001 ± 0.016  | 0.094 ± 0.008 | 0.06 ± 0.17  | 0.09 ± 0.24  | -0.187 ± 0.020 | 0.01 ± 0.08  | -0.0136 ± 0.0011 | 0.77 ± 0.01 | 0.320 ± 0.014  | 0.0 ± 0.0      | 0.314 ± 0.017    |

**Table III — Validation period (2019-01 to 2020-04).**

|                                       | Ann. return    | Ann. vol      | Sharpe       | Sortino      | Max DD         | Calmar       | CVaR5 (daily)    | Avg gross   | Avg turnover   | Margin calls   | Costs (% init)   |
|:--------------------------------------|:---------------|:--------------|:-------------|:-------------|:---------------|:-------------|:-----------------|:------------|:---------------|:---------------|:-----------------|
| EW buy&hold (1x)                      | 0.118 ± 0.000  | 0.289 ± 0.000 | 0.53 ± 0.00  | 0.75 ± 0.00  | -0.330 ± 0.000 | 0.36 ± 0.00  | -0.0466 ± 0.0000 | 1.00 ± 0.00 | 0.003 ± 0.000  | 0.0 ± 0.0      | 0.001 ± 0.000    |
| EW daily-rebal (1x)                   | 0.120 ± 0.000  | 0.293 ± 0.000 | 0.53 ± 0.00  | 0.75 ± 0.00  | -0.336 ± 0.000 | 0.36 ± 0.00  | -0.0471 ± 0.0000 | 1.00 ± 0.00 | 0.012 ± 0.000  | 0.0 ± 0.0      | 0.005 ± 0.000    |
| EW daily-rebal (2x lev)               | 0.138 ± 0.000  | 0.587 ± 0.000 | 0.52 ± 0.00  | 0.74 ± 0.00  | -0.578 ± 0.000 | 0.24 ± 0.00  | -0.0930 ± 0.0000 | 1.99 ± 0.00 | 0.032 ± 0.000  | 0.0 ± 0.0      | 0.015 ± 0.000    |
| Min-variance (monthly)                | 0.067 ± 0.000  | 0.281 ± 0.000 | 0.37 ± 0.00  | 0.51 ± 0.00  | -0.333 ± 0.000 | 0.20 ± 0.00  | -0.0462 ± 0.0000 | 1.00 ± 0.00 | 0.009 ± 0.000  | 0.0 ± 0.0      | 0.004 ± 0.000    |
| Momentum top-7 (monthly)              | 0.057 ± 0.000  | 0.293 ± 0.000 | 0.34 ± 0.00  | 0.46 ± 0.00  | -0.341 ± 0.000 | 0.17 ± 0.00  | -0.0481 ± 0.0000 | 1.00 ± 0.00 | 0.041 ± 0.000  | 0.0 ± 0.0      | 0.019 ± 0.000    |
| PPO long-only 1x (plain reward)       | 0.080 ± 0.055  | 0.307 ± 0.013 | 0.40 ± 0.16  | 0.57 ± 0.24  | -0.334 ± 0.018 | 0.24 ± 0.17  | -0.0492 ± 0.0011 | 1.00 ± 0.00 | 0.422 ± 0.022  | 0.0 ± 0.0      | 0.191 ± 0.009    |
| PPO long-only 2x (paper-style reward) | -0.013 ± 0.090 | 0.604 ± 0.024 | 0.28 ± 0.14  | 0.41 ± 0.21  | -0.570 ± 0.056 | -0.01 ± 0.16 | -0.0954 ± 0.0083 | 2.00 ± 0.00 | 0.841 ± 0.007  | 0.0 ± 0.0      | 0.387 ± 0.006    |
| PPO 50/50 long-short 2x (paper-style) | -0.128 ± 0.049 | 0.120 ± 0.005 | -1.10 ± 0.52 | -1.52 ± 0.58 | -0.248 ± 0.043 | -0.51 ± 0.14 | -0.0171 ± 0.0016 | 2.00 ± 0.00 | 0.803 ± 0.028  | 0.0 ± 0.0      | 0.293 ± 0.011    |
| MARS expert: long                     | 0.045 ± 0.005  | 0.158 ± 0.001 | 0.36 ± 0.03  | 0.47 ± 0.04  | -0.214 ± 0.010 | 0.21 ± 0.03  | -0.0276 ± 0.0001 | 0.89 ± 0.02 | 0.348 ± 0.020  | 0.0 ± 0.0      | 0.159 ± 0.007    |
| MARS expert: long-short               | -0.089 ± 0.062 | 0.082 ± 0.018 | -1.03 ± 0.59 | -1.39 ± 0.76 | -0.167 ± 0.066 | -0.49 ± 0.19 | -0.0121 ± 0.0035 | 0.95 ± 0.01 | 0.387 ± 0.013  | 0.0 ± 0.0      | 0.146 ± 0.008    |
| MARS expert: neutral                  | -0.066 ± 0.032 | 0.038 ± 0.005 | -1.88 ± 1.20 | -2.43 ± 1.33 | -0.108 ± 0.027 | -0.59 ± 0.13 | -0.0055 ± 0.0006 | 0.72 ± 0.02 | 0.318 ± 0.033  | 0.0 ± 0.0      | 0.120 ± 0.010    |
| MARS-RL (retrieval gate)              | -0.040 ± 0.006 | 0.119 ± 0.004 | -0.28 ± 0.04 | -0.36 ± 0.05 | -0.169 ± 0.014 | -0.23 ± 0.04 | -0.0214 ± 0.0007 | 0.73 ± 0.00 | 0.302 ± 0.017  | 0.0 ± 0.0      | 0.124 ± 0.006    |
| MARS uniform mix                      | -0.036 ± 0.027 | 0.095 ± 0.009 | -0.32 ± 0.27 | -0.42 ± 0.35 | -0.137 ± 0.018 | -0.25 ± 0.18 | -0.0158 ± 0.0018 | 0.70 ± 0.02 | 0.291 ± 0.015  | 0.0 ± 0.0      | 0.118 ± 0.006    |
| MARS gate, shuffled memory            | -0.034 ± 0.013 | 0.111 ± 0.004 | -0.25 ± 0.11 | -0.32 ± 0.15 | -0.163 ± 0.005 | -0.21 ± 0.09 | -0.0195 ± 0.0007 | 0.72 ± 0.02 | 0.301 ± 0.011  | 0.0 ± 0.0      | 0.123 ± 0.004    |

Figures 2–3 (`results/figures/fig2_equity_drawdown.png`, `fig3_risk_return.png`) show equity, drawdown and risk–return.

**Table IV — Cost stress (all frictions ×1, ×2, ×3; models trained at ×1).**

| model                                 |   ('sharpe', '1x') |   ('sharpe', '2x') |   ('sharpe', '3x') |   ('ann_return', '1x') |   ('ann_return', '2x') |   ('ann_return', '3x') |
|:--------------------------------------|-------------------:|-------------------:|-------------------:|-----------------------:|-----------------------:|-----------------------:|
| EW buy&hold (1x)                      |              0.847 |              0.846 |              0.845 |                  0.137 |                  0.137 |                  0.137 |
| EW daily-rebal (1x)                   |              0.849 |              0.831 |              0.813 |                  0.138 |                  0.135 |                  0.131 |
| EW daily-rebal (2x lev)               |              0.719 |              0.577 |              0.435 |                  0.204 |                  0.147 |                  0.094 |
| MARS expert: long                     |              0.231 |             -0.48  |             -1.166 |                  0.025 |                 -0.086 |                 -0.182 |
| MARS expert: long-short               |             -1.511 |             -3.382 |             -5.206 |                 -0.093 |                 -0.194 |                 -0.282 |
| MARS expert: neutral                  |             -1.672 |             -4.018 |             -6.281 |                 -0.06  |                 -0.139 |                 -0.21  |
| MARS gate, shuffled memory            |              0.058 |             -0.802 |             -1.605 |                  0.001 |                 -0.086 |                 -0.168 |
| MARS uniform mix                      |             -0.214 |             -1.542 |             -2.844 |                 -0.017 |                 -0.1   |                 -0.175 |
| MARS-RL (retrieval gate)              |              0.095 |             -0.782 |             -1.59  |                  0.004 |                 -0.086 |                 -0.168 |
| Min-variance (monthly)                |              0.789 |              0.78  |              0.77  |                  0.12  |                  0.118 |                  0.116 |
| Momentum top-7 (monthly)              |              0.6   |              0.528 |              0.455 |                  0.097 |                  0.083 |                  0.069 |
| PPO 50/50 long-short 2x (paper-style) |             -1.528 |             -3.69  |             -5.806 |                 -0.151 |                 -0.33  |                 -0.486 |
| PPO long-only 1x (plain reward)       |              0.316 |             -0.384 |             -1.044 |                  0.041 |                 -0.084 |                 -0.189 |
| PPO long-only 2x (paper-style reward) |              0.15  |             -0.588 |             -1.029 |                 -0.012 |                 -0.247 |                 -0.335 |

**Table V — Ablations (test, ×1).**

|                                   | Sharpe       | Ann. return    | Max DD         | Avg gross   |
|:----------------------------------|:-------------|:---------------|:---------------|:------------|
| Ablation: L/S, no CMDP/guard/twin | -0.31 ± 0.64 | -0.001 ± 0.001 | -0.005 ± 0.001 | 0.00 ± 0.00 |
| Ablation: MARS L/S, no twin       | -1.90 ± 0.17 | -0.119 ± 0.021 | -0.329 ± 0.046 | 0.97 ± 0.01 |
| MARS expert: long-short           | -1.51 ± 0.30 | -0.093 ± 0.017 | -0.291 ± 0.025 | 0.95 ± 0.01 |
| fixed-mars_ls                     | -1.51 ± 0.30 | -0.093 ± 0.017 | -0.291 ± 0.025 | 0.95 ± 0.01 |
| MARS expert: long                 | 0.23 ± 0.14  | 0.025 ± 0.022  | -0.273 ± 0.013 | 0.94 ± 0.01 |
| MARS expert: neutral              | -1.67 ± 0.69 | -0.060 ± 0.023 | -0.181 ± 0.065 | 0.71 ± 0.01 |
| MARS uniform mix                  | -0.21 ± 0.21 | -0.017 ± 0.014 | -0.156 ± 0.029 | 0.72 ± 0.01 |
| MARS gate, shuffled memory        | 0.06 ± 0.17  | 0.001 ± 0.016  | -0.187 ± 0.020 | 0.77 ± 0.01 |
| MARS-RL (retrieval gate)          | 0.10 ± 0.20  | 0.004 ± 0.019  | -0.184 ± 0.020 | 0.78 ± 0.01 |

**Table VI — Significance, test period (MARS-RL vs reference; "DSR" row: p-like value = 1 − DSR).**

|                                                            |   sharpe_diff |   ci_low |   ci_high |   p_boot_one_sided |    p_JKM |
|:-----------------------------------------------------------|--------------:|---------:|----------:|-------------------:|---------:|
| (0, 'EW daily-rebal (1x)')                                 |       -0.7155 |  -1.1624 |   -0.2758 |             0.999  |   0.001  |
| (0, 'PPO long-only 1x (plain reward)')                     |       -0.2047 |  -0.7333 |    0.3248 |             0.7795 |   0.4571 |
| (0, 'PPO long-only 2x (paper-style reward)')               |        0.0859 |  -0.4815 |    0.6657 |             0.38   |   0.7536 |
| (0, 'DSR (P[true SR>0] after selection among val trials)') |        0.1337 | nan      |  nan      |             0.9997 | nan      |
| (1, 'EW daily-rebal (1x)')                                 |       -0.5767 |  -1.073  |   -0.0616 |             0.989  |   0.049  |
| (1, 'PPO long-only 1x (plain reward)')                     |        0.0227 |  -0.5946 |    0.6631 |             0.444  |   0.9481 |
| (1, 'PPO long-only 2x (paper-style reward)')               |        0.1491 |  -0.3743 |    0.7014 |             0.282  |   0.646  |
| (1, 'DSR (P[true SR>0] after selection among val trials)') |        0.2724 | nan      |  nan      |             0.9992 | nan      |
| (2, 'EW daily-rebal (1x)')                                 |       -0.9689 |  -1.4879 |   -0.4846 |             1      |   0.0001 |
| (2, 'PPO long-only 1x (plain reward)')                     |       -0.4794 |  -1.0382 |    0.1071 |             0.942  |   0.1255 |
| (2, 'PPO long-only 2x (paper-style reward)')               |       -0.3993 |  -1.0065 |    0.2389 |             0.889  |   0.2056 |
| (2, 'DSR (P[true SR>0] after selection among val trials)') |       -0.1197 | nan      |  nan      |             0.9999 | nan      |

**Table VII — Digital-twin crash stress (8 synthetic 250-day paths).**

|                                       |   Acc. return |   Max DD |   Worst-path MDD |   CVaR5 daily |
|:--------------------------------------|--------------:|---------:|-----------------:|--------------:|
| EW buy&hold (1x)                      |        -0.202 |   -0.27  |           -0.317 |        -0.028 |
| EW daily-rebal (1x)                   |        -0.189 |   -0.261 |           -0.312 |        -0.028 |
| EW daily-rebal (2x lev)               |        -0.376 |   -0.478 |           -0.549 |        -0.055 |
| Min-variance (monthly)                |        -0.183 |   -0.257 |           -0.31  |        -0.027 |
| Momentum top-7 (monthly)              |        -0.268 |   -0.331 |           -0.399 |        -0.032 |
| PPO long-only 1x (plain reward)       |        -0.254 |   -0.311 |           -0.389 |        -0.029 |
| PPO long-only 2x (paper-style reward) |        -0.463 |   -0.542 |           -0.629 |        -0.059 |
| PPO 50/50 long-short 2x (paper-style) |        -0.134 |   -0.177 |           -0.274 |        -0.017 |
| MARS expert: long                     |        -0.249 |   -0.304 |           -0.398 |        -0.027 |
| MARS expert: long-short               |        -0.113 |   -0.132 |           -0.193 |        -0.011 |
| MARS expert: neutral                  |        -0.072 |   -0.081 |           -0.142 |        -0.006 |
| MARS-RL (retrieval gate)              |        -0.135 |   -0.163 |           -0.234 |        -0.014 |
| MARS uniform mix                      |        -0.134 |   -0.161 |           -0.215 |        -0.013 |
| MARS gate, shuffled memory            |        -0.132 |   -0.16  |           -0.228 |        -0.013 |

**Table VIII — Track A: authors' own simulator and agents on our data (not comparable to Tables II–VII).**

| run    |   test_acc_return |   test_ann_return |   test_ann_vol |   test_sharpe |   test_sortino |   test_max_drawdown |   test_calmar |   test_cvar5_daily |
|:-------|------------------:|------------------:|---------------:|--------------:|---------------:|--------------------:|--------------:|-------------------:|
| a2c_s0 |             0.589 |             0.168 |          0.313 |         0.652 |          0.944 |              -0.273 |         0.615 |             -0.044 |
| a2c_s1 |             0.361 |             0.109 |          0.278 |         0.51  |          0.737 |              -0.297 |         0.367 |             -0.038 |
| a2c_s2 |             0.177 |             0.056 |          0.276 |         0.335 |          0.487 |              -0.401 |         0.14  |             -0.037 |
| ppo_s0 |             0.249 |             0.077 |          0.21  |         0.46  |          0.645 |              -0.229 |         0.337 |             -0.031 |
| ppo_s1 |             0.499 |             0.145 |          0.226 |         0.712 |          1.037 |              -0.231 |         0.627 |             -0.031 |
| ppo_s2 |             0.368 |             0.111 |          0.2   |         0.623 |          0.895 |              -0.202 |         0.547 |             -0.027 |

Figures 4–9 give cost stress, ablation, training/dual diagnostics, crash stress, Track A equity and exposure.

## VII. Discussion
**(1) No Sharpe improvement.** MARS-RL's test Sharpe (0.10 ± 0.20) is below equal-weight (0.85) with paired-bootstrap 95% CIs of the difference below zero in all three seeds; against the paper-style PPO agents the differences are not significant. **(2) Tail-risk reduction is real but cheap to obtain.** Max drawdown and crash-stress losses are lower mainly because gross exposure is lower (0.78 average) and because the learned long-short/neutral experts hold offsetting positions; the equally low-exposure uniform mix and shuffled gate give similar drawdowns. **(3) Retrieval not shown to help.** Gate vs shuffled memory: 0.10 vs 0.06 Sharpe, well within seed noise (sd ≈ 0.2). **(4) Cost fragility.** Daily turnover of 0.3–0.8 makes all RL agents lose at 2×–3× costs, while low-turnover baselines do not. **(5) Long-short experts fail** in this sample (validation and test Sharpe negative), consistent with a persistent positive drift in large caps plus short-side costs; we did not tune them. **(6) Track A.** The authors' own agents reach test Sharpe about 0.5–0.6 in their simulator (zero slippage, separate long/short ledgers), above our re-implementations in Track B, but still below buy-and-hold equal weight in our simulator (0.85); the tracks differ in cost model and cannot be compared directly. **(7) Digital twin.** Twin training improved the long-short expert (Sharpe −1.90 → −1.51) but both are negative; only 0% vs 50% twin was tested.

## VIII. Limitations and Threats to Validity
One market, one three-year test window, three seeds; wide confidence intervals. Survivorship bias (current DJIA members). Yahoo adjusted prices; VIX from a third-party mirror (Track A only). Single-ledger margin account in Track B differs from the authors'. Re-implemented baselines may not match the authors' agents. Hyper-parameters (CMDP step sizes, budgets, guard budget B, gate k/β) were not tuned, so the negative result may partly reflect under-tuning, in particular the absence of a turnover/cost-aware reward. Test-set exposure: test metrics from stage-1 (40k-step) models were printed once during pipeline smoke-testing; no design change followed, but the test set is not strictly pristine. Components listed in IV.E were not evaluated; no claim is made about them.

## IX. Conclusion
In a replication-and-extension study with a locked protocol, constrained RL with learned leverage, a conformal guard, twin augmentation and a retrieval gate lowered drawdown and eliminated margin calls but did not improve risk-adjusted return over simple baselines or the base paper's agent family, and all learned agents were fragile to higher trading costs. Future work should add turnover-aware rewards, tune the CMDP budgets on validation, widen the test window (multiple markets/periods), and evaluate the untested components (SSM encoder, distillation, LLM news signals) with the same locked protocol.

## References
[1] J. Gu et al., "Incorporating Realistic Margin Constraints: A Data-Driven DRL Framework for Advanced Portfolio Management," *IEEE Trans. Knowl. Data Eng.*, 2026, doi:10.1109/TKDE.2026.3701681.
[2] X.-Y. Liu et al., "FinRL: A deep reinforcement learning library for automated stock trading in quantitative finance," *NeurIPS Deep RL Workshop*, 2020.
[3] J. Schulman et al., "Proximal policy optimization algorithms," arXiv:1707.06347, 2017.
[4] E. Altman, *Constrained Markov Decision Processes*, CRC Press, 1999.
[5] J. Achiam et al., "Constrained policy optimization," *ICML*, 2017.
[6] V. Vovk, A. Gammerman, G. Shafer, *Algorithmic Learning in a Random World*, Springer, 2005; A. Angelopoulos and S. Bates, "A gentle introduction to conformal prediction," 2021.
[7] D. Politis and J. Romano, "The stationary bootstrap," *JASA*, 89(428), 1994.
[8] P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," *NeurIPS*, 2020.
[9] J. Jobson and B. Korkie, "Performance hypothesis testing with the Sharpe and Treynor measures," *J. Finance*, 1981; M. Memmel, "Performance hypothesis testing with the Sharpe ratio," *Finance Letters*, 2003.
[10] D. Bailey and M. López de Prado, "The deflated Sharpe ratio," *J. Portfolio Management*, 40(5), 2014.
[11] A. Raffin et al., "Stable-Baselines3: Reliable reinforcement learning implementations," *JMLR*, 22, 2021.
