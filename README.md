# MARS-RL — Margin-constrained, Adaptive, Regime-aware, Safe Reinforcement Learning for Portfolio Management

Replication-and-extension study built on **Gu et al., "Incorporating Realistic Margin Constraints: A Data-Driven DRL Framework for Advanced Portfolio Management", IEEE TKDE 2026** (DOI 10.1109/TKDE.2026.3701681; authors' code: github.com/JingyiGu/Margin-Trader, MIT, pinned commit `7b7fa9e`).

> ## Read this first — honest summary of the outcome
> * **MARS-RL did NOT outperform the base paper's setting, nor a plain equal-weight portfolio, on risk-adjusted return.** Test-period Sharpe (3-seed mean, 1× costs): MARS-RL **0.10**, paper-style 2× long-only PPO **0.15**, equal-weight daily-rebalanced **0.85**. The Sharpe deficit vs equal-weight is statistically significant in all 3 seeds (paired-bootstrap 95% CIs of the Sharpe difference all lie below 0: [−1.16,−0.28], [−1.07,−0.06], [−1.49,−0.48]; JKM p = 0.001, 0.049, 0.0001). Against the paper-style PPO agents the differences are not significant (CIs include 0).
> * MARS-RL **did** reduce tail risk: test max drawdown **−18.4%** vs **−54.0%** (paper-style 2× long-only) and −21.1% (equal-weight); **0 margin calls** for all learned models; in synthetic crash stress, MDD −16.3% vs −54.2% (paper-style 2×) and −26.1% (equal-weight). That is a risk-reduction result obtained at the price of almost all of the return.
> * The retrieval gate beats uniform mixing (Sharpe 0.10 vs −0.21) but is **statistically indistinguishable from a shuffled-memory gate** (0.06) → no evidence that retrieval itself adds value.
> * All RL agents are very cost-sensitive (daily turnover 0.3–0.8). At 2×/3× costs every learned model has negative Sharpe, while equal-weight stays ≈ 0.83.
> * The authors' **own** code on our data (Track A, different simulator, zero slippage) gives test Sharpe 0.50 ± 0.16 (A2C) and 0.60 ± 0.13 (PPO) — higher than my re-implemented agents. Track A and Track B must **not** be compared number-to-number.
> * Not delivered / not evaluated: LLM-news agent (code-only), Mamba encoder, knowledge-distillation student, causal reasoning module, factorial 2×2×2 design, twin-ratio sweep (0/25/75%), matched fixed-leverage ablation. See "Limitations / what was NOT done".

## 1. What is in this package

| Path | Content |
|---|---|
| `notebooks/MARS_RL_results.ipynb` | **Executed** notebook: data audit, processed data, all tables, all figures, statistics, conclusions |
| `docs/MANUSCRIPT_MARS_RL.md` | IEEE-style English manuscript (all numbers from the CSVs below) |
| `README.md` / `README_Hinglish.md` | this file (English) / same content in Hinglish |
| `data/raw/` | Raw data: Yahoo-adjusted OHLCV of 28 DJIA tickers (user-downloaded), CBOE VIX (github.com/datasets/finance-vix) |
| `data/manifests/raw_manifest.csv` | SHA-256, rows, source, timestamp of every raw file |
| `data/features/` | standard OHLC indicator tables; `trackA_finrl_format.csv` (exact input format of the authors' script) |
| `data/processed/model_input_features_panel.csv` | **Final processed data actually fed to the models** (OHLCV, 10 stock features, 9 market-state features, vol, conformal quantile, split label) |
| `data/{full,train,val,test}.csv` | the files originally supplied by the user |
| `reports/data_quality.json` | automated data audit |
| `src/` | all code (see §4) · `scripts/` runners · `tests/` unit tests (8, all pass) |
| `models/stage2/` | trained PPO checkpoints (8 roles × 3 seeds) + training logs (`*.json`) |
| `results/trackB/` | per-day equity curves, metrics (`metrics_main.csv`, `metrics_twin_crash.csv`) |
| `results/trackA/` | authors'-code runs (A2C/PPO × 3 seeds): equity curves + metrics |
| `results/tables/`, `results/figures/` | every table (CSV+MD) and figure used in the paper |
| `third_party/margin_trader/` | the authors' unmodified environment/agent code (MIT) |

## 2. Problem and base paper
The base paper trains A2C/PPO agents in a margin-trading environment (long, short, leverage, maintenance-margin warning 40% / strict 30%, 0.1% fees) on 28 DJIA stocks 2009–2023 (train 2009–2018, val 2019/01–2020/04, test 2020/05–2023/05) and reports high returns for margin-aware agents. Gaps I targeted: (i) no slippage/borrow/financing costs in the simulator, (ii) fixed leverage, (iii) no explicit risk constraint — margin calls are only penalised heuristically, (iv) single regime-agnostic policy, (v) single historical path for training and evaluation, (vi) no statistical testing across seeds.

## 3. Method — MARS-RL (what is implemented and tested)
1. **Constrained-MDP experts** (PPO): long-only, long-short, market-neutral; **learned leverage** L = L_max·(a+1)/2; reward = growth − μ_margin·1[margin call] − μ_dd·drawdown-excess − η·downside²; μ's by Lagrangian dual ascent (clip [0,5]; budgets 0.001 margin calls/step, 0.01 dd-excess/step; α untuned).
2. **Conformal exposure guard** at execution: score |r_EW,t|/σ_EW,t−1 (EWMA λ=0.94), rolling 95% quantile (250 d); positions are scaled down so q·√(wᵀΣw) ≤ 3%. Never increases exposure (unit-tested).
3. **Regime-switching block-bootstrap digital twin** (train data only, 3 vol regimes, Markov chain, geometric blocks mean 10): 50% of MARS training episodes are synthetic; crash-stress paths (drift −0.15%/day, vol ×1.3). Fidelity table in `results/tables/table_twin_fidelity.md`.
4. **Retrieval-guided mixture-of-experts gate**: embedding = 9 standardised market-state features; memory of past states + each expert's 20-day forward Sharpe; k=30 neighbours; softmax(β=10), 0.5 shrink toward uniform; **outcome embargo** (s+H ≤ t; unit-tested by corrupting the future).
5. **Common simulator (Track B)**: decision at close t → execute at open t+1; 0.1%/side fee + 2 bp slippage; 0.5%/yr borrow fee; 4%/yr loan; margin warning 40%, call 30% → liquidate to 50% + 0.1% fee; $100k start. All baselines and agents run in the same simulator.
6. **Track A**: the authors' own env/agents, 50k steps, our data. **Track C**: cost stress (1×/2×/3×) and twin crash stress.

Evaluation protocol: train 2009/01–2018/12; validation 2019/01–2020/04 (checkpoint selection only); test 2020/05–2023/05 (single locked run per configuration, valtest run as one account then sliced). 3 seeds. rf=0 (FinRL default). Statistics: paired stationary bootstrap (2000 reps, block 10), Jobson–Korkie–Memmel, deflated Sharpe.

## 4. Code map
`src/config.py` (all assumptions) · `data_prep.py` (provenance, audit, features) · `data.py` (panel, point-in-time features, EWMA cov, conformal quantile) · `margin_env.py` (simulator) · `twin.py` · `train.py` (PPO + dual ascent) · `gating.py` · `baselines.py` · `evaluate.py` · `stress_twin.py` · `metrics.py` · `track_a.py` · `report.py` · `export_processed.py`.
LLM/agentic layer (schemas, staleness decay; LLM never trades) is **code-only and untested** — there were no news data or model weights in the sandbox.

## 5. Reproduce
```bash
pip install -r requirements.txt            # numpy pandas scipy gymnasium stable-baselines3 stockstats matplotlib tabulate nbformat
python -m src.data_prep && python -m src.export_processed
bash scripts/run_stage.sh 150000 models/stage2 "0 1 2" "finrl_long mt_long mt_split mars_long mars_ls mars_neutral plain_ls mars_ls_notwin"
python -m src.evaluate && python -m src.stress_twin && bash scripts/run_track_a.sh && python -m src.report
python -m pytest tests -q
```
CPU-only, ≈ 40 min total on 2 cores. Seeds are fixed, but PyTorch/SB3 results can differ slightly across versions/hardware.

## 6. Key results (test 2020/05–2023/05, 1× costs, mean ± sd over 3 seeds; full tables in `results/tables/`)
| Model | Ann. return | Sharpe | Max DD | Avg gross |
|---|---|---|---|---|
| EW daily-rebalanced 1× | 0.138 | 0.85 | −0.211 | 1.00 |
| EW daily-rebalanced 2× | 0.204 | 0.72 | −0.414 | 2.00 |
| Min-variance (monthly) | 0.120 | 0.79 | −0.212 | 1.00 |
| PPO long-only 1× | 0.041 ± 0.010 | 0.32 ± 0.06 | −0.269 | 1.00 |
| PPO long-only 2× (paper-style) | −0.012 ± 0.045 | 0.15 ± 0.12 | −0.540 | 2.00 |
| PPO 50/50 long-short 2× (paper-style) | −0.151 ± 0.080 | −1.53 ± 0.92 | −0.444 | 2.00 |
| MARS expert: long | 0.025 ± 0.022 | 0.23 ± 0.14 | −0.273 | 0.94 |
| **MARS-RL (retrieval gate)** | 0.004 ± 0.019 | 0.10 ± 0.20 | **−0.184** | 0.78 |
| MARS uniform mix / shuffled memory | −0.017 / 0.001 | −0.21 / 0.06 | −0.156 / −0.187 | 0.72 / 0.77 |
| Track A (authors' code): A2C / PPO | 0.111 / 0.111 | 0.50 / 0.60 | −0.324 / −0.221 | n/a |

## 7. Limitations / what was NOT done (please read)
* **No improvement in risk-adjusted return was obtained.** I do not claim it. The pre-registered hypothesis that MARS improves Sharpe over the base-paper agents is **not supported**.
* Learned long-short and neutral experts lose money on both validation and test (shorting 28 large caps in a bull-dominated sample plus costs); I did not tune them further.
* Agents were trained for 150k steps with untuned hyper-parameters (α, budgets, network, entropy). Heavy turnover is the main failure mode; a turnover/cost-aware reward was **not** tried.
* **Test-period peek (disclosure):** while smoke-testing the evaluation pipeline on 40k-step stage-1 models I printed test metrics once. No design change was made because of it, but the test set is therefore not 100% untouched. Validation-only numbers were used for all subsequent design decisions (and I made none).
* Not implemented/evaluated: Mamba/SSM encoder, distillation student (DAgger), causal-reasoning module, LLM/news agent, factorial 2×2×2 grid, twin-ratio sweep (0/25/75), matched fixed-leverage ablation, 3-window rolling re-estimation. Only twin 0% vs 50% is compared (`mars_ls_notwin` vs `mars_ls`: twin helped L/S: Sharpe −1.90 → −1.51, but both negative).
* Track B baselines are **re-implementations** from the paper's description (not the authors' code) in a simplified single-ledger account; the authors' ledger separates long/short accounts. Track A uses the authors' own code but zero slippage, so the two tracks are not comparable.
* Data: 28 tickers (DOW, WBA missing), yfinance adjusted prices (survivorship: current DJIA members), VIX from a third-party GitHub mirror; no DJIA index or risk-free rate (rf=0). Backtests are not trading advice.
* Single market, single test window (3 years, one regime sequence); 3 seeds only; 95% intervals are wide.

## 8. License / citation
Authors' code under MIT (`third_party/`). Cite the base paper: Gu et al., IEEE TKDE 2026, DOI 10.1109/TKDE.2026.3701681.
