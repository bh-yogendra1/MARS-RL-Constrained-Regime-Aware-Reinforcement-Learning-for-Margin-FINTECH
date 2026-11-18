# MARS-RL — Margin-constrained, Adaptive, Regime-aware, Safe Reinforcement Learning for Portfolio Management (Hinglish README)

Yeh study **Gu et al., "Incorporating Realistic Margin Constraints: A Data-Driven DRL Framework for Advanced Portfolio Management", IEEE TKDE 2026** (DOI 10.1109/TKDE.2026.3701681; authors ka code: github.com/JingyiGu/Margin-Trader, MIT, pinned commit `7b7fa9e`) ko base paper maan kar banayi gayi replication + extension hai.

> ## Sabse pehle yeh padho — result ka honest summary
> * **MARS-RL, base paper ki setting aur simple equal-weight portfolio, dono ko risk-adjusted return me beat NAHI kar paya.** Test-period Sharpe (3-seed mean, 1× cost): MARS-RL **0.10**, paper-style 2× long-only PPO **0.15**, equal-weight daily-rebalanced **0.85**. Equal-weight ke muqable Sharpe ki kami statistically significant hai (paired-bootstrap 95% CI of Sharpe difference teeno seeds me 0 se neeche: [−1.16,−0.28], [−1.07,−0.06], [−1.49,−0.48]; JKM p = 0.001, 0.049, 0.0001). Paper-style PPO agents ke muqable difference significant nahi hai (CI me 0 aata hai).
> * MARS-RL ne **tail risk kam kiya**: test max drawdown **−18.4%** vs **−54.0%** (paper-style 2× long-only) aur −21.1% (equal-weight); saare learned models me **0 margin call**; synthetic crash stress me MDD −16.3% vs −54.2% (paper-style 2×) aur −26.1% (equal-weight). Yeh risk-reduction hai, lekin lagbhag poora return dene ki keemat par.
> * Retrieval gate uniform mixing se behtar hai (Sharpe 0.10 vs −0.21), lekin **shuffled-memory gate (0.06) se statistically alag nahi** → retrieval khud value add karta hai, iska koi saboot nahi.
> * Saare RL agents cost ke prati bahut sensitive hain (daily turnover 0.3–0.8). 2×/3× cost par har learned model ka Sharpe negative ho jata hai, jabki equal-weight ≈ 0.83 par rehta hai.
> * Authors ka **apna** code hamare data par (Track A, alag simulator, zero slippage) test Sharpe 0.50 ± 0.16 (A2C) aur 0.60 ± 0.13 (PPO) deta hai — meri re-implemented agents se zyada. Track A aur Track B ke numbers ko seedha compare **mat** karo.
> * Deliver/evaluate nahi hua: LLM-news agent (sirf code), Mamba encoder, knowledge-distillation student, causal reasoning module, factorial 2×2×2 design, twin-ratio sweep (0/25/75%), matched fixed-leverage ablation. Neeche "Limitations / kya NAHI kiya" dekho.

## 1. Package me kya hai

| Path | Content |
|---|---|
| `notebooks/MARS_RL_results.ipynb` | **Executed** notebook: data audit, processed data, saari tables, saare figures, statistics, conclusions |
| `docs/MANUSCRIPT_MARS_RL.md` | IEEE-style English manuscript (saare numbers neeche ke CSVs se) |
| `README.md` / `README_Hinglish.md` | yeh file (English) / wahi content Hinglish me |
| `data/raw/` | Raw data: 28 DJIA tickers ka Yahoo-adjusted OHLCV (user ne download kiya), CBOE VIX (github.com/datasets/finance-vix) |
| `data/manifests/raw_manifest.csv` | Har raw file ka SHA-256, rows, source, timestamp |
| `data/features/` | standard OHLC indicator tables; `trackA_finrl_format.csv` (authors ki script ka exact input format) |
| `data/processed/model_input_features_panel.csv` | **Final processed data jo models ko actually diya gaya** (OHLCV, 10 stock features, 9 market-state features, vol, conformal quantile, split label) |
| `data/{full,train,val,test}.csv` | user ki di hui original files |
| `reports/data_quality.json` | automated data audit |
| `src/` | saara code (§4) · `scripts/` runners · `tests/` unit tests (8, sab pass) |
| `models/stage2/` | trained PPO checkpoints (8 roles × 3 seeds) + training logs (`*.json`) |
| `results/trackB/` | daily equity curves, metrics (`metrics_main.csv`, `metrics_twin_crash.csv`) |
| `results/trackA/` | authors-code runs (A2C/PPO × 3 seeds): equity curves + metrics |
| `results/tables/`, `results/figures/` | paper me use hui har table (CSV+MD) aur figure |
| `third_party/margin_trader/` | authors ka unmodified environment/agent code (MIT) |

## 2. Problem aur base paper
Base paper A2C/PPO agents ko margin-trading environment me train karta hai (long, short, leverage, maintenance-margin warning 40% / strict 30%, 0.1% fees), 28 DJIA stocks par 2009–2023 (train 2009–2018, val 2019/01–2020/04, test 2020/05–2023/05) aur margin-aware agents ke liye high returns report karta hai. Meine yeh gaps target kiye: (i) simulator me slippage/borrow/financing cost nahi, (ii) fixed leverage, (iii) explicit risk constraint nahi — margin call sirf heuristically penalise hota hai, (iv) ek hi regime-agnostic policy, (v) training aur evaluation dono ke liye ek hi historical path, (vi) seeds par statistical testing nahi.

## 3. Method — MARS-RL (jo implement aur test hua)
1. **Constrained-MDP experts** (PPO): long-only, long-short, market-neutral; **learned leverage** L = L_max·(a+1)/2; reward = growth − μ_margin·1[margin call] − μ_dd·drawdown-excess − η·downside²; μ Lagrangian dual ascent se (clip [0,5]; budgets 0.001 margin calls/step, 0.01 dd-excess/step; α untuned).
2. **Conformal exposure guard** execution par: score |r_EW,t|/σ_EW,t−1 (EWMA λ=0.94), rolling 95% quantile (250 din); positions ko scale-down kiya jata hai taaki q·√(wᵀΣw) ≤ 3%. Exposure kabhi badhata nahi (unit-tested).
3. **Regime-switching block-bootstrap digital twin** (sirf train data, 3 vol regimes, Markov chain, geometric blocks mean 10): MARS training ke 50% episodes synthetic; crash-stress paths (drift −0.15%/day, vol ×1.3). Fidelity table: `results/tables/table_twin_fidelity.md`.
4. **Retrieval-guided mixture-of-experts gate**: embedding = 9 standardised market-state features; past states ki memory + har expert ka 20-din forward Sharpe; k=30 neighbours; softmax(β=10), uniform ki taraf 0.5 shrink; **outcome embargo** (s+H ≤ t; future corrupt karke unit-test kiya).
5. **Common simulator (Track B)**: decision close t par → execution open t+1 par; 0.1%/side fee + 2 bp slippage; 0.5%/yr borrow fee; 4%/yr loan; margin warning 40%, call 30% → liquidate to 50% + 0.1% fee; $100k start. Saare baselines aur agents isi simulator me.
6. **Track A**: authors ke apne env/agents, 50k steps, hamara data. **Track C**: cost stress (1×/2×/3×) aur twin crash stress.

Evaluation protocol: train 2009/01–2018/12; validation 2019/01–2020/04 (sirf checkpoint selection); test 2020/05–2023/05 (har configuration ka single locked run, valtest ek account me chalaya phir date se slice kiya). 3 seeds. rf=0 (FinRL default). Statistics: paired stationary bootstrap (2000 reps, block 10), Jobson–Korkie–Memmel, deflated Sharpe.

## 4. Code map
`src/config.py` (saari assumptions) · `data_prep.py` (provenance, audit, features) · `data.py` (panel, point-in-time features, EWMA cov, conformal quantile) · `margin_env.py` (simulator) · `twin.py` · `train.py` (PPO + dual ascent) · `gating.py` · `baselines.py` · `evaluate.py` · `stress_twin.py` · `metrics.py` · `track_a.py` · `report.py` · `export_processed.py`.
LLM/agentic layer (schemas, staleness decay; LLM kabhi trade nahi karta) **sirf code hai, untested** — sandbox me news data ya model weights nahi the.

## 5. Reproduce kaise karein
```bash
pip install -r requirements.txt            # numpy pandas scipy gymnasium stable-baselines3 stockstats matplotlib tabulate nbformat
python -m src.data_prep && python -m src.export_processed
bash scripts/run_stage.sh 150000 models/stage2 "0 1 2" "finrl_long mt_long mt_split mars_long mars_ls mars_neutral plain_ls mars_ls_notwin"
python -m src.evaluate && python -m src.stress_twin && bash scripts/run_track_a.sh && python -m src.report
python -m pytest tests -q
```
CPU-only, 2 cores par total ≈ 40 min. Seeds fixed hain, lekin PyTorch/SB3 version ya hardware badalne par results thode alag ho sakte hain.

## 6. Key results (test 2020/05–2023/05, 1× costs, 3 seeds ka mean ± sd; poori tables `results/tables/` me)
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
| Track A (authors ka code): A2C / PPO | 0.111 / 0.111 | 0.50 / 0.60 | −0.324 / −0.221 | n/a |

## 7. Limitations / kya NAHI kiya (zaroor padho)
* **Risk-adjusted return me koi improvement nahi mila.** Main iska claim nahi karta. Pre-registered hypothesis ki MARS base-paper agents se Sharpe improve karega — **support nahi hui**.
* Learned long-short aur neutral experts validation aur test dono me paisa lose karte hain (28 large-cap stocks ko bull-dominated sample me short karna + costs); inhe aur tune nahi kiya.
* Agents 150k steps train hue, hyper-parameters (α, budgets, network, entropy) untuned. Zyada turnover main failure mode hai; turnover/cost-aware reward **try nahi kiya**.
* **Test-period peek (disclosure):** evaluation pipeline ko 40k-step stage-1 models par smoke-test karte waqt meine ek baar test metrics print kiye the. Uski wajah se koi design change nahi hua, lekin isliye test set 100% untouched nahi hai. Uske baad ke saare design decisions validation-only numbers par the (aur maine koi decision nahi liya).
* Implement/evaluate nahi hue: Mamba/SSM encoder, distillation student (DAgger), causal-reasoning module, LLM/news agent, factorial 2×2×2 grid, twin-ratio sweep (0/25/75), matched fixed-leverage ablation, 3-window rolling re-estimation. Sirf twin 0% vs 50% compare hua (`mars_ls_notwin` vs `mars_ls`: twin se L/S ko madad mili: Sharpe −1.90 → −1.51, lekin dono negative).
* Track B baselines paper ke description se **re-implementations** hain (authors ka code nahi), simplified single-ledger account me; authors ka ledger long/short accounts alag rakhta hai. Track A authors ka apna code hai lekin zero slippage, isliye dono tracks comparable nahi.
* Data: 28 tickers (DOW, WBA missing), yfinance adjusted prices (survivorship: current DJIA members), VIX third-party GitHub mirror se; DJIA index ya risk-free rate nahi (rf=0). Backtests trading advice nahi hain.
* Ek market, ek test window (3 saal, ek regime sequence); sirf 3 seeds; 95% intervals bahut chaude (wide) hain.

## 8. License / citation
Authors ka code MIT ke tahat (`third_party/`). Base paper cite karein: Gu et al., IEEE TKDE 2026, DOI 10.1109/TKDE.2026.3701681.
