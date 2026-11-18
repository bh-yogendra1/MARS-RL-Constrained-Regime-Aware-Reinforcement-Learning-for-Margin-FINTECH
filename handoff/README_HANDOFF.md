# MARS-RL — Handoff package (for Gemini Antigravity / any new agent)

Read order: **1) `PROJECT_CONTEXT.xml` → 2) this file → 3) `docs/ARCHITECTURE.md` → 4) `docs/EXPERIMENT_LOG.md` → 5) `docs/NEXT_STEPS_NEW_DATASET.md` → 6) `docs/ANTIGRAVITY_STARTER_PROMPT.md`** (copy-paste first prompt).

## 1. One-paragraph summary
Base paper: Gu et al., IEEE TKDE 2026 (margin-trading DRL on 28 DJIA stocks; best test Sharpe 1.05–1.16, ann. return 19–31%). We built **MARS-RL** (CMDP experts + learned leverage + conformal guard + digital twin + retrieval gate) and a stronger **v2** (residual-on-equal-weight PPO with turnover control, seed-ensemble, train+val refit). On the same 2020-05→2023-05 test window our best is **Sharpe 0.85, ann. return 11.3%, max DD −19.2%** with realistic costs — i.e. equal to an equal-weight portfolio, far lower drawdown than the paper's agents, **but it does NOT beat the paper's return/Sharpe**. The user wants to beat it; the next step is a much bigger dataset (already downloaded by the user) to look for a real predictive edge.

## 2. Status table (test 2020-05→2023-05)
| Model | Ann. return | Sharpe | Max DD | Calmar |
|---|---|---|---|---|
| Paper: Margin Trader Long PPO | 30.9% | 1.05 | −25.8% | 1.20 |
| Paper: Margin Trader L&S PPO | 19.2% | 1.16 | −18.2% | 1.06 |
| Equal-weight 1× (ours, common costs) | 13.8% | 0.85 | −21.1% | 0.65 |
| MARS-RL v1 | 0.4% | 0.10 | −18.4% | 0.03 |
| **v2_b refit ensemble (best)** | **11.3%** | **0.85** | **−19.2%** | **0.59** |
| Authors' own code on our data (Track A, A2C / PPO) | 11.1% / 11.1% | 0.50 / 0.60 | −32% / −22% | – |

## 3. What is in the package
```
handoff/PROJECT_CONTEXT.xml      machine-readable context (goal, results, protocol, tasks)
handoff/README_HANDOFF.md        this file
handoff/docs/…                   architecture, experiment log, next steps, starter prompt
src/                             all code (see §4)
tests/test_core.py               8 unit tests (ledger, no-look-ahead, guard, margin call, determinism, gate embargo)
scripts/                         resumable runners (run_stage.sh, run_track_a.sh, refit_cmds.txt)
colab/MARS_RL_mega_dataset_downloader.ipynb   builds the NEW big dataset (user already ran it)
notebooks/MARS_RL_results.ipynb  executed results notebook (old 28-stock data)
results/{tables,figures,trackA,trackB,trackB_v2}   measured outputs
data/raw, data/processed, data/*.csv              OLD 28-ticker data (+ manifest)
models/{stage2,v2,v3,v4,refit}   trained PPO checkpoints (large; ship separately)
docs/MANUSCRIPT_MARS_RL.md, README.md, README_Hinglish.md, docs/ADDENDUM_v2.md   earlier write-ups
third_party/margin_trader        authors' unmodified code (MIT)
```

## 4. Code map (src/)
| File | Role |
|---|---|
| `config.py` | ALL assumptions: CostConfig, EnvConfig (mode long/ls/neutral/split/**resid**, reward plain/mt/mars, guard none/vol/conformal, turn_pen, smooth, kappa), PPOConfig, GateConfig, ROLES (v1: finrl_long, mt_long, mt_split, mars_long/ls/neutral, plain_ls, mars_ls_notwin; v2_a–f; v3_a–d; v4_a–b), SPLITS |
| `data.py` | `Panel`, `build_panel` (10 stock features, 9 market-state features, EWMA covariance, conformal quantile), `load_panel(csv)` from long OHLCV csv |
| `data_prep.py` | provenance (SHA-256 manifest), audit, OHLC indicators (stockstats), Track A file |
| `margin_env.py` | Track B simulator `MarginTradingEnv` + `weights_from_action` + `run_episode` (ledger, margin calls, costs, guard, twin sampling) |
| `twin.py` | `DigitalTwin` (regime block bootstrap), `fidelity_table` |
| `train.py` | PPO + Lagrangian dual ascent; `--refit` trains on train+val (last checkpoint) |
| `gating.py` | `RetrievalGate`, `expert_paths` (kNN retrieval MoE with embargo) |
| `baselines.py` | EW, EW-rebalanced, 2× EW, min-variance, momentum |
| `evaluate.py`, `stress_twin.py` | locked evaluation (cost 1×/2×/3×) and twin crash stress |
| `ensemble_val.py`, `final_ensemble.py`, `final_refit.py` | seed-ensembles; validation-only compare; locked tests |
| `alpha.py` | supervised cross-sectional alpha (ridge/GBM) + rank-IC |
| `metrics.py` | perf metrics, stationary bootstrap Sharpe diff, JKM, deflated Sharpe |
| `track_a.py` | authors' own env/agents on our data |
| `report.py`, `compare_base_paper.py`, `export_processed.py` | tables/figures/exports |

## 5. How to run (CPU, ~1 min per 150k-step model)
```bash
pip install -r requirements.txt
python -m pytest tests -q
python -m src.train --role v2_b --seed 0 --steps 150000 --out models/v2          # validation-selected checkpoint
python -m src.train --role v2_b --seed 0 --steps 150000 --refit --out models/refit  # train+val refit
python src/ensemble_val.py models/v2 v2_b                                          # validation-only ensemble check
python src/final_refit.py                                                          # LOCKED test (run once per final config)
```
Always keep jobs resumable (the sandbox kills background jobs on pauses).

## 6. Rules the new agent MUST follow (honesty protocol)
1. Select models on **validation only**; run the **test once** per final configuration; log every variant tried (current n_trials after v1 = 12 + alpha experiment).
2. Never claim "beats the paper" unless the locked test shows it (multi-seed, CI). Report negative results.
3. No look-ahead: features at t use data ≤ close t; macro/fundamentals use release lags.
4. Keep Track A (authors' simulator) vs Track B (our common-cost simulator) separate.
5. Disclose: S&P 500 list = today's constituents (survivorship bias); test window was peeked once in the v1 stage-1 smoke test.

## 7. Next work (new dataset) — see `docs/NEXT_STEPS_NEW_DATASET.md`
Inspect the new data → alpha IC check (go/no-go) → universe selection + macro/FF/CBOE features → retrain v2-style ensemble → validation selection → refit → single locked test → regenerate paper/README/notebook.

---
## Hinglish saaransh (same content, short)
* Base paper ka best Sharpe 1.05–1.16 aur return 19–31% hai. Hamara best (v2_b refit ensemble) **Sharpe 0.85, return 11.3%, drawdown −19.2%** hai — equal-weight ke barabar, drawdown kam, lekin **paper beat nahi hua**.
* Authors ka apna code hamare data par sirf 11.1% deta hai, isliye paper ka headline number reproduce nahi hua.
* Supervised alpha ka validation IC ≈ 0 tha → 28 stocks + technical features me asli edge nahi mila. Isiliye ab bada dataset (S&P 500, macro, factors, VIX, news) use karna hai.
* Rules: validation par select, test sirf ek baar, sab variants ginna, negative result honestly likhna, look-ahead nahi, Track A/B alag.
* Aage ka kaam: naya data inspect → alpha IC → universe + features → retrain → refit → locked test → tab hi final files generate.
