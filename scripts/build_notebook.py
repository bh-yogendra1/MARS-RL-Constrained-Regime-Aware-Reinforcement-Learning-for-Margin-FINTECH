import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s)); code = lambda s: C.append(nbf.v4.new_code_cell(s))
md("""# MARS-RL — Results notebook (executed)
**M**argin-constrained, **A**daptive, **R**egime-aware **S**afe RL on the Gu et al. (TKDE 2026) setting.

This notebook loads the saved, locked results and shows every table/figure of the paper. **All numbers are measured outputs, including negative ones.**
Set `RETRAIN=True` in the last section to regenerate everything from scratch (≈ 40 min on 2 CPU cores).""")
code("""import sys, os, json, glob; from pathlib import Path
import pandas as pd, numpy as np
from IPython.display import display, Image, Markdown
ROOT = Path(os.getcwd()).parent if Path(os.getcwd()).name=='notebooks' else Path(os.getcwd())
sys.path.insert(0, str(ROOT)); pd.set_option('display.width', 250, 'display.max_columns', 30)
T = ROOT/'results/tables'; F = ROOT/'results/figures'
show = lambda name: display(Markdown((T/f'{name}.md').read_text()))""")
md("## 1. Data provenance and audit")
code("""print(open(ROOT/'data/manifests/raw_manifest.csv').read())
q = json.load(open(ROOT/'reports/data_quality.json'))
print({k:v for k,v in q.items() if k not in ('first_date_per_ticker','trackA_columns','tickers')})
raw = pd.read_csv(ROOT/'data/raw/ohlcv_yahoo_adjusted_28tickers.csv'); display(raw.head()); print(raw.shape)""")
md("## 2. Final processed model input (what the RL agents consume)")
code("""P = pd.read_csv(ROOT/'data/processed/model_input_features_panel.csv'); display(P[P.tic=='AAPL'].iloc[100:105].T)
display(P.groupby('split').date.agg(['min','max','nunique']))""")
md("## 3. Digital-twin fidelity (train-only regime-switching block bootstrap, 8 paths)")
code("show('table_twin_fidelity')")
md("## 4. Main results — test 2020-05 → 2023-05, common costs (Track B), mean ± sd over 3 seeds")
code("show('table_main_test_cost1x')")
code("display(Image(str(F/'fig2_equity_drawdown.png'))); display(Image(str(F/'fig3_risk_return.png')))")
md("### Validation period (2019-01 → 2020-04), used for checkpoint selection only")
code("show('table_main_val_cost1x')")
md("## 5. Cost stress (1×, 2×, 3× all frictions) — models trained at 1×")
code("show('table_cost_stress'); display(Image(str(F/'fig4_cost_stress.png')))")
md("## 6. Ablations")
code("show('table_ablation'); display(Image(str(F/'fig5_ablation.png')))")
md("## 7. Statistical tests (test period; paired stationary bootstrap, Jobson–Korkie–Memmel, deflated Sharpe)")
code("show('table_significance')")
md("## 8. Digital-twin crash stress (250-day synthetic crash paths, drift −0.15%/day, vol ×1.3)")
code("show('table_twin_crash'); display(Image(str(F/'fig7_twin_crash.png')))")
md("## 9. Training diagnostics (validation Sharpe, CMDP dual multipliers) and exposure")
code("display(Image(str(F/'fig6_training_dual.png'))); display(Image(str(F/'fig9_exposure.png')))")
md("## 10. Track A — the authors' own simulator and A2C/PPO code, on our data (NOT comparable to Track B numbers)")
code("show('table_trackA_runs'); display(Image(str(F/'fig8_trackA.png')))")
md("""## 11. Conclusions (from the measured numbers above)
* MARS-RL did **not** beat the equal-weight baseline or the authors' agents on risk-adjusted return in this replication; the Sharpe gap vs equal-weight is statistically significant against MARS-RL.
* It **did** lower drawdown/CVaR relative to leveraged long-only agents and in the synthetic crash stress; it has zero margin calls.
* The gate beats uniform mixing but is statistically indistinguishable from shuffled memory → no evidence that retrieval itself adds value.
* RL agents are highly cost-sensitive (turnover 0.3–0.8/day); at 2×–3× costs they turn clearly negative.
* The LLM/agentic layer is code-only and was not evaluated here (no news data / model weights in the sandbox).""")
md("## 12. Optional: regenerate everything")
code("""RETRAIN = False
if RETRAIN:
    import subprocess
    for c in ['python -m src.data_prep','python -m src.export_processed','bash scripts/run_stage.sh 150000 models/stage2 "0 1 2" "finrl_long mt_long mt_split mars_long mars_ls mars_neutral plain_ls mars_ls_notwin"',
              'python -m src.evaluate','python -m src.stress_twin','bash scripts/run_track_a.sh','python -m src.report']:
        print('>>', c); subprocess.run(c, shell=True, cwd=ROOT, check=True)
else:
    print('Retraining skipped (RETRAIN=False).')
print('Unit tests:'); import subprocess; print(subprocess.run('python -m pytest tests -q', shell=True, cwd=ROOT, capture_output=True, text=True).stdout[-300:])""")
nb.cells = C; nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/MARS_RL_results.ipynb")
