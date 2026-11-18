"""Chart: base-paper Table I (typed from the PDF, TKDE 2026, Table I, PPO rows + DJI) vs our measured results."""
import sys; from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
# ---- base paper Table I (verbatim): ann return %, vol %, MDD %, Sharpe, Calmar
paper = {"Paper: DJI index": (12.9, 17.3, -21.9, 0.790, 0.587),
         "Paper: FinRL long PPO": (10.6, 17.0, -13.9, 0.680, 0.764),
         "Paper: Margin Trader L&S PPO": (19.2, 16.3, -18.2, 1.160, 1.057),
         "Paper: Margin Trader Long PPO": (30.9, 30.1, -25.8, 1.046, 1.199)}
M = pd.concat([pd.read_csv(ROOT / "results/trackB/metrics_main.csv"), pd.read_csv(ROOT / "results/trackB_v2/metrics_main.csv").query("model.str.startswith('v2')", engine="python")]); M = M[(M.split == "test") & (M.cost_mult == 1)]
g = M.groupby("model")[["ann_return", "ann_vol", "max_drawdown", "sharpe", "calmar"]].mean()
A = pd.read_csv(ROOT / "results/tables/table_trackA_runs.csv", index_col=0); A["algo"] = [i[:3] for i in A.index]
ga = A.groupby("algo")[["test_ann_return", "test_ann_vol", "test_max_drawdown", "test_sharpe", "test_calmar"]].mean()
row = lambda r: (100 * r.iloc[0], 100 * r.iloc[1], 100 * r.iloc[2], r.iloc[3], r.iloc[4])
ours = {"Ours TrackB: Equal-weight 1x": row(g.loc["EW daily-rebal (1x)"]), "Ours TrackB: PPO long 2x (paper-style)": row(g.loc["mt_long"]),
        "Ours TrackB: MARS-RL v1": row(g.loc["MARS-gate"]),
        "Ours TrackB: v2_b (val-selected)": row(g.loc["v2_b"]),
        "Ours TrackB: v2_c (NOT val-selected)": row(g.loc["v2_c"]),
        "Ours TrackA: authors' A2C": row(ga.loc["a2c"]), "Ours TrackA: authors' PPO": row(ga.loc["ppo"])}
T = pd.DataFrame({**paper, **ours}, index=["Annual return %", "Annual vol %", "Max drawdown %", "Sharpe", "Calmar"]).T
T.round(3).to_csv(ROOT / "results/tables/table_vs_base_paper.csv"); open(ROOT / "results/tables/table_vs_base_paper.md", "w").write(T.round(3).to_markdown())
col = ["#9aa0a6"] * 4 + ["#444", "#d62728", "#2ca02c", "#ff7f0e", "#bcbd22", "#1f77b4", "#17becf"]
fig, ax = plt.subplots(1, 4, figsize=(17, 5.2), sharey=True)
for a, m in zip(ax, ["Annual return %", "Max drawdown %", "Sharpe", "Calmar"]):
    a.barh(T.index, T[m], color=col); a.set_title(m); a.axvline(0, c="k", lw=.6)
    for i, v in enumerate(T[m]): a.text(v, i, f" {v:.2f}" if abs(v) < 5 else f" {v:.1f}", va="center", fontsize=7)
ax[0].invert_yaxis(); fig.suptitle("Base paper (grey) vs our results (test period)\nPaper: DJIA stocks 2020/05–2023/05, no slippage, authors' own agents. Ours: same stock universe, same split, Track B adds slippage+financing", fontsize=9)
fig.tight_layout(); fig.savefig(ROOT / "results/figures/fig1_vs_base_paper.png", dpi=130)
print(T.round(3).to_string())
