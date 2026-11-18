"""Builds every table and figure of the paper from the saved CSV/JSON results only (no hard-coded numbers)."""
from __future__ import annotations
import json, sys, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.metrics import bootstrap_sharpe_diff, jobson_korkie_memmel, deflated_sharpe, daily_returns
from src.train import load_all
from src.twin import DigitalTwin, fidelity_table

RES, TAB, FIG = ROOT / "results/trackB", ROOT / "results/tables", ROOT / "results/figures"
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})
KEY = ["EW buy&hold (1x)", "EW daily-rebal (1x)", "EW daily-rebal (2x lev)", "Min-variance (monthly)", "Momentum top-7 (monthly)",
       "finrl_long", "mt_long", "mt_split", "mars_long", "mars_ls", "mars_neutral", "MARS-gate", "MARS-uniform", "MARS-shuffled"]
NICE = {"finrl_long": "PPO long-only 1x (plain reward)", "mt_long": "PPO long-only 2x (paper-style reward)",
        "mt_split": "PPO 50/50 long-short 2x (paper-style)", "mars_long": "MARS expert: long", "mars_ls": "MARS expert: long-short",
        "mars_neutral": "MARS expert: neutral", "MARS-gate": "MARS-RL (retrieval gate)", "MARS-uniform": "MARS uniform mix",
        "MARS-shuffled": "MARS gate, shuffled memory", "plain_ls": "Ablation: L/S, no CMDP/guard/twin",
        "mars_ls_notwin": "Ablation: MARS L/S, no twin"}
nice = lambda x: NICE.get(x, x)
M = pd.read_csv(RES / "metrics_main.csv"); TW = pd.read_csv(RES / "metrics_twin_crash.csv")


def ms(df, col, f="%.2f"):
    g = df.groupby("model")[col]
    return (g.mean().map(lambda v: f % v) + " ± " + g.std().fillna(0).map(lambda v: f % v))


def md(df, path):
    df.to_csv(path.with_suffix(".csv")); path.with_suffix(".md").write_text(df.to_markdown())


def main_table(split, cost=1.0, name=None, models=KEY):
    d = M[(M.split == split) & (M.cost_mult == cost) & M.model.isin(models)]
    t = pd.DataFrame({"Ann. return": ms(d, "ann_return", "%.3f"), "Ann. vol": ms(d, "ann_vol", "%.3f"), "Sharpe": ms(d, "sharpe"),
                      "Sortino": ms(d, "sortino"), "Max DD": ms(d, "max_drawdown", "%.3f"), "Calmar": ms(d, "calmar"),
                      "CVaR5 (daily)": ms(d, "cvar5_daily", "%.4f"), "Avg gross": ms(d, "avg_gross"),
                      "Avg turnover": ms(d, "avg_turnover", "%.3f"), "Margin calls": ms(d, "margin_calls", "%.1f"),
                      "Costs (% init)": ms(d, "total_cost_pct_init", "%.3f")}).reindex(models)
    t.index = [nice(i) for i in t.index]
    md(t, TAB / (name or f"table_main_{split}_cost{cost:g}x")); return t


def curves(name, seed, cost=1.0):
    return pd.read_csv(RES / "curves" / f"cost{cost:g}x__{name}__s{seed}.csv", parse_dates=["date"])


def test_slice(df):
    d = df[df.date >= "2020-05-01"].reset_index(drop=True); return d


def main():
    t1 = main_table("test"); main_table("val")
    # ---- cost stress table
    rows = []
    for k in sorted(M.cost_mult.unique()):
        d = M[(M.split == "test") & (M.cost_mult == k) & M.model.isin(KEY)]
        for m, g in d.groupby("model"): rows.append(dict(cost=f"{k:g}x", model=nice(m), sharpe=g.sharpe.mean(), ann_return=g.ann_return.mean(), max_dd=g.max_drawdown.mean()))
    cs = pd.DataFrame(rows).pivot(index="model", columns="cost", values=["sharpe", "ann_return"]).round(3); md(cs, TAB / "table_cost_stress")
    # ---- ablation table (test, 1x)
    abl = ["plain_ls", "mars_ls_notwin", "mars_ls", "fixed-mars_ls", "mars_long", "mars_neutral", "MARS-uniform", "MARS-shuffled", "MARS-gate"]
    d = M[(M.split == "test") & (M.cost_mult == 1) & M.model.isin(abl)]
    ta = pd.DataFrame({"Sharpe": ms(d, "sharpe"), "Ann. return": ms(d, "ann_return", "%.3f"), "Max DD": ms(d, "max_drawdown", "%.3f"),
                       "Avg gross": ms(d, "avg_gross")}).reindex(abl); ta.index = [nice(i) for i in ta.index]; md(ta, TAB / "table_ablation")
    # ---- twin crash table
    g = TW[TW.model.isin(KEY)].groupby("model")
    tc = pd.DataFrame({"Acc. return": g.acc_return.mean(), "Max DD": g.max_drawdown.mean(), "Worst-path MDD": g.max_drawdown.min(),
                       "CVaR5 daily": g.cvar5_daily.mean()}).reindex(KEY).round(3); tc.index = [nice(i) for i in tc.index]; md(tc, TAB / "table_twin_crash")
    # ---- significance (test, 1x cost)
    ref = ["EW daily-rebal (1x)", "finrl_long", "mt_long"]; rows = []
    allS = M[(M.split == "test") & (M.cost_mult == 1)].groupby(["model", "seed"]).sharpe.first()
    for s in range(3):
        a = daily_returns(test_slice(curves("MARS-gate", s)))
        for r in ref:
            rs = 0 if r.startswith("EW") else s
            b = daily_returns(test_slice(curves(r, rs)))
            bs, jk = bootstrap_sharpe_diff(a, b), jobson_korkie_memmel(a, b)
            rows.append(dict(seed=s, vs=nice(r), sharpe_diff=bs["diff"], ci_low=bs["ci_low"], ci_high=bs["ci_high"], p_boot_one_sided=bs["p_one_sided"], p_JKM=jk["p_two_sided"]))
        trials = M[(M.split == "val") & (M.cost_mult == 1)].sharpe.values / np.sqrt(252)
        dsr = deflated_sharpe(a, trials); rows.append(dict(seed=s, vs="DSR (P[true SR>0] after selection among val trials)", sharpe_diff=dsr["sr_daily"] * np.sqrt(252), ci_low=np.nan, ci_high=np.nan, p_boot_one_sided=1 - dsr["dsr"], p_JKM=np.nan))
    md(pd.DataFrame(rows).round(4).set_index(["seed", "vs"]), TAB / "table_significance")
    # ---- Track A table
    ra = []
    for f in sorted(glob.glob(str(ROOT / "results/trackA/*_metrics.json"))):
        j = json.load(open(f)); n = Path(f).stem.replace("trackA_", "").replace("_metrics", "")
        ra.append(dict(run=n, **{f"test_{k}": v for k, v in j["test"].items()}))
    if ra:
        A = pd.DataFrame(ra).set_index("run"); md(A.round(3), TAB / "table_trackA_runs")
        A["algo"] = [i.split("_")[0] for i in A.index]
        md(A.groupby("algo").agg(["mean", "std"]).round(3), TAB / "table_trackA_summary")
    # ---- twin fidelity
    p, hi, lo = load_all(); tr = p.split_range("train"); tw = DigitalTwin(p, tr, hi, lo)
    fid = pd.concat([fidelity_table(p, tw.generate(2500, 100 + i), tr).loc["twin"] for i in range(8)], axis=1).mean(axis=1)
    ft = pd.DataFrame({"real_train": fidelity_table(p, tw.generate(300, 1), tr).loc["real_train"], "twin (mean of 8 paths)": fid}).round(4); md(ft, TAB / "table_twin_fidelity")

    # ================= FIGURES
    C = {"EW daily-rebal (1x)": "#444", "EW daily-rebal (2x lev)": "#999", "finrl_long": "#1f77b4", "mt_long": "#d62728", "MARS-gate": "#2ca02c", "mars_long": "#17becf", "mt_split": "#ff7f0e"}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for n, c in C.items():
        d = test_slice(curves(n, 0)); e = d.equity / d.equity.iloc[0]
        ax[0].plot(d.date, e, label=nice(n), color=c, lw=1.4); ax[1].plot(d.date, e / e.cummax() - 1, color=c, lw=1.2)
    ax[0].set_title("Test equity (rebased, seed 0, 1x costs)"); ax[1].set_title("Test drawdown"); ax[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(FIG / "fig2_equity_drawdown.png"); plt.close(fig)

    d = M[(M.split == "test") & (M.cost_mult == 1) & M.model.isin(KEY)].groupby("model")[["ann_vol", "ann_return", "max_drawdown"]].mean()
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for m, r in d.iterrows(): ax.scatter(r.ann_vol, r.ann_return, s=30); ax.annotate(nice(m)[:22], (r.ann_vol, r.ann_return), fontsize=5.5)
    ax.axhline(0, c="k", lw=.5); ax.set_xlabel("Ann. volatility"); ax.set_ylabel("Ann. return"); ax.set_title("Risk-return (test, 1x costs)"); fig.tight_layout(); fig.savefig(FIG / "fig3_risk_return.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for m in ["EW daily-rebal (1x)", "finrl_long", "mt_long", "mars_long", "MARS-gate"]:
        g = M[(M.split == "test") & (M.model == m)].groupby("cost_mult")[["sharpe", "ann_return"]].mean()
        ax[0].plot(g.index, g.sharpe, marker="o", label=nice(m)); ax[1].plot(g.index, g.ann_return, marker="o")
    ax[0].set_title("Sharpe vs cost multiplier"); ax[1].set_title("Ann. return vs cost multiplier"); ax[0].set_xlabel("cost x"); ax[1].set_xlabel("cost x"); ax[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(FIG / "fig4_cost_stress.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.4)); dd = M[(M.split == "test") & (M.cost_mult == 1) & M.model.isin(abl)]
    g = dd.groupby("model").sharpe; ax.barh([nice(i) for i in g.mean().reindex(abl).index], g.mean().reindex(abl), xerr=g.std().reindex(abl).fillna(0)); ax.set_title("Ablation: test Sharpe (mean ± sd over 3 seeds)"); fig.tight_layout(); fig.savefig(FIG / "fig5_ablation.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for f in sorted(glob.glob(str(ROOT / "models/stage2/mars_*_s0.json"))):
        j = json.load(open(f)); h = pd.DataFrame(j["val_history"]); ax[0].plot(h.step, h.sharpe, marker="o", label=j["role"])
        if j["dual_log"]:
            dl = pd.DataFrame(j["dual_log"]); ax[1].plot(dl.step, dl.mu_dd, label=j["role"] + " μ_dd")
    ax[0].set_title("Validation Sharpe during training (seed 0)"); ax[1].set_title("CMDP multiplier μ_dd (dual ascent)"); ax[0].legend(fontsize=6); ax[1].legend(fontsize=6); fig.tight_layout(); fig.savefig(FIG / "fig6_training_dual.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6)); k = [m for m in KEY if m in set(TW.model)]
    ax.boxplot([TW[TW.model == m].max_drawdown for m in k], tick_labels=[nice(m)[:18] for m in k], vert=False); ax.set_title("Twin crash stress: max drawdown over 8 synthetic paths"); fig.tight_layout(); fig.savefig(FIG / "fig7_twin_crash.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.4)); i = 0
    for f in sorted(glob.glob(str(ROOT / "results/trackA/*_test_equity.csv")))[:6]:
        d = pd.read_csv(f); ax.plot(d.equity.values / d.equity.values[0], lw=1, label=Path(f).stem.replace("trackA_", "").replace("_test_equity", ""))
    ax.set_title("Track A (authors' own simulator/agents): test equity, rebased"); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(FIG / "fig8_trackA.png"); plt.close(fig)

    d = test_slice(curves("MARS-gate", 0)); fig, ax = plt.subplots(2, 1, figsize=(8, 3.6), sharex=True)
    ax[0].plot(d.date, d.gross, lw=1); ax[0].set_ylabel("gross"); ax[1].plot(d.date, d.net, lw=1, c="C1"); ax[1].set_ylabel("net"); ax[0].set_title("MARS-RL exposure over the test period (seed 0)"); fig.tight_layout(); fig.savefig(FIG / "fig9_exposure.png"); plt.close(fig)
    print(t1.to_string()); print((TAB / "table_significance.md").read_text())


if __name__ == "__main__":
    main()
