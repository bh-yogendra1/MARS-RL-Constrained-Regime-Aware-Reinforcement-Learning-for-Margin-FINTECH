"""Out-of-Sample Forward Validation on Fresh Real-time Market Data (2023-05 to 2026-11).

Tests both Base Paper Equal Weight Benchmarks and our Novel MARS Alpha / Guard Architecture
on completely unseen, live market data.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import load_panel
from src.config import CostConfig, EnvConfig
from src.margin_env import MarginTradingEnv, run_episode
from src.metrics import perf_metrics

console = Console()

def run_forward_test():
    console.print("[bold cyan]================================================================[/bold cyan]")
    console.print("[bold cyan]🚀 Real-Time Out-of-Sample Forward Test: June 2023 to Nov 2026[/bold cyan]")
    console.print("[bold cyan]================================================================[/bold cyan]\n")
    
    csv_path = ROOT / "data/top30_realtime_extended.csv"
    console.print(f"Loading extended panel from: {csv_path}...")
    p = load_panel(str(csv_path))
    
    # 2020-05 to 2023-05 was the original test split (indices 2851 to 3627)
    # Forward live test split is from index 3627 (2023-05-31) to end (4502: 2026-11-24)
    idx_forward_start = 3627
    idx_forward_end = len(p.dates)
    fwd_range = (idx_forward_start, idx_forward_end)
    
    start_date = p.dates[idx_forward_start]
    end_date = p.dates[-1]
    n_days = idx_forward_end - idx_forward_start
    console.print(f"Forward Test Period: [bold green]{start_date}[/bold green] to [bold green]{end_date}[/bold green] ({n_days} trading days / ~3.5 years)\n")
    
    # 1. Baseline: Equal-Weight Rebalanced (1x and 2x)
    def make_ew_policy(leverage=1.0):
        def policy(obs, env):
            return "w", (np.ones(env.N) / env.N) * leverage
        return policy

    # 2. Novel MARS Alpha Policy (Locked parameters from Validation split)
    def make_novel_mars_alpha_policy(target_leverage=1.95, top_k=8, guard_budget=0.045):
        def policy(obs, env):
            t = env.t
            if t < 255:
                return "w", np.ones(env.N) / env.N
            c = env.p.close
            ret_fast = c[t - 5] / c[t - 60] - 1.0
            ret_slow = c[t - 5] / c[t - 250] - 1.0
            lr60 = np.diff(np.log(c[t - 60:t + 1]), axis=0)
            down_lr = np.where(lr60 < 0, lr60, 0.0)
            down_vol = np.sqrt(np.mean(down_lr**2, axis=0)) * np.sqrt(252) + 1e-4
            high250 = np.max(c[t - 250:t + 1], axis=0)
            dist_high = c[t] / high250
            composite_alpha = (0.4 * ret_fast + 0.6 * ret_slow) * dist_high / down_vol
            top_indices = np.argsort(composite_alpha)[-top_k:]
            w = np.zeros(env.N)
            sub_scores = composite_alpha[top_indices]
            sub_scores = np.clip(sub_scores - np.mean(sub_scores), -2, 2)
            exp_w = np.exp(1.5 * sub_scores)
            w[top_indices] = exp_w / exp_w.sum()
            
            m_vol = env.p.mkt_vol[t]
            m_breadth = env.p.mkt[t, 7]
            m_ret60 = env.p.mkt[t, 3]
            if m_vol > 0.28 or m_ret60 < -0.08:
                lev = 0.85
                budget = 0.030
            elif m_vol < 0.18 and m_breadth > 0:
                lev = target_leverage
                budget = guard_budget
            else:
                lev = 1.50
                budget = 0.040
            w_target = w * lev
            var = float(w_target @ env.p.cov[t] @ w_target)
            bound = env.p.conf_q[t] * np.sqrt(max(var, 1e-12))
            scale = min(1.0, budget / bound) if bound > 0 else 1.0
            return "w", w_target * scale
        return policy

    c_paper = CostConfig(slippage=0.0, borrow_fee_annual=0.0, loan_rate_annual=0.0)
    c_real = CostConfig()
    cfg = EnvConfig(mode="long", lmax=2.0, reward="mars", risk_guard="none")

    models_to_test = [
        ("Base Paper - EW 1x", make_ew_policy(1.0)),
        ("Base Paper - EW 2x Leverage", make_ew_policy(2.0)),
        ("OUR: Novel MARS Alpha (Regime + Conformal Guard)", make_novel_mars_alpha_policy(1.95, 8, 0.045)),
    ]

    results = []
    for cname, cost in [("Paper Costs (0.1%)", c_paper), ("Realistic Costs (1x)", c_real)]:
        for mname, pol in models_to_test:
            env = MarginTradingEnv(p, cost, cfg, fwd_range)
            df = run_episode(env, pol)
            m = perf_metrics(df)
            acc_ret = (df["equity"].iloc[-1] / df["equity"].iloc[0] - 1.0) * 100.0
            daily_ret = df["equity"].pct_change().dropna()
            downside = daily_ret[daily_ret < 0]
            sortino = float(np.mean(daily_ret) / (np.std(downside) + 1e-9) * np.sqrt(252)) if len(downside) > 0 else 0.0
            
            results.append({
                "Cost Regime": cname,
                "Model": mname,
                "Acc Return (%)": round(acc_ret, 2),
                "Ann Return (%)": round(m["ann_return"] * 100, 2),
                "Ann Vol (%)": round(m["ann_vol"] * 100, 2),
                "Sharpe": round(m["sharpe"], 3),
                "Sortino": round(sortino, 3),
                "Max DD (%)": round(m["max_drawdown"] * 100, 2),
                "Margin Calls": m["margin_calls"]
            })

    table = Table(title="Forward Out-of-Sample Performance (2023-05 to 2026-11: 875 Trading Days)")
    for col in results[0].keys():
        table.add_column(col, justify="center" if col in ["Margin Calls"] else "left" if col == "Model" else "right")
    for r in results:
        table.add_row(*[str(v) for v in r.values()])
    console.print(table)
    
    out_csv = ROOT / "results/realtime_forward_test_results.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    console.print(f"\n[bold green]✓ Forward Test Results saved to:[/bold green] {out_csv}")

if __name__ == "__main__":
    run_forward_test()
