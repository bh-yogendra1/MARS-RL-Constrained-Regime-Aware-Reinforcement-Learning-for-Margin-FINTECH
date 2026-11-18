"""End-to-End Master GPU Training, Validation, Testing and Benchmarking Suite.

Executes:
1. Multi-Horizon Cross-Sectional Alpha + Conformal Risk Guard Grid Search on Validation Split.
2. Full Dataset Train, Validation, and Out-of-Sample Test Evaluation across both Realistic and Paper Cost models.
3. Safe Shared-GPU PPO RL Training with Live Terminal Progress Bar and Val Checkpoint Selection.
4. Final Validation & Test Benchmark comparison.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.gpu_utils import setup_safe_gpu
from src.data import load_panel
from src.config import CostConfig, EnvConfig
from src.margin_env import MarginTradingEnv, run_episode
from src.evaluate import slice_metrics
from src.train import train

console = Console()

def run_full_suite():
    console.print(Panel.fit("[bold cyan]MARS-RL Full GPU End-to-End Training, Validation & Testing Suite[/bold cyan]\n[green]Safe Shared-GPU Allocation Active[/green]"))
    
    # 1. Setup Safe GPU
    dev = setup_safe_gpu("auto", max_mem_fraction=0.25)
    console.print(f"[bold yellow]✓ Assigned Safe Device:[/bold yellow] {dev}\n")
    
    # 2. Dataset Loading
    console.print("[bold blue]1. Loading Market Panel Data (Top 30 Liquid Dataset)...[/bold blue]")
    p = load_panel("data/top30_liquid.csv")
    console.print(f"   Dates range: {p.dates[0]} to {p.dates[-1]} ({len(p.dates)} trading days)")
    console.print(f"   Assets: {len(p.tickers)} liquid tickers | Split ranges: Train={p.split_range('train')}, Val={p.split_range('val')}, Test={p.split_range('test')}\n")
    
    # 3. Novel MARS Alpha Policy Search on Validation Split
    console.print("[bold blue]2. Running Novel MARS Alpha Multi-Horizon Grid Search on Validation Split...[/bold blue]")
    def make_mars_alpha_policy(target_leverage=1.95, top_k=8, guard_budget=0.045):
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
    val_rng = p.split_range("val")
    best_val_sharpe, best_params = -1e9, None

    for lev in [1.85, 1.95]:
        for k in [6, 8]:
            for b in [0.045, 0.050]:
                pol = make_mars_alpha_policy(target_leverage=lev, top_k=k, guard_budget=b)
                env = MarginTradingEnv(p, c_paper, cfg, val_rng)
                df = run_episode(env, pol)
                m = slice_metrics(df, p, "val")
                if m["sharpe"] > best_val_sharpe:
                    best_val_sharpe = m["sharpe"]
                    best_params = (lev, k, b)
                    console.print(f"   [green]New Best Validation Checkpoint:[/green] Sharpe={m['sharpe']:.3f} | Ret={m['ann_return']*100:.2f}% | MDD={m['max_drawdown']*100:.2f}% | Params: Lev={lev}, TopK={k}, Budget={b}")

    console.print(f"\n[bold green]✓ Locked Best Validation Hyperparameters:[/bold green] Lev={best_params[0]}, TopK={best_params[1]}, Budget={best_params[2]} (Val Sharpe: {best_val_sharpe:.3f})\n")

    # 4. Out-of-Sample Test & Multi-Split Evaluation
    console.print("[bold blue]3. Running Full Multi-Split Evaluation (Train, Val, and Out-of-Sample Test)...[/bold blue]")
    lev, k, b = best_params
    final_pol = make_mars_alpha_policy(target_leverage=lev, top_k=k, guard_budget=b)
    alpha_results = []
    
    for cname, cost in [("Paper-style (0.1%)", c_paper), ("Realistic (Full 1x)", c_real)]:
        for split in ["train", "val", "test"]:
            rng = p.split_range(split)
            env = MarginTradingEnv(p, cost, cfg, rng)
            df = run_episode(env, final_pol)
            m = slice_metrics(df, p, split)
            acc_ret = (df["equity"].iloc[-1] / df["equity"].iloc[0] - 1.0) * 100.0
            daily_ret = df["equity"].pct_change().dropna()
            downside = daily_ret[daily_ret < 0]
            sortino = float(np.mean(daily_ret) / (np.std(downside) + 1e-9) * np.sqrt(252)) if len(downside) > 0 else 0.0
            alpha_results.append({
                "Cost Regime": cname,
                "Split": split.upper(),
                "Acc Return (%)": round(acc_ret, 2),
                "Ann Return (%)": round(m["ann_return"] * 100, 2),
                "Ann Vol (%)": round(m["ann_vol"] * 100, 2),
                "Sharpe": round(m["sharpe"], 3),
                "Sortino": round(sortino, 3),
                "Max DD (%)": round(m["max_drawdown"] * 100, 2),
                "Margin Calls": m["margin_calls"]
            })

    table_alpha = Table(title="Novel MARS Alpha Performance Across All Dataset Splits")
    for col in alpha_results[0].keys():
        table_alpha.add_column(col, justify="center" if col in ["Split", "Margin Calls"] else "right")
    for r in alpha_results:
        table_alpha.add_row(*[str(v) for v in r.values()])
    console.print(table_alpha)
    console.print()

    # 5. Full Heavy GPU Reinforcement Learning Training
    console.print("[bold blue]4. Starting Full GPU Deep RL (PPO + Lagrangian Dual Ascent) Training...[/bold blue]")
    out_dir = ROOT / "models/gpu_master"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Train primary MARS constrained policy v2_b (50,000 steps on GPU)
    console.print("[bold yellow]Executing Heavy GPU Training: Role=v2_b, Steps=50,000, Chunk=10,000[/bold yellow]")
    best_rl = train(
        role="v2_b",
        seed=0,
        steps=50000,
        out=out_dir,
        chunk=10000,
        device="auto",
        max_mem_fraction=0.25
    )
    
    console.print(f"[bold green]✓ RL Training Complete![/bold green] Best Validation Sharpe: {best_rl[0]:.3f} at step {best_rl[1]}\n")
    
    # Summary of artifacts
    console.print(Panel.fit("[bold green]ALL TRAINING, VALIDATION & TESTING TASKS FINISHED SUCCESSFULLY ON GPU![/bold green]\nModels saved in: models/gpu_master/"))

if __name__ == "__main__":
    run_full_suite()
