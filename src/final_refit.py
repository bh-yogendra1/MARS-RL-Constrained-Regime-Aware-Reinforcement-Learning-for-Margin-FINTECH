"""Locked test of the train+val refit ensemble (5 seeds, last checkpoint, v2_b config chosen on validation)."""
import sys, numpy as np, pandas as pd; sys.path.insert(0, '.')
from stable_baselines3 import PPO
from src.config import ROLES, CostConfig
from src.train import load_all
from src.margin_env import MarginTradingEnv, run_episode
from src.ensemble_val import ens_fn
from src.evaluate import slice_metrics
from src.baselines import policy_ew_rebalanced
from src.gpu_utils import setup_safe_gpu

dev = setup_safe_gpu("auto", max_mem_fraction=0.15)
p, _, _ = load_all(); vt = p.split_range("valtest")
ms = [PPO.load(f"models/refit/v2_b_refit_s{s}.zip", device=dev) for s in range(5)]
costs = {"full 1x": CostConfig(), "full 2x": CostConfig().scaled(2), "full 3x": CostConfig().scaled(3),
         "paper-style (0.1% fee only)": CostConfig(slippage=0.0, borrow_fee_annual=0.0, loan_rate_annual=0.0)}
rows = []
for n, c in costs.items():
    df = run_episode(MarginTradingEnv(p, c, ROLES["v2_b"], vt), ens_fn(ms)); df.to_csv(f"results/trackB_v2/curves/refit_ens__{n.split()[0]}{n.split()[1] if n.startswith('full') else ''}.csv", index=False)
    rows.append(dict(model="REFIT-ENS v2_b", costs=n, **slice_metrics(df, p, "test")))
    from src.config import EnvConfig
    e = run_episode(MarginTradingEnv(p, c, EnvConfig(mode="ls", reward="plain", risk_guard="none"), vt), policy_ew_rebalanced(1.0))
    rows.append(dict(model="EW 1x", costs=n, **slice_metrics(e, p, "test")))
o = pd.DataFrame(rows); o.to_csv("results/trackB_v2/metrics_refit_ensemble.csv", index=False)
print(o[["model", "costs", "ann_return", "ann_vol", "sharpe", "max_drawdown", "calmar", "avg_gross", "margin_calls"]].round(3).to_string())
