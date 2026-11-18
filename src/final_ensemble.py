"""Locked test of the validation-selected seed-ensemble (mean deterministic action over seeds 0-2)."""
import sys, numpy as np, pandas as pd; sys.path.insert(0, '.')
from stable_baselines3 import PPO
from src.config import ROLES, CostConfig
from src.train import load_all
from src.margin_env import MarginTradingEnv, run_episode
from src.metrics import perf_metrics
from src.ensemble_val import ens_fn
from src.evaluate import slice_metrics
p, _, _ = load_all(); vt = p.split_range("valtest"); rows = []
for role, d in [("v2_b", "models/v2"), ("v4_a", "models/v4")]:
    ms = [PPO.load(f"{d}/{role}_s{s}.zip", device="cpu") for s in range(3)]
    for k in (1.0, 2.0, 3.0):
        df = run_episode(MarginTradingEnv(p, CostConfig().scaled(k), ROLES[role], vt), ens_fn(ms))
        df.to_csv(f"results/trackB_v2/curves/cost{k:g}x__ENS-{role}__s0.csv", index=False)
        for sp in ("val", "test"):
            rows.append(dict(model=f"ENS-{role}", cost_mult=k, split=sp, **slice_metrics(df, p, sp)))
out = pd.DataFrame(rows); out.to_csv("results/trackB_v2/metrics_ensemble.csv", index=False)
print(out[out.split == "test"][["model", "cost_mult", "ann_return", "ann_vol", "sharpe", "max_drawdown", "calmar", "avg_gross", "avg_turnover", "margin_calls"]].round(3).to_string())
