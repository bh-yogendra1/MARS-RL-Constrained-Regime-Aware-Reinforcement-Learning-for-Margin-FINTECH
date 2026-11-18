"""Validation-only comparison of single seeds vs seed-ensembles (mean of deterministic actions)."""
import sys, numpy as np, collections; sys.path.insert(0, '.')
from stable_baselines3 import PPO
from src.config import ROLES, CostConfig
from src.train import load_all
from src.margin_env import MarginTradingEnv, run_episode
from src.metrics import perf_metrics

def ens_fn(models):
    def f(obs, env):
        return "a", np.mean([m.predict(obs, deterministic=True)[0] for m in models], axis=0)
    return f

if __name__ == "__main__":
    roles, d = sys.argv[2:], sys.argv[1]
    p, _, _ = load_all()
    for r in roles:
        ms = [PPO.load(f"{d}/{r}_s{s}.zip", device="cpu") for s in range(3)]
        env = lambda: MarginTradingEnv(p, CostConfig(), ROLES[r], p.split_range("val"))
        single = [perf_metrics(run_episode(env(), lambda o, e, m=m: ("a", m.predict(o, deterministic=True)[0]))) for m in ms]
        en = perf_metrics(run_episode(env(), ens_fn(ms)))
        print(r, "VAL single SR", [round(x["sharpe"], 2) for x in single], "mean %.2f" % np.mean([x["sharpe"] for x in single]),
              "| ENSEMBLE SR %.2f ret %.3f mdd %.3f turn %.3f" % (en["sharpe"], en["ann_return"], en["max_drawdown"], en["avg_turnover"]))
