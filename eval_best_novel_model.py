import numpy as np
import pandas as pd
from src.data import load_panel
from src.config import CostConfig, EnvConfig
from src.margin_env import MarginTradingEnv, run_episode
from src.evaluate import slice_metrics

print("Loading Top 30 Liquid Panel...")
p = load_panel("data/top30_liquid.csv")

def make_best_mars_alpha_policy(target_leverage=2.0, mom_lookback=120, rev_lookback=5, guard_budget=0.055, top_frac=0.30):
    def policy(obs, env):
        t = env.t
        if t < mom_lookback + 5:
            return "w", np.ones(env.N) / env.N
        
        c = env.p.close
        # Intermediate momentum minus short reversal
        mom = (c[t - rev_lookback] / c[t - mom_lookback] - 1.0)
        lr20 = np.diff(np.log(c[t - 20:t + 1]), axis=0)
        vol20 = np.std(lr20, axis=0) + 1e-6
        score = mom / vol20
        
        ranks = np.argsort(np.argsort(score))
        percentile = ranks / (len(ranks) - 1.0)
        
        threshold = 1.0 - top_frac
        w_raw = np.maximum(percentile - threshold, 0.0)
        if w_raw.sum() > 0:
            w_raw = w_raw / w_raw.sum()
        else:
            w_raw = np.ones(env.N) / env.N
            
        m_vol = env.p.mkt_vol[t]
        m_breadth = env.p.mkt[t, 7]
        m_ret20 = env.p.mkt[t, 2]
        
        # Smooth macro vol and trend scaling
        vol_scale = np.clip(0.18 / max(m_vol, 0.10), 0.6, 1.25)
        trend_scale = 1.15 if m_breadth > 0 and m_ret20 > 0 else 0.70
        lev = float(np.clip(target_leverage * vol_scale * trend_scale, 0.5, 2.0))
        
        w_target = w_raw * lev
        
        # Conformal Exposure Guard
        var = float(w_target @ env.p.cov[t] @ w_target)
        bound = env.p.conf_q[t] * np.sqrt(max(var, 1e-12))
        scale = min(1.0, guard_budget / bound) if bound > 0 else 1.0
        w_final = w_target * scale
        
        return "w", w_final
    return policy

c_paper = CostConfig(slippage=0.0, borrow_fee_annual=0.0, loan_rate_annual=0.0)
c_real = CostConfig()
cfg = EnvConfig(mode="long", lmax=2.0, reward="mars", risk_guard="none")

pol = make_best_mars_alpha_policy()

results = []
for cname, cost in [("Paper-style (0.1% fee)", c_paper), ("Realistic (Full 1x)", c_real)]:
    for split in ["train", "val", "test"]:
        rng = p.split_range(split)
        env = MarginTradingEnv(p, cost, cfg, rng)
        df = run_episode(env, pol)
        m = slice_metrics(df, p, split)
        acc_ret = (df["equity"].iloc[-1] / df["equity"].iloc[0] - 1.0) * 100.0
        daily_ret = df["equity"].pct_change().dropna()
        downside = daily_ret[daily_ret < 0]
        sortino = float(np.mean(daily_ret) / (np.std(downside) + 1e-9) * np.sqrt(252)) if len(downside) > 0 else 0.0
        results.append({
            "costs": cname,
            "split": split,
            "acc_return_pct": round(acc_ret, 2),
            "ann_return_pct": round(m["ann_return"] * 100, 2),
            "ann_vol_pct": round(m["ann_vol"] * 100, 2),
            "sharpe": round(m["sharpe"], 3),
            "sortino": round(sortino, 3),
            "max_dd_pct": round(m["max_drawdown"] * 100, 2),
            "calmar": round(m["calmar"], 3),
            "margin_calls": m["margin_calls"],
            "avg_gross": round(m["avg_gross"], 3)
        })

df_res = pd.DataFrame(results)
print(df_res.to_string())
df_res.to_csv("results/best_mars_alpha_model_results.csv", index=False)
