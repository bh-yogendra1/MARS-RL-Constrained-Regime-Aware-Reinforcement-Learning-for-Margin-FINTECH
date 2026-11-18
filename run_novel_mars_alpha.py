import numpy as np
import pandas as pd
from src.data import load_panel
from src.config import CostConfig, EnvConfig
from src.margin_env import MarginTradingEnv, run_episode
from src.evaluate import slice_metrics

print("Loading Top 30 Liquid Panel...")
p = load_panel("data/top30_liquid.csv")

def make_optimized_mars_alpha_policy(
    target_leverage=1.95,
    mom_fast=60,
    mom_slow=250,
    rev_win=5,
    top_k=8,
    guard_budget=0.048
):
    """
    Novel Architecture: Multi-Horizon Cross-Sectional Alpha + 
    Macro-Regime Adaptive Leverage + Conformal Loss-Bounded Guard.
    """
    def policy(obs, env):
        t = env.t
        if t < mom_slow + 5:
            return "w", np.ones(env.N) / env.N
        
        c = env.p.close
        # Multi-horizon relative momentum
        ret_fast = c[t - rev_win] / c[t - mom_fast] - 1.0
        ret_slow = c[t - rev_win] / c[t - mom_slow] - 1.0
        
        # Realized downside volatility over 60 days
        lr60 = np.diff(np.log(c[t - 60:t + 1]), axis=0)
        down_lr = np.where(lr60 < 0, lr60, 0.0)
        down_vol = np.sqrt(np.mean(down_lr**2, axis=0)) * np.sqrt(252) + 1e-4
        
        # Proximity to 52-week high (George & Hwang effect)
        high250 = np.max(c[t - 250:t + 1], axis=0)
        dist_high = c[t] / high250
        
        # Composite Alpha Score: (0.5 * fast + 0.5 * slow) * dist_high / down_vol
        composite_alpha = (0.4 * ret_fast + 0.6 * ret_slow) * dist_high / down_vol
        
        # Select top_k assets
        top_indices = np.argsort(composite_alpha)[-top_k:]
        w = np.zeros(env.N)
        # Softmax / exponential weighting among top winners
        sub_scores = composite_alpha[top_indices]
        sub_scores = np.clip(sub_scores - np.mean(sub_scores), -2, 2)
        exp_w = np.exp(1.5 * sub_scores)
        w[top_indices] = exp_w / exp_w.sum()
        
        # Macro Regime Adaptive Leverage
        m_vol = env.p.mkt_vol[t]      # 20d annualized market vol
        m_breadth = env.p.mkt[t, 7]  # Market breadth (-1 to 1)
        m_ret60 = env.p.mkt[t, 3]    # 60d market return
        
        # Volatility targeting leverage:
        # In calm bull regimes (m_vol <= 0.16 and m_ret60 >= 0): leverage = target_leverage (up to 1.95x)
        # In elevated vol (0.16 < m_vol <= 0.28): leverage = 1.3x - 1.6x
        # In panic / bear (m_vol > 0.28 or m_ret60 < -0.08): leverage = 0.6x - 0.9x
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
        
        # Conformal Exposure Guard (Step 7)
        var = float(w_target @ env.p.cov[t] @ w_target)
        bound = env.p.conf_q[t] * np.sqrt(max(var, 1e-12))
        scale = min(1.0, budget / bound) if bound > 0 else 1.0
        w_final = w_target * scale
        
        return "w", w_final
    return policy

c_paper = CostConfig(slippage=0.0, borrow_fee_annual=0.0, loan_rate_annual=0.0)
c_real = CostConfig()

cfg = EnvConfig(mode="long", lmax=2.0, reward="mars", risk_guard="none")

print("\nRunning Parameter Grid Search across Validation Split to lock best hyperparams...")
val_rng = p.split_range("val")
best_val_sharpe, best_params = -1e9, None

for lev in [1.85, 1.95]:
    for k in [6, 8, 10]:
        for b in [0.045, 0.050]:
            pol = make_optimized_mars_alpha_policy(target_leverage=lev, top_k=k, guard_budget=b)
            env = MarginTradingEnv(p, c_paper, cfg, val_rng)
            df = run_episode(env, pol)
            m = slice_metrics(df, p, "val")
            if m["sharpe"] > best_val_sharpe:
                best_val_sharpe = m["sharpe"]
                best_params = (lev, k, b)
                print(f"New Best Val: Sharpe={m['sharpe']:.3f}, Ret={m['ann_return']*100:.2f}%, MDD={m['max_drawdown']*100:.2f}%, Params={best_params}")

print(f"\nLocked Hyperparameters from Validation: {best_params} with Val Sharpe = {best_val_sharpe:.3f}")

# Final evaluation with locked params
lev, k, b = best_params
final_pol = make_optimized_mars_alpha_policy(target_leverage=lev, top_k=k, guard_budget=b)

results = []
for cname, cost in [("Paper-style (0.1% fee)", c_paper), ("Realistic (Full 1x)", c_real)]:
    for split in ["train", "val", "test"]:
        rng = p.split_range(split)
        env = MarginTradingEnv(p, cost, cfg, rng)
        df = run_episode(env, final_pol)
        m = slice_metrics(df, p, split)
        acc_ret = (df["equity"].iloc[-1] / df["equity"].iloc[0] - 1.0) * 100.0
        daily_ret = df["equity"].pct_change().dropna()
        downside = daily_ret[daily_ret < 0]
        sortino = float(np.mean(daily_ret) / (np.std(downside) + 1e-9) * np.sqrt(252)) if len(downside) > 0 else 0.0
        results.append({
            "costs": cname,
            "split": split,
            "acc_ret": round(acc_ret, 2),
            "ann_ret": round(m["ann_return"] * 100, 2),
            "vol": round(m["ann_vol"] * 100, 2),
            "sharpe": round(m["sharpe"], 3),
            "sortino": round(sortino, 3),
            "mdd": round(m["max_drawdown"] * 100, 2),
            "calmar": round(m["calmar"], 3),
            "calls": m["margin_calls"],
            "avg_gross": round(m["avg_gross"], 3)
        })

res_df = pd.DataFrame(results)
print("\n=== FINAL RESULTS ACROSS ALL SPLITS ===")
print(res_df.to_string())
res_df.to_csv("results/novel_mars_alpha_all_splits.csv", index=False)
