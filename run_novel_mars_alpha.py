import numpy as np
import pandas as pd
from pathlib import Path
from src.data import load_panel
from src.config import CostConfig, EnvConfig
from src.margin_env import MarginTradingEnv, run_episode
from src.metrics import evaluate_metrics

print("Loading Top 30 Liquid Panel...")
p = load_panel("data/top30_liquid.csv")

def make_apex_mars_alpha_policy(
    target_leverage=1.95,
    top_k=6,
    guard_budget=0.060,
    hold_days=14,
    core_weight=0.45,
    crash_lev=0.75,
    crash_trig=0.28
):
    """
    Apex MARS-RL Architecture with User Improvements:
    1. Cross-Sectional Z-Score Normalized Alpha (Invariant across market regimes)
    2. Core-Satellite Portfolio Decomposition (45% Bedrock Beta + 55% Alpha Satellite)
    3. Low-Churn Turnover Throttle (14-day Holding Period)
    4. Two-Tier Regime-Aware Leverage (0.75x Crash, 1.50x Normal, 1.95x Bull)
    5. Conformal Tail-Risk Guard with real-time covariance calibration
    """
    last_w = [None]
    last_t = [-1]

    def policy(obs, env):
        t = env.t
        if t < 65:
            return "w", np.ones(env.N) / env.N
        
        # Low-churn band holding
        if last_w[0] is not None and (t - last_t[0] < hold_days):
            return "w", last_w[0]
        
        c = env.p.close
        m_vol = env.p.mkt_vol[t]      # 20d annualized market vol
        m_breadth = env.p.mkt[t, 7]  # Market breadth (-1 to 1)
        m_ret60 = env.p.mkt[t, 3]    # 60d market return
        
        # 1. 60-day relative return + cross-sectional z-score normalization
        ret60 = c[t - 3] / c[t - 60] - 1.0
        ret60_z = (ret60 - np.mean(ret60)) / (np.std(ret60) + 1e-6)
        
        # 2. Downside Volatility (Sortino denominator)
        lr20 = np.diff(np.log(c[t - 20:t + 1]), axis=0)
        down_lr = np.where(lr20 < 0, lr20, 0.0)
        down_vol = np.sqrt(np.mean(down_lr**2, axis=0)) * np.sqrt(252) + 1e-4
        
        # 3. Composite Asymmetric Score
        alpha_score = (ret60 + 0.1 * ret60_z) / down_vol
        top_idx = np.argsort(alpha_score)[-top_k:]
        w_sat = np.zeros(env.N)
        w_sat[top_idx] = 1.0 / top_k
        
        # 4. Core Bedrock (Equal-Weight diversification)
        w_core = np.ones(env.N) / env.N
        w_base = core_weight * w_core + (1.0 - core_weight) * w_sat
        
        # 5. Two-Tier Regime-Conditioned Asymmetric Leverage
        if m_vol > crash_trig or m_ret60 < -0.08:
            lev = crash_lev
            budget = 0.030
        elif m_vol < 0.17 and (m_ret60 > 0 or m_breadth > 0):
            lev = target_leverage
            budget = guard_budget
        else:
            lev = 1.50
            budget = 0.045
            
        w_target = w_base * lev
        
        # 6. Conformal Tail-Risk Guard
        var = float(w_target @ env.p.cov[t] @ w_target)
        bound = env.p.conf_q[t] * np.sqrt(max(var, 1e-12))
        scale = min(1.0, budget / bound) if bound > 0 else 1.0
        w_final = w_target * scale
        
        last_w[0] = w_final
        last_t[0] = t
        return "w", w_final
        
    return policy

c_paper = CostConfig(slippage=0.0, borrow_fee_annual=0.0, loan_rate_annual=0.0)
c_real = CostConfig()
cfg = EnvConfig(mode="long", lmax=2.0, reward="mars", risk_guard="none")

print("\nRunning Parameter Grid Search across Validation Split to lock best hyperparams...")
val_rng = p.split_range("val")
param_grid = [
    (1.95, 6, 0.060, 14, 0.45, 0.75, 0.28),
    (1.85, 6, 0.055, 14, 0.45, 0.75, 0.28),
    (1.95, 8, 0.060, 14, 0.50, 0.75, 0.28),
    (1.85, 6, 0.050, 10, 0.50, 0.75, 0.26),
]

val_results = []
for p_tuple in param_grid:
    lev, k, b, hd, cw, cr_l, cr_t = p_tuple
    pol = make_apex_mars_alpha_policy(target_leverage=lev, top_k=k, guard_budget=b, hold_days=hd, core_weight=cw, crash_lev=cr_l, crash_trig=cr_t)
    env = MarginTradingEnv(p, c_paper, cfg, val_rng)
    df = run_episode(env, pol)
    m = evaluate_metrics(df)
    print(f"Config (Lev={lev}, k={k}, b={b}, hd={hd}, cw={cw}) -> Val Sharpe: {m['sharpe']:.3f}, Ret: {m['ann_return']*100:.1f}%, MDD: {m['max_drawdown']*100:.1f}%")
    val_results.append((m["sharpe"], p_tuple))

val_results.sort(key=lambda x: x[0], reverse=True)
best_val_sharpe, best_params = val_results[0]
print(f"\nLocked Best Hyperparameters from Validation: {best_params} with Val Sharpe = {best_val_sharpe:.3f}")

lev, k, b, hd, cw, cr_l, cr_t = best_params
final_pol = make_apex_mars_alpha_policy(target_leverage=lev, top_k=k, guard_budget=b, hold_days=hd, core_weight=cw, crash_lev=cr_l, crash_trig=cr_t)

results = []
for cname, cost in [("Paper-style (0.1% fee)", c_paper), ("Realistic (Full 1x)", c_real)]:
    for split in ["train", "val", "test"]:
        rng = p.split_range(split)
        env = MarginTradingEnv(p, cost, cfg, rng)
        # Create fresh policy instance per split
        pol_inst = make_apex_mars_alpha_policy(target_leverage=lev, top_k=k, guard_budget=b, hold_days=hd, core_weight=cw, crash_lev=cr_l, crash_trig=cr_t)
        df = run_episode(env, pol_inst)
        m = evaluate_metrics(df)
        results.append({
            "costs": cname,
            "split": split,
            "acc_ret": round(m["acc_return"] * 100, 2),
            "ann_ret": round(m["ann_return"] * 100, 2),
            "vol": round(m["ann_vol"] * 100, 2),
            "sharpe": round(m["sharpe"], 3),
            "sortino": round(m["sortino"], 3),
            "mdd": round(m["max_drawdown"] * 100, 2),
            "calmar": round(m["calmar"], 3),
            "calls": m["margin_calls"],
            "avg_gross": round(m["avg_gross"], 3),
            "verdict": m["verdict"]
        })

res_df = pd.DataFrame(results)
print("\n=== FINAL RESULTS ACROSS ALL SPLITS (UNIFIED EVALUATE_METRICS) ===")
print(res_df.to_string())
Path("results").mkdir(exist_ok=True)
res_df.to_csv("results/novel_mars_alpha_all_splits.csv", index=False)
