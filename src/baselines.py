"""Rule-based baselines. They run inside the SAME margin environment (same costs, financing, margin rules)
as the RL agents, so comparisons are apples-to-apples.

Every policy is a function (obs, env) -> ('w', weights | None). weights=None means 'hold current shares'.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


def policy_buy_hold_ew():
    st = {"first": True}

    def f(obs, env):
        if st["first"]:
            st["first"] = False
            return "w", np.ones(env.N) / env.N
        return "w", None

    return f


def policy_ew_rebalanced(leverage: float = 1.0):
    """Equal weight, rebalanced daily to `leverage` x equity (leverage=2 -> naive 2x leveraged EW, no RL)."""
    return lambda obs, env: ("w", np.ones(env.N) / env.N * leverage)


def _logret(env, t, win):
    c = env.p.close[max(0, t - win): t + 1]
    return np.diff(np.log(c), axis=0)


def policy_min_variance(rebal: int = 21, win: int = 60):
    st = {"k": 0, "w": None}

    def f(obs, env):
        if st["w"] is None or st["k"] % rebal == 0:
            R = _logret(env, env.t, win)
            n = R.shape[1]
            cov = LedoitWolf().fit(R).covariance_
            res = minimize(lambda w: w @ cov @ w, np.ones(n) / n, jac=lambda w: 2 * cov @ w,
                           bounds=[(0, 1)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                           method="SLSQP", options={"maxiter": 200})
            st["w"] = res.x if res.success else np.ones(n) / n
            st["k"] += 1
            return "w", st["w"].copy()
        st["k"] += 1
        return "w", None

    return f


def policy_momentum(top: int = 7, rebal: int = 21, win: int = 60, leverage: float = 1.0):
    st = {"k": 0}

    def f(obs, env):
        if st["k"] % rebal == 0:
            t = env.t
            mom = env.p.close[t] / env.p.close[max(0, t - win)] - 1
            w = np.zeros(env.N)
            w[np.argsort(-mom)[:top]] = leverage / top
            st["k"] += 1
            return "w", w
        st["k"] += 1
        return "w", None

    return f


BASELINES = {
    "EW buy&hold (1x)": policy_buy_hold_ew,
    "EW daily-rebal (1x)": lambda: policy_ew_rebalanced(1.0),
    "EW daily-rebal (2x lev)": lambda: policy_ew_rebalanced(2.0),
    "Min-variance (monthly)": policy_min_variance,
    "Momentum top-7 (monthly)": policy_momentum,
}
