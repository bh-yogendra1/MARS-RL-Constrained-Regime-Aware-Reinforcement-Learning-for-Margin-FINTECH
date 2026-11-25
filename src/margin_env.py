"""Common realistic margin-trading simulator (Track B) with a full execution ledger.

Timeline (no look-ahead):
    decision at close of day t  ->  order executed at OPEN of day t+1  ->  marked to market at CLOSE of t+1.

Single margin account with signed shares (shares < 0 are shorts):
    equity   = cash + sum(shares * price)
    gross    = sum(|shares| * price)
    margin level ML = equity / gross
Frictions (identical for every strategy, see config.CostConfig):
    * proportional commission + slippage on every traded value
    * loan interest on borrowed cash  = max(0, long_value - equity) * loan_rate / 252
    * borrow fee on short market value = short_value * borrow_fee / 252
Margin rules:
    * ML < warning (40%): risk-increasing trades are blocked (weights are scaled down to the current gross exposure)
    * ML < strict  (30%): margin call -> positions scaled down until ML = liquidation_target (50%), extra liquidation fee
NOTE: this is a simplified single-ledger account. It is NOT claimed to be numerically identical to the authors' separate
long/short ledgers (see Track A, which runs the authors' own simulator unchanged).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import CostConfig, EnvConfig
from .data import Panel


def weights_from_action(a: np.ndarray, mode: str, lmax: float, fixed_leverage: float | None = None, kappa: float = 1.5) -> np.ndarray:
    """Map a raw policy action in [-1,1]^(N+1) to signed portfolio weights (fractions of equity).

    The last component is the leverage request; sum(|w|) = leverage (gross exposure / equity).
    """
    a = np.asarray(a, dtype=np.float64)
    aw, aL = a[:-1], a[-1]
    L = fixed_leverage if fixed_leverage is not None else lmax * (np.clip(aL, -1, 1) + 1.0) / 2.0
    if mode == "resid":  # long-only tilt around equal weight: a_w=0 -> equal weight
        x = np.exp(kappa * np.clip(aw, -1, 1))
        return x / x.sum() * L
    if mode == "long":
        x = np.maximum(aw, 0.0)
        s = x.sum()
        return x / s * L if s > 1e-8 else np.zeros_like(aw)
    if mode == "ls":
        s = np.abs(aw).sum()
        return aw / s * L if s > 1e-8 else np.zeros_like(aw)
    if mode == "neutral":
        x = aw - aw.mean()
        s = np.abs(x).sum()
        return x / s * L if s > 1e-8 else np.zeros_like(aw)
    if mode == "split":  # half of the exposure long, half short (base paper's 50/50 long-short mode)
        pl, ps = np.maximum(aw, 0.0), np.maximum(-aw, 0.0)
        w = np.zeros_like(aw)
        if pl.sum() > 1e-8:
            w += pl / pl.sum() * (L / 2.0)
        if ps.sum() > 1e-8:
            w -= ps / ps.sum() * (L / 2.0)
        return w
    raise ValueError(mode)


class MarginTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, panel: Panel, cost: CostConfig, cfg: EnvConfig, split_range: tuple[int, int], seed: int = 0,
                 twin_panels: list[Panel] | None = None, twin_prob: float = 0.0, record_ledger: bool = False,
                 guard_override: str | None = None):
        super().__init__()
        if guard_override is not None:
            cfg = dataclasses.replace(cfg, risk_guard=guard_override)
        self.p0, self.p, self.cost, self.cfg = panel, panel, cost, cfg
        self.twins, self.twin_prob = twin_panels or [], twin_prob
        self.rng_split = split_range
        self.t_lo, self.t_hi = split_range
        self.N, self.F, self.M = panel.N, panel.feats.shape[2], panel.mkt.shape[1]
        obs_dim = self.N * self.F + self.M + 6 + self.N
        self.observation_space = spaces.Box(-5.0, 5.0, (obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.N + 1,), np.float32)
        self.rng = np.random.default_rng(seed)
        self.mu_margin, self.mu_dd = cfg.mu_margin, cfg.mu_dd
        self.record_ledger = record_ledger
        self.ledger: list[dict] = []
        self._init_state()

    # ------------------------------------------------------------------ helpers
    def set_mu(self, mu_margin: float, mu_dd: float) -> None:
        self.mu_margin, self.mu_dd = float(mu_margin), float(mu_dd)

    def get_mu(self):
        return self.mu_margin, self.mu_dd

    def _init_state(self) -> None:
        self.t = self.t_lo
        self.t_end = self.t_hi - 1
        self.cash = self.cost.initial_equity
        self.shares = np.zeros(self.N)
        self.eq = self.cost.initial_equity
        self.peak = self.eq
        self.last_L = 0.0
        self.prev_w = np.zeros(self.N)
        self.ret_hist: list[float] = []
        self.n_margin_calls = 0
        self.ledger = []

    def _margin_level(self, px: np.ndarray) -> float:
        mv = self.shares * px
        g = np.abs(mv).sum()
        eq = self.cash + mv.sum()
        return eq / g if g > 1e-9 else 10.0

    def current_weights(self) -> np.ndarray:
        px = self.p.close[self.t]
        eq = self.cash + (self.shares * px).sum()
        return self.shares * px / eq if eq > 0 else np.zeros(self.N)

    def _obs(self) -> np.ndarray:
        t, p = self.t, self.p
        mv = self.shares * p.close[t]
        g = np.abs(mv).sum()
        eq = self.cash + mv.sum()
        gross = g / eq if eq > 1e-9 else 5.0
        ml = min(eq / g, 1.0) if g > 1e-9 else 1.0
        net = mv.sum() / eq if eq > 1e-9 else 0.0
        dd = 1.0 - eq / self.peak
        acct = np.array([gross / 2, ml, net / 2, dd * 5, eq / self.cost.initial_equity - 1.0, self.last_L / 2],
                        dtype=np.float32)
        o = np.concatenate([p.feats[t].ravel(), p.mkt[t], acct, p.extra[t]])
        return np.clip(o, -5, 5).astype(np.float32)

    def apply_guard(self, w: np.ndarray, t: int) -> np.ndarray:
        """Execution-time exposure guard. Uses only information known at close t. NEVER increases gross exposure."""
        g = self.cfg.risk_guard
        if g == "none" or g is False or g is None:
            return w
        p = self.p
        gross = float(np.abs(w).sum())
        if gross <= 1e-12:
            return w
        if g == "vol" or g is True:
            gmax = self.cfg.vol_target / max(p.mkt_vol[t], 0.05)
            return w * (gmax / gross) if gross > gmax else w
        if g == "conformal":
            var = float(w @ p.cov[t] @ w)
            bound = p.conf_q[t] * np.sqrt(max(var, 1e-12))
            scale = min(1.0, self.cfg.conformal_budget / bound) if bound > 0 else 1.0
            return w * scale
        raise ValueError(g)

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        full = options.get("full", False)
        use_twin = (not full) and self.twins and self.rng.random() < self.twin_prob
        if use_twin:
            self.p = self.twins[int(self.rng.integers(len(self.twins)))]
            self.t_lo, self.t_hi = 60, self.p.T
        else:
            self.p = self.p0
            self.t_lo, self.t_hi = self.rng_split
        self._init_state()
        if not full:
            L = self.cfg.episode_len
            hi = self.t_hi - 1 - L
            self.t = int(self.rng.integers(self.t_lo, max(self.t_lo + 1, hi)))
            self.t_end = min(self.t + L, self.t_hi - 1)
        return self._obs(), {}

    def step(self, action):
        w = weights_from_action(action, self.cfg.mode, self.cfg.lmax, self.cfg.fixed_leverage, self.cfg.kappa)
        return self.step_weights(w)

    def step_weights(self, w: np.ndarray | None):
        """Advance one day. w=None means 'hold current shares' (used by buy-and-hold)."""
        p, c, cfg = self.p, self.cost, self.cfg
        t = self.t
        n = t + 1
        op, cl = p.open[n], p.close[n]
        eq_prev = self.eq
        eq_open = self.cash + (self.shares * op).sum()
        tv = 0.0
        fee = 0.0
        w_exec = None
        trade = np.zeros(self.N)
        if w is not None and eq_open > 0:
            w = np.asarray(w, dtype=np.float64).copy()
            if cfg.smooth < 1.0:
                w = cfg.smooth * w + (1.0 - cfg.smooth) * self.prev_w
            self.prev_w = w.copy()
            w = self.apply_guard(w, t)
            # pre-action constraint: below warning level no risk-increasing trades
            if self._margin_level(op) < c.maintenance_warning:
                cur_g = np.abs(self.shares * op).sum() / eq_open
                g = np.abs(w).sum()
                if g > cur_g > 0:
                    w *= cur_g / g
                elif cur_g == 0:
                    w[:] = 0.0
            w_exec = w
            tgt = w * eq_open / op
            trade = tgt - self.shares
            tv = float(np.abs(trade * op).sum())
            fee = (c.trade_cost + c.slippage) * tv
            self.cash -= float(trade @ op) + fee
            self.shares = tgt
            self.last_L = float(np.abs(w).sum())
        # ---- mark to market at close n
        mv = self.shares * cl
        long_v, short_v = np.maximum(mv, 0).sum(), np.maximum(-mv, 0).sum()
        eq_c = self.cash + mv.sum()
        loan = max(0.0, long_v - max(eq_c, 0.0))
        carry = loan * c.loan_rate_annual / 252 + short_v * c.borrow_fee_annual / 252
        self.cash -= carry
        eq_c -= carry
        # ---- maintenance check at close
        margin_call = 0
        g = long_v + short_v
        liq_fee = 0.0
        sold = np.zeros(self.N)
        if g > 1e-9 and eq_c > 0 and eq_c / g < c.maintenance_strict:
            s = min(1.0, (eq_c / c.liquidation_target) / g)
            sold = self.shares * (1.0 - s)
            liq_fee = c.liquidation_fee * float(np.abs(sold * cl).sum())
            self.cash += float(sold @ cl) - liq_fee
            self.shares = self.shares * s
            eq_c = self.cash + float((self.shares * cl).sum())
            margin_call = 1
            self.n_margin_calls += 1
            fee += liq_fee
        if self.record_ledger:
            self.ledger.append(dict(day=n, trade=trade.copy(), open=op.copy(), close=cl.copy(), fee=fee - liq_fee,
                                    carry=carry, sold=sold.copy(), liq_fee=liq_fee, close_px=cl.copy(),
                                    equity=eq_c, cash=self.cash, shares=self.shares.copy(), margin_call=margin_call))
        self.eq = eq_c
        self.peak = max(self.peak, eq_c)
        dd = 1.0 - eq_c / self.peak
        lr = float(np.log(max(eq_c, 1e-6) / max(eq_prev, 1e-6)))
        self.ret_hist.append(lr)
        # ---- reward
        if cfg.reward == "plain":
            r = 100.0 * lr
        elif cfg.reward == "mt":  # base paper: profit + 5-step Sharpe
            h = np.array(self.ret_hist[-5:])
            sr = h.mean() / (h.std() + 1e-8) if len(h) > 1 else 0.0
            r = 100.0 * lr + 0.5 * float(np.clip(sr, -5, 5))
        else:  # 'mars' / asymmetric sortino: growth - Lagrangian cost terms - downside deviation + cash drag penalty
            dd_ex = max(0.0, dd - cfg.dd_budget)
            downside_pen = cfg.eta_down * (100.0 * min(lr, 0.0)) ** 2
            r = 100.0 * lr - self.mu_margin * margin_call - self.mu_dd * 100.0 * dd_ex - downside_pen
            # Opportunity-cost penalty: in calm bull market (low vol, positive return), penalize idling in cash (< 1.2x gross)
            m_vol = p.mkt_vol[t]
            m_ret60 = p.mkt[t, 3]
            gross_now = float(np.abs(self.shares * cl).sum() / eq_c) if eq_c > 0 else 0.0
            if m_ret60 > 0 and m_vol < 0.18 and gross_now < 1.2:
                r -= 1.5 * (1.2 - gross_now)
        if cfg.turn_pen > 0 and eq_open > 0:
            r -= cfg.turn_pen * tv / eq_open
        ruined = eq_c < c.ruin_fraction * c.initial_equity
        if ruined:
            r -= 10.0
        self.t = n
        terminated = bool(ruined)
        truncated = bool(self.t >= self.t_end)
        gross_now = float(np.abs(self.shares * cl).sum() / eq_c) if eq_c > 0 else 0.0
        info = dict(equity=eq_c, lr=lr, margin_call=margin_call, dd=dd, dd_excess=max(0.0, dd - cfg.dd_budget),
                    gross=gross_now, net=float((self.shares * cl).sum() / eq_c) if eq_c > 0 else 0.0,
                    turnover=tv / eq_open if eq_open > 0 else 0.0, cost=fee, carry=carry, idx=n,
                    margin_level=self._margin_level(cl), w_exec=w_exec)
        return self._obs(), float(r), terminated, truncated, info


def run_episode(env: MarginTradingEnv, act_fn, split_full: bool = True):
    """Run one deterministic pass over the env's split. act_fn(obs, env) -> ('a', action) | ('w', weights|None)."""
    import pandas as pd

    obs, _ = env.reset(options={"full": split_full})
    rows = [dict(date=env.p.dates[env.t], equity=env.eq, lr=0.0, gross=0.0, net=0.0, turnover=0.0,
                 margin_call=0, cost=0.0, carry=0.0, margin_level=10.0)]
    W = []
    done = False
    while not done:
        kind, val = act_fn(obs, env)
        if kind == "a":
            w = weights_from_action(val, env.cfg.mode, env.cfg.lmax, env.cfg.fixed_leverage, env.cfg.kappa)
        else:
            w = val
        W.append(np.zeros(env.N) if w is None else np.asarray(w).copy())
        obs, r, term, trunc, info = env.step_weights(w)
        rows.append(dict(date=env.p.dates[info["idx"]], equity=info["equity"], lr=info["lr"], gross=info["gross"],
                         net=info["net"], turnover=info["turnover"], margin_call=info["margin_call"],
                         cost=info["cost"], carry=info["carry"], margin_level=info["margin_level"]))
        done = term or trunc
    df = pd.DataFrame(rows)
    df.attrs["weights"] = np.array(W)
    return df
