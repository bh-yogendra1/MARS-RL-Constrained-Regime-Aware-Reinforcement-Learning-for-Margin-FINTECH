"""Central configuration for MARS-RL.

All economic assumptions (costs, financing rates, margin rules) live here so that every
strategy (baselines, experts, final model) is evaluated under exactly the same market frictions.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field


@dataclass
class CostConfig:
    # --- trading frictions (applied to EVERY strategy) ---
    trade_cost: float = 0.001          # 0.1% of traded value per side (same as base paper)
    slippage: float = 0.0002           # 2 bps per side (extra, NOT in base paper)
    borrow_fee_annual: float = 0.005   # 0.5%/yr on short market value (assumption, extra)
    loan_rate_annual: float = 0.04     # 4%/yr on borrowed cash (assumption, extra)
    # --- margin rules (base paper: warning 40%, strict 30%) ---
    maintenance_warning: float = 0.40  # below this margin level no risk-increasing trades
    maintenance_strict: float = 0.30   # below this -> margin call / forced liquidation
    liquidation_target: float = 0.50   # after a margin call margin level is restored to 50% (=2x leverage)
    liquidation_fee: float = 0.001     # extra 0.1% on forcibly liquidated value
    initial_equity: float = 100_000.0  # base paper: $100,000
    ruin_fraction: float = 0.2         # episode terminates if equity < 20% of initial

    def scaled(self, k: float) -> "CostConfig":
        """Cost-stress copy: all frictions multiplied by k."""
        d = asdict(self)
        for key in ("trade_cost", "slippage", "borrow_fee_annual", "loan_rate_annual"):
            d[key] = d[key] * k
        return CostConfig(**d)


@dataclass
class EnvConfig:
    mode: str = "ls"                   # 'long' | 'ls' | 'neutral' | 'split' (see margin_env.weights_from_action)
    lmax: float = 2.0                  # maximum gross leverage the policy may request
    fixed_leverage: float | None = None  # if set, leverage is not learned (base-paper style fixed leverage)
    reward: str = "mars"               # 'plain' | 'mt' (profit + 5-step Sharpe, base paper) | 'mars' (CMDP-shaped)
    risk_guard: str = "none"           # 'none' | 'vol' (vol-target cap) | 'conformal' (conformal loss-bound cap)
    vol_target: float = 0.20           # annualised target used by the 'vol' guard
    conformal_budget: float = 0.03     # max tolerated 1-day loss (fraction of equity) at the conformal quantile
    episode_len: int = 504             # training episodes: random 2-year windows
    # Constrained-MDP (Lagrangian) parameters, only used by reward == 'mars'
    mu_margin: float = 0.0             # multiplier on margin-call indicator (updated by dual ascent)
    mu_dd: float = 0.0                 # multiplier on drawdown excess (updated by dual ascent)
    dd_budget: float = 0.10            # drawdown level above which cost accrues
    eta_down: float = 0.05             # small downside-deviation penalty
    turn_pen: float = 0.0              # extra reward penalty per unit daily turnover (v2 roles)
    smooth: float = 1.0                # execution smoothing: w = s*target + (1-s)*previous (1 = off)
    kappa: float = 1.5                 # 'resid' mode: tilt strength around equal weight
    budget_margin_call: float = 0.001  # allowed margin calls per step (~1 per 4 years)
    budget_dd_excess: float = 0.01     # allowed mean drawdown excess per step


@dataclass
class PPOConfig:
    total_timesteps: int = 150_000
    n_envs: int = 4
    n_steps: int = 512
    batch_size: int = 256
    n_epochs: int = 8
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.005
    net_arch: tuple = (128, 128)


@dataclass
class GateConfig:
    horizon: int = 20                  # H: forward window used to score experts in memory
    k: int = 30                        # retrieved neighbours
    beta: float = 10.0                 # softmax temperature on neighbour-averaged expert score
    shrink: float = 0.5                # shrink gate weights towards uniform
    min_memory: int = 60               # use uniform weights until memory has this many entries


# Roles = (trained model name) -> env configuration
ROLES: dict[str, EnvConfig] = {
    # ---- baselines re-implemented from the base paper's description (NOT the authors' code) ----
    "finrl_long": EnvConfig(mode="long", lmax=1.0, fixed_leverage=1.0, reward="plain"),
    "mt_long": EnvConfig(mode="long", lmax=2.0, fixed_leverage=2.0, reward="mt"),
    "mt_split": EnvConfig(mode="split", lmax=2.0, fixed_leverage=2.0, reward="mt"),
    # ---- MARS-RL experts: learned leverage + CMDP shaping + volatility guard ----
    "mars_long": EnvConfig(mode="long", lmax=2.0, reward="mars", risk_guard="conformal"),
    "mars_ls": EnvConfig(mode="ls", lmax=2.0, reward="mars", risk_guard="conformal"),
    "mars_neutral": EnvConfig(mode="neutral", lmax=1.5, reward="mars", risk_guard="conformal"),
    # ---- ablations (trained separately) ----
    "plain_ls": EnvConfig(mode="ls", lmax=2.0, reward="plain", risk_guard="none"),       # no CMDP, no guard
    "mars_ls_notwin": EnvConfig(mode="ls", lmax=2.0, reward="mars", risk_guard="conformal"),  # no twin episodes
    # ---- v2: residual-on-equal-weight policies with turnover control (design iteration, selected on VALIDATION only) ----
    "v2_a": EnvConfig(mode="resid", lmax=2.0, reward="mars", risk_guard="conformal", turn_pen=0.05, smooth=0.3),
    "v2_b": EnvConfig(mode="resid", lmax=2.0, reward="mars", risk_guard="conformal", turn_pen=0.2, smooth=0.3),
    "v2_c": EnvConfig(mode="resid", lmax=2.0, reward="mars", risk_guard="none", turn_pen=0.05, smooth=0.3),
    "v2_d": EnvConfig(mode="resid", lmax=2.0, reward="mt", risk_guard="none", turn_pen=0.05, smooth=0.3),
    "v2_e": EnvConfig(mode="resid", lmax=2.0, reward="mars", risk_guard="conformal", turn_pen=0.0, smooth=1.0),
    "v2_f": EnvConfig(mode="resid", lmax=1.0, fixed_leverage=1.0, reward="plain", risk_guard="none", turn_pen=0.05, smooth=0.3),
    # ---- v3: relax conformal budget / stronger tilt (selected on VALIDATION only; v2_b is the reference) ----
    "v3_a": EnvConfig(mode="resid", lmax=2.0, reward="mars", risk_guard="conformal", conformal_budget=0.05, turn_pen=0.2, smooth=0.3),
    "v3_b": EnvConfig(mode="resid", lmax=2.0, reward="mars", risk_guard="conformal", conformal_budget=0.08, turn_pen=0.2, smooth=0.3),
    "v3_c": EnvConfig(mode="resid", lmax=2.0, reward="mars", risk_guard="conformal", conformal_budget=0.05, turn_pen=0.2, smooth=0.3, kappa=3.0),
    "v3_d": EnvConfig(mode="resid", lmax=2.0, reward="mars", risk_guard="conformal", conformal_budget=0.08, turn_pen=0.2, smooth=0.3, kappa=3.0),
    # ---- v4: tighter conformal budget (validation trend: B 0.08 < 0.05 < 0.03) ----
    "v4_a": EnvConfig(mode="resid", lmax=2.0, reward="mars", risk_guard="conformal", conformal_budget=0.02, turn_pen=0.2, smooth=0.3),
    "v4_b": EnvConfig(mode="resid", lmax=2.0, reward="mars", risk_guard="conformal", conformal_budget=0.015, turn_pen=0.2, smooth=0.3),
}
EXPERT_ROLES = ["mars_long", "mars_ls", "mars_neutral"]
# probability that a training episode is drawn from a digital-twin path instead of the real train history
TWIN_PROB: dict[str, float] = {r: 0.5 for r in ROLES}
for _r in ("finrl_long", "mt_long", "mt_split", "plain_ls", "mars_ls_notwin", "v2_c", "v2_d", "v2_f"):
    TWIN_PROB[_r] = 0.0

SPLITS = {
    "train": ("2009-01-01", "2019-01-01"),   # 2009/01 - 2018/12   (same as base paper)
    "val": ("2019-01-01", "2020-05-01"),     # 2019/01 - 2020/04
    "test": ("2020-05-01", "2023-06-01"),    # 2020/05 - 2023/05   (never used for any tuning)
}
WARMUP = 60                                  # first usable day index (feature warm-up)
