"""Track A: the AUTHORS' OWN simulator and agent code (pinned commit 7b7fa9e, MIT) run on our 28-ticker data.

This mirrors `trading_margin_a2c.py` from the authors' repo, with two changes only:
  * data comes from data/features/trackA_finrl_format.csv (no Yahoo download in the sandbox)
  * metrics are computed by our own `metrics.py` from the returned account-value curve
Everything else (env, reward, hyper-parameters, 50k A2C steps, VIX/turbulence=70 rule at test time) is untouched.
Their simulator assumes zero slippage/impact and a different (separate long/short) account ledger, so Track A numbers are
NOT directly comparable to Track B (our common-cost simulator). Both are reported separately.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party"))
sys.path.insert(0, str(ROOT / "third_party" / "margin_trader"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402

from env.agent import DRLAgent  # noqa: E402  (authors)
from env.marginEnv import MarginTradingEnv  # noqa: E402  (authors)
from finrl.config import INDICATORS  # noqa: E402
from finrl.meta.preprocessor.preprocessors import data_split  # noqa: E402
from src.metrics import perf_metrics  # noqa: E402

TRAIN = ("2009-01-01", "2018-12-31")
VAL = ("2019-01-01", "2020-04-30")
TRADE = ("2020-05-01", "2023-05-01")


def run(seed: int, algo: str, timesteps: int, out: Path, initial_amount: int = 1_000_000):
    torch.set_num_threads(1)
    set_random_seed(seed)
    processed = pd.read_csv(ROOT / "data/features/trackA_finrl_format.csv", index_col=0)
    train = data_split(processed, *TRAIN)
    val = data_split(processed, *VAL)
    trade = data_split(processed, *TRADE)
    n = len(train.tic.unique())
    state_space = 2 * 3 + 2 * n + len(INDICATORS) * n
    env_kwargs = dict(hmax=100, initial_amount=initial_amount, num_stock_shares=[0] * n,
                      buy_cost_pct=[0.001] * n, sell_cost_pct=[0.001] * n, state_space=state_space, stock_dim=n,
                      tech_indicator_list=INDICATORS, action_space=2 * n, reward_scaling=1e-4, penalty_sharpe=0.05)
    env_train, _ = MarginTradingEnv(df=train, **env_kwargs).get_sb_env()
    agent = DRLAgent(env=env_train)
    params = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.005, "gamma": 0.99} if algo == "a2c" else \
             {"n_steps": 2048, "ent_coef": 0.005, "learning_rate": 0.0003, "batch_size": 64}
    model = agent.get_model(algo, model_kwargs=params, seed=seed)
    trained = agent.train_model(model=model, tb_log_name=algo, total_timesteps=timesteps)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, df in [("val", val), ("test", trade)]:
        e = MarginTradingEnv(df=df, turbulence_threshold=70, risk_indicator_col="vix", **env_kwargs)
        av, act, st = DRLAgent.DRL_prediction(model=trained, environment=e)
        av = av.rename(columns={"account_value": "equity"})
        av.to_csv(out / f"trackA_{algo}_s{seed}_{name}_equity.csv", index=False)
        res[name] = perf_metrics(av)
    json.dump(res, open(out / f"trackA_{algo}_s{seed}_metrics.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--algo", default="a2c")
    ap.add_argument("--timesteps", type=int, default=50_000)
    ap.add_argument("--out", default=str(ROOT / "results/trackA"))
    a = ap.parse_args()
    r = run(a.seed, a.algo, a.timesteps, Path(a.out))
    print(json.dumps(r, indent=1))
