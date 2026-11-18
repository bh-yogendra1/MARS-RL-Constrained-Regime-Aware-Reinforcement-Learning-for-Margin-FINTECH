"""PPO training of one role with Lagrangian (CMDP) dual ascent; checkpoint chosen on VALIDATION only."""
from __future__ import annotations
import argparse, json, sys, dataclasses, time
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm
from src.config import CostConfig, ROLES, TWIN_PROB, PPOConfig
from src.data import load_panel
from src.margin_env import MarginTradingEnv, run_episode
from src.metrics import perf_metrics
from src.twin import DigitalTwin
from src.gpu_utils import setup_safe_gpu

CSV = ROOT / "data/raw/ohlcv_yahoo_adjusted_28tickers.csv"
ALPHA_MARGIN, ALPHA_DD, MU_MAX = 200.0, 10.0, 5.0


def load_all():
    p = load_panel(str(CSV))
    df = pd.read_csv(CSV); df["date"] = pd.to_datetime(df["date"])
    hi = df.pivot(index="date", columns="tic", values="high").sort_index()[p.tickers].values
    lo = df.pivot(index="date", columns="tic", values="low").sort_index()[p.tickers].values
    return p, hi, lo


class DualAscent(BaseCallback):
    """After each rollout: mu <- clip(mu + alpha*(mean cost - budget), 0, MU_MAX), broadcast to all envs."""
    def __init__(self, cfg, log):
        super().__init__(); self.cfg, self.log = cfg, log; self.mc = []; self.dd = []
    def _on_step(self):
        for i in self.locals["infos"]:
            self.mc.append(i["margin_call"]); self.dd.append(i["dd_excess"])
        return True
    def _on_rollout_end(self):
        if not self.mc: return
        envs = [e.unwrapped for e in self.training_env.envs]
        m, d = envs[0].get_mu()
        m = float(np.clip(m + ALPHA_MARGIN * (np.mean(self.mc) - self.cfg.budget_margin_call) * 0.01, 0, MU_MAX))
        d = float(np.clip(d + ALPHA_DD * (np.mean(self.dd) - self.cfg.budget_dd_excess) * 0.1, 0, MU_MAX))
        for e in envs: e.set_mu(m, d)
        self.log.append(dict(step=self.num_timesteps, mu_margin=m, mu_dd=d, mc_rate=float(np.mean(self.mc)), dd_ex=float(np.mean(self.dd))))
        self.mc, self.dd = [], []


def policy_fn(model):
    def f(obs, env):
        a, _ = model.predict(obs, deterministic=True)
        return "a", a
    return f


def evaluate(model, p, role, split, cost=None):
    env = MarginTradingEnv(p, cost or CostConfig(), ROLES[role], p.split_range(split), guard_override=None)
    return run_episode(env, policy_fn(model))


def train(role: str, seed: int, steps: int, out: Path, twin_override: float | None = None, n_twin=8, chunk=20_000, refit: bool = False, device: str = "auto", max_mem_fraction: float = 0.25):
    torch.set_num_threads(1)
    dev = setup_safe_gpu(device, max_mem_fraction=max_mem_fraction)
    pc, cost, cfg = PPOConfig(), CostConfig(), ROLES[role]
    p, hi, lo = load_all()
    tr = p.split_range("train")
    if refit:  # final model: hyper-parameters fixed on validation; refit on train+val, LAST checkpoint, no selection
        tr = (tr[0], p.split_range("val")[1])
    tp = TWIN_PROB[role] if twin_override is None else twin_override
    twins = DigitalTwin(p, tr, hi, lo).paths(n_twin, 1500, seed0=1000 + 100 * seed) if tp > 0 else []
    mk = lambda i: (lambda: MarginTradingEnv(p, cost, cfg, tr, seed=seed * 100 + i, twin_panels=twins, twin_prob=tp))
    venv = DummyVecEnv([mk(i) for i in range(pc.n_envs)])
    model = PPO("MlpPolicy", venv, n_steps=pc.n_steps, batch_size=pc.batch_size, n_epochs=pc.n_epochs, learning_rate=pc.learning_rate,
                gamma=pc.gamma, gae_lambda=pc.gae_lambda, clip_range=pc.clip_range, ent_coef=pc.ent_coef,
                policy_kwargs=dict(net_arch=list(pc.net_arch)), seed=seed, verbose=0, device=dev)
    log, hist, best = [], [], (-1e9, None)
    cb = DualAscent(cfg, log) if cfg.reward == "mars" else None
    out.mkdir(parents=True, exist_ok=True)
    done, t0 = 0, time.time()
    
    print(f"\n🚀 [TRAINING START] Role: {role} | Seed: {seed} | Total Steps: {steps:,} | Chunk: {chunk:,} | Refit: {refit}")
    print(f"📊 Training device: {dev} | Twin Prob: {tp} | Parallel Envs: {pc.n_envs}")
    
    pbar = tqdm(total=steps, desc=f"Training [{role}-s{seed}]", unit="step", dynamic_ncols=True)
    while done < steps:
        n = min(chunk, steps - done)
        model.learn(n, callback=cb, reset_num_timesteps=False)
        done += n
        pbar.update(n)
        if refit:
            continue
        m = perf_metrics(evaluate(model, p, role, "val"))
        hist.append(dict(step=done, **{k: m[k] for k in ("sharpe", "ann_return", "max_drawdown", "margin_calls")}))
        is_best = m["sharpe"] > best[0]
        if is_best:
            best = (m["sharpe"], done); model.save(out / f"{role}_s{seed}")
        
        status_str = f"SR={m['sharpe']:.3f} | Ret={m['ann_return']*100:.1f}% | MDD={m['max_drawdown']*100:.1f}% | Calls={m['margin_calls']}"
        if is_best:
            status_str += " ⭐ (New Best)"
        pbar.set_postfix_str(status_str)
        tqdm.write(f"  [Step {done:6d}/{steps}] Val Metrics -> {status_str}")
        
    pbar.close()
    if refit:
        model.save(out / f"{role}_refit_s{seed}"); best = (float("nan"), done)
        print(f"💾 Saved refit model to: {out / f'{role}_refit_s{seed}.zip'}")
    else:
        print(f"🏆 Best Checkpoint: Step {best[1]} with Val Sharpe = {best[0]:.3f} -> Saved to: {out / f'{role}_s{seed}.zip'}")
        
    elapsed = time.time() - t0
    print(f"⏱️ Total Time Elapsed: {elapsed:.2f}s ({steps / max(elapsed, 1e-6):.1f} steps/s)\n")
    json.dump(dict(role=role, seed=seed, steps=steps, refit=refit, best_val_sharpe=best[0], best_step=best[1], val_history=hist,
                   dual_log=log[::max(1, len(log) // 200)], twin_prob=tp, secs=elapsed), open(out / f"{role}_{'refit_' if refit else ''}s{seed}.json", "w"), indent=1)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5000); ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--twin", type=float, default=None); ap.add_argument("--refit", action="store_true"); ap.add_argument("--chunk", type=int, default=20_000)
    ap.add_argument("--device", default="auto", help="auto, cuda:X, or cpu")
    ap.add_argument("--max-mem-fraction", type=float, default=0.25, help="Safe shared GPU memory ceiling (0.0 - 1.0)")
    a = ap.parse_args()
    print(train(a.role, a.seed, a.steps, Path(a.out), a.twin, chunk=a.chunk, refit=a.refit, device=a.device, max_mem_fraction=a.max_mem_fraction))
