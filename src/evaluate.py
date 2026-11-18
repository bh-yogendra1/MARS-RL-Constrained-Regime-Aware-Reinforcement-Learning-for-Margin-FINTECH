"""Locked evaluation: baselines + RL roles + MARS gate under common costs; cost stress; twin crash stress.
Models are chosen/trained on train(+val for checkpoint pick) ONLY. Val and test are run as ONE contiguous roll-out
('valtest', one account) and sliced by date; the test slice is evaluated once per configuration (no tuning on test).
"""
from __future__ import annotations
import argparse, json, sys, dataclasses
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from stable_baselines3 import PPO
from src.config import CostConfig, EnvConfig, GateConfig, ROLES, EXPERT_ROLES
from src.margin_env import MarginTradingEnv, run_episode
from src.metrics import perf_metrics
from src.baselines import BASELINES
from src.gating import RetrievalGate, expert_paths
from src.train import load_all, policy_fn
from src.gpu_utils import setup_safe_gpu

GATE_CFG = EnvConfig(mode="ls", reward="plain", risk_guard="conformal")   # execution env for mixed weights
BASE_CFG = EnvConfig(mode="ls", reward="plain", risk_guard="none")
SPLIT_NAMES = ("val", "test")


def rollout(panel, cost, cfg, rng, fn):
    return run_episode(MarginTradingEnv(panel, cost, cfg, rng), fn)


def slice_metrics(df, panel, which):
    lo, hi = panel.split_range(which)
    d0, d1 = panel.dates[lo], panel.dates[hi - 1]
    s = df[(df["date"] >= d0) & (df["date"] <= d1)]
    return perf_metrics(s.reset_index(drop=True)) if len(s) > 5 else None


def mixed_curve(panel, cost, W, R, lo, hi, mem_lo, gate, kind, seed=0, fixed_k=None):
    """kind: gate | uniform | fixed | shuffled"""
    K = W.shape[0]
    if kind == "fixed":
        Wm = W[fixed_k, lo:hi - 1]
    elif kind == "shuffled":
        S = gate.scores(R); rng = np.random.default_rng(seed)
        valid = np.arange(mem_lo, panel.T - gate.cfg.horizon)
        S2 = S.copy(); S2[:, valid] = S[:, rng.permutation(valid)]
        Wm = np.zeros((hi - 1 - lo, W.shape[2]))
        for i, t in enumerate(range(lo, hi - 1)):
            Wm[i] = np.tensordot(gate.gate_weights(t, S2, mem_lo), W[:, t, :], axes=1)
    else:
        Wm, _ = gate.mix(W, R, lo, hi, mem_lo, uniform=(kind == "uniform"))
    fn = lambda obs, env: ("w", Wm[env.t - lo])
    return rollout(panel, cost, GATE_CFG, (lo, hi), fn)


def run_config(panel, cost, models, seeds, rng=None, train_rng=None, mem_lo=None, tag="", extra_roles=()):
    """Return dict name -> equity DataFrame for baselines, every role/seed, MARS gate variants."""
    lo, hi = rng
    curves = {}
    for n, mk in BASELINES.items():
        curves[(n, 0)] = rollout(panel, cost, BASE_CFG, rng, mk())
    for role in ROLES:
        for s in seeds:
            m = models.get((role, s))
            if m is None: continue
            curves[(role, s)] = rollout(panel, cost, ROLES[role], rng, policy_fn(m))
    for s in seeds:
        if not all((r, s) in models for r in EXPERT_ROLES): continue
        dfs = [curves[(r, s)] for r in EXPERT_ROLES]
        W, R = expert_paths(panel, lo, hi, dfs)
        gate = RetrievalGate(panel, train_rng, GateConfig())
        for kind in ("gate", "uniform", "shuffled"):
            curves[(f"MARS-{kind}", s)] = mixed_curve(panel, cost, W, R, lo, hi, mem_lo, gate, kind, seed=s)
        for k, r in enumerate(EXPERT_ROLES):
            curves[(f"fixed-{r}", s)] = mixed_curve(panel, cost, W, R, lo, hi, mem_lo, gate, "fixed", fixed_k=k)
    return curves


def load_models(model_dir, seeds, device=None):
    out = {}
    for role in ROLES:
        for s in seeds:
            f = Path(model_dir) / f"{role}_s{s}.zip"
            if f.exists():
                out[(role, s)] = PPO.load(str(f), device=device or "cpu")
    return out


def main(model_dir, out_dir, seeds, cost_mults=(1.0, 2.0, 3.0), device="auto", max_mem_fraction=0.20):
    torch.set_num_threads(1)
    dev = setup_safe_gpu(device, max_mem_fraction=max_mem_fraction)
    panel, hi_, lo_ = load_all()
    train_rng = panel.split_range("train")
    vt = panel.split_range("valtest"); mem_lo = panel.split_range("val")[0]
    models = load_models(model_dir, seeds, device=dev)
    out = Path(out_dir); (out / "curves").mkdir(parents=True, exist_ok=True)
    rows = []
    for k in cost_mults:
        cost = CostConfig().scaled(k)
        curves = run_config(panel, cost, models, seeds, vt, train_rng, mem_lo)
        for (name, s), df in curves.items():
            df.to_csv(out / "curves" / f"cost{k:g}x__{name}__s{s}.csv", index=False)
            for sp in SPLIT_NAMES:
                m = slice_metrics(df, panel, sp)
                if m: rows.append(dict(cost_mult=k, model=name, seed=s, split=sp, **m))
    pd.DataFrame(rows).to_csv(out / "metrics_main.csv", index=False)
    print("wrote", len(rows), "rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT / "models/stage2")); ap.add_argument("--out", default=str(ROOT / "results/trackB"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="auto", help="auto, cuda:X, or cpu")
    ap.add_argument("--max-mem-fraction", type=float, default=0.20, help="Safe memory cap fraction")
    a = ap.parse_args(); main(a.models, a.out, a.seeds, device=a.device, max_mem_fraction=a.max_mem_fraction)
