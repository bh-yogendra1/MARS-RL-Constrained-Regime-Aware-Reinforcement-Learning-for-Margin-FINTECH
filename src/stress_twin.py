"""Crash stress on digital-twin paths (train-only twin, drift -0.15%/day, vol x1.3, 250 days, 8 paths).
Reports per-path metrics for baselines, roles and the MARS gate variants (gate memory = first 60 days of each path)."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.config import CostConfig
from src.evaluate import load_models, run_config
from src.metrics import perf_metrics
from src.train import load_all
from src.twin import DigitalTwin


def main(model_dir, out_dir, seeds, n_paths=8, T=310):
    torch.set_num_threads(1)
    p, hi, lo = load_all(); tr = p.split_range("train")
    models = load_models(model_dir, seeds)
    paths = DigitalTwin(p, tr, hi, lo).paths(n_paths, T, seed0=5000, stress=True)
    rows = []
    for i, q in enumerate(paths):
        rng = (60, q.T)
        curves = run_config(q, CostConfig(), models, seeds, rng, (60, q.T), 120)
        for (name, s), df in curves.items():
            sub = df[df.index >= 60 + 0].reset_index(drop=True)   # evaluate whole path after warm-up
            m = perf_metrics(sub); rows.append(dict(path=i, model=name, seed=s, **m))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "metrics_twin_crash.csv", index=False); print("rows", len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT / "models/stage2")); ap.add_argument("--out", default=str(ROOT / "results/trackB"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args(); main(a.models, a.out, a.seeds)
