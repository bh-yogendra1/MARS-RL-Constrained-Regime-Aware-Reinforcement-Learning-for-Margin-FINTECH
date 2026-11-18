"""Exports the FINAL processed data the RL models actually consume (long format, one row per date x ticker)."""
import sys; from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.data import load_panel, STOCK_FEATURES, MKT_FEATURES
p = load_panel(str(ROOT / "data/raw/ohlcv_yahoo_adjusted_28tickers.csv"))
T, N = p.T, p.N
d = pd.DataFrame({"date": np.repeat(p.dates, N), "tic": np.tile(p.tickers, T), "open": p.open.ravel(), "close": p.close.ravel(), "volume": p.volume.ravel()})
for j, f in enumerate(STOCK_FEATURES): d[f] = p.feats[:, :, j].ravel()
for j, f in enumerate(MKT_FEATURES): d[f] = np.repeat(p.mkt[:, j], N)
d["mkt_vol20_ann"] = np.repeat(p.mkt_vol, N); d["conformal_q95"] = np.repeat(p.conf_q, N)
lo, hi = p.split_range("train"); v0, v1 = p.split_range("val"); t0, t1 = p.split_range("test")
sp = np.array(["warmup"] * T, dtype=object); sp[lo:hi] = "train"; sp[v0:v1] = "val"; sp[t0:t1] = "test"
d["split"] = np.repeat(sp, N)
d.to_csv(ROOT / "data/processed/model_input_features_panel.csv", index=False, float_format="%.6g")
pd.DataFrame(p.cov[-1], index=p.tickers, columns=p.tickers).to_csv(ROOT / "data/processed/ewma_cov_last_day.csv")
print(d.shape, d.split.value_counts().to_dict())
