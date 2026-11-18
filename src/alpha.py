"""Supervised cross-sectional alpha model (train-only fit, embargoed) -> predictions for every day.
Target: demeaned forward H-day log return of close(t+1)->close(t+1+H) (decision at close t, execute at next open ~ close proxy)."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

def build_xy(panel, H):
    T, N, F = panel.feats.shape
    X = np.concatenate([panel.feats, np.repeat(panel.mkt[:, None, :], N, 1)], axis=2)       # (T,N,F+M)
    lc = np.log(panel.close)
    fwd = np.full((T, N), np.nan); fwd[:T - H - 1] = lc[1 + H:T] - lc[1:T - H]
    y = fwd - np.nanmean(fwd, axis=1, keepdims=True)
    return X, y

def fit_predict(panel, H, fit_range, kind="ridge", seed=0):
    X, y = build_xy(panel, H); lo, hi = fit_range
    hi_eff = hi - H - 1                                  # embargo: labels must be realised before fit_range end
    Xtr = X[lo:hi_eff].reshape(-1, X.shape[2]); ytr = y[lo:hi_eff].reshape(-1); ok = ~np.isnan(ytr)
    if kind == "ridge":
        m = Ridge(alpha=50.0).fit(Xtr[ok], ytr[ok])
    else:
        m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0, random_state=seed).fit(Xtr[ok], ytr[ok])
    P = m.predict(X.reshape(-1, X.shape[2])).reshape(X.shape[0], X.shape[1])
    return P

def rank_ic(P, y, lo, hi):
    from scipy.stats import spearmanr
    ic = [spearmanr(P[t], y[t])[0] for t in range(lo, hi) if not np.isnan(y[t]).any()]
    return float(np.mean(ic)), float(np.mean(ic) / (np.std(ic) + 1e-9) * np.sqrt(len(ic))), len(ic)
