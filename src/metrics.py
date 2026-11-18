"""Performance metrics and statistical tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ANN = 252


def perf_metrics(df: pd.DataFrame) -> dict:
    """df must contain 'equity' (daily, first row = initial) and optionally gross/turnover/margin_call/cost."""
    eq = df["equity"].values.astype(float)
    r = np.diff(eq) / eq[:-1]
    n = len(r)
    acc = eq[-1] / eq[0] - 1.0
    cagr = (eq[-1] / eq[0]) ** (ANN / max(n, 1)) - 1.0 if eq[-1] > 0 else -1.0
    vol = r.std(ddof=1) * np.sqrt(ANN)
    sharpe = r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(ANN)
    dn = r[r < 0]
    sortino = r.mean() / (np.sqrt((dn ** 2).sum() / max(n, 1)) + 1e-12) * np.sqrt(ANN)
    peak = np.maximum.accumulate(eq)
    mdd = float(((eq - peak) / peak).min())
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    q = np.quantile(r, 0.05)
    cvar5 = float(r[r <= q].mean())
    out = dict(acc_return=acc, ann_return=cagr, ann_vol=vol, sharpe=sharpe, sortino=sortino, max_drawdown=mdd,
               calmar=calmar, cvar5_daily=cvar5)
    if "gross" in df:
        out["avg_gross"] = float(df["gross"].mean())
    if "turnover" in df:
        out["avg_turnover"] = float(df["turnover"].mean())
    if "margin_call" in df:
        out["margin_calls"] = int(df["margin_call"].sum())
    if "cost" in df:
        out["total_cost_pct_init"] = float(df["cost"].sum() / eq[0])
    return out


def daily_returns(df: pd.DataFrame) -> np.ndarray:
    eq = df["equity"].values.astype(float)
    return np.diff(eq) / eq[:-1]


def _sharpe(r: np.ndarray) -> float:
    return r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(ANN)


def stationary_bootstrap_idx(n: int, B: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices, shape (B, n)."""
    p = 1.0 / mean_block
    idx = np.empty((B, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, B)
    new = rng.random((B, n)) < p
    jumps = rng.integers(0, n, (B, n))
    for t in range(1, n):
        idx[:, t] = np.where(new[:, t], jumps[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def bootstrap_sharpe_diff(r1: np.ndarray, r2: np.ndarray, B: int = 2000, mean_block: int = 10, seed: int = 0) -> dict:
    """Paired stationary bootstrap of Sharpe(r1)-Sharpe(r2); returns diff, 95% CI and one-sided p (diff<=0)."""
    n = min(len(r1), len(r2))
    r1, r2 = r1[:n], r2[:n]
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_idx(n, B, mean_block, rng)
    a, b = r1[idx], r2[idx]
    sa = a.mean(1) / (a.std(1, ddof=1) + 1e-12) * np.sqrt(ANN)
    sb = b.mean(1) / (b.std(1, ddof=1) + 1e-12) * np.sqrt(ANN)
    d = sa - sb
    return dict(diff=float(_sharpe(r1) - _sharpe(r2)), ci_low=float(np.quantile(d, 0.025)),
                ci_high=float(np.quantile(d, 0.975)), p_one_sided=float((d <= 0).mean()))


def jobson_korkie_memmel(r1: np.ndarray, r2: np.ndarray) -> dict:
    """Jobson-Korkie test with Memmel (2003) correction for equality of two Sharpe ratios (daily, non-annualised)."""
    n = min(len(r1), len(r2))
    r1, r2 = r1[:n], r2[:n]
    m1, m2 = r1.mean(), r2.mean()
    s1, s2 = r1.std(ddof=1), r2.std(ddof=1)
    rho = np.corrcoef(r1, r2)[0, 1]
    sr1, sr2 = m1 / s1, m2 / s2
    var = (1.0 / n) * (2 - 2 * rho + 0.5 * (sr1 ** 2 + sr2 ** 2 - 2 * sr1 * sr2 * rho ** 2))
    z = (sr1 - sr2) / np.sqrt(max(var, 1e-18))
    return dict(z=float(z), p_two_sided=float(2 * (1 - stats.norm.cdf(abs(z)))))


def deflated_sharpe(r: np.ndarray, trial_sharpes_daily: np.ndarray) -> dict:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio.

    r: daily returns of the selected strategy. trial_sharpes_daily: daily (non-annualised) Sharpe ratios of ALL
    configurations that were tried when selecting it. Returns the probability that the true Sharpe is > 0 after
    accounting for multiple testing and non-normality.
    """
    n = len(r)
    sr = r.mean() / (r.std(ddof=1) + 1e-12)
    g3, g4 = stats.skew(r), stats.kurtosis(r, fisher=False)
    N = max(len(trial_sharpes_daily), 2)
    V = float(np.var(trial_sharpes_daily, ddof=1)) if N > 1 else 0.0
    emc = 0.5772156649
    sr0 = np.sqrt(V) * ((1 - emc) * stats.norm.ppf(1 - 1.0 / N) + emc * stats.norm.ppf(1 - 1.0 / (N * np.e)))
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    z = (sr - sr0) * np.sqrt(n - 1) / denom
    return dict(dsr=float(stats.norm.cdf(z)), sr0_daily=float(sr0), sr_daily=float(sr), n_trials=int(N))
