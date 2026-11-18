"""Regime-switching block-bootstrap digital twin (train-period data ONLY).

Whole cross-sections (day rows) are resampled so cross-asset correlation is preserved. Regimes = terciles of the
past-only market vol (panel.mkt_vol) on the TRAIN range; the regime follows a Markov chain estimated on train; within a
regime, geometric blocks (mean length 10) of consecutive same-regime-start days are drawn. Synthetic OHLC are rebuilt from
resampled (open/prev_close, high/close, low/close, close/prev_close) ratios, then pushed through the SAME build_panel path.
Stress paths: demeaned returns + drift -0.15%/day and vol x1.3.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .data import Panel, build_panel


class DigitalTwin:
    def __init__(self, panel: Panel, train_range: tuple[int, int], high: np.ndarray, low: np.ndarray, block: int = 10):
        lo, hi = train_range
        self.p, self.block = panel, block
        self.hi_, self.lo_ = high, low
        self.idx = np.arange(max(lo, 1), hi)
        v = panel.mkt_vol[self.idx]
        q = np.quantile(v, [1 / 3, 2 / 3])
        self.reg = np.digitize(v, q)                       # 0,1,2
        K = 3
        P = np.ones((K, K))                                # Laplace smoothing
        for a, b in zip(self.reg[:-1], self.reg[1:]):
            P[a, b] += 1
        self.P = P / P.sum(1, keepdims=True)
        self.by_reg = [np.where(self.reg == k)[0] for k in range(K)]
        o, c = panel.open, panel.close
        pc = np.vstack([c[:1], c[:-1]])
        self.r_open = o / pc; self.r_hi = high / c; self.r_lo = low / c; self.r_cl = c / pc
        self.vol = panel.volume

    def sample_rows(self, T: int, rng: np.random.Generator) -> np.ndarray:
        rows, k = [], int(rng.choice(3))
        while len(rows) < T:
            L = int(rng.geometric(1.0 / self.block))
            pool = self.by_reg[k]
            s = int(rng.choice(pool))
            rows.extend(self.idx[min(s + j, len(self.idx) - 1)] for j in range(L))
            k = int(rng.choice(3, p=self.P[k]))
        return np.array(rows[:T])

    def generate(self, T: int, seed: int, stress: bool = False) -> Panel:
        rng = np.random.default_rng(seed)
        rows = self.sample_rows(T, rng)
        p = self.p
        rc = self.r_cl[rows].copy()
        if stress:
            lr = np.log(rc); lr = (lr - lr.mean(0)) * 1.3 - 0.0015
            rc = np.exp(lr)
        close = p.close[self.idx[0]] * np.cumprod(rc, axis=0)
        prev = np.vstack([p.close[self.idx[0]], close[:-1]])
        gap = self.r_open[rows]  # overnight gap ratio vs prev close
        op = prev * gap
        hi = np.maximum(close * self.r_hi[rows], np.maximum(op, close))
        lo = np.minimum(close * self.r_lo[rows], np.minimum(op, close))
        dates = pd.bdate_range("2000-01-03", periods=T)
        return build_panel(dates, p.tickers, op, hi, lo, close, self.vol[rows])

    def paths(self, n: int, T: int, seed0: int = 1000, stress: bool = False) -> list[Panel]:
        return [self.generate(T, seed0 + i, stress) for i in range(n)]


def fidelity_table(real: Panel, twin: Panel, rng_real: tuple[int, int]) -> pd.DataFrame:
    """Stylised facts: real train vs twin (mean over assets): ann. vol, kurtosis, lag-1 |r| autocorr, avg pairwise corr, max DD EW."""
    def stats(close):
        r = np.diff(np.log(close), axis=0)
        ew = r.mean(1); cum = np.cumsum(ew)
        dd = (np.exp(cum) / np.maximum.accumulate(np.exp(cum)) - 1).min()
        k = pd.DataFrame(r).kurt().mean()
        ac = np.mean([np.corrcoef(np.abs(r[1:, j]), np.abs(r[:-1, j]))[0, 1] for j in range(r.shape[1])])
        C = np.corrcoef(r.T); cc = (C.sum() - len(C)) / (len(C) ** 2 - len(C))
        return dict(ann_vol=r.std(0).mean() * np.sqrt(252), excess_kurtosis=k, abs_ret_autocorr=ac, avg_corr=cc, ew_maxdd=dd)
    lo, hi = rng_real
    return pd.DataFrame({"real_train": stats(real.close[lo:hi]), "twin": stats(twin.close)}).T
