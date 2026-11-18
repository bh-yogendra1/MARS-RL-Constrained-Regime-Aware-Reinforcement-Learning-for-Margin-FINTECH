"""Retrieval-augmented, regime-aware expert gating.

Memory = past market situations (standardised market-state embedding) together with how each RL expert performed
in the H days AFTER that situation. At decision time t only situations s with s + H <= t are retrievable
(outcome embargo -> no future information). The gate retrieves the k most similar past situations, averages each
expert's realised forward risk-adjusted score and turns it into mixture weights.

The retrieval step is the 'RAG' component; an LLM can optionally be placed on top of the retrieved evidence
(see src/news_signals.py: `LLMExpertSelector`), but all reported results use the deterministic kNN selector.
"""
from __future__ import annotations

import numpy as np

from .config import GateConfig
from .data import Panel


def expert_paths(panel: Panel, lo: int, hi: int, dfs: list):
    """Convert expert shadow roll-outs (DataFrames) over [lo,hi) into arrays aligned with panel indices.

    Returns W (K,T,N) pre-guard target weights decided at close t, and R (K,T) daily log-return realised on day t.
    """
    K = len(dfs)
    W = np.zeros((K, panel.T, panel.N))
    R = np.zeros((K, panel.T))
    for k, df in enumerate(dfs):
        w = df.attrs["weights"]                 # weights decided at t = lo .. hi-2
        W[k, lo: lo + len(w)] = w
        lr = df["lr"].values[1:]                # realised on days lo+1 .. hi-1
        R[k, lo + 1: lo + 1 + len(lr)] = lr
    return W, R


class RetrievalGate:
    def __init__(self, panel: Panel, train_range: tuple[int, int], cfg: GateConfig):
        self.p, self.cfg = panel, cfg
        lo, hi = train_range
        mu = panel.mkt[lo:hi].mean(0)
        sd = panel.mkt[lo:hi].std(0) + 1e-6
        self.emb = (panel.mkt - mu) / sd

    def scores(self, R: np.ndarray) -> np.ndarray:
        """S[k,s] = H-day forward Sharpe (daily units) of expert k starting after day s. NaN if window incomplete."""
        K, T = R.shape
        H = self.cfg.horizon
        S = np.full((K, T), np.nan)
        for s in range(T - H):
            seg = R[:, s + 1: s + 1 + H]
            S[:, s] = seg.mean(1) / (seg.std(1) + 1e-6)
        return S

    def gate_weights(self, t: int, S: np.ndarray, mem_lo: int) -> np.ndarray:
        K = S.shape[0]
        c = self.cfg
        valid_hi = t - c.horizon                       # embargo: s + H <= t
        if valid_hi - mem_lo + 1 < c.min_memory:
            return np.ones(K) / K
        idx = np.arange(mem_lo, valid_hi + 1)
        d = np.linalg.norm(self.emb[idx] - self.emb[t], axis=1)
        nb = idx[np.argsort(d)[: min(c.k, len(idx))]]
        e = np.nanmean(S[:, nb], axis=1)
        z = c.beta * (e - e.max())
        soft = np.exp(z) / np.exp(z).sum()
        return c.shrink * np.ones(K) / K + (1 - c.shrink) * soft

    def mix(self, W: np.ndarray, R: np.ndarray, lo: int, hi: int, mem_lo: int, uniform: bool = False):
        """Mixed target weights for decision times t in [lo, hi-1) and the gate weights used."""
        S = self.scores(R)
        K = W.shape[0]
        Wm = np.zeros((hi - 1 - lo, W.shape[2]))
        G = np.zeros((hi - 1 - lo, K))
        for i, t in enumerate(range(lo, hi - 1)):
            g = np.ones(K) / K if uniform else self.gate_weights(t, S, mem_lo)
            G[i] = g
            Wm[i] = np.tensordot(g, W[:, t, :], axes=1)
        return Wm, G
