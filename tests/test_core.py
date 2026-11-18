"""Unit tests: ledger identities, no look-ahead, guard monotonicity, margin events, determinism, gate embargo."""
import sys, dataclasses
from pathlib import Path
import numpy as np, pytest
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.data import load_panel, build_panel
from src.config import CostConfig, ROLES, GateConfig
from src.margin_env import MarginTradingEnv, weights_from_action
from src.gating import RetrievalGate

CSV = ROOT / "data/raw/ohlcv_yahoo_adjusted_28tickers.csv"

@pytest.fixture(scope="module")
def P(): return load_panel(str(CSV))

def _env(P, role="mars_ls", rng="val", **kw):
    return MarginTradingEnv(P, CostConfig(), ROLES[role], P.split_range(rng), record_ledger=True, **kw)

def test_ledger_identity(P):
    e = _env(P); e.reset(options={"full": True}); rs = np.random.default_rng(0)
    for _ in range(200):
        e.step(rs.uniform(-1, 1, P.N + 1))
        L = e.ledger[-1]
        assert abs(L["equity"] - (L["cash"] + (L["shares"] * L["close"]).sum())) < 1e-6

def test_determinism(P):
    out = []
    for _ in range(2):
        e = _env(P); e.reset(seed=3, options={"full": True}); rs = np.random.default_rng(5)
        for _ in range(150): e.step(rs.uniform(-1, 1, P.N + 1))
        out.append(e.eq)
    assert out[0] == out[1]

def test_no_lookahead_scramble(P):
    """Features/cov/conf_q at t must not change if all prices after t are scrambled."""
    import pandas as pd
    t0 = 1500
    df = __import__("pandas").read_csv(CSV); df["date"] = df["date"].astype("datetime64[ns]")
    piv = {c: df.pivot(index="date", columns="tic", values=c).sort_index()[P.tickers].values.copy()
           for c in ["open", "high", "low", "close", "volume"]}
    rs = np.random.default_rng(0)
    for k in piv:
        piv[k][t0 + 1:] = piv[k][t0 + 1:][rs.permutation(len(piv[k]) - t0 - 1)] * rs.uniform(0.5, 2)
    dates = P.dates
    Q = build_panel(dates, P.tickers, piv["open"], piv["high"], piv["low"], piv["close"], piv["volume"])
    for name in ["feats", "mkt", "cov", "conf_q", "mkt_vol"]:
        assert np.allclose(getattr(P, name)[: t0 + 1], getattr(Q, name)[: t0 + 1], atol=1e-6, equal_nan=True), name

def test_guard_never_increases_exposure(P):
    rs = np.random.default_rng(1)
    for g in ["vol", "conformal"]:
        e = MarginTradingEnv(P, CostConfig(), ROLES["mars_ls"], P.split_range("val"), guard_override=g)
        for _ in range(300):
            t = int(rs.integers(100, P.T - 2)); w = rs.normal(size=P.N); w = w / np.abs(w).sum() * rs.uniform(0.2, 2)
            assert np.abs(e.apply_guard(w, t)).sum() <= np.abs(w).sum() + 1e-12

def test_margin_call_triggers_and_restores(P):
    cost = CostConfig(); e = MarginTradingEnv(P, cost, dataclasses.replace(ROLES["finrl_long"], fixed_leverage=None, lmax=50.0, risk_guard="none"),
                                              P.split_range("val"), record_ledger=True)
    e.reset(options={"full": True}); calls = 0
    for _ in range(120):
        w = np.ones(P.N) / P.N * 8.0       # absurd 8x leverage to force events
        _, _, te, tr, i = e.step_weights(w); calls += i["margin_call"]
        if i["margin_call"]: assert 0.45 < i["margin_level"] <= 0.55 or i["equity"] <= 0
        if te or tr: break
    assert calls > 0

def test_costs_charged(P):
    e = _env(P, "finrl_long"); e.reset(options={"full": True})
    _, _, _, _, i = e.step_weights(np.ones(P.N) / P.N)
    assert i["cost"] > 0 and i["turnover"] > 0.9

def test_weights_gross_equals_leverage():
    a = np.random.default_rng(0).uniform(-1, 1, 29)
    for m in ["long", "ls", "neutral", "split"]:
        w = weights_from_action(a, m, 2.0, 1.5); assert abs(np.abs(w).sum() - 1.5) < 1e-9
        if m == "long": assert (w >= 0).all()
        if m == "neutral": assert abs(w.sum()) < 1e-9

def test_gate_embargo(P):
    g = RetrievalGate(P, P.split_range("train"), GateConfig(min_memory=5))
    rs = np.random.default_rng(2); R = rs.normal(0, 0.01, (3, P.T)); S = g.scores(R)
    t, lo = 2600, P.split_range("val")[0]
    w1 = g.gate_weights(t, S, lo)
    R2 = R.copy(); R2[:, t + 1:] += 5.0          # corrupt the future
    assert np.allclose(w1, g.gate_weights(t, g.scores(R2), lo))
