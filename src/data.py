"""Data loading and point-in-time feature engineering.

Every feature at day t uses only information available at the close of day t
(rolling windows over the past). Trades decided at close t are executed at the
open of t+1 (see margin_env.py).

Indicators (MACD, RSI-30, CCI-30, DX-30 as "ADX") are the standard OHLC stockstats definitions used by the base
paper/FinRL. They are computed from (T,N) arrays so real and synthetic (digital-twin) panels go through exactly the
same code path (see data_prep.indicator_arrays).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .config import SPLITS, WARMUP
from .data_prep import indicator_arrays

STOCK_FEATURES = ["ret1", "ret5", "ret20", "ret60", "vol20", "sma50_gap", "macd", "rsi", "cci", "adx"]
MKT_FEATURES = ["m_ret1", "m_ret5", "m_ret20", "m_ret60", "m_vol20", "m_vol60", "m_dd", "m_breadth", "m_disp"]
EWMA_LAMBDA = 0.94
CONF_WINDOW = 250
CONF_LEVEL = 0.95


@dataclass
class Panel:
    dates: np.ndarray          # (T,) datetime64[D]
    tickers: list[str]
    open: np.ndarray           # (T,N) split/dividend adjusted
    close: np.ndarray          # (T,N)
    volume: np.ndarray         # (T,N)
    feats: np.ndarray          # (T,N,F)
    mkt: np.ndarray            # (T,M) market-state features (also used as retrieval embedding)
    mkt_vol: np.ndarray        # (T,) annualised 20d vol of the equal-weight index (past only)
    extra: np.ndarray          # (T,N) optional external signal (e.g. LLM news sentiment); zeros by default
    cov: np.ndarray            # (T,N,N) EWMA covariance of daily log-returns, known at close t
    conf_q: np.ndarray         # (T,) conformal quantile of |standardised 1-day EW-portfolio return|, known at close t

    @property
    def T(self) -> int:
        return len(self.dates)

    @property
    def N(self) -> int:
        return len(self.tickers)

    def split_range(self, name: str) -> tuple[int, int]:
        """[lo, hi) index range of a split. 'valtest' = val and test contiguous."""
        if name == "valtest":
            lo, _ = self.split_range("val")
            _, hi = self.split_range("test")
            return lo, hi
        s, e = SPLITS[name]
        lo = int(np.searchsorted(self.dates, np.datetime64(s), side="left"))
        hi = int(np.searchsorted(self.dates, np.datetime64(e), side="left"))
        return (max(lo, WARMUP) if name == "train" else lo), hi


def _ema(x: pd.DataFrame, span: int) -> pd.DataFrame:
    return x.ewm(span=span, adjust=False).mean()


def build_panel(dates, tickers, open_, high_, low_, close_, volume_, extra: np.ndarray | None = None) -> Panel:
    """Build a Panel (features, market state, covariance, conformal quantile) from raw price arrays (T,N)."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    close = pd.DataFrame(close_, index=idx, columns=tickers)
    lr = np.log(close).diff().fillna(0.0)

    f = {}
    f["ret1"] = lr * 20
    f["ret5"] = (close / close.shift(5) - 1) * 10
    f["ret20"] = (close / close.shift(20) - 1) * 5
    f["ret60"] = (close / close.shift(60) - 1) * 3
    f["vol20"] = lr.rolling(20).std() * 30
    f["sma50_gap"] = (close / close.rolling(50).mean() - 1) * 5
    ind = indicator_arrays(np.asarray(open_), np.asarray(high_), np.asarray(low_), np.asarray(close_),
                           np.asarray(volume_), list(tickers), idx, ["macd", "rsi_30", "cci_30", "dx_30"])
    f["macd"] = pd.DataFrame(ind["macd"] / np.asarray(close_) * 50, index=idx, columns=tickers)
    f["rsi"] = pd.DataFrame((ind["rsi_30"] - 50) / 25, index=idx, columns=tickers)
    f["cci"] = pd.DataFrame(ind["cci_30"] / 150, index=idx, columns=tickers)
    f["adx"] = pd.DataFrame((ind["dx_30"] - 20) / 30, index=idx, columns=tickers)
    feats = np.stack([f[k].fillna(0.0).values for k in STOCK_FEATURES], axis=-1)
    feats = np.clip(feats, -3, 3).astype(np.float32)

    ew = lr.mean(axis=1)                              # equal-weight index log return
    cum = np.exp(ew.cumsum())
    m = pd.DataFrame(index=close.index)
    m["m_ret1"] = ew * 20
    m["m_ret5"] = ew.rolling(5).sum() * 10
    m["m_ret20"] = ew.rolling(20).sum() * 5
    m["m_ret60"] = ew.rolling(60).sum() * 3
    vol20 = ew.rolling(20).std() * np.sqrt(252)
    m["m_vol20"] = vol20 / 0.2
    m["m_vol60"] = ew.rolling(60).std() * np.sqrt(252) / 0.2
    m["m_dd"] = (cum / cum.rolling(252, min_periods=1).max() - 1) * 5
    m["m_breadth"] = ((close > close.rolling(50).mean()).mean(axis=1) - 0.5) * 2
    m["m_disp"] = lr.std(axis=1) * 50
    mkt = np.clip(m[MKT_FEATURES].fillna(0.0).values, -5, 5).astype(np.float32)

    # ---- EWMA covariance (known at close t) and conformal quantile of standardised EW-portfolio return
    R = lr.values
    T, N = R.shape
    cov = np.zeros((T, N, N))
    init = np.cov(R[1:61].T) if T > 61 else np.eye(N) * 1e-4
    c = init.copy()
    for t in range(T):
        if t >= 1:
            c = EWMA_LAMBDA * c + (1 - EWMA_LAMBDA) * np.outer(R[t], R[t])
        cov[t] = c
    w_ew = np.ones(N) / N
    sig_ew = np.sqrt(np.einsum("i,tij,j->t", w_ew, cov, w_ew))        # forecast (made at t) for day t+1
    score = np.full(T, np.nan)
    score[1:] = np.abs(R[1:] @ w_ew) / np.maximum(sig_ew[:-1], 1e-8)  # realised on day t, forecast made at t-1
    q = pd.Series(score).rolling(CONF_WINDOW, min_periods=30).quantile(CONF_LEVEL).bfill().fillna(2.0).values

    Tn = close.shape[0]
    if extra is None:
        extra = np.zeros((Tn, N), dtype=np.float32)
    assert extra.shape == (Tn, N)
    return Panel(
        dates=idx.values.astype("datetime64[D]"),
        tickers=list(tickers),
        open=np.asarray(open_, dtype=np.float64),
        close=np.asarray(close_, dtype=np.float64),
        volume=np.asarray(volume_, dtype=np.float64),
        feats=feats, mkt=mkt,
        mkt_vol=vol20.fillna(0.2).values.astype(np.float64),
        extra=extra.astype(np.float32),
        cov=cov, conf_q=q.astype(np.float64),
    )


def load_panel(csv_path: str, extra: np.ndarray | None = None) -> Panel:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    tickers = sorted(df["tic"].unique())
    piv = {c: df.pivot(index="date", columns="tic", values=c).sort_index()[tickers]
           for c in ["open", "high", "low", "close", "volume"]}
    for k, v in piv.items():
        if v.isna().any().any():
            raise ValueError(f"missing values in '{k}': panel must be complete (clean the data first)")
    return build_panel(piv["close"].index, tickers, piv["open"].values, piv["high"].values, piv["low"].values,
                       piv["close"].values, piv["volume"].values, extra)
