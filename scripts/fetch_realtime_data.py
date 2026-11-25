"""Download and prepare continuous real-time dataset extending through current date.

Combines historical top30_liquid.csv data (2009 to 2023-05) with fresh live data (2023-05 to 2026-11)
to create an extended dataset for forward out-of-sample stress testing.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]

TICKERS = ['AAPL', 'AMZN', 'BA', 'BAC', 'BKNG', 'CSCO', 'CVX', 'DIS', 'FCX', 'GE', 
           'GILD', 'GOOGL', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MSFT', 'MU', 
           'NFLX', 'ORCL', 'PFE', 'PG', 'QCOM', 'SLB', 'V', 'WFC', 'WMT', 'XOM']

def fetch_and_build():
    print(f"1. Fetching fresh market data for {len(TICKERS)} tickers from Yahoo Finance...")
    # Fetch from 2009 to present to get unified, clean continuous historical adjustments
    raw = yf.download(TICKERS, start="2009-01-01", auto_adjust=True, progress=True)
    
    # Reshape multi-index dataframe to standard format: date, tic, open, high, low, close, volume
    dfs = []
    dates = raw.index
    for tic in TICKERS:
        df_tic = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "tic": tic,
            "open": raw["Open"][tic].values,
            "high": raw["High"][tic].values,
            "low": raw["Low"][tic].values,
            "close": raw["Close"][tic].values,
            "volume": raw["Volume"][tic].values
        })
        dfs.append(df_tic)
        
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.dropna(subset=["close"])
    full_df = full_df.sort_values(["date", "tic"]).reset_index(drop=True)
    
    # Validate date ranges
    min_date = full_df["date"].min()
    max_date = full_df["date"].max()
    print(f"\n2. Continuous dataset ready: {min_date} to {max_date} | Total rows: {len(full_df):,}")
    
    out_path = ROOT / "data/top30_realtime_extended.csv"
    full_df.to_csv(out_path, index=False)
    print(f"3. Saved to: {out_path}")
    return out_path

if __name__ == "__main__":
    fetch_and_build()
