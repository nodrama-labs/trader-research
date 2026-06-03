"""Fetch BTCUSDT daily candles from Binance into data/btcusdt_daily.csv.

Phase 0 needs 2021-06 → 2023-06 (~2 years wrapping the 2022 bear).
Later phases extend to 2017-08 → 2025-12. Keeping the fetch window
parametric here so Day 2 can widen without rewriting.
"""

import os
import sys
import time
import requests
import pandas as pd

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1d"

# Phase 0 window: 2021-06-01 → 2023-06-30 inclusive.
START_MS = int(pd.Timestamp("2021-06-01", tz="UTC").timestamp() * 1000)
END_MS = int(pd.Timestamp("2023-06-30", tz="UTC").timestamp() * 1000)

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "btcusdt_daily.csv")

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_base_volume", "taker_quote_volume", "ignore",
]


def fetch_klines(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Page through klines 1000 at a time until we cover the window."""
    rows = []
    cursor = start_ms
    while cursor <= end_ms:
        resp = requests.get(
            BINANCE_KLINES_URL,
            params={
                "symbol": SYMBOL,
                "interval": INTERVAL,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        # advance cursor past last candle (open_time + 1 day in ms)
        cursor = last_open + 24 * 3600 * 1000
        if len(batch) < 1000:
            break
        time.sleep(0.1)  # be polite to the public endpoint
    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = df["open_time"].astype("int64")
    df["close_time"] = df["close_time"].astype("int64")
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.date
    return df[["date", "open_time", "open", "high", "low", "close", "volume"]]


def main():
    print(f"Fetching {SYMBOL} {INTERVAL} candles "
          f"{pd.Timestamp(START_MS, unit='ms', tz='UTC').date()} → "
          f"{pd.Timestamp(END_MS, unit='ms', tz='UTC').date()}")
    df = fetch_klines(START_MS, END_MS)
    print(f"  got {len(df)} candles, {df['date'].min()} → {df['date'].max()}")
    expected = (pd.Timestamp(END_MS, unit="ms", tz="UTC")
                - pd.Timestamp(START_MS, unit="ms", tz="UTC")).days + 1
    print(f"  expected ~{expected} daily candles, gap = {expected - len(df)}")
    df.to_csv(OUT_PATH, index=False)
    print(f"  wrote {OUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
