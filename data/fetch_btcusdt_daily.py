"""Fetch BTCUSDT daily candles from Binance into data/btcusdt_daily.csv.

The fetch window is parametric. Defaults cover the full dashboard range
(2017-08-17 → present); pass --start / --end to override. Binance caps a
single klines response at 1000 candles, so we page with a startTime cursor.

    python data/fetch_btcusdt_daily.py                       # full range
    python data/fetch_btcusdt_daily.py --start 2021-06-01 --end 2023-06-30
"""

import argparse
import os
import sys
import time

import pandas as pd
import requests

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1d"

# Full dashboard range. BTCUSDT spot on Binance starts 2017-08-17.
DEFAULT_START = "2017-08-17"

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "btcusdt_daily.csv")

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_base_volume", "taker_quote_volume", "ignore",
]


def fetch_klines(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Page through klines 1000 at a time until we cover [start_ms, end_ms]."""
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
    # de-dup on date (cursor overlap can repeat a boundary candle)
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return df[["date", "open_time", "open", "high", "low", "close", "volume"]]


def parse_args(argv):
    p = argparse.ArgumentParser(description="Fetch BTCUSDT daily candles from Binance.")
    p.add_argument("--start", default=DEFAULT_START,
                   help=f"UTC start date YYYY-MM-DD (default {DEFAULT_START})")
    p.add_argument("--end", default=None,
                   help="UTC end date YYYY-MM-DD (default: today)")
    p.add_argument("--out", default=OUT_PATH, help="output CSV path")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = (pd.Timestamp(args.end, tz="UTC") if args.end
              else pd.Timestamp.now(tz="UTC").normalize())
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)

    print(f"Fetching {SYMBOL} {INTERVAL} candles {start_ts.date()} → {end_ts.date()}")
    df = fetch_klines(start_ms, end_ms)
    print(f"  got {len(df)} candles, {df['date'].min()} → {df['date'].max()}")
    expected = (end_ts - start_ts).days + 1
    print(f"  expected ~{expected} daily candles, gap = {expected - len(df)}")
    df.to_csv(args.out, index=False)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
