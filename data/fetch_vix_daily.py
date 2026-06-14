"""Fetch VIX daily OHLC from CBOE into data/vix_daily.csv.

CBOE publishes the full ^VIX historical series back to 1990 as a
no-auth CSV at the URL below. (Stooq now requires an API key; FRED
intermittently times out from this location.)

The NH-HMM transition covariate is VIX_t, used by sweep.py. The
autoresearch iteration-2 window is 2017-08-17 → 2025-12-31; we fetch
the full CBOE series and trim. VIX is published only on US trading
days, so the harness forward-fills to crypto-calendar (7-day-a-week)
coverage.
"""

import os
import sys
import io
import requests
import pandas as pd

CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vix_daily.csv")

WINDOW_START = pd.Timestamp("2017-08-17")
WINDOW_END = pd.Timestamp("2025-12-31")


def main():
    print(f"Fetching ^VIX daily from CBOE")
    resp = requests.get(CBOE_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  full series: {len(df)} rows, {df['date'].min().date()} → {df['date'].max().date()}")

    mask = (df["date"] >= WINDOW_START) & (df["date"] <= WINDOW_END)
    df = df.loc[mask, ["date", "open", "high", "low", "close"]].reset_index(drop=True)
    print(f"  trimmed to {WINDOW_START.date()} → {WINDOW_END.date()}: {len(df)} rows")

    df["date"] = df["date"].dt.date
    df.to_csv(OUT_PATH, index=False)
    print(f"  wrote {OUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
