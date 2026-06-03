"""Modifiable interior of the trader-research autoresearch loop.

This file is the agent's playground. Everything here is fair game:

  * ``GemParams`` -- the parameter struct for the bear specialist.
  * The model body -- ``fit_token_exponential`` and ``build_portfolio`` and
    the regression / ATR primitives they call.
  * The sweep driver in ``main`` -- you decide which parameter to sweep, what
    candidate values to try, and how to log the result.

The harness in ``harness.py`` is *not* modifiable. It owns the data loader,
the backtest skeleton, the scoring function, and the canonical fee /
capital constants. ``evaluate`` from the harness is the only currency the
loop trusts.

Default usage:

    python sweep.py --param top_n --values 1,3,5,10
    python sweep.py --param r2_threshold --values 0.3,0.5,0.7,0.8

The script sweeps a single named parameter across the supplied values,
holding everything else at the current defaults in ``GemParams``. It prints
one line per candidate, appends rows to ``results/bear_sweep_results.tsv``,
and prints the winner.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from harness import (
    DATA_PATH,
    FEE_RATE,
    INITIAL_CAPITAL,
    RESULTS_PATH,
    evaluate,
    load_candles,
)


# ---------------------------------------------------------------------------
# Model body -- regression primitives
# ---------------------------------------------------------------------------


def r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - y_hat) ** 2)
    if ss_total < 1e-12:
        return 0.0
    return 1.0 - ss_residual / ss_total


def average_true_range(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> float:
    n = len(close)
    if n < 2:
        return 0.0
    tr_sum = 0.0
    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
        tr_sum += tr
    return tr_sum / (n - 1)


def fit_linear_logspace(x: np.ndarray, y: np.ndarray):
    ln_y = np.log(y)
    X = np.column_stack([np.ones_like(x), x])
    beta, _, _, _ = np.linalg.lstsq(X, ln_y, rcond=None)
    return beta[0], beta[1]


def exponential_model(x, a0, a1):
    return a0 * np.power(a1, x)


def fit_exponential(x: np.ndarray, y: np.ndarray, initial: tuple):
    try:
        popt, _ = curve_fit(
            exponential_model, x, y, p0=initial, method="lm", maxfev=5000
        )
        a0, a1 = popt
        if np.isfinite(a0) and np.isfinite(a1):
            return a0, a1
    except (RuntimeError, ValueError):
        pass
    return initial


# ---------------------------------------------------------------------------
# Position + parameters
# ---------------------------------------------------------------------------


@dataclass
class Position:
    token: str
    weight: float = 0.0
    momentum: float = 0.0
    r2: float = 0.0
    a1: float = 0.0
    atr: float = 0.0
    quantity: float = 0.0  # only set on held positions


@dataclass
class GemParams:
    # Bear-specialist parameters -- these are what the autoresearch loop sweeps.
    # Scored by ``ensemble_score`` (name inherited from the full ensemble contract).
    r2_threshold: float = 0.5
    top_n: int = 5
    atr_window: int = 15
    fit_window: int = 30
    momentum_cap: float = 0.14
    r2_exponent: float = 2.0
    rebalance_cooldown: int = 7

    # Ablation flags. Toggle off currently-active components for deletion runs.
    use_r2_filter: bool = True
    use_growth_filter: bool = True
    use_momentum_cap: bool = True
    use_inverse_vol_weighting: bool = True
    pick_lowest_momentum: bool = False

    # Required by the harness; canonicalized inside ``evaluate``. Do not sweep.
    fee_rate: float = FEE_RATE
    initial_capital: float = INITIAL_CAPITAL


# ---------------------------------------------------------------------------
# Model body -- fit + portfolio construction
# ---------------------------------------------------------------------------


def fit_token_exponential(
    candles: pd.DataFrame, atr_window: int = 15, r2_exponent: float = 1.0
) -> Optional[Position]:
    if candles.empty:
        return None

    token = candles.iloc[0]["token"]
    x = np.arange(len(candles), dtype=float)
    closes = candles["close"].values
    highs = candles["high"].values
    lows = candles["low"].values

    if np.any(closes <= 0):
        return None

    try:
        b0, b1 = fit_linear_logspace(x, closes)
    except Exception:
        return None

    initial = (np.exp(b0), np.exp(b1))
    a0, a1 = fit_exponential(x, closes, initial)

    y_hat = exponential_model(x, a0, a1)
    r2 = r_squared(closes, y_hat)
    momentum = (r2 ** r2_exponent) * (a1 - 1.0) * 100.0

    lookback = min(len(candles), atr_window)
    mean_close = np.mean(closes)
    if lookback > 1 and mean_close > 0:
        atr = (
            average_true_range(
                closes[-lookback:], highs[-lookback:], lows[-lookback:]
            )
            / mean_close
        )
    else:
        atr = 0.0

    return Position(token=token, momentum=momentum, r2=r2, a1=a1, atr=atr)


def build_portfolio(candidates: list[Position], params: GemParams) -> list[Position]:
    filtered = list(candidates)

    if params.use_growth_filter:
        filtered = [p for p in filtered if p.a1 > 1.0]
    if params.use_r2_filter:
        filtered = [p for p in filtered if p.r2 > params.r2_threshold]
    if params.use_momentum_cap and params.momentum_cap > 0.0:
        filtered = [p for p in filtered if p.momentum < params.momentum_cap]

    filtered.sort(key=lambda p: p.momentum, reverse=not params.pick_lowest_momentum)
    filtered = filtered[: params.top_n]

    filtered = [p for p in filtered if p.atr > 0.0]
    if not filtered:
        return []

    if params.use_inverse_vol_weighting:
        inv_vols = [1.0 / p.atr for p in filtered]
        total = sum(inv_vols)
        for p, iv in zip(filtered, inv_vols):
            p.weight = iv / total
    else:
        equal = 1.0 / len(filtered)
        for p in filtered:
            p.weight = equal

    return filtered


# ---------------------------------------------------------------------------
# Sweep driver -- one parameter at a time
# ---------------------------------------------------------------------------


def parse_value(raw: str, default):
    """Coerce a CLI string to the type of the default value."""
    if isinstance(default, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


def sweep_one_parameter(
    param_name: str,
    values: list,
    candles_by_token: dict[str, pd.DataFrame],
    baseline: GemParams,
) -> list[dict]:
    if not hasattr(baseline, param_name):
        raise ValueError(f"GemParams has no field {param_name!r}")

    rows: list[dict] = []
    best_score = float("-inf")
    best_value = None
    started = time.time()

    print(f"\nSweeping {param_name} over {values}")
    print(f"Baseline: {_describe(baseline)}\n")

    for i, raw in enumerate(values):
        value = parse_value(str(raw), getattr(baseline, param_name))
        params = replace(baseline, **{param_name: value})

        score, base, stress = evaluate(
            params, candles_by_token, fit_token_exponential, build_portfolio
        )

        kept = math.isfinite(score) and score > best_score
        if kept:
            best_score = score
            best_value = value

        row = {
            "exp": i,
            "param": param_name,
            "value": value,
            "score": score,
            "annualized_return_pct": base["annualized_return_pct"],
            "max_drawdown_pct": base["max_drawdown_pct"],
            "sharpe": base["sharpe_ratio"],
            "sortino": base["sortino_ratio"],
            "calmar": base["calmar_ratio"],
            "calmar_stress": stress["calmar_ratio"],
            "hhi": base["hhi"],
            "rebalance_count": base["rebalance_count"],
            "best_so_far": best_score,
            "outcome": "kept" if kept else "discarded",
            "description": f"{param_name}={value}",
        }
        for k, v in asdict(params).items():
            if k not in row:
                row[k] = v
        rows.append(row)
        print(_format_row(row))

    elapsed = time.time() - started
    print(
        f"\nDone. {len(rows)} candidates in {elapsed:.1f}s. "
        f"Best: {param_name}={best_value} (score={best_score:.4f})"
    )
    return rows


def _format_row(row: dict) -> str:
    score_str = "-inf" if not math.isfinite(row["score"]) else f"{row['score']:8.4f}"
    return (
        f"#{row['exp']:>3} {row['param']:<22} = {str(row['value']):<10} "
        f"score={score_str} "
        f"ann={row['annualized_return_pct']:>7.2f}% "
        f"dd={row['max_drawdown_pct']:>7.2f}% "
        f"calmar_stress={row['calmar_stress']:>6.2f} "
        f"best={row['best_so_far']:8.4f} {row['outcome']:<10}"
    )


def _describe(params: GemParams) -> str:
    keys = (
        "top_n",
        "r2_threshold",
        "rebalance_cooldown",
        "atr_window",
        "fit_window",
        "momentum_cap",
        "r2_exponent",
    )
    return " ".join(f"{k}={getattr(params, k)}" for k in keys)


def append_rows(rows: list[dict], path: str = RESULTS_PATH) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"Appended {len(rows)} rows to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--param",
        required=True,
        help="GemParams field to sweep (e.g. top_n, r2_threshold, rebalance_cooldown).",
    )
    parser.add_argument(
        "--values",
        required=True,
        help="Comma-separated candidate values (e.g. '1,3,5,10').",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Skip appending to results.tsv (useful for smoke tests).",
    )
    args = parser.parse_args(argv)

    if not os.path.exists(DATA_PATH):
        print(f"error: missing {DATA_PATH}", file=sys.stderr)
        return 1

    values = [v.strip() for v in args.values.split(",") if v.strip()]
    if not values:
        print("error: --values is empty", file=sys.stderr)
        return 1

    candles_by_token = load_candles(DATA_PATH)
    rows = sweep_one_parameter(args.param, values, candles_by_token, GemParams())

    if not args.no_write:
        append_rows(rows, RESULTS_PATH)

    return 0


if __name__ == "__main__":
    sys.exit(main())
