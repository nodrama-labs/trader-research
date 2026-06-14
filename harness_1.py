"""Untouchable evaluation harness for the trader-research autoresearch loop.

This file is the contract surface. Do not modify it from inside the
autoresearch loop. It owns:

  * The data loader and split definition (causal walk-forward over the bear
    portfolio CSV).
  * Portfolio bookkeeping primitives (cash + held positions + snapshots).
  * The backtest skeleton ``gem_backtest`` that walks day-by-day and dispatches
    fit / portfolio decisions to caller-supplied callables.
  * The scoring rule ``ensemble_score`` and the metric aggregator that feeds
    it.
  * The ``evaluate`` harness that runs base + 1.5x fee-stress backtests and
    returns the scalar ``ensemble_score``.

External constraints that do not move with the strategy:

  * ``FEE_RATE`` -- 10 bps exchange + 20 bps slippage = 30 bps round-trip.
  * ``FEE_STRESS_MULTIPLIER`` -- 1.5x multiplier applied to fees in the
    stress-test pass that drives the hard-rejection gate.
  * ``INITIAL_CAPITAL`` -- portfolio sizing, not a strategy parameter.

``evaluate`` overrides whatever the caller's params struct has for
``fee_rate`` and ``initial_capital`` with the canonical values above; this
keeps the autoresearch loop from accidentally optimizing on fees or capital.

Anything the autoresearch loop wants to tune lives in ``sweep.py``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(REPO_ROOT, "data", "bear_portfolio_candles.csv")
RESULTS_PATH = os.path.join(REPO_ROOT, "results", "bear_sweep_results_1.tsv")


# ---------------------------------------------------------------------------
# Fixed constants (do not modify from sweep.py)
# ---------------------------------------------------------------------------

FEE_RATE = 0.003              # 10 bps exchange + 20 bps slippage
FEE_STRESS_MULTIPLIER = 1.5   # multiplier in the calmar hard-rejection gate
INITIAL_CAPITAL = 10_000.0    # USD-equivalent starting bankroll


# ---------------------------------------------------------------------------
# Portfolio bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class HeldPosition:
    quantity: float
    entry_price: float


@dataclass
class DaySnapshot:
    total_value: float
    positions: dict


class PortfolioState:
    def __init__(self, initial_capital: float):
        self.cash = initial_capital
        self.positions: dict[str, HeldPosition] = {}
        self.peak_value = initial_capital
        self.history: list[DaySnapshot] = []

    def buy(self, token: str, price: float, amount: float, fee_rate: float):
        fee = amount * fee_rate
        net_amount = amount - fee
        quantity = net_amount / price
        self.cash -= amount
        if token in self.positions:
            self.positions[token].quantity += quantity
        else:
            self.positions[token] = HeldPosition(quantity=quantity, entry_price=price)

    def sell(self, token: str, price: float, quantity: float, fee_rate: float):
        gross = quantity * price
        fee = gross * fee_rate
        self.cash += gross - fee
        if token in self.positions:
            self.positions[token].quantity -= quantity
            if self.positions[token].quantity <= 1e-12:
                del self.positions[token]

    def total_value(self, prices: dict[str, float]) -> float:
        pos_value = sum(
            pos.quantity * prices.get(token, pos.entry_price)
            for token, pos in self.positions.items()
        )
        return self.cash + pos_value

    def actual_weights(self, prices: dict[str, float]) -> dict[str, float]:
        total = self.total_value(prices)
        if total <= 0:
            return {}
        return {
            token: (pos.quantity * prices.get(token, pos.entry_price)) / total
            for token, pos in self.positions.items()
        }

    def record_snapshot(self, prices: dict[str, float]):
        total = self.total_value(prices)
        self.peak_value = max(self.peak_value, total)
        pos_values = {
            token: pos.quantity * prices.get(token, pos.entry_price)
            for token, pos in self.positions.items()
        }
        self.history.append(DaySnapshot(total_value=total, positions=pos_values))


def should_rebalance(
    current_holdings: dict[str, float],
    momentums: dict[str, float],
    entry_threshold: float = 0.0,
    exit_threshold: float = 0.0,
) -> bool:
    for token, momentum in momentums.items():
        is_held = token in current_holdings
        if not is_held and momentum > entry_threshold:
            return True
        if is_held and momentum < exit_threshold:
            return True
    return False


def is_in_cooldown(
    current_day: int, last_rebalance_day: Optional[int], cooldown_days: int
) -> bool:
    if last_rebalance_day is None:
        return False
    return cooldown_days > 0 and (current_day - last_rebalance_day) < cooldown_days


def rebalance_to_targets(
    state: PortfolioState,
    target_weights: dict[str, float],
    prices: dict[str, float],
    fee_rate: float,
):
    total_value = state.total_value(prices)
    min_trade_value = total_value * 0.001

    for token in list(state.positions.keys()):
        price = prices.get(token, state.positions[token].entry_price)
        current_qty = state.positions[token].quantity
        current_value = current_qty * price
        target_value = target_weights.get(token, 0.0) * total_value
        sell_value = current_value - target_value
        if sell_value > min_trade_value:
            sell_qty = sell_value / price
            state.sell(token, price, sell_qty, fee_rate)

    total_value = state.total_value(prices)

    for token, target_weight in target_weights.items():
        price = prices.get(token)
        if price is None:
            continue
        target_value = total_value * target_weight
        current_value = 0.0
        if token in state.positions:
            current_value = state.positions[token].quantity * price
        buy_amount = target_value - current_value
        if buy_amount > min_trade_value:
            buy_amount = min(buy_amount, state.cash)
            if buy_amount > 0:
                state.buy(token, price, buy_amount, fee_rate)


# ---------------------------------------------------------------------------
# Backtest skeleton
# ---------------------------------------------------------------------------


def gem_backtest(
    params,
    candles_by_token: dict[str, pd.DataFrame],
    fit_fn: Callable,
    portfolio_fn: Callable,
) -> dict:
    """Day-by-day causal walk-forward.

    ``fit_fn(candles_slice, atr_window, r2_exponent)`` -> Position-like with
    ``.momentum``. ``portfolio_fn(positions, params)`` -> list of
    Position-like with ``.token`` and ``.weight`` summing to 1.

    The backtest harness is fixed: it reads ``params.fit_window``,
    ``params.atr_window``, ``params.r2_exponent``, ``params.rebalance_cooldown``,
    ``params.fee_rate``, ``params.initial_capital``. The agent's GemParams
    must keep these field names; their values come from ``evaluate`` below.
    """
    eligible = candles_by_token

    all_timestamps = sorted(
        set(ts for c in eligible.values() for ts in c["timestamp"].values)
    )
    if not all_timestamps:
        return _empty_metrics()

    price_index = {
        token: dict(zip(c["timestamp"].values, c["close"].values))
        for token, c in eligible.items()
    }

    state = PortfolioState(params.initial_capital)
    rebalance_count = 0
    last_rebalance_day = None
    target_weights: dict[str, float] = {}

    for day_idx, timestamp in enumerate(all_timestamps):
        prices = {
            token: ts_map[timestamp]
            for token, ts_map in price_index.items()
            if timestamp in ts_map
        }

        fit_results = {}
        for token, candles in eligible.items():
            up_to = candles[candles["timestamp"] <= timestamp]
            if params.fit_window > 0 and len(up_to) > params.fit_window:
                up_to = up_to.iloc[-params.fit_window :]
            if len(up_to) >= 2:
                pos = fit_fn(up_to, params.atr_window, params.r2_exponent)
                if pos is not None:
                    fit_results[token] = pos

        need_rebalance = False
        if day_idx == 0:
            need_rebalance = True
        elif not is_in_cooldown(day_idx, last_rebalance_day, params.rebalance_cooldown):
            momentums = {t: p.momentum for t, p in fit_results.items()}
            holdings = {t: p.quantity for t, p in state.positions.items()}
            need_rebalance = should_rebalance(holdings, momentums)

        if need_rebalance:
            candidates = list(fit_results.values())
            portfolio = portfolio_fn(candidates, params)
            target_weights = {p.token: p.weight for p in portfolio}
            rebalance_to_targets(state, target_weights, prices, params.fee_rate)
            rebalance_count += 1
            last_rebalance_day = day_idx

        state.record_snapshot(prices)

    return _compute_metrics(state, rebalance_count)


# ---------------------------------------------------------------------------
# Metrics + scoring (do not modify -- the denominator in every comparison)
# ---------------------------------------------------------------------------


def _empty_metrics() -> dict:
    return {
        "total_return_pct": 0.0,
        "annualized_return_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "calmar_ratio": 0.0,
        "rebalance_count": 0,
        "hhi": 1.0,
        "final_value": 0.0,
    }


def _compute_metrics(state: PortfolioState, rebalance_count: int) -> dict:
    history = state.history
    if len(history) < 2:
        return _empty_metrics()

    values = np.array([s.total_value for s in history])
    initial_value = values[0]
    final_value = values[-1]
    total_return = (final_value - initial_value) / initial_value

    days = len(history)
    years = days / 365.0
    annualized_return = (1.0 + total_return) ** (1.0 / max(years, 1e-9)) - 1.0

    peak = np.maximum.accumulate(values)
    drawdowns = (peak - values) / peak
    max_drawdown = float(np.max(drawdowns))

    daily_returns = np.diff(values) / values[:-1]
    mean_r = float(np.mean(daily_returns))
    std_r = float(np.std(daily_returns))
    sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 1e-12 else 0.0

    downside = daily_returns[daily_returns < 0]
    downside_dev = (
        np.sqrt(np.sum(downside ** 2) / len(daily_returns))
        if len(daily_returns) > 0
        else 0.0
    )
    sortino = (mean_r / downside_dev * np.sqrt(252)) if downside_dev > 1e-12 else 0.0
    calmar = (annualized_return / max_drawdown) if max_drawdown > 1e-12 else 0.0

    last_snap = history[-1]
    last_total = last_snap.total_value if last_snap.total_value > 0 else 1.0
    weights = [v / last_total for v in last_snap.positions.values()]
    cash_weight = 1.0 - sum(weights)
    if cash_weight > 1e-9:
        weights.append(cash_weight)
    hhi = float(sum(w * w for w in weights)) if weights else 1.0

    return {
        "total_return_pct": total_return * 100,
        "annualized_return_pct": annualized_return * 100,
        "max_drawdown_pct": -max_drawdown * 100,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "rebalance_count": rebalance_count,
        "hhi": hhi,
        "final_value": final_value,
    }


def ensemble_score(base: dict, stress: dict) -> float:
    """Single scalar metric: return * dd_dampener * div_bonus, with hard gates.

    Hard rejection (returns -inf) on either:
      * annualized return below -50%
      * stress-test calmar ratio is negative
    """
    annualized = base["annualized_return_pct"] / 100.0
    if annualized < -0.50:
        return float("-inf")
    if stress["calmar_ratio"] < 0.0:
        return float("-inf")

    dd = abs(base["max_drawdown_pct"]) / 100.0
    dampener = 1.0 / (1.0 + max(0.0, dd - 0.15)) ** 2
    div_bonus = 1.0 + 0.1 * (1.0 - base["hhi"])
    return annualized * dampener * div_bonus


# ---------------------------------------------------------------------------
# Evaluate harness -- the loop's only currency
# ---------------------------------------------------------------------------


def evaluate(
    params,
    candles_by_token: dict[str, pd.DataFrame],
    fit_fn: Callable,
    portfolio_fn: Callable,
) -> tuple[float, dict, dict]:
    """Run base + fee-stress backtests and return (score, base_metrics, stress_metrics).

    Pins ``fee_rate`` and ``initial_capital`` to the canonical values; whatever
    the caller's params held for these fields is overridden. This is what
    keeps the autoresearch loop honest.
    """
    base_params = replace(params, fee_rate=FEE_RATE, initial_capital=INITIAL_CAPITAL)
    stress_params = replace(
        params,
        fee_rate=FEE_RATE * FEE_STRESS_MULTIPLIER,
        initial_capital=INITIAL_CAPITAL,
    )
    base = gem_backtest(base_params, candles_by_token, fit_fn, portfolio_fn)
    stress = gem_backtest(stress_params, candles_by_token, fit_fn, portfolio_fn)
    score = ensemble_score(base, stress)
    return score, base, stress


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_candles(path: str = DATA_PATH) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(path)
    return {
        token: g.sort_values("timestamp").reset_index(drop=True)
        for token, g in df.groupby("token")
    }
