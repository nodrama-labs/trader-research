"""Read-only contract surface for the HMM regime-detection autoresearch loop.

Responsibilities (per program.md):

- Load BTC daily candles, VIX daily, consensus regime labels.
- Compute the canonical `regime_score` (smooth severity-weighted Brier
  `Q · R` over the 5 consensus periods; see
  docs/plans/2026-06-14-regime-score-fit-statistic-design.md).
- Drive a causal walk-forward evaluation: fit the model on `[0..t-1]`, score
  posterior at `t`, slide forward with a refit cadence.
- Append a single TSV row per `evaluate()` call with the canonical schema.

Anything in this file is fixed. The optimiser may not edit it; that's the
whole point of the contract surface — `regime_score` cannot be rewritten by
the loop to flatter the loop's own output.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BTC_PATH = os.path.join(REPO_ROOT, "data", "btcusdt_daily.csv")
VIX_PATH = os.path.join(REPO_ROOT, "data", "vix_daily.csv")
LABELS_PATH = os.path.join(REPO_ROOT, "data", "consensus_labels.tsv")
RESULTS_PATH = os.path.join(REPO_ROOT, "results", "regime_sweep_results.tsv")


# ---------------------------------------------------------------------------
# Constants — scoring / evaluation discipline (NOT sweep-able)
# ---------------------------------------------------------------------------

# Rolling drawdown warmup. Days where the rolling-200 max isn't fully populated
# are excluded from scoring regardless of label (the drawdown feature itself
# uses a shorter rolling window during these days but the autoresearch baseline
# fixes the feature window at 200, so any feature requiring it is uninformative
# before day 200).
WARMUP_DAYS = 200

# Refit cadence in days. Paper 2's "single hidden state typically persists
# 30-60 days" suggests a refit cadence inside that range gives the fit a
# chance to absorb regime shifts without burning compute on near-duplicate
# refits. Monthly (30 days) sits at the lower end of that range and gives a
# ~10x compute saving over weekly. Not sweep-able to keep cross-experiment
# comparisons honest.
REFIT_CADENCE_DAYS = 30

# Severity-weighted Brier role weights (mirror program.md §"Scoring rule").
# Each scored day's loss weights the three label classes by their ROLE relative
# to the period's true label: the correct class, the cautious middle (ranging),
# and the opposite extreme (the catastrophe — e.g. bull-mass during a bear).
# Not sweep-able: the loop must not be able to soften its own penalty.
BRIER_HIT = 1.0        # weight on the CORRECT class (calibration / "hit" term)
BRIER_RANGING = 0.25   # weight on RANGING (abstention costs a little, not a lot)
BRIER_OPP = 8.0        # weight on the OPPOSITE extreme (catastrophe / mislabel);
                       # a half-mass wrong-way bet then scores ~1.8x worse than
                       # staying flat (all-ranging).

# Reliability decay. The score is multiplied by exp(-NONCONV_LAMBDA * rate);
# 7.0 makes a 10% non-convergence rate ~halve the score (exp(-0.7) ≈ 0.50),
# replacing the old >10% -> -inf cliff with a smooth penalty.
NONCONV_LAMBDA = 7.0

# The five consensus periods named in the grant proposal (§4.1). Macro-average
# over these — and ONLY these — defines `regime_score`. The four ranging windows
# in the labels file mark "neither bear nor bull" days but do not contribute to
# the score directly.
CONSENSUS_PERIODS = (
    "2018_bear",
    "2020q1_covid",
    "bull_2020_2021",
    "2022_bear",
    "2024_etf_bull",
)

# Canonical label values. The model's posterior is a (T, 3) array with these
# columns in this order; sweep.py-side models are responsible for mapping their
# K-state output into this 3-class label space (μ-sort for K=3, custom rules for
# K=4+).
LABEL_VALUES = ("bear", "ranging", "bull")
LABEL_TO_IDX = {lab: i for i, lab in enumerate(LABEL_VALUES)}

# The "opposite extreme" of each directional label — the catastrophic confusion
# the Brier `BRIER_OPP` term punishes. Ranging has no opposite (it is the middle).
OPPOSITE_LABEL = {"bear": "bull", "bull": "bear"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_btc() -> pd.DataFrame:
    """Return BTC daily with `date`, `close`, `log_return`, `log_price`."""
    df = pd.read_csv(BTC_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["log_price"] = np.log(df["close"])
    df["log_return"] = df["log_price"].diff()
    return df[["date", "open", "high", "low", "close", "log_return", "log_price"]]


def load_vix() -> pd.DataFrame:
    """Return VIX daily with `date`, `vix_close`. US trading days only — caller
    forward-fills to the BTC calendar via `load_joined()`."""
    df = pd.read_csv(VIX_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df.rename(columns={"close": "vix_close"})[["date", "vix_close"]]


def load_labels() -> pd.DataFrame:
    """Return the per-day labels expanded from the period table.

    Output: `date`, `label` (one of bear/ranging/bull), `period_id` (one of the
    five CONSENSUS_PERIODS for labelled bear/bull periods, or "ranging_<n>" for
    intermediate ranging windows).
    """
    periods = pd.read_csv(
        LABELS_PATH, sep="\t", header=None,
        names=["start_date", "end_date", "label", "justification"],
    )
    periods["start_date"] = pd.to_datetime(periods["start_date"])
    periods["end_date"] = pd.to_datetime(periods["end_date"])
    periods = periods.sort_values("start_date").reset_index(drop=True)

    # Tag each non-ranging period with a stable id from CONSENSUS_PERIODS, in
    # chronological order. Ranging windows get ranging_<n> ids.
    consensus_iter = iter(CONSENSUS_PERIODS)
    ranging_counter = 0
    period_ids = []
    for _, row in periods.iterrows():
        if row["label"] == "ranging":
            period_ids.append(f"ranging_{ranging_counter}")
            ranging_counter += 1
        else:
            try:
                period_ids.append(next(consensus_iter))
            except StopIteration:
                raise ValueError(
                    f"More non-ranging periods than {len(CONSENSUS_PERIODS)} "
                    "consensus periods named in the grant; check the labels file."
                )
    periods["period_id"] = period_ids

    rows = []
    for _, row in periods.iterrows():
        for day in pd.date_range(row["start_date"], row["end_date"], freq="D"):
            rows.append({"date": day, "label": row["label"], "period_id": row["period_id"]})
    return pd.DataFrame(rows)


def load_joined() -> pd.DataFrame:
    """Join BTC + VIX (forward-filled to crypto calendar) + labels on `date`.

    Rows where the join produces no label (e.g. days outside the period
    coverage) are dropped. Rows where VIX is unavailable for the very first
    days are kept with NaN VIX — the harness leaves it to sweep.py how to
    handle that (paper-2-style: shift VIX forward by 1 to use yesterday's
    close, which also forward-fills weekends).
    """
    btc = load_btc()
    vix = load_vix()
    labels = load_labels()

    # Forward-fill VIX onto BTC's 7-day calendar
    vix = vix.sort_values("date").set_index("date").asfreq("D").ffill().reset_index()

    out = btc.merge(vix, on="date", how="left")
    out = out.merge(labels, on="date", how="inner")  # inner join: drop unlabelled days
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model interface (a duck-typing contract; sweep.py implements concrete models)
# ---------------------------------------------------------------------------

class FittedModelProtocol:
    """A fitted HMM the harness can call `forward_posterior` on.

    Models in sweep.py construct an instance of (or anything with) this
    interface from a training slice of the data. The harness then calls
    `forward_posterior(data_up_to_t)` to get the posterior at every day up to
    `t`, and uses the row at `t` as the model's call for day `t`.

    The returned array has shape (T, 3) with columns in LABEL_VALUES order
    (bear, ranging, bull). For K > 3 fits sweep.py is responsible for
    aggregating to 3 columns (e.g. summing two ranging sub-states).
    """

    def forward_posterior(self, data: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


ModelFactory = Callable[[pd.DataFrame], FittedModelProtocol]
# A ModelFactory takes a training slice (a DataFrame with the same columns as
# `load_joined()` returns) and returns a fitted model implementing
# FittedModelProtocol.


# ---------------------------------------------------------------------------
# Causal walk-forward driver
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    posteriors: np.ndarray          # (T_eval, 3) — posterior over labels on each scored day
    scored_dates: np.ndarray        # (T_eval,) — pd.Timestamp values
    scored_labels: np.ndarray       # (T_eval,) — string label per day
    scored_periods: np.ndarray      # (T_eval,) — period_id per day
    n_refits: int
    n_nonconvergence: int


def _log(msg: str) -> None:
    """Print with explicit flush so progress is visible in real time even when
    stdout is redirected to a file (the autoresearch loop runs everything via
    `> run.log 2>&1`)."""
    print(msg, flush=True)
    sys.stderr.flush()


def causal_walk_forward(
    data: pd.DataFrame,
    model_factory: ModelFactory,
    refit_cadence: int = REFIT_CADENCE_DAYS,
    warmup_days: int = WARMUP_DAYS,
    log_every: int = 5,
) -> WalkForwardResult:
    """Walk forward over `data`, refitting every `refit_cadence` days.

    At each evaluation day `t`, the posterior reported is `forward_posterior(
    data[0..t])[t]` using the most recent fit's parameters. The fit is updated
    every `refit_cadence` days using the expanding window `data[0..t_fit-1]`.

    Skips the first `warmup_days` rows (rolling-feature warmup).

    Logs progress every `log_every` refits with elapsed time + ETA so the
    research agent can gauge whether it's running, stuck, or about to finish.
    """
    T = len(data)
    eval_start = warmup_days
    if eval_start >= T:
        raise ValueError(f"warmup_days={warmup_days} exceeds data length {T}")

    posteriors = np.full((T - eval_start, 3), np.nan)
    n_refits = 0
    n_nonconvergence = 0

    current_model: Optional[FittedModelProtocol] = None
    next_refit_t = eval_start  # initial fit before the first scored day

    n_refits_expected = max(1, (T - eval_start) // refit_cadence + 1)
    t_start = time.time()
    _log(f"[harness] Walk-forward start: T={T}, eval_start={eval_start}, "
         f"expected ~{n_refits_expected} refits at cadence {refit_cadence}.")

    last_refit_secs = None

    for t in range(eval_start, T):
        if t >= next_refit_t:
            train = data.iloc[:t].copy()
            t_refit = time.time()
            try:
                current_model = model_factory(train)
                n_refits += 1
                last_refit_secs = time.time() - t_refit
            except Exception as e:
                n_nonconvergence += 1
                last_refit_secs = time.time() - t_refit
                _log(f"[harness] Refit at t={t} ({data['date'].iloc[t].date()}) "
                     f"FAILED: {type(e).__name__}: {e}")
            next_refit_t = t + refit_cadence

            # Progress log every `log_every` refits (counting both successful
            # and failed attempts — the absolute index drives cadence).
            attempts = n_refits + n_nonconvergence
            if attempts % log_every == 0 or t == eval_start:
                elapsed = time.time() - t_start
                avg_per_refit = elapsed / max(1, attempts)
                eta = avg_per_refit * max(0, n_refits_expected - attempts)
                _log(f"[harness] refit {attempts}/{n_refits_expected}  "
                     f"t={data['date'].iloc[t].date()}  "
                     f"elapsed={elapsed:.1f}s  "
                     f"last_refit={last_refit_secs:.2f}s  "
                     f"ETA~{eta:.0f}s  "
                     f"({n_nonconvergence} non-converged so far)")

        if current_model is not None:
            try:
                post = current_model.forward_posterior(data.iloc[: t + 1])
                posteriors[t - eval_start] = post[-1]
            except Exception as e:
                n_nonconvergence += 1
                _log(f"[harness] forward_posterior at t={t} "
                     f"({data['date'].iloc[t].date()}) FAILED: "
                     f"{type(e).__name__}: {e}")

    total_elapsed = time.time() - t_start
    _log(f"[harness] Walk-forward done: {n_refits} successful refits, "
         f"{n_nonconvergence} failures, total {total_elapsed:.1f}s "
         f"({total_elapsed / max(1, n_refits):.2f}s/refit avg).")

    scored_dates = data["date"].iloc[eval_start:].to_numpy()
    scored_labels = data["label"].iloc[eval_start:].to_numpy()
    scored_periods = data["period_id"].iloc[eval_start:].to_numpy()

    return WalkForwardResult(
        posteriors=posteriors,
        scored_dates=scored_dates,
        scored_labels=scored_labels,
        scored_periods=scored_periods,
        n_refits=n_refits,
        n_nonconvergence=n_nonconvergence,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class ScoreReport:
    regime_score: float                 # Q * R, in [0, 1] (higher is better)
    quality: float                      # Q: geomean over consensus periods of s_P
    reliability: float                  # R: exp(-NONCONV_LAMBDA * nonconv_rate)
    per_period: dict                    # period_id -> s_P (per-period score in [0,1])
    argmax_acc: float                   # over all scored, labelled days
    n_valid: int                        # rows contributing to the score
    n_nan_uniform: int                  # NaN posterior rows scored as uniform
    nonconvergence_rate: float


def regime_score(result: WalkForwardResult, n_refits_attempted: int = None) -> ScoreReport:
    """Compute the canonical smooth regime_score from a walk-forward.

        regime_score = Q * R

        s_P = 1 - mean_t( L_t ) / (BRIER_HIT + BRIER_OPP)   per consensus period
        L_t = BRIER_HIT     * (1 - p[correct])**2           severity-weighted
            + BRIER_RANGING *      p[ranging]**2             per-day Brier loss
            + BRIER_OPP     *      p[opposite]**2
        Q   = ( Π_P s_P )^(1/N)                              geomean (soft-min)
        R   = exp(-NONCONV_LAMBDA * nonconvergence_rate)     reliability factor

    Smooth and finite in [0, 1] — no -inf hard rejections. Scores only the five
    CONSENSUS_PERIODS (ranging windows excluded). NaN / numerically-failed
    posteriors are scored as the uniform "don't know" distribution rather than
    dropped, so a model cannot inflate its score by failing on hard days. See
    docs/plans/2026-06-14-regime-score-fit-statistic-design.md.
    """
    # NaN posteriors -> uniform (max-entropy), not dropped.
    posteriors = result.posteriors
    nan_rows = np.any(np.isnan(posteriors), axis=1)
    n_nan_uniform = int(nan_rows.sum())
    if n_nan_uniform:
        posteriors = posteriors.copy()
        posteriors[nan_rows] = 1.0 / len(LABEL_VALUES)

    denom = BRIER_HIT + BRIER_OPP  # max possible per-day loss

    per_period_scores = {}
    ranging_idx = LABEL_TO_IDX["ranging"]
    for pid in CONSENSUS_PERIODS:
        mask = result.scored_periods == pid
        if mask.sum() == 0:
            per_period_scores[pid] = np.nan
            continue
        label_str = result.scored_labels[mask][0]  # days in a period share the label
        correct_idx = LABEL_TO_IDX[label_str]
        opposite_idx = LABEL_TO_IDX[OPPOSITE_LABEL[label_str]]
        p = posteriors[mask]
        loss = (
            BRIER_HIT * (1.0 - p[:, correct_idx]) ** 2
            + BRIER_RANGING * p[:, ranging_idx] ** 2
            + BRIER_OPP * p[:, opposite_idx] ** 2
        )
        per_period_scores[pid] = float(1.0 - np.mean(loss) / denom)

    # Quality: geometric mean (soft minimum) over the present consensus periods.
    valid_periods = [v for v in per_period_scores.values() if not np.isnan(v)]
    if valid_periods:
        clipped = np.clip(valid_periods, 0.0, 1.0)
        quality = float(np.prod(clipped) ** (1.0 / len(clipped)))
    else:
        quality = 0.0

    # Reliability: smooth penalty on the non-convergence rate.
    if n_refits_attempted is None:
        n_refits_attempted = result.n_refits + result.n_nonconvergence
    nonconv_rate = (
        result.n_nonconvergence / n_refits_attempted if n_refits_attempted > 0 else 0.0
    )
    reliability = float(np.exp(-NONCONV_LAMBDA * nonconv_rate))

    regime_sc = quality * reliability

    # Argmax accuracy (sanity diagnostic, all scored days):
    if len(posteriors) > 0:
        argmax_pred = np.argmax(posteriors, axis=1)
        true_idx = np.array([LABEL_TO_IDX[lab] for lab in result.scored_labels])
        argmax_acc = float((argmax_pred == true_idx).mean())
    else:
        argmax_acc = float("nan")

    in_consensus = np.isin(result.scored_periods, np.array(CONSENSUS_PERIODS))

    return ScoreReport(
        regime_score=regime_sc,
        quality=quality,
        reliability=reliability,
        per_period=per_period_scores,
        argmax_acc=argmax_acc,
        n_valid=int(in_consensus.sum()),
        n_nan_uniform=n_nan_uniform,
        nonconvergence_rate=nonconv_rate,
    )


# ---------------------------------------------------------------------------
# TSV append (the durable record)
# ---------------------------------------------------------------------------

TSV_COLUMNS = (
    "experiment_id", "observation", "K", "emission", "transitions",
    "regime_score",
    "score_2018_bear", "score_2020q1_covid", "score_bull_2020_2021",
    "score_2022_bear", "score_2024_etf_bull",
    "ll_mean", "argmax_acc", "flip_count", "comment",
)


def append_result(
    experiment_id: str,
    observation: str,
    K: int,
    emission: str,
    transitions: str,
    score_report: ScoreReport,
    ll_mean: float,
    flip_count: int,
    comment: str,
) -> None:
    """Append a single experiment row to RESULTS_PATH. Creates header if new."""
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    new_file = not os.path.exists(RESULTS_PATH)

    def fmt(v):
        if isinstance(v, float):
            if np.isnan(v):
                return "nan"
            if np.isinf(v):
                return "-inf" if v < 0 else "inf"
            return f"{v:.6f}"
        return str(v)

    row = {
        "experiment_id": experiment_id,
        "observation": observation,
        "K": K,
        "emission": emission,
        "transitions": transitions,
        "regime_score": fmt(score_report.regime_score),
        "score_2018_bear": fmt(score_report.per_period.get("2018_bear", float("nan"))),
        "score_2020q1_covid": fmt(score_report.per_period.get("2020q1_covid", float("nan"))),
        "score_bull_2020_2021": fmt(score_report.per_period.get("bull_2020_2021", float("nan"))),
        "score_2022_bear": fmt(score_report.per_period.get("2022_bear", float("nan"))),
        "score_2024_etf_bull": fmt(score_report.per_period.get("2024_etf_bull", float("nan"))),
        "ll_mean": fmt(ll_mean),
        "argmax_acc": fmt(score_report.argmax_acc),
        "flip_count": str(flip_count),
        "comment": comment.replace("\t", " ").replace("\n", " "),
    }

    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Top-level evaluation entry point — what sweep.py calls
# ---------------------------------------------------------------------------

def evaluate(
    experiment_id: str,
    model_factory: ModelFactory,
    *,
    observation: str,
    K: int,
    emission: str,
    transitions: str,
    comment: str = "",
    refit_cadence: int = REFIT_CADENCE_DAYS,
    warmup_days: int = WARMUP_DAYS,
    extra_ll_fn: Optional[Callable[[WalkForwardResult, pd.DataFrame], float]] = None,
) -> ScoreReport:
    """Top-level entry point. Loads data, runs walk-forward, scores, appends TSV.

    `model_factory` produces a fitted HMM from a training slice.
    `observation` / `K` / `emission` / `transitions` are descriptive strings
    (written verbatim to the TSV) so the leaderboard remains human-readable
    without having to introspect sweep.py.

    `extra_ll_fn` optionally lets sweep.py supply a mean-per-window
    log-likelihood diagnostic. If None, ll_mean is recorded as nan.
    """
    t_total_start = time.time()
    data = load_joined()
    _log(f"[harness] Experiment {experiment_id}: loaded joined frame "
         f"({len(data)} rows, {data['date'].min().date()} → "
         f"{data['date'].max().date()})")
    _log(f"[harness] Running causal walk-forward "
         f"(refit_cadence={refit_cadence}, warmup={warmup_days})...")

    result = causal_walk_forward(data, model_factory, refit_cadence, warmup_days)
    _log(f"[harness] Walk-forward complete: {result.n_refits} refits, "
         f"{result.n_nonconvergence} non-converged in "
         f"{time.time() - t_total_start:.1f}s.")

    report = regime_score(result)
    ll = extra_ll_fn(result, data) if extra_ll_fn is not None else float("nan")

    # Flip count: how many times the Viterbi-argmax of the posterior changes
    # over the scored window. A noisy classifier flips a lot; a smooth one
    # doesn't. Diagnostic only.
    valid_post = result.posteriors[~np.isnan(result.posteriors).any(axis=1)]
    flip_count = int(np.sum(np.diff(np.argmax(valid_post, axis=1)) != 0)) if len(valid_post) > 1 else 0

    append_result(
        experiment_id=experiment_id,
        observation=observation,
        K=K,
        emission=emission,
        transitions=transitions,
        score_report=report,
        ll_mean=ll,
        flip_count=flip_count,
        comment=comment,
    )

    print(f"\nScore: regime_score={report.regime_score:.4f} "
          f"(quality={report.quality:.4f} x reliability={report.reliability:.4f})  "
          f"argmax_acc={report.argmax_acc:.4f}  flips={flip_count}")
    print(f"Per-period s_P: {report.per_period}")
    print("Done.")

    return report
