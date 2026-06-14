"""Tests for the smooth severity-weighted Brier `regime_score` (harness.py).

Encodes the contract from
docs/plans/2026-06-14-regime-score-fit-statistic-design.md:

    regime_score = Q * R                                       in [0,1]
    Q   = geomean over consensus periods of s_P
    s_P = 1 - mean_t(L_t) / (a + c)
    L_t = a*(1-p_correct)^2 + b*p_ranging^2 + c*p_opposite^2    weighted Brier
    R   = exp(-lambda * nonconv_rate)

    defaults a=1, b=0.25, c=8, lambda=7 ; NaN posteriors -> uniform.

Run: python test_regime_score.py   (or: pytest test_regime_score.py)
"""
from __future__ import annotations

import numpy as np

import harness

# Spec constants the contract is expected to expose.
A, B, C, LAM = 1.0, 0.25, 8.0, 7.0
DENOM = A + C  # 9.0 — max per-day loss


def _brier_bear(p):
    """Per-day weighted Brier on a bear day, p = (bear, ranging, bull)."""
    return A * (1 - p[0]) ** 2 + B * p[1] ** 2 + C * p[2] ** 2


def _s(L):
    return 1.0 - L / DENOM


def _result(posteriors, label, period_id, n_refits=10, n_nonconvergence=0):
    """Build a single-period WalkForwardResult from a list of 3-vectors."""
    post = np.asarray(posteriors, dtype=float)
    T = len(post)
    return harness.WalkForwardResult(
        posteriors=post,
        scored_dates=np.arange(T),
        scored_labels=np.array([label] * T),
        scored_periods=np.array([period_id] * T),
        n_refits=n_refits,
        n_nonconvergence=n_nonconvergence,
    )


# --- per-period weighted Brier --------------------------------------------

def test_perfect_period_scores_one():
    r = _result([(1.0, 0.0, 0.0)] * 5, "bear", "2018_bear")
    rep = harness.regime_score(r)
    assert abs(rep.regime_score - 1.0) < 1e-9, rep.regime_score
    print("[ok] perfect period -> regime_score == 1.0")


def test_confident_correct_period():
    p = (0.8, 0.2, 0.0)
    r = _result([p] * 4, "bear", "2018_bear")
    rep = harness.regime_score(r)
    expected = _s(_brier_bear(p))  # 0.994444...
    assert abs(rep.regime_score - expected) < 1e-9, (rep.regime_score, expected)
    print(f"[ok] confident-correct period -> {expected:.6f}")


def test_nan_posteriors_scored_as_uniform():
    # An all-NaN period must be scored as max-entropy uniform, NOT dropped
    # (dropping let a model inflate its score by failing on hard days, and an
    # empty period collapsed to -inf under the old contract).
    r = _result([(np.nan, np.nan, np.nan)] * 6, "bear", "2018_bear")
    rep = harness.regime_score(r)
    expected = _s(_brier_bear((1 / 3, 1 / 3, 1 / 3)))  # 0.848765...
    assert np.isfinite(rep.regime_score), rep.regime_score
    assert abs(rep.regime_score - expected) < 1e-9, (rep.regime_score, expected)
    print(f"[ok] NaN posteriors scored as uniform -> {expected:.6f}")


# --- discrimination ordering (the c=8 calibration) ------------------------

def test_caution_beats_hedge_beats_catastrophe():
    def score(p):
        return harness.regime_score(_result([p] * 5, "bear", "2018_bear")).regime_score

    perfect = score((1.0, 0.0, 0.0))
    cautious = score((0.5, 0.5, 0.0))
    abstain = score((0.0, 1.0, 0.0))
    hedged = score((0.5, 0.0, 0.5))
    catastrophe = score((0.0, 0.0, 1.0))

    assert perfect > cautious > abstain > hedged > catastrophe, (
        perfect, cautious, abstain, hedged, catastrophe)
    # c=8 means a wrong-direction half-bet scores worse than staying flat.
    assert hedged < abstain
    assert abs(catastrophe - 0.0) < 1e-9
    print("[ok] ordering perfect>cautious>abstain>hedged>catastrophe; "
          f"hedged={hedged:.3f} < abstain={abstain:.3f}")


# --- cross-period aggregation (geometric mean / soft-min) -----------------

def test_aggregation_is_geometric_not_arithmetic():
    # Two bear periods with different per-period scores.
    p1, p2 = (0.5, 0.5, 0.0), (0.5, 0.0, 0.5)  # s = 0.96528, 0.75
    posteriors = [p1, p1, p2, p2]
    labels = ["bear"] * 4
    periods = ["2018_bear", "2018_bear", "2022_bear", "2022_bear"]
    r = harness.WalkForwardResult(
        posteriors=np.array(posteriors, float),
        scored_dates=np.arange(4),
        scored_labels=np.array(labels),
        scored_periods=np.array(periods),
        n_refits=10, n_nonconvergence=0,
    )
    s1, s2 = _s(_brier_bear(p1)), _s(_brier_bear(p2))
    geo = (s1 * s2) ** 0.5
    arith = (s1 + s2) / 2
    rep = harness.regime_score(r)
    assert abs(rep.regime_score - geo) < 1e-9, (rep.regime_score, geo)
    assert abs(rep.regime_score - arith) > 1e-3, "must be geomean, not arithmetic"
    print(f"[ok] cross-period geomean={geo:.6f} (not arith {arith:.6f})")


def test_one_blind_period_tanks_score_smoothly():
    # ace one period, blind on the other -> soft-min pulls it well below mean.
    good, blind = (1.0, 0.0, 0.0), (0.05, 0.0, 0.95)
    r = harness.WalkForwardResult(
        posteriors=np.array([good, blind], float),
        scored_dates=np.arange(2),
        scored_labels=np.array(["bear", "bear"]),
        scored_periods=np.array(["2018_bear", "2022_bear"]),
        n_refits=10, n_nonconvergence=0,
    )
    rep = harness.regime_score(r)
    s_blind = _s(_brier_bear(blind))
    assert np.isfinite(rep.regime_score)
    assert abs(rep.regime_score - (1.0 * s_blind) ** 0.5) < 1e-9
    print(f"[ok] blind period drags geomean to {rep.regime_score:.4f}")


# --- reliability factor R = exp(-lambda * nonconv_rate) --------------------

def test_reliability_halves_score_at_ten_percent_nonconvergence():
    # 1 failure out of 10 attempts -> rate 0.10 -> exp(-7*0.10) ~ 0.4966.
    r = _result([(1.0, 0.0, 0.0)] * 5, "bear", "2018_bear",
                n_refits=9, n_nonconvergence=1)
    rep = harness.regime_score(r)
    expected = np.exp(-LAM * 0.10)  # 0.496585...
    assert abs(rep.regime_score - expected) < 1e-9, (rep.regime_score, expected)
    print(f"[ok] 10% non-convergence -> score {expected:.4f} (halved)")


def test_full_nonconvergence_does_not_produce_neg_inf():
    r = _result([(1.0, 0.0, 0.0)] * 5, "bear", "2018_bear",
                n_refits=0, n_nonconvergence=10)
    rep = harness.regime_score(r)
    assert np.isfinite(rep.regime_score) and rep.regime_score >= 0.0
    print(f"[ok] 100% non-convergence stays finite -> {rep.regime_score:.4f}")


# --- smoothness / range guarantees ----------------------------------------

def test_score_is_always_finite_and_in_unit_interval():
    cases = [
        (0.0, 0.0, 1.0),   # catastrophe (old floor would give -inf)
        (0.0, 1.0, 0.0),   # abstention
        (1 / 3, 1 / 3, 1 / 3),
        (0.39, 0.61, 0.0),  # just under old 0.40 floor — must NOT be -inf
    ]
    for p in cases:
        rep = harness.regime_score(_result([p] * 3, "bear", "2018_bear"))
        assert np.isfinite(rep.regime_score), (p, rep.regime_score)
        assert 0.0 <= rep.regime_score <= 1.0, (p, rep.regime_score)
    print("[ok] score finite and in [0,1] across adversarial inputs (no -inf floor)")


# --- ScoreReport surface --------------------------------------------------

def test_report_exposes_quality_and_reliability():
    r = _result([(0.7, 0.3, 0.0)] * 4, "bear", "2018_bear",
                n_refits=9, n_nonconvergence=1)
    rep = harness.regime_score(r)
    assert hasattr(rep, "quality") and hasattr(rep, "reliability")
    assert abs(rep.regime_score - rep.quality * rep.reliability) < 1e-9
    assert abs(rep.reliability - np.exp(-LAM * 0.10)) < 1e-9
    # per_period holds the per-period s_P contribution.
    assert "2018_bear" in rep.per_period
    assert abs(rep.per_period["2018_bear"] - _s(_brier_bear((0.7, 0.3, 0.0)))) < 1e-9
    print("[ok] ScoreReport exposes quality, reliability, per-period s_P")


if __name__ == "__main__":
    test_perfect_period_scores_one()
    test_confident_correct_period()
    test_nan_posteriors_scored_as_uniform()
    test_caution_beats_hedge_beats_catastrophe()
    test_aggregation_is_geometric_not_arithmetic()
    test_one_blind_period_tanks_score_smoothly()
    test_reliability_halves_score_at_ten_percent_nonconvergence()
    test_full_nonconvergence_does_not_produce_neg_inf()
    test_score_is_always_finite_and_in_unit_interval()
    test_report_exposes_quality_and_reliability()
    print("\nAll regime_score contract tests passed.")
