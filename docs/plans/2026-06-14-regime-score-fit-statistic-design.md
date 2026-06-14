# Regime-score fit statistic redesign

**Date:** 2026-06-14
**Status:** Design validated, ready for implementation
**Scope:** Replace the `regime_score` fit statistic in `harness.py` (the contract
surface) that the HMM architecture selection/ranking loop optimises against.

---

## Problem

The current `regime_score` is a macro-average over the five consensus periods of
*mean posterior mass on the correct label*, gated by two hard rejections to
`-inf`:

- any single period's mean posterior `< 0.40`,
- non-convergence rate `> 10 %`.

The git log shows the consequence: `exp_002`…`exp_006` all read
`regime_score=-inf below exp_001_baseline`. When most candidates collapse to
`-inf`, the statistic stops being a **ranking** function and becomes a pass/fail
gate with **no gradient** — the selection loop cannot tell which architecture is
*closer* to good, so it cannot climb. The fitness landscape is broken.

We want a statistic that is:

1. **Smooth** — finite everywhere, no `-inf` cliffs, differentiable in the
   posteriors.
2. **A better notion of "good fit"** — rewards confident *correct* calls,
   punishes confident *wrong-direction* calls, forgives caution.
3. **Two-plus terms in tension** (a product), one of which is a severity-aware
   mislabelling penalty.

## Why a single mislabelling term is not enough

The driving question was: is a mislabelling penalty *alone* sufficient if it is
continuous and smooth? No — both readings of "mislabelling alone" fail:

- **Mislabelling = mass on any non-correct label.** This is just
  `1 − (current score)`: the same statistic flipped, no new notion of good, and
  one term carries no tension.
- **Mislabelling = mass on the opposite extreme** (the useful, severity-aware
  reading — calling a bear a bull). Minimising this alone is trivially won by a
  do-nothing model that always predicts **ranging**: opposite-mass `= 0`, score
  perfect, model useless.

So the severity-aware mislabelling term is the *right* term, but it **must** be
held in tension with a correctness term, or the loop converges to a coward. The
chosen statistic encodes that tension.

## Decision: severity-weighted Brier × smooth reliability

Per-day prediction is a 3-vector `p = (p_bear, p_ranging, p_bull)`; the true
label `y` of a consensus period is always `bear` or `bull`. Weight each class by
its **role** relative to `y`:

```
a = weight on the CORRECT class       (hit / calibration term)
b = weight on RANGING                 (the cautious middle)
c = weight on the OPPOSITE extreme    (catastrophe — bull-mass in a bear)
```

**Per-day weighted Brier loss** (a strictly proper scoring rule, made
severity-aware and cost-sensitive via the role weights):

```
bear day:  L = a·(1 − p_bear)²  + b·p_ranging²  + c·p_bull²
bull day:  L = a·(1 − p_bull)²  + b·p_ranging²  + c·p_bear²
```

- `a·(1 − p_correct)²` is the **hit** term — rewards confident correctness
  (quadratic, so 0.9 on-target beats 0.6 on-target by more than linearly).
- `c·p_opposite²` is the **catastrophe** / mislabelling term — heavily, quadratically
  penalises wrong-direction mass.
- `b·p_ranging²` is the **cautious middle** — abstaining costs a little, but far
  less than betting the wrong way.

The two-terms-in-tension live *inside* `L` (hit `a` vs. catastrophe `c`); the
product structure lives in the aggregation below.

**Per period** (`N = 5` consensus periods), normalise the mean loss to `[0, 1]`
(the maximum possible per-day loss is `a + c`, attained by all-mass-on-opposite):

```
s_P = 1 − mean_{t ∈ P}( L_t ) / (a + c)        # 1 = perfect, 0 = maximally wrong
```

**Across periods**, geometric mean (a soft minimum — one blind regime tanks the
score smoothly, replacing the old 0.40 floor's cliff with a differentiable one):

```
Q = ( Π_{P}  s_P )^(1/N)
```

**Reliability factor** — fold the old non-convergence cliff into a smooth
multiplicative term:

```
R = exp( −λ · nonconv_rate )
```

**Final statistic:**

```
regime_score = Q · R          ∈ [0, 1],  higher is better,  smooth & finite
```

`Q` and `R` are each in `[0, 1]`, so the range matches the old contract
(`[0, 1]`, higher better) — only the *meaning* changes. There is no `-inf`
anywhere.

## Constants (fixed in the contract surface, NOT sweep-able)

| const | value | meaning / rationale |
|-------|-------|---------------------|
| `a`   | 1     | hit / calibration weight (reference scale) |
| `b`   | 0.25  | cautious-middle weight; abstention costs a little, not a lot |
| `c`   | 8     | catastrophe weight; a half-mass wrong-way bet scores ~1.8× worse than staying flat (matches trading reality: going long in a bear loses money, flat does not) |
| `λ`   | 7     | reliability decay; chosen so 10 % non-convergence halves the score (`exp(−7·0.10) ≈ 0.50`) |
| `N`   | 5     | the five consensus periods |
| NaN policy | uniform | a NaN / numerically-failed posterior is scored as `(1/3, 1/3, 1/3)` (max-entropy "don't know"), so a model cannot inflate its score by failing on hard days |

These belong in `harness.py` so the optimiser cannot rewrite them to flatter
itself.

## Worked discrimination (a bear period, defaults `a=1, b=0.25, c=8`, denom `9`)

| model `(bear, rng, bull)` | `L` | `s_P` |
|---------------------------|-----|-------|
| perfect `(1, 0, 0)`             | 0.00  | **1.000** |
| confident-correct `(.8, .2, 0)` | 0.05  | 0.994 |
| cautious `(.5, .5, 0)`          | 0.31  | 0.965 |
| all-ranging `(0, 1, 0)`         | 1.25  | 0.861 |
| uniform `(⅓, ⅓, ⅓)`             | 1.36  | 0.849 |
| hedged `(.5, 0, .5)`            | 2.25  | 0.750 |
| confident-wrong `(0, 0, 1)`     | 9.00  | **0.000** |

The ordering is exactly what we want: **caution > abstention > hedging >
catastrophe**, all on a smooth continuum, nothing at `-inf`.

Soft-min illustration (cross-period geomean vs. arithmetic mean): a model acing
four periods but blind on one, `s = [.99, .99, .99, .99, .10]`, scores
`Q ≈ 0.63` (geomean) vs. `0.81` (arithmetic) — the blind spot is amplified, not
averaged away.

## Properties

- **Smooth & finite:** `L_t` is quadratic in `p_t`; `s_P` is linear in the mean
  loss; `Q` is smooth where `s_P > 0` and continuous at 0; `R = exp(...)` is
  smooth and strictly positive. No `-inf`, no cliffs.
- **Tension / anti-degeneracy:** the hit term `a` defeats the all-ranging
  coward; the catastrophe term `c` defeats the bear/bull hedger; the geomean
  defeats the "ace 4, ignore 1" specialist.
- **Un-gameable:** stays a single scalar over frozen labels and frozen
  constants in `harness.py`; the NaN→uniform policy closes the
  "fail-on-hard-days" loophole.

## Implementation sketch (`harness.py`)

1. **Constants** (`harness.py:60–62`): replace `PER_PERIOD_FLOOR` and
   `MAX_NONCONVERGENCE_RATE` with the weight block `A, B, C, LAMBDA` and a
   `NAN_POLICY = "uniform"` marker.
2. **`regime_score()`** (`harness.py:328–391`): rewrite the body —
   - substitute uniform `(1/3,1/3,1/3)` for any NaN posterior row instead of
     dropping it (`harness.py:335–336`);
   - compute per-day `L_t` via the role-weighted Brier (correct / ranging /
     opposite chosen per the day's true label);
   - `s_P = 1 − mean(L_t)/(A+C)` per consensus period;
   - `Q = geomean(s_P over the 5 periods)`;
   - `R = exp(−LAMBDA · nonconv_rate)`;
   - return `regime_score = Q · R`.
3. **`ScoreReport`** (`harness.py:317–325`): drop `hard_rejection`; keep
   `per_period` (now holding `s_P` for the TSV `score_<period>` columns,
   `harness.py:438–442`); add `quality` (`Q`) and `reliability` (`R`) for
   transparency. Keep `argmax_acc` as a diagnostic.
4. **`program.md` §"Scoring rule"** (`program.md:69–91`): update the contract
   prose to describe `Q · R`, the weights, and the smooth reliability factor;
   remove the `0.40` floor and `>10 %` `-inf` language. (This is a
   human-authorised contract change — the *optimiser* still may not edit it.)

## Operational note: re-baseline before resuming the loop

The TSV's historical `regime_score` column changes meaning (old = gated
mean-posterior with many `-inf`; new = `Q · R`). Old rows are **not** comparable
to new ones. Before resuming the selection loop, **re-score `exp_001_baseline`
(and any retained candidates) under the new statistic** to re-establish the
baseline row that future experiments are compared against.

## Out of scope / follow-ups

- **State→label projection non-smoothness.** The score is smooth in the
  posteriors, but the K-state→3-label projection in `sweep.py` (μ-sort for K=3,
  custom rules for K≥4) is a discrete `argmax`/sort. A hyperparameter nudge that
  flips the sort relabels the posterior columns and can still make the score
  *jump across architectures*. This redesign smooths the within-experiment
  landscape; it cannot fully smooth the cross-architecture one. Candidate
  follow-up: soft/rank-based projection — separate effort.
- **Refit-cadence doc drift.** `program.md:97` says weekly; `harness.py:58` sets
  monthly (`REFIT_CADENCE_DAYS = 30`). Pre-existing, unrelated to this redesign;
  reconcile separately.
- **Scoring the ranging windows.** Deferred. Kept the score on the five
  consensus periods only to preserve the `program.md` contract and TSV
  comparability. The Brier form handles `true = ranging` naturally
  (`a·(1−p_rng)² + b·p_bear² + b·p_bull²`) if we later choose to add them.
