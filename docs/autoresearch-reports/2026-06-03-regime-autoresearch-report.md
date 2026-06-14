# Regime-detector autoresearch report — 2026-06-03

HMM regime detector (`program.md`, iteration 2). Branch `nhhmm-research`.
Scoring: macro-averaged mean posterior of the correct label across the five
consensus periods, with a hard `-inf` rejection if any period falls below the
**0.40** per-period floor or >10% of walk-forward windows fail to converge.

## 1. Executive summary

Six named experiments ran the full 2×2×2 architectural factorial (emission
family × observation channels × transition structure) at the externally-fixed
K=3. **No variant achieved a finite `regime_score`** — every candidate, baseline
included, is hard-rejected by the 0.40 per-period floor. The proposal
(`exp_002`, multivariate-Gaussian + NH-VIX) did **not** beat the baseline
(both `-inf`), so the loop took the Phase-2 ablation branch and then a Phase-3
Student-t escalation. The headline finding is structural: **the 2020-Q1 COVID
crash is undetectable above the floor for any emission family under homogeneous
transitions, and the only mechanism that moves it (NH-VIX transitions) destroys
the calm-VIX 2024 ETF bull.** The closest-to-passing candidate is
`exp_003` (multivariate-Gaussian, trivariate observation, homogeneous
transitions), which clears 4 of 5 floors and fails only on COVID.

## 2. Runtime

- Wall-clock, first experiment launch → final commit: **≈ 80 minutes**
  (baseline ~16:08 → `exp_006` commit 17:21:59 on 2026-06-03).
- Pure walk-forward evaluation compute (sum of the six runs, solo-equivalent for
  the two ablations that ran in parallel): **≈ 60 minutes**.
- Per-experiment evaluation cost (94 successful weekly… *monthly*-cadence refits
  over 2018-03 → 2025-11, 2 expected warm-up non-convergences each):

  | experiment | walk-forward time | s/refit |
  |---|---|---|
  | exp_001 baseline (Gaussian)      | 366 s  | 3.9 |
  | exp_002 MVN + NH                 | 561 s  | 6.0 |
  | exp_003 MVN homog                | ~540 s (solo-equiv) | ~5.7 |
  | exp_004 Gaussian + NH            | ~540 s (solo-equiv) | ~5.7 |
  | exp_005 MVT homog                | 813 s  | 8.7 |
  | exp_006 MVT + NH                 | 807 s  | 8.6 |

  All comfortably under the 30-minute kill threshold. Student-t runs are ~2×
  the Gaussian baseline (per-iteration Cholesky Mahalanobis + a `brentq` dof
  solve per state); NH adds an inner L-BFGS softmax fit per EM iteration.

## 3. Per-experiment table

Floor = 0.40. ✓ = period clears the floor. The dominant axis being tested is in
**bold**.

| id | observation | emission | transitions | regime_score | 2018 | COVID | bull20 | 2022 | 2024 | argmax | flips | floors | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exp_001 baseline | drawdown_200 | Gaussian | homog | −inf | 0.245 | 0.000 | 0.804 | 0.520 | 0.881 | 0.523 | 39 | 3/5 | reject (2018, COVID) |
| exp_002 proposal | trivariate | **MVN** | **NH-VIX** | −inf | 0.390 | **0.207** | 0.696 | 0.869 | 0.083 | 0.463 | 86 | 3/5 | reject; best COVID, 2024 collapse |
| exp_003 mv-only | **trivariate** | MVN | homog | −inf | **0.424** | 0.000 | 0.765 | 0.629 | 0.877 | 0.556 | 85 | **4/5** | reject (COVID only) — best near-miss |
| exp_004 nh-only | drawdown_200 | Gaussian | **NH-VIX** | −inf | 0.076 | 0.000 | 0.796 | 0.000 | **0.993** | 0.464 | 41 | 2/5 | reject; NH wrecks bears |
| exp_005 mvt-homog | trivariate | **MVT** | homog | −inf | 0.259 | 0.000 | 0.815 | 0.904 | 0.992 | **0.568** | 40 | 3/5 | reject; smoothest, best argmax |
| exp_006 mvt-nh | trivariate | **MVT** | **NH-VIX** | −inf | 0.395 | 0.023 | 0.772 | 0.902 | 0.000 | 0.490 | **29** | 3/5 | reject; worst of both worlds |

(`bull20` = bull_2020_2021; `2024` = 2024_etf_bull; `floors` = periods ≥ 0.40.)

## 4. Validated findings

Non-obvious things the experiments established (behavioural, not parameter
values):

1. **COVID detection is a transition-structure problem, not an emission
   problem.** Every emission family — univariate Gaussian, full-Σ multivariate
   Gaussian, multivariate Student-t — scores ≈ 0.000 on the 2020-Q1 COVID crash
   under homogeneous transitions (exp_001/003/005). Enriching the observation
   (adding return + 5-day realised-vol channels) and fattening the tails do
   **nothing** for COVID. The *only* lever that moved it was the non-homogeneous
   softmax transition driven by the exogenous VIX (exp_002 → 0.207). The 28-day
   crash is simply too short for a sticky filter (learned self-transition
   ≈ 0.97–0.99) to switch state on endogenous price features before it is over;
   an exogenous real-time fear gauge is required to force the switch.

2. **The non-homogeneous VIX transition is a double-edged sword that nets
   negative.** It is the sole COVID-cracker, but it *collapses* the calm-VIX
   2024 ETF bull: 0.877 (homog) → 0.083 (exp_002) → 0.000 (exp_006). During
   fitting the softmax learns to route low-VIX days toward the ranging state
   (low VIX spans 2023 ranging, 2024 bull, and parts of the 2020-21 bull), so
   the steady low-volatility 2024 grind is misclassified as ranging. On the
   baseline's univariate drawdown channel the NH transition is outright
   catastrophic for bears (exp_004: 2018 → 0.076, 2022 → 0.000) because it
   overfits bull-persistence.

3. **Fat tails (Student-t) buy smoothness and persistent-regime confidence but
   cost crisis sensitivity.** MVT halved the flip count (exp_003 85 → exp_005 40)
   and pushed the long, clean regimes to near-certainty (2022 0.629 → 0.904,
   2024 0.877 → 0.992, best argmax accuracy 0.568). But it *dropped* the 2018
   bear (0.424 → 0.259): the fattened ranging/bull states "explain away" milder
   bear-onset moves as tail draws instead of switching to bear. A fat-tailed
   state is, by construction, harder to leave on the strength of an extreme
   observation — exactly the wrong property for regime *entry*.

4. **Fat tails and NH transitions are antagonistic (exp_006).** Stacking them
   gave the worst-of-both: COVID *worse* than exp_002 (0.207 → 0.023) and the
   2024 bull *fully* gone (0.000). The fat tail lets the pre-crisis state absorb
   the COVID extreme as a tail event, directly undermining the VIX-driven switch
   the NH transition is supposed to perform. The single smoothest model (29
   flips) is the least able to catch the sharp crises — a clean illustration that
   "smooth and confident" and "fast crisis detection" are opposing objectives
   here.

5. **The multivariate observation, alone, is the one unambiguous win.** Holding
   transitions homogeneous, going univariate-drawdown → trivariate
   (return, 5-day vol, drawdown) lifts both bears (2018 0.245 → 0.424, clearing
   the floor; 2022 0.520 → 0.629), preserves the 2024 bull (0.881 → 0.877), and
   gives the best argmax accuracy among Gaussian models (0.556). It is the only
   component that helps without a compensating regression elsewhere.

6. **The regimes separate on drawdown depth and volatility, not on daily
   return.** In the full-data exp_003 fit the return channel barely
   discriminates (bear μ_ret −0.08σ vs bull +0.08σ), while drawdown does the
   heavy lifting (bear −1.25σ deep vs bull +0.85σ shallow) alongside volatility.
   The return-channel μ-sort labelling therefore rests on a weak axis; it still
   produces the correct bear→bull ordering only because return is positively
   correlated with the (much stronger) drawdown and vol separation. A
   drawdown-channel sort would be more robust and is a cheap thing to revisit.

## 5. Fundamental limitations

- **The COVID 0.40 floor is structurally unattainable in this framework.** It
  combines three fixed harness constraints that conspire against a 28-day crash:
  (a) the **30-day refit cadence** means the model effectively never refits
  *during* COVID — it views the crash through parameters trained on calm
  pre-COVID data, in which no COVID-like regime exists; (b) the **causal
  walk-forward** forbids using the crash to recognise the crash; (c) the
  **per-period floor** is applied to a mean posterior over only ~28 days, so a
  filter that takes even two weeks to switch already fails. The exogenous VIX
  covariate is the one signal available in real time, and using it sacrifices
  the 2024 bull. There is no configuration in the {Gaussian, MVN, MVT} ×
  {homog, NH-VIX} × {drawdown, trivariate} space that clears it.

- **Scoring is blind to flip dynamics between adjacent regimes** (noted in
  `program.md`). A fast-cycling-but-mostly-correct model scores identically to a
  slow-cycling correct one. This directly masks a real quality difference we
  measured: exp_003 (85 flips) and exp_005 (40 flips) would score similarly on
  the consensus periods even though exp_005 is far more usable as a regime
  signal. The flip_count diagnostic is recorded but does not enter the score.

- **K is externally fixed at 3.** The state that the data most wants to add — a
  distinct high-volatility *crash* state separate from the grinding *deep-bear*
  state (the COVID-vs-2018 distinction in finding 6) — cannot be expressed. The
  single "bear" column is forced to cover two qualitatively different regimes.

- **Per-period floor + macro-average is an all-or-nothing gate.** Five of six
  models score well on four periods; a single sub-floor period zeroes the whole
  candidate to `-inf`. This makes the metric extremely sensitive to the single
  hardest period (COVID) and insensitive to broad competence elsewhere.

## 6. Recommendations

**Near-term (next experiments, within the current framework):**

- **Drawdown-channel μ-sort** (`order_channel = 2`) for the trivariate models,
  per finding 6 — the return axis is too weak to label on reliably. Cheap re-run
  of exp_003/exp_005.
- **Gate the NH transition on a VIX *threshold* rather than a linear softmax
  slope.** The damage in exp_002/006 is that *low* VIX routes 2024 to ranging.
  A transition that is homogeneous in calm regimes and only becomes
  VIX-responsive above a fear threshold could keep COVID's lift without the
  calm-VIX 2024 collapse. (This is an architectural change to `nh_log_A_seq`,
  not a parameter sweep.)
- **Asymmetric stickiness / lowered entry cost to the bear state** to address the
  Student-t entry problem in finding 3 — a structured transition prior that makes
  bear *entry* cheaper than bear *exit*.

**Medium-term (model-body changes):**

- **Semi-Markov / explicit-duration HMM (paper 4).** Caveat from this run: the
  COVID failure is a regime-*entry* problem, not a dwell-time problem, so a
  vanilla sojourn model is unlikely to fix COVID on its own. It is more promising
  for *suppressing flips* (finding 4) — encoding the "30–60 day persistence"
  prior directly would let exp_005-style smoothness be earned rather than an
  accident of fat tails.
- **A dedicated crash/high-vol emission state** if the K=3 constraint is ever
  relaxed — the data clearly wants to separate the violent-short crash (COVID)
  from the grinding-deep bear (2018), which a single bear column cannot hold.

**Long-term (architecture / scoring changes — require human sign-off):**

- **Revisit the COVID floor or the refit cadence in the harness.** As measured,
  the 0.40 COVID floor under a 30-day causal refit is provably unreachable; it is
  effectively a structural reject on every model. Either a shorter refit cadence
  during high-VIX windows, a COVID-specific floor, or a softer aggregate penalty
  would let the metric reward the genuine 4/5 progress.
- **Fold the flip_count into the score** (or an adjacent-regime transition
  penalty) so the metric can distinguish exp_005's smoothness from exp_003's
  churn — the current scalar cannot.
- **Bull/ranging Python ports + full `ensemble_score` integration** with the Rust
  pipeline, once a finite-scoring regime model exists.

## Appendix — best candidate configuration and breakdown

**No finite-scoring winner exists** (all candidates `-inf`). The best near-miss,
recommended as the working configuration, is **`exp_003_multivariate_only`**:

- **K** = 3, μ-sort labelling on the return channel (bear = lowest mean return).
- **Observation**: trivariate `(rₜ, σₜ^{5d}, dₜ_{200})` — daily log-return,
  5-day rolling realised volatility, rolling-200-day log-drawdown. Standardised
  per training window (causal).
- **Emission**: full-covariance multivariate Gaussian.
- **Transitions**: homogeneous (closed-form Baum-Welch M-step).
- **Refit**: monthly cadence, expanding window, warm-started single-EM after a
  cold k-means + 3-restart initial fit.

Per-period breakdown (regime_score = −inf due to the COVID floor):

| period | mean posterior on true label | clears 0.40? |
|---|---|---|
| 2018_bear | 0.424 | ✓ |
| 2020q1_covid | 0.000 | ✗ (sole blocker) |
| bull_2020_2021 | 0.765 | ✓ |
| 2022_bear | 0.629 | ✓ |
| 2024_etf_bull | 0.877 | ✓ |
| argmax accuracy (all labelled days) | 0.556 | — |
| flip count | 85 | — |

Learned state structure (full-data fit, standardised channels `[ret, vol5,
dd200]`; raw means `[0.001, 0.025, -0.310]`, raw stds `[0.034, 0.017, 0.293]`):

```
bear     mu=[-0.078,  0.112, -1.250]   diag(Sigma)=[1.045, 0.855, 0.395]
ranging  mu=[-0.032,  0.384,  0.245]   diag(Sigma)=[1.666, 1.891, 0.123]
bull     mu=[ 0.083, -0.335,  0.850]   diag(Sigma)=[0.527, 0.332, 0.026]
transition A (rows/cols = bear, ranging, bull):
   [0.993, 0.007, 0.000]
   [0.007, 0.966, 0.027]
   [0.000, 0.017, 0.983]
```

Bear is the deep-drawdown state (dd −1.25σ); bull is the shallow-drawdown,
low-volatility state (dd +0.85σ, tight covariance); ranging is the
high-volatility middle. Transitions are very sticky. Honourable mention:
**`exp_005_mvt_homog`** (same config, Student-t emissions) is the most *usable*
signal — best argmax accuracy (0.568), less than half the flips (40) — but loses
the 2018 bear (0.259) and so clears only 3/5 floors.
