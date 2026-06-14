# Regime-detector autoresearch report — 2026-06-14

HMM regime detector (`program.md`, iteration 2). Branch `nhhmm-research`.
Scoring: the **new** smooth severity-weighted Brier `regime_score = Q·R`
(geometric mean over the five consensus periods of a per-day Brier loss weighted
by class role — `BRIER_HIT=1`, `BRIER_RANGING=0.25`, `BRIER_OPP=8` — times a
non-convergence reliability factor `R=exp(-7·rate)`). No `-inf` hard rejections.
Scores here are **not** comparable to the 2026-06-03 run (old arithmetic-mean +
0.40-floor metric); `exp_001_baseline` was re-scored under the new statistic
before any comparison.

## 1. Executive summary

Eleven named experiments ran on top of the re-scored baseline, exploring the
emission family, emission covariance structure, observation channels, transition
structure, state-count, and state→label projection — all at the externally-fixed
operational K=3. **The literature-tier proposal beat the baseline decisively, and
the winner is `exp_004_diag_sigma` — a K=3 multivariate-Gaussian NH-HMM on
`(rₜ, σₜ^{5d}, dₜ_{200})` with diagonal Σ and softmax(VIX) transitions, scoring
0.704 vs the baseline's 0.285 (+147%).** The single most important finding is
that **diagonal Σ decisively beats full Σ** (0.704 vs 0.517): the cross-channel
covariance terms overfit the abundant calm-regime data and crippled detection of
the rare 2020-Q1 COVID crash. This also **corrects the prior run's headline** —
COVID is *not* a pure transition-structure problem; it is jointly an
emission-conditioning and a transition problem, a distinction the prior loop
could not see because it only ever used full Σ. Every Phase-3 attempt to improve
on the winner (drawdown-sort labelling, a relu-gated VIX transition,
identity-pinned labels, Student-t emissions) was a clean negative, each
illuminating *why* the winner's specific choices are right.

## 2. Runtime

- Wall-clock, first experiment commit → last experiment commit: **≈ 102 min**
  (`exp_002` 18:34:46 → `exp_005/006` 20:16:56 on 2026-06-14).
- Eleven experiments, **≈ 9 min/experiment wall-clock**. Experiments were run
  2–3 at a time in parallel, so wall-clock is well below the summed compute.
- Per-experiment walk-forward compute (own clock, **inflated by parallel
  co-runs**; 94 successful monthly refits + 2 warm-up non-convergences each):

  | experiment | walk-forward time | s/refit | notes |
  |---|---|---|---|
  | exp_002 MVN-full + NH        | 513 s  | 5.5  | ~2× parallel |
  | exp_003 MVN-full K=4 + NH    | 1088 s | 11.6 | K=4 heavier |
  | exp_004 MVN-diag + NH (win)  | 1050 s | 11.2 | 2× parallel |
  | exp_007 diag + dd-sort       | 836 s  | 8.9  | |
  | exp_008 diag + relu-VIX      | 910 s  | 9.7  | |
  | exp_009 diag + identity-pin  | 540 s  | 5.7  | solo |
  | exp_010 diag + homog         | 1419 s | 15.1 | 3× parallel |
  | exp_004_nh_only (dd + NH)    | 1420 s | 15.1 | 3× parallel |
  | exp_003_multivariate_only    | 1344 s | 14.3 | 3× parallel |
  | exp_005 MVT-full homog       | 1666 s | 17.7 | Student-t ~2× |
  | exp_006 MVT-full + NH        | ~1600 s| ~17  | Student-t ~2× |

  All comfortably under the 30-minute kill threshold even with contention. Solo
  Gaussian refits run ~5–9 s; Student-t ~2× (per-iteration Cholesky Mahalanobis +
  a per-state dof fixed-point); NH adds an inner L-BFGS softmax fit per EM step.

## 3. Per-experiment table

Baseline = 0.285. Δ = `regime_score` − baseline. Winner in **bold**. The axis
under test is in *italics*.

| id | one-line architecture | regime_score | Δ vs base | flips | verdict |
|---|---|---|---|---|---|
| exp_001_baseline | K=3 Gaussian, drawdown-200, homog | 0.285 | — | 39 | baseline |
| exp_002_proposal_k3 | K=3 *full*-Σ MVN, trivar, *NH-VIX* | 0.517 | +0.232 | 86 | pass (Phase-1 proposal beats base) |
| exp_003_proposal_k4 | *K=4* full-Σ MVN, trivar, NH-VIX | 0.667 | +0.382 | 96 | pass; 4th state earns but K fixed=3 |
| **exp_004_diag_sigma** | **K=3 *diag*-Σ MVN, trivar, NH-VIX** | **0.704** | **+0.419** | 132 | **WINNER** |
| exp_007_diag_ddsort | exp_004 + *drawdown-sort* labels | 0.637 | +0.352 | 135 | fail (hurts 2018; return-sort wins) |
| exp_008_diag_vixgate | exp_004 + *relu(VIX)* gate | 0.437 | +0.152 | 84 | fail (kills COVID 0.755→0.059) |
| exp_009_diag_pinned | exp_004 + *identity-pinned* labels | 0.356 | +0.071 | 131 | fail (relabel is real adaptation) |
| exp_010_diag_homog | exp_004 − NH (*homogeneous*) | 0.593 | +0.308 | 178 | ablation: NH earns +0.111 |
| exp_004_nh_only | *drawdown-only* + NH (no multivar) | 0.401 | +0.116 | 41 | ablation: multivar earns +0.303 |
| exp_003_multivariate_only | full-Σ trivar, *homog* | 0.153 | −0.132 | 85 | ablation: homog kills COVID even full-Σ |
| exp_005_mvt_homog | *Student-t* full-scale, homog | 0.065 | −0.221 | 40 | fail (COVID 4e-6) |
| exp_006_mvt_nh | *Student-t* full-scale, NH | 0.383 | +0.098 | 29 | fail; smoothest but fat tails block entry |

Per-period `s_P` for the winner and the key comparators (bear/COVID/bull20/2022/2024):

| id | 2018 | COVID | bull20 | 2022 | 2024 |
|---|---|---|---|---|---|
| exp_001_baseline | 0.494 | 0.009 | 0.972 | 0.869 | 0.985 |
| exp_002 (full-Σ NH) | 0.638 | 0.211 | 0.735 | 0.908 | 0.848 |
| **exp_004 (winner)** | **0.698** | **0.755** | **0.788** | **0.884** | **0.974** |
| exp_010 (diag homog) | 0.746 | 0.315 | 0.858 | 0.793 | 0.948 |
| exp_003_mvonly (full homog) | 0.837 | 0.0003 | 0.914 | 0.896 | 0.983 |

## 4. Validated findings

Non-obvious, behavioural knowledge the experiments established:

1. **Diagonal Σ decisively beats full Σ — cross-channel covariance overfits and
   cripples crisis detection.** `exp_004` (diag, 0.704) vs `exp_002` (full,
   0.517), a +0.19 gap driven almost entirely by COVID (0.211→0.755) and the
   2024 bull (0.848→0.974). At K=3, D=3 the full Σ carries 18 covariance
   parameters vs the diagonal's 9; the off-diagonal terms, estimated on abundant
   calm-regime days, distort the Mahalanobis geometry for the rare crisis vectors
   and make the bear state reject them. Fewer emission parameters generalise far
   better out-of-sample in a causal walk-forward. *This was the single
   highest-leverage change in the entire ladder, and it makes the model both
   simpler and better.*

2. **COVID is jointly an emission-conditioning AND a transition problem —
   correcting the 2026-06-03 report.** That run concluded "COVID detection is a
   transition-structure problem, not an emission problem." The diagonal × {homog,
   NH} grid refutes this: COVID `s_P` runs full+homog **0.0003** → diag+homog
   **0.315** → full+NH **0.211** → diag+NH **0.755**. Diagonalising the emission
   alone recovers COVID from ≈0 to 0.315 *with homogeneous transitions*; the NH
   switch then adds the rest. The full covariance was a hidden confound the prior
   loop never controlled for (it had no diagonal variant). Both levers are
   necessary; neither alone suffices.

3. **The NH-VIX transition only works as a continuous linear signal; gating it
   destroys COVID.** `exp_008`'s one-sided `relu(vix_std)` gate did exactly what
   it was designed to for the bull periods — confirming the *calm*-VIX linear
   response is what taxes them (2018 0.698→0.823, bull20 0.788→0.876) — but it
   annihilated COVID (0.755→0.059). The gradual VIX rise *into* the crisis is the
   usable signal; a gate that activates only above the training mean relies on a
   slope estimated from sparse high-VIX days and stays too sticky to switch in 28
   days. You cannot remove the calm-VIX cost without losing the crisis signal.

4. **Per-refit μ-relabeling is necessary adaptation, not spurious noise —
   identity-pinning craters the model.** (Directly answering BACKLOG **BL-001**'s
   caveat that flip_count must be de-spuriated before judging churn.) Freezing the
   cold-start (early-2018) state→label map for the whole 7-year walk-forward
   collapses 2022 (0.884→0.118) and COVID (0.755→0.172), with argmax barely above
   random (0.339). Warm-start preserves *parameter continuity* but **not semantic
   state identity** across years; the per-refit `argsort(μ)` correctly tracks
   genuine state drift (keeping bear = lowest-μ). So the bulk of the winner's 132
   flips is real adaptation — full identity-pinning is far worse than the disease.
   A smoke test did confirm real bear↔bull column swaps between adjacent refits on
   early data, so a *middle* ground — cross-refit Hungarian matching against the
   **previous** refit rather than a frozen anchor — remains untested and is the
   one label-stability idea still open.

5. **Return-sort labelling beats drawdown-sort — refuting the prior report's
   recommendation.** The 2026-06-03 report (finding 6) recommended labelling on
   the drawdown channel because the return axis "barely discriminates." But
   `exp_007` (drawdown-sort) changed *only* 2018 (0.698→0.424; the other four
   periods are byte-identical) and made it worse. Return-sort, despite being the
   weaker-*separating* axis, lands the bear/bull assignment more *reliably* here —
   the strong-separation channel is not the same thing as the reliable-labelling
   channel.

6. **Student-t fat tails hurt this score by blocking crisis entry.** MVT
   variants: homog 0.065 (COVID 4e-6), NH 0.383 (COVID 0.039). A fat-tailed bear
   state, by construction, "explains away" the COVID extreme as a tail draw rather
   than switching into it — exactly the wrong property for regime *entry*.
   Tellingly `exp_006` is the **smoothest** model produced (29 flips) yet scores
   0.383, because the flip-blind metric gives no credit for smoothness.
   Diagonal-scale MVT was not pursued: the failure mode is tail down-weighting (an
   emission-*family* property), orthogonal to covariance structure, so
   diagonalising the scale matrix would not fix COVID.

7. **A fourth state earns, but diagonalising the covariance earns more.**
   `exp_003_proposal_k4` (K=4 full-Σ, 0.667) clearly beats K=3 full-Σ (`exp_002`,
   0.517): the extra state absorbs a distinct violent-crash regime (COVID
   0.211→0.625), echoing paper 3's 4-state result. But **K=3 diagonal (0.704)
   beats K=4 full (0.667)** — removing the off-diagonal overfit is worth more than
   adding a state. Since K is operationally fixed at 3 (the Rust pipeline's
   bear/bull/ranging specialists), the deployable winner is the K=3 diagonal
   model; the K=4 result is evidence for the crash-state recommendation below.

## 5. Fundamental limitations

- **The score is flip-blind (BACKLOG BL-001), and now demonstrably prefers the
  churnier model.** `regime_score` is invariant to any time-permutation of days
  within a period; `flip_count` is recorded but never enters `Q·R`. With the full
  ladder in hand the cost is concrete: the **smoothest** model (`exp_006`, 29
  flips) scores 0.383 while the **churniest** (`exp_004`, 132 flips) scores 0.704
  and is selected — because its hard-regime skill dominates the geometric-mean
  soft-min and the metric has no counter-pressure on churn. Each K=3→specialist
  flip is a real capital reroute in the pipeline. This is a scoring-layer
  limitation (the scalar is frozen by design); see recommendations.

- **COVID remains the structural hard ceiling (winner's joint-lowest at 0.755).**
  Three frozen harness constraints conspire: the 30-day refit cadence means the
  model effectively never refits *during* the 28-day crash; the causal
  walk-forward forbids using the crash to recognise it; and the period is short,
  so a filter that takes even ~2 weeks to switch is heavily penalised. The
  exogenous VIX is the only real-time lever and the linear softmax extracts it
  only gradually. No configuration in the explored space pushes COVID materially
  past ~0.76 without sacrificing another period.

- **K=3 is an operational constraint that costs measurable score.** `exp_003_k4`
  shows the data wants a fourth, distinct violent-crash state separate from the
  grinding-deep-bear state (the COVID-vs-2018 distinction); the single "bear"
  column is forced to cover two qualitatively different regimes. K is fixed at 3
  by the Rust pipeline's specialist mapping, not by the data.

- **A fixed warm-up reliability tax caps every trivariate model at R = 0.864.**
  The first two refits (t=200 with 1 valid obs; t=230 with 31) fall below the
  factory's ≥50-observation fit threshold and count as non-convergences, so
  `R = exp(-7·2/96) = 0.864` multiplies every trivariate score (and the
  baseline's). It is uniform — it does not affect ranking — but it caps absolute
  scores at 0.864·Q. The winner's quality is Q = 0.814; the headline 0.704 is
  0.814 × 0.864.

## 6. Recommendations

**Near-term (next experiments, within the current framework):**

- **Cross-refit Hungarian/greedy label matching** — the one label-stability idea
  the ladder did not test. Match each refit's states to the *previous* refit's
  labels (not a frozen cold-start anchor, which `exp_009` proved fatal). This
  could clean the genuine bear↔bull column swaps confirmed on early data without
  freezing out the necessary long-horizon state drift — potentially lifting score
  *and* `flip_count` together.
- **Lower the factory's ≥50-observation fit threshold to ~30** to recover the
  t=230 warm-up refit (31 valid obs is fittable for K=3). That drops failures
  2→1 and lifts R 0.864→0.929 — a uniform +7.5% across all trivariate scores
  (winner 0.704→~0.757). Requires a uniform re-baseline of the ladder to keep
  comparisons honest.
- **Diagonal-scale Student-t** as a completeness check only (low expected value
  given finding 6 — the tail-entry problem is family-level, not covariance-level).

**Medium-term (model-body changes):**

- **Asymmetric transition prior: make bear *entry* cheaper than bear *exit*.**
  This addresses the sticky/fat-tail entry problem (findings 3, 6) in the
  *transition* layer rather than the emission, where it belongs — a structured
  prior that lowers the cost of switching *into* bear on a stress signal while
  preserving bear persistence once there.
- **Richer transition covariates** (paper 3: VIX + 10Y Treasury yield; paper 2:
  DXY / CNYUSD) to sharpen the crisis switch without the relu-gate's COVID cost —
  more *continuous* signal, not a gate.
- **Semi-Markov / explicit-duration sojourn (paper 4)** — primarily a flip
  suppressor (would let `exp_006`-style smoothness be *earned* structurally
  rather than bought with fat tails), only valuable once flips count in the score.

**Long-term (architecture / scoring changes — require human sign-off):**

- **Resolve BL-001 in the frozen scalar.** As measured, the metric *prefers* the
  churnier model. Fold a smooth persistence term into the score — a **TV-norm on
  the posterior path** (`Σ_d |p_{d+1} − p_d|`), per BL-001's own recommendation,
  is preferable to a thresholded flip-count because it stays smooth (preserves
  "always a ranking gradient"). Start γ small (shave ≤10–15% off a 2×-flippy
  model); BL-001 quantifies that a γ > ~0.0127/flip would be needed to overturn
  the proposal, so a reasonable penalty reshapes the gradient without erasing the
  winner's real skill. Until then, treat `flip_count` as a documented secondary
  axis (it is in the TSV).
- **Relax the K=3 constraint.** K=4 earns (+0.15 over K=3-full) and the data
  wants a distinct crash state; revisit once the pipeline can map a 4-state model
  to its specialists (e.g. crash + deep-bear → one bear specialist).
- **Bull/ranging Python ports + full `ensemble_score` integration** with the Rust
  pipeline, now that a strong finite-scoring K=3 regime detector (0.704) exists.

## Appendix — winning configuration and breakdown

**`exp_004_diag_sigma`** — `regime_score = 0.7039` (quality 0.8144 × reliability
0.8643), argmax accuracy 0.474, flip_count 132.

- **K** = 3; state→label by per-refit μ-sort on the **return** channel
  (col 0 = lowest-μ = bear), recomputed each refit (do *not* pin — `exp_009`).
- **Observation**: trivariate `(rₜ, σₜ^{5d}, dₜ_{200})` — daily log-return, 5-day
  rolling realised volatility, rolling-200-day log-drawdown; standardised per
  training window (causal).
- **Emission**: multivariate Gaussian with **diagonal** Σ (the decisive choice).
- **Transitions**: non-homogeneous softmax, `log A_t = softmax_j(W_i·(1, VIX_std_t))`,
  **linear** continuous VIX (not gated — `exp_008`).
- **Refit**: monthly cadence, expanding window, warm-started single-EM after a
  cold k-means + 3-restart initial fit.

Per-period breakdown:

| period | s_P | role |
|---|---|---|
| 2018_bear | 0.698 | joint-lowest |
| 2020q1_covid | 0.755 | joint-lowest (the hard period; +0.75 over baseline) |
| bull_2020_2021 | 0.788 | |
| 2022_bear | 0.884 | |
| 2024_etf_bull | 0.974 | |
| **Q (geomean)** | **0.814** | |
| **R (reliability)** | **0.864** | 2 warm-up non-convergences / 96 |
| **regime_score = Q·R** | **0.704** | |

Learned state structure (full-data fit; channels `[ret, vol5, dd200]`, raw means
`[0.001, 0.025, -0.310]`, raw stds `[0.034, 0.017, 0.293]`; μ and diag(Σ)
standardised):

```
bear     mu=[-0.081,  0.123, -1.254]   diag(Sigma)=[1.052, 0.860, 0.392]
ranging  mu=[ 0.006,  0.452,  0.300]   diag(Sigma)=[1.755, 1.773, 0.159]
bull     mu=[ 0.063, -0.419,  0.833]   diag(Sigma)=[0.419, 0.244, 0.029]

implied A @ calm VIX (z=0)      implied A @ crisis VIX (z=+3)
 [0.993 0.007 0.000]             [0.993 0.007 0.000]
 [0.005 0.956 0.038]             [0.017 0.953 0.030]
 [0.000 0.028 0.972]             [0.000 0.073 0.927]
 (rows/cols = bear, ranging, bull)
```

Bear is the deep-drawdown state (dd −1.25σ); bull is the shallow-drawdown,
low-volatility state (dd +0.83σ, tight covariance); ranging is the high-volatility
middle. The return channel barely separates the states (bear −0.08σ vs bull
+0.06σ) — drawdown and volatility do the discriminating work, which is precisely
why diagonalising Σ (letting each channel vote independently) helps and why
return-*sort* labelling nonetheless suffices. Elevated VIX raises the downward
transition rates (ranging→bear 0.005→0.017, bull→ranging 0.028→0.073) — the
gradual, continuous COVID-cracking mechanism that the relu-gate destroyed.
