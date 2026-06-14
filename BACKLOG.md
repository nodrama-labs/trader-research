# Backlog — trader-research

Tracked ideas and known limitations not yet on the experiment ladder.
Status: `open` / `in-progress` / `done` / `wontfix`.

---

## BL-001 — `regime_score` is flip-blind (no penalty for regime back-and-forth)

**Status:** open
**Raised:** 2026-06-14
**Area:** scoring (`harness.py`, frozen) + modeling layer (`sweep.py`)

### Problem

The per-day loss `L_d` (`harness.py:385-389`) depends only on the *marginal*
posterior at day `d`:

```
L_d = 1·(1−p_correct)² + 0.25·p_ranging² + 8·p_opposite²
```

`s_P = 1 − mean_d(L_d)/9` averages over days, so the score is **invariant to
any time-permutation of days within a period**. A smoothly-held correct
posterior and a day-to-day jittering posterior with the same time-averaged
marginals score identically. `flip_count` is computed (`harness.py:537-541`)
but is **diagnostic only** — it never enters the `Q·R` objective.

The blind spot is concentrated on **adjacent flips** (bull→ranging→bull): the
ranging day pays only `BRIER_RANGING = 0.25`, while opposite-extreme flips are
already crushed by `BRIER_OPP = 8`. So the metric is silent exactly where churn
is cheap.

### Why it matters

K=3 maps to bear/bull/ranging specialists in the Rust pipeline; each flip
routes capital between specialists → whipsaw / transaction cost. The metric
ranks a fast-cycling-but-mostly-correct detector equal to a slow-cycling
correct one, while the pipeline pays real money for the difference.

### Evidence

| exp | regime_score | argmax_acc | flip_count |
|---|---|---|---|
| exp_001_baseline | 0.285 | **0.533** | **39** |
| exp_002_proposal_k3 | **0.517** | 0.474 | **86** |

The selected winner has 2.2× the flips and *lower* argmax accuracy. Per-period,
exp_002's bull_2020_2021 *dropped* 0.972→0.735 (the flip-tax: reactive model
dips into ranging on pullbacks) but bought big hard-regime gains
(covid 0.009→0.211, 2018 bear 0.494→0.638) that the geomean soft-min amplifies.
The loop has no counter-pressure on churn, so the selection gradient drifts
toward reactive models.

### Magnitude

A multiplicative `exp(−γ·flips)` penalty would need `γ > ~0.0127/flip` (~1.3%
per flip) to overturn exp_002 below exp_001 — aggressive. So a *reasonable*
flip penalty shrinks the margin but does **not** overturn exp_002 (its
hard-regime skill is real); it mainly reshapes the gradient for the rest of the
ladder.

### Options (the scalar is frozen by design — `program.md:66-71`)

1. **Human metric change (outside the loop).** Fold a smooth persistence term
   into the score — prefer a **TV-norm** on the posterior path
   (`Σ|p_{d+1}−p_d|`) over a thresholded flip-count: stays smooth (preserves
   "always a ranking gradient"), penalizes the path not the argmax. Start γ
   small (shave ≤10–15% off a 2×-flippy model).
2. **Use the existing diagnostic.** `flip_count` is already in the TSV — use it
   as a human tiebreaker / secondary axis in the shutdown report without
   touching the scalar.
3. **Modeling-layer pressure (allowed in the loop).** Stickier transitions
   (Dirichlet self-transition prior / higher diagonal init), Student-t
   emissions (exp_005 already halved flips as a side-effect), semi-Markov
   sojourn (paper 4). But the loop won't prefer these unless flips count.

### Caveat / dependency

Some of the 86 flips may be **spurious relabeling**, not model jitter — the
`argsort(μ)` state→label map can swap columns between refits (see the
"State→label projection stability" item, `program.md:180`). Pin the label map
(identity-pin / Hungarian cross-refit matching) to clean `flip_count` **before**
deciding how hard to penalize on it.

### Next step

Decision needed (human): change the frozen metric vs. apply modeling-layer
pressure vs. treat flips as a review-only tiebreaker. Optional quantification:
re-run exp_001/exp_002 with posteriors dumped, decompose each period's loss
into hit/ranging/opposite channels + TV-norm of the path, to size exactly how
much of exp_002's score rides on the cheap ranging-flip channel.
