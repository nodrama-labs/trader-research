# autoresearch — trader-research (HMM regime detector)

Autonomously iterate on the **HMM regime detector** in this repo. The
goal is to find the highest-scoring HMM or NH-HMM variant for BTC
regime classification, where score is the macro-averaged mean
posterior probability assigned to the correct label over five
consensus regime periods.

This iteration is **not a parameter sweep** (unlike `program_1.md`).
It is a small ladder of architectural hypothesis tests, each a full
named experiment. K is fixed at 3 by external decision (operational
mapping to bear/bull/ranging specialists in the Rust pipeline); the
loop varies emission family, observation channels, and transition
structure.

## Setup

The human starts you in a git repo already on an `autoresearch/<tag>`
branch. **Do not create branches.** Work in the current directory,
commit to the current branch.

On startup, read these files for context:

- `README.md` — repo context.
- `harness.py` — data loader, label loader, scoring, causal walk-forward
  driver, TSV append protocol. **DO NOT MODIFY.**
- `sweep.py` — HMM model body (forward-backward, Baum-Welch, M-steps,
  emissions) and named-experiment runners. This is your playground.
- `data/btcusdt_daily.csv` — extended 2017-08-17 → 2025-12-31 BTC daily.
- `data/vix_daily.csv` — VIX daily over the same window (transition
  covariate for NH-HMM variants).
- `data/consensus_labels.tsv` — ground-truth regime period table.

If a stale `results/regime_sweep_results.tsv` exists from a prior run,
move it aside
(`mv results/regime_sweep_results.tsv results/regime_sweep_results.tsv.bak`)
so this run starts clean.

The prior iteration's scaffolding lives at `program_1.md`,
`harness_1.py`, `sweep_1.py`, `results/bear_sweep_results_1.tsv`. You
do not need to read them; they're preserved as historical record.

## What you CAN edit

- `sweep.py` — everything in it is fair game.
  - The HMM core: forward-backward, Viterbi, Baum-Welch driver.
  - Emission log-pdfs and M-steps (Gaussian, Student-t, multivariate
    Gaussian with full Σ).
  - Transition-matrix M-step (homogeneous closed-form; NH-HMM softmax
    via `scipy.optimize`).
  - Feature pipelines (drawdown, realised vol, scaled log-price, etc.).
  - Named-experiment runners that map an experiment ID to a (feature,
    emission, K, transitions, init) configuration.
- New helper modules under the repo root if a hypothesis needs them.

## What you CANNOT edit

- `harness.py` — the contract surface: data loader, label loader,
  scoring function (`regime_score`), causal walk-forward driver, TSV
  append helper.
- The scoring rule. `regime_score` is the ground-truth metric. It is
  intentionally a single scalar so the optimizer cannot rewrite it.
- `data/**` — the BTC + VIX series and the consensus labels are frozen
  test sets.
- `program.md` — these instructions.
- Dependencies. The stack is `numpy`, `pandas`, `scipy`. Do not add
  more without explicit human confirmation.

## Scoring rule

`regime_score` from `harness.py`:

```
regime_score = mean over labelled periods P of:
                   mean over days d in P of: posterior[d, true_label(P)]
```

Range [0, 1], higher is better. Macro-average over the **five
consensus periods** (2018 bear, 2020-Q1 COVID, 2020-Q2→2021-Q4 bull,
2022 bear, 2024 post-ETF bull) means a long period doesn't dominate a
short one.

Hard rejection (`-inf`) on either:

- Any labelled period scores below **0.40** mean posterior on its true
  label (the "doesn't even argmax to the right regime, on average"
  floor).
- More than 10 % of causal walk-forward windows fail to converge.

A candidate beats the current best if its `regime_score` is strictly
higher.

## Discipline locked into `harness.py` (not sweep-able)

- **Causal walk-forward**: at each evaluation day t, the HMM is fit on
  `[0..t-1]` and scores posterior at `t`. Refit cadence is **weekly**
  (every 7 days) — paper 2's "30-60 day single-regime persistence"
  finding makes weekly refit a sensible default. Posteriors between
  refits use the most recent fit's parameters applied forward.
- **Baseline**: established by `exp_001_baseline` (K=3 Gaussian on
  rolling-200-day drawdown, homogeneous transitions). Every subsequent
  experiment is compared against this row in the TSV.
- **Drawdown** uses a **rolling-200-day** max (`d_t = log(p_t) −
  max_{s ∈ [t-199, t]} log(p_s)`). This kills the window-edge artefact
  the running-max-since-start version had. The first 199 days of the
  series are inside the warm-up and excluded from scoring.

## Research directions (the experiment ladder)

This is a **ladder of named experiments**, not a parameter sweep.
Don't enumerate candidate values; run the next named experiment from
the ladder, score it, decide where to go next.

**Phase 1 — establish baseline, fire main comparison** (the minimum):

1. `exp_001_baseline` — K=3 Gaussian on rolling-200 drawdown
   (univariate), homogeneous transitions. Re-runs the current
   best-in-class HMM with the new misclassification scoring on the
   extended 2017-2024 window.
2. `exp_002_proposal_k3` — K=3 Gaussian on trivariate
   (rₜ, σₜ^{5d}, dₜ_{200}), full Σ, NH-HMM transitions with VIX as the
   covariate (x_t = (1, VIX_t)). The literature-tier proposal.

**Phase 2 — branch on Phase 1 outcome**:

If `exp_002` beats `exp_001` (substantively, not by a hair): *robustness
branch*:

3a. `exp_003_proposal_k4` — same as exp_002, K=4. Does the extra
    state earn under misclassification (where it earned under BIC in
    `regime_drawdown_k4_2022.org`)?
4a. `exp_004_diag_sigma` — same as exp_002, diagonal Σ. Does full Σ
    actually buy anything, or did diagonal suffice?

If `exp_002` does NOT beat `exp_001`: *ablation branch*:

3b. `exp_003_multivariate_only` — trivariate + homogeneous transitions.
    Isolates whether the multivariate observation alone helps.
4b. `exp_004_nh_only` — drawdown univariate + NH-HMM with VIX. Isolates
    whether the non-homogeneous transitions alone help.

**Phase 3** — your call. Based on Phase 2 results, propose 1-3 follow-up
experiments that close remaining ambiguities, or move toward Student-t
emissions (paper 3) / semi-Markov sojourn (paper 4) variants if the
mass of evidence suggests they're worth trying. Stop when (a) a winner
has been established AND its dominant components have been ablated, or
(b) the human interrupts.

## TSV row schema

`results/regime_sweep_results.tsv` is **append-only**. Each row is one
full experiment. Tab-separated. The append helper in `harness.py`
writes one row per `evaluate()` call.

Columns (in order):

```
experiment_id  observation  K  emission  transitions  regime_score
score_2018_bear  score_2020q1_covid  score_bull_2020_2021
score_2022_bear  score_2024_etf_bull
ll_mean  argmax_acc  flip_count  comment
```

The `comment` column is free-text — use it to record the hypothesis
the experiment tested and the takeaway, in ≤ 120 chars.

## Output format

Each `python sweep.py --experiment <experiment_id>` run prints one
human-readable block to stdout AND appends a row to
`results/regime_sweep_results.tsv`. Extract the result with:

```
grep -E "^Score:|^Done\." run.log
tail -5 results/regime_sweep_results.tsv
```

## Experiment loop

LOOP OVER EXPERIMENTS:

1. Look at TSV state: which experiments have run, which won, which
   failed the hard rejection.
2. Pick the next experiment from the ladder above. If Phase 1 not
   complete, run the next Phase 1 experiment. If Phase 1 complete,
   pick the Phase 2 branch based on `exp_001` vs `exp_002` outcome.
   If Phase 2 complete, propose a Phase 3 experiment.
3. If the experiment requires new code in `sweep.py` (e.g. multivariate
   M-step for exp_002 the first time you reach it), implement it.
   Sanity-check on a tiny synthetic window before running the full
   evaluation.
4. Run the experiment:
   `python sweep.py --experiment <experiment_id> > run.log 2>&1`
   (Redirect everything — do NOT `tee` or let output flood your
   context.)
5. Read out the result: `grep -E "^Score:|^Done\." run.log`.
6. If the run crashed: `tail -50 run.log` to read the traceback, attempt
   a fix. If you can't unblock after 2-3 tries, mark the experiment
   FAILED in the TSV's `comment` column and move on.
7. If the new experiment beats the current best, commit your code:
   `git commit -am "<experiment_id> regime_score=<score> beats <baseline_id>"`.
   If it doesn't beat baseline, still commit the code (the TSV row is
   the durable record):
   `git commit -am "<experiment_id> regime_score=<score> below <baseline_id>"`.
8. Decide the next experiment based on Phase 2 / Phase 3 branching.
   Repeat.

**Timeout**: a single experiment evaluation should finish in well
under 10 minutes (the causal walk-forward over 2017-2024 with weekly
refit is ~470 fits per evaluation × seconds-per-fit). If a run exceeds
30 minutes, kill it, debug, and either narrow the evaluation window
or simplify the model body.

**NEVER STOP**: The loop runs until the human interrupts you OR Phase
3's stopping condition is met (winner established + ablations
explained). Do not pause to ask "should I keep going?" — the human
might be asleep.

## Shutdown report (mandatory)

When the experiment loop ends — interrupted, winner-confirmed, or
context-exhausted — **always** generate a shutdown report before doing
anything else.

Write the report to
`docs/autoresearch-reports/<YYYY-MM-DD>-regime-autoresearch-report.md`.

### Required sections

1. **Executive summary** (2-3 sentences): experiments run, headline
   finding, whether the proposal beat baseline, what the winner was.
2. **Runtime**: wall-clock time (first to last commit), per-experiment
   average.
3. **Per-experiment table**: one row per experiment — experiment_id,
   one-line architectural summary, regime_score, vs-baseline delta,
   pass/fail/borderline verdict, brief comment.
4. **Validated findings**: non-obvious engineering knowledge
   discovered. Things that would surprise a reader who hadn't run the
   experiments. Insights about behavior, not parameter values.
5. **Fundamental limitations**: structural constraints that prevent
   further progress within the current framework. In particular: the
   misclassification scoring is silent about flips between adjacent
   regimes (bull → ranging → bull); a fast-cycling but-mostly-correct
   variant scores the same as a slow-cycling correct variant.
6. **Recommendations**: organized as near-term (next experiments to
   try), medium-term (model-body changes — Student-t emissions,
   semi-Markov sojourn), long-term (architecture / scoring changes
   — bull/ranging Python ports, full ensemble_score integration with
   the Rust pipeline).

### Appendix

Final winning experiment's full configuration and its per-period
score breakdown.
