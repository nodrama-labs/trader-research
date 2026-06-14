# autoresearch — trader-research (bear specialist)

Autonomously iterate on the **bear specialist** strategy in this repo. The
full production stack is an HMM regime detector + per-regime specialists +
meta-allocator; this repo carves out only the bear specialist, evaluated on
the 2022 bear-market window over a small subset of tokens
(`data/bear_portfolio_candles.csv`).

The loop sweeps the parameters of the bear specialist one at a time. After
each sweep, the best-scoring value is locked in as the new default and the
loop moves to the next parameter. The scalar metric is `ensemble_score`
from `harness.py` — the name is inherited from the full ensemble's scoring
contract and stays fixed even though only one specialist is under test
here.

## Setup

The human starts you in a git repo already on an `autoresearch/<tag>`
branch. **Do not create branches.** Work in the current directory, commit to
the current branch.

On startup, read these files for context:

- `README.md` — repo context.
- `harness.py` — data loader, backtest skeleton, scoring, fee / capital
  constants. **DO NOT MODIFY.**
- `sweep.py` — `GemParams`, model body (`fit_token_exponential`,
  `build_portfolio`), single-parameter sweep driver. This is your playground.
- `data/bear_portfolio_candles.csv` — the only dataset. Verify it exists; if
  not, stop and tell the human.

If a stale `results/bear_sweep_results.tsv` exists from a prior run, move it
aside (`mv results/bear_sweep_results.tsv results/bear_sweep_results.tsv.bak`)
so this run starts clean.

## What you CAN edit

- `sweep.py` — everything in it is fair game.
  - `GemParams` defaults and fields.
  - The model body: `fit_token_exponential`, `build_portfolio`, the
    regression / ATR primitives.
  - The sweep driver, candidate values, output formatting.
- New helper modules under the repo root if a hypothesis needs them.

## What you CANNOT edit

- `harness.py` — the contract surface (data loader, backtest skeleton,
  metric aggregator, `ensemble_score`, `FEE_RATE` / `FEE_STRESS_MULTIPLIER` /
  `INITIAL_CAPITAL` constants).
- The scoring rule. `ensemble_score` is the ground-truth metric. It is
  intentionally a single scalar so the optimizer cannot rewrite it.
- `fee_rate` or `initial_capital` — these are external constraints
  (exchange fees, portfolio sizing), not strategy parameters. The harness
  pins them inside `evaluate`; whatever your `GemParams` holds for these
  fields is overridden.
- Dependencies. The stack is `numpy`, `pandas`, `scipy`. Do not add more
  without explicit human confirmation.

## Scoring rule

`ensemble_score` from `harness.py`:

```
score = annualized_return × drawdown_dampener × diversification_bonus
```

- `drawdown_dampener = 1 / (1 + max(0, dd - 0.15))²` — 15% free zone, then
  quadratic decay.
- `diversification_bonus = 1 + 0.1 × (1 - hhi)` — up to +10% for portfolio
  diversity.

Hard rejection (`-inf`) on either:

- Annualized return below -50%.
- Stress-test calmar ratio (1.5× fees) is negative.

A candidate beats the current best iff its `ensemble_score` is strictly
higher.

## Research directions (parameter priority)

Sweep the bear-specialist parameters in this order. After the winner of
one parameter is locked in as the new default, move to the next.

**Tier 1 — structural knobs, sweep first:**

1. `top_n` over `[1, 3, 5, 10]` — portfolio breadth.
2. `r2_threshold` over `[0.3, 0.5, 0.7, 0.8]` — fit-quality cutoff.
3. `rebalance_cooldown` over `[3, 7, 14, 21]` — churn rate.
4. `atr_window` over `[7, 14, 21, 30]` — ATR weighting window.
5. `fit_window` over `[20, 30, 60, 90]` — exponential-fit history length.

**Tier 2 — finer knobs:**

6. `momentum_cap` over `[0.05, 0.10, 0.14, 0.20, 0.50]`.
7. `r2_exponent` over `[1.0, 1.5, 2.0, 3.0]`.

After Tier 2, re-sweep Tier 1 with new defaults locked in to catch
interactions, or zoom in around a Tier 1 winner with a finer grid.

## Output format

Each `python sweep.py --param X --values v1,v2,...` run prints one line per
candidate to stdout AND appends a row to `results/bear_sweep_results.tsv`.
Extract the result with:

```
grep -E "^Best:|^Done\." run.log
tail -10 results/bear_sweep_results.tsv
```

## Experiment loop

LOOP OVER PARAMETERS:

1. Look at git state: which parameters have already been swept and locked
   in (look at `GemParams` defaults and recent commits).
2. Pick the next parameter from the priority order above. If all are done,
   start a second pass with the new defaults locked in, or zoom into a
   prior winner with a finer grid.
3. Pick a candidate list. Use the suggested list, or narrow / widen it
   based on prior wins.
4. Run the sweep:
   `python sweep.py --param <name> --values <v1,v2,...> > run.log 2>&1`
   (Redirect everything — do NOT `tee` or let output flood your context.)
5. Read out the result: `grep -E "^Best:|^Done\." run.log`.
6. If the run crashed: `tail -50 run.log` to read the traceback, attempt a
   fix. If you can't unblock after 2-3 tries, drop the parameter and move
   on.
7. If a finite-score winner exists and beats the current default, update
   the default in `GemParams` in `sweep.py` to the winning value and
   `git commit -am "<param>=<value> (score <best> from <baseline>)"`.
   If every candidate returned `-inf`, do not modify `GemParams`; treat
   the parameter zone as hostile under current defaults and move on.
8. Repeat.

**Timeout:** a single parameter sweep should finish in well under a
minute. If a run exceeds 5 minutes, kill it, narrow the grid, and treat
the run as a failure.

**NEVER STOP:** The loop runs until the human interrupts you. Do not pause
to ask "should I keep going?" — the human might be asleep. If you run out
of priority-list parameters, re-read this file and `sweep.py` for new
angles — try a deletion (toggle a `use_*` flag off), zoom into the
neighborhood of a Tier 1 winner, or try combinations. The loop runs until
interrupted, period.

## Shutdown report (mandatory)

When the experiment loop ends — whether interrupted by the human, hitting
a proven ceiling, or running out of context — **always** generate a
shutdown report before doing anything else. This is not optional.

Write the report to
`docs/autoresearch-reports/<YYYY-MM-DD>-autoresearch-<tag>-report.md`
where `<tag>` is the branch name or loop identifier.

### Required sections

1. **Executive summary** (2-3 sentences): parameters swept, kept /
   discarded count, headline finding.
2. **Runtime**: wall-clock time (first to last commit), per-experiment
   average, session breakdown if the loop spanned multiple conversations.
3. **Per-parameter table**: one row per parameter swept — candidate values
   tried, winner, score before, score after, delta.
4. **Validated findings**: non-obvious engineering knowledge discovered.
   Things that would surprise a reader who hadn't run the experiments.
   Insights about behavior, not parameter values.
5. **Fundamental limitations**: structural constraints that prevent
   further progress within the current framework. In particular: parameter
   interactions a one-at-a-time sweep cannot discover.
6. **Recommendations**: organized as near-term (next sweeps to try),
   medium-term (model-body changes), long-term (architecture / scoring
   changes). Concrete and actionable.

### Appendix

Final `GemParams` (winning configuration) and its scores across base and
stress backtests.
