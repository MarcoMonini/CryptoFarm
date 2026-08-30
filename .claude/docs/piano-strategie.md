# Plan — improving the strategies, and making new ones

Decided on **2026-08-27** with the user, via `AskUserQuestion`, on branch
`claude/ricerca-quant-ml-cinque-asset`. It replaces the "what I would do next" list in
[`HANDOFF.md`](HANDOFF.md), which remains as context.

> **Status as of 2026-08-30.** Step 1 (multiplicity, `scripts/multiplicity.py`) is done and its
> numbers are below. Step 2bis became `trading/confluence.py`, measured and **negative**
> (`strategia-confluenza.md`). Steps 2, 3, 4 and 5 have **not been executed**: the work moved to the
> model strand, which meanwhile produced the project's only positive result
> (`modello-ingresso.md`). They remain valid, unexpired proposals — in particular step 5, the
> 2017-2020 cycle, which is **the last clean verification window left** and must be spent
> deliberately. The "two decisions taken, not to be reopened" below still hold.

## Where it comes from

The measurements on five assets say three things that together choose the plan:

- **per-asset timing is nearly dead**: 24% of out-of-sample cells beat passive holding, and 9 of
  those 12 wins are in windows where passive was losing;
- **the cross section is the only family that transfers**: 89% of configurations profitable out of
  sample, and the real advantage is on risk (DD 45.7% against 91.0%), not on return;
- **choosing the parameters is harmful**: ρ = −0.69 between in-sample and out-of-sample return.

The third point is the one nobody has yet taken fully seriously. If selecting makes things worse, the
answer is not "take central values" — it is **not selecting at all**.

## Two decisions taken, not to be reopened

1. **The 2017-2020 cycle is spent after steps 2-4**, not now. It is the last clean verification
   window left: burning it on the current design takes it away from the new one. Accepted cost: steps
   2-4 are built on top of a result not yet confirmed on a second cycle.
2. **Start from step 1**, multiplicity, because it is the prerequisite for the others: without it,
   every number steps 2-4 produce is again an uncorrected grid maximum.

---

## Step 1 — multiplicity *(in progress, commit `d49f46c`)*

**What.** `deflated_sharpe_ratio` accepts `trial_variance`; `scripts/multiplicity.py` applies DSR to
the grids in `reports/cs_*.csv` and combinatorial PBO to the (year × configuration) matrices already
in `analysis_cache/*/*_annuale.parquet`. No sweep needs re-running.

**Done here**, on the cross-sectional rotation (160 configurations, 2021-2026, 2,057 observations):

| trials counted | chance threshold | DSR of the maximum | DSR of the median |
|---|---|---|---|
| 160 (the family alone) | 0.77 annual | **0.976** | 0.811 |
| 12,000 (the whole project) | 1.12 annual | 0.875 | 0.523 |
| out of sample 2024-2026 | 0.84 annual | 0.736 | 0.486 |

**How to read it.** The maximum only survives if you count the trials of the family alone. With the
honest count it does not survive, and **the median survives under neither count** — and the median is
the configuration one would keep, given that optimising is harmful. It is not a condemnation of the
rotation: 82% of the grid clears the chance threshold, which is a fact about the *family* and not
about the lucky cell. It is a condemnation of reading that grid from its maximum.

**What is missing**, and is not runnable here or by the user until it is on the machine holding the
clone (`analysis_cache/` is gitignored, and the remote session has neither candles nor network access
to the exchanges):

```bash
python -m scripts.multiplicity --cache          # PBO over all the grids already cached
python -m scripts.multiplicity --selfcheck      # 5 checks, runs anywhere
```

If PBO comes out above 0.5 on a family, that family must be read **only** by its median: its
selection procedure does worse than chance, and that is a measurement, not an opinion.

---

## Step 2 — grid ensemble

**The idea.** ρ = −0.69 says that choosing a configuration destroys value. The mechanical consequence
is not "choose better": it is **keep them all**. An equal-weight portfolio of every configuration in
the grid has no parameters to choose, so it has nothing to overfit, and its expected return is the
grid median — which is exactly the column the project has already learned to read as the honest one.

**Why it is plausible, not just elegant.** Neighbouring configurations are almost the same strategy
(the dispersion measured in step 1 says so: 0.0149 against 0.0221 for independent trials), so
averaging them does not diversify the return much — but it smooths the entry timing, which is where
the variance that does not transfer lives.

**Cost.** ~20 lines on top of `rotation.py`. No new dependencies.
**Verification.** Out of sample 2024-2026, against three references: the current central
configuration, the best in-sample one, and the equal-weight universe. The success criterion is
declared in advance: the ensemble must beat **the best in-sample one**, which is the procedure being
replaced. If it does not, the idea is dead and that gets written down.

---

## Step 3 — volatility targeting

**The idea.** Every measurement in this project is at full capital, always. The only cross-sectional
advantage ever found is **on risk** (DD halved). The layer that acts directly on that axis has never
been tried: scaling exposure by the inverse of realised volatility, with a cap.

**Why here and not elsewhere.** Crypto has volatility that varies by a factor of five across regimes.
At fixed exposure the portfolio's risk is entirely decided by the market; at targeted risk it is
decided by whoever writes the rule. And it composes with everything: rotation and per-asset
strategies.

**Cost.** `pnl.simulate_positions` already knows about leverage — it needs to become per-bar instead
of constant. Few lines, but they touch the engine: tests first.
**Verification.** At equal drawdown, does the return go up? That is the only question. The comparison
must be made by rescaling both curves to the same drawdown, not at equal nominal leverage.
**Known trap.** Realised volatility must be computed on bars closed at `i-1`. It is the same defect
already found once in the trailing stop, and `test_no_look_ahead` **would not see it**.

---

## Step 4 — residual momentum, and rank averaging

**The idea.** The rotation today ranks by raw strength. In crypto almost all the return is BTC beta,
so that ranking mostly orders by *how much beta* each asset has — not by which one is doing better
than it should. Ranking on the **residual against BTC** is different information.

Second piece: instead of a single signal, the **average of the ranks** of several signals (residual
momentum, low volatility, trend quality). Rank averaging, not estimated weights: estimating weights
is exactly the selection that step 1 and ρ = −0.69 say not to do.

**Why it is the right direction.** It is the point of the qlib benchmark read in
`ricerca-quant-ml.md` §1: the ceiling is IC ≈ 0.05, and it is monetised **in the cross section**, not
in time, because the error averages out over several assets.

**Cost.** One new signal in `rotation.py` plus the rank combination. ~40 lines.
**Verification.** Same grid, same references as step 2, plus the control that counts: the residual
must beat raw strength **on the median**, not on the maximum.

---

## Step 5 — the second cycle (2017-2020)

**What.** Redo rotation and the meta filter on 2017-2020 with the universe that existed then (BTC,
ETH, XRP, BNB, LTC), **with the design that comes out of steps 2-4**, not with the current one.

**Why it is last.** It is the only truly independent verification left, and it is spent once: every
measurement made on that window contaminates it for the next. The data is already in the user's
store.

**Criterion, declared before looking.** On 2017-2020 the new design must: have a positive median,
beat the equal-weight universe, and have a drawdown below passive's. Three conditions, decided now
precisely because deciding them afterwards would be another selection.

---

---

## Step 2bis — consensus among strategies

Requested by the user on 2026-08-27: an algorithm that recognises market conditions, derives
**truthfulness weights** for each strategy from them, and acts when the weighted sum exceeds a
**dynamic threshold**. It is the same idea as step 2 one level up — there configurations of one
strategy are averaged, here strategies are averaged against each other — and the same rule applies:
**the weights are not estimated**.

### The data that was already there, and that nobody had read this way

`live_bot.py` — the only code in the project that moves real money — already votes: `NUM_CONDITIONS`
decides how many of the ATR band and RSI must agree (`live_bot.py:441`, `:458`). And the
`close_buy_sell_limits` grid in `strategy_sweep` **sweeps `num_cond` between 1 and 2**, 864
configurations per side, over all five symbols and three intervals. It has been in
`reports/sensibilita_*.csv` since day one.

Asking for two conditions instead of one, median return:

| interval | improves | unchanged | worsens | median trades/year |
|---|---|---|---|---|
| 15m | BTC, ETH | — | — | 279 → 53 |
| 4h | BNB, BTC, ETH, SOL | — | XRP | 15-17 → 2-3 |
| 1d | BNB, ETH, SOL | BTC | XRP | 3 → **0** |

Nine out of twelve improve, the median Sharpe rises in ten. **But it is not proof that voting adds
information**: it cuts trades by five to ten times, and this project has already established that
trading frequency explains almost everything. At one day the median goes to 0.0% with zero median
trades: the strategy did not improve, it stopped trading.

**The control that is missing, and that decides:** compare two-condition voting against **a single
condition tuned to the same trading frequency**. If voting does not beat that reference, it is not
selecting better — it is just trading less, and trading less costs one line, not an algorithm.

### The diagnostic to do first, before writing the algorithm

**The correlation matrix of the menu strategies' bar-by-bar positions.** They are almost all
trend-following on the same price: if the average pairwise correlation is high, the vote is a single
opinion counted ten times, and no weighting scheme changes that. It is the cheapest measurement in
the plan and it can close it in an afternoon. **It has not been done yet**, and it is not derivable
from `reports/`, which holds summary rows and not series.

### Three nested versions, one extra degree of freedom each

Each is measured against the previous one, and one moves to the next **only** if it gains:

1. **V0 — equal-weight consensus.** k out of N strategies at fixed parameters (the
   `tuned_defaults`). A single parameter: k. References: each individual strategy, and — what counts
   — each individual strategy retuned to the same trading frequency.
2. **V1 — online weights.** Each strategy's weight exponential in its recent return (Hedge /
   multiplicative weights). A single parameter: the learning rate. The weights are produced by a
   rule, not by a search, and the theoretical guarantee is exactly the one needed given ρ = −0.69:
   asymptotically one does no worse than the best individual strategy.
3. **V2 — regime-conditioned weights** — the version requested. Only if V1 beats V0. This is where
   the parameter count explodes (a regime classifier × N strategies), and it is the version that
   ρ = −0.69 predicts will fail.

The **dynamic threshold** follows the same rule: not a number tuned per regime, but a scale-free
function (for example asking for more consensus when volatility is high), added one at a time and
measured as one extra degree of freedom.

### Is it a known idea?

Yes, and with precise names: prediction with expert advice (Hedge, multiplicative weights),
"sleeping" experts or specialists — which vote only in their own context, i.e. exactly "which
strategies are trustworthy in this regime" — universal portfolios, Markov regime-switching models,
stacking, and Lopez de Prado's meta-labeling, which in this repository is already implemented as
`scripts/meta_gate.py`.

The evidence closest to this project is already in the §1 artifact: in the qlib table the **top spot
for IC is DoubleEnsemble** (an ensemble) and the **top for Rank IC is TRA**, which is a router that
sends each sample to a different predictor — i.e. the learned version of "the market condition that
chooses the weights". The family is the right one. But the annual returns of those two rows are 11.6%
and 7.2%: the ensemble wins the ranking **and stays under the same ceiling**. It does not turn a
losing family into a winning one.

---

## What this plan does not do, and why

- **It does not integrate quant and ML.** The verdict of `ricerca-quant-ml.md` §6 holds: the filter
  did not pass its own random control, and composing an unproven layer with a proven one can only
  make the second worse. It reopens if the filter passes the control on a second cycle.
- **It does not widen the universe.** Measured: out-of-sample median from +62% to −0.9%.
- **It does not add the short side.** Measured at a loss on all five strategies.
- **It does not look for deep architectures.** The qlib benchmarks show them below gradient boosting,
  and the cross section is not exhausted.
- **It does not optimise the parameters.** That is the point of the whole plan.
