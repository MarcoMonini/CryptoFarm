# The swing model — from the triple barrier to the shape of the extremes

**Date:** 2026-08-28. **Status:** superseded on 2026-08-29 by `entry_model_veloce`
(`modello-ingresso.md`), which sits ahead of it in `MODEL_PRECEDENCE`. This model stays in the chain
below it and serves the page again if the entry artifacts are missing. The document remains valid as
a measurement: it is what established that the signal exists but does not beat chance at matched
exposure, and that is where the new question came from.

**Updated 2026-08-30**: the label the model is trained on is now `labeling.swing_leg_target` with
the temporal smoothing at 0.7 — see §2 and `.claude/docs/labeling-strategy.md`. The verdict does not
change; the honest out-of-sample number improves slightly.

Wired in at the time *knowing* what §5 says. What was wired in, and what deliberately was not, is in
§5.4. **The new model does not replace it by being better but by asking a different question**: at
equal selectivity the leg label identifies lows better than the direct forward return (37.2% against
23.0%) and returns 2.4× less.

This document closes three things in one session: the audit of the previous model (`leg_model`), the
replacement of the labeling, and the measurement that says what to do with it. The three parts
should be read in that order, because each is the reason for the next.

---

## 1. Why `leg_model` left the chain

> **The code in this section no longer exists (deleted 2026-08-30).** `ml/leg_trainer.py`, the
> `signals.leg_signals` function and the dispatch branch that served it were removed: a
> `leg_model.joblib` left on disk no longer brings the page back to this model, and a test verifies
> that. `labeling.swing_leg_target`, on the other hand, **stays** — it is the label
> `swing_trainer.py` trains on today and the one `modello-ingresso.md` uses as a comparison. The
> measurement below still holds and is why the section stays written.

A reviewer working in a fresh context, without the author's conclusions, examined
`leg_trainer.py`, `bar_features.py`, `positioning.py`, `leg_signals` and `barrier_widths` against a
nine-requirement contract. Out of some twenty findings the noise was **zero**. The four structural
ones are the same defect seen from four sides — *the validation loop was not measuring the strategy
being shipped*:

| # | finding | verified |
|---|---|---|
| 1 | The random control samples i.i.d. rows from a population where each row overlaps the next by 7/8 of the horizon and 15 symbols share the timestamp. With a weekly block bootstrap the p95 goes from −0.22 to **+0.06** and the model drops to the 80th percentile: it does not pass. | yes |
| 2 | Both thresholds are chosen **on the verification sample** (`idxmax` of the sweep over `fuori`, quantile of the predictions on `fuori`). | yes |
| 3 | The `PASSA` verdict settles for beating a p95 that is itself negative. The average net per entry is **negative at all six thresholds** (−0.149 … −0.091). | yes, read from the metadata |
| 4 | The sweep optimises `rendimento_%`, which closes at +1.5 ATR — i.e. **with** the take profit, the variant the author's own measurement rates worst of six. And the exit head is never evaluated against a P&L. | yes |

`percentile: 100.0` repeated six times out of six was the alarm bell: a control that always answers
the same thing is not measuring anything.

Two contamination defects confirmed by hand, both one-liners:

- **`forza_su_btc` is an exact identity marker for BTC.** It equals `0.0` on 19,691 BTC rows out of
  19,691, and never on ETH. A single split in a tree isolates BTC perfectly — which is exactly what
  the module declares it wants to avoid in its opening docstring.
- **`sopra_ema200` lied on the warm-up bars.** `NaN > x` is `False` and `False.astype(float)` is
  `0.0`: all 199 bars before the EMA200 exists said "below the EMA200". It is the same defect
  already fixed for `atr_rel` **one line above**, left standing one line below. On the page it is
  worse than in training, because the loaded window is short.

`sopra_ema200` fixed; `forza_su_btc` fell with the cross-sectional columns (§3). `leg_model` is out
of `MODEL_PRECEDENCE` with the reason written next to the constant.

**Consequence for the AUC.** The 0.5639 declared "the highest ever produced by the project, above
the ~0.54 ceiling" is not a result: **it is an alarm**, and the two defects above are concrete
candidates to explain it.

---

## 2. The new question

The triple barrier asks *"does the price move 1.5 ATR within the horizon?"*. That is a question
about **volatility**, and since the barriers are already scaled on the ATR the label normalises away
precisely the predictable part. Measured: future amplitude has |IC| 0.42 with 10/10 assets agreeing,
direction 0.06.

The swing label asks instead *where this bar sits between the local extremes around it*: a value in
`[-1, 1]`, −1 on a local low, +1 on the high that follows, and it slides along the leg in between.

**That last property is the point.** Inside a steady rise the label does not sit at +1 for the whole
climb: it saturates only where the rise runs out. A "maximum of future prices", or a distance from
that maximum, would mark the entire climb as *near the high*, and that is what makes those labels
unusable.

Two forms of this label exist, and the distinction matters:

- **`swing_target`** — the centered rank of the close among the `W` bars on each side. It is
  implemented with two causal rollings instead of a centered window: the backward one gives the
  position among the `W` preceding bars, the one on the reversed series among the following, and the
  sum minus one is the exact centered rank. It costs `O(n log W)` instead of materialising
  `n × (2W+1)` values, which at 5m across fifteen symbols does not fit in memory. **It has no
  temporal component by construction**, and today it is used only as the yardstick (§2.1);
- **`swing_leg_target`** — the leg between one extreme and the next, where the position along the
  leg is **70% elapsed bars and 30% price** (`labeling.TIME_WEIGHT = 0.7`). This is the label the
  model is trained on and the one the chart draws. The full treatment is
  `.claude/docs/labeling-strategy.md`.

The time weight is what makes the label *lead* the price rather than follow it: a price that stalls
mid-leg keeps advancing towards the extreme that is coming, because the bars are being spent.
Without it, the target is reproducible by a Stochastic — which is exactly the measurement in §2.1.

### 2.1 Half the target is free, and the yardstick has to know it

The centered rank also uses the `W` **past** bars, which the features already describe:

| | IC against the full target | IC against the forward half alone |
|---|---|---|
| a **Stochastic** (past rank, zero model, zero future) | **+0.703** | +0.050 |
| the gradient-boosted model | +0.670 | +0.054 |

93% of the target is reconstructible from the past. Scoring there mostly measures how well the model
reproduces a Stochastic — something it **loses** at. Hence the `verso` parameter of `swing_target`,
and the rule that every figure is measured against `verso="avanti"`.

A useful note: training **on the centered target** and measuring on the future gives 0.053; training
directly on the future alone gives 0.032. The past half acts as a regulariser.

---

## 3. The design decisions, all measured

Seven symbols, verification 2024–2026, Spearman IC against the forward half:

| variant | columns | IC |
|---|---|---|
| `pos_canale` alone, no model | 1 | +0.0433 |
| 5m base | 15 | +0.0502 |
| + explicit history at −1 and −2 bars | 45 | +0.0509 |
| + history up to −8 bars | 75 | +0.0510 |
| + history up to −32 bars | 105 | +0.0498 |
| + target delayed by `W+1` (the only legitimate delay) | 16 | +0.0497 |
| **+ 1h and 1d aggregation** | **41** | **+0.0540** |
| + all four scales (15m, 1h, 4h, 1d) | 67 | +0.0539 |
| + target delayed by **1** bar | 16 | +0.6729 |

**No explicit history.** Copying the features backwards costs three times the columns for two
thousandths, and past two bars it gets worse. The history is already there, compressed into EMA200,
ADX and OBV over 20 bars.

**No target among the features.** At the only legitimate delay it is worth −0.0005. The last row of
the table is not a result but **the measurement of the damage**: at delay 1 the target shares 143 of
its 144 bars with today's, so that +0.67 is the information leak. It is worth keeping written down
because it is the most dangerous idea in the whole design: it would produce a model that is
spectacular in a table and useless in production.

**No cross-sectional columns.** They depend on the other fourteen assets and the page loads one
symbol at a time. `forza_su_btc` fell with them (§1).

**Aggregation at 1h and 1d**, aligned with `mtf.align_to_lower` so the long bar is only read after it
has closed. 15m and 4h add nothing: they sit too close to what EMA200 and ADX on the base already
describe.

---

## 4. The trained model

Fifteen symbols, 5m **from 2018** — that is where the store reaches, and positioning stays NaN
before 2021-12, which teaches the model the "positioning absent" state, i.e. the production
condition. Roughly 1.25 million training rows against 691 thousand for verification.

The number of reinforcement rounds — retraining on labels revised with the model's own **out-of-fold**
predictions — is chosen on a validation slice carved out of the training set, never on the
out-of-sample set. It is the direct correction of finding §1.2.

Between training and out-of-sample there is an embargo of **three windows** (`EMBARGO_FINESTRE = 3`,
432 bars). One window was enough for the centered rank, whose horizon stops exactly at `W`; the leg
label looks ahead to the next extreme, which is further away and variable.

```
round 0: validation IC +0.0790  →  round 3: +0.0835   (chosen: 3)

Out of sample 2024-01 .. 2026-08
  IC against the forward half   +0.0433
  causal reference              +0.0296   (pos_canale, no model)
  excess                        +0.0137
  per-symbol median             +0.0500   (14/15 agreeing in sign)
```

For comparison, the same trainer on the centered rank (the label in use until 2026-08-30) gave IC
+0.0405 against a reference of +0.0297, i.e. an excess of +0.0108. The time-weighted label predicts
the future half slightly better, and it is the label the chart actually draws.

Reinforcement works but only a little: **+0.0045 over three rounds**.

**Robustness — and this is the real improvement over `leg_model`:**

| scenario | IC |
|---|---|
| everything present | +0.0540 |
| without `@1d` (short page window) | +0.0524 |
| without `@1h` and `@1d` | +0.0542 |
| without positioning | +0.0539 |

It does not degrade. The previous model without the cross-sectional columns collapsed from +1.9% to
−39.5%.

**Inference: 283 ms for 20,000 bars** (234 for features, 49 for prediction).

---

## 5. Why it is not wired in as a directional rule

`scripts/swing_lab.py` makes three measurements, and each decides whether the next one makes sense.

### 5.1 The shape of the signal is U-shaped, not monotone

Excess 48-hour return by prediction decile:

| window | deciles 0 → 9 |
|---|---|
| validation | **+0.184** −0.096 −0.030 −0.010 −0.063 −0.068 −0.000 +0.067 **+0.093** −0.076 |
| out of sample | **+0.088** −0.093 −0.087 −0.098 −0.093 −0.107 −0.021 +0.079 **+0.180** +0.152 |

Both the lowest decile — which the model reads as "near a local low" — and the highest ones — "near
a high" — precede above-average returns; the middle sits below. It replicates in both windows.

**The model does not predict direction, it predicts structure.** The +1 pole is not "sell": it is
"strong trend in progress", and in crypto continuation pays. Selling the predicted highs — the
natural reading of a target in `[-1, 1]`, and the one the specification asked for — **sells exactly
the best bars**. Hence the P&L of the directional rule: it loses at every threshold and every
cadence, in validation as out of sample, from −0.05% to −0.42% net per trade.

### 5.2 The rule the shape supports is an exposure filter

In when `|prediction|` is high, out when it is low, with hysteresis.

| window | configuration | net/trade | compounded | passive |
|---|---|---|---|---|
| out of sample | 0.50 / 0.40 / 288 | **+0.086%** | −15.3% | −33.5% |
| validation | the same | −0.194% | −9.7% | +43.7% |
| validation | 0.35 / 0.25 / 288 | +0.311% | +1.8% | +43.7% |
| out of sample | the same | −0.191% | −57.0% | −33.5% |

**No configuration works in both windows**, and picking the one that works out of sample would be
tuning on the verification sample — finding §1.2 again.

### 5.3 The control that settles it

Staying out of the market 76% of the time beats passive holding inside a bear market **by
construction**. The real question is whether it beats placing the same exposure, with the same
durations, at random. Two hundred draws per symbol:

> **1 symbol out of 15 in validation, 1 out of 15 out of sample** exceeds the p95, against **0.75**
> expected from chance.

The rule's merit is abstention, and no model is needed for that.

### 5.4 What was wired in, and what was not

The model was at the head of `MODEL_PRECEDENCE` at the time and votes in Confluence. The three
measurements above did not become favourable: what changed is that **the wrong reading is no longer
reachable from the code**. Before, the risk was that someone would read a target in `[-1, 1]` and
wire in the sign; now the only road that exists is `|prediction|`, and the three docstrings that
implement it say why.

| where | what is wired in | what is **not** |
|---|---|---|
| `ml/signals.swing_exposure` | high `|prediction|` → in, with hysteresis | `sign(prediction)` as direction (§5.1: it loses at every threshold) |
| `trading/strategies.ai_model_simulation` | the exit is the entry read backwards | barriers, take profit, stop: the model was measured with none of the three |
| `trading/confluence._modello` | a +1 or 0 vote, in a family of its own | the −1 vote, which would go short on the best bars |

Three protocol choices, all taken so as not to repeat §1.2:

- **the thresholds are 0.35/0.25**, chosen on validation. Out of sample they return −0.191% per trade
  against the +0.086% of 0.50/0.40. Taking the latter *because* they return over 2024-2026 would be
  tuning on the verification sample, i.e. the defect that got `leg_model` removed. In Confluence they
  are two knobs (`CONF_MODELLO_ENTRA`/`ESCI`) because §5.2 measures that the good pair changes from
  one window to the next: keeping it in a constant would suggest a right one exists;
- **without an artifact the voter stays out of the default set**, not merely silent. Weights are
  normalised over the voters present, so an eighth one that is always silent would effectively raise
  the threshold for the other seven — and in production `models/` is empty by construction. It stays
  in the registry, so `selezione("modello")` can reach it for measurement;
- **the panel's caption says it does not beat passive holding.** It is the only part of this document
  that reaches whoever is looking at the chart.

Two things emerged while writing the serving path, and neither was visible from reading the trainer:

- **long scales must only be taken if they are longer than the base.** At 4h, aggregating to an hour
  means resampling upwards, i.e. inventing bars. The columns left out become NaN, and that is the
  degradation already measured in §4;
- **and only if they have at least 28 bars.** `ExtraCache.adx(14)` goes through `ta`, which below two
  windows raises `IndexError` instead of returning NaN. It is invisible in training — the series are
  hundreds of thousands of bars long — but the page loads 240 hours by default, i.e. ten daily bars,
  and there the "AI Model" entry fell over as soon as it was selected.

**Measured after wiring in**, and to be read as confirmation of §5.3 and not as a result: BTC at 1h
from 2025, 104 trades, −21.1% against a passive −27.2%. That is the merit of abstention. Inside
Confluence, over 92,321 15m bars from 2024, the voter is long 56.4% of the bars, never short, and
**necessary in 10% of entries**: it adds without dominating, which is the only condition under which
adding it was worth it.

---

## 6. What remains true

- **The statistical signal exists**: IC +0.0433 out of sample against a causal reference of +0.0296,
  14/15 symbols agreeing in sign, and a U shape replicated in two disjoint windows. It is not noise.
- **It is not profitable at these frequencies.** The best measured excess is +0.20% over 48 hours
  against a round trip of commissions costing 0.20%. It is the confirmation tax of `strategy.md`
  §13, for the third independent time in this project.
- **(c) has been done** (§5.4): the model is a Confluence voter, where it does not have to beat
  passive holding on its own. Whether it pays is not measured yet — it needs the
  `scripts/confluence_lab.py` grid redone with and without the voter, on the same assets and the same
  window. Until that comparison exists, the only thing known is that the voter does not dominate the
  decision.
- **Two roads remain**, in order of cost: (a) use `|prediction|` to **size** the position rather than
  as a switch — the only form that does not truncate the right tail; (b) move the decision to a daily
  scale, where the ratio between excess and commissions changes by an order of magnitude.

## 7. Reproducing

```bash
.venv312/bin/python -m cryptofarm.data.positioning --update     # positioning store, 400 MB
.venv312/bin/python -m cryptofarm.ml.swing_trainer --selfcheck  # runs without the store
.venv312/bin/python -m cryptofarm.ml.swing_trainer              # ~12 minutes
.venv312/bin/python -m scripts.swing_lab                        # the three measurements in §5
```
