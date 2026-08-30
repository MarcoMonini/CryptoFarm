# The labeling strategy

> **Language rule.** Every document in this repository is written in English. See
> `CLAUDE.md` § *Documentation language*. This is not a style preference: it is a
> standing instruction, and it has been restated more than once.

Last measured: 2026-08-30. Code: `ml/labeling.py`, `ml/swing_trainer.py`, `trading/panels.py`.

## 1. In one paragraph

The label is a continuous number in **[-1, +1] that oscillates between local lows and local
highs**. It is `-1` on a local low, `+1` on the local high that follows, and it slides along the
leg in between. How far along the leg a bar sits is **not** decided by price alone: 70% of it is
decided by the **bars elapsed** since the previous extreme, and only 30% by how much of the price
move has been covered. That 70% is the **temporal smoothing**, `peso_tempo` / `TIME_WEIGHT = 0.7`.
The extremes themselves are found with a **window**: a bar is a high if its close is the highest
among the `W` bars on each side. Different windows produce different labels, and `W` is the knob
that selects which timescale of swing the model is asked to learn.

The function is `labeling.swing_leg_target(close, window, peso_tempo, saturazione)`.

## 2. Why the time weight is the whole point

Without it the label is a function of price position only, and a price that stalls produces a
label that stalls with it. A model trained on that learns to **follow** the price: it can only say
"we are low right now", which any Stochastic already says, and it says it at the same moment the
price does — too late to be worth a commission.

With the time weight in, a price that goes sideways at mid-leg **keeps advancing towards the
extreme that is coming**, because the bars are being spent. That is the part of the label that can
be anticipated, and it is the reason the target is not just a repackaged oscillator.

Concretely, at `peso_tempo = 0.7`, a price that has retraced half of the leg while consuming three
quarters of its bars sits at **0.68** of the scale rather than 0.50 — the label already calls it
close to the coming extreme.

| `peso_tempo` | what the label is | why it is not used |
|---|---|---|
| 0.0 | pure price position along the leg | the label follows the price; this was the defect of the centered-rank version |
| **0.7** | **70% elapsed bars, 30% price** | **in use, for training, chart and measurement** |
| 1.0 | linear ramp in time between extremes | ignores that price can retrace; the leg becomes a metronome |

`tests/test_swing_target.py::test_il_tempo_conta_quanto_il_prezzo` pins the behaviour: on a
synthetic series where price freezes at the top for 50 bars, the label with time weighting keeps
rising monotonically and the one without does not move at all.

## 3. How the extremes are found, and why they alternate

`labeling.swing_pivots(close, window)` returns the extremes **already alternated**, with their
direction.

1. `scipy.signal.argrelextrema` gives highs and lows separately, with `order=window`.
2. Bars in the first and last `window` positions are **dropped**. At the edges `argrelextrema`
   compares against clipped indices, i.e. against the bar repeated, so it declares extremes that
   the future has not confirmed yet. That would be the only true look-ahead in the whole label.
3. The two sequences are merged and sorted. Nothing guarantees they alternate — two highs in a row
   with no low between them are common once the window is wide — so when two extremes of the same
   direction follow each other, only the more extreme of the two survives. The other is not the
   vertex of any leg.

The result is a strict low, high, low, high … sequence, which is what makes "the leg between one
extreme and the next" well defined.

## 4. The value at an extreme is not always ±1

An extreme is worth `±tanh(prominence / reference)`:

- **prominence** is the *smaller* of the two legs touching that extreme. A vertex reached by a
  huge rise and left by a tiny fall is not a vertex, it is a pause;
- **reference** is `σ · √window · saturazione`, i.e. how far a random walk with the local
  volatility would travel in `window` bars. `σ` is a causal rolling standard deviation of log
  returns ending on the extreme.

This is what makes the label comparable **across the fifteen symbols and across the years**: the
same 5% leg is enormous on a stablecoin and noise on an altcoin in 2021. A swing inside the local
noise scores around 0.3; a real leg saturates towards 1.

`saturazione` (default 1.0) scales the reference: raising it demands bigger legs for the same
value.

## 5. Windows: which ones, and where

"Different time windows" is a real degree of freedom, and there are three distinct places where a
window is chosen. They are not the same number and are not required to be.

| where | window | on which bars | meaning |
|---|---|---|---|
| `swing_trainer.W` | **144** | 5m | half a trading day each side — the swing the model is trained to see |
| `config.SWING_TARGET_WINDOW` | **50** | the chart's interval | what the chart draws, at whatever timeframe is on screen |
| `bar_features` aggregation | 1h, 1d | — | feature scales, not label windows |

The trainer's `W = 144` is a floor-and-ceiling choice: below it the target chases five-minute
noise, above it the embargo eats the sample and the extreme stops being *local*. The chart's
default of 50 is deliberately smaller because the chart is usually looked at on 15m–4h bars, where
50 bars is already half a day to a week.

Both are exposed: `--w` on the trainer, a sidebar field on the page. When the two disagree with
the artifact on disk, the page says so in orange rather than drawing a curve the model never saw
(`signals.swing_etichetta_addestrata`).

## 6. What the label costs: look-ahead, embargo, and the empty tail

The label **looks ahead to the next extreme**. That horizon is *variable* and is **not** bounded by
`window` — it is at least `window` bars and usually more. Three consequences, all of them load
bearing:

- **the tail is empty.** Bars after the last confirmed extreme come out `NaN`. No strategy can
  trade this label, and it is not meant to be traded: it is what the model is trained to predict,
  not a signal;
- **the train/test split needs a real embargo.** `swing_trainer.EMBARGO_FINESTRE = 3` puts three
  windows between the end of the training set and the start of the out-of-sample set. One window —
  which was enough for the centered rank, whose horizon stops exactly at `W` — would leave the last
  training rows with a target that has already read inside the test set, and the out-of-sample
  number would come out inflated;
- **the target must never be a feature.** Delayed by one bar it shares 143 of its 144 bars with
  today's target, and the IC jumps to +0.67. That number is the measurement of the leak, not a
  result.

## 7. The measuring stick is not the label

This is the subtlest part of the design and the easiest to get wrong.

The label uses the **past** as well as the future — where the leg started is behind the bar. The
features already describe the past: a plain Stochastic with no model at all scores **IC 0.70**
against the full centered rank. So a model scored against its own training target is being measured
mostly on its ability to reproduce a Stochastic, and it will look excellent while predicting
nothing.

Every figure the trainer prints is therefore scored against **`swing_target(..., verso="avanti")`**
— the forward-only half, the part that genuinely cannot be known. There the causal reference
(`pos_canale`, no model) is what has to be cleared: **+0.046** in validation, **+0.0296** out of
sample.

The trainer prints the honest number first, the causal reference next to it, the excess, the
per-symbol median, and only then the IC against the full target, labelled as not being a result.

**Measured, 2026-08-30**, 15 symbols at 5m from 2018, `W = 144`, `peso_tempo = 0.7`, out-of-sample
cut at 2024-01-01 with a 432-bar embargo:

| | with smoothing (0.7) | previous label (centered rank) |
|---|---|---|
| IC against the forward half | **+0.0433** | +0.0405 |
| causal reference (`pos_canale`) | +0.0296 | +0.0297 |
| **excess over the reference** | **+0.0137** | +0.0108 |
| per-symbol median | +0.0500, 14/15 agreeing in sign | — |
| IC against the full target | +0.5036 (not a result: 93% of it is the past) | — |

The time-weighted label predicts the future half slightly better than the rank did, and it does so
while being the label the chart actually draws. It is an improvement in the honest number, not a
change of verdict: the swing model still does not clear the commission (`modello-swing.md` §5).

## 8. Where the same label is used, and why they must not drift

Three consumers, one constant:

| consumer | code | how it gets the value |
|---|---|---|
| training | `ml/swing_trainer.py` | `PESO_TEMPO = TIME_WEIGHT` |
| chart | `trading/panels.py` via `config.SWING_TARGET_TEMPO` | copied literal, pinned by a test |
| self-check / tests | `tests/test_swing_target.py` | `TIME_WEIGHT` |

`trading/config.py` deliberately imports nothing, so the value is **copied** there rather than
imported, and `test_la_pagina_parte_dallo_smoothing_con_cui_si_addestra` fails if the copy drifts.

**This is not hypothetical.** Until 2026-08-30 the trainer learned `labeling.swing_target` — the
centered rank inside a fixed window, which has no time weighting by construction — while the page
drew `swing_leg_target` with the smoothing at 0.5. Two different labels under one name: the
sidebar knob could be moved all day without changing anything the model had learned, because the
model had never seen a time-weighted label at all. `test_il_trainer_etichetta_con_le_gambe_e_non_col_rango`
exists to stop that specific regression from coming back.

## 9. The competing labels, and where they stand

`ml/labeling.py` holds three families. They are not interchangeable and each answers a different
question.

| label | question | used by |
|---|---|---|
| `triple_barrier_labels` | "does price move 1.5 ATR up before 1.0 ATR down?" | `trainer.py` (`signal_model`), `meta_trainer.py` |
| `swing_target` | "where does this bar rank among its neighbours?" | the **yardstick** only, as `verso="avanti"` |
| **`swing_leg_target`** | **"where along the leg between two extremes are we, in price and in time?"** | **`swing_trainer.py` (`swing_model`), the chart** |
| `rendimento_futuro` (in `entry_trainer.py`) | "what does buying here return over H bars?" | `entry_model`, `entry_model_veloce` |

An honest note that belongs next to this strategy rather than buried: on 2026-08-29, at equal
selectivity, the leg label identified true lows **better** than the direct forward return (37.2%
vs 23.0%) and made **2.4× less money** (+0.025% vs +0.059% per signalled bar). Picking better lows
and making more money are not the same objective. The leg label is the shape of the market; the
forward return is the till. Both are trained here, and `entry_model_veloce` is still the artifact
that leads `MODEL_PRECEDENCE`.

## 10. Reproducing

```bash
.venv312/bin/python -m cryptofarm.ml.swing_trainer --selfcheck   # no store needed
.venv312/bin/python -m cryptofarm.ml.swing_trainer               # ~12 min, 15 symbols from 2018
.venv312/bin/python -m cryptofarm.ml.swing_trainer --w 72 --peso-tempo 0.5   # another window/weight
.venv312/bin/python -m pytest tests/test_swing_target.py -q
```

The artifact's `labeling` metadata block records `method`, `window`, `peso_tempo` and
`embargo_finestre`, so a saved model always says which label it was trained on.
