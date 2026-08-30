# The entry model: the question changed, and the first numbers that pass the control

**Status: wired in (2026-08-29).** It is the model at the head of `MODEL_PRECEDENCE`, and the two
artifacts are `entry_model_veloce` (which trades) and `entry_model` (which gates).

## 1. Where it comes from: a premise of the user's, verified

The previous model (`swing_model`, `.claude/docs/modello-swing.md`) predicted **proximity to local
extremes**. The user challenged the measurement that came out of it — +0.026% per signalled bar over
twenty bars — with an argument that was not about taste but about orders of magnitude: *"crypto
moves are bigger than 1% even in a sideways market, so this measurement does not hold up: it means
the local highs and lows are not really being identified"*.

Checking it proved them right about the fact and wrong about the cause, and the two must be kept
apart:

- **the ceiling was there.** Bars the *label* marks as the lowest decile return **+0.765%** over
  twenty bars, 87.1% of them upward. It is not a problem of move size;
- **the model caught a third of them.** Of the signalled bars, 30.3% were true lows (chance gives
  10%). On the intersection of signalled and true, the return is +0.786%, i.e. **as much as the
  oracle**: where the model is right, it is entirely right;
- **the rest dragged it down.** False positives return −0.30%, and the arithmetic works out:
  `0.303 × 0.786 + 0.697 × (−0.305) = +0.026%`.

So the signal was there and the margin was not. The question that follows is not "how do I make it
more precise", and that is where the direction was corrected.

## 2. Precision and money are not the same question

At equal selectivity (10% of bars, verification from 2024, 15 symbols):

| target | return of the signalled bars | is it really a low |
|---|---|---|
| leg label (`swing_leg_target`) | +0.025% | **37.2%** |
| within 10 bars of a low | +0.012% | 28.4% |
| low already reached within 10 bars | −0.012% | 22.7% |
| **direct forward return** | **+0.059%** | 23.0% |
| forward return / ATR | +0.012% | 12.8% |

The leg label **wins on precision and loses by 2.4× on money**. Asking "is this a low" and asking
"does it pay" are two different questions, and the second is the one that gets banked. The new
model's target is the log return of the next `H` bars, and nothing else.

Three roads that looked obvious were tried and brought nothing:

- **new features**: four families (wick rejection, leg exhaustion, volume capitulation,
  price/oscillator divergence) move precision from 30.0% to 31.3%. The 16 extra columns make the net
  *worse* (+1.046% against +1.188%), so `bar_features.py` was **not touched**: the same 41 columns
  as `swing_model` are served;
- **more data**: 4.4 million rows instead of 366 thousand give 30.6%;
- **more capacity**: more iterations make it worse, 29.0%.

The swing model was, when measured, **an oscillator**: its prediction correlated +0.932 with
`dist_ema50_atr`, and removing that column left +0.87 because the other forty are of the same family.

## 3. The lever is selectivity, not accuracy

The commission is fixed and the return is not. With the direct target, over 150 bars:

| bars signalled | average return of the signalled |
|---|---|
| 10% | +0.047% (below the commission) |
| 2% | +0.90% |
| 0.5% | **+2.07%** |

The model does not get better: it only trades where it says a lot. Hence the three choices in
`ml/entry_trainer.py`: **threshold from the training-set quantile** (taking it from the
out-of-sample set would be look-ahead), **fixed holding period** instead of a signal-driven exit,
**no overlap** — while in a position, subsequent signals are ignored, or one is measuring capital
that does not exist.

## 4. The control, which here is mandatory

Out of sample, median passive holding does **−34%**: a strategy in the market 17% of the time beats
that almost on its own. "It beats passive" is therefore not a result. The comparison is with
**random entries at the same count and the same holding period**, 200-400 draws, with the same
anti-overlap filter.

| model | trades | average net | profitable | chance | percentile | symbols |
|---|---|---|---|---|---|---|
| `entry_model` (H=150, hold 150) | 427 | **+1.529%** | 59.5% | +0.004% | 100th | 14/15 |
| `entry_model_veloce` (H=20, hold 20) | 223 | **+1.360%** | 63.2% | −0.173% | 100th | 12/15 |

It is the first result in this project that passes the matched-exposure control. The previous
families did not: `swing_model` 1 symbol out of 15, `leg_model` negative net at every threshold, the
RL policy only weakly (rank 0.588).

## 5. The two models compose in one direction only

The request was two complementary models, one narrow (10-20 bars, microstructure) and one wide (~150
bars, macro moves). The shape of the composition is the one the user indicated — **the fast one
makes the trades, the slow one says inside which moves it may make them** — and the measurement
confirms it. Verification from 2024, 20-bar hold, 0.2% commission, 200 draws for the control:

| slow model's gate | trades | average net | profitable | symbols | chance | percentile |
|---|---|---|---|---|---|---|
| none | 223 | +1.360% | 63.2% | 12/15 | −0.173% | 100th |
| training median | 161 | +1.806% | 65.2% | 12/15 | −0.143% | 100th |
| 80th of training | 156 | +2.019% | 65.4% | 13/15 | −0.161% | 100th |
| **90th of training** | **148** | **+2.071%** | **65.5%** | **14/15** | −0.165% | 100th |
| 95th of training | 128 | +2.243% | 65.6% | 13/15 | −0.170% | 100th |
| 98th of training | 100 | +2.464% | 68.0% | 13/15 | −0.172% | 100th |

**Why the 90th and not the 98th.** The curve is monotone: taking the highest value means picking the
maximum of the verification sample, which is the error already measured elsewhere in this project
(correlation −0.69 between in-sample and out-of-sample return on the rotation). The 90th is chosen on
**agreement across symbols**, 14 out of 15, which is the difference between a model and an episode.

The inverse — the slow one trading inside the fast one's indications — was not tried and makes no
operational sense: a 150-bar trade does not fit inside a 20-bar one.

`scripts/entry_lab.py` reproduces this table.

## 6. What was wired in

- **`MODEL_PRECEDENCE`**: `entry_model_veloce`, then `entry_model`, then the earlier families. The
  fast one goes first because it is what generates the signals; the slow one stays in the chain
  because on its own it is still a measured strategy.
- **`ml/signals.entry_signals`**: entry above threshold and with the gate open, exit after the
  holding period. Threshold, gate and holding period are read **from the artifact's metadata** and
  not from the widgets: they are the model, not two knobs. The sidebar's `threshold` does not enter
  this strategy, on purpose.
- **The holding period is a time, not a candle count.** 150 bars at 5m are twelve and a half hours,
  and at 1h they are still twelve and a half hours (13 bars).
- **The gate applies only on the entry bar.** An open position is not closed because the wide plane
  changed: truncating it would measure a different strategy.
- **The confluence's `modello` voter** votes +1 while an entry-model trade is open, 0 otherwise. It
  stays long-only, as before, and `entra`/`esci` have no effect: the selectivity lives in the
  metadata.
- **The panel's caption** states the number that matters and what it is measured against.
- **The page lets you choose which of the two**, `Fast (trades)` or `Slow (gates)`, and the choice
  reaches the strategy as `ai_model_simulation(..., famiglia=...)`. It is not a tuning: they are two
  strategies, and put together you cannot see how they differ. The other families are still chosen by
  `MODEL_PRECEDENCE`, i.e. by moving artifacts around.
- **The "Entry model: prediction vs realised return" panel** puts on the same axis what the model
  predicts for each candle and what it was taught — the return of the next `h` bars, which looks
  forward, so the tail comes out empty — plus the buy threshold. Same axis because it is **the same
  unit**: the swing target lives in [-1, 1] and would flatten returns of the order of a hundredth,
  which is why it stays in a panel of its own.
  What one should expect to see, measured on five symbols from 2024: the two curves **do not look
  alike** (rank IC +0.0074 for the fast one, +0.0223 for the slow one) and the mean over all bars is
  −0.004%, but above the threshold the average realised return is **+1.99%**. A flat, uncorrelated
  blue line is not a fault: it is the model staying silent, and that is where the edge comes from.

## 7. What is still open

- **The confluence grid with and without the `modello` voter**, on the same assets and the same
  window. It is the only way to say whether the model adds to the confluence strategy or merely
  reduces its exposure. The same holds for the RL policy, and it has not been done for either yet.
- **The block control.** The random control samples rows that overlap each other and across symbols;
  on `swing_model` a weekly block bootstrap had moved the verdict. Here the holding period is shorter
  and the trades are few, but the measurement has to be redone in that form before saying
  "significant".
- **Below 5 minutes there is no measurement.** The model is trained and measured at 5m. Above half an
  hour the point has been measured and closed: see §8.

## 8. Two things measured after wiring it in

### 8.1 Trading more often costs, and by how much is known

The question was explicit: trades are wanted on the short intervals, and the model as served makes
few of them. Making more is possible, and it has a price. Here **only** the fast model's threshold
moves — taken on the training sample, not on the out-of-sample one — with the slow model's gate held
at the 90th:

| bars marked | trades | average net | profitable | symbols profitable |
|---|---|---|---|---|
| 5.0% | 1,570 | +0.166% | 46.9% | 10/15 |
| 2.0% | 677 | +0.523% | 51.8% | 13/15 |
| 1.0% | 330 | +1.068% | 59.7% | 13/15 |
| **0.5%** | **148** | **+2.071%** | **65.5%** | **14/15** |
| 0.1% | 23 | +3.710% | 73.9% | 8/11 |

All at the 100th percentile against random entries at matched exposure. Three readings:

- **the commission is fixed at 0.2%, the return is not.** Lowering the threshold does not add trades
  at the same return: it adds worse ones, monotonically and steeply;
- **the accumulated total peaks elsewhere.** Sum of per-trade returns: 261 at 5%, 354 at 2%, 352 at
  1%, 307 at 0.5%. Whoever wants more trades *can* have them — 330 instead of 148 — and the
  accumulation does not get worse. What gets worse is the quality of the individual trade and the
  agreement across symbols;
- **0.5% remains the choice**, and for the same reason as the gate: 14/15 symbols profitable is the
  column's maximum, and this project has already measured that chasing the return maximum transfers
  worse (correlation −0.69 between in-sample and out-of-sample on the rotation).

`scripts/entry_lab.py --frequenza` reproduces the table.

### 8.2 The threshold is a return, and above half an hour it no longer means the same thing

`entry_signals` scaled the holding period with the interval but served the **same absolute
threshold** everywhere. The threshold, however, is not a quantile: it is the predicted return above
which one enters, and the model predicts the return of the next **twenty five-minute bars**. On
longer bars those twenty bars are a different horizon, the predictions grow, and the threshold stops
selecting the same bars. Measured on five symbols from 2024:

| interval | 5m | 15m | 30m | 1h | 4h | 1d |
|---|---|---|---|---|---|---|
| bars marked | 0.063% | 0.270% | 0.722% | 2.98% | 14.1% | 28.1% |

At 1d the "selective" model was marking one bar in four: the opposite of what it was chosen for, and
the +2.07% per trade no longer described anything. Hence `signals.entry_fuori_misura`, which serves
the model **up to 30 minutes** and above that stays silent saying why — the same shape as
`confluence.scala_fuori_misura`. The `modello` voter does not go silent out of scale: it falls
through to the next entry in `MODEL_PRECEDENCE`.

A reading note for the chart: at 5m the threshold marks 0.063% of bars, i.e. **one in sixteen
hundred**. A 240-hour window does not contain enough bars for a trade to be likely, and on BTCUSDT —
the symbol with a single trade in the whole out-of-sample set — the maximum prediction over 2,880
bars stays below the threshold. Zero trades on the chart is the expected behaviour, not a fault.
