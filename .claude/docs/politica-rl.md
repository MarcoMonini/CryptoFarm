# The RL policy: the cost inside the reward

*Measured 2026-08-28, 15 symbols, 5-minute bars from 2019.*
Code: `ml/rl.py`, `ml/rl_trainer.py`, lab `scripts/rl_lab.py`, serving `ml/signals.rl_*`.

**Status: wired in.** `rl_model` is at the head of `trainer.MODEL_PRECEDENCE`, the "AI Model" entry
runs it and the confluence's `modello` voter uses it when the artifact is there.

---

## 1. The starting question, and why the obvious answer was wrong

The request was: *the model makes good trades but does not avoid the crashes, it buys shortly
before them, and the secret to beating passive holding is avoiding it while keeping the behaviour on
the sideways stretches and exploiting the upward legs.*

Both premises are measurable. Previously wired rule (`|prediction| ≥ 0.35` to enter, `< 0.25` to
exit, one decision per day), 15 symbols, out of sample from 2024-01.

### 1.1 "It buys shortly before the crashes" — **false**

Maximum drawdown in the three days after an entry, against the one after any daily bar in the same
period:

| | n | median | p10 | p05 | share below −10% |
|---|---|---|---|---|---|
| after an entry | 3,000 | −3.88% | −11.87% | −15.82% | 15.1% |
| after any bar | 14,385 | −3.94% | −11.84% | −15.72% | 14.7% |

The two distributions are indistinguishable. Entries do **not** land in front of crashes more often
than chance does. What was visible on the page was true — there are entries followed by crashes —
but it is the base rate, not a signature of the model.

### 1.2 "Cut the crashes with a stop" — **harmful**

Same entries, early exit from a stop. Sum of net out-of-sample returns:

| exit | sum | trades stopped out | worst |
|---|---|---|---|
| model only | **−201%** | 0% | −33.8% |
| fixed stop 3% | −229% | 52% | −3.2% |
| fixed stop 5% | −314% | 34% | −5.2% |
| fixed stop 8% | −327% | 18% | −8.2% |
| trailing 5% | −563% | 57% | −5.2% |
| trailing 12% | −418% | 15% | −12.2% |

Every level makes it worse, and worse **monotonically as it bites harder**. The stop cuts the right
tail more than the left one: it is the same thing the U shape of the swing model says
(`modello-swing.md` §5.1), seen from another angle.

### 1.3 Where the money actually goes — **into the commission**

| | gross | net | trades | implicit costs |
|---|---|---|---|---|
| out of sample | **+401%** | **−201%** | 3,009 | 3,009 × 0.2% = **602%** |

The signal is there on a gross basis. The number of round trips eats it. The clearest confirmation:
taking the *same* rule, *the same model*, from one decision a day to one every two days, the net
goes from **−201% to +306%**. No training, no new features: half the round trips.

**Hence the shape of the agent.** Not a risk filter — measured harmful — but a policy that chooses
the position knowing what changing it costs.

---

## 2. The formulation

State `s` = the same 41 columns as the swing model (`bar_features.SWING_COLUMNS`) plus the
**current position**. Action `a ∈ {out, in}`. One-step reward:

```
r(s, p, a) = a · log(P'/P) − cost · |a − p|
```

Three consequences, and they are the three reasons this shape differs from the ones already closed
with a negative result:

- **the do-nothing band is not hand-written.** The two thresholds of the previous rule were a
  parameter; here the hysteresis is what emerges when changing costs and the position is in the state;
- **the policy class contains passive holding** (`a ≡ 1`), which is the benchmark to beat. The agent
  can only add to a policy it can already represent;
- **the economic constraint sits inside the target.** It is the only reformulation `strategy.md`
  §13.4 left open after the three-action policy turned out to be zero-sum by construction: that one
  entered on the confirmation of a low and exited on the confirmation of a high, and the confirmation
  is paid twice against a median leg worth 1.76-2.05 thresholds.

**Algorithm:** offline fitted Q-iteration on a fixed batch — no interaction with the environment, so
no distribution shift from an exploring policy. Two `HistGradientBoostingRegressor`, one per action,
on the state `[features, position]`. The target of each round is `r + γ · max_a' Q(s', a')`, where
`s'` carries **the action just taken** as its position: it is that link that makes the cost an
investment rather than an instantaneous tax.

### 2.1 The three constants, and who chose them

They live in `ml/rl.py` and are **not re-chosen at every training run**: the grid that fixed them
(12 cells) was walked once, in validation.

| constant | value | why |
|---|---|---|
| `CADENZA` | 288 bars = 1 day | the cadence the previous rule is measured at |
| `FEE` | 0.001 per side | the true cost, the one it is **measured** with |
| `COSTO` | **0.012** | the cost the agent sees, twelve times the true one |
| `GAMMA` | 0.95 | ≈ twenty days of horizon at a daily cadence |

`COSTO ≠ FEE` is not a mistake. It is the term that widens the do-nothing band, chosen in validation
among 0.001 / 0.004 / 0.012: with the true cost the policy turns over 203 times in two and a half
years and stays below chance, at 0.012 it turns over 184 times and gets above it. Whoever changes it
should know they are changing the problem.

### 2.2 The defect that removed 85% of the sample

The NaN filter discarded the **whole** row if one column was missing. The two positioning columns
(`data/positioning.py`) do not exist for entire years on symbols that entered futures late, so the
training sample dropped from 165,605 rows to 29,234 — and what disappeared were precisely the early
years, i.e. the only complete cycle inside the training period.
`HistGradientBoostingRegressor` handles NaN by itself. The condition is now `any`, not `all`, and
there is a test.

---

## 3. The three periods

| period | when | what it is for |
|---|---|---|
| training | 2019-01 → 2022-06 | trains the two regressors (121,806 transitions, 8 offsets) |
| validation | 2022-06 → 2024-01 | **chooses the number of iteration rounds** (3 out of 1/2/3/5) |
| out of sample | 2024-01 → today | looked at once and decides nothing |

Validation starts at 2022-06 on purpose: it contains the **2022 bear market and the 2023 rally**.
Choosing inside a single regime would have chosen "stay out of the market", which always wins in a
bear market and is not a skill. Between training and validation there is an embargo of
`144 + cadence` bars — 144 being the swing target's window.

---

## 4. The results, and the right yardstick

`python -m scripts.rl_lab`. **The control is tighter than `swing_lab`'s**: instead of placing
similar durations at random, it shuffles the **policy's own blocks**. Total exposure, number of
blocks and their durations stay identical; only *where* they fall changes. What remains is exactly
the value of the *when*.

And what is measured is the **percentile rank** among 400 draws, not how many times p95 is exceeded:
with 15 symbols the count throws away almost all the information, and 0.75 expected successes
against 2 observed distinguish nothing. The expected mean rank, if timing does not matter, is
**0.500**.

| | beats holding | mean rank | Wilcoxon | max drawdown | avg exposure | exposure in the 10 worst steps |
|---|---|---|---|---|---|---|
| validation | 9/15 | 0.588 | p=0.277 | **−40.8%** against −58.3% | 39% | 48% |
| out of sample | 11/15 | 0.602 | p=0.169 | **−48.8%** against −76.0% | 37% | **25%** |

### 4.1 What they say, read honestly

- **The maximum drawdown halves in both windows.** It is the most solid and most consistent result,
  and it is the operational translation of the starting question: the capital does not avoid the
  crashes by betting on *when* they arrive, it goes through them half exposed.
- **The *when* is above chance in both windows** (0.588 and 0.602 against 0.500) but **does not reach
  significance** in either. The sign is consistent, the strength is not. It is still more than the
  previous rule had, which was at chance level.
- **It beats passive holding 11/15 out of sample, 9/15 in validation.** The second number is a coin
  flip, and that should be said: in a rising market a policy exposed 39% of the time does not keep
  up. The advantage concentrates where passive holding loses.
- **Exposure conditional on crashes is inconsistent**: 25% against a 37% average out of sample (the
  policy *is* less exposed in the worst steps), but 48% against 39% in validation. It is an ex-post
  computation and describes a behaviour, it does not declare a capability.

### 4.2 What was tried and does not work

- **Market columns in the state.** The hypothesis was that crypto crashes are systemic and that the
  state, all of it per single asset, cannot see them. Adding market breadth (share of the 15 above
  their daily EMA200) and BTC's structure, the out-of-sample median goes from +14.7% to **−25.5%**
  and the rank drops. **Rejected.**
- **More iteration rounds.** Past the third the target contains an estimate of itself and the
  variance grows more than the horizon gain.

---

## 5. What is wired in

| where | what |
|---|---|
| `trainer.MODEL_PRECEDENCE` | `rl_model` at the head, ahead of `swing_model` |
| `strategies.ai_model_simulation` | `rl_model` branch → `signals.rl_signals`, with no threshold and no barriers |
| `confluence._modello` | **the same voter**, which serves whichever model heads the chain |
| `panels.STRATEGIE["AI Model"].note` | says what it does and with what strength, on the page |

**Why not a ninth voter.** A second model voter answers the same question starting from the same 41
columns: they would vote together, and the confluence's minimum breadth is counted in families
precisely so the same opinion does not weigh twice. With the policy present, `CONF_MODELLO_ENTRA`
and `CONF_MODELLO_ESCI` have no effect — the policy has the threshold inside its objective — and
they matter again if only the swing model is left.

**In service the cadence is one day at any interval** (`signals.swing_cadenza`). It is not a knob:
it is the step the cost inside the reward is calibrated on. At an hourly cadence the same policy
would pay twenty-four times the costs its objective had budgeted for.

---

## 6. What is missing

- **The basket.** Everything here is per single asset. `strategy.md` §6.1 counts 1,691 daily
  allocation episodes, which is very few for RL, and points at fractional Kelly as an alternative
  that is almost certainly sufficient.
- **The significance of the *when*.** Two windows agreeing in sign and neither significant. It needs
  either more history or a control with more power — for instance the rank across all the folds of a
  CPCV instead of across two periods.
- **The confluence grid with and without the voter**, on the same assets and the same window. It is
  still outstanding for the swing model too (`modello-swing.md` §6): until it exists, there is no
  telling whether the model voter pays.
