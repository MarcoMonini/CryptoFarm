# Capability map — retraining the AI model (2026-08-28)

> **Status: executed, and the target has moved twice since.** This document remains as a
> **pre-registration**: it declares the success criteria *before* the measurements, which is what
> makes it possible to check they were not rewritten after seeing them. What actually happened:
> `model-legs` was closed with a negative result (`modello-swing.md` §1) and its code deleted;
> `model-swing` replaced it and did not beat chance at matched exposure; the model leading today is
> the entry model (`modello-ingresso.md`), which asks a different question from either — not "how
> close are we to an extreme" but "what does buying here return".
> The only modules from this map that are in service today are `positioning` and `features-bar`.
> **It is not a specification to execute: it is the yardstick for judging what was executed.** No
> module spec should be written before this map is approved: getting the boundaries wrong is
> expensive, revising fifteen lines is not.

## Why a map and not a single spec

The request packages five capabilities that are verified separately: a new data store, a set of
features, a training run, and **two** consumers of the model asking different questions (the menu
entry decides *when to enter and exit on its own*; the voter decides *how to vote in a college*). A
single spec would force every task to be judged against the whole contract.

## The modules

| id | responsibility | depends on |
|---|---|---|
| `positioning` | local store of funding rate, open interest, long/short and taker ratio from Binance bulk dumps, twin of `data/klines.py` | — |
| `features-bar` | per-bar, scale-free features, with cross-sectional context and positioning; one shared definition between training and inference | `positioning` |
| `model-legs` | training of the two heads (`P_su`, `P_giu`), purged validation + temporal verification + random control, artifact with metadata | `features-bar` |
| `strategy-ai` | the "AI Model" menu entry: entry on `P_su`, exit on `P_giu` or barrier; and moving `policy_model` out of the precedence list | `model-legs` |
| `voter-ai` | the per-bar `Votante` inside the confluence, voting +1/−1 and registering like the other six | `model-legs` |

Build order: `positioning` → `features-bar` → `model-legs` → `strategy-ai`, `voter-ai`

No cycles. `strategy-ai` and `voter-ai` are parallel and do not know each other: both read the
`model-legs` artifact, which is the interface.

## The initiative's success criterion, declared beforehand

Not "AUC goes up". The project has already measured three times that a real ranking advantage does
not pay. The number to beat, declared now and not afterwards:

1. **out of sample** (trained before the cut, measured after, a single cut declared in advance), the
   average net per trade must sit above the **p95 of 500 random selections of the same size** — the
   same control as `meta_gate`/`ai_voter`, which no design has passed stably so far;
2. on **two adjacent thresholds**, not one: a single peak between neighbouring thresholds is noise,
   and it has happened before (`ai_voter` at 0.45 returns −1.5% between 0.40 and 0.50, which return
   +0.8% and +2.0%);
3. the median entry must fall **before 43% of the leg** — the number the confluence achieves today.
   It is the only criterion that translates "anticipate" into a measurement.

If 1 and 2 do not pass, the result is written up and the strand is closed with a measurement, not
with an opinion. Criterion 3 can pass on its own and is information either way.

## What the map deliberately excludes

- **no three-action policy**: `strategy.md` §12-13, closed with a negative result and a known cause;
- **no `aggTrades`**: `sum_taker_long_short_vol_ratio` is the same information already aggregated to
  5 minutes, and in the panel it did not pass the sign check — so it is not worth hundreds of GB;
- **no deep architectures**: qlib benchmark, `ricerca-quant-ml.md` §1.1;
- **no optimisation of the voters' parameters** together with the model.

---

## Decisions taken with the user (2026-08-28)

| fork | choice | consequence |
|---|---|---|
| positioning data | **yes, only `retail_pos` and `top_pos`** | `positioning` downloads and keeps every column (they arrive in the same file, it costs nothing), but `features-bar` uses two of them. The other ten — funding included — did not pass the sign check on the 5 assets × 2 windows panel |
| scale | **1h + 4h + 1d, with `TIMEFRAME` as a feature** | a single model covers the planes the confirmation, structure and regime voters run on. Below the hour stays excluded: it is the region already measured to be a loser |
| heads | **one only, three classes on symmetric barriers** | from every bar: `+k·ATR` first (UP), `−k·ATR` first (DOWN), neither within `H` (FLAT). `P_su` enters, `P_giu` exits and votes −1 |
| position state | **out of the features** | the model does not know whether a position is open. It is an opinion about the bar, independent of the trading in progress — and that is what makes the artifact identical for the two consumers |

### Why three classes here is not the three classes already rejected

The difference is the **symmetry of the barriers**, not the number of classes. With
`TP_ATR_MULTIPLE = 1.5` and `SL_ATR_MULTIPLE = 1.0` the "sell" class means "the stop of a long
position was hit first": it covers ~60% of the bars and confuses "it goes down" with "it goes down a
little and then up". It is the reason written in `ml/signals.py` for why that class must not be used
as a sell signal.

With symmetric barriers the DOWN class means "it fell `k·ATR` before rising as much", i.e. exactly
the bearish leg to avoid. The two directional classes become comparable with each other, which is
the property `P_su` against `P_giu` needs.

The price of symmetry is that the break-even argument in `labeling.py` disappears (with 2:1
barriers the break-even precision drops from 66.7% to 44.4%). It does not apply the same way here:
the model picks a **direction**, not only whether to enter, and the commission floor remains the
economic constraint inside the label.
