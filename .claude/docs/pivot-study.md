# The pivot study and the swing structure strategy

Branch **`claude/decomposer-pivot-study`**, 2026-10-12 to 2026-11-21. It starts from the decomposer
(`trading/decomposer.py`, merged with PR #12) and asks one question in three steps: *what are the
legs between the market's turns, can they be found without picking a window, and does a rule read on
them trade?* The answer to the last step is **a candidate, not a result**: the best cell beats random
placement at matched exposure on 2018-2022 (98th percentile) and sits on the 95th on 2023-2026, not past
it, and it was picked out of 192 cells.

The code is documented where it lives: the `decomposer.py` and `swing_structure.py` module
docstrings carry every number below with its date, and `src/cryptofarm/trading/README.md` lists the
functions. This document holds the thread between them and what is left open.

## 1. Legs with a window: TradingVision's pivots

The decomposer's section *What is left of the leg* (`pivots`, `legs_of`, `to_pivot`). A pivot is a
close strictly beyond the `W` closes on both sides, two of a kind in a row merged into the more
extreme — TradingVision's `data.pivots.find_pivots`, verified identical on 50,000 random-walk candles
at every window. On BTC 15m the median leg is 0.92%, 1.43%, 2.21% and 3.48% at W = 6, 12, 24 and 48.

**Pivots of a size** slices the legs by their change (1% to 1.2% by default, W = 24) and reads every
indicator on the pivot the leg starts from, in deciles over every candle. Against every candle the
indicators recognise a pivot easily (a low sits in the bottom decile of the Donchian position at +37
standard errors), but that is hindsight. Against **every pivot of the same kind**, the only thing the
1% legs have in common is that they are small: they start from shallower extremes in quieter
periods (bottom decile of the distance from PSAR, EMA, VWAP and KAMA at −6.6 to −7.2 se on BTC, up to
−10.2 on ETH). A correlation of the legs' paths was measured and dropped: the slice is no more alike
than any leg of its side (0.620 against 0.670).

Side effects that stayed:

- **the indicator menu keeps 27 of 42 columns**, the ones that held in at least one study (a zone of
  `zones` that holds, or a filter the pivot study chose); the other fifteen are `DROPPED`. They are
  still computed;
- **the RSI section went.** It read the 80-90 band as momentum worth a fee (+0.56% 48 candles later
  on BTC 15m, 5.5 se) on the whole history with nothing held out. The zones study then measured
  momentum on held-out years and it did not hold.

## 2. Legs with no window: the swing hierarchy

A window fixed in candles finds 1% legs in a quiet month and 5% legs in a wild one. Two windowless
readings of the close, both built on `turns` (every candle where the close changes direction):

- **persistence** (`decomposer.persistence`): the two neighbouring turns with the smallest move
  between them go, over and over, and that move is their persistence. The turns with persistence ≥ p
  are the swings of p or more, nested across p;
- **the swing hierarchy** (`swing_structure.hierarchy`): level 0 is every turn; a high climbs a level
  when it is higher than the highs either side of it on the level below, a low when it is lower —
  Dow's higher highs and lower lows applied to themselves, until nothing climbs.

The hierarchy is the one that went further. Each level's legs are about **3.2 times** the length of
the one below; the median leg is 5 / 18 / 58 / 185 bars at levels 1-4, almost the same at 15m and 1h,
which is why `swing_structure.HORIZON` is one table for both. On BTC 15m level 4 has 1,702 legs of
7.59% over 185.2 candles on average (the decomposer's *Legs of the swing hierarchy* table,
`level_legs`).

**It can be dated live.** `swing_structure.climb` gives every level of every pivot the bar it became
final on, and nothing after that bar changes it: rerunning the hierarchy on every prefix agrees on
every decision (tested, and checked on 252,880 decisions of BTC 15m). The price is the lag: at levels
2-4 a pivot is confirmed 19 / 63 / 199 bars after it happened, longer than the leg itself, because
Dow's rule confirms a high only with the lower high that follows it. About 60% of the next leg is
gone by the confirmation. A threshold zigzag confirms the same legs in a quarter of the bars at the
same cost in price, so the lag is the question's, not the method's.

The decomposer reads its hierarchy from `swing_structure`; until 2026-10-12 it held a second copy.

## 3. Wyckoff on the hierarchy: what was measured

The Wyckoff rules were translated onto the levels and measured with every pivot used only from its
confirmation, on the fifteen `klines.DEFAULT_SYMBOLS`, discovery 2018-2022 against confirmation
2023-2026, per event and before the fee. **192 cells**: 15m and 1h, levels 1-4, four entry rules
(spring, upthrust, backup to the creek, last point of support), two exits, four contexts.

- the rank in discovery says nothing about the rank in confirmation (**Spearman −0.06**);
- only five cells were above 1.5 standard errors in both periods, all of them the last point of
  support with its stop;
- the spring and the upthrust lose once the break restarts; the backup to the creek loses at every
  small level; the length of a range does not predict the move after it.

> **Trap.** The code of that 192-cell sweep **is not in the repository**: it was run ad hoc. What can
> be rerun is `swing_structure.main` (the LPS rule, levels 3-4, the four filter combinations, both
> timeframes: 16 cells) and `--control`. Whoever wants to redo the multiplicity correction over the
> 192 has to rebuild the sweep first.

## 4. The rule: the last point of support

`swing_structure.posizioni`, in the simulator's menu as **«Swing Structure (Wyckoff LPS)»**. At level
k, on the bar a higher low is confirmed in an up structure (the last two highs and the last two lows
rising) go long; a lower high in a down structure goes short. Out at the first close past that
pivot, or after `HORIZON[k]` bars; an opposite signal reverses. Two filters, both on by default:

- `choch`: only the first such pivot after a change of character;
- `aligned`: only when level k+1 points the same way.

An entry whose close is already past its pivot is skipped (one entry in seven was a trade stopped on
the next bar). The stop is read and filled at the close, unlike `strategies_ls`.

**The entry is not at the low.** At level 4 on 15m the trade opens a median 245 bars after the low it
is named for, with the stop a median 5-6% away. A faster confirmation of the same pivots (an adaptive
zigzag at a quarter of the level's median leg) enters ten bars after the pivot, trades 23 times a
year per pair and makes nothing gross.

Two defects were found and fixed on the way, both by measuring:

- **a confirmation bar is read whole** (`_walk`). 43% of a level's pivots share their confirmation bar
  with another one. Reading them one at a time decided on a stale state: on BTC 15m level 4, 116 of
  546 events existed only that way and 91 were missed. No future was read, but no trader would have
  taken those decisions;
- **every exit is pinned by a test** to the first close past the stop or exactly `HORIZON` bars.

### Measured, 2026-10-12 (after the fix)

Gross per trade, one standard error over blocks of `HORIZON` bars, fifteen pairs, 0.1% a side:

| 15m, level 4 | 2018-2022 | 2023-2026 |
|---|---|---|
| **choch and aligned (default)** | **+126.6 ± 56.3 bp** (254 trades, 10/15 up) | **+62.5 ± 40.0** (249, 8/15) |
| choch only | +65.5 ± 30.4 | +19.0 ± 18.9 |
| aligned only | +47.2 ± 35.2 | +6.5 ± 25.1 |
| 1h, choch and aligned | +281.9 ± 200.0 | −52.7 ± 173.1 |

The default makes 4.5 trades a year per pair, +107 / +43 bp net of the round trip, 2.2 and 1.6
standard errors. At 1h no variant of level 4 kept on 2023-2026 what it made on 2018-2022.

### Against chance at matched exposure, 2026-11-21

`swing_structure.control`. Every hold keeps its pair, side and length; all holds of a period move by
**one common random fraction** of it. Exposure, the long/short mix (51% long) and the timing across
pairs, which move together, are all kept; only the *when* is random. This is stricter than
`swing_lab.collocazioni_casuali` and `entry_trainer.controllo_casuale`, which scatter each trade
independently and so break the clustering across pairs. The random placements make +0 to +3 bp a
hold.

| 15m, level 4 | 2018-2022 | 2023-2026 |
|---|---|---|
| **choch and aligned (default)** | **98.2°** | **94.3°** (seeds 1-3 at 2,000 draws: 94.8-95.2) |
| choch only | 99.1° | 82.4° |
| aligned only | 92.3° | 59.4° |
| neither | 92.7° | 50.3° |
| 1h, every variant | 84.5-97.6° | 26.8-38.8° |

## 5. Reading it

- **The edge is real money against a placement that makes nothing**: +62.5 bp gross a hold on
  2023-2026 against +0.9.
- **It does not pass the project's bar.** The entry model, the only thing this project wired in on a
  positive result, sits at the 100th percentile (`modello-ingresso.md`). This cell sits on the 95th.
- **The 2023-2026 percentile is not out of sample.** The cell was chosen out of 192 looking at both
  periods.
- **The filters carry everything.** Each one removed halves what is left; with neither, 2023-2026 is
  the median of chance. The level-4 structure alone does not time anything; the change of character
  plus the level above does, a little.
- **1h fails everywhere** on 2023-2026, though its legs have the same length in bars.

## 6. What is open

In order of what each would decide:

1. **A clean confirmation window.** 2023-2026 is spent, and so is the 2017-2020 cycle that
   `piano-strategie.md` step 5 keeps for the other strands: this study's discovery period covers it.
   The only untouched data is what accrues from here on. Freeze the default cell now and price it on
   data after 2026-11-21 only.
2. **The multiplicity correction over the 192 cells** (`scripts/multiplicity.py`, DSR/PBO). It needs
   the sweep rebuilt first (§3, trap).
3. **Passive holding as a second yardstick.** Not run; with 51% long and four-and-a-half trades a
   year per pair the strategy is out of the market most of the time, so it would mostly measure
   exposure — the reason the matched control comes first.
4. **Level 5 of the same rule at 15m**, out of curiosity rather than hope. Level 3 is in `main`'s
   table and loses net of the fee in every filter combination on both periods (−1.6 to −19.9 bp).

```bash
.venv312/bin/python -m cryptofarm.trading.swing_structure             # the 16-cell table
.venv312/bin/python -m cryptofarm.trading.swing_structure --control   # the random control, ~15 s
streamlit run src/cryptofarm/trading/decomposer.py                    # the pivot study, on the page
```
