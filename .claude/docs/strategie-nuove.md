# New strategies, the short side, and the right dataset

Sequel to [`backtest-strategie.md`](backtest-strategie.md), which measured the simulator's
strategies over nine years. Here: the four corrections applied to the code, the choice of a different
dataset and why, a trader's reading of the strengths and weaknesses of the historical strategies,
five new strategies built on that reading, and an engine that can also go **short**. Full tables in
`reports/` (`lab_*` files).

## The result in three lines

On BTC's 2021-2026 cycle, **at daily scale and with the corrections applied, the historical
strategies are not losers at all**: median +20% and 70% of configurations profitable for "Close ATR",
against the −96% they gave on 15 minutes. The disaster measured earlier was largely an artifact of
trading frequency and timeframe, not of the rules.

**The five new strategies — channel breakout, squeeze, trend pullback, Ichimoku, and mean reversion
with a regime filter — do not beat passive holding**, neither in sample nor out. They do beat its
**risk**, though: 22% drawdown against 76%, which at leverage 2 becomes +196% against +166% with half
the drawdown.

**The short side, on this asset and in this period, subtracts rather than adds**: the median gets
worse in all five strategies and the short leg's contribution is negative. It only pays in 2022, the
one genuinely bearish year.

---

## 1. The four corrections

| defect | correction | measured effect |
|---|---|---|
| the menu entry `"Supetrend"` did not match the dispatch string (`"Supertrend"`) | fixed the string in `config.STRATEGIES` | the entry runs: on BTC 2021-2026 at 4h the best configuration returns **+450%** (Sharpe 1.01, drawdown 34%) |
| `"ATR Bands"` had a dispatch branch and no menu entry | added the entry | selectable: **+678%** at 4h, the best of the historical ones over this period |
| the stop loss in `buy_sell_limits_close_simulation` was commented out | restored: stop fixed at entry, exit on the first close below | with the 99% default it stays inert (no golden changes); at operational values it now acts |
| `EMA200` was the EMA **of the open** over the short window, and "Trend Zones" compared it with `EMA20`, i.e. a moving average against itself | column removed, the three functions that read it now use `EMA100` (the real long average) | see below |

**Trend Zones, before and after** (BTC 2021-2026, commission 0.05%):

| interval | window | before: trades/year | before: return | after: trades/year | after: return |
|---|---:|---:|---:|---:|---:|
| 15m | 10 | 3,603 | −100% | — | — |
| 4h | 10 | 202 | −21.9% | 10.6 | **+309.3%** |
| 4h | 20 | 135 | +30.6% | 8.5 | **+231.5%** |
| 1d | 20 | 21 | +166.1% | 1.2 | +189.4% |
| 1d | 50 | 12 | +78.5% | 0.5 | +204.1% |

The golden master was regenerated: the only 17 entries that changed are `add_technical_indicator`
(one column fewer) and the three functions that read `EMA200`, across the four scenarios. No other
strategy moved.

## 2. The dataset: why not 2017 any more

The previous measurement used 2017-2026 because it was all the history available. It is the wrong
choice for deciding what to do now, and that can be demonstrated rather than asserted:

| period | passive holding | CAGR | Sharpe | drawdown |
|---|---:|---:|---:|---:|
| BTC 2017-2020 | +2,803% | 132.3% | 1.44 | 83.2% |
| **BTC 2021-2026** | **+166%** | **18.9%** | **0.59** | **76.5%** |
| ETH 2017-2019 | +1,479% | 151.3% | 1.37 | 93.8% |

A market growing 132% a year forgives any systematic error; one growing 19% does not. And the
parameters do not carry across regimes: choosing the best configuration on the 2017-2020 cycle and
measuring it on 2021-2026, **four of the five new strategies end up at a loss** (from −73% to +21%,
with passive holding at +166%). In the opposite direction, the same strategies chosen on 2017-2018
and verified on 2019-2020 returned **+180%** with the median of the top five at +172%: in the old
cycle trend-following worked, in this one it does not.

**The dataset used here is therefore BTC/USD from 2021-01-01 to 2026-08-24**, which contains a
complete cycle — November 2021 top, −64% in 2022, 2023-2024 recovery, 2025-2026 distribution — at 1h,
4h and 1d. The source is the same as the previous work (public Bitstamp one-minute dump), with
0.03%-0.37% of flat bars per year in this period, i.e. clean data.

**What is missing, and why.** This session's environment has egress blocked towards every exchange
(`data.binance.vision`, `api.binance.com`, Kraken, Coinbase, Bybit, Kucoin, Gate, MEXC), towards
Kaggle and towards every aggregator (CoinGecko, CryptoCompare, Messari, DefiLlama): it answers 403 on
CONNECT. Only GitHub and PyPI remain reachable, and **no public repository contains recent intraday
candles for SOL and BNB** — those that publish them stop at 2019 (Bitfinex) or publish on Kaggle
(WISEPLAT). The requested multi-asset comparison therefore remained half-done: BTC on the recent
cycle and ETH on 2017-2019, which is the only second market available.

It is not a limitation of the code: `data/klines.py` already downloads BTCUSDT, ETHUSDT, SOLUSDT and
BNBUSDT from the Binance dumps, and every script here accepts `--symbol`. On a machine with open
network access:

```bash
python -m cryptofarm.data.klines --update --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT
for s in BTCUSDT ETHUSDT SOLUSDT BNBUSDT; do
  python -m scripts.strategy_lab --all --symbol $s --interval 1d --since 2021-01-01
  python -m scripts.lab_report --symbol $s --interval 1d
done
```

## 3. The historical ones, read by a trader

What each does, where it is right and where it breaks, with the measurement next to it (BTC
2021-2026).

**Mean reversion on ATR bands** — *Close ATR, ATR Bands, Close Buy/Sell Limits, ATR Live Trade*. They
buy when the price moves `k` ATR away from the mean and sell on the way back. *Strength*: in sideways
markets reversion to the mean is the most statistically reliable phenomenon there is, and indeed on
daily bars in this cycle they are the best of all (Close ATR +575%, Sharpe 1.25, drawdown 25%).
*Weakness*: they buy **every** low, including the first of a structural decline, and have no way to
tell a pullback from a reversal; on 15 minutes the margin per trade (+0.08%) is below the round-trip
cost (0.10%-0.20%) and the median result collapses to −96%.

**Trend-following with moving averages** — *Close EMA Crossover, Trend Zones, Close Bullish EMA*.
*Strength*: no prediction, you stay in as long as the structure holds; on the 2017-2020 cycle they
were the winning family. *Weakness*: in a market oscillating without direction every crossing is a
false signal, and 2021-2026 is exactly that market; with the starting parameters (10/50/200) on 15
minutes they lost everything.

**Breakout with target and stop** — *TP/SL with ATR, Supertrend*. *Strength*: the risk per trade is
defined before entering, the only family where it is. *Weakness*: a target symmetric to the stop
(1:1) or at 1.618 cuts exactly the long moves that pay a breakout strategy, and the win rate needed to
break even rises above 50% net of costs.

**Pure price patterns** — *Green Candles, Close RSI Reverse*. *Strength*: none. *Weakness*: at 15
minutes they lose **even at zero commission** (−10% and −76%), which says there is no signal, not that
the cost eats it. At daily scale they turn positive, but for the same reason everything does: 15-25
trades a year instead of 1,500.

The cross-cutting defect, already measured in the previous document: **none of them knows what regime
it is in**. Among the columns produced there is not a single indicator saying whether a trend exists,
whether volatility is compressed, whether volume confirms. The five new strategies come from there.

## 4. The five new strategies

All in `src/cryptofarm/trading/strategies_ls.py`, all with a three-state position (+1 / 0 / −1), all
measurable with and without the short side. The new indicators are in `indicators_extra.py`: ADX,
Donchian channel, Bollinger + Keltner (squeeze), StochRSI, MFI, OBV, Ichimoku — all from `ta`, none
was used by the project.

### 4.1 `donchian_breakout` — channel breakout with a strength filter
*Hypothesis*: you lose because you buy against the trend; entering **in** the direction of the move
and letting it run inverts the problem.
*Rules*: long on a close above the high of the last `channel` bars (channel shifted by one bar: no
look-ahead), with `ADX ≥ adx_min` and the price on the right side of the long EMA; short symmetrically.
Exit on a **chandelier stop**: highest reached minus `k·ATR`, which follows the price.
*New indicators*: Donchian, ADX.

### 4.2 `squeeze_breakout` — compression and release
*Hypothesis*: there is too much trading; volatility compression selects a few moments a year by
construction, with no arbitrary filters.
*Rules*: when the Bollinger bands move inside the Keltner channel the market is in a *squeeze*; on the
first bar where the squeeze opens, enter in the direction the price sits relative to the mean of the
bands, with optional confirmation from the OBV slope. Exit on an ATR trailing stop.
*New indicators*: Bollinger, Keltner, OBV.

### 4.3 `trend_pullback` — oversold bounce inside a trend
*Hypothesis*: mean reversion works, but only on the trend's side.
*Rules*: above the long EMA, buy when StochRSI climbs back above the oversold threshold; below the
long EMA, sell short on the bounce down from overbought. Fixed stop at `k·ATR`, profitable exit when
the oscillator returns to the opposite zone. With `regime_ema=0` the filter turns off: that is the
ablation that measures what it is worth.
*New indicators*: StochRSI.

### 4.4 `ichimoku_trend` — the yardstick
*Hypothesis*: none. It is a complete trend system, already available and widespread; if a purpose-built
strategy does not beat it, it is not worth the work it costs.
*Rules*: Tenkan/Kijun crossing with the price on the right side of the cloud (spans already shifted
forward, as on the chart); exit on the opposite crossing or on a break of the Kijun.

### 4.5 `band_reversion_gated` — the combination
*Hypothesis*: "Close ATR" fails because of the regime, not because of the idea. Same entry, but only
where it makes sense.
*Rules*: entry identical to the historical bands (KAMA ± `k·ATR`) **only when `ADX < adx_max`**, i.e.
in the absence of a trend; exit on the return to the KAMA or on the `k·ATR` stop. Optional regime
filter on the long EMA.
*New indicators*: ADX on top of the historical structure.

## 5. The short side: how it is simulated, and what it is worth

`pnl.simulate_trading_with_commisions` pairs two lists of signals and knows only one direction: a
direct reversal from long to short cannot be represented. The new engine, `pnl.simulate_positions`,
takes a list of **position changes** `(time, price, target)` with the target in `{+1, 0, −1}` and
produces closed trades with their side. Conventions:

- notional equal to capital times `leverage` (default 1), commission on both legs computed on the
  notional traded;
- daily **carry cost** (`carry`, default 0.03% per day) charged on both directions: it is the funding
  of a perpetual, which on Binance oscillates around 0.01% every eight hours. In reality it is a
  transfer and whoever is on the right side receives it; charging it always is the prudent choice;
- capital hitting zero: simulation stops. That is liquidation, and at leverage 3 an adverse move of a
  third is enough.

**What the short side is worth** (BTC 2021-2026, 1h + 4h + 1d, same configurations with and without):

| strategy | pairs | long-only median | median with short | where short improves | median contribution of the short leg | short win rate |
|---|---:|---:|---:|---:|---:|---:|
| donchian_breakout | 384 | −22.1% | −56.9% | 2.6% | −53.3% | 31.6% |
| squeeze_breakout | 162 | −41.6% | −71.9% | 6.8% | −71.7% | 29.6% |
| trend_pullback | 108 | −36.1% | −60.0% | 8.3% | −35.4% | 49.3% |
| ichimoku_trend | 18 | +15.2% | −25.1% | 5.6% | −56.9% | 29.6% |
| band_reversion_gated | 216 | −0.5% | −5.5% | 23.6% | −3.6% | **52.3%** |

The reading is not "shorting does not work": it is that **on an asset with positive drift, and in a
period whose only bearish year is 2022, the short side pays the cost of being on the wrong side of
the drift** for four years out of five. The one exception is mean reversion
(`band_reversion_gated`), where the short has a 52% win rate and costs almost nothing: selling an
extension above the mean in a sideways market is symmetric to buying one below it.

Anyone who still wants the short side has two roads measurable with these tools: enable it only when
the long moving average is **falling** (price below the average is not enough), or use it only in
mean-reversion strategies.

## 6. The results

**Ranking on BTC 2021-2026, daily bars, commission 0.05%** (passive holding: +166%, drawdown 76.5%,
Sharpe 0.59):

| family | strategy | best | Sharpe | drawdown | trades/year | grid median | profitable |
|---|---|---:|---:|---:|---:|---:|---:|
| historical | Close ATR | +575% | 1.25 | 25.5% | 3.4 | +20.5% | 70.6% |
| historical | Close Buy/Sell Limits | +335% | 0.86 | 59.3% | 3.9 | +25.0% | 71.6% |
| historical | TP/SL with ATR | +288% | 0.87 | 54.3% | 2.8 | +87.6% | 82.1% |
| historical | ATR Bands *(now in the menu)* | +212% | 0.67 | 61.4% | 5.5 | +33.1% | 78.2% |
| new | squeeze_breakout | +120% | 0.59 | 55.6% | 3.4 | −34.0% | 19.8% |
| new | ichimoku_trend | +106% | 0.58 | 32.8% | 7.3 | +11.7% | **75.0%** |
| new | trend_pullback | +89% | 0.50 | 37.4% | 26.8 | −24.2% | 29.2% |
| new | band_reversion_gated | +84% | **0.78** | **22.1%** | 4.4 | −7.0% | 43.6% |
| new | donchian_breakout | +63% | 0.45 | 42.2% | 3.2 | −25.4% | 15.6% |
| historical | Trend Zones *(fixed)* | +60% | 0.42 | 49.9% | 2.7 | +60.2% | 100% |

The "best" figures are maxima over grids of very different size (1,728 configurations for Close
Buy/Sell Limits, 12 for Ichimoku): the honest column is the median, and there long-only Ichimoku with
75% of configurations profitable is the most solid of the new ones.

**Out of sample — chosen on 2021-2023, returned on 2024-2026** (passive holding: +46% then +80%):

| family | strategy | chosen in sample | out-of-sample return | median of the top 5 | ρ in/out |
|---|---|---:|---:|---:|---:|
| historical | Close RSI Reverse | +99% | **+57.5%** | +46.5% | 0.58 |
| historical | ATR Live Trade | +118% | +47.2% | +14.7% | 0.61 |
| historical | Close Bullish EMA | +38% | +38.2% | +42.1% | 0.57 |
| historical | Supertrend | +80% | +35.1% | +35.1% | 0.02 |
| historical | Close ATR | +487% | +15.0% | +0.7% | 0.08 |
| new | band_reversion_gated | +65% | **+11.1%** | +9.4% | 0.52 |
| new | squeeze_breakout | +31% | −3.5% | −3.5% | 0.49 |
| new | ichimoku_trend | +161% | −21.3% | −2.5% | −0.36 |
| new | trend_pullback | +154% | −35.1% | −15.9% | 0.25 |
| new | donchian_breakout | +77% | −39.9% | −23.6% | 0.10 |

**None, in any family, beats passive holding out of sample.** Mean-reversion strategies transfer
better than trend ones, which is consistent with the regime: in a cycle without a clear direction,
the bet on reversion pays more than the bet on continuation.

**The ablations** (same three intervals) say the new indicators are useful, but for robustness, not
for the peak:

| strategy | filter turned off | median without | median with | trades/year without → with |
|---|---|---:|---:|---:|
| ichimoku_trend | cloud confirmation | −20.7% | **+8.4%** | 60 → 23 |
| donchian_breakout | trend filter (long EMA) | −46.8% | −34.8% | 20.4 → 19.3 |
| donchian_breakout | ADX filter | −40.3% | −41.5% | 23.9 → 18.8 |
| squeeze_breakout | volume confirmation (OBV) | −60.5% | −49.8% | 22.2 → 15.9 |
| trend_pullback | trend filter (long EMA) | −57.1% | −44.7% | 116 → 52 |
| band_reversion_gated | range filter (ADX) | −11.0% | −1.9% | 5.8 → 1.1 |
| band_reversion_gated | trend filter | −9.0% | 0.0% | 3.9 → 0.5 |

Every filter improves the median and reduces the number of trades; the only irrelevant one is ADX as
a minimum threshold in the channel breakout, where the wide channel already does that job.

**Leverage and costs.** A strategy with a quarter of passive holding's drawdown is not worse: it is
the same risk at a different leverage.

| configuration | leverage 1 | leverage 2 | leverage 3 |
|---|---|---|---|
| `band_reversion_gated` 1d | +84% / DD 22% | **+196% / DD 41%** | +319% / DD 58% |
| `ichimoku_trend` 1d | +106% / DD 33% | +179% / DD 54% | +169% / DD 71% |
| passive holding | +166% / DD 76.5% | — | — |

At leverage 2 mean reversion with a regime filter **beats passive holding on both axes** (+196%
against +166%, drawdown 41% against 76%). That holds in sample: out of sample the same configuration
returns +11% against +80%. On cost sensitivity, the 1d strategies lose 15-30% of the result going
from 0.02% to 0.10% per leg; the 4h ones lose half or more.

## 7. What I would do with it

1. **Daily scale, not 15 minutes.** It is the most robust conclusion of both documents: the same rule
   changes sign when the timeframe changes, and always in the same direction.
2. **Mean reversion with a regime filter, long only, at leverage 1.5-2.** It is the only combination
   that in the measurements beats passive holding at equal risk, and it is also the only one of the
   new strategies that transfers out of sample with a positive sign.
3. **No shorting BTC in a market without a confirmed downtrend.** The cost is measured, not
   theoretical.
4. **Long-only Ichimoku as the reference**: 75% of its configurations close profitable at 1d. Any new
   strategy that does not beat it on that metric does not deserve to go into production.
5. **Repeat everything on SOL and BNB before deciding.** They are the assets where the 2021-2026 cycle
   had the highest volatility, and none of the conclusions here has been verified on them.

## 8. Limitations

- **The numbers for `donchian_breakout` and `squeeze_breakout` predate a correction to the trailing
  stop, and must be redone.** The stop in force during a bar was built with that same bar's high and
  ATR, then compared with its low: it assumed the favourable extreme arrived first within the bar. For
  a single fill the bias is one-directional — you exited at an unobtainable price, +0.9% on the test
  scenario perturbing only the high of the exit bar, +2.6% on a wider perturbation. On the **portfolio
  net**, by contrast, the sign is not predictable, because the inflated stop also fired earlier than it
  should: on the tests' synthetic series the correction takes `donchian_breakout` from −6.5% to −1.3%
  and `squeeze_breakout` from −2.6% to −2.2%, i.e. it improves. On BTC 2021-2026 it could not be
  re-measured (it needs the candle store, absent in the environment where the correction was made):
  rerun the commands in §9 and redo §6 for those two rows. The other three strategies do not use the
  trailing stop and are untouched.
- **One asset and one cycle.** The conclusions on the short side and on regimes hold for BTC
  2021-2026.
- **Selection.** The "best" columns are maxima over grids: they must be read with the median next to
  them.
- **Ideal execution.** Entries at the bar's close, stops executed at the exact level, no slippage and
  no impact. On crypto liquidation gaps that is optimistic.
- **Fixed funding.** 0.03% per day charged on both directions; in reality it varies and changes sign.
- **Liquidation is evaluated at the close of the trade**, not bar by bar: at high leverage it
  understates the risk of being closed out during the excursion.

## 9. Reproducing

```bash
python -m scripts.strategy_lab --all --interval 1d --since 2021-01-01     # the new ones
python -m scripts.strategy_sweep --all --interval 1d --since 2021-01-01 \
    --fee 0.05 --suffix _2021_fee005                                       # the historical ones, same cost
python -m scripts.lab_report --symbol BTCUSD --interval 1d                 # tables in reports/
```
