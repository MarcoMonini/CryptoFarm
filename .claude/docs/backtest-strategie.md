# Backtest of the indicator strategies — 3,129 configurations over nine years

Measurements produced by `scripts/strategy_sweep.py`, `scripts/sweep_report.py` and
`scripts/strategy_focus.py`; the full tables are in `reports/`. Every number here comes from those
tables: no estimates, no results quoted from memory.

## The result in one line

Of the **3,129 configurations** tried — every indicator strategy in the simulator, each over a grid
of its own parameters, on **BTC at 15 minutes from 2017-01-01 to 2026-08-24** — **14.9% close
profitable, 45.2% lose more than 90% of the capital, and five (0.2%) beat passive holding**, which
over the same period did **+7,947%**. The median is **−87.3%**.

The five that beat passive holding are not five different strategies: they are three configurations
of "Close Buy/Sell Limits" with 3-4 trades a year, one of "Close ATR" with 5.9, and one of "ATR
Bands". And none of them survives out-of-sample verification (§5).

## 1. What was measured, and how

| | |
|---|---|
| Market | BTC/USD, 5-minute candles aggregated to the requested intervals |
| Period | 2017-01-01 → 2026-08-24, 9.65 years, **338,114 bars** at 15 minutes |
| Source | public Bitstamp one-minute dataset ([`ff137/bitstamp-btcusd-minute-data`](https://github.com/ff137/bitstamp-btcusd-minute-data)), imported with `scripts/import_candles.py` |
| Capital | 100, always fully reinvested, as in `pnl.simulate_trading_with_commisions` |
| Commission | 0.1% per leg (the page default); the sensitivity is in §6 |
| Strategies | the 10 functions in `trading/strategies.py` reachable from `trading_analysis`'s dispatch |
| Control | the same grids on ETH/USD 2017-2019, a different exchange and a different period (§9) |
| Metrics | compounded return, CAGR, Sharpe and drawdown on equity **marked to market bar by bar**, win rate, profit factor, exposure, commissions paid |

**Why Bitstamp and not Binance.** `data/klines.py` takes its candles from the `data.binance.vision`
dumps. In this session that host is blocked by network policy (403 on CONNECT, like
`api.binance.com`), so the store was filled from an alternative source with the same structure.
BTC/USD on Bitstamp is not BTCUSDC on Binance: prices differ by fractions of a point and the fee
schedule is different. It is fine for measuring **the behaviour of the strategies**; it is not a
to-the-cent replica of a Binance account.

**The strategies were not rewritten.** The sweep calls the functions in `trading/strategies.py` and
the P&L in `trading/pnl.py` exactly as they are. The only part reimplemented is the indicator
computation, so columns can be reused across configurations instead of recomputed (the PSAR alone
costs 26 seconds over 338,000 bars and does not depend on any swept parameter):
`tests/test_strategy_sweep.py` verifies column by column that it produces **the same table** as
`indicators.add_technical_indicator`, and over seven combinations of strategy and parameters that the
whole path produces **the same trades** `trading_analysis` would write into the page's table, dispatch
included.

**The benchmark.** The period is not neutral: BTC went from ~$1,000 to ~$77,500. A long-only strategy
that stays out of the market half the time starts at a disadvantage, and must also be judged on
drawdown, not on return alone.

| period | passive holding | drawdown |
|---|---|---|
| 2017-2026 (whole) | +7,947% | 84.0% |
| 2017-2021 (in sample) | +4,744% | 84.0% |
| 2022-2026 (verification) | +67.0% | 67.5% |
| 2019-2026 (walk-forward) | +1,970% | 77.3% |

## 2. Overview by strategy

Best and median are computed over configurations with at least 30 trades in nine years: below that
threshold one is not measuring a strategy but a single position held for years.

| strategy | configs | best | median | profitable | beats B&H | best Sharpe | DD of the best |
|---|---:|---:|---:|---:|---:|---:|---:|
| ATR Bands *(not in the menu)* | 168 | **+20,020%** | −70.2% | 33.3% | 0.6% | 1.36 | 63.7% |
| Close Buy/Sell Limits | 1,728 | +13,230% | −90.4% | 9.4% | 0.2% | 1.11 | 83.8% |
| Close ATR | 504 | +8,778% | −96.7% | 4.6% | 0.2% | 1.25 | 63.2% |
| Close Bullish EMA | 420 | +3,392% | −63.7% | 22.6% | 0% | 0.88 | 83.0% |
| Close EMA Crossover | 7 | +2,834% | −64.2% | 42.9% | 0% | 0.97 | 54.0% |
| ATR Live Trade | 18 | +1,056% | −99.6% | 5.9% | 0% | 0.74 | 85.5% |
| Supertrend *(not in the menu)* | 126 | +354% | −12.8% | 45.2% | 0% | 0.75 | 55.0% |
| TP/SL with ATR | 126 | +120% | −97.3% | 12.2% | 0% | 0.49 | 35.2% |
| Trend Zones | 6 | **−100%** | −100% | 0% | 0% | −1.41 | 100% |
| Close RSI Reverse *(not in the menu)* | 25 | **−100%** | −100% | 0% | 0% | −8.06 | 100% |
| Green Candles | 1 | **−100%** | −100% | 0% | 0% | −5.63 | 100% |

Three strategies lose **all** the capital in every configuration tried. It is not a special case: they
trade 2,080, 2,916 and 1,459 times a year respectively (§3).

### With the page's starting parameters

This is the case that matters most, because it is what you see when you open the simulator: ATR 5 /
1.6, EMA 10-50-200, RSI 12, limits 25/75, one condition, stop loss disabled.

| strategy | return | trades/year | win rate | profit factor |
|---|---:|---:|---:|---:|
| ATR Bands *(not selectable)* | +1,331% | 494 | 65.8% | 1.01 |
| Close Bullish EMA | −54.6% | 94 | 68.6% | 0.93 |
| Supertrend *(not selectable)* | −91.6% | 88 | 36.1% | 0.80 |
| Close ATR | −97.6% | 284 | 60.7% | 0.95 |
| Close Buy/Sell Limits | −98.2% | 291 | 60.8% | 0.95 |
| TP/SL with ATR | −99.9% | 392 | 50.4% | 0.88 |
| Green Candles | −100% | 1,459 | 27.2% | 0.72 |

**With the starting values, every strategy reachable from the menu loses money over the long run.**
Note the win rate: 60-69% of trades profitable and a profit factor below 1. The wins are there, but
they are smaller than the losses by more than the commissions allow.

## 3. Trading frequency explains almost everything

All 3,129 configurations, across all strategies, grouped by number of trades per year:

| trades/year | configs | median | profitable | average trade | commissions paid | median drawdown |
|---|---:|---:|---:|---:|---:|---:|
| < 10 | 313 | −2.5% | 39.9% | −0.18% | 5% | 61.4% |
| 10-30 | 265 | **+19.5%** | **53.6%** | +0.38% | 48% | 79.8% |
| 30-100 | 595 | −64.9% | 22.5% | −0.07% | 101% | 89.8% |
| 100-300 | 1,303 | −90.5% | 4.2% | −0.09% | 217% | 95.8% |
| 300-1,000 | 576 | −99.6% | 1.6% | −0.11% | 315% | 99.9% |
| > 1,000 | 57 | −100% | 0% | −0.20% | 96% | 100% |

The number of trades predicts the return better than any parameter, and predicts it inversely. The
"average trade" column says why: **the average gross margin per trade, on short timeframes, is of the
same order as the transaction cost** (0.2% round trip). A strategy trading 300 times a year pays 60%
of the initial capital in commissions every year, and has to earn that before earning anything.

The "commissions paid" column is cumulative over the nine years and expressed relative to the
**initial** capital: above 100% means the commissions paid are worth more than all the starting
capital.

## 4. Parameter sensitivity

The median swing is the difference between a parameter's best and worst value, holding all others
fixed, averaged over all their combinations. It is the answer to "how much does this widget matter".

| grid | most influential parameter | median swing |
|---|---|---:|
| Supertrend | `atr_multiplier` | 187.6 points |
| ATR Bands | `atr_multiplier` | 169.0 points |
| TP/SL with ATR | `atr_multiplier` | 140.6 points |
| Close ATR | `atr_multiplier` | 93.4 points |
| Close Bullish EMA | `rsi_window` | 83.9 points |
| Close Buy/Sell Limits | `rsi_sell_limit` | 25.7 points |

**`atr_multiplier` dominates wherever it appears**, and always in the same direction: wide bands,
few trades, fewer losses.

| `atr_multiplier` | Close ATR (median) | ATR Bands (median) | TP/SL (median) | trades/year (ATR Bands) |
|---|---:|---:|---:|---:|
| 0.8 | −100% | −100% | −100% | 991 |
| 1.2 | −99.8% | −99.8% | −100% | 624 |
| **1.6 (default)** | −98.6% | −95.2% | −99.7% | 414 |
| 2.0 | −95.5% | −67.6% | −96.9% | 280 |
| 2.5 | −85.5% | +16.6% | −72.0% | 205 |
| 3.0 | −56.6% | +45.0% | −16.6% | 133 |
| 4.0 | −7.7% | +37.0% | +28.8% | 43 |

**The 1.6 default is in the worst part of the range for all three strategies that use it.** The
highest multiplier tried (4.0) is the best or second best everywhere, which also says the optimum
might lie beyond the grid's edge.

The other parameters, briefly:

- **`atr_window`** matters little by comparison (5.2 points on Close ATR): it moves the band's noise,
  not its relative width.
- **`num_cond`** in "Close Buy/Sell Limits" is the difference between "RSI **or** band" (1) and "RSI
  **and** band" (2): median −96.2% against −69.1%, profitable 0.6% against 23.8%. Two conditions cut
  trades from 279 to 53 a year. Frequency again.
- **`rsi_sell_limit`** is monotone: 60 → median −94.1%, 85 → −68.0%, with the profitable share going
  from 0.7% to 35.1%. Exiting late is systematically better than exiting early — on a market that did
  +7,947%, where every exit is a bet against the trend.
- **`stop_loss`** never helps: on ATR Bands, median −86.5% with a 5% stop against −50.5% without. The
  stop closes at a loss positions that would have come back, and on Close Buy/Sell Limits it has **no
  effect at all**, because the code that would apply it is commented out (§7).
- **the EMA triples** are the decisive parameter of "Close EMA Crossover": 50/100/200 returns
  +2,834%, 10/50/200 +1,441%, 8/13/21 **−100%**. Seven values, four of which lose everything: not a
  parameter to leave to a default.

## 5. Out of sample: this is where it all falls apart

The previous sections look at the whole period, i.e. they choose the parameters already knowing how it
went. The honest verification is to choose on the early years and measure on the later ones.

**Chosen on 2017-2021, returned on 2022-2026** (passive holding over the same period: **+67.0%**):

| grid | in-sample return | out-of-sample return | median of the in-sample top 10 | top 10 profitable | best possible out of sample | Spearman ρ in/out |
|---|---:|---:|---:|---:|---:|---:|
| Close Buy/Sell Limits | +4,462% | **+192.2%** | −6.3% | 40% | +197.4% | 0.47 |
| Close ATR | +4,257% | **+103.8%** | −48.1% | 20% | +107.1% | 0.78 |
| ATR Bands | +10,327% | **−86.2%** | +6.6% | 50% | +169.6% | 0.65 |
| Close Bullish EMA | +2,775% | −8.7% | +20.4% | 70% | +67.1% | 0.49 |
| Supertrend | +662% | −49.8% | −50.8% | 10% | +58.2% | 0.23 |
| TP/SL with ATR | +334% | −68.0% | −73.2% | 0% | +34.4% | 0.90 |
| Close EMA Crossover | +6,032% | −74.9% | −83.0% | 14% | +0.9% | 0.86 |

Two configurations out of seven transfer, and they also beat passive holding. **But the column that
matters is the fifth**: the median of the in-sample top ten is negative in five cases out of seven.
The top-ranked "Close Buy/Sell Limits" returns +192% in verification while its nine ranking neighbours
have a median of −6.3%: it is not a parameter region that works, it is a lucky row. On "ATR Bands",
the best in sample — the +10,327% one, the most profitable of the whole study over the full period —
loses **86%** in verification.

**Walk-forward.** More realistic still: at the end of every year one re-optimises on the years seen so
far and keeps that configuration for the following year.

| grid | 2019-2026 | profitable years | worst year | configuration changes |
|---|---:|---:|---:|---:|
| ATR Bands | +1,111% | 75% | −33.7% | 3 |
| Close ATR | **+914%** | 87.5% | −5.6% | 2 |
| Close Buy/Sell Limits | +366% | 50% | −50.2% | 3 |
| Close Bullish EMA | +354% | 62.5% | −65.4% | 4 |
| Close EMA Crossover | +126% | 75% | −66.0% | 2 |
| Supertrend | −43.9% | 50% | −37.1% | 2 |
| TP/SL with ATR | −76.1% | 37.5% | −55.1% | 3 |
| Trend Zones | −99.9% | 0% | −85.8% | 1 |
| Green Candles | −100% | 0% | −97.4% | 1 |

None reaches passive holding's **+1,970%** over the same span. Close ATR comes closest with much less
suffering: 87.5% positive years, worst year −5.6%, against a passive holding that lost 64.3% in 2022
and went through a 77.3% drawdown. On a risk-return basis it is the only result in this study that
deserves a second look — with the caveat that it is still 4-9 trades a year decided by two parameters
re-optimised twice in eight years, i.e. a tiny sample.

## 6. Commissions: where the margin really is

The same configurations, rerun varying only the commission (`reports/commissioni.csv`):

| grid (best config) | trades/year | 0% | 0.04% | 0.075% | 0.1% | 0.2% |
|---|---:|---:|---:|---:|---:|---:|
| ATR Bands | 141 | +307,578% | +103,283% | +39,697% | +20,020% | +1,212% |
| Trend Zones | 681 | **+10,672%** | −43.7% | −99.4% | **−100%** | −100% |
| Close EMA Crossover | 51 | +7,737% | +5,191% | +3,651% | +2,834% | +997% |
| Close Buy/Sell Limits | 3.9 | +14,283% | +13,853% | +13,486% | +13,230% | +12,253% |
| Close ATR | 5.9 | +9,850% | +9,407% | +9,035% | +8,778% | +7,820% |
| Close Bullish EMA | 13.9 | +4,466% | +4,002% | +3,635% | +3,392% | +2,570% |
| Supertrend | 28 | +681% | +529% | +420% | +354% | +164% |
| TP/SL with ATR | 29 | +284% | +207% | +153% | +120% | +26% |
| Green Candles | 1,459 | −10.1% | −100% | −100% | −100% | −100% |
| Close RSI Reverse | 4,075 | −76.2% | −100% | −100% | −100% | −100% |

Three distinct groups:

1. **Those with a gross margin that is entirely lost to commissions**: "Trend Zones" makes 10,672% at
   zero cost and loses 100% at 0.04%. "ATR Bands" divides its result by 250 going from 0% to 0.2%.
   These are strategies whose signal contains something, but not enough to pay for execution.
2. **Those with no margin even gross**: "Green Candles" (−10% at zero commission) and "Close RSI
   Reverse" (−76%) lose even in a world without costs. No tuning saves them.
3. **Those insensitive because they trade little**: Close ATR and Close Buy/Sell Limits in their best
   configurations change by less than 20% between zero commission and 0.2%. It is the other face of
   §3.

## 7. Changing the interval: the same rule, a different job

The best configurations at 15 minutes, rerun **without touching anything** on the menu's other
intervals (`reports/intervalli.csv`):

| grid | 5m | 15m | 30m | 1h | 4h | 1d |
|---|---:|---:|---:|---:|---:|---:|
| ATR Bands | +5,629% | **+20,020%** | +9,143% | +977% | +1,938% | +248% |
| Close ATR | +286% | **+8,778%** | +1,124% | +125% | +34% | 0% |
| Close Buy/Sell Limits | +5,251% | **+13,230%** | +843% | +2,278% | 0% | +1,494% |
| Close EMA Crossover | +461% | +2,834% | +3,986% | +2,855% | **+6,740%** | +628% |
| Close Bullish EMA | +4,028% | +3,392% | +2,838% | +3,384% | +3,646% | **+4,142%** |
| Supertrend | −88% | +354% | +344% | +1,624% | **+2,073%** | +175% |
| TP/SL with ATR | −96% | **+120%** | +44% | +79% | −18% | 0% |
| Trend Zones | −100% | −100% | −42% | +810% | **+9,378%** | +4,123% |
| Green Candles | −100% | −100% | −100% | −100% | −55% | **+1,765%** |
| Close RSI Reverse | −100% | −100% | −100% | −100% | −42% | **+3,284%** |

Two opposite readings, one cause.

**What was chosen at 15 minutes loses almost everything elsewhere.** Close ATR goes from +8,778% to
+125% at one hour and to zero on the day; Close Buy/Sell Limits from +13,230% to zero on 4 hours. A
parameter chosen on a timeframe is not a parameter: it is a parameter **and** a timeframe.

**What lost everything at 15 minutes becomes the best on the day.** "Green Candles" — buying after a
green candle that exceeds the previous high — is worth −100% at 15 minutes and **+1,765%** on the day.
"Close RSI Reverse" goes from −100% to **+3,284%**. "Trend Zones" from −100% to **+9,378%** on 4
hours, i.e. **more than passive holding**. The rule did not change: the trades per year did, from
1,459 to 15, from 4,159 to 36, from 681 to 32. It is §3 seen from another angle, and it is the
clearest confirmation that the problem with these strategies is not the signal but its frequency.

The absolute best of each interval, among the configurations re-examined:

| interval | strategy | trades/year | return | Sharpe | drawdown |
|---|---|---:|---:|---:|---:|
| 5m | ATR Bands | 506 | +5,629% | 1.1 | 90.9% |
| 15m | ATR Bands | 141 | +20,020% | 1.4 | 63.7% |
| 30m | ATR Bands | 65 | +9,143% | 1.2 | 77.2% |
| 1h | Close EMA Crossover | 27 | +7,186% | 1.2 | 60.4% |
| 4h | Close EMA Crossover | 6.8 | +11,524% | 1.2 | 63.7% |
| 1d | Trend Zones | 6.7 | +8,303% | 1.2 | 75.2% |

Six intervals, six different winners, all with a Sharpe between 1.1 and 1.4 and drawdowns between 60%
and 91%: none of the six is distinguishable from the others, and none is really distinguishable from
passive holding (+7,947%, 84% drawdown). "ATR Live Trade" does not appear in this table: it simulates
thirty sub-steps per candle and on 5-minute bars it alone would cost more than twenty hours.

## 8. Defects found in the code, by measuring

> **Update**: all four were fixed in the following session; the measured effect of each correction is
> in [`strategie-nuove.md`](strategie-nuove.md) §1. The measurements in this document remain those of
> the code **before** the corrections.

1. **The `"Supetrend"` menu entry runs nothing.** `config.STRATEGIES` writes `"Supetrend"`,
   `trading_analysis`'s dispatch compares against `"Supertrend"`. Selecting it produces no signals and
   the page shows an empty backtest. The function exists and works: in the grid it returns up to
   +354%, with the least negative median of all (−12.8%).
2. **`"ATR Bands"` is not in the menu**, yet it is the strategy with the best result in the study
   (+20,020% in the optimal configuration, +1,331% already with the starting parameters). Like
   `"Close RSI Reverse"`, it has a dispatch branch and no entry that reaches it.
3. **The "Close Buy/Sell Limits" stop loss does not exist.** `buy_sell_limits_close_simulation`
   accepts `stop_loss_percent` and has the three lines that would use it commented out: the "Stop
   Loss %" widget is inert for that strategy. For Close ATR and ATR Bands it does work, and it
   systematically makes the result worse.
4. **"Trend Zones" compares a moving average with itself.** The condition is `EMA20 > EMA200`, but
   `add_technical_indicator` builds `EMA200` as the EMA **of the open** over the **same window** as
   `EMA20` (`ema_window`, default 10), not over 200 periods. The two series differ only by open
   against close, so they cross continuously: 2,080 trades a year with `ema_window=10`, and −100% in
   all six configurations. At zero commission the same strategy would return +10,672% (§6): the signal
   is there, it is the frequency that devours it.
5. **The strategies that enter at the band price assume ideal execution.**
   `atr_buy_sell_simulation` buys at `Lower_Band` when the candle's low touches it, and
   `tp_sl_simulation`/`supertrend_simulation` do the same with their levels. That is a limit order
   filled exactly at the price, with no slippage and no queue. The average trade of the best ATR Bands
   is worth **+0.46%**: a few basis points of slippage per leg cancel it. The `close_*` strategies,
   which use the close, do not have this problem.
6. **The last open position is not counted.** `simulate_trading_with_commisions` pairs signals by
   index: if you are in the market at the end of the period, that trade is not recorded. The sweep
   inherits the page's behaviour; it is one more reason to distrust configurations with very few
   trades.

## 9. Control on a second market

Everything above is measured on BTC. To know how much depends on that, the same ten grids (3,111
configurations) were rerun on **Bitfinex's ETH/USD, 2017-2019** — different asset, different exchange,
different period, different regime: passive holding does +1,482% with a 94.1% drawdown, and 2017 alone
is worth +8,902%. The store is built with
`python -m scripts.import_candles --format bitfinex --symbol ETHUSD`; the tables are the
`*_ETHUSD.csv` files in `reports/`.

The main result reproduces:

| trades/year | BTC 2017-2026: median / profitable | ETH 2017-2019: median / profitable |
|---|---:|---:|
| < 10 | −2.5% / 39.9% | −1.0% / **50.0%** |
| 10 – 30 | +19.5% / 53.6% | −14.2% / 36.8% |
| 30 – 100 | −64.9% / 22.5% | −25.3% / 38.2% |
| 100 – 300 | −90.5% / 4.2% | −66.0% / 19.0% |
| 300 – 1,000 | −99.6% / 1.6% | −88.6% / 3.0% |
| > 1,000 | −100% / 0% | −100% / 0% |

The rest holds too, with the same signs:

- **`atr_multiplier`**: below 2.0 the median is between −30% and −99% on every grid; the best is again
  2.5-3.0 (Supertrend +273%, TP/SL +127%, ATR Bands +54%). The 1.6 default stays in the losing part.
- **`num_cond=2`** beats `num_cond=1` (median −61.5% against −82.8%, profitable 19.4% against 1.3%),
  and **`rsi_sell_limit`** is again monotone: from 60 to 85 the Close Buy/Sell Limits median goes from
  −84.6% to −51.5%, the Close Bullish EMA one from −14.3% to +291%.
- **The same three strategies lose everything**: Trend Zones, Green Candles, Close RSI Reverse, with
  the same four-digit frequencies.

The differences are of level, not of direction: on ETH the profitable share is higher (21.6% against
14.9%) and eighteen configurations beat passive holding instead of five, which is explained by the
period — three years including a −81% 2018 reward staying out of the market far more than nine years
with a +7,947% do. The strategy ranking changes (here Close EMA Crossover wins with +4,992%), which is
further evidence that **choosing the best strategy does not transfer**: what transfers is the
relationship with trading frequency.

## 10. Limitations of this measurement

- **Two markets, not fifteen.** BTC/USD over the whole period, ETH/USD as a control over three years
  (§9). The relationship with frequency and the direction of the parameters reproduce on both; the
  strategy ranking and the optimal values do not.
- **One direction only.** All the strategies are long-only on an asset that did +7,947% over the
  period: staying out of the market costs, and the comparison is severe by construction.
- **No slippage, no book.** The commissions are there, the rest is not (§7.5).
- **Extreme wicks are compressed** by `clip_wicks` on read, as everywhere else in the project.
- **The sample of best configurations is small.** The two that pass out-of-sample verification make
  4-6 trades a year: 38 and 57 trades in total. With numbers like that, the difference between
  "strategy" and "luck" is not measurable with the data available.

## 11. How to reproduce

```bash
git clone https://github.com/ff137/bitstamp-btcusd-minute-data /path/to/data
.venv312/bin/python -m scripts.import_candles --source /path/to/data
.venv312/bin/python -m scripts.strategy_sweep --all --interval 15m --workers 4   # ~45 minutes
.venv312/bin/python -m scripts.sweep_report --interval 15m                       # tables in reports/
.venv312/bin/python -m scripts.strategy_focus --top 3                            # commissions and intervals

# the control on a second market (§9)
git clone https://github.com/Zombie-3000/Bitfinex-historical-data /path/to/bitfinex
.venv312/bin/python -m scripts.import_candles --format bitfinex --symbol ETHUSD \
    --source /path/to/bitfinex/ETHUSD/Candles_1m
.venv312/bin/python -m scripts.strategy_sweep --symbol ETHUSD --interval 15m --since 2017-01-01 --all
.venv312/bin/python -m scripts.sweep_report --interval 15m --symbol ETHUSD
```

With the Binance store already populated (`python -m cryptofarm.data.klines --update`) it is enough to
change `SYMBOL` in `scripts/strategy_sweep.py` to redo everything on BTCUSDT.
