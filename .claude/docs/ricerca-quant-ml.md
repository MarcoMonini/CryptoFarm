# Quantitative and ML research — two strands, measured on five assets

Date: **2026-08-26**. Universe: BTC, ETH, SOL, XRP, BNB against USDT. Spot only, **long only**, no
leverage. Data: the project's local store — 15 symbols at 5 minutes, from 614,732 to 945,675 candles
per symbol, up to **2026-08-19** — aggregated to 4h and 1d by `data/klines.py`.

A sequel to [`backtest-strategie.md`](backtest-strategie.md) and
[`strategie-nuove.md`](strategie-nuove.md), which measured **one asset**. This document closes the
two points those left open (SOL and BNB never measured; `donchian_breakout` and `squeeze_breakout` to
be re-measured after the trailing-stop fix) and opens two families the project had never tried.

---

## The result in five lines

1. **Across five assets, almost nothing beats passive holding.** 7,516 configurations of the
   historical strategies at 1d: 4.7% beat buy-and-hold. 50 out-of-sample cells of the new strategies:
   42% are profitable, 24% beat passive, and **9 of those 12 wins are in windows where passive was
   negative** — you win by standing still, not by earning more.
2. **The previous session's recommendation does not generalise.** `band_reversion_gated`, the
   strategy named as the most promising on BTC, has a negative median on 4 assets out of 5 at both
   intervals. The two that hold up everywhere are `ichimoku_trend` and `donchian_breakout` at 4h.
3. **The conclusion about the timeframe needs correcting too.** Out of sample **4h beats 1d** (52%
   against 32% of profitable cells, median +0.6% against −27.5%). "Slower is always better" was true
   between 15m and 1d on one asset; it is not monotone.
4. **The cross-sectional family — choosing *which* asset instead of *when* — is the only new thing
   that transfers.** Out of sample, on the five majors, the whole grid has a median of +62% with 89%
   of configurations profitable, against +55.6% for BTC and +37.3% for the equal-weight universe. But
   **choosing the parameters does not transfer at all** (ρ = −0.69) and the effect **disappears when
   the universe widens to 15 assets** (median −0.9%).
5. **The reformulated ML strand has a real advantage, but below the significance bar.** A meta filter
   on top of a genuine primary strategy gets AUC 0.537 both in and out of sample — not 0.50 — and
   raises the net per trade. Against a **random selection of the same size**, though, it stays at the
   80th-98th percentile, and having tried some twenty combinations, a 98th percentile is what chance
   is expected to produce.

---

## 1. What the state of the art says, read in the repositories

Nine repositories read directly (README, documentation and benchmark tables downloaded on
2026-08-26, not recalled). What follows is **knowledge derived from the sources**, kept distinct from
the hypotheses further down.

### 1.1 The single most useful number: the ceiling on ML alpha

`microsoft/qlib` publishes the benchmark table of ~20 models on `Alpha158` and `Alpha360` (CSI300,
mean and deviation over 20 seeds). It is the most honest measurement available of what a prediction
model is really worth on a heavily studied market:

| model | dataset | IC | Rank IC | annual return |
|---|---|---:|---:|---:|
| DoubleEnsemble | Alpha158 | **0.0521** | 0.0502 | 11.6% |
| XGBoost | Alpha158 | 0.0498 | 0.0505 | 7.8% |
| LightGBM | Alpha158 | 0.0448 | 0.0469 | 9.0% |
| TRA | Alpha158 | 0.0440 | **0.0540** | 7.2% |
| Localformer | Alpha158 | 0.0356 | 0.0468 | 4.4% |
| Transformer | Alpha158 | 0.0264 | 0.0407 | 2.7% |
| TabNet | Alpha158 | 0.0204 | 0.0333 | 2.3% |
| Transformer | Alpha360 | 0.0114 | 0.0327 | **−2.7%** |
| TabNet | Alpha360 | 0.0099 | 0.0290 | **−3.7%** |

Three things this table establishes, and that no document in the project had:

- **the ceiling is IC ≈ 0.05.** A 5% correlation between prediction and future return is the most
  that twenty architectures produce on a broad, clean equity panel. The AUC 0.54 that `strategy.md`
  §Synthesis reported as a failure **is the field's normal level**, not an anomaly;
- **gradient boosting is not beaten by deep models.** The top three places by IC are tree ensembles;
  Transformer and TabNet on `Alpha360` produce a *negative* annual return. It confirms the `gbdt`
  choice already made in `ml/models.py`;
- **that ceiling is monetised in the cross section, not in time.** All those numbers come from a
  portfolio built by ranking ~300 stocks every day. An IC of 0.05 on *one* asset produces nothing
  executable; on a wide cross section it does, because the error averages out. **This is the point
  that separates the state of the art from everything the project has tried so far**, where every
  measurement has always concerned one symbol at a time.

### 1.2 `machine-learning-for-trading` (third edition) — the method, not the techniques

The third edition is reorganised around a single process with an *evidence boundary* separating
tuning from evaluation, plus a retrain/pause/retire cycle for when the edge decays. What is relevant
here:

- **transaction costs and risk management are whole chapters** (18 and 19), not appendices, and they
  come on top of portfolio construction (17) and strategy synthesis (20): a raw signal has to be
  carried all the way to a sized portfolio, with costs and risk, before saying it works;
- **explicit anti-overfitting tools**: Deflated Sharpe Ratio, Rademacher Anti-Serum, White's Reality
  Check, conformal prediction, walk-forward everywhere. The project already has DSR and PBO in
  `ml/validation.py`, and did not use them on the three-action policy (`strategy.md` §14);
- **nine case studies on the same process**, among them one on **8-hour crypto perpetuals based on
  the funding rate** and one intraday at 15 minutes on order-book microstructure. The first is the
  collection's only crypto case, and it is not a directional strategy: it is funding arbitrage;
- ML4T deals explicitly with *multiple testing* in factor research. It is the right lens for reading
  the results of §5 below.

### 1.3 `freqtrade` / FreqAI — the only crypto ML system in production

FreqAI is freqtrade's ML module, and it is interesting for the shape of the system, not for the
models:

- **self-adaptive retraining** on a sliding window, in a thread separate from inference, and a
  backtest that *emulates* periodic retraining instead of training once on the past;
- **automatic feature expansion** along four axes: `indicator_periods_candles` ×
  `include_timeframes` × `include_shifted_candles` × **`include_corr_pairs`**. The last is decisive:
  in FreqAI's canonical design, *every* pair's features include the indicators of the correlated
  pairs. It is the same principle as §1.1 — the cross-sectional context enters the model;
- **outlier removal** as part of the pipeline (Dissimilarity Index, SVM, DBSCAN) and PCA for
  dimensionality reduction;
- an operational constraint worth recording: FreqAI does **not** combine with dynamic pairlists,
  because the training data has to be downloaded at startup.

### 1.4 The other five

- **`jesse`** — crypto framework for backtesting and live trading. Two tools the project does not
  have and that would be useful immediately: a **significance test for the entry rule** ("could this
  edge have appeared by chance?") and **Monte Carlo analysis** with shuffling of the trade order. It
  also has partial fills, optimisation with Optuna+Ray, and an integrated ML pipeline.
- **`AI4Finance/FinRL`** — financial RL at three levels (environment / agent / application). The
  repository itself now describes itself as a "classic workflow for teaching, experimentation and
  research prototyping" and points to FinRL-X for production. To be read as a laboratory, not as a
  ready-made system.
- **`vnpy`** — a trading platform (gateways, execution, order management). It solves the
  infrastructure, not the alpha.
- **`TauricResearch/TradingAgents`** — LLM agents in roles (analysts, researchers, traders, risk).
  The repository documents its own reproducibility limits (reasoning models ignore temperature). Not
  a source of signal measurable on OHLCV.
- **`Rachnog/Deep-Trading`** — the README states "released part one - simple time series
  forecasting". Historical interest (2017): it is the generation of work the idea that a deep network
  on price is enough comes from, which the qlib benchmarks of §1.1 refute.
- **`awesome-quant`** — an index. Useful for the backtesting, portfolio-optimisation and factor
  analysis sections.

### 1.5 What comes out of it, in three principles

These are not hypotheses: they are conclusions supported by the sources above.

1. **The cross section is where weak alpha becomes executable.** (qlib, FreqAI `corr_pairs`, ML4T
   ch. 17)
2. **The economic constraint goes inside the target, not checked afterwards.** (ML4T ch. 18; and in
   this repository `strategy.md` §13 learned it the hard way)
3. **Gradient boosting is the reference; deep architectures have to be justified.** (qlib benchmarks)

---

## 2. The quantitative strand — what the measurements produced

Command: `strategy_lab` (5 two-sided strategies, 592 configurations) and `strategy_sweep` (the 11
historical ones, 3,129 configurations) on each of the five symbols, at 1d and 4h, from 2021-01-01,
commission 0.05% per leg. Long side only, as mandated.

### 2.1 The references to beat

| symbol | 2021-2023 | 2024-today | whole period | drawdown |
|---|---:|---:|---:|---:|
| BTC | +44.2% | +55.6% | +134.4% | 76.6% |
| ETH | +213.1% | −10.9% | +187.5% | 79.3% |
| SOL | +5,422.0% | −25.5% | +4,346.0% | 96.3% |
| XRP | +159.2% | +69.7% | +349.8% | 83.2% |
| BNB | +725.4% | +97.4% | +1,538.0% | 70.9% |

### 2.2 The new strategies, on five assets (in sample)

Grid median, long only, ≥10 trades:

**1 day**

| strategy | BNB | BTC | ETH | SOL | XRP | profitable (mean) | above passive |
|---|---:|---:|---:|---:|---:|---:|---:|
| squeeze_breakout | +450.9% | +85.6% | +34.2% | +168.0% | +74.5% | 83% | 0-26% |
| donchian_breakout | +34.4% | +27.1% | +17.0% | +315.1% | +28.9% | 80% | 0-4% |
| ichimoku_trend | +36.8% | −5.3% | +35.1% | +113.7% | −14.0% | 74% | 0% |
| band_reversion_gated | −34.8% | +9.2% | −39.1% | −81.6% | +10.3% | 32% | 0% |
| trend_pullback | −46.9% | −9.3% | −26.1% | +119.7% | −77.5% | 27% | 0% |

**4 hours**

| strategy | BNB | BTC | ETH | SOL | XRP | profitable (mean) | above passive |
|---|---:|---:|---:|---:|---:|---:|---:|
| ichimoku_trend | +1,008.4% | +35.9% | +38.0% | +434.9% | +177.7% | 83% | 0-33% |
| donchian_breakout | +374.5% | +40.0% | +68.7% | +199.7% | +171.1% | 87% | 0-23% |
| squeeze_breakout | +16.1% | −0.9% | −13.8% | +29.9% | +20.0% | 53% | 0% |
| trend_pullback | 0.0% | −21.5% | −15.8% | +17.8% | +24.7% | 51% | 0% |
| band_reversion_gated | −16.1% | −10.6% | −29.8% | +11.6% | −11.3% | 31% | 0% |

Three readings, all against a prior expectation:

- **`band_reversion_gated` does not generalise.** On BTC 1d it stays acceptable (median +9.2%, Sharpe
  0.20), but it is negative on BNB, ETH and SOL at both intervals, with a median Sharpe down to
  −0.57. Recommendation #2 of `strategie-nuove.md` §7 holds for one asset, not for the family.
- **The trailing-stop fix promoted `donchian_breakout`.** Re-measured, it is the second most regular
  (75-100% of configurations profitable at 4h). The open point in `strategie-nuove.md` §8 is closed:
  the fix improves things, as the synthetic test suggested.
- **`ichimoku_trend` at 4h is the most solid in sample**: a positive median on all five, median
  Sharpe from 0.33 to 1.08, and the only cell that beats passive in a third of cases (BNB).

### 2.3 Out of sample: 50 cells, and the verdict

Chosen on 2021-2023, performance on 2024-today, for each of the 5 strategies × 5 symbols × 2
intervals:

| | value |
|---|---:|
| cells with a positive return | **21 / 50 (42%)** |
| cells that beat passive holding | **12 / 50 (24%)** |
| … of which in windows where passive was **negative** | **9 out of 12** |
| median out-of-sample return | **−8.9%** |
| median in↔out correlation (ρ) | 0.26 (positive in 70% of cells) |

By strategy:

| strategy | positive | beat passive | median return | median ρ |
|---|---:|---:|---:|---:|
| ichimoku_trend | 6/10 | 4/10 | **+13.5%** | 0.38 |
| squeeze_breakout | 7/10 | 3/10 | +6.5% | −0.04 |
| band_reversion_gated | 4/10 | 3/10 | −2.5% | 0.38 |
| donchian_breakout | 2/10 | 2/10 | −31.5% | −0.02 |
| trend_pullback | 2/10 | 0/10 | −39.4% | 0.22 |

By interval:

| interval | positive cells | beat passive | median |
|---|---:|---:|---:|
| 1d | 32% | 20% | −27.5% |
| **4h** | **52%** | **28%** | **+0.6%** |

**`ichimoku_trend` is the only one that transfers**: positive median, ρ 0.38, and 4 wins out of 10
against passive. And it is also the only one whose in-sample maximum is not a grid artefact — its
grid has 11-12 configurations, not 256.

**The reversal on the timeframe has to be stated plainly**: `strategie-nuove.md` §7 point 1 concluded
"daily scale, not 15 minutes", and the implicit generalisation was "slower is better". Across five
assets it is not so: 4h beats 1d on every out-of-sample metric. The true rule is narrower — **there
is an intermediate interval where the margin per trade exceeds the cost and the trades stay numerous
enough not to depend on three of them**; at 15m the cost wins, at 1d the sample gets too small.

### 2.4 The historical strategies, on five assets

7,516 configurations at 1d, commission 0.05%: median **+43.9%**, **72%** profitable, and **only 4.7%
beat passive holding**. The only cell with a systematic advantage is `Close EMA Crossover` on BTC
(75% of configurations above passive), which does not repeat on any other asset (0%).

On SOL and BNB the medians are spectacular in absolute terms (`Green Candles` +1,778%, `Close RSI
Reverse` +908%) and irrelevant in relative ones: passive did +4,346% and +1,538%.

### 2.5 The new family: cross-sectional rotation

`scripts/cross_section.py`. At each rebalance the assets are ranked by relative strength (return over
`lookback` bars) and the top `top` are held at equal weight; anything with negative strength is not
bought and its share stays in cash. A variant with a single regime switch (out of the market when BTC
is below its 50-bar average). Commission 0.1% per leg (spot fee schedule). 160 configurations:
`lookback` ∈ {10,20,30,60,90}, `top` ∈ {1,2,3,5}, rebalance every {1,3,7,14} bars, regime ∈ {none,
BTC}.

**In sample, 2021-2026, five majors** (BTC passive +134.4%; equal-weight universe +1,311.1%, Sharpe
0.98, DD 91.0%):

| | value |
|---|---:|
| grid median | +1,179.7% |
| profitable configurations | 100% |
| above BTC | 95.6% |
| **above the equal-weight universe** | **44.4%** |
| best by Sharpe (lb 20, top 3, weekly, BTC regime) | +3,508.2%, **Sharpe 1.60**, DD **45.7%** |

The row that counts is the fourth: **the rotation's median does not beat holding the same five at
equal weight.** The huge absolute numbers come from the universe, not from the rotation. The real
advantage is on risk: DD 45.7% against 91.0%, Sharpe 1.60 against 0.98 — again the same shape of
result already seen on `band_reversion_gated`.

**Out of sample, 2024-today** (BTC +55.6%; equal-weight universe +37.3%, Sharpe 0.50):

| | five majors | fifteen assets |
|---|---:|---:|
| median of the whole grid | **+62.0%** | −0.9% |
| profitable configurations | 89% | 49% |
| above BTC | 52% | 16% |
| above their own universe | 65% | 56% |
| median Sharpe | 0.66 (against 0.50) | 0.28 (against 0.23) |
| in↔out ρ over the top 10 | **−0.69** | −0.15 |

Four conclusions, in order of solidity:

1. **The family transfers where single-asset strategies do not.** 89% of configurations profitable
   out of sample against 42% of positive cells in §2.3 is the largest difference measured in this
   document.
2. **Choosing the parameters does not transfer, and in fact harms.** ρ = −0.69: taking the best
   in-sample configuration is worse than taking one at random. The operational consequence is
   precise — **do not optimise**: take a central configuration (lookback ~20-30 bars, top 2-3, weekly
   rebalance) and leave it alone.
3. **The wide universe does not work.** At 15 assets the out-of-sample median is −0.9% and only 16%
   beat BTC. More assets is not more diversification: it is more alt-coins that lost over 2024-2026
   (their passive basket does −9.5%). The effect lives in the large caps.
4. **The cost bites but does not kill**, at a weekly rebalance: the share above the universe goes
   from 50.6% (0.02%/leg) to 44.4% (0.1%) to 28.1% (0.3%).

### 2.6 The pairs (BTC/ETH, ETH/SOL, …)

Holding the stronger of two is the same procedure with a universe of two and `top=1`. No short leg,
so it is compatible with the mandate.

**In sample** it beats the passive "half and half" in **9 pairs out of 10**. But the pair without an
extraordinary winner is also the only one that loses:

| pair | rotation | half and half | outcome |
|---|---:|---:|---|
| **BTC/ETH** | +156.0% | +160.9% | **loses** |
| ETH/SOL | +29,325.7% | +2,266.7% | wins (SOL does ×44) |
| BTC/SOL | +8,882.0% | +2,240.2% | wins (SOL) |
| SOL/BNB | +9,639.2% | +2,942.0% | wins |

That is: in sample, rotating between two "works" when there is an asset that multiplies by forty and
the rule finds it. It is not an edge, it is concentration with hindsight on the universe.

**Out of sample (2024-today) the picture is more interesting and more credible**: it beats half-and-
half in 7 pairs out of 10, and **on BTC/ETH — the pair without an outlier — it does +136.2% against
+22.4% for passive and +55.6% for BTC alone**, avoiding ETH in its weak phase. It is the cleanest
out-of-sample result in the whole document, and it is also a single sample: one pair, one window.

### 2.7 The quantitative architecture that follows

**Not** a single strategy: two layers with different roles.

```
   layer 1 — cross-sectional selection (what)
      rank the 5 majors by relative strength over 20-30 daily bars
      hold the top 2-3 at equal weight, rebalance every 7 bars
      negative strength -> cash, not "the least bad"
      single regime switch: BTC below its 50-bar average -> all to cash

   layer 2 — per-asset timing (when)  [optional, see §5]
      ichimoku_trend at 4h, long only, central grid parameters
      applied only to the assets layer 1 has selected
```

The reasons for each piece are measured, not chosen for symmetry: layer 1 because it is the only
family that transfers (§2.5); central parameters instead of optimal ones because ρ = −0.69 (§2.5);
the single switch because in crypto correlation goes to one on the way down, and selecting the best
of five that are falling does not protect; `ichimoku_trend` because it is the only per-asset rule
with a positive out-of-sample median (§2.3); the ban on shorts because it is measured at a loss in
all five strategies (`strategie-nuove.md` §5) and is outside the mandate anyway.

---

## 3. The ML strand — the reformulation, and what it produced

### 3.1 Why we do not restart from where we had got to

`strategy.md` §13 measured that entering on the confirmation of a low and exiting on the confirmation
of a high captures **zero on average** across 15 symbols at every threshold, *before* costs: the
confirmation is paid twice and the median leg is worth 1.76-2.05 of it. That is not a result about
the model, it is a property of the scheme. No feature, architecture or threshold moves it, and that
strand stays closed.

What remained open is another formulation, which `strategy.md` §2.3 recommended and §13.4
reformulated: **the model does not decide when to buy — it decides whether to let through a signal
that a strategy has already produced.**

### 3.2 The design: `scripts/meta_gate.py`

- **Primary**: a strategy from `strategies_ls.py`, fixed central parameters, never optimised here
  (optimising primary and filter together is the classic way of reading the noise twice).
- **Sample**: one row per **trade**, not per bar. Thousands instead of millions, and that is an
  advantage: the bars inside a trend are the same observation repeated.
- **Label**: `1` if the trade, executed as the strategy would execute it, closes **above costs**. The
  economic constraint is inside the target: you cannot be right on the sign and lose money.
- **Features (16)**: all scale-free and known at the entry bar — distances from EMA50/EMA200 in ATR
  units, position in the Donchian channel and in the Bollinger bands, relative ATR, ADX, band width,
  StochRSI, MFI, OBV slope, relative volume, relative range, above/below EMA200 — **plus three
  cross-sectional ones no model in the project had ever had**: relative-strength rank in the
  universe, market breadth (share of assets above their own 50-bar average), strength against BTC. It
  is principle §1.5.1, applied.
- **Universe**: all 15 symbols in common, 4 hours.
- **Model**: `HistGradientBoostingClassifier` (§1.5.3).
- **Validation**: `PurgedKFold` with embargo from `ml/validation.py` — trades overlap, and an
  ordinary k-fold would measure on a future already seen — plus a separate temporal check (trained up
  to 2024-01-01, measured afterwards).
- **Control**: for each threshold, 500 **random selections of the same size**. With a long-tailed
  primary a few lucky trades are enough to raise the average net: the number to beat is not zero, it
  is the high percentile of chance.

### 3.3 The results

| primary | trades | AUC (purged CV) | AUC (temporal check) | precision without → with filter |
|---|---:|---:|---:|---:|
| `trend_pullback` | 3,098 | 0.537 | 0.538 | 50.6% → 53.4% |
| `donchian_breakout` | 1,718 | 0.512 | 0.514 | 35.7% → 38.2% |
| `squeeze_breakout` | 1,634 | **0.504** | 0.504 | no advantage |
| `band_reversion_gated` | 156 | 0.531 | 0.574 | sample too small |

The economic account out of sample (2024-today), net per trade, with the random control:

**`trend_pullback`** — 1,383 trades in the check window

| threshold | trades | precision | mean net | p95 of chance | percentile within chance |
|---|---:|---:|---:|---:|---:|
| none | 1,383 | 50.6% | −0.124% | — | — |
| 0.50 | 864 | 52.9% | −0.022% | +0.044% | 84th |
| 0.60 | 579 | 53.4% | **+0.064%** | +0.152% | 87th |

**`donchian_breakout`** — 802 trades in the check window

| threshold | trades | precision | mean net | p95 of chance | percentile within chance |
|---|---:|---:|---:|---:|---:|
| none | 802 | 35.7% | +0.146% | — | — |
| 0.45 | 275 | 38.5% | +0.818% | +0.716% | 97th |
| **0.50** | 217 | 38.2% | **+1.062%** | +0.886% | **98th** |
| 0.55 | 164 | 37.8% | +0.510% | +0.905% | 78th |
| 0.60 | 116 | 37.1% | +0.694% | +1.164% | 82nd |

### 3.4 How they should be read

**The ranking advantage is real.** AUC 0.537 in purged cross-validation and 0.538 in a completely
separate temporal check is not noise: two independent measurements agreeing to three decimals. And it
is exactly the ceiling of §1.1 — this is not a disappointment, it is the field.

**The economic advantage does not clear the control.** The best case (`donchian_breakout`, threshold
0.50) sits at the 98th percentile of 500 random selections of the same size. On its own that would be
p ≈ 0.02. But **four primaries by five thresholds were tried**: among twenty combinations, a 98th
percentile is what chance is expected to produce. The adjacent thresholds of the same primary drop to
the 78th and 82nd, which is the behaviour of noise, not of an effect.

**AUC is the wrong metric for a long-tailed primary.** `donchian_breakout` has AUC 0.512 — nearly
indistinguishable from chance — and the largest economic improvement of all, because a selection
barely better than chance on a right-tailed distribution captures a disproportionate share of the few
moves that pay. It is also the reason it cannot be trusted: the same mechanism makes the result
depend on a handful of trades.

**The filter improves a losing primary without making it a winning one.** `trend_pullback` out of
sample goes from −0.124% to +0.064% per trade: the filter removes the minus sign and stops there.

---

## 4. Critical analysis

### 4.1 Limits that apply to both strands

- **Survivorship in the universe.** BTC, ETH, SOL, XRP, BNB are the large caps **of 2026**. In
  January 2021 SOL was not a major. Every number in §2.5 and §2.6 contains this selection, and §2.5
  shows it: most of the in-sample result comes from SOL's ×44. The partial defence used here —
  always comparing against **the equal-weight universe**, which carries the same bias — is the right
  one, and indeed it lowers the verdict from "95.6% above BTC" to "44.4% above the universe".
- **A single cycle.** 2021-2026. `strategie-nuove.md` §2 has already measured that the parameters do
  not carry over from the 2017-2020 cycle to this one. There is no reason to believe they will carry
  over to the next.
- **Test multiplicity.** Between this document and the two before it, over 12,000 configurations have
  been evaluated. No result here has been corrected for multiplicity with Deflated Sharpe or PBO —
  the tools are in `ml/validation.py` and have not been applied to the quant side. It is the biggest
  methodological debt.
- **Ideal execution.** Entries at the bar close, fills at the exact price, no slippage or impact. On
  crypto liquidation gaps that is optimistic, and more so for cross-sectional rotation, which at
  `top=1` concentrates all the capital in one asset.
- **No liquidity model.** On BTC/ETH irrelevant at household size; on smaller assets with a daily
  rebalance, not.

### 4.2 Where each approach fails

| | quantitative (rotation + rules) | ML (meta filter) |
|---|---|---|
| **fails when** | the market has no dispersion between assets (everything moves together): the ranking becomes noise and only commissions get paid | the regime changes shape relative to training — the features stay defined but their relation to the outcome does not |
| **fails silently?** | no: high turnover and a flat result are visible immediately | **yes**: the probability keeps coming out well calibrated while it no longer selects anything |
| **how much it costs to find out** | weeks | months, unless the random control is kept running continuously |
| **overfitting risk** | high on the choice of parameters (ρ = −0.69), low on the family | high: 16 features, 3,098 rows, long tail |
| **interpretability** | total: you can say why you own an asset | medium: the model gives a rank, not a reason |

### 4.3 What it would take to validate

In order of value-to-cost ratio:

1. **A second cycle.** Redo §2.5 and §3.3 on 2017-2020 with the universe that existed then (BTC, ETH,
   XRP, BNB, LTC). If the cross-sectional rotation transfers there too, the state of the evidence
   changes; the data is already in the store.
2. **Deflated Sharpe and PBO on the quant side.** The code exists. It should be applied to the
   rotation grid, which is where a decision is about to be taken.
3. **Rule significance testing and Monte Carlo on the trade order**, as in `jesse` (§1.4). The random
   control of §3.2 is the same idea applied only to the ML filter: it should be extended to the
   strategies.
4. **A fill model**, which `strategy.md` Phase 0.3 lists as a gate never satisfied. Until it exists,
   every number in maker mode remains unverifiable.

---

## 5. Direct comparison

| criterion | cross-sectional quant | per-asset quant | ML (meta filter) |
|---|---|---|---|
| **what it detects well** | which asset is leading; the absence of strength across the whole universe | price structure (breakouts, cloud, bands) on a single instrument | which signals of a primary have the wrong context |
| **what it does not detect** | the moment of entry; nothing below the rebalance bar | the relative context: it buys the breakout of a weak asset just like a strong one | it generates nothing: without a primary it has no input |
| **out-of-sample evidence** | **89% of configurations profitable**, median +62% (5 assets) | 42% of cells positive, median −8.9% | AUC 0.538 stable; economic advantage at the 80th-98th percentile of chance |
| **does it beat passive holding?** | 52% of configurations beat BTC, 65% beat the universe | 24% of cells, and 9 out of 12 only because passive was losing | not applicable on its own |
| **overfitting risk** | low on the family, **very high on the parameters** | high (grids of up to 256 configurations per cell) | high |
| **interpretability** | high | high | medium |
| **implementation complexity** | low (~250 lines, no new dependencies) | already in production | medium (features + purged validation + random control) |
| **compute cost** | seconds | minutes | minutes |
| **adaptation to regime change** | the regime switch is explicit and checkable | depends on the internal filters | requires retraining; degrades silently |
| **trading frequency** | 17-30 annual turnover (weekly rebalance) | 3-27 trades/year per asset | inherits the primary's |

The comparison is not like-for-like on one point that has to be said: **cross-sectional quant and ML
do not answer the same question.** The first allocates between assets, the second filters the signals
of a strategy. They are complementary by construction, which is also why the integration test of §6
has to be approached with suspicion — apparent complementarity is almost always just the fact that
the two are looking at different things.

---

## 6. A hybrid: is it worth it?

The criterion declared up front was: **integrate only if the two bring genuinely different
information or decision-making capability.** Let us apply it.

**What the hybrid would be.** The cross-sectional layer chooses *which* assets to own; the meta
filter decides whether a primary signal on that asset deserves to pass. The information is formally
distinct (rank between assets ≠ probability of a single signal succeeding) and so are the decisions
(allocation ≠ entry).

**But the evidence is not there.** Three measured reasons:

1. **The meta filter did not clear its own control** (§3.4). Composing a layer that has not proved
   its worth with one that has can only make the second worse or leave it as it is.
2. **The cross-sectional features are already inside the filter** (strength rank, breadth, strength
   against BTC): the model already has the information layer 1 would use, and the AUC that comes out
   is 0.537. The information channel the hybrid is supposed to open is therefore already open, and it
   is worth little.
3. **The layers eat each other's sample.** The rotation holds 2-3 assets out of 5; applying a filter
   that discards half the signals on a sixth of the original opportunities leads to a number of
   trades on which nothing can be measured any more. It is the defect that made
   `band_reversion_gated` unassessable in §3.3 (156 trades in five years across fifteen assets).

**Conclusion: do not integrate now.** A weak form of hybrid should be kept, though — one that is
effectively free and that the data supports: **layer 1's single regime switch is already a
conditional filter**, and it applies to the whole portfolio. It is the only composition between the
two worlds for which a measurement exists (Sharpe 1.60 and DD 45.7% with the switch, against 0.98 and
91.0% for passive).

The moment to reopen the question is precise: **if and when the meta filter beats the random control
on a second market cycle.** Not before.

---

## 7. Recommendations

### The strongest quantitative approach

**Cross-sectional rotation on the five large caps, at fixed central parameters, with a regime
switch.**

- lookback 20-30 daily bars, `top` 2-3, weekly rebalance, negative strength to cash, out of the
  market when BTC is below its 50-day average;
- **do not optimise the parameters** — it is the most unusual and best supported operational
  recommendation in this document: ρ = −0.69 between in-sample and out-of-sample return;
- universe of **five, not fifteen**: widening destroys the result (out-of-sample median from +62% to
  −0.9%);
- declared expectation: not a return above passive, but **the same order of return with roughly half
  the drawdown**. Out of sample it gave more than passive (median +62% against +55.6% for BTC), but
  over a single window and with a universe chosen in hindsight.

As a per-asset reference, and **only** as a reference: `ichimoku_trend` at 4h, long only, central
parameters. It is the only one of the per-asset rules with a positive out-of-sample median (+13.5%)
and ρ 0.38. Any new strategy that does not beat it on those two columns does not deserve to go into
production — it is the same bar `strategie-nuove.md` §7 set, now verified on five assets instead of
one.

### The strongest ML approach

**A meta filter on top of a genuine primary, with cross-sectional context in the features — to be
kept in research, not in production.**

The design is correct and should be preserved: the economic constraint inside the label, the sample
per trade, the assets in common, the purged validation, and above all **the control with a random
selection of the same size**, which is what stopped a +1.06% per trade being mistaken for a
discovery. The measured ranking advantage (AUC 0.537, stable across two independent validations) is
consistent with the state of the art's ceiling and is not nothing; it is simply not yet enough to
pay.

The next step is **one**, not three: redo the same measurement on the 2017-2020 cycle. If the
economic advantage clears the random control there too, then it is an effect. If it does not, the
strand closes with a measurement instead of an opinion.

### The hybrid

**No, for now.** With the single exception of the regime switch, which is already inside the
quantitative layer. Reopen only after the point above.

### What not to redo

- **Do not reopen the three-action policy on directional change.** `strategy.md` §13: it captures
  zero on average, before costs, on every symbol and at every threshold.
- **Do not add the short side.** Measured at a loss on all five strategies (`strategie-nuove.md` §5),
  and outside the mandate anyway.
- **Do not widen the universe to "diversify".** Measured: it gets worse.
- **Do not look for deep architectures before the cross section is exhausted.** The qlib benchmarks
  (§1.1) show Transformer and TabNet with a negative annual return where gradient boosting is
  positive.
- **Do not trust a grid's maximum.** It still holds, and §2.6 shows the new shape of the same error: a
  pair that "wins" because it contained an asset that did ×44.

---

## 7bis. What was applied to the simulator

The recommendations above were put into the code on 2026-08-26, with this criterion: **an entry stays
in the menu if, out of sample, it has a positive median or at least two cells out of ten above passive
holding.**

**Seven entries removed.** Trend Pullback (0/10 above passive, median −39.4%), Close ATR (0/10,
−17.2%), Close Buy/Sell Limits (0/10, −1.8%), Close Bullish EMA (0/5, −2.8%), ATR Live Trade (1/10,
−26.2%), Band Reversion (negative on four assets out of five), Green Candles. The last one deserves a
line: it beats passive in 3 cells out of 10, but its **top ten configurations are 0% profitable** —
the wins come from configurations that lose in the rest of the grid, which is the signature of luck
and not of a rule. All of them stay in `strategies.py` and in the golden master: they left the menu,
not the repository.

**One entry put back: Close RSI Reverse.** It was excluded *on purpose*, with a written reason —
"at a total loss in all 25 configurations tried". That measurement was at **15 minutes**. At a daily
scale the same rule does 24-27 trades a year, has a positive median on all five symbols (from +44.9%
to +906.9%) and 72-92% of configurations profitable; at 4 hours it does 160 a year and on BTC loses
45.8%.

> The old verdict was not wrong: it was **tied to the interval**, and nobody had written that next to
> it. It is the clearest case of the rule these documents have been repeating for three sessions —
> trading frequency explains almost everything — and it is worth generalising: **a strategy excluded
> on one interval is not excluded on all of them**, and an exclusion without the interval next to it
> is a wasted measurement.

**One new view.** Cross-sectional rotation is in the page as a second view (`trading/rotation.py` +
`simulator.rotation_page`), not as a menu entry: it chooses between assets, while the menu chooses
within one asset. The initial values are the central ones of §7, the sidebar explains why they should
not be optimised, and the reference drawn alongside is the equal-weight universe. It reads the local
store and not the exchange, so in production it warns instead of raising.

What was **not** put into the page: the meta filter of §3. It did not clear its own control, and a
view would make it look like a result.

## 8. Reproducing

```bash
# per-asset grids (5 symbols × 2 intervals), then the tables
for S in BTCUSDT ETHUSDT SOLUSDT XRPUSDT BNBUSDT; do
  for I in 1d 4h; do
    .venv312/bin/python -m scripts.strategy_lab --all --symbol $S --interval $I \
        --since 2021-01-01 --workers 8
    .venv312/bin/python -m scripts.lab_report --symbol $S --interval $I
  done
  .venv312/bin/python -m scripts.strategy_sweep --all --symbol $S --interval 1d \
      --since 2021-01-01 --fee 0.05 --workers 8 --suffix _2021_fee005
done

# cross-sectional rotation
.venv312/bin/python -m scripts.cross_section --selfcheck
.venv312/bin/python -m scripts.cross_section --universe majors --interval 1d --grid --save cs_majors_1d
.venv312/bin/python -m scripts.cross_section --universe majors --interval 1d --split --save cs_majors_1d_oos
.venv312/bin/python -m scripts.cross_section --universe wide   --interval 1d --split
.venv312/bin/python -m scripts.cross_section --pairs --lookback 20 --every 7
.venv312/bin/python -m scripts.cross_section --pairs --lookback 20 --every 7 --since 2024-01-01

# meta filter
.venv312/bin/python -m scripts.meta_gate --selfcheck
.venv312/bin/python -m scripts.meta_gate --strategy trend_pullback    --universe wide --interval 4h --oos 2024-01-01
.venv312/bin/python -m scripts.meta_gate --strategy donchian_breakout --universe wide --interval 4h --oos 2024-01-01
```

Tables in `reports/` (`lab_*_*USDT_*.csv`, `cs_*.csv`, `meta_*.csv`).

### One code fix, found by running it

`strategy_lab` and `strategy_sweep` passed the candles to the child processes through global
variables filled by the parent, relying on `fork`. **On macOS `ProcessPoolExecutor` uses `spawn`**,
the globals do not arrive and every multi-worker run died with `KeyError: ('SOLUSDT', '1d')`. Fixed in
both by having the worker rebuild what it needs (`prepare` is idempotent, so under `fork` the line
costs nothing). Without this fix none of the measurements in this document would be runnable on the
user's machine.

### New code

| file | what | verification |
|---|---|---|
| `scripts/cross_section.py` | cross-sectional rotation: grid, out of sample, pairs; accounting in value per asset (not in normalised weights, which erased the cash share) | `--selfcheck`: 5 assertions — capital flat at flat prices, expected return on a ramp, monotonicity with respect to commission, **the cash share does not disappear**, no look-ahead when the series is truncated |
| `scripts/meta_gate.py` | meta filter: sample per trade, 16 scale-free features of which 3 cross-sectional, `PurgedKFold`, control with random selection | `--selfcheck`: a signal planted in the features must be found (AUC > 0.75) and pure noise must not be (0.40 < AUC < 0.60) |

`ruff` and `black` clean; 543 tests pass.
