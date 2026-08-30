# Handoff — CryptoFarm

Date: **2026-08-30**. Working branch: **`claude/audit-confluenza`**.
The previous `claude/ricerca-quant-ml-cinque-asset` was pushed. The earlier
`claude/trading-strategies-performance-fb39oc` was merged into `main` with PR #7.
The older branch `ai-labeling-rewrite` (3-state ML pipeline) is **closed with a negative result**
and was never merged: see `.claude/docs/strategy.md` §10-13 and the "The ML strand" section
below.

## Do not duplicate: read these first

| document | what it holds |
|---|---|
| `CLAUDE.md` | repo architecture, commands, environment variables, Docker/Render constraints. **Every folder then has its own `README.md`** with the files and functions it contains |
| `.claude/docs/README.md` | the reading order for everything else |
| `.claude/docs/labeling-strategy.md` | **the labeling: labels from −1 to +1 oscillating between local extremes, several windows, and a temporal smoothing.** Read it before touching labels or training |
| `.claude/docs/modello-ingresso.md` | **the first document to read on the model strand** (2026-08-29). The model at the head today, and the project's first numbers that pass the paired-exposure control |
| `.claude/docs/modello-swing.md` | (2026-08-28, updated 2026-08-30) the audit that closed `leg_model`, the extreme-proximity label, and the measurements by which that model does **not** beat chance at equal exposure |
| `.claude/docs/politica-rl.md` | (2026-08-28) the RL policy, wired in: it beats passive holding 11/15 out of sample and halves the maximum drawdown, but the *when* is above chance only weakly |
| `.claude/docs/strategia-confluenza.md` | (2026-08-27, **measured 2026-08-28**) the multi-timeframe multi-signal strategy. No look-ahead, uncorrelated voters, and **it does not beat passive holding**: every parameter's gradient points at not trading. The conclusions are at the end |
| `.claude/docs/ricerca-quant-ml.md` | (2026-08-26) state of the art from the nine repositories, the two strands measured on BTC/ETH/SOL/XRP/BNB, cross-sectional rotation, the meta filter. It corrects two conclusions of `strategie-nuove.md` that do not generalise |
| `.claude/docs/strategie-nuove.md` | the four corrections and what they changed, the 2021-2026 cycle as the dataset and why, five new strategies, the two-sided engine with the short, leverage and costs |
| `.claude/docs/backtest-strategie.md` | **the simulator's strategies measured over nine years.** 3,129 configurations, parameter sensitivity, out-of-sample behaviour, four code defects found by measuring (§8, now fixed) |
| `.claude/docs/strategy.md` | source of truth for the decisions on the **ML strand** (labeling, features, model, validation). Closed negative, but the traps still hold |
| `git log main..HEAD` | the commit messages explain the *why* of every choice and the bugs found |

Do not summarise that content: it is already written and up to date.

---

## Last session (2026-08-30): the temporal smoothing, put back

The user pointed out that a **temporal smoothing between maxima and minima** had once been defined
in the labeling and was no longer there. It was deliberate and it worked at **0.7**. It is back, in
the labeling, in training and on the page.

**What the label is now**, in one line: `labeling.swing_leg_target` slides continuously from −1 at a
local minimum to +1 at the next local maximum, and the position along the leg is **70% elapsed time
and 30% price covered** (`TIME_WEIGHT = 0.7`). Full description in
[`labeling-strategy.md`](labeling-strategy.md).

**Measured**, 15 symbols at 5m from 2018, `W = 144`, out of sample from 2024-01 with a 432-bar
embargo:

| | with smoothing (0.7) | previous label (centered rank) |
|---|---|---|
| IC against the forward half | **+0.0433** | +0.0405 |
| causal reference (`pos_canale`) | +0.0296 | +0.0297 |
| **excess over the reference** | **+0.0137** | +0.0108 |
| per-symbol median | +0.0500, 14/15 agreeing in sign | — |

The honest number improves; **the verdict does not change** — the swing model still does not clear
the commission, and the reason is in `modello-swing.md` §5.3.

**Three things not to break.**

1. **`labeling.TIME_WEIGHT = 0.7` has three consumers** and they must not drift apart:
   `swing_trainer.PESO_TEMPO`, `config.SWING_TARGET_TEMPO` (a copied literal, because
   `trading/config.py` imports nothing by design) and `tests/test_swing_target.py`. A drift here is
   silent: the page would draw one label and the model would be trained on another. It has already
   happened once, in this very session, which is why the constant now has a test.
2. **The embargo is three windows** (`EMBARGO_FINESTRE = 3`, 432 bars), not one. The leg label looks
   ahead as far as the *next extreme*, a variable distance longer than `W`; a single window was
   enough only for the centered rank.
3. **The yardstick is the forward-only half.** The label's past half is reproducible by a Stochastic
   (IC 0.70 against the full target), so an IC measured against the full label is 93% memory of the
   past. Every figure above is scored with `verso="avanti"`.

**The artifact now records its label** in the metadata
(`labeling: {method, window, peso_tempo, embargo_finestre, base_interval}`), so a drift is
detectable from disk instead of by reading the code.

## Documentation language

**Everything in this repository is written in English** — `CLAUDE.md`, `.claude/docs/`, every
folder `README.md`. The rule is at the top of `CLAUDE.md` and it is not a style preference: it had
been given before and the documents drifted back to Italian anyway. Answer in Italian in the chat if
the user writes in Italian; write the files in English. Code identifiers, comments and docstrings are
still Italian and are out of scope — changing them touches tests that assert on Italian names.

---

## Session (2026-08-30): simplification and documentation

No new measurement and no behaviour change: **−9,570 lines** of code and a `README.md` in every
folder.

**Deleted** (git is the archive, `git log --diff-filter=D --name-only` finds it): all of `backup/`
(`unused/` and `v2/`, ~7,100 lines no test ever ran), the two families closed with a negative result
along with their tests (`policy.py`, `dagger.py`, `policy_trainer.py`, `leg_trainer.py`, and the two
dispatch branches in `ml/signals.py` and `trading/strategies.py`), `trading/live_frames.py`,
`uv.lock` and `requirements.txt` (neither was read by anything), the two dated handoffs.
Every comment that pointed at that code was **repointed to the document holding the measurement**,
not deleted: the measurement still holds, the code does not.

**Documented**: fourteen new or rewritten `README.md`, one per folder, listing the files and their
public functions — derived by reading the syntax tree, not from memory. `models/README.md` still
described the keras era; `reports/README.md` did not mention the `cs_*` and `meta_*` tables; the
root `README.md` declared `policy_model` as the active model. `.claude/docs/INDEX.md` became
`README.md`.

**Local data, deleted with approval**: 4.7 GB, from 9.3 down to 4.6. Out went
`market_data/rl_stati.pkl` (3.5 GB of cache keyed on the artifact signature, `scripts/rl_lab.py`
rebuilds it), `swing_previsioni.pkl`, `tuner_logs/` (keras-tuner is no longer a dependency), `.venv`
and `.venv3.13`, the `models/` artifacts of the two closed families and the three `.keras` files of
the previous era, `.claude/RESUME.md` and the pytest/ruff caches.

**Four things that looked deletable and were not**, each found by verifying instead of trusting the
size:

| | why it stays |
|---|---|
| `.venv3.12` (1.9 GB) | it is the project interpreter in PyCharm (`Python 3.12 (CryptoFarm)`), not `.venv312`. Deleting it breaks the IDE silently |
| `analysis_cache/` (31 MB) | it is not only output: `scripts/multiplicity.py --cache` **reads** `analysis_cache/*/*_annuale.parquet` as input |
| `.claude/.headroom_*` | state of a live process (PID 69612), not leftovers |
| `.serena/` | project configuration of a tool in use |

`models/.gitignore` still named `policy_trainer` among the ways to regenerate the artifacts.

---

## Previous session (2026-08-29): the entry model, wired in with a positive result

All of it in [`modello-ingresso.md`](modello-ingresso.md). Four things before restarting:

1. **The head of the chain changed**: `entry_model_veloce`, then `entry_model`, then the earlier
   families. The fast one generates the trades, the slow one acts as a **gate** on the entry bar
   only.
2. **The question changed, not the model.** Not "how close are we to a local extreme" but "how much
   does buying here return". At equal selection the first identifies lows better (37.2% against
   23.0%) and returns **2.4 times less**. Precision and money are not the same thing, and that is the
   measurement that moved the target.
3. **These are the first numbers that pass the paired-exposure control**: +2.071% net per trade out
   of sample, 148 trades, 14/15 symbols profitable, 100th percentile over 200 draws. The comparison
   with passive holding does not count — out of sample the median passive does −34%, so staying out
   pays by itself.
4. **The lever is selectivity, not accuracy.** At 10% of bars flagged the net is below the
   commission, at 0.5% it is ten times above it. Hence threshold, gate and holding live in the
   artifact metadata and not in the widgets: moving them calls for a different strategy.
5. **It serves up to 30 minutes and above that it stays quiet** (`signals.entry_fuori_misura`, §8.2).
   The threshold is a return, not a quantile: on the same threshold the flagged bars go from 0.063%
   at 5m to 2.98% at 1h and 28.1% at 1d. Above the half hour it was serving a different strategy
   under the name of the measured one.
6. **Trading more is possible and the cost is known** (§8.1). At 1% of bars flagged it is 330 trades
   instead of 148 and the compounding does not get worse, but the net per trade halves and the
   profitable symbols drop to 13/15. `entry_lab --frequenza` redoes the table.

**Two things remain open and are written in §7 of that document**: the block control (rows overlap
each other and across symbols, and this has already flipped a verdict once) and the confluence grid
with and without the `modello` voter.

**Not to be mistaken for a fault:** at 5m the model flags one bar in fifteen hundred. Over the page's
initial window, 240 hours, zero trades is what to expect, and on BTCUSDT — a single trade in the
whole out-of-sample period — the maximum prediction over 2,880 bars stays below the threshold.

**New trap:** reusing a variable name inside a printing loop overwrote the overall average return
with the last symbol's, and the wrong number ended up in the metadata, i.e. in the service. There is
a test that runs `addestra` on two fake symbols and requires the saved average to lie **between** the
two per-symbol averages.

---

## Session (2026-08-28): the AI model, redone and closed negative

All of it in [`modello-swing.md`](modello-swing.md). The three things to know before restarting:

1. **`leg_model` is out of `MODEL_PRECEDENCE`, on purpose.** A reviewer in fresh context found that
   its two thresholds were tuned on the verification sample, that the random control sampled i.i.d.
   rows from a population overlapping for 7/8 of the horizon, and that the average net per entry is
   **negative at all six thresholds**. The artifact stays on disk; the reason is written next to the
   constant. Whoever puts it back must redo the yardstick first.
2. **`swing_model` exists, is trained, and is not wired in.** The statistical signal is there — IC
   +0.0433 out of sample against a causal reference of +0.0296, 14/15 symbols agreeing — but against
   a random control **at paired exposure** it wins on 1 symbol out of 15, i.e. chance. The apparent
   advantage over passive holding was entirely "staying out of the market".
3. **The signal is U-shaped, and that changes how it is used.** The +1 pole is not "sell": it is
   "strong trend in progress", and in crypto continuation pays. Selling the predicted highs — the
   natural reading of a target in [−1, 1] — sells the best bars. Whoever picks this up must start
   here, not from the directional rule.

**New trap to know:** `swing_target` looks `W` bars into the future, so it does not go among the
features unless delayed by **at least `W`+1**. At a one-bar delay the IC goes from 0.050 to
**0.673**, which is not a model but the leak. It is written in the docstring and there is a test.

**Three untried paths**, in order of cost: sizing the position with `|prediction|` instead of using
it as a switch; taking the decision to a daily scale; using the model as a **voter** inside
Confluence, where it does not have to beat passive holding on its own.

---

## Current state of the work: the trading strand

Three consecutive sessions. **The third (2026-08-26/27) is the most recent and partly corrects the
first two**; its measurements are in [`ricerca-quant-ml.md`](ricerca-quant-ml.md). The two sections
below remain because they describe how we got there, not because they are the last word.

Two choices from that session were taken **with the user** via `AskUserQuestion`, with an explicit
answer, and are not to be reopened without asking: pruning the menu with a "medium cut" (seven
entries out, the three borderline cases in — see `CLAUDE.md`, "Functions of `strategies.py` the menu
does not reach"), and adding cross-sectional rotation **as a view of the page**, not as a script
only. Only one thing remains offered and undecided: whether to drop the wide 15-asset universe
entirely from the rotation view, where today it serves as a control on what makes things worse.

### Session 1 — measuring the existing strategies (`d82b3db`, `8f4ccd8`)

3,129 configurations of the simulator's 12 strategies, on BTC 2017-2026 at 15m, plus controls on ETH
and on other intervals. Result: **at 15 minutes almost everything loses**, and trading frequency
explains almost all of it (strategies making thousands of trades a year pay more in commissions than
their margin per trade). Out of sample, even what looked solid in sample collapses.
Scripts kept: `scripts/strategy_sweep.py`, `scripts/sweep_report.py`, `scripts/strategy_focus.py`.
Tables: `reports/*_15m*.csv`.

### Session 2 — corrections, new dataset, new strategies, short (`61603cc`)

**The four corrections** the user asked for, all applied and measured:

| defect | correction | effect |
|---|---|---|
| menu entry `"Supetrend"` ≠ dispatch `"Supertrend"` | string fixed in `config.STRATEGIES` | the entry runs: +450% at 4h in the best configuration |
| `"ATR Bands"` had the dispatch branch but no menu entry | entry added | selectable: +678% at 4h |
| stop loss of `buy_sell_limits_close_simulation` commented out | restored | inert at the 99% default, active at operational values |
| `EMA200` was the EMA **of the open** over the short window, and Trend Zones compared it with `EMA20` (an average against itself) | column removed, the three functions read `EMA100` | Trend Zones 4h from **−21.9% to +309.3%**, from 202 to 10.6 trades/year |

The golden master was regenerated **once only** and the diff verified entry by entry: 17 entries, all
of `add_technical_indicator` and of the three functions that read `EMA200`, across the four
scenarios.

**The dataset changed**: no longer 2017-2026 but **2021-01-01 → today**, because the two cycles are
different markets (BTC 2017-2020: +2,803%, CAGR 132%, Sharpe 1.44 — 2021-2026: +166%, CAGR 19%,
Sharpe 0.59) and **the parameters do not carry from one to the other**: choosing on the old cycle and
measuring on the new one, four strategies out of five go into loss.

**Five new strategies** in `trading/strategies_ls.py`, on seven indicators never used before (ADX,
Donchian, Bollinger, Keltner, StochRSI, OBV/MFI, Ichimoku): `donchian_breakout`, `squeeze_breakout`,
`trend_pullback`, `ichimoku_trend`, `band_reversion_gated`.

**The short side is simulable**: `pnl.simulate_positions` takes position changes
`(time, price, +1|0|−1)` — the two-signal-list format cannot express a direct reversal — with
commission on both legs, daily carry cost (funding, 0.03%), leverage and liquidation at zero capital.

### The results, unsweetened

- **The corrected historical ones beat the new ones in sample** (Close ATR +575% at 1d against +120%
  for the best new one), but on grids 100 times larger: the honest column is the median.
- **Out of sample (chosen 2021-2023, returned 2024-2026) no strategy, of any family, beats passive
  holding.** The only new one with a positive sign is `band_reversion_gated` (+11.1%).
- **The real advantage is on risk, not on return**: `band_reversion_gated` does +84% with 22%
  drawdown against +166% with 76.5%; **at 2× leverage it becomes +196% with 41% DD**, i.e. it beats
  passive holding on both axes (in sample).
- **The short takes away instead of adding**: the median gets worse in all five strategies (donchian
  −22% → −57%; ichimoku +15% → −25%), it improves only in 2.6-23.6% of the pairs, and it only pays in
  2022. The only exception is mean reversion, where the short side has a 52.3% win rate.
- **The ablations** say every new filter improves the median and reduces the trades; the only
  irrelevant one is ADX as a minimum threshold in the channel breakout.

### What remains open

> **Update 2026-08-26 — the two points below are closed.** On the user's machine the store has all 15
> symbols up to 2026-08-19: SOL and BNB were measured and `donchian_breakout`/`squeeze_breakout`
> remeasured after the trailing-stop fix. The results are in `.claude/docs/ricerca-quant-ml.md` §2,
> and **two conclusions of `strategie-nuove.md` do not hold on five assets**:
> `band_reversion_gated` is negative on 4 assets out of 5, and out of sample 4h beats 1d. The
> commands below remain valid, with a correction needed in `strategy_lab`/`strategy_sweep` for
> macOS's `spawn` (see §8 of that document).

**SOL and BNB were not measured.** The user had asked for them explicitly. It is not a choice: in the
remote environment egress towards *every* exchange and aggregator answers 403 on CONNECT (Binance,
Bybit, Kraken, Coinbase, Kucoin, MEXC, Gate, CoinGecko, CryptoCompare, Messari, Yahoo, Kaggle,
HuggingFace), and no reachable public repository has recent intraday candles for those two assets.
The code already supports them. **Locally this is enough:**

```bash
python -m cryptofarm.data.klines --update --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT
for s in BTCUSDT ETHUSDT SOLUSDT BNBUSDT; do
  python -m scripts.strategy_lab --all --symbol $s --interval 1d --since 2021-01-01
  python -m scripts.lab_report --symbol $s --interval 1d
done
```

The conclusions on regimes and on the short side hold for **one asset and one cycle** until that
runs.

**Remeasure `donchian_breakout` and `squeeze_breakout`.** The trailing stop was fixed (see the traps
below and `.claude/docs/strategie-nuove.md` §8): their rows in §6 and the `reports/lab_*.csv` predate
the fix. They need the same candles as the point above, so the two jobs are done together. The other
three strategies are untouched.

### The page, rebuilt around the registry

The simulator used to show everything: fifteen parameters in the sidebar and a dozen traces, the same
for every strategy. Now `trading/panels.py` holds the map strategy → indicators → parameters →
traces, and the page reads it: you see only what the chosen strategy actually uses, and everything
only when none is selected. `trading_analysis` went from thirty keyword arguments to a dictionary,
the chain of `if strategia == ...` is gone, and `simulator.py` went from 672 to 402 lines.

The five new strategies are in the menu, **always and only long**: the short side is measured at a
loss. They go through the classic engine via an adapter from position changes to two lists, exact
without the short side and verified against `simulate_positions`. **Their numbers on the page are
more optimistic than `reports/lab_*.csv`**, because that engine does not charge funding and does not
know about leverage: the page says so next to the result.

Traps found by looking at the rendered figure, not at the code: aquamarine over the candles is
confused with a bullish body (the palette validator approves the pair, because it does not know one
of the two is a filled body), and the overview put two "short EMA" and two "upper band" entries in
the legend. Both pinned by a test.

### New code in the trading strand

| file | what |
|---|---|
| `trading/indicators_extra.py` | `ExtraParams` + `ExtraCache`: ADX, EMA, ATR, KAMA, Donchian, Bollinger, Keltner, StochRSI, MFI, OBV slope, Ichimoku, memoised per parameter. **Donchian is shifted by one bar** (`.shift(1)`): without it, the channel contains the bar that breaks it |
| `trading/strategies_ls.py` | the five new strategies, all with `allow_short`; they return position changes, not two lists |
| `trading/pnl.py` | `simulate_positions` alongside `simulate_trading_with_commisions`. `CARRY_DAILY_PERCENT = 0.03` |
| `scripts/strategy_lab.py` | grids for the new ones (592 configurations), `ProcessPoolExecutor` with candles inherited by fork, short-aware metrics (`n_long`/`n_short`, contribution per side) |
| `scripts/lab_report.py` | overview, paired short effect, ablations, transfer between datasets, out of sample with configurable windows, historical vs new ranking, leverage and costs |
| `tests/test_long_short.py` | 14 tests: long/short symmetry, costs on both legs and over time, leverage and wipe-out, events ignored after liquidation, **no look-ahead** (truncated series → identical events) and "long-only never goes short", parametrised over the five strategies |

`tests/test_simulator_golden.py` now also covers `simulate_positions` (a sequence with a direct
long→short reversal).

Tables produced: `reports/lab_*.csv` (overview, short effect, ablations, ranking, out of sample,
leverage and costs; suffixes `_1d`, `_4h`, `_4h_ciclo2017`, `_ETHUSD_4h`).

### A fault that had been there for a day: rotation with an empty store

`rotation.load_universe` built `pd.DataFrame({})` from an empty dictionary — which is born with an
**integer** RangeIndex — and then filtered it with `frame.index >= since`. Comparing int64 with a
string raises `TypeError`, so with an empty `market_data/` the rotation view **crashed** instead of
showing the warning that has sat next to it all along. That is the normal condition in production:
Render's free plan has no persistent disks.

The same fault hid the collection of `tests/test_simulator_page.py`, which exercises it at module
level. In that session I had attributed the failed collection to the Python version: **that was
wrong**. With the fix the suite runs whole, without `--ignore`.

### The confluence (2026-08-27, measured 2026-08-28)

The full design and **the verdict** are in [`strategia-confluenza.md`](strategia-confluenza.md):
measured on 15 assets over seven years, **it does not beat passive holding**. No look-ahead and
uncorrelated voters, but every parameter's gradient points at not trading. Only the map of what
exists remains here.

| file | what |
|---|---|
| `trading/mtf.py` | `align_to_lower`: reads the **closed** long bar, shifting it by one period. It is the only defence against look-ahead between intervals |
| `trading/voters.py` | `held_state` (events → held state) and `decayed_vote` (state → fading vote). It is the signal's memory, i.e. what makes the confluence able to trigger |
| `trading/confluence.py` | the strategy: six voters on four planes, weighted score, breadth by family, threshold that moves with the higher planes, three-condition exit. `stati_dei_votanti` isolates the expensive part |
| `trading/portfolio.py` | one capital across several assets, with missed opportunities and concentration |
| `scripts/confluence_lab.py` | the bench: three grids, basket, three references, `--selfcheck` that runs without the store |
| the "Confluence" entry in the simulator | two panels (decision and voters) and the explanation attached to every marker |

**Four things already known that must be kept in mind when reading the results:**

1. the expectation declared *beforehand* is the same order of return as passive holding with a much
   smaller drawdown, **not** a higher return. If the result were much better, the first hypothesis to
   check is look-ahead, not success;
2. the most likely risk is not that it loses: it is that it **trades too little** for anything to be
   said. The number of trades per year goes next to every result;
3. the **necessity per voter** is in the results table (`nec_*`, `necessarieta_max`). Above 0.60 the
   ensemble is that voter in disguise, and the metrics are talking about it;
4. the trial count for `multiplicity.py` is at the top of every CSV produced. `ampia` is 4,800 cells:
   looking at the best one and reporting its Sharpe without discounting it is not a measurement.

---

## The ML strand, briefly

Closed negative and **not to be reopened without reading `strategy.md` §10-13**. In one line:
entering on the confirmation of a low and exiting on the confirmation of a high captures **zero on
average**, on all 15 symbols, at every threshold, *before* costs. The confirmation is paid twice and
the median leg is worth 1.76-2.05 of them. No choice of model, features or hyperparameters changes
it.

What **not** to redo: retuning the decision threshold (§12.6), adding DAgger iterations (it works but
cures a different problem), trying a different architecture (in-sample is already below cost, it is
not overfitting), trusting an attribution with a "perfect" exit (use `confirmed_reversal_rows` and the
control with random entries).

Still open, in this order: `capture` beyond 0.40 (never measured up to 0.85); the formulation of
§13.4 (at the confirmation bar, predict whether *this* leg will exceed `2 × threshold + cost`); then
microstructure data (`aggTrades`) and the maker fill model (Phase 0.3).

---

## Things that are not in the documents and are needed right away

- **Use `.venv312/bin/python`.** The pre-existing `.venv` is Python 3.9 without `scikit-learn`; the
  project requires ≥3.12. Normal install: `pip install -e ".[app,data,dev]"`.
- **Network: it depends on where the session runs.** On the user's machine `raw.githubusercontent.com`
  and `api.github.com` answer, and the repositories' READMEs download (done 2026-08-26). The
  paragraph below applies to the **remote environment**, not to the local one, and should be read
  that way.
- **Network blocked in a remote session.** No exchange and no aggregator is reachable (403 on the
  proxy's CONNECT); even GitHub's *search API* is denied because the session is bound to its own
  repositories. PyPI, the contents of the configured repositories and release assets remain
  reachable. Do not waste time retrying new hosts: it has already been done exhaustively.
- **`market_data/` on the user's machine has all 15 symbols** at 5 minutes, up to 2026-08-19 (284 MB,
  gitignored): from 614,732 to 945,675 candles per symbol. It is with this store that the
  `ricerca-quant-ml.md` measurements were made. The paragraph below describes the **remote
  environment**, where there were two.
- **`market_data/` in the remote environment holds only two files** (55 MB, gitignored):
  `BTCUSD-5m.parquet` (1,540,397 candles, 2012-01-01 → today, source Bitstamp) and
  `ETHUSD-5m.parquet` (342,929 candles, 2016-03 → 2019-12, source Bitfinex). ETH **does not cover the
  recent cycle**: that is why the control on a second asset was done on the 2017-2019 cycle and not
  on 2021-2026. On the user's machine the Binance store is much wider (15 symbols, ~11.8 million 5m
  candles).
- **`models/*.joblib` and `*.json` are not tracked** (`models/.gitignore`, extended in 2026-08).
  `meta_model.*` is the previous strategy's model: do not delete it, `load_signal_model()` still
  loads it. `MODEL_PRECEDENCE` and `active_model_name()` are the only source of truth.
- **Tests: 1,024 passed, 3 skipped, in 35 files.** `ruff check src tests scripts` and `black --check`
  clean. CI runs both jobs on every PR.
- The two long measurements (`strategy_sweep`, `strategy_lab`) take tens of minutes: start them in
  the background and wait with a polling loop, never with a chained `sleep`.
- **`analysis_cache/` is gitignored and is the input of `scripts/tune_defaults.py`.** Without it,
  `trading/tuned_defaults.py` cannot be regenerated: it would take the grids on four intervals for
  five symbols again, roughly two hours of computation. The `reports/*.csv` files are tracked.

## Rules of engagement set by the user

- Before structural changes: a written plan, then confirmation. (Suspended when the user explicitly
  says "go ahead with the implementation".)
- **Every number must be measured on the project's data, never estimated nor taken from the prompts.**
  The user has repeated: *"if a measurement contradicts a thesis in the prompt, report it — I prefer a
  correct strategy to one that confirms what I asked for."* It happened again in that session (the
  short side, which the user expected would double the opportunities, is measured at a loss) and it
  must be said unsweetened.
- Cascading controls on every result, and suspicion towards results that are too good.
- Incremental commits with a summary after every block.
- The final deliverables must also be published as a readable artifact, besides being written in the
  repo.
- **The documentation is written in English**, always. See the top of `CLAUDE.md`.

## Traps already met

**On the simulator and the backtests**

- **Look-ahead in the channels**: a rolling maximum that includes the current bar makes the breakout
  impossible to miss. `indicators_extra` shifts Donchian; the test `test_no_look_ahead` verifies that
  truncating the series does not change the events already emitted.
- **Look-ahead *inside* the bar**: a trailing stop built from the high and the ATR of the very bar it
  is tested on assumes that within that bar the favourable extreme comes first. **`test_no_look_ahead`
  does not see it**, because it truncates the series *between* bars and the bar triggering the event
  stays identical: it takes the control that perturbs the high of the exit bar alone
  (`test_trailing_stop_ignores_the_high_of_its_own_bar`). The stop must be computed on data closed at
  `i-1`.
- **`ta` fills, it does not leave NaN.** `IchimokuIndicator(visual=True)` builds span B with
  `min_periods=0` and fills the shift's first `slow` rows with the mean of the **whole** series: span
  B is never NaN, no guard catches it, and that fill is look-ahead. In `ichimoku_trend` the protection
  is `start = slow + span + 2`, and it must not be lowered.
- **The golden master accepts any difference** if regenerated. Regenerate only after checking the
  difference by hand, and verify the diff contains only the expected lines.
- **The golden scenarios are not interchangeable**: `close_ema_crossover_simulation` demands three
  crossovers in sequence, `close_bullish_ema_simulation` only the sideways one. Removing a scenario
  uncovers strategies.
- **`indicators._atr_ema` replicates `ta` 0.11 line by line.** If touched, it must be reverified
  against `ta`: a silent divergence moves every signal.
- **Functions decorated with `@st.cache_data` are called with `.__wrapped__`** outside Streamlit.
- **The maximum over a grid is not a result**: it must always be read with the median and the share of
  profitable configurations next to it.

**On the ML pipeline**

- **Retrospective pivots**: using `extreme_bar` instead of `confirm_bar` in a feature is look-ahead
  (`strategy.md` §7.1: median delay 1-8 bars, p99 up to 101, at its worst precisely on the wide
  moves).
- **Overlapping labels**: `t_exit` is the next confirmed pivot, a variable and potentially long
  horizon. The embargo must be sized on the high percentile. For the leg label it is **three
  windows**, not one (see the top of this document).
- **Non-causal rolling**: `labeling.py` uses `[::-1].rolling(...)[::-1]` on purpose. Correct there,
  disastrous anywhere else.
- **A label constant with several consumers drifts silently.** `TIME_WEIGHT` is copied into
  `trading/config.py` because that module imports nothing by design; there is a test that pins the
  two together. Without it, the page draws one label and the model trains on another.
- Two target defects have already **flipped the sign** of the results once (commit `7ebb2e0`).

## Suggested skills

- **`tdd`** — for any extension of the position engine or of the strategies: the costliest bugs of
  both sessions were found by tests.
- **`diagnosing-bugs`** — when a measurement does not add up.
- **`dataviz`** — before adding charts to the simulator.
- **`artifact-design`** — the user expects a visual report at the close of every block of
  measurements.
- **`ponytail:ponytail`** — the user launches it themselves at the start of a session; in the third
  one it was active at `full` level throughout.

`codebase-design` is not needed (the module structure is decided and documented). `research` is not
needed for the state of the art, which is already collected in `ricerca-quant-ml.md` §1 — but **on
the user's machine the network works**, so the old note "no external source reachable" applies only
remotely.
