# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation language — standing rule

**Every document in this repository is written in English.** This covers `CLAUDE.md`, everything
under `.claude/docs/`, every `README.md` in every folder, and any new document added later.

This rule is not a style preference and it is not up for renegotiation. It has been given more
than once and the documentation drifted back to Italian anyway, so it is written here, at the top,
where it is read first:

- **do not translate a document back into Italian**, in whole or in part, for any reason;
- **write new documents in English from the first draft**, not in Italian to be translated later;
- when you edit an existing document, the edited lines stay English;
- if a request arrives in Italian, answer in Italian in the chat, but **write the files in
  English**. The language of the conversation and the language of the repository are separate.

Code identifiers, comments and docstrings are currently Italian and are *out of scope* for this
rule — changing them is a separate, larger job that touches tests asserting on Italian names. If
that is wanted, it has to be asked for explicitly.

## Working documentation

Project decisions and the state of the work live in **`.claude/docs/`**:

- `.claude/docs/labeling-strategy.md` — **the labeling strategy, in full (2026-08-30).** Labels in
  [-1, +1] oscillating between local lows and highs, the pivot windows, and the **temporal
  smoothing** (`TIME_WEIGHT = 0.7`) that makes the label lead the price instead of following it.
  Also: the embargo the variable look-ahead demands, and why the training target is not the
  measuring stick. Read this before touching `labeling.py` or `swing_trainer.py`.
- `.claude/docs/strategy.md` — source of truth for decisions on labeling, features, model and
  validation, with the measurements that justify them. Update it in place when something is decided.
- `.claude/docs/HANDOFF.md` — current state of the work and environment traps for whoever picks it up.
- `.claude/docs/backtest-strategie.md` — the indicator strategies measured over nine years: 3,129
  configurations, parameter sensitivity, out-of-sample behaviour, defects found by measuring.
- `.claude/docs/strategia-confluenza.md` — the multi-timeframe multi-signal strategy: four planes
  with disjoint questions, six voters chosen by family, signal memory, threshold set by the higher
  planes. **Measured (2026-08-28) on 15 assets and seven years: it does not beat passive holding.**
  No look-ahead, uncorrelated voters, but the gradient of every parameter points at not trading.
  The conclusions and what to do with them are at the bottom of that document.
- `.claude/docs/strategie-nuove.md` — the sequel: the four corrections applied, the 2021-2026 cycle
  as a dataset, five new strategies and the engine that can also go short.
- `.claude/docs/politica-rl.md` — **the reinforcement policy, wired in (2026-08-28).** The three
  measurements that rule out the stop and point at the commission as the cause, the reward with the
  cost inside it, and the results: it beats passive holding 11/15 out of sample and **halves the
  maximum drawdown**, but the *when* is only weakly above chance.
- `.claude/docs/modello-swing.md` — **the swing model, measured and wired in.** The audit that
  removed `leg_model` from the chain, the labeling, and the measurements for which the signal
  exists (IC +0.0385 out of sample, 14/15 symbols agreeing) but does not beat chance at matched
  exposure. §5.4 covers **what was wired in and what deliberately was not**.
- `.claude/docs/modello-ingresso.md` — **the model leading the chain today, wired in (2026-08-29).**
  It changes the question: not "how close are we to an extreme" but "what does buying here return".
  The measurements that moved the target (at equal selectivity the leg label identifies lows better
  and returns 2.4× less), selectivity as the only lever, and the **first numbers in this project
  that pass the matched-exposure control**: +2.071% net per trade out of sample, 14/15 symbols
  profitable, 100th percentile. The fast one trades, the slow one gates it.
- `.claude/docs/README.md` — suggested reading order.

Before changing the ML pipeline, read `strategy.md`: it contains measurements that explicitly rule
out several roads that look reasonable at first sight.

## Environment

Use **`.venv312/bin/python`**: it is the only complete environment, and the project requires Python
>= 3.12. A second one, `.venv3.12`, remains locally — it is the interpreter registered in PyCharm
("Python 3.12 (CryptoFarm)") and for that reason it was not deleted with the others on 2026-08-30 —
but it is frozen at July 2025. Whoever works from the IDE should update it or repoint it to
`.venv312`.

Installation is split into extras: `pip install -e ".[app,data,dev]"` is the normal case. The core
(`pip install -e .`) is enough for features, labels, `gbdt` models and the live bot; `[app]` adds
Streamlit and Plotly (only `trading/simulator.py` and the modules it decorates with
`st.cache_data`); `[data]` adds pyarrow, i.e. the parquet engine that `data/klines.py` and
`scripts/analysis.py` want; `[dl]` adds TensorFlow, about 1 GB, and is only needed for
`--model gru|cnn|lstm`.

`MODELS_DIR` and `MARKET_DATA_DIR` in `paths.py` move with `CRYPTOFARM_MODELS_DIR` and
`CRYPTOFARM_MARKET_DATA_DIR`. Without those two variables they stay relative to the repo root.

## Project overview

CryptoFarm trains a signal model on Binance market data and backtests trading strategies against it.
There are two things that matter — **`trading/simulator.py`** (research) and **`ml/trainer.py`**
(training) — plus their dependencies, plus one live bot. Whatever was not reachable from there was
**deleted** (2026-08-30): git is the archive, and a folder of dead code that no test exercises costs
more than it is worth. Every folder has a `README.md` listing its files and their functions; this
document keeps only what holds for the whole project.

```
src/cryptofarm/
├── data/klines.py        local candle store, built on Binance bulk dumps
├── ml/                   training pipeline (below)
└── trading/
    ├── market_data.py    ad-hoc download from Binance for the Streamlit page
    ├── indicators.py     indicators + the numpy ATR/EMA core
    ├── indicators_extra.py  ADX, Donchian, Bollinger/Keltner, StochRSI, OBV/MFI, Ichimoku
    ├── panels.py         the registry: which strategy uses which indicators and which parameters
    ├── strategies.py     from candles with indicators to (buy_signals, sell_signals)
    ├── strategies_ls.py  two-sided strategies: from candles to position changes (+1/0/-1)
    ├── pnl.py            from signals to trades: `simulate_trading_with_commisions` (long only)
    │                     and `simulate_positions` (long/short, with leverage and carry cost)
    ├── mtf.py            alignment across intervals: reads the **closed** long bar, never the current one
    ├── voters.py         from position changes to a per-bar vote, with memory and decay
    ├── confluence.py     the confluence strategy: six voters on four planes, dynamic threshold
    ├── portfolio.py      one pot of capital across several assets: it opens on the first that speaks
    ├── rotation.py       cross-sectional rotation: it picks *which* asset, not *when*
    ├── tuned_defaults.py generated: measured starting values, per interval
    ├── config.py         starting values for the page's widgets
    ├── simulator.py      the Streamlit page: two views, `trading_analysis` + `rotation_page`
    └── live_bot.py       headless bot that places real orders
scripts/analysis.py       command-line measurements producing the numbers in strategy.md
```

### Entry points

```bash
# Simulator / backtest (the main research tool)
streamlit run src/cryptofarm/trading/simulator.py

# Training. It downloads its own data; the parameters are constants at the top of the file
.venv312/bin/python -m cryptofarm.ml.trainer               # default: gbdt
.venv312/bin/python -m cryptofarm.ml.trainer --model gru   # sequential model
.venv312/bin/python -m cryptofarm.ml.meta_trainer          # meta-labeling

# Candle store (prerequisite for training)
.venv312/bin/python -m cryptofarm.data.klines --update

# Swing model: where along the leg between two local extremes we are
# (see .claude/docs/labeling-strategy.md and .claude/docs/modello-swing.md)
.venv312/bin/python -m cryptofarm.data.positioning --update     # futures positioning, 400 MB
.venv312/bin/python -m cryptofarm.ml.swing_trainer --selfcheck  # runs without the store
.venv312/bin/python -m cryptofarm.ml.swing_trainer              # ~12 minutes, 15 symbols from 2018
.venv312/bin/python -m cryptofarm.ml.swing_trainer --w 72 --peso-tempo 0.5   # another window/weight
.venv312/bin/python -m scripts.swing_lab                        # deciles, P&L, random control

# Entry model: what buying here returns (see .claude/docs/modello-ingresso.md)
.venv312/bin/python -m cryptofarm.ml.entry_trainer --selfcheck  # runs without the store
.venv312/bin/python -m cryptofarm.ml.entry_trainer              # ~12 minutes, the slow one (H=150)
.venv312/bin/python -m cryptofarm.ml.entry_trainer --h 20 --quantile 0.995 --nome entry_model_veloce
.venv312/bin/python -m scripts.entry_lab                        # what the slow model's gate is worth
.venv312/bin/python -m scripts.entry_lab --frequenza            # what trading more often costs

# Reinforcement policy: picks the position with the cost inside the reward (.claude/docs/politica-rl.md)
.venv312/bin/python -m cryptofarm.ml.rl                         # self-check of the algorithm alone
.venv312/bin/python -m cryptofarm.ml.rl_trainer --selfcheck     # runs without the store
.venv312/bin/python -m cryptofarm.ml.rl_trainer                 # ~5 minutes, 15 symbols from 2019
.venv312/bin/python -m scripts.rl_lab                           # shuffled-blocks control

# The measurements in strategy.md
.venv312/bin/python -m scripts.analysis

# Cross-sectional rotation and meta filter (see .claude/docs/ricerca-quant-ml.md)
.venv312/bin/python -m scripts.cross_section --universe majors --interval 1d --grid
.venv312/bin/python -m scripts.meta_gate --strategy donchian_breakout --interval 4h --oos 2024-01-01

# Measured starting values per interval (regenerates trading/tuned_defaults.py)
.venv312/bin/python -m scripts.tune_defaults --all-intervals --save

# Backtest of the indicator strategies over the whole history (see .claude/docs/backtest-strategie.md)
.venv312/bin/python -m scripts.strategy_sweep --all --interval 15m   # parameter grids
.venv312/bin/python -m scripts.sweep_report --interval 15m           # tables in reports/
.venv312/bin/python -m scripts.strategy_focus --top 3                # commissions and intervals

# Confluence strategy (see .claude/docs/strategia-confluenza.md)
.venv312/bin/python -m scripts.confluence_lab --selfcheck             # runs without the store, fake data
.venv312/bin/python -m scripts.confluence_lab --grid coordinate --symbol BTCUSDT --interval 15m
.venv312/bin/python -m scripts.confluence_lab --grid ampia --interval 15m --since 2021-01-01
.venv312/bin/python -m scripts.confluence_lab --grid veloce --paniere majors

# Two-sided strategies, long and short (see .claude/docs/strategie-nuove.md)
.venv312/bin/python -m scripts.strategy_lab --all --interval 1d --since 2021-01-01
.venv312/bin/python -m scripts.lab_report --symbol BTCUSD --interval 1d

# Candle store from an alternative source, where data.binance.vision is unreachable
.venv312/bin/python -m scripts.import_candles --source /path/to/clone

# Live bot — places real orders, requires the environment variables (see .env.example)
.venv312/bin/python src/cryptofarm/trading/live_bot.py
```

Tests: `.venv312/bin/python -m pytest` (35 files: 1,024 passed, 3 skipped). Lint/format:
`ruff check src scripts tests` and `black src scripts tests` (config in `pyproject.toml`).

## The simulator

`trading/simulator.py` used to be a single 2,028-line file and was split into the modules above. The
dependencies form a DAG: `market_data`, `indicators`, `pnl` and `config` depend on nothing,
`strategies` depends on `indicators`, `simulator` on all of them. **There is no re-export facade**:
whoever needs a strategy imports it from the module that contains it.

- All OHLCV DataFrames are indexed on `Open time` (`DatetimeIndex`) with columns
  `Open, High, Low, Close, Volume`.
- The functions in `strategies.py` return `(buy_signals, sell_signals)`, lists of
  `(timestamp, price)`, which `trading_analysis` passes to `pnl.simulate_trading_with_commisions` or
  `simulate_trading_with_commisions_multiple_buy`. Those in `strategies_ls.py` return position
  changes `(timestamp, price, +1|0|-1)` for `pnl.simulate_positions` instead: that is the format
  needed to express a direct reversal and short selling.
- Per-row reads go through numpy arrays extracted before the loop, not `df["Col"].iloc[i]`. That is
  where most of the speed comes from (the whole simulator: 4,295 ms → 125 ms). Keep the style.
- `indicators._atr_ema` replicates the `ta` 0.11 formulas in numpy line by line (ATR seeded on the
  mean of the first `window` true ranges, then Wilder; EMA as `ewm(span, adjust=False)`).
  **If it changes, it must be re-verified against `ta`**: it is what makes `simulate_candles` 40
  times faster, and a silent divergence here moves every signal.

### The two views

The page has a switch at the top of the sidebar (`config.ROTATION_MODES`), and the two entries are
not two strategies but **two different questions**:

- **Single asset** — `trading_analysis`: loads a symbol from the exchange and runs a strategy from
  the menu on it. It picks *when* to be in.
- **Cross-asset rotation** — `rotation_page` on `trading/rotation.py`: loads the universe **from the
  local store**, ranks it by relative strength and keeps the top names. It picks *which*.

Three consequences to know before touching them:

- **rotation does not use the network.** It reads `market_data/`, so in production (no persistent
  disk) it has no data and says so, instead of attempting fifteen downloads. A test verifies this;
- **the initial values are central, not optimal.** The correlation between in-sample and
  out-of-sample return on the first ten configurations is **-0.69**: chasing the in-sample maximum
  transfers worse than picking a configuration at random. Anyone changing them to "the ones that
  return most on the chart" is making exactly the measured mistake;
- **the benchmark to beat is the equal-weight universe, not BTC.** It carries the same survivorship
  bias as the rotation, so the comparison isolates what rotation adds. Against BTC the rotation wins
  in 95.6% of configurations; against the universe, in 44.4%.

### The starting values depend on the interval

`trading/tuned_defaults.py` is **generated** by `scripts/tune_defaults.py` and is not edited by
hand. It holds, for each of the four measured intervals (15m, 1h, 4h, 1d), the starting value of
every parameter of every strategy in the menu.

**How they are chosen, and why it is not the grid maximum.** The maximum is the luckiest cell: on
this data picking the maximum transfers worse than the median, and on the rotation the correlation
between in-sample and out-of-sample return is −0.69. Here one coordinate is chosen at a time: every
configuration gets its **percentile rank within its own symbol** (the only way to add up assets
whose passive holdings range from +134% to +4,346%), and for every value of every parameter the
median of those ranks across five assets is taken. The best value is adopted **only if** it passes
two checks: it moves the median rank by at least 0.06, and it picks the same value when looking at
2021-2023 alone. Whatever fails them keeps the hand-written default.

**The `panels.ANCORA_MISURATA` map** says which measurement covers which interval: the menu offers
nine, the grids cover four. It is data and not a computation, because "the closest one" is already
a decision (30m sits between 15m and 1h).

Three things to know before touching it:

- **the widget key includes the interval** (`par_{name}_{interval}`). Streamlit preserves the state
  of a widget with the same key: without it, changing timeframe leaves the fields on the previous
  interval's numbers and the measured default never appears. The defect is invisible when reading
  the code and `AppTest` does not see it, since it rebuilds state on every run — which is why the
  test asserts on the **key**, not on the value;
- **windows grow as bars get shorter**, which is the mechanical reading of the result: the same rule
  wants a 20-bar channel on a daily and 150 on an hourly to cover the same stretch of calendar. A
  test pins the direction of that inequality;
- **two parameters have no coherent reading across intervals** and should be treated with suspicion:
  `ATR Bands / atr_multiplier` (3.0 at 15m, 1.6 at 1h, 1.2 at 4h, 3.0 at 1d) and
  `Donchian Breakout / adx_min`. They are adopted because they pass the two checks on each interval
  taken alone, but the overall picture does not support them. `tune_defaults` prints the
  cross-interval agreement table specifically to make them visible.

**Below the hour, no measurement in this project has ever found anything that beats passive
holding.** The 15m defaults are the best *among those tried*, not good.

### The `panels.py` registry

The page no longer decides on its own what to show. `trading/panels.py` holds, as data, which
indicators each strategy uses, which parameters each of them needs and how they are drawn;
`simulator.py` reads it and lays out widgets and traces. Adding a strategy means adding a row there
and the entry in `config.STRATEGIES` — a test verifies the two lists match.

Three things to know before touching it:

- **The map is verified by hand.** A static scan of the columns read is not enough:
  `close_bullish_ema_simulation` takes the moving averages with `(df[c].to_numpy() for c in (...))`,
  a variable slice that syntax-tree analysis does not see.
- **Dependencies matter more than names.** `Upper_Band`/`Lower_Band` are `KAMA ± multiplier × ATR`
  and `KAMA` uses `ema_window`: a band strategy depends on "EMA Short" even if it draws no moving
  average at all.
- **There are three colours**, blue/orange/aquamarine: the only ones that pass every validator pair
  on a dark surface. The fourth slot against orange drops to 4.8 ΔE for deuteranopia. Aquamarine is
  not used over the candles, where it blends with the bullish body. Green and red stay reserved for
  state. Three tests hold these rules in place.

### The confluence strategy

`trading/confluence.py` is the only menu entry that is not an indicator strategy: it reads **four
time planes derived from the chosen interval** (`FATTORI` — ×1 trigger, ×4 confirmation, ×16
structure, ×96 regime, i.e. 15m/1h/4h/1d starting from 15m) and has eight different strategies vote.
Things to know before touching it:

- **adding a voter is `confluence.registra(Votante(...))`, almost and that is all.** Families,
  weights, necessity, sidebar panels, strategy parameters and the lab grid adapt on their own, and
  that is the point. The only list left to keep aligned by hand is the traces of the *Voters* panel
  in `panels.INDICATORI`, and there is a test that notices: it counts the traces with `·` against
  `len(VOTANTI)`;
- **the `modello` voter is in the default only if an artifact exists.** `votanti_predefiniti()`
  removes it when none of the four (`entry_model_veloce`, `entry_model`, `rl_model`, `swing_model`)
  is on disk, which is the production condition: weights are normalised over the voters present, so
  an eighth one that is always silent would effectively raise the threshold for the other seven. It
  stays in the registry, so `selezione("modello")` still reaches it. It is also the only **long
  only** voter: it votes +1 or 0, never −1. With the entry model it votes +1 while one of its trades
  is open and the two thresholds have no effect — the selectivity lives in the artifact's metadata;
- **voter parameters resolve in three layers**: the function default (`config.CONF_*`), the value
  measured in `tuned_defaults` for the interval of the **plane** the voter runs on — not the page's
  — and the caller's override. The second layer is the one that is easy to get wrong: on a 15m base
  a structure voter runs at 4h and wants the 4h values;
- **moving them costs.** The freeze kept the free parameters at nine; with the 31 knobs open it goes
  past forty, and `scripts/multiplicity.py` says what happens there. Move them to understand,
  measure with the voters frozen;
- **the threshold is continuous, not stepped.** The long planes enter as distance from the mean
  normalised by the ATR of the same plane, not as `np.sign`: with the sign the threshold jumped by
  0.15 at a time and one score-driven exit in four was decided by that jump;
- **the hysteresis has a floor and a ceiling** (`barre_minime`, `pazienza`), and they apply **only
  to the score-driven exit**. The stop and the gate do not: they are risk rules, not opinions;
- **the expensive part does not depend on the grid.** Frozen voters have a state that depends only
  on (symbol, interval): `stati_dei_votanti` computes it once and `scripts/confluence_lab.py` reuses
  it across every cell. Measured over 11,520 bars: 351 ms per cell against 104 ms;
- **there is exactly one place where it can cheat**, `_stato_del_votante`, and the defence is
  `mtf.align_to_lower`, which shifts the long plane's state by a whole period before reading it. The
  test that protects it cuts **inside** a long bar that has already started and compares the states:
  a cut aligned to the boundaries passes even with the defect reintroduced, and that is how it was
  written the first time;
- **zero trades is not a result, it is a question.** The entry conditions are four in `and` and
  `Confluenza.perche_non_entra()` says which one never came true, with the numbers. It is needed
  because the most common case is not the strategy being cautious but the history: at 15m the regime
  plane is daily and its mean asks for fifty bars, i.e. **1,200 hours**, against the 240 of the
  page's starting value;
- **the x1/x4/x16/x96 scale holds around fifteen minutes.** At 1m the "regime" lasts an hour and a
  half, at 1d it asks for 96-day bars. The written rule is that the regime plane must last between
  half a day and a week (`scala_fuori_misura`), which leaves 15m, 30m and 1h;
- **the explanation travels with the signal.** Confluence signals are `(when, price, text)` instead
  of `(when, price)`, and the chart shows the text on hover. That is why `pnl` unpacks with `[:2]`:
  any strategy may append elements after the two the engine uses. The text **distinguishes entries
  from exits**: four exits out of five are the trailing stop, and showing the voters on top of them
  reads as "sold while five voters were saying buy", which is true and completely misleading;
- **the four panels are not interchangeable.** `regime` and `struttura` are worth ±1 and the score
  sits within ±0.5: on the same axis the first flattens the second, and you see a line stuck at 1
  while buying and selling happens. Hence the separate *Higher planes* panel, and the trailing stop
  drawn on the candles — without it, 80% of the sells are unexplainable from the chart.

`trading/portfolio.py` answers a different question and must not be confused with `rotation.py`:
rotation picks *which* asset to hold and is always in; the shared-capital basket stays out until
someone speaks and then puts all the capital on the first that gives the signal. It always reports
the **missed opportunities** while the capital was committed and the **concentration**, i.e. the
share of the most traded asset: above 0.9 the basket is fiction.

### Functions in `strategies.py` the menu does not reach

`buy_sell_limits_simulation` reads `MACD`, which is still commented out in
`add_technical_indicator`, and therefore raises `KeyError` as soon as it is called: it is the only
one excluded because it is broken.

The other seven **left the menu by measurement** (2026-08-26, `.claude/docs/ricerca-quant-ml.md`
§2): Close Buy/Sell Limits, Close ATR, Close Bullish EMA, Green Candles, ATR Live Trade, Trend
Pullback, Band Reversion. They stay in the module and in the golden master — the measurement is
redone with `scripts/strategy_sweep` — but they are not selectable.

**`close_rsi_buy_sell_limits_simulation` came back instead** ("Close RSI Reverse"). The reason it
was excluded — "at a total loss in all 25 configurations tried" — holds at 15 minutes and not at
daily scale: at 1d it makes 24-27 trades a year, a positive median on all five symbols and 72-92%
of configurations profitable; at 4h it makes 160 a year and loses 45.8% on BTC. It is the clearest
case of the already known rule that trading frequency explains almost everything: **a strategy
excluded on one interval is not excluded on all of them**.

### The golden master

`tests/test_simulator_golden.py` pins the behaviour of 21 functions across four synthetic market
scenarios, comparing it against `tests/data/simulator_golden.json`. It covers the **behaviour of the
functions**: **before touching it, this must pass; afterwards, it must pass again without
regenerating it**.

The **assembly** is covered by `tests/test_simulator_page.py` instead, which runs the page with
`streamlit.testing.v1.AppTest`. That is the level the failure that took the simulator out of
production went through: every function had its tests and they all passed, while
`load_signal_model()` called unconditionally inside `__main__` prevented the page from opening. It
also covers degradation without the candle store, which is the condition the public service runs in.

Regenerating (`SIMULATOR_GOLDEN_REGEN=1 pytest tests/test_simulator_golden.py`) **accepts any
behavioural difference**. Only do it after verifying by hand that the difference is intended, and
check that the JSON diff contains only the expected lines.

The scenarios are not interchangeable: `close_ema_crossover_simulation` demands three EMA crossings
in sequence and only fires on a real reversal (`regimi`, `sbandate`), `close_bullish_ema_simulation`
only in a range. Removing one scenario uncovers strategies.

## The ML pipeline

`ml/trainer.py` contains no logic of its own: it assembles the pieces and holds the configuration.
Features live in `features.py`, labels in `labeling.py` and `directional_change.py`, the matrix in
`dataset.py`, models in `models.py`, metrics in `evaluate.py`, validation in `validation.py`,
simulated execution in `execution.py`. `meta.py` + `meta_trainer.py` do the meta-labeling.
`ml/README.md` lists the public functions of each file one by one.

**Two families were closed with a negative result and their code is no longer here**: the
three-action policy (`policy.py`, `dagger.py`, `policy_trainer.py` — `strategy.md` §12-13) and the
leg model trainer (`leg_trainer.py` — `modello-swing.md` §1). Putting their name back in
`MODEL_PRECEDENCE` is not enough to make them run, because the dispatch branch is gone, and a test
verifies that. The measurement that closed them is in the documents, and that is where it must be
re-read before redoing them.

Note the distinction, because it is easy to get wrong: what was closed is the *leg trainer*, not the
*leg label*. `labeling.swing_leg_target` is alive and is what `swing_model` is trained on today —
see `.claude/docs/labeling-strategy.md`.

The default model is **`gbdt`** (`HistGradientBoostingClassifier`), no longer an LSTM; `models.py`
still keeps `gru`/`cnn`/`lstm` behind `--model`. The prerequisite for training is the candle store
(`data/klines.py`), not an on-the-fly download.

### How the data is labeled

Three label families live in `ml/labeling.py`, and they are not interchangeable. The full treatment
is in **`.claude/docs/labeling-strategy.md`**; the short version:

| label | question | used by |
|---|---|---|
| `triple_barrier_labels` | does price move 1.5 ATR up before 1.0 ATR down? | `trainer.py`, `meta_trainer.py` |
| `swing_leg_target` | where along the leg between two local extremes are we? | **`swing_trainer.py`**, the chart |
| `swing_target` | where does this bar rank among its neighbours? | the yardstick only, as `verso="avanti"` |
| `rendimento_futuro` | what does buying here return over H bars? | `entry_trainer.py` |

The swing label is **[-1, +1] oscillating between local lows and highs**, with two knobs that must
be understood together:

- **the window** (`W`, 144 five-minute bars in training) selects which timescale of swing counts as
  an extreme. Different windows give different labels, and that is intended;
- **the temporal smoothing** (`peso_tempo` / `labeling.TIME_WEIGHT` = **0.7**) decides how much of
  the position along the leg is told by the **elapsed bars** rather than by the price. At 0.7 a
  price that stalls mid-leg keeps advancing towards the extreme that is coming, which is the part
  that can be anticipated. At 0 the label follows the price and the model learns an oscillator.

**The three consumers of that constant must never drift apart.** Training reads
`swing_trainer.PESO_TEMPO = TIME_WEIGHT`, the chart reads `config.SWING_TARGET_TEMPO` (a copy,
because `config.py` deliberately imports nothing), and the tests read `TIME_WEIGHT` directly. Until
2026-08-30 they had drifted: the trainer learned the centered rank, which has no time weighting at
all, while the page drew the time-weighted leg. Two different labels under one name, and the sidebar
knob changed nothing the model had seen. `tests/test_swing_target.py` pins all of it.

Because the leg label looks ahead to the **next extreme** — a variable horizon, longer than `W` —
the train/test split uses an embargo of `EMBARGO_FINESTRE = 3` windows, not one. One window was
enough for the centered rank and is not enough here.

### Which model the simulator uses

`ml/trainer.MODEL_PRECEDENCE` is
`("entry_model_veloce", "entry_model", "rl_model", "swing_model", "meta_model", "signal_model")` and
`active_model_name()` is the single source of truth: `load_signal_model` loads that model and
`ai_model_simulation` picks the strategy based on that name, so the two cannot diverge. To go back
to the previous model it is enough to move the most recent artifact elsewhere.

`meta_parameters()` reads barriers, CUSUM threshold and execution parameters **from the artifact's
metadata**, not from constants: they must be exactly the ones the model was trained with.

**The model leading today is `entry_model_veloce`, and the two entry artifacts work as a pair.** It
predicts the return of the next H bars — not the shape of the chart — and its advantage is
**selectivity**: at 10% of bars signalled the net is below the commission, at 0.5% it is ten times
above it. It follows that threshold, gate and holding period live in the **artifact's metadata** and
not in the widgets: changing them does not tune a knob, it asks for a different strategy. The fast
one (20-bar hold) generates the trades, the slow one (`entry_model`, 150-bar hold) acts as a gate on
the entry bar alone: +2.071% net per trade out of sample against +1.360% without it, 14 symbols out
of 15 profitable, 100th percentile against random entries at matched exposure
(`modello-ingresso.md`). Without the slow artifact the fast one trades alone, and it is back to
+1.360%.

**It is served up to 30 minutes and stays silent above that.** The threshold is a return, not a
quantile, and the model predicts the return of the next twenty *five-minute* bars: on the same
threshold the marked bars go from 0.063% at 5m to 2.98% at 1h and 28.1% at 1d, against the 0.5% it
is measured for. `signals.entry_fuori_misura` is the scale gate, twin of
`confluence.scala_fuori_misura`. On the page the two artifacts are chosen with a switch
(`Fast (trades)` / `Slow (gates)`) and the choice reaches the strategy as
`ai_model_simulation(..., famiglia=...)`: they are two strategies, not two tunings. The *Entry
model* panel puts prediction and target side by side in the same unit — the two curves do not look
alike (rank IC +0.007) and that is not a fault: above the threshold the average realised return is
+1.99% against −0.004% across all bars.
A consequence to know before saying "it does not work": at 5m it marks **one bar in fifteen
hundred**, so over a 240-hour window zero trades is the expected behaviour.

The earlier families stay in the chain below it. `swing_model` predicts where along the leg between
two local extremes the bar sits. The measured shape of that signal is U-shaped: *both* poles precede
above-average returns, so the sign **does not tell the direction**. `ml/signals.swing_exposure`
wires in the only reading the measurement supports — `|prediction|` as an exposure switch, with
hysteresis. Wiring `sign(prediction)`, which is the natural reading of a target in `[-1, 1]`, sells
exactly the best bars: it is measured at a loss at every threshold and every cadence
(`modello-swing.md` §5.1). That model **does not beat passive holding**.

## Data/model artifacts

`models/` contains the artifacts (`.joblib` + `.json` metadata) and **tracks none of them**:
`models/.gitignore` covers `*.keras`, `*.joblib` and `*.json`, and keeps only the `README.md`. A
clone of the repository therefore has no models, and that is the condition the public service runs
in. Regenerate with the trainers, do not edit by hand.

## Docker and CI

A single `Dockerfile` with four targets: **`runtime`** (simulator, trainer, candle store,
`scripts.analysis`), **`dev`** (`runtime` + pytest/ruff/black, the image the CI runs on), **`dl`**
(`runtime` + TensorFlow, for the sequential models) and **`web`**, which is what goes to production
and is identical to `runtime`.

**`web` is the last stage in the file, and must stay there**: a build without `--target` takes the
last stage, and Render has no field to choose one. Moving it means shipping the TensorFlow image to
production. The CI also builds without `--target` precisely to notice. A new stage goes above `web`,
never below.

A leaner image for the page alone is not achievable by dropping pyarrow: `streamlit` depends on
`pyarrow>=7.0`, so the 141 MB of the parquet engine are there anyway.

```bash
mkdir -p models market_data                     # first time only: the bind mounts must exist
docker compose up simulator                     # http://localhost:8501
docker compose --profile data  run --rm klines
docker compose --profile train run --rm trainer
docker compose --profile ci    run --rm tests
```

Inside the image the package sits in `site-packages`, not editable: the root `paths.py` would infer
from the file's location would point inside the virtualenv, so the image sets
`CRYPTOFARM_MODELS_DIR=/app/models` and `CRYPTOFARM_MARKET_DATA_DIR=/app/market_data`, which is
where `compose.yaml` mounts the host's `./models` and `./market_data`. Whoever touches `paths.py`
must keep the override working, otherwise models trained in a container end up in a throwaway layer.

The public deployment is in `render.yaml` (free plan, `frankfurt` region). Three constraints that
are not visible from the code: the service must bind to **`$PORT`** on `0.0.0.0` (the image's
command uses `${PORT:-8501}`); Binance blocks US IPs on `api.binance.com`, which is where the
simulator gets its candles, so the region is not a detail; the free plan has no persistent disks,
and with `models/*.joblib` gitignored the classic strategies are what run online.

The model is **optional** for the page: the artifacts are gitignored, so a clone of the repository
and the production image have none. `simulator.available_strategies` removes the
`config.AI_STRATEGY` entry from the menu when `active_model_name()` finds nothing, and the load at
startup is conditioned on the same check. Whoever touches that point should keep in mind that
`load_signal_model()` used to be unconditional and brought down the whole page, not just that
strategy.

The four `@st.cache_data` in `trading/` have `ttl`/`max_entries` for an operational reason: the
parameters come from the widgets, so cardinality is decided by whoever moves the sliders, and
without a cap a 512 MB instance ends in OOM while in use. Do not remove them.

`live_bot.py` is **not** a compose service, on purpose: it starts the `while True` loop at import,
with no `main()` and no signal handling, so a container that restarts on its own would put it back
to placing orders unsupervised. That refactor comes first.

The CI (`.github/workflows/ci.yml`) runs on every pull request and on pushes to `main`, in two jobs.
The first installs `.[app,data,dev]` on Python 3.12 and passes `ruff check`, `black --check` and
`pytest` over `src`, `tests` and `scripts`. The second builds the images and verifies four things
that are not visible from the source: that the package imports and resolves the data directories to
`/app/...`, that the tests pass inside the image, that the build **without `--target`** does not
carry TensorFlow (i.e. that `web` is still the last stage), and that the container really binds to
`$PORT` — it starts it with `PORT=10000` and queries `/_stcore/health`.

No image is published to a registry: Render builds the Dockerfile itself on every push to `main`.

## Configuration

Binance credentials and the bot's parameters come from environment variables — see `.env.example`.
Nothing in the repo loads `.env` on its own (there is no `python-dotenv`): export them in the shell
or in the IDE's run configuration.

- `API_KEY`, `API_SECRET` — only `trading/live_bot.py`.
- `live_bot.py` also reads `ASSET`, `CURRENCY`, `CANDLES_TIME`, `SMA_WINDOW`, `ATR_WINDOW`,
  `ATR_MULTIPLIER`, `RSI_WINDOW`, `RSI_BUY_LIMIT`, `RSI_SELL_LIMIT`, `NUM_CONDITIONS`.
- `MARKET_DATA_CSV` — path of the historical CSV in the Streamlit page (`trading/config.py`).
- The simulator and the trainers use Binance's public endpoints and need no credentials.

`.streamlit/config.toml` sets the dark theme.

### Claude Code plugins

`.claude/settings.json` is tracked and declares three marketplaces with the plugins enabled for the
project: `ponytail`, `agent-skills` (collections of general-purpose skills) and three plugins from
`anthropics/financial-services` — `financial-analysis`, `equity-research`, `market-researcher` —
chosen because the work here is quantitative financial analysis.

Each marketplace is **pinned to a commit** (`ref`, a 40-character SHA): it is the only way to pin
plugin versions, because `enabledPlugins` only accepts a boolean and the version is declared by the
marketplace manifest. At the time of pinning: ponytail 4.9.0, agent-skills 0.6.7,
financial-analysis 0.1.1, equity-research 0.1.2, market-researcher 0.1.1. To update them the `ref`
is moved to a more recent commit, deliberately — it does not happen on its own.

Plugin skills become available from the session after installation, not the one in which the file is
edited.

## What was deleted, and how to find it again

There is no archive left in the repository. `backup/unused/` (live dashboard, two-account bot, grid
search, results viewer, analysis dashboard), `backup/v2/` (the multi-timeframe simulator of the
previous rewrite), `trading/live_frames.py` and the two model families closed with a negative result
were deleted on 2026-08-30. **git is the archive**: `git log --diff-filter=D --name-only` finds
them, and `git checkout <commit>^ -- <path>` puts them back with their history intact.
