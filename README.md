<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:000C3D,100:0A4FD6&height=180&section=header&text=CryptoFarm&fontSize=48&fontColor=ffffff&desc=95-module%20ML%20trading%20platform%20%7C%20ingestion%20%E2%86%92%20training%20%E2%86%92%20signals%20%E2%86%92%20Streamlit%20app&descSize=15&descAlignY=72" />

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&pause=1000&color=7CFFA0&center=true&vCenter=true&width=700&lines=95+modules%3A+data+acquisition+to+live+order+placement;Parquet+store+of+5m+candles%2C+15+pairs%2C+2017+%E2%86%92+today;GRU+%2F+LSTM+%2F+CNN+%2F+gradient+boosting+on+the+same+pipeline;355+tests+in+35+suites%2C+enforced+by+GitHub+Actions;Deployed+on+Render+%E2%80%94+multi-stage+Docker" /></a>

[![CI](https://github.com/MarcoMonini/CryptoFarm/actions/workflows/ci.yml/badge.svg)](https://github.com/MarcoMonini/CryptoFarm/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/Live_demo-cryptofarm.onrender.com-7CFFA0?style=for-the-badge&logo=render&logoColor=white)](https://cryptofarm.onrender.com)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

<p>
  <img src="https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white" />
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Apache_Parquet-50ABF1?style=flat-square&logo=apacheparquet&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white" />
  <img src="https://img.shields.io/badge/Ruff-D7FF64?style=flat-square&logo=ruff&logoColor=black" />
  <img src="https://img.shields.io/badge/Black-000000?style=flat-square" />
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white" />
  <img src="https://img.shields.io/badge/Render-46E3B7?style=flat-square&logo=render&logoColor=black" />
</p>

</div>

## 🎯 What this is

**An end-to-end platform, and a measurement log.** It acquires Binance candles, trains signal models
on them, serves those models as trading rules, and backtests everything against indicator strategies
over nine years and fifteen assets. There is also a headless bot that places real orders.

**The result in one line.** Almost everything that has been tried does *not* beat passive holding,
and it is written down where it was measured. The one thing that clears the matched-exposure control
is the entry model: **+2.071% net per trade out of sample, 14 symbols out of 15 profitable, 100th
percentile against random entries** — and its edge is not the prediction, it is the **selectivity**.
It flags one bar in two hundred.

The decision log lives in [`.claude/docs/`](.claude/docs/) (English, 5,500 lines). Every number in
this README comes from a bench in [`scripts/`](scripts/) that reproduces it.

<div align="center">

| 15 USDT pairs | 5m base interval | 95 modules | 355 tests / 35 suites | 3,129 backtests |
|:---:|:---:|:---:|:---:|:---:|
| **2017 → today** | **Parquet, ~4 GB** | **ruff · black · pytest in CI** | **1,024 collected cases** | **9.65 years, 338,114 bars** |

</div>

## 🧱 The platform — 95 modules

40 package modules, 18 bench modules, 36 files under `tests/` and one Docker entrypoint helper:
**95 Python files** end to end, from the first HTTP request to the order sent to Binance. Every
folder carries its own `README.md` listing its files and their functions.

```
src/cryptofarm/
├── data/      the candle store: bulk dumps, Parquet, incremental updates
├── ml/        features → labels → dataset → model → evaluation → serving
└── trading/   indicators, strategies, P&L, the Streamlit page, the live bot
scripts/       17 measurement benches: they produce the documents' numbers
tests/         35 suites, no network, no candle store required
.claude/docs/  the decisions, and the measurements that justify them
```

<details>
<summary><b>Module map</b></summary>

| Layer | Modules | What it does |
|---|---|---|
| **Acquisition** | `data.klines` · `data.positioning` | Monthly bulk dumps from `data.binance.vision` (static CDN ZIPs, no credentials, 2017+), REST only for today's tail. Futures positioning: long/short ratio, open interest, funding, basis. |
| **Features** | `ml.features` · `ml.bar_features` | Scale-free columns only — an ATR of ~300 on BTC and ~0.0002 on DOGE would otherwise teach the model the asset's identity. `bar_features` serves the current models: 41 columns, 15 on the base bar plus 13 per long scale, in **one** definition shared by training and inference. |
| **Labels** | `ml.labeling` · `ml.directional_change` | Four families, not interchangeable: triple barrier, swing leg, centred rank, forward return. See the table below. |
| **Assembly** | `ml.dataset` · `ml.validation` | Sequence windows, purged splits, an embargo sized on the label's look-ahead (`EMBARGO_FINESTRE = 3`, because the leg label reaches a *variable* horizon). |
| **Models** | `ml.models` · `ml.trainer` · `ml.meta_trainer` | GRU, Bi-LSTM, dilated CNN (TensorFlow/Keras) and `HistGradientBoostingClassifier` (scikit-learn) behind one `build_model(kind=...)`. Meta-labelling on top. |
| **Research heads** | `ml.swing_trainer` · `ml.entry_trainer` · `ml.rl` · `ml.rl_trainer` | Three question families: *where along the leg are we*, *what does buying here return*, *what position should I hold with the fee inside the reward*. |
| **Serving** | `ml.signals` · `ml.evaluate` · `ml.execution` | Artifact → trading rule. Thresholds, gates and holding periods come from the artifact's **metadata**, never from a widget. |
| **Trading** | `trading.indicators(+_extra)` · `strategies` · `strategies_ls` · `pnl` · `mtf` | Indicators with a numpy ATR/EMA core, one-sided and two-sided strategies, commission- and carry-aware P&L, multi-timeframe alignment that reads the **closed** long bar. |
| **Portfolio** | `trading.confluence` · `voters` · `portfolio` · `rotation` | Six voters on four disjoint time planes; one pot of capital across assets; cross-sectional rotation that picks *which* asset rather than *when*. |
| **App** | `trading.simulator` · `panels` · `config` · `tuned_defaults` | Streamlit + Plotly, two views, a data-driven panel registry, and per-interval defaults that are **generated, not hand-tuned**. |
| **Live** | `trading.live_bot` | Headless bot on `python-binance`. Real orders, credentials from the environment. |
| **Benches** | `scripts/*` (17) | `analysis`, `strategy_sweep`, `entry_lab`, `swing_lab`, `rl_lab`, `confluence_lab`, `cross_section`, `tune_defaults`, … — each one regenerates a document's tables. |

</details>

## 🗄️ The data layer — a Parquet store built for latency, not bandwidth

**15 USDT pairs, one 5-minute archive per symbol, from 2017-08 to today, ~4 GB of Parquet.** The
selection spans liquidity profiles on purpose: store of value (BTC, LTC), high-throughput L1s (SOL,
AVAX, NEAR, TRX), majors (ETH, BNB, XRP, ADA, DOT, ATOM), DeFi (LINK, UNI) and one meme (DOGE).

**One interval, not four.** 15m/30m/1h are *derived* by aggregation, verified against the official
dumps at zero difference on OHLC — one source of truth instead of four archives drifting apart, and
a quarter of the network traffic.

**Bulk dumps for the history, REST for the tail.** The cost of this ingestion is entirely
per-request latency (~2.7 s measured, identical on both routes), not bandwidth:

| route | payload per request | rate limit | full history |
|---|---|---|---|
| REST `klines` | 1,000 candles | yes | ~18,700 sequential calls ≈ **14 hours** |
| Monthly dump (CDN) | ~8,900 candles | none | ~1,350 files, 32 workers ≈ **under 10 minutes** |

So the store backfills from static S3 ZIPs in parallel and uses the REST API **incrementally**, for
the candles of today that the dumps do not carry yet. `update_store` is idempotent: it reads the
manifest, downloads only the missing months, and appends.

```bash
python -m cryptofarm.data.klines --update                     # build / refresh, ~10 min
python -m cryptofarm.data.klines --manifest                   # what is on disk
python -m cryptofarm.data.positioning --update                # futures positioning, ~400 MB
python -m scripts.import_candles --source /path/to/clone      # where the CDN is unreachable
```

## 🤖 Models — four architectures, one pipeline

`ml/models.py` builds all four behind a single `build_model(kind=...)`, so features, labels, purged
splits and metrics stay identical and the comparison means something.

| `--model` | Architecture | Framework | Status |
|---|---|---|---|
| `gbdt` *(default)* | `HistGradientBoostingClassifier` | scikit-learn | **The default, by measurement** |
| `gru` | `GRU(64) → Dropout(0.2) → Dense(32)` | TensorFlow / Keras | behind the `dl` extra |
| `lstm` | `Bidirectional(LSTM(64)) → Dropout(0.2) → Dense(32)` | TensorFlow / Keras | behind the `dl` extra |
| `cnn` | stacked causal `Conv1D(48, k=3)` with growing dilation | TensorFlow / Keras | behind the `dl` extra |

**Why the boosted trees lead.** On the same dataset and the same labels the `HistGradientBoosting`
fit took **3.9 seconds** against roughly **25 minutes** for a three-layer LSTM with 747,267
parameters — for an equivalent result. The sequential models stay in the repository, one flag away,
because that comparison is worth being able to redo; a Keras Tuner `Hyperband` search over the LSTM's
depth, width, dropout and learning rate is preserved in `trainer.py`'s history for the same reason.
The lesson kept from it: on this signal-to-noise ratio, capacity was never the binding constraint.

Keras is imported **inside the functions**, not at module top level, so the 1 GB TensorFlow install
is optional and the default path runs without it.

### How the data is labelled

| label | question | trained by |
|---|---|---|
| `triple_barrier_labels` | does price move 1.5 ATR up before 1.0 ATR down? | `trainer.py`, `meta_trainer.py` |
| `swing_leg_target` | where along the leg between two local extremes are we? | `swing_trainer.py`, the chart |
| `swing_target` | where does this bar rank among its neighbours? | the yardstick only |
| `rendimento_futuro` | what does buying here return over H bars? | `entry_trainer.py` |

The swing label oscillates in [−1, +1] between local lows and highs, with **temporal smoothing**
(`TIME_WEIGHT = 0.7`) deciding how much of the position along the leg is told by elapsed bars rather
than by price. At 0.7 a price that stalls mid-leg keeps advancing towards the extreme that is
coming — the part that can be anticipated. At 0 the label follows the price and the model learns an
oscillator. Full treatment in
[`labeling-strategy.md`](.claude/docs/labeling-strategy.md).

## 📈 Signal generation — selectivity is the product

`MODEL_PRECEDENCE` is the single source of truth: `active_model_name()` decides both which artifact
loads and which strategy runs, so the two cannot diverge. At the head sit **two entry artifacts that
work as a pair — the fast one trades, the slow one gates it.**

**The lever is selectivity, not accuracy** (H = 150, out of sample):

| bars signalled | average return of the signalled |
|---|---|
| 10% | +0.047% — *below the commission* |
| 2% | +0.90% |
| **0.5%** | **+2.07%** |

**The control, which here is mandatory.** Median passive holding is −34% over the same window, so
"it beats passive" proves nothing: a strategy in the market 17% of the time clears that on exposure
alone. The comparison is against **random entries at the same trade count and the same holding
period**, 200–400 draws, same anti-overlap filter:

| configuration | trades | avg net | profitable | chance | percentile | symbols |
|---|---|---|---|---|---|---|
| `entry_model_veloce` alone | 223 | +1.360% | 63.2% | −0.173% | 100th | 12/15 |
| + gate at the 80th | 156 | +2.019% | 65.4% | −0.161% | 100th | 13/15 |
| **+ gate at the 90th** | **148** | **+2.071%** | **65.5%** | −0.165% | 100th | **14/15** |
| + gate at the 98th | 100 | +2.464% | 68.0% | −0.172% | 100th | 13/15 |

The 90th is chosen on **agreement across symbols**, not on the maximum — the curve is monotone, and
taking its top is the mistake this project already measured elsewhere (§ *starting values*, below).

**It is served up to 30 minutes and stays silent above that.** The threshold is a return, not a
quantile, and the model predicts the return of the next twenty *five-minute* bars. On the same
threshold the marked bars go 0.063% at 5m → 2.98% at 1h → 28.1% at 1d, against the 0.5% it was
measured for, so `signals.entry_fuori_misura` gates the scale and says why. A consequence worth
knowing before calling it broken: at 5m it marks **one bar in sixteen hundred**, so zero trades over
a 240-hour window is the expected behaviour.

## 📊 What else was measured

| Family | Verdict |
|---|---|
| **RL policy** (`rl_trainer`, fee inside the reward) | Beats passive holding **11/15** out of sample and **halves the maximum drawdown** (−48.8% against −76.0%, at 37% average exposure). The *when* is only weakly above chance (mean rank 0.602, Wilcoxon p = 0.169). |
| **Swing model** (`swing_trainer`) | The statistical signal exists — IC **+0.0433** out of sample against a causal reference of +0.0296 — and it does **not** beat chance at matched exposure. The measured shape is U-shaped: both poles precede above-average returns, so the sign does not carry direction, and `swing_exposure` wires in `\|prediction\|` as a switch. |
| **Indicator strategies** (3,129 configurations, 9.65 years, BTC 15m) | 14.9% close profitable, 45.2% lose more than 90% of capital, **five (0.2%) beat passive holding** — and none survives out-of-sample verification. |
| **Confluence** (six voters, four planes, 15 assets, 7 years) | No look-ahead, uncorrelated voters, and at 15 minutes only **6.4%** of configurations beat passive. The gradient of every parameter points at not trading. |
| **Rotation** vs the right benchmark | Against BTC it wins in 95.6% of configurations; against the **equal-weight universe**, which carries the same survivorship bias, in 44.4%. The second number is the honest one. |

### ❌ Closed with a negative result, code deleted

Kept here because re-reading it costs less than re-measuring it. The three-action policy
(`policy.py`, `dagger.py`, `policy_trainer.py`) and the leg-model trainer (`leg_trainer.py`) are
gone: putting their name back in `MODEL_PRECEDENCE` does not bring them back, the dispatch branch is
deleted, and a test enforces that. `git log --diff-filter=D --name-only` is the archive.

- **Precision and money are different questions.** At equal selectivity the leg label identifies real
  lows far better (37.2% against 23.0%) and returns **2.4× less**. The target moved to the forward
  return and nothing else.
- **`sign(prediction)` on the swing model sells exactly the best bars** — the natural reading of a
  target in [−1, 1], measured at a loss at every threshold and every cadence.
- **The stop that was supposed to cut the crashes made it worse**, monotonically: −201% model-only
  against −229% at a 3% fixed stop and −563% on a 5% trailing. Entries do not land in front of
  crashes more often than chance does; the fee was the cause.
- **More features, more data, more capacity: none of them moved precision.** 30.0% → 31.3% with 16
  extra columns, and the *net* got worse; 4.4M rows instead of 366k gave 30.6%; more iterations gave
  29.0%.
- **Chasing the in-sample maximum transfers worse than picking at random.** On the rotation the
  correlation between in-sample and out-of-sample return over the first ten configurations is
  **−0.69**.

## 🖥️ The analysis app — Streamlit + Plotly

[**cryptofarm.onrender.com**](https://cryptofarm.onrender.com) · `streamlit run src/cryptofarm/trading/simulator.py`

Two views that ask two different questions, not two strategies:

- **Single asset** — loads a symbol from the exchange, runs a strategy from an 11-entry menu, prices
  every trade with commissions. It picks *when* to be in.
- **Cross-asset rotation** — reads the universe from the local store, ranks it by relative strength,
  holds the top names. It picks *which*. It uses no network, so in production it says so instead of
  attempting fifteen downloads.

Three things that make it more than a chart:

- **`panels.py` is a registry, not code.** Which indicators a strategy uses, which parameters each
  needs and how they are drawn are *data*; the page lays out widgets and traces from it. Adding a
  strategy is a row there. Three colours only — blue/orange/aquamarine, the single triple that clears
  every colour-vision validator pair on a dark surface; green and red stay reserved for state.
- **`tuned_defaults.py` is generated.** For each of the four measured intervals every parameter gets
  a starting value chosen one coordinate at a time, by **percentile rank within its own symbol**, and
  adopted only if it moves the median rank by ≥ 0.06 *and* picks the same value on 2021-2023 alone.
  The widget key includes the interval, or Streamlit silently keeps the previous timeframe's numbers.
- **Zero trades is a question, not a result.** `Confluenza.perche_non_entra()` reports which of the
  four `and` conditions never came true, with the numbers — usually the history is too short for the
  regime plane, not the strategy being cautious.

Per-row reads go through numpy arrays extracted before the loop, never `df["Col"].iloc[i]`: that is
where the speed comes from — **4,295 ms → 125 ms** for the whole simulator, and `simulate_candles`
40× faster since `indicators._atr_ema` replicates `ta` 0.11 in numpy line by line.

## ⚙️ Engineering

- **355 test functions across 35 suites, 1,024 collected cases** (parametrisation does the rest), all
  green in CI. No network, no candle store, no model artifact required: synthetic data and
  monkeypatching throughout.
- **A golden master pins behaviour, not coverage.** `tests/test_simulator_golden.py` compares 21
  functions across four synthetic market scenarios against a committed JSON. It must pass before a
  change to `trading/` and pass again afterwards **without being regenerated** — regenerating accepts
  any difference, including a regression. The scenarios are not interchangeable: removing one
  uncovers strategies.
- **The assembly is tested too.** `tests/test_simulator_page.py` drives the real page with
  `streamlit.testing.v1.AppTest`. That is the level at which the failure that took the simulator out
  of production slipped through: every function had tests, all passing, while an unconditional
  `load_signal_model()` inside `__main__` stopped the page from opening at all.
- **GitHub Actions on every push and PR**, two jobs. `quality`: `ruff check`, `black --check`,
  `pytest -q` on Python 3.12. `docker`: builds the targets, runs the suite *inside* the image, and
  checks three things no source file shows — that the package resolves its data directories to
  `/app/...`, that a build **without `--target`** does not carry TensorFlow, and that the container
  really binds `$PORT` (started at `PORT=10000`, queried on `/_stcore/health`).
- **Multi-stage Docker, four targets.** `runtime` (page, trainer, store), `dev` (+ pytest/ruff/black,
  the CI image), `dl` (+ TensorFlow, ~1 GB, only for the sequential models) and **`web`**, which goes
  to production and must stay the **last stage in the file** — Render builds without `--target` and
  has no field to choose one. A new stage goes above it, never below, and CI builds without
  `--target` precisely to notice. `tini` as PID 1, non-root user, two named volumes.
- **Deployed on Render** from [`render.yaml`](render.yaml), free plan, region `frankfurt` — Binance
  blocks US IPs on `api.binance.com`, which is where the page gets its candles, so the region is not
  a detail. `MALLOC_ARENA_MAX=2` because glibc allocates an arena per thread and Streamlit is
  multi-threaded: on a 512 MB instance heap fragmentation alone costs tens of MB. The four
  `@st.cache_data` in `trading/` carry `ttl`/`max_entries` for the same reason — cardinality is
  decided by whoever moves the sliders.
- **The model is optional for the page.** Artifacts are gitignored, so a fresh clone and the
  production image have none; `available_strategies` removes the AI entry from the menu instead of
  crashing.

```bash
mkdir -p models market_data                      # the bind mounts must exist first
docker compose up simulator                      # http://localhost:8501
docker compose --profile data  run --rm klines
docker compose --profile train run --rm trainer
docker compose --profile ci    run --rm tests
```

## 🚀 Quickstart

Python ≥ 3.12. The reference environment is **`.venv312`**.

```bash
pip install -e ".[app,data,dev]"

streamlit run src/cryptofarm/trading/simulator.py          # the app
python -m cryptofarm.data.klines --update                  # the store, prerequisite for training

python -m cryptofarm.ml.trainer                            # gbdt (default)
python -m cryptofarm.ml.trainer --model gru                # needs the `dl` extra

python -m cryptofarm.ml.entry_trainer --selfcheck          # runs without the store
python -m cryptofarm.ml.entry_trainer                      # the slow one, H=150, ~12 min
python -m cryptofarm.ml.entry_trainer --h 20 --quantile 0.995 --nome entry_model_veloce
python -m scripts.entry_lab                                # reproduces the gate table above

pytest -q && ruff check src scripts tests && black --check src scripts tests
```

<details>
<summary><b>Installation extras</b></summary>

| extra | holds | needed for |
|---|---|---|
| (core) | numpy, pandas, scipy, ta, requests, python-binance, scikit-learn | features, labels, `gbdt`, live bot |
| `app` | streamlit, plotly | `trading/simulator.py` and the modules it decorates with `st.cache_data` |
| `data` | pyarrow (141 MB) | the store's Parquet engine |
| `dl` | tensorflow (~1 GB) | only `--model gru\|cnn\|lstm` |
| `dev` | pytest, ruff, black, pre-commit | |

A leaner image for the page alone is not obtained by dropping pyarrow: `streamlit` depends on
`pyarrow>=7.0`, so those 141 MB are there anyway.

</details>

The remaining commands — the other trainers, the strategy sweeps, the 17 measurement benches — are
in [`CLAUDE.md`](CLAUDE.md) and in the READMEs of [`scripts/`](scripts/) and
[`src/cryptofarm/ml/`](src/cryptofarm/ml/).

## 🔭 Open points

1. **The confluence grid with and without the `modello` voter**, same assets, same window. It is the
   only way to say whether the model *adds* to the strategy or merely reduces its exposure. Same for
   the RL policy; neither has been done.
2. **The block control.** The random control samples overlapping rows across symbols; on
   `swing_model` a weekly block bootstrap already moved a verdict once. Until that is redone, the
   entry model's numbers are strong but not yet "significant".
3. **Below 5 minutes nothing is measured**, and below the hour **no measurement in this project has
   ever found anything that beats passive holding** — the 15m defaults are the best among those
   tried, not good.
4. **`live_bot.py` starts its `while True` at import**, with no `main()` and no signal handling,
   which is why it is deliberately *not* a compose service: a container restarting on its own would
   put it back to placing orders unsupervised. That refactor comes first.
5. **A known defect.** `buy_sell_limits_simulation` reads a `MACD` column still commented out in
   `add_technical_indicator`, so it raises `KeyError` on call. No menu entry reaches it; making it
   usable means restoring the indicator **and** adding the entry.

## 📝 Conventions

**Documentation is written in English** — `CLAUDE.md`, `.claude/docs/` and every folder `README.md`.
The rule is at the top of [`CLAUDE.md`](CLAUDE.md). Identifiers are English where they are domain
terms (`simulate_positions`, `swing_target`) and Italian where they name a decision taken here
(`perche_non_entra`, `scala_fuori_misura`, `votanti_predefiniti`); renaming those touches tests that
assert on the Italian names and is a separate job.

Credentials and bot parameters come from environment variables only — see `.env.example`; nothing in
the repository loads `.env` by itself. `API_KEY`/`API_SECRET` are read by `live_bot.py` alone: the
simulator and the trainers use Binance's public endpoints and want no credentials.

---

<div align="center">

**Marco Monini** · [LinkedIn](https://www.linkedin.com/in/marco-monini/) · [marco.monini98@gmail.com](mailto:marco.monini98@gmail.com)

*Research code. Nothing here is investment advice, and the project's own conclusion is that almost
everything tried does not beat holding the asset.*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0A4FD6,100:000C3D&height=100&section=footer" width="100%" />

</div>
