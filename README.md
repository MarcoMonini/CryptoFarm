# CryptoFarm

Trains a signal model on Binance market data and backtests trading strategies against it, with one
headless bot that can trade the result live.

Two files matter — `trading/simulator.py` (research) and `ml/trainer.py` (training) — plus their
dependencies. Everything not reachable from those lives in `backup/unused/`.

## Project structure

```
src/cryptofarm/
├── paths.py                  where models/, market_data/ and friends live
├── data/
│   └── klines.py             local candle store, built from Binance bulk dumps
├── ml/                       training pipeline
│   ├── features.py           per-bar features (returns, RSI, STOCH, ATR, TSI, volume)
│   ├── dataset.py            design matrix, lags, CUSUM events, time splits
│   ├── labeling.py           ATR-based triple barrier
│   ├── directional_change.py confirmed pivots and soft labels
│   ├── models.py             model construction (gbdt default; gru/cnn/lstm behind --model)
│   ├── evaluate.py           per-class and per-score-quantile metrics
│   ├── validation.py         purged k-fold, CPCV, embargo, PBO, Deflated Sharpe
│   ├── execution.py          fill simulation (maker entry, taker exit)
│   ├── meta.py               cost-net meta-labelling targets
│   ├── policy.py             position state and the three-action policy
│   ├── dagger.py             DAgger rollouts and state coverage
│   ├── signals.py            from model to signals, for the simulator
│   ├── trainer.py            * trains the signal classifier
│   ├── meta_trainer.py       trains the meta-labelling secondary
│   └── policy_trainer.py     trains the three-action policy
└── trading/
    ├── market_data.py        one-off Binance downloads for the page
    ├── indicators.py         indicators + the numpy ATR/EMA core, PSAR
    ├── strategies.py         from indicator table to (buy_signals, sell_signals)
    ├── pnl.py                from signals to trades, fees included
    ├── config.py             starting values for the sidebar widgets
    ├── simulator.py          * Streamlit page: trading_analysis + layout
    └── live_bot.py           headless bot that places real orders

scripts/analysis.py           reproducible measurements behind .claude/docs/strategy.md
tests/                        188 tests; test_simulator_golden.py pins the simulator
models/                       .joblib artifacts + .json metadata (untracked)
backup/unused/                modules removed from src/ because nothing imported them
backup/v2/                    multi-timeframe simulator, read-only reference
```

Dependencies inside `trading/` form a DAG: `market_data`, `indicators`, `pnl` and `config` depend on
nothing, `strategies` depends on `indicators`, `simulator` on all of them. There is no re-export
facade — each strategy is imported from the module that holds it.

## Running

Python >= 3.12. Use `.venv312/bin/python` — the older `.venv` is 3.9 and lacks `scikit-learn`.

```bash
pip install -e ".[app,data,dev]"   # see "Dependency extras" below

# Backtest / strategy simulator
streamlit run src/cryptofarm/trading/simulator.py

# Kline store first, then train
.venv312/bin/python -m cryptofarm.data.klines --update
.venv312/bin/python -m cryptofarm.ml.trainer                # gbdt by default
.venv312/bin/python -m cryptofarm.ml.meta_trainer
.venv312/bin/python -m cryptofarm.ml.policy_trainer

# Measurements (--help lists every measure)
.venv312/bin/python -m scripts.analysis --barrier-capacity

# Live bot — places real orders, needs env vars
.venv312/bin/python src/cryptofarm/trading/live_bot.py
```

Tests: `.venv312/bin/python -m pytest` (188 tests). Lint/format: `ruff check src tests`,
`black src tests`.

### Dependency extras

The install is split so each image carries only what it runs.

| extra | contents | needed by |
|---|---|---|
| (core) | numpy, pandas, scipy, ta, requests, python-binance, scikit-learn, colorama | features, labels, gbdt models, live bot |
| `app` | streamlit, plotly | `trading/simulator.py` and the modules it decorates with `st.cache_data` |
| `data` | pyarrow (141 MB) | the parquet kline store — `data/klines.py` and `scripts/analysis.py`; arrives anyway with `app`, since streamlit requires `pyarrow>=7.0` |
| `dl` | tensorflow, keras-tuner (~1 GB) | only `--model gru|cnn|lstm`; `ml/models.py` imports keras inside the functions |
| `dev` | pytest, ruff, black, pre-commit | |

`data` is separate because the core install has no reason to carry a parquet engine — not because
the deployed image can shed it. Streamlit depends on pyarrow, so any image with the page has it.

## Docker

```bash
mkdir -p models market_data          # first run only, so the bind mounts exist

docker compose up simulator                     # http://localhost:8501
docker compose --profile data  run --rm klines  # populate the kline store
docker compose --profile train run --rm trainer # train (gbdt)
docker compose --profile ci    run --rm tests   # the suite, in the CI image
```

One `Dockerfile`, four targets:

| target | contents | for |
|---|---|---|
| `runtime` | core + `app` + `data` | simulator, trainers, kline store, analysis |
| `dev` | `runtime` + pytest/ruff/black | the image CI runs |
| `dl` | `runtime` + TensorFlow | `--model gru|cnn|lstm` |
| **`web`** | same as `runtime` | **production — it is the file's last stage, so a plain `docker build .` produces it** |

The stage order matters: a build without `--target` takes the last stage, and Render has no field
to choose one. `web` exists to occupy that position, so the platform gets the page image instead of
the TensorFlow one — CI builds without `--target` and fails if that stops being true. Any new stage
belongs above it.

`models/` and `market_data/` are bind-mounted from the host, so artifacts and the hundreds of MB of
candles survive `docker compose down`. Inside the image they live at `/app/models` and
`/app/market_data`, pointed there by `CRYPTOFARM_MODELS_DIR` and `CRYPTOFARM_MARKET_DATA_DIR` —
`paths.py` reads both, because the package is installed in `site-packages` and the project root it
would otherwise derive from its own location points inside the virtualenv.

The container runs as uid 1000. On Linux, if your user has a different uid, the bind-mounted files
will be written as 1000; uncomment the `user:` line in `compose.yaml` to avoid it.

The live bot is deliberately not a compose service: `live_bot.py` runs its loop at import time with
no `main()` and no signal handling, so it is not safe to restart automatically. Containerising it
needs that refactor first.

## Deploy (Render, free plan)

`render.yaml` describes the public service: the simulator alone, built from the Dockerfile's last
stage. Points worth knowing before touching it:

- **Port.** Render assigns the port through `$PORT` (10000 by default) and requires binding to
  `0.0.0.0`. The image's command reads `${PORT:-8501}`, so the same image serves Render and
  `docker compose` locally.
- **Region `frankfurt`.** Binance blocks US IP addresses on `api.binance.com`, which is where the
  simulator fetches candles on every interaction. A US region silently breaks the page.
- **No model artifact.** The free plan has no persistent disk and `models/*.joblib` is gitignored,
  so the deployed page runs the classic strategies; picking "AI Model" reports the missing
  artifact. The simulator loads the model inside the page, not at import, so nothing else breaks.
- **Memory.** The free instance has 512 MB. `MALLOC_ARENA_MAX=2` caps glibc's per-thread arenas,
  and the `st.cache_data` decorators carry `ttl`/`max_entries` — an unbounded cache filled by
  sliding the sidebar widgets is what pushes the process into an OOM restart.
- **What the plan still costs you.** Free services spin down after 15 minutes idle and take about a
  minute to wake. Nothing in the image changes that; only a paid instance does.

Changing region or runtime on an existing Render service is not possible — those require creating
a new service.

## CI

`.github/workflows/ci.yml` runs on every pull request and on pushes to `main`, in two jobs:

- **quality** — installs `.[app,dev]` on Python 3.12 and runs `ruff check`, `black --check` and
  `pytest` over `src`, `tests` and `scripts`. No network and no kline store: the tests are synthetic
  data and monkeypatches.
- **docker** — builds the `runtime` and `dev` targets with buildx (GitHub Actions cache), checks the
  image can import the package and resolves its data directories to `/app/...`, then runs the suite
  inside the `dev` image.

Nothing is pushed anywhere: image publishing and deployment come when there is somewhere to deploy.

## The model the simulator uses

`ml/trainer.MODEL_PRECEDENCE` is `("policy_model", "meta_model", "signal_model")`; `active_model_name()`
is the single source of truth for both loading and strategy dispatch, so they cannot diverge. To fall
back to an earlier model, move the newer artifact out of `models/`.

**Currently active: `policy_model`** — the three-action, position-conditioned policy. `ai_model_simulation`
routes it to `policy_signals`, which decides entries *and* exits (the barriers do not apply).

| | |
|---|---|
| Type | `HistGradientBoostingClassifier`, 3 classes (hold / buy / sell) |
| Trained | 2026-08-20, 36 min fit |
| Input features | 83 |
| Boosting iterations | 400 (no early stop) — 1.200 trees, 150.000 nodes, 75.600 leaves, max depth 27 |
| Hyperparameters | `learning_rate=0.06`, `max_leaf_nodes=63`, `min_samples_leaf=200`, `l2=1.0` |
| Labelling | directional change, `capture=0.30`, 8–12 confirmed extremes/day |
| Data | 15 symbols, 5m bars since 2022-01-01 — 3.653.165 rows, plus 2,4 M added over 2 DAgger rounds |
| Decision threshold | 0.50 · assumed round-trip cost 0,08% |

A gradient boosting has no weights: the closest analogue to a parameter count is the leaf count
(~75.600 values) plus the split thresholds. It is not comparable to an LSTM's parameter count.

### Measured performance — negative

No artifact stores accuracy, and it would mislead here: with these class balances a model that always
says "hold" scores well and trades nothing. What is recorded:

| Model | Metric | Value |
|---|---|---|
| `policy_model` | holdout, 10.483 trades | gross −0,123%, **net −0,203%/trade**, win rate 32,6% |
| `policy_model` | in-sample, 58.866 trades | gross +0,032%, net −0,048%/trade |
| `policy_bassa` | holdout, 63.293 trades | net −0,091%/trade, win rate 33,5% |
| `meta_model` | CPCV, 28 splits | PBO 0,00 · Deflated Sharpe 1,00 · net +0,097%/trade |
| `signal_model` | validation | **AUC 0,5401** · win rate 39,6% vs break-even 47,3% |

The in-sample gross edge of +0,032% sits below the 0,08% round-trip cost. That gap — not a modelling
failure — is why none of this is profitable; see §13 of `.claude/docs/strategy.md`.

Three caveats worth carrying:

- `meta_model`'s PBO of 0,00 and Deflated Sharpe of 1,00 are saturated values. Its
  `mean_uniqueness` is 0,034, meaning each label overlaps ~29 others, so the effective sample is far
  smaller than 1,5 M events and significance metrics skew optimistic.
- The policy artifacts were trained with `--no-cpcv`, so they carry holdout numbers only. The CPCV
  figures in `strategy.md` §12.1/§12.3 predate a leak fix in `_cpcv` and **need rerunning**; the
  negative verdict stands, the values do not. See §14 of `strategy.md`.
- `policy_alta.joblib` is byte-identical to `policy_model.joblib` — the same operating point saved
  twice.

## Configuration

Environment variables only — see `.env.example`. Nothing loads `.env` automatically.
`API_KEY`/`API_SECRET` and the strategy parameters are read by `live_bot.py`; `MARKET_DATA_CSV`
points the Streamlit page at a historical CSV. The simulator and the trainers use Binance's public
endpoints and need no credentials.

## Known issue

`add_technical_indicator` still has its `MACD` block commented out, and `buy_sell_limits_simulation`
reads that column, so it raises `KeyError` when called. No menu entry reaches it — the dispatch binds
it to `"Buy/Sell Limits"`, which is not in `config.STRATEGIES`, as with `"ATR Bands"` and the
`"Dinamic *"` variants. Making it usable means restoring the `MACD` block *and* adding the menu entry.

`PSAR` was in the same state and has been restored: `"Close ATR"` and the `"ATR Live Trade"`
stop-loss no longer raise.

## Notes for contributors

`tests/test_simulator_golden.py` pins the simulator's behaviour against
`tests/data/simulator_golden.json`. It must pass before and after any change to `trading/`, without
being regenerated. Regenerating accepts any behaviour change, so do it only deliberately.

Row-wise reads in `trading/` go through numpy arrays hoisted out of the loop, not
`df["Col"].iloc[i]`; that is where the simulator's speed comes from (4295 ms → 125 ms). Keep the
style.

`indicators._atr_ema` reproduces `ta` 0.11's ATR and EMA formulas in numpy. If you change it,
re-verify against `ta` — a silent divergence there moves every signal.
