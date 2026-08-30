# CryptoFarm

Trains signal models on Binance data and checks them against indicator strategies, over nine years of
history and fifteen assets. There is also a headless bot that can trade live.

**The result in one line.** Almost everything that has been tried does not beat passive holding, and
it is written down where it was measured. The only thing that passes the paired-exposure control is
the entry model: **+2.071% net per trade out of sample, 14 symbols out of 15 profitable**, and its
advantage is not the prediction but the **selectivity** — it flags one bar in two hundred.

## How it is built

```
src/cryptofarm/
├── data/      the local candle store (bulk dumps, parquet)
├── ml/        features → labels → model → evaluation → serving
└── trading/   strategies, P&L accounting, the Streamlit page, the live bot
scripts/       eighteen measurement benches: they produce the documents' numbers
tests/         1,024 cases, no network, no store required
.claude/docs/  the decisions and the measurements that justify them
```

Every folder has its own README listing the files and functions:
[`src/cryptofarm/data/`](src/cryptofarm/data/) ·
[`src/cryptofarm/ml/`](src/cryptofarm/ml/) ·
[`src/cryptofarm/trading/`](src/cryptofarm/trading/) ·
[`scripts/`](scripts/) · [`tests/`](tests/) · [`models/`](models/) · [`reports/`](reports/)

Two things really matter: **`trading/simulator.py`** (the research) and **`ml/trainer.py`** (the
training), plus their dependencies and one bot.

## Getting started

Python >= 3.12, environment **`.venv312`** — the pre-existing `.venv` is 3.9 and has no
`scikit-learn`.

```bash
pip install -e ".[app,data,dev]"

# The page: two views, "when to be in" and "which asset to hold"
streamlit run src/cryptofarm/trading/simulator.py

# The candle store, a prerequisite for any training (~10 minutes)
.venv312/bin/python -m cryptofarm.data.klines --update

# The model at the head today: the fast one trades, the slow one gates it
.venv312/bin/python -m cryptofarm.ml.entry_trainer --selfcheck   # runs without the store
.venv312/bin/python -m cryptofarm.ml.entry_trainer               # the slow one (H=150)
.venv312/bin/python -m cryptofarm.ml.entry_trainer --h 20 --quantile 0.995 --nome entry_model_veloce
.venv312/bin/python -m scripts.entry_lab                         # what the gate is worth

# The live bot — it places real orders, it wants the variables from .env.example
.venv312/bin/python src/cryptofarm/trading/live_bot.py
```

Tests: `.venv312/bin/python -m pytest`. Lint: `ruff check src scripts tests` and
`black src scripts tests`.

The other commands — the other models, the strategy sweeps, the measurement benches — are in
[`CLAUDE.md`](CLAUDE.md) and in the READMEs of [`scripts/`](scripts/) and
[`src/cryptofarm/ml/`](src/cryptofarm/ml/).

## The installation extras

| extra | what it holds | what it is for |
|---|---|---|
| (core) | numpy, pandas, scipy, ta, requests, python-binance, scikit-learn | features, labels, `gbdt` models, live bot |
| `app` | streamlit, plotly | only `trading/simulator.py` and the modules it decorates with `st.cache_data` |
| `data` | pyarrow (141 MB) | the store's parquet engine: `data/klines.py`, `scripts/analysis.py` |
| `dl` | tensorflow (~1 GB) | only `--model gru\|cnn\|lstm` |
| `dev` | pytest, ruff, black, pre-commit | |

A leaner image for the page alone is not obtained by dropping pyarrow: `streamlit` depends on
`pyarrow>=7.0`, so the 141 MB are there anyway.

## Docker and deploy

```bash
mkdir -p models market_data                      # the bind mounts must exist
docker compose up simulator                      # http://localhost:8501
docker compose --profile data  run --rm klines
docker compose --profile train run --rm trainer
docker compose --profile ci    run --rm tests
```

A single `Dockerfile`, four targets: `runtime` (page, trainer, store), `dev` (`runtime` +
pytest/ruff/black, the CI image), `dl` (`runtime` + TensorFlow) and **`web`**, which is the one that
goes to production. **`web` is the last stage and must stay there**: a build without `--target` takes
the last one, and Render has no field to choose it. A new stage goes above, never below — and CI
builds without `--target` precisely to notice.

The public deploy is in [`render.yaml`](render.yaml), free plan, region **`frankfurt`**: Binance
blocks US IPs on `api.binance.com`, which is where the simulator gets its candles, so the region is
not a detail. The free plan has no persistent disks, and `models/*.joblib` is gitignored: **online
the classic strategies run**, and the page removes the "AI Model" entry from the menu by itself
instead of crashing.

## What to know before making changes

**`.claude/docs/` contains measurements that explicitly rule out several paths that look reasonable
at first sight.** The reading order is in [`.claude/docs/README.md`](.claude/docs/README.md);
whoever picks the work up cold reads [`HANDOFF.md`](.claude/docs/HANDOFF.md) and nothing else.
Whoever touches the ML pipeline reads [`strategy.md`](.claude/docs/strategy.md) first, and whoever
touches labels or training reads
[`labeling-strategy.md`](.claude/docs/labeling-strategy.md).

**The golden master must be respected.** `tests/test_simulator_golden.py` pins the behaviour of 21
functions: it must pass before a change to `trading/` and pass again afterwards, **without being
regenerated**. Regenerating it accepts any difference, including a regression.

**The starting values are central, not optimal.** Looking for the in-sample maximum transfers worse
than picking a configuration at random: on the rotation the correlation between in-sample and
out-of-sample return is **−0.69**. Whoever changes the defaults to "the ones that return most in the
chart" is making exactly the measured mistake.

**Per-row reads go through numpy arrays** extracted before the loop, not through
`df["Col"].iloc[i]`: that is where most of the speed comes from (the whole simulator: 4295 ms → 125
ms). And `indicators._atr_ema` replicates `ta` 0.11's formulas in numpy line by line — if it is
changed, it must be reverified against `ta`, because a silent divergence there moves every signal.

## A known defect

`buy_sell_limits_simulation` reads the `MACD` column, which remains commented out in
`add_technical_indicator`: it raises `KeyError` as soon as it is called. No menu entry reaches it.
Making it usable means restoring the `MACD` block **and** adding the entry.

## Configuration

Environment variables only, see [`.env.example`](.env.example) — nothing in the repository loads
`.env` by itself. `API_KEY`/`API_SECRET` and the strategy parameters are read by `live_bot.py`
alone; `MARKET_DATA_CSV` is the page's historical CSV. The simulator and the trainers use Binance's
public endpoints and want no credentials.

## Language

**The documentation of this project is written in English** — `CLAUDE.md`, `.claude/docs/` and every
folder `README.md`. The rule is at the top of [`CLAUDE.md`](CLAUDE.md) and it is not a style
preference: it had been given before and the documents drifted back anyway.

Code and function names are English where they are domain terms (`simulate_positions`,
`swing_target`) and Italian where they describe a decision taken here (`perche_non_entra`,
`scala_fuori_misura`, `votanti_predefiniti`). Renaming those is a separate job that touches tests
asserting on the Italian names, and it is out of scope for the documentation rule.
