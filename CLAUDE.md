# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

CryptoFarm simulates and trades cryptocurrency strategies against the Binance API. It lives in a `src/cryptofarm/`
package with four sub-packages, plus an archived legacy UI. Each entry point below shares indicator/simulation
building blocks but is otherwise independent:

- **Live dashboard** (`src/cryptofarm/app/dashboard_live.py`) — Streamlit app, connects to Binance via REST +
  WebSocket, plots candles with PSAR/ATR bands, and can place real orders.
- **Headless trading bots** (`src/cryptofarm/app/live_bot.py`, `live_bot_dual.py`) — same signal logic as the
  dashboard but run as a console loop (auto-reconnecting WebSocket, colorama output) and call
  `proceed_buy`/`proceed_sell` to place real orders unattended (e.g. on Render, see `render.yaml`). `live_bot.py`
  trades one Binance account; `live_bot_dual.py` mirrors the same signal across two accounts (`API_KEY1`/`API_KEY2`).
  The two files are ~90% duplicated code — deliberately not deduplicated (see "Known duplication" below).
- **Backtesting/simulation Streamlit app** (`src/cryptofarm/trading/simulator.py`) — the main research tool.
  Downloads historical klines, computes many indicator sets, and evaluates ~10 different trading strategies
  (`trading_analysis`), including one driven by the trained LSTM (`ai_model_simulation`, "AI Model" strategy).
- **Batch/grid-search simulation** (`src/cryptofarm/trading/grid_search.py`, `simulator_opt.py`) — runs
  `trading_analysis`-equivalent logic across a cartesian product of parameter values/assets to find good strategy
  parameters; results are analyzed with `src/cryptofarm/app/grid_results_viewer.py` (a small Streamlit app reading a
  CSV export and plotting heatmaps/violin plots).
- **ML training** (`src/cryptofarm/ml/trainer.py`) — builds/trains the bidirectional-LSTM classifier (predicts
  buy/sell/hold at each candle) that `trading/simulator.py`'s "AI Model" strategy consumes. It assembles its own
  dataset from a cartesian product of assets × timeframes downloaded from Binance's public market-data endpoints
  (`TRAIN_ASSETS` / `TRAIN_INTERVALS` / `TRAIN_HOURS` at the top of the file), plus any local CSVs listed in
  `EXTRA_CSV_FILES`. Shares its feature engineering (`add_technical_indicator`,
  `normalize_scale_dependent_features`, `calculate_percentage_changes`, `create_sequences`) with the prediction
  path (`get_model_predictions`), so changing feature logic here requires retraining before any consumer of
  `models/trained_model.keras` / `models/optimized_model.keras` will behave correctly.

`src/cryptofarm/data/` is an empty placeholder package, reserved for a future split of the data-fetching logic
currently living inside `trading/simulator.py`.

### Known duplication (intentional, not yet resolved)

There is no shared module between `trading/simulator.py` and `app/live_bot.py` / `app/dashboard_live.py` —
indicator calculation (`add_technical_indicator`) and signal logic are duplicated per file with slightly different
parameter sets. `app/live_bot.py` and `app/live_bot_dual.py` are themselves ~90% duplicated (fetch/reconnect/order
logic identical, `live_bot_dual.py` adds a second account). This was a deliberate choice when restructuring the
repo (2026-08) to keep the physical reorganization low-risk on code that places real orders — extracting a shared
`trading/live_common.py` is a legitimate follow-up, not an oversight. When changing signal/indicator logic, check
whether the same logic needs to be mirrored in the sibling file(s).

### Archived v2

`backup/v2/` holds a more advanced, multi-timeframe Streamlit simulator (`simulator_v2_app.py` +
`simulator_v2_backend.py` + `simulator_v2_ai_trainer.py` + `simulator_v2_massive.py`, see its own
`backup/v2/README.md`). It was the active development target before the repo was restructured back onto v1; it is
not imported by anything in `src/cryptofarm/` and is excluded from lint/format tooling. Treat it as read-only
reference material, not something to extend.

## Running

Install dependencies (Python >= 3.12):
```bash
pip install -e ".[dev]"
```
(`requirements.txt` carries the same pinned runtime deps and is what the existing Render deployment installs from.)

```bash
# Backtesting / strategy simulator (primary research tool)
streamlit run src/cryptofarm/trading/simulator.py

# Live trading dashboard
streamlit run src/cryptofarm/app/dashboard_live.py

# Grid-search results viewer (edit the hardcoded CSV path at the top of the file first)
streamlit run src/cryptofarm/app/grid_results_viewer.py

# Headless live bot(s) — places real orders, requires env vars (see Configuration below)
python src/cryptofarm/app/live_bot.py
python src/cryptofarm/app/live_bot_dual.py

# Grid-search over strategy parameters (parameters are edited directly in the
# if __name__ == "__main__": block)
python src/cryptofarm/trading/grid_search.py

# Train/retrain the LSTM model. Downloads its own data from Binance; data sources, labeling
# parameters and hyperparameters are module-level constants at the top of the file
python src/cryptofarm/ml/trainer.py
```

Tests: `pytest` (11 tests, `tests/test_indicators.py`, covering the `ml/trainer.py` feature, labeling and
balancing pipeline with synthetic OHLC data). Lint/format: `ruff check src tests` and `black src tests` (config in
`pyproject.toml`; `backup/` is excluded from both). `.pre-commit-config.yaml` runs black + `ruff --fix` on commit
if installed (`pre-commit install`).

## Configuration

Binance credentials and bot parameters are passed via environment variables — see `.env.example` for the full list
and working defaults. Nothing in the repo loads `.env` automatically (no `python-dotenv` dependency); export the
variables into your shell or your IDE's run configuration before running anything that trades.

- `API_KEY`, `API_SECRET` — used by `dashboard_live.py` and `live_bot.py`.
- `API_KEY1`/`API_SECRET1`/`API_KEY2`/`API_SECRET2` — used by `live_bot_dual.py` (two accounts).
- `live_bot.py`/`live_bot_dual.py` additionally read: `ASSET`, `CURRENCY`, `CANDLES_TIME`, `SMA_WINDOW`,
  `ATR_WINDOW`, `ATR_MULTIPLIER`, `RSI_WINDOW`, `RSI_BUY_LIMIT`, `RSI_SELL_LIMIT`, `NUM_CONDITIONS`.
- `trading/simulator.py`/`grid_search.py` call Binance's public market-data endpoints with a placeholder client and
  don't need real credentials.

`.streamlit/config.toml` sets the dark theme for the Streamlit apps.

## Data/model artifacts

- `models/optimized_model.keras`, `models/trained_model.keras`, `models/trained_model1.keras` — pretrained Keras
  models. Gitignored (`models/.gitignore`), not tracked — a fresh clone starts with an empty `models/` dir.
  `trading/simulator.py` loads `models/optimized_model.keras` for the "AI Model" strategy;
  `ml/trainer.py`'s `MODEL_PATH` points at `models/trained_model.keras`. Regenerate with `ml/trainer.py` rather
  than hand-editing.
- `tuner_logs/` — Keras Tuner search state from a (currently commented-out) hyperparameter search in
  `ml/trainer.py`.
- Historical CSVs used for analysis are expected as local files at hardcoded paths (see the
  `if __name__ == "__main__":` block in `app/grid_results_viewer.py`, and `EXTRA_CSV_FILES` in `ml/trainer.py`) —
  these paths are machine-specific and must be updated per environment.

## Working with the indicator/strategy code

- All OHLCV DataFrames are indexed by `Open time` (a `DatetimeIndex`) with columns `Open, High, Low, Close,
  Volume`.
- Strategy functions in `trading/simulator.py` (`atr_buy_sell_simulation`, `close_atr_buy_sell_simulation`,
  `buy_sell_limits_simulation`, `supertrend_simulation`, `trend_zone_simulation`, `ai_model_simulation`, etc.) all
  return `(buy_signals, sell_signals)` as lists of `(timestamp, price)` tuples, which `trading_analysis` then feeds
  into `simulate_trading_with_commisions` / `simulate_trading_with_commisions_multiple_buy` to compute P&L
  including trading fees.
- `ml/trainer.py`'s feature pipeline rescales the scale-dependent features
  (`normalize_scale_dependent_features`: ATR as a percentage of Close, the bounded oscillators onto ~[-1,1]) and
  converts OHLC to percentage changes relative to the previous close (`calculate_percentage_changes`) before
  windowing into sequences (`create_sequences`, window size `WINDOW_SIZE = 50`) — the model consumes relative
  price movement, not absolute price, which is what lets one model span several assets. `get_model_predictions`
  in the same file mirrors this preprocessing at inference time, and `trading/simulator.py`'s
  `ai_model_simulation` calls into it directly via `from cryptofarm.ml.trainer import get_model_predictions, ...`;
  keep both paths in sync. Any new feature in absolute price units must be rescaled inside
  `normalize_scale_dependent_features`, which is the one place both paths go through.
- Labels come from `calculate_relative_extrema`, a three-stage cascade run on *absolute* prices: local extrema
  (`EXT_WINDOW_SIZE`), then a minimum forward return within a horizon (`LABEL_MIN_RETURN`,
  `LABEL_RETURN_HORIZON`), then a minimum spacing between consecutive signals (`LABEL_COOLDOWN`). Retuning these
  changes how many signals exist and how large the move behind each one is — check the printed per-stage class
  distribution rather than guessing. The train/validation split (`split_train_val`) applies an embargo sized from
  those lookaheads; shrinking it reintroduces leakage across the split. Class balancing is done on the data
  (`balance_signal_classes`, `downsample_holds`) and applied to the *training set only*, so validation metrics
  still reflect the real market distribution.
