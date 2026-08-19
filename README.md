# CryptoFarm

Crypto trading simulator, live Binance trading bots, and an LSTM-based buy/sell signal
model, built on `python-binance`, `ta`, TensorFlow/Keras and Streamlit.

> **Status**: v1 is the only active codebase. Simulator v2 (a more advanced multi-timeframe
> Streamlit UI) has been archived to [`backup/v2/`](backup/v2/README.md) for reference — see
> that folder's own README for its architecture. It is not part of the active package and is
> excluded from linting/formatting.

## What's in here

| Tool | Entry point | What it does |
|---|---|---|
| Backtesting / strategy simulator | `src/cryptofarm/trading/simulator.py` | Streamlit UI — downloads historical klines and backtests ~10 strategies (ATR bands, RSI/MACD limits, Supertrend, an LSTM-driven strategy, ...). Primary research tool. |
| Live trading dashboard | `src/cryptofarm/app/dashboard_live.py` | Streamlit UI — PSAR/ATR signals on a live Binance WebSocket feed, can place real orders. |
| Headless live bot (single account) | `src/cryptofarm/app/live_bot.py` | Console bot, same ATR/RSI signal logic, places real orders unattended. |
| Headless live bot (dual account) | `src/cryptofarm/app/live_bot_dual.py` | Same as above, mirrors the same signal across two Binance accounts. |
| Grid-search backtester | `src/cryptofarm/trading/grid_search.py` | Runs the strategy backtest across a cartesian product of parameters/assets to search for good settings. |
| Grid-search results viewer | `src/cryptofarm/app/grid_results_viewer.py` | Streamlit UI — heatmaps/violin plots over a CSV export produced by the grid search. |
| LSTM trainer | `src/cryptofarm/ml/trainer.py` | Feature engineering (RSI/ATR/Stochastic/TSI via `ta`, percentage-change encoding) and training of the bidirectional-LSTM buy/sell/hold classifier consumed by `simulator.py`'s "AI Model" strategy. |

There is no shared module between the simulator and the live bots yet — indicator/signal
logic is intentionally duplicated per entry point rather than partially refactored (see
`CLAUDE.md` for the reasoning). `src/cryptofarm/data/` is an empty placeholder package,
reserved for a future split of the data-fetching logic out of `trading/simulator.py`.

## Project structure

```
src/cryptofarm/
    data/          # placeholder, reserved for a future split of market-data fetching
    trading/       # backtesting engine, strategies, grid search
    ml/            # feature engineering + LSTM training
    app/           # Streamlit apps and live-trading entry points
tests/             # pytest suite
models/            # trained .keras models (gitignored — regenerate with ml/trainer.py)
backup/v2/         # archived Simulator v2, not active
```

## Setup

Requires **Python >= 3.12**.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

This installs the pinned runtime dependencies plus the dev tools (`pytest`, `ruff`,
`black`, `pre-commit`) from `pyproject.toml`. `requirements.txt` carries the same runtime
pins and is what the existing Render deployment installs from.

Optionally enable the formatting/lint pre-commit hook:
```bash
pre-commit install
```

### Environment variables

Copy `.env.example` to `.env` and fill in real values before running anything that places
orders or reads account balances (`dashboard_live.py`, `live_bot.py`, `live_bot_dual.py`).
Nothing in this repo loads `.env` automatically yet — export the variables into your shell
(`export $(grep -v '^#' .env | xargs)`) or set them via your IDE's run configuration.

The backtesting/simulation tools (`simulator.py`, `grid_search.py`) call Binance's public
market-data endpoints directly and don't need real credentials to run.

| Variable | Used by | Notes |
|---|---|---|
| `API_KEY`, `API_SECRET` | `dashboard_live.py`, `live_bot.py` | Binance API credentials, single account. |
| `API_KEY1`, `API_SECRET1`, `API_KEY2`, `API_SECRET2` | `live_bot_dual.py` | Two Binance accounts traded in parallel with the same signal. |
| `ASSET`, `CURRENCY`, `CANDLES_TIME` | `live_bot.py`, `live_bot_dual.py` | e.g. `AMP`, `USDT`, `15m`. |
| `SMA_WINDOW`, `ATR_WINDOW`, `ATR_MULTIPLIER`, `RSI_WINDOW`, `RSI_BUY_LIMIT`, `RSI_SELL_LIMIT`, `NUM_CONDITIONS` | `live_bot.py`, `live_bot_dual.py` | Strategy parameters — see `.env.example` for working defaults. |

No credential has ever been hardcoded in this repository or its git history (verified
across the full commit history, not just the current tree).

## Running the apps

```bash
# Backtesting / strategy simulator (primary research tool)
streamlit run src/cryptofarm/trading/simulator.py

# Live trading dashboard
streamlit run src/cryptofarm/app/dashboard_live.py

# Grid-search results viewer (edit the hardcoded CSV path at the top of the file first)
streamlit run src/cryptofarm/app/grid_results_viewer.py

# Headless live bot(s) — place real orders, require the env vars above
python src/cryptofarm/app/live_bot.py
python src/cryptofarm/app/live_bot_dual.py

# Grid-search over strategy parameters (edited directly in the
# if __name__ == "__main__": block)
python src/cryptofarm/trading/grid_search.py

# Train/retrain the LSTM model (CSV input path is hardcoded in the
# if __name__ == "__main__": block — edit it first)
python src/cryptofarm/ml/trainer.py
```

## Tests

```bash
pytest
```

3 tests cover the technical-indicator feature pipeline in `cryptofarm.ml.trainer`
(`add_technical_indicator`, `calculate_relative_extrema`, `calculate_percentage_changes`)
against synthetic OHLC data — no network calls.

## Linting & formatting

```bash
black src tests
ruff check src tests
```

`backup/v2/` is excluded from both — it's frozen legacy code, not touched.
