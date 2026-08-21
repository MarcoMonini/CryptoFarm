# CryptoFarm

Trains a signal model on Binance market data and backtests trading strategies against it, with one
headless bot that can trade the result live.

Two files matter — `trading/simulator.py` (research) and `ml/trainer.py` (training) — plus their
dependencies. Everything not reachable from those lives in `backup/unused/`.

## Layout

| What | Where |
|---|---|
| Backtest / strategy simulator (Streamlit) | `src/cryptofarm/trading/simulator.py` |
| Strategies, indicators, P&L, market data, defaults | `src/cryptofarm/trading/{strategies,indicators,pnl,market_data,config}.py` |
| Training pipeline | `src/cryptofarm/ml/` (`trainer.py`, `meta_trainer.py`, `policy_trainer.py`) |
| Local kline store (Binance bulk dumps) | `src/cryptofarm/data/klines.py` |
| Headless live bot (places real orders) | `src/cryptofarm/trading/live_bot.py` |
| Measurements behind `.claude/docs/strategy.md` | `scripts/analysis.py` |

## Running

Python >= 3.12. Use `.venv312/bin/python` — the older `.venv` is 3.9 and lacks `scikit-learn`.

```bash
pip install -e ".[dev]"

# Backtest / strategy simulator
streamlit run src/cryptofarm/trading/simulator.py

# Kline store first, then train
.venv312/bin/python -m cryptofarm.data.klines --update
.venv312/bin/python -m cryptofarm.ml.trainer                # gbdt by default
.venv312/bin/python -m cryptofarm.ml.meta_trainer
.venv312/bin/python -m cryptofarm.ml.policy_trainer

# Measurements
.venv312/bin/python -m scripts.analysis

# Live bot — places real orders, needs env vars
.venv312/bin/python src/cryptofarm/trading/live_bot.py
```

Tests: `.venv312/bin/python -m pytest` (185 tests). Lint/format: `ruff check src tests`,
`black src tests`.

## Which model the simulator uses

`ml/trainer.MODEL_PRECEDENCE` is `("policy_model", "meta_model", "signal_model")`. Whichever exists
in `models/` first wins, for both loading and strategy dispatch. To fall back to the previous model,
move the newer artifact out of `models/`.

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
