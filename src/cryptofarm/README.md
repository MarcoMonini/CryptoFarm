# `cryptofarm/` — the package

Three subpackages and one module. The dependencies go in one direction only:

```
paths.py   where data and models live   (depends on nothing)
data/      the local candle store
   ↓
ml/  ⇄  trading/
```

| what | what it is for |
|---|---|
| [`data/`](data/) | downloads and stores the Binance candles. A prerequisite for training |
| [`ml/`](ml/) | from candles to model to signal: features, labels, models, evaluation, serving |
| [`trading/`](trading/) | strategies, P&L accounting, the Streamlit page, the live bot |
| `paths.py` | `MODELS_DIR` and `MARKET_DATA_DIR`, with the environment-variable override |

**`ml/` and `trading/` are not in a hierarchy, they are joined at two points.** `trading/` asks `ml/`
for the model serving (`signals`, `trainer`, and in `panels` also `labeling` and `entry_trainer`, to
draw the target and the pivots next to the prediction). In the other direction there is a single
module: `ml/bar_features.py` imports `trading.indicators_extra` and `trading.mtf`, and that is
deliberate — the model's features must be **the same** indicators and **the same** alignment between
intervals that the page uses, not a second implementation that can silently diverge. Those are the
two points to look at first if you are thinking of moving something between the two packages.

Both read `data/`: `ml/` for the training samples, `trading/` for the cross-sectional rotation and
for interval conversion.

## `paths.py`

Two constants and nothing else. They are relative to the repository root, and they move with
`CRYPTOFARM_MODELS_DIR` and `CRYPTOFARM_MARKET_DATA_DIR`. The override is not a luxury: inside the
image the package is installed in `site-packages`, so the root inferred from the file's location
would point **inside the virtualenv** instead of at the mounted volume, and models trained in a
container would end up in a throwaway layer. Whoever touches this file has to keep it working.

## The installation extras

The core (`pip install -e .`) is enough for features, labels, `gbdt` models and the live bot. `[app]`
adds Streamlit and Plotly, which are needed **only** by `trading/simulator.py` and the modules it
decorates with `st.cache_data`. `[data]` adds pyarrow, i.e. the parquet engine `data/klines.py` and
`scripts/analysis.py` want. `[dl]` adds TensorFlow, roughly 1 GB, and is only needed for
`--model gru|cnn|lstm`. The normal case is `pip install -e ".[app,data,dev]"`.
