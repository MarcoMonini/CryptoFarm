# `data/` — the local candle store

Two stores, the same infrastructure: bulk dumps from `data.binance.vision`, a parquet archive in
`market_data/`, incremental updates. It is the **prerequisite for training**: the trainers read from
here, they do not download on the fly.

| file | lines | what it stores |
|---|---|---|
| `klines.py` | 431 | the OHLCV candles, **a single interval per symbol** (5m), from which 15m/30m/1h are derived by aggregation |
| `positioning.py` | 289 | futures positioning: long/short ratio, open interest, funding, basis |

## The functions

**`klines.py`** — `update_store` and `update_symbol` to build and update, `load_klines` and
`resample_klines` to read, `store_path` and `store_manifest` to know what is there,
`interval_to_minutes` for the conversion, `clip_wicks` and `wick_outliers` for candles with
implausible wicks. `main` is the entry point of `python -m cryptofarm.data.klines --update`.

**`positioning.py`** — the same shape: `update_store`, `update_symbol`, `load_positioning`,
`store_path`, `store_manifest`, `main`, plus the two downloaders `fetch_metrics_day` and
`fetch_funding_month`, which have different cadences (daily and monthly) because that is how Binance
publishes them.

## Why a single interval, and why the dumps and not the REST

**A single interval** because aggregation from 5m is exact — verified against the official dumps,
zero difference on OHLC. One source of truth instead of four archives to keep aligned, and a quarter
of the network requests.

**The dumps and not the REST** because at this scale the cost is entirely per-request latency
(~2.7 s measured, identical on both routes) and not bandwidth. REST gives at most 1000 candles per
call and is rate limited: ~18,700 sequential requests for the history, about 14 hours. A monthly dump
contains ~8,900 of them and is a static file on a CDN with no rate limit, hence parallelisable:
~1,350 files with 32 workers, under ten minutes. REST is still used **only for the tail**, i.e.
today's candles, which are not in the dumps yet.

## Where the data ends up

In `market_data/`, which is gitignored and weighs about 4 GB with a full store. The location moves
with `CRYPTOFARM_MARKET_DATA_DIR` (see [`../paths.py`](../paths.py)); in a container the image sets
it to `/app/market_data`, where `compose.yaml` mounts the host folder.

Where `data.binance.vision` is unreachable, `scripts/import_candles.py` builds the same store from a
local clone of a one-minute dataset.
