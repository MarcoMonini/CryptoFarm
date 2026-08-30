# `market_data/` — the local store

**Not tracked.** Around 4.2 GB with a full store, 15 symbols from 2017. A clone of the repository
finds this folder empty (apart from this file), and in production — where Render has no persistent
disks — it stays empty forever: that is the condition for which the *Cross-asset rotation* view says
it has no data instead of attempting fifteen downloads.

## What ends up in here

| shape | produced by | what it holds |
|---|---|---|
| `<SYMBOL>-5m.parquet` | `python -m cryptofarm.data.klines --update` | the OHLCV candles. **5m only**: 15m/30m/1h are derived by aggregation, and that is exact |
| `<SYMBOL>-positioning.parquet` | `python -m cryptofarm.data.positioning --update` | long/short ratio, open interest, funding, basis. ~400 MB |
| `*.pkl` | the benches in `scripts/` | cache of a model's predictions, keyed on the artifact's `created` signature |

The `.pkl` files are **cache, not data**: `rl_stati.pkl` alone weighs 3.5 GB and is rebuilt by
rerunning `scripts/rl_lab.py`. Deleting them costs CPU time, not information. The candle `.parquet`
files, on the other hand, cost hours of network: those are kept.

## Rebuilding it

```bash
.venv312/bin/python -m cryptofarm.data.klines --update        # under 10 minutes with 32 workers
.venv312/bin/python -m cryptofarm.data.positioning --update   # ~400 MB
```

The data comes from the monthly dumps of `data.binance.vision`, not from the REST API — the reason,
which is a factor of eighty on time, is in
[`../src/cryptofarm/data/README.md`](../src/cryptofarm/data/README.md).
Where that domain is unreachable, `scripts/import_candles.py` builds the same store from a local
clone.

## Moving it

`CRYPTOFARM_MARKET_DATA_DIR`. Without the variable, the location stays relative to the repository
root. In a container the image sets it to `/app/market_data`, where `compose.yaml` mounts this folder
from the host.
