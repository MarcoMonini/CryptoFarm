# `data/` — lo store locale delle candele

Due store, stessa infrastruttura: dump bulk da `data.binance.vision`, archivio parquet in
`market_data/`, aggiornamento incrementale. È il **prerequisito dell'addestramento**: i trainer
leggono da qui, non scaricano al volo.

| file | righe | cosa archivia |
|---|---|---|
| `klines.py` | 431 | le candele OHLCV, **un solo intervallo per simbolo** (5m), da cui 15m/30m/1h si derivano aggregando |
| `positioning.py` | 289 | il posizionamento sui futures: long/short ratio, open interest, funding, base |

## Le funzioni

**`klines.py`** — `update_store` e `update_symbol` per costruire e aggiornare, `load_klines` e
`resample_klines` per leggere, `store_path` e `store_manifest` per sapere cosa c'è,
`interval_to_minutes` per la conversione, `clip_wicks` e `wick_outliers` per le candele con ombre
implausibili. `main` è il punto d'ingresso di `python -m cryptofarm.data.klines --update`.

**`positioning.py`** — la stessa forma: `update_store`, `update_symbol`, `load_positioning`,
`store_path`, `store_manifest`, `main`, più i due scaricatori `fetch_metrics_day` e
`fetch_funding_month`, che hanno cadenze diverse (giornaliera e mensile) perché Binance le
pubblica così.

## Perché un intervallo solo, e perché i dump e non la REST

**Un intervallo solo** perché l'aggregazione da 5m è esatta — verificata contro i dump ufficiali,
differenza nulla su OHLC. Una sola fonte di verità invece di quattro archivi da tenere allineati,
e un quarto delle richieste di rete.

**I dump e non la REST** perché su questa scala il costo è interamente latenza per richiesta
(~2,7 s misurati, identici sulle due strade) e non banda. La REST dà al massimo 1000 candele per
chiamata ed è soggetta a rate limit: ~18.700 richieste sequenziali per lo storico, circa 14 ore.
Un dump mensile ne contiene ~8.900 ed è un file statico su CDN senza rate limit, quindi
parallelizzabile: ~1.350 file con 32 worker, sotto i dieci minuti. La REST resta usata **solo per
la coda**, cioè le candele di oggi, che nei dump non ci sono ancora.

## Dove finiscono i dati

In `market_data/`, che è gitignorato e pesa circa 4 GB a store pieno. La posizione si sposta con
`CRYPTOFARM_MARKET_DATA_DIR` (vedi [`../paths.py`](../paths.py)); in container l'immagine la
imposta a `/app/market_data`, dove `compose.yaml` monta la cartella dell'host.

Dove `data.binance.vision` non è raggiungibile, `scripts/import_candles.py` costruisce lo stesso
store da un clone locale di un dataset a un minuto.
