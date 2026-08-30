# `market_data/` — lo store locale

**Non tracciato.** Circa 4,2 GB a store pieno, 15 simboli dal 2017. Un clone del repository trova
questa cartella vuota (a parte questo file), e in produzione — dove Render non ha dischi
persistenti — resta vuota per sempre: è la condizione per cui la vista *Cross-asset rotation* dice
che non ha dati invece di provare quindici scarichi.

## Cosa ci finisce dentro

| forma | prodotto da | cosa contiene |
|---|---|---|
| `<SIMBOLO>-5m.parquet` | `python -m cryptofarm.data.klines --update` | le candele OHLCV. **Solo 5m**: 15m/30m/1h si derivano aggregando, ed è esatto |
| `<SIMBOLO>-positioning.parquet` | `python -m cryptofarm.data.positioning --update` | long/short ratio, open interest, funding, base. ~400 MB |
| `*.pkl` | i banchi di `scripts/` | cache delle previsioni di un modello, chiavata sulla firma `created` dell'artefatto |

I `.pkl` sono **cache, non dati**: `rl_stati.pkl` da solo pesa 3,5 GB e si ricostruisce
rilanciando `scripts/rl_lab.py`. Cancellarli costa tempo di CPU, non informazione. I `.parquet`
delle candele invece costano ore di rete: quelli si tengono.

## Ricostruirlo

```bash
.venv312/bin/python -m cryptofarm.data.klines --update        # sotto i 10 minuti con 32 worker
.venv312/bin/python -m cryptofarm.data.positioning --update   # ~400 MB
```

I dati arrivano dai dump mensili di `data.binance.vision`, non dalla REST API — la ragione, che è
un fattore ottanta sul tempo, sta in [`../src/cryptofarm/data/README.md`](../src/cryptofarm/data/README.md).
Dove quel dominio non è raggiungibile, `scripts/import_candles.py` costruisce lo stesso store da un
clone locale.

## Spostarlo

`CRYPTOFARM_MARKET_DATA_DIR`. Senza la variabile, la posizione resta relativa alla radice del
repository. In container l'immagine la imposta a `/app/market_data`, dove `compose.yaml` monta
questa cartella dell'host.
