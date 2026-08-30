# `cryptofarm/` — il pacchetto

Tre sottopacchetti e un modulo. Le dipendenze vanno in una direzione sola:

```
paths.py   dove stanno dati e modelli   (non dipende da nulla)
data/      lo store locale delle candele
   ↓
ml/  ⇄  trading/
```

| cosa | a cosa serve |
|---|---|
| [`data/`](data/) | scarica e archivia le candele Binance. Prerequisito dell'addestramento |
| [`ml/`](ml/) | dalle candele al modello al segnale: feature, etichette, modelli, valutazione, servizio |
| [`trading/`](trading/) | strategie, conto del profitto, la pagina Streamlit, il bot live |
| `paths.py` | `MODELS_DIR` e `MARKET_DATA_DIR`, con l'override da variabile d'ambiente |

**`ml/` e `trading/` non sono in gerarchia, si tengono per due punti.** `trading/` chiede a `ml/`
il servizio del modello (`signals`, `trainer`, e in `panels` anche `labeling` e `entry_trainer`,
per disegnare bersaglio e pivot accanto alla previsione). Nell'altro verso c'è un solo modulo:
`ml/bar_features.py` importa `trading.indicators_extra` e `trading.mtf`, ed è deliberato — le
feature del modello devono essere **gli stessi** indicatori e **lo stesso** allineamento fra
intervalli che usa la pagina, non una seconda implementazione che può divergere in silenzio.
Sono i due punti da guardare per primi se si pensa di spostare qualcosa fra i due pacchetti.

Entrambi leggono `data/`: `ml/` per i campioni di addestramento, `trading/` per la rotazione
trasversale e per la conversione degli intervalli.

## `paths.py`

Due costanti e nient'altro. Sono relative alla radice del repository, e si spostano con
`CRYPTOFARM_MODELS_DIR` e `CRYPTOFARM_MARKET_DATA_DIR`. L'override non è un lusso: dentro
l'immagine il pacchetto è installato in `site-packages`, quindi la radice dedotta dalla posizione
del file punterebbe **dentro il virtualenv** invece che al volume montato, e i modelli addestrati
in container finirebbero in un layer usa e getta. Chi tocca questo file deve tenerlo funzionante.

## Gli extra di installazione

Il nucleo (`pip install -e .`) basta a feature, etichette, modelli `gbdt` e bot live. `[app]`
aggiunge Streamlit e Plotly, che servono **solo** a `trading/simulator.py` e ai moduli che decora
con `st.cache_data`. `[data]` aggiunge pyarrow, cioè il motore parquet che vogliono `data/klines.py`
e `scripts/analysis.py`. `[dl]` aggiunge TensorFlow, circa 1 GB, e serve solo a
`--model gru|cnn|lstm`. Il caso normale è `pip install -e ".[app,data,dev]"`.
