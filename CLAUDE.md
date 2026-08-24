# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentazione di lavoro

Le decisioni di progetto e lo stato del lavoro stanno in **`.claude/docs/`**:

- `.claude/docs/strategy.md` — fonte di verità delle decisioni su labeling, feature, modello e
  validazione, con le misure che le giustificano. Da aggiornare in luogo quando si decide qualcosa.
- `.claude/docs/HANDOFF.md` — stato corrente del lavoro e trappole ambientali per chi riprende.
- `.claude/docs/INDEX.md` — ordine di lettura consigliato.

Prima di modificare la pipeline ML, leggere `strategy.md`: contiene misure che escludono
esplicitamente diverse strade che sembrano ragionevoli a prima vista.

## Ambiente

Usare **`.venv312/bin/python`**. Il `.venv` preesistente è Python 3.9 senza `scikit-learn`,
mentre il progetto richiede Python >= 3.12.

L'installazione è divisa in extra: `pip install -e ".[app,data,dev]"` è il caso normale. Il nucleo
(`pip install -e .`) basta a feature, etichette, modelli `gbdt` e bot live; `[app]` aggiunge
Streamlit e Plotly (solo `trading/simulator.py` e i moduli che decora con `st.cache_data`);
`[data]` aggiunge pyarrow, cioè il motore parquet che vogliono `data/klines.py` e
`scripts/analysis.py`; `[dl]` aggiunge TensorFlow, circa 1 GB, e serve solo a `--model gru|cnn|lstm`.

`MODELS_DIR` e `MARKET_DATA_DIR` di `paths.py` si spostano con `CRYPTOFARM_MODELS_DIR` e
`CRYPTOFARM_MARKET_DATA_DIR`. Senza le due variabili restano relative alla radice del repo.

## Project overview

CryptoFarm trains a signal model on Binance market data and backtests trading strategies against it.
There are two things that matter — **`trading/simulator.py`** (research) and **`ml/trainer.py`**
(training) — plus their dependencies, plus one live bot. Anything not reachable from those was moved
to `backup/unused/` in 2026-08; see `backup/unused/README.md` for what and why.

```
src/cryptofarm/
├── data/klines.py        store locale delle candele, costruito sui dump bulk di Binance
├── ml/                   pipeline di addestramento (sotto)
└── trading/
    ├── market_data.py    scarico puntuale da Binance per la pagina Streamlit
    ├── indicators.py     indicatori + il nucleo numpy ATR/EMA
    ├── strategies.py     da candele con indicatori a (buy_signals, sell_signals)
    ├── pnl.py            da segnali a operazioni, commissioni incluse
    ├── config.py         valori di partenza dei widget della pagina
    ├── simulator.py      la pagina Streamlit: `trading_analysis` + layout
    └── live_bot.py       bot headless che piazza ordini veri
scripts/analysis.py       misure da riga di comando che producono i numeri di strategy.md
```

### Entry points

```bash
# Simulatore / backtest (strumento di ricerca principale)
streamlit run src/cryptofarm/trading/simulator.py

# Addestramento. Scarica da solo i dati; i parametri sono costanti in cima al file
.venv312/bin/python -m cryptofarm.ml.trainer               # default: gbdt
.venv312/bin/python -m cryptofarm.ml.trainer --model gru   # modello sequenziale
.venv312/bin/python -m cryptofarm.ml.meta_trainer          # meta-labeling
.venv312/bin/python -m cryptofarm.ml.policy_trainer        # politica a tre azioni

# Store delle candele (prerequisito dell'addestramento)
.venv312/bin/python -m cryptofarm.data.klines --update

# Misure di strategy.md
.venv312/bin/python -m scripts.analysis

# Bot live — piazza ordini veri, richiede le variabili d'ambiente (vedi .env.example)
.venv312/bin/python src/cryptofarm/trading/live_bot.py
```

Test: `.venv312/bin/python -m pytest` (188 test in 12 file). Lint/format: `ruff check src tests` e
`black src tests` (config in `pyproject.toml`; `backup/` è escluso da entrambi).

## Il simulatore

`trading/simulator.py` era un file solo da 2028 righe ed è stato spezzato nei moduli sopra. Le
dipendenze formano un DAG: `market_data`, `indicators`, `pnl` e `config` non dipendono da nulla,
`strategies` dipende da `indicators`, `simulator` da tutti. **Non c'è una facciata di
ri-esportazione**: chi serve una strategia la importa dal modulo che la contiene.

- Tutti i DataFrame OHLCV sono indicizzati su `Open time` (`DatetimeIndex`) con colonne
  `Open, High, Low, Close, Volume`.
- Le funzioni in `strategies.py` restituiscono `(buy_signals, sell_signals)`, liste di
  `(timestamp, prezzo)`, che `trading_analysis` passa a `pnl.simulate_trading_with_commisions` o
  `simulate_trading_with_commisions_multiple_buy`.
- Le letture per riga sono in array numpy estratti prima del ciclo, non `df["Col"].iloc[i]`. È da lì
  che viene il grosso della velocità (il simulatore intero: 4295 ms → 125 ms). Mantenere lo stile.
- `indicators._atr_ema` replica in numpy le formule di `ta` 0.11 riga per riga (seme dell'ATR sulla
  media dei primi `window` true range, poi Wilder; EMA come `ewm(span, adjust=False)`).
  **Se si cambia, va riverificato contro `ta`**: è ciò che rende `simulate_candles` 40 volte più
  veloce, e una divergenza silenziosa qui sposta ogni segnale.

### `MACD`: un ramo di dispatch irraggiungibile

`add_technical_indicator` calcola di nuovo `PSAR` (era commentato: le strategie "Close ATR" e
"ATR Live Trade" si rompevano con `KeyError`). Resta commentato il solo `MACD`, letto da
`buy_sell_limits_simulation`, che quindi solleva `KeyError` appena chiamata.

**Nessuna voce del menu la raggiunge**: `trading_analysis` la lega alla stringa `"Buy/Sell Limits"`,
che non è in `config.STRATEGIES`. Lo stesso vale per `"ATR Bands"` e per le varianti `"Dinamic *"`.
Sono rami morti; la funzione resta perché il codice attorno la documenta, ma per renderla usabile
servirebbe ripristinare il blocco `MACD` **e** aggiungere la voce al menu.

### Il golden master

`tests/test_simulator_golden.py` fissa il comportamento di 21 funzioni su quattro scenari di mercato
sintetici, confrontandolo con `tests/data/simulator_golden.json`. Serve perché il simulatore non ha
altri test: **prima di toccarlo, questo deve passare; dopo, deve passare ancora senza rigenerarlo**.

Rigenerare (`SIMULATOR_GOLDEN_REGEN=1 pytest tests/test_simulator_golden.py`) **accetta qualunque
differenza di comportamento**. Farlo solo dopo aver verificato a mano che la differenza sia voluta, e
controllare che il diff del JSON contenga solo le righe attese.

Gli scenari non sono intercambiabili: `close_ema_crossover_simulation` pretende tre incroci EMA in
sequenza e scatta solo su un'inversione vera (`regimi`, `sbandate`), `close_bullish_ema_simulation`
solo in laterale. Togliere uno scenario scopre delle strategie.

## La pipeline ML

`ml/trainer.py` non contiene logica propria: assembla i pezzi e tiene la configurazione. Le feature
stanno in `features.py`, le etichette in `labeling.py` e `directional_change.py`, la matrice in
`dataset.py`, i modelli in `models.py`, le metriche in `evaluate.py`, la validazione in
`validation.py`, l'esecuzione simulata in `execution.py`. `meta.py` + `meta_trainer.py` fanno il
meta-labeling; `policy.py` + `dagger.py` + `policy_trainer.py` la politica a tre azioni.

Il modello di default è **`gbdt`** (`HistGradientBoostingClassifier`), non più un LSTM; `models.py`
tiene ancora `gru`/`cnn`/`lstm` dietro `--model`. Prerequisito dell'addestramento è lo store di
candele (`data/klines.py`), non un download al volo.

### Quale modello usa il simulatore

`ml/trainer.MODEL_PRECEDENCE` è `("policy_model", "meta_model", "signal_model")` e
`active_model_name()` è l'unica fonte di verità: `load_signal_model` carica quel modello e
`ai_model_simulation` sceglie la strategia in base a quel nome, quindi i due non possono divergere.
Per tornare al modello precedente basta spostare altrove l'artefatto di quello più recente.

`meta_parameters()` legge barriere, soglia CUSUM e parametri di esecuzione **dai metadata
dell'artefatto**, non da costanti: devono essere esattamente quelli con cui il modello è stato
addestrato.

## Data/model artifacts

`models/` contiene gli artefatti (`.joblib` + `.json` di metadata). `models/.gitignore` copre solo
`*.keras`, per cui i `.joblib` sono finiti tracciati nel repository — circa 24 MB, con
`policy_alta` e `policy_model` byte per byte identici. **Da sistemare**: estendere il `.gitignore` e
fare `git rm --cached`. Rigenerare con i trainer, non modificare a mano.

## Docker e CI

Un solo `Dockerfile` con quattro target: **`runtime`** (uso locale completo: simulatore, trainer,
store delle candele, `scripts.analysis`), **`dev`** (`runtime` + pytest/ruff/black, è l'immagine con
cui gira la CI), **`dl`** (`runtime` + TensorFlow, per i modelli sequenziali) e **`web`** (la sola
pagina, senza pyarrow: è quella che va in produzione).

**`web` è l'ultimo stage del file, e deve restarci**: una build senza `--target` prende l'ultimo
stage, e Render non ha un campo per sceglierlo. Spostarlo significa spedire in produzione
l'immagine con TensorFlow. La CI costruisce anche senza `--target` proprio per accorgersene.

```bash
mkdir -p models market_data                     # solo la prima volta: i bind mount devono esistere
docker compose up simulator                     # http://localhost:8501
docker compose --profile data  run --rm klines
docker compose --profile train run --rm trainer
docker compose --profile ci    run --rm tests
```

Dentro l'immagine il pacchetto sta in `site-packages`, non in editable: la radice che `paths.py`
dedurrebbe dalla posizione del file punterebbe dentro il virtualenv, quindi l'immagine imposta
`CRYPTOFARM_MODELS_DIR=/app/models` e `CRYPTOFARM_MARKET_DATA_DIR=/app/market_data`, che è dove
`compose.yaml` monta `./models` e `./market_data` dell'host. Chi tocca `paths.py` deve tenere
funzionante l'override, altrimenti i modelli addestrati in container finiscono in un layer usa e
getta.

Il deploy pubblico sta in `render.yaml` (piano gratuito, regione `frankfurt`). Tre vincoli che
non si vedono dal codice: il servizio deve legarsi a **`$PORT`** su `0.0.0.0` (il comando
dell'immagine usa `${PORT:-8501}`); Binance blocca gli IP statunitensi su `api.binance.com`, da cui
il simulatore prende le candele, quindi la regione non è un dettaglio; il piano gratuito non ha
dischi persistenti, e con `models/*.joblib` gitignorato online girano le strategie classiche mentre
la voce "AI Model" segnala l'artefatto mancante.

I quattro `@st.cache_data` di `trading/` hanno `ttl`/`max_entries` per una ragione operativa: i
parametri arrivano dai widget, quindi la cardinalità la decide chi muove gli slider, e senza tetto
un'istanza da 512 MB finisce in OOM mentre la si usa. Non toglierli.

`live_bot.py` **non** è un servizio di compose, di proposito: fa partire il ciclo `while True`
all'import, senza `main()` e senza gestione dei segnali, quindi un container che si riavvia da solo
lo rimetterebbe a piazzare ordini senza controllo. Prima serve quel refactor.

La CI (`.github/workflows/ci.yml`) gira su ogni pull request e sui push a `main`, in due job. Il
primo installa `.[app,data,dev]` su Python 3.12 e passa `ruff check`, `black --check` e `pytest` su
`src`, `tests` e `scripts`. Il secondo costruisce le immagini e verifica quattro cose che dal
sorgente non si vedono: che il pacchetto si importi e risolva le directory dei dati a `/app/...`,
che i test passino dentro l'immagine, che la build **senza `--target`** non porti pyarrow (cioè che
`web` sia ancora l'ultimo stage), e che il container si leghi davvero a `$PORT` — lo avvia con
`PORT=10000` e interroga `/_stcore/health`.

Nessuna immagine viene pubblicata su un registry: Render costruisce il Dockerfile da sé a ogni
push su `main`.

## Configuration

Le credenziali Binance e i parametri del bot passano da variabili d'ambiente — vedi `.env.example`.
Nulla nel repo carica `.env` da solo (non c'è `python-dotenv`): esportarle nella shell o nella
configurazione di esecuzione dell'IDE.

- `API_KEY`, `API_SECRET` — solo `trading/live_bot.py`.
- `live_bot.py` legge anche `ASSET`, `CURRENCY`, `CANDLES_TIME`, `SMA_WINDOW`, `ATR_WINDOW`,
  `ATR_MULTIPLIER`, `RSI_WINDOW`, `RSI_BUY_LIMIT`, `RSI_SELL_LIMIT`, `NUM_CONDITIONS`.
- `MARKET_DATA_CSV` — percorso del CSV storico nella pagina Streamlit (`trading/config.py`).
- Il simulatore e i trainer usano gli endpoint pubblici di Binance e non hanno bisogno di credenziali.

`.streamlit/config.toml` imposta il tema scuro.

## Archived

- `backup/unused/` — moduli rimossi da `src/` perché nessuno li importava (dashboard live, bot a due
  account, grid search, visualizzatore dei risultati, dashboard di analisi). `git mv` li rimette a
  posto con la storia intatta.
- `backup/v2/` — simulatore multi-timeframe, precedente riscrittura. Materiale di riferimento in sola
  lettura, escluso da lint e format.
