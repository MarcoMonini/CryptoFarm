# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentazione di lavoro

Le decisioni di progetto e lo stato del lavoro stanno in **`.claude/docs/`**:

- `.claude/docs/strategy.md` — fonte di verità delle decisioni su labeling, feature, modello e
  validazione, con le misure che le giustificano. Da aggiornare in luogo quando si decide qualcosa.
- `.claude/docs/HANDOFF.md` — stato corrente del lavoro e trappole ambientali per chi riprende.
- `.claude/docs/backtest-strategie.md` — le strategie a indicatori misurate su nove anni: 3.129
  configurazioni, sensibilità ai parametri, tenuta fuori campione, difetti trovati misurando.
- `.claude/docs/strategie-nuove.md` — il seguito: le quattro correzioni applicate, il ciclo
  2021-2026 come dataset, cinque strategie nuove e il motore che sa stare anche corto.
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
    ├── indicators_extra.py  ADX, Donchian, Bollinger/Keltner, StochRSI, OBV/MFI, Ichimoku
    ├── panels.py         il registro: quale strategia usa quali indicatori e quali parametri
    ├── strategies.py     da candele con indicatori a (buy_signals, sell_signals)
    ├── strategies_ls.py  strategie a due versi: da candele a cambi di posizione (+1/0/-1)
    ├── pnl.py            da segnali a operazioni: `simulate_trading_with_commisions` (solo long)
    │                     e `simulate_positions` (long/short, con leva e costo di mantenimento)
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

# Backtest delle strategie a indicatori su tutto lo storico (vedi .claude/docs/backtest-strategie.md)
.venv312/bin/python -m scripts.strategy_sweep --all --interval 15m   # griglie di parametri
.venv312/bin/python -m scripts.sweep_report --interval 15m           # tabelle in reports/
.venv312/bin/python -m scripts.strategy_focus --top 3                # commissioni e intervalli

# Strategie a due versi, long e short (vedi .claude/docs/strategie-nuove.md)
.venv312/bin/python -m scripts.strategy_lab --all --interval 1d --since 2021-01-01
.venv312/bin/python -m scripts.lab_report --symbol BTCUSD --interval 1d

# Store delle candele da fonte alternativa, dove data.binance.vision non è raggiungibile
.venv312/bin/python -m scripts.import_candles --source /percorso/al/clone

# Bot live — piazza ordini veri, richiede le variabili d'ambiente (vedi .env.example)
.venv312/bin/python src/cryptofarm/trading/live_bot.py
```

Test: `.venv312/bin/python -m pytest` (430 test in 15 file). Lint/format: `ruff check src tests` e
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
  `simulate_trading_with_commisions_multiple_buy`. Quelle in `strategies_ls.py` restituiscono invece
  cambi di posizione `(timestamp, prezzo, +1|0|-1)` per `pnl.simulate_positions`: è il formato che
  serve a rappresentare l'inversione diretta e la vendita allo scoperto.
- Le letture per riga sono in array numpy estratti prima del ciclo, non `df["Col"].iloc[i]`. È da lì
  che viene il grosso della velocità (il simulatore intero: 4295 ms → 125 ms). Mantenere lo stile.
- `indicators._atr_ema` replica in numpy le formule di `ta` 0.11 riga per riga (seme dell'ATR sulla
  media dei primi `window` true range, poi Wilder; EMA come `ewm(span, adjust=False)`).
  **Se si cambia, va riverificato contro `ta`**: è ciò che rende `simulate_candles` 40 volte più
  veloce, e una divergenza silenziosa qui sposta ogni segnale.

### Il registro di `panels.py`

La pagina non decide piu' da sola cosa mostrare. `trading/panels.py` tiene, in forma di dati, quali
indicatori usa ogni strategia, quali parametri servono a ognuno e come si disegnano; `simulator.py`
lo legge e dispone widget e tracce. Aggiungere una strategia vuol dire aggiungere una riga li' e la
voce in `config.STRATEGIES` — un test verifica che le due liste coincidano.

Tre cose da sapere prima di toccarlo:

- **La mappa e' verificata a mano.** Uno scan statico delle colonne lette non basta:
  `close_bullish_ema_simulation` prende le medie con `(df[c].to_numpy() for c in (...))`, uno slice
  variabile che l'analisi dell'albero sintattico non vede.
- **Le dipendenze contano piu' dei nomi.** `Upper_Band`/`Lower_Band` sono `KAMA ± moltiplicatore ×
  ATR` e `KAMA` usa `ema_window`: una strategia a bande dipende da "EMA Short" anche se di medie non
  ne disegna nessuna.
- **I colori sono tre**, blu/arancio/acquamarina: le uniche che passano tutte le coppie del
  validatore su superficie scura. Il quarto slot contro l'arancio scende a 4,8 di ΔE per
  deuteranopia. L'acquamarina non si usa sopra le candele, dove si confonde con il corpo rialzista.
  Verde e rosso restano allo stato. Tre test tengono ferme queste regole.

### Funzioni di `strategies.py` che il menu non raggiunge

`buy_sell_limits_simulation` legge `MACD`, che resta commentata in `add_technical_indicator`, e
quindi solleva `KeyError` appena chiamata. `close_rsi_buy_sell_limits_simulation` e'
irraggiungibile per scelta: misurata su nove anni, e' in perdita totale in tutte le 25
configurazioni provate. Nessuna delle due sta nel registro, quindi non compare nel menu; restano
nel modulo perche' il golden master le copre e il codice attorno le documenta.

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

Un solo `Dockerfile` con quattro target: **`runtime`** (simulatore, trainer, store delle candele,
`scripts.analysis`), **`dev`** (`runtime` + pytest/ruff/black, è l'immagine con cui gira la CI),
**`dl`** (`runtime` + TensorFlow, per i modelli sequenziali) e **`web`**, che è quello che va in
produzione ed è identico a `runtime`.

**`web` è l'ultimo stage del file, e deve restarci**: una build senza `--target` prende l'ultimo
stage, e Render non ha un campo per sceglierlo. Spostarlo significa spedire in produzione
l'immagine con TensorFlow. La CI costruisce anche senza `--target` proprio per accorgersene. Uno
stage nuovo va aggiunto sopra `web`, mai sotto.

Un'immagine più magra per la sola pagina non è ottenibile togliendo pyarrow: `streamlit` dipende da
`pyarrow>=7.0`, quindi i 141 MB del motore parquet ci sono comunque.

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
che i test passino dentro l'immagine, che la build **senza `--target`** non porti TensorFlow (cioè
che `web` sia ancora l'ultimo stage), e che il container si leghi davvero a `$PORT` — lo avvia con
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

### Plugin di Claude Code

`.claude/settings.json` è tracciato e dichiara tre marketplace con i plugin abilitati per il
progetto: `ponytail`, `agent-skills` (raccolte di skill generaliste) e tre plugin di
`anthropics/financial-services` — `financial-analysis`, `equity-research`, `market-researcher` —
scelti perché il lavoro qui è di analisi finanziaria quantitativa.

Ogni marketplace è **agganciato a un commit** (`ref`, SHA a 40 caratteri): è l'unico modo di fissare
le versioni dei plugin, perché `enabledPlugins` accetta solo un booleano e la versione la dichiara
il manifesto del marketplace. Al momento dell'aggancio: ponytail 4.9.0, agent-skills 0.6.7,
financial-analysis 0.1.1, equity-research 0.1.2, market-researcher 0.1.1. Per aggiornarli si sposta
il `ref` su un commit più recente, deliberatamente — non succede da solo.

Le skill dei plugin sono disponibili dalla sessione successiva all'installazione, non da quella in
cui si modifica il file.

## Archived

- `backup/unused/` — moduli rimossi da `src/` perché nessuno li importava (dashboard live, bot a due
  account, grid search, visualizzatore dei risultati, dashboard di analisi). `git mv` li rimette a
  posto con la storia intatta.
- `backup/v2/` — simulatore multi-timeframe, precedente riscrittura. Materiale di riferimento in sola
  lettura, escluso da lint e format.
