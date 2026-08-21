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

Test: `.venv312/bin/python -m pytest` (185 test in 12 file). Lint/format: `ruff check src tests` e
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
