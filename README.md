# CryptoFarm

Addestra modelli di segnale su dati Binance e li verifica contro strategie a indicatori, su nove
anni di storico e quindici asset. C'è anche un bot headless che può operare dal vivo.

**Il risultato in una riga.** Quasi tutto ciò che è stato provato non batte il possesso passivo, ed
è scritto dove è stato misurato. L'unica cosa che passa il controllo a esposizione appaiata è il
modello d'ingresso: **+2,071% netti per operazione fuori campione, 14 simboli su 15 in utile**, e
il suo vantaggio non è la previsione ma la **selettività** — marca una barra su duecento.

## Come è fatto

```
src/cryptofarm/
├── data/      lo store locale delle candele (dump bulk, parquet)
├── ml/        feature → etichette → modello → valutazione → servizio
└── trading/   strategie, conto del profitto, la pagina Streamlit, il bot live
scripts/       diciotto banchi di misura: producono i numeri dei documenti
tests/         1.022 casi, nessuna rete, nessuno store richiesto
.claude/docs/  le decisioni e le misure che le giustificano
```

Ogni cartella ha il suo README con l'elenco dei file e delle funzioni:
[`src/cryptofarm/data/`](src/cryptofarm/data/) ·
[`src/cryptofarm/ml/`](src/cryptofarm/ml/) ·
[`src/cryptofarm/trading/`](src/cryptofarm/trading/) ·
[`scripts/`](scripts/) · [`tests/`](tests/) · [`models/`](models/) · [`reports/`](reports/)

Due cose contano davvero: **`trading/simulator.py`** (la ricerca) e **`ml/trainer.py`**
(l'addestramento), più le loro dipendenze e un bot.

## Far partire

Python >= 3.12, ambiente **`.venv312`** — il `.venv` preesistente è 3.9 e non ha `scikit-learn`.

```bash
pip install -e ".[app,data,dev]"

# La pagina: due viste, "quando stare dentro" e "quale asset tenere"
streamlit run src/cryptofarm/trading/simulator.py

# Lo store delle candele, prerequisito di ogni addestramento (~10 minuti)
.venv312/bin/python -m cryptofarm.data.klines --update

# Il modello in testa oggi: il veloce opera, il lento gli fa da cancello
.venv312/bin/python -m cryptofarm.ml.entry_trainer --selfcheck   # gira senza store
.venv312/bin/python -m cryptofarm.ml.entry_trainer               # il lento (H=150)
.venv312/bin/python -m cryptofarm.ml.entry_trainer --h 20 --quantile 0.995 --nome entry_model_veloce
.venv312/bin/python -m scripts.entry_lab                         # quanto vale il cancello

# Il bot live — piazza ordini veri, vuole le variabili di .env.example
.venv312/bin/python src/cryptofarm/trading/live_bot.py
```

Test: `.venv312/bin/python -m pytest`. Lint: `ruff check src scripts tests` e
`black src scripts tests`.

Gli altri comandi — gli altri modelli, gli sweep delle strategie, i banchi di misura — stanno in
[`CLAUDE.md`](CLAUDE.md) e nei README di [`scripts/`](scripts/) e [`src/cryptofarm/ml/`](src/cryptofarm/ml/).

## Gli extra di installazione

| extra | cosa contiene | serve a |
|---|---|---|
| (nucleo) | numpy, pandas, scipy, ta, requests, python-binance, scikit-learn | feature, etichette, modelli `gbdt`, bot live |
| `app` | streamlit, plotly | solo `trading/simulator.py` e i moduli che decora con `st.cache_data` |
| `data` | pyarrow (141 MB) | il motore parquet dello store: `data/klines.py`, `scripts/analysis.py` |
| `dl` | tensorflow (~1 GB) | solo `--model gru\|cnn\|lstm` |
| `dev` | pytest, ruff, black, pre-commit | |

Un'immagine più magra per la sola pagina non si ottiene togliendo pyarrow: `streamlit` dipende da
`pyarrow>=7.0`, quindi i 141 MB ci sono comunque.

## Docker e deploy

```bash
mkdir -p models market_data                      # i bind mount devono esistere
docker compose up simulator                      # http://localhost:8501
docker compose --profile data  run --rm klines
docker compose --profile train run --rm trainer
docker compose --profile ci    run --rm tests
```

Un solo `Dockerfile`, quattro target: `runtime` (pagina, trainer, store), `dev` (`runtime` +
pytest/ruff/black, l'immagine della CI), `dl` (`runtime` + TensorFlow) e **`web`**, che è quello
che va in produzione. **`web` è l'ultimo stage e deve restarci**: una build senza `--target` prende
l'ultimo, e Render non ha un campo per sceglierlo. Uno stage nuovo va aggiunto sopra, mai sotto —
e la CI costruisce anche senza `--target` proprio per accorgersene.

Il deploy pubblico sta in [`render.yaml`](render.yaml), piano gratuito, regione **`frankfurt`**:
Binance blocca gli IP statunitensi su `api.binance.com`, che è da dove il simulatore prende le
candele, quindi la regione non è un dettaglio. Il piano gratuito non ha dischi persistenti, e
`models/*.joblib` è gitignorato: **online girano le strategie classiche**, e la pagina toglie da sé
la voce «AI Model» dal menu invece di cadere.

## Cosa sapere prima di modificare

**`.claude/docs/` contiene misure che escludono esplicitamente diverse strade che sembrano
ragionevoli a prima vista.** L'ordine di lettura sta in [`.claude/docs/README.md`](.claude/docs/README.md);
chi riprende a freddo legge [`HANDOFF.md`](.claude/docs/HANDOFF.md) e basta. Chi tocca la pipeline
ML legge prima [`strategy.md`](.claude/docs/strategy.md).

**Il golden master va rispettato.** `tests/test_simulator_golden.py` fissa il comportamento di 21
funzioni: deve passare prima di una modifica a `trading/` e passare ancora dopo, **senza essere
rigenerato**. Rigenerarlo accetta qualunque differenza, anche una regressione.

**I valori di partenza sono centrali, non ottimi.** Cercare il massimo in campione trasferisce
peggio che prendere una configurazione a caso: sulla rotazione la correlazione fra resa in stima e
resa in verifica è **−0,69**. Chi cambia i default in «quelli che rendono di più nel grafico» sta
facendo esattamente l'errore misurato.

**Le letture per riga passano da array numpy** estratti prima del ciclo, non da
`df["Col"].iloc[i]`: da lì viene il grosso della velocità (il simulatore intero: 4295 ms → 125 ms).
E `indicators._atr_ema` replica in numpy le formule di `ta` 0.11 riga per riga — se si cambia, va
riverificato contro `ta`, perché una divergenza silenziosa lì sposta ogni segnale.

## Un difetto noto

`buy_sell_limits_simulation` legge la colonna `MACD`, che resta commentata in
`add_technical_indicator`: solleva `KeyError` appena chiamata. Nessuna voce del menu la raggiunge.
Renderla usabile vuol dire ripristinare il blocco `MACD` **e** aggiungere la voce.

## Configurazione

Solo variabili d'ambiente, vedi [`.env.example`](.env.example) — niente nel repository carica
`.env` da solo. `API_KEY`/`API_SECRET` e i parametri della strategia li legge il solo
`live_bot.py`; `MARKET_DATA_CSV` è il CSV storico della pagina. Il simulatore e i trainer usano gli
endpoint pubblici di Binance e non vogliono credenziali.

## Lingua

I documenti di questo progetto sono in italiano, il codice e i nomi delle funzioni in inglese dove
sono di dominio (`simulate_positions`, `swing_target`) e in italiano dove descrivono una decisione
presa qui (`perche_non_entra`, `scala_fuori_misura`, `votanti_predefiniti`).
