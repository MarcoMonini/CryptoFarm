# Handoff — CryptoFarm

Data: **2026-08-27**. Branch di lavoro: **`claude/ricerca-quant-ml-cinque-asset`**
(spinto). Il precedente
`claude/trading-strategies-performance-fb39oc` e' stato unito in `main` con la PR #7.
Il branch precedente `ai-labeling-rewrite` (pipeline ML a 3 stati) e' **chiuso con esito negativo**
e non e' mai stato unito: vedi `.claude/docs/strategy.md` §10-13 e la sezione "Il filone ML" qui
sotto.

## Non duplicare: leggi prima questi

| documento | cosa contiene |
|---|---|
| `CLAUDE.md` | architettura del repo, comandi, variabili d'ambiente, vincoli Docker/Render |
| `.claude/docs/backtest-strategie.md` | **le strategie del simulatore misurate su nove anni.** 3.129 configurazioni, sensibilita' ai parametri, tenuta fuori campione, quattro difetti del codice trovati misurando (§8, ora corretti) |
| `.claude/docs/strategie-nuove.md` | **lo stato piu' recente del filone trading.** Le quattro correzioni e cosa hanno cambiato, il ciclo 2021-2026 come dataset e il perche', cinque strategie nuove, il motore a due versi con lo short, leva e costi |
| `.claude/docs/sessione-2026-08-27.md` | **chiusura dell'ultima sessione**: le due decisioni prese con l'utente, le trappole d'ambiente scoperte misurando, i due test che passavano a vuoto e come sono stati corretti, e cosa farei dopo in ordine |
| `.claude/docs/strategia-confluenza.md` | **il filone piu' recente (2026-08-27): la strategia multi-timeframe a piu' segnali.** Disegno, sei votanti, memoria del segnale, soglia decisa dai piani alti, paniere a capitale condiviso, e le tre cose che scrivendola si sono rivelate diverse dal disegno. **Il codice c'e' e gira; la misura su dati veri no** |
| `.claude/docs/ricerca-quant-ml.md` | **il documento piu' recente (2026-08-26), e quello da leggere per primo sui risultati.** Stato dell'arte dai nove repository, i due filoni misurati su BTC/ETH/SOL/XRP/BNB, la rotazione trasversale, il filtro meta. Corregge due conclusioni di `strategie-nuove.md` che non generalizzano |
| `.claude/docs/strategy.md` | fonte di verita' delle decisioni sul **filone ML** (etichettatura, feature, modello, validazione). Chiuso in negativo, ma le trappole valgono ancora |
| `git log main..HEAD` | i messaggi di commit spiegano il *perche'* di ogni scelta e i bug trovati |

Non riassumere quei contenuti: sono gia' scritti e aggiornati.

---

## Stato del lavoro corrente: il filone trading

Tre sessioni consecutive. **La terza (2026-08-26/27) e' quella piu' recente e in parte corregge le
prime due**: sta in [`sessione-2026-08-27.md`](sessione-2026-08-27.md) per lo stato operativo e in
[`ricerca-quant-ml.md`](ricerca-quant-ml.md) per le misure. Le due sezioni qui sotto restano perche'
descrivono come ci si e' arrivati, non perche' siano l'ultima parola.

### Sessione 1 — misurare le strategie esistenti (`d82b3db`, `8f4ccd8`)

3.129 configurazioni delle 12 strategie del simulatore, su BTC 2017-2026 a 15m, piu' controlli su
ETH e su altri intervalli. Risultato: **a 15 minuti quasi tutto perde**, e la frequenza operativa
spiega quasi tutto (le strategie che fanno migliaia di operazioni l'anno pagano in commissioni piu'
del margine per operazione). Fuori campione crolla anche cio' che in campione sembrava solido.
Script conservati: `scripts/strategy_sweep.py`, `scripts/sweep_report.py`, `scripts/strategy_focus.py`.
Tabelle: `reports/*_15m*.csv`.

### Sessione 2 — correzioni, dataset nuovo, strategie nuove, short (`61603cc`)

**Le quattro correzioni** chieste dall'utente, tutte applicate e misurate:

| difetto | correzione | effetto |
|---|---|---|
| voce di menu `"Supetrend"` ≠ dispatch `"Supertrend"` | stringa corretta in `config.STRATEGIES` | la voce esegue: +450% a 4h nella migliore configurazione |
| `"ATR Bands"` aveva il ramo di dispatch ma nessuna voce di menu | voce aggiunta | selezionabile: +678% a 4h |
| stop loss di `buy_sell_limits_close_simulation` commentato | ripristinato | inerte al default 99%, attivo ai valori operativi |
| `EMA200` era l'EMA **dell'apertura** sulla finestra corta, e Trend Zones la confrontava con `EMA20` (una media con se stessa) | colonna eliminata, le tre funzioni leggono `EMA100` | Trend Zones 4h da **−21,9% a +309,3%**, da 202 a 10,6 operazioni/anno |

Il golden master e' stato rigenerato **una volta sola** e il diff verificato voce per voce: 17 voci,
tutte di `add_technical_indicator` e delle tre funzioni che leggevano `EMA200`, sui quattro scenari.

**Il dataset e' cambiato**: non piu' 2017-2026 ma **2021-01-01 → oggi**, perche' i due cicli sono
mercati diversi (BTC 2017-2020: +2.803%, CAGR 132%, Sharpe 1,44 — 2021-2026: +166%, CAGR 19%,
Sharpe 0,59) e **i parametri non passano dall'uno all'altro**: scegliendo sul ciclo vecchio e
misurando sul nuovo, quattro strategie su cinque vanno in perdita.

**Cinque strategie nuove** in `trading/strategies_ls.py`, su sette indicatori mai usati prima
(ADX, Donchian, Bollinger, Keltner, StochRSI, OBV/MFI, Ichimoku): `donchian_breakout`,
`squeeze_breakout`, `trend_pullback`, `ichimoku_trend`, `band_reversion_gated`.

**Il verso corto e' simulabile**: `pnl.simulate_positions` prende cambi di posizione
`(tempo, prezzo, +1|0|−1)` — il formato a due liste di segnali non sa esprimere l'inversione
diretta — con commissione su entrambe le gambe, costo di mantenimento giornaliero (funding, 0,03%),
leva e liquidazione a capitale zero.

### I risultati, senza addolcirli

- **Le storiche corrette battono le nuove in campione** (Close ATR +575% a 1d contro +120% della
  migliore nuova), ma su griglie 100 volte piu' grandi: la colonna onesta e' la mediana.
- **Fuori campione (scelta 2021-2023, resa 2024-2026) nessuna strategia, di nessuna famiglia, batte
  il possesso passivo.** La sola nuova con segno positivo e' `band_reversion_gated` (+11,1%).
- **Il vantaggio reale e' sul rischio, non sul rendimento**: `band_reversion_gated` fa +84% con 22%
  di drawdown contro +166% con 76,5%; **a leva 2 diventa +196% con DD 41%**, cioe' batte il possesso
  passivo su entrambi gli assi (in campione).
- **Lo short toglie invece di aggiungere**: la mediana peggiora in tutte e cinque le strategie
  (donchian −22% → −57%; ichimoku +15% → −25%), migliora solo nel 2,6-23,6% delle coppie, e paga
  solo nel 2022. L'unica eccezione e' il ritorno alla media, dove il corto ha win rate 52,3%.
- **Le ablazioni** dicono che ogni filtro nuovo migliora la mediana e riduce le operazioni; l'unico
  ininfluente e' l'ADX come soglia minima nella rottura di canale.

### Cosa resta aperto

> **Aggiornamento 2026-08-26 — i due punti qui sotto sono chiusi.** Sulla macchina dell'utente lo
> store ha tutti e 15 i simboli fino al 2026-08-19: SOL e BNB sono stati misurati e
> `donchian_breakout`/`squeeze_breakout` rimisurate dopo la correzione dello stop a trailing. I
> risultati stanno in `.claude/docs/ricerca-quant-ml.md` §2, e **due conclusioni di
> `strategie-nuove.md` non reggono su cinque asset**: `band_reversion_gated` e' negativa su 4 asset
> su 5, e fuori campione 4h batte 1d. Restano validi i comandi qui sotto, con una correzione
> necessaria a `strategy_lab`/`strategy_sweep` per il `spawn` di macOS (vedi §8 di quel documento).

**SOL e BNB non sono stati misurati.** L'utente li aveva chiesti esplicitamente. Non e' una scelta:
in ambiente remoto l'egress verso *ogni* exchange e aggregatore risponde 403 sul CONNECT (Binance,
Bybit, Kraken, Coinbase, Kucoin, MEXC, Gate, CoinGecko, CryptoCompare, Messari, Yahoo, Kaggle,
HuggingFace), e nessun repository pubblico raggiungibile ha candele intraday recenti di quei due
asset. Il codice li supporta gia'. **In locale bastano:**

```bash
python -m cryptofarm.data.klines --update --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT
for s in BTCUSDT ETHUSDT SOLUSDT BNBUSDT; do
  python -m scripts.strategy_lab --all --symbol $s --interval 1d --since 2021-01-01
  python -m scripts.lab_report --symbol $s --interval 1d
done
```

Le conclusioni su regimi e verso corto valgono per **un asset e un ciclo** finche' quello non gira.

**Rimisurare `donchian_breakout` e `squeeze_breakout`.** Lo stop a trailing e' stato corretto
(vedi le trappole sotto e `.claude/docs/strategie-nuove.md` §8): le loro righe in §6 e i
`reports/lab_*.csv` sono precedenti alla correzione. Servono le stesse candele del punto sopra,
quindi le due cose si fanno insieme. Le altre tre strategie non sono toccate.

### La pagina, rifatta attorno al registro

Il simulatore mostrava sempre tutto: quindici parametri nella barra laterale e una dozzina di
tracce, uguali per ogni strategia. Ora `trading/panels.py` tiene la mappa strategia → indicatori →
parametri → tracce, e la pagina la legge: si vede solo cio' che la strategia scelta usa davvero, e
tutto solo quando non ne e' selezionata nessuna. `trading_analysis` e' passata da trenta argomenti
nominali a un dizionario, la catena di `if strategia == ...` non c'e' piu', e `simulator.py` da 672
a 402 righe.

Le cinque strategie nuove sono nel menu, **sempre e solo lunghe**: il verso corto e' misurato in
perdita. Passano dal motore classico tramite un adattatore da cambi di posizione a due liste,
esatto senza il verso corto e verificato contro `simulate_positions`. **I loro numeri nella pagina
sono piu' ottimisti di `reports/lab_*.csv`**, perche' quel motore non addebita il funding e non
conosce la leva: la pagina lo dice accanto al risultato.

Trappole trovate guardando la figura renderizzata, non il codice: l'acquamarina sopra le candele si
confonde con il corpo rialzista (il validatore di palette approva la coppia, perche' non sa che una
delle due e' un corpo pieno), e la panoramica metteva in legenda due "EMA corta" e due "Banda
superiore". Entrambe fissate da un test.

### Codice nuovo del filone trading

| file | cosa |
|---|---|
| `trading/indicators_extra.py` | `ExtraParams` + `ExtraCache`: ADX, EMA, ATR, KAMA, Donchian, Bollinger, Keltner, StochRSI, MFI, pendenza OBV, Ichimoku, memoizzati per parametro. **Donchian e' shiftato di una barra** (`.shift(1)`): senza, il canale contiene la barra che lo rompe |
| `trading/strategies_ls.py` | le cinque strategie nuove, tutte con `allow_short`; restituiscono cambi di posizione, non due liste |
| `trading/pnl.py` | `simulate_positions` accanto a `simulate_trading_with_commisions`. `CARRY_DAILY_PERCENT = 0.03` |
| `scripts/strategy_lab.py` | griglie delle nuove (592 configurazioni), `ProcessPoolExecutor` con candele ereditate per fork, metriche short-aware (`n_long`/`n_short`, contributo per lato) |
| `scripts/lab_report.py` | panoramica, effetto short appaiato, ablazioni, trasferimento fra dataset, fuori campione con finestre configurabili, classifica storiche vs nuove, leva e costi |
| `tests/test_long_short.py` | 14 test: simmetria long/short, costi su entrambe le gambe e nel tempo, leva e azzeramento, eventi ignorati dopo la liquidazione, **no look-ahead** (serie troncata → eventi identici) e "long-only non va mai corto", parametrizzati sulle cinque strategie |

`tests/test_simulator_golden.py` copre ora anche `simulate_positions` (una sequenza con inversione
diretta lungo→corto).

Tabelle prodotte: `reports/lab_*.csv` (panoramica, effetto short, ablazioni, classifica, fuori
campione, leva e costi; suffissi `_1d`, `_4h`, `_4h_ciclo2017`, `_ETHUSD_4h`).

### Un guasto che era li' da un giorno: la rotazione con lo store vuoto

`rotation.load_universe` costruiva `pd.DataFrame({})` da un dizionario vuoto -- che nasce con un
RangeIndex di **interi** -- e poi lo filtrava con `frame.index >= since`. Il confronto fra int64 e
una stringa solleva `TypeError`, quindi con `market_data/` vuoto la vista di rotazione **cadeva**
invece di mostrare l'avviso che le sta accanto da sempre. E' la condizione normale in produzione:
il piano gratuito di Render non ha dischi persistenti.

Lo stesso guasto nascondeva la raccolta di `tests/test_simulator_page.py`, che lo esercita a
livello di modulo. In questa sessione avevo attribuito quella raccolta fallita alla versione di
Python: **era sbagliato**. Con la correzione la suite gira intera, 911 test, senza `--ignore`.

### La confluenza (2026-08-27) — scritta, non misurata

Il filone piu' recente. Il disegno completo sta in
[`strategia-confluenza.md`](strategia-confluenza.md); qui solo cosa esiste e cosa manca.

| file | cosa |
|---|---|
| `trading/mtf.py` | `align_to_lower`: legge la barra lunga **chiusa**, spostandola di un periodo. E' l'unica difesa contro il look-ahead fra intervalli |
| `trading/live_frames.py` | le barre lunghe *in formazione*, in forma chiusa (`groupby` + `cummax`/`cummin`/`cumsum`). 103 ms per cinque anni e tre intervalli, contro ore del ciclo ingenuo |
| `trading/voters.py` | `held_state` (eventi → stato tenuto) e `decayed_vote` (stato → voto che sfuma). E' la memoria del segnale, cioe' cio' che rende la confluenza capace di innescare |
| `trading/confluence.py` | la strategia: sei votanti su quattro piani, punteggio pesato, ampiezza per famiglie, soglia che si muove coi piani alti, uscita a tre condizioni. `stati_dei_votanti` isola la parte cara |
| `trading/portfolio.py` | un capitale solo su piu' asset, con occasioni perse e concentrazione |
| `scripts/confluence_lab.py` | il banco: tre griglie, paniere, tre riferimenti, `--selfcheck` che gira senza store |
| voce «Confluence» nel simulatore | due riquadri (decisione e votanti) e la spiegazione attaccata a ogni marcatore |

**Quello che manca e' la misura**, e serve la macchina con lo store delle candele. Da fare, in
ordine: `--grid coordinate` su BTCUSDT a 15m (79 celle, e' anche S0 travestito), poi `--grid ampia`
sui cinque asset, poi `--paniere majors`. I tre riferimenti si stampano da soli.

**Quattro cose gia' sapute che vanno tenute a mente leggendo i risultati:**

1. l'aspettativa dichiarata *prima* e' lo stesso ordine di rendimento del possesso passivo con un
   drawdown molto minore, **non** un rendimento superiore. Se il risultato fosse molto migliore, la
   prima ipotesi da verificare e' il look-ahead, non il successo;
2. il rischio piu' probabile non e' che perda: e' che **operi troppo poco** perche' si possa dire.
   Il numero di operazioni all'anno va riportato accanto a ogni risultato;
3. la **necessarieta' per votante** e' nella tabella dei risultati (`nec_*`, `necessarieta_max`).
   Sopra 0,60 l'insieme e' quel votante travestito, e le metriche parlano di lui;
4. il conteggio delle prove per `multiplicity.py` e' in testa a ogni CSV prodotto. `ampia` sono
   4.800 celle: guardarne la migliore e riportarne lo Sharpe senza scontarlo non e' una misura.

---

## Il filone ML, in breve

Chiuso in negativo e **non va riaperto senza leggere `strategy.md` §10-13**. In una riga: entrare
alla conferma di un minimo e uscire alla conferma di un massimo cattura **zero in media**, su tutti
e 15 i simboli, a ogni soglia, *prima* dei costi. La conferma si paga due volte e la gamba mediana
ne vale 1,76-2,05. Nessuna scelta di modello, feature o iperparametro lo cambia.

Cosa **non** rifare: ritarare la soglia di decisione (§12.6), aggiungere iterazioni DAgger (funziona
ma cura un altro problema), provare un'architettura diversa (l'in-sample e' gia' sotto il costo,
non e' overfitting), fidarsi di un'attribuzione con uscita "perfetta" (usare `confirmed_reversal_rows`
e il controllo con ingressi casuali).

Restano aperte, in quest'ordine: `capture` oltre 0,40 (mai misurata fino a 0,85); la formulazione
di §13.4 (alla barra di conferma, prevedere se *questa* gamba superera' `2 × soglia + costo`); poi
i dati di microstruttura (`aggTrades`) e il modello di riempimento maker (Fase 0.3).

---

## Cose che non stanno nei documenti e servono subito

- **Usa `.venv312/bin/python`.** Il `.venv` preesistente e' Python 3.9 senza `scikit-learn`; il
  progetto richiede ≥3.12. Installazione normale: `pip install -e ".[app,data,dev]"`.
- **Rete: dipende da dove gira la sessione.** Sulla macchina dell'utente `raw.githubusercontent.com`
  e `api.github.com` rispondono, e i README dei repository si scaricano (fatto il 2026-08-26). Il
  paragrafo qui sotto vale per l'**ambiente remoto**, non per il locale, e va letto cosi'.
- **Rete bloccata in sessione remota.** Nessun exchange e nessun aggregatore e' raggiungibile
  (403 sul CONNECT del proxy); anche la *search API* di GitHub e' negata perche' la sessione e'
  legata ai suoi repository. Restano raggiungibili PyPI, i contenuti dei repository configurati e
  gli asset di release. Non perdere tempo a riprovare host nuovi: e' gia' stato fatto in modo
  esaustivo.
- **`market_data/` sulla macchina dell'utente ha tutti e 15 i simboli** a 5 minuti, fino al
  2026-08-19 (284 MB, gitignorato): da 614.732 a 945.675 candele per simbolo. E' con questo store
  che sono state fatte le misure di `ricerca-quant-ml.md`. Il paragrafo qui sotto descrive
  l'**ambiente remoto**, dove ce n'erano due.
- **`market_data/` nell'ambiente remoto contiene solo due file** (55 MB, gitignorato):
  `BTCUSD-5m.parquet` (1.540.397 candele, 2012-01-01 → oggi, fonte Bitstamp) e
  `ETHUSD-5m.parquet` (342.929 candele, 2016-03 → 2019-12, fonte Bitfinex). ETH **non copre il
  ciclo recente**: e' per questo che il controllo su un secondo asset e' stato fatto sul ciclo
  2017-2019 e non sul 2021-2026. Sulla macchina dell'utente lo store Binance e' molto piu' ampio
  (15 simboli, ~11,8 milioni di candele 5m).
- **`models/*.joblib` e `*.json` non sono tracciati** (`models/.gitignore`, esteso nel 2026-08).
  `meta_model.*` e' il modello della strategia precedente: non cancellarlo, `load_signal_model()`
  lo carica ancora. `MODEL_PRECEDENCE` e `active_model_name()` sono l'unica fonte di verita'.
- **Test: 695 in 20 file.** `ruff check src tests scripts` e `black --check` puliti. La CI gira
  entrambi i job su ogni PR.
- Le due misure lunghe (`strategy_sweep`, `strategy_lab`) impiegano decine di minuti: farle partire
  in background e attendere con un ciclo di controllo, mai con `sleep` in catena.
- **`analysis_cache/` e' gitignorata ed e' l'input di `scripts/tune_defaults.py`.** Senza,
  `trading/tuned_defaults.py` non e' rigenerabile: servono di nuovo le griglie su quattro intervalli
  per cinque simboli, circa due ore di calcolo. I `reports/*.csv` invece sono tracciati.

## Regole di ingaggio stabilite dall'utente

- Prima di modifiche strutturali: piano scritto, poi conferma. (Sospeso quando l'utente dice
  esplicitamente "procedi con l'implementazione".)
- **Ogni numero va misurato sui dati del progetto, mai stimato ne' ripreso dai prompt.** L'utente
  ha ripetuto: *"se una misura contraddice una tesi del prompt, riportalo — preferisco una
  strategia corretta a una che conferma quello che ho chiesto."* E' successo di nuovo in questa
  sessione (lo short, che l'utente si aspettava raddoppiasse le occasioni, e' misurato in perdita)
  e va detto senza addolcire.
- Controlli a cascata su ogni risultato, e sospetto verso i risultati troppo buoni.
- Commit incrementali con riepilogo dopo ogni blocco.
- I deliverable finali vanno anche pubblicati come artifact leggibile, oltre che scritti nel repo.

## Trappole gia' incontrate

**Sul simulatore e i backtest**

- **Look-ahead nei canali**: un massimo mobile che include la barra corrente rende la rottura
  impossibile da mancare. `indicators_extra` shifta Donchian; il test `test_no_look_ahead` verifica
  che troncare la serie non cambi gli eventi gia' emessi.
- **Look-ahead *dentro* la barra**: uno stop a trailing costruito con il massimo e l'ATR della
  barra su cui viene testato assume che dentro quella barra l'estremo favorevole arrivi per primo.
  **`test_no_look_ahead` non lo vede**, perche' tronca la serie *fra* le barre e la barra che
  scatena l'evento resta identica: serve il controllo che perturba il massimo della sola barra
  dell'uscita (`test_trailing_stop_ignores_the_high_of_its_own_bar`). Lo stop va calcolato su dati
  chiusi a `i-1`.
- **`ta` riempie, non lascia NaN.** `IchimokuIndicator(visual=True)` costruisce span B con
  `min_periods=0` e riempie le prime `slow` righe dello shift con la media dell'**intera** serie:
  span B non e' mai NaN, nessuna guardia lo intercetta, e quel riempimento e' look-ahead. In
  `ichimoku_trend` la protezione e' `start = slow + span + 2`, e non va abbassata.
- **Il golden master accetta qualunque differenza** se lo si rigenera. Rigenerare solo dopo aver
  verificato a mano la differenza, e controllare che il diff contenga solo le righe attese.
- **Gli scenari del golden non sono intercambiabili**: `close_ema_crossover_simulation` pretende tre
  incroci in sequenza, `close_bullish_ema_simulation` solo il laterale. Togliere uno scenario scopre
  delle strategie.
- **`indicators._atr_ema` replica `ta` 0.11 riga per riga.** Se si tocca, va riverificato contro
  `ta`: una divergenza silenziosa sposta ogni segnale.
- **Le funzioni decorate con `@st.cache_data` si chiamano con `.__wrapped__`** fuori da Streamlit.
- **Il massimo su una griglia non e' un risultato**: va sempre letto con la mediana e con la quota
  di configurazioni in utile accanto.

**Sulla pipeline ML**

- **Pivot retrospettivi**: usare `extreme_bar` invece di `confirm_bar` in una feature e' look-ahead
  (`strategy.md` §7.1: ritardo mediano 1-8 barre, p99 fino a 101, massimo proprio sui movimenti ampi).
- **Etichette sovrapposte**: `t_exit` e' il pivot successivo confermato, orizzonte variabile e
  potenzialmente lungo. L'embargo va dimensionato sul percentile alto.
- **Rolling non causali**: `labeling.py` usa `[::-1].rolling(...)[::-1]` di proposito. Corretto li',
  disastroso altrove.
- Due difetti del target hanno gia' **invertito il segno** dei risultati una volta (commit `7ebb2e0`).

## Suggested skills

- **`tdd`** — per qualunque estensione del motore di posizioni o delle strategie: i bug piu' costosi
  di entrambe le sessioni sono stati trovati dai test.
- **`diagnosing-bugs`** — quando una misura non torna.
- **`dataviz`** — prima di aggiungere grafici al simulatore.
- **`artifact-design`** — l'utente si aspetta un report visuale a chiusura di ogni blocco di misure.

- **`ponytail:ponytail`** — l'utente la lancia da se' a inizio sessione; nella terza era attiva a
  livello `full` per tutto il tempo.

Non serve `codebase-design` (la struttura a moduli e' decisa e documentata). `research` non serve
per lo stato dell'arte, che e' gia' raccolto in `ricerca-quant-ml.md` §1 — ma **sulla macchina
dell'utente la rete funziona**, quindi la vecchia nota "nessuna fonte esterna raggiungibile" vale
solo in remoto.
