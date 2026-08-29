# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentazione di lavoro

Le decisioni di progetto e lo stato del lavoro stanno in **`.claude/docs/`**:

- `.claude/docs/strategy.md` — fonte di verità delle decisioni su labeling, feature, modello e
  validazione, con le misure che le giustificano. Da aggiornare in luogo quando si decide qualcosa.
- `.claude/docs/HANDOFF.md` — stato corrente del lavoro e trappole ambientali per chi riprende.
- `.claude/docs/backtest-strategie.md` — le strategie a indicatori misurate su nove anni: 3.129
  configurazioni, sensibilità ai parametri, tenuta fuori campione, difetti trovati misurando.
- `.claude/docs/strategia-confluenza.md` — la strategia multi-timeframe a più segnali: quattro
  piani con domande disgiunte, sei votanti scelti per famiglia, memoria del segnale, soglia decisa
  dai piani alti. **Misurata (2026-08-28) su 15 asset e sette anni: non batte il possesso passivo.**
  Niente look-ahead, votanti non correlati, ma il gradiente di ogni parametro punta al non-operare.
  Le conclusioni e cosa farne stanno in fondo a quel documento.
- `.claude/docs/strategie-nuove.md` — il seguito: le quattro correzioni applicate, il ciclo
  2021-2026 come dataset, cinque strategie nuove e il motore che sa stare anche corto.
- `.claude/docs/politica-rl.md` — **la politica a rinforzo, cablata (2026-08-28).** Le tre misure
  che escludono lo stop e indicano la commissione come causa, la ricompensa col costo dentro, e i
  risultati: batte il possesso passivo 11/15 fuori campione e **dimezza la discesa massima**, ma il
  *quando* sta sopra il caso solo debolmente.
- `.claude/docs/modello-swing.md` — **il modello AI rifatto, misurato e cablato (2026-08-28).**
  L'audit che ha tolto `leg_model` dalla catena, l'etichettatura nuova a prossimità degli
  estremi, le tre misure per cui il segnale esiste (IC +0,0385 fuori campione, 14/15 simboli
  concordi) ma non batte il caso a esposizione appaiata, e il §5.4 su **cosa è stato cablato e
  cosa deliberatamente no** nella pagina e in Confluence.
- `.claude/docs/modello-ingresso.md` — **il modello in testa oggi, cablato (2026-08-29).** Cambia
  la domanda: non «quanto siamo vicini a un estremo» ma «quanto rende comprare qui». Le misure che
  hanno spostato il bersaglio (a pari selezione l'etichetta a gambe individua meglio i minimi e
  rende 2,4 volte meno), la selettività come unica leva, e i **primi numeri di questo progetto che
  passano il controllo a esposizione appaiata**: +2,071% netti per operazione fuori campione,
  14/15 simboli in utile, 100° percentile. Il veloce opera, il lento gli fa da cancello.
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
    ├── mtf.py            allineamento fra intervalli: legge la barra lunga **chiusa**, mai quella corrente
    ├── live_frames.py    le barre lunghe *in formazione*, in forma chiusa — **oggi non lo importa nessuno**
    ├── voters.py         da cambi di posizione a voto per barra, con memoria e decadimento
    ├── confluence.py     la strategia a confluenza: sei votanti su quattro piani, soglia dinamica
    ├── portfolio.py      un capitale solo su più asset: si apre sul primo che parla
    ├── rotation.py       rotazione trasversale: sceglie *quale* asset, non *quando*
    ├── tuned_defaults.py generato: valori di partenza misurati, per intervallo
    ├── config.py         valori di partenza dei widget della pagina
    ├── simulator.py      la pagina Streamlit: due viste, `trading_analysis` + `rotation_page`
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

# Modello a swing: prossimità agli estremi locali (vedi .claude/docs/modello-swing.md)
.venv312/bin/python -m cryptofarm.data.positioning --update     # posizionamento futures, 400 MB
.venv312/bin/python -m cryptofarm.ml.swing_trainer --selfcheck  # gira senza store
.venv312/bin/python -m cryptofarm.ml.swing_trainer              # ~12 minuti, 15 simboli dal 2018
.venv312/bin/python -m scripts.swing_lab                        # decili, P&L, controllo casuale

# Modello d'ingresso: quanto rende comprare qui (vedi .claude/docs/modello-ingresso.md)
.venv312/bin/python -m cryptofarm.ml.entry_trainer --selfcheck  # gira senza store
.venv312/bin/python -m cryptofarm.ml.entry_trainer              # ~12 minuti, il lento (H=150)
.venv312/bin/python -m cryptofarm.ml.entry_trainer --h 20 --quantile 0.995 --nome entry_model_veloce
.venv312/bin/python -m scripts.entry_lab                        # quanto vale il cancello del lento
.venv312/bin/python -m scripts.entry_lab --frequenza           # quanto costa operare di piu'

# Politica a rinforzo: sceglie la posizione col costo dentro la ricompensa (.claude/docs/politica-rl.md)
.venv312/bin/python -m cryptofarm.ml.rl                         # selfcheck del solo algoritmo
.venv312/bin/python -m cryptofarm.ml.rl_trainer --selfcheck     # gira senza store
.venv312/bin/python -m cryptofarm.ml.rl_trainer                 # ~5 minuti, 15 simboli dal 2019
.venv312/bin/python -m scripts.rl_lab                           # controllo a blocchi rimescolati

# Misure di strategy.md
.venv312/bin/python -m scripts.analysis

# Rotazione trasversale e filtro meta (vedi .claude/docs/ricerca-quant-ml.md)
.venv312/bin/python -m scripts.cross_section --universe majors --interval 1d --grid
.venv312/bin/python -m scripts.meta_gate --strategy donchian_breakout --interval 4h --oos 2024-01-01

# Valori di partenza misurati per intervallo (rigenera trading/tuned_defaults.py)
.venv312/bin/python -m scripts.tune_defaults --all-intervals --save

# Backtest delle strategie a indicatori su tutto lo storico (vedi .claude/docs/backtest-strategie.md)
.venv312/bin/python -m scripts.strategy_sweep --all --interval 15m   # griglie di parametri
.venv312/bin/python -m scripts.sweep_report --interval 15m           # tabelle in reports/
.venv312/bin/python -m scripts.strategy_focus --top 3                # commissioni e intervalli

# Strategia a confluenza (vedi .claude/docs/strategia-confluenza.md)
.venv312/bin/python -m scripts.confluence_lab --selfcheck             # gira senza store, dati finti
.venv312/bin/python -m scripts.confluence_lab --grid coordinate --symbol BTCUSDT --interval 15m
.venv312/bin/python -m scripts.confluence_lab --grid ampia --interval 15m --since 2021-01-01
.venv312/bin/python -m scripts.confluence_lab --grid veloce --paniere majors

# Strategie a due versi, long e short (vedi .claude/docs/strategie-nuove.md)
.venv312/bin/python -m scripts.strategy_lab --all --interval 1d --since 2021-01-01
.venv312/bin/python -m scripts.lab_report --symbol BTCUSD --interval 1d

# Store delle candele da fonte alternativa, dove data.binance.vision non è raggiungibile
.venv312/bin/python -m scripts.import_candles --source /percorso/al/clone

# Bot live — piazza ordini veri, richiede le variabili d'ambiente (vedi .env.example)
.venv312/bin/python src/cryptofarm/trading/live_bot.py
```

Test: `.venv312/bin/python -m pytest` (994 test in 36 file, tutti verificati). Lint/format: `ruff check src tests` e
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

### Le due viste

La pagina ha un interruttore in cima alla barra laterale (`config.ROTATION_MODES`), e le due voci
non sono due strategie ma **due domande diverse**:

- **Single asset** — `trading_analysis`: carica un simbolo dall'exchange e ci esegue sopra una
  strategia del menu. Sceglie *quando* stare dentro.
- **Cross-asset rotation** — `rotation_page` su `trading/rotation.py`: carica l'universo **dallo
  store locale**, lo ordina per forza relativa e tiene i primi. Sceglie *quale*.

Tre conseguenze da conoscere prima di toccarle:

- **la rotazione non usa la rete.** Legge `market_data/`, quindi in produzione (nessun disco
  persistente) non ha dati e lo dice, invece di provare quindici scarichi. Un test lo verifica;
- **i valori iniziali sono centrali, non ottimi.** La correlazione fra resa in stima e resa in
  verifica sulle prime dieci configurazioni e' **-0,69**: cercare il massimo in campione trasferisce
  peggio che prendere una configurazione a caso. Chi li cambia in "quelli che rendono di piu' nel
  grafico" sta facendo esattamente l'errore misurato;
- **il riferimento da battere e' l'universo a peso uguale, non BTC.** Porta la stessa distorsione da
  sopravvivenza della rotazione, quindi il confronto isola cio' che la rotazione aggiunge. Contro
  BTC la rotazione vince nel 95,6% delle configurazioni; contro l'universo, nel 44,4%.

### I valori di partenza dipendono dall'intervallo

`trading/tuned_defaults.py` e' **generato** da `scripts/tune_defaults.py` e non si modifica a mano.
Tiene, per ognuno dei quattro intervalli misurati (15m, 1h, 4h, 1d), il valore di partenza di ogni
parametro di ogni strategia del menu.

**Come sono scelti, e perche' non e' il massimo della griglia.** Il massimo e' la cella piu'
fortunata: su questi dati la scelta del massimo trasferisce peggio della mediana, e sulla rotazione
la correlazione fra resa in stima e in verifica e' −0,69. Qui si sceglie una coordinata alla volta:
ogni configurazione riceve il suo **rango percentile dentro il proprio simbolo** (unico modo di
sommare asset i cui possessi passivi vanno da +134% a +4.346%), e per ogni valore di ogni parametro
si prende la mediana di quei ranghi su cinque asset. Si adotta il valore migliore **solo se** supera
due controlli: sposta la mediana dei ranghi di almeno 0,06, e sceglie lo stesso valore anche
guardando il solo 2021-2023. Chi non li supera tiene il default scritto a mano.

**La mappa `panels.ANCORA_MISURATA`** dice quale misura copre quale intervallo: il menu ne offre
nove, le griglie ne coprono quattro. E' un dato e non un calcolo, perche' "il piu' vicino" e' gia'
una decisione (30m sta in mezzo fra 15m e 1h).

Tre cose da sapere prima di toccarlo:

- **la chiave dei widget include l'intervallo** (`par_{nome}_{intervallo}`). Streamlit conserva lo
  stato di un widget con la stessa chiave: senza, cambiando timeframe i campi restano fermi sui
  numeri del precedente e il default misurato non compare mai. Il difetto e' invisibile leggendo il
  codice e non lo vede `AppTest`, che ricostruisce lo stato a ogni run — per questo il test asserisce
  sulla **chiave**, non sul valore;
- **le finestre crescono quando le barre si accorciano**, ed e' la lettura meccanica del risultato:
  la stessa regola vuole un canale di 20 barre a un giorno e di 150 a un'ora per coprire lo stesso
  tratto di calendario. Un test fissa il verso di quella disuguaglianza;
- **due parametri non hanno una lettura coerente fra intervalli** e vanno trattati con sospetto:
  `ATR Bands / atr_multiplier` (3,0 a 15m, 1,6 a 1h, 1,2 a 4h, 3,0 a 1d) e
  `Donchian Breakout / adx_min`. Sono scelte adottate perche' superano i due controlli su ogni
  intervallo preso da solo, ma il quadro d'insieme non le sostiene. `tune_defaults` stampa la
  tabella dell'accordo fra intervalli apposta per renderle visibili.

**Sotto l'ora nessuna misura di questo progetto ha mai trovato qualcosa che batta il possesso
passivo.** I default a 15m sono i migliori *fra quelli provati*, non buoni.

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

### La strategia a confluenza

`trading/confluence.py` è l'unica voce del menu che non è una strategia a indicatore: legge
**quattro piani temporali ricavati dall'intervallo scelto** (`FATTORI` — ×1 innesco, ×4 conferma,
×16 struttura, ×96 regime, cioè 15m/1h/4h/1d partendo da 15m) e fa votare otto strategie diverse.
Quattro cose da sapere prima di toccarla:

- **aggiungere un votante è `confluence.registra(Votante(...))`, quasi e basta.** Da lì si adattano
  da soli famiglie, pesi, necessarietà, riquadri della barra laterale, parametri della strategia e
  griglia del banco, ed è quello il punto. L'unico elenco rimasto da tenere allineato a mano sono
  le tracce del riquadro *Voters* in `panels.INDICATORI`, e c'è un test che se ne accorge: conta
  le tracce col `·` contro `len(VOTANTI)`;
- **il votante `modello` sta nel default solo se un artefatto c'è.** `votanti_predefiniti()` lo
  toglie quando nessuno dei quattro (`entry_model_veloce`, `entry_model`, `rl_model`,
  `swing_model`) è su disco, che è la condizione della produzione: i pesi si normalizzano sui
  votanti presenti, quindi un ottavo che tace sempre alzerebbe di fatto la soglia per gli altri
  sette. Nel registro ci resta, così `selezione("modello")` lo raggiunge. È anche l'unico votante
  **solo lungo**: vota +1 o 0, mai −1. Col modello d'ingresso vota +1 mentre una sua operazione è
  aperta e le due soglie non hanno effetto — la selettività sta nei metadata dell'artefatto;
- **i parametri dei votanti si risolvono in tre strati**: default della funzione (`config.CONF_*`),
  valore misurato in `tuned_defaults` per l'intervallo del **piano** su cui il votante gira — non
  quello della pagina — e override di chi chiama. Il secondo strato è quello che si sbaglia
  facilmente: a base 15m un votante di struttura gira a 4h e vuole i valori di 4h;
- **muoverli costa.** Il congelamento teneva a nove i parametri liberi; con le 31 manopole aperte
  si superano i quaranta, e `scripts/multiplicity.py` dice cosa succede lì. Muoverli per capire,
  misurare con i votanti fermi;
- **la soglia è continua, non a gradini.** I piani lunghi entrano come distanza dalla media
  normalizzata sull'ATR dello stesso piano, non come `np.sign`: con il segno la soglia saltava di
  0,15 per volta e una uscita per punteggio su quattro era decisa da quel salto;
- **l'isteresi ha un pavimento e un soffitto** (`barre_minime`, `pazienza`), e valgono **solo per
  l'uscita dal punteggio**. Lo stop e il cancello no: sono regole di rischio, non di opinione;
- **la parte cara non dipende dalla griglia.** I votanti congelati hanno uno stato che dipende solo
  da (simbolo, intervallo): `stati_dei_votanti` lo calcola una volta e `scripts/confluence_lab.py`
  lo riusa su tutte le celle. Misurato su 11.520 barre: 351 ms per cella contro 104 ms;
- **il punto in cui può barare è uno solo**, `_stato_del_votante`, e la difesa è
  `mtf.align_to_lower`, che sposta lo stato del piano lungo di un periodo intero prima di leggerlo.
  Il test che lo protegge taglia **dentro** una barra lunga già cominciata e confronta gli stati:
  un taglio allineato ai confini passa anche col difetto reintrodotto, ed è com'era scritto la
  prima volta;
- **`live_frames.py` oggi non lo importa nessuno.** Era lo stadio S1 e serviva a sollevare i piani
  lunghi a valore provvisorio; scrivendo la confluenza si e' visto che su un confronto di *segno* il
  sollevamento e' algebricamente un non-fare (`confluence._sign_su_media` lo dimostra in tre
  righe), e che sollevare una *strategia* qualunque non e' generico. Il modulo resta perche' e' il
  pezzo che serve appena un votante debba leggere il **valore** di una barra lunga parziale -- una
  distanza, una banda, uno stop -- e perche' contiene il test contro il difetto da tre caratteri
  (`transform("max")` invece di `cummax`). Se si decide che non servira', si cancella: e' codice
  vivo solo nei suoi test, e va detto invece che lasciato credere il contrario;
- **zero operazioni non e' un risultato, e' una domanda.** Le condizioni d'ingresso sono quattro in
  `and` e `Confluenza.perche_non_entra()` dice quale non si e' mai avverata, con i numeri. Serve
  perche' il caso piu' comune non e' la prudenza della strategia ma la storia: a 15m il piano di
  regime e' giornaliero e la sua media ne chiede cinquanta barre, cioe' **1.200 ore**, contro le
  240 del valore di partenza della pagina;
- **la scala x1/x4/x16/x96 vale attorno ai quindici minuti.** A 1m il «regime» dura un'ora e mezza,
  a 1d chiede barre da 96 giorni. La regola scritta e' che il piano di regime duri fra mezza
  giornata e una settimana (`scala_fuori_misura`), il che lascia 15m, 30m e 1h;
- **la spiegazione viaggia col segnale.** I segnali della confluenza sono `(quando, prezzo, testo)`
  invece di `(quando, prezzo)`, e il grafico mostra il testo al passaggio del mouse. Per questo
  `pnl` scompatta con `[:2]`: qualunque strategia può aggiungere elementi dopo i due che il motore
  usa. Il testo **distingue gli ingressi dalle uscite**: quattro uscite su cinque sono lo stop a
  trailing, e mostrarci sopra i votanti fa leggere «venduto mentre cinque votanti dicevano di
  comprare», che è vero e del tutto fuorviante;
- **i quattro riquadri non sono intercambiabili.** `regime` e `struttura` valgono ±1 e il punteggio
  sta in ±0,5: sullo stesso asse il primo schiaccia il secondo, e si vede una linea ferma a 1
  mentre si compra e si vende. Da qui il riquadro *Higher planes* separato, e lo stop a trailing
  disegnato sulle candele — senza, l'80% delle vendite è inspiegabile dal grafico.

`trading/portfolio.py` risponde a una domanda diversa e non va confuso con `rotation.py`: la
rotazione sceglie *quale* asset tenere e ci sta dentro sempre; il paniere a capitale condiviso sta
fuori finché nessuno parla e mette tutto il capitale sul primo che dà il segnale. Riporta sempre le
**occasioni perse** mentre il capitale era impegnato e la **concentrazione**, cioè la quota
dell'asset più operato: sopra 0,9 il paniere è finzione.

### Funzioni di `strategies.py` che il menu non raggiunge

`buy_sell_limits_simulation` legge `MACD`, che resta commentata in `add_technical_indicator`, e
quindi solleva `KeyError` appena chiamata: e' l'unica esclusa perche' rotta.

Le altre sette sono **uscite dal menu misurando** (2026-08-26, `.claude/docs/ricerca-quant-ml.md`
§2): Close Buy/Sell Limits, Close ATR, Close Bullish EMA, Green Candles, ATR Live Trade, Trend
Pullback, Band Reversion. Restano nel modulo e nel golden master -- la misura si rifa' con
`scripts/strategy_sweep` -- ma non sono selezionabili.

**`close_rsi_buy_sell_limits_simulation` e' invece rientrata** ("Close RSI Reverse"). La ragione
per cui era esclusa -- "in perdita totale in tutte le 25 configurazioni provate" -- vale a 15
minuti e non a scala giornaliera: a 1d fa 24-27 operazioni l'anno, mediana positiva su tutti e
cinque i simboli e 72-92% di configurazioni in utile; a 4h ne fa 160 l'anno e su BTC perde il
45,8%. E' il caso piu' netto della regola gia' nota, che la frequenza operativa spiega quasi tutto:
**una strategia esclusa su un intervallo non e' esclusa su tutti**.

### Il golden master

`tests/test_simulator_golden.py` fissa il comportamento di 21 funzioni su quattro scenari di mercato
sintetici, confrontandolo con `tests/data/simulator_golden.json`. Copre il **comportamento delle
funzioni**: **prima di toccarlo, questo deve passare; dopo, deve passare ancora senza rigenerarlo**.

L'**assemblaggio** invece lo copre `tests/test_simulator_page.py`, che esegue la pagina con
`streamlit.testing.v1.AppTest`. E' il livello da cui e' passato il guasto che tolse il simulatore
dalla produzione: ogni funzione aveva i suoi test e passavano tutti, mentre `load_signal_model()`
chiamata senza condizione dentro `__main__` impediva alla pagina di aprirsi. Copre anche la
degradazione senza store delle candele, che e' la condizione in cui gira il servizio pubblico.

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

`ml/trainer.MODEL_PRECEDENCE` è `("rl_model", "swing_model", "meta_model", "signal_model")` e
`active_model_name()` è l'unica fonte di verità: `load_signal_model` carica quel modello e
`ai_model_simulation` sceglie la strategia in base a quel nome, quindi i due non possono divergere.
Per tornare al modello precedente basta spostare altrove l'artefatto di quello più recente.

`meta_parameters()` legge barriere, soglia CUSUM e parametri di esecuzione **dai metadata
dell'artefatto**, non da costanti: devono essere esattamente quelli con cui il modello è stato
addestrato.

**Il modello in testa oggi è `entry_model_veloce`, e i due artefatti d'ingresso lavorano in
coppia.** Prevede il rendimento delle prossime H barre — non la forma del grafico — e il suo
vantaggio è la **selettività**: al 10% di barre segnalate il netto è sotto la commissione, allo
0,5% è dieci volte sopra. Ne segue che soglia, cancello e tenuta stanno nei **metadata
dell'artefatto** e non nei widget: cambiarli non regola una manopola, serve un'altra strategia.
Il veloce (tenuta 20 barre) genera le operazioni, il lento (`entry_model`, tenuta 150) fa da
cancello sulla sola barra d'ingresso: +2,071% netti per operazione fuori campione contro +1,360%
senza, 14 simboli su 15 in utile, 100° percentile contro ingressi a caso a pari esposizione
(`modello-ingresso.md`). Senza l'artefatto lento il veloce opera da solo, e si torna a +1,360%.

**Si serve fino a 30 minuti e sopra tace.** La soglia è un rendimento, non un quantile, e il
modello prevede quello delle prossime venti barre *da cinque minuti*: sulla stessa soglia le barre
marcate passano da 0,063% a 5m a 2,98% a 1h e 28,1% a 1d, contro lo 0,5% per cui è misurato.
`signals.entry_fuori_misura` è il cancello di scala, gemello di `confluence.scala_fuori_misura`.
Conseguenza da conoscere prima di dire «non funziona»: a 5m marca **una barra su millecinquecento**,
quindi su una finestra da 240 ore zero operazioni è il comportamento atteso.

Le famiglie precedenti restano nella catena sotto di lui. `swing_model` prevede la prossimità agli
estremi locali e la forma misurata di quel segnale è a U: *entrambi* i poli precedono rendimenti
sopra la media, quindi il segno **non dice il verso**. `ml/signals.swing_exposure` cabla l'unica
lettura che la misura sostiene — `|previsione|` come interruttore di esposizione, con isteresi.
Cablare `sign(previsione)`, che è la lettura naturale di un target in `[-1, 1]`, vende esattamente
le barre migliori: è misurato in perdita a tutte le soglie e tutte le cadenze
(`modello-swing.md` §5.1). Quel modello **non batte il possesso passivo**.

## Data/model artifacts

`models/` contiene gli artefatti (`.joblib` + `.json` di metadata) e **non ne traccia nessuno**:
`models/.gitignore` copre `*.keras`, `*.joblib` e `*.json`, e tiene solo il `README.md`. Un clone
del repository quindi non ha modelli, ed è la condizione in cui gira il servizio pubblico.
Rigenerare con i trainer, non modificare a mano.

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
dischi persistenti, e con `models/*.joblib` gitignorato online girano le strategie classiche.

Il modello è **opzionale** per la pagina: gli artefatti sono gitignorati, quindi un clone del
repository e l'immagine in produzione non ne hanno. `simulator.available_strategies` toglie la voce
`config.AI_STRATEGY` dal menu quando `active_model_name()` non trova niente, e il caricamento
all'avvio è condizionato allo stesso controllo. Chi tocca quel punto tenga presente che prima il
`load_signal_model()` era incondizionato e faceva cadere l'intera pagina, non solo quella strategia.

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
