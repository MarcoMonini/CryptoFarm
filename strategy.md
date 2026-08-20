# Strategia di training — analisi e decisioni

Documento di riferimento per le fasi di sviluppo successive. Contiene l'analisi, le scelte
motivate e i trade-off espliciti. **Nessun codice è stato modificato per produrlo.**

Ogni numero qui riportato è misurato sui dati effettivamente presenti nel progetto, non stimato.
Gli script di misura stanno nella scratchpad di sessione; i comandi per riprodurli sono indicati
dove serve.

---

## Sintesi esecutiva

Tre conclusioni, in ordine di importanza.

**1. La modalità di esecuzione domina ogni altra scelta.** Con barriera 2:1 e stop allo 0,6%, la
precision di break-even è 44,4% con commissioni taker e 35,6% con commissioni maker. Il base rate
teorico di una barriera 2:1 è 33,3%. Significa che con ordini a mercato il modello deve battere il
caso di **11,1 punti**, con ordini limite di **2,2 punti**. Il primo è un obiettivo che in questo
dominio non si raggiunge stabilmente; il secondo è alla portata di un modello con AUC 0,55.
Nessun lavoro su feature o architettura sposta questo rapporto.

**2. Il target di >10 trade/giorno è incompatibile con "target ≥ 2× fee round-trip" su un singolo
simbolo in modalità taker, ed è banale su un portafoglio.** Con fee taker il target minimo richiesto
è 0,40%, che su BTC 5m si raggiunge in una mediana di 27 barre (2,2 ore): il tetto teorico è 10,7
trade/giorno *assumendo di entrare a ogni occasione e centrare sempre il target*. Con posizioni non
sovrapposte e un modello selettivo si scende a una frazione. La stessa frequenza su 15 simboli
richiede meno di 1 trade/giorno per simbolo, che è ampiamente sostenibile con barriere larghe e
selettive. **La frequenza va cercata nella larghezza del portafoglio, non nella brevità del trade.**

**3. Il collo di bottiglia attuale non è il modello.** Un LSTM da 747k parametri e un gradient
boosting hanno prodotto lo stesso risultato sulle stesse etichette (macro F1 0,092 contro 0,111,
in 25 minuti contro 3,9 secondi). L'ultimo modello addestrato ha AUC 0,54 ed edge lordo +0,017%
per operazione. Le leve che restano sono, in ordine: esecuzione maker, campionamento a eventi,
dati di microstruttura. Il tuning architetturale è l'ultima, non la prima.

---

## 1. Verifica di idoneità dei dati

### 1.1 Cosa c'è

Store locale in `market_data/`, popolato da `python -m cryptofarm.data.klines --update`:

| | |
|---|---|
| Simboli | 15 (BTC, ETH, BNB, SOL, XRP, ADA, DOGE, AVAX, LINK, DOT, LTC, TRX, ATOM, NEAR, UNI, tutti USDT) |
| Granularità archiviata | 5m — 15m/30m/1h derivati per aggregazione esatta |
| Candele | 11.770.246 |
| Dimensione | 298 MB (parquet) |
| Copertura | BTC/ETH dal 2017-08; il più recente (AVAX/UNI) dal 2020-09 |
| Campi | Open, High, Low, Close, Volume |

Aggiornamento incrementale in secondi; il download completo richiede minuti (dump CDN paralleli).

### 1.2 Copertura dei regimi — **adeguata**

BTC, rendimento su finestra mobile di 30 giorni, soglia ±10% (quota di giorni per anno):

| anno | bear | sideways | bull |
|---|---|---|---|
| 2017 | 5% | 18% | 78% |
| 2018 | **48%** | 39% | 13% |
| 2019 | 22% | 40% | 39% |
| 2020 | 13% | 33% | 55% |
| 2021 | 25% | 29% | 46% |
| 2022 | **44%** | 45% | 11% |
| 2023 | 10% | 48% | 43% |
| 2024 | 8% | 56% | 36% |
| 2025 | 17% | **67%** | 16% |
| 2026 | 27% | 61% | 12% |

**Complessivo su 9,0 anni: sideways 45%, bull 32%, bear 23%.**

I tre regimi sono tutti rappresentati, e — cosa più importante per la validazione — sono
*concentrati in periodi distinti*. 2018 e 2022 sono anni prevalentemente ribassisti, 2017 e 2020–21
prevalentemente rialzisti, 2024–26 prevalentemente laterali. Questo consente una CPCV in cui ogni
combinazione di fold vede una miscela di regimi diversa: è esattamente la condizione che rende la
distribuzione degli Sharpe out-of-sample informativa invece che un artefatto di un singolo ciclo.

Il rovescio della medaglia: **la validation attualmente in uso (dal 2024-12-31) cade in un periodo
per il 61–67% laterale.** Un modello valutato solo lì è valutato su un regime solo. È una ragione
ulteriore, indipendente dal leakage, per abbandonare lo split singolo.

### 1.3 Granularità sufficiente per >10 trade/giorno? — **sì, ma non nel modo ovvio**

Misura su BTCUSDT 5m dal 2023 (382.066 candele): tempo mediano perché il prezzo raggiunga un
target rialzista di ampiezza data, entro un orizzonte di 24 ore.

| target | barre mediane | ore | trade/giorno teorici (1 simbolo) |
|---|---|---|---|
| 0,08% | 3 | 0,2 | 96,0 |
| **0,40%** | **27** | **2,2** | **10,7** |
| 0,60% | 45 | 3,8 | 6,4 |
| 1,00% | 77 | 6,4 | 3,7 |
| 2,00% | 126 | 10,5 | 2,3 |

Il vincolo richiesto — target ≥ 2× fee round-trip — si legge direttamente qui:

- **Taker (0,10%/lato)**: round-trip 0,20%, target minimo 0,40% → **tetto di 10,7 trade/giorno**,
  e quello è il limite *fisico* assumendo entrata a ogni barra e successo sistematico. Con un
  modello che seleziona il 5% delle candele e vince il 40% delle volte, il numero reale è
  nell'ordine di 1–2 al giorno per simbolo.
- **Maker (0,02%/lato)**: round-trip 0,04%, target minimo 0,08% → 96 trade/giorno teorici. Qui
  >10/giorno su un simbolo solo è realistico.

**Conclusione operativa: con 15 simboli, >10 trade/giorno si ottiene con meno di 1 trade/giorno
per simbolo, mantenendo barriere allo 0,6–1% che hanno un margine sano sopra le fee anche in
modalità taker.** Questa è la configurazione da perseguire. Forzare >10 trade/giorno su un simbolo
solo obbliga a barriere sottili, dove le fee mangiano tutto — è il modo tipico in cui una
strategia con win rate apparentemente buono perde soldi.

### 1.4 Barre a eventi: dollar bar e CUSUM — **entrambe fattibili, CUSUM preferibile**

Le time-bar fisse attualmente usate campionano il mercato a intervalli regolari, mentre
l'informazione arriva a raffiche. Su un mercato 24/7 questo significa che gran parte delle barre
notturne è rumore e le barre nei momenti di attività aggregano troppo.

**Dollar bar** — fattibili con i dati presenti (`Close × Volume` è già in archivio):

| soglia per barra (BTC) | barre/giorno |
|---|---|
| $214,2 M | ~10 |
| $42,8 M | ~50 |
| $21,4 M | ~100 |
| $7,1 M | ~300 |

Volume in dollari medio giornaliero di BTC: $2,14 miliardi. La soglia va calibrata **per simbolo**
(un valore unico produrrebbe 100 barre/giorno su BTC e 2 su NEAR) e **riadattata nel tempo** (il
volume in dollari di BTC è cresciuto di ordini di grandezza dal 2017: una soglia fissa produce
poche barre nei primi anni e moltissime negli ultimi, distorcendo la densità del dataset per
epoca). Meglio una soglia mobile ancorata a una mediana su finestra lunga.

**Filtro CUSUM** — fattibile e, a mio avviso, la scelta migliore:

| soglia | eventi/giorno (BTC) |
|---|---|
| 1× σ (0,121%) | 99,3 |
| 2× σ (0,241%) | 49,7 |
| 3× σ (0,362%) | 30,0 |
| **5× σ (0,603%)** | **14,7** |

σ = deviazione standard dei log-return a 5m su finestra di 24 ore (mediana 0,121% per barra).

Il CUSUM è preferibile ai dollar bar per tre ragioni: si auto-normalizza sulla volatilità (quindi
è già confrontabile tra simboli ed epoche, senza calibrazione per simbolo); campiona *quando è
successo qualcosa di dimensione rilevante*, che è esattamente la condizione in cui un ingresso ha
senso; e la soglia si mappa direttamente sul vincolo economico — a 5×σ si generano ~14,7
eventi/giorno su BTC con un movimento accumulato dello 0,6%, cioè 3× la fee round-trip taker.

**Trade-off da mettere in conto:** con il CUSUM il dataset non è più una serie regolare. Le feature
a ritardo fisso in barre (`RET_1`, `RET_5`, … in `dataset.py`) cambiano significato, perché la
distanza temporale fra due eventi consecutivi è variabile. Vanno affiancate da feature calcolate su
finestre *temporali* fisse, e il tempo trascorso dall'evento precedente diventa esso stesso una
feature (è informativo: eventi ravvicinati segnalano un regime attivo).

### 1.5 Dati mancanti

**L'order book non è presente e non è archiviato.** Verificato su `data.binance.vision`:

| fonte | disponibile | note |
|---|---|---|
| spot `aggTrades` | **sì** | ~409 MB/mese per BTC |
| spot `trades` | **sì** | ~696 MB/mese per BTC |
| futures `bookDepth` | **sì** | snapshot di profondità, solo futures |
| futures `metrics` (open interest) | **sì** | piccolo |
| futures `fundingRate` | **sì** | piccolo |
| spot `bookTicker` | no | 404 sul percorso mensile spot |

`aggTrades` è la fonte realistica di microstruttura: contiene il flag "buyer is maker", da cui si
ricavano volume delta, trade flow imbalance, dimensione media degli ordini e VPIN. **Non va
archiviato in forma grezza** — 409 MB/mese × ~100 mesi × 15 simboli sono centinaia di gigabyte. Va
processato in streaming file per file, riducendolo ad aggregati per barra da 5m, e conservato solo
il derivato (poche decine di MB per simbolo).

`fundingRate` e `metrics` (open interest) sono piccoli e vale la pena scaricarli comunque: sono
feature di regime a costo quasi nullo. Sono dati futures, non spot, ma questo non è un problema —
si usano come contesto di mercato, non come strumento operativo.

---

## 2. Metodo di labeling

### 2.1 Triple-Barrier — **già implementato, da estendere**

Presente in `src/cryptofarm/ml/labeling.py` (`triple_barrier_labels`), con barriere già
parametrizzate su ATR e con pavimento legato alle fee (`barrier_widths`). Le costanti attuali:

```
TP_ATR_MULTIPLE   = 1.5      SL_ATR_MULTIPLE = 1.0
HORIZON_BARS      = 96       ROUND_TRIP_FEE  = 0.002
FEE_FLOOR_MULTIPLE = 3.0
```

Le proprietà già acquisite e da mantenere: etichetta definita su ogni candela, look-ahead che parte
da t+1 (mai la barra di ingresso), risoluzione pessimistica quando entrambe le barriere risultano
toccate nella stessa barra, coda non osservabile lasciata a HOLD.

**Modifiche da apportare:**

1. **Ripristinare `TP_ATR_MULTIPLE = 2.0`.** Il valore corrente 1.5 è stato messo in un'ultima
   iterazione e ha peggiorato il risultato: l'unica configurazione con edge lordo positivo e volume
   di operazioni decente misurata è tp/sl = 2:1 su 15m/30m/1h (+0,056% lordo su 1.066 operazioni).
2. **Barriere verticali in tempo, non in barre.** Con barre a eventi (CUSUM) `HORIZON_BARS`
   perde significato. Va sostituito da un orizzonte espresso in ore.
3. **Pavimento parametrico per regime di esecuzione.** `FEE_FLOOR_MULTIPLE = 3.0` con
   `ROUND_TRIP_FEE = 0.002` fissa un pavimento allo 0,6%, che su 5m morde l'84% delle volte —
   di fatto disattivando lo scaling su ATR. In modalità maker il pavimento scende a 0,12% e lo
   scaling torna attivo. Il regime di esecuzione va quindi reso un parametro esplicito della
   configurazione, non una costante.

### 2.2 Il vincolo economico, in tabella

Precision di break-even, `p* = (sl + f) / ((tp − f) + (sl + f))`, con barriera 2:1:

| stop-loss | take-profit | taker 0,20% RT | BNB 0,15% RT | maker 0,04% RT |
|---|---|---|---|---|
| 0,20% | 0,40% | 66,7% | 58,3% | 40,0% |
| 0,40% | 0,80% | 50,0% | 45,8% | 36,7% |
| **0,60%** | **1,20%** | **44,4%** | 41,7% | **35,6%** |
| 1,00% | 2,00% | 40,0% | 38,3% | 34,7% |
| 2,00% | 4,00% | 36,7% | 35,8% | 34,0% |

Il base rate teorico di una barriera 2:1 sotto random walk è **33,3%**. Quanto il modello deve
battere il caso:

| barriera SL | taker | BNB | maker |
|---|---|---|---|
| 0,40% | +16,7 punti | +12,5 punti | **+3,3 punti** |
| 0,60% | +11,1 punti | +8,3 punti | **+2,2 punti** |
| 1,00% | +6,7 punti | +5,0 punti | **+1,3 punti** |

Questa è la tabella più importante del documento. Con esecuzione maker e barriere ≥0,6% il compito
diventa "battere il caso di due punti percentuali", che è realistico. Con esecuzione taker e
barriere sottili è "battere il caso di sedici punti", che non lo è.

### 2.3 Meta-labeling — **raccomandato, e con una motivazione precisa**

Struttura proposta:

- **Modello primario — direzione.** Non necessariamente un modello ML: può essere una regola
  semplice (breakout del range, rottura di banda, momentum) che genera *candidati* con recall alto
  e precision bassa. Definisce il lato dell'operazione.
- **Modello secondario — eseguire o no.** Classificatore binario addestrato **solo sui candidati
  del primario**, con etichetta 1 se quel trade avrebbe chiuso in profitto netto (barriera TP
  toccata prima della SL, al netto delle fee), 0 altrimenti. Il suo output è una probabilità che
  serve sia al filtro sia al dimensionamento della posizione.

Perché conviene qui, concretamente:

1. **Il problema secondario è meglio bilanciato e meglio posto.** Sui candidati del primario il
   base rate è molto più alto del 33% di partenza, e il modello impara "questo setup regge" invece
   di "il mercato salirà", che è una domanda più facile e più stabile.
2. **Il vincolo fee entra nell'etichetta invece che in un filtro a valle.** L'etichetta del
   secondario è già definita al netto delle commissioni, quindi la precision del secondario *è* il
   win rate netto. Nessuna traduzione da fare.
3. **Risolve il problema di calibrazione che ho già incontrato.** Il modello attuale produce
   P(buy) con mediana 0,377 e p99 0,468 — una distribuzione compressa in cui nessuna soglia
   assoluta è significativa, tanto che ho dovuto passare ai quantili. Un secondario binario su
   classi bilanciate produce probabilità distribuite su tutto l'intervallo e direttamente usabili
   per il dimensionamento (Kelly frazionario o simili).
4. **Separa due domande che oggi sono confuse.** Ho misurato AUC 0,71 su una configurazione a
   orizzonte breve, salvo scoprire che veniva in gran parte dal prevedere *la volatilità* (se le
   barriere verranno toccate) e non *la direzione* (quale). Nel meta-labeling la direzione è
   responsabilità del primario e il secondario è esplicitamente un giudizio sulla qualità del
   setup: le due capacità non si mescolano più in una metrica sola.

**Trade-off:** due modelli da mantenere e validare, e il secondario eredita ogni bias del primario
(se il primario non genera mai candidati in un regime, il secondario è cieco lì). Il primario va
quindi tenuto deliberatamente permissivo.

---

## 3. Feature engineering

Notazione del rischio di leakage: **A** = calcolo puramente causale, sicuro; **B** = sicuro solo
con implementazione attenta; **C** = rischio alto.

### 3.1 Indicatori tecnici

Già presenti in `features.py`: RSI(12), ATR(6), Stocastico(12,3), TSI(25,13), volume relativo,
timeframe. Tutti normalizzati scale-free — proprietà da preservare, è ciò che rende possibile un
modello unico su 15 asset.

| feature | motivazione | leakage |
|---|---|---|
| RSI, Stocastico, TSI | oscillatori di momento già normalizzati; **da soli notoriamente non robusti** — le soglie classiche (30/70) non sopravvivono al cambio di regime. Come input a un modello che li condiziona su altro contesto restano informativi | A |
| ATR / Close | volatilità normalizzata. Doppio ruolo: feature e dimensionamento delle barriere | A |
| MACD | **da aggiungere**: cattura convergenza/divergenza fra scale, informazione non presente nel set attuale. Va normalizzato per Close come l'ATR, altrimenti dipende dalla scala del prezzo | A |
| Bande di Bollinger (%B e ampiezza) | **da aggiungere**: %B posiziona il prezzo nella distribuzione recente, l'ampiezza è un indicatore di compressione. Complementari all'ATR, che misura escursione ma non compressione | A |
| OBV | **da aggiungere con cautela**: è una somma cumulata senza reset, quindi il livello assoluto è privo di significato e cresce indefinitamente. Usare solo la variazione su finestra, mai il livello | **B** |
| Breakout di range | `POS_n` in `dataset.py` già lo approssima (posizione del Close nel range delle ultime n barre). Da estendere con distanza dal massimo/minimo di n barre in unità di ATR | A |

**Non aggiungere indicatori a raffica.** Il set attuale genera già 80 colonne dai lag. Ogni feature
in più aumenta il rischio di overfitting e il numero di configurazioni testate, che entra nel
calcolo del PBO. Meglio poche feature giustificate.

### 3.2 Microstruttura — richiede `aggTrades`

Nessuna di queste è calcolabile con i dati attualmente in archivio.

| feature | motivazione | leakage |
|---|---|---|
| Volume delta (buy − sell aggressivo) | il flag "buyer is maker" di `aggTrades` separa il volume aggressivo dai due lati. È il segnale di microstruttura più diretto sulla pressione | A |
| Trade flow imbalance | volume delta normalizzato dal volume totale della barra: scale-free, confrontabile fra simboli | A |
| Dimensione media e dispersione dei trade | distingue attività retail frammentata da flusso istituzionale | A |
| Conteggio trade / volume per trade | proxy di frammentazione dell'ordine | A |
| VPIN | probabilità di trading informato; misura consolidata, ma va calcolata su volume bucket, non su time bar | **B** |
| Spread, order book imbalance | **non disponibili per lo spot**. `bookDepth` esiste solo per i futures. Da valutare in una fase separata | — |

**Il rischio di leakage vero in questa categoria è di natura pratica, non statistica:** gli
aggregati per barra vanno calcolati usando solo i trade con timestamp `< fine barra`, e i file
mensili di `aggTrades` vanno tagliati esattamente sui confini di barra. Un errore di allineamento
di un solo trade inserisce informazione futura in ogni riga.

### 3.3 Regime di mercato

| feature | motivazione | leakage |
|---|---|---|
| Volatilità realizzata multi-scala (1h, 4h, 24h, 7g) | il rapporto fra scale distingue compressione da espansione. È il contesto che rende condizionabili gli oscillatori | A |
| Rapporto trend/range (es. ADX, efficiency ratio di Kaufman) | dice se un segnale di momento ha senso *adesso*. Un modello che non lo sa media comportamenti opposti | A |
| Volume relativo alla propria mediana mobile | già presente come `VOLUME` | A |
| Correlazione mobile con BTC | in un mercato dominato da un asset, sapere se un'alt si sta muovendo con o contro BTC è contesto reale | **B** — la finestra mobile deve essere strettamente passata; è facile calcolarla centrata per errore |
| Funding rate (futures) | posizionamento e sentiment di mercato, a costo quasi nullo. Disponibile | A |
| Open interest e sua variazione | conferma o smentisce i movimenti di prezzo (aumento di prezzo con OI in calo = ricopertura, non nuova domanda) | A |
| Dominanza BTC | regime di rotazione verso le alt | A |

### 3.4 Regola generale sul leakage

Tre errori sono i più probabili in questa pipeline, in ordine di probabilità:

1. **Rolling non causale.** Ogni `rolling`, `ewm`, `resample` deve avere la finestra chiusa a
   sinistra e etichetta a sinistra. `labeling.py` usa già il pattern `[::-1].rolling(...)[::-1]`
   per guardare *avanti* deliberatamente: quel pattern è corretto lì e sarebbe un disastro altrove.
2. **Normalizzazione fittata su tutto il dataset.** `features.py` oggi usa solo trasformazioni
   fisse e senza stato appreso, ed è una proprietà **da difendere**: qualunque scaler fittato va
   fittato dentro il fold di training, mai sull'intero dataset.
3. **Feature che incorporano il futuro attraverso l'aggregazione.** Aggregando 5m → 1h, la barra
   oraria delle 10:00 è completa solo alle 11:00. Usarla su una decisione presa alle 10:15 è
   leakage. `resample_klines` etichetta a sinistra correttamente, ma **al momento non esiste nulla
   che impedisca a un consumatore di usare la barra corrente incompleta** — va gestito
   esplicitamente con uno shift.

---

## 4. Selezione del modello

### 4.1 Il confronto è già stato fatto, su questi dati

Misurato in questa sessione, stesse sequenze e stesse etichette:

| modello | parametri | tempo | macro F1 | lift buy |
|---|---|---|---|---|
| LSTM bidirezionale, 3 strati | 747.267 | ~25 min (10 epoche) | 0,092 | 3,5× |
| HistGradientBoostingClassifier | — | **3,9 s** | **0,111** | 3,6× |

Due famiglie molto diverse convergono sullo stesso risultato: il limite era il target, non
l'architettura. Sul dataset completo (1,54 M righe × 80 feature) il gradient boosting si addestra
in **53 secondi**, con il ciclo end-to-end a 1,8 minuti.

### 4.2 Raccomandazione: **gradient boosting come modello di riferimento**

Le ragioni, nell'ordine in cui pesano:

1. **Rischio di overfitting.** Il numero di campioni *effettivamente indipendenti* è molto inferiore
   a quello nominale: con orizzonte di 96 barre, due etichette adiacenti condividono oltre il 99%
   del loro futuro. 1,5 milioni di righe con stride 12 corrispondono a un ordine di grandezza di
   decine di migliaia di osservazioni indipendenti. Su questo numero, una rete da centinaia di
   migliaia di parametri è fuori scala; un GBDT con regolarizzazione e vincolo sulle foglie no.
2. **Costo di iterazione.** Quattro secondi per fit permettono di tarare labeling, feature e
   soglie decine di volte al giorno in locale; venticinque minuti no. Finché il target non è
   stabile, iterare sul target vale più di qualunque architettura — e serve anche alla CPCV, che
   moltiplica il numero di fit per il numero di combinazioni.
3. **Costo di inferenza in produzione.** Con >10 trade/giorno e valutazione a ogni evento su 15
   simboli, un GBDT costa microsecondi per riga e non richiede TensorFlow nel processo del bot
   live. Rilevante per il deploy su Render.
4. **Robustezza sui dati tabellari.** Le feature qui sono tabellari con lag espliciti; è il regime
   in cui gli alberi con boosting sono lo stato dell'arte, non un ripiego.

**XGBoost/LightGBM contro `HistGradientBoostingClassifier`:** l'implementazione di scikit-learn è
già una dipendenza del progetto, usa lo stesso algoritmo a istogrammi di LightGBM e ha prestazioni
comparabili. LightGBM aggiungerebbe supporto nativo per il campionamento pesato per unicità e
callback più ricchi. **Raccomandazione: restare su scikit-learn** finché non serve una funzionalità
specifica; una dipendenza in meno vale più di un margine di prestazione.

### 4.3 Quando riconsiderare le architetture sequenziali

Non ora, ma con due precondizioni chiare: **dopo** che il target è stabile e un GBDT dimostra
expectancy netta positiva out-of-sample, e **solo** se le feature di microstruttura sono in gioco
(il flusso di ordini ha struttura sequenziale fine che i lag tabellari comprimono male).

In quel caso il candidato non è l'LSTM ma la **TCN** (convoluzioni causali dilatate): stesso campo
recettivo, parallela sui timestep, molto più veloce su CPU — è già disponibile in `models.py` come
`kind="cnn"`. Il Transformer è da escludere a questa scala di dati indipendenti: la sua efficienza
campionaria è peggiore, non migliore.

---

## 5. Validazione

### 5.1 Cosa c'è oggi e perché non basta

`dataset.py` implementa `time_split`: taglio cronologico unico all'80% con embargo, **globale su
tutti i simboli** (correttamente: le criptovalute sono troppo correlate perché uno split per
simbolo abbia senso). Rispetto al punto di partenza è già molto meglio, ma resta insufficiente per
tre ragioni:

1. **Una sola stima.** Nessuna nozione della varianza della performance.
2. **Un solo regime.** La validation attuale (dal 2024-12-31) cade in un periodo per oltre il 60%
   laterale.
3. **Nessuna difesa contro il backtest overfitting.** Stiamo già testando molte configurazioni
   (orizzonte, rapporto barriere, ampiezza, soglia): senza correzione, la migliore è selezionata
   anche sul rumore.

### 5.2 Purged K-Fold con embargo — **minimo indispensabile**

Ogni etichetta triple-barrier ha un intervallo di vita `[t, t_exit]`. Un fold di training deve
escludere ogni osservazione il cui intervallo si sovrappone a quello di una qualsiasi osservazione
di test (*purging*), e va aggiunto un embargo temporale dopo il test set per neutralizzare
l'autocorrelazione seriale.

Dimensionamento: l'embargo deve coprire l'orizzonte massimo dell'etichetta sul timeframe più lungo.
`trainer.py` già lo calcola così (`longest × horizon`), ed è la logica giusta da riportare nella
nuova validazione. Il purging invece **richiede il tempo di uscita effettivo di ogni etichetta**,
che oggi `triple_barrier_labels` non restituisce: restituisce solo la classe. **È la prima
modifica strutturale necessaria** — senza `t_exit` il purging non è implementabile.

### 5.3 CPCV — **obiettivo**

Con N gruppi temporali e k gruppi di test per combinazione si ottengono `C(N, k)` split, ciascuno
purged ed embargato, e quindi una *distribuzione* di performance out-of-sample invece di un punto.
Configurazione ragionevole: N = 8–10 gruppi su 9 anni (quindi ~1 anno per gruppo, che rispetta i
cicli di mercato), k = 2, per 28–45 combinazioni.

Costo: 28–45 fit. A 53 s per fit sul dataset completo sono **25–40 minuti** — perfettamente
sostenibile in locale con il GBDT, e completamente fuori portata con una rete neurale. È un altro
argomento a favore della scelta del modello.

### 5.4 Walk-forward — **verifica finale, non metrica primaria**

Rolling o expanding, riaddestrando periodicamente, come simulazione di come il sistema verrebbe
davvero operato. Serve a intercettare la degradazione del modello nel tempo (quanto rapidamente
scade un modello addestrato su dati fino a t?), che la CPCV non misura perché mescola i periodi.
Da usare come conferma di realismo operativo sulla configurazione finale, non per selezionare.

### 5.5 Metriche anti-overfitting

**PBO (Probability of Backtest Overfitting).** Con la CPCV già in piedi, il PBO si calcola per
combinazione: si sceglie la configurazione migliore in-sample e si misura la sua posizione nella
distribuzione out-of-sample. Il PBO è la frazione di combinazioni in cui la migliore in-sample
finisce sotto la mediana out-of-sample. **PBO > 0,5 significa che la procedura di selezione è
peggiore di una scelta casuale** e va riportato accanto a ogni risultato.

**Deflated Sharpe Ratio.** Testando molte configurazioni, il massimo Sharpe osservato è distorto
verso l'alto anche se nessuna ha edge. Il DSR corregge per il numero di prove, per l'asimmetria e
la curtosi dei rendimenti (rilevanti: i rendimenti di questa strategia sono asimmetrici per
costruzione, TP e SL non sono simmetrici) e per la lunghezza della serie. **Va calcolato tenendo il
conto onesto di *tutte* le configurazioni provate**, incluse quelle scartate per strada — è
l'errore più comune nell'applicarlo.

**Regola vincolante:** ogni metrica riportata (WR, expectancy, Sharpe, max drawdown) va calcolata
al netto delle fee simulate e dello slippage, e solo su split mai usati né in training né nella
selezione degli iperparametri. `evaluate.py` già separa `atteso_lordo` da `atteso_per_trade` e
include `fee_sensitivity`: quella separazione va mantenuta, perché distingue "il modello non
prevede nulla" da "l'edge c'è ma non copre i costi", due diagnosi con rimedi opposti.

**Slippage — oggi non modellato affatto.** Le fee sono nel calcolo, lo slippage no. Per ordini
maker lo slippage rilevante non è il costo di attraversare lo spread ma il **rischio di mancato
riempimento**: se il prezzo si muove contro, l'ordine limite non viene eseguito e si perde
l'occasione; se si muove a favore, viene eseguito proprio quando non conviene (selezione avversa).
Una simulazione maker che assume riempimento certo sovrastima sistematicamente. **Va modellata
prima di dichiarare valido qualunque risultato in modalità maker** — ed è particolarmente critico
perché tutta l'analisi economica sopra indica proprio la modalità maker come la via praticabile.

---

## 6. Piano di modifica proposto per il codice

In ordine di priorità. **Da discutere, non da eseguire ora.**

### Priorità 1 — sbloccano la validazione corretta

| # | intervento | file | perché prima |
|---|---|---|---|
| 1.1 | `triple_barrier_labels` restituisce anche `t_exit` (tempo di uscita effettivo) e la barriera toccata | `ml/labeling.py` | senza il tempo di uscita il purging non è implementabile: blocca tutto il resto |
| 1.2 | Nuovo modulo `ml/validation.py`: `PurgedKFold`, `CombinatorialPurgedCV`, `sample_uniqueness_weights` | nuovo | è il punto 5, il requisito critico |
| 1.3 | Ripristinare `TP_ATR_MULTIPLE = 2.0`; rendere il regime di esecuzione (taker/BNB/maker) un parametro esplicito che pilota `ROUND_TRIP_FEE` e il pavimento | `ml/labeling.py` | correzione di una regressione già identificata, costo nullo |

### Priorità 2 — sbloccano l'expectancy positiva

| # | intervento | file | perché |
|---|---|---|---|
| 2.1 | Campionamento a eventi con filtro CUSUM, soglia in multipli di σ | `ml/dataset.py` | ~14,7 eventi/giorno a 5×σ, allineati al target di frequenza; riduce il rumore delle time-bar |
| 2.2 | Modello di riempimento per ordini limite (probabilità di fill, selezione avversa) | `ml/evaluate.py` | tutta l'analisi indica la modalità maker come la sola praticabile: senza questo modello i suoi numeri non sono credibili |
| 2.3 | Meta-labeling: primario direzionale + secondario binario su etichette già nette da fee | `ml/labeling.py`, `ml/models.py` | separa direzione da qualità del setup; produce probabilità calibrate e usabili per il dimensionamento |
| 2.4 | Simulazione a portafoglio: 15 simboli in parallelo con vincolo di capitale | nuovo `ml/portfolio.py` | è così che si raggiungono >10 trade/giorno senza barriere sottili |

### Priorità 3 — nuove fonti di informazione

| # | intervento | file | perché |
|---|---|---|---|
| 3.1 | Scaricare `fundingRate` e `metrics` (open interest) | `data/klines.py` o nuovo `data/derivatives.py` | file piccoli, feature di regime a costo quasi nullo |
| 3.2 | Ingestione `aggTrades` in streaming → aggregati per barra da 5m (volume delta, flow imbalance, dimensione trade) | nuovo `data/microstructure.py` | unica microstruttura disponibile per lo spot; **mai archiviare i grezzi** |
| 3.3 | Aggiungere MACD, %B e ampiezza di Bollinger, variazione OBV, ADX/efficiency ratio, correlazione mobile con BTC | `ml/features.py` | poche feature giustificate, non una raffica |

### Priorità 4 — dopo che l'expectancy è positiva

| # | intervento | file |
|---|---|---|
| 4.1 | PBO e Deflated Sharpe Ratio, con conteggio onesto delle configurazioni provate | `ml/evaluate.py` |
| 4.2 | Walk-forward come verifica finale | `ml/validation.py` |
| 4.3 | Valutare la TCN, solo se la microstruttura è in gioco | `ml/models.py` (già presente come `kind="cnn"`) |

### Cosa **non** va toccato

- La proprietà scale-free di `features.py` e l'assenza di stato appreso.
- Lo split temporale globale (mai per simbolo).
- La separazione fra edge lordo e netto in `evaluate.py`.
- La corrispondenza fra le barriere che definiscono le etichette e quelle che governano le uscite
  in `signals.py`: è ciò che rende il P&L simulato la traduzione diretta del win rate misurato.

---

## 7. Aspettative realistiche

Stima onesta, basata su quanto misurato in questa sessione e non su quanto sarebbe desiderabile.

### Win rate

**Misurato oggi:** win rate 39,6% contro un break-even di 47,3%, AUC 0,54, edge lordo +0,017% per
operazione. Verificato end-to-end sul periodo mai visto: 32–44% per coppia, coerente.

**Realistico dopo gli interventi di priorità 1–2:** **40–45%** con barriera 2:1. Il ragionamento:
il base rate teorico è 33,3%; un modello con AUC 0,55–0,60 sul decile superiore porta la precision
a circa 40–45%. Andare oltre richiederebbe un AUC che su prezzo e indicatori non si osserva.

**Con microstruttura (priorità 3):** forse **45–50%**. È la stima più incerta del documento, perché
non ho misurato nulla su `aggTrades`.

**Un WR >70% in-sample su questo dominio va trattato come un difetto fino a prova contraria.** I
sospetti da verificare, nell'ordine: normalizzazione fittata fuori dal fold, rolling non causale,
uso di una barra aggregata incompleta, purging assente (etichette sovrapposte fra train e test).

### Frequenza

| configurazione | trade/giorno | note |
|---|---|---|
| Oggi (soglia 0,484, 15m, 1 simbolo) | **~0,2** | 99 trade in 20 mesi su BTC 15m |
| Time-bar, taker, barriera 0,6%, 1 simbolo | 1–2 | tetto fisico 6,4 |
| Time-bar, taker, barriera 0,6%, **15 simboli** | **15–30** | **obiettivo raggiunto** |
| CUSUM 5×σ, 15 simboli, filtro selettivo | 20–50 | dipende dalla selettività del meta-modello |
| Maker, barriera 0,08%, 1 simbolo | fino a 96 | tetto teorico, non raccomandato: la selezione avversa non è modellata |

**Il target di >10 trade/giorno è raggiungibile, ma attraverso il portafoglio.** Cercarlo su un
simbolo solo obbliga a barriere che le fee mangiano.

### Expectancy netta

È il criterio vero, e la risposta onesta è che **dipende quasi interamente dal regime di
esecuzione**:

| scenario | WR atteso | break-even (barriera 0,6%) | esito |
|---|---|---|---|
| Taker, WR 42% | 42% | 44,4% | **negativo** |
| Taker + BNB, WR 42% | 42% | 41,7% | **marginalmente positivo** |
| Maker, WR 42% | 42% | 35,6% | **positivo con margine** |
| Maker, WR 40% (conservativo) | 40% | 35,6% | **positivo** |

**La conclusione da portare nella fase di implementazione: l'esecuzione con ordini limite non è
un'ottimizzazione, è la precondizione.** Con ordini a mercato la strategia richiede una capacità
predittiva che non ho osservato in nessuna configurazione testata; con ordini limite richiede un
margine di due punti sul caso, che è alla portata. Di conseguenza **il modello di riempimento
(intervento 2.2) è la singola cosa più importante da costruire dopo la validazione** — senza,
tutti i numeri della colonna "maker" restano non verificati.

---

## Riproducibilità

| misura | comando |
|---|---|
| Copertura dello store | `python -m cryptofarm.data.klines --manifest` |
| Regimi, fattibilità, dollar bar, CUSUM | `scratchpad/feasibility.py` |
| Confronto LSTM / GBDT | `scratchpad/diagnose.py` |
| Sweep configurazioni di labeling | `scratchpad/sweep_labeling.py`, `scratchpad/sweep_barriers.py` |
| Verifica segnali end-to-end | `scratchpad/check_signals.py` |

Gli script della scratchpad sono di sessione: vanno spostati sotto `scripts/` se le misure devono
restare riproducibili nel tempo.
