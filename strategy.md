# Strategia di training — analisi e decisioni

Documento di riferimento per le fasi di sviluppo successive. Contiene l'analisi, le scelte
motivate e i trade-off espliciti.

Ogni numero qui riportato è misurato sui dati effettivamente presenti nel progetto, non stimato.
Gli script di misura stanno nella scratchpad di sessione; i comandi per riprodurli sono indicati
dove serve.

---

### Revisione 2 — cosa è cambiato e perché

**Target operativo aggiornato: ~4 trade/giorno/simbolo** (≈60/giorno su 15 simboli). È un vincolo
di progetto: barriere e campionamento si scelgono per rispettarlo.

Correzioni apportate, in ordine di gravità:

| # | cosa | perché |
|---|---|---|
| 1 | **§1.3 era distorta.** Il tempo mediano al target era calcolato con `np.nanmedian` sui soli casi che il target lo raggiungevano entro 24h — quindi condizionato al successo, e sottostimato del 25–110%. Ricalcolato con i censurati trattati esplicitamente (§1.3) | sovrastimava la capacità di frequenza |
| 2 | **§2.2 usava una formula di break-even incompleta**, che assume chiusura sempre su una barriera di prezzo. Sostituita con l'expectancy misurata sulla distribuzione completa degli esiti, timeout inclusi al loro rendimento reale (§2.2) | il timeout esiste e non è a rendimento nullo |
| 3 | **Il "base rate teorico 33,3%" era teorico.** Sostituito dal base rate **misurato**: 32,1–32,6% (§2.2) | il mercato reale non è una random walk, e la differenza va nella direzione sfavorevole |
| 4 | **Aggiunta §1.5, la tabella di capacità**, con `t_exit` misurato per configurazione di barriere e confronto esplicito con una random walk driftless | è la tabella che vincola tutto il resto |
| 5 | **§1.4 misurava il CUSUM solo su BTC.** Esteso a tutti i 15 simboli, con la soglia per simbolo (§1.4) | serviva per verificare la selettività implicata |
| 6 | **Riformulata la conclusione sulla frequenza** (sintesi esecutiva): non è un limite fisico del mercato, è un limite della commissione | la formulazione precedente era fuorviante |
| 7 | Aggiunte **Parte B** (dove serve il reinforcement learning) e **Parte C** (pattern strutturali), e §6 riscritta come **piano a fasi con gate espliciti** | richiesti dalla revisione |

**Cosa resta valido e non è stato toccato:** l'impianto metodologico (triple-barrier, CPCV con
purging ed embargo, meta-labeling, vincolo economico come criterio primario), l'analisi dei dati
di §1.1–1.2, il set di feature di §3, la scelta del modello di §4 e la struttura di validazione
di §5.

---

## Sintesi esecutiva

Tre conclusioni, in ordine di importanza.

**1. La modalità di esecuzione domina ogni altra scelta.** Con barriera 2:1 e stop allo 0,60%,
la precision di break-even misurata sulla distribuzione completa degli esiti è **44,4%** con
commissioni taker e **35,4%** con commissioni maker, contro un win rate di base **misurato** del
**32,3%**. Il modello deve quindi battere il caso di **12,1 punti** con ordini a mercato e di
**3,1 punti** con ordini limite. Alle barriere strette richieste dal target di frequenza il
divario taker sale a 23,2 punti. Nessun lavoro su feature o architettura sposta questo rapporto.

**2. Il target di 4 trade/giorno/simbolo è raggiungibile, e il limite non è la velocità del
mercato ma la commissione.** Misurando il tempo di uscita reale del triple-barrier (non il tempo
al target, che è un'altra cosa): con barriere 0,60%/0,30% l'holding medio è 1,07 ore, il tetto è
22,3 trade/giorno e per farne 4 basta stare in mercato il **18%** del tempo. Anche a 1,20%/0,60%
il tetto resta 6,9/giorno. **Movimenti sfruttabili in 1–3 ore esistono in abbondanza: ciò che
manca è il margine per pagarli.** A 0,60%/0,30% il divario da colmare è di 23,2 punti percentuali
in modalità taker e di 5,0 punti in modalità maker.

Ne segue la conclusione operativa centrale: **la frequenza richiesta e l'esecuzione con ordini
limite sono lo stesso vincolo.** Quattro trade al giorno per simbolo con ordini a mercato non è un
obiettivo ambizioso, è aritmeticamente fuori portata. Con ordini limite è alla portata di un
modello che batta il caso di 3–5 punti.

**3. Il collo di bottiglia attuale non è il modello.** Un LSTM da 747k parametri e un gradient
boosting hanno prodotto lo stesso risultato sulle stesse etichette (macro F1 0,092 contro 0,111,
in 25 minuti contro 3,9 secondi). L'ultimo modello addestrato ha AUC 0,54 ed edge lordo +0,017%
per operazione. Le leve che restano sono, in ordine: esecuzione maker, campionamento a eventi,
dati di microstruttura. Il tuning architetturale è l'ultima, non la prima.

**Conseguenza sulla priorità: il modello di riempimento degli ordini limite sale a priorità 1,
insieme alla validazione.** Non è un dettaglio di simulazione: tutta la strategia vive in modalità
maker, quindi senza un modello di fill credibile *nessun numero della strategia è verificabile*.

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

### 1.3 Tempo al target — **misura corretta**

> **Correzione.** La versione precedente di questa tabella calcolava la mediana con
> `np.nanmedian` sui soli casi che il target lo raggiungevano entro 24 ore, lasciando gli altri a
> `NaN`. Era quindi una **mediana condizionata al successo**, sottostimata, che sovrastimava la
> capacità di frequenza. Sotto ci sono entrambe le versioni, per rendere visibile l'entità
> dell'errore.

Misura su 5 simboli (BTC, ETH, SOL, XRP, LINK), 5m dal 2022-01-01. La mediana reale usa
Kaplan-Meier con i casi non raggiunti trattati come censurati.

| target | P(raggiunto entro 24h) | mediana condizionata (h) | **mediana reale (h)** | errore |
|---|---|---|---|---|
| 0,30% | 90,6% | 0,67 | **0,83** | +24% |
| 0,40% | 87,5% | 1,08 | **1,42** | +31% |
| 0,60% | 81,5% | 1,92 | **3,00** | +56% |
| 1,00% | 70,1% | 3,67 | **7,75** | +111% |

L'errore cresce con l'ampiezza del target, perché cresce la quota di casi censurati. Nessuna
mediana risulta però censurata via del tutto: anche all'1% il 70,1% dei casi raggiunge il target
entro 24 ore, quindi la mediana esiste ed è finita.

**Ma questa non è la tabella giusta per calcolare la capacità in trade/giorno.** Il tempo al
target non è il tempo di detenzione: con barriere 2:1 la maggior parte dei trade chiude sullo
stop, che è più vicino e quindi più veloce. La misura corretta è §1.5.

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

**Filtro CUSUM** — fattibile e, a mio avviso, la scelta migliore. Misurato su **tutti i 15
simboli**, 5m dal 2022 (σ = deviazione standard dei log-return su finestra di 24 ore, stimata per
simbolo):

| simbolo | σ 5m | eventi/gg a 3σ | eventi/gg a 5σ |
|---|---|---|---|
| BTCUSDT | 0,128% | 30,3 | 14,8 |
| ETHUSDT | 0,169% | 30,2 | 14,9 |
| BNBUSDT | 0,144% | 32,0 | 15,8 |
| SOLUSDT | 0,238% | 33,0 | 16,2 |
| XRPUSDT | 0,186% | 33,3 | 16,6 |
| ADAUSDT | 0,218% | 33,2 | 16,2 |
| DOGEUSDT | 0,223% | 32,4 | 16,1 |
| AVAXUSDT | 0,240% | 32,7 | 16,1 |
| LINKUSDT | 0,232% | 32,9 | 15,8 |
| DOTUSDT | 0,220% | 32,1 | 15,6 |
| LTCUSDT | 0,194% | 32,2 | 15,7 |
| TRXUSDT | 0,093% | 35,3 | 18,6 |
| ATOMUSDT | 0,215% | 34,7 | 16,9 |
| NEARUSDT | 0,280% | 33,2 | 16,9 |
| UNIUSDT | 0,251% | 32,2 | 15,4 |

**Il risultato più utile è l'uniformità: a 3σ tutti e 15 i simboli producono 30–35 eventi/giorno**,
nonostante σ vari di un fattore 3 fra TRX (0,093%) e NEAR (0,280%). È esattamente ciò che ci si
aspetta da una soglia normalizzata sulla volatilità, e significa che **non serve calibrazione per
simbolo**: k = 3,0 va bene per undici simboli, k = 3,5 porta gli altri quattro (XRP, ADA, TRX,
ATOM, NEAR) nella stessa fascia 27–29/giorno.

**Selettività implicata.** Con ~30 eventi/giorno e un target di 4 trade/giorno/simbolo, il
meta-modello deve accettare il **13,3% degli eventi** — una selezione da decile superiore, che è
un punto di lavoro ragionevole per un classificatore calibrato e coerente con la §5.2 di
`evaluate.py` (lo sweep per quantili è già costruito per esprimere esattamente questa scelta).

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

### 1.5 Capacità: holding time reale e tetto di trade/giorno — **la tabella che vincola tutto**

Misura del triple-barrier first-touch sui dati reali (5 simboli, 5m dal 2022): per ogni
configurazione, la distribuzione degli esiti e il tempo di uscita effettivo `t_exit`.

| TP / SL | orizzonte | P(TP) | P(SL) | P(timeout) | E[r &#124; timeout] | holding medio | tetto trade/gg | in mercato per 4/gg |
|---|---|---|---|---|---|---|---|---|
| 0,60% / 0,30% | 8h | 31,9% | 66,1% | 2,0% | +0,111% | **1,07 h** | **22,3** | **18%** |
| 0,80% / 0,40% | 8h | 30,8% | 64,8% | 4,4% | +0,144% | 1,62 h | 14,8 | 27% |
| 1,00% / 0,50% | 12h | 30,8% | 65,1% | 4,1% | +0,179% | 2,40 h | 10,0 | 40% |
| 1,20% / 0,60% | 24h | 31,6% | 66,3% | 2,0% | +0,214% | 3,50 h | 6,9 | 58% |

**Tutte e quattro le configurazioni sostengono 4 trade/giorno/simbolo**, con un tempo in mercato
fra il 18% e il 58%. Il vincolo di frequenza da solo non esclude nessuna barriera fino all'1,2%.

#### Confronto con una random walk driftless

La tabella di riferimento fornita nella revisione veniva da una simulazione first-touch su random
walk con σ(5m) = 0,121%. Il confronto con la misura reale:

| TP / SL | orizzonte | holding random walk | holding reale | rapporto | P(TP) random walk | P(TP) reale |
|---|---|---|---|---|---|---|
| 0,60% / 0,30% | 8h | 1,43 h | 1,07 h | **0,75×** | 35,5% | **31,9%** |
| 0,80% / 0,40% | 8h | 2,29 h | 1,62 h | **0,71×** | 33,9% | **30,8%** |
| 1,00% / 0,50% | 12h | 3,40 h | 2,40 h | **0,70×** | 33,9% | **30,8%** |
| 1,20% / 0,60% | 24h | 4,85 h | 3,50 h | **0,72×** | 34,6% | **31,6%** |

**I miei numeri differiscono da quelli di riferimento in modo sistematico e nella stessa direzione,
e la spiegazione è una sola.** Il mercato reale tocca le barriere circa il **30% più in fretta**
della random walk e centra il take-profit circa **3 punti meno spesso**. Entrambe le deviazioni
sono la firma delle code grasse e del clustering di volatilità: i rendimenti a 5m non sono
gaussiani, quindi movimenti abbastanza grandi da toccare una barriera arrivano prima di quanto la
gaussiana preveda, e la barriera più vicina — lo stop, a metà distanza — ne beneficia in modo
sproporzionato.

Ne segue una lettura precisa: **la capacità è migliore di quanto la random walk suggerisca, e
l'economia è peggiore.** Sono due conseguenze dello stesso fenomeno. La differenza sul tetto di
trade/giorno (22,3 contro 17,8 nel riferimento) è a favore; quella sul P(TP) (31,9% contro 35,5%)
è a sfavore, e pesa di più perché entra direttamente nel break-even.

Un secondo motivo di differenza, minore: σ = 0,121% è la mediana di BTC, mentre la misura reale
mescola cinque simboli fra cui alcune alt con σ fino al doppio. Ripetendo su BTC solo, il rapporto
di holding sale a ~0,85×, quindi circa metà dello scarto viene dalle code e metà dalla
composizione del campione.

#### Posizioni concorrenti a portafoglio

Con 4 trade/giorno/simbolo su 15 simboli, ingressi simulati come processo di Poisson e holding
dalla tabella sopra:

| TP / SL | posizioni medie | picco mediano | picco 99° percentile |
|---|---|---|---|
| 0,60% / 0,30% | 2,7 | 13 | **15** |
| 0,80% / 0,40% | 4,0 | 16 | **19** |
| 1,00% / 0,50% | 6,0 | 20 | **23** |
| 1,20% / 0,60% | 8,8 | 25 | **28** |

**Il picco è 4–5 volte la media.** Dimensionare il capitale sulla media significa non poter aprire
metà delle posizioni proprio nei momenti in cui il modello vede più occasioni. E il numero reale è
peggiore di così: la simulazione assume arrivi indipendenti fra simboli, mentre le criptovalute
sono fortemente correlate e i segnali arrivano in raffiche sincronizzate. **Il picco misurato qui è
un limite inferiore.**

### 1.6 Dati mancanti

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

### 2.2 Il vincolo economico — **expectancy misurata, non formula analitica**

> **Correzione.** La versione precedente usava `p* = (sl + f) / ((tp − f) + (sl + f))`, che
> assume che ogni trade chiuda su una barriera di prezzo. Non è vero: le uscite sulla barriera
> temporale chiudono a mercato, con un rendimento qualunque fra −sl e +tp. Inoltre il base rate
> usato (33,3%) era il valore teorico da random walk, non quello misurato.

La forma corretta include i timeout al loro rendimento reale:

```
E[netto] = P(TP)·(tp − f) + P(SL)·(−sl − f) + P(timeout)·(E[r | timeout] − f)
```

Il **win rate** è definito come la quota di trade risolti su barriera di prezzo che chiudono in
take-profit, `P(TP) / (P(TP) + P(SL))`. Il break-even è il valore di quel win rate che annulla
`E[netto]`, tenendo fissi `P(timeout)` e `E[r | timeout]` misurati.

Con i valori di §1.5 (5 simboli, 5m dal 2022):

| TP / SL | WR misurato | taker 0,20% | BNB 0,15% | maker 0,04% |
|---|---|---|---|---|
| 0,60% / 0,30% | 32,6% | be 55,8% (**+23,2 pt**) · E −0,204% | be 50,1% (+17,5 pt) · E −0,154% | be 37,6% (**+5,0 pt**) · E −0,044% |
| 0,80% / 0,40% | 32,2% | be 50,2% (+18,0 pt) · E −0,207% | be 45,9% (+13,7 pt) · E −0,157% | be 36,3% (**+4,1 pt**) · E −0,047% |
| 1,00% / 0,50% | 32,1% | be 46,7% (+14,6 pt) · E −0,211% | be 43,3% (+11,2 pt) · E −0,161% | be 35,6% (**+3,5 pt**) · E −0,051% |
| 1,20% / 0,60% | 32,3% | be 44,4% (+12,1 pt) · E −0,214% | be 41,6% (+9,3 pt) · E −0,164% | be 35,4% (**+3,1 pt**) · E −0,054% |

`E` è l'expectancy netta per operazione **senza alcuna capacità predittiva** — cioè entrando a
caso. È il punto di partenza che il modello deve ribaltare.

**Tre osservazioni, in ordine di importanza:**

1. **Il divario in modalità maker è di 3,1–5,0 punti, non di 0,8–2,0.** La tabella di riferimento
   fornita nella revisione era ottimista, per due ragioni sommate: usava il base rate teorico
   (33,3%) invece di quello misurato (32,1–32,6%), e ignorava il termine di timeout. La differenza
   è di circa 2–3 punti, che su un divario di questa dimensione è metà del problema.
2. **Il termine di timeout è piccolo ma non nullo.** `P(timeout)` è fra il 2,0% e il 4,4% — molto
   meno di quanto il sospetto della revisione suggerisse — e `E[r | timeout]` è **positivo**
   (+0,11% … +0,21%). La formula analitica era quindi una buona approssimazione, e sbagliava nella
   direzione *pessimista*. Va comunque sostituita: il fatto che l'errore fosse piccolo su questa
   configurazione non garantisce che lo resti su orizzonti più stretti, dove la quota di timeout
   cresce rapidamente.
3. **Il divario si stringe con barriere più larghe, ma la frequenza no.** Passando da 0,60% a
   1,20% di take-profit il divario maker scende da 5,0 a 3,1 punti, e il tetto di trade/giorno da
   22,3 a 6,9 — che resta sopra il target di 4. **La configurazione 1,20%/0,60% è quindi
   dominante sotto entrambi i vincoli** e va presa come punto di partenza, non 0,60%/0,30%.

**Nota sull'interpretazione del vincolo "target ≥ 2× fee round-trip".** In modalità maker il
round-trip è 0,04% e il vincolo imporrebbe un target minimo dello 0,08%, che sarebbe soddisfatto
da qualunque barriera qui. Il vincolo non è quindi binding: a vincolare è il **divario di win
rate**, che la tabella sopra rende esplicito e che il vincolo sulle fee non cattura.

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
3. **Costo di inferenza in produzione.** Con ~60 trade/giorno a portafoglio e valutazione a ogni evento su 15
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

## 6. Reinforcement learning: dove serve davvero

### 6.1 La tesi, verificata con un conteggio

La tesi della revisione — il RL è debole sulla generazione di alfa e forte sull'esecuzione — è
**confermata**, e il modo più netto per mostrarlo è contare le osservazioni indipendenti
disponibili per ciascun problema. Il RL deve stimare non solo una mappa stato→etichetta ma anche
la dinamica delle conseguenze, e ha quindi un'efficienza campionaria **peggiore** del
supervisionato sullo stesso numero di campioni. Quel numero cambia però di ordini di grandezza a
seconda di dove lo si applica.

Periodo di riferimento: 2022-01-01 → 2026-08, cioè 1.691 giorni × 15 simboli = **25.365
simbolo-giorni**.

| problema | cos'è un episodio | durata | episodi indipendenti disponibili |
|---|---|---|---|
| **Generazione di alfa** (direzione) | una decisione di ingresso, il cui esito si risolve nell'orizzonte dell'etichetta | 8–24 h | 25.000 – 76.000 |
| **Esecuzione** (piazzamento/riprezzamento del limite) | un ordine da piazzare, riprezzare o abbandonare | minuti | **~756.000** (30 eventi CUSUM/gg × 25.365) |
| **Gestione dell'uscita** | una posizione aperta da gestire fino alla chiusura | 1–3,5 h | 25.000 – 76.000 |
| **Dimensionamento a portafoglio** | una giornata di allocazione sotto vincolo di capitale | 1 giorno | **1.691** |

**Il problema di esecuzione ha circa trenta volte i dati del problema di alfa**, e i suoi episodi
sono quasi indipendenti fra loro (l'esito di un piazzamento si risolve in minuti, quindi due
episodi consecutivi non condividono futuro — a differenza delle etichette di trading, che si
sovrappongono pesantemente). È la ragione quantitativa, non stilistica, per cui l'esecuzione è il
punto giusto in cui far entrare il RL.

### 6.2 Raccomandazione: **esecuzione degli ordini limite, come primo e unico candidato ora**

**Perché per primo.** Tre ragioni che si sommano:

1. **È il pezzo che manca ed è bloccante.** Tutta la strategia vive in modalità maker (§2.2), e in
   modalità maker la domanda "dove metto il limite e quanto aspetto" *è* la strategia. Oggi il
   progetto non ha nulla in quel punto: `signals.py` assume implicitamente riempimento al prezzo di
   chiusura.
2. **Ha i dati.** ~756.000 episodi quasi indipendenti, contro le decine di migliaia degli altri.
3. **Il suo risultato è verificabile in produzione a costo basso.** Una politica di esecuzione si
   può confrontare con la baseline (limite al mid, timeout fisso) misurando il prezzo di
   riempimento effettivo, senza dover attendere che una strategia direzionale maturi. È l'unico dei
   tre candidati che dà un segnale di validità in giorni invece che in mesi.

**Perché la gestione dell'uscita viene dopo.** È un vero problema sequenziale che il labeling
supervisionato non sa esprimere — la triple-barrier fissa TP e SL all'ingresso e non li tocca più —
ma ha lo stesso numero di episodi del problema di alfa (25k–76k) e un difetto aggiuntivo: **una
politica di uscita appresa può migliorare l'expectancy solo se l'expectancy di partenza non è
troppo negativa.** Oggi è −0,044% per operazione anche in modalità maker (§2.2). Ottimizzare
l'uscita di una strategia che entra a caso è ottimizzare il modo di perdere. Ha senso quando la
Fase 2 ha prodotto expectancy positiva.

**Perché il dimensionamento a portafoglio viene per ultimo.** 1.691 episodi giornalieri sono
pochissimi per il RL, e il problema è per giunta quello con la maggiore dimensionalità dell'azione
(15 pesi simultanei sotto vincolo di capitale). In compenso è il problema con le alternative non-RL
migliori: il dimensionamento tramite Kelly frazionario sulla probabilità calibrata del
meta-modello, con un tetto sull'esposizione, è quasi certamente sufficiente. **Da valutare solo se
si dimostra che l'allocazione a regola fissa lascia sul tavolo qualcosa di misurabile.**

### 6.3 Vincoli non negoziabili

Sono i vincoli della revisione, che accetto integralmente, con un'annotazione su ciascuno:

- **Offline, mai online.** Il distribution shift va gestito esplicitamente: la politica proporrà
  piazzamenti che nei dati storici non compaiono, e stimarne il valore è extrapolazione.
  Contromisura minima: vincolo di prossimità alla politica di comportamento (BCQ/CQL o un
  penalty term), più un intervallo di confidenza sulle azioni fuori supporto che va **riportato**,
  non nascosto.
- **Il simulatore è il collo di bottiglia, non l'algoritmo.** Un agente addestrato su un
  simulatore con riempimento certo impara a sfruttare un'assunzione falsa, e lo fa benissimo. **Il
  modello di fill va costruito e validato contro `aggTrades` reali prima di qualunque
  addestramento RL.** Se non è pronto, il RL non parte — è un gate, non una raccomandazione.
- **Reward al netto dei costi reali.** Fee per lato corretto a seconda di come l'ordine si è
  riempito, mancato riempimento come costo opportunità esplicito, slippage sulle uscite a mercato.
  Una valutazione a fee piatta sovrastima sistematicamente, e in modalità maker sovrastima
  *proprio la variabile che si sta ottimizzando*.
- **Reward risk-adjusted.** Sharpe differenziale o utilità con penalizzazione del drawdown. Il PnL
  grezzo produce politiche che accettano code di rischio arbitrarie, e in esecuzione questo si
  manifesta come "aspetta indefinitamente un riempimento migliore".
- **Stessa validazione del resto: CPCV con purging ed embargo.** Un agente valutato su un solo
  split non è valutato. L'overfitting a un regime è l'esito di default.
- **Baseline obbligatoria.** Per l'esecuzione: limite al mid con timeout fisso. Per l'uscita: la
  triple-barrier statica. Se non batte la regola fissa sulla **distribuzione** CPCV — non sulla
  media di uno split — non entra.

---

## 7. Pattern strutturali (ABC, ABCD, onde)

### 7.1 Il problema bloccante è reale, e ora è quantificato

Un pivot non è conoscibile quando si forma: lo zigzag lo colloca retroattivamente sull'estremo
esatto, mentre in tempo reale diventa noto solo quando il ritracciamento raggiunge la soglia.
Misura del **ritardo di conferma** (barre fra l'estremo e la barra in cui il pivot è conoscibile),
15 simboli dal 2022:

| timeframe | soglia | pivot totali | ritardo mediano | p90 | p99 |
|---|---|---|---|---|---|
| 15m | 0,5% | 702.771 | 1 | 2 | 8 |
| 15m | 1,0% | 305.823 | 1 | 6 | 21 |
| 15m | 2,0% | 76.412 | 4 | 19 | **65** |
| 15m | 3,0% | 16.202 | 8 | 36 | **101** |
| 1h | 1,0% | 147.610 | 1 | 2 | 7 |
| 1h | 2,0% | 56.247 | 1 | 6 | 19 |
| 1h | 3,0% | 14.512 | 2 | 9 | 26 |

**Il ritardo mediano è piccolo (1–8 barre) ma la coda è pesante**: al 99° percentile si arriva a
101 barre su 15m al 3%, cioè oltre 25 ore. Una pipeline che usa i pivot "come disegnati" inserisce
quindi un look-ahead che nella maggior parte dei casi vale poco e in una frazione non trascurabile
vale un giorno intero di informazione futura — ed è esattamente nei casi di movimento ampio, cioè
quelli su cui si guadagna, che il look-ahead è massimo. **È il meccanismo per cui i backtest su
pattern sembrano eccellenti e non sopravvivono al live.**

**Requisiti vincolanti** (da rispettare in implementazione):

- Ogni pivot va spostato alla **barra di conferma**; il ritardo è un parametro esplicito.
- La gamba in formazione non è utilizzabile. Nel caso ABCD questo significa che **il punto D è
  esattamente ciò che non si conosce quando servirebbe**: un ABCD è utilizzabile solo come
  *proiezione* di D a partire da A, B, C confermati, mai come struttura completa.
- Test di regressione dedicato: la pipeline deve produrre, per ogni timestamp, le stesse strutture
  che sarebbero state visibili in tempo reale. Il confronto fra output causale e retrospettivo va
  misurato e riportato — **è la misura diretta di quanto vale il look-ahead in questo dominio**, ed
  è un numero che vale la pena avere prima di fidarsi di qualunque risultato su pattern.

### 7.2 Il conteggio: quante strutture ci sono davvero

Istanze **confermate causalmente**, 15 simboli dal 2022 (1.691 giorni × 15 = 25.365
simbolo-giorni). "Indipendenti" = non sovrapposte nel tempo, perché istanze sovrapposte
condividono futuro e non sono osservazioni separate.

| timeframe | soglia | tipo | nominali | **indipendenti** | durata mediana | istanze/gg/simbolo |
|---|---|---|---|---|---|---|
| 15m | 1,0% | ABC (3 pivot) | 305.793 | **26.388** | 2,0 h | 1,04 |
| 15m | 1,0% | doppio max/min | 251.788 | **25.121** | 2,0 h | 0,99 |
| 15m | 1,0% | ABCD con Fibonacci | 52.422 | **10.356** | 3,2 h | 0,41 |
| 15m | 1,0% | impulso 5 onde | 9.012 | **2.534** | 6,2 h | 0,10 |
| 15m | 2,0% | ABC (3 pivot) | 76.386 | 8.790 | 6,2 h | 0,35 |
| 15m | 2,0% | ABCD con Fibonacci | 13.205 | **3.266** | 9,5 h | 0,13 |
| 15m | 2,0% | impulso 5 onde | 2.233 | **757** | 19,5 h | 0,030 |
| 15m | 3,0% | ABCD con Fibonacci | 2.772 | **1.055** | 20,0 h | 0,042 |
| 15m | 3,0% | impulso 5 onde | 434 | **213** | 36,9 h | 0,008 |
| 1h | 1,0% | ABC (3 pivot) | 147.582 | 11.723 | 5,0 h | 0,46 |
| 1h | 2,0% | ABCD con Fibonacci | 9.553 | 2.319 | 13,0 h | 0,091 |

Il termine di paragone è il numero di osservazioni indipendenti dell'intero dataset (§4.2): decine
di migliaia. Ne segue una classificazione netta:

- **ABC e doppio massimo/minimo a soglia 1%** producono ~25.000 istanze indipendenti — dello stesso
  ordine dell'intero dataset. Sono utilizzabili.
- **ABCD è marginale**: 10.356 a soglia 1%, 3.266 al 2%, 1.055 al 3%. Un modello dedicato su
  qualche migliaio di osservazioni indipendenti, con la libertà di scelta dei parametri di Fibonacci
  che quel tipo di struttura porta con sé, è una macchina per l'overfitting.
- **L'impulso a 5 onde è fuori scala e va escluso, non ottimizzato**: 213–2.534 istanze
  indipendenti. **Da escludere.**

### 7.3 Scale temporali: sono due sistemi, non uno — **raccomando l'Opzione 1**

Il dato che chiude la questione è l'ultima colonna di §7.2: **il tipo di struttura più frequente
produce 1,04 istanze/giorno/simbolo**, contro un target di 4. ABCD al 2% ne produce 0,13, cioè
**trenta volte meno del target**. Non è un difetto dei pattern: è che strutture e alta frequenza
sono due sistemi con orizzonti diversi, e mescolarli in un target unico li rompe entrambi.

**Raccomandazione: Opzione 1 — i pattern entrano come feature di stato nel sistema veloce
esistente**, non come strategia autonoma. Concretamente:

- distanza dall'ultimo pivot confermato, in unità di ATR (non in prezzo: dev'essere scale-free);
- tempo trascorso dall'ultimo pivot confermato;
- rapporto fra le ultime due gambe (è il numero che sta sotto ad ABC e ABCD, senza imporre una
  classificazione discreta);
- posizione del prezzo dentro la struttura candidata corrente;
- flag di struttura invalidata (il prezzo ha superato un livello che nega la struttura).

Non cambia la frequenza, non frammenta il dataset, e aggiunge informazione strutturale che le
feature attuali — momento, oscillatori, volatilità — non contengono. È a basso rischio e
compatibile con tutto il resto.

**L'Opzione 2 (sistema lento separato)** resta legittima come progetto distinto, con target di
frequenza proprio (~1 trade ogni pochi giorni per simbolo), barriere ampie dove il vincolo fee
sparisce (a TP 4% le commissioni taker sono rumore) e validazione propria. **Non va mescolata con
il sistema a 4 trade/giorno.** È anche il regime in cui il RL sulla gestione dell'uscita ha più
spazio, perché la posizione resta aperta abbastanza da poter essere gestita.

### 7.4 Modelli specialisti per tipo di pattern — **sconsigliato**

La proposta di "un grande chunk per tipo di movimento" va valutata contro l'alternativa di **un
modello unico con il tipo di struttura come feature categorica**. Il criterio proposto nella
revisione è il conteggio di §7.2, e il conteggio risponde: **migliaia di istanze indipendenti per
bucket, non decine di migliaia.** Un ABCD a soglia 2% ha 3.266 osservazioni indipendenti.

Separare in specialisti su quei numeri fa due danni contemporaneamente: frammenta un conteggio di
campioni indipendenti già scarso, e impedisce la condivisione di forza statistica fra tipi che
condividono struttura (un ABC è il prefisso di un ABCD, un doppio massimo è un ABC con vincolo di
uguaglianza sui livelli — non sono categorie disgiunte).

**Raccomandazione: modello unico, tipo di struttura come feature categorica**, e verifica *a
posteriori* che il modello usi davvero quella feature — importanza per permutazione calcolata
**dentro il fold**, mai sull'intero dataset. Se l'importanza è nulla, la conclusione è che le
strutture non aggiungono niente sopra le feature esistenti, ed è un risultato utile da avere.

---

## 8. Strategia di addestramento: fasi e gate

Ogni fase ha un **gate**. Se il criterio non è soddisfatto non si passa alla successiva: si torna
indietro. La colonna "configurazioni" conta le prove da sommare nel calcolo del Deflated Sharpe
Ratio finale — **il conteggio va tenuto onesto, incluse le configurazioni scartate.**

### Fase 0 — Infrastruttura di verità

| # | intervento | file |
|---|---|---|
| 0.1 | `triple_barrier_labels` restituisce anche `t_exit` e la barriera toccata | `ml/labeling.py` |
| 0.2 | `PurgedKFold`, `CombinatorialPurgedCV`, `sample_uniqueness_weights` | nuovo `ml/validation.py` |
| 0.3 | Modello di riempimento degli ordini limite: probabilità di fill e selezione avversa | nuovo `ml/execution.py` |
| 0.4 | Ingestione `aggTrades` in streaming, ridotta ad aggregati per barra | nuovo `data/microstructure.py` |

`0.1` è bloccante per `0.2`: senza il tempo di uscita il purging non è calcolabile. `0.4` è
bloccante per la validazione di `0.3`: il modello di fill va tarato contro riempimenti reali.

**Gate:** la CPCV gira end-to-end su tutto il dataset, e il modello di fill è validato contro
`aggTrades` reali su un campione (almeno 3 simboli × 3 mesi, coprendo un regime calmo e uno
volatile). *Configurazioni testate: 0 — è infrastruttura, non selezione.*

### Fase 1 — Punto di lavoro economico

| # | intervento | file |
|---|---|---|
| 1.1 | Ripristinare `TP_ATR_MULTIPLE = 2.0`; regime di esecuzione come parametro esplicito | `ml/labeling.py` |
| 1.2 | Campionamento a eventi con filtro CUSUM, soglia in multipli di σ | `ml/dataset.py` |
| 1.3 | Scelta di barriere e soglia CUSUM per 4 trade/giorno/simbolo | configurazione |
| 1.4 | Expectancy misurata sulla distribuzione completa degli esiti, in modalità maker | `ml/evaluate.py` |

Punto di partenza indicato dalle misure: **barriere 1,20%/0,60% con orizzonte 24h** (§2.2: divario
maker più basso, 3,1 punti, con tetto 6,9 trade/giorno che resta sopra il target) e **CUSUM a
k = 3,0–3,5** (§1.4: 27–33 eventi/giorno su tutti i simboli, selettività implicata 13%).

**Gate:** distribuzione CPCV dell'expectancy netta con **mediana positiva** e **PBO < 0,5**.
*Configurazioni: ~12 (4 barriere × 3 soglie CUSUM).*

### Fase 2 — Meta-labeling — **fermarsi qui e chiedere conferma**

| # | intervento | file |
|---|---|---|
| 2.1 | Primario direzionale permissivo (regola o modello leggero) | `ml/labeling.py` |
| 2.2 | Secondario binario su etichette già nette da fee | `ml/models.py` |
| 2.3 | Calibrazione delle probabilità e dimensionamento (Kelly frazionario con tetto) | `ml/evaluate.py` |

**Gate:** il secondario batte la soglia applicata direttamente al primario sulla **distribuzione**
CPCV, non su un singolo split. *Configurazioni: ~8.*

### Fase 3 — Feature strutturali e microstruttura

| # | intervento | file |
|---|---|---|
| 3.1 | Feature strutturali causali (§7.3, Opzione 1) con test di regressione causale/retrospettivo | `ml/features.py` |
| 3.2 | Feature di microstruttura da `aggTrades` (volume delta, flow imbalance) | `ml/features.py` |
| 3.3 | `fundingRate` e open interest come feature di regime | `data/derivatives.py` |
| 3.4 | MACD, %B e ampiezza di Bollinger, variazione OBV, ADX/efficiency ratio | `ml/features.py` |

**Gate:** miglioramento dell'expectancy netta out-of-sample che **sopravvive al Deflated Sharpe
Ratio** con il conteggio onesto di tutte le configurazioni provate fino a qui.
*Configurazioni: ~10.*

### Fase 4 — RL sull'esecuzione

Il candidato scelto in §6.2. RL offline con vincolo di prossimità alla politica di comportamento,
reward risk-adjusted al netto dei costi reali.

**Gate:** batte la baseline a regola fissa (limite al mid con timeout fisso) sulla **distribuzione**
CPCV. *Configurazioni: ~6.*

### Fase 5 — RL sull'uscita e dimensionamento a portafoglio

Solo dopo che le fasi precedenti hanno prodotto expectancy netta positiva stabile. Include il
vincolo di capitale sul **picco** di posizioni concorrenti (§1.5: 15–28 al 99° percentile, e il
numero reale è più alto perché i segnali arrivano correlati).

**Gate:** batte il dimensionamento a regola fissa sulla distribuzione CPCV. *Configurazioni: ~6.*

### Cosa **non** va toccato

- La proprietà scale-free di `features.py` e l'assenza di stato appreso.
- Lo split temporale globale (mai per simbolo).
- La separazione fra edge lordo e netto in `evaluate.py`.
- La corrispondenza fra le barriere che definiscono le etichette e quelle che governano le uscite
  in `signals.py`: è ciò che rende il P&L simulato la traduzione diretta del win rate misurato.

---

## 9. Aspettative realistiche

Stima onesta, basata su quanto misurato in questa sessione e non su quanto sarebbe desiderabile.

### Win rate

**Misurato oggi:** win rate 39,6% contro un break-even di 47,3%, AUC 0,54, edge lordo +0,017% per
operazione. Verificato end-to-end sul periodo mai visto: 32–44% per coppia, coerente.

**Realistico dopo le Fasi 1–2:** **40–45%**. Il ragionamento: il base rate **misurato** è
32,1–32,6% (§2.2, non il 33,3% teorico); un modello con AUC 0,55–0,60 sul decile superiore porta
la precision a circa 40–45%. Andare oltre richiederebbe un AUC che su prezzo e indicatori non si
osserva.

**Con microstruttura (Fase 3):** forse **45–50%**. È la stima più incerta del documento, perché non
ho misurato nulla su `aggTrades`.

**Un WR >70% in-sample su questo dominio va trattato come un difetto fino a prova contraria.** I
sospetti da verificare, nell'ordine: normalizzazione fittata fuori dal fold, rolling non causale,
uso di una barra aggregata incompleta, purging assente (etichette sovrapposte fra train e test).

### Frequenza

| configurazione | trade/gg/simbolo | tetto fisico | note |
|---|---|---|---|
| Oggi (soglia 0,484, 15m) | **~0,2** | — | 99 trade in 20 mesi su BTC 15m |
| Barriere 0,60%/0,30%, orizzonte 8h | 4 | 22,3 | 18% di tempo in mercato |
| Barriere 0,80%/0,40%, orizzonte 8h | 4 | 14,8 | 27% di tempo in mercato |
| **Barriere 1,20%/0,60%, orizzonte 24h** | **4** | **6,9** | 58% in mercato — **raccomandata** |
| CUSUM k=3σ, accettando il 13% degli eventi | 4 | ~30 | selettività da decile superiore |

**Il target di 4 trade/giorno/simbolo è raggiungibile in tutte e quattro le configurazioni di
barriere misurate**, quindi il vincolo di frequenza non seleziona da solo la barriera: a
selezionare è l'economia (§2.2), che indica 1,20%/0,60%. A portafoglio sono ~60 trade/giorno con
un picco di 25–28 posizioni concorrenti (§1.5).

### Expectancy netta

È il criterio vero, e la risposta onesta è che **dipende quasi interamente dal regime di
esecuzione**. Con la configurazione raccomandata (1,20%/0,60%, orizzonte 24h) e WR di base
misurato 32,3%:

| scenario | WR richiesto | WR atteso | esito |
|---|---|---|---|
| Taker | 44,4% | 40–45% | **negativo o al limite** |
| Taker + sconto BNB | 41,6% | 40–45% | **marginale** |
| **Maker** | **35,4%** | 40–45% | **positivo con margine** |
| Maker, ipotesi conservativa | 35,4% | 37% | **positivo, margine sottile** |

**La conclusione da portare in implementazione: l'esecuzione con ordini limite non è
un'ottimizzazione, è la precondizione.** Con ordini a mercato la strategia richiede di battere il
caso di 12,1–23,2 punti, capacità che non ho osservato in nessuna configurazione testata; con
ordini limite richiede 3,1–5,0 punti, che è alla portata.

**Il numero più fragile del documento è il "WR atteso 40–45%".** Poggia su un AUC osservato di
0,54 e su un'estrapolazione a 0,55–0,60 dopo le Fasi 1–3. Se l'AUC reale resta a 0,54, il WR sul
decile superiore si ferma intorno al 36–38% — **appena sopra il break-even maker e sotto ogni
altro**. In quello scenario la strategia è marginale anche nel suo regime migliore, e la decisione
onesta è fermarsi al gate della Fase 1 invece di proseguire.

E resta la riserva metodologica principale: **tutti i numeri della colonna maker sono non
verificati finché il modello di riempimento (Fase 0.3) non esiste.** Una simulazione maker che
assume riempimento certo sovrastima sistematicamente, e sovrastima proprio la variabile su cui
l'intera strategia poggia.

---

## Riproducibilità

| misura | script |
|---|---|
| Copertura dello store | `python -m cryptofarm.data.klines --manifest` |
| Regimi, dollar bar, CUSUM su BTC (rev. 1) | `feasibility.py` |
| §1.3 tempo al target censurato, §1.5 holding e capacità | `m1_capacity.py` |
| §2.2 break-even misurato, confronto random walk, §1.4 CUSUM 15 simboli, concorrenza | `m2_economics.py` |
| §7.1 ritardo di conferma dei pivot, §7.2 conteggio strutture | `m3_patterns.py` |
| §4.1 confronto LSTM / GBDT | `diagnose.py` |
| Sweep configurazioni di labeling | `sweep_labeling.py`, `sweep_barriers.py` |
| Verifica segnali end-to-end | `check_signals.py` |

**Gli script stanno nella scratchpad di sessione e sono quindi effimeri.** Vanno spostati sotto
`scripts/` perché le misure di questo documento restino riproducibili — è un prerequisito pratico
della Fase 0, non un abbellimento: senza, ogni numero qui è verificabile solo a memoria.
