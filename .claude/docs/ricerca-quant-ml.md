# Ricerca quantitativa e ML — due filoni, misurati su cinque asset

Data: **2026-08-26**. Universo: BTC, ETH, SOL, XRP, BNB contro USDT. Solo **pronti**, solo
**lunghi**, nessuna leva. Dati: lo store locale del progetto — 15 simboli a 5 minuti, da 614.732 a
945.675 candele per simbolo, fino al **2026-08-19** — aggregato a 4h e 1d da `data/klines.py`.

Seguito di [`backtest-strategie.md`](backtest-strategie.md) e [`strategie-nuove.md`](strategie-nuove.md),
che misuravano **un asset**. Questo documento chiude i due punti che quelli lasciavano aperti (SOL e
BNB mai misurati; `donchian_breakout` e `squeeze_breakout` da rimisurare dopo la correzione dello
stop a trailing) e apre due famiglie che il progetto non aveva mai provato.

---

## Il risultato in cinque righe

1. **Su cinque asset, quasi niente batte il possesso passivo.** 7.516 configurazioni delle strategie
   storiche a 1d: il 4,7% batte il comprare-e-tenere. 50 celle fuori campione delle strategie nuove:
   il 42% è in utile, il 24% batte il passivo, e **9 di quelle 12 vittorie sono in finestre dove il
   passivo era negativo** — si vince stando fermi, non guadagnando di più.
2. **La raccomandazione della sessione precedente non generalizza.** `band_reversion_gated`, la
   strategia indicata come la più promettente su BTC, ha mediana negativa su 4 asset su 5 a
   entrambi gli intervalli. Le due che reggono ovunque sono `ichimoku_trend` e `donchian_breakout`
   a 4h.
3. **Anche la conclusione sul timeframe va corretta.** Fuori campione **4h batte 1d** (52% contro
   32% di celle in utile, mediana +0,6% contro −27,5%). "Sempre più lento è meglio" era vero fra 15m
   e 1d su un asset; non è monotono.
4. **La famiglia trasversale — scegliere *quale* asset invece di *quando* — è l'unica cosa nuova che
   trasferisce.** Fuori campione, sui cinque grandi, l'intera griglia ha mediana +62% con l'89% di
   configurazioni in utile, contro +55,6% di BTC e +37,3% dell'universo a peso uguale. Ma **la
   scelta dei parametri non trasferisce affatto** (ρ = −0,69) e l'effetto **sparisce allargando
   l'universo a 15 asset** (mediana −0,9%).
5. **Il filone ML riformulato ha un vantaggio reale ma sotto la soglia di significatività.** Il
   filtro meta sopra una strategia primaria vera ottiene AUC 0,537 dentro e fuori campione — non
   0,50 — e alza il netto per operazione. Contro una **selezione casuale della stessa numerosità**
   però resta all'80°-98° percentile, e avendo provato una ventina di combinazioni, un 98° percentile
   è quello che ci si aspetta dal caso.

---

## 1. Cosa dice lo stato dell'arte, letto nei repository

Nove repository letti direttamente (README, documentazione e tabelle dei benchmark scaricate il
2026-08-26, non ricordate). Quello che segue è **conoscenza derivata dalle fonti**, distinta più
avanti dalle ipotesi.

### 1.1 Il numero più utile di tutti: il soffitto dell'alpha ML

`microsoft/qlib` pubblica la tabella dei benchmark di ~20 modelli su `Alpha158` e `Alpha360`
(CSI300, media e deviazione su 20 semi). È la misura più onesta disponibile di quanto valga
davvero un modello di previsione su un mercato molto studiato:

| modello | dataset | IC | Rank IC | rendimento annuo |
|---|---|---:|---:|---:|
| DoubleEnsemble | Alpha158 | **0,0521** | 0,0502 | 11,6% |
| XGBoost | Alpha158 | 0,0498 | 0,0505 | 7,8% |
| LightGBM | Alpha158 | 0,0448 | 0,0469 | 9,0% |
| TRA | Alpha158 | 0,0440 | **0,0540** | 7,2% |
| Localformer | Alpha158 | 0,0356 | 0,0468 | 4,4% |
| Transformer | Alpha158 | 0,0264 | 0,0407 | 2,7% |
| TabNet | Alpha158 | 0,0204 | 0,0333 | 2,3% |
| Transformer | Alpha360 | 0,0114 | 0,0327 | **−2,7%** |
| TabNet | Alpha360 | 0,0099 | 0,0290 | **−3,7%** |

Tre cose che questa tabella stabilisce, e che nessun documento del progetto aveva:

- **il soffitto è IC ≈ 0,05.** Una correlazione del 5% fra previsione e rendimento futuro è il
  massimo che venti architetture producono su un panel azionario ampio e pulito. L'AUC 0,54 che
  `strategy.md` §Sintesi riportava come fallimento **è il livello normale del campo**, non
  un'anomalia;
- **il gradient boosting non è battuto dai modelli profondi.** I primi tre posti per IC sono
  ensemble di alberi; Transformer e TabNet su `Alpha360` producono rendimento annuo *negativo*.
  Conferma la scelta `gbdt` già fatta in `ml/models.py`;
- **quel soffitto si monetizza in sezione, non nel tempo.** Tutti quei numeri vengono da un
  portafoglio costruito ordinando ~300 titoli ogni giorno. Un IC di 0,05 su *un* asset non produce
  niente di eseguibile; su una sezione larga sì, perché l'errore si media. **Questo è il punto che
  separa lo stato dell'arte da tutto ciò che il progetto ha provato finora**, dove ogni misura ha
  sempre riguardato un simbolo alla volta.

### 1.2 `machine-learning-for-trading` (terza edizione) — il metodo, non le tecniche

La terza edizione è riorganizzata attorno a un processo unico con una *evidence boundary* che separa
la messa a punto dalla valutazione, più un ciclo di riaddestramento/pausa/ritiro quando l'edge
decade. Novità rilevanti qui:

- **costi di transazione e gestione del rischio sono capitoli interi** (18 e 19), non appendici, e
  si aggiungono a costruzione di portafoglio (17) e sintesi di strategia (20): un segnale grezzo va
  portato fino a un portafoglio dimensionato, con costi e rischio, prima di dire che funziona;
- **strumenti anti-overfitting espliciti**: Deflated Sharpe Ratio, Rademacher Anti-Serum, White's
  Reality Check, predizione conformale, walk-forward ovunque. Il progetto ha già DSR e PBO in
  `ml/validation.py`, e non li usava nella politica a tre azioni (`strategy.md` §14);
- **nove casi di studio sullo stesso processo**, fra cui uno su **perpetui cripto a 8 ore basato sul
  funding rate** e uno intraday a 15 minuti su microstruttura del book. Il primo è l'unico caso
  cripto della raccolta, e non è una strategia direzionale: è arbitraggio di finanziamento;
- ML4T tratta esplicitamente il *multiple testing* nella ricerca sui fattori. È la lente giusta per
  leggere i risultati della §5 qui sotto.

### 1.3 `freqtrade` / FreqAI — l'unico impianto ML cripto in produzione

FreqAI è il modulo ML di freqtrade, ed è interessante per la forma dell'impianto, non per i modelli:

- **riaddestramento auto-adattivo** su finestra scorrevole, in un thread separato dall'inferenza, e
  un backtest che *emula* il riaddestramento periodico invece di addestrare una volta sul passato;
- **espansione automatica delle feature** su quattro assi: `indicator_periods_candles` ×
  `include_timeframes` × `include_shifted_candles` × **`include_corr_pairs`**. Quest'ultimo è
  decisivo: nel disegno canonico di FreqAI, le feature di *ogni* coppia includono gli indicatori
  delle coppie correlate. È lo stesso principio della §1.1 — il contesto trasversale entra nel
  modello;
- **rimozione degli outlier** come parte della pipeline (Dissimilarity Index, SVM, DBSCAN) e PCA per
  la riduzione di dimensionalità;
- un vincolo operativo che vale la pena registrare: FreqAI **non** si combina con pairlist dinamiche,
  perché i dati di addestramento vanno scaricati all'avvio.

### 1.4 Gli altri cinque

- **`jesse`** — framework cripto per backtest e live. Due strumenti che il progetto non ha e che
  sarebbero utili subito: **test di significatività della regola d'ingresso** ("questo edge poteva
  comparire per caso?") e **analisi Monte Carlo** con rimescolamento dell'ordine delle operazioni.
  Ha anche fill parziali, ottimizzazione con Optuna+Ray, e una pipeline ML integrata.
- **`AI4Finance/FinRL`** — RL finanziario a tre livelli (ambiente / agente / applicazione). Il
  repository stesso ora si dichiara "workflow classico per didattica, sperimentazione e
  prototipazione di ricerca" e rimanda a FinRL-X per la produzione. Da leggere come laboratorio, non
  come impianto pronto.
- **`vnpy`** — piattaforma di trading (gateway, esecuzione, gestione ordini). Risolve l'infrastruttura,
  non l'alpha.
- **`TauricResearch/TradingAgents`** — agenti LLM in ruoli (analisti, ricercatori, trader, rischio).
  Il repository documenta da sé i limiti di riproducibilità (i modelli di ragionamento ignorano la
  temperatura). Non è una fonte di segnale misurabile su OHLCV.
- **`Rachnog/Deep-Trading`** — il README dichiara "released part one - simple time series
  forecasting". Interesse storico (2017): è la generazione di lavori da cui viene l'idea che una rete
  profonda sul prezzo basti, che i benchmark qlib di §1.1 smentiscono.
- **`awesome-quant`** — indice. Utile per le sezioni backtesting, ottimizzazione di portafoglio e
  analisi fattoriale.

### 1.5 Cosa se ne ricava, in tre principi

Non sono ipotesi: sono conclusioni sostenute dalle fonti sopra.

1. **La sezione trasversale è dove l'alpha debole diventa eseguibile.** (qlib, FreqAI `corr_pairs`,
   ML4T cap. 17)
2. **Il vincolo economico va dentro il target, non verificato dopo.** (ML4T cap. 18; e in questo
   repository `strategy.md` §13 lo ha imparato a sue spese)
3. **Il gradient boosting è il riferimento; le architetture profonde vanno giustificate.** (qlib
   benchmarks)

---

## 2. Filone quantitativo — cosa hanno prodotto le misure

Comando: `strategy_lab` (5 strategie a due versi, 592 configurazioni) e `strategy_sweep` (le 11
storiche, 3.129 configurazioni) su ognuno dei cinque simboli, a 1d e 4h, dal 2021-01-01, commissione
0,05% per gamba. Solo il lato lungo, come da mandato.

### 2.1 I riferimenti da battere

| simbolo | 2021-2023 | 2024-oggi | intero periodo | drawdown |
|---|---:|---:|---:|---:|
| BTC | +44,2% | +55,6% | +134,4% | 76,6% |
| ETH | +213,1% | −10,9% | +187,5% | 79,3% |
| SOL | +5.422,0% | −25,5% | +4.346,0% | 96,3% |
| XRP | +159,2% | +69,7% | +349,8% | 83,2% |
| BNB | +725,4% | +97,4% | +1.538,0% | 70,9% |

### 2.2 Le strategie nuove, su cinque asset (in campione)

Mediana della griglia, solo lunghe, ≥10 operazioni:

**1 giorno**

| strategia | BNB | BTC | ETH | SOL | XRP | in utile (media) | sopra il passivo |
|---|---:|---:|---:|---:|---:|---:|---:|
| squeeze_breakout | +450,9% | +85,6% | +34,2% | +168,0% | +74,5% | 83% | 0-26% |
| donchian_breakout | +34,4% | +27,1% | +17,0% | +315,1% | +28,9% | 80% | 0-4% |
| ichimoku_trend | +36,8% | −5,3% | +35,1% | +113,7% | −14,0% | 74% | 0% |
| band_reversion_gated | −34,8% | +9,2% | −39,1% | −81,6% | +10,3% | 32% | 0% |
| trend_pullback | −46,9% | −9,3% | −26,1% | +119,7% | −77,5% | 27% | 0% |

**4 ore**

| strategia | BNB | BTC | ETH | SOL | XRP | in utile (media) | sopra il passivo |
|---|---:|---:|---:|---:|---:|---:|---:|
| ichimoku_trend | +1.008,4% | +35,9% | +38,0% | +434,9% | +177,7% | 83% | 0-33% |
| donchian_breakout | +374,5% | +40,0% | +68,7% | +199,7% | +171,1% | 87% | 0-23% |
| squeeze_breakout | +16,1% | −0,9% | −13,8% | +29,9% | +20,0% | 53% | 0% |
| trend_pullback | 0,0% | −21,5% | −15,8% | +17,8% | +24,7% | 51% | 0% |
| band_reversion_gated | −16,1% | −10,6% | −29,8% | +11,6% | −11,3% | 31% | 0% |

Tre letture, tutte contro un'aspettativa precedente:

- **`band_reversion_gated` non generalizza.** Su BTC 1d resta accettabile (mediana +9,2%, Sharpe
  0,20), ma è negativa su BNB, ETH e SOL a entrambi gli intervalli, con Sharpe mediano fino a −0,57.
  La raccomandazione #2 di `strategie-nuove.md` §7 vale per un asset, non per la famiglia.
- **La correzione dello stop a trailing ha promosso `donchian_breakout`.** Rimisurata, è la seconda
  più regolare (75-100% di configurazioni in utile a 4h). Il punto aperto in
  `strategie-nuove.md` §8 è chiuso: la correzione migliora, come il test sintetico suggeriva.
- **`ichimoku_trend` a 4h è la più solida in campione**: mediana positiva su tutti e cinque, Sharpe
  mediano da 0,33 a 1,08, e l'unica cella che supera il passivo in un terzo dei casi (BNB).

### 2.3 Fuori campione: 50 celle, e il verdetto

Scelta sul 2021-2023, resa sul 2024-oggi, per ognuna delle 5 strategie × 5 simboli × 2 intervalli:

| | valore |
|---|---:|
| celle con resa positiva | **21 / 50 (42%)** |
| celle che battono il possesso passivo | **12 / 50 (24%)** |
| … di cui in finestre dove il passivo era **negativo** | **9 su 12** |
| mediana della resa fuori campione | **−8,9%** |
| correlazione mediana stima↔verifica (ρ) | 0,26 (positiva nel 70% delle celle) |

Per strategia:

| strategia | positive | battono il passivo | mediana della resa | ρ mediano |
|---|---:|---:|---:|---:|
| ichimoku_trend | 6/10 | 4/10 | **+13,5%** | 0,38 |
| squeeze_breakout | 7/10 | 3/10 | +6,5% | −0,04 |
| band_reversion_gated | 4/10 | 3/10 | −2,5% | 0,38 |
| donchian_breakout | 2/10 | 2/10 | −31,5% | −0,02 |
| trend_pullback | 2/10 | 0/10 | −39,4% | 0,22 |

Per intervallo:

| intervallo | celle positive | battono il passivo | mediana |
|---|---:|---:|---:|
| 1d | 32% | 20% | −27,5% |
| **4h** | **52%** | **28%** | **+0,6%** |

**`ichimoku_trend` è l'unica che trasferisce**: mediana positiva, ρ 0,38, e 4 vittorie su 10 contro
il passivo. Ed è anche l'unica il cui massimo in campione non è un artefatto di griglia — la sua
griglia ha 11-12 configurazioni, non 256.

**Il ribaltamento sul timeframe va detto chiaramente**: `strategie-nuove.md` §7 punto 1 concludeva
"scala giornaliera, non 15 minuti", e la generalizzazione implicita era "più lento è meglio". Su
cinque asset non è così: 4h batte 1d su ogni metrica fuori campione. La regola vera è più stretta —
**esiste un intervallo intermedio dove il margine per operazione supera il costo e le operazioni
restano abbastanza numerose da non dipendere da tre trade**; a 15m il costo vince, a 1d il campione
diventa troppo piccolo.

### 2.4 Le strategie storiche, su cinque asset

7.516 configurazioni a 1d, commissione 0,05%: mediana **+43,9%**, **72%** in utile, e **solo il
4,7% batte il possesso passivo**. L'unica cella con un vantaggio sistematico è `Close EMA Crossover`
su BTC (75% delle configurazioni sopra il passivo), che non si ripete su nessun altro asset (0%).

Su SOL e BNB le mediane sono spettacolari in assoluto (`Green Candles` +1.778%, `Close RSI Reverse`
+908%) e irrilevanti in relativo: il passivo faceva +4.346% e +1.538%.

### 2.5 La famiglia nuova: rotazione trasversale

`scripts/cross_section.py`. A ogni ribilanciamento si ordinano gli asset per forza relativa
(rendimento su `lookback` barre) e si tengono i primi `top` a peso uguale; chi ha forza negativa non
si compra e la sua quota resta in contanti. Variante con un interruttore di regime unico (fuori dal
mercato quando BTC sta sotto la sua media a 50 barre). Commissione 0,1% per gamba (listino a
pronti). 160 configurazioni: `lookback` ∈ {10,20,30,60,90}, `top` ∈ {1,2,3,5}, ribilanciamento ogni
{1,3,7,14} barre, regime ∈ {nessuno, BTC}.

**In campione, 2021-2026, cinque grandi** (BTC passivo +134,4%; universo a peso uguale +1.311,1%,
Sharpe 0,98, DD 91,0%):

| | valore |
|---|---:|
| mediana della griglia | +1.179,7% |
| configurazioni in utile | 100% |
| sopra BTC | 95,6% |
| **sopra l'universo a peso uguale** | **44,4%** |
| migliore per Sharpe (lb 20, top 3, settimanale, regime BTC) | +3.508,2%, **Sharpe 1,60**, DD **45,7%** |

La riga che conta è la quarta: **la mediana della rotazione non batte il tenere gli stessi cinque a
peso uguale.** I numeri assoluti enormi vengono dall'universo, non dalla rotazione. Il vantaggio
vero è sul rischio: DD 45,7% contro 91,0%, Sharpe 1,60 contro 0,98 — di nuovo la stessa forma di
risultato già vista su `band_reversion_gated`.

**Fuori campione, 2024-oggi** (BTC +55,6%; universo a peso uguale +37,3%, Sharpe 0,50):

| | cinque grandi | quindici asset |
|---|---:|---:|
| mediana dell'intera griglia | **+62,0%** | −0,9% |
| configurazioni in utile | 89% | 49% |
| sopra BTC | 52% | 16% |
| sopra il proprio universo | 65% | 56% |
| Sharpe mediano | 0,66 (contro 0,50) | 0,28 (contro 0,23) |
| ρ stima↔verifica sulle prime 10 | **−0,69** | −0,15 |

Quattro conclusioni, in ordine di solidità:

1. **La famiglia trasferisce dove le strategie a un asset non trasferiscono.** L'89% di
   configurazioni in utile fuori campione contro il 42% di celle positive della §2.3 è la differenza
   più grande misurata in questo documento.
2. **La scelta dei parametri non trasferisce, e anzi danneggia.** ρ = −0,69: prendere la migliore
   configurazione in stima è peggio che prenderne una a caso. La conseguenza operativa è precisa —
   **non ottimizzare**: prendere una configurazione centrale (lookback ~20-30 barre, top 2-3,
   ribilanciamento settimanale) e lasciarla ferma.
3. **L'universo largo non funziona.** A 15 asset la mediana fuori campione è −0,9% e solo il 16%
   batte BTC. Più asset non è più diversificazione: sono più alt-coin che nel 2024-2026 hanno perso
   (il loro paniere passivo fa −9,5%). L'effetto vive nei grandi capitalizzati.
4. **Il costo morde ma non uccide**, a ribilanciamento settimanale: la quota sopra l'universo passa
   da 50,6% (0,02%/gamba) a 44,4% (0,1%) a 28,1% (0,3%).

### 2.6 Le coppie (BTC/ETH, ETH/SOL, …)

Tenere la più forte fra due è la stessa procedura con universo di due e `top=1`. Nessuna gamba
corta, quindi compatibile col mandato.

**In campione** batte il "metà e metà" passivo in **9 coppie su 10**. Ma la coppia senza un vincitore
straordinario è anche l'unica che perde:

| coppia | rotazione | metà e metà | esito |
|---|---:|---:|---|
| **BTC/ETH** | +156,0% | +160,9% | **perde** |
| ETH/SOL | +29.325,7% | +2.266,7% | vince (SOL fa ×44) |
| BTC/SOL | +8.882,0% | +2.240,2% | vince (SOL) |
| SOL/BNB | +9.639,2% | +2.942,0% | vince |

Cioè: in campione la rotazione fra due "funziona" quando c'è un asset che moltiplica per quaranta e
la regola lo trova. Non è un edge, è concentrazione con il senno di poi sull'universo.

**Fuori campione (2024-oggi) il quadro è più interessante e più credibile**: batte il metà-e-metà in
7 coppie su 10, e **su BTC/ETH — la coppia senza outlier — fa +136,2% contro +22,4% del passivo e
+55,6% di BTC da solo**, evitando ETH nella sua fase debole. È il singolo risultato fuori campione
più pulito di tutto il documento, ed è anche un solo campione: una coppia, una finestra.

### 2.7 L'architettura quantitativa che ne segue

**Non** una strategia sola: due strati con ruoli diversi.

```
   strato 1 — selezione trasversale (che cosa)
      ordina i 5 grandi per forza relativa a 20-30 barre giornaliere
      tieni i primi 2-3 a peso uguale, ribilancia ogni 7 barre
      forza negativa -> contanti, non "il meno peggio"
      interruttore di regime unico: BTC sotto la media a 50 -> tutto in contanti

   strato 2 — tempificazione per asset (quando)  [opzionale, vedi §5]
      ichimoku_trend a 4h, solo lungo, parametri centrali della griglia
      applicato solo agli asset che lo strato 1 ha selezionato
```

Le ragioni di ogni pezzo sono misurate, non scelte per simmetria: lo strato 1 perché è l'unica
famiglia che trasferisce (§2.5); i parametri centrali invece degli ottimi perché ρ = −0,69 (§2.5);
l'interruttore unico perché in cripto la correlazione in caduta va a uno e selezionare il migliore
di cinque che scendono non protegge; `ichimoku_trend` perché è l'unica per-asset con mediana
positiva fuori campione (§2.3); il divieto di short perché è misurato in perdita in tutte e cinque
le strategie (`strategie-nuove.md` §5) ed è comunque fuori mandato.

---

## 3. Filone ML — la riformulazione, e cosa ha prodotto

### 3.1 Perché non si riparte da dove si era arrivati

`strategy.md` §13 ha misurato che entrare alla conferma di un minimo e uscire alla conferma di un
massimo cattura **zero in media** su 15 simboli a ogni soglia, *prima* dei costi: la conferma si paga
due volte e la gamba mediana ne vale 1,76-2,05. Non è un risultato sul modello, è una proprietà dello
schema. Nessuna feature, architettura o soglia lo sposta, e quel filone resta chiuso.

Quello che restava aperto è un'altra formulazione, che `strategy.md` §2.3 raccomandava e §13.4
riformulava: **il modello non decide quando comprare — decide se lasciar passare un segnale che una
strategia ha già prodotto.**

### 3.2 Il disegno: `scripts/meta_gate.py`

- **Primaria**: una strategia di `strategies_ls.py`, parametri centrali fissi, mai ottimizzati qui
  (ottimizzare primaria e filtro insieme è il modo classico di leggere il rumore due volte).
- **Campione**: una riga per **operazione**, non per barra. Migliaia invece di milioni, ed è un
  vantaggio: le barre dentro un trend sono la stessa osservazione ripetuta.
- **Etichetta**: `1` se l'operazione, eseguita come la strategia la eseguirebbe, chiude **sopra i
  costi**. Il vincolo economico è dentro il target: non si può avere ragione sul segno e perdere.
- **Feature (16)**: tutte scale-free e note alla barra d'ingresso — distanze da EMA50/EMA200 in unità
  di ATR, posizione nel canale di Donchian e nelle bande di Bollinger, ATR relativo, ADX, larghezza
  delle bande, StochRSI, MFI, pendenza OBV, volume relativo, escursione relativa, sopra/sotto
  EMA200 — **più tre trasversali che nessun modello del progetto aveva mai avuto**: rango di forza
  relativa nell'universo, ampiezza di mercato (quota di asset sopra la propria media a 50), forza
  contro BTC. È il principio §1.5.1, applicato.
- **Universo**: tutti e 15 i simboli in comune, 4 ore.
- **Modello**: `HistGradientBoostingClassifier` (§1.5.3).
- **Validazione**: `PurgedKFold` con embargo da `ml/validation.py` — le operazioni si sovrappongono,
  il k-fold ordinario misurerebbe su futuro già visto — più una verifica temporale separata
  (addestrato fino al 2024-01-01, misurato dopo).
- **Controllo**: per ogni soglia, 500 selezioni **casuali della stessa numerosità**. Con una primaria
  a coda lunga bastano poche operazioni fortunate per alzare il netto medio: il numero da battere non
  è zero, è il percentile alto del caso.

### 3.3 I risultati

| primaria | operazioni | AUC (CV purgata) | AUC (verifica temporale) | precisione senza → con filtro |
|---|---:|---:|---:|---:|
| `trend_pullback` | 3.098 | 0,537 | 0,538 | 50,6% → 53,4% |
| `donchian_breakout` | 1.718 | 0,512 | 0,514 | 35,7% → 38,2% |
| `squeeze_breakout` | 1.634 | **0,504** | 0,504 | nessun vantaggio |
| `band_reversion_gated` | 156 | 0,531 | 0,574 | campione insufficiente |

Il conto economico fuori campione (2024-oggi), netto per operazione, con il controllo casuale:

**`trend_pullback`** — 1.383 operazioni in verifica

| soglia | operazioni | precisione | netto medio | p95 del caso | percentile nel caso |
|---|---:|---:|---:|---:|---:|
| nessuna | 1.383 | 50,6% | −0,124% | — | — |
| 0,50 | 864 | 52,9% | −0,022% | +0,044% | 84° |
| 0,60 | 579 | 53,4% | **+0,064%** | +0,152% | 87° |

**`donchian_breakout`** — 802 operazioni in verifica

| soglia | operazioni | precisione | netto medio | p95 del caso | percentile nel caso |
|---|---:|---:|---:|---:|---:|
| nessuna | 802 | 35,7% | +0,146% | — | — |
| 0,45 | 275 | 38,5% | +0,818% | +0,716% | 97° |
| **0,50** | 217 | 38,2% | **+1,062%** | +0,886% | **98°** |
| 0,55 | 164 | 37,8% | +0,510% | +0,905% | 78° |
| 0,60 | 116 | 37,1% | +0,694% | +1,164% | 82° |

### 3.4 Come vanno letti

**Il vantaggio di ranking è reale.** AUC 0,537 in cross-validation purgata e 0,538 in una verifica
temporale completamente separata non è rumore: due misure indipendenti che coincidono a tre
decimali. Ed è esattamente il soffitto di §1.1 — questa non è una delusione, è il campo.

**Il vantaggio economico non supera il controllo.** Il caso migliore (`donchian_breakout`, soglia
0,50) sta al 98° percentile di 500 selezioni casuali della stessa numerosità. Preso da solo sarebbe
p ≈ 0,02. Ma **sono state provate quattro primarie per cinque soglie**: fra venti combinazioni, un
98° percentile è quello che ci si aspetta dal caso. Le soglie adiacenti della stessa primaria
scendono al 78° e all'82°, che è il comportamento del rumore, non di un effetto.

**L'AUC è la metrica sbagliata per una primaria a coda lunga.** `donchian_breakout` ha AUC 0,512 —
quasi indistinguibile dal caso — e il miglioramento economico più grande di tutti, perché una
selezione appena migliore del caso su una distribuzione a coda destra cattura una quota
sproporzionata dei pochi movimenti che pagano. È anche il motivo per cui non ci si può fidare: lo
stesso meccanismo rende il risultato dipendente da una manciata di operazioni.

**Il filtro migliora una primaria perdente senza renderla vincente.** `trend_pullback` fuori campione
va da −0,124% a +0,064% per operazione: il filtro toglie il segno meno e si ferma lì.

---

## 4. Analisi critica

### 4.1 Limiti che valgono per entrambi i filoni

- **Sopravvivenza nell'universo.** BTC, ETH, SOL, XRP, BNB sono i grandi capitalizzati **del 2026**.
  Nel gennaio 2021 SOL non era un maggiore. Ogni numero della §2.5 e della §2.6 contiene questa
  selezione, e la §2.5 lo mostra: il grosso del risultato in campione viene dal ×44 di SOL. La
  difesa parziale usata qui — confrontare sempre con **l'universo a peso uguale**, che porta la
  stessa distorsione — è quella giusta, e infatti abbassa il verdetto dal "95,6% sopra BTC" al
  "44,4% sopra l'universo".
- **Un ciclo solo.** 2021-2026. `strategie-nuove.md` §2 ha già misurato che i parametri non passano
  dal ciclo 2017-2020 a questo. Non c'è ragione di credere che passeranno al prossimo.
- **Molteplicità dei test.** Fra questo documento e i due precedenti sono state valutate oltre
  12.000 configurazioni. Nessun risultato qui è stato corretto per molteplicità con Deflated Sharpe
  o PBO — gli strumenti sono in `ml/validation.py` e non sono stati applicati alla parte quant. È il
  debito metodologico più grosso.
- **Esecuzione ideale.** Ingressi alla chiusura della barra, riempimento al prezzo esatto, niente
  slippage né impatto. Sui gap di liquidazione cripto è ottimistico, e lo è di più per la rotazione
  trasversale, che a `top=1` concentra tutto il capitale in un asset.
- **Nessun modello di liquidità.** Su BTC/ETH irrilevante ai capitali domestici; su asset minori a
  ribilanciamento giornaliero, no.

### 4.2 Dove ciascun approccio fallisce

| | quantitativo (rotazione + regole) | ML (filtro meta) |
|---|---|---|
| **fallisce quando** | il mercato è privo di dispersione fra asset (tutti si muovono insieme): la classifica diventa rumore e si pagano solo commissioni | il regime cambia forma rispetto all'addestramento — le feature restano definite ma la relazione con l'esito no |
| **fallisce in silenzio?** | no: turnover alto e risultato piatto sono visibili subito | **sì**: la probabilità continua a uscire ben calibrata mentre non seleziona più niente |
| **quanto costa scoprirlo** | settimane | mesi, se non si tiene il controllo casuale acceso in continuo |
| **rischio di sovradattamento** | alto sulla scelta dei parametri (ρ = −0,69), basso sulla famiglia | alto: 16 feature, 3.098 righe, coda lunga |
| **interpretabilità** | totale: si può dire perché si possiede un asset | media: il modello dà un rango, non una ragione |

### 4.3 Cosa servirebbe per convalidare

In ordine di rapporto valore/costo:

1. **Un secondo ciclo.** Rifare §2.5 e §3.3 su 2017-2020 con l'universo che esisteva allora
   (BTC, ETH, XRP, BNB, LTC). Se la rotazione trasversale trasferisce anche lì, cambia lo stato
   della prova; i dati ci sono già nello store.
2. **Deflated Sharpe e PBO sulla parte quant.** Il codice esiste. Va applicato alla griglia della
   rotazione, che è dove si sta per prendere una decisione.
3. **Test di significatività della regola e Monte Carlo sull'ordine delle operazioni**, come in
   `jesse` (§1.4). Il controllo casuale della §3.2 è la stessa idea applicata solo al filtro ML: va
   esteso alle strategie.
4. **Un modello di riempimento**, che `strategy.md` Fase 0.3 elenca come gate mai soddisfatto.
   Finché non c'è, ogni numero in modalità maker resta non verificabile.

---

## 5. Confronto diretto

| criterio | quant trasversale | quant per-asset | ML (filtro meta) |
|---|---|---|---|
| **cosa rileva bene** | quale asset sta guidando; assenza di forza in tutto l'universo | struttura di prezzo (rotture, nuvola, bande) su un singolo strumento | quali segnali di una primaria hanno il contesto sbagliato |
| **cosa non rileva** | il momento dell'ingresso; niente sotto la barra di ribilanciamento | il contesto relativo: compra la rottura di un asset debole come di uno forte | non genera niente: senza una primaria non ha input |
| **prova fuori campione** | **89% di configurazioni in utile**, mediana +62% (5 asset) | 42% di celle positive, mediana −8,9% | AUC 0,538 stabile; vantaggio economico all'80°-98° percentile del caso |
| **batte il possesso passivo?** | 52% delle configurazioni battono BTC, 65% l'universo | 24% delle celle, e 9 su 12 solo perché il passivo perdeva | non applicabile da solo |
| **rischio di sovradattamento** | basso sulla famiglia, **altissimo sui parametri** | alto (griglie fino a 256 configurazioni per cella) | alto |
| **interpretabilità** | alta | alta | media |
| **complessità realizzativa** | bassa (~250 righe, nessuna dipendenza nuova) | già in produzione | media (feature + validazione purgata + controllo casuale) |
| **costo di calcolo** | secondi | minuti | minuti |
| **adattamento al cambio di regime** | l'interruttore di regime è esplicito e verificabile | dipende dai filtri interni | richiede riaddestramento; degrada in silenzio |
| **frequenza operativa** | 17-30 di turnover annuo (ribilanciamento settimanale) | 3-27 operazioni/anno per asset | eredita quella della primaria |

Il confronto non è alla pari su un punto che va detto: **il quant trasversale e il ML non
rispondono alla stessa domanda.** Il primo alloca fra asset, il secondo filtra i segnali di una
strategia. Sono complementari per costruzione, il che è anche il motivo per cui il test di
integrazione della §6 va fatto con sospetto — la complementarità apparente è quasi sempre solo il
fatto che i due si guardano cose diverse.

---

## 6. Ibrido: conviene?

Il criterio dichiarato in partenza era: **integrare solo se i due portano informazione o capacità
decisionale genuinamente diversa.** Applichiamolo.

**Cosa sarebbe l'ibrido.** Lo strato trasversale sceglie *quali* asset possedere; il filtro meta
decide se un segnale della primaria su quell'asset merita di passare. Le informazioni sono
formalmente distinte (rango fra asset ≠ probabilità di successo di un singolo segnale) e le
decisioni pure (allocazione ≠ ingresso).

**Ma la prova non c'è.** Tre ragioni misurate:

1. **Il filtro meta non ha superato il proprio controllo** (§3.4). Comporre uno strato che non ha
   dimostrato di valere con uno che ha dimostrato di valere non può che peggiorare il secondo o
   lasciarlo com'è.
2. **Le feature trasversali sono già dentro il filtro** (rango di forza, ampiezza, forza su BTC): il
   modello ha già l'informazione che lo strato 1 userebbe, e l'AUC che ne esce è 0,537. Il canale di
   informazione che l'ibrido dovrebbe aprire è quindi già aperto, e vale poco.
3. **Gli strati si mangiano il campione a vicenda.** La rotazione tiene 2-3 asset su 5; applicare
   un filtro che scarta metà dei segnali su un sesto delle occasioni originarie porta a un numero di
   operazioni su cui non si misura più niente. È il difetto che ha reso `band_reversion_gated`
   invalutabile nella §3.3 (156 operazioni in cinque anni su quindici asset).

**Conclusione: non integrare adesso.** Va tenuta però una forma debole di ibrido, che è di fatto
gratuita e che i dati sostengono: **l'interruttore di regime unico dello strato 1 è già un filtro
condizionale**, e vale su tutto il portafoglio. È la sola composizione fra i due mondi per cui esiste
una misura (Sharpe 1,60 e DD 45,7% con l'interruttore, contro 0,98 e 91,0% del passivo).

Il momento in cui riaprire la questione è preciso: **se e quando il filtro meta batte il controllo
casuale su un secondo ciclo di mercato.** Non prima.

---

## 7. Raccomandazioni

### Il più forte approccio quantitativo

**Rotazione trasversale sui cinque grandi capitalizzati, a parametri centrali fissi, con
interruttore di regime.**

- lookback 20-30 barre giornaliere, `top` 2-3, ribilanciamento settimanale, forza negativa in
  contanti, fuori dal mercato quando BTC sta sotto la sua media a 50 giorni;
- **non ottimizzare i parametri** — è la raccomandazione operativa più insolita e la meglio
  sostenuta di questo documento: ρ = −0,69 fra resa in stima e in verifica;
- universo **cinque, non quindici**: allargare distrugge il risultato (mediana fuori campione da
  +62% a −0,9%);
- aspettativa dichiarata: non un rendimento superiore al passivo, ma **lo stesso ordine di
  rendimento con circa metà del drawdown**. Fuori campione ha dato più del passivo (mediana +62%
  contro +55,6% di BTC), ma su una finestra sola e con l'universo scelto col senno di poi.

Come riferimento per-asset, e **solo** come riferimento: `ichimoku_trend` a 4h, solo lungo,
parametri centrali. È l'unica delle regole per-asset con mediana positiva fuori campione (+13,5%) e
ρ 0,38. Qualunque strategia nuova che non la batta su quelle due colonne non merita di essere messa
in produzione — è la stessa soglia che `strategie-nuove.md` §7 fissava, ora verificata su cinque
asset invece che su uno.

### Il più forte approccio ML

**Filtro meta sopra una primaria vera, con contesto trasversale nelle feature — da tenere in
ricerca, non in produzione.**

Il disegno è corretto e va conservato: il vincolo economico dentro l'etichetta, il campione per
operazione, gli asset in comune, la validazione purgata, e soprattutto **il controllo con selezione
casuale della stessa numerosità**, che è ciò che ha impedito di scambiare un +1,06% per operazione
per una scoperta. Il vantaggio di ranking misurato (AUC 0,537, stabile fra due validazioni
indipendenti) è coerente con il soffitto dello stato dell'arte e non è nulla; semplicemente non è
ancora abbastanza per pagare.

Il prossimo passo è **uno**, non tre: rifare la stessa misura sul ciclo 2017-2020. Se il vantaggio
economico supera il controllo casuale anche lì, allora è un effetto. Se non lo supera, il filone si
chiude con una misura invece che con un'opinione.

### L'ibrido

**No, per ora.** Con l'unica eccezione dell'interruttore di regime, che è già dentro lo strato
quantitativo. Riaprire solo dopo il punto sopra.

### Cosa non rifare

- **Non riaprire la politica a tre azioni sul directional change.** `strategy.md` §13: cattura zero
  in media, prima dei costi, su ogni simbolo e a ogni soglia.
- **Non aggiungere il verso corto.** Misurato in perdita su tutte e cinque le strategie
  (`strategie-nuove.md` §5), e comunque fuori mandato.
- **Non allargare l'universo per "diversificare".** Misurato: peggiora.
- **Non cercare architetture profonde prima di aver esaurito la sezione trasversale.** I benchmark
  qlib (§1.1) mostrano Transformer e TabNet con rendimento annuo negativo dove il gradient boosting
  è positivo.
- **Non fidarsi del massimo di una griglia.** Vale ancora, e la §2.6 mostra la forma nuova dello
  stesso errore: una coppia che "vince" perché conteneva un asset che ha fatto ×44.

---

## 8. Riprodurre

```bash
# griglie per-asset (5 simboli × 2 intervalli), poi le tabelle
for S in BTCUSDT ETHUSDT SOLUSDT XRPUSDT BNBUSDT; do
  for I in 1d 4h; do
    .venv312/bin/python -m scripts.strategy_lab --all --symbol $S --interval $I \
        --since 2021-01-01 --workers 8
    .venv312/bin/python -m scripts.lab_report --symbol $S --interval $I
  done
  .venv312/bin/python -m scripts.strategy_sweep --all --symbol $S --interval 1d \
      --since 2021-01-01 --fee 0.05 --workers 8 --suffix _2021_fee005
done

# rotazione trasversale
.venv312/bin/python -m scripts.cross_section --selfcheck
.venv312/bin/python -m scripts.cross_section --universe majors --interval 1d --grid --save cs_majors_1d
.venv312/bin/python -m scripts.cross_section --universe majors --interval 1d --split --save cs_majors_1d_oos
.venv312/bin/python -m scripts.cross_section --universe wide   --interval 1d --split
.venv312/bin/python -m scripts.cross_section --pairs --lookback 20 --every 7
.venv312/bin/python -m scripts.cross_section --pairs --lookback 20 --every 7 --since 2024-01-01

# filtro meta
.venv312/bin/python -m scripts.meta_gate --selfcheck
.venv312/bin/python -m scripts.meta_gate --strategy trend_pullback    --universe wide --interval 4h --oos 2024-01-01
.venv312/bin/python -m scripts.meta_gate --strategy donchian_breakout --universe wide --interval 4h --oos 2024-01-01
```

Tabelle in `reports/` (`lab_*_*USDT_*.csv`, `cs_*.csv`, `meta_*.csv`).

### Una correzione al codice, trovata eseguendo

`strategy_lab` e `strategy_sweep` passavano le candele ai processi figli tramite variabili globali
riempite dal padre, contando sul `fork`. **Su macOS `ProcessPoolExecutor` usa `spawn`**, i globali
non arrivano e ogni esecuzione multi-worker moriva con `KeyError: ('SOLUSDT', '1d')`. Corretto in
entrambi facendo ricostruire al worker ciò che gli serve (`prepare` è idempotente, quindi sotto
`fork` la riga non costa nulla). Senza questa correzione nessuna misura di questo documento sarebbe
eseguibile sulla macchina dell'utente.

### Codice nuovo

| file | cosa | verifica |
|---|---|---|
| `scripts/cross_section.py` | rotazione trasversale: griglia, fuori campione, coppie; contabilità in valore per asset (non in pesi normalizzati, che cancellavano la quota in contanti) | `--selfcheck`: 5 asserzioni — capitale fermo a prezzi fermi, rendimento atteso su una rampa, monotonia rispetto alla commissione, **la quota in contanti non sparisce**, nessun look-ahead troncando la serie |
| `scripts/meta_gate.py` | filtro meta: campione per operazione, 16 feature scale-free di cui 3 trasversali, `PurgedKFold`, controllo con selezione casuale | `--selfcheck`: un segnale piantato nelle feature deve essere trovato (AUC > 0,75) e il rumore puro no (0,40 < AUC < 0,60) |

`ruff` e `black` puliti; 543 test passano.
