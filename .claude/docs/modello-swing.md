# Il modello a swing — dalla barriera tripla alla forma degli estremi

**Data:** 2026-08-28. **Stato:** superato il 2026-08-29 da `entry_model_veloce`
(`modello-ingresso.md`), che sta davanti in `MODEL_PRECEDENCE`. Questo modello resta nella catena
sotto di lui e torna a servire la pagina se gli artefatti d'ingresso non ci sono. Il documento
resta valido come misura: è quello che ha stabilito che il segnale esiste ma non batte il caso a
esposizione appaiata, ed è da lì che è nata la domanda nuova.

Cablato all'epoca *sapendo* cosa dice il §5. Cosa era stato cablato, e cosa deliberatamente no,
sta al §5.4. **Perché il modello nuovo non lo sostituisce per bravura ma per domanda**: a pari
selezione l'etichetta a gambe individua i minimi meglio del rendimento futuro diretto (37,2%
contro 23,0%) e rende 2,4 volte meno.

Questo documento chiude tre cose in una sessione: l'audit del modello precedente (`leg_model`),
la sostituzione dell'etichettatura, e la misura che dice cosa farne. Le tre parti vanno lette in
quest'ordine, perché ognuna è la ragione della successiva.

---

## 1. Perché `leg_model` è uscito dalla catena

> **Il codice di questa sezione non c'è più (cancellato il 2026-08-30).** `ml/leg_trainer.py`, la
> funzione `signals.leg_signals` e il ramo di dispatch che la serviva sono stati tolti: un
> `leg_model.joblib` rimasto su disco non riporta più la pagina su questo modello, e un test lo
> verifica. `labeling.swing_leg_target` invece **resta**, perché è l'etichetta che
> `modello-ingresso.md` usa come termine di confronto. La misura sotto vale ancora ed è il motivo
> per cui la sezione resta scritta.

Un revisore in contesto fresco, senza le conclusioni dell'autore, ha esaminato
`leg_trainer.py`, `bar_features.py`, `positioning.py`, `leg_signals` e `barrier_widths` contro un
contratto di nove requisiti. Su una ventina di rilievi il rumore è stato **zero**. I quattro
strutturali sono lo stesso difetto visto da quattro lati — *il ciclo di validazione non misurava
la strategia spedita*:

| # | rilievo | verificato |
|---|---|---|
| 1 | Il controllo casuale campiona righe i.i.d. da una popolazione dove ogni riga si sovrappone alla successiva per 7/8 dell'orizzonte e 15 simboli condividono il timestamp. Con un bootstrap a blocchi settimanali il p95 passa da −0,22 a **+0,06** e il modello scende all'80° percentile: non passa. | sì |
| 2 | Entrambe le soglie sono scelte **sul campione di verifica** (`idxmax` dello sweep su `fuori`, quantile delle predizioni di `fuori`). | sì |
| 3 | Il verdetto `PASSA` si accontenta di battere un p95 anch'esso negativo. Il netto medio per ingresso è **negativo a tutte e sei le soglie** (−0,149 … −0,091). | sì, letto dai metadata |
| 4 | Lo sweep ottimizza `rendimento_%`, che chiude a +1,5 ATR — cioè **con** il take profit, la variante che la misura dell'autore dà per peggiore fra sei. E la testa d'uscita non è mai valutata contro un P&L. | sì |

`percentile: 100.0` ripetuto sei volte su sei era il campanello: un controllo che risponde sempre
la stessa cosa non sta misurando niente.

Due difetti di contaminazione confermati a mano, entrambi da una riga:

- **`forza_su_btc` è un marcatore d'identità esatto per BTC.** Vale `0,0` su 19.691 righe BTC su
  19.691, e mai su ETH. Una sola divisione in un albero isola BTC alla perfezione — cioè
  esattamente ciò che il modulo dichiara di voler evitare nel proprio docstring d'apertura.
- **`sopra_ema200` mentiva sulle barre di riscaldamento.** `NaN > x` è `False` e
  `False.astype(float)` è `0,0`: tutte e 199 le barre prima che la EMA200 esista dicevano «sotto
  la EMA200». È lo stesso difetto già corretto per `atr_rel` **una riga sopra**, lasciato in piedi
  una riga sotto. In pagina è peggio che in addestramento, perché la finestra caricata è corta.

Corretto `sopra_ema200`; `forza_su_btc` è caduta con le colonne trasversali (§3). `leg_model` è
fuori da `MODEL_PRECEDENCE` con la ragione scritta accanto alla costante.

**Conseguenza sull'AUC.** Lo 0,5639 dichiarato «il più alto mai prodotto dal progetto, sopra il
soffitto ~0,54» non è un risultato: **è un allarme**, e i due difetti sopra sono candidati
concreti a spiegarlo.

---

## 2. La domanda nuova

La barriera tripla chiede *«il prezzo si muove di 1,5 ATR entro l'orizzonte?»*. È una domanda
sulla **volatilità**, e siccome le barriere sono già scalate sull'ATR l'etichetta normalizza via
proprio la parte prevedibile. Misurato: l'ampiezza futura ha |IC| 0,42 con 10/10 asset concordi,
la direzione 0,06.

`labeling.swing_target` chiede invece *dove sta questa barra rispetto alle sue vicine*: il rango
centrato della chiusura in `[-1, 1]`, −1 su un minimo locale, +1 su un massimo, ~0 dentro una
tendenza regolare.

**Quell'ultima proprietà è il punto.** Dentro una salita costante la barra centrale ha metà
finestra sopra e metà sotto per costruzione, quindi il target vale 0 e non +1: satura solo dove la
salita si esaurisce. Un «massimo dei prezzi futuri», o una distanza da quel massimo, marcherebbero
come *vicino al massimo* tutta la salita, ed è ciò che rende quelle etichette inservibili.

Implementata con due rolling causali invece che con una finestra centrata: quello all'indietro dà
la posizione fra le `W` barre precedenti, quello sulla serie rovesciata fra le successive, e la
somma meno uno è il rango centrato esatto. Costa `O(n log W)` invece di materializzare
`n × (2W+1)` valori, che a 5m su quindici simboli non ci sta in memoria.

### 2.1 Metà del target è gratis, e il metro deve saperlo

Il rango centrato usa anche le `W` barre **passate**, che le feature già descrivono:

| | IC contro il Target pieno | IC contro la sola metà futura |
|---|---|---|
| uno **Stochastic** (rango passato, zero modello, zero futuro) | **+0,703** | +0,050 |
| il modello a gradienti | +0,670 | +0,054 |

Il 93% del target è ricostruibile dal passato. Valutare lì misura soprattutto quanto bene il
modello rifà uno Stochastic — cosa in cui **perde**. Da qui il parametro `verso` di
`swing_target`, e la regola che ogni cifra si misura contro `verso="avanti"`.

Nota utile: addestrare **sul target centrato** e misurare sul futuro dà 0,053; addestrare
direttamente sul solo futuro dà 0,032. La metà passata fa da regolarizzatore.

---

## 3. Le decisioni di disegno, tutte misurate

Sette simboli, verifica 2024–2026, IC di Spearman contro la metà futura:

| variante | colonne | IC |
|---|---|---|
| `pos_canale` da solo, nessun modello | 1 | +0,0433 |
| base 5m | 15 | +0,0502 |
| + storico esplicito a −1 e −2 barre | 45 | +0,0509 |
| + storico fino a −8 barre | 75 | +0,0510 |
| + storico fino a −32 barre | 105 | +0,0498 |
| + Target ritardato di `W+1` (l'unico ritardo lecito) | 16 | +0,0497 |
| **+ aggregazione 1h e 1d** | **41** | **+0,0540** |
| + tutte e quattro le scale (15m, 1h, 4h, 1d) | 67 | +0,0539 |
| + Target ritardato di **1** barra | 16 | +0,6729 |

**Niente storico esplicito.** Ricopiare le feature indietro costa il triplo delle colonne per due
millesimi, e oltre le due barre peggiora. Lo storico c'è già, compresso in EMA200, ADX e OBV a 20
barre.

**Niente Target fra le feature.** Al solo ritardo lecito vale −0,0005. L'ultima riga della tabella
non è un risultato ma **la misura del danno**: a ritardo 1 il target condivide 143 delle sue 144
barre con quello di oggi, quindi quel +0,67 è la fuga di informazione. Serve tenerla scritta
perché è l'idea più pericolosa dell'intero disegno: produrrebbe un modello spettacolare in tabella
e inservibile in produzione.

**Niente colonne trasversali.** Dipendono dagli altri quattordici asset e in pagina si carica un
simbolo alla volta. Ne è caduto anche `forza_su_btc` (§1).

**Aggregazione a 1h e 1d**, allineate con `mtf.align_to_lower` così che la barra lunga si legga
solo dopo che ha chiuso. 15m e 4h non aggiungono nulla: stanno troppo vicine a ciò che EMA200 e
ADX sulla base già descrivono.

---

## 4. Il modello addestrato

Quindici simboli, 5m **dal 2018** — lo store arriva lì, e il posizionamento resta NaN prima del
2021-12, il che insegna al modello lo stato «posizionamento assente», che è la condizione di
produzione. 1.063.757 righe di stima contro 692.047 di verifica.

Il numero di giri di rinforzo — riaddestramento su etichette riviste con le predizioni **fuori
piega** del modello stesso — è scelto su una fetta di validazione ritagliata dallo stima, mai sul
fuori campione. È la correzione diretta del rilievo §1.2.

```
giro 0: IC validazione +0.0894  →  giro 3: +0.0910   (scelto: 3)

Fuori campione 2024-01 .. 2026-08
  IC contro la metà futura   +0.0385
  riferimento causale        +0.0297   (pos_canale, nessun modello)
  eccesso                    +0.0088
  mediana per simbolo        +0.0458   (14/15 concordi di segno)
```

Il rinforzo funziona ma poco: **+0,0016 in tre giri**.

**Robustezza — ed è il miglioramento vero rispetto a `leg_model`:**

| scenario | IC |
|---|---|
| tutto presente | +0,0540 |
| senza `@1d` (finestra di pagina corta) | +0,0524 |
| senza `@1h` e `@1d` | +0,0542 |
| senza posizionamento | +0,0539 |

Non si degrada. Il modello precedente senza le trasversali crollava da +1,9% a −39,5%.

**Inferenza: 283 ms per 20.000 barre** (234 di feature, 49 di predizione).

---

## 5. Perché non è cablato

`scripts/swing_lab.py` fa tre misure, e ognuna decide se la successiva ha senso.

### 5.1 La forma del segnale è a U, non monotona

Eccesso di rendimento a 48 ore per decile di previsione:

| finestra | decili 0 → 9 |
|---|---|
| validazione | **+0,184** −0,096 −0,030 −0,010 −0,063 −0,068 −0,000 +0,067 **+0,093** −0,076 |
| fuori campione | **+0,088** −0,093 −0,087 −0,098 −0,093 −0,107 −0,021 +0,079 **+0,180** +0,152 |

Sia il decile più basso — che il modello legge come «vicino a un minimo locale» — sia i più alti —
«vicino a un massimo» — precedono rendimenti sopra la media; il centro sta sotto. È replicato in
entrambe le finestre.

**Il modello non prevede la direzione, prevede la struttura.** Il polo +1 non è «vendi»: è
«tendenza forte in corso», e in cripto la continuazione paga. Vendere sui massimi previsti — la
lettura naturale di un target in `[-1, 1]`, e quella che chiedeva la specifica — **vende
esattamente le barre migliori**. Da cui il P&L della regola direzionale: perde a tutte le soglie e
tutte le cadenze, in validazione come fuori campione, da −0,05% a −0,42% netti per operazione.

### 5.2 La regola che la forma sostiene è un filtro di esposizione

Dentro quando `|previsione|` è alta, fuori quando è bassa, con isteresi.

| finestra | configurazione | netto/op | composto | passivo |
|---|---|---|---|---|
| fuori campione | 0,50 / 0,40 / 288 | **+0,086%** | −15,3% | −33,5% |
| validazione | la stessa | −0,194% | −9,7% | +43,7% |
| validazione | 0,35 / 0,25 / 288 | +0,311% | +1,8% | +43,7% |
| fuori campione | la stessa | −0,191% | −57,0% | −33,5% |

**Nessuna configurazione va bene in entrambe le finestre**, e prendere quella che funziona fuori
campione sarebbe tararsi sul campione di verifica — il rilievo §1.2 di nuovo.

### 5.3 Il controllo che chiude la faccenda

Stare fuori dal mercato il 76% del tempo batte il possesso passivo dentro un ribasso **per
costruzione**. La domanda vera è se lo batte meglio di collocare la stessa esposizione, con le
stesse durate, a caso. Duecento estrazioni per simbolo:

> **1 simbolo su 15 in validazione, 1 su 15 fuori campione** supera il p95, contro **0,75** attesi
> dal caso.

Il merito della regola è l'astensione, e per quella non serve un modello.

### 5.4 Cosa è stato cablato, e cosa no

Il modello è ora in testa a `MODEL_PRECEDENCE` e vota in Confluence. Le tre misure sopra non sono
diventate favorevoli: quello che è cambiato è che **la lettura sbagliata non è più raggiungibile
dal codice**. Prima il rischio era che qualcuno leggesse un target in `[-1, 1]` e cablasse il
segno; ora l'unica strada che esiste è `|previsione|`, e le tre docstring che la implementano
dicono perché.

| dove | cosa è cablato | cosa **non** lo è |
|---|---|---|
| `ml/signals.swing_exposure` | `|previsione|` alta → dentro, con isteresi | `sign(previsione)` come direzione (§5.1: perde a tutte le soglie) |
| `trading/strategies.ai_model_simulation` | l'uscita è l'ingresso letto al contrario | barriere, take profit, stop: il modello non è stato misurato con nessuno dei tre |
| `trading/confluence._modello` | voto +1 o 0, in una famiglia sua | il voto −1, che darebbe un corto sulle barre migliori |

Tre scelte di protocollo, tutte prese per non ripetere §1.2:

- **le soglie sono 0,35/0,25**, scelte sulla validazione. Fuori campione rendono −0,191% per
  operazione contro il +0,086% di 0,50/0,40. Prendere le seconde *perché* rendono sul 2024-2026
  sarebbe tararsi sul campione di verifica, cioè il difetto per cui `leg_model` è uscito. In
  Confluence sono due manopole (`CONF_MODELLO_ENTRA`/`ESCI`) perché §5.2 misura che la coppia
  buona cambia da una finestra all'altra: tenerla in una costante farebbe credere che ne esista
  una giusta;
- **senza artefatto il votante resta fuori dall'insieme di default**, non semplicemente muto. I
  pesi si normalizzano sui votanti presenti, quindi un ottavo che tace sempre alzerebbe di fatto
  la soglia per gli altri sette — e in produzione `models/` è vuoto per costruzione. Nel registro
  ci resta, così `selezione("modello")` lo raggiunge per misurarlo;
- **la nota del riquadro dice che non batte il possesso passivo.** È l'unica parte di questo
  documento che arriva a chi guarda il grafico.

Due cose sono emerse scrivendo il percorso di servizio, e nessuna si vedeva leggendo il trainer:

- **le scale lunghe vanno prese solo se sono più lunghe della base.** A 4h, aggregare a un'ora
  significa ricampionare all'insù, cioè inventare barre. Le colonne che restano fuori diventano
  NaN, ed è la degradazione già misurata al §4;
- **e solo se hanno almeno 28 barre.** `ExtraCache.adx(14)` passa da `ta`, che sotto due finestre
  solleva `IndexError` invece di restituire NaN. In addestramento non si vede — le serie sono di
  centinaia di migliaia di barre — ma la pagina carica per default 240 ore, cioè dieci barre
  giornaliere, e lì la voce «AI Model» cadeva appena selezionata.

**Misurato dopo il cablaggio**, e da leggere come conferma del §5.3 e non come risultato: BTC a 1h
dal 2025, 104 operazioni, −21,1% contro un passivo di −27,2%. È il merito dell'astensione. Dentro
Confluence, su 92.321 barre 15m dal 2024, il votante è lungo il 56,4% delle barre, mai corto, e
**necessario nel 10% degli ingressi**: aggiunge senza dominare, che è la sola condizione in cui
valeva la pena aggiungerlo.

---

## 6. Cosa resta vero

- **Il segnale statistico esiste**: IC +0,0385 fuori campione contro un riferimento causale di
  +0,0297, 14/15 simboli concordi di segno, e una forma a U replicata in due finestre disgiunte.
  Non è rumore.
- **Non è redditizio a queste frequenze.** Il miglior eccesso misurato è +0,20% su 48 ore contro
  un giro di commissioni che ne costa 0,20%. È la tassa di conferma di `strategy.md` §13, per la
  terza volta indipendente in questo progetto.
- **La (c) è stata fatta** (§5.4): il modello è un votante di Confluence, dove non deve battere
  il possesso passivo da solo. Che paghi non è ancora misurato — serve rifare la griglia di
  `scripts/confluence_lab.py` con e senza il votante, sugli stessi asset e la stessa finestra.
  Finché quel confronto non c'è, l'unica cosa che si sa è che il votante non domina la decisione.
- **Due strade restano**, in ordine di costo: (a) usare `|previsione|` per **dimensionare** la
  posizione invece che come interruttore — l'unica forma che non tronca la coda destra; (b)
  portare la decisione su scala giornaliera, dove il rapporto fra eccesso e commissioni cambia di
  un ordine di grandezza.

## 7. Riprodurre

```bash
.venv312/bin/python -m cryptofarm.data.positioning --update     # store del posizionamento, 400 MB
.venv312/bin/python -m cryptofarm.ml.swing_trainer --selfcheck  # gira senza store
.venv312/bin/python -m cryptofarm.ml.swing_trainer              # ~12 minuti
.venv312/bin/python -m scripts.swing_lab                        # le tre misure del §5
```
