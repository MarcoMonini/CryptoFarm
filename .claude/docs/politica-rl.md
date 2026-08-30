# La politica RL: il costo dentro la ricompensa

*Misurato il 2026-08-28, 15 simboli, barre da 5 minuti dal 2019.*
Codice: `ml/rl.py`, `ml/rl_trainer.py`, banco `scripts/rl_lab.py`, servizio `ml/signals.rl_*`.

**Stato: cablata.** `rl_model` e' in testa a `trainer.MODEL_PRECEDENCE`, la voce «AI Model» la
esegue e il votante `modello` della confluenza la usa quando l'artefatto c'e'.

---

## 1. La domanda di partenza, e perche' la risposta ovvia era sbagliata

La richiesta era: *il modello fa buoni trade ma non evita i crolli, compra poco prima, e il segreto
per battere il possesso passivo e' evitarlo tenendo il comportamento sul laterale e sfruttando le
gambe rialziste.*

Le due premesse si misurano. Regola cablata precedente (`|previsione| ≥ 0,35` per entrare, `< 0,25`
per uscire, una decisione al giorno), 15 simboli, fuori campione dal 2024-01.

### 1.1 «Compra poco prima dei crolli» — **falsa**

Discesa massima nei tre giorni dopo un ingresso, contro quella dopo una barra giornaliera qualunque
dello stesso periodo:

| | n | mediana | p10 | p05 | quota sotto −10% |
|---|---|---|---|---|---|
| dopo un ingresso | 3.000 | −3,88% | −11,87% | −15,82% | 15,1% |
| dopo una barra qualunque | 14.385 | −3,94% | −11,84% | −15,72% | 14,7% |

Le due distribuzioni sono indistinguibili. Gli ingressi **non** cadono davanti ai crolli piu' di
quanto ci cada il caso. Quello che si vedeva in pagina era vero — ci sono ingressi seguiti da
crolli — ma e' il tasso di base, non una firma del modello.

### 1.2 «Tagliare i crolli con uno stop» — **dannosa**

Stessi ingressi, uscita anticipata da uno stop. Somma dei rendimenti netti fuori campione:

| uscita | somma | operazioni fermate dallo stop | peggiore |
|---|---|---|---|
| solo modello | **−201%** | 0% | −33,8% |
| stop fisso 3% | −229% | 52% | −3,2% |
| stop fisso 5% | −314% | 34% | −5,2% |
| stop fisso 8% | −327% | 18% | −8,2% |
| trailing 5% | −563% | 57% | −5,2% |
| trailing 12% | −418% | 15% | −12,2% |

Ogni livello peggiora, e peggiora **monotonamente al crescere di quanto morde**. Lo stop taglia la
coda destra piu' di quella sinistra: e' la stessa cosa che dice la forma a U del modello a swing
(`modello-swing.md` §5.1), vista da un'altra angolazione.

### 1.3 Dove va davvero il denaro — **nella commissione**

| | lordo | netto | operazioni | costi impliciti |
|---|---|---|---|---|
| fuori campione | **+401%** | **−201%** | 3.009 | 3.009 × 0,2% = **602%** |

Il segnale c'e' al lordo. Lo mangia il numero di giri. La conferma piu' netta: passando la *stessa*
regola, *lo stesso modello*, da una decisione al giorno a una ogni due giorni, il netto va da
**−201% a +306%**. Nessun addestramento, nessuna feature nuova: meta' dei giri.

**Da qui la forma dell'agente.** Non un filtro di rischio — misurato dannoso — ma una politica che
sceglie la posizione sapendo quanto costa cambiarla.

---

## 2. La formulazione

Stato `s` = le stesse 41 colonne del modello a swing (`bar_features.SWING_COLUMNS`) piu' la
**posizione corrente**. Azione `a ∈ {fuori, dentro}`. Ricompensa di un passo:

```
r(s, p, a) = a · log(P'/P) − costo · |a − p|
```

Tre conseguenze, e sono le tre ragioni per cui questa forma e' diversa da quelle gia' chiuse in
negativo:

- **la banda di non-fare non e' scritta a mano.** Le due soglie della regola precedente erano un
  parametro; qui l'isteresi e' cio' che emerge quando cambiare costa e la posizione sta nello stato;
- **la classe di politiche contiene il possesso passivo** (`a ≡ 1`), che e' il riferimento da
  battere. L'agente puo' solo aggiungere a una politica che sa gia' rappresentare;
- **il vincolo economico sta dentro il target.** E' l'unica riformulazione che `strategy.md` §13.4
  lasciava aperta dopo che la politica a tre azioni era risultata a somma nulla per costruzione:
  quella entrava alla conferma di un minimo e usciva alla conferma di un massimo, e la conferma si
  paga due volte contro una gamba mediana che vale 1,76-2,05 soglie.

**Algoritmo:** fitted Q-iteration offline su un batch fisso — nessuna interazione con l'ambiente,
quindi nessuno shift di distribuzione da politica che esplora. Due `HistGradientBoostingRegressor`,
uno per azione, sullo stato `[feature, posizione]`. Il bersaglio di ogni giro e'
`r + γ · max_a' Q(s', a')`, dove `s'` porta come posizione **l'azione appena presa**: e' quel
collegamento che rende il costo un investimento invece che una tassa istantanea.

### 2.1 Le tre costanti, e chi le ha scelte

Stanno in `ml/rl.py` e **non si riscelgono a ogni addestramento**: la griglia che le ha fissate
(12 celle) e' stata percorsa una volta sola in validazione.

| costante | valore | perche' |
|---|---|---|
| `CADENZA` | 288 barre = 1 giorno | la cadenza a cui la regola precedente e' misurata |
| `FEE` | 0,001 per lato | il costo vero, quello con cui si **misura** |
| `COSTO` | **0,012** | il costo che l'agente vede, dodici volte quello vero |
| `GAMMA` | 0,95 | ≈ venti giorni di orizzonte a cadenza giornaliera |

`COSTO ≠ FEE` non e' un errore. E' il termine che allarga la banda di non-fare, scelto in
validazione fra 0,001 / 0,004 / 0,012: col costo vero la politica gira 203 volte in due anni e
mezzo e resta sotto il caso, a 0,012 ne gira 184 e ci passa sopra. Chi lo cambia sappia che sta
cambiando il problema.

### 2.2 Il difetto che toglieva l'85% del campione

Il filtro sui NaN scartava la riga **intera** se una colonna mancava. Le due colonne di
posizionamento (`data/positioning.py`) non esistono per interi anni sui simboli entrati tardi nei
futures, quindi il campione di stima scendeva da 165.605 righe a 29.234 — e sparivano proprio i
primi anni, cioe' l'unico ciclo completo dentro il periodo di stima.
`HistGradientBoostingRegressor` i NaN li tratta da solo. La condizione ora e' `any`, non `all`, e
c'e' un test.

---

## 3. I tre periodi

| periodo | quando | a cosa serve |
|---|---|---|
| stima | 2019-01 → 2022-06 | addestra i due regressori (121.806 transizioni, 8 sfasature) |
| validazione | 2022-06 → 2024-01 | **sceglie i giri di iterazione** (3 su 1/2/3/5) |
| fuori campione | 2024-01 → oggi | si guarda una volta e non decide niente |

La validazione parte dal 2022-06 apposta: contiene il **ribasso del 2022 e il rialzo del 2023**.
Sceglierla dentro un regime solo avrebbe scelto «stare fuori dal mercato», che in un ribasso vince
sempre e non e' una capacita'. Fra stima e validazione c'e' un embargo di `144 + cadenza` barre —
144 e' la finestra del target a swing.

---

## 4. I risultati, e il metro giusto

`python -m scripts.rl_lab`. **Il controllo e' piu' stretto di quello di `swing_lab`**: invece di
collocare a caso durate simili, rimescola i **blocchi della politica stessa**. Esposizione totale,
numero di blocchi e loro durate restano identici; cambia solo *dove* cadono. Cio' che resta e'
esattamente il valore del *quando*.

E si misura il **rango percentile** fra 400 estrazioni, non quante volte si supera il p95: con 15
simboli il conteggio butta via quasi tutta l'informazione, e 0,75 successi attesi contro 2 osservati
non distinguono niente. Il rango medio atteso, se il momento non conta, e' **0,500**.

| | batte il possesso | rango medio | Wilcoxon | discesa massima | espos. media | espos. nei 10 passi peggiori |
|---|---|---|---|---|---|---|
| validazione | 9/15 | 0,588 | p=0,277 | **−40,8%** contro −58,3% | 39% | 48% |
| fuori campione | 11/15 | 0,602 | p=0,169 | **−48,8%** contro −76,0% | 37% | **25%** |

### 4.1 Cosa dicono, letti onestamente

- **La discesa massima si dimezza in entrambe le finestre.** E' il risultato piu' solido e piu'
  consistente, ed e' la traduzione operativa della domanda di partenza: il capitale non evita i
  crolli scommettendo *quando* arrivano, li attraversa esposto per meta'.
- **Il *quando* sta sopra il caso in tutte e due le finestre** (0,588 e 0,602 contro 0,500) ma
  **non raggiunge la significativita'** in nessuna delle due. Il segno e' coerente, la forza no.
  E' comunque piu' di quanto avesse la regola precedente, che era al livello del caso.
- **Batte il possesso passivo 11/15 fuori campione, 9/15 in validazione.** Il secondo numero e' una
  monetina, e va detto: in un mercato al rialzo una politica esposta al 39% non tiene il passo. Il
  vantaggio si concentra dove il possesso passivo perde.
- **L'esposizione condizionata ai crolli e' discordante**: 25% contro 37% medio fuori campione (la
  politica *e'* meno esposta nei passi peggiori), ma 48% contro 39% in validazione. E' un conto a
  posteriori e descrive un comportamento, non dichiara una capacita'.

### 4.2 Cosa e' stato provato e non funziona

- **Colonne di mercato nello stato.** L'ipotesi era che i crolli in cripto siano sistemici e che
  lo stato, tutto per singolo asset, non li possa vedere. Aggiungendo ampiezza di mercato (quota
  dei 15 sopra la loro EMA200 giornaliera) e la struttura di BTC, il fuori campione passa da
  +14,7% a **−25,5%** mediano e il rango scende. **Respinta.**
- **Piu' giri di iterazione.** Oltre il terzo il bersaglio contiene la stima di se stesso e la
  varianza cresce piu' del guadagno di orizzonte.

---

## 5. Cosa e' cablato

| dove | cosa |
|---|---|
| `trainer.MODEL_PRECEDENCE` | `rl_model` in testa, davanti a `swing_model` |
| `strategies.ai_model_simulation` | ramo `rl_model` → `signals.rl_signals`, senza soglia ne' barriere |
| `confluence._modello` | **lo stesso votante**, che serve il modello in testa alla catena |
| `panels.STRATEGIE["AI Model"].note` | dice cosa fa e con che forza, in pagina |

**Perche' non un nono votante.** Un secondo votante a modello risponde alla stessa domanda a
partire dalle stesse 41 colonne: voterebbero insieme, e l'ampiezza minima della confluenza si conta
in famiglie proprio per non far pesare due volte la stessa opinione. Con la politica presente,
`CONF_MODELLO_ENTRA` e `CONF_MODELLO_ESCI` non hanno effetto — la soglia la politica ce l'ha dentro
l'obiettivo — e tornano a contare se resta solo il modello a swing.

**In servizio la cadenza e' un giorno a qualunque intervallo** (`signals.swing_cadenza`). Non e'
una manopola: e' il passo su cui il costo dentro la ricompensa e' calibrato. A cadenza oraria la
stessa politica pagherebbe ventiquattro volte i costi che il suo obiettivo aveva messo in conto.

---

## 6. Cosa manca

- **Il paniere.** Tutto qui e' per singolo asset. `strategy.md` §6.1 conta 1.691 episodi
  giornalieri di allocazione, che sono pochissimi per il RL, e indica il Kelly frazionario come
  alternativa quasi certamente sufficiente.
- **La significativita' del *quando*.** Due finestre concordi nel segno e nessuna significativa.
  Serve o piu' storia o un controllo con piu' potenza — per esempio il rango su tutte le finestre
  di una CPCV invece che su due periodi.
- **La griglia della confluenza con e senza il votante**, sugli stessi asset e la stessa finestra.
  E' rimasta da fare anche per il modello a swing (`modello-swing.md` §6): finche' non c'e', non si
  sa se il votante a modello paga.
