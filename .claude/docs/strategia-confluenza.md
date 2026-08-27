# Confluenza — disegno di una strategia multi-timeframe a più segnali

Ipotesi di lavoro del **2026-08-27**, chiesta dall'utente. Non è ancora misurata: niente qui è un
risultato. Quello che è già scritto e verificato è il primo stadio, `trading/mtf.py`.

## Il principio, e perché la struttura conta più dei segnali

L'idea è quella classica dell'analisi dall'alto: il quadro macro decide *se*, gli intervalli
intermedi decidono *se davvero*, l'intervallo breve decide *quando*. La trappola è costruirla come
quattro voti sulla stessa domanda — «c'è un trend?» — su quattro scale. Quello non è un consenso:
è **una sola opinione contata quattro volte**, e il punteggio che ne esce sembra continuo ma è
binario travestito.

Quindi la regola di disegno è: **ogni piano risponde a una domanda diversa**, e i piani non si
possono sostituire fra loro.

| piano | intervallo | domanda | uscita |
|---|---|---|---|
| Regime | 1D | si può stare lunghi, in assoluto? | cancello 0/1 + forza ∈ [0,1] |
| Struttura | 4H | il trend di medio periodo è allineato? | punteggio direzionale ∈ [−1,+1] |
| Conferma | 1H | il movimento è confermato o esaurito? | punteggio ∈ [−1,+1] |
| Innesco | 15m | esattamente quando, e con quale stop | evento + livello |

Il piano 1D produce un **cancello**, non un segnale: non contribuisce al punteggio, lo abilita.
Confonderli è il difetto più comune di questi sistemi — un regime forte finisce per compensare
l'assenza di conferma, e si compra dentro una distribuzione.

## I sei votanti, scelti per famiglia e non per numero

Cinque strategie di prezzo sono cinque modi di misurare la stessa cosa. La scelta qui privilegia
**famiglie diverse**, e in particolare include l'unica famiglia che non guarda il prezzo:

| # | votante | intervallo | famiglia | perché è dentro |
|---|---|---|---|---|
| 1 | `ichimoku_trend` | 4H | inseguimento del trend | **l'unica regola per-asset con mediana positiva fuori campione** (+13,5%, ρ 0,38) |
| 2 | `donchian_breakout` | 4H | rottura di canale | famiglia diversa da 1; 87% in utile in campione, e fuori campione è la peggiore per mediana — sta dentro come voto, mai da sola |
| 3 | `squeeze_breakout` | 1H | regime di volatilità | non misura la direzione ma la **compressione**: è ortogonale per costruzione |
| 4 | `trend_pullback` | 1H | rientro dentro il trend | vota *contro* l'ingresso sull'estensione, cioè corregge il difetto tipico di 1 e 2 |
| 5 | `band_reversion_gated` | 15m | ritorno alla media | l'unico che può opporsi a tutti gli altri; negativa da sola su 4 asset su 5, ed è esattamente il motivo per cui serve come voce di minoranza |
| 6 | `obv_slope` + `mfi` | 4H | **flusso di volume** | l'unico votante che non legge il prezzo. È la mossa che decorrela di più, e il codice esiste già in `ExtraCache` |

Tutti e sei restituiscono uno **stato per barra** in `{−1, 0, +1}`, non eventi: si ottiene
propagando in avanti i cambi di posizione che `strategies_ls` già produce. È un adattatore, non
una riscrittura.

## La memoria del segnale: è ciò che rende possibile la confluenza

Un segnale non vale solo sulla barra in cui scatta. Ed è una necessità, non una comodità: un voto
a 4H e uno a 1H non cadono quasi mai sulla stessa barra da quindici minuti, quindi **senza memoria
il punteggio è quasi sempre sparso e la confluenza non innesca mai**. La memoria converte
«conferme simultanee», che sono rare, in «conferme entro una finestra», che sono frequenti — ed è
il meccanismo che fa aumentare le occasioni invece di ridurle.

```
sᵢ(t) = gᵢ(t)                    se il votante i scatta a t
sᵢ(t) = sᵢ(t−1) · λᵢ             altrimenti,   con  sᵢ = 0  sotto ε
```

`λᵢ` deriva da **una sola** emivita globale, espressa in barre del timeframe del votante: così un
segnale giornaliero resta vivo per giorni e uno a quindici minuti per ore, senza sei parametri. È
una ricorsione, O(N), e decade invece di spegnersi di colpo — un voto vecchio pesa meno di uno
fresco senza che nessuno debba deciderlo caso per caso.

## Il punteggio, e i due freni contro la dipendenza da un solo votante

```
punteggio(t) = Σ wᵢ · sᵢ(t)                          Σwᵢ = 1,  wᵢ ≤ w_max
accordo_alto(t) = (regime_1D(t) + struttura_4H(t)) / 2       ∈ [−1, +1]
soglia(t) = θ_base − θ_macro · accordo_alto(t)
```

Dare più potere ad alcuni segnali va bene; il rischio è che l'insieme diventi *un* segnale con
delle decorazioni. Due freni, entrambi misurabili:

1. **Tetto per votante.** `wᵢ ≤ w_max` (0,30 con sei-sette votanti, contro un peso uguale di
   0,14-0,17), rinormalizzando dopo il taglio. Nessuno può valere più di circa il doppio della
   media, qualunque cosa dica la taratura.
2. **Ampiezza obbligatoria.** Per entrare non basta `punteggio ≥ soglia`: servono anche **almeno
   k famiglie distinte** concordi (k = 2 o 3). Famiglie, non votanti — ed è per questo che i sei
   sono stati scelti per famiglia. Un peso grande, da solo, non può aprire una posizione.

E una diagnosi che va **riportata accanto a ogni risultato**, non tenuta da parte: la
**necessarietà per votante**, cioè in che frazione degli ingressi quel votante era indispensabile
(azzerandolo, l'ingresso non sarebbe avvenuto). Se un votante è necessario in più del 60% degli
ingressi, l'insieme è quel votante travestito, e il numero lo dice prima che lo dica il mercato.

La soglia **non è un numero tarato per regime**: sono i piani alti a decidere quanta conferma
serve. Quando 1D e 4H concordano con forza, `accordo_alto ≈ +1`, la soglia scende e si accetta un
ingresso con meno conferme dal basso. Quando si contraddicono la soglia sale e serve quasi
l'unanimità. È la tua idea — «le condizioni di mercato definiscono i pesi di veridicità» — resa in
**due parametri invece che in un classificatore**.

**Ingresso** quando, sulla stessa barra 15m: il cancello 1D è aperto, `punteggio ≥ soglia`,
almeno k famiglie concordano, e l'innesco 15m scatta. L'innesco serve a rendere l'ingresso davvero a quindici minuti: senza, si sta
solo eseguendo una decisione 4H con risoluzione più fine.

**Dimensione** proporzionale al margine sopra la soglia, moltiplicata per il rapporto fra
volatilità obiettivo e volatilità realizzata. È il punto in cui il passo 3 del piano generale
(`piano-strategie.md`) si innesta senza modifiche.

**Uscita**, la prima delle tre che arriva:
1. `punteggio < soglia − isteresi` — l'isteresi non è un dettaglio: senza, si entra e si esce sulla
   stessa barra ogni volta che il punteggio oscilla attorno alla soglia;
2. stop a trailing ATR su 15m, **calcolato su barre chiuse a `i−1`**;
3. il cancello 1D si chiude — si va flat senza discutere.

## La ricostruzione delle barre lunghe «in formazione»

Il bot live, alle 10:00, non aspetta la mezzanotte: vede una barra 1D aperta all'apertura, con
massimo e minimo correnti e chiusura provvisoria pari all'ultimo prezzo. **Quella barra parziale
non è look-ahead**, perché è costruita solo con dati fino alle 10:00 — ed è una cosa diversa dalla
barra 1D *completa* di quel giorno, che invece lo sarebbe. Il backtest deve replicare la prima.

Non è una raffinatezza: aspettare la chiusura giornaliera vuol dire reagire fino a ventiquattro ore
dopo, e la maggior parte dei segnali muore in quell'attesa. È il secondo meccanismo, dopo la
memoria, che fa **aumentare** le occasioni.

### Il costo, e perché il ciclo non serve

Rifare aggregazione e indicatori a ogni barra breve è quadratico: su cinque anni a quindici minuti
sono 175.200 passi, ognuno che ripercorre la storia. Misurato in questa sessione, estrapolando dal
costo di una riaggregazione: **dell'ordine delle ore, per un solo intervallo e una sola
configurazione.** Su una griglia non si esegue.

Non serve, perché la barra in formazione ha forma chiusa: dentro il periodo l'apertura è la prima,
il massimo è il massimo *corrente*, il minimo il minimo corrente, la chiusura è il prezzo di adesso
e il volume la somma corrente. `groupby` più `cummax`/`cummin`/`cumsum` le producono tutte senza
nessun ciclo Python. Misurato: **103 ms per cinque anni e tre intervalli** (`trading/live_frames.py`).

Due proprietà rendono la cosa economica anche sulla griglia:

- **la parte cara non dipende da nessun parametro di strategia.** Le barre in formazione si
  calcolano una volta per (simbolo, intervallo) e si riusano su tutte le configurazioni;
- **gli indicatori ricorsivi si sollevano in O(1).** Lo stato (EMA, ATR di Wilder, KAMA, ADX) resta
  fermo all'**ultima chiusura**; il valore provvisorio si ricava combinandolo con la barra parziale
  e non viene mai committato finché il periodo non chiude davvero. `provisional_ema` è il modello
  di tutti gli altri. Gli indicatori a finestra che già escludono la barra corrente — Donchian è
  shiftato — dipendono solo da barre chiuse e si calcolano una volta per periodo.

### Il difetto da una lettera

`groupby.transform("max")` restituisce il massimo dell'**intero** periodo, incluse barre non ancora
accadute. Contro `cummax` è un errore di tre caratteri, non lo segnala nessun tipo, e trasforma il
backtest in una macchina che conosce il futuro. `tests/test_live_frames.py` lo intercetta —
verificato reintroducendolo: cadono due test su sei. **Il test che confronta la barra alla chiusura
con quella aggregata continua a passare**, ed è il motivo per cui non basta.

La variante a sole barre chiuse (`mtf.align_to_lower`) resta e non va cancellata: serve come
**ablazione**. La differenza fra le due misura esattamente quanto vale reagire prima della
chiusura, ed è un numero che questo disegno ottiene gratis.

## Lo slot per il modello AI

Il modello è **un votante come gli altri**, non uno strato sopra. Interfaccia: nome, famiglia,
intervallo, e una funzione che restituisce un valore in [−1,+1] per barra. Quattro vincoli, tutti
conseguenza di misure già fatte:

- **causale**, addestrato solo su dati precedenti a `t`, con la validazione purgata che sta già in
  `ml/validation.py`. È l'unico votante che può barare sull'addestramento invece che sui dati;
- **si astiene**. Uscita 0 quando `|p − 0,5| < margine`, ±1 oltre. Con un'AUC misurata a 0,537 un
  modello che vota sempre, debolmente, aggiunge solo rumore: deve parlare poco e quando ha qualcosa
  da dire. È lo stesso schema degli «esperti dormienti»;
- **nessun privilegio di peso**: stesso `w_max` degli altri. Dato che il vantaggio economico del
  filtro meta è finito dentro il rumore del controllo casuale, dargli un peso grande sarebbe
  esattamente l'errore che quelle misure hanno evitato;
- **la sua famiglia è "trasversale"**, e lì sta il suo valore: è l'unico votante che può leggere
  rango di forza nell'universo, ampiezza di mercato e forza contro BTC — informazione che nessun
  votante di prezzo su un solo simbolo possiede. Se lo si addestra sulle stesse feature di prezzo
  degli altri, è ridondante per costruzione.

`meta_gate` produce già probabilità per operazione; serve la versione **per barra**. E va misurato
come tutti: insieme con il votante e insieme senza, più il controllo a selezione casuale.

## Il conteggio onesto dei parametri liberi

È la sezione che decide se questo disegno è misurabile o è un esercizio.

| voce | liberi | come |
|---|---|---|
| parametri dei sei votanti | **0** | **congelati** ai `tuned_defaults` misurati, mai ritarati dentro l'insieme |
| pesi | 0 | uguali nella versione base, sotto `w_max` |
| θ_base, θ_macro | 2 | |
| isteresi | 1 | |
| emivita del decadimento | 1 | una sola, in barre del timeframe di ciascun votante |
| `w_max` | 1 | |
| k famiglie minime | 1 | |
| stop ATR 15m | 0 | dai `tuned_defaults` a 15m |
| volatilità obiettivo | 1 | |
| innesco 15m | 1 | |
| **totale** | **9** | |

Nove, non cinque: la memoria e i due freni costano quattro parametri in più, e vanno dichiarati.
Nove restano trattabili, ma il conto delle prove per la correzione di molteplicità deve includere
**l'intera griglia su questi nove**, non le sole configurazioni finali guardate.

**Congelare i votanti è il vincolo portante.** Ritararli dentro l'insieme porta il conto a oltre
venticinque parametri su cinque anni di dati, e la correzione per molteplicità già applicata alla
rotazione (`multiplicity.py`, DSR della mediana 0,52 contando le prove del progetto) dice cosa
succede a quel punto: niente di quello che esce è distinguibile dalla fortuna.

## Come si misura, e contro cosa — dichiarato prima

1. **Il possesso passivo**, sempre.
2. **`ichimoku_trend` a 4h a parametri centrali**: il riferimento per-asset dichiarato. Una
   strategia complessa che non batte il proprio votante migliore non ha guadagnato niente dalla
   complessità.
3. **Il riferimento a frequenza appaiata**: la migliore singola strategia ritarata per fare lo
   *stesso numero di operazioni all'anno* della confluenza. È il controllo che separa «seleziona
   meglio» da «opera solo di meno», e in questo progetto è la distinzione che ha già spiegato quasi
   tutto.
4. **Ablazioni**: un piano spento alla volta, e un votante spento alla volta. `lab_report` le fa già.

## I tre modi in cui questo fallisce, dichiarati prima

1. **Look-ahead fra intervalli.** È l'unico che produce risultati falsi *positivi*, ed è il motivo
   per cui `trading/mtf.py` esiste ed è il primo pezzo scritto. `resample_klines` etichetta a
   sinistra: la barra 1D di oggi chiude domani, e leggerla stamattina inietta il resto della
   giornata nella decisione. Il test che tronca la serie **non lo vedrebbe**, perché tronca fra le
   barre corte e la barra lunga incriminata resta identica.
2. **Il campione.** Restava il rischio più probabile del disegno precedente: quattro piani di
   conferma simultanea fanno pochissime operazioni. La memoria del segnale e le barre in formazione
   lo attaccano direttamente — sono i due meccanismi che generano occasioni invece di sopprimerle —
   ma **quanto, è da misurare, non da assumere**. Il numero di operazioni all'anno va riportato
   accanto a ogni risultato, e la misura va comunque fatta su cinque asset in comune e per
   operazione, non per curva di equity.
3. **La correlazione fra i votanti.** Se i sei stati sono correlati 0,8 il punteggio ha tre valori
   effettivi e la soglia dinamica non ha niente su cui lavorare.

## Ordine di costruzione, uno stadio per volta

Ogni stadio aggiunge **un** meccanismo e si misura contro il precedente. Se uno non guadagna, si
scrive e ci si ferma li'.

| | cosa | stato |
|---|---|---|
| **S0** | correlazione fra i sei stati barra-per-barra | da fare per primo: può chiudere tutto in un pomeriggio |
| **S1** | allineamento a barre chiuse e barre in formazione | **fatto** (`trading/mtf.py` + `trading/live_frames.py`, 11 test) |
| **S2** | adattatore da cambi di posizione a stato per barra, con memoria e decadimento | **fatto** (`trading/voters.py`, 10 test) |
| **S3** | punteggio a peso uguale, soglia fissa, ampiezza minima, un asset | contro i tre riferimenti |
| **S4** | soglia dinamica dai piani alti | +2 parametri |
| **S5** | innesco 15m, stop ATR 15m, volatilità obiettivo | +2 parametri |
| **S6** | il votante AI, con astensione | misurato con e senza, più il controllo casuale |
| **S7** | pesi online (regola, non ricerca) | solo se S3-S5 hanno guadagnato |

## L'aspettativa, dichiarata prima di misurare

Sulla base di tutto ciò che questo progetto ha già misurato, mi aspetto che la confluenza arrivi
**allo stesso ordine di rendimento del possesso passivo con un drawdown molto minore** — lo stesso
posto in cui è arrivata la rotazione trasversale — e non a un rendimento superiore. Il rischio più
probabile non è che perda: è che **operi troppo poco perché si possa dire se ha funzionato**.

Dichiararlo adesso serve a una cosa sola: se il risultato sarà molto migliore di così, la prima
ipotesi da verificare non sarà il successo, sarà il look-ahead.
