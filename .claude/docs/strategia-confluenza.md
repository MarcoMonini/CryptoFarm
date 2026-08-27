# Confluenza — disegno di una strategia multi-timeframe a più segnali

Ipotesi di lavoro del **2026-08-27**, chiesta dall'utente. **Non è ancora misurata su dati veri:
niente qui è un risultato.** Il codice però c'è tutto — `trading/confluence.py`,
`trading/portfolio.py`, la voce nel simulatore e il banco `scripts/confluence_lab.py` — e gira; ciò
che manca è la macchina con lo store delle candele.

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
| **S0** | correlazione fra i sei stati barra-per-barra | `stati_dei_votanti` la produce; **resta da misurare su dati veri** |
| **S1** | allineamento a barre chiuse e barre in formazione | **scritto** (`trading/mtf.py` + `trading/live_frames.py`, 11 test) |
| **S2** | adattatore da cambi di posizione a stato per barra, con memoria e decadimento | **scritto** (`trading/voters.py`, 10 test) |
| **S3** | punteggio a peso uguale, soglia fissa, ampiezza minima, un asset | **scritto** (`trading/confluence.py`, 16 test) — da misurare |
| **S4** | soglia dinamica dai piani alti | **scritto**: `theta_macro` |
| **S5** | innesco 15m, stop ATR 15m | **scritto**: `innesco`, `atr_multiplier`. La volatilità obiettivo **no** |
| **S5b** | paniere a capitale condiviso | **scritto** (`trading/portfolio.py`, 8 test) — non era nel piano, l'ha chiesto l'utente |
| **S6** | il votante AI, con astensione | lo slot c'è (`Votante`), il votante no |
| **S7** | pesi online (regola, non ricerca) | `_pesi` accetta pesi non uguali e li tiene sotto `w_max`; nessuna regola online |

Gli stadi sono scritti, non misurati. La distinzione non è pedanteria: fino a che non girano sui
cinque asset veri, l'unica cosa che si sa è che il codice fa quello che dice di fare.

## L'aspettativa, dichiarata prima di misurare

Sulla base di tutto ciò che questo progetto ha già misurato, mi aspetto che la confluenza arrivi
**allo stesso ordine di rendimento del possesso passivo con un drawdown molto minore** — lo stesso
posto in cui è arrivata la rotazione trasversale — e non a un rendimento superiore. Il rischio più
probabile non è che perda: è che **operi troppo poco perché si possa dire se ha funzionato**.

Dichiararlo adesso serve a una cosa sola: se il risultato sarà molto migliore di così, la prima
ipotesi da verificare non sarà il successo, sarà il look-ahead.


## Cosa è cambiato scrivendolo

Tre cose che il disegno diceva in un modo e il codice ha costretto a dire in un altro. Stanno qui
perché un disegno che non registra dove ha sbagliato smette di essere una fonte di verità.

### 1. Il sollevamento della media a valore provvisorio è algebricamente un non-fare

Il disegno prometteva le barre in formazione anche per il cancello 1D e la struttura 4H, «perché
lì il sollevamento è esatto e in O(1)». Lo è. È anche inutile, e si vede in una riga:

```
provisional_ema = a·prezzo + (1−a)·chiusa
segno(prezzo − provisional_ema) = segno((1−a)·(prezzo − chiusa)) = segno(prezzo − chiusa)
```

Su un confronto di **segno** il sollevamento non sposta niente, per qualunque `a`. Il primo test
scritto contro `barre_in_formazione` è caduto per questo — accendere e spegnere il meccanismo dava
lo stesso identico risultato — ed era un parametro che sembrava collegato a qualcosa e non lo era.

L'ablazione vera è un'altra, e ora è quella attuata: **quale prezzo** si confronta con la media.
Quello di adesso (ciò che vede il bot live a metà giornata) o quello dell'ultima chiusura del
piano lungo (l'attesa fino a ventiquattro ore). Quella differenza è reale e si misura.

Il sollevamento resta utile dove conta il **valore** e non il lato — una distanza, una banda, uno
stop — e `provisional_ema` resta in `live_frames.py` per quando servirà.

**Conseguenza da dichiarare: `live_frames.py` oggi non lo importa nessuno.** È vivo solo nei suoi
sei test. Sopravvive per due ragioni, e se nessuna delle due regge va cancellato invece che tenuto
per affezione: è il pezzo che serve appena un votante debba leggere il *valore* di una barra lunga
parziale, e contiene il test contro il difetto da tre caratteri (`transform("max")` invece di
`cummax`, che trasforma il backtest in una macchina che conosce il futuro).

### 2. I sei votanti non reagiscono dentro il periodo, e questo è il limite dichiarato

Sollevare a valore provvisorio una *strategia* qualunque non è generico: ogni indicatore ricorsivo
va sollevato a mano, e il disegno **congela i votanti**, cioè vieta di riscriverli. I due vincoli
non stanno insieme. La scelta: i votanti decidono alla chiusura della propria barra lunga, e il
loro stato entra nell'indice breve un periodo dopo (`mtf.align_to_lower`).

La reattività intra-periodo resta dove non costa una riscrittura: nel cancello e nella struttura,
che confrontano il **prezzo di adesso** con la media del piano lungo. È metà del confronto, ed è
la metà che si muove.

### 3. L'isteresi non copriva il rientro

Il secondo test caduto: si usciva e si rientrava sulla stessa barra, 35 volte su 125. L'isteresi
frena il punteggio che oscilla attorno alla soglia, ma uno **stop scattato dentro la barra** lascia
il punteggio dov'era — e la condizione di ingresso, un attimo dopo, è di nuovo vera. Chi esce ora
non rientra prima della barra successiva.

## Il paniere a capitale condiviso

Non era nel disegno; l'ha chiesto l'utente ed è la risposta più diretta al rischio dichiarato —
che la confluenza operi troppo poco perché si possa dire se ha funzionato. Si sorvegliano *N*
asset con la stessa strategia, si sta fuori finché nessuno parla, e quando **uno** dà il segnale
ci si mette tutto il capitale.

È una domanda diversa da quella della rotazione trasversale: la rotazione sceglie *quale* asset
tenere e ci sta dentro sempre; questo sceglie *quando*, su un paniere, e sta fuori per default.

Due numeri vanno riportati accanto a ogni risultato, e sono i due modi in cui la cosa può essere
un'illusione:

- **le occasioni perse.** Ogni segnale che arriva mentre il capitale è impegnato viene buttato. Se
  sono molte più delle operazioni fatte, la scarsità del campione non era il problema;
- **la concentrazione.** Se il 90% delle operazioni sta su un asset solo, non si sta sorvegliando
  un paniere: si sta operando su quell'asset con quattro spettatori.

Le pari merito le vince il **margine del punteggio sopra la soglia**, non l'ordine del dizionario.
Su asset che si muovono insieme — e le criptovalute lo fanno — i segnali cadono spesso sulla stessa
barra, e sceglierli per ordine alfabetico sarebbe una decisione arbitraria travestita da dettaglio
di attuazione.

## Come si misura, in pratica

```bash
python -m scripts.confluence_lab --selfcheck                       # senza store, dati finti
python -m scripts.confluence_lab --grid coordinate --symbol BTCUSDT --interval 15m
python -m scripts.confluence_lab --grid ampia --interval 15m --since 2021-01-01
python -m scripts.confluence_lab --grid veloce --paniere majors
```

Tre griglie, e due modi diversi di guardare lo spazio:

| griglia | celle | cosa vede |
|---|---|---|
| `veloce` | 108 | fumo: che giri, e su che ordine di grandezza di operazioni |
| `ampia` | 4.800 | cartesiana su sei parametri: le **interazioni** fra di loro |
| `coordinate` | 79 | l'intero intervallo di **ognuno** degli undici, uno per volta, gli altri al centro |

La cartesiana su tutti e undici sarebbe mezzo milione di celle, cioè giorni di calcolo per una
risposta che nessuno leggerebbe: un test tiene il tetto a cinquemila. La scansione per coordinata è
anche il metodo con cui questo progetto ha già scelto i valori di partenza
(`scripts/tune_defaults.py`), e per la stessa ragione: il massimo di una griglia è la cella più
fortunata, e su questi dati trasferisce fuori campione peggio della mediana.

Ogni file di risultati porta in testa il **numero di celle girate**, che è il conteggio delle prove
per `scripts/multiplicity.py`. Guardare la cella migliore di 4.800 e riportarne lo Sharpe senza
scontarlo non è una misura.

## Il costo per barra, misurato

| | |
|---|---|
| confluenza completa, 11.520 barre | 351 ms |
| la stessa, riusando gli stati dei votanti | **104 ms** |
| barre in formazione, 5 anni × 3 intervalli | 103 ms |

Il 70% del costo sta nei sei votanti, e i votanti sono congelati: il loro stato non dipende da
nessun parametro della griglia. `stati_dei_votanti` lo calcola una volta per (simbolo, intervallo)
e il banco lo riusa su tutte le celle. È la differenza fra una griglia da 4.800 celle che si lancia
e una che si rimanda.

## Il fallimento silenzioso, e come è stato tolto

Provandola nel simulatore con i valori di partenza della pagina — **240 ore** — non succedeva
niente, e niente diceva perché. La causa, misurata:

| storia caricata | cancello aperto | punteggio massimo | operazioni |
|---|---|---|---|
| 240 ore (10 giorni) | **0%** | 0,17 | 0 |
| 52 giorni | 4% | 0,42 | 0 |
| 208 giorni | 76% | 0,47 | 268 |

Il piano di regime a 15m è giornaliero, e la sua media ne chiede cinquanta barre: con dieci giorni
di storia è tutta NaN, `sign(NaN)` diventa 0, e il cancello **non può** aprirsi. Zero operazioni
non era prudenza della strategia: era una condizione impossibile, che si legge identica.

Le condizioni d'ingresso sono quattro in `and`, e ognuna chiede un rimedio diverso. Ora
`Confluenza.perche_non_entra()` dice quale non si è mai avverata, con i numeri:

```
not enough history: the regime plane has 10 bars and its moving average needs 50.
the gate opened but the score never reached the threshold: peak +0.17 against a threshold
    that never fell below 0.20. Lower «Entry threshold».
score and gate agreed, but never with 3 families at once (at most 2).
```

La pagina lo mostra sotto «No trades», e nella barra laterale dichiara in anticipo **quante ore**
servono a quell'intervallo: 1.200 a 15m, 2.400 a 30m, 4.800 a 1h.

## La scala dei piani vale attorno ai quindici minuti, non ovunque

Il menu offre nove intervalli; la scala ×1/×4/×16/×96 è nata su barre da quindici minuti, dove
cade esatta su 15m/1h/4h/1d. Altrove no:

| base | innesco | conferma | struttura | regime | |
|---|---|---|---|---|---|
| 1m | 1m | 4m | 16m | 96m | il «regime» dura un'ora e mezza |
| **15m** | 15m | 1h | 4h | **1d** | la scala del disegno |
| 30m | 30m | 2h | 8h | 2d | |
| 1h | 1h | 4h | 16h | 4d | |
| 4h | 4h | 16h | 64h | 16d | la media di regime chiede decenni |
| 1d | 1d | 4d | 16d | 96d | idem, in peggio |

La regola scritta: il piano di regime deve durare **fra mezza giornata e una settimana**. Fuori da
lì `scala_fuori_misura` restituisce un avviso che la barra laterale mostra. La strategia gira lo
stesso — non è un errore, è una scelta di chi guarda — ma dal menu non si vedeva.

## Il grafico era un pessimo testimone

Provandola, i grafici sembravano contraddire le operazioni: una linea ferma a 1 mentre si comprava
e si vendeva, e nessuna relazione visibile fra i voti e i trade. L'audit del motore, però, è
pulito: **su 288 ingressi, zero violano le quattro condizioni; su 288 uscite, zero sono senza
causa.** L'incoerenza era tutta nella visualizzazione, e le cause erano quattro.

### 1. Il cancello schiacciava il punteggio

`regime` vale ±1, il punteggio sta in ±0,5. Sullo stesso asse il cancello occupa **2,2 volte**
l'ampiezza del punteggio: si vede una linea piatta a 1 e uno sfrigolio indistinguibile vicino allo
zero. La serie che decide era proprio quella illeggibile. Ora i due piani lunghi hanno il loro
riquadro, **Higher planes**, e *Confluence* tiene solo punteggio e soglia, sulla loro scala.

### 2. Quattro uscite su cinque sono lo stop, che non era disegnato

| chi chiude | quante |
|---|---|
| stop a trailing | **231** |
| punteggio sotto soglia − isteresi | 57 |
| cancello che si chiude | 0 |

L'80% delle vendite avveniva mentre il punteggio era tranquillamente sopra la soglia — corretto, e
del tutto inspiegabile dal grafico, perché il livello dello stop non c'era. Ora è una linea
tratteggiata sulle candele, presente solo dove c'è una posizione.

### 3. La spiegazione delle uscite era attivamente fuorviante

`spiega()` mostrava punteggio e votanti anche sulle uscite. Su un'uscita per stop si leggeva
«venduto mentre cinque votanti dicevano di comprare»: vero, e falso come spiegazione — quella
posizione l'ha chiusa il prezzo, non il voto. Ora ogni marcatore dice a quale famiglia appartiene:

```
entry — score +0.35 / threshold 0.20 · 3 families · ichimoku +0.17, flusso +0.15, pullback +0.04
exit  — trailing stop at 144.26
```

### 4. Il piano di struttura non era disegnato affatto

`struttura` è metà di `accordo_alto`, cioè **muove la soglia**, e non compariva da nessuna parte.
Ora sta accanto al cancello in *Higher planes*: è lì che si vedono le condizioni sui timeframe
lunghi, che prima erano solo un'affermazione nella documentazione.

## Cosa il codice **non** fa, dichiarato

- **la volatilità obiettivo** (dimensione della posizione proporzionale al margine e inversa alla
  volatilità realizzata). Il passo 3 di `piano-strategie.md` si innesta lì senza modifiche, ma non
  c'è: a leva 1 e capitale pieno ogni operazione ha la stessa dimensione;
- **il votante AI**. Lo slot è la dataclass `Votante` e nient'altro: nome, famiglia, piano, una
  funzione che restituisce eventi. Serve la versione **per barra** di `meta_gate`, che oggi produce
  probabilità per operazione;
- **i pesi online**. `_pesi` accetta pesi non uguali e li tiene sotto `w_max`, ma nessuno li muove;
- **le famiglie multiple per votante**. Oggi i sei sono uno per famiglia, quindi contare famiglie o
  votanti è lo stesso. La distinzione è scritta perché morderà appena entra un secondo votante di
  prezzo, ed è più facile scriverla ora che accorgersene dopo.
