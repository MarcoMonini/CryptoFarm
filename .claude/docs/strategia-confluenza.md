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
nessun ciclo Python. Misurato: **103 ms per cinque anni e tre intervalli** (in `trading/live_frames.py`,
cancellato il 2026-08-30 — vedi in fondo: la misura resta, il modulo no).

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
backtest in una macchina che conosce il futuro. `tests/test_live_frames.py` lo intercettava —
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
| **S0** | correlazione fra i sei stati barra-per-barra | **misurato** (2026-08-28): media +0,156, massima 0,476 — i votanti sono vari |
| **S1** | allineamento a barre chiuse e barre in formazione | **scritto** (`trading/mtf.py`, 5 test; la parte in formazione stava in `trading/live_frames.py`, cancellato il 2026-08-30 perche' nessun votante ne aveva bisogno) |
| **S2** | adattatore da cambi di posizione a stato per barra, con memoria e decadimento | **scritto** (`trading/voters.py`, 10 test) |
| **S3** | punteggio a peso uguale, soglia fissa, ampiezza minima, un asset | **scritto e misurato** (`trading/confluence.py`) — non batte il passivo |
| **S4** | soglia dinamica dai piani alti | **scritto e misurato**: `theta_macro` — **toglie valore**, meglio a 0 |
| **S5** | innesco 15m, stop ATR 15m | **scritto**: `innesco`, `atr_multiplier`. La volatilità obiettivo **no** |
| **S5b** | paniere a capitale condiviso | **scritto** (`trading/portfolio.py`, 8 test) — non era nel piano, l'ha chiesto l'utente |
| **S6** | il votante AI, con astensione | lo slot c'è (`Votante`), il votante no |
| **S7** | pesi online (regola, non ricerca) | `_pesi` accetta pesi non uguali e li tiene sotto `w_max`; nessuna regola online |

Gli stadi erano scritti e non misurati. Dal 2026-08-28 lo sono, su quindici asset e sette anni:
vedi «La misura su dati veri» in fondo. Il codice fa quello che dice di fare — e ciò che dice di
fare non basta.

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

Dal 2026-08-27 è anche **girabile dalla pagina** («React inside forming higher-plane bars»), non
solo dalla griglia del banco. Finché era una costante di `config` senza widget, non entrava nel
dizionario che la barra laterale costruisce, e `panels.diagnosi_confluenza` — che quel dizionario
lo riceve così com'è — cadeva con `KeyError` esattamente nel caso per cui esiste, quello senza
operazioni. Ora `panels.confluenza_di` riempie da sé i buchi di ciò che riceve, così la stessa
trappola non si ripresenta col prossimo parametro senza widget.

Il sollevamento resta utile dove conta il **valore** e non il lato — una distanza, una banda, uno
stop — ma nessun votante lo chiede, quindi `provisional_ema` e' andata via con il modulo.

**Conseguenza dichiarata, e poi eseguita: `live_frames.py` non lo importava nessuno.** Era vivo
solo nei suoi sei test, e la regola scritta qui era che andasse cancellato invece che tenuto per
affezione se nessuna delle sue due ragioni avesse retto. **Cancellato il 2026-08-30.** Le due
ragioni restano vere e sono il motivo per cui vale saperlo esistito: è il pezzo che serve appena un
votante debba leggere il *valore* di una barra lunga parziale, e conteneva il test contro il difetto
da tre caratteri (`transform("max")` invece di `cummax`, che trasforma il backtest in una macchina
che conosce il futuro). `git log --diff-filter=D --name-only` lo ritrova con i suoi test.

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

## I tre difetti di valutazione, misurati e corretti

Provandola nel simulatore sono emersi tre problemi distinti. Tutti e tre misurati prima di
toccarli, perché la differenza fra «mi sembra» e «è» è l'unica cosa che rende una correzione
verificabile.

### 1. La soglia saltava, e decideva lei

I due piani lunghi entravano come `np.sign(prezzo − media)`, cioè −1, 0 o +1. Quindi
`accordo_alto` prendeva cinque valori e la soglia **saltava di 0,15 per volta** — il 16%
dell'escursione totale del punteggio, in un istante. Misurato: **una uscita per punteggio su
quattro cadeva sulla barra esatta in cui la soglia era saltata.**

La distanza dalla media ora si normalizza sull'ATR **dello stesso piano** e si schiaccia con una
tangente iperbolica. Il cancello resta lo stesso confronto (`> 0` è ancora «prezzo sopra la
media»); la soglia si muove con continuità.

| | prima | dopo |
|---|---|---|
| salto tipico della soglia | 0,15 | **0,0023** |
| valori distinti | 5 | 2.271 |
| uscite causate dal salto | 24,6% | **0** |

L'ATR di normalizzazione ha finestra fissa a 14 e **non è un parametro libero**: è l'unità di
misura, e la tangente iperbolica rende il risultato insensibile al suo valore esatto.

### 2. L'isteresi sbagliava in tutte e due le direzioni

Verso il basso: si apriva e si chiudeva in due barre da quindici minuti. Verso l'alto: il punteggio
decade piano, quindi restava appena sopra `soglia − isteresi` per ore — mediana 14 barre oltre il
primo calo sotto la soglia, coda a 84, cioè ventun'ore.

Due limiti nuovi, e valgono **solo per l'uscita dal punteggio**: `barre_minime` è il pavimento,
`pazienza` il soffitto sulle barre consecutive sotto la soglia semplice.

| | prima | dopo |
|---|---|---|
| operazioni chiuse entro 2 barre | 3 | **0** |
| ritardo oltre il primo segnale, 90° percentile | 35 | 23 barre |
| ritardo massimo | 84 | 52 barre |
| operazioni totali | 211 | 213 |

L'ultima riga è la più importante: **non si è soppressa attività**, si è tolta quella sbagliata.

**Lo stop e il cancello non sono soggetti al pavimento.** Sono regole di rischio, non di opinione:
un pavimento che tiene aperta una posizione mentre lo stop è saltato non è pazienza, è un difetto
travestito da parametro.

### 3. I pesi non sommavano a uno con pochi votanti

Trovato dai test mentre scrivevo la selezione parziale: con tre votanti un tetto di 0,30 li cappava
tutti e la somma faceva **0,90**. Il punteggio restava sistematicamente sotto la soglia e nessuno
lo diceva. Con i sei di default non si vedeva, perché 0,167 sta sotto 0,30. Il tetto ora non può
scendere sotto `1/n`, che è il limite oltre il quale il vincolo è insoddisfacibile.

## I votanti sono moduli

```python
confluence.registra(confluence.Votante("nome", "famiglia", "conferma", funzione, parametri))
confluence.evaluate(candele, "15m", votanti=confluence.selezione("ichimoku", "flusso"))
```

Registrare basta: da lì in poi si adattano da soli il conteggio delle famiglie, i pesi, la
necessarietà, i riquadri della barra laterale, i parametri della strategia e la griglia del banco.
**Non c'è nessun elenco da tenere allineato a mano** — è l'unica forma di modularità che conta,
quella in cui dimenticarsi un posto non è possibile. Un test lo verifica registrando un votante
finto e controllando che la pagina ne mostri il riquadro.

### Le 31 manopole, e i tre strati da cui prendono il valore

I parametri dei votanti erano congelati **e invisibili**: trentuno manopole sugli indicatori dei
piani lunghi che non si potevano nemmeno leggere. Ora ognuna ha la sua costante, la sua etichetta e
il suo widget, in un riquadro per votante. Il valore si risolve in tre strati:

1. il default della funzione in `strategies_ls`, scritto in `config` come `CONF_*`;
2. il valore **misurato** in `tuned_defaults` per l'intervallo del **piano** su cui il votante gira;
3. l'override esplicito, cioè i widget e la griglia.

Il secondo strato è quello che si sbagliava facilmente: a base 15m un votante di struttura gira a
4h e prende i `tuned_defaults` di 4h, non quelli di 15m. Prenderli dalla pagina sarebbe stato
sbagliato in silenzio. Dove il piano cade fuori dai quattro intervalli misurati — a base 1h la
struttura è 16h — non si sostituisce niente e restano i default scritti a mano.

### Il costo, dichiarato

Il congelamento era **il vincolo portante** che teneva a nove i parametri liberi. Scioglierlo li
porta a oltre quaranta. È una scelta esplicita di chi usa la pagina, non un miglioramento: il
conteggio delle prove per la correzione di molteplicità deve ora includere anche questi, e con
quaranta gradi di libertà su cinque anni di dati la soglia del DSR sale al punto in cui quasi
niente resta distinguibile dalla fortuna. **La raccomandazione resta muovere i votanti per capire,
e misurare con i votanti fermi.**

## Ottimizzare i default: cosa si può dire adesso e cosa no

`--grid votanti` scandisce le 31 manopole una per volta attorno al proprio valore di partenza, con
la griglia ricavata dal registro (cinque multipli del default, ritagliati sui limiti di `config`).
Sono 116 celle.

**Ma i valori "ottimi" non li ha ancora scelti nessuna misura**, e non li si può inventare: i
default di oggi sono quelli delle funzioni più i `tuned_defaults` dove esistono, cioè scelte
ragionevoli e non ottimizzate. Ottimizzarli richiede lo store delle candele. La sequenza è:

```bash
python -m scripts.confluence_lab --grid votanti     --symbol BTCUSDT --interval 15m --since 2021-01-01
python -m scripts.confluence_lab --grid coordinate  --symbol BTCUSDT --interval 15m --since 2021-01-01
python -m scripts.confluence_lab --grid ampia       --interval 15m --since 2021-01-01
```

E la regola con cui leggerle è già scritta in questo progetto, non va reinventata: si adotta un
valore **solo se** sposta la mediana dei ranghi di almeno 0,06 e sceglie lo stesso valore anche
guardando il solo primo sottoperiodo. È il criterio di `scripts/tune_defaults.py`, e nasce dalla
misura che qui vale più di ogni altra: sulla rotazione la correlazione fra resa in stima e resa in
verifica è **−0,69**, cioè prendere il massimo della griglia trasferisce peggio che scegliere a
caso.

## La misura su dati veri, finalmente (2026-08-28)

Fino a qui questo documento diceva «il codice c'è e gira; la misura su dati veri no». Adesso c'è:
`scripts/confluence_audit.py`, **90 configurazioni per coordinata × 15 asset × tre intervalli ×
tre finestre**, più una griglia cartesiana da 4.800 celle e un Monte Carlo per permutazione delle
barre. I dati sono lo store locale, 2019-2026, dai 610.000 ai 950.000 minuti di storia per asset.

### Il risultato, in una riga

**Non batte il possesso passivo, e il modo in cui fallisce è informativo.** Su 1.350 celle a
quindici minuti (90 configurazioni × 15 asset, finestra intera) ne battono il passivo il **6,4%**,
e la mediana rende **−70,3%** dove la mediana del possesso passivo rende **+348,3%**.

### Le tre cose che il banco ha escluso, e non sono poche

Prima di leggere il resto: il fallimento **non** è uno dei tre dichiarati in «I tre modi in cui
questo fallisce».

1. **Non c'è look-ahead.** Verificato su dati veri, non solo sintetici: riscrivendo ×1,7 il futuro
   di BTC a partire da un istante *dentro* una barra giornaliera già cominciata, i 70 eventi
   precedenti restano identici byte per byte, e il troncamento puro dà gli stessi 70. `mtf.py` fa
   il suo lavoro.
2. **I votanti non sono correlati** (S0, che era «resta da misurare»). Correlazione media fra i sei
   stati **+0,156**, massima 0,476, nessuna coppia sopra 0,5; servono **cinque** componenti
   principali su sei per il 90% della varianza. Il timore era 0,8: l'insieme è genuinamente vario.
3. **Non opera troppo poco.** Era il rischio ritenuto più probabile. Opera *troppo*: 116 operazioni
   l'anno di mediana a quindici minuti.

Quindi il disegno ha fatto ciò che prometteva. Perde lo stesso.

### Perché perde: opera, e non dovrebbe

La correlazione fra numero di operazioni e rango della configurazione è **−0,60** su tutte le celle
(p ≈ 1e-130), mediana **−0,72** dentro il simbolo, negativa in 14 asset su 15. La scala per fascia
di frequenza non ha eccezioni:

| operazioni/anno | mediana del rendimento |
|---|---|
| < 10 | −1,1% |
| 10-25 | −14,5% |
| 25-50 | −42,7% |
| 50-100 | −65,0% |
| 100-200 | −75,1% |
| > 200 | −84,7% |

**Ogni parametro migliora nella direzione che fa operare di meno**, senza una sola eccezione:
`theta_base` da 0,05 (rango 0,03) a 0,60 (0,97); `emivita` da 48 (0,06) a 0,5 (0,94);
`atr_multiplier` da 1,0 (0,06) a 8,0 (0,88); `k_famiglie` da 1 (0,42) a 6 (0,97), dove 6 significa
**zero operazioni**. Il gradiente della griglia punta al non-fare, ed è la forma che prende una
strategia il cui vantaggio lordo è minore del proprio costo di transazione.

Che sia il costo e non il tempismo si vede togliendo il costo. Al lordo di **tutto** — nessuna
commissione, nessun mantenimento — contro l'esposizione casuale di pari durata:

| | lordo | esposizione | caso, stessa esposizione | passivo |
|---|---|---|---|---|
| BTC | +25,3% | 7,4% | +24,3% | +1.759,6% |
| ETH | +5,5% | 7,4% | +22,9% | +1.495,7% |
| SOL | −48,2% | 6,5% | +23,5% | +2.487,3% |
| XRP | −13,6% | 5,1% | +5,9% | +207,2% |
| BNB | −62,7% | 6,7% | +36,3% | +10.035,6% |

Su quattro asset su cinque il tempismo è **peggiore** dell'esposizione casuale di pari durata, e
sul quinto è indistinguibile. Le commissioni non rovinano un vantaggio: non c'è un vantaggio da
rovinare. Raddoppiare la commissione da perpetui (0,05%) a spot (0,10%) porta BTC da −59,7% a
−87,0%, che è la firma di una strategia il cui risultato è dominato dal numero di operazioni.

### Il mantenimento non c'entra, contro l'ipotesi ovvia

`confluence_lab` addebita 0,03%/giorno a una strategia che di default è solo lunga, cioè ~11,6%
l'anno di costo. Sembra il colpevole e non lo è: toglierlo del tutto sposta BTC da −62,1% a
−59,7%, perché **l'esposizione è il 7,4%** del calendario. È comunque una stortura del modello —
a pronti una posizione lunga non paga mantenimento — ma non spiega niente.

### Batte il mercato solo dove il mercato scende

La correlazione fra la resa passiva di un asset e la frazione di configurazioni che la battono è
**−0,73**, identica a 15m e a 1h (p = 0,002, n = 15).

- asset con possesso passivo sopra +100% (n = 9): battuti nell'**1,6%** delle configurazioni;
- asset con possesso passivo sotto +50% (n = 5): **44,9%**.

Su BNB, SOL, LINK, BTC, ETH, TRX, ADA, LTC — ognuno con passivo sopra +50% — **nessuna** delle 90
configurazioni batte il possesso passivo. Le uniche vittorie larghe sono DOT (passivo −74,9%,
battuto dal 98,9% delle celle) e ATOM (−65,8%, dall'85,6%). Non è selezione: è **stare fuori**, e
stare fuori paga esattamente quando il mercato scende.

### Il trasferimento fuori campione è positivo, e non significa quello che sembra

Spearman fra rango in stima (2019-2024) e rango in verifica (2024-2026): **+0,65** a 15m. Sembra
la smentita del −0,69 misurato sulla rotazione, e non lo è. La correlazione fra operazioni all'anno
e rango **fuori campione** è **−0,843**: la classifica ordina il *non-fare*, che è una proprietà dei
parametri e non del mercato, quindi trasferisce per costruzione. Fra le prime dieci fuori campione,
quattro fanno meno di due operazioni l'anno e una ne fa **zero**.

Le due finestre non sono nemmeno lo stesso mercato: in stima il possesso passivo rende **+613,9%**
di mediana con **0 asset su 15** in perdita; in verifica rende **−33,8%** con **11 su 15** in
perdita. «Battere il passivo nel 47% delle celle» in verifica vuol dire che stare in contanti ha
battuto le criptovalute durante una discesa. Non è un'abilità.

### I due parametri morti

- **`w_max` non fa niente, mai.** `_pesi` fa `w_max = max(w_max, 1/len(nomi))`: a pesi uniformi il
  tetto non può mordere. Tutti e sette i valori della scansione danno peso 0,1667. Empiricamente lo
  spread di rango è **0,000** e i rendimenti sono identici asset per asset. La scansione spreca
  sette celle su un parametro che è un no-op.
- **`k_famiglie = 2`, il default, è identico a `k_famiglie = 1`** su tutti e 15 gli asset,
  rendimenti e numero di operazioni compresi. Con sei pesi uniformi da 0,167 un punteggio sopra una
  soglia di 0,20+ implica già almeno due votanti concordi: il cancello di ampiezza non vincola mai.
  Morde da 3 in su.

### Due meccanismi del disegno che tolgono valore

- **La soglia adattiva (`theta_macro`) fa danno.** A 15m spegnerla (0,0) è la scelta migliore della
  coordinata: rango 0,867 contro 0,417 del centro, rendimento mediano −33,3% contro −77,1%, e il
  degrado è monotòno fino a 0,40 (rango 0,156). È il meccanismo dello stadio S4.
- **Reagire dentro la barra lunga fa danno.** `barre_in_formazione=False` batte `True` a tutti e due
  gli intervalli: 0,589 contro 0,417 a 15m (−69,8% contro −77,1%), 0,667 contro 0,583 a 1h (+1,6%
  contro −5,3%). È l'ablazione che questo documento chiedeva, e la risposta è che **aspettare la
  chiusura del piano lungo è meglio**.

### Due votanti su sei fanno il lavoro

Necessarietà mediana sulla configurazione centrale, BTC 2019-2026: `flusso` **0,741**, `pullback`
**0,683**, `ichimoku` 0,375, `donchian` 0,224, `squeeze` 0,075, `reversione` 0,057. Su tutti e 15
gli asset la necessarietà massima mediana sta fra **0,74 e 0,80**, cioè sopra la soglia di 0,60 che
questo documento fissa come «l'insieme è quel votante travestito», nell'**88,5%** delle celle a 15m.

Ma attenzione a come si legge, perché la misura di S0 dice che l'insieme *non* è un votante
travestito (correlazione +0,156). Le due cose convivono: con sei pesi uniformi da 0,167 e una
soglia di 0,35 servono due o tre votanti attivi per entrare, quindi **quasi ogni votante attivo è
necessario per costruzione**. `necessarieta` confonde «votante dominante» con «margine sottile
sopra la soglia», e va letta insieme alla correlazione, non da sola.

### La scala impedisce di andare dove il gradiente punta

I tre intervalli che `scala_fuori_misura` ammette, stesse 90 configurazioni e stessi 15 asset:

| intervallo | batte il passivo | in utile | mediana rendimento | operazioni/anno | drawdown mediano |
|---|---|---|---|---|---|
| 15m | 6,4% | 7,4% | −70,3% | 116,0 | 78,4% |
| 30m | 9,0% | 36,6% | −22,0% | 76,5 | 66,2% |
| 1h | 15,9% | 41,9% | −8,6% | 31,2 | 55,2% |

Monotòno in ogni colonna, e coerente con tutto il resto: allungare la barra riduce le operazioni e
migliora tutto. Il gradiente punta **oltre** l'ora — ma il fattore ×96 del piano di regime porta la
media di regime a 4 giorni a 1h e a 16 giorni a 4h, cioè fuori dalla regola che questo documento si
è dato. **Il vincolo di scala del disegno taglia via proprio la regione che le misure indicano**, ed
è il difetto strutturale più importante fra quelli trovati: i quattro piani sono agganciati a una
progressione geometrica fissa (×1/×4/×16/×96) invece che a intervalli scelti, e quella progressione
è ciò che vieta una base a 4h.

### Cosa farne

In ordine di rapporto fra guadagno atteso e lavoro:

1. **Sganciare i piani dalla progressione fissa.** `FATTORI` diventa una mappa da intervallo a
   quattro intervalli scelti, così una base a 4h può avere un regime a 1d invece che a 16d. È la
   sola modifica che apre la regione dove le misure puntano.
2. **Spegnere `theta_macro` di default** (S4 non guadagna) e **mettere `barre_in_formazione=False`**
   come valore di partenza. Sono due default, non due meccanismi da cancellare.
3. **Togliere `w_max` dalla scansione** finché i pesi sono uniformi, e portare il default di
   `k_famiglie` a 3, che è dove comincia a vincolare.
4. **Non cercare il massimo di griglia.** La cella migliore in assoluto è DOGE a 1h con +10.864%
   contro un passivo di +1.752%: una cella su 1.350, su un asset, ed è esattamente la forma che ha
   la fortuna. Vale come promemoria, non come configurazione.

E la cosa da non fare: tarare i sei votanti sperando di trovare la regione buona. Al lordo delle
commissioni il tempismo è peggiore dell'esposizione casuale su quattro asset su cinque; non c'è una
taratura che trasformi quel numero in un vantaggio.

## Il seguito: Monte Carlo, slot concorrenti, votante a modello (2026-08-28)

### Monte Carlo per permutazione — il risultato piu' netto

200 permutazioni per asset, 3.000 simulazioni. Ogni barra conserva la propria geometria e viene
riattaccata alla chiusura della precedente nel nuovo ordine: restano identiche la distribuzione dei
rendimenti di barra e **la deriva dell'asset**, sparisce solo la correlazione seriale.

**Valore-p mediano 0,488.** Su 7 asset su 15 il risultato vero e' *peggiore* della mediana delle
permutazioni; 3 asset stanno sotto 0,05 e 4 sopra 0,95, che e' la forma del caso su quindici prove.
Distruggere ogni struttura temporale non peggiora la strategia in modo misurabile: e' la
confutazione diretta della premessa del disegno.

### Gli slot concorrenti sono de-leva, non diversificazione

`portfolio.simulate_shared_capital` teneva **una** posizione e scartava i segnali che arrivavano a
capitale impegnato: il 36% su cinque asset, il **60%** su quindici -- molto piu' di quanto suggerisca
un conto a eventi indipendenti, perche' in cripto i segnali arrivano insieme. `simulate_slots` divide
il capitale in `n_slot` quote e li recupera tutti.

Il rendimento migliora in modo monotono in tutte le sei combinazioni provate (a 0,40/4,0: da −23,8%
con 3 slot a −6,3% con 12). **Ma lo Sharpe non si muove** -- −0,52 con 3 slot e −0,52 con 12, mentre
il drawdown scende dal 31,1% all'8,7% -- e a 0,35/6,0 *peggiora*, da −1,21 a −1,33. Lo Sharpe e'
invariante di scala: se resta fermo mentre il rendimento sale, quello che e' cambiato e' la
**dimensione della posizione**, non la bonta' della scelta. Con correlazione fra asset attorno a 0,8
le posizioni contemporanee non sono scommesse indipendenti, quindi non c'e' errore da mediare.

Gli slot sono una leva di rischio genuina, e il loro valore e' **condizionato a uno Sharpe positivo**.
Su 48 configurazioni (3 theta × 2 emivita × 2 k_famiglie × 4 n_slot) **nessuna** e' in utile, contro un
possesso passivo mediano di +348,3%; e stringere per compensare non funziona come sembra -- la stretta
e' moltiplicativa fra sei votanti, e nel primo tentativo ha portato gli ingressi da 113,7 a **0,3 per
asset all'anno**, cioe' 35 operazioni in tutto. Non e' operare meno: e' smettere.

### Il votante a modello: vantaggio reale, stabile, insufficiente

`scripts/ai_voter.py` addestra un GBDT sulle **operazioni della confluenza stessa** -- non su quelle di
un'altra primaria, perche' quattro uscite su cinque le chiude lo stop a trailing e un'etichetta presa
altrove risponderebbe su un'operazione che non verra' presa. 21.919 operazioni su 15 asset, etichetta
«chiude sopra i costi», le 16 feature di `meta_gate` comprese le tre trasversali.

| taglio | AUC | verifica | netto base | netto a p ≥ 0,45 | percentile |
|---|---|---|---|---|---|
| 2021-07 | 0,536 | 15.125 | −0,099% | −0,002% | 98,4 |
| 2022-01 | 0,540 | 13.322 | −0,109% | +0,005% | 99,2 |
| 2023-01 | 0,536 | 11.309 | −0,092% | +0,036% | 94,0 |
| 2024-01 | 0,536 | 7.789 | −0,100% | +0,177% | 99,0 |

**Il vantaggio di ordinamento e' reale e stabile**: AUC 0,536-0,540 su quattro tagli indipendenti, ogni
volta fra il 94o e il 99o percentile di 500 selezioni casuali di pari numerosita'. E' il soffitto del
campo (§1.1), non una delusione. **E toglie il segno meno, poi si ferma**: il netto per operazione va da
≈−0,10% a ≈0,00% su tre tagli su quattro. E' parola per parola la conclusione gia' registrata in §3.4
per `trend_pullback`, adesso su un campione sette volte piu' grande e stabile su quattro finestre.

Una trappola da non ripetere: la CV purgata **dentro** la sola finestra 2024-2026 da' AUC 0,495, cioe'
caso. Il purging toglie la sovrapposizione fra operazioni vicine, non il fatto che in una CV il regime
successivo sia gia' nel campione di addestramento. Serve il taglio temporale vero.

Fine a fine sul 2024-2026 (modello seleziona, votanti confermano, slot allocano): da −51,0% senza filtro
a **+2,0%** a soglia 0,50 con 12 slot, Sharpe +0,23 e drawdown 5,2%, contro un passivo di −33,8%. Va
letto per quello che e': in una finestra in cui il passivo perde un terzo, battere il passivo di 36 punti
vuol dire **stare in contanti**, e contro i contanti sono +2,0% in due anni e mezzo. In piu' la soglia
0,45 rende −1,5% mentre la 0,40 e la 0,50 rendono +0,8% e +2,0%: fra soglie adiacenti quel salto e'
rumore. **L'architettura si compone e smette di perdere; un vantaggio non l'ha dimostrato.**

### Il collegio nuovo, misurato e non adottato

Su richiesta dell'utente Donchian e squeeze escono, entrano `atr_band_bounce` (le bande ATR senza il
cancello di range, con uscita alla banda opposta) e `trend_zone` (la macrostruttura come stato). Ognuna
registrata **due volte su piani diversi**: e' la prima volta che `famiglia` codifica qualcosa, perche'
con sei votanti in sei famiglie `k_famiglie=2` era dimostrabilmente identico a `k_famiglie=1`. Adesso
tutti e quattro i piani hanno un votante -- il regime non ne aveva nessuno.

A parametri identici pero' **rende meno**: mediana −79,2% contro −70,3%, con il 25% di operazioni in piu'
(1.077 contro 864). La necessarieta' massima scende da 0,766 a 0,627, cioe' l'insieme e' meno dominato --
ma `flusso` resta il votante piu' necessario su tutti e 15 gli asset. Il confronto giusto tiene fermo il
**numero di operazioni**, non `theta_base`: con sette votanti il peso di ciascuno passa da 1/6 a 1/7 e la
stessa soglia non vuol dire la stessa cosa. Quel confronto non e' stato fatto, e finche' non lo e' il
collegio nuovo non e' ne' meglio ne' peggio: e' diverso.

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
