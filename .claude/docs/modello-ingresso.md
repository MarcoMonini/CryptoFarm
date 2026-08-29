# Il modello d'ingresso: la domanda cambiata, e i primi numeri che passano il controllo

**Stato: cablato (2026-08-29).** È il modello in testa a `MODEL_PRECEDENCE`, e i due artefatti
sono `entry_model_veloce` (opera) e `entry_model` (fa da cancello).

## 1. Da dove viene: una premessa dell'utente, verificata

Il modello precedente (`swing_model`, `.claude/docs/modello-swing.md`) prevedeva la **prossimità
agli estremi locali**. L'utente ha contestato la misura che ne usciva — +0,026% per barra segnalata
su venti barre — con un argomento che non era di gusto ma di ordine di grandezza: *«i movimenti
delle cripto anche su mercato laterale sono maggiori dell'1%, quindi questa misura non regge:
significa che i massimi e i minimi locali non vengono individuati davvero»*.

La verifica gli ha dato ragione sul fatto e torto sulla causa, e le due cose vanno tenute separate:

- **il tetto c'era.** Le barre che l'*etichetta* segna come decile più basso rendono **+0,765%** su
  venti barre, con l'87,1% in salita. Non è un problema di ampiezza dei movimenti;
- **il modello ne prendeva un terzo.** Delle barre segnalate, il 30,3% erano minimi veri (il caso
  ne dà il 10%). Sull'intersezione fra segnalate e vere il rendimento è +0,786%, cioè **quanto
  l'oracolo**: dove il modello ha ragione, ha ragione del tutto;
- **il resto trascinava.** I falsi positivi rendono −0,30%, e l'aritmetica torna:
  `0,303 × 0,786 + 0,697 × (−0,305) = +0,026%`.

Quindi il segnale c'era e il margine no. La domanda che ne segue non è «come lo rendo più
preciso», ed è qui che la direzione è stata corretta.

## 2. Precisione e denaro non sono la stessa domanda

A pari selezione (10% delle barre, verifica dal 2024, 15 simboli):

| bersaglio | rendimento del segnalato | è davvero un minimo |
|---|---|---|
| etichetta a gambe (`swing_leg_target`) | +0,025% | **37,2%** |
| entro 10 barre da un minimo | +0,012% | 28,4% |
| minimo già avvenuto entro 10 barre | −0,012% | 22,7% |
| **rendimento futuro diretto** | **+0,059%** | 23,0% |
| rendimento futuro / ATR | +0,012% | 12,8% |

L'etichetta a gambe **vince sulla precisione e perde di 2,4 volte sul denaro**. Chiedere «è un
minimo» e chiedere «rende» sono due domande diverse, e la seconda è quella che si incassa. Il
bersaglio del modello nuovo è il rendimento logaritmico delle prossime `H` barre, e basta.

Tre strade che sembravano ovvie sono state provate e non portavano niente:

- **feature nuove**: quattro famiglie (rifiuto delle ombre, esaurimento della gamba, capitolazione
  a volume, divergenza prezzo/oscillatore) spostano la precisione da 30,0% a 31,3%. Le 16 colonne
  aggiuntive *peggiorano* il netto (+1,046% contro +1,188%), quindi `bar_features.py` **non è
  stato toccato**: si servono le stesse 41 colonne di `swing_model`;
- **più dati**: 4,4 milioni di righe invece di 366 mila danno 30,6%;
- **più capacità**: più iterazioni peggiorano, 29,0%.

Il modello a swing era, misurandolo, **un oscillatore**: la sua previsione correlava +0,932 con
`dist_ema50_atr`, e togliendo quella colonna restava +0,87 perché le altre quaranta sono della
stessa famiglia.

## 3. La leva è la selettività, non l'accuratezza

La commissione è fissa e il rendimento no. Col bersaglio diretto, su 150 barre:

| barre segnalate | rendimento medio del segnalato |
|---|---|
| 10% | +0,047% (sotto la commissione) |
| 2% | +0,90% |
| 0,5% | **+2,07%** |

Il modello non diventa più bravo: si opera solo dove dice molto. Da qui le tre scelte di
`ml/entry_trainer.py`: **soglia dal quantile dello stima** (prenderlo dal fuori campione sarebbe
look-ahead), **tenuta fissa** invece di un'uscita a segnale, **nessuna sovrapposizione** — mentre
si è dentro, i segnali successivi si ignorano, o si misura un capitale che non si ha.

## 4. Il controllo, che qui è obbligatorio

Fuori campione il possesso passivo mediano fa **−34%**: una strategia dentro il mercato il 17% del
tempo lo batte quasi da sola. «Batte il passivo» non è quindi un risultato. Il confronto è con
**ingressi a caso a pari numero e pari tenuta**, 200-400 estrazioni, con lo stesso filtro
anti-sovrapposizione.

| modello | operazioni | netto medio | in utile | caso | percentile | simboli |
|---|---|---|---|---|---|---|
| `entry_model` (H=150, tenuta 150) | 427 | **+1,529%** | 59,5% | +0,004% | 100° | 14/15 |
| `entry_model_veloce` (H=20, tenuta 20) | 223 | **+1,360%** | 63,2% | −0,173% | 100° | 12/15 |

È il primo risultato di questo progetto che passa il controllo a esposizione appaiata. Le famiglie
precedenti non lo passavano: `swing_model` 1 simbolo su 15, `leg_model` netto negativo a tutte le
soglie, la politica RL solo debolmente (rango 0,588).

## 5. I due modelli si compongono in un verso solo

La richiesta era due modelli complementari, uno stretto (10-20 barre, microstruttura) e uno largo
(~150 barre, macro movimenti). La forma della composizione è quella indicata dall'utente — **il
veloce fa le operazioni, il lento dice dentro quali movimenti può farle** — e la misura la
conferma. Verifica dal 2024, tenuta 20 barre, commissione 0,2%, 200 estrazioni per il controllo:

| cancello del lento | operazioni | netto medio | in utile | simboli | caso | percentile |
|---|---|---|---|---|---|---|
| nessuno | 223 | +1,360% | 63,2% | 12/15 | −0,173% | 100° |
| mediana dello stima | 161 | +1,806% | 65,2% | 12/15 | −0,143% | 100° |
| 80° dello stima | 156 | +2,019% | 65,4% | 13/15 | −0,161% | 100° |
| **90° dello stima** | **148** | **+2,071%** | **65,5%** | **14/15** | −0,165% | 100° |
| 95° dello stima | 128 | +2,243% | 65,6% | 13/15 | −0,170% | 100° |
| 98° dello stima | 100 | +2,464% | 68,0% | 13/15 | −0,172% | 100° |

**Perché il 90° e non il 98°.** La curva è monotona: prendere il valore più alto significa scegliere
il massimo del campione di verifica, che è l'errore già misurato altrove in questo progetto
(correlazione −0,69 fra resa in stima e in verifica sulla rotazione). Il 90° è scelto sulla
**concordanza fra simboli**, 14 su 15, che è la differenza fra un modello e un episodio.

L'inverso — il lento che opera dentro le indicazioni del veloce — non è stato provato e non ha
senso operativo: un'operazione da 150 barre non sta dentro una da 20.

`scripts/entry_lab.py` rifà questa tabella.

## 6. Cosa è stato cablato

- **`MODEL_PRECEDENCE`**: `entry_model_veloce`, poi `entry_model`, poi le famiglie precedenti. Il
  veloce passa davanti perché è quello che genera i segnali; il lento resta nella catena perché da
  solo è comunque una strategia misurata.
- **`ml/signals.entry_signals`**: ingresso sopra soglia e col cancello aperto, uscita dopo la
  tenuta. Soglia, cancello e tenuta si leggono **dai metadata dell'artefatto** e non dai widget:
  sono il modello, non due manopole. Il `threshold` della barra laterale non entra in questa
  strategia, di proposito.
- **La tenuta è un tempo, non un conteggio di candele.** 150 barre a 5m sono dodici ore e mezza, e
  a 1h restano dodici ore e mezza (13 barre).
- **Il cancello vale solo sulla barra d'ingresso.** Una posizione aperta non si chiude perché il
  piano largo è cambiato: troncarla misurerebbe un'altra strategia.
- **Il votante `modello` della confluenza** vota +1 mentre un'operazione del modello d'ingresso è
  aperta, 0 altrimenti. Resta solo lungo, come prima, e `entra`/`esci` non hanno effetto: la
  selettività sta nei metadata.
- **La nota del riquadro** dice il numero che conta e contro cosa è misurato.

## 7. Cosa resta aperto

- **La griglia della confluenza con e senza il votante `modello`**, sugli stessi asset e la stessa
  finestra. È l'unico modo di dire se il modello aggiunge alla strategia a confluenza o si limita a
  ridurne l'esposizione. Vale anche per la politica RL, e non è ancora stata fatta per nessuna
  delle due.
- **Il controllo a blocchi.** Il controllo casuale campiona righe che si sovrappongono fra loro e
  fra simboli; su `swing_model` un bootstrap a blocchi settimanali aveva spostato il verdetto. Qui
  la tenuta è più corta e le operazioni sono poche, ma la misura va rifatta in quella forma prima
  di dire «significativo».
- **Sotto i 5 minuti non c'è misura.** Il modello è addestrato e misurato a 5m. Sopra la
  mezz'ora il punto è stato misurato e chiuso: vedi §8.

## 8. Due cose misurate dopo il cablaggio

### 8.1 Operare più spesso costa, e si sa quanto

La domanda era esplicita: si vogliono operazioni sugli intervalli brevi, e il modello servito ne fa
poche. Fare di più si può, e ha un prezzo. Qui si muove **solo** la soglia del veloce — presa sul
campione di stima, non sul fuori campione — col cancello del lento fermo al 90°:

| barre marcate | operazioni | netto medio | in utile | simboli in utile |
|---|---|---|---|---|
| 5,0% | 1.570 | +0,166% | 46,9% | 10/15 |
| 2,0% | 677 | +0,523% | 51,8% | 13/15 |
| 1,0% | 330 | +1,068% | 59,7% | 13/15 |
| **0,5%** | **148** | **+2,071%** | **65,5%** | **14/15** |
| 0,1% | 23 | +3,710% | 73,9% | 8/11 |

Tutte al 100° percentile contro ingressi casuali a pari esposizione. Tre letture:

- **la commissione è fissa allo 0,2%, il rendimento no.** Abbassare la soglia non aggiunge
  operazioni allo stesso rendimento: ne aggiunge di peggiori, in modo monotono e ripido;
- **il totale accumulato ha il massimo altrove.** Somma dei rendimenti per operazione: 261 a 5%,
  354 a 2%, 352 a 1%, 307 a 0,5%. Chi vuole più operazioni *può* averle — 330 invece di 148 — e
  l'accumulo non peggiora. Quello che peggiora è la qualità della singola operazione e la
  concordanza fra simboli;
- **0,5% resta la scelta**, e per la stessa ragione del cancello: 14/15 simboli in utile è il
  massimo della colonna, e questo progetto ha già misurato che inseguire il massimo del
  rendimento trasferisce peggio (correlazione −0,69 fra stima e verifica sulla rotazione).

`scripts/entry_lab.py --frequenza` rifà la tabella.

### 8.2 La soglia è un rendimento, e sopra la mezz'ora non vuol più dire la stessa cosa

`entry_signals` scalava la tenuta con l'intervallo ma serviva la **stessa soglia assoluta**
ovunque. La soglia però non è un quantile: è il rendimento previsto sopra il quale si entra, e il
modello prevede il rendimento delle prossime **venti barre da cinque minuti**. Su barre più lunghe
quelle venti barre sono un altro orizzonte, le previsioni crescono, e la soglia smette di
selezionare le stesse barre. Misurato su cinque simboli dal 2024:

| intervallo | 5m | 15m | 30m | 1h | 4h | 1d |
|---|---|---|---|---|---|---|
| barre marcate | 0,063% | 0,270% | 0,722% | 2,98% | 14,1% | 28,1% |

A 1d il modello «selettivo» marcava una barra su quattro: l'opposto di ciò per cui è stato scelto,
e il +2,07% per operazione non descriveva più niente. Da qui `signals.entry_fuori_misura`, che
serve il modello **fino a 30 minuti** e sopra tace dicendo perché — stessa forma di
`confluence.scala_fuori_misura`. Il votante `modello` fuori scala non tace: scende al successivo
in `MODEL_PRECEDENCE`.

Nota di lettura per il grafico: a 5m la soglia marca lo 0,063% delle barre, cioè **una su
milleseicento**. Una finestra da 240 ore non contiene abbastanza barre perché un'operazione sia
probabile, e su BTCUSDT — il simbolo con una sola operazione in tutto il fuori campione — la
previsione massima su 2.880 barre resta sotto la soglia. Zero operazioni sul grafico è il
comportamento atteso, non un guasto.
