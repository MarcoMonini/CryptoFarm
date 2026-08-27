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

## Il punteggio e la soglia dinamica

```
punteggio(t) = Σ wᵢ · statoᵢ(t)              wᵢ = 1/6 nella versione base
accordo_alto(t) = (regime_1D(t) + struttura_4H(t)) / 2       ∈ [−1, +1]
soglia(t) = θ_base − θ_macro · accordo_alto(t)
```

La soglia **non è un numero tarato per regime**: sono i piani alti a decidere quanta conferma
serve. Quando 1D e 4H concordano con forza, `accordo_alto ≈ +1`, la soglia scende e si accetta un
ingresso con meno conferme dal basso. Quando si contraddicono la soglia sale e serve quasi
l'unanimità. È la tua idea — «le condizioni di mercato definiscono i pesi di veridicità» — resa in
**due parametri invece che in un classificatore**.

**Ingresso** quando, sulla stessa barra 15m: il cancello 1D è aperto, `punteggio ≥ soglia`, e
l'innesco 15m scatta. L'innesco serve a rendere l'ingresso davvero a quindici minuti: senza, si sta
solo eseguendo una decisione 4H con risoluzione più fine.

**Dimensione** proporzionale al margine sopra la soglia, moltiplicata per il rapporto fra
volatilità obiettivo e volatilità realizzata. È il punto in cui il passo 3 del piano generale
(`piano-strategie.md`) si innesta senza modifiche.

**Uscita**, la prima delle tre che arriva:
1. `punteggio < soglia − isteresi` — l'isteresi non è un dettaglio: senza, si entra e si esce sulla
   stessa barra ogni volta che il punteggio oscilla attorno alla soglia;
2. stop a trailing ATR su 15m, **calcolato su barre chiuse a `i−1`**;
3. il cancello 1D si chiude — si va flat senza discutere.

## Il conteggio onesto dei parametri liberi

È la sezione che decide se questo disegno è misurabile o è un esercizio.

| voce | liberi | come |
|---|---|---|
| parametri dei sei votanti | **0** | **congelati** ai `tuned_defaults` misurati, mai ritarati dentro l'insieme |
| pesi | 0 | uguali nella versione base |
| θ_base, θ_macro | 2 | |
| isteresi | 1 | |
| stop ATR 15m | 0 | dai `tuned_defaults` a 15m |
| volatilità obiettivo | 1 | |
| innesco 15m | 1 | |
| **totale** | **5** | |

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
2. **Il campione.** Quattro piani di conferma su un asset fanno forse 5-15 operazioni l'anno: su
   cinque anni sono 25-75 operazioni, troppo poche per distinguere un effetto da un caso.
   Mitigazione obbligatoria: misurare **su cinque asset in comune** e per operazione, non per curva
   di equity.
3. **La correlazione fra i votanti.** Se i sei stati sono correlati 0,8 il punteggio ha tre valori
   effettivi e la soglia dinamica non ha niente su cui lavorare.

## Ordine di costruzione, uno stadio per volta

Ogni stadio aggiunge **un** meccanismo e si misura contro il precedente. Se uno non guadagna, si
scrive e ci si ferma li'.

| | cosa | stato |
|---|---|---|
| **S0** | correlazione fra i sei stati barra-per-barra | da fare per primo: può chiudere tutto in un pomeriggio |
| **S1** | `mtf.align_to_lower` + adattatore da cambi a stato per barra | **allineamento fatto** (`trading/mtf.py`, 5 test) |
| **S2** | punteggio a peso uguale, soglia fissa, ingresso alla barra 4H, un asset | contro i tre riferimenti |
| **S3** | soglia dinamica dai piani alti | +2 parametri |
| **S4** | innesco 15m, stop ATR 15m, volatilità obiettivo | +2 parametri |
| **S5** | pesi online (regola, non ricerca) | solo se S2-S4 hanno guadagnato |

## L'aspettativa, dichiarata prima di misurare

Sulla base di tutto ciò che questo progetto ha già misurato, mi aspetto che la confluenza arrivi
**allo stesso ordine di rendimento del possesso passivo con un drawdown molto minore** — lo stesso
posto in cui è arrivata la rotazione trasversale — e non a un rendimento superiore. Il rischio più
probabile non è che perda: è che **operi troppo poco perché si possa dire se ha funzionato**.

Dichiararlo adesso serve a una cosa sola: se il risultato sarà molto migliore di così, la prima
ipotesi da verificare non sarà il successo, sarà il look-ahead.
