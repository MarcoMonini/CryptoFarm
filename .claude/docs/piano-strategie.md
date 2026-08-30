# Piano — migliorare le strategie, e farne di nuove

Deciso il **2026-08-27** con l'utente, in `AskUserQuestion`, sul branch
`claude/ricerca-quant-ml-cinque-asset`. Sostituisce la lista "cosa farei dopo" di
[`HANDOFF.md`](HANDOFF.md), che resta come contesto.

## Da dove nasce

Le misure su cinque asset dicono tre cose che insieme scelgono il piano:

- **la tempificazione per-asset e' quasi morta**: 24% delle celle fuori campione batte il possesso
  passivo, e 9 di quelle 12 vittorie stanno in finestre dove il passivo perdeva;
- **la sezione trasversale e' l'unica famiglia che trasferisce**: 89% di configurazioni in utile
  fuori campione, e il vantaggio vero e' sul rischio (DD 45,7% contro 91,0%), non sul rendimento;
- **scegliere i parametri danneggia**: ρ = −0,69 fra resa in stima e in verifica.

Il terzo punto e' quello che nessuno ha ancora preso sul serio fino in fondo. Se selezionare
peggiora, la risposta non e' "prendere valori centrali" — e' **non selezionare affatto**.

## Due decisioni prese, da non riaprire

1. **Il ciclo 2017-2020 si spende dopo i passi 2-4**, non subito. E' l'ultima finestra di verifica
   pulita rimasta: bruciarla sul disegno attuale la toglie al disegno nuovo. Costo accettato: i
   passi 2-4 si costruiscono sopra un risultato non ancora confermato su un secondo ciclo.
2. **Si parte dal passo 1**, la molteplicita', perche' e' il prerequisito degli altri: senza, ogni
   numero che i passi 2-4 produrranno e' di nuovo un massimo di griglia non corretto.

---

## Passo 1 — molteplicita' *(in corso, commit `d49f46c`)*

**Cosa.** `deflated_sharpe_ratio` accetta `trial_variance`; `scripts/multiplicity.py` applica DSR
alle griglie di `reports/cs_*.csv` e PBO combinatorio alle matrici (anno × configurazione) gia' in
`analysis_cache/*/*_annuale.parquet`. Nessuno sweep da rieseguire.

**Fatto qui**, sulla rotazione trasversale (160 configurazioni, 2021-2026, 2057 osservazioni):

| prove contate | soglia del caso | DSR del massimo | DSR della mediana |
|---|---|---|---|
| 160 (la sola famiglia) | 0,77 annuo | **0,976** | 0,811 |
| 12.000 (tutto il progetto) | 1,12 annuo | 0,875 | 0,523 |
| fuori campione 2024-2026 | 0,84 annuo | 0,736 | 0,486 |

**Come si legge.** Il massimo sopravvive solo contando le prove della sola famiglia. Con il conto
onesto non sopravvive, e **la mediana non sopravvive in nessuno dei due conti** — ed e' la mediana
la configurazione che si terrebbe, visto che ottimizzare danneggia. Non e' una condanna della
rotazione: l'82% della griglia supera la soglia del caso, che e' un fatto sulla *famiglia* e non
sulla cella fortunata. E' una condanna del leggere quella griglia dal suo massimo.

**Cosa manca**, e non e' eseguibile ne' qui ne' dall'utente finche' non e' sulla macchina che ha
il clone (`analysis_cache/` e' gitignorata, e la sessione remota non ha ne' candele ne' rete verso
gli exchange):

```bash
python -m scripts.multiplicity --cache          # PBO su tutte le griglie gia' in cache
python -m scripts.multiplicity --selfcheck      # 5 controlli, gira ovunque
```

Se il PBO esce sopra 0,5 su una famiglia, quella famiglia va letta **solo** per mediana: la sua
procedura di selezione fa peggio del caso, ed e' una misura, non un'opinione.

---

## Passo 2 — ensemble di griglia

**L'idea.** ρ = −0,69 dice che scegliere una configurazione distrugge valore. La conseguenza
meccanica non e' "scegliere meglio": e' **tenerle tutte**. Un portafoglio a peso uguale di tutte
le configurazioni della griglia non ha parametri da scegliere, quindi non ha niente da
sovradattare, e la sua resa attesa e' la mediana della griglia — che e' proprio la colonna che il
progetto ha gia' imparato a leggere come quella onesta.

**Perche' e' plausibile, non solo elegante.** Le configurazioni vicine sono quasi la stessa
strategia (lo dice la dispersione misurata al passo 1: 0,0149 contro 0,0221 di prove
indipendenti), quindi mediarle non diversifica molto il rendimento — ma smussa il momento
d'ingresso, che e' dove sta la varianza che non trasferisce.

**Costo.** ~20 righe sopra `rotation.py`. Nessuna dipendenza nuova.
**Verifica.** Fuori campione 2024-2026, contro tre riferimenti: la configurazione centrale
attuale, la migliore in stima, e l'universo a peso uguale. Il criterio di successo e' dichiarato
prima: l'ensemble deve battere **la migliore in stima**, che e' la procedura che si sta
sostituendo. Se non la batte, l'idea e' morta e si scrive.

---

## Passo 3 — volatility targeting

**L'idea.** Ogni misura di questo progetto e' a capitale pieno, sempre. L'unico vantaggio
trasversale mai trovato e' **sul rischio** (DD dimezzato). Lo strato che agisce direttamente su
quell'asse non e' mai stato provato: scalare l'esposizione sull'inverso della volatilita'
realizzata, con un tetto.

**Perche' qui e non altrove.** Cripto ha volatilita' che varia di un fattore cinque fra regimi.
A esposizione fissa il rischio del portafoglio e' interamente deciso dal mercato; a rischio
mirato e' deciso da chi scrive la regola. E si compone con tutto: rotazione e strategie per-asset.

**Costo.** `pnl.simulate_positions` conosce gia' la leva — serve renderla per-barra invece che
costante. Poche righe, ma toccano il motore: test prima.
**Verifica.** A parita' di drawdown, il rendimento sale? E' l'unica domanda. Il confronto va fatto
riscalando entrambe le curve allo stesso drawdown, non a parita' di leva nominale.
**Trappola nota.** La volatilita' realizzata va calcolata su barre chiuse a `i-1`. E' lo stesso
difetto dello stop a trailing gia' trovato una volta, e `test_no_look_ahead` **non lo vedrebbe**.

---

## Passo 4 — momentum residuo, e media dei ranghi

**L'idea.** La rotazione oggi ordina per forza grezza. In cripto quasi tutto il rendimento e' beta
di BTC, quindi quella classifica ordina soprattutto per *quanto beta* ha ogni asset — non per
quale sta facendo meglio del dovuto. Ordinare sul **residuo contro BTC** e' informazione diversa.

Secondo pezzo: invece di un segnale solo, la **media dei ranghi** di piu' segnali (momentum
residuo, bassa volatilita', qualita' del trend). Media dei ranghi, non pesi stimati: stimare pesi
e' esattamente la selezione che il passo 1 e ρ = −0,69 dicono di non fare.

**Perche' e' la direzione giusta.** E' il punto del benchmark qlib letto in `ricerca-quant-ml.md`
§1: il soffitto e' IC ≈ 0,05, e si monetizza **in sezione**, non nel tempo, perche' l'errore si
media su piu' asset.

**Costo.** Un segnale nuovo in `rotation.py` piu' la combinazione per ranghi. ~40 righe.
**Verifica.** Stessa griglia, stessi riferimenti del passo 2, piu' il controllo che conta: il
residuo deve battere la forza grezza **sulla mediana**, non sul massimo.

---

## Passo 5 — il secondo ciclo (2017-2020)

**Cosa.** Rifare rotazione e filtro meta sul 2017-2020 con l'universo che esisteva allora
(BTC, ETH, XRP, BNB, LTC), **con il disegno uscito dai passi 2-4**, non con quello attuale.

**Perche' e' l'ultimo.** E' l'unica verifica veramente indipendente rimasta, e si spende una volta
sola: ogni misura fatta su quella finestra la contamina per la successiva. I dati sono gia' nello
store dell'utente.

**Criterio, dichiarato prima di guardare.** Il disegno nuovo deve, su 2017-2020: mediana positiva,
battere l'universo a peso uguale, e drawdown sotto quello del passivo. Tre condizioni, decise ora
proprio perche' deciderle dopo sarebbe un'altra selezione.

---

---

## Passo 2bis — il consenso fra strategie

Chiesto dall'utente il 2026-08-27: un algoritmo che riconosce le condizioni di mercato, ne ricava
**pesi di veridicita'** per ogni strategia, e agisce quando la somma pesata supera una **soglia
dinamica**. E' la stessa idea del passo 2 un piano sopra -- li' si mediano le configurazioni di una
strategia, qui le strategie fra loro -- e vale la stessa regola: **i pesi non si stimano**.

### Il dato che c'era gia', e che nessuno aveva letto cosi'

`live_bot.py` -- l'unico codice del progetto che muove denaro vero -- vota gia': `NUM_CONDITIONS`
decide quante fra banda ATR e RSI devono concordare (`live_bot.py:441`, `:458`). E la griglia
`close_buy_sell_limits` di `strategy_sweep` **sweepa `num_cond` fra 1 e 2**, 864 configurazioni per
lato, su tutti e cinque i simboli e tre intervalli. E' in `reports/sensibilita_*.csv` dal primo
giorno.

Chiedere due condizioni invece di una, mediana del rendimento:

| intervallo | migliora | invariato | peggiora | trade/anno mediani |
|---|---|---|---|---|
| 15m | BTC, ETH | — | — | 279 → 53 |
| 4h | BNB, BTC, ETH, SOL | — | XRP | 15-17 → 2-3 |
| 1d | BNB, ETH, SOL | BTC | XRP | 3 → **0** |

Nove su dodici migliorano, lo Sharpe mediano sale in dieci. **Ma non e' una prova che il voto
aggiunga informazione**: taglia le operazioni di cinque-dieci volte, e questo progetto ha gia'
stabilito che la frequenza operativa spiega quasi tutto. A un giorno la mediana passa a 0,0% con
zero operazioni mediane: la strategia non e' migliorata, ha smesso di operare.

**Il controllo che manca, e che decide:** confrontare il voto a due condizioni con **una condizione
sola tarata sulla stessa frequenza operativa**. Se il voto non batte quel riferimento, non sta
selezionando meglio -- sta solo operando meno, e operare meno costa una riga, non un algoritmo.

### La diagnosi da fare per prima, prima di scrivere l'algoritmo

**La matrice di correlazione fra le posizioni barra-per-barra delle strategie del menu.** Sono quasi
tutte di inseguimento del trend sullo stesso prezzo: se la correlazione media a coppie e' alta, il
voto e' una sola opinione contata dieci volte, e nessun sistema di pesi lo cambia. E' la misura piu'
economica del piano e puo' chiuderlo in un pomeriggio. **Non e' ancora stata fatta**, e non e'
deducibile da `reports/`, che tiene righe di riepilogo e non serie.

### Tre versioni annidate, una liberta' in piu' ciascuna

Si misura ognuna contro la precedente, e si passa alla successiva **solo** se guadagna:

1. **V0 — consenso a peso uguale.** k fra N strategie a parametri fissi (i `tuned_defaults`).
   Un solo parametro: k. Riferimenti: ogni strategia singola, e -- quello che conta -- ogni
   strategia singola ritarata alla stessa frequenza operativa.
2. **V1 — pesi online.** Peso di ogni strategia esponenziale nella sua resa recente (Hedge /
   pesi moltiplicativi). Un solo parametro: il tasso di apprendimento. I pesi li produce una
   regola, non una ricerca, e la garanzia teorica e' esattamente quella che serve dato ρ = −0,69:
   asintoticamente non si fa peggio della migliore strategia singola.
3. **V2 — pesi condizionati al regime** — la versione chiesta. Solo se V1 batte V0. E' qui che il
   numero di parametri esplode (un classificatore di regime x N strategie), ed e' la versione che
   ρ = −0,69 prevede fallisca.

La **soglia dinamica** segue la stessa regola: non un numero tarato per regime, ma una funzione
scale-free (per esempio chiedere piu' consenso quando la volatilita' e' alta), aggiunta una alla
volta e misurata come una liberta' in piu'.

### E' un'idea nota?

Si', e con nomi precisi: previsione con consulenti esperti (Hedge, pesi moltiplicativi), esperti
"dormienti" o specialisti -- che votano solo nel proprio contesto, cioe' esattamente "quali
strategie sono attendibili in questo regime" -- portafogli universali, modelli a cambio di regime
di Markov, stacking, e il meta-labeling di Lopez de Prado, che in questo repository e' gia'
implementato come `scripts/meta_gate.py`.

La prova piu' vicina a questo progetto sta gia' nell'artifact §1: nella tabella qlib il **primo
posto per IC e' DoubleEnsemble** (un ensemble) e il **primo per Rank IC e' TRA**, che e' un
instradatore che manda ogni campione a un predittore diverso -- cioe' la versione appresa della
"condizione di mercato che sceglie i pesi". La famiglia e' quella giusta. Ma i rendimenti annui di
quelle due righe sono 11,6% e 7,2%: l'ensemble vince la classifica **e resta sotto lo stesso
soffitto**. Non trasforma una famiglia perdente in una vincente.

---

## Cosa questo piano non fa, e perche'

- **Non integra quant e ML.** Il verdetto di `ricerca-quant-ml.md` §6 regge: il filtro non ha
  superato il proprio controllo casuale, e comporre uno strato non dimostrato con uno dimostrato
  non puo' che peggiorare il secondo. Si riapre se il filtro passa il controllo su un secondo ciclo.
- **Non allarga l'universo.** Misurato: mediana fuori campione da +62% a −0,9%.
- **Non aggiunge il verso corto.** Misurato in perdita su tutte e cinque le strategie.
- **Non cerca architetture profonde.** I benchmark qlib le mostrano sotto il gradient boosting, e
  la sezione trasversale non e' esaurita.
- **Non ottimizza i parametri.** E' il punto dell'intero piano.
