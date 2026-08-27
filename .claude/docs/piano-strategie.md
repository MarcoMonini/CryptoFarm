# Piano — migliorare le strategie, e farne di nuove

Deciso il **2026-08-27** con l'utente, in `AskUserQuestion`, sul branch
`claude/ricerca-quant-ml-cinque-asset`. Sostituisce la lista "cosa farei dopo" di
[`sessione-2026-08-27.md`](sessione-2026-08-27.md), che resta come contesto.

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

**Cosa manca**, e richiede la macchina dell'utente perche' `analysis_cache/` e' gitignorata:

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

## Cosa questo piano non fa, e perche'

- **Non integra quant e ML.** Il verdetto di `ricerca-quant-ml.md` §6 regge: il filtro non ha
  superato il proprio controllo casuale, e comporre uno strato non dimostrato con uno dimostrato
  non puo' che peggiorare il secondo. Si riapre se il filtro passa il controllo su un secondo ciclo.
- **Non allarga l'universo.** Misurato: mediana fuori campione da +62% a −0,9%.
- **Non aggiunge il verso corto.** Misurato in perdita su tutte e cinque le strategie.
- **Non cerca architetture profonde.** I benchmark qlib le mostrano sotto il gradient boosting, e
  la sezione trasversale non e' esaurita.
- **Non ottimizza i parametri.** E' il punto dell'intero piano.
