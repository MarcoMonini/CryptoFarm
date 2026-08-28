# Mappa delle capacità — riaddestramento del modello AI (2026-08-28)

Stato: **in attesa di approvazione**. Nessuna spec di modulo va scritta prima che questa mappa
sia approvata: sbagliare i confini costa, rivedere quindici righe no.

## Perché una mappa e non una spec sola

La richiesta impacchetta cinque capacità che si verificano separatamente: uno store di dati
nuovo, un insieme di feature, un addestramento, e **due** consumatori del modello che pongono
domande diverse (la voce di menu decide *quando entrare e uscire da sola*; il votante decide
*come votare in un collegio*). Una spec unica costringerebbe ogni task a valere su tutto il
contratto.

## I moduli

| id | responsabilità | dipende da |
|---|---|---|
| `positioning` | store locale di funding rate, open interest, long/short e taker ratio dai dump bulk di Binance, gemello di `data/klines.py` | — |
| `features-bar` | feature per barra, scale-free, con contesto trasversale e posizionamento; una sola definizione condivisa fra addestramento e inferenza | `positioning` |
| `model-legs` | addestramento delle due teste (`P_su`, `P_giu`), validazione purgata + verifica temporale + controllo casuale, artefatto con metadata | `features-bar` |
| `strategy-ai` | la voce di menu «AI Model»: ingresso su `P_su`, uscita su `P_giu` o barriera; e lo spostamento di `policy_model` fuori dalla precedenza | `model-legs` |
| `voter-ai` | il `Votante` per barra dentro la confluenza, che vota +1/−1 e si registra come gli altri sei | `model-legs` |

Ordine di costruzione: `positioning` → `features-bar` → `model-legs` → `strategy-ai`, `voter-ai`

Nessun ciclo. `strategy-ai` e `voter-ai` sono paralleli e non si conoscono: entrambi leggono
l'artefatto di `model-legs`, che è l'interfaccia.

## Il criterio di successo dell'iniziativa, dichiarato prima

Non «l'AUC sale». Il progetto ha già misurato tre volte che un vantaggio di ordinamento reale non
paga. Il numero da battere, dichiarato adesso e non dopo:

1. **fuori campione** (addestrato < taglio, misurato dopo, un taglio solo dichiarato prima), il
   netto medio per operazione deve stare sopra il **p95 di 500 selezioni casuali di pari
   numerosità** — lo stesso controllo di `meta_gate`/`ai_voter`, che finora nessun disegno ha
   superato in modo stabile;
2. su **due soglie adiacenti**, non una: un solo picco fra soglie vicine è rumore, ed è già
   successo (`ai_voter` a 0,45 rende −1,5% fra 0,40 e 0,50 che rendono +0,8% e +2,0%);
3. la mediana degli ingressi deve cadere **prima del 43% della gamba** — il numero che la
   confluenza fa oggi. È l'unico criterio che traduce «anticipare» in una misura.

Se 1 e 2 non passano, il risultato si scrive e il filone si chiude con una misura, non con
un'opinione. Il criterio 3 può passare da solo ed è comunque un'informazione.

## Quello che la mappa esclude di proposito

- **niente politica a tre azioni**: `strategy.md` §12-13, chiusa in negativo con la causa nota;
- **niente `aggTrades`**: `sum_taker_long_short_vol_ratio` è la stessa informazione già aggregata
  a 5 minuti, e nel pannello non ha superato il controllo di segno — quindi non vale centinaia
  di GB;
- **niente architetture profonde**: benchmark qlib, `ricerca-quant-ml.md` §1.1;
- **niente ottimizzazione dei parametri dei votanti** insieme al modello.

---

## Decisioni prese con l'utente (2026-08-28)

| bivio | scelta | conseguenza |
|---|---|---|
| dati di posizionamento | **sì, solo `retail_pos` e `top_pos`** | `positioning` scarica e conserva tutte le colonne (arrivano nello stesso file, non costa niente), ma `features-bar` ne usa due. Le altre dieci — funding compreso — non hanno superato il controllo di segno sul pannello 5 asset × 2 finestre |
| scala | **1h + 4h + 1d, con `TIMEFRAME` come feature** | un modello solo copre i piani su cui girano i votanti di conferma, struttura e regime. Sotto l'ora resta escluso: è la regione già misurata perdente |
| teste | **una sola, tre classi su barriere simmetriche** | da ogni barra: `+k·ATR` per primo (SU), `−k·ATR` per primo (GIÙ), nessuno dei due entro `H` (FERMO). `P_su` entra, `P_giu` esce e vota −1 |
| stato della posizione | **fuori dalle feature** | il modello non sa se una posizione è aperta. È un'opinione sulla barra, indipendente dal trading in corso — ed è ciò che rende identico l'artefatto per i due consumatori |

### Perché il tre-classi qui non è il tre-classi già bocciato

La differenza è la **simmetria delle barriere**, non il numero di classi. Con `TP_ATR_MULTIPLE = 1.5`
e `SL_ATR_MULTIPLE = 1.0` la classe «sell» significa «lo stop di una posizione lunga è stato
toccato per primo»: copre ~60% delle barre e confonde «scende» con «scende un po' e poi sale». È
la ragione scritta in `ml/signals.py` per cui quella classe non va usata come segnale di vendita.

Con barriere simmetriche la classe GIÙ significa «è sceso di `k·ATR` prima di salirne altrettanti»,
cioè esattamente la gamba ribassista da evitare. Le due classi direzionali diventano confrontabili
fra loro, che è la proprietà che serve a `P_su` contro `P_giu`.

Il prezzo della simmetria è che scompare l'argomento di `labeling.py` sul break-even (con barriere
2:1 la precisione di pareggio scende dal 66,7% al 44,4%). Qui non si applica allo stesso modo:
il modello sceglie una **direzione**, non solo se entrare, e il pavimento sulle commissioni resta
il vincolo economico dentro l'etichetta.
