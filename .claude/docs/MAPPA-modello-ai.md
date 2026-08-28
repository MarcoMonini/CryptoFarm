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
