# Documentazione di lavoro — CryptoFarm

Tutto ciò che serve per riprendere il lavoro sta qui. `CLAUDE.md` nella radice resta dov'è perché
Claude Code lo carica automaticamente da lì, e rimanda a questa cartella.

## Ordine di lettura

Chi riprende da zero legge **`HANDOFF.md`** e basta: è lo stato corrente, e gli altri documenti li
referenzia. Chi deve toccare un pezzo preciso salta al documento di quel pezzo.

Chi vuole capire *dove è arrivato il progetto*, in ordine cronologico di risultato:
`backtest-strategie.md` → `strategie-nuove.md` → `ricerca-quant-ml.md` → `strategia-confluenza.md`
→ `politica-rl.md` → `modello-swing.md` → **`modello-ingresso.md`**, che è l'unico con numeri che
passano il controllo a esposizione appaiata.

## I documenti

| documento | quando serve |
|---|---|
| [`HANDOFF.md`](HANDOFF.md) | **da leggere per primo.** Stato dei due filoni con i risultati più recenti, cosa resta aperto, trappole ambientali e di misura, regole di ingaggio. Non duplica gli altri, li referenzia. **Da aggiornare a fine sessione.** |
| [`strategy.md`](strategy.md) | **fonte di verità delle decisioni** su labeling, feature, modello e validazione, con le misure che le giustificano. Ha una tabella di revisione in testa. Da aggiornare in luogo quando si decide qualcosa. |
| [`backtest-strategie.md`](backtest-strategie.md) | **le strategie a indicatori, misurate.** 3.129 configurazioni su nove anni di BTC: cosa rende, quanto dipende dai parametri, cosa resta fuori campione, e i difetti del codice trovati misurando. Le tabelle stanno in `reports/`, gli script che le producono in `scripts/{strategy_sweep,sweep_report,strategy_focus}.py`. |
| [`strategie-nuove.md`](strategie-nuove.md) | **seguito operativo del backtest.** Le quattro correzioni al codice e cosa hanno cambiato, il ciclo 2021-2026 come dataset, cinque strategie nuove e il motore che sa stare anche corto (`trading/strategies_ls.py`, `pnl.simulate_positions`). |
| [`ricerca-quant-ml.md`](ricerca-quant-ml.md) | le misure su cinque asset: rotazione trasversale (`scripts/cross_section.py`) e filtro meta (`scripts/meta_gate.py`), più §2, che è la ragione per cui sette voci sono uscite dal menu. |
| [`piano-strategie.md`](piano-strategie.md) | il piano deciso con l'utente il 2026-08-27: cosa il piano **non** fa, e perché. Il controllo di molteplicità (DSR/PBO) sta qui. |
| [`strategia-confluenza.md`](strategia-confluenza.md) | **la strategia multi-timeframe a più segnali, misurata.** Quattro piani con domande disgiunte, sei votanti scelti per famiglia, soglia decisa dai piani alti. Su 15 asset e sette anni **non batte il possesso passivo**: niente look-ahead, votanti non correlati, ma il gradiente di ogni parametro punta al non-operare. Le conclusioni stanno in fondo. |
| [`politica-rl.md`](politica-rl.md) | **la politica a rinforzo, cablata** (2026-08-28). Parte da una premessa dell'utente — «compra poco prima dei crolli» — e la misura falsa: gli ingressi hanno lo stesso drawdown di una barra qualunque, e ogni livello di stop peggiora il netto. La causa è la commissione. Da lì la forma dell'agente, con il costo dentro la ricompensa. Batte il possesso passivo 11/15 fuori campione e **dimezza la discesa massima**; il *quando* sta sopra il caso solo debolmente. |
| [`modello-swing.md`](modello-swing.md) | **il modello AI rifatto e misurato** (2026-08-28). L'audit che ha chiuso il modello a gambe (§1), l'etichetta nuova `labeling.swing_target` e perché il 93% di quel target è gratis, e le misure per cui il segnale esiste ma **non batte il caso a esposizione appaiata** (1 simbolo su 15). §5.4: cosa è stato cablato e cosa deliberatamente no. |
| [`modello-ingresso.md`](modello-ingresso.md) | **il modello in testa oggi, cablato** (2026-08-29). Cambia la domanda: non «quanto siamo vicini a un estremo» ma «quanto rende comprare qui». La leva è la **selettività**, non l'accuratezza. Sono i primi numeri del progetto che passano il controllo a esposizione appaiata: +2,071% netti per operazione fuori campione, 14/15 simboli in utile, 100° percentile. Il veloce opera, il lento gli fa da cancello. |
| [`MAPPA-modello-ai.md`](MAPPA-modello-ai.md) | i criteri di successo del lavoro sul modello AI, **dichiarati prima delle misure**. Sta qui apposta: serve a verificare che il bersaglio non sia stato spostato dopo. |

## Regole

- ciò che si decide va scritto **nel documento del pezzo**, non solo nel messaggio di commit;
- `git log` resta la fonte più densa sul *perché* di ogni scelta: i messaggi sono lunghi apposta;
- le misure vanno con i numeri e con il comando che le rifà, altrimenti non sono verificabili;
- un risultato negativo si scrive come si scrive uno positivo. Metà di questi documenti chiude
  una strada, ed è quello che impedisce di riaprirla per la terza volta.
