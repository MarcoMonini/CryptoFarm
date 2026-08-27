# Documentazione di lavoro — CryptoFarm

Tutto ciò che serve per riprendere il lavoro sta qui. `CLAUDE.md` nella radice resta dov'è perché
Claude Code lo carica automaticamente da lì, e rimanda a questa cartella.

| documento | quando serve |
|---|---|
| [`strategy.md`](strategy.md) | **fonte di verità delle decisioni.** Analisi, misure, piano a fasi con gate, risultati ottenuti. Ha una tabella di revisione in testa che elenca cosa è stato corretto e perché. È il documento da leggere per primo e da aggiornare quando si decide qualcosa. |
| [`HANDOFF.md`](HANDOFF.md) | ripartire da zero in una sessione nuova: stato dei due filoni (trading e ML) con i risultati piu' recenti, cosa resta aperto, trappole ambientali e di misura, regole di ingaggio. Non duplica gli altri documenti, li referenzia. **Da aggiornare a fine sessione.** |
| [`backtest-strategie.md`](backtest-strategie.md) | **le strategie a indicatori, misurate.** 3.129 configurazioni su nove anni di BTC: cosa rende, quanto dipende dai parametri, cosa resta fuori campione, e i difetti del codice trovati misurando. Le tabelle complete stanno in `reports/`, gli script che le producono in `scripts/strategy_sweep.py`, `scripts/sweep_report.py`, `scripts/strategy_focus.py`. |
| [`strategie-nuove.md`](strategie-nuove.md) | **seguito operativo del backtest.** Le quattro correzioni al codice e cosa hanno cambiato, la scelta del ciclo 2021-2026 come dataset, cinque strategie nuove (Donchian+ADX, squeeze Bollinger/Keltner, rientro StochRSI, Ichimoku, ritorno alla media con filtro ADX) e il motore a due versi con lo short. Tabelle `lab_*` in `reports/`. |
| [`ricerca-quant-ml.md`](ricerca-quant-ml.md) | **lo stato dell'arte letto nei repository, e i due filoni misurati su cinque asset.** Chiude i punti aperti di `strategie-nuove.md` (SOL e BNB, lo stop a trailing), corregge due sue conclusioni che non generalizzano, e apre due famiglie nuove: rotazione trasversale (`scripts/cross_section.py`) e filtro meta sopra una primaria vera (`scripts/meta_gate.py`). Tabelle `cs_*` e `meta_*` in `reports/`. |
| [`piano-strategie.md`](piano-strategie.md) | **il piano in corso (2026-08-27).** Cinque passi decisi con l'utente per migliorare le strategie e farne di nuove: molteplicita' (DSR/PBO sulle griglie gia' misurate), ensemble di griglia, volatility targeting, momentum residuo, e il ciclo 2017-2020 come verifica finale. Ogni passo ha il suo criterio di successo dichiarato **prima** della misura. Chiude anche cosa il piano non fa e perche'. |
| [`strategia-confluenza.md`](strategia-confluenza.md) | **il disegno della strategia multi-timeframe a piu' segnali** (2026-08-27, ipotesi non ancora misurata). Quattro piani con domande disgiunte (1D regime, 4H struttura, 1H conferma, 15m innesco), sei votanti scelti per famiglia, soglia decisa dai piani alti invece che tarata, conteggio dei parametri liberi, e i tre modi dichiarati in cui puo' fallire. Il primo stadio e' scritto: `trading/mtf.py`. |
| [`sessione-2026-08-27.md`](sessione-2026-08-27.md) | **chiusura dell'ultima sessione.** Le due decisioni prese con l'utente, le trappole d'ambiente scoperte misurando (fra cui: `analysis_cache/` e' gitignorata ed e' l'input di `tune_defaults`), i due test che passavano a vuoto, e cosa fare dopo in ordine. Ha una sezione "Suggested skills". |
| [`sessione-2026-08-21.md`](sessione-2026-08-21.md) | chiusura di una singola sessione: cosa era aperto al momento di staccare e cosa va confermato con l'utente prima di riprendere. Ne nasce uno per sessione, datato; non sostituisce `HANDOFF.md`, che resta il documento sempre valido. |
| `RESUME.md` (cartella sopra) | generato da Claude Code, non modificarlo a mano |

## Ordine di lettura consigliato

1. `CLAUDE.md` nella radice — architettura del repo e comandi
2. `HANDOFF.md` — dove siamo e cosa non è ovvio
3. `piano-strategie.md` — dove stiamo andando, e cosa si è deciso di non fare
4. `strategy.md` — perché le scelte sono quelle
5. `git log main..HEAD` — i messaggi di commit spiegano ogni decisione e i bug trovati

## Regola di manutenzione

`strategy.md` va **aggiornato in luogo**, non riscritto: la tabella di revisione in testa serve a
rendere visibile cosa è cambiato e perché. Le decisioni prese in una sessione e non scritte lì
vanno perse alla successiva.
