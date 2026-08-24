# Documentazione di lavoro — CryptoFarm

Tutto ciò che serve per riprendere il lavoro sta qui. `CLAUDE.md` nella radice resta dov'è perché
Claude Code lo carica automaticamente da lì, e rimanda a questa cartella.

| documento | quando serve |
|---|---|
| [`strategy.md`](strategy.md) | **fonte di verità delle decisioni.** Analisi, misure, piano a fasi con gate, risultati ottenuti. Ha una tabella di revisione in testa che elenca cosa è stato corretto e perché. È il documento da leggere per primo e da aggiornare quando si decide qualcosa. |
| [`HANDOFF.md`](HANDOFF.md) | ripartire da zero in una sessione nuova: stato del lavoro corrente, trappole ambientali, regole di ingaggio. Non duplica `strategy.md`, lo referenzia. |
| [`backtest-strategie.md`](backtest-strategie.md) | **le strategie a indicatori, misurate.** 3.129 configurazioni su nove anni di BTC: cosa rende, quanto dipende dai parametri, cosa resta fuori campione, e i difetti del codice trovati misurando. Le tabelle complete stanno in `reports/`, gli script che le producono in `scripts/strategy_sweep.py`, `scripts/sweep_report.py`, `scripts/strategy_focus.py`. |
| [`strategie-nuove.md`](strategie-nuove.md) | **seguito operativo del backtest.** Le quattro correzioni al codice e cosa hanno cambiato, la scelta del ciclo 2021-2026 come dataset, cinque strategie nuove (Donchian+ADX, squeeze Bollinger/Keltner, rientro StochRSI, Ichimoku, ritorno alla media con filtro ADX) e il motore a due versi con lo short. Tabelle `lab_*` in `reports/`. |
| [`sessione-2026-08-21.md`](sessione-2026-08-21.md) | chiusura di una singola sessione: cosa era aperto al momento di staccare e cosa va confermato con l'utente prima di riprendere. Ne nasce uno per sessione, datato; non sostituisce `HANDOFF.md`, che resta il documento sempre valido. |
| `RESUME.md` (cartella sopra) | generato da Claude Code, non modificarlo a mano |

## Ordine di lettura consigliato

1. `CLAUDE.md` nella radice — architettura del repo e comandi
2. `HANDOFF.md` — dove siamo e cosa non è ovvio
3. `strategy.md` — perché le scelte sono quelle
4. `git log main..HEAD` — i messaggi di commit spiegano ogni decisione e i bug trovati

## Regola di manutenzione

`strategy.md` va **aggiornato in luogo**, non riscritto: la tabella di revisione in testa serve a
rendere visibile cosa è cambiato e perché. Le decisioni prese in una sessione e non scritte lì
vanno perse alla successiva.
