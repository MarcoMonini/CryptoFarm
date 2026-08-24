# Tabelle del backtest delle strategie

Prodotte da `scripts/sweep_report.py` e `scripts/strategy_focus.py` a partire dagli sweep in
`analysis_cache/sweeps/` (che non si tracciano: sono decine di MB e si rigenerano). La lettura
ragionata di questi numeri sta in [`.claude/docs/backtest-strategie.md`](../.claude/docs/backtest-strategie.md).

Tutte le misure sono su BTC/USD a 15 minuti dal 2017-01-01 al 2026-08-24, capitale 100 sempre
reinvestito, commissione 0,1% per gamba salvo dove indicato.

| file | cosa contiene |
|---|---|
| `riferimento_15m.csv` | il possesso passivo per periodo e per anno: il metro di tutto il resto |
| `panoramica_15m.csv` | una riga per strategia: migliore, mediana, peggiore, quota in utile, quota che batte il possesso passivo |
| `frequenza_15m.csv` | tutte le configurazioni raggruppate per operazioni all'anno — la relazione che spiega piu' di ogni altra |
| `sensibilita_15m.csv` | per ogni parametro e ogni suo valore: mediana, migliore, quota in utile |
| `escursione_15m.csv` | quanto sposta ogni parametro a parita' di tutti gli altri |
| `stabilita_15m.csv` | il rendimento anno per anno della configurazione migliore di ogni griglia |
| `fuori_campione_15m.csv` | scelta su 2017-2021, resa su 2022-2026, con la mediana delle prime dieci |
| `walk_forward_15m.csv`, `walk_forward_dettaglio_15m.csv` | riottimizzazione annuale sui soli anni gia' visti |
| `commissioni.csv` | le configurazioni migliori rieseguite a commissione 0%, 0,04%, 0,075%, 0,1%, 0,2% |
| `intervalli.csv` | le stesse configurazioni rieseguite da 5m a 1d, senza ritoccare i parametri |
| `*_ETHUSD.csv` | le stesse viste sul mercato di controllo: ETH/USD di Bitfinex, 2017-2019 (§9 del documento) |
| `lab_panoramica_*.csv` | strategie a due versi: migliore, mediana e quota in utile per intervallo |
| `lab_effetto_short_*.csv` | la stessa configurazione con e senza il verso corto |
| `lab_ablazioni_*.csv` | ogni filtro spento a turno: quanto vale ADX, regime, volume, nuvola |
| `lab_classifica_*.csv` | storiche e nuove sullo stesso periodo e con lo stesso costo |
| `lab_fuori_campione_*.csv` | scelta sul 2021-2023, resa sul 2024-2026, per entrambe le famiglie |
| `lab_leva_costi_*.csv` | le migliori a leva 1, 2 e 3 e a tre livelli di commissione e funding |

Le viste `lab_*` sono su BTC/USD 2021-2026 (e ETH/USD 2017-2019 come controllo), commissione 0,05%
per gamba piu' 0,03% al giorno di mantenimento: la lettura sta in
[`.claude/docs/strategie-nuove.md`](../.claude/docs/strategie-nuove.md).
