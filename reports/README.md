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

## Rotazione trasversale e filtro meta

Due famiglie prodotte da script diversi, che rispondono a domande diverse dalle precedenti. La
lettura sta in [`.claude/docs/ricerca-quant-ml.md`](../.claude/docs/ricerca-quant-ml.md).

| file | cosa contiene |
|---|---|
| `cs_majors_1d.csv` | rotazione sui *majors* a scala giornaliera: sceglie **quale** asset, non quando |
| `cs_majors_1d_oos.csv` | la stessa griglia, scelta in campione e resa fuori |
| `cs_pairs.csv` | l'universo a coppie |
| `cs_pairs_2024.csv` | le stesse coppie sul solo 2024 |
| `cs_wide_1d_oos.csv` | l'universo largo a 15 asset, fuori campione |
| `meta_donchian_breakout_4h*.csv` | il secondario di meta-labeling sopra Donchian Breakout a 4h |
| `meta_trend_pullback_4h*.csv` | lo stesso sopra Trend Pullback a 4h |

Prodotte da `scripts/cross_section.py` e `scripts/meta_gate.py`:

```bash
.venv312/bin/python -m scripts.cross_section --universe majors --interval 1d --grid
.venv312/bin/python -m scripts.meta_gate --strategy donchian_breakout --interval 4h --oos 2024-01-01
```

Due avvertenze che valgono su tutte le tabelle di questa cartella:

**Il riferimento della rotazione è l'universo a peso uguale, non BTC.** Porta la stessa distorsione
da sopravvivenza, quindi il confronto isola ciò che la rotazione aggiunge. Contro BTC la rotazione
vince nel 95,6% delle configurazioni; contro l'universo, nel 44,4%.

**La riga migliore di una griglia non è un risultato.** È la cella più fortunata: la correlazione
fra resa in stima e resa in verifica sulle prime dieci configurazioni della rotazione è **−0,69**.
`scripts/multiplicity.py` (DSR e PBO) esiste per dire quanto di un massimo è la griglia.
