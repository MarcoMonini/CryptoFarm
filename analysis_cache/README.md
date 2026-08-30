# `analysis_cache/` — i risultati intermedi delle misure

**Non tracciata.** Circa 31 MB. Tutto qui dentro si rigenera rilanciando lo script che l'ha
prodotto: è cache, non dato. Cancellarla costa tempo di CPU e nient'altro.

La distinzione con [`../reports/`](../reports/), che invece si traccia, è netta: qui stanno le
misure **grezze**, una riga per configurazione provata; là stanno le tabelle finali, quelle che i
documenti di `.claude/docs/` citano. `scripts/sweep_report.py` è il passaggio fra le due.

## Cosa contiene

| cosa | prodotto da | cosa tiene |
|---|---|---|
| `sweeps/` (342 file, 10 MB) | `scripts/strategy_sweep.py` | ogni configurazione di ogni strategia, per simbolo/intervallo/commissione |
| `lab/` (4,1 MB) | `scripts/strategy_lab.py` | lo stesso per le strategie a due versi |
| `confluence_audit/` (9,1 MB) | `scripts/confluence_audit.py` | le stesse configurazioni su molti asset, dentro e fuori campione |
| `ai_voter/` (7,8 MB) | `scripts/ai_voter.py` | il votante addestrato sulle operazioni della confluenza |
| `*.parquet` sciolti | `scripts/analysis.py` | le singole misure di `strategy.md`: `break_even`, `cusum_rates`, `concurrency`, `market_regimes`, `pivot_delays`, `random_walk`, `time_to_target`, `barrier_capacity`, `capture_sweep`, `operating_points`, `store_coverage` |

Il nome dei file degli sweep porta i parametri della misura
(`atr_bands_15m_BTCUSDT_2021_fee005.parquet`, con `_annuale` per la vista anno per anno): è così
che si sa cosa si sta guardando senza aprirlo.
