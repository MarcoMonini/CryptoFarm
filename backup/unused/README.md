# Moduli rimossi da `src/cryptofarm/` (2026-08-21)

Nessuno di questi file era importato da `trading/simulator.py`, da `ml/trainer.py` o dalle loro
dipendenze, ne' da un test. Sono qui invece che cancellati: `git mv` da questa cartella li rimette
al loro posto con la storia intatta.

| File | Perche' e' qui |
|---|---|
| `app/dashboard_live.py` | Dashboard Streamlit live. Nessun importatore; duplicava indicatori e logica di segnale gia' presenti altrove. |
| `app/live_bot_dual.py` | ~90% identico a `live_bot.py`, con un secondo account. Nessun importatore. |
| `app/grid_results_viewer.py` | Visualizzatore dei CSV della grid search, che se ne va con essa. Percorso CSV cablato nel file. |
| `app/analysis_dashboard.py` | Pagina Streamlit sopra `scripts/analysis.py`, che resta ed espone le stesse misure da riga di comando. Importava `scripts.analysis`, un modulo di primo livello non impacchettato (`packages.find` guarda solo `src`), quindi funzionava solo dalla radice del repo. |
| `trading/grid_search.py` | Ricerca a griglia sui parametri di strategia. Nessun importatore; unico consumatore di `simulator_opt.py`. |
| `trading/simulator_opt.py` | Variante di `trading_analysis` usata solo da `grid_search.py`. |

`app/live_bot.py` non e' qui: e' stato spostato in `src/cryptofarm/trading/live_bot.py`.
