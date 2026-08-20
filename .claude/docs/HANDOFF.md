# Handoff — CryptoFarm, strategia a 3 stati (BUY/SELL/HOLD)

Data: 2026-08-20. Branch: **`ai-labeling-rewrite`** (15 commit avanti su `main`, mai fatto merge).
Repo: `/Users/marcomonini/PycharmProjects/CryptoFarm`.

## Non duplicare: leggi prima questi

| documento | cosa contiene |
|---|---|
| `.claude/docs/strategy.md` | **fonte di verità delle decisioni.** Analisi completa, misure, piano a fasi con gate, risultati dell'implementazione precedente (§8bis). Ha una tabella di revisione in testa che elenca le correzioni già fatte. |
| `CLAUDE.md` | architettura del repo, comandi, variabili d'ambiente |
| `git log main..HEAD` | i messaggi di commit spiegano il *perché* di ogni scelta, inclusi i bug trovati e come |

Non riassumere quei contenuti: sono già scritti e aggiornati.

## Stato del lavoro corrente

L'utente ha **cambiato strategia** con un prompt dettagliato (modello a 3 classi BUY/SELL/HOLD
condizionato sullo stato della posizione, con mascheramento delle azioni non valide a inferenza).
L'ordine di lavoro che ha dato è in 5 punti, con l'istruzione esplicita:

> "1. Etichettatura + distribuzione delle classi e dei ritardi di conferma → **fermati e mostrami i numeri**"

**Fatto:** `src/cryptofarm/ml/directional_change.py` + `tests/test_directional_change.py`
(commit `c8e7b16`). Pivot per directional change con `extreme_bar` e `confirm_bar` separati,
etichetta morbida (`soft_labels`), `capturable_fraction`, `tune_threshold`. 97 test verdi.

**NON fatto, ed è il gate che l'utente aspetta:** le misure sui 15 simboli reali.
Vanno prodotte e mostrate prima di passare al punto 2:

1. ritardo di conferma dei pivot — mediana e 90° percentile, **per simbolo e per soglia**;
2. soglia tarata per simbolo che porta a **8–12 estremi/giorno** (`tune_threshold` esiste già);
3. distribuzione delle classi con l'etichetta morbida al 60% (attesa: positivi al 10–15%);
4. verifica dei numeri di riferimento del punto 2 del prompt (gamba mediana ~0,95% a soglia 0,5%,
   ~60% catturabile alla conferma) — **sono da simulazione e vanno confermati sui dati reali**.

I punti 3–5 del prompt (feature di posizione + baseline, randomizzazione dello stato, DAgger,
CPCV) non sono iniziati.

## Cose che non stanno nei documenti e servono subito

- **Usa `.venv312/bin/python`.** Il `.venv` preesistente è Python 3.9 senza `scikit-learn` e il
  progetto richiede ≥3.12. `.venv312` è gitignorato e ha tutto (`pip install -e ".[dev]"`).
- **`market_data/`** contiene 11.770.246 candele 5m su 15 simboli (298 MB, gitignorato).
  15m/30m/1h si derivano per aggregazione. Rigenerabile con
  `python -m cryptofarm.data.klines --update` (~minuti, dump CDN paralleli).
- **`analysis_cache/`** è popolata (gitignorata). La pagina Streamlit la usa:
  `streamlit run src/cryptofarm/app/analysis_dashboard.py`.
- **`models/*.joblib` e `*.json` non sono tracciati** (gitignore). `meta_model.*` è il modello
  della strategia *precedente* (meta-labeling su eventi CUSUM) — non cancellarlo, il simulatore
  lo carica ancora tramite `load_signal_model()`.
- **Gli script di misura della sessione stavano nella scratchpad ed è effimera.** Quelli salvati
  sono in `scripts/analysis.py`. La tabella "Riproducibilità" in fondo a `strategy.md` è stata
  **corretta**: ora distingue le misure conservate in `scripts/analysis.py` da quelle prodotte da
  script effimeri e non più rieseguibili, elencate esplicitamente come debito.
- 4 errori `ruff` in `trading/simulator.py`, `app/live_bot*.py`, `app/grid_results_viewer.py`
  sono **pre-esistenti** e non vanno confusi con regressioni.

## Regole di ingaggio stabilite dall'utente

- Prima di modifiche strutturali: piano scritto, poi conferma. (Sospeso quando l'utente dice
  esplicitamente "procedi con l'implementazione".)
- **Ogni numero va misurato sui dati del progetto, mai stimato né ripreso dai prompt.** L'utente
  ha ripetuto: *"se una misura contraddice una tesi del prompt, riportalo — preferisco una
  strategia corretta a una che conferma quello che ho chiesto."* È già successo più volte e va
  fatto senza addolcire.
- Controlli a cascata su ogni risultato, e sospetto verso i risultati troppo buoni. Nella sessione
  precedente il test di permutazione ha rivelato che ~1/3 dell'edge apparente non veniva dalle
  etichette; l'edge riportato va corretto per quel controllo.
- Commit incrementali con riepilogo dopo ogni blocco.

## Trappole già incontrate su questo tipo di codice

- **Pivot retrospettivi**: usare `extreme_bar` invece di `confirm_bar` in una feature è look-ahead.
  Misurato in `.claude/docs/strategy.md` §7.1: ritardo mediano 1–8 barre ma p99 fino a 101, ed è massimo
  proprio sui movimenti ampi.
- **Etichette sovrapposte**: `t_exit` qui è il pivot successivo confermato, quindi orizzonte
  **variabile e potenzialmente lungo**. L'embargo va dimensionato sul percentile alto, non sulla
  mediana. `ml/validation.py` ha già `PurgedKFold`, `CombinatorialPurgedCV`, `sample_uniqueness`,
  PBO e Deflated Sharpe.
- **Rolling non causali**: `labeling.py` usa di proposito `[::-1].rolling(...)[::-1]` per guardare
  avanti. Corretto lì, disastroso altrove.
- Nella sessione precedente due difetti del target hanno **invertito il segno** dei risultati
  (addestrare sugli ordini non riempiti; classificatore binario cieco alle magnitudini). Il
  commit `7ebb2e0` li spiega — vale la pena rileggerlo prima di definire il nuovo target.

## Suggested skills

Il prossimo agente dovrebbe invocare via Skill tool:

- **`tdd`** — per i punti 3–4 del prompt (feature di posizione, randomizzazione dello stato,
  DAgger). Sono tutti casi in cui il test scritto prima chiarisce cosa deve valere; i bug più
  costosi di questa sessione sono stati trovati proprio dai test.
- **`diagnosing-bugs`** — quando una misura non torna. È già servito: il rilevatore di pivot non
  trovava nessun estremo perché con direzione indecisa entrambi i rami si eseguivano.
- **`dataviz`** — prima di toccare `app/analysis_dashboard.py` o di aggiungere grafici, per
  mantenere coerenza con le tavolozze e le convenzioni già in uso.
- **`artifact-design`** — solo se l'utente chiede un report visuale pubblicabile; i deliverable
  finora sono file nel repo.

Non serve `research` (non ci sono fonti esterne da consultare) né `codebase-design` (la struttura
a moduli è già decisa e documentata in `.claude/docs/strategy.md` §8).
