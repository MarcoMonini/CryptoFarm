# Handoff — CryptoFarm, strategia a 3 stati (BUY/SELL/HOLD)

Data: 2026-08-21. Branch: **`ai-labeling-rewrite`** (34 commit avanti su `main`, mai fatto merge).
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

**Tutti e 5 i punti del piano sono stati eseguiti fino in fondo.** Il risultato e' **negativo**, e
la causa e' nota e misurata. Sta tutto in `.claude/docs/strategy.md` §10–13; qui solo il minimo per
non ripartire da zero.

### Il risultato in una riga

Entrare alla conferma di un minimo e uscire alla conferma di un massimo cattura **zero in media**,
su tutti e 15 i simboli, a ogni soglia, **prima** dei costi (§13). La conferma si paga due volte e
la gamba mediana ne vale 1,76–2,05. Nessuna scelta di modello, feature o iperparametro lo cambia.

Di conseguenza: 0 split CPCV in utile su 15 in entrambe le bande di frequenza, edge lordo sotto il
costo **anche in-sample**, e il vantaggio degli ingressi del modello misurato con un'uscita
eseguibile e' −0,004% (§12.4).

### Cosa NON rifare

- Non ritarare `capture`, la soglia dei pivot o la soglia di decisione: misurato, nessuna aiuta
  (§12.6, §12.5).
- Non aggiungere iterazioni DAgger: funziona (disaccordo 13–19%, 913.000 righe raccolte) ma
  corregge un problema che non e' quello che abbiamo.
- Non provare un'architettura diversa: l'in-sample e' gia' sotto il costo, non e' overfitting.
- Non fidarsi di un'attribuzione con uscita "perfetta": e' il modo di non pagare la seconda soglia
  e fa sembrare informativi ingressi che non lo sono. Usare sempre la colonna causale
  (`confirmed_reversal_rows`) e il controllo con ingressi casuali.

### La sola direzione aperta (§13.4)

> Alla barra di conferma di un minimo, prevedere se **questa** gamba superera' `2 x soglia + costo`.

Classificazione binaria su ~10 eventi al giorno per simbolo invece di una politica per barra su
487.000 barre, positivi al 33–35% per costruzione, target su quantita' interamente causali. E' la
prima formulazione in cui il vincolo economico sta **dentro** il target. Costa un'ora di calcolo.
Le feature sono le stesse che hanno gia' fallito due volte, quindi non e' garantita: e' solo
l'unica rimasta che sia economica da provare.

Dopo di quella restano le leve di §3.2 e §6.2, in quest'ordine: dati di **microstruttura**
(`aggTrades`) — l'unica informazione che il modello non ha mai avuto — e il **modello di
riempimento maker** (Fase 0.3), senza il quale nessun numero in modalita' maker e' verificabile.

### Codice nuovo di questa sessione

| file | cosa |
|---|---|
| `data/klines.py` | `clip_wicks` / `wick_outliers`, applicati dentro `load_klines` (`clip=False` nel percorso di aggiornamento). Lo store su disco resta grezzo |
| `ml/directional_change.py` | finestra di `soft_labels` spostata alla barra di conferma; `confirmed_reversal_rows` (uscita causale) |
| `ml/policy.py` | stato della posizione, mascheramento delle azioni, randomizzazione dell'ingresso |
| `ml/dagger.py` | rollout con episodi batchati che non attraversano i confini fra simboli |
| `ml/policy_trainer.py` | dataset, DAgger, CPCV, holdout, attribuzione ingresso/uscita |
| `scripts/analysis.py` | `pivot_delays`, `pivot_labels`, `operating_points`, `confirmation_tax` |
| `ml/signals.py` + `trading/simulator.py` | `policy_signals`: la politica gira nel simulatore (commit `bacd384`). `load_signal_model` prova `policy_model` per primo, poi `meta_model`, poi `signal_model` |

`models/policy_model.*` e' una copia di `policy_alta.*`: e' quello che il simulatore carica.
Spostarlo altrove fa tornare in uso `meta_model`. Attenzione a due cose leggendo il grafico: il
simulatore applica **0,1% per lato** contro lo 0,08% andata-e-ritorno maker assunto in §12, e il
modello e' addestrato su **5m** mentre il simulatore gira anche su 15m.

`models/policy_alta.*` e `policy_bassa.*` sono i due modelli addestrati, con i rapporti JSON
completi (CPCV per split, sweep della soglia, attribuzione). Non sono tracciati.

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
- 10 rilievi `ruff` in `trading/simulator.py`, `app/live_bot*.py`, `app/grid_results_viewer.py`
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
