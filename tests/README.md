# `tests/` — 1.022 test in 35 file

> I numeri fra parentesi sono i casi **raccolti** da pytest, non le funzioni scritte: dove c'è
> `parametrize` i due divergono molto (`test_panels.py` ha 25 funzioni e 410 casi).

`.venv312/bin/python -m pytest`. Nessun test tocca la rete e nessuno richiede lo store delle
candele: dove servono dati, sono costruiti in memoria. È una condizione, non una comodità — è
quella in cui gira la CI e in cui deve girare chi ha appena clonato il repository.

## Come sono organizzati

Un file per modulo, con lo stesso nome (`test_confluence.py` ↔ `trading/confluence.py`), più tre
file che coprono un **livello** invece di un modulo.

### I tre di livello

| file | test | cosa protegge |
|---|---|---|
| `test_simulator_golden.py` | 75 | il **comportamento** di 21 funzioni su quattro scenari sintetici, contro `data/simulator_golden.json` |
| `test_simulator_page.py` | 8 | la pagina **come pagina**, eseguita con `streamlit.testing.v1.AppTest` |
| `test_scripts_importabili.py` | 18 | che ogni modulo di `scripts/` almeno si importi |

`test_simulator_page.py` è il livello da cui è passato il guasto che tolse il simulatore dalla
produzione: ogni funzione aveva i suoi test e passavano tutti, mentre un `load_signal_model()`
chiamato senza condizione impediva alla pagina di aprirsi. Copre anche la degradazione senza store
e senza modelli, che è la condizione del servizio pubblico.

### Il modello e i suoi segnali

`test_features.py` (7) · `test_labeling.py` (13) · `test_directional_change.py` (12) ·
`test_dataset.py` (14) · `test_evaluate.py` (14) · `test_validation.py` (16) ·
`test_execution.py` (7) · `test_signals.py` (10) · `test_model_discovery.py` (12)

`test_swing_target.py` (11) · `test_swing_features.py` (4) · `test_swing_signals.py` (8) ·
`test_swing_lab.py` (5) — il modello a swing, dal bersaglio al servizio.

`test_entry_trainer.py` (7) · `test_entry_signals.py` (10) · `test_entry_panel.py` (6) — il
modello d'ingresso, che è quello in testa oggi.

`test_rl.py` (6) · `test_rl_signals.py` (8) — la politica a rinforzo.

### Le strategie e il conto

`test_confluence.py` (56) · `test_confluence_lab.py` (8) · `test_confluence_audit.py` (4) ·
`test_ai_voter.py` (3) · `test_voters.py` (10) · `test_mtf.py` (5) · `test_long_short.py` (25) ·
`test_portfolio.py` (13) · `test_rotation.py` (9) · `test_panels.py` (410) ·
`test_tuned_defaults.py` (177) · `test_strategy_sweep.py` (15)

### I dati

`test_klines_store.py` (12) · `test_positioning.py` (4) — nessuna rete: i dump sono costruiti in
memoria.

## Cinque test che vanno letti prima di modificarne il modulo

Sono quelli che difendono da un difetto **invisibile leggendo il codice**, e riscriverli senza
capirli è il modo più rapido di reintrodurlo.

- **`test_mtf.py`** taglia *dentro* una barra lunga già cominciata. Un taglio allineato ai confini
  passa anche col look-ahead reintrodotto, ed è com'era scritto la prima volta.
- **`test_tuned_defaults.py`** asserisce sulla **chiave** del widget, non sul valore: Streamlit
  conserva lo stato per chiave, e `AppTest` ricostruisce lo stato a ogni run, quindi il difetto
  vero (i campi che restano fermi cambiando intervallo) non lo vedrebbe.
- **`test_panels.py`** conta le tracce del riquadro *Voters* contro `len(VOTANTI)`: è l'unico
  elenco della confluenza che va tenuto allineato a mano.
- **`test_model_discovery.py`** verifica che un artefatto vecchio in `models/` **non** riporti in
  servizio un disegno già chiuso in negativo. È il nome, non il ramo, a decidere cosa si carica.
- **`test_swing_signals.py`** fissa la regola a esposizione *e cosa non è*: `sign(previsione)` è la
  lettura naturale di un target in [−1, 1] ed è misurata in perdita a tutte le soglie.

## Il golden master

`test_simulator_golden.py` **deve passare prima di una modifica e passare ancora dopo, senza
rigenerarlo**. Rigenerare (`SIMULATOR_GOLDEN_REGEN=1 pytest tests/test_simulator_golden.py`)
accetta qualunque differenza di comportamento: farlo solo dopo aver verificato a mano che la
differenza sia voluta, e controllare che il diff del JSON contenga solo le righe attese.

Gli scenari non sono intercambiabili: `close_ema_crossover_simulation` pretende tre incroci EMA in
sequenza e scatta solo su un'inversione vera, `close_bullish_ema_simulation` solo in laterale.
Togliere uno scenario scopre delle strategie senza che nessun test fallisca.

## Lint

`ruff check src scripts tests` e `black src scripts tests`. La configurazione sta in
`pyproject.toml` (riga a 120 caratteri).
