# Simulator v2 - Streamlit UI

## Overview
Questa e' una UI Streamlit pensata per simulazioni multi-timeframe con indicatori
configurabili a runtime. I controlli globali sono in alto, mentre la sezione
principale usa una "matrix" con tab per timeframe (controlli a sinistra e
grafici a destra). Il backend separa fetch dati, calcolo indicatori, strategie
e rendering.
Nota: la simulazione e' manuale. Usa il pulsante "Run simulation" per eseguire
fetch e calcoli; modifiche ai parametri non ricalcolano automaticamente.

## Struttura dei file
- `simulator_v2_app.py`: frontend Streamlit (UI, session_state, orchestrazione).
- `simulator_v2_backend.py`: backend (fetch Binance, indicatori, strategie, plot).
- `simulator_v2_ai_trainer.py`: trainer AI senza UI (multi-timeframe).
- `simulator_v2_massive.py`: ricerca massiva parametri (grid/random).

## Flusso dati (alto livello)
1. `init_session_state()` imposta i default UI.
2. `build_config()` normalizza la config, `validate_config()` valida i parametri.
3. `prepare_timeframe_data()` scarica dati e calcola indicatori per TF.
4. Strategia:
   - Single TF: `strategy.signal_func(...)`.
   - Multi TF: `mtf_close_buy_sell_limits(...)`.
5. `simulate_trades()` calcola i trade, `summarize_trades()` crea le metriche.
6. `build_timeframe_figure()` e `render_timeframe_panel()` rendono i grafici.

## Chiavi session_state (pattern)
- Timeframe:
  - `tf{n}_enabled` (bool), `tf{n}_value` (stringa intervallo).
- Toggle indicatori:
  - `tf{n}_ind_*` (es. `tf1_ind_rsi_short`).
- Parametri indicatori e strategia:
  - `tf{n}_{param}` (es. `tf2_atr_window`, `tf3_rsi_short_buy_limit`).
- Globali:
  - `asset_base`, `asset_quote`, `range_mode`, `range_hours`, date/time,
    `wallet`, `fee_percent`, `strategy`, `conditions_required`,
    `ai_model_path`, `ai_scaler_path`, `ai_metadata_path`,
    `ai_buy_threshold`, `ai_sell_threshold`.

## Indicatori
Il backend calcola solo gli indicatori attivi (flag in `indicator_flags`) per
ridurre il costo. Le colonne aggiunte in `compute_indicators()` sono:
- ATR Bands: `ATR`, `ATR_MID`, `ATR_UPPER`, `ATR_LOWER`.
- RSI: `RSI_SHORT`, `RSI_MED`, `RSI_LONG`.
- EMA: `EMA_SHORT`, `EMA_MED`, `EMA_LONG`.
- KAMA: `KAMA`.
- MACD: `MACD`, `MACD_SIGNAL`, `MACD_HIST`.
  - Note: MACD e' normalizzato come percentuale del prezzo.

Parametri strategia per timeframe (Buy/Sell Limits):
- `rsi_short_buy_limit` / `rsi_short_sell_limit`
- `rsi_medium_buy_limit` / `rsi_medium_sell_limit`
- `rsi_long_buy_limit` / `rsi_long_sell_limit`
- `macd_buy_limit` / `macd_sell_limit`

## Strategie
Le strategie sono definite in `STRATEGIES` tramite `StrategySpec`.
Campi principali:
- `required_indicators`: indicatori necessari per segnali.
- `default_params`: parametri default.
- `signal_func`: funzione che produce i segnali.
- `multi_timeframe`: abilita la pipeline multi-TF.

### Buy/Sell Limits (multi-timeframe)
Condizioni possibili per ogni TF abilitato:
- RSI Short/Medium/Long vs limiti dedicati per RSI
- ATR Bands: close vs `ATR_LOWER` / `ATR_UPPER`
- MACD (hist): `macd_buy_limit` / `macd_sell_limit`
- EMA Cross: incroci S/M, M/L, S/L

Il parametro `conditions_required` indica quante condizioni devono essere vere
per generare buy/sell. Il totale disponibile cambia in base agli indicatori
abilitati per TF. La timeline dei segnali usa il timeframe piu piccolo.

### AI Model (multi-timeframe)
Usa un modello AI addestrato per produrre segnali buy/sell:
- I feature derivano da indicatori multi-timeframe (come nel trainer).
- I parametri e indicatori usati sono definiti nei metadata del modello.
- La UI richiede i path di model/scaler/metadata e le soglie di probabilita'.
- I metadata includono anche regole di gap-handling (ffill e sequenze).

La strategia AI ignora i toggle indicatori nella UI per i segnali (servono
solo per la visualizzazione dei grafici).

## AI Trainer (background)
Eseguire:
```bash
python simulator_v2_ai_trainer.py
```

Output:
- `ai_models/mtf_ai_model_v1.keras`
- `ai_models/mtf_ai_model_v1_scaler.pkl`
- `ai_models/mtf_ai_model_v1_meta.json`

Nella UI selezionare `AI Model` e impostare i path ai file generati.
I timeframe selezionati in UI devono coincidere con quelli nei metadata.

### AI data quality knobs
Nel trainer puoi controllare la qualita' dei dati pre-addestramento:
- `LABEL_METHOD`, `LABEL_MIN_RETURN`, `LABEL_RETURN_HORIZON`, `LABEL_COOLDOWN`
  per ridurre segnali rumorosi e bilanciare buy/sell vs hold.
- `MAX_FFILL_GAP_MULTIPLIER` per evitare forward-fill troppo vecchi.
- `GAP_TOLERANCE`, `SEQUENCE_STRIDE`, `EMBARGO_STEPS` per ridurre leakage.
- `BALANCE_METHOD`, `HOLD_KEEP_RATIO`, `BALANCE_ASSETS` per gestire lo sbilanciamento.

Questi parametri sono salvati nei metadata per riproducibilita'.

## Come aggiungere un indicatore
1. Backend: aggiungi il calcolo in `compute_indicators()` e le colonne.
2. Backend: aggiorna `build_timeframe_figure()` per il rendering.
3. Frontend: aggiungi toggle in `INDICATOR_STATE_KEYS` e default in
   `INDICATOR_TOGGLE_DEFAULTS`.
4. Frontend: aggiungi parametri in `INDICATOR_PARAM_KEYS` e default in
   `INDICATOR_PARAM_DEFAULTS` (e `TIMEFRAME_PARAM_DEFAULTS`).
5. Frontend: aggiungi i controlli in `render_indicator_controls()`.
6. Strategia (opzionale): integra la nuova condizione.

## Come aggiungere una strategia
1. Backend: implementa una funzione segnali `(df, params) -> (buy, sell)`.
2. Backend: registra la strategia in `STRATEGIES` con default e indicatori.
3. Frontend: se servono parametri extra, aggiungi input nella sezione Strategy.
4. Frontend: se multi-timeframe, aggiungi la logica nel ramo `strategy.multi_timeframe`.

## Avvio e configurazione
Eseguire:
```bash
streamlit run simulator_v2_app.py
```

Variabili utili:
- `BINANCE_API_KEY`, `BINANCE_API_SECRET`
- `BINANCE_SSL_VERIFY` (0/1)
- `BINANCE_PROXY_HTTP`, `BINANCE_PROXY_HTTPS`

Nota: `get_binance_client()` applica `requests_params` con SSL/proxy usando
`BINANCE_SSL_VERIFY` e `BINANCE_PROXY_*`.
