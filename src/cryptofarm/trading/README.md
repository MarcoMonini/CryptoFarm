# `trading/` — la pagina, le strategie, il conto

Tutto ciò che va da candele a operazioni, più la pagina Streamlit che lo mostra e il bot che
piazza ordini veri. Le dipendenze formano un DAG e **non c'è una facciata di ri-esportazione**:
chi serve una strategia la importa dal modulo che la contiene.

```
market_data ─┐
indicators ──┼→ strategies ────┐
indicators_extra                ├→ simulator (la pagina)
mtf → voters → confluence ─────┤
strategies_ls → pnl ───────────┘
config, panels, tuned_defaults → simulator
                                  rotation, portfolio (le altre due domande)
```

## I file

| file | righe | a cosa serve |
|---|---|---|
| `config.py` | 255 | i valori di partenza dei widget e gli elenchi del menu. Nessuna logica |
| `market_data.py` | 304 | scarico puntuale da Binance per la pagina (la REST pubblica, non i dump) |
| `indicators.py` | 212 | gli indicatori classici, più il nucleo numpy ATR/EMA |
| `indicators_extra.py` | 186 | ADX, Donchian, Bollinger/Keltner, StochRSI, OBV/MFI, Ichimoku, dietro una cache |
| `strategies.py` | 848 | da candele con indicatori a `(buy_signals, sell_signals)` — solo lungo |
| `strategies_ls.py` | 537 | da candele a **cambi di posizione** `(quando, prezzo, +1\|0\|-1)` — lungo, fuori, corto |
| `pnl.py` | 277 | da segnali a operazioni: commissioni, leva, costo di mantenimento |
| `mtf.py` | 87 | l'allineamento fra intervalli: legge la barra lunga **chiusa**, mai quella corrente |
| `voters.py` | 155 | da cambi di posizione a voto per barra, con memoria e decadimento |
| `confluence.py` | 1004 | la strategia a confluenza: sei votanti su quattro piani, soglia dinamica |
| `portfolio.py` | 308 | un capitale solo su più asset: si apre sul primo che parla |
| `rotation.py` | 215 | rotazione trasversale: sceglie *quale* asset, non *quando* |
| `panels.py` | 1068 | il registro: quale strategia usa quali indicatori, quali parametri, come si disegnano |
| `tuned_defaults.py` | 61 | **generato** da `scripts/tune_defaults.py`: valori di partenza misurati, per intervallo |
| `simulator.py` | 768 | la pagina Streamlit, due viste |
| `live_bot.py` | 472 | il bot headless che piazza ordini veri |

## Le funzioni

**`config.py`** — la classe `Param` e le costanti: `STRATEGIES`, `INTERVALS`, `AI_STRATEGY`,
`CONFLUENCE_STRATEGY`, `ROTATION_MODES`, i `CONF_*` dei votanti, le finestre degli indicatori.

**`market_data.py`** — `get_market_data`, `get_market_data_between_dates`, `download_market_data`,
`interval_to_minutes`.

**`indicators.py`** — `add_technical_indicator`, `latest_bands`, `calculate_latest_indicators`.
`_atr_ema` replica in numpy le formule di `ta` 0.11 riga per riga ed è ciò che rende
`simulate_candles` quaranta volte più veloce: **se si cambia, va riverificato contro `ta`**, perché
una divergenza silenziosa qui sposta ogni segnale.

**`indicators_extra.py`** — la sola `ExtraCache`: calcola una famiglia di indicatori alla volta e
la tiene, così un pannello che ne chiede tre non ricalcola le candele tre volte.

**`strategies.py`** — `simulate_candles` (il motore) e dodici strategie:
`buy_sell_limits_simulation`, `buy_sell_limits_close_simulation`,
`close_rsi_buy_sell_limits_simulation`, `atr_buy_sell_simulation`, `close_atr_buy_sell_simulation`,
`close_ema_crossover_simulation`, `close_bullish_ema_simulation`, `tp_sl_simulation`,
`green_candles_simulation`, `supertrend_simulation`, `trend_zone_simulation`,
`ai_model_simulation`, più gli aiuti `bullish_condition`, `bearish_condition`,
`identify_trend_zones`, `get_green_red_percentage`.
**Il menu ne raggiunge sette.** Delle cinque che restano fuori, quattro sono **uscite
misurandole** (`.claude/docs/ricerca-quant-ml.md` §2) e `buy_sell_limits_simulation` è rotta: legge
`MACD`, che resta commentata in `add_technical_indicator`, quindi solleva `KeyError` appena
chiamata. `close_rsi_buy_sell_limits_simulation` è invece **rientrata**, perché la ragione che
l'aveva esclusa vale a 15 minuti e non a scala giornaliera: una strategia esclusa su un intervallo
non è esclusa su tutti.

**`strategies_ls.py`** — `donchian_breakout`, `squeeze_breakout`, `trend_pullback`,
`ichimoku_trend`, `band_reversion_gated`, `atr_band_bounce`, `trend_zone`. Le prime tre voci del
menu vengono da qui (Ichimoku Trend, Squeeze Breakout, Donchian Breakout); `trend_pullback` e
`band_reversion_gated` sono fra le sette uscite misurando.

**`pnl.py`** — `simulate_trading_with_commisions` e
`simulate_trading_with_commisions_multiple_buy` per i segnali solo lungo,
`simulate_positions` per i cambi di posizione (leva e costo di mantenimento inclusi),
`drawdown` e `annualised` per leggerne l'esito.
Scompatta i segnali con `[:2]`: qualunque strategia può aggiungerci elementi dopo i due che il
motore usa, ed è così che la confluenza fa viaggiare la spiegazione col segnale.

**`mtf.py`** — la sola `align_to_lower`. È il punto in cui una strategia multi-timeframe imbroglia,
e la difesa è spostare lo stato del piano lungo di un periodo intero prima di leggerlo.

**`voters.py`** — `held_state`, `decayed_vote`.

**`confluence.py`** — `piani`, `ore_richieste`, `scala_fuori_misura`, `Par`, `Votante`, `registra`,
`selezione`, `votanti_predefiniti`, `Confluenza`, `valori_del_votante`, `stati_dei_votanti`,
`evaluate`. Aggiungere un votante è `registra(Votante(...))` e da lì si adattano da soli famiglie,
pesi, riquadri e griglia del banco; l'unico elenco da tenere allineato a mano sono le tracce del
riquadro *Voters* in `panels.INDICATORI`, e un test se ne accorge.
`Confluenza.perche_non_entra()` esiste perché **zero operazioni non è un risultato, è una domanda**.

**`portfolio.py`** — `Portafoglio`, `simulate_shared_capital`, `simulate_slots`, `curva_capitale`.
Riporta sempre le occasioni perse mentre il capitale era impegnato e la concentrazione: sopra 0,9
il paniere è finzione.

**`rotation.py`** — `load_universe`, `backtest`, `benchmarks`. Legge lo store locale e **non usa la
rete**: in produzione, dove non c'è disco persistente, lo dice invece di provare quindici scarichi.
Il riferimento da battere è l'universo a peso uguale, non BTC.

**`panels.py`** — i tipi `Traccia`, `Indicatore`, `Strategia` e le funzioni di interrogazione:
`indicatori_di`, `parametri_di`, `pannelli_di`, `pannelli_degli`, `gruppi_di`, `ancora_di`,
`valori_misurati`, `valori_del_piano`, `valori_predefiniti`, `confluenza_di`,
`diagnosi_confluenza`. La mappa è **verificata a mano**: uno scan statico delle colonne lette non
basta, perché almeno una strategia prende le medie con uno slice variabile che l'analisi
dell'albero sintattico non vede.

**`simulator.py`** — `available_strategies`, `trading_analysis` (vista *Single asset*),
`rotation_page` e `rotation_analysis` (vista *Cross-asset rotation*), `modello_di_sessione`,
`modelli_dingresso`, `universo_di_sessione`.

**`live_bot.py`** — `fetch_initial_candles`, `run_socket_with_reconnect`, `add_technical_indicator`,
`get_asset_balance`, `adjust_quantity`, `place_order`, `proceed_buy`, `proceed_sell`,
`print_user_and_wallet_info`. **Piazza ordini veri**, vuole `API_KEY`/`API_SECRET`, e non è un
servizio di compose di proposito: fa partire il ciclo `while True` all'import, senza `main()` e
senza gestione dei segnali. Prima di containerizzarlo serve quel refactor.

## Tre cose che non si vedono dal codice

**Le letture per riga sono array numpy estratti prima del ciclo**, non `df["Col"].iloc[i]`. Da lì
viene il grosso della velocità (il simulatore intero: 4295 ms → 125 ms). Mantenere lo stile.

**I valori di partenza sono centrali, non ottimi.** Sulla rotazione la correlazione fra resa in
stima e resa in verifica è −0,69: cercare il massimo in campione trasferisce peggio che prendere
una configurazione a caso. Chi cambia i default in «quelli che rendono di più nel grafico» sta
facendo esattamente l'errore misurato.

**Sotto l'ora nessuna misura di questo progetto ha mai trovato qualcosa che batta il possesso
passivo.** I default a 15m sono i migliori *fra quelli provati*, non buoni.

## Documenti

[`backtest-strategie.md`](../../../.claude/docs/backtest-strategie.md) ·
[`strategie-nuove.md`](../../../.claude/docs/strategie-nuove.md) ·
[`strategia-confluenza.md`](../../../.claude/docs/strategia-confluenza.md) ·
[`ricerca-quant-ml.md`](../../../.claude/docs/ricerca-quant-ml.md)
