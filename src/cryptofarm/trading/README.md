# `trading/` — the page, the strategies, the accounting

Everything that goes from candles to trades, plus the Streamlit page that shows it and the bot that
places real orders. The dependencies form a DAG and **there is no re-export facade**: whoever needs a
strategy imports it from the module that holds it.

```
market_data ─┐
indicators ──┼→ strategies ────┐
indicators_extra                ├→ simulator (the page)
mtf → voters → confluence ─────┤
strategies_ls → pnl ───────────┘
config, panels, tuned_defaults → simulator
                                  rotation, portfolio (the other two questions)
```

## The files

| file | lines | what it is for |
|---|---|---|
| `config.py` | 255 | the widgets' starting values and the menu lists. No logic |
| `market_data.py` | 304 | point downloads from Binance for the page (the public REST, not the dumps) |
| `indicators.py` | 212 | the classic indicators, plus the numpy ATR/EMA core |
| `indicators_extra.py` | 186 | ADX, Donchian, Bollinger/Keltner, StochRSI, OBV/MFI, Ichimoku, behind a cache |
| `strategies.py` | 848 | from candles with indicators to `(buy_signals, sell_signals)` — long only |
| `strategies_ls.py` | 537 | from candles to **position changes** `(when, price, +1\|0\|-1)` — long, flat, short |
| `pnl.py` | 277 | from signals to trades: commissions, leverage, carry cost |
| `mtf.py` | 87 | the alignment between intervals: it reads the **closed** long bar, never the current one |
| `voters.py` | 155 | from position changes to a vote per bar, with memory and decay |
| `confluence.py` | 1004 | the confluence strategy: six voters on four planes, dynamic threshold |
| `portfolio.py` | 308 | one capital across several assets: it opens on the first that speaks |
| `rotation.py` | 215 | cross-sectional rotation: it chooses *which* asset, not *when* |
| `panels.py` | 1068 | the registry: which strategy uses which indicators, which parameters, how they are drawn |
| `tuned_defaults.py` | 61 | **generated** by `scripts/tune_defaults.py`: measured starting values, per interval |
| `simulator.py` | 768 | the Streamlit page, two views |
| `live_bot.py` | 472 | the headless bot that places real orders |

## The functions

**`config.py`** — the `Param` class and the constants: `STRATEGIES`, `INTERVALS`, `AI_STRATEGY`,
`CONFLUENCE_STRATEGY`, `ROTATION_MODES`, the voters' `CONF_*`, the indicator windows, and
`SWING_TARGET_TEMPO` — the temporal smoothing the page draws the swing label with. That last one is a
**copied literal** of `ml/labeling.TIME_WEIGHT`, because this module imports nothing by design; a
test pins the two together, and without it the page would draw one label while the model trains on
another.

**`market_data.py`** — `get_market_data`, `get_market_data_between_dates`, `download_market_data`,
`interval_to_minutes`.

**`indicators.py`** — `add_technical_indicator`, `latest_bands`, `calculate_latest_indicators`.
`_atr_ema` replicates `ta` 0.11's formulas in numpy line by line and is what makes
`simulate_candles` forty times faster: **if it is changed, it must be reverified against `ta`**,
because a silent divergence here moves every signal.

**`indicators_extra.py`** — just `ExtraCache`: it computes one family of indicators at a time and
keeps it, so a panel asking for three does not recompute the candles three times.

**`strategies.py`** — `simulate_candles` (the engine) and twelve strategies:
`buy_sell_limits_simulation`, `buy_sell_limits_close_simulation`,
`close_rsi_buy_sell_limits_simulation`, `atr_buy_sell_simulation`, `close_atr_buy_sell_simulation`,
`close_ema_crossover_simulation`, `close_bullish_ema_simulation`, `tp_sl_simulation`,
`green_candles_simulation`, `supertrend_simulation`, `trend_zone_simulation`, `ai_model_simulation`,
plus the helpers `bullish_condition`, `bearish_condition`, `identify_trend_zones`,
`get_green_red_percentage`.
**The menu reaches seven of them.** Of the five left out, four **exited by being measured**
(`.claude/docs/ricerca-quant-ml.md` §2) and `buy_sell_limits_simulation` is broken: it reads `MACD`,
which stays commented out in `add_technical_indicator`, so it raises `KeyError` as soon as it is
called. `close_rsi_buy_sell_limits_simulation` has instead **come back**, because the reason that
excluded it holds at 15 minutes and not at daily scale: a strategy excluded on one interval is not
excluded on all of them.

**`strategies_ls.py`** — `donchian_breakout`, `squeeze_breakout`, `trend_pullback`,
`ichimoku_trend`, `band_reversion_gated`, `atr_band_bounce`, `trend_zone`. The menu's first three
entries come from here (Ichimoku Trend, Squeeze Breakout, Donchian Breakout); `trend_pullback` and
`band_reversion_gated` are among the seven that exited by being measured.

**`pnl.py`** — `simulate_trading_with_commisions` and
`simulate_trading_with_commisions_multiple_buy` for the long-only signals, `simulate_positions` for
the position changes (leverage and carry cost included), `drawdown` and `annualised` to read the
outcome.
It unpacks the signals with `[:2]`: any strategy can add elements after the two the engine uses, and
that is how the confluence makes the explanation travel with the signal.

**`mtf.py`** — just `align_to_lower`. It is the point where a multi-timeframe strategy cheats, and
the defence is to shift the long plane's state by a whole period before reading it.

**`voters.py`** — `held_state`, `decayed_vote`.

**`confluence.py`** — `piani`, `ore_richieste`, `scala_fuori_misura`, `Par`, `Votante`, `registra`,
`selezione`, `votanti_predefiniti`, `Confluenza`, `valori_del_votante`, `stati_dei_votanti`,
`evaluate`. Adding a voter is `registra(Votante(...))` and from there families, weights, panels and
the bench's grid adapt by themselves; the only list to keep aligned by hand is the *Voters* panel's
traces in `panels.INDICATORI`, and a test notices.
`Confluenza.perche_non_entra()` exists because **zero trades is not a result, it is a question**.

**`portfolio.py`** — `Portafoglio`, `simulate_shared_capital`, `simulate_slots`, `curva_capitale`.
It always reports the opportunities missed while the capital was committed, and the concentration:
above 0.9 the basket is fiction.

**`rotation.py`** — `load_universe`, `backtest`, `benchmarks`. It reads the local store and **does
not use the network**: in production, where there is no persistent disk, it says so instead of
attempting fifteen downloads. The reference to beat is the equal-weight universe, not BTC.

**`panels.py`** — the types `Traccia`, `Indicatore`, `Strategia` and the query functions:
`indicatori_di`, `parametri_di`, `pannelli_di`, `pannelli_degli`, `gruppi_di`, `ancora_di`,
`valori_misurati`, `valori_del_piano`, `valori_predefiniti`, `confluenza_di`,
`diagnosi_confluenza`. The map is **verified by hand**: a static scan of the columns read is not
enough, because at least one strategy takes the moving averages with a variable slice that syntax
tree analysis does not see.

**`simulator.py`** — `available_strategies`, `trading_analysis` (*Single asset* view),
`rotation_page` and `rotation_analysis` (*Cross-asset rotation* view), `modello_di_sessione`,
`modelli_dingresso`, `universo_di_sessione`.

**`live_bot.py`** — `fetch_initial_candles`, `run_socket_with_reconnect`, `add_technical_indicator`,
`get_asset_balance`, `adjust_quantity`, `place_order`, `proceed_buy`, `proceed_sell`,
`print_user_and_wallet_info`. **It places real orders**, it wants `API_KEY`/`API_SECRET`, and it is
not a compose service on purpose: it starts the `while True` loop at import, with no `main()` and no
signal handling. Before containerising it, that refactor is needed.

## Three things that are not visible from the code

**Per-row reads go through numpy arrays extracted before the loop**, not `df["Col"].iloc[i]`. That is
where most of the speed comes from (the whole simulator: 4295 ms → 125 ms). Keep the style.

**The starting values are central, not optimal.** On the rotation the correlation between in-sample
and out-of-sample return is −0.69: looking for the in-sample maximum transfers worse than picking a
configuration at random. Whoever changes the defaults to "the ones that return most in the chart" is
making exactly the measured mistake.

**Below the hour, no measurement in this project has ever found anything that beats passive
holding.** The 15m defaults are the best *among those tried*, not good.

## Documents

[`backtest-strategie.md`](../../../.claude/docs/backtest-strategie.md) ·
[`strategie-nuove.md`](../../../.claude/docs/strategie-nuove.md) ·
[`strategia-confluenza.md`](../../../.claude/docs/strategia-confluenza.md) ·
[`ricerca-quant-ml.md`](../../../.claude/docs/ricerca-quant-ml.md)
