"""Streamlit frontend per Simulator v2.

Layout (matrix):
- Global settings in alto (market, range, strategy).
- Tab per timeframe con controlli a sinistra e grafico a destra.
- Pulsante "Run simulation" per eseguire fetch e calcoli manualmente.

Flusso dati:
- init_session_state -> build_config -> validate_config
- prepare_timeframe_data -> compute_indicators -> strategia -> render
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Dict, List, Optional

import os
import streamlit as st

from simulator_v2_backend import (
    STRATEGIES,
    TIMEFRAME_MINUTES,
    build_timeframe_figure,
    compute_indicators,
    fetch_market_data_dates,
    fetch_market_data_hours,
    mtf_close_buy_sell_limits,
    simulate_trades,
    summarize_trades,
)


INDICATOR_STATE_KEYS = {
    "atr_bands": "ind_atr_bands",
    "rsi_short": "ind_rsi_short",
    "rsi_medium": "ind_rsi_medium",
    "rsi_long": "ind_rsi_long",
    "ema_short": "ind_ema_short",
    "ema_medium": "ind_ema_medium",
    "ema_long": "ind_ema_long",
    "kama": "ind_kama",
    "macd": "ind_macd",
}

INDICATOR_PARAM_KEYS = [
    "atr_window",
    "atr_multiplier",
    "rsi_short_window",
    "rsi_medium_window",
    "rsi_long_window",
    "ema_short_window",
    "ema_medium_window",
    "ema_long_window",
    "kama_window",
    "kama_pow1",
    "kama_pow2",
    "macd_short_window",
    "macd_long_window",
    "macd_signal_window",
]

STRATEGY_PARAM_KEYS = [
    "rsi_buy_limit",
    "rsi_sell_limit",
]

TIMEFRAME_PARAM_KEYS = INDICATOR_PARAM_KEYS + STRATEGY_PARAM_KEYS

TF_INDEXES = (1, 2, 3)

INDICATOR_TOGGLE_DEFAULTS = {
    "atr_bands": True,
    "rsi_short": True,
    "rsi_medium": True,
    "rsi_long": True,
    "ema_short": True,
    "ema_medium": True,
    "ema_long": True,
    "kama": True,
    "macd": False,
}

INDICATOR_PARAM_DEFAULTS = {
    "atr_window": 5,
    "atr_multiplier": 1.6,
    "rsi_short_window": 12,
    "rsi_medium_window": 24,
    "rsi_long_window": 36,
    "ema_short_window": 10,
    "ema_medium_window": 50,
    "ema_long_window": 200,
    "kama_window": 10,
    "kama_pow1": 2,
    "kama_pow2": 30,
    "macd_short_window": 12,
    "macd_long_window": 26,
    "macd_signal_window": 9,
}

TIMEFRAME_PARAM_DEFAULTS = {
    **INDICATOR_PARAM_DEFAULTS,
    "rsi_buy_limit": 25,
    "rsi_sell_limit": 75,
}


def indicator_state_key(tf_index: int, indicator: str) -> str:
    """Costruisce la chiave session_state per il toggle indicatore.

    Input:
        tf_index: indice timeframe (1..3).
        indicator: nome indicatore (es. rsi_short).
    Output:
        chiave session_state (es. tf1_ind_rsi_short).
    """
    return f"tf{tf_index}_{INDICATOR_STATE_KEYS[indicator]}"


def indicator_param_key(tf_index: int, param: str) -> str:
    """Costruisce la chiave session_state per un parametro numerico.

    Input:
        tf_index: indice timeframe (1..3).
        param: nome parametro (es. atr_window).
    Output:
        chiave session_state (es. tf2_atr_window).
    """
    return f"tf{tf_index}_{param}"


def init_session_state() -> None:
    """Inizializza i valori di default in session_state.

    Input:
        Nessuno (usa st.session_state).
    Output:
        Nessun valore di ritorno.
    Side effects:
        - Popola chiavi mancanti per asset, range, timeframe, indicatori.
        - Normalizza il nome strategia legacy.
    """
    today = datetime.utcnow().date()
    defaults: Dict[str, object] = {
        "asset_base": "BTC",
        "asset_quote": "USDT",
        "range_mode": "Hours",
        "range_hours": 24,
        "range_start_date": today - timedelta(days=7),
        "range_end_date": today,
        "range_start_time": time(0, 0),
        "range_end_time": time(23, 59),
        "tf1_enabled": True,
        "tf1_value": "15m",
        "tf2_enabled": True,
        "tf2_value": "4h",
        "tf3_enabled": False,
        "tf3_value": "1d",
        "strategy": "Custom",
        "wallet": 100.0,
        "fee_percent": 0.1,
        "stop_loss_percent": 99.0,
        "conditions_required": 1,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.get("strategy") == "Close Buy/Sell Limits":
        st.session_state["strategy"] = "Buy/Sell Limits"
    if st.session_state.get("strategy") not in STRATEGIES:
        st.session_state["strategy"] = "Custom"

    for tf_index in TF_INDEXES:
        for indicator, default_value in INDICATOR_TOGGLE_DEFAULTS.items():
            state_key = indicator_state_key(tf_index, indicator)
            if state_key not in st.session_state:
                st.session_state[state_key] = default_value
        for param_key, default_value in TIMEFRAME_PARAM_DEFAULTS.items():
            full_key = indicator_param_key(tf_index, param_key)
            if full_key not in st.session_state:
                st.session_state[full_key] = default_value

    if "last_run_signature" not in st.session_state:
        st.session_state["last_run_signature"] = None
    if "last_run_payload" not in st.session_state:
        st.session_state["last_run_payload"] = None
    if "last_run_at" not in st.session_state:
        st.session_state["last_run_at"] = None


def apply_strategy_defaults() -> None:
    """Applica i default della strategia ai parametri di UI.

    Input:
        Nessuno (legge st.session_state).
    Output:
        Nessun valore di ritorno.
    Side effects:
        - Aggiorna toggle indicatori per ogni timeframe.
        - Reimposta parametri per timeframe ai default strategia.
    """
    strategy_name = st.session_state.get("strategy", "Custom")
    strategy = STRATEGIES.get(strategy_name)
    if not strategy:
        return

    if strategy_name == "Custom":
        for tf_index in TF_INDEXES:
            for indicator, default_value in INDICATOR_TOGGLE_DEFAULTS.items():
                state_key = indicator_state_key(tf_index, indicator)
                st.session_state[state_key] = default_value

            for param_key, param_value in TIMEFRAME_PARAM_DEFAULTS.items():
                full_key = indicator_param_key(tf_index, param_key)
                st.session_state[full_key] = param_value
        st.session_state["conditions_required"] = 1
        return

    if "stop_loss_percent" in strategy.default_params:
        st.session_state["stop_loss_percent"] = strategy.default_params["stop_loss_percent"]
    if "conditions_required" in strategy.default_params:
        st.session_state["conditions_required"] = strategy.default_params["conditions_required"]

    for tf_index in TF_INDEXES:
        for indicator in INDICATOR_STATE_KEYS:
            state_key = indicator_state_key(tf_index, indicator)
            st.session_state[state_key] = indicator in strategy.default_indicators

        for param_key, param_value in TIMEFRAME_PARAM_DEFAULTS.items():
            full_key = indicator_param_key(tf_index, param_key)
            st.session_state[full_key] = param_value

        for param_key, param_value in strategy.default_params.items():
            if param_key in TIMEFRAME_PARAM_KEYS:
                full_key = indicator_param_key(tf_index, param_key)
                st.session_state[full_key] = param_value


def collect_timeframe_configs() -> List[Dict[str, object]]:
    """Raccoglie configurazioni per ogni slot timeframe.

    Output:
        Lista di dict con enabled/value/indicator_flags/timeframe_params.
    """
    configs = []
    for tf_index in TF_INDEXES:
        configs.append(
            {
                "index": tf_index,
                "enabled": bool(st.session_state.get(f"tf{tf_index}_enabled")),
                "value": st.session_state.get(f"tf{tf_index}_value"),
                "indicator_flags": collect_indicator_flags(tf_index),
                "timeframe_params": collect_timeframe_params(tf_index),
            }
        )
    return configs


def collect_indicator_flags(tf_index: int) -> Dict[str, bool]:
    """Ritorna i toggle indicatori per il timeframe specificato.

    Input:
        tf_index: indice timeframe (1..3).
    Output:
        dict indicatore -> bool.
    """
    return {
        indicator: bool(st.session_state[indicator_state_key(tf_index, indicator)])
        for indicator in INDICATOR_STATE_KEYS
    }


def collect_timeframe_params(tf_index: int) -> Dict[str, float]:
    """Ritorna i parametri numerici per il timeframe specificato.

    Input:
        tf_index: indice timeframe (1..3).
    Output:
        dict param -> float.
    """
    return {
        key: float(st.session_state[indicator_param_key(tf_index, key)])
        for key in TIMEFRAME_PARAM_KEYS
    }


def build_config() -> Dict[str, object]:
    """Costruisce un config normalizzato dalla UI.

    Output:
        dict con asset, range, timeframes, strategy e parametri globali.
    Notes:
        Concatena asset base/quote e converte in uppercase.
    """
    asset_base = str(st.session_state["asset_base"]).strip().replace(" ", "")
    asset_quote = str(st.session_state["asset_quote"]).strip().replace(" ", "")
    asset = f"{asset_base}{asset_quote}".upper()

    range_mode = st.session_state["range_mode"]
    hours = int(st.session_state["range_hours"])

    start_date = st.session_state["range_start_date"]
    end_date = st.session_state["range_end_date"]
    start_time = st.session_state["range_start_time"]
    end_time = st.session_state["range_end_time"]

    start_dt = datetime.combine(start_date, start_time)
    end_dt = datetime.combine(end_date, end_time)

    return {
        "asset": asset,
        "range_mode": range_mode,
        "hours": hours,
        "start_dt": start_dt,
        "end_dt": end_dt,
        "timeframes": collect_timeframe_configs(),
        "strategy": st.session_state["strategy"],
        "wallet": float(st.session_state["wallet"]),
        "fee_percent": float(st.session_state["fee_percent"]),
        "stop_loss_percent": float(st.session_state["stop_loss_percent"]),
        "conditions_required": int(st.session_state["conditions_required"]),
    }


def config_signature(config: Dict[str, object]) -> tuple:
    """Crea una signature immutabile per confrontare modifiche config."""
    timeframes_sig = []
    for tf in config["timeframes"]:
        timeframes_sig.append(
            (
                tf["index"],
                bool(tf["enabled"]),
                tf["value"],
                tuple(sorted(tf["indicator_flags"].items())),
                tuple(sorted(tf["timeframe_params"].items())),
            )
        )
    return (
        str(config["asset"]),
        str(config["range_mode"]),
        int(config["hours"]),
        config["start_dt"].isoformat(),
        config["end_dt"].isoformat(),
        tuple(timeframes_sig),
        str(config["strategy"]),
        float(config["wallet"]),
        float(config["fee_percent"]),
        float(config["stop_loss_percent"]),
        int(config["conditions_required"]),
    )


def validate_config(config: Dict[str, object]) -> List[str]:
    """Valida il config e ritorna una lista errori.

    Input:
        config: output di build_config.
    Output:
        lista stringhe (vuota se valido).
    """
    errors = []
    if not config["asset"]:
        errors.append("Asset is empty.")
    elif not str(config["asset"]).isalnum():
        errors.append("Asset must be alphanumeric (e.g., BTCUSDT).")

    enabled_timeframes = [tf for tf in config["timeframes"] if tf["enabled"]]
    if not enabled_timeframes:
        errors.append("Select at least one timeframe.")
    else:
        values = [tf["value"] for tf in enabled_timeframes]
        if len(values) != len(set(values)):
            errors.append("Timeframes must be unique.")

    if config["range_mode"] == "Hours":
        hours = int(config["hours"])
        if hours < 24:
            errors.append("Hours must be at least 24.")
    else:
        if config["start_dt"] >= config["end_dt"]:
            errors.append("Start date must be before end date.")

    if config["strategy"] == "Buy/Sell Limits":
        for tf_data in enabled_timeframes:
            flags = tf_data["indicator_flags"]
            if not flags.get("rsi_short"):
                continue
            buy_limit = float(tf_data["timeframe_params"]["rsi_buy_limit"])
            sell_limit = float(tf_data["timeframe_params"]["rsi_sell_limit"])
            if buy_limit >= sell_limit:
                errors.append(
                    f"TF{tf_data['index']} RSI Buy Limit must be lower than RSI Sell Limit."
                )

    return errors


def collect_warnings(config: Dict[str, object]) -> List[str]:
    """Raccoglie warning non bloccanti dalla config."""
    warnings = []
    enabled_timeframes = [tf for tf in config["timeframes"] if tf["enabled"]]
    for tf_data in enabled_timeframes:
        flags = tf_data["indicator_flags"]
        params = tf_data["timeframe_params"]
        if flags.get("ema_short") and flags.get("ema_long"):
            if params["ema_short_window"] >= params["ema_long_window"]:
                warnings.append(
                    f"TF{tf_data['index']} EMA Short window should be lower than EMA Long."
                )
    return warnings


def format_range_label(config: Dict[str, object]) -> str:
    """Etichetta human-readable per il range selezionato."""
    if config["range_mode"] == "Hours":
        return f"Last {config['hours']} hours"
    return f"{config['start_dt']} -> {config['end_dt']}"


def count_active_conditions(timeframes: List[Dict[str, object]]) -> int:
    """Conta quante condizioni strategia sono attive.

    Input:
        timeframes: lista config per timeframe.
    Output:
        conteggio condizioni attive (RSI, ATR, EMA cross).
    """
    total = 0
    for tf_data in timeframes:
        if not tf_data.get("enabled"):
            continue
        flags = tf_data["indicator_flags"]
        if flags.get("rsi_short"):
            total += 1
        if flags.get("atr_bands"):
            total += 1
        if flags.get("ema_short") and flags.get("ema_long"):
            total += 1
    return total


def build_conditions_summary_text(tf_index: int, strategy_name: str) -> str:
    """Costruisce il testo di riepilogo condizioni per il timeframe.

    Input:
        tf_index: indice timeframe (1..3).
        strategy_name: nome strategia corrente.
    Output:
        testo markdown per st.info.
    """
    tf_enabled = bool(st.session_state.get(f"tf{tf_index}_enabled"))
    if not tf_enabled:
        return "**Strategy Conditions Summary**\nTimeframe disabled."
    if strategy_name != "Buy/Sell Limits":
        return "**Strategy Conditions Summary**\nNo per-timeframe conditions for this strategy."

    flags = collect_indicator_flags(tf_index)
    params = collect_timeframe_params(tf_index)
    ema_cross_enabled = bool(flags.get("ema_short")) and bool(flags.get("ema_long"))
    active_conditions = sum(
        [
            bool(flags.get("rsi_short")),
            bool(flags.get("atr_bands")),
            ema_cross_enabled,
        ]
    )

    lines = [
        "**Strategy Conditions Summary**",
        f"Active conditions in this timeframe: {active_conditions}",
        f"- RSI Short: {'On' if flags.get('rsi_short') else 'Off'}",
        f"- ATR Bands: {'On' if flags.get('atr_bands') else 'Off'}",
        f"- EMA Cross (Short+Long): {'On' if ema_cross_enabled else 'Off'}",
        f"- Global required conditions: {st.session_state.get('conditions_required', 1)}",
    ]
    if flags.get("rsi_short"):
        lines.append(
            f"- RSI Limits: buy {params['rsi_buy_limit']:.0f} / sell {params['rsi_sell_limit']:.0f}"
        )
    return "\n".join(lines)


def prepare_timeframe_data(
    timeframe: str,
    config: Dict[str, object],
    indicator_flags: Dict[str, bool],
    timeframe_params: Dict[str, float],
) -> Dict[str, object]:
    """Fetch dati e calcola indicatori per un timeframe.

    Input:
        timeframe: intervallo Binance.
        config: config globale (asset, range, ecc).
        indicator_flags: indicatori abilitati.
        timeframe_params: parametri indicatori.
    Output:
        dict con df, actual_hours e error.
    Notes:
        In caso di eccezioni ritorna error string.
    """
    try:
        if config["range_mode"] == "Hours":
            df, actual_hours = fetch_market_data_hours(
                asset=config["asset"],
                interval=timeframe,
                hours=config["hours"],
            )
        else:
            df, actual_hours = fetch_market_data_dates(
                asset=config["asset"],
                interval=timeframe,
                start_dt=config["start_dt"],
                end_dt=config["end_dt"],
            )
    except Exception as exc:
        message = str(exc)
        if "Web Page Blocked" in message or "Invalid JSON" in message:
            message = (
                "Network block detected (HTML response). Check proxy or allowlist api.binance.com."
            )
        elif "CERTIFICATE_VERIFY_FAILED" in message:
            message = "SSL verify failed. Set BINANCE_SSL_VERIFY=0 or install CA."
        return {"error": f"Data fetch failed: {message}"}

    if df.empty:
        return {"error": "No data returned for this timeframe."}

    try:
        if not any(indicator_flags.values()):
            df_indicators = df
        else:
            df_indicators = compute_indicators(df, indicator_flags, timeframe_params)
    except Exception as exc:
        return {"error": f"Indicator computation failed: {exc}"}

    return {
        "df": df_indicators,
        "actual_hours": actual_hours,
        "error": None,
    }


def align_signals_to_timeframe(signals: List[tuple], df: "pd.DataFrame") -> List[tuple]:
    """Allinea i segnali al candle precedente nel timeframe target.

    Input:
        signals: lista (timestamp, price) su base timeframe.
        df: DataFrame target con index temporale.
    Output:
        segnali riposizionati sul timeframe target.
    """
    if not signals:
        return []

    aligned: List[tuple] = []
    index = df.index

    for ts, _price in signals:
        pos = index.searchsorted(ts, side="right") - 1
        if pos < 0:
            continue
        aligned.append((index[pos], float(df["Close"].iloc[pos])))

    return aligned


def render_global_settings() -> None:
    """Renderizza i controlli globali (market, range, strategy)."""
    st.subheader("Global Settings")
    market_col, range_col, strategy_col = st.columns([1, 1.2, 1], gap="large")

    with market_col:
        st.markdown("**Market**")
        st.text_input("Asset", key="asset_base")
        st.text_input("Quote", key="asset_quote")

    with range_col:
        st.markdown("**Range**")
        st.radio("Range mode", ["Hours", "Dates"], key="range_mode")
        if st.session_state["range_mode"] == "Hours":
            st.number_input(
                "Total hours (multiple of 24)",
                min_value=24,
                step=24,
                key="range_hours",
            )
        else:
            st.date_input("Start date", key="range_start_date")
            st.time_input("Start time", key="range_start_time")
            st.date_input("End date", key="range_end_date")
            st.time_input("End time", key="range_end_time")

    with strategy_col:
        st.markdown("**Strategy**")
        st.selectbox(
            "Strategy",
            options=list(STRATEGIES.keys()),
            key="strategy",
            on_change=apply_strategy_defaults,
        )
        selected_strategy = STRATEGIES[st.session_state["strategy"]]
        if selected_strategy.description:
            st.caption(selected_strategy.description)

        if selected_strategy.name == "Buy/Sell Limits":
            total_conditions = count_active_conditions(collect_timeframe_configs())
            max_value = max(1, total_conditions)
            if st.session_state["conditions_required"] > max_value:
                st.session_state["conditions_required"] = max_value
            st.number_input(
                "Conditions Required",
                min_value=1,
                max_value=max_value,
                step=1,
                key="conditions_required",
                disabled=total_conditions == 0,
            )
            st.caption(f"Active conditions: {total_conditions}")
            if total_conditions == 0:
                st.caption("Enable RSI/ATR or EMA Short+Long to activate conditions.")

        st.number_input("Wallet", min_value=0.0, step=10.0, key="wallet")
        st.number_input("Fee %", min_value=0.0, max_value=5.0, step=0.01, key="fee_percent")

        stop_loss_disabled = (
            "stop_loss_percent" not in selected_strategy.default_params
            and selected_strategy.name != "Custom"
        )
        st.number_input(
            "Stop Loss %",
            min_value=0.1,
            max_value=100.0,
            step=0.1,
            key="stop_loss_percent",
            disabled=stop_loss_disabled,
        )


def render_timeframe_panel(
    df_indicators: "pd.DataFrame",
    actual_hours: float,
    asset: str,
    timeframe: str,
    indicator_flags: Dict[str, bool],
    buy_signals: List[tuple],
    sell_signals: List[tuple],
    rsi_buy_limit: Optional[float],
    rsi_sell_limit: Optional[float],
    trades_df,
    summary: Dict[str, float],
    execution_note: str = "",
) -> None:
    """Renderizza metriche, grafico e tabella trade per un timeframe.

    Input:
        df_indicators: DataFrame con OHLCV + indicatori.
        actual_hours: ore effettive caricate.
        indicator_flags: toggle indicatori per il grafico.
        buy_signals/sell_signals: marker buy/sell.
        rsi_buy_limit/rsi_sell_limit: linee RSI opzionali.
        trades_df/summary: output simulazione trade.
        execution_note: nota addizionale per il pannello.
    Output:
        Nessun valore di ritorno (scrive in UI).
    """
    metrics = st.columns(4)
    metrics[0].metric("Total Profit", f"{summary['total_profit']:.2f}")
    metrics[1].metric("Win Rate", f"{summary['win_rate']:.2f}%")
    metrics[2].metric("Trades", f"{summary['num_trades']}")
    metrics[3].metric("Final Wallet", f"{summary['final_wallet']:.2f}")

    st.caption(f"Actual hours loaded: {actual_hours:.2f}")
    if execution_note:
        st.caption(execution_note)

    fig = build_timeframe_figure(
        df_indicators,
        asset=asset,
        interval=timeframe,
        indicator_flags=indicator_flags,
        buy_signals=buy_signals,
        sell_signals=sell_signals,
        rsi_buy_limit=rsi_buy_limit,
        rsi_sell_limit=rsi_sell_limit,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Trades", expanded=False):
        st.dataframe(trades_df, use_container_width=True)


def render_indicator_controls(tf_index: int) -> None:
    """Renderizza i controlli indicatori per un timeframe.

    Input:
        tf_index: indice timeframe (1..3).
    Output:
        Nessun valore di ritorno (scrive in UI).
    Notes:
        I limiti RSI sono visibili solo con strategia Buy/Sell Limits
        e RSI Short attivo.
    """
    tf_enabled = bool(st.session_state.get(f"tf{tf_index}_enabled"))

    atr_enabled = st.checkbox(
        "ATR Bands",
        key=indicator_state_key(tf_index, "atr_bands"),
        disabled=not tf_enabled,
    )
    atr_cols = st.columns(2)
    atr_cols[0].number_input(
        "ATR Window",
        min_value=2,
        max_value=500,
        step=1,
        key=indicator_param_key(tf_index, "atr_window"),
        disabled=not tf_enabled or not atr_enabled,
    )
    atr_cols[1].number_input(
        "ATR Multiplier",
        min_value=0.1,
        max_value=50.0,
        step=0.1,
        key=indicator_param_key(tf_index, "atr_multiplier"),
        disabled=not tf_enabled or not atr_enabled,
    )

    st.divider()

    rsi_short = st.checkbox(
        "RSI Short",
        key=indicator_state_key(tf_index, "rsi_short"),
        disabled=not tf_enabled,
    )
    st.number_input(
        "RSI Short Window",
        min_value=2,
        max_value=500,
        step=1,
        key=indicator_param_key(tf_index, "rsi_short_window"),
        disabled=not tf_enabled or not rsi_short,
    )
    rsi_medium = st.checkbox(
        "RSI Medium",
        key=indicator_state_key(tf_index, "rsi_medium"),
        disabled=not tf_enabled,
    )
    st.number_input(
        "RSI Medium Window",
        min_value=2,
        max_value=500,
        step=1,
        key=indicator_param_key(tf_index, "rsi_medium_window"),
        disabled=not tf_enabled or not rsi_medium,
    )
    rsi_long = st.checkbox(
        "RSI Long",
        key=indicator_state_key(tf_index, "rsi_long"),
        disabled=not tf_enabled,
    )
    st.number_input(
        "RSI Long Window",
        min_value=2,
        max_value=500,
        step=1,
        key=indicator_param_key(tf_index, "rsi_long_window"),
        disabled=not tf_enabled or not rsi_long,
    )

    strategy_name = st.session_state.get("strategy", "Custom")
    show_rsi_limits = strategy_name == "Buy/Sell Limits" and tf_enabled and rsi_short
    if show_rsi_limits:
        rsi_limit_cols = st.columns(2)
        rsi_limit_cols[0].number_input(
            "RSI Buy Limit",
            min_value=0,
            max_value=100,
            step=1,
            key=indicator_param_key(tf_index, "rsi_buy_limit"),
        )
        rsi_limit_cols[1].number_input(
            "RSI Sell Limit",
            min_value=0,
            max_value=100,
            step=1,
            key=indicator_param_key(tf_index, "rsi_sell_limit"),
        )

    st.divider()

    ema_short = st.checkbox(
        "EMA Short",
        key=indicator_state_key(tf_index, "ema_short"),
        disabled=not tf_enabled,
    )
    st.number_input(
        "EMA Short Window",
        min_value=1,
        max_value=500,
        step=1,
        key=indicator_param_key(tf_index, "ema_short_window"),
        disabled=not tf_enabled or not ema_short,
    )
    ema_medium = st.checkbox(
        "EMA Medium",
        key=indicator_state_key(tf_index, "ema_medium"),
        disabled=not tf_enabled,
    )
    st.number_input(
        "EMA Medium Window",
        min_value=1,
        max_value=500,
        step=1,
        key=indicator_param_key(tf_index, "ema_medium_window"),
        disabled=not tf_enabled or not ema_medium,
    )
    ema_long = st.checkbox(
        "EMA Long",
        key=indicator_state_key(tf_index, "ema_long"),
        disabled=not tf_enabled,
    )
    st.number_input(
        "EMA Long Window",
        min_value=1,
        max_value=500,
        step=1,
        key=indicator_param_key(tf_index, "ema_long_window"),
        disabled=not tf_enabled or not ema_long,
    )

    st.divider()

    kama_enabled = st.checkbox(
        "KAMA",
        key=indicator_state_key(tf_index, "kama"),
        disabled=not tf_enabled,
    )
    kama_cols = st.columns(3)
    kama_cols[0].number_input(
        "KAMA Window",
        min_value=2,
        max_value=500,
        step=1,
        key=indicator_param_key(tf_index, "kama_window"),
        disabled=not tf_enabled or not kama_enabled,
    )
    kama_cols[1].number_input(
        "KAMA Pow1",
        min_value=1,
        max_value=200,
        step=1,
        key=indicator_param_key(tf_index, "kama_pow1"),
        disabled=not tf_enabled or not kama_enabled,
    )
    kama_cols[2].number_input(
        "KAMA Pow2",
        min_value=1,
        max_value=2000,
        step=1,
        key=indicator_param_key(tf_index, "kama_pow2"),
        disabled=not tf_enabled or not kama_enabled,
    )

    st.divider()

    macd_enabled = st.checkbox(
        "MACD",
        key=indicator_state_key(tf_index, "macd"),
        disabled=not tf_enabled,
    )
    macd_cols = st.columns(3)
    macd_cols[0].number_input(
        "MACD Short",
        min_value=2,
        max_value=200,
        step=1,
        key=indicator_param_key(tf_index, "macd_short_window"),
        disabled=not tf_enabled or not macd_enabled,
    )
    macd_cols[1].number_input(
        "MACD Long",
        min_value=2,
        max_value=300,
        step=1,
        key=indicator_param_key(tf_index, "macd_long_window"),
        disabled=not tf_enabled or not macd_enabled,
    )
    macd_cols[2].number_input(
        "MACD Signal",
        min_value=1,
        max_value=200,
        step=1,
        key=indicator_param_key(tf_index, "macd_signal_window"),
        disabled=not tf_enabled or not macd_enabled,
    )


def render_timeframe_matrix_controls(tf_index: int) -> None:
    """Renderizza i controlli del timeframe in layout matrix."""
    tf_enabled = bool(st.session_state.get(f"tf{tf_index}_enabled"))
    st.markdown(f"**TF{tf_index} Settings**")
    st.checkbox("Enabled", key=f"tf{tf_index}_enabled")
    st.selectbox(
        "Interval",
        options=list(TIMEFRAME_MINUTES.keys()),
        key=f"tf{tf_index}_value",
        disabled=not tf_enabled,
    )
    st.caption("Disabled timeframes are ignored in fetch, indicators and strategy.")

    st.divider()
    with st.expander("Indicators & Parameters", expanded=True):
        render_indicator_controls(tf_index)

    with st.expander("Strategy Conditions Summary", expanded=True):
        summary_text = build_conditions_summary_text(tf_index, st.session_state.get("strategy", "Custom"))
        st.info(summary_text)


def compute_single_timeframe_payload(
    config: Dict[str, object],
    enabled_timeframes: List[Dict[str, object]],
    strategy,
    strategy_params: Dict[str, float],
) -> Dict[str, object]:
    """Calcola dati e segnali per strategie single-timeframe."""
    results = []
    for timeframe_config in enabled_timeframes:
        with st.spinner(f"Fetching {config['asset']} {timeframe_config['value']} data..."):
            data = prepare_timeframe_data(
                timeframe_config["value"],
                config=config,
                indicator_flags=timeframe_config["indicator_flags"],
                timeframe_params=timeframe_config["timeframe_params"],
            )
        if data["error"]:
            return {
                "error": f"TF{timeframe_config['index']} {timeframe_config['value']}: {data['error']}"
            }

        df_indicators = data["df"]
        actual_hours = data["actual_hours"]

        missing = [
            indicator
            for indicator in strategy.required_indicators
            if not timeframe_config["indicator_flags"].get(indicator)
        ]
        if missing:
            warning = (
                f"Strategy requires indicators not enabled: {', '.join(missing)}. Signals skipped."
            )
            buy_signals: List[tuple] = []
            sell_signals: List[tuple] = []
        elif strategy.signal_func:
            signal_params = {**timeframe_config["timeframe_params"], **strategy_params}
            try:
                buy_signals, sell_signals = strategy.signal_func(df_indicators, signal_params)
                warning = ""
            except Exception as exc:
                return {"error": f"Strategy computation failed: {exc}"}
        else:
            buy_signals = []
            sell_signals = []
            warning = ""

        trades_df = simulate_trades(
            buy_signals,
            sell_signals,
            initial_wallet=config["wallet"],
            fee_percent=config["fee_percent"],
        )
        summary = summarize_trades(trades_df, initial_wallet=config["wallet"])
        results.append(
            {
                **timeframe_config,
                "df": df_indicators,
                "actual_hours": actual_hours,
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "trades_df": trades_df,
                "summary": summary,
                "warning": warning,
            }
        )

    return {
        "error": None,
        "mode": "single",
        "timeframe_results": results,
        "timeframe_map": {item["index"]: item for item in results},
    }


def compute_mtf_payload(
    config: Dict[str, object],
    enabled_timeframes: List[Dict[str, object]],
) -> Dict[str, object]:
    """Calcola dati e segnali per la strategia multi-timeframe."""
    timeframe_data = []
    for timeframe_config in enabled_timeframes:
        with st.spinner(f"Fetching {config['asset']} {timeframe_config['value']} data..."):
            data = prepare_timeframe_data(
                timeframe_config["value"],
                config=config,
                indicator_flags=timeframe_config["indicator_flags"],
                timeframe_params=timeframe_config["timeframe_params"],
            )
        if data["error"]:
            return {"error": f"TF{timeframe_config['index']} {timeframe_config['value']}: {data['error']}"}

        timeframe_data.append(
            {
                **timeframe_config,
                "df": data["df"],
                "actual_hours": data["actual_hours"],
            }
        )

    base_config = min(
        timeframe_data,
        key=lambda item: TIMEFRAME_MINUTES[item["value"]],
    )
    frames = []
    inactive_frames = []
    any_active = False
    total_conditions = count_active_conditions(enabled_timeframes)
    for tf_data in timeframe_data:
        use_rsi = bool(tf_data["indicator_flags"].get("rsi_short"))
        use_atr = bool(tf_data["indicator_flags"].get("atr_bands"))
        use_ema = bool(tf_data["indicator_flags"].get("ema_short")) and bool(
            tf_data["indicator_flags"].get("ema_long")
        )
        if not use_rsi and not use_atr and not use_ema:
            inactive_frames.append(f"TF{tf_data['index']} {tf_data['value']}")
        else:
            any_active = True

        frames.append(
            {
                "timeframe": tf_data["value"],
                "df": tf_data["df"],
                "use_rsi": use_rsi,
                "use_atr": use_atr,
                "use_ema": use_ema,
                "rsi_buy_limit": tf_data["timeframe_params"]["rsi_buy_limit"],
                "rsi_sell_limit": tf_data["timeframe_params"]["rsi_sell_limit"],
            }
        )

    if inactive_frames:
        st.warning(
            "Strategy ignores timeframes without ATR/RSI or EMA Short+Long enabled: "
            + ", ".join(inactive_frames)
        )

    if not any_active:
        buy_signals = []
        sell_signals = []
    else:
        try:
            buy_signals, sell_signals = mtf_close_buy_sell_limits(
                frames=frames,
                base_timeframe=base_config["value"],
                conditions_required=config["conditions_required"],
            )
        except Exception as exc:
            return {"error": f"Strategy computation failed: {exc}"}

    trades_df = simulate_trades(
        buy_signals,
        sell_signals,
        initial_wallet=config["wallet"],
        fee_percent=config["fee_percent"],
    )
    summary = summarize_trades(trades_df, initial_wallet=config["wallet"])
    if total_conditions > 0:
        required = min(config["conditions_required"], total_conditions)
        conditions_note = f"Conditions required: {required}/{total_conditions}."
    else:
        conditions_note = ""
    if not any_active:
        execution_note = "Signals skipped due to no active indicators."
    else:
        execution_note = f"Signals computed on base timeframe: {base_config['value']}."
    if conditions_note:
        execution_note = f"{execution_note} {conditions_note}"

    return {
        "error": None,
        "mode": "mtf",
        "timeframe_data": timeframe_data,
        "timeframe_map": {tf["index"]: tf for tf in timeframe_data},
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "trades_df": trades_df,
        "summary": summary,
        "execution_note": execution_note,
    }


def render_timeframe(
    timeframe: str,
    config: Dict[str, object],
    strategy,
    indicator_flags: Dict[str, bool],
    timeframe_params: Dict[str, float],
    strategy_params: Dict[str, float],
) -> None:
    """Gestisce pipeline single-timeframe: fetch -> indicatori -> segnali -> render.

    Input:
        timeframe: intervallo Binance.
        config: config globale.
        strategy: StrategySpec selezionata.
        indicator_flags/timeframe_params: parametri per indicatori.
        strategy_params: parametri strategia globali.
    Output:
        Nessun valore di ritorno (scrive in UI).
    """
    with st.spinner(f"Fetching {config['asset']} {timeframe} data..."):
        data = prepare_timeframe_data(timeframe, config, indicator_flags, timeframe_params)
    if data["error"]:
        st.warning(data["error"])
        return

    df_indicators = data["df"]
    actual_hours = data["actual_hours"]

    missing = [
        indicator for indicator in strategy.required_indicators if not indicator_flags.get(indicator)
    ]
    if missing:
        st.warning(
            f"Strategy requires indicators not enabled: {', '.join(missing)}. Signals skipped."
        )
        buy_signals = []
        sell_signals = []
    elif strategy.signal_func:
        signal_params = {**timeframe_params, **strategy_params}
        try:
            buy_signals, sell_signals = strategy.signal_func(df_indicators, signal_params)
        except Exception as exc:
            st.error(f"Strategy computation failed: {exc}")
            buy_signals = []
            sell_signals = []
    else:
        buy_signals = []
        sell_signals = []

    trades_df = simulate_trades(
        buy_signals,
        sell_signals,
        initial_wallet=config["wallet"],
        fee_percent=config["fee_percent"],
    )
    summary = summarize_trades(trades_df, initial_wallet=config["wallet"])
    show_limits = strategy.name == "Buy/Sell Limits" and indicator_flags.get("rsi_short")
    rsi_buy_limit = timeframe_params["rsi_buy_limit"] if show_limits else None
    rsi_sell_limit = timeframe_params["rsi_sell_limit"] if show_limits else None
    render_timeframe_panel(
        df_indicators=df_indicators,
        actual_hours=actual_hours,
        asset=config["asset"],
        timeframe=timeframe,
        indicator_flags=indicator_flags,
        buy_signals=buy_signals,
        sell_signals=sell_signals,
        rsi_buy_limit=rsi_buy_limit,
        rsi_sell_limit=rsi_sell_limit,
        trades_df=trades_df,
        summary=summary,
    )


def main() -> None:
    """Entrypoint Streamlit per la UI del simulatore.

    Flow:
        - Imposta layout, inizializza session_state.
        - Renderizza global settings e matrix per timeframe.
        - Costruisce config e valida input.
        - Esegue calcoli solo su richiesta ("Run simulation").
    """
    st.set_page_config(
        page_title="CryptoFarm Simulator v2",
        page_icon=":bar_chart:",
        layout="wide",
    )

    init_session_state()
    if str(os.getenv("BINANCE_SSL_VERIFY", "")).strip().lower() in {"0", "false", "no"}:
        st.warning("BINANCE_SSL_VERIFY is disabled. SSL verification is off.")

    st.title("CryptoFarm Simulator v2")
    st.caption(
        "Configure global settings, then tune each timeframe in the tabs below."
    )
    st.info(
        "Quick start:\n"
        "1) Pick asset and time range.\n"
        "2) Enable the timeframes you want to inspect.\n"
        "3) Tune indicators per timeframe.\n"
        "4) Press Run simulation to refresh charts."
    )

    render_global_settings()

    config = build_config()
    signature = config_signature(config)
    errors = validate_config(config)
    if errors:
        for err in errors:
            st.error(err)
    for warning in collect_warnings(config):
        st.warning(warning)

    if errors:
        st.info("Fix configuration errors before running the simulation.")

    enabled_timeframes = [tf for tf in config["timeframes"] if tf["enabled"]]
    timeframe_labels = [f"TF{tf['index']} {tf['value']}" for tf in enabled_timeframes]
    if enabled_timeframes:
        st.caption(
            f"Asset: {config['asset']} | Range: {format_range_label(config)} | "
            f"Timeframes: {', '.join(timeframe_labels)}"
        )
    else:
        st.caption(f"Asset: {config['asset']} | Range: {format_range_label(config)}")

    # Adjust hours to a multiple of 24 if needed
    if config["range_mode"] == "Hours" and config["hours"] % 24 != 0:
        adjusted = config["hours"] - (config["hours"] % 24)
        st.warning(f"Hours adjusted to {adjusted} to keep a multiple of 24.")
        config["hours"] = adjusted

    run_col, status_col = st.columns([0.2, 0.8], gap="large")
    with run_col:
        run_clicked = st.button("Run simulation", type="primary", disabled=bool(errors))
    with status_col:
        last_run_at = st.session_state.get("last_run_at")
        if last_run_at:
            st.caption(f"Last run: {last_run_at}")
        if st.session_state.get("last_run_signature") and signature != st.session_state["last_run_signature"]:
            st.warning("Changes detected. Results shown are from the last run.")

    strategy = STRATEGIES[config["strategy"]]
    strategy_params = {
        "stop_loss_percent": config["stop_loss_percent"],
    }

    if run_clicked and not errors:
        if strategy.multi_timeframe:
            payload = compute_mtf_payload(config, enabled_timeframes)
        else:
            payload = compute_single_timeframe_payload(
                config=config,
                enabled_timeframes=enabled_timeframes,
                strategy=strategy,
                strategy_params=strategy_params,
            )
        if payload.get("error"):
            st.error(payload["error"])
        else:
            payload["strategy_name"] = strategy.name
            payload["strategy_multi"] = strategy.multi_timeframe
            st.session_state["last_run_signature"] = signature
            st.session_state["last_run_payload"] = payload
            st.session_state["last_run_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    payload = st.session_state.get("last_run_payload")
    payload_strategy_name = payload.get("strategy_name") if payload else strategy.name
    if not payload_strategy_name:
        payload_strategy_name = strategy.name
    payload_strategy_multi = payload.get("strategy_multi") if payload else strategy.multi_timeframe
    if payload_strategy_multi is None:
        payload_strategy_multi = strategy.multi_timeframe

    st.divider()
    st.subheader("Timeframes Matrix")
    tab_labels = []
    for tf_index in TF_INDEXES:
        tf_value = st.session_state.get(f"tf{tf_index}_value")
        tf_status = "ON" if st.session_state.get(f"tf{tf_index}_enabled") else "OFF"
        tab_labels.append(f"TF{tf_index} {tf_status} ({tf_value})")

    tabs = st.tabs(tab_labels)
    for tf_index, tab in zip(TF_INDEXES, tabs):
        with tab:
            left, right = st.columns([0.35, 0.65], gap="large")
            with left:
                render_timeframe_matrix_controls(tf_index)
            with right:
                if errors:
                    st.info("Fix configuration errors before running the simulation.")
                    continue

                tf_enabled = bool(st.session_state.get(f"tf{tf_index}_enabled"))
                if not tf_enabled:
                    st.info("Timeframe disabled. Enable it to load data and charts.")
                    continue

                if not payload:
                    st.info("Press Run simulation to load data and charts.")
                    continue

                if payload_strategy_multi:
                    tf_data = payload["timeframe_map"].get(tf_index)
                    if not tf_data:
                        st.info("Timeframe data not available.")
                        continue
                    plot_buy = align_signals_to_timeframe(payload["buy_signals"], tf_data["df"])
                    plot_sell = align_signals_to_timeframe(payload["sell_signals"], tf_data["df"])
                    show_limits = (
                        payload_strategy_name == "Buy/Sell Limits"
                        and tf_data["indicator_flags"].get("rsi_short")
                    )
                    rsi_buy_limit = (
                        tf_data["timeframe_params"]["rsi_buy_limit"] if show_limits else None
                    )
                    rsi_sell_limit = (
                        tf_data["timeframe_params"]["rsi_sell_limit"] if show_limits else None
                    )
                    render_timeframe_panel(
                        df_indicators=tf_data["df"],
                        actual_hours=tf_data["actual_hours"],
                        asset=config["asset"],
                        timeframe=tf_data["value"],
                        indicator_flags=tf_data["indicator_flags"],
                        buy_signals=plot_buy,
                        sell_signals=plot_sell,
                        rsi_buy_limit=rsi_buy_limit,
                        rsi_sell_limit=rsi_sell_limit,
                        trades_df=payload["trades_df"],
                        summary=payload["summary"],
                        execution_note=payload["execution_note"],
                    )
                else:
                    tf_data = payload["timeframe_map"].get(tf_index)
                    if not tf_data:
                        st.info("Timeframe data not available.")
                        continue
                    if tf_data.get("warning"):
                        st.warning(tf_data["warning"])
                    show_limits = (
                        payload_strategy_name == "Buy/Sell Limits"
                        and tf_data["indicator_flags"].get("rsi_short")
                    )
                    rsi_buy_limit = (
                        tf_data["timeframe_params"]["rsi_buy_limit"] if show_limits else None
                    )
                    rsi_sell_limit = (
                        tf_data["timeframe_params"]["rsi_sell_limit"] if show_limits else None
                    )
                    render_timeframe_panel(
                        df_indicators=tf_data["df"],
                        actual_hours=tf_data["actual_hours"],
                        asset=config["asset"],
                        timeframe=tf_data["value"],
                        indicator_flags=tf_data["indicator_flags"],
                        buy_signals=tf_data["buy_signals"],
                        sell_signals=tf_data["sell_signals"],
                        rsi_buy_limit=rsi_buy_limit,
                        rsi_sell_limit=rsi_sell_limit,
                        trades_df=tf_data["trades_df"],
                        summary=tf_data["summary"],
                    )


if __name__ == "__main__":
    main()
