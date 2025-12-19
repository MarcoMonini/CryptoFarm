from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Dict, List

import streamlit as st

from simulator_v2_backend import (
    STRATEGIES,
    TIMEFRAME_MINUTES,
    build_timeframe_figure,
    compute_indicators,
    fetch_market_data_dates,
    fetch_market_data_hours,
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


def init_session_state() -> None:
    """Initialize all widget defaults in session_state."""
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
        "ind_atr_bands": True,
        "ind_rsi_short": True,
        "ind_rsi_medium": True,
        "ind_rsi_long": True,
        "ind_ema_short": True,
        "ind_ema_medium": True,
        "ind_ema_long": True,
        "ind_kama": True,
        "ind_macd": False,
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
        "strategy": "Custom",
        "wallet": 100.0,
        "fee_percent": 0.1,
        "stop_loss_percent": 99.0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_strategy_defaults() -> None:
    """Apply strategy defaults to indicator toggles and parameters."""
    strategy_name = st.session_state.get("strategy", "Custom")
    strategy = STRATEGIES.get(strategy_name)
    if not strategy or strategy_name == "Custom":
        return

    for indicator, state_key in INDICATOR_STATE_KEYS.items():
        st.session_state[state_key] = indicator in strategy.required_indicators

    for param_key, param_value in strategy.default_params.items():
        st.session_state[param_key] = param_value


def collect_timeframes() -> List[str]:
    """Collect up to three timeframes, removing duplicates while keeping order."""
    selected = []
    for idx in range(1, 4):
        if st.session_state.get(f"tf{idx}_enabled"):
            selected.append(st.session_state.get(f"tf{idx}_value"))

    # Preserve order while removing duplicates
    seen = set()
    unique = []
    for tf in selected:
        if tf in seen:
            continue
        seen.add(tf)
        unique.append(tf)
    return unique


def collect_indicator_flags() -> Dict[str, bool]:
    """Return the indicator toggle map for the current UI state."""
    return {
        indicator: bool(st.session_state[state_key])
        for indicator, state_key in INDICATOR_STATE_KEYS.items()
    }


def collect_indicator_params() -> Dict[str, float]:
    """Return numeric indicator parameters for the current UI state."""
    return {key: float(st.session_state[key]) for key in INDICATOR_PARAM_KEYS}


def build_config() -> Dict[str, object]:
    """Build a normalized config dict from the current UI state."""
    asset_base = str(st.session_state["asset_base"]).strip()
    asset_quote = str(st.session_state["asset_quote"]).strip()
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
        "timeframes": collect_timeframes(),
        "indicator_flags": collect_indicator_flags(),
        "indicator_params": collect_indicator_params(),
        "strategy": st.session_state["strategy"],
        "wallet": float(st.session_state["wallet"]),
        "fee_percent": float(st.session_state["fee_percent"]),
        "stop_loss_percent": float(st.session_state["stop_loss_percent"]),
    }


def validate_config(config: Dict[str, object]) -> List[str]:
    """Validate the config and return a list of errors."""
    errors = []
    if not config["asset"]:
        errors.append("Asset is empty.")

    if not config["timeframes"]:
        errors.append("Select at least one timeframe.")

    if config["range_mode"] == "Hours":
        hours = int(config["hours"])
        if hours < 24:
            errors.append("Hours must be at least 24.")
    else:
        if config["start_dt"] >= config["end_dt"]:
            errors.append("Start date must be before end date.")

    return errors


def format_range_label(config: Dict[str, object]) -> str:
    """Human readable label for the selected time range."""
    if config["range_mode"] == "Hours":
        return f"Last {config['hours']} hours"
    return f"{config['start_dt']} -> {config['end_dt']}"


def render_timeframe(
    timeframe: str,
    config: Dict[str, object],
    strategy,
    indicator_flags: Dict[str, bool],
    indicator_params: Dict[str, float],
    strategy_params: Dict[str, float],
) -> None:
    """Fetch data, compute indicators, run strategy, and render charts/stats."""
    try:
        with st.spinner(f"Fetching {config['asset']} {timeframe} data..."):
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
        st.error(f"Data fetch failed: {exc}")
        return

    if df.empty:
        st.warning("No data returned for this timeframe.")
        return

    try:
        df_indicators = compute_indicators(df, indicator_flags, indicator_params)
    except Exception as exc:
        st.error(f"Indicator computation failed: {exc}")
        return

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
        signal_params = {**indicator_params, **strategy_params}
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

    metrics = st.columns(4)
    metrics[0].metric("Total Profit", f"{summary['total_profit']:.2f}")
    metrics[1].metric("Win Rate", f"{summary['win_rate']:.2f}%")
    metrics[2].metric("Trades", f"{summary['num_trades']}")
    metrics[3].metric("Final Wallet", f"{summary['final_wallet']:.2f}")

    st.caption(f"Actual hours loaded: {actual_hours:.2f}")

    fig = build_timeframe_figure(
        df_indicators,
        asset=config["asset"],
        interval=timeframe,
        indicator_flags=indicator_flags,
        buy_signals=buy_signals,
        sell_signals=sell_signals,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Trades", expanded=False):
        st.dataframe(trades_df, use_container_width=True)


def main() -> None:
    """Streamlit entrypoint for the simulator UI."""
    st.set_page_config(
        page_title="CryptoFarm Simulator v2",
        page_icon=":bar_chart:",
        layout="wide",
    )

    init_session_state()

    st.sidebar.title("CryptoFarm Simulator v2")
    st.sidebar.caption("Optimized multi-timeframe simulator")

    st.sidebar.header("Market")
    st.sidebar.text_input("Asset", key="asset_base")
    st.sidebar.text_input("Quote", key="asset_quote")

    st.sidebar.radio("Range mode", ["Hours", "Dates"], key="range_mode")

    if st.session_state["range_mode"] == "Hours":
        st.sidebar.number_input(
            "Total hours (multiple of 24)",
            min_value=24,
            step=24,
            key="range_hours",
        )
    else:
        st.sidebar.date_input("Start date", key="range_start_date")
        st.sidebar.time_input("Start time", key="range_start_time")
        st.sidebar.date_input("End date", key="range_end_date")
        st.sidebar.time_input("End time", key="range_end_time")

    with st.sidebar.expander("Timeframes", expanded=True):
        options = list(TIMEFRAME_MINUTES.keys())
        for idx in range(1, 4):
            cols = st.columns([1, 2])
            with cols[0]:
                st.checkbox(f"TF{idx}", key=f"tf{idx}_enabled")
            with cols[1]:
                st.selectbox("Interval", options=options, key=f"tf{idx}_value")

    with st.sidebar.expander("Indicators", expanded=True):
        atr_enabled = st.checkbox("ATR Bands", key="ind_atr_bands")
        atr_cols = st.columns(2)
        atr_cols[0].number_input(
            "ATR Window",
            min_value=2,
            max_value=500,
            step=1,
            key="atr_window",
            disabled=not atr_enabled,
        )
        atr_cols[1].number_input(
            "ATR Multiplier",
            min_value=0.1,
            max_value=50.0,
            step=0.1,
            key="atr_multiplier",
            disabled=not atr_enabled,
        )

        st.divider()

        rsi_short = st.checkbox("RSI Short", key="ind_rsi_short")
        st.number_input(
            "RSI Short Window",
            min_value=2,
            max_value=500,
            step=1,
            key="rsi_short_window",
            disabled=not rsi_short,
        )
        rsi_medium = st.checkbox("RSI Medium", key="ind_rsi_medium")
        st.number_input(
            "RSI Medium Window",
            min_value=2,
            max_value=500,
            step=1,
            key="rsi_medium_window",
            disabled=not rsi_medium,
        )
        rsi_long = st.checkbox("RSI Long", key="ind_rsi_long")
        st.number_input(
            "RSI Long Window",
            min_value=2,
            max_value=500,
            step=1,
            key="rsi_long_window",
            disabled=not rsi_long,
        )

        st.divider()

        ema_short = st.checkbox("EMA Short", key="ind_ema_short")
        st.number_input(
            "EMA Short Window",
            min_value=1,
            max_value=500,
            step=1,
            key="ema_short_window",
            disabled=not ema_short,
        )
        ema_medium = st.checkbox("EMA Medium", key="ind_ema_medium")
        st.number_input(
            "EMA Medium Window",
            min_value=1,
            max_value=500,
            step=1,
            key="ema_medium_window",
            disabled=not ema_medium,
        )
        ema_long = st.checkbox("EMA Long", key="ind_ema_long")
        st.number_input(
            "EMA Long Window",
            min_value=1,
            max_value=500,
            step=1,
            key="ema_long_window",
            disabled=not ema_long,
        )

        st.divider()

        kama_enabled = st.checkbox("KAMA", key="ind_kama")
        kama_cols = st.columns(3)
        kama_cols[0].number_input(
            "KAMA Window",
            min_value=2,
            max_value=500,
            step=1,
            key="kama_window",
            disabled=not kama_enabled,
        )
        kama_cols[1].number_input(
            "KAMA Pow1",
            min_value=1,
            max_value=200,
            step=1,
            key="kama_pow1",
            disabled=not kama_enabled,
        )
        kama_cols[2].number_input(
            "KAMA Pow2",
            min_value=1,
            max_value=2000,
            step=1,
            key="kama_pow2",
            disabled=not kama_enabled,
        )

        st.divider()

        macd_enabled = st.checkbox("MACD", key="ind_macd")
        macd_cols = st.columns(3)
        macd_cols[0].number_input(
            "MACD Short",
            min_value=2,
            max_value=200,
            step=1,
            key="macd_short_window",
            disabled=not macd_enabled,
        )
        macd_cols[1].number_input(
            "MACD Long",
            min_value=2,
            max_value=300,
            step=1,
            key="macd_long_window",
            disabled=not macd_enabled,
        )
        macd_cols[2].number_input(
            "MACD Signal",
            min_value=1,
            max_value=200,
            step=1,
            key="macd_signal_window",
            disabled=not macd_enabled,
        )

    with st.sidebar.expander("Strategy", expanded=True):
        st.selectbox(
            "Strategy",
            options=list(STRATEGIES.keys()),
            key="strategy",
            on_change=apply_strategy_defaults,
        )
        strategy = STRATEGIES[st.session_state["strategy"]]
        if strategy.description:
            st.caption(strategy.description)

        st.number_input("Wallet", min_value=0.0, step=10.0, key="wallet")
        st.number_input("Fee %", min_value=0.0, max_value=5.0, step=0.01, key="fee_percent")

        stop_loss_disabled = "stop_loss_percent" not in strategy.default_params and strategy.name != "Custom"
        st.number_input(
            "Stop Loss %",
            min_value=0.1,
            max_value=100.0,
            step=0.1,
            key="stop_loss_percent",
            disabled=stop_loss_disabled,
        )

    config = build_config()
    errors = validate_config(config)
    if errors:
        for err in errors:
            st.error(err)
        return

    st.title("CryptoFarm Simulator v2")
    st.caption(
        f"Asset: {config['asset']} | Range: {format_range_label(config)} | Timeframes: {', '.join(config['timeframes'])}"
    )

    # Adjust hours to a multiple of 24 if needed
    if config["range_mode"] == "Hours" and config["hours"] % 24 != 0:
        adjusted = config["hours"] - (config["hours"] % 24)
        st.warning(f"Hours adjusted to {adjusted} to keep a multiple of 24.")
        config["hours"] = adjusted

    strategy = STRATEGIES[config["strategy"]]
    indicator_flags = dict(config["indicator_flags"])
    indicator_params = dict(config["indicator_params"])

    strategy_params = {
        "stop_loss_percent": config["stop_loss_percent"],
    }

    timeframes = config["timeframes"]
    if len(timeframes) == 1:
        render_timeframe(
            timeframes[0],
            config=config,
            strategy=strategy,
            indicator_flags=indicator_flags,
            indicator_params=indicator_params,
            strategy_params=strategy_params,
        )
    else:
        tabs = st.tabs(timeframes)
        for tab, timeframe in zip(tabs, timeframes):
            with tab:
                render_timeframe(
                    timeframe,
                    config=config,
                    strategy=strategy,
                    indicator_flags=indicator_flags,
                    indicator_params=indicator_params,
                    strategy_params=strategy_params,
                )


if __name__ == "__main__":
    main()
