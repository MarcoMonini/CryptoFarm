"""Massive search tool for Simulator v2.

Features:
- Grid or random search over parameter sets.
- Multiple assets and multiple timeframe combinations.
- Supports Binance fetch or CSV data already downloaded.
- Returns the best parameter set by total profit and summary stats.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import itertools
import os
import random
import re

import pandas as pd
import streamlit as st

from simulator_v2_backend import (
    STRATEGIES,
    TIMEFRAME_MINUTES,
    compute_indicators,
    fetch_market_data_dates,
    fetch_market_data_hours,
    mtf_close_buy_sell_limits,
    simulate_trades,
    summarize_trades,
)


INDICATOR_KEYS = [
    "atr_bands",
    "rsi_short",
    "rsi_medium",
    "rsi_long",
    "ema_short",
    "ema_medium",
    "ema_long",
    "kama",
    "macd",
]

DEFAULT_INDICATOR_FLAGS = {
    "atr_bands": True,
    "rsi_short": True,
    "rsi_medium": True,
    "rsi_long": True,
    "ema_short": True,
    "ema_medium": True,
    "ema_long": True,
    "kama": True,
    "macd": True,
}

# Base defaults aligned with simulator_v2_app.py.
BASE_PARAM_DEFAULTS = {
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
    "rsi_short_buy_limit": 25,
    "rsi_short_sell_limit": 75,
    "rsi_medium_buy_limit": 30,
    "rsi_medium_sell_limit": 70,
    "rsi_long_buy_limit": 35,
    "rsi_long_sell_limit": 65,
    "macd_buy_limit": -2.5,
    "macd_sell_limit": 2.5,
    "conditions_required": 1,
    "stop_loss_percent": 99.0,
}


@dataclass(frozen=True)
class ParamSpec:
    """Definition of a parameter that can be searched."""

    name: str
    kind: str  # "int" or "float"
    default: float
    group: str
    description: str


PARAM_SPECS = [
    ParamSpec("atr_window", "int", 5, "ATR", "ATR rolling window"),
    ParamSpec("atr_multiplier", "float", 1.6, "ATR", "ATR band multiplier"),
    ParamSpec("kama_window", "int", 10, "KAMA", "KAMA window"),
    ParamSpec("kama_pow1", "int", 2, "KAMA", "KAMA pow1"),
    ParamSpec("kama_pow2", "int", 30, "KAMA", "KAMA pow2"),
    ParamSpec("rsi_short_window", "int", 12, "RSI windows", "RSI short window"),
    ParamSpec("rsi_medium_window", "int", 24, "RSI windows", "RSI medium window"),
    ParamSpec("rsi_long_window", "int", 36, "RSI windows", "RSI long window"),
    ParamSpec("rsi_short_buy_limit", "int", 25, "RSI limits", "RSI short buy limit"),
    ParamSpec("rsi_short_sell_limit", "int", 75, "RSI limits", "RSI short sell limit"),
    ParamSpec("rsi_medium_buy_limit", "int", 30, "RSI limits", "RSI medium buy limit"),
    ParamSpec("rsi_medium_sell_limit", "int", 70, "RSI limits", "RSI medium sell limit"),
    ParamSpec("rsi_long_buy_limit", "int", 35, "RSI limits", "RSI long buy limit"),
    ParamSpec("rsi_long_sell_limit", "int", 65, "RSI limits", "RSI long sell limit"),
    ParamSpec("ema_short_window", "int", 10, "EMA", "EMA short window"),
    ParamSpec("ema_medium_window", "int", 50, "EMA", "EMA medium window"),
    ParamSpec("ema_long_window", "int", 200, "EMA", "EMA long window"),
    ParamSpec("macd_short_window", "int", 12, "MACD", "MACD short window"),
    ParamSpec("macd_long_window", "int", 26, "MACD", "MACD long window"),
    ParamSpec("macd_signal_window", "int", 9, "MACD", "MACD signal window"),
    ParamSpec("macd_buy_limit", "float", -2.5, "MACD", "MACD buy limit"),
    ParamSpec("macd_sell_limit", "float", 2.5, "MACD", "MACD sell limit"),
    ParamSpec("conditions_required", "int", 1, "Strategy", "Min conditions to trigger"),
    ParamSpec("stop_loss_percent", "float", 99.0, "Strategy", "Stop loss percent"),
]

STRATEGY_PARAM_NAMES = {
    "Buy/Sell Limits": [
        "atr_window",
        "atr_multiplier",
        "kama_window",
        "kama_pow1",
        "kama_pow2",
        "rsi_short_window",
        "rsi_medium_window",
        "rsi_long_window",
        "rsi_short_buy_limit",
        "rsi_short_sell_limit",
        "rsi_medium_buy_limit",
        "rsi_medium_sell_limit",
        "rsi_long_buy_limit",
        "rsi_long_sell_limit",
        "ema_short_window",
        "ema_medium_window",
        "ema_long_window",
        "macd_short_window",
        "macd_long_window",
        "macd_signal_window",
        "macd_buy_limit",
        "macd_sell_limit",
        "conditions_required",
    ],
    "Close ATR": [
        "atr_window",
        "atr_multiplier",
        "stop_loss_percent",
    ],
}
# Add a new strategy here:
# - Add its parameter names to STRATEGY_PARAM_NAMES.
# - Ensure STRATEGIES in simulator_v2_backend has a signal_func (or MTF func).


def parse_list(text: str, kind: str) -> List[float]:
    """Parse a list of numbers from comma list or range format start:stop:step."""
    raw = text.strip()
    if not raw:
        raise ValueError("Empty list.")

    if ":" in raw:
        parts = [p.strip() for p in raw.split(":")]
        if len(parts) != 3:
            raise ValueError("Range format must be start:stop:step.")
        start = float(parts[0])
        stop = float(parts[1])
        step = float(parts[2])
        if step <= 0:
            raise ValueError("Step must be > 0.")
        values: List[float] = []
        current = start
        # Use a small epsilon to include the stop when close.
        while current <= stop + 1e-9:
            values.append(current)
            current += step
    else:
        values = [float(item.strip()) for item in raw.split(",") if item.strip()]

    if kind == "int":
        return [int(round(v)) for v in values]
    return values


def parse_assets(text: str) -> List[str]:
    """Parse asset list from comma or newline separated text."""
    assets = []
    for item in re.split(r"[,\n]+", text):
        token = item.strip().upper()
        if token:
            assets.append(token)
    return assets


def parse_custom_combos(text: str) -> List[Tuple[str, ...]]:
    """Parse custom timeframe combos from lines like '15m,4h'."""
    combos: List[Tuple[str, ...]] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if not parts:
            continue
        combos.append(tuple(parts))
    return combos


def build_timeframe_combos(
    candidate_tfs: List[str],
    sizes: List[int],
    custom_text: str,
    use_custom: bool,
) -> List[Tuple[str, ...]]:
    """Return timeframe combinations based on user selection."""
    if use_custom:
        combos = parse_custom_combos(custom_text)
    else:
        combos = []
        for size in sizes:
            combos.extend(list(itertools.combinations(candidate_tfs, size)))
    # Validate and normalize.
    normalized = []
    for combo in combos:
        unique = tuple(dict.fromkeys(combo))  # preserve order, drop duplicates
        for tf in unique:
            if tf not in TIMEFRAME_MINUTES:
                raise ValueError(f"Unsupported timeframe: {tf}")
        normalized.append(unique)
    if not normalized:
        raise ValueError("No timeframe combinations selected.")
    return normalized


def parse_asset_interval_from_filename(filename: str) -> Optional[Tuple[str, str]]:
    """Extract asset and interval from filename like BTCUSDT_15m.csv."""
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]
    tokens = re.split(r"[_\-]", name)
    for token in tokens:
        if token in TIMEFRAME_MINUTES:
            interval = token
            asset = "".join([t for t in tokens if t != token]).upper()
            return asset, interval
    return None


def normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a CSV DataFrame to OHLCV with datetime index."""
    cols = {c.lower(): c for c in df.columns}
    index_col = None
    for key in ("open time", "open_time", "timestamp", "date", "time"):
        if key in cols:
            index_col = cols[key]
            break
    if index_col:
        df[index_col] = pd.to_datetime(df[index_col], utc=True, errors="coerce")
        df = df.dropna(subset=[index_col]).set_index(index_col)
    else:
        # If there is no explicit time column, assume index is already datetime.
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[~df.index.isna()]

    # Normalize column names to expected casing.
    rename_map = {}
    for key, target in [
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("volume", "Volume"),
    ]:
        if key in cols:
            rename_map[cols[key]] = target
    df = df.rename(columns=rename_map)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    df = df[required].astype(float).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def slice_df_by_range(
    df: pd.DataFrame,
    range_mode: str,
    hours: int,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    """Slice a DataFrame to the selected time range."""
    if df.empty:
        return df
    if range_mode == "Hours":
        end_ts = df.index.max()
        start_ts = end_ts - pd.Timedelta(hours=hours)
        return df.loc[df.index >= start_ts]
    return df.loc[(df.index >= start_dt) & (df.index <= end_dt)]


def get_param_spec(name: str) -> ParamSpec:
    """Return ParamSpec by name."""
    for spec in PARAM_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(name)


def generate_combos(
    param_lists: Dict[str, List[float]],
    mode: str,
    random_samples: int,
    seed: int,
) -> Tuple[List[str], Iterable[Tuple[float, ...]], int]:
    """Generate param combinations for grid or random search."""
    names = list(param_lists.keys())
    values_list = [param_lists[name] for name in names]
    total_grid = 1
    for values in values_list:
        total_grid *= max(1, len(values))

    if mode == "Grid":
        return names, itertools.product(*values_list), total_grid

    random.seed(seed)
    sample_target = min(random_samples, total_grid)
    combos: List[Tuple[float, ...]] = []
    seen = set()
    # Randomly draw combinations from the discrete value lists.
    while len(combos) < sample_target and len(seen) < total_grid:
        combo = tuple(random.choice(values) for values in values_list)
        if combo in seen:
            continue
        seen.add(combo)
        combos.append(combo)
    return names, combos, len(combos)


def build_indicator_flags(strategy_name: str, custom_flags: Dict[str, bool]) -> Dict[str, bool]:
    """Build indicator flags, forcing required ones if needed."""
    flags = dict(custom_flags)
    if strategy_name == "Close ATR":
        # Close ATR requires ATR bands.
        flags["atr_bands"] = True
    return flags


def build_params(param_values: Dict[str, float]) -> Dict[str, float]:
    """Merge param values with defaults to create a full param dict."""
    params = dict(BASE_PARAM_DEFAULTS)
    params.update(param_values)
    return params


def fetch_or_load_data(
    asset: str,
    interval: str,
    source_mode: str,
    range_mode: str,
    hours: int,
    start_dt: datetime,
    end_dt: datetime,
    file_cache: Dict[Tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    """Return OHLCV data for asset/interval using fetch or CSV cache."""
    if source_mode == "Binance":
        if range_mode == "Hours":
            df, _ = fetch_market_data_hours(asset, interval, hours)
        else:
            df, _ = fetch_market_data_dates(asset, interval, start_dt, end_dt)
        return df

    key = (asset, interval)
    if key not in file_cache:
        raise ValueError(f"Missing CSV data for {asset} {interval}.")
    df = file_cache[key]
    return slice_df_by_range(df, range_mode, hours, start_dt, end_dt)


def run_mtf_strategy(
    strategy_name: str,
    asset: str,
    timeframe_combo: Tuple[str, ...],
    indicator_flags: Dict[str, bool],
    params: Dict[str, float],
    data_cache: Dict[str, object],
    range_mode: str,
    hours: int,
    start_dt: datetime,
    end_dt: datetime,
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    """Compute signals and trades for a multi-timeframe strategy."""
    frames = []
    for interval in timeframe_combo:
        df = fetch_or_load_data(
            asset=asset,
            interval=interval,
            source_mode=data_cache["source_mode"],
            range_mode=range_mode,
            hours=hours,
            start_dt=start_dt,
            end_dt=end_dt,
            file_cache=data_cache["file_cache"],
        )
        if df.empty:
            return 0.0, summarize_trades(pd.DataFrame(), params.get("wallet", 0.0)), pd.DataFrame()

        df_ind = compute_indicators(df, indicator_flags, params)
        frames.append(
            {
                "timeframe": interval,
                "df": df_ind,
                "use_rsi_short": indicator_flags.get("rsi_short"),
                "use_rsi_medium": indicator_flags.get("rsi_medium"),
                "use_rsi_long": indicator_flags.get("rsi_long"),
                "use_atr": indicator_flags.get("atr_bands"),
                "use_macd": indicator_flags.get("macd"),
                "ema_short_on": indicator_flags.get("ema_short"),
                "ema_medium_on": indicator_flags.get("ema_medium"),
                "ema_long_on": indicator_flags.get("ema_long"),
                "ema_short_window": params["ema_short_window"],
                "ema_medium_window": params["ema_medium_window"],
                "ema_long_window": params["ema_long_window"],
                "rsi_short_buy_limit": params["rsi_short_buy_limit"],
                "rsi_short_sell_limit": params["rsi_short_sell_limit"],
                "rsi_medium_buy_limit": params["rsi_medium_buy_limit"],
                "rsi_medium_sell_limit": params["rsi_medium_sell_limit"],
                "rsi_long_buy_limit": params["rsi_long_buy_limit"],
                "rsi_long_sell_limit": params["rsi_long_sell_limit"],
                "macd_buy_limit": params["macd_buy_limit"],
                "macd_sell_limit": params["macd_sell_limit"],
            }
        )

    base_timeframe = min(timeframe_combo, key=lambda tf: TIMEFRAME_MINUTES[tf])
    buy_signals, sell_signals = mtf_close_buy_sell_limits(
        frames=frames,
        base_timeframe=base_timeframe,
        conditions_required=int(params["conditions_required"]),
    )
    trades_df = simulate_trades(
        buy_signals,
        sell_signals,
        initial_wallet=float(params["wallet"]),
        fee_percent=float(params["fee_percent"]),
    )
    summary = summarize_trades(trades_df, initial_wallet=float(params["wallet"]))
    return summary["total_profit"], summary, trades_df


def run_single_strategy(
    strategy_name: str,
    asset: str,
    interval: str,
    indicator_flags: Dict[str, bool],
    params: Dict[str, float],
    data_cache: Dict[str, object],
    range_mode: str,
    hours: int,
    start_dt: datetime,
    end_dt: datetime,
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    """Compute signals and trades for a single-timeframe strategy."""
    df = fetch_or_load_data(
        asset=asset,
        interval=interval,
        source_mode=data_cache["source_mode"],
        range_mode=range_mode,
        hours=hours,
        start_dt=start_dt,
        end_dt=end_dt,
        file_cache=data_cache["file_cache"],
    )
    if df.empty:
        return 0.0, summarize_trades(pd.DataFrame(), params.get("wallet", 0.0)), pd.DataFrame()

    df_ind = compute_indicators(df, indicator_flags, params)
    strategy = STRATEGIES[strategy_name]
    if not strategy.signal_func:
        return 0.0, summarize_trades(pd.DataFrame(), params.get("wallet", 0.0)), pd.DataFrame()

    buy_signals, sell_signals = strategy.signal_func(df_ind, params)
    trades_df = simulate_trades(
        buy_signals,
        sell_signals,
        initial_wallet=float(params["wallet"]),
        fee_percent=float(params["fee_percent"]),
    )
    summary = summarize_trades(trades_df, initial_wallet=float(params["wallet"]))
    return summary["total_profit"], summary, trades_df


def main() -> None:
    st.set_page_config(
        page_title="Simulator v2 Massive Search",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
    )

    st.title("Simulator v2 Massive Search")
    st.caption("Grid/random search across assets, timeframes, and strategies.")

    # --- Data source ---
    with st.expander("1) Data Source", expanded=True):
        source_mode = st.radio("Data source", ["Binance", "CSV Upload", "CSV Folder"])
        range_mode = st.radio("Range mode", ["Hours", "Dates"], horizontal=True)
        hours = 24
        start_dt = datetime.utcnow() - timedelta(days=7)
        end_dt = datetime.utcnow()
        if range_mode == "Hours":
            hours = st.number_input("Total hours", min_value=24, step=24, value=24)
        else:
            start_date = st.date_input("Start date", value=(datetime.utcnow() - timedelta(days=7)).date())
            start_time = st.time_input("Start time", value=time(0, 0))
            end_date = st.date_input("End date", value=datetime.utcnow().date())
            end_time = st.time_input("End time", value=time(23, 59))
            start_dt = datetime.combine(start_date, start_time)
            end_dt = datetime.combine(end_date, end_time)

        file_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
        if source_mode == "CSV Upload":
            uploads = st.file_uploader(
                "Upload CSV files (name format: ASSET_INTERVAL.csv)",
                accept_multiple_files=True,
                type=["csv"],
            )
            if uploads:
                for file in uploads:
                    parsed = parse_asset_interval_from_filename(file.name)
                    if not parsed:
                        st.warning(f"Cannot parse asset/interval from {file.name}.")
                        continue
                    asset, interval = parsed
                    df = pd.read_csv(file)
                    file_cache[(asset, interval)] = normalize_ohlcv_df(df)

        if source_mode == "CSV Folder":
            folder_path = st.text_input(
                "CSV folder path (files named ASSET_INTERVAL.csv)",
                value="",
            )
            if folder_path:
                try:
                    for filename in os.listdir(folder_path):
                        if not filename.lower().endswith(".csv"):
                            continue
                        parsed = parse_asset_interval_from_filename(filename)
                        if not parsed:
                            continue
                        asset, interval = parsed
                        df = pd.read_csv(os.path.join(folder_path, filename))
                        file_cache[(asset, interval)] = normalize_ohlcv_df(df)
                except Exception as exc:
                    st.error(f"CSV folder error: {exc}")

    # --- Strategy and assets ---
    with st.expander("2) Strategy and Assets", expanded=True):
        strategy_name = st.selectbox("Strategy", options=list(STRATEGIES.keys()), index=2)
        if strategy_name not in STRATEGY_PARAM_NAMES:
            st.info("This strategy has no search spec. Only Buy/Sell Limits and Close ATR are supported.")

        asset_text = st.text_area("Assets (comma or newline separated)", value="BTCUSDT")
        assets = parse_assets(asset_text)
        if not assets:
            st.error("Please enter at least one asset.")

        wallet = st.number_input("Initial wallet", min_value=0.0, value=100.0, step=10.0)
        fee_percent = st.number_input("Fee %", min_value=0.0, max_value=5.0, value=0.1, step=0.01)

    # --- Timeframes ---
    with st.expander("3) Timeframes", expanded=True):
        if STRATEGIES[strategy_name].multi_timeframe:
            use_custom_combos = st.checkbox("Use custom combinations", value=False)
            if use_custom_combos:
                combos_text = st.text_area("One combo per line (example: 15m,4h)")
                candidate_tfs = list(TIMEFRAME_MINUTES.keys())
                sizes = [1]
            else:
                candidate_tfs = st.multiselect(
                    "Candidate timeframes",
                    options=list(TIMEFRAME_MINUTES.keys()),
                    default=["15m", "4h"],
                )
                sizes = st.multiselect("Combination sizes", options=[1, 2, 3], default=[2])
                combos_text = ""
        else:
            use_custom_combos = False
            candidate_tfs = st.multiselect(
                "Timeframes",
                options=list(TIMEFRAME_MINUTES.keys()),
                default=["15m"],
            )
            sizes = [1]
            combos_text = ""

        timeframe_combos = []
        try:
            timeframe_combos = build_timeframe_combos(candidate_tfs, sizes, combos_text, use_custom_combos)
            st.caption(f"Total combinations: {len(timeframe_combos)}")
        except Exception as exc:
            st.error(f"Timeframe error: {exc}")

    # --- Indicators ---
    with st.expander("4) Indicators Used in Strategy", expanded=True):
        indicator_flags = {}
        cols = st.columns(3)
        for idx, key in enumerate(INDICATOR_KEYS):
            with cols[idx % 3]:
                indicator_flags[key] = st.checkbox(key, value=DEFAULT_INDICATOR_FLAGS.get(key, False))

        indicator_flags = build_indicator_flags(strategy_name, indicator_flags)

    # --- Parameter space ---
    with st.expander("5) Parameter Search Space", expanded=True):
        mode = st.selectbox("Search mode", ["Grid", "Random (from lists)"])
        random_samples = 50
        random_seed = 42
        if mode != "Grid":
            random_samples = st.number_input("Random samples", min_value=1, value=50, step=1)
            random_seed = st.number_input("Random seed", min_value=0, value=42, step=1)

        if strategy_name in STRATEGY_PARAM_NAMES:
            params_to_use = STRATEGY_PARAM_NAMES[strategy_name]
        else:
            params_to_use = []

        # Group parameters by indicator for readability.
        param_values: Dict[str, List[float]] = {}
        grouped: Dict[str, List[ParamSpec]] = {}
        for name in params_to_use:
            spec = get_param_spec(name)
            grouped.setdefault(spec.group, []).append(spec)

        for group, specs in grouped.items():
            st.markdown(f"**{group}**")
            for spec in specs:
                default_list = str(BASE_PARAM_DEFAULTS.get(spec.name, spec.default))
                help_text = f"{spec.description} | list '1,2,3' or range '1:10:1'"
                value_text = st.text_input(f"{spec.name}", value=default_list, help=help_text)
                try:
                    param_values[spec.name] = parse_list(value_text, spec.kind)
                except Exception as exc:
                    st.error(f"Param {spec.name} error: {exc}")

    # --- Run search ---
    st.divider()
    run_clicked = st.button("Run massive search", type="primary")

    if run_clicked:
        if not assets or not timeframe_combos or strategy_name not in STRATEGY_PARAM_NAMES:
            st.error("Missing assets, timeframes, or unsupported strategy.")
            return
        if len(param_values) != len(STRATEGY_PARAM_NAMES[strategy_name]):
            st.error("Please fix parameter errors before running.")
            return

        param_names, combos_iter, combo_count = generate_combos(
            param_lists=param_values,
            mode=mode,
            random_samples=int(random_samples),
            seed=int(random_seed),
        )
        total_runs = len(assets) * len(timeframe_combos) * combo_count
        st.info(f"Total runs: {total_runs}")

        progress = st.progress(0)
        run_idx = 0

        aggregate: Dict[Tuple[float, ...], Dict[str, float]] = {}
        top_runs: List[Dict[str, object]] = []
        top_n = 20

        data_cache = {
            "source_mode": source_mode,
            "file_cache": file_cache,
        }

        for combo in combos_iter:
            param_set = dict(zip(param_names, combo))
            params = build_params(param_set)
            params["wallet"] = float(wallet)
            params["fee_percent"] = float(fee_percent)

            for asset in assets:
                for tf_combo in timeframe_combos:
                    try:
                        if STRATEGIES[strategy_name].multi_timeframe:
                            profit, summary, trades_df = run_mtf_strategy(
                                strategy_name,
                                asset,
                                tf_combo,
                                indicator_flags,
                                params,
                                data_cache,
                                range_mode,
                                int(hours),
                                start_dt,
                                end_dt,
                            )
                        else:
                            profit, summary, trades_df = run_single_strategy(
                                strategy_name,
                                asset,
                                tf_combo[0],
                                indicator_flags,
                                params,
                                data_cache,
                                range_mode,
                                int(hours),
                                start_dt,
                                end_dt,
                            )
                    except Exception as exc:
                        st.warning(f"Run error {asset} {tf_combo}: {exc}")
                        profit, summary, trades_df = 0.0, summarize_trades(pd.DataFrame(), wallet), pd.DataFrame()

                    win_trades = int((trades_df["profit"] > 0).sum()) if not trades_df.empty else 0
                    num_trades = int(summary["num_trades"])

                    key = tuple(combo)
                    if key not in aggregate:
                        aggregate[key] = {
                            "total_profit": 0.0,
                            "total_trades": 0,
                            "win_trades": 0,
                            "runs": 0,
                        }
                    aggregate[key]["total_profit"] += float(summary["total_profit"])
                    aggregate[key]["total_trades"] += num_trades
                    aggregate[key]["win_trades"] += win_trades
                    aggregate[key]["runs"] += 1

                    # Track top single runs by profit.
                    top_runs.append(
                        {
                            "Asset": asset,
                            "Timeframes": ",".join(tf_combo),
                            "Profit": float(summary["total_profit"]),
                            "WinRate": float(summary["win_rate"]),
                            "Trades": int(summary["num_trades"]),
                            "FinalWallet": float(summary["final_wallet"]),
                            **param_set,
                        }
                    )
                    if len(top_runs) > top_n:
                        top_runs = sorted(top_runs, key=lambda x: x["Profit"], reverse=True)[:top_n]

                    run_idx += 1
                    if total_runs > 0:
                        progress.progress(min(run_idx / total_runs, 1.0))

        # Build aggregate DataFrame.
        rows = []
        for key, stats in aggregate.items():
            param_set = dict(zip(param_names, key))
            trades = stats["total_trades"]
            win_rate = (stats["win_trades"] / trades * 100.0) if trades > 0 else 0.0
            rows.append(
                {
                    **param_set,
                    "TotalProfit": stats["total_profit"],
                    "TotalTrades": trades,
                    "WinRate": win_rate,
                    "Runs": stats["runs"],
                }
            )

        result_df = pd.DataFrame(rows).sort_values("TotalProfit", ascending=False)
        st.subheader("Best Parameter Set (by Total Profit Sum)")
        if not result_df.empty:
            st.dataframe(result_df.head(1), use_container_width=True)
        else:
            st.info("No results.")

        st.subheader("Top Parameter Sets")
        st.dataframe(result_df.head(20), use_container_width=True)

        st.subheader("Top Single Runs")
        if top_runs:
            st.dataframe(pd.DataFrame(top_runs), use_container_width=True)

        csv_data = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results CSV",
            data=csv_data,
            file_name="massive_search_results.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
