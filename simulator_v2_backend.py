from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import os
import time

import pandas as pd
import streamlit as st
from binance import Client
from ta.momentum import KAMAIndicator, RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange
from plotly.subplots import make_subplots
import plotly.graph_objects as go

Signal = Tuple[pd.Timestamp, float]

TIMEFRAME_MINUTES = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "1d": 1440,
}


@dataclass(frozen=True)
class StrategySpec:
    name: str
    required_indicators: frozenset[str]
    default_params: Dict[str, float]
    signal_func: Optional[Callable[[pd.DataFrame, Dict[str, float]], Tuple[List[Signal], List[Signal]]]]
    description: str = ""


def _read_secret(key: str) -> str:
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no"}


def _build_requests_params() -> Dict[str, object]:
    verify_ssl = _env_flag("BINANCE_SSL_VERIFY", default=True)
    proxy_http = os.getenv("BINANCE_PROXY_HTTP") or os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    proxy_https = os.getenv("BINANCE_PROXY_HTTPS") or os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    params: Dict[str, object] = {"verify": verify_ssl}
    proxies: Dict[str, str] = {}
    if proxy_http:
        proxies["http"] = proxy_http
    if proxy_https:
        proxies["https"] = proxy_https
    if proxies:
        params["proxies"] = proxies
    return params


def get_binance_client() -> Client:
    """Create a Binance client using env vars or Streamlit secrets."""
    api_key = os.getenv("BINANCE_API_KEY", "")  # or _read_secret("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET", "")  # or _read_secret("BINANCE_API_SECRET")
    # requests_params = _build_requests_params()
    return Client(
        api_key=api_key,
        api_secret=api_secret,
        # requests_params=requests_params,
    )


def interval_to_minutes(interval: str) -> int:
    """Convert a Binance interval string to minutes."""
    if interval not in TIMEFRAME_MINUTES:
        raise ValueError(f"Unsupported interval: {interval}")
    return TIMEFRAME_MINUTES[interval]


def _align_ms(timestamp_ms: int, interval_ms: int) -> int:
    """Align a timestamp down to the nearest interval boundary."""
    if interval_ms <= 0:
        return timestamp_ms
    return timestamp_ms - (timestamp_ms % interval_ms)


@st.cache_data(show_spinner=False)
def fetch_klines_range(asset: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch klines in chunks and return a sorted OHLCV DataFrame."""
    if start_ms >= end_ms:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    client = get_binance_client()
    interval_minutes = interval_to_minutes(interval)
    interval_ms = interval_minutes * 60_000

    all_klines: List[list] = []
    fetch_start = start_ms

    while fetch_start < end_ms:
        chunk = client.get_klines(
            symbol=asset,
            interval=interval,
            limit=1000,
            startTime=fetch_start,
            endTime=end_ms,
        )
        if not chunk:
            break

        all_klines.extend(chunk)

        last_open_time = chunk[-1][0]
        next_open_time = last_open_time + interval_ms

        if next_open_time <= fetch_start:
            break

        fetch_start = next_open_time

        if len(chunk) < 1000:
            break

    if not all_klines:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    columns = [
        "Open time",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Close time",
        "Quote asset volume",
        "Number of trades",
        "Taker buy base asset volume",
        "Taker buy quote asset volume",
        "Ignore",
    ]
    raw_df = pd.DataFrame(all_klines, columns=columns)
    raw_df["Open time"] = pd.to_datetime(raw_df["Open time"], unit="ms")
    raw_df.set_index("Open time", inplace=True)

    df = raw_df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    return df


def fetch_market_data_hours(asset: str, interval: str, hours: int) -> Tuple[pd.DataFrame, float]:
    """Fetch the last N hours of data and return the DataFrame and hours loaded."""
    interval_minutes = interval_to_minutes(interval)
    interval_ms = interval_minutes * 60_000
    end_ms = _align_ms(int(time.time() * 1000), interval_ms)
    start_ms = _align_ms(end_ms - hours * 60 * 60 * 1000, interval_ms)

    df = fetch_klines_range(asset=asset, interval=interval, start_ms=start_ms, end_ms=end_ms)
    actual_hours = (len(df) * interval_minutes / 60.0) if not df.empty else 0.0
    return df, actual_hours


def fetch_market_data_dates(
        asset: str,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
) -> Tuple[pd.DataFrame, float]:
    """Fetch data between two datetimes and return the DataFrame and hours loaded."""
    interval_minutes = interval_to_minutes(interval)
    interval_ms = interval_minutes * 60_000
    start_ms = _align_ms(int(start_dt.timestamp() * 1000), interval_ms)
    end_ms = _align_ms(int(end_dt.timestamp() * 1000), interval_ms)

    df = fetch_klines_range(asset=asset, interval=interval, start_ms=start_ms, end_ms=end_ms)
    actual_hours = (len(df) * interval_minutes / 60.0) if not df.empty else 0.0
    return df, actual_hours


def compute_indicators(
        df: pd.DataFrame,
        indicator_flags: Dict[str, bool],
        indicator_params: Dict[str, float],
) -> pd.DataFrame:
    """Compute only the selected indicators and return a copy with new columns."""
    if df.empty:
        return df
    if not any(indicator_flags.values()):
        return df

    df_out = df.copy()

    if indicator_flags.get("atr_bands"):
        atr_window = int(indicator_params["atr_window"])
        atr_multiplier = float(indicator_params["atr_multiplier"])

        atr_indicator = AverageTrueRange(
            high=df_out["High"],
            low=df_out["Low"],
            close=df_out["Close"],
            window=atr_window,
        )
        df_out["ATR"] = atr_indicator.average_true_range()

        # Use an EMA midline for the bands to keep the band logic independent
        ema_mid = EMAIndicator(close=df_out["Close"], window=atr_window).ema_indicator()
        df_out["ATR_MID"] = ema_mid
        df_out["ATR_UPPER"] = ema_mid + (atr_multiplier * df_out["ATR"])
        df_out["ATR_LOWER"] = ema_mid - (atr_multiplier * df_out["ATR"])

    if indicator_flags.get("rsi_short"):
        window = int(indicator_params["rsi_short_window"])
        df_out["RSI_SHORT"] = RSIIndicator(close=df_out["Close"], window=window).rsi()

    if indicator_flags.get("rsi_medium"):
        window = int(indicator_params["rsi_medium_window"])
        df_out["RSI_MED"] = RSIIndicator(close=df_out["Close"], window=window).rsi()

    if indicator_flags.get("rsi_long"):
        window = int(indicator_params["rsi_long_window"])
        df_out["RSI_LONG"] = RSIIndicator(close=df_out["Close"], window=window).rsi()

    if indicator_flags.get("ema_short"):
        window = int(indicator_params["ema_short_window"])
        df_out["EMA_SHORT"] = EMAIndicator(close=df_out["Close"], window=window).ema_indicator()

    if indicator_flags.get("ema_medium"):
        window = int(indicator_params["ema_medium_window"])
        df_out["EMA_MED"] = EMAIndicator(close=df_out["Close"], window=window).ema_indicator()

    if indicator_flags.get("ema_long"):
        window = int(indicator_params["ema_long_window"])
        df_out["EMA_LONG"] = EMAIndicator(close=df_out["Close"], window=window).ema_indicator()

    if indicator_flags.get("kama"):
        window = int(indicator_params["kama_window"])
        pow1 = int(indicator_params["kama_pow1"])
        pow2 = int(indicator_params["kama_pow2"])
        df_out["KAMA"] = KAMAIndicator(
            close=df_out["Close"],
            window=window,
            pow1=pow1,
            pow2=pow2,
        ).kama()

    if indicator_flags.get("macd"):
        short = int(indicator_params["macd_short_window"])
        long = int(indicator_params["macd_long_window"])
        signal = int(indicator_params["macd_signal_window"])
        macd_indicator = MACD(
            close=df_out["Close"],
            window_slow=long,
            window_fast=short,
            window_sign=signal,
        )
        df_out["MACD"] = macd_indicator.macd()
        df_out["MACD_SIGNAL"] = macd_indicator.macd_signal()
        df_out["MACD_HIST"] = macd_indicator.macd_diff()

    return df_out


def close_atr_signals(df: pd.DataFrame, params: Dict[str, float]) -> Tuple[List[Signal], List[Signal]]:
    """Close ATR strategy: buy below lower band, sell above upper band."""
    buy_signals: List[Signal] = []
    sell_signals: List[Signal] = []
    holding = False
    stop_loss_price: Optional[float] = None

    stop_loss_percent = float(params.get("stop_loss_percent", 99.0))
    stop_loss_decimal = stop_loss_percent / 100.0

    close_series = df["Close"]
    upper_series = df["ATR_UPPER"] if "ATR_UPPER" in df else pd.Series(index=df.index, dtype="float64")
    lower_series = df["ATR_LOWER"] if "ATR_LOWER" in df else pd.Series(index=df.index, dtype="float64")

    for i in range(1, len(df)):
        close = close_series.iloc[i]
        upper = upper_series.iloc[i]
        lower = lower_series.iloc[i]

        if pd.isna(upper) or pd.isna(lower):
            continue

        if not holding and close <= lower:
            buy_signals.append((df.index[i], float(close)))
            holding = True
            stop_loss_price = float(close) * (1.0 - stop_loss_decimal)
            continue

        if holding and close >= upper:
            sell_signals.append((df.index[i], float(close)))
            holding = False
            stop_loss_price = None
            continue

        if holding and stop_loss_price is not None and close <= stop_loss_price:
            sell_signals.append((df.index[i], float(close)))
            holding = False
            stop_loss_price = None

    return buy_signals, sell_signals


def simulate_trades(
        buy_signals: Sequence[Signal],
        sell_signals: Sequence[Signal],
        initial_wallet: float,
        fee_percent: float,
) -> pd.DataFrame:
    """Simulate buy/sell signals with fees and return a trades DataFrame."""
    trades: List[Dict[str, float]] = []

    if not buy_signals or not sell_signals:
        return pd.DataFrame(
            columns=[
                "buy_time",
                "buy_price",
                "sell_time",
                "sell_price",
                "quantity",
                "profit",
                "wallet_after",
            ]
        )

    fee_rate = fee_percent / 100.0
    wallet = float(initial_wallet)
    sell_idx = 0

    for buy_time, buy_price in buy_signals:
        while sell_idx < len(sell_signals) and sell_signals[sell_idx][0] <= buy_time:
            sell_idx += 1

        if sell_idx >= len(sell_signals):
            break

        sell_time, sell_price = sell_signals[sell_idx]
        sell_idx += 1

        wallet_before = wallet
        invested = wallet_before * (1.0 - fee_rate)
        quantity = invested / buy_price

        gross_proceed = quantity * sell_price
        wallet = gross_proceed * (1.0 - fee_rate)

        profit = wallet - wallet_before
        trades.append(
            {
                "buy_time": buy_time,
                "buy_price": float(buy_price),
                "sell_time": sell_time,
                "sell_price": float(sell_price),
                "quantity": float(quantity),
                "profit": float(profit),
                "wallet_after": float(wallet),
            }
        )

    return pd.DataFrame(trades)


def summarize_trades(trades_df: pd.DataFrame, initial_wallet: float) -> Dict[str, float]:
    """Build summary stats for a trades DataFrame."""
    if trades_df.empty:
        return {
            "total_profit": 0.0,
            "win_rate": 0.0,
            "num_trades": 0,
            "final_wallet": initial_wallet,
        }

    total_profit = float(trades_df["profit"].sum())
    num_trades = int(len(trades_df))
    win_rate = float((trades_df["profit"] > 0).mean() * 100.0)
    final_wallet = float(trades_df["wallet_after"].iloc[-1])

    return {
        "total_profit": total_profit,
        "win_rate": win_rate,
        "num_trades": num_trades,
        "final_wallet": final_wallet,
    }


def build_timeframe_figure(
        df: pd.DataFrame,
        asset: str,
        interval: str,
        indicator_flags: Dict[str, bool],
        buy_signals: Sequence[Signal],
        sell_signals: Sequence[Signal],
) -> go.Figure:
    """Build a multi-row Plotly figure with overlays and indicator panels."""
    show_rsi = any(indicator_flags.get(key) for key in ("rsi_short", "rsi_medium", "rsi_long"))
    show_macd = indicator_flags.get("macd")

    rows = 1 + int(show_rsi) + int(show_macd)
    row_heights = [1.0] if rows == 1 else [0.6] + [0.2] * (rows - 1)

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=f"{asset} {interval}",
        ),
        row=1,
        col=1,
    )

    if indicator_flags.get("ema_short"):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA_SHORT"],
                mode="lines",
                line=dict(color="#1f77b4", width=1),
                name="EMA Short",
            ),
            row=1,
            col=1,
        )

    if indicator_flags.get("ema_medium"):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA_MED"],
                mode="lines",
                line=dict(color="#2ca02c", width=1),
                name="EMA Medium",
            ),
            row=1,
            col=1,
        )

    if indicator_flags.get("ema_long"):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA_LONG"],
                mode="lines",
                line=dict(color="#ff7f0e", width=1),
                name="EMA Long",
            ),
            row=1,
            col=1,
        )

    if indicator_flags.get("kama"):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["KAMA"],
                mode="lines",
                line=dict(color="#9467bd", width=1),
                name="KAMA",
            ),
            row=1,
            col=1,
        )

    if indicator_flags.get("atr_bands"):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["ATR_UPPER"],
                mode="lines",
                line=dict(color="#8c564b", width=1, dash="dash"),
                name="ATR Upper",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["ATR_LOWER"],
                mode="lines",
                line=dict(color="#8c564b", width=1, dash="dash"),
                name="ATR Lower",
            ),
            row=1,
            col=1,
        )

    if buy_signals:
        buy_times, buy_prices = zip(*buy_signals)
        fig.add_trace(
            go.Scatter(
                x=buy_times,
                y=buy_prices,
                mode="markers",
                marker=dict(size=10, color="#2ca02c", symbol="triangle-up"),
                name="Buy",
            ),
            row=1,
            col=1,
        )

    if sell_signals:
        sell_times, sell_prices = zip(*sell_signals)
        fig.add_trace(
            go.Scatter(
                x=sell_times,
                y=sell_prices,
                mode="markers",
                marker=dict(size=10, color="#d62728", symbol="triangle-down"),
                name="Sell",
            ),
            row=1,
            col=1,
        )

    next_row = 2
    if show_rsi:
        if indicator_flags.get("rsi_short"):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["RSI_SHORT"],
                    mode="lines",
                    line=dict(color="#17becf", width=1),
                    name="RSI Short",
                ),
                row=next_row,
                col=1,
            )
        if indicator_flags.get("rsi_medium"):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["RSI_MED"],
                    mode="lines",
                    line=dict(color="#bcbd22", width=1),
                    name="RSI Medium",
                ),
                row=next_row,
                col=1,
            )
        if indicator_flags.get("rsi_long"):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["RSI_LONG"],
                    mode="lines",
                    line=dict(color="#7f7f7f", width=1),
                    name="RSI Long",
                ),
                row=next_row,
                col=1,
            )
        next_row += 1

    if show_macd:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD"],
                mode="lines",
                line=dict(color="#1f77b4", width=1),
                name="MACD",
            ),
            row=next_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD_SIGNAL"],
                mode="lines",
                line=dict(color="#ff7f0e", width=1),
                name="MACD Signal",
            ),
            row=next_row,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["MACD_HIST"],
                marker=dict(color="#c7c7c7"),
                name="MACD Hist",
            ),
            row=next_row,
            col=1,
        )

    fig.update_layout(
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        height=520 + (rows - 1) * 180,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


STRATEGIES: Dict[str, StrategySpec] = {
    "Custom": StrategySpec(
        name="Custom",
        required_indicators=frozenset(),
        default_params={},
        signal_func=None,
        description="Manual indicators only. No signals are computed.",
    ),
    "Close ATR": StrategySpec(
        name="Close ATR",
        required_indicators=frozenset({"atr_bands"}),
        default_params={
            "atr_window": 5,
            "atr_multiplier": 1.6,
            "stop_loss_percent": 99.0,
        },
        signal_func=close_atr_signals,
        description="Buy when close is below lower ATR band, sell when close is above upper band.",
    ),
}

# Strategy extension guide:
# - Implement a signal function that returns (buy_signals, sell_signals).
# - Register it in STRATEGIES with required indicators and default params.
