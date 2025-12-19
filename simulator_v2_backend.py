"""Backend per Simulator v2.

Struttura logica:
- Fetch dati Binance in chunk e normalizzazione OHLCV.
- Calcolo indicatori solo se selezionati (per ridurre il costo).
- Strategie e segnali buy/sell (single o multi timeframe).
- Simulazione trade e rendering Plotly.
"""

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
    """Spec dei parametri di una strategia.

    Fields:
        name: nome della strategia in UI.
        required_indicators: indicatori richiesti per segnali.
        default_params: valori di default per parametri strategia.
        signal_func: funzione (df, params) -> (buy, sell), o None.
        description: descrizione breve per la UI.
        multi_timeframe: True se richiede piu timeframe.
        default_indicators: indicatori da attivare quando selezionata.
    """
    name: str
    required_indicators: frozenset[str]
    default_params: Dict[str, float]
    signal_func: Optional[Callable[[pd.DataFrame, Dict[str, float]], Tuple[List[Signal], List[Signal]]]]
    description: str = ""
    multi_timeframe: bool = False
    default_indicators: frozenset[str] = frozenset()


def _read_secret(key: str) -> str:
    """Legge una chiave da st.secrets in modo sicuro.

    Input:
        key: nome della chiave.
    Output:
        stringa (vuota se non disponibile).
    """
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


def _env_flag(name: str, default: bool = True) -> bool:
    """Converte una variabile d'ambiente in boolean.

    Input:
        name: nome della variabile.
        default: valore se non presente.
    Output:
        True/False in base al contenuto (0/false/no -> False).
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no"}


def _build_requests_params() -> Dict[str, object]:
    """Costruisce parametri requests per il client Binance.

    Include:
    - verify SSL (BINANCE_SSL_VERIFY)
    - proxy HTTP/HTTPS (BINANCE_PROXY_* o HTTP(S)_PROXY)
    """
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
    """Crea un client Binance con credenziali e parametri requests.

    Input:
        BINANCE_API_KEY / BINANCE_API_SECRET (env o st.secrets)
        BINANCE_SSL_VERIFY, BINANCE_PROXY_HTTP/HTTPS
    Output:
        Client configurato.
    Notes:
        Per attivare verify/proxy via requests_params, abilita _build_requests_params.
    """
    api_key = os.getenv("BINANCE_API_KEY", "") # or _read_secret("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET", "") # or _read_secret("BINANCE_API_SECRET")
    requests_params = _build_requests_params()
    return Client(
        api_key=api_key,
        api_secret=api_secret,
        requests_params=requests_params,
    )


def interval_to_minutes(interval: str) -> int:
    """Converte intervalli Binance (es. 15m, 4h) in minuti.

    Input:
        interval: stringa Binance (es. 15m, 4h, 1d).
    Output:
        minuti come intero.
    Raises:
        ValueError se l'intervallo non e' supportato.
    """
    if interval not in TIMEFRAME_MINUTES:
        raise ValueError(f"Unsupported interval: {interval}")
    return TIMEFRAME_MINUTES[interval]


def _align_ms(timestamp_ms: int, interval_ms: int) -> int:
    """Allinea un timestamp al boundary dell'intervallo (floor).

    Input:
        timestamp_ms: epoch in millisecondi.
        interval_ms: durata timeframe in millisecondi.
    Output:
        timestamp allineato al boundary inferiore.
    """
    if interval_ms <= 0:
        return timestamp_ms
    return timestamp_ms - (timestamp_ms % interval_ms)


@st.cache_data(show_spinner=False)
def fetch_klines_range(asset: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Scarica klines in chunk da Binance e ritorna OHLCV ordinato.

    Input:
        asset: simbolo (es. BTCUSDT).
        interval: timeframe Binance.
        start_ms/end_ms: range in millisecondi.
    Output:
        DataFrame indicizzato per Open time con colonne OHLCV float.
    Flow:
        - Richieste paged da 1000 klines.
        - Merge chunk, sort index e rimozione duplicati.
    Notes:
        La funzione e' cached per ridurre richieste ripetute.
    """
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
    """Scarica le ultime N ore di dati e ritorna (df, ore_effettive).

    Input:
        asset, interval, hours.
    Output:
        df OHLCV e ore effettive caricate.
    Notes:
        Allinea l'intervallo ai boundary del timeframe.
    """
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
    """Scarica dati tra due date e ritorna (df, ore_effettive).

    Input:
        asset, interval, start_dt, end_dt.
    Output:
        df OHLCV e ore effettive caricate.
    Notes:
        Allinea le date ai boundary del timeframe.
    """
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
    """Calcola solo gli indicatori selezionati.

    Input:
        df: OHLCV indicizzato per data.
        indicator_flags: dict nome -> bool, per abilitazione.
        indicator_params: parametri numerici per indicatori.
    Output:
        DataFrame con colonne indicatori aggiunte.
    Flow:
        - Copia df per evitare side-effects.
        - Calcola indicatori solo se abilitati.
        - Se lunghezza insufficiente, inserisce NaN.
    Note:
        Se la serie e' troppo corta per una finestra, inserisce NaN per
        evitare errori e lascia il grafico consistente.
    """
    if df.empty:
        return df
    if not any(indicator_flags.values()):
        return df

    df_out = df.copy()
    series_len = len(df_out)

    def _nan_series() -> pd.Series:
        return pd.Series(index=df_out.index, dtype="float64")

    def _has_min_len(window: int) -> bool:
        # Some TA indicators need at least window + 1 values to avoid index errors.
        return series_len >= (int(window) + 1)

    if indicator_flags.get("atr_bands"):
        atr_window = int(indicator_params["atr_window"])
        atr_multiplier = float(indicator_params["atr_multiplier"])
        if _has_min_len(atr_window):
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
        else:
            df_out["ATR"] = _nan_series()
            df_out["ATR_MID"] = _nan_series()
            df_out["ATR_UPPER"] = _nan_series()
            df_out["ATR_LOWER"] = _nan_series()

    if indicator_flags.get("rsi_short"):
        window = int(indicator_params["rsi_short_window"])
        if _has_min_len(window):
            df_out["RSI_SHORT"] = RSIIndicator(close=df_out["Close"], window=window).rsi()
        else:
            df_out["RSI_SHORT"] = _nan_series()

    if indicator_flags.get("rsi_medium"):
        window = int(indicator_params["rsi_medium_window"])
        if _has_min_len(window):
            df_out["RSI_MED"] = RSIIndicator(close=df_out["Close"], window=window).rsi()
        else:
            df_out["RSI_MED"] = _nan_series()

    if indicator_flags.get("rsi_long"):
        window = int(indicator_params["rsi_long_window"])
        if _has_min_len(window):
            df_out["RSI_LONG"] = RSIIndicator(close=df_out["Close"], window=window).rsi()
        else:
            df_out["RSI_LONG"] = _nan_series()

    if indicator_flags.get("ema_short"):
        window = int(indicator_params["ema_short_window"])
        if _has_min_len(window):
            df_out["EMA_SHORT"] = EMAIndicator(close=df_out["Close"], window=window).ema_indicator()
        else:
            df_out["EMA_SHORT"] = _nan_series()

    if indicator_flags.get("ema_medium"):
        window = int(indicator_params["ema_medium_window"])
        if _has_min_len(window):
            df_out["EMA_MED"] = EMAIndicator(close=df_out["Close"], window=window).ema_indicator()
        else:
            df_out["EMA_MED"] = _nan_series()

    if indicator_flags.get("ema_long"):
        window = int(indicator_params["ema_long_window"])
        if _has_min_len(window):
            df_out["EMA_LONG"] = EMAIndicator(close=df_out["Close"], window=window).ema_indicator()
        else:
            df_out["EMA_LONG"] = _nan_series()

    if indicator_flags.get("kama"):
        window = int(indicator_params["kama_window"])
        pow1 = int(indicator_params["kama_pow1"])
        pow2 = int(indicator_params["kama_pow2"])
        if _has_min_len(window):
            df_out["KAMA"] = KAMAIndicator(
                close=df_out["Close"],
                window=window,
                pow1=pow1,
                pow2=pow2,
            ).kama()
        else:
            df_out["KAMA"] = _nan_series()

    if indicator_flags.get("macd"):
        short = int(indicator_params["macd_short_window"])
        long = int(indicator_params["macd_long_window"])
        signal = int(indicator_params["macd_signal_window"])
        if series_len >= (max(short, long, signal) + 1):
            macd_indicator = MACD(
                close=df_out["Close"],
                window_slow=long,
                window_fast=short,
                window_sign=signal,
            )
            df_out["MACD"] = macd_indicator.macd()
            df_out["MACD_SIGNAL"] = macd_indicator.macd_signal()
            df_out["MACD_HIST"] = macd_indicator.macd_diff()
        else:
            df_out["MACD"] = _nan_series()
            df_out["MACD_SIGNAL"] = _nan_series()
            df_out["MACD_HIST"] = _nan_series()

    return df_out


def close_atr_signals(df: pd.DataFrame, params: Dict[str, float]) -> Tuple[List[Signal], List[Signal]]:
    """Strategia Close ATR.

    Input:
        df: DataFrame con Close, ATR_UPPER, ATR_LOWER.
        params: stop_loss_percent.
    Output:
        Liste buy/sell (timestamp, prezzo).
    Flow:
        - Buy quando Close <= ATR_LOWER.
        - Sell quando Close >= ATR_UPPER o stop loss.
    """
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


def mtf_close_buy_sell_limits(
    frames: Sequence[Dict[str, object]],
    base_timeframe: str,
    conditions_required: int,
) -> Tuple[List[Signal], List[Signal]]:
    """Strategia Buy/Sell Limits multi-timeframe con soglia condizioni.

    Input:
        frames: lista dict con df per TF + flag uso indicatori e limiti RSI.
        base_timeframe: timeframe usato come timeline dei segnali.
        conditions_required: numero minimo di condizioni vere.
    Output:
        Buy/sell signals allineati al base_timeframe.
    Flow:
        - Costruisce serie booleane per ogni condizione attiva.
        - Allinea condizioni al base_timeframe (ffill su RSI/ATR).
        - Conta condizioni vere per candela e genera segnali.
    Notes:
        EMA cross e' un evento puntuale, quindi non viene ffill.
    Ogni indicatore attivo genera una condizione per TF:
    - RSI Short: confronta con limiti buy/sell
    - ATR Bands: close vs ATR_LOWER/ATR_UPPER
    - EMA Cross: incrocio EMA_SHORT vs EMA_LONG
    La logica finale usa una soglia (conditions_required) sul totale attivo.
    """
    if not frames:
        return [], []
    base_frame = next((frame for frame in frames if frame["timeframe"] == base_timeframe), None)
    if base_frame is None:
        raise ValueError(f"Base timeframe {base_timeframe} not found.")

    base_df = base_frame["df"]
    base_index = base_df.index
    base_close = base_df["Close"]

    buy_conditions: List[pd.Series] = []
    sell_conditions: List[pd.Series] = []

    for frame in frames:
        df = frame["df"]
        use_rsi = bool(frame.get("use_rsi"))
        use_atr = bool(frame.get("use_atr"))
        use_ema = bool(frame.get("use_ema"))

        # Condizione RSI per timeframe
        if use_rsi:
            rsi_buy_limit = float(frame["rsi_buy_limit"])
            rsi_sell_limit = float(frame["rsi_sell_limit"])
            if "RSI_SHORT" not in df:
                raise ValueError("RSI_SHORT indicator missing.")
            buy_cond = (df["RSI_SHORT"] <= rsi_buy_limit)
            sell_cond = (df["RSI_SHORT"] >= rsi_sell_limit)
            buy_cond = buy_cond.reindex(base_index, method="ffill").fillna(False)
            sell_cond = sell_cond.reindex(base_index, method="ffill").fillna(False)
            buy_conditions.append(buy_cond)
            sell_conditions.append(sell_cond)

        # Condizione ATR per timeframe
        if use_atr:
            if "ATR_LOWER" not in df or "ATR_UPPER" not in df:
                raise ValueError("ATR bands indicators missing.")
            buy_cond = (df["Close"] <= df["ATR_LOWER"])
            sell_cond = (df["Close"] >= df["ATR_UPPER"])
            buy_cond = buy_cond.reindex(base_index, method="ffill").fillna(False)
            sell_cond = sell_cond.reindex(base_index, method="ffill").fillna(False)
            buy_conditions.append(buy_cond)
            sell_conditions.append(sell_cond)

        # Condizione incrocio EMA (fast vs slow)
        if use_ema:
            if "EMA_SHORT" not in df or "EMA_LONG" not in df:
                raise ValueError("EMA indicators missing.")
            fast = df["EMA_SHORT"]
            slow = df["EMA_LONG"]
            buy_cross = (fast.shift(1) <= slow.shift(1)) & (fast > slow)
            sell_cross = (fast.shift(1) >= slow.shift(1)) & (fast < slow)
            buy_cross = buy_cross.reindex(base_index).fillna(False)
            sell_cross = sell_cross.reindex(base_index).fillna(False)
            buy_conditions.append(buy_cross)
            sell_conditions.append(sell_cross)

    total_conditions = len(buy_conditions)
    if total_conditions == 0:
        return [], []

    # Clamp della soglia per evitare out of range.
    required = max(1, min(int(conditions_required), total_conditions))

    # Pre-aggregate conditions to speed up per-candle checks.
    buy_counts = pd.concat(buy_conditions, axis=1).sum(axis=1).to_numpy()
    sell_counts = pd.concat(sell_conditions, axis=1).sum(axis=1).to_numpy()
    base_close_values = base_close.to_numpy()

    buy_signals: List[Signal] = []
    sell_signals: List[Signal] = []
    holding = False

    for i, ts in enumerate(base_index):
        if not holding and buy_counts[i] >= required:
            buy_signals.append((ts, float(base_close_values[i])))
            holding = True
        elif holding and sell_counts[i] >= required:
            sell_signals.append((ts, float(base_close_values[i])))
            holding = False

    return buy_signals, sell_signals


def simulate_trades(
        buy_signals: Sequence[Signal],
        sell_signals: Sequence[Signal],
        initial_wallet: float,
        fee_percent: float,
) -> pd.DataFrame:
    """Simula buy/sell con commissioni e ritorna un DataFrame trades.

    Input:
        buy_signals/sell_signals: segnali ordinati per tempo.
        initial_wallet: capitale iniziale.
        fee_percent: fee applicata a buy e sell.
    Output:
        DataFrame con dettagli dei trade.
    Flow:
        - Abbina ogni buy al primo sell successivo.
        - Calcola quantita e profitto netto con fee.
    """
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
    """Calcola metriche aggregate dei trade (profit, winrate, wallet).

    Input:
        trades_df: output di simulate_trades.
        initial_wallet: capitale iniziale per fallback.
    Output:
        dict con total_profit, win_rate, num_trades, final_wallet.
    """
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
    rsi_buy_limit: Optional[float] = None,
    rsi_sell_limit: Optional[float] = None,
) -> go.Figure:
    """Costruisce un grafico Plotly multi-panel con overlay e indicatori.

    Input:
        df: OHLCV + indicatori calcolati.
        indicator_flags: abilitazioni indicatori.
        buy_signals/sell_signals: marker da disegnare.
        rsi_buy_limit/rsi_sell_limit: linee orizzontali RSI opzionali.
    Output:
        Plotly Figure con layout multi-row.
    Flow:
        - Pannello principale: candlestick + overlay (EMA/KAMA/ATR).
        - Pannelli secondari: RSI e/o MACD se selezionati.
        - Linee RSI buy/sell solo se passate.
    """
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
        if indicator_flags.get("rsi_short") and rsi_buy_limit is not None:
            fig.add_trace(
                go.Scatter(
                    x=[df.index.min(), df.index.max()],
                    y=[rsi_buy_limit, rsi_buy_limit],
                    mode="lines",
                    line=dict(color="#2ca02c", width=1, dash="dash"),
                    name="RSI Buy Limit",
                ),
                row=next_row,
                col=1,
            )
        if indicator_flags.get("rsi_short") and rsi_sell_limit is not None:
            fig.add_trace(
                go.Scatter(
                    x=[df.index.min(), df.index.max()],
                    y=[rsi_sell_limit, rsi_sell_limit],
                    mode="lines",
                    line=dict(color="#d62728", width=1, dash="dash"),
                    name="RSI Sell Limit",
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
        default_indicators=frozenset(),
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
        default_indicators=frozenset({"atr_bands"}),
    ),
    "Buy/Sell Limits": StrategySpec(
        name="Buy/Sell Limits",
        required_indicators=frozenset(),
        default_params={
            "atr_window": 5,
            "atr_multiplier": 1.6,
            "rsi_short_window": 12,
            "rsi_buy_limit": 25,
            "rsi_sell_limit": 75,
            "conditions_required": 1,
        },
        signal_func=None,
        description=(
            "Multi-timeframe: buy/sell when at least N enabled conditions are met across TFs."
        ),
        multi_timeframe=True,
        default_indicators=frozenset({"atr_bands", "rsi_short", "ema_short", "ema_long"}),
    ),
}

# Strategy extension guide:
# - Implement a signal function that returns (buy_signals, sell_signals).
# - Register it in STRATEGIES with required indicators and default params.
