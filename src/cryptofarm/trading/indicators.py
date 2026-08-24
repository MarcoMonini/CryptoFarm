"""Indicatori tecnici sulle candele, condivisi da tutte le strategie.

Estratto da `simulator.py` senza modifiche. `add_technical_indicator` calcola l'intera tabella
in una volta; `calculate_latest_indicators` ricalcola solo la finestra attorno a una candela,
ed e' cio' che rende `simulate_candles` lento."""

import numpy as np
import pandas as pd
import streamlit as st
from ta.momentum import KAMAIndicator, RSIIndicator, StochasticOscillator, TSIIndicator
from ta.trend import EMAIndicator, PSARIndicator
from ta.volatility import AverageTrueRange


# I parametri arrivano dai widget della barra laterale: ogni combinazione e' una voce di
# cache, e senza tetto muovere gli slider fa crescere la memoria fino al riavvio del processo.
@st.cache_data(max_entries=32)
def add_technical_indicator(
    df,
    step=0.02,
    max_step=0.4,
    rsi_window=12,
    rsi_window2=24,
    rsi_window3=36,
    macd_long_window=26,
    macd_short_window=12,
    macd_signal_window=9,
    ema_window=10,
    ema_window2=50,
    ema_window3=200,
    atr_window=6,
    atr_multiplier=1.6,
    kama_pow1=2,
    kama_pow2=30,
):
    df_copy = df.copy()
    # Calcolo del SAR utilizzando la libreria "ta" (PSARIndicator).
    # Era commentato, ma `atr_buy_sell_simulation` e `close_atr_buy_sell_simulation` leggono
    # `PSAR` per la loro condizione di stop-loss: senza questa colonna sollevavano `KeyError`
    # non appena quel ramo veniva raggiunto, e "Close ATR" era una voce di menu che si rompeva.
    sar_indicator = PSARIndicator(
        high=df_copy["High"], low=df_copy["Low"], close=df_copy["Close"], step=step, max_step=max_step
    )
    df_copy["PSAR"] = sar_indicator.psar()
    df_copy["PSARVP"] = df_copy["PSAR"] / df_copy["Close"]

    # Calcolo dell'RSI
    rsi_indicator = RSIIndicator(close=df_copy["Close"], window=rsi_window)
    df_copy["RSI"] = rsi_indicator.rsi()
    rsi_indicator = RSIIndicator(close=df_copy["Close"], window=rsi_window2)
    df_copy["RSI2"] = rsi_indicator.rsi()
    rsi_indicator = RSIIndicator(close=df_copy["Close"], window=rsi_window3)
    df_copy["RSI3"] = rsi_indicator.rsi()
    # SMA dell'RSI
    # df_copy['RSI_S'] = df_copy['RSI'].rolling(window=3).mean()
    # df_copy['RSI2_S'] = df_copy['RSI2'].rolling(window=3).mean()
    # df_copy['RSI3_S'] = df_copy['RSI3'].rolling(window=3).mean()

    # Calcolo del MACD
    # macd_indicator = MACD(
    #     close=df_copy['Close'],
    #     window_slow=macd_long_window,
    #     window_fast=macd_short_window,
    #     window_sign=macd_signal_window
    # )
    # df_copy['MACD_L'] = macd_indicator.macd()
    # df_copy['MACD_S'] = macd_indicator.macd_signal()
    # df_copy['MACD'] = macd_indicator.macd_diff()  # Istogramma (differenza tra MACD e Signal Line)
    # # Calcolo del MACD normalizzato come percentuale del prezzo
    # df_copy['MACD_S'] = df_copy['MACD_S'] / df_copy['Close'] * 100  # normalizzato
    # df_copy['MACD_L'] = df_copy['MACD_L'] / df_copy['Close'] * 100  # normalizzato
    # df_copy['MACD'] = df_copy['MACD'] / df_copy['Close'] * 100  # normalizzato

    # ATR
    atr_indicator = AverageTrueRange(
        high=df_copy["High"], low=df_copy["Low"], close=df_copy["Close"], window=atr_window
    )
    df_copy["ATR"] = atr_indicator.average_true_range()

    # EMA (Media Mobile per le Rolling ATR Bands)
    ema_indicator = EMAIndicator(close=df_copy["Close"], window=ema_window)
    df_copy["EMA20"] = ema_indicator.ema_indicator()
    ema_indicator = EMAIndicator(close=df_copy["Close"], window=ema_window2)
    df_copy["EMA50"] = ema_indicator.ema_indicator()
    ema_indicator = EMAIndicator(close=df_copy["Close"], window=ema_window3)
    df_copy["EMA100"] = ema_indicator.ema_indicator()
    emao_indicator = EMAIndicator(close=df_copy["Open"], window=ema_window)
    df_copy["EMA200"] = emao_indicator.ema_indicator()

    kama_indicator = KAMAIndicator(close=df_copy["Close"], window=ema_window, pow1=kama_pow1, pow2=kama_pow2)
    df_copy["KAMA"] = kama_indicator.kama()

    # Rolling ATR Bands
    # df_copy['Upper_Band'] = df_copy['EMA20'] + atr_multiplier * df_copy['ATR']
    # df_copy['Lower_Band'] = df_copy['EMA20'] - atr_multiplier * df_copy['ATR']
    df_copy["Upper_Band"] = df_copy["KAMA"] + atr_multiplier * df_copy["ATR"]
    df_copy["Lower_Band"] = df_copy["KAMA"] - atr_multiplier * df_copy["ATR"]
    # `df["col"][:n] = None` e' un assegnamento concatenato: con il Copy-on-Write di pandas 3.0
    # scriverebbe su una copia intermedia e le prime barre resterebbero valorizzate, in silenzio.
    df_copy.iloc[:atr_window, df_copy.columns.get_loc("Upper_Band")] = None
    df_copy.iloc[:atr_window, df_copy.columns.get_loc("Lower_Band")] = None

    # STOCASTICO
    stoch_indicator = StochasticOscillator(
        high=df_copy["High"], low=df_copy["Low"], close=df_copy["Close"], window=rsi_window, smooth_window=3
    )
    df_copy["STOCH"] = stoch_indicator.stoch()
    df_copy["STOCH_S"] = stoch_indicator.stoch_signal()

    tsi_indicator = TSIIndicator(
        close=df_copy["Close"],
        window_slow=25,
        window_fast=13,
    )
    df_copy["TSI"] = tsi_indicator.tsi()

    return df_copy


def _atr_ema(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """ATR di Wilder ed EMA, con le stesse formule di `ta`, senza costruire oggetti pandas.

    `simulate_candles` chiama questo calcolo dieci volte per candela: con `ta.AverageTrueRange` e
    `ta.EMAIndicator` ogni chiamata costruiva una manciata di Series, ed era li' che se ne andava
    quasi tutto il tempo del simulatore.

    Le formule sono quelle di `ta` 0.11 riga per riga:
    `AverageTrueRange._run` semina l'ATR con la media dei primi `window` true range e poi applica
    lo smorzamento di Wilder, lasciando zero prima; `_ema` e' `ewm(span=window, adjust=False)` con
    `min_periods=window`, quindi NaN prima. Il true range della prima barra vale `high - low`,
    perche' `DataFrame.max(axis=1)` scarta i NaN che nascono dal `close.shift(1)`.
    """
    n = len(close)
    previous_close = np.empty(n)
    previous_close[0] = np.nan
    previous_close[1:] = close[:-1]
    with np.errstate(invalid="ignore"):
        true_range = np.nanmax(
            np.vstack([high - low, np.abs(high - previous_close), np.abs(low - previous_close)]), axis=0
        )

    atr = np.zeros(n)
    atr[window - 1] = true_range[:window].mean()
    for i in range(window, n):
        atr[i] = (atr[i - 1] * (window - 1) + true_range[i]) / window

    alpha = 2.0 / (window + 1.0)
    ema = np.empty(n)
    ema[0] = close[0]
    for i in range(1, n):
        ema[i] = alpha * close[i] + (1 - alpha) * ema[i - 1]
    ema[: window - 1] = np.nan

    return atr, ema


def latest_bands(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int, multiplier: float
) -> tuple[float, float] | tuple[None, None]:
    """Solo l'ultima banda superiore e inferiore della finestra: cio' che serve a `simulate_candles`.

    `(None, None)` quando la finestra e' piu' corta di `window`, come faceva
    `calculate_latest_indicators` restituendo colonne a None.
    """
    if len(close) < window:
        return None, None
    atr, ema = _atr_ema(high, low, close, window)
    return ema[-1] + multiplier * atr[-1], ema[-1] - multiplier * atr[-1]


def calculate_latest_indicators(df: pd.DataFrame, i: int, atr_window: int = 14, atr_multiplier: float = 2.4):
    """
    Calcola SOLO l'ultimo valore di RSI e MACD sulla candela 'i'
    del DataFrame 'df', ritagliando una finestra minima attorno a 'i'.
    """

    # needed_bars = max(atr_window, macd_short_window, macd_long_window, macd_signal_window) + 5
    needed_bars = atr_window + 12
    start_idx = max(0, i - needed_bars + 1)
    end_idx = i + 1  # slice in pandas: end non è incluso
    # Estrai la finestra di dati
    temp_df = df.iloc[start_idx:end_idx].copy()

    # Se la finestra è troppo corta, restituiamo df_copy con tutte None
    if len(temp_df) < atr_window:
        df_copy = df.copy()
        df_copy["ATR"] = None
        df_copy["EMA20"] = None
        df_copy["Upper_Band"] = None
        df_copy["Lower_Band"] = None
        df_copy["PSAR"] = None
        return df_copy

    atr, ema = _atr_ema(temp_df["High"].to_numpy(), temp_df["Low"].to_numpy(), temp_df["Close"].to_numpy(), atr_window)

    # kama_indicator = KAMAIndicator(close=temp_df['Close'],
    #                                window=atr_window,
    #                                pow1=2,
    #                                pow2=30)
    # kama = kama_indicator.kama()

    # temp_df['Upper_Band'] = ema + atr_multiplier * atr
    # temp_df['Lower_Band'] = ema - atr_multiplier * atr

    temp_df["Upper_Band"] = ema + atr_multiplier * atr
    temp_df["Lower_Band"] = ema - atr_multiplier * atr

    return temp_df
