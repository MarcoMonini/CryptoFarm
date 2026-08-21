"""Indicatori tecnici sulle candele, condivisi da tutte le strategie.

Estratto da `simulator.py` senza modifiche. `add_technical_indicator` calcola l'intera tabella
in una volta; `calculate_latest_indicators` ricalcola solo la finestra attorno a una candela,
ed e' cio' che rende `simulate_candles` lento."""

import pandas as pd
import streamlit as st
from ta.momentum import KAMAIndicator, RSIIndicator, StochasticOscillator, TSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange


@st.cache_data
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
    # Calcolo del SAR utilizzando la libreria "ta" (PSARIndicator)
    # sar_indicator = PSARIndicator(
    #     high=df_copy['High'],
    #     low=df_copy['Low'],
    #     close=df_copy['Close'],
    #     step=step,
    #     max_step=max_step
    # )
    # df_copy['PSAR'] = sar_indicator.psar()
    # df_copy['PSARVP'] = df_copy['PSAR'] / df_copy['Close']

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
    df_copy["Upper_Band"][:atr_window] = None
    df_copy["Lower_Band"][:atr_window] = None

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

    # ATR
    atr_indicator = AverageTrueRange(
        high=temp_df["High"], low=temp_df["Low"], close=temp_df["Close"], window=atr_window
    )
    atr = atr_indicator.average_true_range()

    # EMA (Media Mobile per le Rolling ATR Bands)
    ema_indicator = EMAIndicator(close=temp_df["Close"], window=atr_window)
    ema = ema_indicator.ema_indicator()

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
