"""Costruzione delle feature dalle candele grezze.

Modulo puro e senza stato: nessun parametro appreso dai dati, nessun file da salvare accanto
al modello, nessuno scaler da ricaricare. E' una scelta deliberata -- ogni artefatto che
training e inferenza devono condividere e' un modo in cui i due possono divergere in silenzio,
e in questo progetto e' gia' successo due volte.

Tutte le feature sono **scale-free**: confrontabili tra BTC a 100.000 e DOGE a 0,2, e tra un
asset di oggi e lo stesso asset di cinque anni fa. Senza questa proprieta' un modello unico su
piu' asset non ha senso: l'ATR grezzo vale ~300 su BTC e ~0,0002 su DOGE, e il modello
imparerebbe l'identita' dell'asset invece del suo comportamento.

I prezzi restano **assoluti** qui: vengono normalizzati rispetto all'apertura della finestra in
`dataset.create_sequences`, dove la finestra e' nota. Normalizzarli prima richiederebbe una
somma cumulata sull'intera serie, che su milioni di candele accumula magnitudini enormi per poi
sottrarle di nuovo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator, TSIIndicator
from ta.volatility import AverageTrueRange

from cryptofarm.data.klines import BASE_INTERVAL, interval_to_minutes

# Colonne di prezzo: le uniche riportate all'apertura della finestra al momento di creare le
# sequenze. Tutte le altre sono gia' confrontabili cosi' come sono.
PRICE_FEATURES = ("Open", "High", "Low", "Close")

FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "RSI",
    "STOCH",
    "STOCH_S",
    "ATR",
    "TSI",
    "VOLUME",
    "TIMEFRAME",
]

RSI_WINDOW = 12
ATR_WINDOW = 6
STOCH_SMOOTH_WINDOW = 3
TSI_SLOW_WINDOW = 25
TSI_FAST_WINDOW = 13
# Mediana mobile su cui si misura il volume relativo: 96 barre sono 8 ore su 5m, 4 giorni su 1h.
VOLUME_WINDOW = 96

# Candele minime per calcolare gli indicatori. Sotto questa soglia `ta` non degrada: solleva un
# IndexError dall'interno del calcolo dell'ATR. La dashboard puo' passare finestre corte, quindi
# il caso va gestito qui invece di propagare un errore incomprensibile.
MIN_CANDLES = TSI_SLOW_WINDOW + TSI_FAST_WINDOW + 1


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge gli indicatori nelle loro unita' native. La normalizzazione viene dopo."""
    result = df.copy()

    result["RSI"] = RSIIndicator(close=result["Close"], window=RSI_WINDOW).rsi()
    result["ATR"] = AverageTrueRange(
        high=result["High"], low=result["Low"], close=result["Close"], window=ATR_WINDOW
    ).average_true_range()

    stochastic = StochasticOscillator(
        high=result["High"],
        low=result["Low"],
        close=result["Close"],
        window=RSI_WINDOW,
        smooth_window=STOCH_SMOOTH_WINDOW,
    )
    result["STOCH"] = stochastic.stoch()
    result["STOCH_S"] = stochastic.stoch_signal()
    result["TSI"] = TSIIndicator(close=result["Close"], window_slow=TSI_SLOW_WINDOW, window_fast=TSI_FAST_WINDOW).tsi()

    return result


def normalize_indicators(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Porta ogni indicatore su una scala confrontabile tra asset, epoche e timeframe.

    - ATR in percentuale del Close: e' l'unico indicatore in unita' di prezzo, e lasciato grezzo
      da solo impedirebbe l'addestramento multi-asset.
    - RSI, STOCH e STOCH_S da [0, 100] a [-1, 1]; TSI da [-100, 100] a [-1, 1]. Sono gia'
      limitati per costruzione, ma su range uno o due ordini di grandezza sopra le altre feature:
      riscalati pesano quanto le altre invece di dominare.
    - VOLUME come logaritmo del rapporto con la propria mediana mobile: il volume assoluto non
      dice nulla fuori contesto, il volume rispetto al normale recente dice molto.
    - TIMEFRAME come feature esplicita, cosi' un modello unico su 5m/15m/30m/1h puo'
      condizionarsi sulla granularita' invece di mediare comportamenti diversi.
    """
    result = df.copy()

    close = result["Close"].replace(0, np.nan)
    result["ATR"] = (result["ATR"] / close * 100).fillna(0)

    for column in ("RSI", "STOCH", "STOCH_S"):
        result[column] = (result[column] - 50.0) / 50.0
    result["TSI"] = result["TSI"] / 100.0

    if "Volume" in result.columns:
        reference = result["Volume"].rolling(VOLUME_WINDOW, min_periods=1).median()
        result["VOLUME"] = np.log1p(result["Volume"] / reference.replace(0, np.nan)).fillna(0)
    else:
        result["VOLUME"] = 0.0

    # log2 del rapporto col timeframe base: 5m -> 0, 15m -> 1.58, 30m -> 2.58, 1h -> 3.58.
    # Diviso per 4 sta nello stesso ordine di grandezza delle altre feature.
    base_minutes = interval_to_minutes(BASE_INTERVAL)
    result["TIMEFRAME"] = np.log2(interval_to_minutes(interval) / base_minutes) / 4.0

    return result


def build_feature_frame(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Da candele OHLCV grezze al frame di feature, prezzi ancora assoluti.

    Le righe iniziali in cui gli indicatori non si sono ancora scaldati vengono scartate invece
    che riempite con zeri: uno zero in RSI e' un valore plausibile e sbagliato, che il modello
    non ha modo di distinguere da un RSI davvero basso.
    """
    if len(df) < MIN_CANDLES:
        return df.iloc[:0].reindex(columns=list(df.columns) + FEATURES).drop_duplicates()

    result = add_technical_indicators(df)
    result = normalize_indicators(result, interval)
    return result.dropna(subset=[column for column in FEATURES if column in result.columns])
