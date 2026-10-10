"""The ten zones of `reports/zones_study_project.html` (Top 10), as signals the simulator can mark on a chart.

A zone fires on a candle when one indicator, read at that candle's close, sits in a tail of its own history: the
bottom or the top `ZONE_TAIL` percent. The tail's edge is a quantile of the indicator over the pair's **2017-08 to
2020-12** candles in the local store, at the zone's timeframe — the period the studies chose their zones on, so the
edge is the one the measured numbers were priced against. Every zone was measured long, a candle after the signal and
held `tieni` candles; nothing here trades, it only says which candles qualify.

Each zone is valid on one timeframe only: nine on 15m, the Bollinger width also on 5m. On any other timeframe there is
nothing to mark, and `validi` says so with an empty list. Without the pair in the store there is no edge, and the
zone marks nothing rather than guess one from the candles on screen.

**What can be drawn on the candles.** A zone on a distance from a price line — the window's high, KAMA, VWAP — has an
exact twin on the price: the indicator is `log(close / X)`, so the line is `X = close * exp(-value)` and the edge is
`X * exp(edge)`. Reading `X` back from the indicator itself, rather than recomputing it, keeps the line and the signal
the same object. The others — the return over the window, the EMA's slope, the bar's range, the upper wick, the
Bollinger width — are not prices and go in a panel with their edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from cryptofarm.data import klines
from cryptofarm.trading import candle_features, project_features

STORICO = (pd.Timestamp("2017-08-01"), pd.Timestamp("2021-01-01"))  # the studies' discovery period
QUOTE = ("FDUSD", "USDT", "USDC", "BUSD", "EUR", "USD")  # stripped from the page's symbol to find the store's pair
WINDOW_FIXED = 6  # any window computes a column that reads none (bar range, wick): `zones.FIXED`


@dataclass(frozen=True)
class Segnale:
    nome: str  # what the legend and the selector say
    timeframe: str
    colonna: str  # a column of `candle_features` or of `project_features`
    finestra: str | None  # the `config` parameter holding its window; None for a column that reads none
    alto: bool  # True: the top tail fires; False: the bottom one
    tieni: int  # candles held in the measurement, for the reader: nothing here trades
    prezzo: bool  # a distance from a price line, drawn on the candles; otherwise a panel


# Ordered as the report's Top 10, by gross per trade on 2021-01 → 2025-06.
SEGNALI: dict[str, Segnale] = {
    "bb_width": Segnale("#1 Bollinger width, top tail", "15m", "bollinger_width", "ZONE_BB_WINDOW", True, 96, False),
    "window_high": Segnale(
        "#2 Distance from the window high, bottom tail",
        "15m",
        "distance_from_window_high_pct",
        "ZONE_HIGH_WINDOW",
        False,
        24,
        True,
    ),
    "bar_range": Segnale("#3 Bar range, top tail", "15m", "bar_range_pct", None, True, 96, False),
    "upper_wick": Segnale("#4 Upper wick, top tail", "15m", "upper_wick_pct", None, True, 96, False),
    "kama_fast": Segnale(
        "#5 Distance from fast KAMA, bottom tail", "15m", "distance_from_kama_pct", "ZONE_KAMA_FAST", False, 24, True
    ),
    "kama_slow": Segnale(
        "#6 Distance from slow KAMA, bottom tail", "15m", "distance_from_kama_pct", "ZONE_KAMA_SLOW", False, 96, True
    ),
    "bb_width_5m": Segnale("#7 Bollinger width, top tail", "5m", "bollinger_width", "ZONE_BB_WINDOW", True, 96, False),
    "window_return": Segnale(
        "#8 Return over the window, bottom tail",
        "15m",
        "cum_log_return_window",
        "ZONE_RETURN_WINDOW",
        False,
        6,
        False,
    ),
    "ema_slope": Segnale("#9 EMA slope, bottom tail", "15m", "ema_slope", "ZONE_SLOPE_WINDOW", False, 6, False),
    "vwap": Segnale(
        "#10 Distance from VWAP, bottom tail", "15m", "distance_from_vwap_pct", "ZONE_VWAP_WINDOW", False, 6, True
    ),
}


def validi(intervallo: str) -> list[str]:
    """The zones measured at this timeframe, in the Top 10's order; empty where none was."""
    return [chiave for chiave, s in SEGNALI.items() if s.timeframe == intervallo]


def scelti(valori: dict) -> list[str]:
    """The zones the page asked for (`ZONE_SIGNALS`), all the valid ones by default, never one from another
    timeframe."""
    ammessi = validi(str(valori.get("INTERVALLO", "")))
    voluti = valori.get("ZONE_SIGNALS")
    return ammessi if voluti is None else [c for c in voluti if c in ammessi]


def base_di(simbolo: str) -> str:
    """BTCUSDC → BTC: the page's pair, without the quote, to find the store's `BTCUSDT`."""
    for quote in QUOTE:
        if simbolo.endswith(quote) and len(simbolo) > len(quote):
            return simbolo[: -len(quote)]
    return simbolo


def finestra(segnale: Segnale, valori: dict) -> int:
    return int(valori[segnale.finestra]) if segnale.finestra else WINDOW_FIXED


def _modulo(colonna: str):
    return project_features if colonna in project_features.LABELS else candle_features


def valore(df: pd.DataFrame, segnale: Segnale, w: int) -> pd.Series:
    """The zone's indicator on the page's candles (capitalised OHLCV), at window `w`."""
    frame = df[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
    return _modulo(segnale.colonna).features(frame, w)[segnale.colonna]


@lru_cache(maxsize=4)
def _storia(base: str, timeframe: str) -> pd.DataFrame | None:
    """The pair's discovery-period candles at `timeframe` from the store, or None when it has none."""
    df = klines.load_klines(f"{base}USDT", timeframe, clip=False)
    df = df[(df.index >= STORICO[0]) & (df.index < STORICO[1])]
    return df if len(df) > 1000 else None


@lru_cache(maxsize=64)
def soglia(base: str, chiave: str, w: int, coda: float) -> float | None:
    """The edge of the zone's tail: the `coda` percent quantile (or 100 - `coda`, for a top tail) of its indicator on
    the pair's 2017-2020 candles. None without the pair in the store."""
    segnale = SEGNALI[chiave]
    storia = _storia(base, segnale.timeframe)
    if storia is None:
        return None
    q = 1 - coda / 100 if segnale.alto else coda / 100
    return float(np.nanquantile(valore(storia, segnale, w).to_numpy(), q))


def scatta(x: pd.Series, segnale: Segnale, bordo: float) -> pd.Series:
    """Where the indicator is in the tail: above the edge for a top tail, below it for a bottom one."""
    return x > bordo if segnale.alto else x < bordo


def linea(df: pd.DataFrame, x: pd.Series) -> pd.Series:
    """The price line a distance is measured from: `close * exp(-distance)`."""
    return df["Close"] * np.exp(-x)


def _selfcheck() -> None:
    """Symbols, the default selection, the edge's side, and the price line read back from a distance."""
    assert base_di("BTCUSDC") == "BTC" and base_di("ETHUSDT") == "ETH" and base_di("XRPEUR") == "XRP"
    assert base_di("USDT") == "USDT", "a bare quote is not stripped to nothing"
    assert len(validi("15m")) == 9 and validi("5m") == ["bb_width_5m"] and validi("1h") == []
    assert scelti({"INTERVALLO": "15m"}) == validi("15m")
    assert scelti({"INTERVALLO": "5m", "ZONE_SIGNALS": ["vwap", "bb_width_5m"]}) == ["bb_width_5m"]
    assert scelti({"INTERVALLO": "15m", "ZONE_SIGNALS": []}) == []
    x = pd.Series([-0.03, 0.0, 0.03])
    assert list(scatta(x, SEGNALI["vwap"], -0.02)) == [True, False, False]
    assert list(scatta(x, SEGNALI["bb_width"], 0.02)) == [False, False, True]
    # The KAMA read back from its distance is ta's KAMA, to the float.
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    df = pd.DataFrame(
        {"Open": close, "High": close * 1.002, "Low": close * 0.998, "Close": close, "Volume": 1.0},
        index=pd.date_range("2024", periods=300, freq="15min"),
    )
    from ta.momentum import KAMAIndicator

    kama = KAMAIndicator(df.Close, window=6).kama()
    back = linea(df, valore(df, SEGNALI["kama_fast"], 6))
    assert np.allclose(back.iloc[20:], kama.iloc[20:])
    assert soglia("NOSUCHPAIR", "vwap", 12, 1.0) is None


if __name__ == "__main__":
    _selfcheck()
    print("ok")
