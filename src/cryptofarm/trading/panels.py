"""Chi usa cosa: il registro che lega ogni strategia ai suoi indicatori e parametri.

La pagina mostrava sempre tutto -- quindici colonne di indicatori nella barra laterale e una
dozzina di tracce nel grafico -- qualunque strategia fosse selezionata. "Trend Zones" legge due
medie e si vedeva accanto a StochRSI, TSI e tre RSI che non tocca; "Green Candles" non legge
nessun indicatore e li mostrava tutti.

Qui sta la mappa, in forma di dati. `simulator.py` la legge e si limita a disporre widget e
tracce: non contiene piu' ne' la catena di `if strategia == ...` ne' l'elenco fisso delle tracce.

**La mappa e' verificata a mano, non dedotta.** Uno scan statico delle colonne lette non basta:
`close_bullish_ema_simulation` prende le medie con `(df[c].to_numpy() for c in ("EMA20", ...))`,
uno slice variabile che nessuna analisi banale dell'albero sintattico vede, e quella strategia
sarebbe finita nel registro senza le sue tre medie.

**Le dipendenze contano piu' delle apparenze.** `Upper_Band`/`Lower_Band` sono
`KAMA +/- moltiplicatore * ATR`, e `KAMA` usa `ema_window`, cioe' il parametro "EMA Short". Una
strategia a bande come "Close ATR" dipende quindi da EMA Short, dalle due potenze di KAMA e dai
due parametri dell'ATR, anche se di medie non ne disegna nessuna: i `parametri` di `bande_atr` li
elencano tutti, altrimenti la barra laterale nasconderebbe un campo che muove le bande.

## La palette

Tre tinte, non di piu': sono le uniche tre che passano tutte le coppie del validatore su
superficie scura (blu/arancio/acqua, ΔE normale peggiore 20,9, ΔE per daltonismo 9,4). Il quarto
slot -- il giallo -- contro l'arancio scende a 4,8 per deuteranopia e 10,6 a colori pieni, cioe'
due serie che non si distinguono. Quando una vista ha bisogno di piu' di tre linee, la quarta
riusa una tinta cambiando tratteggio, mai una tinta nuova.

Le stesse tre tinte tornano in strategie diverse -- il canale di Donchian e' arancio dove non ci
sono bande, le bande sono arancio dove non c'e' un canale -- perche' due famiglie che non
compaiono mai nella stessa vista possono condividere una tinta senza ambiguita'. Verde e rosso
restano riservati allo stato (candele, acquisti, vendite) e non sono mai il colore di un
indicatore.

**L'acquamarina vive solo nei pannelli.** Sopra le candele e' troppo vicina al verde del corpo
rialzista: una linea di indicatore e una candela in salita si leggono come la stessa cosa. Gli
overlay usano quindi solo blu e arancio, e il terzo slot resta agli oscillatori, che stanno nel
loro riquadro dove le candele non ci sono. Un test lo tiene fermo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from cryptofarm.trading import strategies, strategies_ls
from cryptofarm.trading.indicators_extra import ExtraCache

# Le tre tinte categoriche, validate insieme sulla superficie scura di Streamlit.
BLU = "#3987e5"
ARANCIO = "#d95926"
ACQUA = "#199e70"
# Stato: non sono mai il colore di un indicatore.
RIALZO = "#26a69a"
RIBASSO = "#ef5350"


@dataclass(frozen=True)
class Traccia:
    """Una serie del grafico: da dove viene e come si disegna."""

    serie: str
    nome: str
    colore: str
    tratteggio: str | None = None
    larghezza: float = 1.5
    modo: str = "lines"
    simbolo: str = "circle"


@dataclass(frozen=True)
class Indicatore:
    """Un indicatore: i suoi parametri, dove si disegna, come si ricavano le sue serie.

    `pannello` a `None` significa sovrapposto alle candele; altrimenti e' il titolo del riquadro
    che l'indicatore si prende sotto il grafico principale.
    """

    etichetta: str
    parametri: tuple[str, ...]
    pannello: str | None
    serie: Callable[[pd.DataFrame, ExtraCache, dict], dict[str, pd.Series]]
    tracce: tuple[Traccia, ...]


@dataclass(frozen=True)
class Strategia:
    """Una voce del menu: cosa esegue, quali indicatori usa, quali parametri le servono.

    `esegui` prende il frame, la cache degli indicatori e i valori letti dalla barra laterale, e restituisce
    `(buy_signals, sell_signals)`. E' una chiamata scritta per esteso invece di una tabella di
    corrispondenze fra nomi: le firme delle strategie non sono uniformi, e una riga esplicita si
    legge meglio di un meccanismo che le uniformi.

    `parametri` sono i suoi soli: le finestre degli indicatori arrivano gia' da
    `Indicatore.parametri`, e la barra laterale unisce i due insiemi senza ripetizioni.
    """

    indicatori: tuple[str, ...]
    esegui: Callable[[pd.DataFrame, ExtraCache, dict], tuple[list, list]]
    parametri: tuple[str, ...] = ()
    note: str = ""


def _colonne(*nomi: str):
    """Le serie sono gia' colonne del frame: la sorgente piu' comune.

    Le strategie nuove invece non hanno colonne nel frame -- i loro indicatori li calcola
    `ExtraCache` sul momento, con la finestra scelta nella barra laterale -- e per quelle la
    funzione legge la cache. Da qui la firma a tre argomenti, uguale per tutti.
    """
    return lambda df, cache, valori: {nome: df[nome] for nome in nomi}


def _serie(indice: pd.Index, **colonne) -> dict[str, pd.Series]:
    """Gli array numpy che tornano da `ExtraCache`, allineati all'indice delle candele."""
    return {nome: pd.Series(valori, index=indice) for nome, valori in colonne.items()}


# -------------------------------------------------------------------------------------------------
# Gli indicatori che `indicators.add_technical_indicator` gia' produce
# -------------------------------------------------------------------------------------------------

INDICATORI: dict[str, Indicatore] = {
    "medie": Indicatore(
        etichetta="Medie esponenziali",
        parametri=("EMA_SHORT", "EMA_MEDIUM", "EMA_LONG"),
        pannello=None,
        serie=_colonne("EMA20", "EMA50", "EMA100"),
        tracce=(
            Traccia("EMA20", "EMA corta", BLU, larghezza=1.0),
            Traccia("EMA50", "EMA media", BLU, tratteggio="dot", larghezza=1.5),
            Traccia("EMA100", "EMA lunga", BLU, tratteggio="dash", larghezza=2.0),
        ),
    ),
    "medie_trend": Indicatore(
        # "Trend Zones" confronta la corta con la lunga: la media di mezzo non la tocca.
        etichetta="Medie corta e lunga",
        parametri=("EMA_SHORT", "EMA_LONG"),
        pannello=None,
        serie=_colonne("EMA20", "EMA100"),
        tracce=(
            Traccia("EMA20", "EMA corta", BLU, larghezza=1.0),
            Traccia("EMA100", "EMA lunga", BLU, tratteggio="dash", larghezza=2.0),
        ),
    ),
    "bande_atr": Indicatore(
        # KAMA usa `ema_window`, e le bande sono KAMA +/- moltiplicatore * ATR: i cinque
        # parametri servono tutti, anche se il nome ne cita uno solo.
        etichetta="Bande ATR su KAMA",
        parametri=("ATR_WINDOW", "ATR_MULTIPLIER", "KAMA_POW1", "KAMA_POW2", "EMA_SHORT"),
        pannello=None,
        serie=_colonne("KAMA", "Upper_Band", "Lower_Band"),
        tracce=(
            Traccia("KAMA", "KAMA", ARANCIO, larghezza=1.5),
            Traccia("Upper_Band", "Banda superiore", ARANCIO, tratteggio="dash", larghezza=1.0),
            Traccia("Lower_Band", "Banda inferiore", ARANCIO, tratteggio="dash", larghezza=1.0),
        ),
    ),
    "psar": Indicatore(
        etichetta="Parabolic SAR",
        parametri=(),
        pannello=None,
        serie=_colonne("PSAR"),
        tracce=(Traccia("PSAR", "PSAR", BLU, modo="markers", simbolo="circle", larghezza=0.0),),
    ),
    "estremi": Indicatore(
        # Non e' letto da nessuna strategia: e' il riferimento visivo dei massimi e minimi
        # relativi, e resta disponibile nella panoramica senza strategia selezionata.
        etichetta="Massimi e minimi relativi",
        parametri=("PIVOT_WINDOW",),
        pannello=None,
        serie=lambda df, cache, valori: {},
        tracce=(),
    ),
    "rsi": Indicatore(
        etichetta="RSI",
        parametri=("RSI_SHORT", "RSI_MEDIUM", "RSI_LONG"),
        pannello="RSI",
        serie=_colonne("RSI", "RSI2", "RSI3"),
        tracce=(
            Traccia("RSI", "RSI corto", BLU, larghezza=1.5),
            Traccia("RSI2", "RSI medio", BLU, tratteggio="dot", larghezza=1.0),
            Traccia("RSI3", "RSI lungo", BLU, tratteggio="dash", larghezza=1.0),
        ),
    ),
    "stocastico": Indicatore(
        etichetta="Stocastico",
        parametri=("RSI_SHORT",),
        pannello="Stocastico",
        serie=_colonne("STOCH", "STOCH_S"),
        tracce=(
            Traccia("STOCH", "Stocastico", ARANCIO, larghezza=1.5),
            Traccia("STOCH_S", "Segnale", ARANCIO, tratteggio="dash", larghezza=1.0),
        ),
    ),
    "tsi": Indicatore(
        etichetta="True Strength Index",
        parametri=(),
        pannello="TSI",
        serie=_colonne("TSI"),
        tracce=(Traccia("TSI", "TSI", ACQUA, larghezza=1.5),),
    ),
    "donchian": Indicatore(
        etichetta="Canale di Donchian",
        parametri=("DONCHIAN_CHANNEL",),
        pannello=None,
        serie=lambda df, cache, v: _serie(
            df.index, **dict(zip(("canale_alto", "canale_basso"), cache.donchian(int(v["DONCHIAN_CHANNEL"]))))
        ),
        tracce=(
            Traccia("canale_alto", "Canale superiore", ARANCIO, larghezza=1.5),
            Traccia("canale_basso", "Canale inferiore", ARANCIO, tratteggio="dash", larghezza=1.0),
        ),
    ),
    "media_regime": Indicatore(
        etichetta="Media di regime",
        parametri=("REGIME_EMA",),
        pannello=None,
        serie=lambda df, cache, v: (
            _serie(df.index, regime=cache.ema(int(v["REGIME_EMA"]))) if int(v["REGIME_EMA"]) else {}
        ),
        tracce=(Traccia("regime", "EMA di regime", BLU, tratteggio="dash", larghezza=2.0),),
    ),
    "bollinger": Indicatore(
        etichetta="Bande di Bollinger",
        parametri=("BB_WINDOW", "BB_DEV"),
        pannello=None,
        serie=lambda df, cache, v: _serie(
            df.index,
            **dict(
                zip(
                    ("bb_alta", "bb_bassa", "bb_media"),
                    cache.bollinger(int(v["BB_WINDOW"]), float(v["BB_DEV"])),
                )
            ),
        ),
        tracce=(
            Traccia("bb_alta", "Bollinger superiore", BLU, larghezza=1.0),
            Traccia("bb_media", "Bollinger media", BLU, tratteggio="dot", larghezza=1.0),
            Traccia("bb_bassa", "Bollinger inferiore", BLU, larghezza=1.0),
        ),
    ),
    "keltner": Indicatore(
        # Il canale di Keltner usa la stessa finestra ATR dell'uscita a trailing.
        etichetta="Canale di Keltner",
        parametri=("KC_WINDOW", "KC_MULTIPLIER", "TRAIL_ATR_WINDOW"),
        pannello=None,
        serie=lambda df, cache, v: _serie(
            df.index,
            **dict(
                zip(
                    ("kc_alta", "kc_bassa"),
                    cache.keltner(int(v["KC_WINDOW"]), int(v["TRAIL_ATR_WINDOW"]), float(v["KC_MULTIPLIER"])),
                )
            ),
        ),
        tracce=(
            Traccia("kc_alta", "Keltner superiore", ARANCIO, tratteggio="dash", larghezza=1.0),
            Traccia("kc_bassa", "Keltner inferiore", ARANCIO, tratteggio="dash", larghezza=1.0),
        ),
    ),
    "ichimoku": Indicatore(
        etichetta="Ichimoku",
        parametri=("ICHIMOKU_FAST", "ICHIMOKU_SLOW", "ICHIMOKU_SPAN"),
        pannello=None,
        serie=lambda df, cache, v: _serie(
            df.index,
            **dict(
                zip(
                    ("tenkan", "kijun", "span_a", "span_b"),
                    cache.ichimoku(int(v["ICHIMOKU_FAST"]), int(v["ICHIMOKU_SLOW"]), int(v["ICHIMOKU_SPAN"])),
                )
            ),
        ),
        tracce=(
            Traccia("tenkan", "Tenkan", BLU, larghezza=1.5),
            Traccia("kijun", "Kijun", ARANCIO, larghezza=1.5),
            Traccia("span_a", "Nuvola A", BLU, tratteggio="dot", larghezza=1.0),
            Traccia("span_b", "Nuvola B", ARANCIO, tratteggio="dot", larghezza=1.0),
        ),
    ),
    "bande_kama": Indicatore(
        # Non sono le `Upper_Band`/`Lower_Band` del frame: stessa forma, ma finestra e
        # moltiplicatore sono quelli della strategia di ritorno alla media.
        etichetta="Bande di ritorno alla media",
        parametri=("REVERSION_KAMA_WINDOW", "REVERSION_BAND_MULT", "TRAIL_ATR_WINDOW"),
        pannello=None,
        serie=lambda df, cache, v: _serie(
            df.index,
            kama=cache.kama(int(v["REVERSION_KAMA_WINDOW"])),
            banda_alta=cache.kama(int(v["REVERSION_KAMA_WINDOW"]))
            + float(v["REVERSION_BAND_MULT"]) * cache.atr(int(v["TRAIL_ATR_WINDOW"])),
            banda_bassa=cache.kama(int(v["REVERSION_KAMA_WINDOW"]))
            - float(v["REVERSION_BAND_MULT"]) * cache.atr(int(v["TRAIL_ATR_WINDOW"])),
        ),
        tracce=(
            Traccia("kama", "KAMA", ARANCIO, larghezza=1.5),
            Traccia("banda_alta", "Banda superiore", ARANCIO, tratteggio="dash", larghezza=1.0),
            Traccia("banda_bassa", "Banda inferiore", ARANCIO, tratteggio="dash", larghezza=1.0),
        ),
    ),
    "adx": Indicatore(
        etichetta="ADX (forza del trend)",
        parametri=("ADX_WINDOW",),
        pannello="ADX",
        serie=lambda df, cache, v: _serie(df.index, adx=cache.adx(int(v["ADX_WINDOW"]))),
        tracce=(Traccia("adx", "ADX", ACQUA, larghezza=1.5),),
    ),
    "stochrsi": Indicatore(
        etichetta="StochRSI",
        parametri=("STOCHRSI_WINDOW", "STOCHRSI_SMOOTH"),
        pannello="StochRSI",
        serie=lambda df, cache, v: _serie(
            df.index, stochrsi=cache.stochrsi(int(v["STOCHRSI_WINDOW"]), int(v["STOCHRSI_SMOOTH"]))
        ),
        tracce=(Traccia("stochrsi", "StochRSI", BLU, larghezza=1.5),),
    ),
    "obv": Indicatore(
        etichetta="Pendenza dell'OBV",
        parametri=("OBV_WINDOW",),
        pannello="Volume (OBV)",
        serie=lambda df, cache, v: _serie(df.index, obv=cache.obv_slope(int(v["OBV_WINDOW"]))),
        tracce=(Traccia("obv", "Pendenza OBV", ACQUA, larghezza=1.5),),
    ),
}


def _solo_lunghe(eventi: list) -> tuple[list, list]:
    """Da cambi di posizione alle due liste che la pagina sa gia' trattare.

    Le strategie di `strategies_ls` restituiscono `(tempo, prezzo, obiettivo)`. Chiamate con
    `allow_short=False` l'obiettivo e' solo `+1` o `0`, alternati, ed e' proprio la forma che
    `simulate_trading_with_commisions` si aspetta accoppiando per indice. La conversione e' quindi
    esatta, non un'approssimazione -- ma vale **solo** senza il verso corto: un'inversione diretta
    da lungo a corto in due liste separate non e' rappresentabile, ed e' il motivo per cui quel
    formato esisteva.

    Quello che si perde e' il motore: `simulate_positions` addebita il costo di mantenimento
    giornaliero e conosce la leva, `simulate_trading_with_commisions` no. I risultati che la
    pagina mostra per queste cinque strategie sono percio' piu' ottimisti di quelli misurati in
    `reports/lab_*.csv`, che il funding lo pagano.
    """
    acquisti = [(tempo, prezzo) for tempo, prezzo, obiettivo in eventi if obiettivo > 0]
    uscite = [(tempo, prezzo) for tempo, prezzo, obiettivo in eventi if obiettivo == 0]
    return acquisti, uscite


# -------------------------------------------------------------------------------------------------
# Le strategie del menu
# -------------------------------------------------------------------------------------------------
# Gli indicatori sono stati letti funzione per funzione, non dedotti. Due voci non ne usano
# nessuno: "Green Candles" guarda solo la forma delle candele, "AI Model" chiede al modello.

STRATEGIE: dict[str, Strategia] = {
    "Close Buy/Sell Limits": Strategia(
        indicatori=("bande_atr", "rsi"),
        parametri=("RSI_BUY_LIMIT", "RSI_SELL_LIMIT", "NUM_CONDITIONS", "STOP_LOSS_PERCENT"),
        esegui=lambda df, cache, v: strategies.buy_sell_limits_close_simulation(
            df=df,
            rsi_buy_limit=v["RSI_BUY_LIMIT"],
            rsi_sell_limit=v["RSI_SELL_LIMIT"],
            num_cond=v["NUM_CONDITIONS"],
            stop_loss_percent=v["STOP_LOSS_PERCENT"],
        ),
    ),
    "Close ATR": Strategia(
        indicatori=("bande_atr", "psar"),
        parametri=("STOP_LOSS_PERCENT",),
        esegui=lambda df, cache, v: strategies.close_atr_buy_sell_simulation(
            df=df, stop_loss_percent=v["STOP_LOSS_PERCENT"]
        ),
    ),
    "ATR Bands": Strategia(
        indicatori=("bande_atr", "psar"),
        parametri=("STOP_LOSS_PERCENT",),
        esegui=lambda df, cache, v: strategies.atr_buy_sell_simulation(df=df, stop_loss_percent=v["STOP_LOSS_PERCENT"]),
    ),
    "Close Bullish EMA": Strategia(
        indicatori=("medie", "rsi"),
        parametri=("RSI_BUY_LIMIT", "RSI_SELL_LIMIT"),
        esegui=lambda df, cache, v: strategies.close_bullish_ema_simulation(
            df=df, rsi_buy_limit=v["RSI_BUY_LIMIT"], rsi_sell_limit=v["RSI_SELL_LIMIT"]
        ),
    ),
    "Close EMA Crossover": Strategia(
        indicatori=("medie",),
        esegui=lambda df, cache, v: strategies.close_ema_crossover_simulation(df=df),
    ),
    "Supertrend": Strategia(
        indicatori=("bande_atr",),
        esegui=lambda df, cache, v: strategies.supertrend_simulation(df=df),
    ),
    "Trend Zones": Strategia(
        indicatori=("medie_trend",),
        esegui=lambda df, cache, v: strategies.trend_zone_simulation(df=df),
    ),
    "TP/SL with ATR": Strategia(
        indicatori=("bande_atr",),
        esegui=lambda df, cache, v: strategies.tp_sl_simulation(df=df),
    ),
    "Green Candles": Strategia(
        indicatori=(),
        esegui=lambda df, cache, v: strategies.green_candles_simulation(df=df),
        note="Guarda solo la forma delle candele: nessun indicatore.",
    ),
    "ATR Live Trade": Strategia(
        indicatori=("bande_atr", "psar"),
        parametri=("ATR_WINDOW", "ATR_MULTIPLIER", "STOP_LOSS_PERCENT"),
        esegui=lambda df, cache, v: strategies.simulate_candles(
            raw_df=df,
            atr_window=v["ATR_WINDOW"],
            atr_multiplier=v["ATR_MULTIPLIER"],
            stop_loss_percent=v["STOP_LOSS_PERCENT"],
        ),
        note="Ricalcola il PSAR per conto suo a partire dalle candele grezze.",
    ),
    "AI Model": Strategia(
        indicatori=(),
        esegui=lambda df, cache, v: strategies.ai_model_simulation(df=df, model=v["MODELLO"]),
        note="I segnali vengono dal modello addestrato: nessun indicatore da disegnare.",
    ),
    "Donchian Breakout": Strategia(
        indicatori=("donchian", "media_regime", "adx"),
        parametri=("DONCHIAN_CHANNEL", "ADX_WINDOW", "ADX_MIN", "TRAIL_ATR_WINDOW", "DONCHIAN_ATR_MULT", "REGIME_EMA"),
        esegui=lambda df, cache, v: _solo_lunghe(
            strategies_ls.donchian_breakout(
                df,
                cache,
                channel=int(v["DONCHIAN_CHANNEL"]),
                adx_window=int(v["ADX_WINDOW"]),
                adx_min=float(v["ADX_MIN"]),
                atr_window=int(v["TRAIL_ATR_WINDOW"]),
                atr_multiplier=float(v["DONCHIAN_ATR_MULT"]),
                regime_ema=int(v["REGIME_EMA"]),
                allow_short=False,
            )
        ),
        note="Rottura di canale con filtro ADX e uscita a trailing sull'ATR.",
    ),
    "Squeeze Breakout": Strategia(
        indicatori=("bollinger", "keltner", "obv"),
        parametri=(
            "BB_WINDOW",
            "BB_DEV",
            "KC_WINDOW",
            "KC_MULTIPLIER",
            "TRAIL_ATR_WINDOW",
            "SQUEEZE_ATR_MULT",
            "OBV_WINDOW",
        ),
        esegui=lambda df, cache, v: _solo_lunghe(
            strategies_ls.squeeze_breakout(
                df,
                cache,
                bb_window=int(v["BB_WINDOW"]),
                bb_dev=float(v["BB_DEV"]),
                kc_window=int(v["KC_WINDOW"]),
                kc_multiplier=float(v["KC_MULTIPLIER"]),
                atr_window=int(v["TRAIL_ATR_WINDOW"]),
                atr_multiplier=float(v["SQUEEZE_ATR_MULT"]),
                confirm_volume=bool(v["CONFIRM_VOLUME"]),
                obv_window=int(v["OBV_WINDOW"]),
                allow_short=False,
            )
        ),
        note="Entra quando le bande di Bollinger escono dal canale di Keltner.",
    ),
    "Trend Pullback": Strategia(
        indicatori=("media_regime", "stochrsi"),
        parametri=(
            "REGIME_EMA",
            "STOCHRSI_WINDOW",
            "STOCHRSI_SMOOTH",
            "STOCH_OVERSOLD",
            "STOCH_OVERBOUGHT",
            "TRAIL_ATR_WINDOW",
            "PULLBACK_ATR_MULT",
        ),
        esegui=lambda df, cache, v: _solo_lunghe(
            strategies_ls.trend_pullback(
                df,
                cache,
                regime_ema=int(v["REGIME_EMA"]),
                stochrsi_window=int(v["STOCHRSI_WINDOW"]),
                stochrsi_smooth=int(v["STOCHRSI_SMOOTH"]),
                oversold=float(v["STOCH_OVERSOLD"]),
                overbought=float(v["STOCH_OVERBOUGHT"]),
                atr_window=int(v["TRAIL_ATR_WINDOW"]),
                atr_multiplier=float(v["PULLBACK_ATR_MULT"]),
                allow_short=False,
            )
        ),
        note="Compra il ritracciamento, ma solo sopra la media lunga.",
    ),
    "Ichimoku Trend": Strategia(
        indicatori=("ichimoku",),
        parametri=("ICHIMOKU_FAST", "ICHIMOKU_SLOW", "ICHIMOKU_SPAN"),
        esegui=lambda df, cache, v: _solo_lunghe(
            strategies_ls.ichimoku_trend(
                df,
                cache,
                fast=int(v["ICHIMOKU_FAST"]),
                slow=int(v["ICHIMOKU_SLOW"]),
                span=int(v["ICHIMOKU_SPAN"]),
                require_cloud=bool(v["REQUIRE_CLOUD"]),
                allow_short=False,
            )
        ),
        note="Il sistema di trend preso dal manuale, come metro di paragone.",
    ),
    "Band Reversion": Strategia(
        indicatori=("bande_kama", "adx"),
        parametri=(
            "REVERSION_KAMA_WINDOW",
            "TRAIL_ATR_WINDOW",
            "REVERSION_BAND_MULT",
            "ADX_WINDOW",
            "ADX_MAX",
            "REVERSION_STOP_MULT",
            "REVERSION_REGIME_EMA",
        ),
        esegui=lambda df, cache, v: _solo_lunghe(
            strategies_ls.band_reversion_gated(
                df,
                cache,
                kama_window=int(v["REVERSION_KAMA_WINDOW"]),
                atr_window=int(v["TRAIL_ATR_WINDOW"]),
                band_multiplier=float(v["REVERSION_BAND_MULT"]),
                adx_window=int(v["ADX_WINDOW"]),
                adx_max=float(v["ADX_MAX"]),
                stop_multiplier=float(v["REVERSION_STOP_MULT"]),
                regime_ema=int(v["REVERSION_REGIME_EMA"]),
                allow_short=False,
            )
        ),
        note=(
            "Ritorno alla media, ma solo quando l'ADX dice che non c'e' trend. Il filtro di regime "
            "e' spento di default, come nella misura: in quel caso la media non viene disegnata."
        ),
    ),
}

VUOTA = "-"  # la voce che non seleziona nessuna strategia: si mostra tutto

# Nella panoramica senza strategia questi due si tolgono, perche' sarebbero doppioni visivi:
# `medie_trend` disegna due delle tre linee di `medie`, e `bande_kama` ha la stessa forma di
# `bande_atr` con finestra e moltiplicatore diversi. Mostrarli tutti metteva in legenda due
# "EMA corta" e due "Banda superiore", cioe' due etichette identiche su linee diverse -- che e'
# peggio di un'informazione mancante, perche' sembra un errore di lettura di chi guarda.
PANORAMICA_ESCLUSI = ("medie_trend", "bande_kama")


def valori_predefiniti() -> dict:
    """Il valore iniziale di ogni parametro noto, cioe' cosa vede la pagina prima che si tocchi
    qualcosa. Serve alla pagina come base su cui scrivere le scelte dei widget, e ai test come
    contesto per calcolare le serie."""
    from cryptofarm.trading import config

    valori = {
        nome: getattr(config, nome).value for nome in dir(config) if isinstance(getattr(config, nome), config.Param)
    }
    valori["CONFIRM_VOLUME"] = config.CONFIRM_VOLUME
    valori["REQUIRE_CLOUD"] = config.REQUIRE_CLOUD
    return valori


def indicatori_di(strategia: str) -> tuple[str, ...]:
    """Gli indicatori da mostrare. Senza strategia selezionata si mostra tutto, come richiesto."""
    if strategia not in STRATEGIE:
        return tuple(chiave for chiave in INDICATORI if chiave not in PANORAMICA_ESCLUSI)
    return STRATEGIE[strategia].indicatori


def parametri_di(strategia: str) -> list[str]:
    """I parametri della barra laterale, senza ripetizioni e in ordine stabile.

    Sono l'unione di quelli degli indicatori usati e di quelli propri della strategia: le finestre
    arrivano dagli indicatori, le soglie dalla strategia.
    """
    voluti: list[str] = []
    for chiave in indicatori_di(strategia):
        voluti.extend(INDICATORI[chiave].parametri)
    if strategia in STRATEGIE:
        voluti.extend(STRATEGIE[strategia].parametri)
    else:
        for nota in STRATEGIE.values():
            voluti.extend(nota.parametri)
    return list(dict.fromkeys(voluti))


def pannelli_di(strategia: str) -> list[str]:
    """I titoli dei riquadri sotto le candele, nell'ordine in cui vanno impilati."""
    titoli: list[str] = []
    for chiave in indicatori_di(strategia):
        pannello = INDICATORI[chiave].pannello
        if pannello is not None and pannello not in titoli:
            titoli.append(pannello)
    return titoli


# -------------------------------------------------------------------------------------------------
# Come si presentano nella barra laterale
# -------------------------------------------------------------------------------------------------
# Il nome della costante dice a cosa serve nel codice; l'etichetta dice cosa muove a chi guarda il
# grafico. `STOP_LOSS_PERCENT` e' "Stop loss %", `TRAIL_ATR_WINDOW` e' "Finestra ATR (uscita)" e non
# semplicemente "ATR", perche' nella stessa vista puo' esserci anche l'ATR delle bande.

ETICHETTE: dict[str, str] = {
    "ATR_MULTIPLIER": "Moltiplicatore ATR",
    "ATR_WINDOW": "Finestra ATR",
    "RSI_SHORT": "RSI corto",
    "RSI_MEDIUM": "RSI medio",
    "RSI_LONG": "RSI lungo",
    "EMA_SHORT": "EMA corta",
    "EMA_MEDIUM": "EMA media",
    "EMA_LONG": "EMA lunga",
    "KAMA_POW1": "KAMA potenza 1",
    "KAMA_POW2": "KAMA potenza 2",
    "RSI_BUY_LIMIT": "Soglia RSI di acquisto",
    "RSI_SELL_LIMIT": "Soglia RSI di vendita",
    "STOP_LOSS_PERCENT": "Stop loss %",
    "NUM_CONDITIONS": "Condizioni richieste",
    "PIVOT_WINDOW": "Finestra massimi/minimi",
    "ADX_WINDOW": "Finestra ADX",
    "ADX_MIN": "ADX minimo (serve trend)",
    "ADX_MAX": "ADX massimo (serve intervallo)",
    "REGIME_EMA": "EMA di regime",
    "TRAIL_ATR_WINDOW": "Finestra ATR (uscita)",
    "DONCHIAN_CHANNEL": "Ampiezza del canale",
    "DONCHIAN_ATR_MULT": "Distanza dello stop (ATR)",
    "BB_WINDOW": "Finestra Bollinger",
    "BB_DEV": "Deviazioni Bollinger",
    "KC_WINDOW": "Finestra Keltner",
    "KC_MULTIPLIER": "Moltiplicatore Keltner",
    "OBV_WINDOW": "Finestra OBV",
    "SQUEEZE_ATR_MULT": "Distanza dello stop (ATR)",
    "STOCHRSI_WINDOW": "Finestra StochRSI",
    "STOCHRSI_SMOOTH": "Lisciatura StochRSI",
    "STOCH_OVERSOLD": "Soglia di ipervenduto",
    "STOCH_OVERBOUGHT": "Soglia di ipercomprato",
    "PULLBACK_ATR_MULT": "Distanza dello stop (ATR)",
    "ICHIMOKU_FAST": "Tenkan",
    "ICHIMOKU_SLOW": "Kijun",
    "ICHIMOKU_SPAN": "Senkou B",
    "REVERSION_KAMA_WINDOW": "Finestra KAMA",
    "REVERSION_BAND_MULT": "Ampiezza delle bande (ATR)",
    "REVERSION_STOP_MULT": "Distanza dello stop (ATR)",
    "REVERSION_REGIME_EMA": "EMA di regime (0 = spento)",
}

SOGLIE = "Soglie della strategia"

# Le cinque strategie che nella pagina passano dal motore classico: quello non addebita il costo di
# mantenimento e non conosce la leva, quindi i loro numeri qui sono piu' ottimisti di quelli
# misurati. La pagina lo dice accanto al risultato invece di lasciarlo scoprire per confronto.
NUOVE_SENZA_MANTENIMENTO = (
    "Donchian Breakout",
    "Squeeze Breakout",
    "Trend Pullback",
    "Ichimoku Trend",
    "Band Reversion",
)


def gruppi_di(strategia: str) -> list[tuple[str, list[str]]]:
    """I riquadri della barra laterale: un titolo e i parametri che ci vanno dentro.

    Un parametro compare **una volta sola**, nel primo gruppo che lo rivendica: `EMA_SHORT` serve
    sia alle medie sia alle bande costruite su KAMA, e disegnarlo due volte darebbe due widget con
    la stessa chiave -- cioe' un errore di Streamlit, non un doppione innocuo.
    """
    gruppi: list[tuple[str, list[str]]] = []
    gia_visti: set[str] = set()
    for chiave in indicatori_di(strategia):
        indicatore = INDICATORI[chiave]
        dentro = [nome for nome in indicatore.parametri if nome not in gia_visti]
        gia_visti.update(dentro)
        if dentro:
            gruppi.append((indicatore.etichetta, dentro))
    propri = [nome for nome in parametri_di(strategia) if nome not in gia_visti]
    if propri:
        gruppi.append((SOGLIE, propri))
    return gruppi
