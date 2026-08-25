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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from cryptofarm.trading import strategies

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
    serie: Callable[[pd.DataFrame], dict[str, pd.Series]]
    tracce: tuple[Traccia, ...]


@dataclass(frozen=True)
class Strategia:
    """Una voce del menu: cosa esegue, quali indicatori usa, quali parametri le servono.

    `esegui` prende il frame e il dizionario dei valori letti dalla barra laterale, e restituisce
    `(buy_signals, sell_signals)`. E' una chiamata scritta per esteso invece di una tabella di
    corrispondenze fra nomi: le firme delle strategie non sono uniformi, e una riga esplicita si
    legge meglio di un meccanismo che le uniformi.

    `parametri` sono i suoi soli: le finestre degli indicatori arrivano gia' da
    `Indicatore.parametri`, e la barra laterale unisce i due insiemi senza ripetizioni.
    """

    indicatori: tuple[str, ...]
    esegui: Callable[[pd.DataFrame, dict], tuple[list, list]]
    parametri: tuple[str, ...] = ()
    note: str = ""


def _colonne(*nomi: str) -> Callable[[pd.DataFrame], dict[str, pd.Series]]:
    """Le serie sono gia' colonne del frame: la sorgente piu' comune."""
    return lambda df: {nome: df[nome] for nome in nomi}


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
        tracce=(Traccia("PSAR", "PSAR", ACQUA, modo="markers", simbolo="circle", larghezza=0.0),),
    ),
    "estremi": Indicatore(
        # Non e' letto da nessuna strategia: e' il riferimento visivo dei massimi e minimi
        # relativi, e resta disponibile nella panoramica senza strategia selezionata.
        etichetta="Massimi e minimi relativi",
        parametri=("PIVOT_WINDOW",),
        pannello=None,
        serie=lambda df: {},
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
}


# -------------------------------------------------------------------------------------------------
# Le strategie del menu
# -------------------------------------------------------------------------------------------------
# Gli indicatori sono stati letti funzione per funzione, non dedotti. Due voci non ne usano
# nessuno: "Green Candles" guarda solo la forma delle candele, "AI Model" chiede al modello.

STRATEGIE: dict[str, Strategia] = {
    "Close Buy/Sell Limits": Strategia(
        indicatori=("bande_atr", "rsi"),
        parametri=("RSI_BUY_LIMIT", "RSI_SELL_LIMIT", "NUM_CONDITIONS", "STOP_LOSS_PERCENT"),
        esegui=lambda df, v: strategies.buy_sell_limits_close_simulation(
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
        esegui=lambda df, v: strategies.close_atr_buy_sell_simulation(df=df, stop_loss_percent=v["STOP_LOSS_PERCENT"]),
    ),
    "ATR Bands": Strategia(
        indicatori=("bande_atr", "psar"),
        parametri=("STOP_LOSS_PERCENT",),
        esegui=lambda df, v: strategies.atr_buy_sell_simulation(df=df, stop_loss_percent=v["STOP_LOSS_PERCENT"]),
    ),
    "Close Bullish EMA": Strategia(
        indicatori=("medie", "rsi"),
        parametri=("RSI_BUY_LIMIT", "RSI_SELL_LIMIT"),
        esegui=lambda df, v: strategies.close_bullish_ema_simulation(
            df=df, rsi_buy_limit=v["RSI_BUY_LIMIT"], rsi_sell_limit=v["RSI_SELL_LIMIT"]
        ),
    ),
    "Close EMA Crossover": Strategia(
        indicatori=("medie",),
        esegui=lambda df, v: strategies.close_ema_crossover_simulation(df=df),
    ),
    "Supertrend": Strategia(
        indicatori=("bande_atr",),
        esegui=lambda df, v: strategies.supertrend_simulation(df=df),
    ),
    "Trend Zones": Strategia(
        indicatori=("medie_trend",),
        esegui=lambda df, v: strategies.trend_zone_simulation(df=df),
    ),
    "TP/SL with ATR": Strategia(
        indicatori=("bande_atr",),
        esegui=lambda df, v: strategies.tp_sl_simulation(df=df),
    ),
    "Green Candles": Strategia(
        indicatori=(),
        esegui=lambda df, v: strategies.green_candles_simulation(df=df),
        note="Guarda solo la forma delle candele: nessun indicatore.",
    ),
    "ATR Live Trade": Strategia(
        indicatori=("bande_atr", "psar"),
        parametri=("ATR_WINDOW", "ATR_MULTIPLIER", "STOP_LOSS_PERCENT"),
        esegui=lambda df, v: strategies.simulate_candles(
            raw_df=df,
            atr_window=v["ATR_WINDOW"],
            atr_multiplier=v["ATR_MULTIPLIER"],
            stop_loss_percent=v["STOP_LOSS_PERCENT"],
        ),
        note="Ricalcola il PSAR per conto suo a partire dalle candele grezze.",
    ),
    "AI Model": Strategia(
        indicatori=(),
        esegui=lambda df, v: strategies.ai_model_simulation(df=df, model=v["MODELLO"]),
        note="I segnali vengono dal modello addestrato: nessun indicatore da disegnare.",
    ),
}

VUOTA = "-"  # la voce che non seleziona nessuna strategia: si mostra tutto


def indicatori_di(strategia: str) -> tuple[str, ...]:
    """Gli indicatori da mostrare. Senza strategia selezionata si mostra tutto, come richiesto."""
    if strategia not in STRATEGIE:
        return tuple(INDICATORI)
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
