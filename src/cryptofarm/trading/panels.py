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

import numpy as np
import pandas as pd

from cryptofarm.data.klines import interval_to_minutes
from cryptofarm.trading import confluence, strategies, strategies_ls
from cryptofarm.trading.indicators_extra import ExtraCache

# Le tre tinte categoriche, validate insieme sulla superficie scura di Streamlit.
BLU = "#3987e5"
ARANCIO = "#d95926"
ACQUA = "#199e70"

# Dentro una famiglia le serie sono **ordinate** -- corta, media e lunga si distinguono per la
# finestra, non per identita' -- e la codifica giusta per un ordine e' una rampa di chiarezza, non
# tre tratteggi della stessa tinta. Tre linee blu che differiscono solo per il tratteggio non si
# distinguono a colpo d'occhio: era il difetto della prima versione. Qui chiarezza e spessore
# crescono insieme alla finestra, cosi' due canali dicono la stessa cosa.
# Le due rampe passano tutti i controlli ordinali del validatore sulla superficie scura: chiarezza
# monotona, salti di almeno 0,06, estremo scuro sopra 2:1 di contrasto.
BLU_CHIARO = "#9ec5f4"
BLU_SCURO = "#184f95"
ARANCIO_CHIARO = "#f5a183"
ARANCIO_SCURO = "#8c3413"
FAMIGLIA_BLU = (BLU_CHIARO, BLU, BLU_SCURO)
FAMIGLIA_ARANCIO = (ARANCIO_CHIARO, ARANCIO, ARANCIO_SCURO)
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
    dimensione: float = 5.0  # solo per i marcatori


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
    # `condizionale` dice che questo indicatore puo' legittimamente non disegnare niente su un
    # dato frame -- lo stop a trailing esiste solo dove c'e' una posizione aperta, e su una storia
    # in cui la strategia non e' mai entrata non esiste affatto. Serve al test che verifica che
    # ogni traccia dichiarata abbia davvero la sua serie: senza, quel test dovrebbe accettare il
    # dizionario vuoto da chiunque, e smetterebbe di intercettare il nome di colonna sbagliato che
    # e' la ragione per cui esiste.
    condizionale: bool = False


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


# La confluenza viene chiesta tre volte per ogni ridisegno della pagina -- una dalla strategia e
# una per ciascuno dei suoi due riquadri -- e ogni volta costa quanto sei strategie. La chiave e'
# **derivata dal contenuto**, non da `id()`: due frame con la stessa lunghezza, gli stessi estremi
# di indice, gli stessi prezzi ai bordi e gli stessi parametri danno lo stesso risultato, quindi
# una collisione non e' un errore ma una risposta giusta.
# ponytail: memoria a una cella, basta perche' le tre chiamate sono consecutive; se un giorno la
# pagina dovesse confrontare due configurazioni affiancate, serve una `lru_cache` vera.
_ULTIMA_CONFLUENZA: tuple = ()


def confluenza_di(df: pd.DataFrame, valori: dict):
    """La confluenza sulle candele date, calcolata una volta sola per ridisegno.

    Restituisce `None` quando la storia e' troppo corta perche' i piani lunghi esistano: e' il
    caso della pagina appena aperta con poche ore di dati, e va detto invece che sollevato.
    """
    global _ULTIMA_CONFLUENZA

    # `valori` puo' arrivare **parziale**: la barra laterale lo costruisce da zero mettendoci solo
    # cio' che disegna, e chi chiama da fuori la pagina passa quel che ha. `trading_analysis`
    # riempie i buchi per conto suo, ma non e' l'unico chiamante -- `diagnosi_confluenza` riceve il
    # dizionario della barra laterale cosi' com'e', e cadeva con `KeyError` proprio nel caso per cui
    # esiste, quello senza operazioni. Riempirli qui copre tutti i chiamanti in una volta.
    valori = {**valori_predefiniti(), **valori}

    parametri = {
        "theta_base": float(valori["CONF_THETA_BASE"]),
        "theta_macro": float(valori["CONF_THETA_MACRO"]),
        "isteresi": float(valori["CONF_ISTERESI"]),
        "barre_minime": int(valori["CONF_BARRE_MINIME"]),
        "pazienza": int(valori["CONF_PAZIENZA"]),
        "emivita": float(valori["CONF_EMIVITA"]),
        "w_max": float(valori["CONF_W_MAX"]),
        "k_famiglie": int(valori["CONF_K_FAMIGLIE"]),
        "innesco": int(valori["CONF_INNESCO"]),
        "atr_window": int(valori["CONF_ATR_WINDOW"]),
        "atr_multiplier": float(valori["CONF_ATR_MULT"]),
        "regime_ema": int(valori["CONF_REGIME_EMA"]),
        "struttura_ema": int(valori["CONF_STRUTTURA_EMA"]),
        "barre_in_formazione": bool(valori["CONF_IN_FORMAZIONE"]),
    }
    intervallo = str(valori["INTERVALLO"])
    # Gli override dei votanti: quel che i widget hanno mosso rispetto ai loro default. Passarli
    # sempre tutti sarebbe equivalente ma renderebbe la chiave della memoria enorme.
    parametri["parametri_votanti"] = {
        votante.nome: {
            parametro.kwarg: valori[parametro.config] for parametro in votante.parametri if parametro.config in valori
        }
        for votante in confluence.VOTANTI
    }
    chiave = (
        len(df),
        df.index[0],
        df.index[-1],
        float(df["Close"].iloc[0]),
        float(df["Close"].iloc[-1]),
        intervallo,
        tuple(sorted((k, v) for k, v in parametri.items() if k != "parametri_votanti")),
        tuple(sorted((n, tuple(sorted(d.items()))) for n, d in parametri["parametri_votanti"].items())),
    )
    if _ULTIMA_CONFLUENZA and _ULTIMA_CONFLUENZA[0] == chiave:
        return _ULTIMA_CONFLUENZA[1]

    # Serve almeno una barra del piano piu' lungo, piu' quelle che le medie chiedono.
    minimo = max(confluence.FATTORI.values()) * 3
    risultato = confluence.evaluate(df, intervallo, **parametri) if len(df) >= minimo else None
    _ULTIMA_CONFLUENZA = (chiave, risultato)
    return risultato


def diagnosi_confluenza(df: pd.DataFrame, valori: dict, intervallo: str) -> str:
    """Perche' la confluenza non ha operato, in una riga, o "" se ha operato.

    Zero operazioni non e' un risultato ma una domanda, e le condizioni d'ingresso sono quattro in
    `and`: senza sapere quale non si e' mai avverata, chi guarda non sa se guardare la storia
    caricata, la soglia o l'ampiezza. Riusa il calcolo gia' fatto per il grafico.
    """
    risultato = confluenza_di(df, {**valori, "INTERVALLO": intervallo})
    if risultato is None:
        minimo = max(confluence.FATTORI.values()) * 3
        return f"only {len(df)} bars loaded: the longer planes need at least {minimo}."
    return risultato.perche_non_entra()


def _serie_confluenza(df, cache, valori):
    risultato = confluenza_di(df, valori)
    if risultato is None:
        return {}
    return _serie(df.index, punteggio=risultato.punteggio, soglia=risultato.soglia)


def _serie_piani(df, cache, valori):
    risultato = confluenza_di(df, valori)
    if risultato is None:
        return {}
    return _serie(df.index, regime=risultato.regime, struttura=risultato.struttura)


def _serie_stop(df, cache, valori):
    """Lo stop a trailing, o niente quando non si e' mai stati dentro il mercato.

    Restituire una serie tutta vuota metterebbe in legenda un indicatore che non disegna niente --
    la stessa ragione per cui `media_regime` a finestra zero non restituisce nulla.
    """
    risultato = confluenza_di(df, valori)
    if risultato is None or risultato.stop is None or not np.isfinite(risultato.stop).any():
        return {}
    return _serie(df.index, stop=risultato.stop)


def _serie_votanti(df, cache, valori):
    risultato = confluenza_di(df, valori)
    return _serie(df.index, **risultato.voti) if risultato is not None else {}


def _serie_etichetta(df, cache, valori):
    """L'etichetta con cui il modello a swing viene addestrato, sulle candele caricate.

    Non e' un indicatore: `swing_target` guarda `window` barre **avanti**, quindi non e' operabile
    e le ultime `window` barre escono vuote. Sta qui perche' e' l'unico modo di vedere, sulle
    candele che si stanno guardando, quale numero il modello impara a prevedere su ognuna.

    Le tre tracce sono la stessa formula su tre finestre: quella piena (cio' su cui il modello e'
    addestrato), la sola meta' futura -- l'unica non ricavabile dal passato, e quindi il solo metro
    onesto -- e la sola meta' passata, che e' uno Stochastic e si puo' calcolare senza modello.
    """
    from cryptofarm.ml.labeling import swing_target

    finestra = int(valori["SWING_TARGET_WINDOW"])
    chiusura = df["Close"]
    return {
        "swing_target": swing_target(chiusura, finestra),
        "swing_avanti": swing_target(chiusura, finestra, verso="avanti"),
        "swing_dietro": swing_target(chiusura, finestra, verso="dietro"),
    }


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
        etichetta="Exponential moving averages",
        parametri=("EMA_SHORT", "EMA_MEDIUM", "EMA_LONG"),
        pannello=None,
        serie=_colonne("EMA20", "EMA50", "EMA100"),
        tracce=(
            Traccia("EMA20", "EMA fast", BLU_CHIARO, larghezza=1.0),
            Traccia("EMA50", "EMA mid", BLU, larghezza=2.0),
            Traccia("EMA100", "EMA slow", BLU_SCURO, larghezza=3.2),
        ),
    ),
    "medie_trend": Indicatore(
        # "Trend Zones" confronta la corta con la lunga: la media di mezzo non la tocca.
        etichetta="Fast and slow averages",
        parametri=("EMA_SHORT", "EMA_LONG"),
        pannello=None,
        serie=_colonne("EMA20", "EMA100"),
        tracce=(
            Traccia("EMA20", "EMA fast", BLU_CHIARO, larghezza=1.0),
            Traccia("EMA100", "EMA slow", BLU_SCURO, larghezza=3.2),
        ),
    ),
    "bande_atr": Indicatore(
        # KAMA usa `ema_window`, e le bande sono KAMA +/- moltiplicatore * ATR: i cinque
        # parametri servono tutti, anche se il nome ne cita uno solo.
        etichetta="ATR bands on KAMA",
        parametri=("ATR_WINDOW", "ATR_MULTIPLIER", "KAMA_POW1", "KAMA_POW2", "EMA_SHORT"),
        pannello=None,
        serie=_colonne("KAMA", "Upper_Band", "Lower_Band"),
        tracce=(
            Traccia("KAMA", "KAMA", ARANCIO_SCURO, larghezza=2.2),
            Traccia("Upper_Band", "Upper band", ARANCIO_CHIARO, tratteggio="dash", larghezza=1.3),
            Traccia("Lower_Band", "Lower band", ARANCIO_CHIARO, tratteggio="dash", larghezza=1.3),
        ),
    ),
    "psar": Indicatore(
        etichetta="Parabolic SAR",
        parametri=(),
        pannello=None,
        serie=_colonne("PSAR"),
        tracce=(Traccia("PSAR", "PSAR", BLU_CHIARO, modo="markers", larghezza=0.0, dimensione=3.5),),
    ),
    "estremi": Indicatore(
        # Non e' letto da nessuna strategia: e' il riferimento visivo dei massimi e minimi
        # relativi, e resta disponibile nella panoramica senza strategia selezionata.
        etichetta="Swing highs and lows",
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
            Traccia("RSI", "RSI fast", BLU_CHIARO, larghezza=1.0),
            Traccia("RSI2", "RSI mid", BLU, larghezza=2.0),
            Traccia("RSI3", "RSI slow", BLU_SCURO, larghezza=3.2),
        ),
    ),
    "stocastico": Indicatore(
        etichetta="Stochastic",
        parametri=("RSI_SHORT",),
        pannello="Stochastic",
        serie=_colonne("STOCH", "STOCH_S"),
        tracce=(
            Traccia("STOCH", "Stochastic", ARANCIO, larghezza=1.8),
            Traccia("STOCH_S", "Signal", ARANCIO_CHIARO, tratteggio="dash", larghezza=1.2),
        ),
    ),
    "tsi": Indicatore(
        etichetta="True Strength Index",
        parametri=(),
        pannello="TSI",
        serie=_colonne("TSI"),
        tracce=(Traccia("TSI", "TSI", ACQUA, larghezza=1.8),),
    ),
    "donchian": Indicatore(
        etichetta="Donchian channel",
        parametri=("DONCHIAN_CHANNEL",),
        pannello=None,
        serie=lambda df, cache, v: _serie(
            df.index, **dict(zip(("canale_alto", "canale_basso"), cache.donchian(int(v["DONCHIAN_CHANNEL"]))))
        ),
        tracce=(
            Traccia("canale_alto", "Channel high", ARANCIO, larghezza=1.8),
            Traccia("canale_basso", "Channel low", ARANCIO, larghezza=1.8),
        ),
    ),
    "confluenza": Indicatore(
        # La decisione, disegnata per intero: il punteggio, la soglia che gli si muove incontro
        # quando i piani alti concordano, e il cancello che manda a flat senza discutere. Chi
        # guarda deve poter dire perche' quella barra ha aperto e non la precedente.
        etichetta="Confluence score",
        parametri=("CONF_THETA_BASE", "CONF_THETA_MACRO", "CONF_ISTERESI", "CONF_BARRE_MINIME", "CONF_PAZIENZA"),
        pannello="Confluence",
        serie=_serie_confluenza,
        tracce=(
            Traccia("punteggio", "Score", BLU, larghezza=2.0),
            Traccia("soglia", "Threshold", ARANCIO, tratteggio="dash", larghezza=1.4),
        ),
    ),
    "piani_lunghi": Indicatore(
        # I due piani lunghi stanno in un riquadro **loro**, e non e' una questione di ordine:
        # valgono +-1 e il punteggio sta in +-0,5, quindi sullo stesso asse il cancello occupa il
        # doppio dell'ampiezza del punteggio e lo schiaccia in una riga piatta vicino allo zero. Si
        # vedeva una linea ferma a 1 mentre si comprava e si vendeva, e sembrava incoerente proprio
        # perche' la serie che decide non era leggibile.
        etichetta="Higher planes",
        parametri=("CONF_REGIME_EMA", "CONF_STRUTTURA_EMA"),
        pannello="Higher planes",
        serie=_serie_piani,
        tracce=(
            Traccia("regime", "Regime plane (gate)", ACQUA, larghezza=2.0),
            Traccia("struttura", "Structure plane", ARANCIO, tratteggio="dash", larghezza=1.4),
        ),
    ),
    "stop_confluenza": Indicatore(
        # Quattro uscite su cinque sono lo stop a trailing. Senza questa linea il grafico mostra una
        # vendita mentre il punteggio e' sopra la soglia, e non c'e' modo di capire perche'.
        etichetta="Trailing stop",
        parametri=("CONF_ATR_WINDOW", "CONF_ATR_MULT"),
        pannello=None,
        serie=_serie_stop,
        tracce=(Traccia("stop", "Trailing stop", ARANCIO_CHIARO, tratteggio="dash", larghezza=1.3),),
        condizionale=True,
    ),
    "votanti": Indicatore(
        # **Chi** ha votato, e quanto forte. Con il passaggio del mouse unificato questo riquadro
        # elenca i sei valori sulla barra puntata: e' li' che si legge chi ha generato il segnale,
        # perche' sei linee sovrapposte in [-1, +1] da sole non si distinguono a colpo d'occhio.
        # Sei serie categoriche su tre tinte: la quarta riusa una tinta cambiando tratteggio, mai
        # una tinta nuova, e nessuna e' una rampa -- fra i votanti non c'e' nessun ordine.
        etichetta="Voters",
        parametri=("CONF_EMIVITA", "CONF_W_MAX", "CONF_K_FAMIGLIE"),
        pannello="Voters",
        serie=_serie_votanti,
        tracce=(
            # Il colore dice la **famiglia**, il tratteggio il **piano**. Prima erano sei votanti
            # in sei famiglie e la distinzione non aveva niente da codificare; adesso due famiglie
            # hanno due votanti ciascuna -- le stesse bande a due scale, le stesse zone a due
            # piani -- e il colore condiviso e' cio' che si vuole leggere a colpo d'occhio,
            # perche' e' quello che `k_famiglie` conta.
            Traccia("ichimoku", "Ichimoku · structure", BLU, larghezza=1.4),
            Traccia("flusso", "Volume flow · structure", ARANCIO, larghezza=1.4),
            Traccia("pullback", "Pullback · confirmation", ACQUA, larghezza=1.4),
            Traccia("bande_conferma", "ATR bands · confirmation", BLU, tratteggio="dash", larghezza=1.4),
            Traccia("bande_innesco", "ATR bands · trigger", BLU, tratteggio="dot", larghezza=1.4),
            Traccia("zone_regime", "Trend zones · regime", ARANCIO, tratteggio="dash", larghezza=1.4),
            Traccia("zone_struttura", "Trend zones · structure", ARANCIO, tratteggio="dot", larghezza=1.4),
            # Il votante a modello e' l'unico che non scende mai sotto zero: vota +1 quando
            # `|previsione|` e' alta e 0 altrimenti, perche' la forma misurata del segnale non
            # dice il verso. Una linea che sta solo in [0, 1] si legge a colpo d'occhio come
            # diversa dalle altre, ed e' giusto cosi'. Se l'artefatto non c'e' la serie non
            # esiste e la traccia si salta da se'.
            Traccia("modello", "Swing model · trigger", ACQUA, tratteggio="dot", larghezza=1.4),
        ),
    ),
    "media_regime": Indicatore(
        etichetta="Regime average",
        parametri=("REGIME_EMA",),
        pannello=None,
        serie=lambda df, cache, v: (
            _serie(df.index, regime=cache.ema(int(v["REGIME_EMA"]))) if int(v["REGIME_EMA"]) else {}
        ),
        tracce=(Traccia("regime", "Regime EMA", BLU_CHIARO, tratteggio="dash", larghezza=2.2),),
    ),
    "bollinger": Indicatore(
        etichetta="Bollinger bands",
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
            Traccia("bb_alta", "Bollinger upper", BLU_CHIARO, larghezza=1.6),
            Traccia("bb_media", "Bollinger mid", BLU_SCURO, tratteggio="dot", larghezza=1.2),
            Traccia("bb_bassa", "Bollinger lower", BLU_CHIARO, larghezza=1.6),
        ),
    ),
    "keltner": Indicatore(
        # Il canale di Keltner usa la stessa finestra ATR dell'uscita a trailing.
        etichetta="Keltner channel",
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
            Traccia("kc_alta", "Keltner upper", ARANCIO, tratteggio="dash", larghezza=1.4),
            Traccia("kc_bassa", "Keltner lower", ARANCIO, tratteggio="dash", larghezza=1.4),
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
            Traccia("tenkan", "Tenkan", BLU_CHIARO, larghezza=1.8),
            Traccia("kijun", "Kijun", ARANCIO, larghezza=1.8),
            Traccia("span_a", "Cloud A", BLU_SCURO, tratteggio="dot", larghezza=1.2),
            Traccia("span_b", "Cloud B", ARANCIO_SCURO, tratteggio="dot", larghezza=1.2),
        ),
    ),
    "bande_kama": Indicatore(
        # Non sono le `Upper_Band`/`Lower_Band` del frame: stessa forma, ma finestra e
        # moltiplicatore sono quelli della strategia di ritorno alla media.
        etichetta="Mean-reversion bands",
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
            Traccia("kama", "KAMA", ARANCIO_SCURO, larghezza=2.2),
            Traccia("banda_alta", "Upper band", ARANCIO_CHIARO, tratteggio="dash", larghezza=1.3),
            Traccia("banda_bassa", "Lower band", ARANCIO_CHIARO, tratteggio="dash", larghezza=1.3),
        ),
    ),
    "adx": Indicatore(
        etichetta="ADX (trend strength)",
        parametri=("ADX_WINDOW",),
        pannello="ADX",
        serie=lambda df, cache, v: _serie(df.index, adx=cache.adx(int(v["ADX_WINDOW"]))),
        tracce=(Traccia("adx", "ADX", ACQUA, larghezza=1.8),),
    ),
    "stochrsi": Indicatore(
        etichetta="StochRSI",
        parametri=("STOCHRSI_WINDOW", "STOCHRSI_SMOOTH"),
        pannello="StochRSI",
        serie=lambda df, cache, v: _serie(
            df.index, stochrsi=cache.stochrsi(int(v["STOCHRSI_WINDOW"]), int(v["STOCHRSI_SMOOTH"]))
        ),
        tracce=(Traccia("stochrsi", "StochRSI", ACQUA, larghezza=1.8),),
    ),
    "obv": Indicatore(
        etichetta="OBV slope",
        parametri=("OBV_WINDOW",),
        pannello="Volume (OBV)",
        serie=lambda df, cache, v: _serie(df.index, obv=cache.obv_slope(int(v["OBV_WINDOW"]))),
        tracce=(Traccia("obv", "OBV slope", ACQUA, larghezza=1.8),),
    ),
    # Non e' un indicatore di nessuna strategia: e' l'etichetta del modello, e si mostra a
    # richiesta (`MOSTRA_ETICHETTA`). Sta nel registro perche' da li' prende widget, colori e
    # riquadro come tutti gli altri.
    "etichetta_swing": Indicatore(
        etichetta="Swing target",
        parametri=("SWING_TARGET_WINDOW",),
        pannello="Swing target (label, looks ahead)",
        serie=_serie_etichetta,
        tracce=(
            Traccia("swing_target", "Target", ACQUA, larghezza=2.0),
            Traccia("swing_avanti", "Forward half", BLU, tratteggio="dot", larghezza=1.3),
            Traccia("swing_dietro", "Backward half", ARANCIO, tratteggio="dot", larghezza=1.3),
        ),
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
    "ATR Bands": Strategia(
        indicatori=("bande_atr", "psar"),
        parametri=("STOP_LOSS_PERCENT",),
        esegui=lambda df, cache, v: strategies.atr_buy_sell_simulation(df=df, stop_loss_percent=v["STOP_LOSS_PERCENT"]),
        note="Best out of sample across the five assets: 7 of 10 cells beat buy and hold.",
    ),
    "Close EMA Crossover": Strategia(
        indicatori=("medie",),
        esegui=lambda df, cache, v: strategies.close_ema_crossover_simulation(df=df),
    ),
    "Close RSI Reverse": Strategia(
        indicatori=("rsi",),
        esegui=lambda df, cache, v: strategies.close_rsi_buy_sell_limits_simulation(df=df),
        note="Fast/mid RSI crossover. Daily bars only: at 4h it makes 160 trades a year and loses.",
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
    "AI Model": Strategia(
        indicatori=(),
        esegui=lambda df, cache, v: strategies.ai_model_simulation(
            df=df, model=v["MODELLO"], symbol=v.get("SIMBOLO", "")
        ),
        note="Signals come from the trained model: nothing to plot. The RL policy picks the "
        "position with the switching cost inside its objective: out of sample it beats buy and "
        "hold on 11 of 15 assets and halves max drawdown, but its timing is only weakly above "
        "an exposure-matched random control.",
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
        note="Channel breakout, ADX-filtered, with an ATR trailing exit.",
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
        note="Enters when the Bollinger bands expand out of the Keltner channel.",
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
        note="The textbook trend system, kept as a benchmark.",
    ),
    "Confluence": Strategia(
        indicatori=("stop_confluenza", "confluenza", "piani_lunghi", "votanti"),
        # I parametri dei votanti si ricavano dal registro invece di essere elencati qui: e' cio'
        # che rende «aggiungere un votante» un'operazione sola. Un elenco a mano si sarebbe
        # disallineato al primo votante nuovo, e il disallineamento sarebbe stato invisibile.
        parametri=("CONF_INNESCO", *(p.config for v in confluence.VOTANTI for p in v.parametri)),
        esegui=lambda df, cache, v: _confluenza_lunga(df, v),
        note=(
            "Six voters on four timeframes derived from the one selected: trigger, confirmation, "
            "structure, regime. Confluence shows score against threshold, Higher planes shows the "
            "two long timeframes, Voters shows who drove it, and the dashed line on the candles is "
            "the trailing stop — which closes most trades. Voter parameters are frozen at their "
            "measured values and are not adjustable."
        ),
    ),
}


def _confluenza_lunga(df: pd.DataFrame, valori: dict) -> tuple[list, list]:
    """I segnali della confluenza, ognuno con **chi l'ha generato** attaccato.

    Il terzo elemento e' la sola differenza rispetto alle altre strategie, e c'e' per una ragione
    precisa: qui la decisione viene da sei votanti, e la posizione del marcatore sul grafico non
    dice quali abbiano parlato ne' con che contributo. Il grafico lo mostra al passaggio del
    mouse; senza, chi guarda vedrebbe un triangolo e dovrebbe crederci.
    """
    risultato = confluenza_di(df, valori)
    if risultato is None:
        return [], []
    compra, vende = _solo_lunghe(risultato.eventi)
    return (
        [(quando, prezzo, risultato.spiega(quando)) for quando, prezzo in compra],
        [(quando, prezzo, risultato.spiega(quando)) for quando, prezzo in vende],
    )


VUOTA = "-"  # la voce che non seleziona nessuna strategia: si mostra tutto
CONFLUENZA = "Confluence"

# Nella panoramica senza strategia questi due si tolgono, perche' sarebbero doppioni visivi:
# `medie_trend` disegna due delle tre linee di `medie`, e `bande_kama` ha la stessa forma di
# `bande_atr` con finestra e moltiplicatore diversi. Mostrarli tutti metteva in legenda due
# "EMA fast" e due "Upper band", cioe' due etichette identiche su linee diverse -- che e'
# peggio di un'informazione mancante, perche' sembra un errore di lettura di chi guarda.
PANORAMICA_ESCLUSI = ("medie_trend", "bande_kama", "etichetta_swing")


# Quale misura copre quale intervallo. E' una **decisione**, scritta come dato invece che calcolata:
# le griglie sono girate su quattro intervalli, la pagina ne offre nove, e dire "il piu' vicino" e'
# gia' una scelta -- 30m sta in mezzo fra 15m e 1h, e va deciso da che parte cade.
#
# Sotto l'ora nessuna misura di questo progetto ha mai trovato qualcosa che batta il possesso
# passivo: i default a 15m sono i migliori **fra quelli provati**, non buoni. La pagina lo dice.
ANCORA_MISURATA: dict[str, str] = {
    "1m": "15m",
    "3m": "15m",
    "5m": "15m",
    "15m": "15m",
    "30m": "1h",
    "1h": "1h",
    "2h": "1h",
    "4h": "4h",
    "1d": "1d",
}


def ancora_di(intervallo: str) -> str | None:
    """L'intervallo misurato che fa da riferimento per quello scelto, o `None` se non ce n'e' uno."""
    from cryptofarm.trading.tuned_defaults import PER_INTERVALLO

    ancora = ANCORA_MISURATA.get(intervallo)
    return ancora if ancora in PER_INTERVALLO else None


def valori_misurati(strategia: str, intervallo: str) -> dict:
    """I soli parametri per cui una misura ha scelto un valore diverso da quello di `config`.

    Vuoto quando la strategia non e' stata misurata a quell'intervallo, quando il parametro non
    discrimina, o quando la scelta non regge sulla meta' dei dati: in tutti e tre i casi resta il
    default scritto a mano, che e' la scelta prudente.
    """
    from cryptofarm.trading.tuned_defaults import PER_INTERVALLO

    if strategia == CONFLUENZA:
        # Ogni votante gira sul **suo** piano, quindi il suo valore misurato e' quello
        # dell'intervallo del piano -- non di quello scelto nella pagina. Prendere quello della
        # pagina darebbe alla confluenza i default di un altro timeframe, in silenzio.
        misurati: dict = {}
        for votante in confluence.VOTANTI:
            misurati.update(valori_del_piano(votante, intervallo))
        return misurati

    ancora = ancora_di(intervallo)
    return dict(PER_INTERVALLO.get(ancora, {}).get(strategia, {})) if ancora else {}


def valori_del_piano(votante, intervallo: str) -> dict:
    """I valori misurati di un votante, tradotti nei nomi dei suoi widget.

    L'ancora serve anche qui: le griglie coprono quattro intervalli e un piano puo' caderne fuori
    (a base 1h il piano di struttura e' 16h, che nessuno ha misurato). In quel caso non si
    sostituisce niente e restano i default di `config`, che e' la scelta prudente.
    """
    from cryptofarm.trading.tuned_defaults import PER_INTERVALLO

    minuti = interval_to_minutes(intervallo) * confluence.FATTORI[votante.piano]
    ancora = ancora_di(confluence._intervallo(minuti))
    if not ancora or not votante.menu:
        return {}
    misurati = PER_INTERVALLO.get(ancora, {}).get(votante.menu, {})
    return {
        parametro.config: misurati[parametro.misurato]
        for parametro in votante.parametri
        if parametro.misurato in misurati
    }


def valori_predefiniti(strategia: str = "", intervallo: str = "") -> dict:
    """Il valore iniziale di ogni parametro noto, cioe' cosa vede la pagina prima che si tocchi
    qualcosa. Serve alla pagina come base su cui scrivere le scelte dei widget, e ai test come
    contesto per calcolare le serie.

    Con `strategia` e `intervallo` i valori misurati per quella coppia si sovrappongono a quelli
    scritti a mano. Senza, si ottengono i default di `config` e basta -- che e' quel che serve ai
    test e a chi calcola una serie fuori dalla pagina.
    """
    from cryptofarm.trading import config

    valori = {
        nome: getattr(config, nome).value for nome in dir(config) if isinstance(getattr(config, nome), config.Param)
    }
    valori["CONFIRM_VOLUME"] = config.CONFIRM_VOLUME
    valori["CONF_IN_FORMAZIONE"] = config.CONF_IN_FORMAZIONE
    # L'intervallo e' un parametro come gli altri per la confluenza, che da li' ricava i suoi
    # quattro piani. La pagina lo sovrascrive con quello scelto; fuori dalla pagina resta questo.
    valori["INTERVALLO"] = config.INTERVALS[config.INTERVAL_INDEX]
    valori["REQUIRE_CLOUD"] = config.REQUIRE_CLOUD
    if strategia and intervallo:
        valori.update(valori_misurati(strategia, intervallo))
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
    return pannelli_degli(indicatori_di(strategia))


def pannelli_degli(chiavi) -> list[str]:
    """Come sopra, ma per un elenco qualunque: la pagina puo' aggiungere l'etichetta del modello,
    che non appartiene a nessuna strategia."""
    titoli: list[str] = []
    for chiave in chiavi:
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

# -------------------------------------------------------------------------------------------------
# Come si presentano nella barra laterale
# -------------------------------------------------------------------------------------------------
# Il nome della costante dice a cosa serve nel codice; l'etichetta dice cosa muove a chi guarda il
# grafico. `STOP_LOSS_PERCENT` e' "Stop loss %", `TRAIL_ATR_WINDOW` e' "Finestra ATR (uscita)" e non
# semplicemente "ATR", perche' nella stessa vista puo' esserci anche l'ATR delle bande.

ETICHETTE: dict[str, str] = {
    "ATR_MULTIPLIER": "ATR multiplier",
    "ATR_WINDOW": "ATR window",
    "RSI_SHORT": "RSI fast",
    "RSI_MEDIUM": "RSI mid",
    "RSI_LONG": "RSI slow",
    "EMA_SHORT": "EMA fast",
    "EMA_MEDIUM": "EMA mid",
    "EMA_LONG": "EMA slow",
    "KAMA_POW1": "KAMA power 1",
    "KAMA_POW2": "KAMA power 2",
    "RSI_BUY_LIMIT": "RSI buy threshold",
    "RSI_SELL_LIMIT": "RSI sell threshold",
    "STOP_LOSS_PERCENT": "Stop loss %",
    "NUM_CONDITIONS": "Conditions required",
    "PIVOT_WINDOW": "Swing window",
    "SWING_TARGET_WINDOW": "Target window (bars per side)",
    "CONF_THETA_BASE": "Entry threshold",
    "CONF_THETA_MACRO": "Threshold relief from higher planes",
    "CONF_ISTERESI": "Exit hysteresis",
    "CONF_BARRE_MINIME": "Minimum bars held (score exit only)",
    "CONF_PAZIENZA": "Bars below threshold before giving up",
    "CONF_EMIVITA": "Signal half-life (voter bars)",
    "CONF_W_MAX": "Weight cap per voter",
    "CONF_K_FAMIGLIE": "Families required to agree",
    "CONF_INNESCO": "Trigger breakout window",
    "CONF_ATR_WINDOW": "ATR window (trailing exit)",
    "CONF_ATR_MULT": "Stop distance (ATR)",
    "CONF_REGIME_EMA": "Regime plane EMA",
    "CONF_STRUTTURA_EMA": "Structure plane EMA",
    "CONF_ICHIMOKU_FAST": "Tenkan (fast)",
    "CONF_ICHIMOKU_SLOW": "Kijun (slow)",
    "CONF_ICHIMOKU_SPAN": "Cloud span",
    "CONF_ICHIMOKU_CLOUD": "Require cloud (1 = yes)",
    "CONF_FLOW_WINDOW": "OBV and MFI window",
    "CONF_FLOW_MFI_ALTO": "MFI ceiling for longs",
    "CONF_FLOW_MFI_BASSO": "MFI floor for shorts",
    "CONF_PULLBACK_REGIME_EMA": "Regime EMA",
    "CONF_PULLBACK_STOCH_WINDOW": "StochRSI window",
    "CONF_PULLBACK_STOCH_SMOOTH": "StochRSI smoothing",
    "CONF_PULLBACK_OVERSOLD": "Oversold level",
    "CONF_PULLBACK_OVERBOUGHT": "Overbought level",
    "CONF_PULLBACK_ATR_MULT": "Stop distance (ATR)",
    "CONF_BANDE_KAMA": "KAMA window (confirmation)",
    "CONF_BANDE_BAND_MULT": "Band width (ATR, confirmation)",
    "CONF_BANDE_STOP_MULT": "Stop distance (ATR, confirmation)",
    "CONF_BANDE_KAMA_VELOCE": "KAMA window (trigger)",
    "CONF_BANDE_BAND_MULT_VELOCE": "Band width (ATR, trigger)",
    "CONF_BANDE_STOP_MULT_VELOCE": "Stop distance (ATR, trigger)",
    "CONF_ZONE_FAST": "Fast EMA (regime)",
    "CONF_ZONE_SLOW": "Slow EMA (regime)",
    "CONF_ZONE_FAST_STRUTTURA": "Fast EMA (structure)",
    "CONF_ZONE_SLOW_STRUTTURA": "Slow EMA (structure)",
    "CONF_MODELLO_ENTRA": "Model |prediction| in",
    "CONF_MODELLO_ESCI": "Model |prediction| out",
    "ADX_WINDOW": "ADX window",
    "ADX_MIN": "ADX minimum (trend required)",
    "ADX_MAX": "ADX maximum (range required)",
    "REGIME_EMA": "Regime EMA",
    "TRAIL_ATR_WINDOW": "ATR window (exit)",
    "DONCHIAN_CHANNEL": "Channel length",
    "DONCHIAN_ATR_MULT": "Stop distance (ATR)",
    "BB_WINDOW": "Bollinger window",
    "BB_DEV": "Bollinger deviations",
    "KC_WINDOW": "Keltner window",
    "KC_MULTIPLIER": "Keltner multiplier",
    "OBV_WINDOW": "OBV window",
    "SQUEEZE_ATR_MULT": "Stop distance (ATR)",
    "STOCHRSI_WINDOW": "StochRSI window",
    "STOCHRSI_SMOOTH": "StochRSI smoothing",
    "STOCH_OVERSOLD": "Oversold threshold",
    "STOCH_OVERBOUGHT": "Overbought threshold",
    "PULLBACK_ATR_MULT": "Stop distance (ATR)",
    "ICHIMOKU_FAST": "Tenkan",
    "ICHIMOKU_SLOW": "Kijun",
    "ICHIMOKU_SPAN": "Senkou B",
    "REVERSION_KAMA_WINDOW": "KAMA window",
    "REVERSION_BAND_MULT": "Band width (ATR)",
    "REVERSION_STOP_MULT": "Stop distance (ATR)",
    "REVERSION_REGIME_EMA": "Regime EMA (0 = off)",
}


SOGLIE = "Strategy thresholds"

# Le strategie che nella pagina passano dal motore classico: quello non addebita il costo di
# mantenimento e non conosce la leva, quindi i loro numeri qui sono piu' ottimisti di quelli
# misurati. La pagina lo dice accanto al risultato invece di lasciarlo scoprire per confronto.
NUOVE_SENZA_MANTENIMENTO = (
    "Donchian Breakout",
    "Squeeze Breakout",
    "Ichimoku Trend",
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
    if strategia == CONFLUENZA:
        # Un riquadro per votante invece di trentuno campi in fila sotto un titolo solo. I nomi
        # arrivano dal registro, quindi un votante aggiunto porta con se' il proprio riquadro.
        for votante in confluence.VOTANTI:
            dentro = [p.config for p in votante.parametri if p.config not in gia_visti]
            gia_visti.update(dentro)
            if dentro:
                gruppi.append((f"Voter · {votante.nome}", dentro))
    propri = [nome for nome in parametri_di(strategia) if nome not in gia_visti]
    if propri:
        gruppi.append((SOGLIE, propri))
    return gruppi
