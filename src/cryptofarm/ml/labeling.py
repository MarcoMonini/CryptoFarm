"""Definizione delle etichette: quando comprare, quando vendere, quando stare fermi.

Il metodo di riferimento e' il **triple-barrier**: per ogni candela si fissano tre barriere --
un take-profit sopra, uno stop-loss sotto, e un limite temporale -- e l'etichetta e' quale
viene toccata per prima.

Perche' questo e non i minimi/massimi locali (il metodo precedente, tenuto in fondo al modulo
per confronto):

- E' definito su **ogni** candela, non solo sulle poche che sono estremi. Su 350.000 candele il
  metodo per estremi ne etichettava ~12.000 (3,5%); qui sono 350.000.
- La distribuzione risulta naturalmente equilibrata, quindi non serve nessun downsampling --
  e senza downsampling il modello resta calibrato sulla frequenza reale degli eventi.
- L'etichetta non e' una proprieta' geometrica della curva ma **l'esito di un trade**: "comprando
  qui con questo TP e questo SL, il TP arriva prima dello SL". La precision del modello e' quindi
  direttamente il win rate della strategia, senza bisogno di interpretazioni.
- Non e' una lama di coltello: il metodo per estremi chiede di indovinare la candela esatta, e
  un segnale una candela in anticipo conta come errore totale pur valendo quasi lo stesso.

Le barriere sono **proporzionali alla volatilita' (ATR)** e non percentuali fisse: una soglia
dell'1% e' un movimento raro su BTC in un'ora e rumore su un'altcoin in un giorno. Con un
pavimento legato alle commissioni, perche' un'etichetta "vincente" il cui movimento non copre
0,2% di andata e ritorno insegna al modello a perdere soldi.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import argrelextrema

# Etichette. Coincidono con la convenzione gia' usata da `trading/simulator.py`.
HOLD, BUY, SELL = 0, 1, 2
LABEL_NAMES = {HOLD: "hold", BUY: "buy", SELL: "sell"}

# Parametri di riferimento del triple-barrier.
#
# Le barriere sono **asimmetriche**, e la ragione e' aritmetica. Con commissioni f di andata e
# ritorno, barriera di profitto u_tp e barriera di perdita u_sl, la precision necessaria solo per
# andare in pari e'  p = (u_sl + f) / ((u_tp - f) + (u_sl + f)).  Con barriere simmetriche allo
# 0,6% e f = 0,2% servirebbe il **66,7%** di precision -- un'asticella che nessun modello su dati
# di mercato tiene stabilmente. Con il take-profit al doppio dello stop-loss la stessa soglia
# scende al **44,4%**, che e' un obiettivo realistico.
#
# Il prezzo di questa scelta e' che "profitto prima della perdita" diventa piu' raro (circa un
# terzo delle candele invece di meta'): meno esempi positivi, ma ognuno vale molto di piu'.
TP_ATR_MULTIPLE = 1.5  # take-profit in multipli dell'ATR corrente
SL_ATR_MULTIPLE = 1.0  # stop-loss: meta' del take-profit
HORIZON_BARS = 96  # limite temporale in barre
ROUND_TRIP_FEE = 0.002  # 0,1% per lato su spot Binance
FEE_FLOOR_MULTIPLE = 3.0  # nessuna barriera sotto 3x le commissioni di andata e ritorno

# Le finestre di look-ahead vengono elaborate a blocchi: la matrice completa dei futuri di 11
# milioni di candele x 96 barre non entrerebbe in memoria, un blocco alla volta si'.
CHUNK_ROWS = 200_000


def barrier_widths(
    atr_percent: pd.Series,
    tp_multiple: float = TP_ATR_MULTIPLE,
    sl_multiple: float = SL_ATR_MULTIPLE,
    round_trip_fee: float = ROUND_TRIP_FEE,
    fee_floor_multiple: float = FEE_FLOOR_MULTIPLE,
) -> tuple[np.ndarray, np.ndarray]:
    """Ampiezza delle barriere per ogni candela, in frazione del prezzo.

    `atr_percent` e' l'ATR gia' espresso in percentuale del Close (come lo produce
    `features.normalize_indicators`).
    """
    floor = round_trip_fee * fee_floor_multiple
    atr_fraction = atr_percent.to_numpy(dtype=float) / 100.0

    # Il pavimento si applica allo stop, e il take-profit lo segue mantenendo il rapporto. Se si
    # applicasse a entrambi separatamente, sugli asset a bassa volatilita' (dove il pavimento
    # morde) le due barriere collasserebbero allo stesso valore, riportando il rapporto a 1:1 e
    # con esso il break-even al 66,7% -- esattamente il caso che l'asimmetria evita.
    stop_loss = np.maximum(atr_fraction * sl_multiple, floor)
    take_profit = stop_loss * (tp_multiple / sl_multiple)
    return take_profit, stop_loss


def triple_barrier_events(
    df: pd.DataFrame,
    horizon: int = HORIZON_BARS,
    tp_multiple: float = TP_ATR_MULTIPLE,
    sl_multiple: float = SL_ATR_MULTIPLE,
    round_trip_fee: float = ROUND_TRIP_FEE,
    fee_floor_multiple: float = FEE_FLOOR_MULTIPLE,
) -> pd.DataFrame:
    """Etichetta ogni candela con l'esito del trade che vi si aprirebbe, **e con la sua durata**.

    Ingresso al Close della candela; le barriere vengono verificate sui massimi e minimi delle
    candele **successive** (t+1 .. t+horizon), mai su quella corrente.

    Se entrambe le barriere risultano toccate nella stessa candela l'esito e' ambiguo -- i dati
    OHLC non dicono in che ordine il prezzo le ha raggiunte dentro la barra -- e viene assegnato
    l'esito peggiore (SELL). Assumere il migliore produrrebbe un modello ottimista che in
    esecuzione reale trova sistematicamente meno di quanto si aspetta.

    Colonne restituite:

    - `Label`      esito: 0 hold (timeout), 1 buy (TP per primo), 2 sell (SL per primo)
    - `exit_bar`   posizione della candela di uscita
    - `t_exit`     timestamp della candela di uscita
    - `exit_return` rendimento realizzato, lordo di commissioni
    - `tp_width` / `sl_width`  ampiezza effettiva delle barriere, in frazione del prezzo

    `t_exit` non e' un dettaglio diagnostico: e' il dato senza cui il **purging** della
    cross-validation non e' calcolabile. Due osservazioni le cui vite si sovrappongono
    condividono futuro, e per saperlo serve sapere quando ciascuna finisce.
    """
    if "ATR" not in df.columns:
        raise KeyError("triple_barrier_events richiede la colonna ATR normalizzata in percentuale")

    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    take_profit, stop_loss = barrier_widths(df["ATR"], tp_multiple, sl_multiple, round_trip_fee, fee_floor_multiple)

    total = len(df)
    labels = np.zeros(total, dtype=np.int8)
    exit_bar = np.arange(total, dtype=np.int64)
    exit_return = np.zeros(total, dtype=float)

    labelable = total - horizon
    if labelable <= 0:
        return pd.DataFrame(
            {
                "Label": labels,
                "exit_bar": exit_bar,
                "t_exit": df.index,
                "exit_return": exit_return,
                "tp_width": take_profit,
                "sl_width": stop_loss,
            },
            index=df.index,
        )

    # Finestra dei futuri: la riga i copre le candele i+1 .. i+horizon.
    future_high = sliding_window_view(high[1:], horizon)[:labelable]
    future_low = sliding_window_view(low[1:], horizon)[:labelable]
    future_close = sliding_window_view(close[1:], horizon)[:labelable]

    upper = close[:labelable] * (1.0 + take_profit[:labelable])
    lower = close[:labelable] * (1.0 - stop_loss[:labelable])

    for start in range(0, labelable, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, labelable)
        hit_upper = future_high[start:stop] >= upper[start:stop, None]
        hit_lower = future_low[start:stop] <= lower[start:stop, None]

        # argmax su un array booleano da il primo True; senza nessun True da 0, quindi il
        # risultato va reso valido solo dove un contatto c'e' stato davvero.
        never = horizon + 1
        first_upper = np.where(hit_upper.any(axis=1), hit_upper.argmax(axis=1), never)
        first_lower = np.where(hit_lower.any(axis=1), hit_lower.argmax(axis=1), never)

        size = stop - start
        chunk = np.full(size, HOLD, dtype=np.int8)
        # Timeout: uscita a mercato sull'ultima candela dell'orizzonte.
        bars = np.full(size, horizon, dtype=np.int64)
        returns = future_close[start:stop, -1] / close[start:stop] - 1.0

        won = first_upper < first_lower
        # `<=` e non `<`: a parita' di candela vince lo stop, per la ragione nel docstring.
        lost = (first_lower <= first_upper) & (first_lower != never)

        chunk[won] = BUY
        bars[won] = first_upper[won] + 1
        returns[won] = take_profit[start:stop][won]

        chunk[lost] = SELL
        bars[lost] = first_lower[lost] + 1
        returns[lost] = -stop_loss[start:stop][lost]

        labels[start:stop] = chunk
        exit_bar[start:stop] = np.arange(start, stop) + bars
        exit_return[start:stop] = returns

    # La coda senza futuro osservabile resta HOLD, con uscita su se stessa: non e' un trade.
    exit_bar[labelable:] = np.arange(labelable, total)
    exit_bar = np.minimum(exit_bar, total - 1)

    return pd.DataFrame(
        {
            "Label": labels,
            "exit_bar": exit_bar,
            "t_exit": df.index[exit_bar],
            "exit_return": exit_return,
            "tp_width": take_profit,
            "sl_width": stop_loss,
        },
        index=df.index,
    )


def triple_barrier_labels(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Solo le etichette. Comodita' per i chiamanti che non hanno bisogno delle durate."""
    return triple_barrier_events(df, **kwargs)["Label"].rename("Label")


def label_distribution(labels: np.ndarray | pd.Series) -> dict[str, float]:
    """Conteggi e percentuali per classe, per la diagnostica a ogni stadio della pipeline."""
    values = np.asarray(labels)
    if values.size == 0:
        return {}
    distribution = {}
    for code, name in LABEL_NAMES.items():
        count = int((values == code).sum())
        distribution[name] = count
        distribution[f"{name}_pct"] = count / values.size
    return distribution


def format_distribution(labels: np.ndarray | pd.Series, stage: str) -> str:
    """Riga di diagnostica leggibile con la distribuzione delle classi."""
    values = np.asarray(labels)
    if values.size == 0:
        return f"[{stage}] nessun campione"
    parts = [
        f"{name}={int((values == code).sum())} ({(values == code).mean():.1%})" for code, name in LABEL_NAMES.items()
    ]
    return f"[{stage}] {values.size} campioni | " + ", ".join(parts)


# ---------------------------------------------------------------------------------------------
# Metodo precedente, mantenuto per poter confrontare i due su dati identici.
# ---------------------------------------------------------------------------------------------


def apply_label_cooldown(labels: pd.Series, cooldown: int) -> pd.Series:
    """Impone una distanza minima in candele tra due segnali consecutivi."""
    if cooldown <= 0 or labels.empty:
        return labels
    values = labels.to_numpy(copy=True)
    last_kept = None
    for position in np.flatnonzero(values != HOLD):
        if last_kept is not None and (position - last_kept) <= cooldown:
            values[position] = HOLD
        else:
            last_kept = position
    return pd.Series(values, index=labels.index, name="Label")


def filter_labels_by_future_return(df: pd.DataFrame, labels: pd.Series, min_return: float, horizon: int) -> pd.Series:
    """Scarta i segnali non seguiti da un movimento di almeno `min_return` entro `horizon`."""
    if min_return <= 0 or horizon <= 0:
        return labels

    future_max = df["High"][::-1].rolling(horizon, min_periods=1).max()[::-1].shift(-1)
    future_min = df["Low"][::-1].rolling(horizon, min_periods=1).min()[::-1].shift(-1)
    upside = (future_max / df["Close"]) - 1.0
    downside = (future_min / df["Close"]) - 1.0

    filtered = labels.copy()
    # La forma negata scarta anche le ultime righe, dove il futuro non e' osservabile e il
    # confronto darebbe NaN (un `<` su NaN sarebbe False e terrebbe l'etichetta).
    filtered.loc[(filtered == BUY) & ~(upside >= min_return)] = HOLD
    filtered.loc[(filtered == SELL) & ~(downside <= -min_return)] = HOLD
    return filtered


def extrema_labels(
    df: pd.DataFrame,
    window_pivot: int = 25,
    min_return: float = 0.012,
    return_horizon: int = 48,
    cooldown: int = 8,
) -> pd.Series:
    """Minimi e massimi locali filtrati per rendimento futuro e distanziati da un cooldown."""
    order = max(2, int(window_pivot / 2))
    labels = pd.Series(HOLD, index=df.index, dtype="int64", name="Label")
    labels.iloc[argrelextrema(df["High"].values, np.greater, order=order)[0]] = SELL
    labels.iloc[argrelextrema(df["Low"].values, np.less, order=order)[0]] = BUY
    labels = filter_labels_by_future_return(df, labels, min_return, return_horizon)
    return apply_label_cooldown(labels, cooldown)
