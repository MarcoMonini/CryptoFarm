"""Etichettatura per directional change, con pivot confermati ed etichetta morbida.

Due scelte che distinguono questo modulo dallo zigzag classico, entrambe per la stessa ragione:
allineare l'etichetta a cio' che si puo' davvero sapere e a cio' che rende davvero.

**I pivot sono datati alla barra di conferma, non all'estremo.** Un minimo non e' conoscibile
quando si forma: lo diventa quando il prezzo si e' invertito della soglia. Lo zigzag "come
disegnato" colloca il pivot sull'estremo esatto, e costruirci sopra feature o stato inserisce
look-ahead -- il meccanismo per cui i backtest su pattern sembrano eccellenti e non sopravvivono
al mercato reale. Qui l'estremo e la sua conferma sono due indici distinti e separati per
costruzione.

**L'etichetta e' morbida.** Marcare BUY solo la barra del minimo esatto punisce il modello per
aver segnalato una barra a due candele di distanza, che pero' economicamente vale quasi lo
stesso. Qui e' BUY ogni barra da cui si cattura almeno una frazione data della gamba successiva:
la funzione di perdita torna allineata all'economia, e i positivi passano da qualche punto
percentuale a una frazione che non richiede pesature artificiose per essere apprendibile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOLD, BUY, SELL = 0, 1, 2
LABEL_NAMES = {HOLD: "hold", BUY: "buy", SELL: "sell"}

DEFAULT_THRESHOLD = 0.005  # 0,5% di inversione per confermare un estremo
DEFAULT_CAPTURE = 0.60  # frazione minima della gamba catturabile perche' la barra sia un segnale


def directional_change_pivots(high: np.ndarray, low: np.ndarray, threshold: float) -> pd.DataFrame:
    """Estremi locali per inversione di direzione, con la barra in cui diventano noti.

    Colonne: `extreme_bar` (dove il prezzo ha girato), `confirm_bar` (quando lo si e' potuto
    sapere), `price`, `kind` (+1 massimo, -1 minimo).

    La differenza fra le due colonne e' il **ritardo di conferma**, ed e' esattamente
    l'informazione che uno zigzag retrospettivo nasconde.
    """
    pivots: list[tuple[int, int, float, int]] = []
    direction = 0
    max_bar, max_price = 0, float(high[0])
    min_bar, min_price = 0, float(low[0])

    for bar in range(1, len(high)):
        if direction > 0:
            # In salita si insegue il massimo finche' il prezzo non ritraccia della soglia.
            if high[bar] > max_price:
                max_bar, max_price = bar, float(high[bar])
            elif low[bar] <= max_price * (1 - threshold):
                pivots.append((max_bar, bar, max_price, 1))
                direction = -1
                min_bar, min_price = bar, float(low[bar])
        elif direction < 0:
            if low[bar] < min_price:
                min_bar, min_price = bar, float(low[bar])
            elif high[bar] >= min_price * (1 + threshold):
                pivots.append((min_bar, bar, min_price, -1))
                direction = 1
                max_bar, max_price = bar, float(high[bar])
        else:
            # Direzione non ancora decisa: si tengono entrambi gli estremi e vince il primo che
            # viene ritracciato della soglia. Tenerli separati e' necessario -- aggiornarli nello
            # stesso ramo farebbe sovrascrivere l'uno all'altro.
            if high[bar] > max_price:
                max_bar, max_price = bar, float(high[bar])
            if low[bar] < min_price:
                min_bar, min_price = bar, float(low[bar])
            if low[bar] <= max_price * (1 - threshold):
                pivots.append((max_bar, bar, max_price, 1))
                direction = -1
                min_bar, min_price = bar, float(low[bar])
            elif high[bar] >= min_price * (1 + threshold):
                pivots.append((min_bar, bar, min_price, -1))
                direction = 1
                max_bar, max_price = bar, float(high[bar])

    return pd.DataFrame(pivots, columns=["extreme_bar", "confirm_bar", "price", "kind"])


def leg_table(pivots: pd.DataFrame) -> pd.DataFrame:
    """Gambe fra estremi consecutivi, con ampiezza e quanto ne resta alla conferma.

    `capturable_at_confirm` e' la domanda centrale della strategia: quando il minimo diventa noto
    il prezzo e' gia' risalito della soglia, quindi **il modello non puo' imparare "questo e' il
    minimo"** -- puo' solo imparare "questo movimento appena iniziato prosegue". Questa colonna
    misura quanto di quel movimento resta effettivamente da prendere.
    """
    if len(pivots) < 2:
        return pd.DataFrame(
            columns=[
                "start_bar",
                "end_bar",
                "confirm_bar",
                "start_price",
                "end_price",
                "direction",
                "size",
                "capturable_at_confirm",
            ]
        )
    rows = []
    for position in range(len(pivots) - 1):
        start, end = pivots.iloc[position], pivots.iloc[position + 1]
        size = abs(end["price"] - start["price"]) / start["price"]
        rows.append(
            {
                "start_bar": int(start["extreme_bar"]),
                "end_bar": int(end["extreme_bar"]),
                "confirm_bar": int(start["confirm_bar"]),
                "start_price": float(start["price"]),
                "end_price": float(end["price"]),
                "direction": 1 if start["kind"] == -1 else -1,
                "size": size,
            }
        )
    return pd.DataFrame(rows)


def capturable_fraction(legs: pd.DataFrame, close: np.ndarray) -> pd.DataFrame:
    """Frazione della gamba ancora disponibile alla barra di conferma del suo inizio."""
    result = legs.copy()
    if result.empty:
        result["capturable_at_confirm"] = []
        return result
    entry = close[result["confirm_bar"].to_numpy()]
    start_price = result["start_price"].to_numpy()
    end_price = result["end_price"].to_numpy()
    span = end_price - start_price
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(np.abs(span) > 0, (end_price - entry) / span, 0.0)
    result["entry_price"] = entry
    result["capturable_at_confirm"] = fraction
    return result


def soft_labels(
    close: np.ndarray,
    pivots: pd.DataFrame,
    capture: float = DEFAULT_CAPTURE,
) -> np.ndarray:
    """Etichetta morbida: BUY/SELL su ogni barra da cui resta catturabile almeno `capture`.

    Per ogni gamba, la frazione catturabile dalla barra `b` e' `(P_fine - Close_b) / (P_fine -
    P_inizio)` su una gamba al rialzo, e simmetricamente al ribasso. La finestra considerata va
    dall'estremo precedente all'estremo finale della gamba, quindi la zona di segnale copre sia
    la discesa verso il minimo sia l'inizio della risalita -- che e' il punto: due barre prima o
    dopo il minimo esatto valgono quasi lo stesso, e l'etichetta deve dirlo.
    """
    labels = np.zeros(len(close), dtype=np.int8)
    if len(pivots) < 2:
        return labels

    for position in range(len(pivots) - 1):
        start, end = pivots.iloc[position], pivots.iloc[position + 1]
        start_bar, end_bar = int(start["extreme_bar"]), int(end["extreme_bar"])
        span = end["price"] - start["price"]
        if end_bar <= start_bar or span == 0:
            continue
        # La finestra parte dall'estremo precedente: la discesa finale verso un minimo e' gia'
        # zona di acquisto, perche' da li' si cattura quasi tutta la gamba successiva.
        window_start = int(pivots.iloc[position - 1]["extreme_bar"]) if position > 0 else start_bar
        window = np.arange(window_start, end_bar + 1)
        fraction = (end["price"] - close[window]) / span
        signal = BUY if span > 0 else SELL
        labels[window[fraction >= capture]] = signal
    return labels


def label_distribution(labels: np.ndarray) -> dict[str, float]:
    return {name: float((labels == code).mean()) for code, name in LABEL_NAMES.items()}


def tune_threshold(
    high: np.ndarray,
    low: np.ndarray,
    days: float,
    target_per_day: tuple[float, float] = (8.0, 12.0),
    candidates: tuple[float, ...] = (0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.015),
) -> tuple[float, float]:
    """Soglia che porta il numero di estremi al giorno dentro la fascia richiesta.

    Si sceglie la soglia il cui tasso cade nella fascia; se nessuna ci cade, quella che le si
    avvicina di piu'. Va tarata **per simbolo**: la stessa soglia percentuale produce tassi molto
    diversi su asset con volatilita' diverse.
    """
    lower, upper = target_per_day
    centre = (lower + upper) / 2
    best, best_rate, best_distance = candidates[0], 0.0, float("inf")
    for threshold in candidates:
        rate = len(directional_change_pivots(high, low, threshold)) / days
        if lower <= rate <= upper:
            distance = abs(rate - centre)
        else:
            distance = abs(rate - centre) + 1000
        if distance < best_distance:
            best, best_rate, best_distance = threshold, rate, distance
    return best, best_rate
