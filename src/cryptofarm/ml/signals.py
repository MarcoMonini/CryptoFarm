"""Dai punteggi del modello alle operazioni.

Il modello risponde a una sola domanda: **se compro su questa candela, il take-profit arriva
prima dello stop-loss?** E' un segnale di ingresso, e non ne esiste uno simmetrico di uscita --
la classe "sell" delle etichette significa "brutto momento per comprare", non "buon momento per
vendere". Trattarla come un segnale di vendita produce un diluvio di vendite (quella classe copre
circa il 60% delle candele) e, cosa peggiore, rompe la corrispondenza fra le etichette su cui il
modello e' stato valutato e le operazioni effettivamente simulate: i numeri di aspettativa non
descriverebbero piu' nulla.

L'uscita e' definita dalle **stesse barriere che definiscono le etichette**: take-profit,
stop-loss e limite temporale, calcolati dall'ATR al momento dell'ingresso. E' cio' che rende il
P&L simulato la traduzione diretta del win rate misurato in validation.

Come effetto collaterale i segnali risultano perfettamente alternati (un acquisto, la sua
vendita, il successivo acquisto), che e' anche l'unico modo in cui l'accoppiamento per indice di
`simulate_trading_with_commisions` produce operazioni sensate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptofarm.data.klines import interval_to_minutes
from cryptofarm.ml.dataset import build_design_matrix, cusum_events
from cryptofarm.ml.features import build_feature_frame
from cryptofarm.ml.labeling import BUY, HORIZON_BARS, SL_ATR_MULTIPLE, TP_ATR_MULTIPLE, barrier_widths
from cryptofarm.ml.models import predict_proba


def interval_from_index(index: pd.DatetimeIndex) -> str:
    """Deduce l'intervallo delle candele dalla loro spaziatura mediana."""
    if len(index) < 2:
        return "15m"
    minutes = int(round(np.median(np.diff(index.to_numpy()).astype("timedelta64[m]").astype(float))))
    return f"{minutes}m" if minutes < 60 else f"{minutes // 60}h"


def buy_probabilities(df: pd.DataFrame, model) -> pd.Series:
    """P(take-profit prima dello stop-loss) per ogni candela in cui e' calcolabile.

    Le feature vengono ricostruite dai soli OHLCV con le costanti di questo pacchetto invece di
    riusare le colonne che il chiamante ha gia' in tabella: la dashboard le calcola con i periodi
    scelti dai suoi slider, e un modello alimentato con feature diverse da quelle
    dell'addestramento sbaglia senza dare nessun segno.
    """
    features = build_feature_frame(df, interval_from_index(df.index))
    matrix = build_design_matrix(features)
    matrix = matrix[matrix.notna().all(axis=1)]
    if matrix.empty:
        return pd.Series(dtype=float)
    probabilities = predict_proba(model, matrix.to_numpy())
    return pd.Series(probabilities[:, BUY], index=matrix.index, name="P_buy")


def barrier_signals(
    df: pd.DataFrame,
    model,
    threshold: float,
    horizon: int = HORIZON_BARS,
    tp_multiple: float = TP_ATR_MULTIPLE,
    sl_multiple: float = SL_ATR_MULTIPLE,
) -> tuple[list[tuple], list[tuple]]:
    """Genera le operazioni: ingresso sul punteggio del modello, uscita sulle barriere.

    Restituisce `(buy_signals, sell_signals)` come liste di `(timestamp, prezzo)`, alternate e
    della stessa lunghezza salvo una posizione ancora aperta a fine serie.

    Il prezzo di uscita e' il livello della barriera toccata, non la chiusura della candela: e'
    l'approssimazione che corrisponde a un ordine gia' piazzato sul book. Se in una stessa candela
    risultano toccate entrambe le barriere l'esito e' ambiguo -- l'OHLC non dice in che ordine il
    prezzo le ha raggiunte -- e viene assegnato lo stop, la stessa convenzione pessimistica usata
    nell'etichettatura.
    """
    features = build_feature_frame(df, interval_from_index(df.index))
    if features.empty:
        return [], []

    scores = buy_probabilities(df, model)
    if scores.empty:
        return [], []

    take_profit, stop_loss = barrier_widths(features["ATR"], tp_multiple=tp_multiple, sl_multiple=sl_multiple)
    probability = scores.reindex(features.index).to_numpy()
    high = features["High"].to_numpy(dtype=float)
    low = features["Low"].to_numpy(dtype=float)
    close = features["Close"].to_numpy(dtype=float)
    timestamps = features.index

    buy_signals: list[tuple] = []
    sell_signals: list[tuple] = []

    position = 0
    while position < len(close):
        if not (probability[position] >= threshold):
            position += 1
            continue

        entry_price = close[position]
        target = entry_price * (1.0 + take_profit[position])
        stop = entry_price * (1.0 - stop_loss[position])
        deadline = min(position + horizon, len(close) - 1)
        buy_signals.append((timestamps[position], float(entry_price)))

        exit_position = deadline
        exit_price = close[deadline]
        for step in range(position + 1, deadline + 1):
            if low[step] <= stop:
                exit_position, exit_price = step, stop
                break
            if high[step] >= target:
                exit_position, exit_price = step, target
                break

        sell_signals.append((timestamps[exit_position], float(exit_price)))
        # Nessuna nuova posizione prima che la precedente sia chiusa: il modello stima l'esito di
        # un ingresso isolato, non di posizioni sovrapposte.
        position = exit_position + 1

    return buy_signals, sell_signals


def meta_signals(
    df: pd.DataFrame,
    model,
    threshold: float,
    horizon_hours: float = 24.0,
    tp_multiple: float = 1.5,
    sl_multiple: float = 1.0,
    round_trip_fee: float = 0.0012,
    fee_floor_multiple: float = 5.0,
    cusum_sigma: float = 3.0,
    limit_offset_atr: float = 0.5,
    limit_patience: int = 12,
) -> tuple[list[tuple], list[tuple]]:
    """Catena completa della strategia meta: primario CUSUM, secondario, esecuzione a limite.

    Riproduce esattamente cio' che il modello e' stato addestrato a prevedere, e nell'ordine in
    cui e' stato valutato:

    1. il **primario** (filtro CUSUM) propone un candidato quando il prezzo ha accumulato un
       movimento di dimensione rilevante;
    2. il **secondario** assegna la probabilita' che quell'ingresso chiuda in profitto netto, e
       si opera solo sopra soglia;
    3. l'ingresso e' un **ordine limite** sotto il prezzo, che puo' non riempirsi -- e in quel
       caso non c'e' nessun trade, non un trade a prezzo di mercato;
    4. l'uscita e' la barriera toccata per prima, calcolata dall'ATR al momento dell'ingresso.

    Rispettare questa catena non e' pedanteria: e' cio' che rende il P&L simulato la traduzione
    diretta dell'aspettativa misurata in cross-validation. Cambiare un anello -- entrare a
    mercato invece che a limite, uscire su un segnale invece che su una barriera -- scollega i
    due numeri senza dare nessun segnale che sia successo.
    """
    from cryptofarm.ml.execution import limit_fills
    from cryptofarm.ml.labeling import barrier_widths

    interval = interval_from_index(df.index)
    minutes = interval_to_minutes(interval)
    features = build_feature_frame(df, interval)
    if features.empty:
        return [], []

    matrix = build_design_matrix(features)
    usable = matrix.notna().all(axis=1).to_numpy()
    events = cusum_events(features["Close"], cusum_sigma)
    events = events[usable[events]]
    if len(events) == 0:
        return [], []

    scores = predict_proba(model, matrix.iloc[events].to_numpy())[:, 1]
    candidates = events[scores >= threshold]
    if len(candidates) == 0:
        return [], []

    fills = limit_fills(features, candidates, offset_atr=limit_offset_atr, patience=limit_patience)
    take_profit, stop_loss = barrier_widths(
        features["ATR"], tp_multiple, sl_multiple, round_trip_fee, fee_floor_multiple
    )

    high = features["High"].to_numpy(dtype=float)
    low = features["Low"].to_numpy(dtype=float)
    close = features["Close"].to_numpy(dtype=float)
    timestamps = features.index
    horizon_bars = int(horizon_hours * 60 / minutes)

    buy_signals: list[tuple] = []
    sell_signals: list[tuple] = []
    busy_until = -1

    for row, position in enumerate(candidates):
        if position <= busy_until or not fills["filled"].iloc[row]:
            continue
        entry_bar = int(fills["fill_bar"].iloc[row])
        entry_price = float(fills["fill_price"].iloc[row])
        target = entry_price * (1.0 + take_profit[position])
        stop = entry_price * (1.0 - stop_loss[position])
        deadline = min(entry_bar + horizon_bars, len(close) - 1)

        exit_position, exit_price = deadline, close[deadline]
        for step in range(entry_bar + 1, deadline + 1):
            if low[step] <= stop:
                exit_position, exit_price = step, stop
                break
            if high[step] >= target:
                exit_position, exit_price = step, target
                break

        buy_signals.append((timestamps[entry_bar], entry_price))
        sell_signals.append((timestamps[exit_position], float(exit_price)))
        # Una posizione alla volta: il modello stima l'esito di un ingresso isolato.
        busy_until = exit_position

    return buy_signals, sell_signals
