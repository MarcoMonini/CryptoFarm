"""Costruzione della matrice di addestramento a partire da feature ed etichette.

Il percorso principale e' **tabellare**: una riga per candela, con la storia recente compressa
in ritardi (lag) a scale diverse invece che in una finestra completa. La ragione e' di scala --
con milioni di etichette il tensore tridimensionale di una finestra da 50 barre occuperebbe
decine di gigabyte, mentre la stessa informazione a ritardi crescenti sta in poche centinaia di
megabyte. I ritardi crescono in modo quasi geometrico (1, 2, 3, 5, 8, 13, ...): la risoluzione
e' fine sul passato prossimo e grossolana su quello remoto, che e' come l'informazione di mercato
e' effettivamente distribuita.

`create_sequences` resta disponibile per i modelli sequenziali, che hanno bisogno del tensore.

Due precauzioni contro il leakage, entrambe necessarie e nessuna sufficiente da sola:

- **Split temporale globale**, non per simbolo. Le criptovalute sono fortemente correlate:
  mettere il 2026 di BTC in training e il 2026 di ETH in validation significa validare su un
  periodo che il modello ha gia' visto attraverso un altro asset.
- **Embargo** dimensionato sull'orizzonte delle etichette. L'etichetta di una candela dipende
  dalle `horizon` candele successive, quindi le righe a cavallo del taglio condividono futuro
  con entrambi i lati.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptofarm.ml.features import FEATURES, PRICE_FEATURES

# Ritardi in barre su cui si guarda indietro. La progressione quasi geometrica copre da una
# barra a ~4,5 ore su 5m con sole 9 colonne per feature.
LAGS = (1, 2, 3, 5, 8, 13, 21, 34, 55)

# Una riga ogni quante candele. Le etichette di barre adiacenti condividono quasi tutto il loro
# futuro, quindi campionarle tutte moltiplica la dimensione del dataset senza aggiungere
# informazione -- e gonfia la fiducia nelle metriche di validation.
DEFAULT_STRIDE = 12

# Le feature non di prezzo di cui si tiene anche la storia.
_LAGGED_INDICATORS = ("RSI", "STOCH", "STOCH_S", "ATR", "TSI", "VOLUME")


def cusum_events(close: pd.Series, threshold_sigma: float = 3.0, volatility_window: int = 288) -> np.ndarray:
    """Posizioni degli eventi CUSUM: campiona quando **e' successo qualcosa**, non a orologio.

    Le time-bar campionano a intervalli regolari mentre l'informazione arriva a raffiche: su un
    mercato 24/7 gran parte delle barre notturne e' rumore e le barre nei momenti di attivita'
    aggregano troppo. Il filtro accumula i rendimenti in due somme, una per direzione, e segnala
    un evento quando una delle due supera la soglia, azzerandola.

    La soglia e' in **multipli della volatilita' locale**, non in percentuale fissa: misurato sui
    15 simboli in archivio, 3 sigma produce 30-35 eventi/giorno su ognuno, nonostante sigma vari
    di un fattore tre fra TRX e NEAR. E' cio' che rende il campionamento confrontabile fra asset
    ed epoche senza calibrazione per simbolo.
    """
    values = close.to_numpy(dtype=float)
    returns = np.zeros(len(values))
    returns[1:] = np.diff(np.log(values))
    sigma = pd.Series(returns).rolling(volatility_window, min_periods=volatility_window // 2).std().to_numpy()

    events = []
    positive = negative = 0.0
    for position in range(len(returns)):
        limit = sigma[position]
        if not np.isfinite(limit) or limit <= 0:
            continue
        limit *= threshold_sigma
        positive = max(0.0, positive + returns[position])
        negative = min(0.0, negative + returns[position])
        if positive > limit:
            positive = 0.0
            events.append(position)
        elif negative < -limit:
            negative = 0.0
            events.append(position)
    return np.array(events, dtype=np.int64)


def build_design_matrix(features: pd.DataFrame, lags: tuple[int, ...] = LAGS) -> pd.DataFrame:
    """Da frame di feature a matrice di progetto, una riga per candela.

    I prezzi entrano solo come **rendimenti logaritmici** su ciascun ritardo, mai come livelli:
    un livello assoluto non e' confrontabile tra asset ne' tra epoche, un rendimento si'.
    """
    columns: dict[str, pd.Series] = {}
    close = features["Close"]

    # Momento a scale diverse: quanto e' salito o sceso il prezzo negli ultimi L intervalli.
    for lag in lags:
        columns[f"RET_{lag}"] = np.log(close / close.shift(lag)) * 100

    # Forma della candela corrente, relativa alla sua chiusura.
    for column in PRICE_FEATURES:
        if column == "Close":
            continue
        columns[f"BAR_{column}"] = np.log(features[column] / close) * 100

    # Escursione recente: dove sta la chiusura dentro il range delle ultime L barre. Cattura
    # rotture e compressioni che i soli rendimenti non mostrano.
    for lag in lags:
        if lag < 3:
            continue
        highest = features["High"].rolling(lag).max()
        lowest = features["Low"].rolling(lag).min()
        span = (highest - lowest).replace(0, np.nan)
        columns[f"POS_{lag}"] = ((close - lowest) / span).fillna(0.5)

    # Indicatori: valore corrente e storia ai ritardi.
    for indicator in _LAGGED_INDICATORS:
        columns[indicator] = features[indicator]
        for lag in lags:
            columns[f"{indicator}_{lag}"] = features[indicator].shift(lag)

    columns["TIMEFRAME"] = features["TIMEFRAME"]

    matrix = pd.DataFrame(columns, index=features.index)
    return matrix.astype(np.float32)


def build_samples(
    features: pd.DataFrame,
    labels: pd.Series,
    expected_minutes: float,
    horizon: int,
    lags: tuple[int, ...] = LAGS,
    stride: int = DEFAULT_STRIDE,
    gap_tolerance: float = 1.5,
) -> tuple[pd.DataFrame, pd.Series]:
    """Matrice ed etichette pronte per una singola coppia simbolo/timeframe.

    Scarta le righe la cui storia o il cui futuro attraversano un buco temporale nella serie: i
    ritardi e le etichette ragionano in posizioni, non in timestamp, quindi attraverso un buco
    metterebbero in relazione momenti scorrelati.
    """
    matrix = build_design_matrix(features, lags)
    usable = matrix.notna().all(axis=1)

    # Le ultime `horizon` candele non hanno futuro osservabile: la loro etichetta e' HOLD per
    # mancanza di dati, non perche' non sia successo nulla.
    usable.iloc[-horizon:] = False

    if expected_minutes > 0:
        deltas = matrix.index.to_series().diff().dt.total_seconds().to_numpy() / 60.0
        is_gap = np.zeros(len(matrix), dtype=np.int64)
        is_gap[1:] = (deltas[1:] > expected_minutes * gap_tolerance).astype(np.int64)
        cumulative = np.concatenate([[0], np.cumsum(is_gap)])
        # Una riga e' valida se non ci sono buchi ne' nella sua storia (i ritardi) ne' nel suo
        # futuro (l'orizzonte dell'etichetta).
        positions = np.arange(len(matrix))
        starts = np.clip(positions - max(lags) + 1, 0, len(matrix))
        stops = np.clip(positions + horizon + 1, 0, len(matrix))
        usable &= (cumulative[stops] - cumulative[starts]) == 0

    selected = np.flatnonzero(usable.to_numpy())
    if stride > 1:
        selected = selected[::stride]

    return matrix.iloc[selected], labels.iloc[selected]


def time_split(
    index: pd.DatetimeIndex,
    train_fraction: float,
    embargo: pd.Timedelta,
) -> tuple[np.ndarray, np.ndarray]:
    """Maschere di training e validation separate da una data di taglio, con embargo.

    Il taglio e' sulla **data**, uguale per tutti i simboli: la correlazione tra criptovalute
    rende inutile qualunque separazione fatta simbolo per simbolo.
    """
    ordered = np.sort(index.unique())
    cutoff = pd.Timestamp(ordered[int(len(ordered) * train_fraction)])
    train = np.asarray(index < (cutoff - embargo))
    validation = np.asarray(index >= (cutoff + embargo))
    return train, validation


def create_sequences(
    features: pd.DataFrame,
    labels: pd.Series,
    window_size: int,
    columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Tensore (campioni, finestra, feature) per i modelli sequenziali.

    I prezzi sono normalizzati **dentro la finestra** come rendimenti logaritmici rispetto alla
    sua apertura: ogni finestra parte da zero, e non serve nessuna somma cumulata sull'intera
    serie che accumulerebbe magnitudini enormi per poi sottrarle di nuovo.
    """
    columns = columns or FEATURES
    values = features[columns].to_numpy(dtype=np.float32)
    count = len(values) - window_size
    if count <= 0:
        return np.empty((0, window_size, len(columns)), dtype=np.float32), np.empty(0, dtype=np.int8)

    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=window_size, axis=0)
    windows = np.moveaxis(windows, -1, 1)[:count].copy()

    price_columns = [position for position, name in enumerate(columns) if name in PRICE_FEATURES]
    open_column = columns.index("Open")
    # Copia obbligatoria: `windows` viene riscritto in place e senza copia questa sarebbe una
    # vista sui dati che stiamo modificando -- dopo la prima colonna la base sarebbe gia' zero.
    base = windows[:, 0, open_column].copy()[:, None]
    for position in price_columns:
        windows[:, :, position] = np.log(windows[:, :, position] / base) * 100

    return windows, labels.to_numpy()[window_size:].astype(np.int8)
