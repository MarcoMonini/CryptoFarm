"""Addestramento del classificatore LSTM bidirezionale (hold / buy / sell).

Il modello consuma *movimento relativo* di prezzo (variazioni percentuali) piu' un set di
indicatori tecnici scale-free, su finestre scorrevoli di `WINDOW_SIZE` candele.

La qualita' del modello dipende quasi interamente dalla qualita' delle etichette: marcare ogni
minimo/massimo locale come buy/sell (anche una fluttuazione di rumore dello 0,1%) produce un
dataset in cui la classe "hold" e' oltre il 98% e i segnali sono in gran parte impredicibili.
Il labeling qui e' quindi filtrato in tre stadi -- estremi locali, rendimento futuro minimo,
cooldown tra segnali consecutivi -- e il bilanciamento e' fatto a livello di *dati*
(downsampling di "hold") invece che solo a livello di loss.

Lo stesso preprocessing e' usato in inferenza da `get_model_predictions`, che
`trading/simulator.py` chiama per la strategia "AI Model": ogni modifica alle feature o alla
loro normalizzazione va fatta nelle funzioni condivise, non duplicata nei due path.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import argrelextrema
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from ta.momentum import RSIIndicator, StochasticOscillator, TSIIndicator
from ta.volatility import AverageTrueRange
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, BatchNormalization, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from cryptofarm.paths import MODELS_DIR

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
]

# Colonne che `create_sequences` riporta a zero sull'apertura della finestra: sono le uniche
# espresse in punti percentuali cumulati, il resto delle feature e' gia' scale-free.
PRICE_FEATURES = ("Open", "High", "Low", "Close")

# --- Indicatori tecnici -------------------------------------------------------------------
ATR_WINDOW = 6  # Periodo dell'ATR
RSI_WINDOW = 12  # Periodo dell'RSI

# --- Sequenze -----------------------------------------------------------------------------
WINDOW_SIZE = 50  # Numero di candele in ingresso al modello per ogni predizione
GAP_TOLERANCE = 1.5  # Multiplo del passo atteso oltre il quale due candele sono "non contigue"

# --- Labeling -----------------------------------------------------------------------------
# EXT_WINDOW_SIZE controlla quanto deve essere isolato un estremo per essere candidato
# (argrelextrema usa order = EXT_WINDOW_SIZE / 2 candele per lato). Il filtro di rendimento
# minimo scarta poi i candidati che non sono seguiti da un movimento davvero tradabile, e il
# cooldown evita grappoli di segnali a distanza di poche candele.
# Valori calibrati misurando la cascata su BTCUSDC 15m, 2 anni (70.080 candele): producono
# ~1000 buy e ~1000 sell (naturalmente bilanciati, ~2,7 segnali al giorno), con un movimento
# disponibile dopo il segnale di 2,1% mediano e 1,55% al 25esimo percentile -- cioe' un ordine
# di grandezza sopra le commissioni anche nel caso peggiore. Alzare LABEL_MIN_RETURN alza il
# caso peggiore ma riduce il numero di esempi quasi linearmente.
EXT_WINDOW_SIZE = 25
LABEL_MIN_RETURN = 0.012  # Movimento minimo (1,2%) richiesto entro l'orizzonte per tenere il segnale
LABEL_RETURN_HORIZON = 48  # Candele di look-ahead per verificare il movimento
LABEL_COOLDOWN = 8  # Distanza minima (candele) tra due segnali consecutivi

# --- Split train / validation --------------------------------------------------------------
TRAIN_SPLIT = 0.8
# Le etichette hanno look-ahead strutturale: dipendono dalle candele successive (order
# dell'argrelextrema e orizzonte del filtro di rendimento), e ogni sequenza copre WINDOW_SIZE
# candele passate. Senza embargo le sequenze a cavallo dello split condividono informazione tra
# train e validation e la validation diventa ottimista.
EMBARGO_STEPS = WINDOW_SIZE + max(EXT_WINDOW_SIZE // 2, LABEL_RETURN_HORIZON)

# --- Bilanciamento --------------------------------------------------------------------------
RANDOM_SEED = 42
# Quante sequenze "hold" tenere per ogni sequenza di segnale (buy + sell) nel *training set*.
# 2.0 = il doppio di hold rispetto ai segnali: la classe maggioritaria resta tale (il mercato
# passa piu' tempo fermo che sui punti di svolta) ma non schiaccia piu' l'apprendimento.
HOLD_TO_SIGNAL_RATIO = 2.0
BALANCE_SIGNAL_CLASSES = True  # Pareggia il numero di buy e sell nel training set
# Dentro ogni timeframe, tutti gli asset contribuiscono lo stesso numero di sequenze di
# training. Il raggruppamento e' per timeframe e non globale di proposito: le sorgenti 15m
# hanno ~4x le candele delle 1h sullo stesso periodo, quindi un cap globale al minimo
# butterebbe il 75% dei dati 15m per un problema -- il predominio di un asset -- che si risolve
# confrontando gli asset tra loro, non i timeframe tra loro.
BALANCE_ASSETS = True

# --- Sorgenti dati ---------------------------------------------------------------------------
# Prodotto cartesiano asset x timeframe scaricato da Binance (market data pubblici, nessuna
# credenziale necessaria). I CSV locali eventualmente elencati sono aggiunti alle sorgenti.
TRAIN_ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
TRAIN_INTERVALS = ["15m", "1h"]
TRAIN_HOURS = 24 * 365 * 2  # ~2 anni di storico per ogni coppia (asset, intervallo)
EXTRA_CSV_FILES = []  # es. ["/percorso/BTCUSDC_2anni_15m.csv"]

# --- Inferenza ---------------------------------------------------------------------------------
# Sotto questa confidenza `get_model_predictions` degrada la predizione a "hold". Va ricalibrata
# guardando la distribuzione delle probabilita' del modello addestrato, non lasciata a caso.
PREDICTION_CONFIDENCE_THRESHOLD = 0.6

# --- Training ------------------------------------------------------------------------------------
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001


# =================================================================================================
# Preparazione dati
# =================================================================================================


def prepare_df_from_csv(csv_file: str, verbose: bool = True) -> pd.DataFrame:
    """Legge un CSV di candele e restituisce il frame di feature + etichette pronto per le sequenze."""
    raw_df = pd.read_csv(csv_file)
    raw_df["Open time"] = pd.to_datetime(raw_df["Open time"])
    raw_df.set_index("Open time", inplace=True)
    return prepare_labeled_frame(raw_df, verbose=verbose)


def prepare_labeled_frame(raw_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """OHLC assoluto -> feature relative + colonna `Label`.

    L'ordine conta: indicatori ed etichette vanno calcolati mentre i prezzi sono ancora
    assoluti (il filtro di rendimento minimo ragiona in percentuale sul prezzo reale), la
    conversione in variazioni percentuali viene per ultima.
    """
    df = raw_df[["Open", "High", "Low", "Close"]].astype(float)
    df = add_technical_indicator(df=df, rsi_window=RSI_WINDOW, atr_window=ATR_WINDOW)
    df = normalize_scale_dependent_features(df)
    df = calculate_relative_extrema(df, verbose=verbose)

    df_transformed = calculate_percentage_changes(df)
    df_transformed.dropna(inplace=True)
    return df_transformed[FEATURES + ["Label"]]


def add_technical_indicator(df, rsi_window=12, atr_window=6):
    df_copy = df.copy()

    # Calcolo dell'RSI
    rsi_indicator = RSIIndicator(close=df_copy["Close"], window=rsi_window)
    df_copy["RSI"] = rsi_indicator.rsi()

    # ATR
    atr_indicator = AverageTrueRange(
        high=df_copy["High"], low=df_copy["Low"], close=df_copy["Close"], window=atr_window
    )
    df_copy["ATR"] = atr_indicator.average_true_range()

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

    df_copy.fillna(0, inplace=True)

    return df_copy


def normalize_scale_dependent_features(df: pd.DataFrame) -> pd.DataFrame:
    """Porta tutte le feature non di prezzo su una scala confrontabile.

    Due problemi distinti, entrambi risolti qui:

    1. L'ATR e' in unita' di prezzo: vale ~300 su BTC e ~1,5 su SOL, e sullo stesso asset
       cambia di un fattore 3 tra due anni fa e oggi. Lasciato grezzo rende impossibile
       addestrare su piu' asset. Espresso come percentuale del Close diventa confrontabile.
    2. RSI, STOCH e TSI sono limitati per costruzione ma su range (0..100, +/-100) che sono
       uno o due ordini di grandezza sopra le altre feature: un LSTM senza normalizzazione
       degli input lascia che siano loro a dominare i gradienti. Ricentrati e riscalati su
       circa [-1, 1] pesano quanto le altre.

    Le trasformazioni sono fisse e senza stato appreso, non uno scaler fittato: non c'e' nulla
    da salvare accanto al modello e nulla che possa divergere tra training e inferenza. La
    funzione va chiamata mentre `Close` e' ancora un prezzo assoluto, cioe' prima di
    `calculate_percentage_changes`, ed e' condivisa dai due path proprio per questo.
    """
    df_copy = df.copy()
    close = df_copy["Close"].replace(0, np.nan)
    df_copy["ATR"] = (df_copy["ATR"] / close * 100).fillna(0)
    for column in ("RSI", "STOCH", "STOCH_S"):
        df_copy[column] = (df_copy[column] - 50.0) / 50.0
    df_copy["TSI"] = df_copy["TSI"] / 100.0
    return df_copy


# trasforma il dataframe in ingresso in variazioni percentuali rispetto alla chiusura precedente
def calculate_percentage_changes(df):
    # Copia del DataFrame per non sovrascrivere i dati originali
    df_transformed = df.copy()
    # Calcolo delle variazioni percentuali rispetto alla chiusura precedente
    df_transformed["Open_Perc"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1) * 100
    df_transformed["High_Perc"] = (df["High"] - df["Close"].shift(1)) / df["Close"].shift(1) * 100
    df_transformed["Low_Perc"] = (df["Low"] - df["Close"].shift(1)) / df["Close"].shift(1) * 100
    df_transformed["Close_Perc"] = (df["Close"] - df["Close"].shift(1)) / df["Close"].shift(1) * 100

    # Gestione della prima riga: uso df.loc[0] (o il primo index se diverso)
    df_transformed.iloc[0, df_transformed.columns.get_loc("Open_Perc")] = 0
    base_open = df.iloc[0]["Open"]

    df_transformed.iloc[0, df_transformed.columns.get_loc("High_Perc")] = (
        (df.iloc[0]["High"] - base_open) / base_open * 100
    )
    df_transformed.iloc[0, df_transformed.columns.get_loc("Low_Perc")] = (
        (df.iloc[0]["Low"] - base_open) / base_open * 100
    )
    df_transformed.iloc[0, df_transformed.columns.get_loc("Close_Perc")] = (
        (df.iloc[0]["Close"] - base_open) / base_open * 100
    )

    # Rimuove i valori NaN (la prima riga avrà NaN dopo la trasformazione)
    df_transformed = df_transformed.dropna()
    # Aggiustamento per garantire la continuità: ogni riga accumula la chiusura percentuale di
    # tutte le righe precedenti. Equivalente alla cumsum di Close_Perc (dimostrabile per induzione:
    # prev_close dopo la riga i == Close_Perc_raw[0] + ... + Close_Perc_raw[i] == cumsum(Close_Perc_raw)[i]),
    # ma vettorizzato invece di un loop Python riga-per-riga (che con decine di migliaia di righe
    # diventa il collo di bottiglia della pipeline).
    prev_close = df_transformed["Close_Perc"].cumsum().shift(1, fill_value=0)
    df_transformed["Open_Perc"] = df_transformed["Open_Perc"] + prev_close
    df_transformed["High_Perc"] = df_transformed["High_Perc"] + prev_close
    df_transformed["Low_Perc"] = df_transformed["Low_Perc"] + prev_close
    df_transformed["Close_Perc"] = df_transformed["Close_Perc"].cumsum()
    df_transformed["Open"] = df_transformed["Open_Perc"]
    df_transformed["High"] = df_transformed["High_Perc"]
    df_transformed["Low"] = df_transformed["Low_Perc"]
    df_transformed["Close"] = df_transformed["Close_Perc"]

    df_transformed.fillna(0, inplace=True)

    return df_transformed


# =================================================================================================
# Labeling
# =================================================================================================


def apply_label_cooldown(labels: pd.Series, cooldown: int) -> pd.Series:
    """Impone una distanza minima in candele tra due segnali consecutivi.

    Gli estremi locali arrivano spesso in grappoli (piu' candele adiacenti che soddisfano la
    condizione su lati diversi dell'oscillazione): tenerli tutti significa chiedere al modello
    di distinguere il minimo "vero" dal suo vicino a una candela di distanza, che e' rumore.
    Tiene il primo segnale di ogni grappolo e azzera quelli troppo vicini.
    """
    if cooldown <= 0 or labels.empty:
        return labels
    values = labels.to_numpy(copy=True)
    signal_positions = np.flatnonzero(values != 0)
    last_kept = None
    for position in signal_positions:
        if last_kept is not None and (position - last_kept) <= cooldown:
            values[position] = 0
        else:
            last_kept = position
    return pd.Series(values, index=labels.index, dtype="int64")


def filter_labels_by_future_return(
    df: pd.DataFrame,
    labels: pd.Series,
    min_return: float,
    horizon: int,
) -> pd.Series:
    """Scarta i segnali non seguiti da un movimento di almeno `min_return` entro `horizon` candele.

    E' il filtro che separa uno swing tradabile da una fluttuazione di rumore: senza di esso un
    minimo locale seguito da un rimbalzo dello 0,1% riceve la stessa etichetta di uno seguito da
    un +5%, e il modello impara una relazione che non esiste.

    Il massimo/minimo futuro e' calcolato con un rolling su serie rovesciata e poi spostato di
    una posizione, cosi' la finestra copre le candele i+1..i+horizon: la candela corrente non
    entra nel proprio criterio, e nessuna riga precedente vede dati futuri.
    """
    if min_return <= 0 or horizon <= 0:
        return labels

    future_max = df["High"][::-1].rolling(horizon, min_periods=1).max()[::-1].shift(-1)
    future_min = df["Low"][::-1].rolling(horizon, min_periods=1).min()[::-1].shift(-1)
    close = df["Close"]

    upside = (future_max / close) - 1.0
    downside = (future_min / close) - 1.0

    filtered = labels.copy()
    # La forma negata scarta anche le ultime `horizon` righe, dove il futuro non e' osservabile
    # e il confronto restituirebbe NaN (un `<` su NaN sarebbe False e terrebbe l'etichetta).
    filtered.loc[(filtered == 1) & ~(upside >= min_return)] = 0
    filtered.loc[(filtered == 2) & ~(downside <= -min_return)] = 0
    return filtered


def calculate_relative_extrema(
    data,
    window_pivot=EXT_WINDOW_SIZE,
    min_return=LABEL_MIN_RETURN,
    return_horizon=LABEL_RETURN_HORIZON,
    cooldown=LABEL_COOLDOWN,
    verbose=False,
):
    """Etichetta le candele: 0 = hold, 1 = minimo relativo (buy), 2 = massimo relativo (sell).

    Tre stadi in cascata: candidati per isolamento locale (`argrelextrema`), filtro di
    rendimento futuro minimo, cooldown. Passare `min_return=0` e `cooldown=0` riproduce il
    labeling puramente locale originale.

    Richiede prezzi assoluti (`High`/`Low`/`Close` non ancora convertiti in percentuali).
    """
    order = max(2, int(window_pivot / 2))
    max_idx = argrelextrema(data["High"].values, np.greater, order=order)[0]
    min_idx = argrelextrema(data["Low"].values, np.less, order=order)[0]

    labels = pd.Series(0, index=data.index, dtype="int64")
    labels.iloc[max_idx] = 2
    labels.iloc[min_idx] = 1
    if verbose:
        summarize_labels(labels.to_numpy(), "estremi locali grezzi")

    labels = filter_labels_by_future_return(data, labels, min_return, return_horizon)
    if verbose:
        summarize_labels(labels.to_numpy(), f"dopo filtro rendimento >= {min_return:.1%}")

    labels = apply_label_cooldown(labels, cooldown)
    if verbose:
        summarize_labels(labels.to_numpy(), f"dopo cooldown di {cooldown} candele")

    data["Label"] = labels
    return data


def summarize_labels(y: np.ndarray, stage: str) -> None:
    """Stampa la distribuzione delle classi a uno stadio della pipeline."""
    names = {0: "hold", 1: "buy", 2: "sell"}
    if len(y) == 0:
        print(f"[{stage}] nessun campione")
        return
    unique, counts = np.unique(y, return_counts=True)
    stats = {int(k): int(v) for k, v in zip(unique, counts)}
    parts = [f"{names.get(k, k)}={stats.get(k, 0)} ({stats.get(k, 0) / len(y):.2%})" for k in (0, 1, 2)]
    print(f"[{stage}] {len(y)} campioni | " + ", ".join(parts))


# =================================================================================================
# Sequenze
# =================================================================================================


# Crea le sequenze da passare al modello per l'addestramento
# ogni sequenza inzia da 0 e varia in punti percentuale
def create_sequences(data, features, window_size):
    """
    Crea le sequenze temporali scorrevoli da passare al modello.
    Sottrae a ogni sequenza il valore di apertura della prima riga della finestra.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset contenente i dati con le colonne richieste.
    features : list
        Lista di colonne usate come feature per creare le sequenze.
    window_size : int
        Lunghezza della finestra temporale (numero di step).

    Returns
    -------
    X : numpy.ndarray
        Sequenze (shape: [num_samples, window_size, num_features]).
    y : numpy.ndarray
        Etichetta della candela immediatamente successiva a ciascuna finestra.
    """
    df_copy = data[features + ["Label"]].copy()
    num_sequences = len(df_copy) - window_size
    if num_sequences <= 0:
        return np.empty((0, window_size, len(features))), np.empty((0,))

    # Finestre scorrevoli vettorizzate (equivalenti al loop Python precedente, ma senza il costo di
    # migliaia di accessi .iloc riga per riga, che diventa proibitivo su dataset di decine di
    # migliaia di candele).
    values = df_copy[features].to_numpy(dtype=float)
    windows = sliding_window_view(values, window_shape=window_size, axis=0)
    windows = np.moveaxis(windows, -1, 1)[:num_sequences].copy()

    # Sottrae a ogni finestra il valore di apertura della sua prima riga (solo colonne di prezzo).
    open_index = features.index("Open")
    window_open = values[:num_sequences, open_index]
    price_columns = [j for j, feature in enumerate(features) if feature in PRICE_FEATURES]
    for j in price_columns:
        windows[:, :, j] -= window_open[:, None]

    X = windows
    # Etichetta del punto immediatamente successivo a ciascuna finestra.
    y = df_copy["Label"].to_numpy()[window_size:]

    return X, y


def infer_expected_minutes(index: pd.DatetimeIndex) -> float:
    """Passo temporale tipico della serie, in minuti (mediana, robusta ai buchi)."""
    if len(index) < 2:
        return 0.0
    deltas = index.to_series().diff().dt.total_seconds().to_numpy()[1:] / 60.0
    return float(np.median(deltas))


def build_sequence_valid_mask(
    index: pd.DatetimeIndex,
    expected_minutes: float,
    window_size: int,
    gap_tolerance: float = GAP_TOLERANCE,
) -> np.ndarray:
    """Maschera le sequenze che attraversano un buco temporale nella serie.

    `create_sequences` lavora su posizioni, non su timestamp: se il CSV o il download hanno un
    buco (manutenzione dell'exchange, righe mancanti), la finestra risultante incolla insieme
    due momenti scorrelati e la sequenza e' spazzatura. La maschera copre le righe
    i..i+window_size, cioe' la finestra piu' la candela da cui viene l'etichetta.
    """
    n = len(index)
    num_sequences = max(0, n - window_size)
    if num_sequences == 0:
        return np.zeros(0, dtype=bool)
    if expected_minutes <= 0:
        return np.ones(num_sequences, dtype=bool)

    deltas = index.to_series().diff().dt.total_seconds().to_numpy() / 60.0
    is_gap = np.zeros(n, dtype=np.int64)
    is_gap[1:] = (deltas[1:] > expected_minutes * gap_tolerance).astype(np.int64)

    cumulative = np.concatenate([[0], np.cumsum(is_gap)])
    gaps_in_span = cumulative[window_size + 1 : window_size + 1 + num_sequences] - cumulative[1 : 1 + num_sequences]
    return gaps_in_span == 0


def build_sequences_for_source(
    raw_df: pd.DataFrame,
    name: str,
    window_size: int = WINDOW_SIZE,
    verbose: bool = True,
):
    """Da candele grezze a sequenze (X, y) per una singola coppia asset/timeframe."""
    print(f"\n--- {name} ---")
    df = prepare_labeled_frame(raw_df, verbose=verbose)
    X, y = create_sequences(df, FEATURES, window_size)

    expected_minutes = infer_expected_minutes(df.index)
    valid = build_sequence_valid_mask(df.index, expected_minutes, window_size)
    dropped = int((~valid).sum())
    if dropped:
        print(
            f"[{name}] scartate {dropped} sequenze a cavallo di buchi temporali "
            f"(passo atteso {expected_minutes:g} min)"
        )
    X, y = X[valid], y[valid]

    summarize_labels(y, f"{name} sequenze valide")
    return X, y


# =================================================================================================
# Bilanciamento e split
# =================================================================================================


def split_train_val(X: np.ndarray, y: np.ndarray, train_split: float, embargo_steps: int):
    """Split cronologico con embargo simmetrico attorno al punto di taglio.

    Le sequenze scartate nell'embargo sono quelle la cui finestra o la cui etichetta
    attraversano il confine: senza rimuoverle, la validation misura anche informazione gia'
    vista in training.
    """
    split_idx = int(len(X) * train_split)
    embargo = max(0, int(embargo_steps))
    train_end = max(0, split_idx - embargo)
    val_start = min(len(X), split_idx + embargo)
    return X[:train_end], y[:train_end], X[val_start:], y[val_start:]


def cap_sequences(X: np.ndarray, y: np.ndarray, max_samples, rng: np.random.Generator):
    """Riduce le sequenze a `max_samples` campioni estratti a caso (bilanciamento tra sorgenti)."""
    if max_samples is None or len(y) <= max_samples:
        return X, y
    selected = np.sort(rng.choice(len(y), size=int(max_samples), replace=False))
    return X[selected], y[selected]


def downsample_holds(X: np.ndarray, y: np.ndarray, hold_to_signal_ratio: float, rng: np.random.Generator):
    """Riduce la classe "hold" a un multiplo del numero di segnali.

    Complementa (non sostituisce) il class weight: pesare la loss di 140:1 spinge il modello
    verso minimi locali degeneri in cui predice sempre la stessa classe, mentre ridurre lo
    sbilanciamento nei dati lascia al class weight solo una correzione residua piccola.
    """
    if hold_to_signal_ratio <= 0 or len(y) == 0:
        return X, y
    hold_idx = np.flatnonzero(y == 0)
    signal_idx = np.flatnonzero(y != 0)
    if len(signal_idx) == 0 or len(hold_idx) == 0:
        return X, y

    keep_holds = int(len(signal_idx) * hold_to_signal_ratio)
    if keep_holds >= len(hold_idx):
        return X, y
    selected_holds = rng.choice(hold_idx, size=keep_holds, replace=False)
    selected = np.sort(np.concatenate([signal_idx, selected_holds]))
    return X[selected], y[selected]


def balance_signal_classes(X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """Pareggia il numero di buy e sell tagliando la classe piu' numerosa."""
    buy_idx = np.flatnonzero(y == 1)
    sell_idx = np.flatnonzero(y == 2)
    if len(buy_idx) == 0 or len(sell_idx) == 0:
        return X, y
    target = min(len(buy_idx), len(sell_idx))
    if len(buy_idx) > target:
        buy_idx = rng.choice(buy_idx, size=target, replace=False)
    if len(sell_idx) > target:
        sell_idx = rng.choice(sell_idx, size=target, replace=False)
    selected = np.sort(np.concatenate([np.flatnonzero(y == 0), buy_idx, sell_idx]))
    return X[selected], y[selected]


# =================================================================================================
# Inferenza
# =================================================================================================


def get_model_predictions(df, model, confidence_threshold: float = PREDICTION_CONFIDENCE_THRESHOLD):
    """Applica il modello a un DataFrame di mercato e restituisce le predizioni allineate all'indice.

    Gli indicatori vengono ricalcolati qui dai soli OHLC, con le costanti di questo modulo,
    invece di riusare le colonne che il chiamante ha gia' in tabella: `trading/simulator.py`
    calcola le sue con i periodi scelti dagli slider della dashboard (l'ATR Window ha default 5
    nella UI contro i 6 usati in addestramento), e un modello alimentato con feature calcolate
    diversamente da come e' stato addestrato sbaglia in silenzio, senza nessun errore visibile.

    Il resto del preprocessing e' identico al training perche' passa per le stesse funzioni:
    riscalatura delle feature, variazioni percentuali, sequenze scorrevoli.
    """
    data = df[["Open", "High", "Low", "Close"]].astype(float).copy()
    data.fillna(0, inplace=True)
    data = add_technical_indicator(data, rsi_window=RSI_WINDOW, atr_window=ATR_WINDOW)
    data = normalize_scale_dependent_features(data)
    data = data[FEATURES]
    data["Label"] = 0
    df_transformed = calculate_percentage_changes(data)

    X, _ = create_sequences(df_transformed, FEATURES, WINDOW_SIZE)

    probabilities = model.predict(X, verbose=0)
    probabilities = np.nan_to_num(probabilities, nan=0.0)
    # Verifica che la lunghezza coincida
    if len(df) - WINDOW_SIZE != probabilities.shape[0]:
        raise ValueError("Dimension mismatch: df vs model predictions")

    # Allinea con l'indice originale del DataFrame
    df_preds = df.iloc[WINDOW_SIZE:].copy()

    # Sotto la soglia di confidenza la predizione degrada a "hold": meglio non operare che
    # operare su un segnale incerto.
    max_probs = np.max(probabilities, axis=1)
    max_classes = np.argmax(probabilities, axis=1)
    df_preds["Prediction"] = np.where(max_probs > confidence_threshold, max_classes, 0)

    return df_preds


# Funzione per normalizzare le feature numeriche
def normalize_features(df, scaler=None):
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    if scaler is None:
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df, scaler


# =================================================================================================
# Costruzione del dataset multi-asset / multi-timeframe
# =================================================================================================


def load_market_sources(assets, intervals, hours):
    """Scarica il prodotto cartesiano asset x intervallo dai market data pubblici di Binance.

    L'import di `trading.simulator` e' locale: quel modulo importa a sua volta questo file (per
    `get_model_predictions`), quindi un import in testa creerebbe un ciclo.
    """
    from cryptofarm.trading.simulator import get_market_data

    sources = []
    for asset in assets:
        for interval in intervals:
            name = f"{asset}-{interval}"
            try:
                df, _ = get_market_data(asset=asset, interval=interval, time_hours=hours)
            except Exception as exc:
                print(f"[{name}] download fallito: {exc}")
                continue
            if df is None or df.empty:
                print(f"[{name}] nessun dato disponibile, sorgente saltata")
                continue
            # Il gruppo e' l'intervallo: e' la chiave su cui viene fatto il bilanciamento fra
            # sorgenti (vedi build_training_dataset).
            sources.append((name, interval, df))
    return sources


def build_training_dataset(sources, window_size=WINDOW_SIZE, rng=None):
    """Assembla train/validation da piu' sorgenti.

    Lo split cronologico e' fatto *per sorgente* e non sul dataset concatenato: concatenare
    prima manderebbe intere coppie asset/timeframe interamente in validation, misurando la
    generalizzazione tra asset invece che nel tempo.
    """
    rng = rng if rng is not None else np.random.default_rng(RANDOM_SEED)

    per_source = []
    for name, group, raw_df in sources:
        X, y = build_sequences_for_source(raw_df, name, window_size=window_size)
        if len(y) == 0:
            print(f"[{name}] nessuna sequenza utilizzabile, sorgente saltata")
            continue
        X_train, y_train, X_val, y_val = split_train_val(X, y, TRAIN_SPLIT, EMBARGO_STEPS)
        if len(y_train) == 0 or len(y_val) == 0:
            print(f"[{name}] dati insufficienti dopo lo split con embargo, sorgente saltata")
            continue
        per_source.append((name, group, X_train, y_train, X_val, y_val))

    if not per_source:
        raise RuntimeError("Nessuna sorgente utilizzabile: impossibile costruire il dataset.")

    # Dentro ogni gruppo (timeframe) tutte le sorgenti scendono al conteggio della piu' piccola:
    # e' quello che impedisce all'asset con piu' storico di dominare, senza penalizzare i
    # timeframe piu' fitti rispetto a quelli piu' radi.
    caps = {}
    if BALANCE_ASSETS:
        for _, group, _, y_train, _, _ in per_source:
            caps[group] = min(caps.get(group, len(y_train)), len(y_train))
        print("\nBilanciamento per timeframe:", {g: c for g, c in caps.items()})

    X_train_parts, y_train_parts, X_val_parts, y_val_parts = [], [], [], []
    for name, group, X_train, y_train, X_val, y_val in per_source:
        X_train, y_train = cap_sequences(X_train, y_train, caps.get(group), rng)
        print(f"[{name}] train={len(y_train)} val={len(y_val)}")
        X_train_parts.append(X_train)
        y_train_parts.append(y_train)
        X_val_parts.append(X_val)
        y_val_parts.append(y_val)

    X_train = np.concatenate(X_train_parts)
    y_train = np.concatenate(y_train_parts).astype(int)
    X_val = np.concatenate(X_val_parts)
    y_val = np.concatenate(y_val_parts).astype(int)

    print()
    summarize_labels(y_train, "train concatenato")
    summarize_labels(y_val, "validation concatenata")

    # Il bilanciamento tocca solo il training set: la validation resta alla distribuzione reale
    # del mercato, altrimenti le metriche riportate non direbbero nulla su come il modello si
    # comporta in produzione.
    if BALANCE_SIGNAL_CLASSES:
        X_train, y_train = balance_signal_classes(X_train, y_train, rng)
        summarize_labels(y_train, "train dopo pareggio buy/sell")

    X_train, y_train = downsample_holds(X_train, y_train, HOLD_TO_SIGNAL_RATIO, rng)
    summarize_labels(y_train, "train dopo downsampling hold")

    # Le sequenze arrivano ordinate per sorgente e per tempo: senza shuffle i batch sarebbero
    # tutti dello stesso asset e dello stesso regime di mercato.
    shuffled = rng.permutation(len(y_train))
    return X_train[shuffled], y_train[shuffled], X_val, y_val


# Funzione per costruire il modello (usata da Keras Tuner)
def build_model(hp, input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(Bidirectional(LSTM(units=hp.Int("lstm_units1", 64, 256, step=64), return_sequences=True)))
    model.add(Dropout(hp.Float("dropout1", 0.1, 0.3, step=0.1)))
    model.add(BatchNormalization())

    model.add(LSTM(units=hp.Int("lstm_units2", 64, 256, step=64), return_sequences=True))
    model.add(Dropout(hp.Float("dropout2", 0.1, 0.3, step=0.1)))
    model.add(BatchNormalization())

    model.add(LSTM(units=hp.Int("lstm_units3", 64, 256, step=64), return_sequences=False))
    model.add(Dropout(hp.Float("dropout3", 0.1, 0.3, step=0.1)))
    model.add(BatchNormalization())

    model.add(Dense(3, activation="softmax"))

    model.compile(
        optimizer=Adam(hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lstm(input_shape):
    return Sequential(
        [
            Input(shape=input_shape),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(192, return_sequences=True),
            Dropout(0.1),
            BatchNormalization(),
            LSTM(256, return_sequences=False),
            Dropout(0.1),
            BatchNormalization(),
            Dense(3, activation="softmax"),
        ]
    )


if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_SEED)

    sources = load_market_sources(TRAIN_ASSETS, TRAIN_INTERVALS, TRAIN_HOURS)
    for csv_file in EXTRA_CSV_FILES:
        raw_df = pd.read_csv(csv_file)
        raw_df["Open time"] = pd.to_datetime(raw_df["Open time"])
        raw_df.set_index("Open time", inplace=True)
        sources.append((csv_file.split("/")[-1], "csv", raw_df))

    X_train, y_train, X_val, y_val = build_training_dataset(sources, WINDOW_SIZE, rng)
    print(f"\nX_train {X_train.shape} | X_val {X_val.shape}")

    # Dopo il downsampling lo sbilanciamento residuo e' dell'ordine di 2:1, non piu' 140:1: i
    # pesi "balanced" grezzi bastano e non serve piu' smorzarli (il vecchio np.sqrt serviva a
    # tamponare uno squilibrio che ora e' risolto a monte, nei dati).
    present_classes = np.unique(y_train)
    # zip sulle classi presenti, non enumerate: se una classe mancasse dal training set,
    # enumerate assegnerebbe i pesi alle etichette sbagliate in silenzio.
    class_weights = compute_class_weight(class_weight="balanced", classes=present_classes, y=y_train)
    class_weights = {int(k): float(v) for k, v in zip(present_classes, class_weights)}
    print("Class weights:", {k: round(v, 3) for k, v in class_weights.items()})

    model = build_lstm((X_train.shape[1], X_train.shape[2]))
    model.compile(optimizer=Adam(LEARNING_RATE), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-5)
    checkpoint = ModelCheckpoint(
        str(MODELS_DIR / "optimized_model.keras"), monitor="val_loss", save_best_only=True, verbose=1
    )

    print("Final Training")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        class_weight=class_weights,
    )

    # La accuracy grezza e' fuorviante su una validation che resta sbilanciata verso "hold": un
    # classificatore banale che predice sempre "hold" la ottiene gratis. Il classification report
    # per classe (precision/recall/F1) e la confusion matrix mostrano se il modello sta davvero
    # riconoscendo buy/sell o e' collassato su una predizione costante.
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")

    val_probabilities = model.predict(X_val, verbose=0)
    val_predictions = np.argmax(val_probabilities, axis=1)
    print(classification_report(y_val, val_predictions, target_names=["hold", "buy", "sell"], zero_division=0))
    print("Confusion matrix (righe=reale, colonne=predetto):")
    print(confusion_matrix(y_val, val_predictions))

    # Con quale confidenza il modello emette i segnali? Serve a ricalibrare
    # PREDICTION_CONFIDENCE_THRESHOLD invece di lasciarlo al valore storico di 0.6.
    max_probs = np.max(val_probabilities, axis=1)
    print(
        f"Confidenza max: media={max_probs.mean():.3f} p50={np.percentile(max_probs, 50):.3f} "
        f"p90={np.percentile(max_probs, 90):.3f} p99={np.percentile(max_probs, 99):.3f}"
    )
    for threshold in (0.4, 0.5, 0.6, 0.7, 0.8):
        gated = np.where(max_probs > threshold, val_predictions, 0)
        signals = int((gated != 0).sum())
        print(f"  soglia {threshold}: {signals} segnali su {len(gated)} sequenze di validation")

    model.save(str(MODELS_DIR / "trained_model.keras"))
    print(f"Modello salvato in {MODELS_DIR / 'trained_model.keras'}")
