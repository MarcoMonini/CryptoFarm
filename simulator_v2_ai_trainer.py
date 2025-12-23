"""Background AI trainer for Simulator v2 (multi-timeframe).

This script is configured directly in code (no UI) and trains a model to
classify buy/sell/hold. It includes data-quality controls (label filtering,
gap handling, class balancing) to reduce overfitting and leakage.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from simulator_v2_backend import (
    TIMEFRAME_MINUTES,
    compute_indicators,
    fetch_market_data_dates,
    fetch_market_data_hours,
)


# ---------------------------
# Training configuration
# ---------------------------

DATA_SOURCE = "Binance"  # "Binance" or "CSV"
CSV_FOLDER = "data/ohlcv"  # Used when DATA_SOURCE == "CSV"

ASSETS = ["BTCUSDT"]
TIMEFRAMES = ["15m", "4h", "1d"]  # Multi-timeframe inputs

RANGE_MODE = "Hours"  # "Hours" or "Dates"
HOURS = 24 * 60  # Used if RANGE_MODE == "Hours"
START_DT = datetime.utcnow() - timedelta(days=365)  # Used if RANGE_MODE == "Dates"
END_DT = datetime.utcnow()

SEQUENCE_LENGTH = 50  # LSTM sequence length
EXTREMA_WINDOW = 80  # Window used to mark local minima/maxima
TRAIN_SPLIT = 0.8  # Chronological split

# Data quality and labeling controls
RANDOM_SEED = 42
LABEL_METHOD = "extrema"  # "extrema" or "future_return"
LABEL_MIN_RETURN = 0.01  # Minimum forward return required to keep a label (0 to disable)
LABEL_RETURN_HORIZON = 24  # Candles to look ahead when filtering labels
LABEL_COOLDOWN = 6  # Minimum gap (candles) between consecutive labels

# Sequence and gap handling
SEQUENCE_STRIDE = 1  # Use >1 to reduce overlapping sequences
GAP_TOLERANCE = 1.5  # Allowable multiple of expected delta between candles
MAX_FFILL_GAP_MULTIPLIER = 2.0  # Max age (in TF candles) for ffill alignment
EMBARGO_STEPS = 5  # Drop sequences around train/val split to reduce leakage

# Class/asset balancing
BALANCE_METHOD = "both"  # "weights", "downsample", "both", "none"
HOLD_KEEP_RATIO = 0.25  # Keep this fraction of hold sequences (0..1)
BALANCE_ASSETS = True  # Ensure assets contribute similar sample counts
MAX_SAMPLES_PER_ASSET = None  # Optional cap per asset
MIN_CLASS_SAMPLES = 50  # Warn if a class has fewer samples

MODEL_DIR = "ai_models"
MODEL_NAME = "mtf_ai_model_v1"
SCALER_TYPE = "standard"  # "standard", "minmax", or "robust"

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 1e-3

# Indicator flags (same for all timeframes)
INDICATOR_FLAGS = {
    "atr_bands": True,
    "rsi_short": True,
    "rsi_medium": True,
    "rsi_long": True,
    "ema_short": True,
    "ema_medium": True,
    "ema_long": True,
    "kama": True,
    "macd": True,
}

# Per-timeframe parameters (override defaults if needed).
TIMEFRAME_PARAMS: Dict[str, Dict[str, float]] = {
    "15m": {},
    "4h": {},
    "1d": {},
}

# Base defaults (aligned with simulator_v2_app.py).
BASE_PARAM_DEFAULTS = {
    "atr_window": 5,
    "atr_multiplier": 1.6,
    "rsi_short_window": 12,
    "rsi_medium_window": 24,
    "rsi_long_window": 36,
    "ema_short_window": 10,
    "ema_medium_window": 50,
    "ema_long_window": 200,
    "kama_window": 10,
    "kama_pow1": 2,
    "kama_pow2": 30,
    "macd_short_window": 12,
    "macd_long_window": 26,
    "macd_signal_window": 9,
    "rsi_short_buy_limit": 25,
    "rsi_short_sell_limit": 75,
    "rsi_medium_buy_limit": 30,
    "rsi_medium_sell_limit": 70,
    "rsi_long_buy_limit": 35,
    "rsi_long_sell_limit": 65,
    "macd_buy_limit": -2.5,
    "macd_sell_limit": 2.5,
    "conditions_required": 1,
    "stop_loss_percent": 99.0,
}


@dataclass(frozen=True)
class DatasetBundle:
    features: pd.DataFrame
    labels: pd.Series
    base_timeframe: str


def seed_everything(seed: int) -> None:
    """Imposta i seed per riproducibilita' del preprocessing e training."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        # If TF is not available, skip without failing.
        pass


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def get_timeframe_params(timeframe: str) -> Dict[str, float]:
    params = dict(BASE_PARAM_DEFAULTS)
    params.update(TIMEFRAME_PARAMS.get(timeframe, {}))
    return params


def find_csv_file(asset: str, interval: str) -> str:
    filename = f"{asset}_{interval}.csv"
    candidate = os.path.join(CSV_FOLDER, filename)
    if os.path.isfile(candidate):
        return candidate
    # Try lowercase or different separators.
    for name in os.listdir(CSV_FOLDER):
        if name.lower() == filename.lower():
            return os.path.join(CSV_FOLDER, name)
    raise FileNotFoundError(f"CSV not found for {asset} {interval} in {CSV_FOLDER}.")


def normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    index_col = None
    for key in ("open time", "open_time", "timestamp", "date", "time"):
        if key in cols:
            index_col = cols[key]
            break

    if index_col:
        df[index_col] = pd.to_datetime(df[index_col], utc=True, errors="coerce")
        df = df.dropna(subset=[index_col]).set_index(index_col)
    else:
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[~df.index.isna()]

    rename_map = {}
    for key, target in [
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("volume", "Volume"),
    ]:
        if key in cols:
            rename_map[cols[key]] = target
    df = df.rename(columns=rename_map)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    df = df[required].astype(float).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def fetch_data(asset: str, interval: str) -> pd.DataFrame:
    if DATA_SOURCE == "Binance":
        if RANGE_MODE == "Hours":
            df, _ = fetch_market_data_hours(asset, interval, HOURS)
        else:
            df, _ = fetch_market_data_dates(asset, interval, START_DT, END_DT)
        return df

    csv_path = find_csv_file(asset, interval)
    df = pd.read_csv(csv_path)
    return normalize_ohlcv_df(df)


def required_feature_columns(indicator_flags: Dict[str, bool]) -> List[str]:
    columns = ["Open", "High", "Low", "Close", "Volume"]
    if indicator_flags.get("atr_bands"):
        columns.extend(["ATR", "ATR_MID", "ATR_UPPER", "ATR_LOWER"])
    if indicator_flags.get("rsi_short"):
        columns.append("RSI_SHORT")
    if indicator_flags.get("rsi_medium"):
        columns.append("RSI_MED")
    if indicator_flags.get("rsi_long"):
        columns.append("RSI_LONG")
    if indicator_flags.get("ema_short"):
        columns.append("EMA_SHORT")
    if indicator_flags.get("ema_medium"):
        columns.append("EMA_MED")
    if indicator_flags.get("ema_long"):
        columns.append("EMA_LONG")
    if indicator_flags.get("kama"):
        columns.append("KAMA")
    if indicator_flags.get("macd"):
        columns.extend(["MACD", "MACD_SIGNAL", "MACD_HIST"])
    return columns


def apply_label_cooldown(labels: pd.Series, cooldown: int) -> pd.Series:
    """Applica una distanza minima tra etichette buy/sell.

    Input:
        labels: Serie int (0=hold, 1=buy, 2=sell).
        cooldown: numero minimo di candle tra due segnali.
    Output:
        labels filtrate con cooldown applicato.
    """
    if cooldown <= 0 or labels.empty:
        return labels
    filtered = labels.copy()
    last_label_idx = None
    for idx in range(len(filtered)):
        if filtered.iat[idx] == 0:
            continue
        if last_label_idx is not None and (idx - last_label_idx) <= cooldown:
            # Drop labels too close to previous to reduce noise/overfitting.
            filtered.iat[idx] = 0
        else:
            last_label_idx = idx
    return filtered


def filter_labels_by_future_return(
    df: pd.DataFrame,
    labels: pd.Series,
    min_return: float,
    horizon: int,
) -> pd.Series:
    """Rimuove etichette con profitto futuro insufficiente.

    Input:
        df: DataFrame con colonne High/Low/Close.
        labels: Serie int (0/1/2).
        min_return: ritorno minimo richiesto (es. 0.01 = 1%).
        horizon: numero di candle per guardare avanti.
    Output:
        labels filtrate.
    """
    if min_return <= 0 or horizon <= 0:
        return labels

    # Forward-looking max/min usando reverse-rolling per evitare lookback.
    future_max = df["High"].iloc[::-1].rolling(horizon, min_periods=1).max().iloc[::-1]
    future_min = df["Low"].iloc[::-1].rolling(horizon, min_periods=1).min().iloc[::-1]
    close = df["Close"]

    max_return = (future_max / close) - 1.0
    min_return_series = (future_min / close) - 1.0

    filtered = labels.copy()
    buy_mask = filtered == 1
    sell_mask = filtered == 2
    filtered.loc[buy_mask & (max_return < min_return)] = 0
    filtered.loc[sell_mask & (min_return_series > -min_return)] = 0
    return filtered


def build_extrema_labels(
    df: pd.DataFrame,
    window_size: int,
    min_return: float,
    return_horizon: int,
    cooldown: int,
) -> pd.Series:
    """Costruisce etichette usando minimi/massimi locali filtrati.

    Input:
        df: DataFrame OHLCV con indice datetime.
        window_size: finestra di ricerca estremi.
        min_return: ritorno minimo per mantenere l'etichetta.
        return_horizon: orizzonte forward per il filtro ritorno.
        cooldown: gap minimo tra etichette.
    Output:
        Serie int (0=hold, 1=buy, 2=sell).
    """
    order = max(2, int(window_size / 2))
    highs = df["High"].values
    lows = df["Low"].values
    max_idx = argrelextrema(highs, np.greater, order=order)[0]
    min_idx = argrelextrema(lows, np.less, order=order)[0]
    labels = pd.Series(0, index=df.index, dtype="int64")
    labels.iloc[max_idx] = 2  # Sell
    labels.iloc[min_idx] = 1  # Buy
    labels = filter_labels_by_future_return(df, labels, min_return, return_horizon)
    labels = apply_label_cooldown(labels, cooldown)
    return labels


def build_future_return_labels(
    df: pd.DataFrame,
    horizon: int,
    buy_threshold: float,
    sell_threshold: float,
    cooldown: int,
) -> pd.Series:
    """Etichette basate sul ritorno futuro a orizzonte fisso.

    Input:
        df: DataFrame OHLCV.
        horizon: candle future per calcolare il return.
        buy_threshold: return >= soglia -> buy.
        sell_threshold: return <= soglia -> sell.
        cooldown: gap minimo tra etichette.
    Output:
        Serie int (0=hold, 1=buy, 2=sell).
    """
    if horizon <= 0:
        return pd.Series(0, index=df.index, dtype="int64")
    future_close = df["Close"].shift(-horizon)
    future_return = (future_close - df["Close"]) / df["Close"]
    labels = pd.Series(0, index=df.index, dtype="int64")
    labels.loc[future_return >= buy_threshold] = 1
    labels.loc[future_return <= sell_threshold] = 2
    labels = apply_label_cooldown(labels, cooldown)
    return labels


def build_labels(df: pd.DataFrame) -> pd.Series:
    """Seleziona il metodo di labeling configurato."""
    if LABEL_METHOD == "future_return":
        return build_future_return_labels(
            df=df,
            horizon=LABEL_RETURN_HORIZON,
            buy_threshold=LABEL_MIN_RETURN,
            sell_threshold=-LABEL_MIN_RETURN,
            cooldown=LABEL_COOLDOWN,
        )
    if LABEL_METHOD == "extrema":
        return build_extrema_labels(
            df=df,
            window_size=EXTREMA_WINDOW,
            min_return=LABEL_MIN_RETURN,
            return_horizon=LABEL_RETURN_HORIZON,
            cooldown=LABEL_COOLDOWN,
        )
    raise ValueError(f"Unsupported LABEL_METHOD: {LABEL_METHOD}")


def build_sequence_valid_mask(
    index: pd.Index,
    expected_minutes: int,
    sequence_length: int,
    gap_tolerance: float,
) -> np.ndarray:
    """Crea maschera per scartare sequenze che attraversano gap temporali."""
    if expected_minutes <= 0:
        return np.ones(len(index), dtype=bool)
    if sequence_length <= 1:
        return np.ones(len(index), dtype=bool)
    if len(index) == 0:
        return np.zeros(0, dtype=bool)

    expected_gap = expected_minutes * gap_tolerance
    deltas = index.to_series().diff().dt.total_seconds() / 60.0
    valid_step = (deltas <= expected_gap).fillna(False).to_numpy()
    invalid_steps = (~valid_step).astype(int)
    window = np.ones(sequence_length - 1, dtype=int)
    invalid_count = np.convolve(invalid_steps, window, mode="full")[: len(invalid_steps)]
    valid_seq_end = invalid_count == 0
    valid_seq_end[: sequence_length - 1] = False
    return valid_seq_end


def downsample_sequences(
    X: np.ndarray,
    y: np.ndarray,
    hold_keep_ratio: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Riduce la classe 'hold' per attenuare lo sbilanciamento."""
    if hold_keep_ratio >= 1.0 or len(y) == 0:
        return X, y
    hold_idx = np.where(y == 0)[0]
    keep_idx = np.where(y != 0)[0]
    if len(hold_idx) == 0:
        return X, y
    keep_hold = int(len(hold_idx) * max(0.0, hold_keep_ratio))
    if keep_hold <= 0:
        selected = keep_idx
    else:
        selected_hold = rng.choice(hold_idx, size=keep_hold, replace=False)
        selected = np.sort(np.concatenate([keep_idx, selected_hold]))
    return X[selected], y[selected]


def cap_sequences(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: Optional[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Taglia il numero di campioni per asset per bilanciare la contribuzione."""
    if max_samples is None or len(y) <= max_samples:
        return X, y
    selected = rng.choice(np.arange(len(y)), size=max_samples, replace=False)
    return X[selected], y[selected]


def summarize_labels(y: np.ndarray, label: str) -> None:
    """Stampa distribuzione classi per debug."""
    if len(y) == 0:
        print(f"[{label}] No samples.")
        return
    unique, counts = np.unique(y, return_counts=True)
    stats = {int(k): int(v) for k, v in zip(unique, counts)}
    print(f"[{label}] class distribution: {stats}")


def build_feature_frame(asset: str) -> DatasetBundle:
    base_timeframe = min(TIMEFRAMES, key=lambda tf: TIMEFRAME_MINUTES[tf])
    base_df = fetch_data(asset, base_timeframe)
    if base_df.empty:
        raise ValueError(f"No data for {asset} {base_timeframe}")

    base_params = get_timeframe_params(base_timeframe)
    base_df = compute_indicators(base_df, INDICATOR_FLAGS, base_params)
    base_index = base_df.index

    features = pd.DataFrame(index=base_index)
    feature_columns: List[str] = []

    # Max acceptable staleness (in minutes) for forward-filled data.
    # This prevents using very old TF values when candles are missing.
    max_ffill_gap_minutes: Dict[str, float] = {
        tf: TIMEFRAME_MINUTES[tf] * MAX_FFILL_GAP_MULTIPLIER for tf in TIMEFRAMES
    }

    for tf in TIMEFRAMES:
        df = fetch_data(asset, tf)
        if df.empty:
            raise ValueError(f"No data for {asset} {tf}")
        tf_params = get_timeframe_params(tf)
        df_ind = compute_indicators(df, INDICATOR_FLAGS, tf_params)
        columns = required_feature_columns(INDICATOR_FLAGS)
        tf_slice = df_ind[columns].reindex(base_index, method="ffill")
        # Track the last timestamp from the TF to detect stale ffill.
        last_ts = pd.Series(df_ind.index, index=df_ind.index).reindex(base_index, method="ffill")
        age_minutes = base_index.to_series().sub(last_ts).dt.total_seconds() / 60.0
        gap_limit = max_ffill_gap_minutes[tf]
        if gap_limit > 0:
            stale_mask = age_minutes > gap_limit
            if stale_mask.any():
                tf_slice.loc[stale_mask, :] = np.nan
        tf_columns = [f"tf_{tf}_{col}" for col in columns]
        tf_slice.columns = tf_columns
        features = features.join(tf_slice)
        feature_columns.extend(tf_columns)

    labels = build_labels(base_df)
    dataset = features.join(labels.rename("Label"))
    dataset = dataset.dropna()

    return DatasetBundle(
        features=dataset[feature_columns],
        labels=dataset["Label"],
        base_timeframe=base_timeframe,
    )


def create_sequences(
    features: pd.DataFrame,
    labels: pd.Series,
    sequence_length: int,
    expected_minutes: int,
    gap_tolerance: float,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Costruisce sequenze LSTM evitando gap temporali."""
    X, y = [], []
    if len(features) == 0:
        return np.array(X), np.array(y)
    values = features.values
    label_values = labels.values
    stride = max(1, int(stride))
    valid_mask = build_sequence_valid_mask(
        features.index, expected_minutes, sequence_length, gap_tolerance
    )
    for i in range(sequence_length - 1, len(features), stride):
        if not valid_mask[i]:
            continue
        start = i - sequence_length + 1
        X.append(values[start : i + 1])
        y.append(label_values[i])
    return np.array(X), np.array(y)


def split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    train_split: float,
    embargo_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split cronologico con embargo per ridurre leakage."""
    split_idx = int(len(X) * train_split)
    embargo = max(0, int(embargo_steps))
    train_end = max(0, split_idx - embargo)
    val_start = min(len(X), split_idx + embargo)
    return X[:train_end], X[val_start:], y[:train_end], y[val_start:]


def scale_sequences(
    X_train: np.ndarray,
    X_val: np.ndarray,
    scaler_type: str,
) -> Tuple[np.ndarray, np.ndarray, object]:
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    # Fit on train only to avoid leakage.
    train_2d = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(train_2d)

    def transform(X: np.ndarray) -> np.ndarray:
        flat = X.reshape(-1, X.shape[-1])
        scaled = scaler.transform(flat)
        return scaled.reshape(X.shape)

    return transform(X_train), transform(X_val), scaler


def build_model(sequence_length: int, num_features: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=(sequence_length, num_features)),
            LSTM(96, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(3, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    ensure_dir(MODEL_DIR)
    seed_everything(RANDOM_SEED)
    print("AI training started.")
    print(f"Assets: {ASSETS}")
    print(f"Timeframes: {TIMEFRAMES}")
    print(f"Label method: {LABEL_METHOD}, min_return={LABEL_MIN_RETURN}, horizon={LABEL_RETURN_HORIZON}")
    print(f"Hold keep ratio: {HOLD_KEEP_RATIO}, balance method: {BALANCE_METHOD}")

    all_X_train, all_y_train = [], []
    all_X_val, all_y_val = [], []
    feature_columns: List[str] = []
    base_timeframe = min(TIMEFRAMES, key=lambda tf: TIMEFRAME_MINUTES[tf])
    expected_minutes = TIMEFRAME_MINUTES[base_timeframe]
    rng = np.random.default_rng(RANDOM_SEED)

    asset_sequences = []
    for asset in ASSETS:
        bundle = build_feature_frame(asset)
        if bundle.features.empty:
            print(f"No data after preprocessing for {asset}")
            continue
        feature_columns = list(bundle.features.columns)
        X, y = create_sequences(
            bundle.features,
            bundle.labels,
            SEQUENCE_LENGTH,
            expected_minutes=expected_minutes,
            gap_tolerance=GAP_TOLERANCE,
            stride=SEQUENCE_STRIDE,
        )
        summarize_labels(y, f"{asset} raw")

        if BALANCE_METHOD in {"downsample", "both"}:
            X, y = downsample_sequences(X, y, HOLD_KEEP_RATIO, rng)
            summarize_labels(y, f"{asset} downsampled")

        if MAX_SAMPLES_PER_ASSET:
            X, y = cap_sequences(X, y, MAX_SAMPLES_PER_ASSET, rng)
            summarize_labels(y, f"{asset} capped")

        asset_sequences.append({"asset": asset, "X": X, "y": y})

    if BALANCE_ASSETS and asset_sequences:
        # Balance assets by trimming to the smallest sample count.
        min_samples = min(len(item["y"]) for item in asset_sequences if len(item["y"]) > 0)
        if min_samples > 0:
            for item in asset_sequences:
                item["X"], item["y"] = cap_sequences(item["X"], item["y"], min_samples, rng)
                summarize_labels(item["y"], f"{item['asset']} balanced")

    for item in asset_sequences:
        X, y = item["X"], item["y"]
        if len(y) == 0:
            print(f"Skipping {item['asset']} (no sequences).")
            continue
        X_train, X_val, y_train, y_val = split_train_val(
            X, y, TRAIN_SPLIT, embargo_steps=EMBARGO_STEPS
        )
        all_X_train.append(X_train)
        all_X_val.append(X_val)
        all_y_train.append(y_train)
        all_y_val.append(y_val)

    if not all_X_train or not any(len(x) for x in all_X_train):
        raise RuntimeError("No training data available.")

    X_train = np.concatenate(all_X_train, axis=0)
    X_val = np.concatenate(all_X_val, axis=0)
    y_train = np.concatenate(all_y_train, axis=0)
    y_val = np.concatenate(all_y_val, axis=0)

    if X_train.size == 0:
        raise RuntimeError("Training set is empty after preprocessing.")
    if X_val.size == 0:
        raise RuntimeError("Validation set is empty. Adjust TRAIN_SPLIT or EMBARGO_STEPS.")

    summarize_labels(y_train, "Train")
    summarize_labels(y_val, "Val")

    # Warn if one class is severely underrepresented.
    for label_name, label_value in {"hold": 0, "buy": 1, "sell": 2}.items():
        count = int(np.sum(y_train == label_value))
        if count < MIN_CLASS_SAMPLES:
            print(f"Warning: {label_name} has only {count} samples in train set.")

    X_train, X_val, scaler = scale_sequences(X_train, X_val, SCALER_TYPE)

    class_weight_map = None
    if BALANCE_METHOD in {"weights", "both"}:
        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )
        class_weight_map = {int(cls): float(weight) for cls, weight in zip(classes, class_weights)}

    model = build_model(SEQUENCE_LENGTH, X_train.shape[-1])
    model.summary(print_fn=lambda x: print(x))

    checkpoint_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.keras")
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-5),
        ModelCheckpoint(checkpoint_path, save_best_only=True),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_map,
        callbacks=callbacks,
        verbose=1,
    )

    # Save scaler and metadata.
    scaler_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_scaler.pkl")
    joblib.dump(scaler, scaler_path)

    metadata = {
        "model_name": MODEL_NAME,
        "data_source": DATA_SOURCE,
        "assets": ASSETS,
        "range_mode": RANGE_MODE,
        "hours": HOURS if RANGE_MODE == "Hours" else None,
        "start_dt": START_DT.isoformat() if RANGE_MODE == "Dates" else None,
        "end_dt": END_DT.isoformat() if RANGE_MODE == "Dates" else None,
        "timeframes": TIMEFRAMES,
        "base_timeframe": base_timeframe,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_columns": feature_columns,
        "indicator_flags": INDICATOR_FLAGS,
        "timeframe_params": {tf: get_timeframe_params(tf) for tf in TIMEFRAMES},
        "label_method": LABEL_METHOD,
        "label_window": EXTREMA_WINDOW,
        "label_min_return": LABEL_MIN_RETURN,
        "label_return_horizon": LABEL_RETURN_HORIZON,
        "label_cooldown": LABEL_COOLDOWN,
        "scaler_type": SCALER_TYPE,
        "sequence_stride": SEQUENCE_STRIDE,
        "gap_tolerance": GAP_TOLERANCE,
        "max_ffill_gap_multiplier": MAX_FFILL_GAP_MULTIPLIER,
        "embargo_steps": EMBARGO_STEPS,
        "balance_method": BALANCE_METHOD,
        "hold_keep_ratio": HOLD_KEEP_RATIO,
        "balance_assets": BALANCE_ASSETS,
        "max_samples_per_asset": MAX_SAMPLES_PER_ASSET,
        "class_labels": {"hold": 0, "buy": 1, "sell": 2},
    }
    metadata_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_meta.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Training complete.")
    print(f"Model saved: {checkpoint_path}")
    print(f"Scaler saved: {scaler_path}")
    print(f"Metadata saved: {metadata_path}")


if __name__ == "__main__":
    main()
