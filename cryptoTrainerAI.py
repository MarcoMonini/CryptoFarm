import numpy as np
import pandas as pd
from pyarrow import nulls
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator, StochasticOscillator, TSIIndicator
from scipy.signal import argrelextrema

# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

FEATURES = ['Open', 'High', 'Low', 'Close', 'RSI', 'STOCH', 'STOCH_S','ATR','TSI']

# Configurazioni principali
EXT_WINDOW_SIZE = 50  # Dimensione della finestra temporale per min max

WINDOW_SIZE = 20 # Dimensione della finestra temporale per le sequenze

ATR_WINDOW = 6   # Periodo dell'ATR
RSI_WINDOW = 12   # Periodo dell'RSI
# MACD_SHORT_WINDOW = 12
# MACD_LONG_WINDOW = 26
# MACD_SIGNAL_WINDOW = 9

# file = '/Users/marcomonini/Documents/BTC_1anno_15m.csv'
file = '/Users/marcomonini/Documents/BTC_12kh.csv'

def add_technical_indicator(df,  rsi_window=12, atr_window=6):
    df_copy = df.copy()

    # Calcolo dell'RSI
    rsi_indicator = RSIIndicator(close=df_copy['Close'], window=rsi_window)
    df_copy['RSI'] = rsi_indicator.rsi()

    # ATR
    atr_indicator = AverageTrueRange(
        high=df_copy['High'],
        low=df_copy['Low'],
        close=df_copy['Close'],
        window=atr_window
    )
    df_copy['ATR'] = atr_indicator.average_true_range()

    # STOCASTICO
    stoch_indicator = StochasticOscillator(
        high=df_copy['High'],
        low=df_copy['Low'],
        close=df_copy['Close'],
        window=rsi_window,
        smooth_window=3
    )
    df_copy['STOCH'] = stoch_indicator.stoch()
    df_copy['STOCH_S'] = stoch_indicator.stoch_signal()

    tsi_indicator = TSIIndicator(
        close=df_copy['Close'],
        window_slow=25,
        window_fast=13,
    )
    df_copy['TSI'] = tsi_indicator.tsi()

    df_copy.fillna(0, inplace=True)

    return df_copy

# trasforma il dataframe in ingresso in variazioni percentuali rispetto alla chiusura precedente
def calculate_percentage_changes(df):
    # Copia del DataFrame per non sovrascrivere i dati originali
    df_transformed = df.copy()
    # Calcolo delle variazioni percentuali rispetto alla chiusura precedente
    df_transformed['Open_Perc'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
    df_transformed['High_Perc'] = (df['High'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
    df_transformed['Low_Perc'] = (df['Low'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
    df_transformed['Close_Perc'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
    # df_transformed['Volume_Perc'] = (df['Volume'] - df['Volume'].shift(1)) / df['Volume'].shift(1) * 100
    # Rimuove i valori NaN (la prima riga avrà NaN dopo la trasformazione)
    df_transformed = df_transformed.dropna()
    # Aggiustamento per garantire la continuità
    prev_close = 0  # Punto iniziale di riferimento
    prev_vol_close = 0  # Punto iniziale di riferimento per il volume
    for i in range(len(df_transformed)):
        df_transformed.iloc[i, df_transformed.columns.get_loc('Open_Perc')] += prev_close
        df_transformed.iloc[i, df_transformed.columns.get_loc('High_Perc')] += prev_close
        df_transformed.iloc[i, df_transformed.columns.get_loc('Low_Perc')] += prev_close
        df_transformed.iloc[i, df_transformed.columns.get_loc('Close_Perc')] += prev_close
        # df_transformed.iloc[i, df_transformed.columns.get_loc('Volume_Perc')] += prev_vol_close
        # Aggiorna il valore di chiusura precedente
        prev_close = df_transformed.iloc[i, df_transformed.columns.get_loc('Close_Perc')]
        # prev_vol_close = df_transformed.iloc[i, df_transformed.columns.get_loc('Volume_Perc')]
    df_transformed['Open'] = df_transformed['Open_Perc']
    df_transformed['High'] = df_transformed['High_Perc']
    df_transformed['Low'] = df_transformed['Low_Perc']
    df_transformed['Close'] = df_transformed['Close_Perc']
    # df_transformed['Volume'] = df_transformed['Volume_Perc']
    df_final = df_transformed[['Open', 'High', 'Low', 'Close']].astype(float)

    return df_final


# Calcolo dei massimi e minimi relativi
def calculate_relative_extrema(data, window_pivot=EXT_WINDOW_SIZE):
    price_high = data['High']
    price_low = data['Low']
    order = int(window_pivot / 2)
    # Trova gli indici dei massimi e minimi relativi
    max_idx = argrelextrema(price_high.values, np.greater, order=order)[0]
    min_idx = argrelextrema(price_low.values, np.less, order=order)[0]
    # Inizializza colonna etichette
    data['Label'] = 0
    for i in max_idx:
        # data.loc[data.index[i-1], 'Label'] = 2  # Massimo relativo
        data.loc[data.index[i], 'Label'] = 2  # Massimo relativo
        # data.loc[data.index[i+1], 'Label'] = 2  # Massimo relativo
    for i in min_idx:
        # data.loc[data.index[i-1], 'Label'] = 1  # Minimo relativo
        data.loc[data.index[i], 'Label'] = 1  # Minimo relativo
        # data.loc[data.index[i+1], 'Label'] = 1  # Minimo relativo
    return data


# Ora bilancia solo il TRAIN SET
def balance_data(X, y):
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    idx2 = np.where(y == 2)[0]
    min_len = min(len(idx0), len(idx1), len(idx2))

    idx0 = np.random.choice(idx0, min_len*3, replace=False)
    idx1 = np.random.choice(idx1, min_len, replace=False)
    idx2 = np.random.choice(idx2, min_len, replace=False)

    idx_total = np.concatenate([idx0, idx1, idx2])
    np.random.shuffle(idx_total)

    return X[idx_total], y[idx_total]


def create_sequences(data, features, window_size):
    """
    Crea sequenze temporali bilanciate con target equamente distribuiti tra le classi 0, 1, 2.
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
    X_balanced : numpy.ndarray
        Sequenze bilanciate (shape: [num_samples, window_size, num_features]).
    y_balanced : numpy.ndarray
        Etichette bilanciate corrispondenti alle sequenze.
    """
    X, y = [], []

    # Creazione delle sequenze e delle etichette
    for i in range(len(data) - window_size):
        open = data['Open'].iloc[i]
        window = data[features].iloc[i:i + window_size].values

        for j, feature in enumerate(features):
            if feature in ['Open', 'High', 'Low', 'Close']:
                    window[:, j] = window[:, j] - open

        # window = data[features].iloc[i:i + window_size].values - open
        X.append(window)
        if 'Label' in data.columns:
            # Aggiungi l'etichetta corrispondente alla sequenza
            y.append(data['Label'].iloc[i + window_size])  # Etichetta del punto corrente

    # Converti in numpy array
    X = np.array(X)
    y = np.array(y)

    return X, y

def get_model_predictions(df, model):

    # data = calculate_percentage_changes(df)
    #data = add_technical_indicators(data)
    # data['Label'] = 0

    # data = df[features].values
    # X = []
    # y = []

    data = df.copy()
    data = data[FEATURES]
    X, y = create_sequences(data, FEATURES, WINDOW_SIZE)


    # X = np.array(X)
    y = model.predict(X, verbose=0)
    # preds_class = np.argmax(preds, axis=1)  # Se output one-hot, es: [0, 1, 0]
    y = np.nan_to_num(y, nan=0.0)
    # Verifica che la lunghezza coincida
    if len(df) - WINDOW_SIZE != y.shape[0]:
        raise ValueError("Dimension mismatch: df vs model predictions")

    # Allinea con l'indice originale del DataFrame
    df_preds = df.iloc[WINDOW_SIZE:].copy()

    preds = np.zeros(y.shape[0], dtype=int)
    # Troviamo l'indice della classe con probabilità massima
    max_probs = np.max(y, axis=1)
    # max_probs[not max_probs] = 0  # Imposta a 0 le probabilità sotto la soglia
    max_classes = np.argmax(y, axis=1)

    preds = np.where(max_probs > 0.8, max_classes, 0)

    df_preds['Prediction'] = preds

    # df_preds['Prediction'] = np.argmax(y, axis=1)  # Converte le probabilità in classi

    return df_preds


if __name__ == "__main__":
    # Caricamento dati
    raw_df = pd.read_csv(file)
    raw_df['Open time'] = pd.to_datetime(raw_df['Open time'])
    raw_df.set_index('Open time', inplace=True)
    # Mantieni solo le colonne essenziali, converti a float
    df = raw_df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

    print("add_technical_indicators")
    df = add_technical_indicator(df=df, rsi_window=RSI_WINDOW, atr_window=ATR_WINDOW)

    print("calculate_relative_extrema")
    df = calculate_relative_extrema(df)

    print("calculate_percentage_changes")
    df_transformed = calculate_percentage_changes(df)

    # Selezione delle feature
    print("Create_sequences")
    X, y = create_sequences(df, FEATURES, WINDOW_SIZE)

    # Dividi i dati in train e test
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # print("balance_data")
    X_train, y_train = balance_data(X_train, y_train)

    model = keras.Sequential([
        Input(shape=(X.shape[1], X.shape[2])),

        Bidirectional(LSTM(256, return_sequences=True)),
        Dropout(0.1),
        BatchNormalization(),

        LSTM(128, return_sequences=True),
        Dropout(0.1),
        BatchNormalization(),

        LSTM(64, return_sequences=False),
        Dropout(0.1),
        BatchNormalization(),

        Dense(3, activation='relu'),

        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # Addestramento del modello
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stopping])

    # Valutazione del modello
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    # Previsione su nuovi dati
    # predictions = get_model_predictions(df, model, FEATURES, WINDOW_SIZE)
    predictions = model.predict(X_train, verbose=1)

    # predictions = model.predict(X)
    print(predictions[:10])

    # Salva il modello in un file HDF5
    model.save('trained_model.keras')
    # # Carica il modello salvato
    # loaded_model = load_model('trained_model.h5')
    # # Utilizzo del modello per fare previsioni
    # predictions = loaded_model.predict(X_test)
