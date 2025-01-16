import numpy as np
import pandas as pd
from ta.trend import PSARIndicator, SMAIndicator, MACD, VortexIndicator
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from scipy.signal import argrelextrema

# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# Configurazioni principali
EXT_WINDOW_SIZE = 80  # Dimensione della finestra temporale per min max

WINDOW_SIZE = 10
# ATR_WINDOW = 14   # Periodo dell'ATR
# RSI_WINDOW = 14   # Periodo dell'RSI
MACD_SHORT_WINDOW = 12
MACD_LONG_WINDOW = 26
MACD_SIGNAL_WINDOW = 9
ATR_MULTIPLIER = 2
STEP = 0.02
MAX_STEP = 0.2

file = 'C:/Users/marco/Documents/BTC_1ANNO_15m.csv'


def calculate_percentage_changes(df):
    # Copia del DataFrame per non sovrascrivere i dati originali
    df_transformed = df.copy()
    # Calcolo delle variazioni percentuali rispetto alla chiusura precedente
    df_transformed['Open_Perc'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
    df_transformed['High_Perc'] = (df['High'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
    df_transformed['Low_Perc'] = (df['Low'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
    df_transformed['Close_Perc'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
    # Rimuove i valori NaN (la prima riga avrà NaN dopo la trasformazione)
    df_transformed = df_transformed.dropna()
    # Aggiustamento per garantire la continuità
    prev_close = 0  # Punto iniziale di riferimento
    for i in range(len(df_transformed)):
        df_transformed.iloc[i, df_transformed.columns.get_loc('Open_Perc')] += prev_close
        df_transformed.iloc[i, df_transformed.columns.get_loc('High_Perc')] += prev_close
        df_transformed.iloc[i, df_transformed.columns.get_loc('Low_Perc')] += prev_close
        df_transformed.iloc[i, df_transformed.columns.get_loc('Close_Perc')] += prev_close
        # Aggiorna il valore di chiusura precedente
        prev_close = df_transformed.iloc[i, df_transformed.columns.get_loc('Close_Perc')]
    df_transformed['Open'] = df_transformed['Open_Perc']
    df_transformed['High'] = df_transformed['High_Perc']
    df_transformed['Low'] = df_transformed['Low_Perc']
    df_transformed['Close'] = df_transformed['Close_Perc']
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
        data.loc[data.index[i], 'Label'] = 2  # Massimo relativo
    for i in min_idx:
        data.loc[data.index[i], 'Label'] = 1  # Minimo relativo
    return data


# Calcolo degli indicatori tecnici
def add_technical_indicators(data):
    # Calcolo del SAR utilizzando la libreria "ta" (PSARIndicator)
    sar_indicator = PSARIndicator(
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        step=STEP,
        max_step=MAX_STEP
    )
    data['SAR'] = sar_indicator.psar()

    # ATR
    atr_indicator = AverageTrueRange(
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        window=WINDOW_SIZE
    )
    data['ATR'] = atr_indicator.average_true_range()

    # SMA (Media Mobile per le Rolling ATR Bands)
    sma_indicator = SMAIndicator(close=data['Close'], window=WINDOW_SIZE)
    data['SMA'] = sma_indicator.sma_indicator()

    # Calcolo dell'RSI
    # Impostazione classica RSI(14). Se vuoi segnali più veloci, puoi provare RSI(7) o RSI(9).
    rsi_indicator = RSIIndicator(
        close=data['Close'],
        window=WINDOW_SIZE
    )
    data['RSI'] = rsi_indicator.rsi()

    # Calcolo delle linee MACD
    # Calcolo del MACD
    macd_indicator = MACD(
        close=data['Close'],
        window_slow=MACD_LONG_WINDOW,
        window_fast=MACD_SHORT_WINDOW,
        window_sign=MACD_SIGNAL_WINDOW
    )
    data['MACD'] = macd_indicator.macd_diff()  # Istogramma (differenza tra MACD e Signal Line)

    # Vortex Indicator
    vi = VortexIndicator(
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        window=WINDOW_SIZE)
    vip = vi.vortex_indicator_pos()
    vim = vi.vortex_indicator_neg()
    data['VI'] = vip - vim

    final_data = data.dropna()

    return final_data


# Creazione delle sequenze temporali
def create_sequences_with_target(data, features, window_size):
    X, y = [], []
    for i in range(len(data) - window_size - 1):
        # Include la finestra temporale precedente e il punto corrente
        window = data[features].iloc[i:i+window_size+1].values
        X.append(window)
        y.append(data['Label'].iloc[i+window_size])  # Etichetta del punto corrente
    return np.array(X), np.array(y)


def create_balanced_sequences_with_target(data, features, window_size):
    """
    Crea sequenze temporali bilanciate con target equamente distribuiti tra le classi 0, 1, 2.

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
    for i in range(len(data) - window_size - 1):
        # Include la finestra temporale precedente
        window = data[features].iloc[i:i + window_size].values
        X.append(window)
        y.append(data['Label'].iloc[i + window_size])  # Etichetta del punto corrente

    # Converti in numpy array
    X = np.array(X)
    y = np.array(y)

    # Trova gli indici per ogni classe
    indices_0 = np.where(y == 0)[0]
    indices_1 = np.where(y == 1)[0]
    indices_2 = np.where(y == 2)[0]

    # Trova il numero minimo di campioni tra le classi
    min_samples = min(len(indices_0), len(indices_1), len(indices_2))

    # Sottocampiona per bilanciare le classi
    balanced_indices_0 = np.random.choice(indices_0, min_samples, replace=False)
    balanced_indices_1 = np.random.choice(indices_1, min_samples, replace=False)
    balanced_indices_2 = np.random.choice(indices_2, min_samples, replace=False)

    # Combina gli indici bilanciati
    balanced_indices = np.concatenate([balanced_indices_0, balanced_indices_1, balanced_indices_2])

    # Mescola gli indici per evitare che le classi siano ordinate
    np.random.shuffle(balanced_indices)

    # Seleziona le sequenze bilanciate e le etichette
    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]

    return X_balanced, y_balanced


if __name__ == "__main__":
    # Caricamento dati
    raw_df = pd.read_csv(file)
    raw_df['Open time'] = pd.to_datetime(raw_df['Open time'])
    raw_df.set_index('Open time', inplace=True)
    # Mantieni solo le colonne essenziali, converti a float
    df = raw_df[['Open', 'High', 'Low', 'Close']].astype(float)

    print("calculate_percentage_changes")
    df_transformed = calculate_percentage_changes(df)
    print("calculate_relative_extrema")
    df = calculate_relative_extrema(df_transformed)
    print("add_technical_indicators")
    df = add_technical_indicators(df)

    # print(df)
    # Selezione delle feature
    features = ['Open', 'High', 'Low', 'Close', 'RSI', 'ATR', 'SAR', 'SMA', 'MACD', 'VI']

    X, y = create_balanced_sequences_with_target(df, features, WINDOW_SIZE)

    # Dividi i dati in train e test
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Creazione del modello LSTM
    # model = keras.Sequential([
    #     LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    #     Dropout(0.2),
    #     LSTM(50),
    #     Dropout(0.2),
    #     Dense(3, activation='softmax')  # Output (0, 1, 2) (MIN, none, MAX)
    # ])
    # Creazione del modello LSTM
    model = keras.Sequential([
        Input(shape=(X.shape[1], X.shape[2])),  # Definizione dell'input
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(3, activation='softmax')  # Output (0, 1, 2) (MIN, none, MAX)
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Addestramento del modello
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

    # Valutazione del modello
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    # Previsione su nuovi dati
    predictions = model.predict(X_test)
    print(predictions[:10])

    # Salva il modello in un file HDF5
    model.save('trained_model.keras')
    # # Carica il modello salvato
    # loaded_model = load_model('trained_model.h5')
    # # Utilizzo del modello per fare previsioni
    # predictions = loaded_model.predict(X_test)
