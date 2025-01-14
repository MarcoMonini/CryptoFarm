import numpy as np
import pandas as pd
from ta.trend import PSARIndicator, SMAIndicator, MACD
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from scipy.signal import argrelextrema

# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Configurazioni principali
WINDOW_SIZE = 50  # Dimensione della finestra temporale
ATR_WINDOW = 14   # Periodo dell'ATR
RSI_WINDOW = 14   # Periodo dell'RSI
MACD_SHORT_WINDOW = 12
MACD_LONG_WINDOW = 26
MACD_SIGNAL_WINDOW = 9
ATR_MULTIPLIER = 2
STEP = 0.02
MAX_STEP = 0.2

file = 'market_data.csv'


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

    return df_transformed

# Calcolo dei massimi e minimi relativi
def calculate_relative_extrema(data, window_pivot=WINDOW_SIZE):
    price_high = data['High']
    price_low = data['Low']
    order = int(window_pivot / 2)
    # Trova gli indici dei massimi e minimi relativi
    max_idx = argrelextrema(price_high.values, np.greater, order=order)[0]
    min_idx = argrelextrema(price_low.values, np.less, order=order)[0]
    # Inizializza colonna etichette
    data['Label'] = 0
    for i in max_idx:
        data.loc[data.index[i], 'Label'] = 1  # Massimo relativo
    for i in min_idx:
        data.loc[data.index[i], 'Label'] = -1  # Minimo relativo
    return data


# Calcolo degli indicatori tecnici
def add_technical_indicators(data):
    # Parabolic SAR
    sar_indicator = PSARIndicator(high=data['High'], low=data['Low'], close=data['Close'], step=STEP, max_step=MAX_STEP)
    data['PSAR'] = sar_indicator.psar()

    # ATR
    atr_indicator = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=ATR_WINDOW)
    data['ATR'] = atr_indicator.average_true_range()

    # SMA
    sma_indicator = SMAIndicator(close=data['Close'], window=ATR_WINDOW)
    data['SMA'] = sma_indicator.sma_indicator()
    # Rolling ATR Bands
    # data['Upper_Band'] = data['SMA'] + ATR_MULTIPLIER * data['ATR']
    # data['Lower_Band'] = data['SMA'] - ATR_MULTIPLIER * data['ATR']

    # RSI
    rsi_indicator = RSIIndicator(close=data['Close'], window=RSI_WINDOW)
    data['RSI'] = rsi_indicator.rsi()

    # MACD
    macd_indicator = MACD(close=data['Close'], window_slow=MACD_LONG_WINDOW, window_fast=MACD_SHORT_WINDOW, window_sign=MACD_SIGNAL_WINDOW)
    data['MACD'] = macd_indicator.macd_diff()
    # data['MACD'] = macd_indicator.macd_diff() / data['Close'] * 100  # Normalizzato

    return data


# Creazione delle sequenze temporali
def create_sequences_with_target(data, features, window_size):
    X, y = [], []
    for i in range(len(data) - window_size - 1):
        # Include la finestra temporale precedente e il punto corrente
        window = data[features].iloc[i:i+window_size+1].values
        X.append(window)
        y.append(data['Label'].iloc[i+window_size])  # Etichetta del punto corrente
    return np.array(X), np.array(y)


# Caricamento dati
df = pd.read_csv(file)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Applicazione della funzione al DataFrame
df_transformed = calculate_percentage_changes(df)

df = calculate_relative_extrema(df)
df = add_technical_indicators(df)

# Selezione delle feature
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'ATR', 'PSAR', 'SMA', 'MACD', 'VI']
X, y = create_sequences_with_target(df, features, WINDOW_SIZE)

# Dividi i dati in train e test
train_size = int(0.7 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Creazione del modello LSTM
model = keras.Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1, activation='tanh')  # Output (-1, 0, 1)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Addestramento del modello
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# Valutazione del modello
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Previsione su nuovi dati
predictions = model.predict(X_test)
print(predictions[:10])
