import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator, StochasticOscillator, TSIIndicator
from scipy.signal import argrelextrema
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

FEATURES = ['Open', 'High', 'Low', 'Close', 'RSI', 'STOCH', 'STOCH_S','ATR','TSI']#,'EMA20','EMA50','EMA100','EMA200']

# Configurazioni principali
EXT_WINDOW_SIZE = 100  # Dimensione della finestra temporale per min max

WINDOW_SIZE = 50 # Dimensione della finestra temporale per le sequenze

ATR_WINDOW = 6   # Periodo dell'ATR
RSI_WINDOW = 12   # Periodo dell'RSI
# MACD_SHORT_WINDOW = 12
# MACD_LONG_WINDOW = 26
# MACD_SIGNAL_WINDOW = 9

# file = '/Users/marcomonini/Documents/BTC_1anno_15m.csv'

def prepare_df_from_csv(csv_file:str):
    # Caricamento dati
    raw_df = pd.read_csv(csv_file)
    raw_df['Open time'] = pd.to_datetime(raw_df['Open time'])
    raw_df.set_index('Open time', inplace=True)
    # Mantieni solo le colonne essenziali, converti a float
    df = raw_df[['Open', 'High', 'Low', 'Close']].astype(float)
    df = add_technical_indicator(df=df, rsi_window=RSI_WINDOW, atr_window=ATR_WINDOW)
    df = calculate_relative_extrema(df)

    # data normalization
    df_transformed = calculate_percentage_changes(df)
    df_transformed.dropna(inplace=True)
    features = FEATURES
    features.append('Label')
    df_transformed = df_transformed[features]
    # df_transformed, scaler = normalize_features(df)

    return df_transformed


def add_technical_indicator(df,  rsi_window=12, atr_window=6):
    print("add_technical_indicators")
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

    # EMA (Media Mobile per le Rolling ATR Bands)
    # ema_indicator = EMAIndicator(close=df_copy['Close'], window=20)
    # df_copy['EMA20'] = ema_indicator.ema_indicator()
    # df_copy['EMA20'] = df_copy['EMA20'] / df_copy['Close']
    # ema_indicator = EMAIndicator(close=df_copy['Close'], window=50)
    # df_copy['EMA50'] = ema_indicator.ema_indicator()
    # df_copy['EMA50'] = df_copy['EMA50'] / df_copy['Close']
    # ema_indicator = EMAIndicator(close=df_copy['Close'], window=100)
    # df_copy['EMA100'] = ema_indicator.ema_indicator()
    # df_copy['EMA100'] = df_copy['EMA100'] / df_copy['Close']
    # emao_indicator = EMAIndicator(close=df_copy['Open'], window=200)
    # df_copy['EMA200'] = emao_indicator.ema_indicator()
    # df_copy['EMA200'] = df_copy['EMA200'] / df_copy['Open']

    df_copy.fillna(0, inplace=True)

    return df_copy

# trasforma il dataframe in ingresso in variazioni percentuali rispetto alla chiusura precedente
def calculate_percentage_changes(df):
    print("calculate_percentage_changes")

    # Copia del DataFrame per non sovrascrivere i dati originali
    df_transformed = df.copy()
    # Calcolo delle variazioni percentuali rispetto alla chiusura precedente
    df_transformed['Open_Perc'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
    df_transformed['High_Perc'] = (df['High'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
    df_transformed['Low_Perc'] = (df['Low'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
    df_transformed['Close_Perc'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100

    # Gestione della prima riga: uso df.loc[0] (o il primo index se diverso)
    df_transformed.iloc[0, df_transformed.columns.get_loc('Open_Perc')] = 0
    base_open = df.iloc[0]['Open']

    df_transformed.iloc[0, df_transformed.columns.get_loc('High_Perc')] = (df.iloc[0][
                                                                               'High'] - base_open) / base_open * 100
    df_transformed.iloc[0, df_transformed.columns.get_loc('Low_Perc')] = (df.iloc[0][
                                                                              'Low'] - base_open) / base_open * 100
    df_transformed.iloc[0, df_transformed.columns.get_loc('Close_Perc')] = (df.iloc[0][
                                                                                'Close'] - base_open) / base_open * 100

    # df_transformed['Volume_Perc'] = (df['Volume'] - df['Volume'].shift(1)) / df['Volume'].shift(1) * 100
    # Rimuove i valori NaN (la prima riga avrà NaN dopo la trasformazione)
    df_transformed = df_transformed.dropna()
    #df_transformed = df_transformed.fillna(0, inplace=True)
    # Aggiustamento per garantire la continuità
    prev_close = 0  # Punto iniziale di riferimento
    #prev_vol_close = 0  # Punto iniziale di riferimento per il volume
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

    df_transformed.fillna(0, inplace=True)
    # df_transformed['Volume'] = df_transformed['Volume_Perc']
    # df_final = df_transformed[['Open', 'High', 'Low', 'Close']].astype(float)

    return df_transformed


# Calcolo dei massimi e minimi relativi
def calculate_relative_extrema(data, window_pivot=EXT_WINDOW_SIZE):
    print("calculate_relative_extrema")
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

    idx0 = np.random.choice(idx0, min_len*10, replace=False)
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
    print("Create_sequences")

    X, y = [], []
    df_copy = data.copy()
    df_copy = df_copy[features]
    # Creazione delle sequenze e delle etichette
    for i in range(len(df_copy) - window_size):
        open = df_copy['Open'].iloc[i]
        window = df_copy[features].iloc[i:i + window_size].values

        for j, feature in enumerate(features):
            if feature in ['Open', 'High', 'Low', 'Close']:
                    window[:, j] = window[:, j] - open

        # window = df_copy[features].iloc[i:i + window_size].values - open
        X.append(window)
        if 'Label' in df_copy.columns:
            # Aggiungi l'etichetta corrispondente alla sequenza
            y.append(df_copy['Label'].iloc[i + window_size])  # Etichetta del punto corrente

    # Converti in numpy array
    X = np.array(X)
    y = np.array(y)

    return X, y

def get_model_predictions(df, model):
    data = df.copy()
    data.fillna(0, inplace=True)
    data = data[FEATURES]

    df_transformed = calculate_percentage_changes(data)
    #df_transformed.dropna(inplace=True)
    #df_transformed, scaler = normalize_features(df_transformed)

    X, y = create_sequences(df_transformed, FEATURES, WINDOW_SIZE)

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
    preds = np.where(max_probs > 0.6, max_classes, 0)
    df_preds['Prediction'] = preds
    # df_preds['Prediction'] = np.argmax(y, axis=1)  # Converte le probabilità in classi

    return df_preds

# Funzione per normalizzare le feature numeriche
def normalize_features(df, scaler=None):
    print("Normalize features")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if scaler is None:
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df, scaler

# Funzione per costruire il modello (usata da Keras Tuner)
def build_model(hp):
    print("Build model")
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))

    model.add(Bidirectional(LSTM(
        units=hp.Int('lstm_units1', 64, 256, step=64),
        return_sequences=True)))
    model.add(Dropout(hp.Float('dropout1', 0.1, 0.3, step=0.1)))
    model.add(BatchNormalization())

    model.add(LSTM(
        units=hp.Int('lstm_units2', 64, 256, step=64),
        return_sequences=True))
    model.add(Dropout(hp.Float('dropout2', 0.1, 0.3, step=0.1)))
    model.add(BatchNormalization())

    model.add(LSTM(
        units=hp.Int('lstm_units3', 64, 256, step=64),
        return_sequences=False))
    model.add(Dropout(hp.Float('dropout3', 0.1, 0.3, step=0.1)))
    model.add(BatchNormalization())

    model.add(Dense(3, activation='softmax'))

    model.compile(
        optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":

    # Selezione delle feature
    fileBTC = '/Users/marcomonini/Documents/BTCUSDC_2anni_15m.csv'
    #fileETH = '/Users/marcomonini/Documents/ETH_2anni_15m.csv'
    df = prepare_df_from_csv(fileBTC)
    #df2 = prepare_df_from_csv(fileETH)
    #df = pd.concat([df1, df2], axis=0)

    X, y = create_sequences(df, FEATURES, WINDOW_SIZE)

    # Dividi i dati in train e test
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # X_train, y_train = balance_data(X_train, y_train)

    # Class weights per gestione sbilanciamento
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))

    # === TUNING CON KERAS TUNER ===
    # tuner = kt.Hyperband(
    #     build_model,
    #     objective='val_accuracy',
    #     max_epochs=20,
    #     factor=3,
    #     directory='tuner_logs',
    #     project_name='crypto_lstm',
    #     overwrite = True
    # )

    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),

        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        BatchNormalization(),

        LSTM(192, return_sequences=True),
        Dropout(0.1),
        BatchNormalization(),

        LSTM(256, return_sequences=False),
        Dropout(0.1),
        BatchNormalization(),

        #Dense(3, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(0.01),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-5)
    checkpoint = ModelCheckpoint('optimized_model.keras', monitor='val_loss', save_best_only=True, verbose=1)

    # print("Searching for best model")
    # tuner.search(
    #     X_train, y_train,
    #     epochs=50,
    #     validation_data=(X_test, y_test),
    #     callbacks=[early_stopping, reduce_lr],
    #     class_weight=class_weights,
    #     batch_size=32
    # )
    #
    # # Miglior modello trovato
    # model = tuner.get_best_models(num_models=1)[0]

    print("Final Training")
    # === ADDDESTRAMENTO FINALE CON IL MIGLIOR MODELLO ===
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        class_weight=class_weights
    )

    #
    # # 1. Compilazione del modello con learning rate definito
    # optimizer = Adam(learning_rate=0.001)
    #
    # model.compile(
    #     optimizer=optimizer,
    #     loss='sparse_categorical_crossentropy',  # per target con etichette intere
    #     metrics=['accuracy']
    # )
    #
    # # 2. Callback
    # early_stopping = EarlyStopping(
    #     monitor='val_loss',
    #     patience=10,
    #     restore_best_weights=True
    # )
    #
    # reduce_lr = ReduceLROnPlateau(
    #     monitor='val_loss',
    #     factor=0.5,
    #     patience=3,
    #     verbose=1,
    #     min_lr=1e-5
    # )
    #
    # checkpoint = ModelCheckpoint(
    #     filepath='optimized_model.keras',
    #     monitor='val_loss',
    #     save_best_only=True,
    #     verbose=1
    # )

    # 3. Addestramento
    # history = model.fit(
    #     X_train, y_train,
    #     validation_data=(X_test, y_test),
    #     epochs=50,
    #     batch_size=32,
    #     callbacks=[early_stopping, reduce_lr, checkpoint],
    #     class_weight=class_weights,
    #     #verbose=2
    # )

    #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # Addestramento del modello
    #model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stopping])

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
