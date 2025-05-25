import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator, StochasticOscillator, KAMAIndicator, TSIIndicator
from ta.trend import MACD, PSARIndicator, EMAIndicator
from binance import Client
import streamlit as st
import numpy as np
import math
import time
import warnings

# from scipy.signal import argrelextrema
# from tensorflow.keras.models import load_model
# from cryptoTrainerAI import calculate_percentage_changes, add_technical_indicators

# Disattiva i FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)


def interval_to_minutes(interval: str) -> int:
    """
    Converte l'intervallo di Binance (es. "1m", "15m", "1h") in minuti.
    """
    if interval.endswith('m'):
        # Intervalli tipo "1m", "3m", "15m", "30m", ecc.
        return int(interval.replace('m', ''))
    elif interval.endswith('h'):
        # Intervalli tipo "1h", "2h", ecc.
        hours = int(interval.replace('h', ''))
        return hours * 60
    else:
        # Se non corrisponde a 'm' o 'h', gestisci come preferisci (o ritorna 0)
        return 0


@st.cache_data
def get_market_data(asset: str, interval: str, time_hours: int) -> tuple:
    """
    Scarica i dati di mercato di 'asset' per le ultime 'time_hours' ore,
    basandosi sull'intervallo 'interval' (es. '1m', '5m', '1h').

    Se il numero di candele necessarie supera 1000 (limite Binance),
    fa più richieste e unisce i dati in un unico DataFrame ordinato.

    Parameters
    ----------
    asset : str
        Il simbolo dell'asset da scaricare (es. "BTCUSDC").
    interval : str
        L'intervallo tra le candele (es. "1m", "3m", "5m", "1h").
    time_hours : int
        Numero di ore di dati da scaricare.

    Returns
    -------
    pd.DataFrame
        DataFrame con colonne ['Open', 'High', 'Low', 'Close', 'Volume']
        e indice temporale (Open time). Ordinato dalla candela più vecchia
        a quella più recente, senza duplicati.
    """
    print(f"Scarico ~{time_hours} ore di dati per {asset}, intervallo={interval}")

    # Inizializza il client (personalizza se hai già un'istanza altrove)
    client = Client(api_key="<api_key>", api_secret="<api_secret>")

    # 1. Converte l'intervallo (es. "5m") in minuti. Gestisce possibili errori.
    candlestick_minutes = interval_to_minutes(interval)
    if candlestick_minutes <= 0:
        raise ValueError(f"Intervallo '{interval}' non supportato o non valido.")

    # 2. Calcola quante candele totali sono necessarie per coprire `time_hours`.
    #    Esempio: se time_hours=24 e interval="1m", candlestick_minutes=1 => servono 24*60=1440 candele
    needed_candles = math.ceil((time_hours * 60) / candlestick_minutes)

    print(f"Servono ~{needed_candles} candele totali (max 1000 per singola fetch).")

    # 3. Determina l'istante attuale (fine periodo), e da lì il "start_time" in millisecondi.
    now_ms = int(time.time() * 1000)  # adesso in ms
    # Ogni candela dura candlestick_minutes. Quindi totalNeededMs:
    totalNeededMs = needed_candles * candlestick_minutes * 60_000
    start_ms = now_ms - totalNeededMs

    # 4. Scarica i dati in più chunk da 1000 candele, se necessario
    all_klines = []
    fetch_start = start_ms
    candles_left = needed_candles

    while candles_left > 0:
        # Quante candele proviamo a prendere in questa fetch
        chunk_size = min(1000, candles_left)

        # Esegui la fetch
        chunk_klines = client.get_klines(
            symbol=asset,
            interval=interval,
            limit=chunk_size,  # max 1000
            startTime=fetch_start,  # in ms
            endTime=now_ms  # in ms
        )

        if not chunk_klines:
            # Se è vuoto, vuol dire che non ci sono più dati (o l'asset è troppo giovane)
            break

        # Aggiungiamo quanto scaricato alla lista generale
        all_klines.extend(chunk_klines)

        # Diminuiamo il numero di candele da richiedere
        real_fetched = len(chunk_klines)
        candles_left -= real_fetched

        # Calcoliamo l'open time dell'ultima candela (in ms)
        last_open_time = chunk_klines[-1][0]  # colonna 0 è "Open time"
        # Saltiamo all'open time successivo (cioè la candela dopo l'ultima)
        # in modo da non duplicare dati nel prossimo loop
        next_open_time = last_open_time + (candlestick_minutes * 60_000)

        # Se non abbiamo recuperato 1000 candele,
        # è probabile che siamo già arrivati oltre i dati disponibili
        if real_fetched < chunk_size:
            break

        # Aggiorna start time per il prossimo ciclo
        fetch_start = next_open_time

        # Se siamo già andati oltre la data "now_ms", possiamo uscire
        if fetch_start >= now_ms:
            break

    if not all_klines:
        # Nessun dato trovato
        print(f"Nessun dato trovato per {asset} su {interval} per le ultime {time_hours} ore.")
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume']), 0

    # 5. Costruisci il DataFrame da all_klines
    columns = [
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
        'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
        'Taker buy quote asset volume', 'Ignore'
    ]
    raw_df = pd.DataFrame(all_klines, columns=columns)

    # 6. Converte i timestamp e imposta l'indice
    raw_df['Open time'] = pd.to_datetime(raw_df['Open time'], unit='ms')
    raw_df.set_index('Open time', inplace=True)

    # Mantieni solo le colonne essenziali, converti a float
    df = raw_df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

    # 7. Ordina per data (dalla più vecchia alla più recente) e rimuovi duplicati
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]  # elimina eventuali duplicati su 'Open time'

    # Calcolo delle ore effettive di dati disponibili
    if not df.empty:
        actual_hours = len(df) * candlestick_minutes / 60
    else:
        actual_hours = 0

    print(f"Scaricate {len(df)} ({actual_hours} ore) candele reali per {asset} (richieste iniziali: {needed_candles}).")

    return df, actual_hours


@st.cache_data
def get_market_data_between_dates(asset: str, interval: str, start_date: str, end_date: str) -> tuple:
    """
       Scarica i dati di mercato di 'asset' per l'intervallo temporale specificato,
       basandosi sull'intervallo 'interval' (es. '1m', '5m', '1h').

       Se il numero di candele necessarie supera 1000 (limite Binance),
       fa più richieste e unisce i dati in un unico DataFrame ordinato.

       Parameters
       ----------
       asset : str
           Il simbolo dell'asset da scaricare (es. "BTCUSDC").
       interval : str
           L'intervallo tra le candele (es. "1m", "3m", "5m", "1h").
       start_date : str
           Data di inizio (es. "2023-01-01 00:00:00").
       end_date : str
           Data di fine (es. "2023-01-02 00:00:00").

       Returns
       -------
       pd.DataFrame
           DataFrame con colonne ['Open', 'High', 'Low', 'Close', 'Volume']
           e indice temporale (Open time), ordinato dalla candela più vecchia a quella più recente.
       """
    print(f"Scarico dati per {asset}, intervallo={interval}, da {start_date} a {end_date}")

    # Converte le date in timestamp in millisecondi
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    # Inizializza il client (personalizza se hai già un'istanza altrove)
    client = Client(api_key="<api_key>", api_secret="<api_secret>")

    # 1. Converte l'intervallo (es. "5m") in minuti.
    candlestick_minutes = interval_to_minutes(interval)
    if candlestick_minutes <= 0:
        raise ValueError(f"Intervallo '{interval}' non supportato o non valido.")

    # 2. Calcola quante candele totali sono necessarie per coprire l'intervallo specificato.
    total_minutes = (end_ms - start_ms) / 60000  # totale minuti
    needed_candles = math.ceil(total_minutes / candlestick_minutes)
    print(f"Servono ~{needed_candles} candele totali.")

    # 3. Scarica i dati in chunk da 1000 candele (limite Binance)
    all_klines = []
    fetch_start = start_ms
    candles_left = needed_candles

    while candles_left > 0:
        # Quante candele proviamo a prendere in questa fetch
        chunk_size = min(1000, candles_left)

        # Esegui la fetch, limitando i dati a quelli disponibili fino a end_ms
        chunk_klines = client.get_klines(
            symbol=asset,
            interval=interval,
            limit=chunk_size,  # max 1000
            startTime=fetch_start,  # in ms
            endTime=end_ms  # in ms
        )

        if not chunk_klines:
            # Se è vuoto, non ci sono più dati disponibili
            break

        # Aggiungi i dati scaricati alla lista generale
        all_klines.extend(chunk_klines)
        real_fetched = len(chunk_klines)
        candles_left -= real_fetched

        # Calcola l'open time dell'ultima candela per impostare il prossimo fetch
        last_open_time = chunk_klines[-1][0]
        next_open_time = last_open_time + (candlestick_minutes * 60_000)

        # Se il numero di candele recuperato è inferiore a quello richiesto, esci dal loop
        if real_fetched < chunk_size:
            break

        fetch_start = next_open_time
        if fetch_start >= end_ms:
            break

    if not all_klines:
        print(f"Nessun dato trovato per {asset} nell'intervallo specificato.")
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume']), 0

    # 4. Costruisci il DataFrame dai dati scaricati
    columns = [
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
        'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
        'Taker buy quote asset volume', 'Ignore'
    ]
    raw_df = pd.DataFrame(all_klines, columns=columns)

    # 5. Converte i timestamp e imposta l'indice
    raw_df['Open time'] = pd.to_datetime(raw_df['Open time'], unit='ms')
    raw_df.set_index('Open time', inplace=True)
    df = raw_df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

    # 6. Ordina per data e rimuove eventuali duplicati
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    # Calcola le ore effettive di dati disponibili
    actual_hours = len(df) * candlestick_minutes / 60 if not df.empty else 0
    print(f"Scaricate {len(df)} candele ({actual_hours} ore) per {asset}.")

    return df, actual_hours


def download_market_data(assets: list, intervals: list, hours: int):
    """
    Scarica i dati di mercato per tutti gli asset e intervalli specificati.
    I dati vengono salvati in un dizionario per un utilizzo futuro.

    Parameters
    ----------
    assets : list
        Lista di asset (es. ["BTCUSDT", "ETHUSDT"]).
    intervals : list
        Lista di intervalli (es. ["1m", "5m"]).
    hours : int
        Numero di ore di dati da scaricare.

    Returns
    -------
    dict
        Dizionario con i dati scaricati organizzati come dati[asset][interval].
    """
    dati = {}
    for asset in assets:
        dati[asset] = {}
        for interval in intervals:
            try:
                print(f"Scarico dati per {asset} - {interval}")
                df, _ = get_market_data(asset=asset, interval=interval, time_hours=hours)
                dati[asset][interval] = df
            except Exception as e:
                print(f"Errore durante il download dei dati per {asset} - {interval}: {e}")
                dati[asset][interval] = None
    return dati


@st.cache_data
def add_technical_indicator(df, step, max_step, rsi_window, rsi_window2, rsi_window3,
                            macd_long_window, macd_short_window, macd_signal_window,
                            ema_window, ema_window2, ema_window3,
                            atr_window, atr_multiplier,
                            kama_pow1, kama_pow2
                            ):
    df_copy = df.copy()
    # Calcolo del SAR utilizzando la libreria "ta" (PSARIndicator)
    sar_indicator = PSARIndicator(
        high=df_copy['High'],
        low=df_copy['Low'],
        close=df_copy['Close'],
        step=step,
        max_step=max_step
    )
    df_copy['PSAR'] = sar_indicator.psar()
    df_copy['PSARVP'] = df_copy['PSAR'] / df_copy['Close']

    # Calcolo dell'RSI
    rsi_indicator = RSIIndicator(close=df_copy['Close'], window=rsi_window)
    df_copy['RSI'] = rsi_indicator.rsi()
    rsi_indicator = RSIIndicator(close=df_copy['Close'], window=rsi_window2)
    df_copy['RSI2'] = rsi_indicator.rsi()
    rsi_indicator = RSIIndicator(close=df_copy['Close'], window=rsi_window3)
    df_copy['RSI3'] = rsi_indicator.rsi()
    # SMA dell'RSI
    # df_copy['RSI_S'] = df_copy['RSI'].rolling(window=3).mean()
    # df_copy['RSI2_S'] = df_copy['RSI2'].rolling(window=3).mean()
    # df_copy['RSI3_S'] = df_copy['RSI3'].rolling(window=3).mean()

    # Calcolo del MACD
    # macd_indicator = MACD(
    #     close=df_copy['Close'],
    #     window_slow=macd_long_window,
    #     window_fast=macd_short_window,
    #     window_sign=macd_signal_window
    # )
    # df_copy['MACD_L'] = macd_indicator.macd()
    # df_copy['MACD_S'] = macd_indicator.macd_signal()
    # df_copy['MACD'] = macd_indicator.macd_diff()  # Istogramma (differenza tra MACD e Signal Line)
    # # Calcolo del MACD normalizzato come percentuale del prezzo
    # df_copy['MACD_S'] = df_copy['MACD_S'] / df_copy['Close'] * 100  # normalizzato
    # df_copy['MACD_L'] = df_copy['MACD_L'] / df_copy['Close'] * 100  # normalizzato
    # df_copy['MACD'] = df_copy['MACD'] / df_copy['Close'] * 100  # normalizzato

    # ATR
    atr_indicator = AverageTrueRange(
        high=df_copy['High'],
        low=df_copy['Low'],
        close=df_copy['Close'],
        window=atr_window
    )
    df_copy['ATR'] = atr_indicator.average_true_range()

    # EMA (Media Mobile per le Rolling ATR Bands)
    ema_indicator = EMAIndicator(close=df_copy['Close'], window=ema_window)
    df_copy['EMA'] = ema_indicator.ema_indicator()
    ema_indicator = EMAIndicator(close=df_copy['Close'], window=ema_window2)
    df_copy['EMA2'] = ema_indicator.ema_indicator()
    ema_indicator = EMAIndicator(close=df_copy['Close'], window=ema_window3)
    df_copy['EMA3'] = ema_indicator.ema_indicator()
    emao_indicator = EMAIndicator(close=df_copy['Open'], window=ema_window)
    df_copy['EMAO'] = emao_indicator.ema_indicator()

    kama_indicator = KAMAIndicator(close=df_copy['Close'],
                                   window=ema_window,
                                   pow1=kama_pow1,
                                   pow2=kama_pow2)
    df_copy['KAMA'] = kama_indicator.kama()

    # Rolling ATR Bands
    # df_copy['Upper_Band'] = df_copy['EMA'] + atr_multiplier * df_copy['ATR']
    # df_copy['Lower_Band'] = df_copy['EMA'] - atr_multiplier * df_copy['ATR']
    df_copy['Upper_Band'] = df_copy['KAMA'] + atr_multiplier * df_copy['ATR']
    df_copy['Lower_Band'] = df_copy['KAMA'] - atr_multiplier * df_copy['ATR']
    df_copy['Upper_Band'][:atr_window] = None
    df_copy['Lower_Band'][:atr_window] = None

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

    return df_copy


def calculate_latest_indicators(df: pd.DataFrame, i: int, atr_window: int = 14, atr_multiplier: float = 2.4):
    """
    Calcola SOLO l'ultimo valore di RSI e MACD sulla candela 'i'
    del DataFrame 'df', ritagliando una finestra minima attorno a 'i'.
    """

    # needed_bars = max(atr_window, macd_short_window, macd_long_window, macd_signal_window) + 5
    needed_bars = atr_window + 12
    start_idx = max(0, i - needed_bars + 1)
    end_idx = i + 1  # slice in pandas: end non è incluso
    # Estrai la finestra di dati
    temp_df = df.iloc[start_idx:end_idx].copy()

    # Se la finestra è troppo corta, restituiamo df_copy con tutte None
    if len(temp_df) < atr_window:
        df_copy = df.copy()
        df_copy['ATR'] = None
        df_copy['EMA'] = None
        df_copy['Upper_Band'] = None
        df_copy['Lower_Band'] = None
        df_copy['PSAR'] = None
        return df_copy

    # ATR
    atr_indicator = AverageTrueRange(
        high=temp_df['High'],
        low=temp_df['Low'],
        close=temp_df['Close'],
        window=atr_window
    )
    atr = atr_indicator.average_true_range()

    # EMA (Media Mobile per le Rolling ATR Bands)
    ema_indicator = EMAIndicator(close=temp_df['Close'], window=atr_window)
    ema = ema_indicator.ema_indicator()

    # kama_indicator = KAMAIndicator(close=temp_df['Close'],
    #                                window=atr_window,
    #                                pow1=2,
    #                                pow2=30)
    # kama = kama_indicator.kama()

    # temp_df['Upper_Band'] = ema + atr_multiplier * atr
    # temp_df['Lower_Band'] = ema - atr_multiplier * atr

    temp_df['Upper_Band'] = ema + atr_multiplier * atr
    temp_df['Lower_Band'] = ema - atr_multiplier * atr

    return temp_df


@st.cache_data
def simulate_candles(raw_df, atr_window: int = 6, atr_multiplier: float = 2, step: float = 0.01, max_step: float = 0.4,
                     stop_loss_percent: float = 99.0):
    """
    raw_df: DataFrame con colonne ['Open', 'High', 'Low', 'Close', 'Volume']
    Altre parametri: come nel tuo add_technical_indicator.
    stop_loss: % di stop loss
    strategia: stringa per la strategia
    """

    df = raw_df.copy()

    # Inizializza strutture per salvare i segnali
    buy_signals = []
    sell_signals = []
    holding = False
    last_signal_candle_index = -1  # inizialmente
    # variabili per lo Stop Loss
    stop_loss_price = None
    got_stop_loss = False
    stop_loss_decimal = stop_loss_percent / 100

    # Per evitare di ricalcolare da zero,
    # puoi calcolare UNA SOLA volta gli indicatori "storici" fino alla prima candela.
    # Tuttavia, nella logica semplificata, rifaremo "add_technical_indicator" in ogni step.

    # Loop sulle candele
    for i in range(len(df)):

        o = df['Open'].iloc[i]
        h = df['High'].iloc[i]
        l = df['Low'].iloc[i]
        c = df['Close'].iloc[i]

        n_steps = 10
        is_green = (c >= o)

        # Verifichiamo se è candela verde o rossa
        # (puoi anche decidere con un'altra logica, es: "Close >= Open => verde" di default)
        # Definiamo i 3 segmenti e costruiamo i 30 step di prezzo
        # Candela verde: open -> low (10 step), low -> high (10 step), high -> close (10 step)
        # Candela rossa: open -> high (10 step), high -> low (10 step), low -> close (10 step)
        # Per comodità, uso una piccola funzione di supporto per generare "n" step dal prezzo A al prezzo B
        def linspace_steps(a, b, n=n_steps):
            return np.linspace(a, b, n, endpoint=False)[1:]  # escludiamo la "prima" perché corrisponde a a

        prices_sequence = []
        if is_green:
            # Segmento 1: open -> low
            segment1 = linspace_steps(o, l, n=int(n_steps / 2))
            # Segmento 2: low -> high
            segment2 = linspace_steps(l, h, n=int(n_steps * 2))
            # Segmento 3: high -> close
            segment3 = linspace_steps(h, c, n=int(n_steps / 2))
        else:
            # Segmento 1: open -> high
            segment1 = linspace_steps(o, h, n=int(n_steps / 2))
            # Segmento 2: high -> low
            segment2 = linspace_steps(h, l, n=int(n_steps * 2))
            # Segmento 3: low -> close
            segment3 = linspace_steps(l, c, n=int(n_steps / 2))
        prices_sequence = list(segment1) + list(segment2) + list(segment3)

        # A questo punto abbiamo 3*9 = 27 prezzi intermedi,
        # se vogliamo esattamente 30 step (includendo anche l'ultimo?),
        # possiamo aggiungere l'ultimo prezzo "Close" come step finale,
        # così da totalizzare 28 (oppure gestire diversamente).
        # Per semplicità, qui aggiungo manualmente l'ultimo step = c
        # (ma dipende da come preferisci gestire i conti).
        prices_sequence.append(c)
        # Inizializza i valori "in costruzione" della candela:
        step_open = o
        step_high = o
        step_low = o
        step_close = o

        # Ora eseguiamo la simulazione step-by-step
        for price in prices_sequence:
            # Aggiorniamo SOLO l'ultima candela con un "Close" fittizio = price
            # e lasciamo invariati Open, High, Low "finali" della candela,
            # in modo che eventuali indicatori che usano 'High', 'Low'
            # vedano la candela 'per intero'.

            # Aggiorna i valori di High e Low dinamicamente
            if price > step_high:
                step_high = price
            if price < step_low:
                step_low = price
            # Aggiorna la chiusura
            step_close = price

            temp_df = df.copy()
            # Sovrascrivi sulla candela i-esima i valori dinamici
            temp_df.at[temp_df.index[i], 'Open'] = step_open
            temp_df.at[temp_df.index[i], 'High'] = step_high
            temp_df.at[temp_df.index[i], 'Low'] = step_low
            temp_df.at[temp_df.index[i], 'Close'] = step_close

            df_utile = calculate_latest_indicators(i=i, df=temp_df, atr_window=atr_window,
                                                   atr_multiplier=atr_multiplier)

            row = df_utile.iloc[-1]
            # Condizione di BUY
            if (not holding and last_signal_candle_index != i and
                    row['Lower_Band'] is not None and row['Close'] <= row['Lower_Band'] and
                    not (got_stop_loss and row['PSAR'] is not None and row['PSAR'] > row['Close'])):
                buy_signals.append((df.index[i], float(row['Close'])))
                holding = True
                last_signal_candle_index = i
                got_stop_loss = False
                stop_loss_price = float(row['Close']) * (1 - stop_loss_decimal)
            # Condizione di SELL
            if (holding and last_signal_candle_index != i and
                    row['Upper_Band'] is not None and row['Close'] >= row['Upper_Band']):
                sell_signals.append((df.index[i], float(row['Close'])))
                holding = False
                last_signal_candle_index = i
                stop_loss_price = None
                got_stop_loss = False
            # Condizione STOP LOSS
            if (holding and stop_loss_price is not None and row['Close'] < stop_loss_price and
                    row['PSAR'] > row['Close']):
                sell_signals.append((df.index[i], float(row['Close'])))
                holding = False
                last_signal_candle_index = i
                got_stop_loss = True
                stop_loss_price = None

    return buy_signals, sell_signals


def buy_sell_limits_simulation(df, macd_buy_limit, macd_sell_limit, rsi_buy_limit, rsi_sell_limit,
                               num_cond):
    buy_signals = []
    sell_signals = []
    holding = False

    for i in range(1, len(df)):
        # CONDIZIONI DI BUY
        cond_buy_macd = 1 if df['MACD'].iloc[i] <= macd_buy_limit else 0
        # cond_buy_macd2 = 1 if df['MACD'].iloc[i] > df['MACD'].tail(
        #     10).min() else 0  # il MACD ha invertito direzione
        cond_buy_rsi = 1 if df['RSI'].iloc[i] <= rsi_buy_limit else 0
        # cond_buy_vi = 1 if df['VI'].iloc[i] <= vi_buy_limit else 0
        # cond_buy_psarvp = 1 if df['PSARVP'].iloc[i] >= psarvp_buy_limit else 0
        # cond_buy_atr = 1 if df['Low'].iloc[i] <= df['Lower_Band'].iloc[i] else 0
        # cond_buy_srsi = 1 if df['StochRSI'].iloc[i] <= srsi_buy_limit else 0
        # cond_buy_tsi = 1 if df['TSI'].iloc[i] <= tsi_buy_limit else 0
        # cond_buy_roc = 1 if df['ROC'].iloc[i] <= roc_buy_limit else 0
        # cond_buy_pvo = 1 if df['PVO'].iloc[i] <= pvo_buy_limit else 0
        # cond_buy_mfi = 1 if df['MFI'].iloc[i] <= mfi_buy_limit else 0
        sum_buy = cond_buy_macd + cond_buy_rsi
        # + cond_buy_vi + cond_buy_psarvp + cond_buy_atr + cond_buy_srsi +
        # cond_buy_tsi + cond_buy_roc + cond_buy_pvo + cond_buy_mfi)
        if not holding and sum_buy >= num_cond:
            if df['Low'].iloc[i] < df['Lower_Band'].iloc[i]:
                buy_signals.append((df.index[i], float(df['Lower_Band'].iloc[i])))
            else:
                buy_signals.append((df.index[i], float(df['Close'].iloc[i])))
            holding = True
        # CONDIZIONI DI SELL
        cond_sell_macd = 1 if df['MACD'].iloc[i] >= macd_sell_limit else 0
        # cond_sell_macd2 = 1 if df['MACD'].iloc[i] < df['MACD'].tail(
        #    10).max() else 0  # il MACD ha invertito direzione
        cond_sell_rsi = 1 if df['RSI'].iloc[i] >= rsi_sell_limit else 0
        # cond_sell_vi = 1 if df['VI'].iloc[i] >= vi_sell_limit else 0
        # cond_sell_psavp = 1 if df['PSARVP'].iloc[i] <= psarvp_sell_limit else 0
        # cond_sell_atr = 1 if df['High'].iloc[i] >= df['Upper_Band'].iloc[i] else 0
        # cond_sell_srsi = 1 if df['StochRSI'].iloc[i] >= srsi_sell_limit else 0
        # cond_sell_tsi = 1 if df['TSI'].iloc[i] >= tsi_sell_limit else 0
        # cond_sell_roc = 1 if df['ROC'].iloc[i] >= roc_sell_limit else 0
        # cond_sell_pvo = 1 if df['PVO'].iloc[i] >= pvo_sell_limit else 0
        # cond_sell_mfi = 1 if df['MFI'].iloc[i] >= mfi_sell_limit else 0
        sum_sell = cond_sell_macd + cond_sell_rsi
        # + cond_sell_vi + cond_sell_psavp + cond_sell_atr +
        # cond_sell_srsi + cond_sell_tsi + cond_sell_roc + cond_sell_pvo + cond_sell_mfi)
        if holding and sum_sell >= num_cond:
            if df['High'].iloc[i] > df['Upper_Band'].iloc[i]:
                sell_signals.append((df.index[i], float(df['Upper_Band'].iloc[i])))
            else:
                sell_signals.append((df.index[i], float(df['Close'].iloc[i])))
            holding = False

    return buy_signals, sell_signals


def buy_sell_limits_close_simulation(df, rsi_buy_limit: int = 25, rsi_sell_limit: int = 75,
                                     macd_buy_limit: float = -2.5, macd_sell_limit: float = 2.5,
                                     num_cond: int = 1, stop_loss_percent: float = 99.0):
    buy_signals = []
    sell_signals = []
    holding = False
    last_signal_candle_index = -1
    # stop_loss_price = None
    # got_stop_loss = False
    # stop_loss_decimal = stop_loss_percent / 100

    for i in range(1, len(df)):
        # CONDIZIONI DI BUY
        cond_buy_atr = 1 if df['Close'].iloc[i] <= df['Lower_Band'].iloc[i] else 0
        cond_buy_rsi = 1 if df['RSI'].iloc[i] <= rsi_buy_limit else 0
        sum_buy = cond_buy_rsi + cond_buy_atr
        if not holding and last_signal_candle_index != i and sum_buy >= num_cond:
            buy_signals.append((df.index[i], float(df['Close'].iloc[i])))
            holding = True
            last_signal_candle_index = i
        # CONDIZIONI DI SELL
        cond_sell_rsi = 1 if df['RSI'].iloc[i] >= rsi_sell_limit else 0
        cond_sell_atr = 1 if df['Close'].iloc[i] >= df['Upper_Band'].iloc[i] else 0
        sum_sell = cond_sell_rsi + cond_sell_atr
        if holding and last_signal_candle_index != i and sum_sell >= num_cond:
            sell_signals.append((df.index[i], float(df['Close'].iloc[i])))
            holding = False
            last_signal_candle_index = i

    return buy_signals, sell_signals


def close_rsi_buy_sell_limits_simulation(df):
    buy_signals = []
    sell_signals = []
    holding = False
    last_signal_candle_index = -1

    for i in range(1, len(df)):
        # CONDIZIONI DI BUY
        # if (not holding and last_signal_candle_index != i and
        #         df['RSI'].iloc[i - 1] > df['RSI2'].iloc[i - 1] and
        #         df['RSI'].iloc[i] < df['RSI2'].iloc[i]):
        if (not holding and last_signal_candle_index != i and
                df['RSI'].iloc[i - 1] < df['RSI2'].iloc[i - 1] and
                df['RSI'].iloc[i] > df['RSI2'].iloc[i]):
            buy_signals.append((df.index[i], float(df['Close'].iloc[i])))
            holding = True
            last_signal_candle_index = i
        # CONDIZIONI DI SELL
        # if (holding and last_signal_candle_index != i and
        #         df['RSI'].iloc[i - 1] < df['RSI2'].iloc[i - 1] and
        #         df['RSI'].iloc[i] > df['RSI2'].iloc[i]):
        if (holding and last_signal_candle_index != i and
                df['RSI'].iloc[i - 1] > df['RSI2'].iloc[i - 1] and
                df['RSI'].iloc[i] < df['RSI2'].iloc[i]):
            sell_signals.append((df.index[i], float(df['Close'].iloc[i])))
            holding = False
            last_signal_candle_index = i

    return buy_signals, sell_signals


def atr_buy_sell_simulation(df, stop_loss_percent):
    # Identificazione dei segnali di acquisto e vendita
    buy_signals = []
    sell_signals = []
    holding = False
    last_signal_candle_index = -1
    stop_loss_price = None
    got_stop_loss = False
    stop_loss_decimal = stop_loss_percent / 100

    for i in range(1, len(df)):
        if (not holding and last_signal_candle_index != i and df['Low'].iloc[i] <= df['Lower_Band'].iloc[i]
                and not (got_stop_loss and df['PSAR'].iloc[i] > df['Close'].iloc[i])):
            buy_signals.append((df.index[i], float(df['Lower_Band'].iloc[i])))
            holding = True
            last_signal_candle_index = i
            got_stop_loss = False
            stop_loss_price = df['Lower_Band'].iloc[i] * (1 - stop_loss_decimal)
        if holding and last_signal_candle_index != i and df['High'].iloc[i] >= df['Upper_Band'].iloc[i]:
            sell_signals.append((df.index[i], float(df['Upper_Band'].iloc[i])))
            holding = False
            last_signal_candle_index = i
            stop_loss_price = None
            got_stop_loss = False
        if (holding and stop_loss_price is not None and df['Low'].iloc[i] < stop_loss_price and
                df['PSAR'].iloc[i] > df['Close'].iloc[i]):
            # devo vendere per STOP LOSS
            sell_signals.append((df.index[i], stop_loss_price))
            holding = False
            last_signal_candle_index = i
            got_stop_loss = True
            stop_loss_price = None

    return buy_signals, sell_signals


def close_atr_buy_sell_simulation(df, stop_loss_percent):
    # Identificazione dei segnali di acquisto e vendita
    buy_signals = []
    sell_signals = []
    holding = False
    last_signal_candle_index = -1
    stop_loss_price = None
    got_stop_loss = False
    stop_loss_decimal = stop_loss_percent / 100

    for i in range(1, len(df)):
        if (not holding and last_signal_candle_index != i and df['Close'].iloc[i] <= df['Lower_Band'].iloc[i]
                and not (got_stop_loss and df['PSAR'].iloc[i] > df['Close'].iloc[i])):
            buy_signals.append((df.index[i], float(df['Close'].iloc[i])))
            holding = True
            last_signal_candle_index = i
            got_stop_loss = False
            stop_loss_price = float(df['Close'].iloc[i]) * (1 - stop_loss_decimal)
        if holding and last_signal_candle_index != i and df['Close'].iloc[i] >= df['Upper_Band'].iloc[i]:
            sell_signals.append((df.index[i], float(df['Close'].iloc[i])))
            holding = False
            last_signal_candle_index = i
            stop_loss_price = None
            got_stop_loss = False
        if (holding and stop_loss_price is not None and df['Close'].iloc[i] < stop_loss_price and
                df['PSAR'].iloc[i] > df['Close'].iloc[i]):
            # devo vendere per STOP LOSS
            sell_signals.append((df.index[i], float(df['Close'].iloc[i])))
            holding = False
            last_signal_candle_index = i
            got_stop_loss = True
            stop_loss_price = None

    return buy_signals, sell_signals


def close_ema_crossover_simulation(df):
    buy_signals = []
    sell_signals = []
    holding = False
    first_break = False
    second_break = False
    for i in range(1, len(df)):
        ema50ema100up = (df["EMA"].iloc[i - 1] <= df["EMA2"].iloc[i - 1] and df["EMA"].iloc[i] > df["EMA2"].iloc[i])
        ema50ema200up = (df["EMA"].iloc[i - 1] <= df["EMA3"].iloc[i - 1] and df["EMA"].iloc[i] > df["EMA3"].iloc[i])
        ema100ema200up = (df["EMA2"].iloc[i - 1] <= df["EMA3"].iloc[i - 1] and df["EMA2"].iloc[i] > df["EMA3"].iloc[i])

        ema50ema100down = (df["EMA"].iloc[i - 1] >= df["EMA2"].iloc[i - 1] and df["EMA"].iloc[i] < df["EMA2"].iloc[i])
        ema50ema200down = (df["EMA"].iloc[i - 1] >= df["EMA3"].iloc[i - 1] and df["EMA"].iloc[i] < df["EMA3"].iloc[i])
        ema100ema200down = (
                df["EMA2"].iloc[i - 1] >= df["EMA3"].iloc[i - 1] and df["EMA2"].iloc[i] < df["EMA3"].iloc[i])

        if not holding:
            # non si verifica le sequenza esatta
            if first_break and (ema100ema200up or ema50ema100down or ema50ema200down or ema100ema200down):
                first_break = False
            if second_break and (ema50ema100up or ema50ema100down or ema50ema200down or ema100ema200down):
                second_break = False
            # controllo la sequenza esatta
            if ema50ema100up:
                first_break = True
            if first_break and ema50ema200up:
                second_break = True
            if second_break and ema100ema200up:
                first_break = False
                second_break = False
                buy_signals.append((df.index[i], float(df['Close'].iloc[i])))
                holding = True

        if holding:
            # non si verifica le sequenza esatta
            if first_break and (ema100ema200down or ema50ema100up or ema50ema200up or ema100ema200up):
                first_break = False
            if second_break and (ema50ema100down or ema50ema100up or ema50ema200up or ema100ema200up):
                second_break = False
            # controllo la sequenza esatta
            if ema50ema100down:
                first_break = True
            if first_break and ema50ema200down:
                second_break = True
            if second_break and ema100ema200down:
                first_break = False
                second_break = False
                sell_signals.append((df.index[i], float(df['Close'].iloc[i])))
                holding = False

    return buy_signals, sell_signals


def close_bullish_ema_simulation(df, rsi_buy_limit: int = 50, rsi_sell_limit: int = 70):
    buy_signals = []
    sell_signals = []
    holding = False
    n = 30
    for i in range(1, len(df)):
        cond_1 = df['EMA'][i - n:i] > df['EMA2'][i - n:i]
        cond_2 = df['EMA2'][i - n:i] > df['EMA3'][i - n:i]
        cond_ema = (cond_1 & cond_2).all()
        if (not holding and cond_ema and (
                df['EMA'].iloc[i] > df['EMA2'].iloc[i] > df['EMA3'].iloc[i])  # trend rialzista nel breve termine
                # and df['ADX'].iloc[i] > 30  # conferma della forza del trend
                # and df['EMA2'].iloc[i] < df['Upper_Band3'].iloc[i]  # il prezzo oscilla attorno alla media lunga
                and df['Close'].iloc[i] > df['EMA3'].iloc[i]  # il prezzo sta sopra alla media lunga
                and rsi_buy_limit <= df['RSI'].iloc[
                    i] < rsi_sell_limit  # RSI compreso in una fascia che conferma il trend
                # controlli sulle candele precedenti
                and ((df['Low'].iloc[i - 1] < df['EMA2'].iloc[i - 1] < df['Close'].iloc[i - 1]) or
                     (df['Low'].iloc[i - 1] < df['EMA3'].iloc[i - 1] < df['Close'].iloc[i - 1]))
                and ((df['EMA2'].iloc[i - 2] < df['Low'].iloc[i - 2] < df['Close'].iloc[i - 2]) or
                     (df['EMA3'].iloc[i - 2] < df['Low'].iloc[i - 2] < df['Close'].iloc[i - 2]))):
            buy_signals.append((df.index[i], float(df['Close'].iloc[i])))
            holding = True
        if holding and df['RSI'].iloc[i] > rsi_sell_limit:
            sell_signals.append((df.index[i], float(df['Close'].iloc[i])))
            holding = False

    return buy_signals, sell_signals


def tp_sl_simulation(df):
    # Identificazione dei segnali di acquisto e vendita
    buy_signals = []
    sell_signals = []
    holding = False
    last_signal_candle_index = -1
    take_profit_price = None
    stop_loss_price = None

    for i in range(1, len(df)):
        if (not holding and last_signal_candle_index != i and
                df['Close'].iloc[i] >= df['Upper_Band'].iloc[i]):
            buy_signals.append((df.index[i], float(df['Close'].iloc[i])))
            holding = True
            last_signal_candle_index = i
            stop_loss_price = float(df['Lower_Band'].iloc[i])
            take_profit_price =  df['Close'].iloc[i] + (df['Close'].iloc[i] - stop_loss_price)
        if (holding and take_profit_price is not None and last_signal_candle_index != i and
                df['High'].iloc[i] >= take_profit_price):
            # vengo per TAKE PROFIT
            sell_signals.append((df.index[i], take_profit_price))
            holding = False
            last_signal_candle_index = i
            stop_loss_price = None
            take_profit_price = None
        if (holding and stop_loss_price is not None and last_signal_candle_index != i and
                df['Low'].iloc[i] <= stop_loss_price):
            # devo vendere per STOP LOSS
            sell_signals.append((df.index[i], stop_loss_price))
            holding = False
            last_signal_candle_index = i
            stop_loss_price = None
            take_profit_price = None

    return buy_signals, sell_signals


def green_candles_simulation(df):
    buy_signals = []
    sell_signals = []
    holding = False

    for i in range(1, len(df)):
        if (not holding and df['Close'].iloc[i-1] < df['Open'].iloc[i-1] and
                df['Close'].iloc[i] > df['High'].iloc[i-1]):
            buy_signals.append((df.index[i], float(df['Close'].iloc[i])))
            holding = True
        if (holding and df['Close'].iloc[i-1] > df['Open'].iloc[i-1] and
                df['Close'].iloc[i] < df['Low'].iloc[i-1]):
            sell_signals.append((df.index[i],  float(df['Close'].iloc[i])))
            holding = False

    return buy_signals, sell_signals

def simulate_trading_with_commisions(buy_signals: list, sell_signals: list, wallet: float = 100,
                                     fee_percent: float = 0.1):
    operations = []
    holding = False  # Flag che indica se stiamo detenendo l'asset
    quantity = 0.0  # Quantità dell'asset comprata
    working_wallet = wallet  # Capitale di partenza (USDT/USDC)
    # Converto fee_percent in forma decimale (es. 1% -> 0.01)
    fee_decimal = fee_percent / 100.0
    # Per semplicità, assumiamo che numero di buy_signals e sell_signals
    # siano (in media) abbinati, usando lo stesso indice i in parallelo.
    for i in range(len(buy_signals)):
        # Se NON stiamo detenendo nulla e c'è un segnale di BUY, compriamo
        if not holding and i < len(buy_signals):
            buy_time, buy_price = buy_signals[i]
            if working_wallet > 0:
                # Paghiamo la commissione in USDT/USDC: se abbiamo working_wallet,
                # dopo la fee rimane working_wallet*(1 - fee_decimal) per comprare
                net_invested = working_wallet * (1 - fee_decimal)
                # quantità di crypto ottenuta
                quantity = net_invested / buy_price
                # Ora working_wallet = 0 (tutto investito)
                working_wallet = 0.0
                holding = True
        # Se ABBIAMO una posizione aperta e c'è un segnale di SELL, vendiamo
        if holding and i < len(sell_signals):
            sell_time, sell_price = sell_signals[i]
            # Ricaviamo USDT vendendo la quantity di crypto
            gross_proceed = quantity * sell_price
            # Applichiamo la commissione di vendita
            # commissions = gross_proceed * fee_decimal
            net_proceed = gross_proceed * (1 - fee_decimal)
            # Calcoliamo il profit: differenza fra l'importo netto incassato e l'importo speso in fase di BUY e le commissioni
            cost_in_usd = (quantity * buy_price) * (1 + fee_decimal)  # spesa inziale
            profit = net_proceed - cost_in_usd
            # Aggiorniamo working_wallet
            working_wallet = net_proceed
            # Registriamo il trade in un'unica riga
            operations.append({
                'Buy_Time': buy_time,
                'Buy_Price': buy_price,
                'Sell_Time': sell_time,
                'Sell_Price': sell_price,
                'Quantity': quantity,
                'Profit': profit,
                'Wallet_After': working_wallet
            })
            # Resettiamo lo stato
            holding = False
            quantity = 0.0
    return operations


def simulate_trading_with_commisions_multiple_buy(buy_signals: list, sell_signals: list, wallet: float = 100,
                                                  fee_percent: float = 0.1):
    operations = []
    holding = False  # Flag che indica se stiamo detenendo l'asset
    quantity = 0.0  # Quantità dell'asset comprata
    total_buy = 0
    working_wallet = wallet  # Capitale di partenza (USDT/USDC)
    # Converto fee_percent in forma decimale (es. 1% -> 0.01)
    fee_decimal = fee_percent / 100.0
    buy_percentage = 0.6
    # Per semplicità, assumiamo che numero di buy_signals e sell_signals
    # siano (in media) abbinati, usando lo stesso indice i in parallelo.
    s = 0
    b = 0
    while b < len(buy_signals):
        # for b in range(len(buy_signals)):
        # if i < len(buy_signals):
        buy_time, buy_price = buy_signals[b]
        # if working_wallet > 0:
        # Paghiamo la commissione in USDT/USDC: se abbiamo working_wallet,
        # utilizzo metà del working wallet
        total_buy = working_wallet * buy_percentage
        net_invested = (working_wallet * buy_percentage) * (1 - fee_decimal)
        # quantità di crypto ottenuta
        quantity = net_invested / buy_price
        working_wallet -= (working_wallet * buy_percentage)
        holding = True
        next = b + 1
        while next < len(buy_signals):
            if s < len(sell_signals):
                buy_time, buy_price = buy_signals[next]
                sell_time, sell_price = sell_signals[s]
                if buy_time < sell_time:
                    # Paghiamo la commissione in USDT/USDC: se abbiamo working_wallet,
                    # utilizzo metà del working wallet
                    total_buy += (working_wallet * buy_percentage)
                    net_invested = (working_wallet * buy_percentage) * (1 - fee_decimal)
                    # quantità di crypto ottenuta
                    quantity += net_invested / buy_price
                    working_wallet -= (working_wallet * buy_percentage)
                    b = next
                else:
                    break
            else:
                break
            next += 1
        b += 1

        if holding and s < len(sell_signals):
            sell_time, sell_price = sell_signals[s]
            # mean_cost = total_buy / quantity
            # Ricaviamo USDT vendendo la quantity di crypto
            gross_proceed = quantity * sell_price
            # Applichiamo la commissione di vendita
            # commissions = gross_proceed * fee_decimal
            net_proceed = gross_proceed * (1 - fee_decimal)
            # Calcoliamo il profit: differenza fra l'importo netto incassato e l'importo speso in fase di BUY
            # cost_in_usd = (quantity * buy_price) * (1 + fee_decimal)  # spesa inziale
            # profit = net_proceed - cost_in_usd
            profit = net_proceed - total_buy
            # Aggiorniamo working_wallet
            working_wallet += net_proceed
            # Registriamo il trade in un'unica riga
            operations.append({
                'Buy_Time': buy_time,
                'Buy_Price': buy_price,
                'Sell_Time': sell_time,
                'Sell_Price': sell_price,
                'Quantity': quantity,
                'Profit': profit,
                'Wallet_After': working_wallet
            })
            # Resettiamo lo stato
            holding = False
            quantity = 0.0
            s += 1

    return operations


def bullish_condition(df,i) -> bool:
    # cond_bullish = (df['EMA'].iloc[i] > df['EMA2'].iloc[i] > df['EMA3'].iloc[i] and
    #                 df['RSI'].iloc[i] > df['RSI2'].iloc[i] > df['RSI3'].iloc[i] and
    #                 df['STOCH'].iloc[i] > df['STOCH_S'].iloc[i])

    # cond_bullish = df['Close'].iloc[i] >= df['Upper_Band'].iloc[i]
    cond_bullish = df['EMA'].iloc[i] >= df['EMAO'].iloc[i]

    return cond_bullish


def bearish_condition (df, i) -> bool:
    # cond_bearish = (df['EMA'].iloc[i] < df['EMA2'].iloc[i] < df['EMA3'].iloc[i] and
    #                 df['RSI'].iloc[i] < df['RSI2'].iloc[i] < df['RSI3'].iloc[i] and
    #                 df['STOCH'].iloc[i] < df['STOCH_S'].iloc[i])
    #
    # cond_bearish = df['Close'].iloc[i] <= df['Lower_Band'].iloc[i]
    cond_bearish = df['EMA'].iloc[i] < df['EMAO'].iloc[i]

    return cond_bearish


def identify_trend_zones(df: pd.DataFrame) -> list:
    """
    Identifica intervalli di trend rialzista e ribassista basandosi su parametri di input,
    e restituisce una lista di dizionari "shapes" Plotly.

    Parametri
    ---------
    df : pd.DataFrame
        Il DataFrame con le colonne richieste (RSI, EMA, etc.)

    Ritorna
    -------
    list
        Lista di shapes (rettangoli) da passare successivamente a fig.update_layout(shapes=...).
        Ogni shape è un dict con parametri per disegnare la zona colorata su Plotly.
    """

    shapes = []
    current_trend = None  # Possibili valori: 'bullish', 'bearish', oppure None
    zone_start_index = None
    # cond_bullish = False
    # cond_bearish = False

    for i in range(len(df)):
        # Valuta le condizioni per bullish e bearish

        cond_bullish = bullish_condition(df,i)
        cond_bearish = bearish_condition(df,i)

        # if current_trend != "bearish" and cond_bearish:
        #     new_trend = "bearish"
        # elif current_trend != "bullish" and cond_bullish:
        #     new_trend = "bullish"
        # else:
        #     new_trend = None

        if cond_bullish:
            new_trend = "bullish"
        elif cond_bearish:
            new_trend = "bearish"
        else:
            new_trend = None

        if new_trend is not None and new_trend != current_trend:
            # Se stiamo cambiando "trend" rispetto al bar/candela precedente,
            # chiudiamo eventuale zona precedente e ne apriamo un'altra se necessario.
            if current_trend is not None and zone_start_index is not None:
                # Costruisci shape per la zona precedente [zone_start_index ... i-1]
                x0 = df.index[zone_start_index]
                # i potrebbe essersi spostato di 1 troppo in avanti, usiamo df.index[i-1] se i>0
                x1 = df.index[i - 1] if i > 0 else x0

                if current_trend == "bullish":
                    fillcolor = "green"
                    opacity_val = 0.15
                else:
                    fillcolor = "red"
                    opacity_val = 0.15

                shapes.append(
                    dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=x0,
                        x1=x1,
                        y0=0,  # intera altezza del grafico
                        y1=1,
                        fillcolor=fillcolor,
                        opacity=opacity_val,
                        layer="below",  # la zona resta sotto le candele
                        line_width=0  # niente contorno
                    )
                )

            # Apertura della nuova zona, se c'è un nuovo trend
            if new_trend is not None:
                zone_start_index = i
            else:
                zone_start_index = None

            current_trend = new_trend

    # Chiudi eventuale ultima zona rimasta aperta
    if current_trend is not None and zone_start_index is not None:
        x0 = df.index[zone_start_index]
        x1 = df.index[-1]  # fino alla fine
        if current_trend == "bullish":
            fillcolor = "green"
            opacity_val = 0.15
        else:
            fillcolor = "red"
            opacity_val = 0.15

        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=x0,
                x1=x1,
                y0=0,
                y1=1,
                fillcolor=fillcolor,
                opacity=opacity_val,
                layer="below",
                line_width=0
            )
        )

    return shapes


def supertrend_simulation(df):
    buy_signals = []
    sell_signals = []
    holding = False

    current_trend = None  # Possibili valori: 'bullish', 'bearish', oppure None

    take_profit_price = None
    stop_loss_price = None

    for i in range(1, len(df)):
        # cond_bullish = bullish_condition(df, i)
        cond_bullish = df['Close'].iloc[i] >= df['Upper_Band'].iloc[i]
        # cond_bearish = bearish_condition(df, i)
        cond_bearish = df['Close'].iloc[i] <= df['Lower_Band'].iloc[i]
        if cond_bullish:
            new_trend = "bullish"
        elif cond_bearish:
            new_trend = "bearish"
        else:
            new_trend = None

        if new_trend is not None and new_trend != current_trend:
            if not holding and new_trend == 'bullish':
                buy_signals.append((df.index[i], float(df['Close'].iloc[i])))
                holding = True
                stop_loss_price = df['Lower_Band'].iloc[i]
                take_profit_price = df['Close'].iloc[i] + (df['Close'].iloc[i] - stop_loss_price)*1.618

            if holding and new_trend == 'bearish':
                sell_signals.append((df.index[i], float(df['Close'].iloc[i])))
                holding = False

        if holding and take_profit_price is not None and df['High'].iloc[i] >= take_profit_price:
            # vengo per TAKE PROFIT
            sell_signals.append((df.index[i], take_profit_price))
            holding = False
            stop_loss_price = None
            take_profit_price = None

        if holding and stop_loss_price is not None and df['Low'].iloc[i] <= stop_loss_price:
            # devo vendere per STOP LOSS
            sell_signals.append((df.index[i], stop_loss_price))
            holding = False
            stop_loss_price = None
            take_profit_price = None

            current_trend = new_trend

        # if not bearish_condition(df, i) and bearish_condition(df, i - 1):
        #     buy_signals.append((df.index[i], float(df['Close'].iloc[i])))
        #     holding = True
        #
        # if holding and not bullish_condition(df, i) and bullish_condition(df, i - 1):
        #     sell_signals.append((df.index[i], float(df['Close'].iloc[i])))
        #     holding = False

    return buy_signals, sell_signals

def trend_zone_simulation(df):
    buy_signals = []
    sell_signals = []
    holding = False
    for i in range(1, len(df)):
        if not holding and df['EMA'].iloc[i] > df['EMAO'].iloc[i]:
            buy_signals.append((df.index[i], float(df['Close'].iloc[i])))
            holding = True

        if holding and df['EMA'].iloc[i] <= df['EMAO'].iloc[i]:
            sell_signals.append((df.index[i], float(df['Close'].iloc[i])))
            holding = False

    return buy_signals, sell_signals

def trading_analysis(
        asset: str, interval: str, wallet: float, time_hours: int = 24,
        fee_percent: float = 0.1,  # Commissione % per ogni operazione (buy e sell)
        show: bool = True,
        step: float = 0.01, max_step: float = 0.4,
        atr_multiplier: float = 1.5, atr_window: int = 12, window_pivot: int = 10,
        rsi_window: int = 10, rsi_window2: int = 20, rsi_window3: int = 30,
        ema_window: int = 12, ema_window2: int = 24, ema_window3: int = 36,
        macd_short_window: int = 12, macd_long_window: int = 26, macd_signal_window: int = 9,
        kama_pow1:int = 2, kama_pow2:int = 30,
        rsi_buy_limit: int = 40, rsi_sell_limit: int = 60,
        macd_buy_limit: float = -0.4, macd_sell_limit: float = 0.4,
        num_cond: int = 1, stop_loss: int = 99,
        strategia: str = "", market_data: dict = None,
        # din_macd_div: float = 1.2, modello = None
):
    """
    Scarica le candele di 'asset' con intervallo 'interval' (tramite una funzione
    esterna get_market_data), calcola il SAR con i parametri 'step' e 'max_step',
    identifica segnali di acquisto/vendita, simula le operazioni in base al 'wallet'
    iniziale e restituisce un grafico Plotly con candlestick, SAR e segnali,
    oltre al DataFrame con tutte le operazioni, decurtando una commissione
    su ogni BUY e SELL (fee_percent).

    Parameters
    ----------
    asset : str
        Nome dell'asset (es. "BTCUSDT").
    interval : str
        Intervallo di tempo delle candele (es. "1h", "15m", ecc.).
    wallet : float
        Quantità di USDC/USDT a disposizione per le operazioni di trading.
    step : float
        Passo (step) per il calcolo del SAR (param. 'step' in PSARIndicator).
    max_step : float
        Valore massimo di step (param. 'max_step' in PSARIndicator).
    time_hours: int, optional
        tempo in ore che si vuole scaricare
    fee_percent : float, optional
        Percentuale di commissione per operazione (default 1.0, cioè 1%).

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Il grafico con candlestick, SAR e segnali di acquisto/vendita.
    trades_df : pandas.DataFrame
        Un DataFrame con tutte le operazioni effettuate, incluse informazioni
        su buy_time, sell_time, profit, volatilità del periodo, ecc.
    """

    # ======================================
    # Scarica i dati di mercato e calcola il SAR
    if market_data is None:
        # Otteniamo i dati di mercato (funzione esterna da definire)
        # df = get_market_data(asset=asset, interval=interval, limit=limit)
        df, actual_hours = get_market_data(asset=asset, interval=interval, time_hours=time_hours)
    else:
        df = market_data
        actual_hours = time_hours

    # Aggiungiamo una colonna per i massimi e i minimi relativi
    # # Utilizziamo i prezzi massimi ('High') e minimi ('Low')
    # price_high = df['High']
    # price_low = df['Low']
    # # Trova gli indici dei massimi e minimi relativi
    # order = int(window_pivot / 2)
    # max_idx = argrelextrema(price_high.values, np.greater, order=order)[0]
    # min_idx = argrelextrema(price_low.values, np.less, order=order)[0]
    # # Inizializza gli array per massimi e minimi
    # rel_max = []
    # rel_min = []
    # # Popola gli array con tuple (indice, prezzo)
    # for i in min_idx:
    #     rel_min.append((df.index[i], df.loc[df.index[i], 'Low']))
    # for i in max_idx:
    #     rel_max.append((df.index[i], df.loc[df.index[i], 'High']))

    # dinamic_atr = False
    # if strategia == "Dinamic ATR Bands" or strategia == "Dinamic Close ATR":
    #     dinamic_atr = True

    df = add_technical_indicator(df, step=step, max_step=max_step,
                                 rsi_window=rsi_window, rsi_window2=rsi_window2, rsi_window3=rsi_window3,
                                 ema_window=ema_window, ema_window2=ema_window2, ema_window3=ema_window3,
                                 macd_long_window=macd_long_window, macd_short_window=macd_short_window,
                                 macd_signal_window=macd_signal_window,
                                 atr_window=atr_window, atr_multiplier=atr_multiplier,
                                 kama_pow1=kama_pow1, kama_pow2=kama_pow2
                                 )

    # ======================================
    # Identificazione dei segnali di acquisto e vendita in base alla strategia
    buy_signals = []
    sell_signals = []

    if strategia == "ATR Bands" or strategia == "Dinamic ATR Bands":
        buy_signals, sell_signals = atr_buy_sell_simulation(df=df, stop_loss_percent=stop_loss)

    if strategia == "Close ATR" or strategia == "Dinamic Close ATR":
        buy_signals, sell_signals = close_atr_buy_sell_simulation(df=df, stop_loss_percent=stop_loss)

    if strategia == "Buy/Sell Limits":
        buy_signals, sell_signals = (
            buy_sell_limits_simulation(df=df,
                                       macd_buy_limit=macd_buy_limit, macd_sell_limit=macd_sell_limit,
                                       rsi_buy_limit=rsi_buy_limit, rsi_sell_limit=rsi_sell_limit,
                                       num_cond=num_cond))

    if strategia == "Close Buy/Sell Limits":
        buy_signals, sell_signals = (
            buy_sell_limits_close_simulation(df=df,
                                             macd_buy_limit=macd_buy_limit, macd_sell_limit=macd_sell_limit,
                                             rsi_buy_limit=rsi_buy_limit, rsi_sell_limit=rsi_sell_limit,
                                             num_cond=num_cond, stop_loss_percent=stop_loss))

    if strategia == "ATR Live Trade":
        buy_signals, sell_signals = simulate_candles(raw_df=df, atr_window=atr_window, atr_multiplier=atr_multiplier,
                                                     step=step, max_step=max_step, stop_loss_percent=stop_loss)

    if strategia == "Close EMA Crossover":
        buy_signals, sell_signals = close_ema_crossover_simulation(df=df)

    if strategia == "Close Bullish EMA":
        buy_signals, sell_signals = close_bullish_ema_simulation(df=df, rsi_buy_limit=rsi_buy_limit,
                                                                 rsi_sell_limit=rsi_sell_limit)

    if strategia == "Close RSI Reverse":
        buy_signals, sell_signals = close_rsi_buy_sell_limits_simulation(df=df)

    if strategia == "Supertrend":
        buy_signals, sell_signals = supertrend_simulation(df=df)

    if strategia == "Trend Zones":
        buy_signals, sell_signals = trend_zone_simulation(df=df)

    if strategia == "TP/SL with ATR":
        buy_signals, sell_signals = tp_sl_simulation(df=df)

    if strategia == "Green Candles":
        buy_signals, sell_signals = green_candles_simulation(df=df)

    # ======================================
    # Simulazione di trading con commissioni
    if strategia == "Close MACD Retest": #or  strategia == "Trend Zones"
        operations = simulate_trading_with_commisions_multiple_buy(wallet=wallet, buy_signals=buy_signals,
                                                                   sell_signals=sell_signals, fee_percent=fee_percent)
    else:
        operations = simulate_trading_with_commisions(wallet=wallet, buy_signals=buy_signals, sell_signals=sell_signals,
                                                      fee_percent=fee_percent)




    # ======================================
    # 4. Creazione del grafico
    rows = 2
    candlestick_height_px = 400
    indicators_height_px = candlestick_height_px / 2
    total_height = candlestick_height_px + ((rows - 1) * indicators_height_px)
    nominal_height = 1 / (rows + 1)
    candle_height = 2 * nominal_height
    row_heights = [candle_height] + [nominal_height] * (rows - 1)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=row_heights,
                        subplot_titles=("Candlestick",
                                        "True and Relative Strength Index and Stochastic (TSI / RSI / STOCH)",)
                        )
    if show:
        trend_shapes = identify_trend_zones(df=df)
        index = 1
        # Candele (candlestick)
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name=f"{asset}"
        ), row=index, col=1)

        # Punti SAR (marker rossi)
        # fig.add_trace(go.Scatter(
        #     x=df.index,
        #     y=df['PSAR'],
        #     mode='markers',
        #     marker=dict(size=2, color='yellow', symbol='circle'),
        #     name='PSAR'
        # ),
        #     row=index, col=1
        # )

        #EMA SHORT
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA'], mode='lines',
            line=dict(color='Green', width=1),
            name='EMA SHORT'
        ), row=index, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA2'], mode='lines',
            line=dict(color='purple', width=1),
            name='EMA MED'
        ), row=index, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA3'], mode='lines',
            line=dict(color='coral', width=1),
            name='EMA LONG'
        ), row=index, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMAO'], mode='lines',
            line=dict(color='Red', width=1),
            name='EMA OPEN'
        ), row=index, col=1)

        #KAMA
        fig.add_trace(go.Scatter(
            x=df.index, y=df['KAMA'], mode='lines',
            line=dict(color='yellow', width=1),
            name='KAMA'
        ), row=index, col=1)

        # Rolling ATR Bands
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Upper_Band'], mode='lines',
            line=dict(color='yellow', width=1, dash='dash'),
            name='Upper ATR'
        ), row=index, col=1
        )
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Lower_Band'], mode='lines',
            line=dict(color='yellow', width=1, dash='dash'),
            name='Lower ATR'
        ), row=index, col=1)

        # Massimi relativi
        # if rel_max:
        #     max_times, max_prices = zip(*rel_max)
        #     fig.add_trace(go.Scatter(
        #         x=max_times,
        #         y=max_prices,
        #         mode='markers',
        #         marker=dict(size=10, color='red', symbol='square-open'),
        #         name='Local Max'
        #     ),
        #         row=index, col=1
        #     )
        # Minimi relativi
        # if rel_min:
        #     min_times, min_prices = zip(*rel_min)
        #     fig.add_trace(go.Scatter(
        #         x=min_times,
        #         y=min_prices,
        #         mode='markers',
        #         marker=dict(size=10, color='green', symbol='square-open'),
        #         name='Local Min'
        #     ),
        #         row=index, col=1
        #     )

        # Segnali di acquisto
        if buy_signals:
            buy_times, buy_prices = zip(*buy_signals)
            fig.add_trace(go.Scatter(
                x=buy_times, y=buy_prices, mode='markers',
                marker=dict(size=14, color='green', symbol='triangle-up'),
                name='Buy Signal'
            ), row=index, col=1)

        # Segnali di vendita
        if sell_signals:
            sell_times, sell_prices = zip(*sell_signals)
            fig.add_trace(go.Scatter(
                x=sell_times, y=sell_prices, mode='markers',
                marker=dict(size=14, color='red', symbol='triangle-down'),
                name='Sell Signal'
            ), row=index, col=1)

        #STOCASTICO
        index += 1
        fig.add_trace(go.Scatter(
            x=df.index, y=df['STOCH'], mode='lines',
            line=dict(color='darkblue', width=1),
            name='STOCH'
        ), row=index, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['STOCH_S'], mode='lines',
            line=dict(color='darkcyan', width=1),
            name='STOCH S'
        ), row=index, col=1)

        #TSI
        fig.add_trace(go.Scatter(
            x=df.index, y=df['TSI'], mode='lines',
            line=dict(color='yellow', width=1),
            name='TSI'
        ), row=index, col=1)

        # RSI
        # fig.add_trace(go.Scatter(
        #     x=df.index, y=df['RSI_S'], mode='lines',
        #     line=dict(color='orange', width=1),
        #     name='RSI SMOOTH'
        # ), row=index, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI'], mode='lines',
            line=dict(color='salmon', width=1),
            name='RSI SHORT'
        ), row=index, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI2'], mode='lines',
            line=dict(color='pink', width=1),
            name='RSI MED'
        ), row=index, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI3'], mode='lines',
            line=dict(color='purple', width=1),
            name='RSI LONG'
        ), row=index, col=1)
        fig.add_trace(go.Scatter(
            x=[df.index.min(), df.index.max()], y=[rsi_sell_limit, rsi_sell_limit], mode='lines',
            line=dict(color='red', width=1, dash='dash'),
            name='Sell Limit'
        ), row=index, col=1)
        fig.add_trace(go.Scatter(
            x=[df.index.min(), df.index.max()], y=[rsi_buy_limit, rsi_buy_limit], mode='lines',
            line=dict(color='green', width=1, dash='dash'),
            name='Buy Limit'
        ), row=index, col=1)

        # MACD
        # index += 1
        # fig.add_trace(go.Scatter(
        #     x=df.index, y=df['MACD_L'], mode='lines',
        #     line=dict(color='fuchsia', width=1, dash='dot'),
        #     name='MACD Line'
        # ), row=index, col=1)
        # fig.add_trace(go.Scatter(
        #     x=df.index, y=df['MACD_S'], mode='lines',
        #     line=dict(color='blue', width=1, dash='dot'),
        #     name='MACD Signal'
        # ), row=index, col=1)
        # fig.add_trace(go.Bar(
        #     x=df.index, y=df['MACD'], name='MACD',
        #     marker=dict(color='lightyellow')
        # ), row=index, col=1
        # )
        #
        # fig.add_trace(go.Scatter(
        #     x=[df.index.min(), df.index.max()], y=[macd_buy_limit, macd_buy_limit], mode='lines',
        #     line=dict(color='green', width=1, dash='dash'),
        #     name='Buy Limit'
        # ), row=index, col=1
        # )
        # fig.add_trace(go.Scatter(
        #     x=[df.index.min(), df.index.max()], y=[macd_sell_limit, macd_sell_limit], mode='lines',
        #     line=dict(color='red', width=1, dash='dash'),
        #     name='Sell Limit'
        # ), row=index, col=1)

        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=total_height,
            shapes=trend_shapes
        )

    # ======================================
    # Creazione del DataFrame finale con le operazioni
    if operations:
        trades_df = pd.DataFrame(operations)
        # Aggiungiamo qualche metrica sul periodo analizzato
        apertura = df['Open'].iloc[0]  # Prezzo di apertura (prima candela)
        chiusura = df['Close'].iloc[-1]  # Prezzo di chiusura (ultima candela)
        high_max = df['High'].max()
        low_min = df['Low'].min()
        # Variazione percentuale (close finale su open iniziale)
        variazione = (chiusura - apertura) / apertura * 100
        # Volatilità: std dei rendimenti "Close-to-Close", in termini %
        volatilita = df['Close'].pct_change().std() * 100
        # Inseriamo questi valori su ogni riga del DataFrame trades_df.
        trades_df['massimo'] = high_max
        trades_df['minimo'] = low_min
        trades_df['variazione(%)'] = variazione
        trades_df['volatilita(%)'] = volatilita
    else:
        # Nessun trade effettuato
        trades_df = pd.DataFrame(columns=[
            'Buy_Time', 'Buy_Price', 'Sell_Time', 'Sell_Price',
            'Quantity', 'Profit', 'Wallet_After'
        ])

    print(f"{wallet} USDC su {asset}, fee={fee_percent}%, {interval}, strategia: {strategia}, "
          f"profitto totale={round(trades_df['Profit'].sum())} USD")

    return fig, trades_df, actual_hours


if __name__ == "__main__":
    # ------------------------------
    # Configura il titolo della pagina e il logo
    st.set_page_config(
        page_title="CryptoFarm Simulator",  # Titolo della scheda del browser
        page_icon="📈",  # Icona (grafico che sale, simbolico per un mercato finanziario)
        layout="wide",  # Layout: "centered" o "wide"
        initial_sidebar_state="expanded"  # Stato iniziale della sidebar: "expanded", "collapsed", "auto"
    )
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    # if 'model' not in st.session_state:
    #     st.session_state['model'] = load_model('trained_model.keras')

    text_placeholder = st.empty()
    fig_placeholder = st.empty()
    st.sidebar.title("Market parameters")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        asset = st.text_input(label="Asset", placeholder="es. BTC, ETH, XRP...", max_chars=8)
        time_hours = st.number_input(label="Time Hours", min_value=0, value=24, step=24)

    with col2:
        currency = st.text_input(label="Currency", placeholder="es. USDC, USDT, EUR...", max_chars=8, value="USDT")
        interval = st.selectbox(label="Candle Interval",
                                options=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"],
                                index=3)

    symbol = asset + currency
    wallet = st.sidebar.number_input(label=f"Wallet ({currency})", min_value=0, value=100, step=1)
    st.sidebar.title("Indicators parameters")
    strategia = st.sidebar.selectbox(label="Strategia",
                                     options=["-",
                                              "Close Buy/Sell Limits",
                                              "Close ATR",
                                              "Close Bullish EMA",
                                              "Close EMA Crossover",
                                              "Supetrend",
                                              "Trend Zones",
                                              "TP/SL with ATR",
                                              "Green Candles",
                                              "ATR Live Trade"
                                              ],
                                     index=0)
    if st.sidebar.button("SIMULATE"):
        st.session_state['df'], _ = get_market_data(asset=symbol, interval=interval, time_hours=time_hours)

    col1, col2 = st.sidebar.columns(2)

    # step = col1.number_input(label="PSAR Step", min_value=0.001, max_value=1.000, value=0.01, step=0.001, format="%.3f")
    # max_step = col2.number_input(label="PSAR Max Step", min_value=0.01, max_value=1.0, value=0.4, step=0.01)
    atr_multiplier = col1.number_input(label="ATR Multiplier", min_value=0.1, max_value=50.0, value=1.6, step=0.1)
    atr_window = col2.number_input(label="ATR Window", min_value=1, max_value=100, value=5, step=1)

    col1, col2, col3 = st.sidebar.columns(3)
    rsi_window = col1.number_input(label="RSI Short", min_value=2, max_value=500, value=12, step=1)
    rsi_window2 = col2.number_input(label="Medium", min_value=2, max_value=500, value=24, step=1)
    rsi_window3 = col3.number_input(label="Long", min_value=2, max_value=500, value=36, step=1)

    ema_window = col1.number_input(label="EMA Short", min_value=1, max_value=500, value=10, step=1)
    ema_window2 = col2.number_input(label="Medium", min_value=1, max_value=500, value=50, step=1)
    ema_window3 = col3.number_input(label="Long", min_value=1, max_value=500, value=200, step=1)

    # macd_short_window = col1.number_input(label="MACD Short", min_value=0, max_value=500, value=12, step=1)
    # macd_long_window = col2.number_input(label="Long", min_value=0, max_value=500, value=26, step=1)
    # macd_signal_window = col3.number_input(label="Signal", min_value=0, max_value=500, value=9, step=1)

    col1, col2 = st.sidebar.columns(2)
    kama_pow1 = col1.number_input(label="KAMA Pow 1", min_value=1, max_value=1000, value=2, step=1)
    kama_pow2 = col2.number_input(label="Pow 2", min_value=1, max_value=1000, value=30, step=1)
    rsi_buy_limit = col1.number_input(label="RSI Buy limit", min_value=0, max_value=100, value=25, step=1)
    rsi_sell_limit = col2.number_input(label="RSI Sell limit", min_value=0, max_value=100, value=75, step=1)

    # macd_buy_limit = col1.number_input(label="MACD Buy Limit", min_value=-10.0, max_value=10.0, value=-2.5,
                                       # value=-0.66,
    #                                    step=0.01)
    # macd_sell_limit = col2.number_input(label="MACD Sell Limit", min_value=-10.0, max_value=10.0, value=2.5,
                                        # value=0.66,
    #                                    step=0.01)
    # din_macd_div = col1.number_input(label="ATR Dividend", min_value=-10.0, max_value=10.0, value=1.2,
    #                                  step=0.1)

    stop_loss = col2.number_input(label="Stop Loss %", min_value=0.1, max_value=100.0, value=99.0, step=1.0)

    num_cond = col1.number_input(label="Numero di condizioni", min_value=1, max_value=10, value=1, step=1)
    window_pivot = col2.number_input(label="Min-Max Window", min_value=2, max_value=500, value=100, step=2)

    if st.session_state['df'] is not None:
        if st.sidebar.button("SAVE DATA"):
            st.write(st.session_state['df'])

    csv_file = st.sidebar.text_input(label="CSV File", value="C:/Users/monini.m/Documents/market_data.csv")
    if st.sidebar.button("Read from CSV"):
        st.session_state['df'] = pd.read_csv(csv_file)
        st.session_state['df'].set_index('Open time', inplace=True)
        # Mantieni solo le colonne essenziali, converti a float
        st.session_state['df'] = st.session_state['df'][['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

    show_graph = st.sidebar.checkbox(label="Show Graphs", value=1)

    if st.session_state['df'] is not None:
        (fig, trades_df, actual_hours) = trading_analysis(
            asset=symbol,
            interval=interval,
            wallet=wallet,  # Wallet iniziale
            # step=step,
            # max_step=max_step,
            time_hours=time_hours,
            fee_percent=0.1,  # %
            atr_multiplier=atr_multiplier, atr_window=atr_window,
            window_pivot=window_pivot,
            rsi_window=rsi_window, rsi_window2=rsi_window2, rsi_window3=rsi_window3,
            ema_window=ema_window, ema_window2=ema_window2, ema_window3=ema_window3,
            # macd_short_window=macd_short_window, macd_long_window=macd_long_window,
            # macd_signal_window=macd_signal_window,
            kama_pow1=kama_pow1, kama_pow2=kama_pow2,
            rsi_buy_limit=rsi_buy_limit, rsi_sell_limit=rsi_sell_limit,  # OK
            # macd_buy_limit=macd_buy_limit, macd_sell_limit=macd_sell_limit,  # NO, DA TOGLIERE
            num_cond=num_cond,
            stop_loss=stop_loss,
            strategia=strategia,
            # din_macd_div=din_macd_div,
            market_data=st.session_state['df'],
        )
        text_placeholder.subheader("Operations Report")
        if not trades_df.empty:
            # text_placeholder.write(trades_df)
            total_profit = trades_df['Profit'].sum()
            num_trades = len(trades_df)
            profitable_trades = trades_df[trades_df['Profit'] > 0]
            num_profitable = len(profitable_trades)
            win_rate = (
                    num_profitable / num_trades * 100) if num_trades > 0 else 0.0
            text_placeholder.write(f"Total profit: {total_profit:.2f} {currency}, Winrate: {win_rate:.2f}%")
        else:
            text_placeholder.write("No operation performed.")
        if show_graph:
            fig_placeholder.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.sidebar.columns(2)
        start_date = col1.date_input(label="Start Date")
        end_date = col2.date_input(label="End Date")
        if st.sidebar.button("Get Data from Dates"):
            data, _ = get_market_data_between_dates(asset=symbol,
                                   interval=interval,
                                   start_date=start_date,
                                   end_date=end_date)
            st.write(data)
