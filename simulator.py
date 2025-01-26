import pandas as pd
import plotly.graph_objects as go
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator, TSIIndicator, ROCIndicator, AwesomeOscillatorIndicator, StochRSIIndicator, PercentageVolumeOscillator
from ta.trend import MACD, SMAIndicator, PSARIndicator, VortexIndicator
from ta.volume import AccDistIndexIndicator, OnBalanceVolumeIndicator, ForceIndexIndicator, VolumePriceTrendIndicator, MFIIndicator
from binance import Client
import streamlit as st
import numpy as np
import math
import time
from scipy.signal import argrelextrema
import warnings
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
def add_technical_indicator(df, step, max_step, rsi_window, macd_long_window, macd_short_window, macd_signal_window, atr_window, atr_multiplier):
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
    rsi_indicator = RSIIndicator(
        close=df_copy['Close'],
        window=rsi_window
    )
    df_copy['RSI'] = rsi_indicator.rsi()

    # Vortex Indicator
    vi = VortexIndicator(
        high=df_copy['High'],
        low=df_copy['Low'],
        close=df_copy['Close'],
        window=rsi_window)
    vip = vi.vortex_indicator_pos()
    vim = vi.vortex_indicator_neg()
    df_copy['VI'] = vip - vim

    # Calcolo del MACD
    macd_indicator = MACD(
        close=df_copy['Close'],
        window_slow=macd_long_window,
        window_fast=macd_short_window,
        window_sign=macd_signal_window
    )
    macd = macd_indicator.macd_diff()  # Istogramma (differenza tra MACD e Signal Line)
    # Calcolo del MACD normalizzato come percentuale del prezzo
    df_copy['MACD'] = macd / df_copy['Close'] * 100  # normalizzato

    # ATR
    atr_indicator = AverageTrueRange(
        high=df_copy['High'],
        low=df_copy['Low'],
        close=df_copy['Close'],
        window=atr_window
    )
    df_copy['ATR'] = atr_indicator.average_true_range()

    # SMA (Media Mobile per le Rolling ATR Bands)
    sma_indicator = SMAIndicator(close=df_copy['Close'], window=atr_window)
    df_copy['SMA'] = sma_indicator.sma_indicator()

    # Rolling ATR Bands
    df_copy['Upper_Band'] = df_copy['SMA'] + atr_multiplier * df_copy['ATR']
    df_copy['Lower_Band'] = df_copy['SMA'] - atr_multiplier * df_copy['ATR']

    # TSI
    tsi_indicator = TSIIndicator(close=df_copy['Close'])
    df_copy['TSI'] = tsi_indicator.tsi()

    # ROC
    roc_indicator = ROCIndicator(close=df_copy['Close'], window=rsi_window)
    df_copy['ROC'] = roc_indicator.roc()

    # Awesome Oscillator
    ao_indicator = AwesomeOscillatorIndicator(
        high=df_copy['High'],
        low=df_copy['Low']
    )
    df_copy['AO'] = ao_indicator.awesome_oscillator()

    # Stochastic RSI
    stoch_rsi_indicator = StochRSIIndicator(close=df_copy['Close'], window=rsi_window)
    df_copy['StochRSI'] = stoch_rsi_indicator.stochrsi()

    # Percentage Volume Oscillator
    pvo_indicator = PercentageVolumeOscillator(volume=df_copy['Volume'])
    df_copy['PVO'] = pvo_indicator.pvo()

    # Accumulation/Distribution Index
    adi_indicator = AccDistIndexIndicator(
        high=df_copy['High'],
        low=df_copy['Low'],
        close=df_copy['Close'],
        volume=df_copy['Volume']
    )
    df_copy['ADI'] = adi_indicator.acc_dist_index()

    # On-Balance Volume
    obv_indicator = OnBalanceVolumeIndicator(
        close=df_copy['Close'],
        volume=df_copy['Volume']
    )
    df_copy['OBV'] = obv_indicator.on_balance_volume()

    # Force Index
    fi_indicator = ForceIndexIndicator(
        close=df_copy['Close'],
        volume=df_copy['Volume']
    )
    df_copy['ForceIndex'] = fi_indicator.force_index()

    # Volume Price Trend
    vpt_indicator = VolumePriceTrendIndicator(
        close=df_copy['Close'],
        volume=df_copy['Volume']
    )
    df_copy['VPT'] = vpt_indicator.volume_price_trend()

    # Money Flow Index
    mfi_indicator = MFIIndicator(
        high=df_copy['High'],
        low=df_copy['Low'],
        close=df_copy['Close'],
        volume=df_copy['Volume'],
        window=rsi_window
    )
    df_copy['MFI'] = mfi_indicator.money_flow_index()

    return df_copy


def trading_analysis(
        asset: str,
        interval: str,
        wallet: float,
        step: float,  # compreso tra 0.001 e 0.1
        max_step: float,  # compreso tra 0.1 e 1
        time_hours: int = 24,
        fee_percent: float = 0.1,  # Commissione % per ogni operazione (buy e sell)
        show: bool = True,
        atr_multiplier: float = 1.5,  # Moltiplicatore per le Rolling ATR Bands
        atr_window: int = 14,  # compreso tra 2 e 30
        window_pivot: int = 10,  # compreso tra 2 e 30 (numeri pari)
        rsi_window: int = 10,  # compreso tra 2 e 50
        macd_short_window: int = 12,  # compreso tra 4 e 20
        macd_long_window: int = 26,  # compreso tra 20 e 50
        macd_signal_window: int = 9,  # short < signal < long
        rsi_buy_limit: int = 40,
        rsi_sell_limit: int = 60,
        macd_buy_limit: float = -0.4,
        macd_sell_limit: float = 0.4,
        vi_buy_limit: float = -0.5,
        vi_sell_limit: float = 0.5,
        psarvp_buy_limit: float = -0.1,
        psarvp_sell_limit: float = 10.1,
        num_cond: int = 3,
        strategia: str = "",
        market_data: dict = None,
        # modello = None
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
    # Utilizziamo i prezzi massimi ('High') e minimi ('Low')
    price_high = df['High']
    price_low = df['Low']
    # Trova gli indici dei massimi e minimi relativi
    order = int(window_pivot / 2)
    max_idx = argrelextrema(price_high.values, np.greater, order=order)[0]
    min_idx = argrelextrema(price_low.values, np.less, order=order)[0]
    # Inizializza gli array per massimi e minimi
    rel_max = []
    rel_min = []
    # Popola gli array con tuple (indice, prezzo)
    for i in min_idx:
        rel_min.append((df.index[i], df.loc[df.index[i], 'Low']))
    for i in max_idx:
        rel_max.append((df.index[i], df.loc[df.index[i], 'High']))

    df = add_technical_indicator(df,
                                 step=step,
                                 max_step=max_step,
                                 rsi_window=rsi_window,
                                 macd_long_window=macd_long_window,
                                 macd_short_window=macd_short_window,
                                 macd_signal_window=macd_signal_window,
                                 atr_window=atr_window,
                                 atr_multiplier=atr_multiplier)

    # ======================================
    # Identificazione dei segnali di acquisto e vendita
    buy_signals = []
    sell_signals = []
    holding = False
    for i in range(1, len(df)):
        # Se hai un modello e vuoi usarlo:
        # assicuriamoci di avere abbastanza dati per creare la finestraf
        # if (modello is not None) and (i >= EXT_WINDOW_SIZE) and False:
        #     # Costruisci la sequenza dagli ultimi 'window_size_for_model' punti
        #     # Finestra: df[features].iloc[i-window_size_for_model:i]
        #     X_seq = df_perc[FEATURES].iloc[i - EXT_WINDOW_SIZE:i].values
        #     X_seq = np.expand_dims(X_seq, axis=0)  # shape (1, window_size, n_eatures)
        #
        #     # Previsione
        #     predicted_probs = modello.predict(X_seq)
        #     predicted_class = np.argmax(predicted_probs, axis=1)[0]  # 0,1,2
        #
        #     # Esempio di interpretazione:
        #     #  - 1 => segnale di buy
        #     #  - 2 => segnale di sell
        #     #  - 0 => no action
        #     if not holding and predicted_class == 1:
        #         buy_signals.append((df.index[i], float(df['Close'].iloc[i])))
        #         holding = True
        #
        #     elif holding and predicted_class == 2:
        #         sell_signals.append((df.index[i], float(df['Close'].iloc[i])))
        #         holding = False
        # else:
        # ------------------------------------------------------------
        if strategia == "ATR Bands":
            if not holding and (df['PSAR'].iloc[i] > df['Close'].iloc[i]) and df['Low'].iloc[i] < df['Lower_Band'].iloc[i]:
                buy_signals.append((df.index[i], float(df['Lower_Band'].iloc[i])))
                holding = True
            if holding and (df['PSAR'].iloc[i] < df['Close'].iloc[i]) and df['High'].iloc[i] > df['Upper_Band'].iloc[i]:
                sell_signals.append((df.index[i], float(df['Upper_Band'].iloc[i])))
                holding = False
        # ------------------------------------------------------------
        if strategia == "Buy/Sell Limits":
            cond_buy_1 = 1 if df['MACD'].iloc[i] <= macd_buy_limit else 0
            cond_buy_2 = 1 if df['RSI'].iloc[i] <= rsi_buy_limit else 0
            cond_buy_3 = 1 if df['VI'].iloc[i] <= vi_buy_limit else 0
            cond_buy_4 = 1 if df['PSARVP'].iloc[i] >= psarvp_buy_limit else 0
            cond_buy_5 = 1 if df['Low'].iloc[i] <= df['Lower_Band'].iloc[i] else 0
            sum_buy = cond_buy_1 + cond_buy_2 + cond_buy_3 + cond_buy_4 + cond_buy_5
            if not holding and sum_buy >= num_cond:
                if df['Low'].iloc[i] < df['Lower_Band'].iloc[i]:
                    buy_signals.append((df.index[i], float(df['Lower_Band'].iloc[i])))
                else:
                    buy_signals.append((df.index[i], float(df['Close'].iloc[i])))
                holding = True
            cond_sell_1 = 1 if df['MACD'].iloc[i] >= macd_sell_limit else 0
            cond_sell_2 = 1 if df['RSI'].iloc[i] >= rsi_sell_limit else 0
            cond_sell_3 = 1 if df['VI'].iloc[i] >= vi_sell_limit else 0
            cond_sell_4 = 1 if df['PSARVP'].iloc[i] <= psarvp_sell_limit else 0
            cond_sell_5 = 1 if df['High'].iloc[i] >= df['Upper_Band'].iloc[i] else 0
            sum_sell = cond_sell_1 + cond_sell_2 + cond_sell_3 + cond_sell_4 + cond_sell_5
            if holding and sum_sell >= num_cond:
                if df['High'].iloc[i] > df['Upper_Band'].iloc[i]:
                    sell_signals.append((df.index[i], float(df['Upper_Band'].iloc[i])))
                else:
                    sell_signals.append((df.index[i], float(df['Close'].iloc[i])))
                holding = False

    valori_ottimi = []  # Lista per salvare i risultati
    for item in rel_min:
        index = item[0]  # L'indice è il primo elemento della tupla
        if index in df.index:  # Verifica che l'indice sia presente nel DataFrame
            valori_ottimi.append({'Type': "Min",
                                  'Prezzo': df.loc[index, 'Low'],
                                  'RSI': df.loc[index, 'RSI'],
                                  'PSAR': df.loc[index, 'PSAR'],
                                  'SMA': df.loc[index, 'SMA'],
                                  'ATR': df.loc[index, 'ATR'],
                                  'MACD': df.loc[index, 'MACD'],
                                  'VI': df.loc[index, 'VI'],
                                  })
        else:
            print(f"Index {index} not found in DataFrame.")
    for item in rel_max:
        index = item[0]  # L'indice è il primo elemento della tupla
        if index in df.index:  # Verifica che l'indice sia presente nel DataFrame
            valori_ottimi.append({'Type': "Max",
                                  'Prezzo': df.loc[index, 'Low'],
                                  'RSI': df.loc[index, 'RSI'],
                                  'PSAR': df.loc[index, 'PSAR'],
                                  'SMA': df.loc[index, 'SMA'],
                                  'ATR': df.loc[index, 'ATR'],
                                  'MACD': df.loc[index, 'MACD'],
                                  'VI': df.loc[index, 'VI'],
                                  })
        else:
            print(f"Index {index} not found in DataFrame.")

    # ======================================
    # Simulazione di trading con commissioni
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

    # ======================================
    # 4. Creazione del grafico
    fig = go.Figure()
    fig_rsi = go.Figure()
    fig_macd = go.Figure()
    fig_vi = go.Figure()
    fig_psarvp = go.Figure()
    fig_tsi = go.Figure()
    fig_roc = go.Figure()
    fig_ao = go.Figure()
    fig_stochrsi = go.Figure()
    fig_pvo = go.Figure()
    fig_adi = go.Figure()
    fig_obv = go.Figure()
    fig_fi = go.Figure()
    fig_vpt = go.Figure()
    fig_mfi = go.Figure()
    if show:
        # Candele (candlestick)
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=f"{asset}"
        ))
        # Punti SAR (marker rossi)
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['PSAR'],
            mode='markers',
            marker=dict(size=2, color='yellow', symbol='circle'),
            name='PSAR'
        ))
        # Rolling ATR Bands
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Upper_Band'],
            mode='lines',
            line=dict(color='red', width=1),
            name='Upper ATR'
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Lower_Band'],
            mode='lines',
            line=dict(color='green', width=1),
            name='Lower ATR'
        ))

        # Massimi relativi
        if rel_max:
            max_times, max_prices = zip(*rel_max)
            fig.add_trace(go.Scatter(
                x=max_times,
                y=max_prices,
                mode='markers',
                marker=dict(size=10, color='red', symbol='square-open'),
                name='Local Max'
            ))
        # Minimi relativi
        if rel_min:
            min_times, min_prices = zip(*rel_min)
            fig.add_trace(go.Scatter(
                x=min_times,
                y=min_prices,
                mode='markers',
                marker=dict(size=10, color='green', symbol='square-open'),
                name='Local Min'
            ))

        # Segnali di acquisto
        if buy_signals:
            buy_times, buy_prices = zip(*buy_signals)
            fig.add_trace(go.Scatter(
                x=buy_times,
                y=buy_prices,
                mode='markers',
                marker=dict(size=14, color='green', symbol='triangle-up'),
                name='Buy Signal'
            ))

        # Segnali di vendita
        if sell_signals:
            sell_times, sell_prices = zip(*sell_signals)
            fig.add_trace(go.Scatter(
                x=sell_times,
                y=sell_prices,
                mode='markers',
                marker=dict(size=14, color='red', symbol='triangle-down'),
                name='Sell Signal'
            ))
        # RSI
        fig_rsi.add_trace(go.Scatter(
            x=df.index,
            y=df['RSI'],
            mode='lines',
            line=dict(color='purple', width=2),
            name='RSI'
        ))
        # Linea tratteggiata a 70 (overbought)
        fig_rsi.add_trace(go.Scatter(
            x=[df.index.min(), df.index.max()],
            y=[rsi_sell_limit, rsi_sell_limit],
            mode='lines',
            line=dict(color='red', width=1, dash='dash'),
            name='Sell Limit'
        ))

        # Linea tratteggiata a 30 (oversold)
        fig_rsi.add_trace(go.Scatter(
            x=[df.index.min(), df.index.max()],
            y=[rsi_buy_limit, rsi_buy_limit],
            mode='lines',
            line=dict(color='green', width=1, dash='dash'),
            name='Buy Limit'
        ))

        # MACD
        fig_macd.add_trace(go.Bar(
            x=df.index,
            y=df['MACD'],
            name='MACD',
            marker=dict(color='yellow')
        ))
        fig_macd.add_trace(go.Scatter(
            x=[df.index.min(), df.index.max()],
            y=[macd_buy_limit, macd_buy_limit],
            mode='lines',
            line=dict(color='green', width=1, dash='dash'),
            name='Buy Limit'
        ))
        fig_macd.add_trace(go.Scatter(
            x=[df.index.min(), df.index.max()],
            y=[macd_sell_limit, macd_sell_limit],
            mode='lines',
            line=dict(color='red', width=1, dash='dash'),
            name='Sell Limit'
        ))
        # Aggiungi VI
        fig_vi.add_trace(go.Scatter(
            x=df.index,
            y=df['VI'],
            mode='lines',
            line=dict(color='yellow', width=1),
            name='VI'
        ))
        fig_vi.add_trace(go.Scatter(
            x=[df.index.min(), df.index.max()],
            y=[vi_buy_limit, vi_buy_limit],
            mode='lines',
            line=dict(color='green', width=1, dash='dash'),
            name='Buy Limit'
        ))
        fig_vi.add_trace(go.Scatter(
            x=[df.index.min(), df.index.max()],
            y=[vi_sell_limit, vi_sell_limit],
            mode='lines',
            line=dict(color='red', width=1, dash='dash'),
            name='Sell Limit'
        ))
        # PSAR versus Price
        fig_psarvp.add_trace(go.Scatter(
            x=df.index,
            y=df['PSARVP'],
            name='PSAR vs Price',
            marker=dict(color='yellow')
        ))
        fig_psarvp.add_trace(go.Scatter(
            x=[df.index.min(), df.index.max()],
            y=[psarvp_buy_limit, psarvp_buy_limit],
            mode='lines',
            line=dict(color='green', width=1, dash='dash'),
            name='Buy Limit'
        ))
        fig_psarvp.add_trace(go.Scatter(
            x=[df.index.min(), df.index.max()],
            y=[psarvp_sell_limit, psarvp_sell_limit],
            mode='lines',
            line=dict(color='red', width=1, dash='dash'),
            name='Sell Limit'
        ))

        # TSI
        fig_tsi.add_trace(go.Scatter(
            x=df.index,
            y=df['TSI'],
            mode='lines',
            line=dict(color='blue', width=2),
            name='TSI'
        ))

        # ROC
        fig_roc.add_trace(go.Scatter(
            x=df.index,
            y=df['ROC'],
            mode='lines',
            line=dict(color='orange', width=2),
            name='ROC'
        ))

        # Awesome Oscillator
        fig_ao.add_trace(go.Scatter(
            x=df.index,
            y=df['AO'],
            mode='lines',
            line=dict(color='cyan', width=2),
            name='Awesome Oscillator'
        ))

        # Stochastic RSI
        fig_stochrsi.add_trace(go.Scatter(
            x=df.index,
            y=df['StochRSI'],
            mode='lines',
            line=dict(color='magenta', width=2),
            name='Stochastic RSI'
        ))

        # Percentage Volume Oscillator
        fig_pvo.add_trace(go.Scatter(
            x=df.index,
            y=df['PVO'],
            mode='lines',
            line=dict(color='brown', width=2),
            name='PVO'
        ))

        # Accumulation/Distribution Index
        fig_adi.add_trace(go.Scatter(
            x=df.index,
            y=df['ADI'],
            mode='lines',
            line=dict(color='gold', width=2),
            name='Accumulation/Distribution Index'
        ))

        # On-Balance Volume
        fig_obv.add_trace(go.Scatter(
            x=df.index,
            y=df['OBV'],
            mode='lines',
            line=dict(color='teal', width=2),
            name='On-Balance Volume'
        ))

        # Force Index
        fig_fi.add_trace(go.Scatter(
            x=df.index,
            y=df['ForceIndex'],
            mode='lines',
            line=dict(color='darkblue', width=2),
            name='Force Index'
        ))

        # Volume Price Trend
        fig_vpt.add_trace(go.Scatter(
            x=df.index,
            y=df['VPT'],
            mode='lines',
            line=dict(color='darkgreen', width=2),
            name='Volume Price Trend'
        ))

        # Money Flow Index
        fig_mfi.add_trace(go.Scatter(
            x=df.index,
            y=df['MFI'],
            mode='lines',
            line=dict(color='darkred', width=2),
            name='Money Flow Index'
        ))

        # Layout e aspetto del grafico principale
        fig.update_layout(
            title=f"{asset} ({interval}) CandleChart",
            xaxis_title="Date",
            yaxis_title=f"Price",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=600
        )
        # Configurare il layout del grafico MACD
        fig_macd.update_layout(
            title='MACD, Signal Line, and Histogram',
            xaxis_title='Date',
            yaxis_title='Value',
            template="plotly_dark",
            height=300
        )
        # Configurare il layout del grafico per l'RSI
        fig_rsi.update_layout(
            title='Relative Strength Index (RSI)',
            xaxis_title='Date',
            yaxis_title='RSI',
            yaxis=dict(
                range=[0, 100],  # L'RSI va tipicamente da 0 a 100
            ),
            template="plotly_dark",
            height=300
        )
        # Configurare il layout del grafico per l'VI
        fig_vi.update_layout(
            title='Vortex Indicator (VI)',
            xaxis_title='Date',
            yaxis_title='Value',
            template="plotly_dark",
            height=300
        )
        # Configurare il layout del grafico per l'PSARPVP
        fig_psarvp.update_layout(
            title='PSAR vs Price (PSARVP)',
            xaxis_title='Date',
            yaxis_title='Value',
            template="plotly_dark",
            height=300
        )

        # Configurare il layout del grafico per il TSI
        fig_tsi.update_layout(
            title='True Strength Index (TSI)',
            xaxis_title='Date',
            yaxis_title='Value',
            template="plotly_dark",
            height=300
        )
        # Configurare il layout del grafico per il ROC
        fig_roc.update_layout(
            title='Rate of Change (ROC)',
            xaxis_title='Date',
            yaxis_title='Value',
            template="plotly_dark",
            height=300
        )
        # Configurare il layout del grafico per l'Awesome Oscillator
        fig_ao.update_layout(
            title='Awesome Oscillator (AO)',
            xaxis_title='Date',
            yaxis_title='Value',
            template="plotly_dark",
            height=300
        )
        # Configurare il layout del grafico per lo Stochastic RSI
        fig_stochrsi.update_layout(
            title='Stochastic RSI',
            xaxis_title='Date',
            yaxis_title='Value',
            template="plotly_dark",
            height=300
        )
        # Configurare il layout del grafico per il PVO
        fig_pvo.update_layout(
            title='Percentage Volume Oscillator (PVO)',
            xaxis_title='Date',
            yaxis_title='Value',
            template="plotly_dark",
            height=300
        )
        # Configurare il layout del grafico per l'ADI
        fig_adi.update_layout(
            title='Accumulation/Distribution Index (ADI)',
            xaxis_title='Date',
            yaxis_title='Value',
            template="plotly_dark",
            height=300
        )
        # Configurare il layout del grafico per l'OBV
        fig_obv.update_layout(
            title='On-Balance Volume (OBV)',
            xaxis_title='Date',
            yaxis_title='Value',
            template="plotly_dark",
            height=300
        )
        # Configurare il layout del grafico per il Force Index
        fig_fi.update_layout(
            title='Force Index',
            xaxis_title='Date',
            yaxis_title='Value',
            template="plotly_dark",
            height=300
        )
        # Configurare il layout del grafico per il VPT
        fig_vpt.update_layout(
            title='Volume Price Trend (VPT)',
            xaxis_title='Date',
            yaxis_title='Value',
            template="plotly_dark",
            height=300
        )
        # Configurare il layout del grafico per il MFI
        fig_mfi.update_layout(
            title='Money Flow Index (MFI)',
            xaxis_title='Date',
            yaxis_title='Value',
            template="plotly_dark",
            height=300
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

    print(f"{wallet} USDC su {asset}, fee={fee_percent}%, {interval}, step={step}, max_step={max_step}, "
          f"atr_multiplier={atr_multiplier}, atr_window={atr_window}, rsi_window={rsi_window}, "
          f"rsi_buy_limit={rsi_buy_limit}, rsi_sell_limit={rsi_sell_limit}, "
          f"profitto totale={round(trades_df['Profit'].sum())} USD")

    return (fig, fig_rsi, fig_macd, fig_vi, fig_psarvp, fig_tsi, fig_roc,
            fig_ao, fig_stochrsi, fig_pvo, fig_adi, fig_obv,
            fig_fi, fig_vpt, fig_mfi, trades_df, actual_hours)


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
    # csv_file = st.sidebar.text_input(label="CSV File", value="C:/Users/monini.m/Documents/2025-01-13T08-47_export.csv")
    # if st.sidebar.button("Read from CSV"):
    #     st.session_state['df'] = pd.read_csv(csv_file)
    #     st.session_state['df'].set_index('Open time', inplace=True)
    #     # Mantieni solo le colonne essenziali, converti a float
    #     st.session_state['df'] = st.session_state['df'][['Open', 'High', 'Low', 'Close']].astype(float)

    text_placeholder = st.empty()
    fig_placeholder = st.empty()
    fig_rsi_placeholder = st.empty()
    fig_stochrsi_placeholder = st.empty()
    fig_tsi_placeholder = st.empty()
    fig_macd_placeholder = st.empty()
    fig_vi_placeholder = st.empty()
    fig_psarvp_placeholder = st.empty()
    fig_roc_placeholder = st.empty()
    fig_ao_placeholder = st.empty()
    fig_pvo_placeholder = st.empty()
    fig_adi_placeholder = st.empty()
    fig_obv_placeholder = st.empty()
    fig_fi_placeholder = st.empty()
    fig_vpt_placeholder = st.empty()
    fig_mfi_placeholder = st.empty()
    st.sidebar.title("Market parameters")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        asset = st.text_input(label="Asset", placeholder="es. BTC, ETH, XRP...", max_chars=8)
        time_hours = st.number_input(label="Time Hours", min_value=0, value=24, step=24)
    with col2:
        currency = st.text_input(label="Currency", placeholder="es. USDC, USDT, EUR...", max_chars=8, value="USDC")
        interval = st.selectbox(label="Candle Interval",
                                options=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"],
                                index=3)
    symbol = asset + currency
    wallet = st.sidebar.number_input(label=f"Wallet ({currency})", min_value=0, value=100, step=1)
    st.sidebar.title("Indicators parameters")
    strategia = st.sidebar.selectbox(label="Strategia",
                                options=["ATR Bands", "Buy/Sell Limits"],
                                index=0)
    col1, col2 = st.sidebar.columns(2)
    with col1:
        step = st.number_input(label="PSAR Step", min_value=0.001, max_value=1.000, value=0.010, step=0.001, format="%.3f")
        atr_multiplier = st.number_input(label="ATR Multiplier", min_value=1.0, max_value=5.0, value=2.4, step=0.1)
        rsi_window = st.number_input(label="RSI Window", min_value=2, max_value=500, value=12, step=1)
        rsi_buy_limit = st.number_input(label="RSI Buy limit", min_value=1, max_value=99, value=30, step=1)
        macd_buy_limit = st.number_input(label="MACD Buy Limit", min_value=-10.0, max_value=10.0, value=-0.48, step=0.01)
        vi_buy_limit = st.number_input(label="VI Buy Limit", min_value=-10.0, max_value=10.0, value=-0.45, step=0.01)
        psarvp_buy_limit = st.number_input(label="PSARVP Buy Limit", min_value=-10.0, max_value=10.0, value=1.01, step=0.01)
    with col2:
        max_step = st.number_input(label="PSAR Max Step", min_value=0.01, max_value=1.0, value=0.4, step=0.01)
        atr_window = st.number_input(label="ATR Window", min_value=1, max_value=100, value=6, step=1)
        window_pivot = st.number_input(label="Min-Max Window", min_value=2, max_value=500, value=50, step=2)
        rsi_sell_limit = st.number_input(label="RSI Sell limit", min_value=1, max_value=99, value=79, step=1)
        macd_sell_limit = st.number_input(label="MACD Sell Limit", min_value=-10.0, max_value=10.0, value=0.4, step=0.01)
        vi_sell_limit = st.number_input(label="VI Sell Limit", min_value=-10.0, max_value=10.0, value=0.6, step=0.01)
        psarvp_sell_limit = st.number_input(label="PSARVP Sell Limit", min_value=-10.0, max_value=10.0, value=0.99, step=0.01)
    # col1, col2, col3 = st.sidebar.columns(3)
    # with col1:
    #     macd_short_window = st.number_input(label="MACD Short Window", min_value=1, max_value=100, value=12, step=1)
    # with col2:
    #     macd_long_window = st.number_input(label="MACD Long Window", min_value=1, max_value=100, value=26, step=1)
    # with col3:
    #     macd_signal_window = st.number_input(label="MACD Signal Window", min_value=1, max_value=100, value=9, step=1)
    num_cond = st.sidebar.number_input(label="Numero di condizioni", min_value=1, max_value=5, value=2, step=1)
    if st.sidebar.button("SIMULATE"):
        df, _ = get_market_data(asset=symbol, interval=interval, time_hours=time_hours)
        st.session_state['df'] = df

    # if st.sidebar.button("Print Data"):
    #     if st.session_state['df'] is not None:
    #         st.write(st.session_state['df'])

    if st.session_state['df'] is not None:
        (fig, fig_rsi, fig_macd, fig_vi, fig_psarvp, fig_tsi, fig_roc,
         fig_ao, fig_stochrsi, fig_pvo, fig_adi, fig_obv,
         fig_fi, fig_vpt, fig_mfi, trades_df, actual_hours) = trading_analysis(
            asset=symbol,
            interval=interval,
            wallet=wallet,  # Wallet iniziale
            step=step,
            max_step=max_step,
            time_hours=time_hours,
            fee_percent=0.1,  # %
            atr_multiplier=atr_multiplier,
            atr_window=atr_window,
            window_pivot=window_pivot,
            rsi_window=rsi_window,
            # macd_short_window=macd_short_window,
            # macd_long_window=macd_long_window,
            # macd_signal_window=macd_signal_window,
            rsi_buy_limit=rsi_buy_limit,
            rsi_sell_limit=rsi_sell_limit,
            macd_buy_limit=macd_buy_limit,
            macd_sell_limit=macd_sell_limit,
            vi_buy_limit=vi_buy_limit,
            vi_sell_limit=vi_sell_limit,
            psarvp_buy_limit=psarvp_buy_limit,
            psarvp_sell_limit=psarvp_sell_limit,
            num_cond=num_cond,
            strategia=strategia,
            market_data=st.session_state['df'],
            # modello=st.session_state['model']
        )
        text_placeholder.subheader("Operations Report")
        if not trades_df.empty:
            text_placeholder.write(trades_df)
            total_profit = trades_df['Profit'].sum()
            text_placeholder.write(f"Total profit: {total_profit:.2f} {currency}")
        else:
            text_placeholder.write("No operation performed.")

        fig_placeholder.plotly_chart(fig, use_container_width=True)
        fig_rsi_placeholder.plotly_chart(fig_rsi, use_container_width=True)
        fig_stochrsi_placeholder.plotly_chart(fig_stochrsi, use_container_width=True)
        fig_tsi_placeholder.plotly_chart(fig_tsi, use_container_width=True)
        fig_macd_placeholder.plotly_chart(fig_macd, use_container_width=True)
        fig_vi_placeholder.plotly_chart(fig_vi, use_container_width=True)
        fig_psarvp_placeholder.plotly_chart(fig_psarvp, use_container_width=True)
        fig_roc_placeholder.plotly_chart(fig_roc, use_container_width=True)
        fig_ao_placeholder.plotly_chart(fig_ao, use_container_width=True)
        fig_pvo_placeholder.plotly_chart(fig_pvo, use_container_width=True)
        fig_adi_placeholder.plotly_chart(fig_adi, use_container_width=True)
        fig_obv_placeholder.plotly_chart(fig_obv, use_container_width=True)
        fig_fi_placeholder.plotly_chart(fig_fi, use_container_width=True)
        fig_vpt_placeholder.plotly_chart(fig_vpt, use_container_width=True)
        fig_mfi_placeholder.plotly_chart(fig_mfi, use_container_width=True)
