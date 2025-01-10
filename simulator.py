import pandas as pd
import plotly.graph_objects as go
from ta.trend import PSARIndicator
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volatility import KeltnerChannel
from ta.trend import MACD
from binance import Client
import streamlit as st
import numpy as np
import math
import time
from scipy.signal import argrelextrema


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
def get_market_data(
        asset: str,
        interval: str,
        time_hours: int
) -> tuple:
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


def download_market_data(assets, intervals, hours):
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


def sar_trading_analysis(
        asset: str,
        interval: str,
        wallet: float,
        step: float, # compreso tra 0.001 e 0.1
        max_step: float, # compreso tra 0.1 e 1
        time_hours: int = 24,
        fee_percent: float = 0.1,  # Commissione % per ogni operazione (buy e sell)
        show: bool = True,
        atr_multiplier: float = 1.5,  # Moltiplicatore per le Rolling ATR Bands
        atr_window: int = 14, # compreso tra 2 e 30
        window_pivot: int = 10, # compreso tra 2 e 30 (numeri pari)
        rsi_window: int = 10, # compreso tra 2 e 50
        macd_short_window: int = 12, # compreso tra 4 e 20
        macd_long_window: int = 26, # compreso tra 20 e 50
        macd_signal_window: int = 9, # short < signal < long
        market_data: dict = None
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
    order = int(window_pivot/2)
    max_idx = argrelextrema(price_high.values, np.greater, order=order)[0]
    min_idx = argrelextrema(price_low.values, np.less, order=order)[0]
    # Inizializza gli array per massimi e minimi
    rel_max = []
    rel_min = []
    # Popola gli array con tuple (indice, prezzo)
    for i in max_idx:
        rel_max.append((df.index[i], df.loc[df.index[i], 'High']))
    for i in min_idx:
        rel_min.append((df.index[i], df.loc[df.index[i], 'Low']))

    # Calcolo del SAR utilizzando la libreria "ta" (PSARIndicator)
    sar_indicator = PSARIndicator(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        step=step,
        max_step=max_step
    )
    df['SAR'] = sar_indicator.psar()

    # ATR
    atr_indicator = AverageTrueRange(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=atr_window
    )
    df['ATR'] = atr_indicator.average_true_range()

    # SMA (Media Mobile per le Rolling ATR Bands)
    sma_indicator = SMAIndicator(close=df['Close'], window=atr_window)
    df['SMA'] = sma_indicator.sma_indicator()

    # Rolling ATR Bands
    df['Upper_Band'] = df['SMA'] + atr_multiplier * df['ATR']
    df['Lower_Band'] = df['SMA'] - atr_multiplier * df['ATR']

    kc = KeltnerChannel(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=atr_window,  # finestra EMA
        window_atr=atr_window,  # finestra ATR
        original_version=False
    )

    df['KC_middle'] = kc.keltner_channel_mband()  # Banda centrale (EMA base)
    df['KC_high'] = kc.keltner_channel_hband()  # Banda superiore
    df['KC_low'] = kc.keltner_channel_lband()  # Banda inferiore

    # Calcolo dell'RSI
    # Impostazione classica RSI(14). Se vuoi segnali più veloci, puoi provare RSI(7) o RSI(9).
    # rsi_window = 30
    rsi_indicator = RSIIndicator(
        close=df['Close'],
        window=rsi_window
    )
    df['RSI'] = rsi_indicator.rsi()

    # Pivot Points dinamici
    # window_pivot = 10  # dimensione della finestra per cercare minimi/massimi locali
    # Creiamo colonne che rappresentano il massimo e minimo degli ultimi N periodi
    df['rolling_max'] = df['High'].rolling(window_pivot).max()
    df['rolling_min'] = df['Low'].rolling(window_pivot).min()

    # Calcolo delle linee MACD
    # Calcolo del MACD
    macd_indicator = MACD(
        close=df['Close'],
        window_slow=macd_long_window,
        window_fast=macd_short_window,
        window_sign=macd_signal_window
    )
    # Aggiungere le colonne del MACD al DataFrame
    df['MACD'] = macd_indicator.macd()  # Linea MACD
    df['Signal_Line'] = macd_indicator.macd_signal()  # Linea di segnale
    df['MACD_Hist'] = macd_indicator.macd_diff()  # Istogramma (differenza tra MACD e Signal Line)

    # ======================================
    # Identificazione dei segnali di acquisto e vendita
    buy_signals = []
    sell_signals = []
    holding = False
    # upper_trend = False
    # lower_trend = False
    for i in range(1, len(df)):
        # ------------------------------------------------------------
        # # Segnale di acquisto: quando il SAR passa da > prezzo a < prezzo (tra candela precedente e attuale)
        # if (not holding and (df['SAR'].iloc[i] < df['Open'].iloc[i]) and
        #         (df['SAR'].iloc[i - 1] > df['Open'].iloc[i - 1]) and
        #         (float(df['Close'].iloc[i-1]) >= float(df['Low'].iloc[i]))):
        #     # Salviamo la chiusura della candela come prezzo di buy
        #     buy_signals.append((df.index[i], float(df['Close'].iloc[i-1])))
        #     # buy_signals.append((df.index[i], float(df['Low'].iloc[i]))) # Best
        #     # buy_signals.append((df.index[i], float(df['High'].iloc[i]))) # Worst
        #     holding = True
        # # Segnale di vendita: quando il SAR passa da < prezzo a > prezzo (tra candela precedente e attuale)
        # if (holding and (df['SAR'].iloc[i - 1] < df['Open'].iloc[i - 1]) and
        #         (df['SAR'].iloc[i] > df['Open'].iloc[i]) and
        #         (float(df['Close'].iloc[i-1]) <= float(df['High'].iloc[i]))):
        #     # Salviamo la chiusura della candela precedente come prezzo di sell
        #     # print(f"Sell signal al prezzo di {float(df['Close'].iloc[i])}, mentre era stato acquisto a {buy_signals[-1][1]}")
        #     # print(f"il profitto è {(float(df['Close'].iloc[i])-buy_signals[-1][1])}")
        #     sell_signals.append((df.index[i], float(df['Close'].iloc[i-1])))
        #     # sell_signals.append((df.index[i], float(df['High'].iloc[i]))) # Best
        #     # sell_signals.append((df.index[i], float(df['Low'].iloc[i]))) # Worst
        #     holding = False
        # ------------------------------------------------------------
        # # Segnale di acquisto: quando il SAR passa da > prezzo a < prezzo (tra candela precedente e attuale)
        # if (not holding and (df['SAR'].iloc[i] < df['Close'].iloc[i]) and
        #         (df['SAR'].iloc[i - 1] > df['Close'].iloc[i - 1])):
        #     # Salviamo la chiusura della candela come prezzo di buy
        #     buy_signals.append((df.index[i], float(df['Close'].iloc[i])))
        #     holding = True
        # # Segnale di vendita: quando il SAR passa da < prezzo a > prezzo (tra candela precedente e attuale)
        # if (holding and (df['SAR'].iloc[i - 1] < df['Close'].iloc[i - 1]) and
        #         (df['SAR'].iloc[i] > df['Close'].iloc[i])):
        #     # Salviamo la chiusura della candela come prezzo di sell
        #     sell_signals.append((df.index[i], float(df['Close'].iloc[i])))
        #     holding = False
        #------------------------------------------------------------
        # upper trend: quando il SAR passa da > prezzo a < prezzo
        # if ((df['SAR'].iloc[i] < df['Close'].iloc[i]) and
        #         (df['SAR'].iloc[i - 1] > df['Close'].iloc[i - 1])):
        #     upper_trend = True
        #     lower_trend = False
        # # lower trend: quando il SAR passa da < prezzo a > prezzo (tra candela precedente e attuale)
        # if ((df['SAR'].iloc[i - 1] < df['Close'].iloc[i - 1]) and
        #         (df['SAR'].iloc[i] > df['Close'].iloc[i])):
        #     upper_trend = False
        #     lower_trend = True
        if not holding and (df['SAR'].iloc[i] > df['Close'].iloc[i]) and df['Low'].iloc[i] < df['Lower_Band'].iloc[i]:
            buy_signals.append((df.index[i], float(df['Lower_Band'].iloc[i])))
            holding = True
        if holding and (df['SAR'].iloc[i] < df['Close'].iloc[i]) and df['High'].iloc[i] > df['Upper_Band'].iloc[i]:
            sell_signals.append((df.index[i], float(df['Upper_Band'].iloc[i])))
            holding = False
        # ------------------------------------------------------------

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
            y=df['SAR'],
            mode='markers',
            marker=dict(size=4, color='yellow', symbol='circle'),
            name='SAR'
        ))
        # Rolling ATR Bands
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Upper_Band'],
            mode='lines',
            line=dict(color='red', width=1),
            name='Upper ATR Band'
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Lower_Band'],
            mode='lines',
            line=dict(color='green', width=1),
            name='Lower ATR Band'
        ))
        # KELTNER CHANNELS
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['KC_middle'],
            mode='lines',
            line=dict(color='blue', width=1, dash='dot'),
            name='KC Middle'
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['KC_high'],
            mode='lines',
            line=dict(color='blue', width=1),
            name='KC High'
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['KC_low'],
            mode='lines',
            line=dict(color='blue', width=1),
            name='KC Low'
        ))
        # PIVOT POINTS DINAMICI (Rolling Min/Max)
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['rolling_max'],
            mode='lines',
            line=dict(color='orange', width=1),
            name='Rolling Max'
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['rolling_min'],
            mode='lines',
            line=dict(color='orange', width=1),
            name='Rolling Min'
        ))

        # Massimi relativi
        if rel_max:
            max_times, max_prices = zip(*rel_max)
            fig.add_trace(go.Scatter(
                x=max_times,
                y=max_prices,
                mode='markers',
                marker=dict(size=14, color='red', symbol='square-open'),
                name='Local Maximum'
            ))
        # Minimi relativi
        if rel_min:
            min_times, min_prices = zip(*rel_min)
            fig.add_trace(go.Scatter(
                x=min_times,
                y=min_prices,
                mode='markers',
                marker=dict(size=14, color='green', symbol='square-open'),
                name='Local Minimum'
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

        # MACD
        fig_macd.add_trace(go.Scatter(
            x=df.index,
            y=df['MACD'],
            mode='lines',
            name='MACD Line',
            line=dict(color='blue')
        ))
        fig_macd.add_trace(go.Scatter(
            x=df.index,
            y=df['Signal_Line'],
            mode='lines',
            name='Signal Line',
            line=dict(color='red')
        ))
        fig_macd.add_trace(go.Bar(
            x=df.index,
            y=df['MACD_Hist'],
            name='MACD Histogram',
            marker=dict(color='green')
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
            height=200
        )
        # Configurare il layout del grafico per l'RSI
        fig_rsi.update_layout(
            title='Relative Strength Index (RSI)',
            xaxis_title='Date',
            yaxis_title='RSI',
            yaxis=dict(
                range=[0, 100],  # L'RSI va tipicamente da 0 a 100
                showgrid=True,  # Mostrare una griglia per facilitare la lettura
            ),
            template="plotly_dark",
            height=200
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
          f"atr_multiplier={atr_multiplier}, atr_window={atr_window}, profitto totale={round(trades_df['Profit'].sum())} USD")

    return fig, fig_rsi, fig_macd, trades_df, actual_hours


def run_simulation(wallet: float,
                   hours: int,
                   assets: list,
                   intervals: list,
                   steps: list,
                   max_steps: list,
                   atr_multipliers: list,
                   atr_windows: list,
                   market_data: dict = None):
    """
    Esegue una simulazione massiva di strategie Buy/Sell basate sul SAR
    (Parabolic SAR). Per ogni combinazione di:
      - Intervallo (es. 1m, 3m, 5m)
      - Asset (lista di crypto)
      - Step e Max Step per il calcolo del SAR
    viene chiamata la funzione sar_trading_analysis, e si registrano
    statistiche relative ai risultati di trading.

    Infine, mostra una tabella 'simulazioni' con i risultati.
    """

    st.title("Simulazione Buy/Sell su indice SAR")
    st.write("Avvio simulazione massiva...")

    simulazioni = []

    for interval in intervals:
        # Calcolo del tempo totale coperto dalle candele (utile per ROI giornaliero)
        try:
            candlestick_minutes = interval_to_minutes(interval)
            if candlestick_minutes <= 0:
                st.warning(f"Intervallo '{interval}' non valido (non termina in m/h?), salto.")
                continue
        except Exception as e:
            st.error(f"Errore calcolo minuti per '{interval}': {e}")
            continue

        for asset in assets:
            # Usa i dati già scaricati
            df = market_data.get(asset, {}).get(interval, None)
            for step in steps:
                for max_step in max_steps:
                    for atr_multiplier in atr_multipliers:
                        for atr_window in atr_windows:
                            # Richiamiamo la funzione che fa l'analisi col SAR e
                            # unisce buy/sell in un'unica riga per trade
                            try:
                                fig, trades_df, actual_hours = sar_trading_analysis(
                                    asset=asset,
                                    interval=interval,
                                    wallet=wallet,  # Wallet iniziale in USDT
                                    step=step,
                                    max_step=max_step,
                                    time_hours=hours,
                                    show=False,
                                    atr_multiplier=atr_multiplier,
                                    atr_window=atr_window,
                                    market_data=df
                                )
                            except Exception as e:
                                st.error(f"Errore durante sar_trading_analysis({asset}, {interval}): {e}")
                                continue

                            total_days = actual_hours / 24
                            time_string = f"{actual_hours:.2f} ore ({total_days:.2f} giorni)"
                            # Se trades_df è vuoto, nessuna operazione
                            if trades_df.empty:
                                simulazioni.append({
                                    'Asset': asset,
                                    'Intervallo': interval,
                                    'Tempo': time_string,
                                    'Step': step,
                                    'Max Step': max_step,
                                    'Operazioni Chiuse': 0,
                                    'Profitto Totale': 0,
                                    'ROI totale (%)': 0,
                                    'ROI giornaliero (%)': 0
                                    # ...e altre colonne a zero/nan
                                })
                                continue

                            # Calcola statistiche principali
                            # (ogni riga di trades_df è un trade completo: Buy+Sell)
                            num_trades = len(trades_df)
                            total_profit = trades_df['Profit'].sum()

                            # Trade in profitto/perdita/pareggio
                            profitable_trades = trades_df[trades_df['Profit'] > 0]
                            losing_trades = trades_df[trades_df['Profit'] < 0]
                            break_even_trades = trades_df[trades_df['Profit'] == 0]

                            num_profitable = len(profitable_trades)
                            num_losing = len(losing_trades)
                            num_break_even = len(break_even_trades)

                            # Win rate
                            win_rate = (num_profitable / num_trades * 100) if num_trades > 0 else 0.0
                            # Profitto medio
                            avg_profit = trades_df['Profit'].mean() if num_trades > 0 else 0.0
                            # Profitto medio (Gain)
                            avg_win = profitable_trades['Profit'].mean() if num_profitable > 0 else 0.0
                            # Perdita media (Loss)
                            avg_loss = losing_trades['Profit'].mean() if num_losing > 0 else 0.0

                            # Max e Min profit
                            max_profit_trade = trades_df['Profit'].max() if num_trades > 0 else 0.0
                            min_profit_trade = trades_df['Profit'].min() if num_trades > 0 else 0.0

                            # ROI totale
                            roi_percent = (total_profit / wallet * 100) if wallet > 0 else 0.0

                            # ROI giornaliero (composto)
                            final_wallet = wallet + total_profit
                            if final_wallet > 0:
                                daily_roi = (final_wallet / wallet) ** (1 / total_days) - 1
                                daily_roi_percent = daily_roi * 100
                            else:
                                daily_roi_percent = -100.0  # Nel caso di portafoglio azzerato o negativo

                            # Infine, ricaviamo eventuali metriche sul prezzo (se esistono nel trades_df)
                            if 'massimo' in trades_df.columns:
                                prezzo_massimo = trades_df['massimo'].max()
                            else:
                                prezzo_massimo = np.nan

                            if 'minimo' in trades_df.columns:
                                prezzo_minimo = trades_df['minimo'].min()
                            else:
                                prezzo_minimo = np.nan

                            if 'variazione(%)' in trades_df.columns:
                                # L'ultima riga o la prima: dipende da come hai popolato i dati
                                variazione_prezzo = trades_df['variazione(%)'].iloc[-1]
                            else:
                                variazione_prezzo = np.nan

                            if 'volatilita(%)' in trades_df.columns:
                                volatilita = trades_df['volatilita(%)'].iloc[-1]
                            else:
                                volatilita = np.nan

                            # Salviamo i risultati
                            simulazioni.append({
                                'Asset': asset,
                                'Intervallo': interval,
                                'Tempo': time_string,
                                'Step': step,
                                'Max Step': max_step,
                                'Moltiplicatore ATR': atr_multiplier,
                                'Finestra ATR': atr_window,
                                'Prezzo Massimo': prezzo_massimo,
                                'Prezzo Minimo': prezzo_minimo,
                                'Variazione di prezzo (%)': variazione_prezzo,
                                'Volatilità (%)': volatilita,
                                'Profitto Totale': round(total_profit, 4),
                                'Profitto Medio': round(avg_profit, 4),
                                'Operazioni Chiuse': num_trades,
                                'Operazioni in Profitto': num_profitable,
                                'Operazioni in Perdita': num_losing,
                                'Pareggi': num_break_even,
                                'Win Rate (%)': round(win_rate, 2),
                                'Profitto Medio (Gain)': round(avg_win, 4),
                                'Perdita Media (Loss)': round(avg_loss, 4),
                                'Max Profit Trade': round(max_profit_trade, 4),
                                'Min Profit Trade': round(min_profit_trade, 4),
                                'ROI totale (%)': round(roi_percent, 2),
                                'ROI giornaliero (%)': round(daily_roi_percent, 2)
                            })

    # Terminati tutti i loop, mostriamo i risultati
    if simulazioni:
        results_df = pd.DataFrame(simulazioni)
        st.write("## Risultati delle simulazioni:")
        st.dataframe(results_df)
    else:
        st.warning("Nessuna simulazione eseguita o nessun trade effettuato.")


if __name__ == "__main__":
    # ------------------------------
    # Configura il titolo della pagina e il logo
    st.set_page_config(
        page_title="CryptoFarm Simulator",  # Titolo della scheda del browser
        page_icon="📈",  # Icona (grafico che sale, simbolico per un mercato finanziario)
        layout="wide",  # Layout: "centered" o "wide"
        initial_sidebar_state="expanded"  # Stato iniziale della sidebar: "expanded", "collapsed", "auto"
    )
    text_placeholder = st.empty()
    fig_placeholder = st.empty()
    fig_rsi_placeholder = st.empty()
    fig_macd_placeholder = st.empty()
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
    symbol = asset+currency
    wallet = st.sidebar.number_input(label=f"Wallet ({currency})", min_value=0, value=1000, step=1)
    st.sidebar.title("Indicators parameters")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        step = st.number_input(label="PSAR Step", min_value=0.001, max_value=1.0, value=0.04, step=0.01)
        atr_multiplier = st.number_input(label="ATR Multiplier", min_value=1.0, max_value=5.0, value=3.2, step=0.1)
        rsi_window = st.number_input(label="RSI Window", min_value=2, max_value=500, value=10, step=1)
    with col2:
        max_step = st.number_input(label="PSAR Max Step", min_value=0.01, max_value=1.0, value=0.4, step=0.01)
        atr_window = st.number_input(label="ATR Window", min_value=1, max_value=100, value=6, step=1)
        window_pivot = st.number_input(label="Min-Max Window", min_value=2, max_value=500, value=10, step=2)
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        macd_short_window = st.number_input(label="MACD Short Window", min_value=1, max_value=100, value=12, step=1)
    with col2:
        macd_long_window = st.number_input(label="MACD Long Window", min_value=1, max_value=100, value=26, step=1)
    with col3:
        macd_signal_window = st.number_input(label="MACD Signal Window", min_value=1, max_value=100, value=9, step=1)

    if st.sidebar.button("SIMULATE"):
        fig, fig_rsi, fig_macd, trades_df, actual_hours = sar_trading_analysis(
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
            macd_short_window=macd_short_window,
            macd_long_window=macd_long_window,
            macd_signal_window=macd_signal_window
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
        fig_macd_placeholder.plotly_chart(fig_macd, use_container_width=True)

    # ------------------------------
    # Parametri fissi per l'ottimizzazione
    # wallet = 1000.0  # Capitale iniziale
    # hours = 4320  # Numero di ore
    # assets = ["AAVEUSDC", "AMPUSDT","ADAUSDC","AVAXUSDC", "BNBUSDC", "BTCUSDC", "DEXEUSDT", "DOGEUSDC", "DOTUSDC",
    #            "ETHUSDC", "LINKUSDC","SOLUSDC", "PEPEUSDC", "RUNEUSDC", "SUIUSDC", "ZENUSDT", "XRPUSDT"]
    # intervals = ["5m","15m"]
    # steps = [0.04]
    # max_steps = [0.4]
    # atr_multipliers = [2.35,2.4,2.45]
    # atr_windows = [3,6,9]
    # dati = download_market_data(assets, intervals, hours)
    # run_simulation(wallet=wallet,
    #                hours=hours,
    #                assets=assets,
    #                intervals=intervals,
    #                steps=steps,
    #                max_steps=max_steps,
    #                atr_multipliers=atr_multipliers,
    #                atr_windows=atr_windows,
    #                market_data=dati)
    # print("Finito.")
