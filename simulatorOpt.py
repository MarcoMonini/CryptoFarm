import streamlit as st
import pandas as pd
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, PSARIndicator, VortexIndicator
from simulator import get_market_data, interval_to_minutes

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
    sar = sar_indicator.psar()
    df_copy['PSARVP'] = sar / df_copy['Close']

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
    # Aggiungere le colonne del MACD al DataFrame
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

    return df_copy

def sar_trading_analysis(
        asset: str,
        interval: str,
        wallet: float,
        time_hours: int = 24,
        fee_percent: float = 0.1,  # Commissione % per ogni operazione (buy e sell)
        step: float = 0.001,  # compreso tra 0.001 e 0.1
        max_step: float = 0.4,  # compreso tra 0.1 e 1
        atr_multiplier: float = 2.4,  # Moltiplicatore per le Rolling ATR Bands
        atr_window: int = 6,  # compreso tra 2 e 30
        rsi_window: int = 12,  # compreso tra 2 e 50
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
        market_data: dict = None,
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

    if market_data is None:
        # Scarica i dati di mercato e calcola il SAR
        df, actual_hours = get_market_data(asset=asset, interval=interval, time_hours=time_hours)
    else:
        candlestick_minutes = interval_to_minutes(interval)
        df = market_data
        actual_hours = candlestick_minutes * len(df) / 60

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
        # ------------------------------------------------------------
        cond_buy_1 = 1 if df['MACD'].iloc[i] < macd_buy_limit else 0
        cond_buy_2 = 1 if df['RSI'].iloc[i] < rsi_buy_limit else 0
        cond_buy_3 = 1 if df['VI'].iloc[i] < vi_buy_limit else 0
        cond_buy_4 = 1 if df['PSARVP'].iloc[i] > psarvp_buy_limit else 0
        cond_buy_5 = 1 if df['Low'].iloc[i] < df['Lower_Band'].iloc[i] else 0
        sum_buy = cond_buy_1+cond_buy_2+cond_buy_3+cond_buy_4+cond_buy_5
        if not holding and sum_buy >= num_cond:
            if df['Low'].iloc[i] < df['Lower_Band'].iloc[i]:
                buy_signals.append((df.index[i], float(df['Lower_Band'].iloc[i])))
            else:
                buy_signals.append((df.index[i], float(df['Close'].iloc[i])))
            holding = True
        cond_sell_1 = 1 if df['MACD'].iloc[i] > macd_sell_limit else 0
        cond_sell_2 = 1 if df['RSI'].iloc[i] > rsi_sell_limit else 0
        cond_sell_3 = 1 if df['VI'].iloc[i] > vi_sell_limit else 0
        cond_sell_4 = 1 if df['PSARVP'].iloc[i] < psarvp_sell_limit else 0
        cond_sell_5 = 1 if df['High'].iloc[i] > df['Upper_Band'].iloc[i] else 0
        sum_sell = cond_sell_1+cond_sell_2+cond_sell_3+cond_sell_4+cond_sell_5
        if holding and sum_sell >= num_cond:
            if df['High'].iloc[i] > df['Upper_Band'].iloc[i]:
                sell_signals.append((df.index[i], float(df['Upper_Band'].iloc[i])))
            else:
                sell_signals.append((df.index[i], float(df['Close'].iloc[i])))
            holding = False

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

    print(f"{wallet} su {asset}, profitto totale={round(trades_df['Profit'].sum())}, num_cond={num_cond},"
          f" rsi_sell_limit = {rsi_sell_limit}, rsi_buy_limit = {rsi_buy_limit}, "
          f"macd_buy_limit = {macd_buy_limit}, macd_sell_limit = {macd_sell_limit}, "
          f"vi_buy_limit = {vi_buy_limit}, vi_sell_limit = {vi_sell_limit}, "
          f"sarvp_buy_limit = {psarvp_buy_limit}, psarvp_sell_limit = {psarvp_sell_limit}")

    return trades_df, actual_hours

