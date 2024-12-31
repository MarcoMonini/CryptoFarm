import pandas as pd
import plotly.graph_objects as go
from ta.trend import PSARIndicator
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from binance import Client
import streamlit as st
import numpy as np
import math
import time


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
            limit=chunk_size,      # max 1000
            startTime=fetch_start, # in ms
            endTime=now_ms         # in ms
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
        step: float,
        max_step: float,
        time_hours: int = 24,
        fee_percent: float = 0.1,  # Commissione % per ogni operazione (buy e sell)
        show: bool = True,
        atr_multiplier: float = 1.5,  # Moltiplicatore per le Rolling ATR Bands
        market_data:dict = None
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
    limit : int, optional !!!!!!!!NOOOOOOO!!!!!!!!!
        Numero di candele da scaricare (default 500).
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
    # 1. Scarica i dati di mercato e calcola il SAR
    # ======================================
    if market_data is None:
        # Otteniamo i dati di mercato (funzione esterna da definire)
        # df = get_market_data(asset=asset, interval=interval, limit=limit)
        df, actual_hours = get_market_data(asset=asset, interval=interval, time_hours=time_hours)
    else:
        df = market_data
        actual_hours = time_hours


    print(f"[INFO] Analisi con {wallet} USDC su {asset}, fee={fee_percent}%, {interval}, step={step}, max_step={max_step}, atr_multiplier={atr_multiplier}")

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
        window=14
    )
    df['ATR'] = atr_indicator.average_true_range()

    # SMA (Media Mobile per le Rolling ATR Bands)
    sma_indicator = SMAIndicator(close=df['Close'], window=14)
    df['SMA'] = sma_indicator.sma_indicator()

    # Rolling ATR Bands
    df['Upper_Band'] = df['SMA'] + atr_multiplier * df['ATR']
    df['Lower_Band'] = df['SMA'] - atr_multiplier * df['ATR']

    # RSI
    rsi_indicator = RSIIndicator(
        close=df['Close'],
        window=14
    )
    df['RSI'] = rsi_indicator.rsi()

    # ======================================
    # 2. Identificazione dei segnali di acquisto e vendita
    # ======================================
    buy_signals = []
    sell_signals = []
    holding = False
    upper_trend = False
    lower_trend = False
    for i in range(1, len(df)):

        # Segnale di acquisto: quando il SAR passa da > prezzo a < prezzo (tra candela precedente e attuale)
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

        #upper trend: quando il SAR passa da > prezzo a < prezzo
        if ((df['SAR'].iloc[i] < df['Close'].iloc[i]) and
                (df['SAR'].iloc[i - 1] > df['Close'].iloc[i - 1])):
            upper_trend = True
            lower_trend = False
        # lower trend: quando il SAR passa da < prezzo a > prezzo (tra candela precedente e attuale)
        if ((df['SAR'].iloc[i - 1] < df['Close'].iloc[i - 1]) and
                (df['SAR'].iloc[i] > df['Close'].iloc[i])):
            upper_trend = False
            lower_trend = True

        if (not holding and lower_trend and df['Low'].iloc[i]<df['Lower_Band'].iloc[i]):
            buy_signals.append((df.index[i], float(df['Lower_Band'].iloc[i])))
            holding = True
        if (holding and upper_trend and df['High'].iloc[i]>df['Upper_Band'].iloc[i]):
            sell_signals.append((df.index[i], float(df['Upper_Band'].iloc[i])))
            holding = False


    # ======================================
    # 3. Simulazione di trading con commissioni
    # ======================================
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
            net_proceed = gross_proceed * (1 - fee_decimal)
            # Calcoliamo il profit: differenza fra l'importo netto incassato e l'importo speso in fase di BUY
            cost_in_usdt = quantity * buy_price  # spesa "teorica" (senza la fee di buy)
            profit = net_proceed - cost_in_usdt
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

    # Nota: se alla fine rimaniamo in holding=True, cioè con una crypto comprata
    # ma senza vendere, non vendiamo automaticamente a fine periodo.
    # Se desideri chiudere forzatamente la posizione all'ultima candela,
    # puoi aggiungere una logica extra qui in coda.

    # ======================================
    # 4. Creazione del grafico (Plotly)
    # ======================================
    fig = go.Figure()
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

        # # RSI
        # fig.add_trace(go.Scatter(
        #     x=df.index,
        #     y=df['RSI'],
        #     mode='lines',
        #     line=dict(color='purple', width=2),
        #     name='RSI'
        # ))


        # Segnali di acquisto
        if buy_signals:
            buy_times, buy_prices = zip(*buy_signals)
            fig.add_trace(go.Scatter(
                x=buy_times,
                y=buy_prices,
                mode='markers',
                marker=dict(size=14, color='green', symbol='triangle-up'),
                name='Segnale Acquisto'
            ))

        # Segnali di vendita
        if sell_signals:
            sell_times, sell_prices = zip(*sell_signals)
            fig.add_trace(go.Scatter(
                x=sell_times,
                y=sell_prices,
                mode='markers',
                marker=dict(size=14, color='red', symbol='triangle-down'),
                name='Segnale Vendita'
            ))

        # Layout e aspetto del grafico
        fig.update_layout(
            title=f"Grafico {asset} ({interval}) con SAR e Segnali di Trading",
            xaxis_title="Data e Ora",
            yaxis_title=f"Prezzo ({asset[-4:]})",  # esempio: USDT, USDC, ecc.
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=600
        )

    # ======================================
    # 5. Creazione del DataFrame finale con le operazioni
    # ======================================
    if operations:
        trades_df = pd.DataFrame(operations)
        # Opzionale: profitto cumulato
        # trades_df['CumulativeProfit'] = trades_df['Profit'].cumsum()
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

    # ======================================
    # 6. Restituiamo il grafico e il DataFrame
    # ======================================
    return fig, trades_df, actual_hours


def optimize_parameters_with_cached_data(wallet, hours, assets, intervals, step_range, max_step_range, atr_multiplier_range, dati):
    """
    Ottimizza i parametri di trading usando i dati già scaricati.

    Parameters
    ----------
    wallet : float
        Capitale iniziale in USDT/USDC.
    hours : int
        Numero di ore di dati storici da considerare.
    assets : list
        Lista di asset su cui eseguire le simulazioni.
    intervals : list
        Lista di intervalli di tempo (es. ["1m", "3m", "5m"]).
    step_range : tuple
        Range per il parametro `step` (es. (0.01, 0.1, 0.001)).
    max_step_range : tuple
        Range per il parametro `max_step` (es. (0.1, 1, 0.01)).
    atr_multiplier_range : tuple
        Range per il parametro `atr_multiplier` (es. (1, 4, 0.1)).
    dati : dict
        Dizionario con i dati scaricati.

    Returns
    -------
    pd.DataFrame
        Un DataFrame con tutte le combinazioni di parametri e i risultati.
    dict
        La combinazione di parametri con il profitto massimo.
    """
    results = []
    best_result = None
    max_profit = float('-inf')  # Inizializza il massimo profitto

    # Cicli per ogni combinazione di parametri
    for interval in intervals:
        for step in np.arange(*step_range):
            for max_step in np.arange(*max_step_range):
                for atr_multiplier in np.arange(*atr_multiplier_range):
                    total_profit = 0.0
                    for asset in assets:
                        # Usa i dati già scaricati
                        df = dati.get(asset, {}).get(interval, None)
                        if df is None or df.empty:
                            continue

                        try:
                            # Esegui la simulazione per l'asset corrente
                            _, trades_df, _ = sar_trading_analysis(
                                asset=asset,
                                interval=interval,
                                wallet=wallet,
                                step=step,
                                max_step=max_step,
                                time_hours=hours,
                                show=False,
                                atr_multiplier=atr_multiplier
                            )

                            # Accumula il profitto totale per l'asset
                            if not trades_df.empty:
                                total_profit += trades_df['Profit'].sum()
                        except Exception as e:
                            st.warning(f"Errore durante simulazione per {asset} ({interval}, {step}, {max_step}, {atr_multiplier}): {e}")
                            continue

                    # Salva i risultati della simulazione
                    result = {
                        'Interval': interval,
                        'Step': step,
                        'Max_Step': max_step,
                        'ATR_Multiplier': atr_multiplier,
                        'Total_Profit': total_profit
                    }
                    results.append(result)

                    # Aggiorna il miglior risultato
                    if total_profit > max_profit and total_profit > 0:
                        max_profit = total_profit
                        best_result = result

    # Converti i risultati in un DataFrame
    results_df = pd.DataFrame(results)

    # Ordina per profitto totale
    results_df.sort_values(by='Total_Profit', ascending=False, inplace=True)

    return results_df, best_result


def run_simulation(wallet: float,
                   hours:int,
                   assets:list,
                   intervals:list,
                   steps:list,
                   max_steps:list,
                   atr_multipliers:list,
                   market_data:dict = None):
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
                            'Moltiplicatore ATR':atr_multiplier,
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
        page_title="Simulator",  # Titolo della scheda del browser
        page_icon="📈",  # Icona (grafico che sale, simbolico per un mercato finanziario)
        layout="wide",  # Layout: "centered" o "wide"
        initial_sidebar_state="expanded"  # Stato iniziale della sidebar: "expanded", "collapsed", "auto"
    )
    # ------------------------------
    # fig, trades_df, actual_hours = sar_trading_analysis(
    #     asset='BTCUSDC',
    #     interval='5m',
    #     wallet=1000.0,  # Wallet iniziale in USDT
    #     step=0.04,
    #     max_step=0.4,
    #     time_hours=720,
    #     fee_percent=0.1, # %
    #     atr_multiplier=2.5
    # )
    # st.plotly_chart(fig, use_container_width=True)
    # st.subheader("Resoconto Operazioni")
    # if not trades_df.empty:
    #     st.write(trades_df)
    #     total_profit = trades_df['Profit'].sum()
    #     st.write(f"Profitto Totale: {total_profit:.2f} USDT")
    # else:
    #     st.write("Nessuna operazione effettuata.")
    # ------------------------------

    # Parametri fissi per la simulazione
    # wallet = 1000.0  # Capitale iniziale in USDT/USDC
    # assets = ["AAVEUSDC","AMPUSDT","AVAXUSDC","BTCUSDC","BTTCUSDT","DOGEUSDC","DOTUSDC","ETHUSDC",
    #           "LINKUSDC","PEPEUSDC","PNUTUSDC","RUNEUSDC","SUIUSDC","ZENUSDT","TRXUSDC","XRPUSDT"]
    # steps = [0.06, 0.065, 0.07, 0.075, 0.085, 0.09, 0.095]
    # max_steps = [0.2, 0.4, 0.6, 0.8, 1.0]
    # intervals = ["3m", "5m"]
    # steps = [0.02, 0.04, 0.08]
    # max_steps = [0.2, 0.4, 0.8]
    # atr_multipliers = [2.2, 2.4, 2.6, 2.8]

    # assets = ["AAVEUSDC", "AMPUSDT", "AVAXUSDC", "BTCUSDC", "BTTCUSDT", "DOGEUSDC", "DOTUSDC",
    #           "ETHUSDC", "LINKUSDC", "PEPEUSDC", "PNUTUSDC", "RUNEUSDC", "SUIUSDC", "ZENUSDT",
    #           "TRXUSDC", "XRPUSDT"]
    # intervals = ["1m", "3m", "5m", "15m"]
    # steps = (0.01, 0.1, 0.001)  # Da 0.01 a 0.1 con passi di 0.001
    # max_steps = (0.1, 1.0, 0.01)  # Da 0.1 a 1.0 con passi di 0.01
    # atr_multipliers = (1.0, 4.0, 0.1)  # Da 1 a 4 con passi di 0.1

    # hours = 720 #24ore = 1giorno, 168ore = 1settimana, 720ore = 1mese
    # run_simulation(wallet=wallet,
    #                hours=hours,
    #                assets=assets,
    #                intervals=intervals,
    #                steps=steps,
    #                max_steps=max_steps,
    #                atr_multipliers=atr_multipliers)
    # print("Finito.")

    # Parametri fissi per l'ottimizzazione
    wallet = 1000.0  # Capitale iniziale
    hours = 720  # Numero di ore (1 mese)
    assets = ["AAVEUSDC", "AMPUSDT", "AVAXUSDC", "BTCUSDC", "BTTCUSDT", "DOGEUSDC", "DOTUSDC",
              "ETHUSDC", "LINKUSDC", "PEPEUSDC", "PNUTUSDC", "RUNEUSDC", "SUIUSDC", "ZENUSDT",
              "TRXUSDC", "XRPUSDT"]
    intervals = ["3m", "5m", "15m"]
    steps = [0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]
    max_steps = [0.2,0.4,0.5,0.6,0.7,0.8,0.9]
    atr_multipliers = [2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8]
    dati = download_market_data(assets, intervals, hours)
    run_simulation(wallet=wallet,
                   hours=hours,
                   assets=assets,
                   intervals=intervals,
                   steps=steps,
                   max_steps=max_steps,
                   atr_multipliers=atr_multipliers,
                   market_data=dati)
    print("Finito.")


    # step_range = (0.01, 0.1, 0.005)  # Da 0.01 a 0.1 con passi di 0.001
    # max_step_range = (0.1, 0.8, 0.05)  # Da 0.1 a 1.0 con passi di 0.01
    # atr_multiplier_range = (1.0, 3.0, 0.2)  # Da 1 a 4 con passi di 0.1

    # Scarica tutti i dati necessari
    # st.write("Scaricamento dati di mercato...")
    #
    #
    # # Esegui l'ottimizzazione
    # st.write("Inizio ottimizzazione...")
    # results_df, best_result = optimize_parameters_with_cached_data(
    #     wallet=wallet,
    #     hours=hours,
    #     assets=assets,
    #     intervals=intervals,
    #     step_range=step_range,
    #     max_step_range=max_step_range,
    #     atr_multiplier_range=atr_multiplier_range,
    #     dati=dati
    # )
    #
    # # Mostra i risultati
    # st.title("Risultati dell'Ottimizzazione")
    # st.write("### Miglior Configurazione:")
    # st.write(best_result)
    # st.write("### Tutti i Risultati:")
    # st.dataframe(results_df)

