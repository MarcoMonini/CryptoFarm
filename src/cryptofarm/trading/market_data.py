"""Scarico delle candele da Binance: REST pubblico, nessuna credenziale.

Estratto da `simulator.py` senza modifiche. `data/klines.py` copre lo store locale dei dump
storici; qui restano le richieste puntuali che la pagina Streamlit fa a ogni interazione."""

import asyncio
import math
import time

import pandas as pd
import streamlit as st
from binance import Client


def interval_to_minutes(interval: str) -> int:
    """
    @brief Converte l'intervallo di Binance (es. "1m", "15m", "1h") in minuti.
    @param interval Stringa che rappresenta l'intervallo (es. "1m", "15m", "1h").
    @return Numero di minuti corrispondenti all'intervallo specificato.
    """
    if interval.endswith("m"):
        # Intervalli tipo "1m", "3m", "15m", "30m", ecc.
        return int(interval.replace("m", ""))
    elif interval.endswith("h"):
        # Intervalli tipo "1h", "2h", ecc.
        hours = int(interval.replace("h", ""))
        return hours * 60
    else:
        # Se non corrisponde a 'm' o 'h', gestisci come preferisci (o ritorna 0)
        return 0


@st.cache_data
def get_market_data(asset: str, interval: str, time_hours: int) -> tuple:
    """
    @brief Scarica i dati di mercato per un asset specifico per un determinato intervallo e periodo.
    @param asset Simbolo dell'asset da scaricare (es. "BTCUSDC").
    @param interval Intervallo tra le candele (es. "1m", "5m", "1h").
    @param time_hours Numero di ore di dati da scaricare.
    @return Una tupla contenente un DataFrame con i dati di mercato e altre informazioni utili.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

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
            endTime=now_ms,  # in ms
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
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]), 0

    # 5. Costruisci il DataFrame da all_klines
    columns = [
        "Open time",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Close time",
        "Quote asset volume",
        "Number of trades",
        "Taker buy base asset volume",
        "Taker buy quote asset volume",
        "Ignore",
    ]
    raw_df = pd.DataFrame(all_klines, columns=columns)

    # 6. Converte i timestamp e imposta l'indice
    raw_df["Open time"] = pd.to_datetime(raw_df["Open time"], unit="ms")
    raw_df.set_index("Open time", inplace=True)

    # Mantieni solo le colonne essenziali, converti a float
    df = raw_df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

    # 7. Ordina per data (dalla più vecchia alla più recente) e rimuovi duplicati
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="first")]  # elimina eventuali duplicati su 'Open time'

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
    @brief Scarica i dati di mercato di un asset per un intervallo temporale specificato.
    @param asset Simbolo dell'asset da scaricare (es. "BTCUSDC").
    @param interval Intervallo tra le candele (es. "1m", "5m", "1h").
    @param start_date Data di inizio (es. "2023-01-01 00:00:00").
    @param end_date Data di fine (es. "2023-01-02 00:00:00").
    @return Una tupla contenente un DataFrame con i dati di mercato e le ore effettive disponibili.
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
            endTime=end_ms,  # in ms
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
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]), 0

    # 4. Costruisci il DataFrame dai dati scaricati
    columns = [
        "Open time",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Close time",
        "Quote asset volume",
        "Number of trades",
        "Taker buy base asset volume",
        "Taker buy quote asset volume",
        "Ignore",
    ]
    raw_df = pd.DataFrame(all_klines, columns=columns)

    # 5. Converte i timestamp e imposta l'indice
    raw_df["Open time"] = pd.to_datetime(raw_df["Open time"], unit="ms")
    raw_df.set_index("Open time", inplace=True)
    df = raw_df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

    # 6. Ordina per data e rimuove eventuali duplicati
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="first")]

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
