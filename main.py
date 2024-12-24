import streamlit as st
from binance import ThreadedWebsocketManager, Client
import pandas as pd
import plotly.graph_objects as go
from ta.trend import PSARIndicator
import time
import queue

# Configura le tue chiavi API Binance
api_key = '<api_key>'
api_secret = '<api_secret>'

# Parametri di configurazione
symbol = "XRPUSDT"
interval = "1m"

# Coda thread-safe per comunicare tra il WebSocket e il thread principale
data_queue = queue.Queue()

# Stato globale per memorizzare i dati
if "df" not in st.session_state:
    st.session_state["df"] = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

if "buy_signals" not in st.session_state:
    st.session_state["buy_signals"] = []

if "sell_signals" not in st.session_state:
    st.session_state["sell_signals"] = []

# Variabile per tenere traccia dell'ultima candela su cui è stato generato un segnale
# None indica nessun segnale generato finora
if "last_signal_candle_time" not in st.session_state:
    st.session_state["last_signal_candle_time"] = None

# Inizializza il client REST per Binance
client = Client(api_key, api_secret)

# Funzione per ottenere le candele precedenti
def fetch_initial_candles():
    try:
        print("Fetching initial candles...")  # Debug
        klines = client.get_klines(symbol=symbol, interval=interval, limit=30)
        candles = []
        for kline in klines:
            candles.append({
                "Open time": pd.to_datetime(kline[0], unit="ms"),
                "Open": float(kline[1]),
                "High": float(kline[2]),
                "Low": float(kline[3]),
                "Close": float(kline[4]),
                "Volume": float(kline[5]),
            })
        initial_df = pd.DataFrame(candles)
        initial_df.set_index("Open time", inplace=True)
        print("Fetched initial candles:", initial_df.tail())  # Debug
        return initial_df
    except Exception as e:
        print(f"Error fetching initial candles: {e}")
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

# Aggiorna il DataFrame iniziale con le candele precedenti
st.session_state["df"] = fetch_initial_candles()

# Funzione per gestire i messaggi del WebSocket
def handle_socket_message(msg):
    print("New message received:", msg)  # Debug

    if msg["e"] == "kline":
        kline = msg["k"]

        # Aggiungi il messaggio ricevuto alla coda
        data = {
            "timestamp": pd.to_datetime(kline["t"], unit="ms"),
            "open": float(kline["o"]),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": float(kline["c"]),
            "volume": float(kline["v"]),
            "closed": kline["x"],  # Flag che indica se la candela è chiusa
        }
        print(f"Adding to queue: {data}")  # Debug
        data_queue.put(data)

# Funzione per avviare il WebSocket
def start_websocket():
    twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
    twm.start()
    # Avvia il WebSocket per le candele
    twm.start_kline_socket(callback=handle_socket_message, symbol=symbol, interval=interval)
    print("WebSocket started.")  # Debug
    return twm

# Avvia il WebSocket
twm = start_websocket()

# Placeholder per il grafico
placeholder = st.empty()

# Loop principale per aggiornare il grafico
while True:
    # Gestisci i dati ricevuti dal WebSocket
    while not data_queue.empty():
        data = data_queue.get()
        print(f"Processing data: {data}")  # Debug

        # Aggiungi o aggiorna i dati nel DataFrame
        timestamp = data["timestamp"]
        st.session_state["df"].loc[timestamp] = [
            data["open"],
            data["high"],
            data["low"],
            data["close"],
            data["volume"],
        ]

        print("Updated DataFrame:", st.session_state["df"].tail())

    # Preleva i dati aggiornati
    df = st.session_state["df"].copy()

    # Calcola PSAR
    if len(df) >= 2:
        sar_indicator = PSARIndicator(high=df["High"], low=df["Low"], close=df["Close"], step=0.02, max_step=0.2)
        df["PSAR"] = sar_indicator.psar()

        # Genera segnali di acquisto e vendita se non già generati nella stessa candela
        if len(df) > 1:
            # i è l'ultimo indice
            i = len(df) - 1
            current_candle_time = df.index[i]

            # Controlliamo se su questa candela è già stato generato un segnale
            if st.session_state["last_signal_candle_time"] is None or st.session_state["last_signal_candle_time"] != current_candle_time:
                # Segnale di acquisto: PSAR passa da > Open a < Open
                if df["PSAR"].iloc[i] < df["Open"].iloc[i] and df["PSAR"].iloc[i - 1] > df["Open"].iloc[i - 1]:
                    st.session_state["buy_signals"].append((current_candle_time, df["Close"].iloc[i]))
                    print("Buy signal detected at time ", current_candle_time," and price", df["Close"].iloc[i])
                    st.session_state["last_signal_candle_time"] = current_candle_time
                # Segnale di vendita: PSAR passa da < Open a > Open
                elif df["PSAR"].iloc[i - 1] < df["Open"].iloc[i - 1] and df["PSAR"].iloc[i] > df["Open"].iloc[i]:
                    st.session_state["sell_signals"].append((current_candle_time, df["Close"].iloc[i]))
                    print("Sell signal detected at time ", current_candle_time," and price", df["Close"].iloc[i])
                    st.session_state["last_signal_candle_time"] = current_candle_time

    # Crea il grafico
    fig = go.Figure()

    # Aggiungi le candele
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Candlestick"
    ))

    # Aggiungi PSAR
    if "PSAR" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["PSAR"],
            mode="markers",
            marker=dict(size=6, color="red", symbol="circle"),
            name="PSAR"
        ))

    # Aggiungi segnali di acquisto
    if len(st.session_state["buy_signals"]) > 0:
        fig.add_trace(go.Scatter(
            x=[s[0] for s in st.session_state["buy_signals"]],
            y=[s[1] for s in st.session_state["buy_signals"]],
            mode="markers",
            marker=dict(size=10, color="green", symbol="triangle-up"),
            name="Buy Signal"
        ))

    # Aggiungi segnali di vendita
    if len(st.session_state["sell_signals"]) > 0:
        fig.add_trace(go.Scatter(
            x=[s[0] for s in st.session_state["sell_signals"]],
            y=[s[1] for s in st.session_state["sell_signals"]],
            mode="markers",
            marker=dict(size=10, color="red", symbol="triangle-down"),
            name="Sell Signal"
        ))

    fig.update_layout(
        title=f"Grafico {symbol} con PSAR",
        xaxis_title="Data e Ora",
        yaxis_title="Prezzo",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600
    )

    # Aggiorna il grafico nel placeholder
    placeholder.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{time.time()}")

    # Debug aggiornamento grafico
    print("Updated plot.")  # Debug

    # Attendi un breve intervallo prima del prossimo aggiornamento
    time.sleep(1)
