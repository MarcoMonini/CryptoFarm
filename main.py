import streamlit as st
from binance import ThreadedWebsocketManager, Client
import pandas as pd
import plotly.graph_objects as go
from ta.trend import PSARIndicator
import time
import queue


# Inserire qui le chiavi API fornite da Binance
api_key = '<api_key>'
api_secret = '<api_secret>'

# Permette all'utente di selezionare l'asset (symbol)
symbol = st.sidebar.selectbox(
    "Seleziona l'asset (symbol)",
    options=["BTCUSDT", "ETHUSDT", "XRPUSDT", "BNBUSDT", "ADAUSDT"],
    index=2  # di default "XRPUSDT"
)

# Permette all'utente di selezionare l'intervallo di tempo (interval)
interval = st.sidebar.selectbox(
    "Seleziona l'intervallo di tempo (candlestick)",
    options=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"],
    index=0  # di default "1m"
)

# Permette all'utente di selezionare i parametri PSAR: step e max_step
step = st.sidebar.number_input(
    "PSAR step",
    min_value=0.001,
    max_value=1.0,
    value=0.02,
    step=0.01
)

max_step = st.sidebar.number_input(
    "PSAR max_step",
    min_value=0.01,
    max_value=1.0,
    value=0.2,
    step=0.01
)

# Permette all'utente di selezionare il tempo di aggiornamento (in secondi)
update_time = st.sidebar.number_input(
    "Tempo di aggiornamento (secondi)",
    min_value=1,
    max_value=60,
    value=2,
    step=1
)


# Coda thread-safe per comunicare tra il WebSocket e il thread principale.
# Conterrà i dati delle candele man mano che arrivano.
data_queue = queue.Queue()

# Inizializza lo stato globale per memorizzare i dati se non esiste già
if "df" not in st.session_state:
    st.session_state["df"] = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

# Inizializza le liste di segnali Buy e Sell
if "buy_signals" not in st.session_state:
    st.session_state["buy_signals"] = []

if "sell_signals" not in st.session_state:
    st.session_state["sell_signals"] = []

# Inizializza la variabile che memorizza il timestamp dell'ultima candela che ha generato un segnale
# (usata per evitare segnali multipli sulla stessa candela)
if "last_signal_candle_time" not in st.session_state:
    st.session_state["last_signal_candle_time"] = None

# Inizializza il client REST di Binance con le chiavi
client = Client(api_key, api_secret)


def fetch_initial_candles():
    """
    Recupera un certo numero di candele storiche da Binance (limite=30)
    per popolare inizialmente il grafico.
    """
    try:
        # Debug: messaggio di log
        print("Fetching initial candles...")

        # Ottiene le candele storiche
        klines = client.get_klines(symbol=symbol, interval=interval, limit=30)

        # Crea una lista di dict, ognuno rappresenta una candela
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

        # Converte in DataFrame
        initial_df = pd.DataFrame(candles)

        # Imposta l'indice sul "tempo di apertura" della candela
        initial_df.set_index("Open time", inplace=True)

        # Debug: stampa le ultime righe recuperate
        print("Fetched initial candles:", initial_df.tail())

        return initial_df
    except Exception as e:
        print(f"Error fetching initial candles: {e}")
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


# Richiama la funzione per ottenere le candele iniziali e aggiorna lo stato globale
st.session_state["df"] = fetch_initial_candles()


def handle_socket_message(msg):
    """
    Viene chiamata automaticamente per ogni nuovo messaggio ricevuto dal WebSocket.
    Se il messaggio contiene una candela (kline), estraiamo i dati principali e li
    inseriamo in una coda condivisa (data_queue) da cui il thread principale li legge.
    """
    # Debug: stampa il nuovo messaggio
    print("New message received:", msg)

    # Controlliamo se il messaggio è di tipo "kline"
    if msg["e"] == "kline":
        kline = msg["k"]

        # Prepara i dati essenziali della candela
        data = {
            "timestamp": pd.to_datetime(kline["t"], unit="ms"),
            "open": float(kline["o"]),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": float(kline["c"]),
            "volume": float(kline["v"]),
            "closed": kline["x"],  # Indica se la candela è chiusa
        }

        # Debug: stampa i dati che stiamo per inserire in coda
        # print(f"Adding to queue: {data}")

        # Inserisce i dati nella coda
        data_queue.put(data)


def start_websocket():
    """
    Crea e avvia il ThreadedWebsocketManager di Binance.
    Avvia il canale per ricevere i dati delle candele in tempo reale.
    Restituisce l'oggetto twm per permettere eventuali altre operazioni in futuro.
    """
    twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
    twm.start()

    # Avvia il WebSocket specifico per le candele (kline)
    twm.start_kline_socket(
        callback=handle_socket_message,
        symbol=symbol,
        interval=interval
    )

    # Debug: stampa che il WebSocket è partito
    print("WebSocket started.")

    return twm


# Avvia il WebSocket una sola volta
twm = start_websocket()

# Crea un placeholder su Streamlit per il grafico
placeholder = st.empty()


# Nota: in un vero ambiente Streamlit, l'uso di "while True:"
#       può entrare in conflitto con la gestione interna dell'evento.
#       Tuttavia, per semplicità qui usiamo un ciclo infinito
#       con time.sleep(update_time).
#       Se necessario, si potrebbe usare st.timeout o una strategia a callback.
while True:
    # 1) CONTROLLA SE LA CODA CONTIENE NUOVI DATI DAL WEBSOCKET
    while not data_queue.empty():
        # Leggi i dati dalla coda
        data = data_queue.get()

        # Debug: stampa i dati in arrivo
        # print(f"Processing data: {data}")

        # Estrai timestamp (che useremo come indice del DataFrame)
        timestamp = data["timestamp"]

        # Aggiorna il DataFrame in session_state:
        #  - se esiste già la riga per quel timestamp, la sovrascrive
        #  - se non esiste, ne crea una nuova
        st.session_state["df"].loc[timestamp] = [
            data["open"],
            data["high"],
            data["low"],
            data["close"],
            data["volume"],
        ]

        # Debug: verifica la parte finale del DataFrame aggiornato
        # print("Updated DataFrame:", st.session_state["df"].tail())

    # 2) LAVORA SUL DATAFRAME (COPIA)
    df = st.session_state["df"].copy()

    # 3) CALCOLA PSAR (se sono presenti almeno 2 candele)
    if len(df) >= 2:
        sar_indicator = PSARIndicator(
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            step=step,
            max_step=max_step
        )
        df["PSAR"] = sar_indicator.psar()

        # 4) GENERA SEGNALI DI BUY/SELL
        #    (controllando che non sia già stato generato un segnale nella medesima candela)
        if len(df) > 1:
            i = len(df) - 1  # l'indice dell'ultima riga
            current_candle_time = df.index[i]

            # Verifichiamo se su questa candela è già stato generato un segnale
            if (st.session_state["last_signal_candle_time"] is None or
                    st.session_state["last_signal_candle_time"] != current_candle_time):

                # Segnale di acquisto: PSAR passa da > Open a < Open
                if df["PSAR"].iloc[i] < df["Open"].iloc[i] and df["PSAR"].iloc[i - 1] > df["Open"].iloc[i - 1]:
                    st.session_state["buy_signals"].append((current_candle_time, df["Close"].iloc[i]))
                    print("Buy signal detected at time", current_candle_time, "price", df["Close"].iloc[i])
                    st.session_state["last_signal_candle_time"] = current_candle_time

                # Segnale di vendita: PSAR passa da < Open a > Open
                elif df["PSAR"].iloc[i - 1] < df["Open"].iloc[i - 1] and df["PSAR"].iloc[i] > df["Open"].iloc[i]:
                    st.session_state["sell_signals"].append((current_candle_time, df["Close"].iloc[i]))
                    print("Sell signal detected at time", current_candle_time, "price", df["Close"].iloc[i])
                    st.session_state["last_signal_candle_time"] = current_candle_time

    # 5) COSTRUISCI IL GRAFICO
    fig = go.Figure()

    # Aggiungi le candele (candlestick)
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Candlestick"
    ))

    # Aggiungi PSAR come punti rossi
    if "PSAR" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["PSAR"],
            mode="markers",
            marker=dict(size=6, color="red", symbol="circle"),
            name="PSAR"
        ))

    # Aggiungi segnali di acquisto (triangoli verdi)
    if len(st.session_state["buy_signals"]) > 0:
        fig.add_trace(go.Scatter(
            x=[s[0] for s in st.session_state["buy_signals"]],
            y=[s[1] for s in st.session_state["buy_signals"]],
            mode="markers",
            marker=dict(size=10, color="green", symbol="triangle-up"),
            name="Buy Signal"
        ))

    # Aggiungi segnali di vendita (triangoli rossi)
    if len(st.session_state["sell_signals"]) > 0:
        fig.add_trace(go.Scatter(
            x=[s[0] for s in st.session_state["sell_signals"]],
            y=[s[1] for s in st.session_state["sell_signals"]],
            mode="markers",
            marker=dict(size=10, color="red", symbol="triangle-down"),
            name="Sell Signal"
        ))

    # 6) CONFIGURA IL LAYOUT DEL GRAFICO
    fig.update_layout(
        title=f"Grafico {symbol} con PSAR",
        xaxis_title="Data e Ora",
        yaxis_title="Prezzo",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600
    )

    # 7) AGGIORNA IL GRAFICO NELLA PAGINA STREAMLIT
    placeholder.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{time.time()}")

    # Debug: log dell'aggiornamento
    print("Updated plot.")

    # 8) ATTESA PRIMA DEL PROSSIMO AGGIORNAMENTO
    time.sleep(update_time)
