import streamlit as st
from binance import ThreadedWebsocketManager, Client
import pandas as pd
import plotly.graph_objects as go
from ta.trend import PSARIndicator
import time
import queue
import threading
import os
from datetime import datetime

# Inserire qui le chiavi API fornite da Binance
# API_KEY = '<api_key>'
# API_SECRET = '<api_secret>'

# Configura il titolo della pagina e il logo
st.set_page_config(
    page_title="CryptoFarm",  # Titolo della scheda del browser
    page_icon="🤑",  # Icona (può essere un emoji o il percorso di un file immagine)
    layout="wide",  # Layout: "centered" o "wide"
    initial_sidebar_state="expanded"  # Stato iniziale della sidebar: "expanded", "collapsed", "auto"
)

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
print(API_KEY, API_SECRET)

# Inizializza le variabili per contenere il socket
if "client" not in st.session_state:
    # Inizializza il client REST di Binance con le chiavi
    st.session_state["client"] = Client(API_KEY, API_SECRET)
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
if "last_update" not in st.session_state:
    st.session_state["last_update"] = None

###############################################################################
# Funzione che gira in un thread dedicato e ascolta il WebSocket
###############################################################################
def run_socket(data_queue, stop_event, symbol:str, interval:str):
    """
    Questo è il "worker thread": si collega a Binance, ascolta i messaggi kline
    e li mette nella data_queue. Rimane attivo finché stop_event non è impostato.
    """
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()

    def handle_socket_message(msg):
        """
            Viene chiamata automaticamente per ogni nuovo messaggio ricevuto dal WebSocket.
            Se il messaggio contiene una candela (kline), estraiamo i dati principali e li
            inseriamo in una coda condivisa (data_queue) da cui il thread principale li legge.
            """
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
            # Inserisce i dati nella coda
            data_queue.put(data)

    # Avviamo il socket delle candele
    socket_id = twm.start_kline_socket(
        callback=handle_socket_message,
        symbol=symbol,
        interval=interval
    )

    # Loop finché non riceviamo lo stop
    while not stop_event.is_set():
        # Dormo leggermente per non occupare la CPU
        time.sleep(0.5)

    # Quando stop_event è True, interrompiamo il socket e il manager
    twm.stop_socket(socket_id)
    twm.stop()

###############################################################################
# Funzione per avviare il thread (se non esiste già)
###############################################################################
def start_socket_thread(symbol:str, interval:str):
    """
    Se il thread non è già attivo, lo crea e lo avvia.
    """
    if "socket_thread" not in st.session_state:
        # Coda dove verranno inseriti i messaggi
        st.session_state["data_queue"] = queue.Queue()
        # Evento per segnalare lo stop
        st.session_state["stop_event"] = threading.Event()

        # Creiamo il thread
        thread = threading.Thread(
            target=run_socket,
            args=(st.session_state["data_queue"],
                  st.session_state["stop_event"],
                  symbol,
                  interval),
            daemon=True  # il thread si chiude se l'app si chiude
        )
        # Avviamo il thread
        thread.start()

        # Salviamo il riferimento al thread in session_state
        st.session_state["socket_thread"] = thread

###############################################################################
# Funzione per fermare il thread (se attivo)
###############################################################################
def stop_socket_thread():
    """
    Imposta lo stop_event, aspetta che il thread finisca e rimuove tutto da session_state.
    """
    if "socket_thread" in st.session_state:
        # Manda il segnale di STOP
        st.session_state["stop_event"].set()
        # Aspetta che il thread muoia
        st.session_state["socket_thread"].join()

        # Pulizia delle chiavi
        if "socket_thread" in st.session_state:
            del st.session_state["socket_thread"]
        if "stop_event" in st.session_state:
            del st.session_state["stop_event"]
        if "data_queue" in st.session_state:
            del st.session_state["data_queue"]
        if "df" in st.session_state:
            del st.session_state["df"]
        if "buy_signals" in st.session_state:
            del st.session_state["buy_signals"]
        if "sell_signals" in st.session_state:
            del st.session_state["sell_signals"]
        if "last_signal_candle_time" in st.session_state:
            del st.session_state["last_signal_candle_time"]


@st.cache_data
def fetch_initial_candles(symbol:str, interval:str) -> pd.DataFrame:
    """
    Recupera un certo numero di candele storiche da Binance (limite=30)
    per popolare inizialmente il grafico.
    """
    try:
        # Debug: messaggio di log
        print("Fetching initial candles...")

        # Ottiene le candele storiche
        klines = st.session_state["client"].get_klines(symbol=symbol, interval=interval, limit=30)

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
        # print("Fetched initial candles:", initial_df.tail())
        return initial_df
    except Exception as e:
        print(f"Error fetching initial candles: {e}")
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


# Aggiungi una sezione per le informazioni dell'utente e del wallet
def display_user_and_wallet_info():
    try:
        # Ottieni i dettagli dell'account
        account_info = st.session_state["client"].get_account()

        # Informazioni sull'utente
        st.sidebar.subheader("Informazioni Utente")
        st.sidebar.write(f"**UID**: {account_info['uid']}")
        st.sidebar.write(f"**Tipo Account**: {account_info['accountType']}")
        st.sidebar.write(f"**Permessi**: {'✅' if account_info['canTrade'] else '❌'} Trade")
        st.sidebar.write(f"**Commissioni**: {account_info['makerCommission'] / 100:.2}%")
        # st.sidebar.write(f"- Maker: {account_info['makerCommission'] / 100:.2}%")
        # st.sidebar.write(f"- Taker: {account_info['takerCommission'] / 100:.2}%")

        # Informazioni sul wallet
        st.sidebar.subheader("Saldo Disponibile")
        balances = account_info.get("balances", [])
        non_zero_balances = [
            {"asset": b["asset"], "free": float(b["free"]), "locked": float(b["locked"])}
            for b in balances if float(b["free"]) > 0 or float(b["locked"]) > 0
        ]
        if non_zero_balances:
            for balance in non_zero_balances:
                st.sidebar.write(f"- {balance['free']} **{balance['asset']}**")
        else:
            st.sidebar.write("Nessun saldo disponibile.")
    except Exception as e:
        st.sidebar.error(f"Errore nel recupero delle informazioni: {e}")


disabled = False if "socket_thread" not in st.session_state else True

# SEZIONE PARAMETRI
# Permette all'utente di selezionare l'asset (symbol)
symbol = st.sidebar.selectbox(
    "Seleziona l'asset (symbol)",
    options=["AAVEUSDC", "ADAUSDT", "AMPUSDT", "AVAXUSDC", "BTCUSDC", "DOGEUSDC","ETHUSDT", "PNUTUSDC", "TRXUSDC", "XRPUSDT"],
    index=3,  # default
    disabled=disabled
)

# Permette all'utente di selezionare l'intervallo di tempo (interval)
interval = st.sidebar.selectbox(
    "Seleziona l'intervallo di tempo (candlestick)",
    options=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"],
    index=0,  # di default "1m"
    disabled=disabled
)

# Bottoni Start/Stop
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("START Socket"):
        if "socket_thread" not in st.session_state:
            start_socket_thread(symbol=symbol, interval=interval)
            print("Socket avviato!")
            st.rerun()
        else:
            print("Il socket è già in esecuzione.")

with col2:
    if st.button("STOP Socket"):
        if "socket_thread" in st.session_state:
            stop_socket_thread()
            print("Socket fermato!")
            # st.rerun()
        else:
            print("Nessun socket da fermare.")

# Permette all'utente di selezionare i parametri PSAR: step e max_step
step = st.sidebar.number_input(
    "PSAR step",
    min_value=0.001,
    max_value=1.0,
    value=0.07,
    step=0.01,
    disabled=disabled
)

max_step = st.sidebar.number_input(
    "PSAR max_step",
    min_value=0.01,
    max_value=1.0,
    value=0.4,
    step=0.01,
    disabled=disabled
)

# Permette all'utente di selezionare il tempo di aggiornamento (in secondi)
update_time = st.sidebar.number_input(
    "Tempo di aggiornamento (secondi)",
    min_value=1,
    max_value=60,
    value=10,
    step=1,
    # disabled=disabled
)

# Richiama la funzione per ottenere le candele iniziali e aggiorna lo stato globale
st.session_state["df"] = fetch_initial_candles(symbol=symbol, interval=interval)

# Richiama la funzione per mostrare le informazioni
display_user_and_wallet_info()

# Crea un placeholder su Streamlit per il grafico
placeholder = st.empty()


# Main loop per aggiornare il grafico
while True:
    if "data_queue" not in st.session_state:
        continue
    # CONTROLLA SE LA CODA CONTIENE NUOVI DATI DAL WEBSOCKET
    while not st.session_state["data_queue"].empty():
        # Leggi i dati dalla coda
        data = st.session_state["data_queue"].get()
        # Estrai timestamp (che useremo come indice del DataFrame)
        timestamp = data["timestamp"]
        # Ottieni l'orario corrente in formato leggibile
        current_time = datetime.now().strftime("%H:%M:%S")
        st.session_state["last_update"] = current_time
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

    # LAVORA SUL DATAFRAME (COPIA)
    df = st.session_state["df"].copy()

    # CALCOLA PSAR (se sono presenti almeno 2 candele)
    if len(df) >= 2:
        sar_indicator = PSARIndicator(
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            step=step,
            max_step=max_step
        )
        df["PSAR"] = sar_indicator.psar()

        # GENERA SEGNALI DI BUY/SELL
        if len(df) > 1:
            i = len(df) - 1  # l'indice dell'ultima riga
            current_candle_time = df.index[i]
            current_price = df["Close"].iloc[i]
            print(current_candle_time, current_price)

            # Verifichiamo se su questa candela è già stato generato un segnale
            if (st.session_state["last_signal_candle_time"] is None or
                    st.session_state["last_signal_candle_time"] != current_candle_time):

                # # Segnale di acquisto: PSAR passa da > Open a < Open
                # if df["PSAR"].iloc[i] < df["Close"].iloc[i] and df["PSAR"].iloc[i - 1] > df["Close"].iloc[i - 1]:
                #     st.session_state["buy_signals"].append((current_candle_time, df["Close"].iloc[i]))
                #     print("Buy signal detected at time", current_candle_time, "price", df["Close"].iloc[i])
                #     st.session_state["last_signal_candle_time"] = current_candle_time
                #
                # # Segnale di vendita: PSAR passa da < Open a > Open
                # elif df["PSAR"].iloc[i - 1] < df["Close"].iloc[i - 1] and df["PSAR"].iloc[i] > df["Close"].iloc[i]:
                #     st.session_state["sell_signals"].append((current_candle_time, df["Close"].iloc[i]))
                #     print("Sell signal detected at time", current_candle_time, "price", df["Close"].iloc[i])
                #     st.session_state["last_signal_candle_time"] = current_candle_time

                # Segnale di acquisto:
                # PSAR passa da > prezzo a < prezzo e prezzo attuale (Close[i]) < Close[i-1]
                if df["PSAR"].iloc[i] < df["Close"].iloc[i] <= df["Close"].iloc[i - 1] < df["PSAR"].iloc[i - 1]:
                    st.session_state["buy_signals"].append((current_candle_time, df["Close"].iloc[i]))
                    print("Buy signal detected at time", current_candle_time, "price", df["Close"].iloc[i])
                    st.session_state["last_signal_candle_time"] = current_candle_time

                # Segnale di vendita:
                # PSAR passa da < prezzo a > prezzo e prezzo attuale (Close[i]) > Close[i-1]
                elif df["PSAR"].iloc[i - 1] < df["Close"].iloc[i - 1] <= df["Close"].iloc[i] < df["PSAR"].iloc[i]:
                    st.session_state["sell_signals"].append((current_candle_time, df["Close"].iloc[i]))
                    print("Sell signal detected at time", current_candle_time, "price", df["Close"].iloc[i])
                    st.session_state["last_signal_candle_time"] = current_candle_time

    # COSTRUISCI IL GRAFICO
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

    # CONFIGURA IL LAYOUT DEL GRAFICO
    fig.update_layout(
        title=f"Grafico {symbol} con PSAR, ultimo aggiornamento alle ore {st.session_state["last_update"]}",
        xaxis_title="Data e Ora",
        yaxis_title="Prezzo",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600
    )

    # AGGIORNA IL GRAFICO NELLA PAGINA STREAMLIT
    placeholder.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{time.time()}")

    # Debug: log dell'aggiornamento
    print("Updated plot.")

    # ATTESA PRIMA DEL PROSSIMO AGGIORNAMENTO
    time.sleep(update_time)
