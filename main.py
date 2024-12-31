import streamlit as st
from binance import ThreadedWebsocketManager, Client
import pandas as pd
import plotly.graph_objects as go
from ta.trend import PSARIndicator, SMAIndicator
from ta.volatility import AverageTrueRange
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
# print(API_KEY, API_SECRET)
assets = ["AAVEUSDC","AMPUSDT","AVAXUSDC","BTTCUSDT","DOGEUSDC","DOTUSDC",
              "LINKUSDC","PEPEUSDC","RUNEUSDC","SUIUSDC","ZENUSDT"]

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
if "holding" not in st.session_state:
    st.session_state["holding"] = False

###############################################################################
# Funzione che gira in un thread dedicato e ascolta il WebSocket
###############################################################################
def run_socket(data_queue, stop_event, symbol:str, interval:str):
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()

    def handle_socket_message(msg):
        if msg["e"] == "kline":
            kline = msg["k"]
            data = {
                "timestamp": pd.to_datetime(kline["t"], unit="ms"),
                "open": float(kline["o"]),
                "high": float(kline["h"]),
                "low": float(kline["l"]),
                "close": float(kline["c"]),
                "volume": float(kline["v"]),
                "closed": kline["x"],
            }
            data_queue.put(data)

    socket_id = twm.start_kline_socket(
        callback=handle_socket_message,
        symbol=symbol,
        interval=interval
    )

    while not stop_event.is_set():
        time.sleep(0.5)

    twm.stop_socket(socket_id)
    twm.stop()

###############################################################################
# Funzione per avviare il thread (se non esiste già)
###############################################################################
def start_socket_thread(symbol:str, interval:str):
    if "socket_thread" not in st.session_state:
        st.session_state["data_queue"] = queue.Queue()
        st.session_state["stop_event"] = threading.Event()

        thread = threading.Thread(
            target=run_socket,
            args=(st.session_state["data_queue"],
                  st.session_state["stop_event"],
                  symbol,
                  interval),
            daemon=True
        )
        thread.start()

        st.session_state["socket_thread"] = thread

###############################################################################
# Funzione per fermare il thread (se attivo)
###############################################################################
def stop_socket_thread():
    if "socket_thread" in st.session_state:
        st.session_state["stop_event"].set()
        st.session_state["socket_thread"].join()

        del st.session_state["socket_thread"]
        del st.session_state["stop_event"]
        del st.session_state["data_queue"]
        del st.session_state["df"]
        del st.session_state["buy_signals"]
        del st.session_state["sell_signals"]
        del st.session_state["last_signal_candle_time"]

@st.cache_data
def fetch_initial_candles(symbol:str, interval:str) -> pd.DataFrame:
    try:
        klines = st.session_state["client"].get_klines(symbol=symbol, interval=interval, limit=30)
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
        return initial_df
    except Exception as e:
        print(f"Error fetching initial candles: {e}")
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


def display_user_and_wallet_info():
    try:
        account_info = st.session_state["client"].get_account()
        st.sidebar.subheader("Informazioni Utente")
        st.sidebar.write(f"**UID**: {account_info['uid']}")
        st.sidebar.write(f"**Tipo Account**: {account_info['accountType']}")
        st.sidebar.write(f"**Permessi**: {'✅' if account_info['canTrade'] else '❌'} Trade")
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


# SEZIONE PARAMETRI
disabled = False if "socket_thread" not in st.session_state else True
symbol = st.sidebar.selectbox(
    "Seleziona l'asset",
    options=assets,
    index=0,
    disabled=disabled
)

interval = st.sidebar.selectbox(
    "Seleziona l'intervallo di tempo delle candele",
    options=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"],
    index=3,
    disabled=disabled
)

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("START Socket"):
        if "socket_thread" not in st.session_state:
            start_socket_thread(symbol=symbol, interval=interval)
            st.rerun()
    # Permette all'utente di selezionare i parametri PSAR: step e max_step
    step = st.number_input(
        "PSAR step",
        min_value=0.001,
        max_value=1.0,
        value=0.04,
        step=0.01,
        disabled=disabled
    )

with col2:
    if st.button("STOP Socket"):
        if "socket_thread" in st.session_state:
            stop_socket_thread()
    max_step = st.number_input(
            "PSAR Max Step",
            min_value=0.01,
            max_value=1.0,
            value=0.4,
            step=0.01,
            disabled=disabled
        )


atr_multiplier = st.sidebar.number_input(
    "Moltiplicatore ATR",
    min_value=1.0,
    max_value=5.0,
    value=3.2,
    step=0.1,
    disabled=disabled
)

update_time = st.sidebar.number_input(
    "Tempo di aggiornamento (secondi)",
    min_value=1,
    max_value=60,
    value=10,
    step=1
)


placeholder = st.empty()

st.session_state["df"] = fetch_initial_candles(symbol=symbol, interval=interval)

display_user_and_wallet_info()

while True:
    if "data_queue" not in st.session_state:
        continue

    while not st.session_state["data_queue"].empty():
        data = st.session_state["data_queue"].get()
        timestamp = data["timestamp"]
        current_time = datetime.now().strftime("%H:%M:%S")
        st.session_state["last_update"] = current_time
        st.session_state["df"].loc[timestamp] = [
            data["open"],
            data["high"],
            data["low"],
            data["close"],
            data["volume"],
        ]

    df = st.session_state["df"].copy()

    if len(df) >= 14:
        atr_indicator = AverageTrueRange(
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            window=14
        )
        df["ATR"] = atr_indicator.average_true_range()

        sma_indicator = SMAIndicator(close=df["Close"], window=14)
        df["SMA"] = sma_indicator.sma_indicator()
        df["Upper_Band"] = df["SMA"] + atr_multiplier * df["ATR"]
        df["Lower_Band"] = df["SMA"] - atr_multiplier * df["ATR"]

        sar_indicator = PSARIndicator(
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            step=step,
            max_step=max_step
        )
        df["PSAR"] = sar_indicator.psar()

        if len(df) > 1:
            i = len(df) - 1
            current_candle_time = df.index[i]

            if (st.session_state["last_signal_candle_time"] is None or
                    st.session_state["last_signal_candle_time"] != current_candle_time):

                if (not st.session_state["holding"] and
                        df["PSAR"].iloc[i] < df["Close"].iloc[i] and
                        df["Low"].iloc[i] < df["Lower_Band"].iloc[i]):
                    st.session_state["buy_signals"].append((current_candle_time, df["Lower_Band"].iloc[i]))
                    st.session_state["last_signal_candle_time"] = current_candle_time
                    st.session_state["holding"] = True

                elif (st.session_state["holding"] and
                      df["PSAR"].iloc[i] > df["Close"].iloc[i] and
                      df["High"].iloc[i] > df["Upper_Band"].iloc[i]):
                    st.session_state["sell_signals"].append((current_candle_time, df["Upper_Band"].iloc[i]))
                    st.session_state["last_signal_candle_time"] = current_candle_time
                    st.session_state["holding"] = False

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Candlestick"
    ))

    if "PSAR" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["PSAR"],
            mode="markers",
            marker=dict(size=4, color="yellow", symbol="circle"),
            name="PSAR"
        ))

    if "Upper_Band" in df.columns and "Lower_Band" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Upper_Band"],
            mode="lines",
            line=dict(color="red", width=1),
            name="Upper ATR Band"
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Lower_Band"],
            mode="lines",
            line=dict(color="green", width=1),
            name="Lower ATR Band"
        ))

    if len(st.session_state["buy_signals"]) > 0:
        fig.add_trace(go.Scatter(
            x=[s[0] for s in st.session_state["buy_signals"]],
            y=[s[1] for s in st.session_state["buy_signals"]],
            mode="markers",
            marker=dict(size=10, color="green", symbol="triangle-up"),
            name="Buy Signal"
        ))

    if len(st.session_state["sell_signals"]) > 0:
        fig.add_trace(go.Scatter(
            x=[s[0] for s in st.session_state["sell_signals"]],
            y=[s[1] for s in st.session_state["sell_signals"]],
            mode="markers",
            marker=dict(size=10, color="red", symbol="triangle-down"),
            name="Sell Signal"
        ))

    fig.update_layout(
        title=f"Grafico {symbol} con PSAR e ATR Bands, ultimo aggiornamento alle ore {st.session_state['last_update']}",
        xaxis_title="Data e Ora",
        yaxis_title="Prezzo",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600
    )

    placeholder.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{time.time()}")

    time.sleep(update_time)
