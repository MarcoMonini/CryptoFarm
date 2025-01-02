# import streamlit as st
from binance import ThreadedWebsocketManager, Client
import pandas as pd
# from streamlit import number_input
# import plotly.graph_objects as go
from ta.trend import PSARIndicator, SMAIndicator
from ta.volatility import AverageTrueRange
import time
import queue
import threading
import os
from datetime import datetime
import keyboard
from colorama import Fore, Back, Style, init

init(autoreset=True)


# stampa in debug le informazioni dell'utente collegato tramite API
def print_user_and_wallet_info(client:Client):
    try:
        account_info = client.get_account()
        print(Style.BRIGHT + Fore.GREEN + f"UID: {account_info['uid']}")
        print(Style.BRIGHT + f"Tipo Account: {account_info['accountType']}")
        print(f"Permessi: {'✅' if account_info['canTrade'] else '❌'} Trade")
        print(Style.BRIGHT + "Saldo Disponibile")
        balances = account_info.get("balances", [])
        non_zero_balances = [
            {"asset": b["asset"], "free": float(b["free"]), "locked": float(b["locked"])}
            for b in balances if float(b["free"]) > 0 or float(b["locked"]) > 0
        ]
        if non_zero_balances:
            for balance in non_zero_balances:
                print(Style.BRIGHT + Fore.GREEN + f"- {balance['free']} {balance['asset']}")
        else:
            print(Style.BRIGHT + Fore.RED +"Nessun saldo disponibile.")
    except Exception as e:
        print(Style.BRIGHT + Fore.RED + f"Errore nel recupero delle informazioni: {e}")


# funzione per ottenere i dati iniziali
def fetch_initial_candles(client:Client, symbol:str, interval:str) -> pd.DataFrame:
    print("Fetching initial candles...")
    try:
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
        print("Initial candles fetched correctly.")
        return initial_df
    except Exception as e:
        print(Style.BRIGHT + Fore.RED + f"Error fetching initial candles: {e}")
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


# Funzione che gira in un thread dedicato e ascolta il WebSocket
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
            print(f"@KlineMessage: {datetime.now().strftime("%H:%M:%S")}, {msg['s']}, {data['close']}$")
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


def numberInput(placeholder:str) -> float:
    while True:
        try:
            user_input = float(input(placeholder))
            # Controlla che sia maggiore di 0
            if user_input > 0:
                print(Fore.GREEN + f"Hai inserito un valore valido: {user_input}")
                return user_input
            else:
                print(Fore.RED + "Errore: Per favore, inserisci un numero valido.")
        except ValueError:
            print(Fore.RED + "Errore: Per favore, inserisci un numero valido.")

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
print(Style.BRIGHT + f"KEY = {API_KEY}, SECRET = {API_SECRET}")

# Inizializza e avvia il Thread per il socket collegato a binance
symbol = input("Inserisci l'asset da utilizzare (es. BTCUSDC): ")
interval = input("Inserisci l'intervallo di tempo delle candele (3m, 5m, 15m...): ")
step = numberInput("Inserisci lo Step per il calcolo del PSAR (consigliato 0.04): ")
max_step = numberInput("Inserisci il Max Step per il calcolo del PSAR (consigliato 0.4): ")
atr_multiplier = numberInput("Inserisci il moltiplicatore per l'ATR (consigliato 3.2): ")
# df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

client = Client(API_KEY, API_SECRET)
print_user_and_wallet_info(client=client)
df = fetch_initial_candles(client=client, symbol=symbol, interval=interval)

data_queue = queue.Queue()
stop_event = threading.Event()
socket_thread = threading.Thread(
            target=run_socket,
            args=(data_queue,
                  stop_event,
                  symbol,
                  interval),
            daemon=True
        )
socket_thread.start()
running = True

buy_signals = []
sell_signals = []
last_signal_candle_time = None
last_update = None
holding = False


# # Thread per ascoltare l'input
# listener = threading.Thread(target=wait_for_exit)
# listener.start()
print(Style.BRIGHT + Fore.YELLOW + "Il Job sta per inziare. Per terminarlo premi 'q'.")

while True:
    while not data_queue.empty():
        data = data_queue.get()
        timestamp = data["timestamp"]
        current_time = datetime.now().strftime("%H:%M:%S")
        last_update = current_time
        df.loc[timestamp] = [
            data["open"],
            data["high"],
            data["low"],
            data["close"],
            data["volume"],
        ]

    df_copy = df.copy()

    if len(df_copy) >= 14:
        atr_indicator = AverageTrueRange(
            high=df_copy["High"],
            low=df_copy["Low"],
            close=df_copy["Close"],
            window=14
        )
        df_copy["ATR"] = atr_indicator.average_true_range()

        sma_indicator = SMAIndicator(close=df_copy["Close"], window=14)
        df_copy["SMA"] = sma_indicator.sma_indicator()
        df_copy["Upper_Band"] = df_copy["SMA"] + atr_multiplier * df_copy["ATR"]
        df_copy["Lower_Band"] = df_copy["SMA"] - atr_multiplier * df_copy["ATR"]

        sar_indicator = PSARIndicator(
            high=df_copy["High"],
            low=df_copy["Low"],
            close=df_copy["Close"],
            step=step,
            max_step=max_step
        )
        df_copy["PSAR"] = sar_indicator.psar()

        if len(df_copy) > 1:
            i = len(df_copy) - 1
            current_candle_time = df_copy.index[i]

            if last_signal_candle_time != current_candle_time:
                if (not holding and
                        df_copy["PSAR"].iloc[i] < df_copy["Close"].iloc[i] and
                        df_copy["Low"].iloc[i] < df_copy["Lower_Band"].iloc[i]):
                    buy_signals.append((current_candle_time, df_copy["Close"].iloc[i]))
                    print(Style.BRIGHT + Fore.GREEN + f"Buy Signal detected at {current_candle_time} and price {df_copy["Close"].iloc[i]}")
                    last_signal_candle_time = current_candle_time
                    holding = True

                elif (holding and
                      df_copy["PSAR"].iloc[i] > df_copy["Close"].iloc[i] and
                      df_copy["High"].iloc[i] > df_copy["Upper_Band"].iloc[i]):
                    sell_signals.append((current_candle_time, df_copy["Close"].iloc[i]))
                    print(Style.BRIGHT + Fore.RED + f"Sell Signal detected at {current_candle_time} and price {df_copy["Close"].iloc[i]}")
                    last_signal_candle_time = current_candle_time
                    holding = False

    if keyboard.is_pressed('q'):
        print(Style.BRIGHT + Fore.RED + "\nHai premuto 'q'. Sto terminando il Job...")
        break

    time.sleep(1)

print(Style.BRIGHT + Fore.GREEN + "Job terminato.")



