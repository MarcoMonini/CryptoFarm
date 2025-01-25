from binance import ThreadedWebsocketManager, Client
import pandas as pd
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, PSARIndicator, VortexIndicator
import time
import queue
import threading
import os
from datetime import datetime
import keyboard
from colorama import Fore, Back, Style, init
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

init(autoreset=True)


# stampa in debug le informazioni dell'utente collegato tramite API
def print_user_and_wallet_info(client:Client):
    try:
        account_info = client.get_account()
        print(Style.BRIGHT + Fore.GREEN + f"UID: {account_info['uid']}, Tipo: {account_info['accountType']}, "
                                          f"Trade: {'✅' if account_info['canTrade'] else '❌'} ")
        # print(Style.BRIGHT + f"Tipo Account: {account_info['accountType']}")
        # print(f"Permessi: {'✅' if account_info['canTrade'] else '❌'} Trade")
        print(Style.BRIGHT + "Saldo Disponibile:")
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

        return non_zero_balances

    except Exception as e:
        print(Style.BRIGHT + Fore.RED + f"Errore nel recupero delle informazioni: {e}")
        return None


# funzione per ottenere i dati iniziali
def fetch_initial_candles(client:Client, symbol:str, interval:str) -> pd.DataFrame:
    print("Fetching initial candles...")
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=50)
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
            if kline["x"]:
                print(f"@KlineMessage: {datetime.now().strftime("%H:%M:%S")}, {msg['s']}, {data['close']}")

            data_queue.put(data)

    socket_id = twm.start_kline_socket(
        callback=handle_socket_message,
        symbol=symbol,
        interval=interval
    )

    while not stop_event.is_set():
        try:
            time.sleep(1)
        except Exception as e:
            print(f"Errore nella gestione del WebSocket: {e}")

    # while not stop_event.is_set():
    #     try:
    #         run_socket(data_queue, stop_event, symbol, interval)
    #     except Exception as e:
    #         print(f"Errore durante l'esecuzione del WebSocket: {e}")
    #         print("Tentativo di riconnessione in 5 secondi...")
    #         time.sleep(5)  # Attendi prima di riconnettere

    twm.stop_socket(socket_id)
    twm.stop()
    print("WebSocket chiuso.")

# Funzione per inizializzare e avviare il WebSocket
def run_socket_with_reconnect(data_queue, stop_event, symbol: str, interval: str):
    """
    Avvia il WebSocket e gestisce la riconnessione automatica in caso di errore o chiusura.
    """
    while not stop_event.is_set():
        try:
            print(Style.BRIGHT + Fore.YELLOW + "Avvio WebSocket...")
            # Crea un nuovo ThreadedWebsocketManager
            twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
            twm.start()

            # Funzione per gestire i messaggi ricevuti
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
                    if kline["x"]:
                        print(f"@KlineMessage: {datetime.now().strftime("%H:%M:%S")}, {msg['s']}, {data['close']}$")
                    data_queue.put(data)

            # Funzione per gestire la chiusura del WebSocket
            # def handle_socket_close(ws, close_status_code, close_msg):
            #     print(f"WebSocket chiuso. Codice: {close_status_code}, Messaggio: {close_msg}")
            #     if close_status_code == 1001:
            #         print("Chiusura volontaria (1001 - Going Away).")
            #     else:
            #         print("Chiusura inaspettata. Tentativo di riconnessione...")
            # Associa il gestore di chiusura al WebSocket
            #  twm._ws.on_close = handle_socket_close

            # Avvia il WebSocket per il kline
            socket_id = twm.start_kline_socket(
                callback=handle_socket_message,
                symbol=symbol,
                interval=interval
            )

            print(Style.BRIGHT + Fore.GREEN + "WebSocket avviato con successo.")

            # Mantieni il WebSocket attivo fino a quando non viene richiesto di fermarsi
            while not stop_event.is_set():
                time.sleep(1)

            # Ferma il WebSocket in modo pulito
            twm.stop_socket(socket_id)
            twm.stop()
            print(Style.BRIGHT + Fore.RED + "WebSocket terminato.")

        except Exception as e:
            print(Style.BRIGHT + Fore.RED + f"Errore nel WebSocket: {e}")
            print(Style.BRIGHT + Fore.YELLOW + "Tentativo di riconnessione in 5 secondi...")
            time.sleep(5)  # Aspetta prima di tentare di riconnettere


def get_asset_balance(balance, asset):
    # Trova l'asset desiderato
    asset_info = next((item for item in balance if item['asset'] == asset), None)
    # Verifica e stampa i risultati
    if asset_info:
        return asset_info['free']
    else:
        return 0


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


def adjust_quantity(quantity, min_qty, max_qty, step_size):
    """
    Regola la quantità per rispettare i parametri LOT_SIZE.

    Args:
        quantity (float): Quantità disponibile.
        min_qty (float): Quantità minima accettabile.
        max_qty (float): Quantità massima accettabile.
        step_size (float): Incremento consentito.

    Returns:
        float: Quantità regolata per rispettare i parametri.
    """
    # Assicurati che la quantità sia all'interno dei limiti
    print("adjust quantity",quantity,min_qty,max_qty,step_size)
    if quantity < min_qty:
        return 0.0  # Non abbastanza per effettuare un ordine
    if quantity > max_qty:
        quantity = max_qty  # Limita alla quantità massima

    # Arrotolamento alla precisione del stepSize
    precision = len(str(step_size).split(".")[1])  # Numero di cifre decimali di step_size
    adjusted_quantity = (quantity // step_size) * step_size  # Allinea al multiplo inferiore
    return round(adjusted_quantity, precision)


def place_order(client:Client, symbol:str, side:str, order_type:str, quantity:float, price:float=None) -> bool:
    """
    Crea un ordine su Binance.

    Args:
        client (Client): Client di Binance collegato tramite API keys
        symbol (str): Coppia di trading (es. "BTCUSDT").
        side (str): "BUY" o "SELL".
        order_type (str): Tipo di ordine ("MARKET" o "LIMIT").
        quantity (float): Quantità da acquistare/vendere.
        price (float, optional): Prezzo (solo per ordini LIMIT).
    Returns:
        dict: Risposta dell'API Binance.
    """
    try:
        # Per ordini LIMIT è necessario specificare il prezzo
        if order_type == "LIMIT" and price is not None:
            order = client.create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                timeInForce="GTC",  # "Good Till Cancelled"
                quantity=quantity,
                price=price
            )
        elif order_type == "MARKET":
            order = client.create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity
            )
        else:
            print(Style.BRIGHT + Fore.RED + "Errore durante l'esecuzione dell'ordine: Tipo di ordine non supportato.")
            return False

        print(Style.BRIGHT + Fore.CYAN + f"Ordine {side} eseguito con successo:")
        print(Style.BRIGHT + Fore.CYAN + f"  Asset: {order['symbol']}")
        print(Style.BRIGHT + Fore.CYAN + f"  Quantità: {order['fills'][0]['qty']}")
        print(Style.BRIGHT + Fore.CYAN + f"  Commissione: {order['fills'][0]['commission']}")
        print(Style.BRIGHT + Fore.CYAN + f"  Prezzo: {order['fills'][0]['price']}")
        return True
    except Exception as e:
        print(Style.BRIGHT + Fore.RED + f"Errore durante l'esecuzione dell'ordine: {e}")
        return False


API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
print(Style.BRIGHT + f"KEY = {API_KEY}, SECRET = {API_SECRET}")
asset = os.getenv("ASSET", "BTC")
valuta = os.getenv("VALUTA", "USDC")
symbol = asset + valuta
interval = os.getenv("CANDLES_TIME", "15m")
step = float(os.getenv("PSAR_STEP", 0.01))
max_step = float(os.getenv("PSAR_MAX_STEP", 0.4))
atr_multiplier = float(os.getenv("ATR_MULTIPLIER", 2.4))
atr_window = int(os.getenv("ATR_WINDOW", 6))
rsi_window=int(os.getenv("RSI_WINDOW", 12))
macd_long_window=int(os.getenv("MACD_LONG_WINDOW", 26))
macd_short_window=int(os.getenv("MACD_SHORT_WINDOW", 12))
macd_signal_window=int(os.getenv("MACD_SIGNAL_WINDOW", 9))
rsi_buy_limit=float(os.getenv("RSI_BUY_LIMIT", 27))
rsi_sell_limit=float(os.getenv("RSI_SELL_LIMIT", 79))
macd_buy_limit=float(os.getenv("MACD_BUY_LIMIT", -0.44))
macd_sell_limit=float(os.getenv("MACD_SElL_LIMIT", 0.48))
vi_buy_limit=float(os.getenv("VI_BUY_LIMIT", -0.41))
vi_sell_limit=float(os.getenv("VI_SELL_LIMIT", 0.62))
psarvp_buy_limit=float(os.getenv("PSAVP_BUY_LIMIT", 1.01))
psarvp_sell_limit=float(os.getenv("PSARVP_SELL_LIMIT", 0.97))
num_cond=float(os.getenv("NUM_CONDITIONS", 2))

minQty = 0
maxQty = 0
stepQty = 0

client = Client(API_KEY, API_SECRET)
balance = print_user_and_wallet_info(client=client)
asset_balance = get_asset_balance(balance=balance, asset=asset)
usd_balance = get_asset_balance(balance=balance, asset=valuta)
exchange_info = client.get_exchange_info()
symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
if symbol_info:
    for filter_info in symbol_info['filters']:
        if filter_info['filterType'] == 'LOT_SIZE':
            minQty = float(filter_info['minQty'])
            maxQty = float(filter_info['maxQty'])
            stepQty = float(filter_info['stepSize'])
            print(Style.BRIGHT + Fore.GREEN + f"Informazioni per {symbol}:")
            print(f"  MinQty: {minQty}")
            print(f"  MaxQty: {maxQty}")
            print(f"  stepQty: {stepQty}")
else:
    print(Fore.RED + f"La coppia {symbol} non è disponibile.")


df = fetch_initial_candles(client=client, symbol=symbol, interval=interval)
data_queue = queue.Queue()
stop_event = threading.Event()
# socket_thread = threading.Thread(
#             target=run_socket,
#             args=(data_queue,
#                   stop_event,
#                   symbol,
#                   interval),
#             daemon=True
#         )
# socket_thread.start()

# Avvia il WebSocket in un thread separato
socket_thread = threading.Thread(
    target=run_socket_with_reconnect,
    args=(data_queue, stop_event, symbol, interval),
    daemon=True
)
socket_thread.start()

running = True
# buy_signals = []
# sell_signals = []
last_signal_candle_time = None
last_update = None
holding = False

if asset_balance > usd_balance:
    holding = True
else:
    holding = False

# Thread per ascoltare l'input
print(Style.BRIGHT + Fore.YELLOW + "Il Job sta per iniziare.")
print(Style.BRIGHT + "Riepilogo parametri")
print(f" Simbolo: {symbol} ({asset_balance}), holding: {holding}")
print(f" USD disponibili: {usd_balance}")
print(f" Intervallo: {interval}")
print(f" PSAR Step: {step}")
print(f" PSAR Max Step: {max_step}")
print(f" ATR Moltiplicatore: {atr_multiplier}")
print(f" ATR Window: {atr_window}")

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

    # df_copy = df.copy()
    df_copy = add_technical_indicator(df,
                                 step=step,
                                 max_step=max_step,
                                 rsi_window=rsi_window,
                                 macd_long_window=macd_long_window,
                                 macd_short_window=macd_short_window,
                                 macd_signal_window=macd_signal_window,
                                 atr_window=atr_window,
                                 atr_multiplier=atr_multiplier)

    if len(df_copy) > 1:
        i = len(df_copy) - 1
        current_candle_time = df_copy.index[i]
        current_candle_price = df_copy["Close"].iloc[i]

        if last_signal_candle_time != current_candle_time:
            #CONDIZIONI PER IL BUY
            cond_buy_1 = 1 if df_copy['MACD'].iloc[i] <= macd_buy_limit else 0
            cond_buy_2 = 1 if df_copy['RSI'].iloc[i] <= rsi_buy_limit else 0
            cond_buy_3 = 1 if df_copy['VI'].iloc[i] <= vi_buy_limit else 0
            cond_buy_4 = 1 if df_copy['PSARVP'].iloc[i] >= psarvp_buy_limit else 0
            cond_buy_5 = 1 if current_candle_price <= df_copy['Lower_Band'].iloc[i] else 0
            sum_buy = cond_buy_1 + cond_buy_2 + cond_buy_3 + cond_buy_4 + cond_buy_5
            if not holding and sum_buy >= num_cond:
            # if (not holding and df_copy["PSAR"].iloc[i] > current_candle_price
            #        and (current_candle_price <= df_copy["Lower_Band"].iloc[i])):
            #   buy_signals.append((current_candle_time, current_candle_price))
                print(Style.BRIGHT + Fore.GREEN + f"Buy Signal detected at {current_candle_time} and price {current_candle_price}")
                balance = print_user_and_wallet_info(client=client)
                usd_balance = get_asset_balance(balance=balance, asset=valuta)
                quantity = usd_balance / current_candle_price
                adjusted_quantity = adjust_quantity(quantity, minQty, maxQty, stepQty)
                print(Style.BRIGHT + Fore.GREEN + f"Procceding with BUY Order, quantity={adjusted_quantity} (={usd_balance}$)")
                # Piazza l'ordine di acquisto
                response = place_order(client=client,
                                       symbol=symbol,
                                       side="BUY",
                                       order_type="MARKET",
                                       quantity=adjusted_quantity)
                # aspetto e verifico che l'ordine è andato a buon fine
                time.sleep(10)
                balance = print_user_and_wallet_info(client=client)
                asset_balance = get_asset_balance(balance=balance, asset=asset)
                usd_balance = get_asset_balance(balance=balance, asset=valuta)
                if asset_balance > usd_balance:
                    last_signal_candle_time = current_candle_time
                    holding = True
                    print(Style.BRIGHT + f"BUY Order Completed, holding: {holding}")

            # CONDIZIONI PER IL SELL
            cond_sell_1 = 1 if df_copy['MACD'].iloc[i] >= macd_sell_limit else 0
            cond_sell_2 = 1 if df_copy['RSI'].iloc[i] >= rsi_sell_limit else 0
            cond_sell_3 = 1 if df_copy['VI'].iloc[i] >= vi_sell_limit else 0
            cond_sell_4 = 1 if df_copy['PSARVP'].iloc[i] <= psarvp_sell_limit else 0
            cond_sell_5 = 1 if current_candle_price >= df_copy['Upper_Band'].iloc[i] else 0
            sum_sell = cond_sell_1 + cond_sell_2 + cond_sell_3 + cond_sell_4 + cond_sell_5
            if holding and sum_sell >= num_cond:
            # if (holding and df_copy["PSAR"].iloc[i] < current_candle_price
            #       and (current_candle_price >= df_copy["Upper_Band"].iloc[i])):
            #   sell_signals.append((current_candle_time, current_candle_price))
                print(Style.BRIGHT + Fore.RED + f"Sell Signal detected at {current_candle_time} and price {current_candle_price}")
                balance = print_user_and_wallet_info(client=client)
                asset_balance = get_asset_balance(balance=balance, asset=asset)
                adjusted_quantity = adjust_quantity(asset_balance, minQty, maxQty, stepQty)
                print(Style.BRIGHT + Fore.RED + f"Procceding with SELL Order, quantity={adjusted_quantity} (={adjusted_quantity * current_candle_price}$)")
                # Piazza l'ordine di vendita
                response = place_order(client=client,
                                       symbol=symbol,
                                       side="SELL",
                                       order_type="MARKET",
                                       quantity=adjusted_quantity)
                # aspetto e verifico che l'ordine è andato a buon fine
                time.sleep(10)
                balance = print_user_and_wallet_info(client=client)
                asset_balance = get_asset_balance(balance=balance, asset=asset)
                usd_balance = get_asset_balance(balance=balance, asset=valuta)
                if asset_balance < usd_balance:
                    last_signal_candle_time = current_candle_time
                    holding = False
                    print(Style.BRIGHT + f"SELL Order Completed, holding: {holding}")
                # if response:
                #     last_signal_candle_time = current_candle_time
                #     holding = False

# if keyboard.is_pressed('q'):
#     print(Style.BRIGHT + Fore.RED + "\nHai premuto 'q'. Sto terminando il Job...")
#     break

time.sleep(1)

# print(Style.BRIGHT + Fore.GREEN + "Job terminato.")



