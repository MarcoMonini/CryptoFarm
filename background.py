from binance import ThreadedWebsocketManager, Client
import pandas as pd
from ta.trend import PSARIndicator, SMAIndicator
from ta.volatility import AverageTrueRange
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


def get_asset_balance(balance, asset):
    # Trova l'asset desiderato
    asset_info = next((item for item in balance if item['asset'] == asset), None)
    # Verifica e stampa i risultati
    if asset_info:
        return asset_info['free']
    else:
        return 0


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
        print(Style.BRIGHT + Fore.CYAN + f" Asset: {order['symbol']}")
        print(Style.BRIGHT + Fore.CYAN + f" Quantità: {order['fills'][0]['qty']}")
        print(Style.BRIGHT + Fore.CYAN + f" Commissione: {order['fills'][0]['commission']}")
        print(Style.BRIGHT + Fore.CYAN + f" Prezzo: {order['fills'][0]['price']}")
        print(Style.BRIGHT + Fore.CYAN + f" Valuta: {order['fills'][0]['commissionAsset']}")
        print(Style.BRIGHT + Fore.CYAN + f" Costo totale: {order['fills'][0]['qty'] * order['fills'][0]['price']}")
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
step = float(os.getenv("PSAR_STEP", 0.04))
max_step = float(os.getenv("PSAR_MAX_STEP", 0.4))
atr_multiplier = float(os.getenv("ATR_MULTIPLIER", 2.4))
atr_window = int(os.getenv("ATR_WINDOW", 6))

# asset = input("Inserisci l'asset da utilizzare (es. BTC): ")
# valuta = input("Inserisci la valuta da utilizzare (USDC/USDT): ")
# symbol = asset + valuta
# interval = input("Inserisci l'intervallo di tempo delle candele (3m, 5m, 15m...): ")
# step = numberInput("Inserisci lo Step per il calcolo del PSAR (consigliato 0.04): ")
# max_step = numberInput("Inserisci il Max Step per il calcolo del PSAR (consigliato 0.4): ")
# atr_multiplier = numberInput("Inserisci il moltiplicatore per l'ATR (consigliato 3.2): ")
# atr_window = int(numberInput("Inserisci la finestra per l'ATR (consigliato 10): "))

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

if asset_balance > usd_balance:
    holding = True
else:
    holding = False

# Thread per ascoltare l'input
print(Style.BRIGHT + Fore.YELLOW + "Il Job sta per inziare. Per terminarlo premi 'q'.")
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

    df_copy = df.copy()

    if len(df_copy) >= atr_window:
        atr_indicator = AverageTrueRange(
            high=df_copy["High"],
            low=df_copy["Low"],
            close=df_copy["Close"],
            window=atr_window
        )
        df_copy["ATR"] = atr_indicator.average_true_range()

        sma_indicator = SMAIndicator(close=df_copy["Close"], window=atr_window)
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
            current_candle_price = df_copy["Close"].iloc[i]

            # print(f"-) Price: {current_candle_price}, PSAR: {df_copy["PSAR"].iloc[i]} ")
            # print(f"-) LowerBand: {df_copy["Lower_Band"].iloc[i]}, UpperBand: {df_copy["Upper_Band"].iloc[i]}")
            # print(f"-) holding: {holding}, Time: {current_candle_time}")

            if last_signal_candle_time != current_candle_time:
                if (not holding and df_copy["PSAR"].iloc[i] > current_candle_price
                        and (current_candle_price <= df_copy["Lower_Band"].iloc[i])):
                    buy_signals.append((current_candle_time, current_candle_price))
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
                    # if response:
                    #     last_signal_candle_time = current_candle_time
                    #     holding = True

                elif (holding and df_copy["PSAR"].iloc[i] < current_candle_price
                      and (current_candle_price >= df_copy["Upper_Band"].iloc[i])):
                    sell_signals.append((current_candle_time, current_candle_price))
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



