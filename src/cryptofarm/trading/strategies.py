"""Le strategie: da una tabella di candele con indicatori a due liste di segnali.

Estratte da `simulator.py` senza modifiche. Ognuna restituisce `(buy_signals, sell_signals)`,
liste di `(timestamp, prezzo)` che `pnl.py` trasforma in operazioni.

Attenzione: `buy_sell_limits_simulation`, `atr_buy_sell_simulation` e
`close_atr_buy_sell_simulation` leggono le colonne `MACD` e `PSAR`, che
`indicators.add_technical_indicator` non produce piu' (i calcoli sono commentati). Sollevano
`KeyError` appena chiamate, ed e' cosi' anche prima di questa riorganizzazione."""

import numpy as np
import pandas as pd
import streamlit as st

from cryptofarm.ml.signals import barrier_signals, meta_signals, policy_signals
from cryptofarm.ml.trainer import active_model_name, meta_parameters, stored_decision_threshold
from cryptofarm.trading.indicators import calculate_latest_indicators


@st.cache_data
def simulate_candles(
    raw_df,
    atr_window: int = 6,
    atr_multiplier: float = 2,
    step: float = 0.01,
    max_step: float = 0.4,
    stop_loss_percent: float = 99.0,
):
    """
    raw_df: DataFrame con colonne ['Open', 'High', 'Low', 'Close', 'Volume']
    Altre parametri: come nel tuo add_technical_indicator.
    stop_loss: % di stop loss
    strategia: stringa per la strategia
    """

    df = raw_df.copy()

    # Inizializza strutture per salvare i segnali
    buy_signals = []
    sell_signals = []
    holding = False
    last_signal_candle_index = -1  # inizialmente
    # variabili per lo Stop Loss
    stop_loss_price = None
    got_stop_loss = False
    stop_loss_decimal = stop_loss_percent / 100

    # Per evitare di ricalcolare da zero,
    # puoi calcolare UNA SOLA volta gli indicatori "storici" fino alla prima candela.
    # Tuttavia, nella logica semplificata, rifaremo "add_technical_indicator" in ogni step.

    # Loop sulle candele
    for i in range(len(df)):

        o = df["Open"].iloc[i]
        h = df["High"].iloc[i]
        l = df["Low"].iloc[i]
        c = df["Close"].iloc[i]

        n_steps = 10
        is_green = c >= o

        # Verifichiamo se è candela verde o rossa
        # (puoi anche decidere con un'altra logica, es: "Close >= Open => verde" di default)
        # Definiamo i 3 segmenti e costruiamo i 30 step di prezzo
        # Candela verde: open -> low (10 step), low -> high (10 step), high -> close (10 step)
        # Candela rossa: open -> high (10 step), high -> low (10 step), low -> close (10 step)
        # Per comodità, uso una piccola funzione di supporto per generare "n" step dal prezzo A al prezzo B
        def linspace_steps(a, b, n=n_steps):
            return np.linspace(a, b, n, endpoint=False)[1:]  # escludiamo la "prima" perché corrisponde a a

        prices_sequence = []
        if is_green:
            # Segmento 1: open -> low
            segment1 = linspace_steps(o, l, n=int(n_steps / 2))
            # Segmento 2: low -> high
            segment2 = linspace_steps(l, h, n=int(n_steps * 2))
            # Segmento 3: high -> close
            segment3 = linspace_steps(h, c, n=int(n_steps / 2))
        else:
            # Segmento 1: open -> high
            segment1 = linspace_steps(o, h, n=int(n_steps / 2))
            # Segmento 2: high -> low
            segment2 = linspace_steps(h, l, n=int(n_steps * 2))
            # Segmento 3: low -> close
            segment3 = linspace_steps(l, c, n=int(n_steps / 2))
        prices_sequence = list(segment1) + list(segment2) + list(segment3)

        # A questo punto abbiamo 3*9 = 27 prezzi intermedi,
        # se vogliamo esattamente 30 step (includendo anche l'ultimo?),
        # possiamo aggiungere l'ultimo prezzo "Close" come step finale,
        # così da totalizzare 28 (oppure gestire diversamente).
        # Per semplicità, qui aggiungo manualmente l'ultimo step = c
        # (ma dipende da come preferisci gestire i conti).
        prices_sequence.append(c)
        # Inizializza i valori "in costruzione" della candela:
        step_open = o
        step_high = o
        step_low = o
        step_close = o

        # Ora eseguiamo la simulazione step-by-step
        for price in prices_sequence:
            # Aggiorniamo SOLO l'ultima candela con un "Close" fittizio = price
            # e lasciamo invariati Open, High, Low "finali" della candela,
            # in modo che eventuali indicatori che usano 'High', 'Low'
            # vedano la candela 'per intero'.

            # Aggiorna i valori di High e Low dinamicamente
            if price > step_high:
                step_high = price
            if price < step_low:
                step_low = price
            # Aggiorna la chiusura
            step_close = price

            temp_df = df.copy()
            # Sovrascrivi sulla candela i-esima i valori dinamici
            temp_df.at[temp_df.index[i], "Open"] = step_open
            temp_df.at[temp_df.index[i], "High"] = step_high
            temp_df.at[temp_df.index[i], "Low"] = step_low
            temp_df.at[temp_df.index[i], "Close"] = step_close

            df_utile = calculate_latest_indicators(
                i=i, df=temp_df, atr_window=atr_window, atr_multiplier=atr_multiplier
            )

            row = df_utile.iloc[-1]
            # Condizione di BUY
            if (
                not holding
                and last_signal_candle_index != i
                and row["Lower_Band"] is not None
                and row["Close"] <= row["Lower_Band"]
                and not (got_stop_loss and row["PSAR"] is not None and row["PSAR"] > row["Close"])
            ):
                buy_signals.append((df.index[i], float(row["Close"])))
                holding = True
                last_signal_candle_index = i
                got_stop_loss = False
                stop_loss_price = float(row["Close"]) * (1 - stop_loss_decimal)
            # Condizione di SELL
            if (
                holding
                and last_signal_candle_index != i
                and row["Upper_Band"] is not None
                and row["Close"] >= row["Upper_Band"]
            ):
                sell_signals.append((df.index[i], float(row["Close"])))
                holding = False
                last_signal_candle_index = i
                stop_loss_price = None
                got_stop_loss = False
            # Condizione STOP LOSS
            if (
                holding
                and stop_loss_price is not None
                and row["Close"] < stop_loss_price
                and row["PSAR"] > row["Close"]
            ):
                sell_signals.append((df.index[i], float(row["Close"])))
                holding = False
                last_signal_candle_index = i
                got_stop_loss = True
                stop_loss_price = None

    return buy_signals, sell_signals


def buy_sell_limits_simulation(df, macd_buy_limit, macd_sell_limit, rsi_buy_limit, rsi_sell_limit, num_cond):
    buy_signals = []
    sell_signals = []
    holding = False

    for i in range(1, len(df)):
        # CONDIZIONI DI BUY
        cond_buy_macd = 1 if df["MACD"].iloc[i] <= macd_buy_limit else 0
        # cond_buy_macd2 = 1 if df['MACD'].iloc[i] > df['MACD'].tail(
        #     10).min() else 0  # il MACD ha invertito direzione
        cond_buy_rsi = 1 if df["RSI"].iloc[i] <= rsi_buy_limit else 0
        # cond_buy_vi = 1 if df['VI'].iloc[i] <= vi_buy_limit else 0
        # cond_buy_psarvp = 1 if df['PSARVP'].iloc[i] >= psarvp_buy_limit else 0
        # cond_buy_atr = 1 if df['Low'].iloc[i] <= df['Lower_Band'].iloc[i] else 0
        # cond_buy_srsi = 1 if df['StochRSI'].iloc[i] <= srsi_buy_limit else 0
        # cond_buy_tsi = 1 if df['TSI'].iloc[i] <= tsi_buy_limit else 0
        # cond_buy_roc = 1 if df['ROC'].iloc[i] <= roc_buy_limit else 0
        # cond_buy_pvo = 1 if df['PVO'].iloc[i] <= pvo_buy_limit else 0
        # cond_buy_mfi = 1 if df['MFI'].iloc[i] <= mfi_buy_limit else 0
        sum_buy = cond_buy_macd + cond_buy_rsi
        # + cond_buy_vi + cond_buy_psarvp + cond_buy_atr + cond_buy_srsi +
        # cond_buy_tsi + cond_buy_roc + cond_buy_pvo + cond_buy_mfi)
        if not holding and sum_buy >= num_cond:
            if df["Low"].iloc[i] < df["Lower_Band"].iloc[i]:
                buy_signals.append((df.index[i], float(df["Lower_Band"].iloc[i])))
            else:
                buy_signals.append((df.index[i], float(df["Close"].iloc[i])))
            holding = True
        # CONDIZIONI DI SELL
        cond_sell_macd = 1 if df["MACD"].iloc[i] >= macd_sell_limit else 0
        # cond_sell_macd2 = 1 if df['MACD'].iloc[i] < df['MACD'].tail(
        #    10).max() else 0  # il MACD ha invertito direzione
        cond_sell_rsi = 1 if df["RSI"].iloc[i] >= rsi_sell_limit else 0
        # cond_sell_vi = 1 if df['VI'].iloc[i] >= vi_sell_limit else 0
        # cond_sell_psavp = 1 if df['PSARVP'].iloc[i] <= psarvp_sell_limit else 0
        # cond_sell_atr = 1 if df['High'].iloc[i] >= df['Upper_Band'].iloc[i] else 0
        # cond_sell_srsi = 1 if df['StochRSI'].iloc[i] >= srsi_sell_limit else 0
        # cond_sell_tsi = 1 if df['TSI'].iloc[i] >= tsi_sell_limit else 0
        # cond_sell_roc = 1 if df['ROC'].iloc[i] >= roc_sell_limit else 0
        # cond_sell_pvo = 1 if df['PVO'].iloc[i] >= pvo_sell_limit else 0
        # cond_sell_mfi = 1 if df['MFI'].iloc[i] >= mfi_sell_limit else 0
        sum_sell = cond_sell_macd + cond_sell_rsi
        # + cond_sell_vi + cond_sell_psavp + cond_sell_atr +
        # cond_sell_srsi + cond_sell_tsi + cond_sell_roc + cond_sell_pvo + cond_sell_mfi)
        if holding and sum_sell >= num_cond:
            if df["High"].iloc[i] > df["Upper_Band"].iloc[i]:
                sell_signals.append((df.index[i], float(df["Upper_Band"].iloc[i])))
            else:
                sell_signals.append((df.index[i], float(df["Close"].iloc[i])))
            holding = False

    return buy_signals, sell_signals


def buy_sell_limits_close_simulation(
    df,
    rsi_buy_limit: int = 25,
    rsi_sell_limit: int = 75,
    macd_buy_limit: float = -2.5,
    macd_sell_limit: float = 2.5,
    num_cond: int = 1,
    stop_loss_percent: float = 99.0,
):
    buy_signals = []
    sell_signals = []
    holding = False
    last_signal_candle_index = -1
    # stop_loss_price = None
    # got_stop_loss = False
    # stop_loss_decimal = stop_loss_percent / 100

    for i in range(1, len(df)):
        # CONDIZIONI DI BUY
        cond_buy_atr = 1 if df["Close"].iloc[i] <= df["Lower_Band"].iloc[i] else 0
        cond_buy_rsi = 1 if df["RSI"].iloc[i] <= rsi_buy_limit else 0
        sum_buy = cond_buy_rsi + cond_buy_atr
        if not holding and last_signal_candle_index != i and sum_buy >= num_cond:
            buy_signals.append((df.index[i], float(df["Close"].iloc[i])))
            holding = True
            last_signal_candle_index = i
        # CONDIZIONI DI SELL
        cond_sell_rsi = 1 if df["RSI"].iloc[i] >= rsi_sell_limit else 0
        cond_sell_atr = 1 if df["Close"].iloc[i] >= df["Upper_Band"].iloc[i] else 0
        sum_sell = cond_sell_rsi + cond_sell_atr
        if holding and last_signal_candle_index != i and sum_sell >= num_cond:
            sell_signals.append((df.index[i], float(df["Close"].iloc[i])))
            holding = False
            last_signal_candle_index = i

    return buy_signals, sell_signals


def close_rsi_buy_sell_limits_simulation(df):
    buy_signals = []
    sell_signals = []
    holding = False
    last_signal_candle_index = -1

    for i in range(1, len(df)):
        # CONDIZIONI DI BUY
        # if (not holding and last_signal_candle_index != i and
        #         df['RSI'].iloc[i - 1] > df['RSI2'].iloc[i - 1] and
        #         df['RSI'].iloc[i] < df['RSI2'].iloc[i]):
        if (
            not holding
            and last_signal_candle_index != i
            and df["RSI"].iloc[i - 1] < df["RSI2"].iloc[i - 1]
            and df["RSI"].iloc[i] > df["RSI2"].iloc[i]
        ):
            buy_signals.append((df.index[i], float(df["Close"].iloc[i])))
            holding = True
            last_signal_candle_index = i
        # CONDIZIONI DI SELL
        # if (holding and last_signal_candle_index != i and
        #         df['RSI'].iloc[i - 1] < df['RSI2'].iloc[i - 1] and
        #         df['RSI'].iloc[i] > df['RSI2'].iloc[i]):
        if (
            holding
            and last_signal_candle_index != i
            and df["RSI"].iloc[i - 1] > df["RSI2"].iloc[i - 1]
            and df["RSI"].iloc[i] < df["RSI2"].iloc[i]
        ):
            sell_signals.append((df.index[i], float(df["Close"].iloc[i])))
            holding = False
            last_signal_candle_index = i

    return buy_signals, sell_signals


def atr_buy_sell_simulation(df, stop_loss_percent):
    # Identificazione dei segnali di acquisto e vendita
    buy_signals = []
    sell_signals = []
    holding = False
    last_signal_candle_index = -1
    stop_loss_price = None
    got_stop_loss = False
    stop_loss_decimal = stop_loss_percent / 100

    for i in range(1, len(df)):
        if (
            not holding
            and last_signal_candle_index != i
            and df["Low"].iloc[i] <= df["Lower_Band"].iloc[i]
            and not (got_stop_loss and df["PSAR"].iloc[i] > df["Close"].iloc[i])
        ):
            buy_signals.append((df.index[i], float(df["Lower_Band"].iloc[i])))
            holding = True
            last_signal_candle_index = i
            got_stop_loss = False
            stop_loss_price = df["Lower_Band"].iloc[i] * (1 - stop_loss_decimal)
        if holding and last_signal_candle_index != i and df["High"].iloc[i] >= df["Upper_Band"].iloc[i]:
            sell_signals.append((df.index[i], float(df["Upper_Band"].iloc[i])))
            holding = False
            last_signal_candle_index = i
            stop_loss_price = None
            got_stop_loss = False
        if (
            holding
            and stop_loss_price is not None
            and df["Low"].iloc[i] < stop_loss_price
            and df["PSAR"].iloc[i] > df["Close"].iloc[i]
        ):
            # devo vendere per STOP LOSS
            sell_signals.append((df.index[i], stop_loss_price))
            holding = False
            last_signal_candle_index = i
            got_stop_loss = True
            stop_loss_price = None

    return buy_signals, sell_signals


def close_atr_buy_sell_simulation(df, stop_loss_percent):
    # Identificazione dei segnali di acquisto e vendita
    buy_signals = []
    sell_signals = []
    holding = False
    last_signal_candle_index = -1
    stop_loss_price = None
    got_stop_loss = False
    stop_loss_decimal = stop_loss_percent / 100

    for i in range(1, len(df)):
        if (
            not holding
            and last_signal_candle_index != i
            and df["Close"].iloc[i] <= df["Lower_Band"].iloc[i]
            and not (got_stop_loss and df["PSAR"].iloc[i] > df["Close"].iloc[i])
        ):
            buy_signals.append((df.index[i], float(df["Close"].iloc[i])))
            holding = True
            last_signal_candle_index = i
            got_stop_loss = False
            stop_loss_price = float(df["Close"].iloc[i]) * (1 - stop_loss_decimal)
        if holding and last_signal_candle_index != i and df["Close"].iloc[i] >= df["Upper_Band"].iloc[i]:
            sell_signals.append((df.index[i], float(df["Close"].iloc[i])))
            holding = False
            last_signal_candle_index = i
            stop_loss_price = None
            got_stop_loss = False
        if (
            holding
            and stop_loss_price is not None
            and df["Close"].iloc[i] < stop_loss_price
            and df["PSAR"].iloc[i] > df["Close"].iloc[i]
        ):
            # devo vendere per STOP LOSS
            sell_signals.append((df.index[i], float(df["Close"].iloc[i])))
            holding = False
            last_signal_candle_index = i
            got_stop_loss = True
            stop_loss_price = None

    return buy_signals, sell_signals


def close_ema_crossover_simulation(df):
    buy_signals = []
    sell_signals = []
    holding = False
    first_break = False
    second_break = False
    for i in range(1, len(df)):
        ema50ema100up = df["EMA20"].iloc[i - 1] <= df["EMA50"].iloc[i - 1] and df["EMA20"].iloc[i] > df["EMA50"].iloc[i]
        ema50ema200up = (
            df["EMA20"].iloc[i - 1] <= df["EMA100"].iloc[i - 1] and df["EMA20"].iloc[i] > df["EMA100"].iloc[i]
        )
        ema100ema200up = (
            df["EMA50"].iloc[i - 1] <= df["EMA100"].iloc[i - 1] and df["EMA50"].iloc[i] > df["EMA100"].iloc[i]
        )

        ema50ema100down = (
            df["EMA20"].iloc[i - 1] >= df["EMA50"].iloc[i - 1] and df["EMA20"].iloc[i] < df["EMA50"].iloc[i]
        )
        ema50ema200down = (
            df["EMA20"].iloc[i - 1] >= df["EMA100"].iloc[i - 1] and df["EMA20"].iloc[i] < df["EMA100"].iloc[i]
        )
        ema100ema200down = (
            df["EMA50"].iloc[i - 1] >= df["EMA100"].iloc[i - 1] and df["EMA50"].iloc[i] < df["EMA100"].iloc[i]
        )

        if not holding:
            # non si verifica le sequenza esatta
            if first_break and (ema100ema200up or ema50ema100down or ema50ema200down or ema100ema200down):
                first_break = False
            if second_break and (ema50ema100up or ema50ema100down or ema50ema200down or ema100ema200down):
                second_break = False
            # controllo la sequenza esatta
            if ema50ema100up:
                first_break = True
            if first_break and ema50ema200up:
                second_break = True
            if second_break and ema100ema200up:
                first_break = False
                second_break = False
                buy_signals.append((df.index[i], float(df["Close"].iloc[i])))
                holding = True

        if holding:
            # non si verifica le sequenza esatta
            if first_break and (ema100ema200down or ema50ema100up or ema50ema200up or ema100ema200up):
                first_break = False
            if second_break and (ema50ema100down or ema50ema100up or ema50ema200up or ema100ema200up):
                second_break = False
            # controllo la sequenza esatta
            if ema50ema100down:
                first_break = True
            if first_break and ema50ema200down:
                second_break = True
            if second_break and ema100ema200down:
                first_break = False
                second_break = False
                sell_signals.append((df.index[i], float(df["Close"].iloc[i])))
                holding = False

    return buy_signals, sell_signals


def close_bullish_ema_simulation(df, rsi_buy_limit: int = 50, rsi_sell_limit: int = 70):
    buy_signals = []
    sell_signals = []
    holding = False
    n = 30
    for i in range(1, len(df)):
        cond_1 = df["EMA20"][i - n : i] > df["EMA50"][i - n : i]
        cond_2 = df["EMA50"][i - n : i] > df["EMA100"][i - n : i]
        cond_ema = (cond_1 & cond_2).all()
        if (
            not holding
            and cond_ema
            and (df["EMA20"].iloc[i] > df["EMA50"].iloc[i] > df["EMA100"].iloc[i])  # trend rialzista nel breve termine
            # and df['ADX'].iloc[i] > 30  # conferma della forza del trend
            # and df['EMA50'].iloc[i] < df['Upper_Band3'].iloc[i]  # il prezzo oscilla attorno alla media lunga
            and df["Close"].iloc[i] > df["EMA100"].iloc[i]  # il prezzo sta sopra alla media lunga
            and rsi_buy_limit <= df["RSI"].iloc[i] < rsi_sell_limit  # RSI compreso in una fascia che conferma il trend
            # controlli sulle candele precedenti
            and (
                (df["Low"].iloc[i - 1] < df["EMA50"].iloc[i - 1] < df["Close"].iloc[i - 1])
                or (df["Low"].iloc[i - 1] < df["EMA100"].iloc[i - 1] < df["Close"].iloc[i - 1])
            )
            and (
                (df["EMA50"].iloc[i - 2] < df["Low"].iloc[i - 2] < df["Close"].iloc[i - 2])
                or (df["EMA100"].iloc[i - 2] < df["Low"].iloc[i - 2] < df["Close"].iloc[i - 2])
            )
        ):
            buy_signals.append((df.index[i], float(df["Close"].iloc[i])))
            holding = True
        if holding and df["RSI"].iloc[i] > rsi_sell_limit:
            sell_signals.append((df.index[i], float(df["Close"].iloc[i])))
            holding = False

    return buy_signals, sell_signals


def tp_sl_simulation(df):
    # Identificazione dei segnali di acquisto e vendita
    buy_signals = []
    sell_signals = []
    holding = False
    last_signal_candle_index = -1
    take_profit_price = None
    stop_loss_price = None

    for i in range(1, len(df)):
        if not holding and last_signal_candle_index != i and df["Close"].iloc[i] >= df["Upper_Band"].iloc[i]:
            buy_signals.append((df.index[i], float(df["Close"].iloc[i])))
            holding = True
            last_signal_candle_index = i
            stop_loss_price = float(df["Lower_Band"].iloc[i])
            take_profit_price = df["Close"].iloc[i] + (df["Close"].iloc[i] - stop_loss_price)
        if (
            holding
            and take_profit_price is not None
            and last_signal_candle_index != i
            and df["High"].iloc[i] >= take_profit_price
        ):
            # vengo per TAKE PROFIT
            sell_signals.append((df.index[i], take_profit_price))
            holding = False
            last_signal_candle_index = i
            stop_loss_price = None
            take_profit_price = None
        if (
            holding
            and stop_loss_price is not None
            and last_signal_candle_index != i
            and df["Low"].iloc[i] <= stop_loss_price
        ):
            # devo vendere per STOP LOSS
            sell_signals.append((df.index[i], stop_loss_price))
            holding = False
            last_signal_candle_index = i
            stop_loss_price = None
            take_profit_price = None

    return buy_signals, sell_signals


def green_candles_simulation(df):
    buy_signals = []
    sell_signals = []
    holding = False

    for i in range(1, len(df)):
        if (
            not holding
            and df["Close"].iloc[i - 1] < df["Open"].iloc[i - 1]
            and df["Close"].iloc[i] > df["High"].iloc[i - 1]
        ):
            buy_signals.append((df.index[i], float(df["Close"].iloc[i])))
            holding = True
        if holding and df["Close"].iloc[i - 1] > df["Open"].iloc[i - 1] and df["Close"].iloc[i] < df["Low"].iloc[i - 1]:
            sell_signals.append((df.index[i], float(df["Close"].iloc[i])))
            holding = False

    return buy_signals, sell_signals


def bullish_condition(df, i) -> bool:
    # cond_bullish = (df['EMA20'].iloc[i] > df['EMA2'].iloc[i] > df['EMA3'].iloc[i] and
    #                 df['RSI'].iloc[i] > df['RSI2'].iloc[i] > df['RSI3'].iloc[i] and
    #                 df['STOCH'].iloc[i] > df['STOCH_S'].iloc[i])

    # cond_bullish = df['Close'].iloc[i] >= df['Upper_Band'].iloc[i]
    cond_bullish = df["EMA20"].iloc[i] >= df["EMA200"].iloc[i]

    return cond_bullish


def bearish_condition(df, i) -> bool:
    # cond_bearish = (df['EMA20'].iloc[i] < df['EMA2'].iloc[i] < df['EMA3'].iloc[i] and
    #                 df['RSI'].iloc[i] < df['RSI2'].iloc[i] < df['RSI3'].iloc[i] and
    #                 df['STOCH'].iloc[i] < df['STOCH_S'].iloc[i])
    #
    # cond_bearish = df['Close'].iloc[i] <= df['Lower_Band'].iloc[i]
    cond_bearish = df["EMA20"].iloc[i] < df["EMA200"].iloc[i]

    return cond_bearish


def identify_trend_zones(df: pd.DataFrame) -> list:
    """
    Identifica intervalli di trend rialzista e ribassista basandosi su parametri di input,
    e restituisce una lista di dizionari "shapes" Plotly.

    Parametri
    ---------
    df : pd.DataFrame
        Il DataFrame con le colonne richieste (RSI, EMA, etc.)

    Ritorna
    -------
    list
        Lista di shapes (rettangoli) da passare successivamente a fig.update_layout(shapes=...).
        Ogni shape è un dict con parametri per disegnare la zona colorata su Plotly.
    """

    shapes = []
    current_trend = None  # Possibili valori: 'bullish', 'bearish', oppure None
    zone_start_index = None
    # cond_bullish = False
    # cond_bearish = False

    for i in range(len(df)):
        # Valuta le condizioni per bullish e bearish

        cond_bullish = bullish_condition(df, i)
        cond_bearish = bearish_condition(df, i)

        # if current_trend != "bearish" and cond_bearish:
        #     new_trend = "bearish"
        # elif current_trend != "bullish" and cond_bullish:
        #     new_trend = "bullish"
        # else:
        #     new_trend = None

        if cond_bullish:
            new_trend = "bullish"
        elif cond_bearish:
            new_trend = "bearish"
        else:
            new_trend = None

        if new_trend is not None and new_trend != current_trend:
            # Se stiamo cambiando "trend" rispetto al bar/candela precedente,
            # chiudiamo eventuale zona precedente e ne apriamo un'altra se necessario.
            if current_trend is not None and zone_start_index is not None:
                # Costruisci shape per la zona precedente [zone_start_index ... i-1]
                x0 = df.index[zone_start_index]
                # i potrebbe essersi spostato di 1 troppo in avanti, usiamo df.index[i-1] se i>0
                x1 = df.index[i - 1] if i > 0 else x0

                if current_trend == "bullish":
                    fillcolor = "green"
                    opacity_val = 0.15
                else:
                    fillcolor = "red"
                    opacity_val = 0.15

                shapes.append(
                    dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=x0,
                        x1=x1,
                        y0=0,  # intera altezza del grafico
                        y1=1,
                        fillcolor=fillcolor,
                        opacity=opacity_val,
                        layer="below",  # la zona resta sotto le candele
                        line_width=0,  # niente contorno
                    )
                )

            # Apertura della nuova zona, se c'è un nuovo trend
            if new_trend is not None:
                zone_start_index = i
            else:
                zone_start_index = None

            current_trend = new_trend

    # Chiudi eventuale ultima zona rimasta aperta
    if current_trend is not None and zone_start_index is not None:
        x0 = df.index[zone_start_index]
        x1 = df.index[-1]  # fino alla fine
        if current_trend == "bullish":
            fillcolor = "green"
            opacity_val = 0.15
        else:
            fillcolor = "red"
            opacity_val = 0.15

        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=x0,
                x1=x1,
                y0=0,
                y1=1,
                fillcolor=fillcolor,
                opacity=opacity_val,
                layer="below",
                line_width=0,
            )
        )

    return shapes


def supertrend_simulation(df):
    buy_signals = []
    sell_signals = []
    holding = False

    current_trend = None  # Possibili valori: 'bullish', 'bearish', oppure None

    take_profit_price = None
    stop_loss_price = None

    for i in range(1, len(df)):
        # cond_bullish = bullish_condition(df, i)
        cond_bullish = df["Close"].iloc[i] >= df["Upper_Band"].iloc[i]
        # cond_bearish = bearish_condition(df, i)
        cond_bearish = df["Close"].iloc[i] <= df["Lower_Band"].iloc[i]
        if cond_bullish:
            new_trend = "bullish"
        elif cond_bearish:
            new_trend = "bearish"
        else:
            new_trend = None

        if new_trend is not None and new_trend != current_trend:
            if not holding and new_trend == "bullish":
                buy_signals.append((df.index[i], float(df["Close"].iloc[i])))
                holding = True
                stop_loss_price = df["Lower_Band"].iloc[i]
                take_profit_price = df["Close"].iloc[i] + (df["Close"].iloc[i] - stop_loss_price) * 1.618

            if holding and new_trend == "bearish":
                sell_signals.append((df.index[i], float(df["Close"].iloc[i])))
                holding = False

        if holding and take_profit_price is not None and df["High"].iloc[i] >= take_profit_price:
            # vengo per TAKE PROFIT
            sell_signals.append((df.index[i], take_profit_price))
            holding = False
            stop_loss_price = None
            take_profit_price = None

        if holding and stop_loss_price is not None and df["Low"].iloc[i] <= stop_loss_price:
            # devo vendere per STOP LOSS
            sell_signals.append((df.index[i], stop_loss_price))
            holding = False
            stop_loss_price = None
            take_profit_price = None

            current_trend = new_trend

        # if not bearish_condition(df, i) and bearish_condition(df, i - 1):
        #     buy_signals.append((df.index[i], float(df['Close'].iloc[i])))
        #     holding = True
        #
        # if holding and not bullish_condition(df, i) and bullish_condition(df, i - 1):
        #     sell_signals.append((df.index[i], float(df['Close'].iloc[i])))
        #     holding = False

    return buy_signals, sell_signals


def trend_zone_simulation(df):
    buy_signals = []
    sell_signals = []
    holding = False
    for i in range(1, len(df)):
        if not holding and df["EMA20"].iloc[i] > df["EMA200"].iloc[i]:
            buy_signals.append((df.index[i], float(df["Close"].iloc[i])))
            holding = True

        if holding and df["EMA20"].iloc[i] <= df["EMA200"].iloc[i]:
            sell_signals.append((df.index[i], float(df["Close"].iloc[i])))
            holding = False

    return buy_signals, sell_signals


def get_green_red_percentage(df: pd.DataFrame):
    # calcola la percentuale di candele verdi dopo una candela verde e dopo una candela rossa
    green = 0
    green_after_green = 0
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Open"].iloc[i]:
            # candela verde
            green += 1
            if df["Close"].iloc[i - 1] > df["Open"].iloc[i - 1]:
                # candela verde precedente
                green_after_green += 1

    return green_after_green / green


def ai_model_simulation(df, model, threshold: float = None):
    """Strategia "AI Model": ingresso sul punteggio del modello, uscita sulle barriere.

    Il modello produce solo segnali di ingresso; l'uscita e' il take-profit, lo stop-loss o il
    limite temporale con cui sono state costruite le etichette. Rispettare quella corrispondenza
    e' cio' che rende il P&L qui sotto la traduzione diretta del win rate misurato in validation.

    I segnali risultano alternati per costruzione, che e' anche l'unico caso in cui
    l'accoppiamento per indice di `simulate_trading_with_commisions` ha senso.
    """
    threshold = threshold if threshold is not None else stored_decision_threshold()
    family = active_model_name()
    if family == "policy_model":
        # La politica a tre azioni decide anche l'uscita, quindi le barriere qui non entrano.
        return policy_signals(df, model, threshold=threshold)
    if family == "meta_model":
        return meta_signals(df, model, threshold=threshold, **meta_parameters())
    return barrier_signals(df, model, threshold=threshold)
