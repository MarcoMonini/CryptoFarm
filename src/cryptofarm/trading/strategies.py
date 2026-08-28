"""Le strategie: da una tabella di candele con indicatori a due liste di segnali.

Estratte da `simulator.py` senza modifiche. Ognuna restituisce `(buy_signals, sell_signals)`,
liste di `(timestamp, prezzo)` che `pnl.py` trasforma in operazioni.

`buy_sell_limits_simulation` legge `MACD`, che resta commentata in `add_technical_indicator`, e
quindi solleva `KeyError` appena chiamata. Nessuna voce del menu la raggiunge: il dispatch di
`trading_analysis` la lega alla stringa "Buy/Sell Limits", che non e' in `config.STRATEGIES`.
Lo stesso vale per `close_rsi_buy_sell_limits_simulation`: la misura su nove anni la da' in
perdita totale in tutte le 25 configurazioni provate, quindi la voce non e' stata aggiunta."""

import numpy as np
import pandas as pd
import streamlit as st
from ta.trend import PSARIndicator

from cryptofarm.ml.signals import (
    barrier_signals,
    leg_signals,
    meta_signals,
    policy_signals,
    rl_signals,
    swing_signals,
)
from cryptofarm.ml.trainer import active_model_name, meta_parameters, stored_decision_threshold
from cryptofarm.trading.indicators import latest_bands


# Come sopra: la cardinalita' la decidono i widget, quindi il numero di voci va limitato.
@st.cache_data(max_entries=32)
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

    # Letture per riga: `df["Open"].iloc[i]` passa dal motore di indicizzazione di pandas a ogni
    # accesso. Le colonne servono per intero, quindi si estraggono una volta.
    opens, highs, lows, closes = (df[c].to_numpy() for c in ("Open", "High", "Low", "Close"))
    # Il PSAR governa la condizione di stop-loss piu' sotto. Si calcola una volta sulle candele
    # reali: e' un indicatore ricorsivo, e ricalcolarlo sui prezzi simulati dentro la candela
    # darebbe un valore che nel mercato vero non esiste. `step` e `max_step`, dichiarati nella
    # firma e fin qui inutilizzati, sono i suoi parametri.
    psar = (
        PSARIndicator(high=df["High"], low=df["Low"], close=df["Close"], step=step, max_step=max_step).psar().to_numpy()
    )
    needed_bars = atr_window + 12

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

        o = opens[i]
        h = highs[i]
        low = lows[i]
        c = closes[i]

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
            segment1 = linspace_steps(o, low, n=int(n_steps / 2))
            # Segmento 2: low -> high
            segment2 = linspace_steps(low, h, n=int(n_steps * 2))
            # Segmento 3: high -> close
            segment3 = linspace_steps(h, c, n=int(n_steps / 2))
        else:
            # Segmento 1: open -> high
            segment1 = linspace_steps(o, h, n=int(n_steps / 2))
            # Segmento 2: high -> low
            segment2 = linspace_steps(h, low, n=int(n_steps * 2))
            # Segmento 3: low -> close
            segment3 = linspace_steps(low, c, n=int(n_steps / 2))
        prices_sequence = list(segment1) + list(segment2) + list(segment3)

        # A questo punto abbiamo 3*9 = 27 prezzi intermedi,
        # se vogliamo esattamente 30 step (includendo anche l'ultimo?),
        # possiamo aggiungere l'ultimo prezzo "Close" come step finale,
        # così da totalizzare 28 (oppure gestire diversamente).
        # Per semplicità, qui aggiungo manualmente l'ultimo step = c
        # (ma dipende da come preferisci gestire i conti).
        prices_sequence.append(c)
        # Inizializza i valori "in costruzione" della candela:
        step_high = o
        step_low = o
        step_close = o

        # Le bande dipendono solo dalle ultime `needed_bars` barre che finiscono in `i`, e di quelle
        # solo da High/Low/Close. Prima ogni sotto-passo copiava tutto il DataFrame e ricostruiva
        # gli indicatori con `ta`: qui la finestra si ritaglia una volta per candela e i sotto-passi
        # ne riscrivono solo l'ultima barra, che sovrascrivono comunque per intero.
        start = max(0, i - needed_bars + 1)
        window_high = highs[start : i + 1].copy()
        window_low = lows[start : i + 1].copy()
        window_close = closes[start : i + 1].copy()

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

            # Sovrascrivi sulla candela i-esima i valori dinamici. L'apertura non entra nel
            # calcolo: ATR ed EMA guardano solo High, Low e Close.
            window_high[-1] = step_high
            window_low[-1] = step_low
            window_close[-1] = step_close

            upper_band, lower_band = latest_bands(window_high, window_low, window_close, atr_window, atr_multiplier)

            # Condizione di BUY
            if (
                not holding
                and last_signal_candle_index != i
                and lower_band is not None
                and step_close <= lower_band
                and not (got_stop_loss and psar[i] is not None and psar[i] > step_close)
            ):
                buy_signals.append((df.index[i], float(step_close)))
                holding = True
                last_signal_candle_index = i
                got_stop_loss = False
                stop_loss_price = float(step_close) * (1 - stop_loss_decimal)
            # Condizione di SELL
            if holding and last_signal_candle_index != i and upper_band is not None and step_close >= upper_band:
                sell_signals.append((df.index[i], float(step_close)))
                holding = False
                last_signal_candle_index = i
                stop_loss_price = None
                got_stop_loss = False
            # Condizione STOP LOSS
            if holding and stop_loss_price is not None and step_close < stop_loss_price and psar[i] > step_close:
                sell_signals.append((df.index[i], float(step_close)))
                holding = False
                last_signal_candle_index = i
                got_stop_loss = True
                stop_loss_price = None

    return buy_signals, sell_signals


def buy_sell_limits_simulation(df, macd_buy_limit, macd_sell_limit, rsi_buy_limit, rsi_sell_limit, num_cond):
    buy_signals = []
    sell_signals = []
    holding = False

    index = df.index
    closes = df["Close"].to_numpy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    lower_band = df["Lower_Band"].to_numpy()
    macd = df["MACD"].to_numpy()
    rsi = df["RSI"].to_numpy()
    upper_band = df["Upper_Band"].to_numpy()
    for i in range(1, len(df)):
        # CONDIZIONI DI BUY
        cond_buy_macd = 1 if macd[i] <= macd_buy_limit else 0
        # cond_buy_macd2 = 1 if df['MACD'].iloc[i] > df['MACD'].tail(
        #     10).min() else 0  # il MACD ha invertito direzione
        cond_buy_rsi = 1 if rsi[i] <= rsi_buy_limit else 0
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
            if lows[i] < lower_band[i]:
                buy_signals.append((index[i], float(lower_band[i])))
            else:
                buy_signals.append((index[i], float(closes[i])))
            holding = True
        # CONDIZIONI DI SELL
        cond_sell_macd = 1 if macd[i] >= macd_sell_limit else 0
        # cond_sell_macd2 = 1 if df['MACD'].iloc[i] < df['MACD'].tail(
        #    10).max() else 0  # il MACD ha invertito direzione
        cond_sell_rsi = 1 if rsi[i] >= rsi_sell_limit else 0
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
            if highs[i] > upper_band[i]:
                sell_signals.append((index[i], float(upper_band[i])))
            else:
                sell_signals.append((index[i], float(closes[i])))
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
    # Lo stop loss era dichiarato e mai applicato: le tre righe che lo usavano erano commentate,
    # quindi il widget "Stop Loss %" per questa strategia non faceva nulla. Ora vale come in
    # `close_atr_buy_sell_simulation`: prezzo di stop fissato all'ingresso, uscita alla prima
    # chiusura sotto. Il default 99% lo lascia di fatto disattivato, come prima.
    stop_loss_price = None
    stop_loss_decimal = stop_loss_percent / 100

    index = df.index
    closes = df["Close"].to_numpy()
    lower_band = df["Lower_Band"].to_numpy()
    rsi = df["RSI"].to_numpy()
    upper_band = df["Upper_Band"].to_numpy()
    for i in range(1, len(df)):
        # CONDIZIONI DI BUY
        cond_buy_atr = 1 if closes[i] <= lower_band[i] else 0
        cond_buy_rsi = 1 if rsi[i] <= rsi_buy_limit else 0
        sum_buy = cond_buy_rsi + cond_buy_atr
        if not holding and last_signal_candle_index != i and sum_buy >= num_cond:
            buy_signals.append((index[i], float(closes[i])))
            holding = True
            last_signal_candle_index = i
            stop_loss_price = float(closes[i]) * (1 - stop_loss_decimal)
        # CONDIZIONI DI SELL
        cond_sell_rsi = 1 if rsi[i] >= rsi_sell_limit else 0
        cond_sell_atr = 1 if closes[i] >= upper_band[i] else 0
        sum_sell = cond_sell_rsi + cond_sell_atr
        if holding and last_signal_candle_index != i and sum_sell >= num_cond:
            sell_signals.append((index[i], float(closes[i])))
            holding = False
            last_signal_candle_index = i
            stop_loss_price = None
        # CONDIZIONE DI STOP LOSS
        if holding and stop_loss_price is not None and last_signal_candle_index != i and closes[i] < stop_loss_price:
            sell_signals.append((index[i], float(closes[i])))
            holding = False
            last_signal_candle_index = i
            stop_loss_price = None

    return buy_signals, sell_signals


def close_rsi_buy_sell_limits_simulation(df):
    buy_signals = []
    sell_signals = []
    holding = False
    last_signal_candle_index = -1

    index = df.index
    closes = df["Close"].to_numpy()
    rsi = df["RSI"].to_numpy()
    rsi2 = df["RSI2"].to_numpy()
    for i in range(1, len(df)):
        # CONDIZIONI DI BUY
        # if (not holding and last_signal_candle_index != i and
        #         df['RSI'].iloc[i - 1] > df['RSI2'].iloc[i - 1] and
        #         df['RSI'].iloc[i] < df['RSI2'].iloc[i]):
        if not holding and last_signal_candle_index != i and rsi[i - 1] < rsi2[i - 1] and rsi[i] > rsi2[i]:
            buy_signals.append((index[i], float(closes[i])))
            holding = True
            last_signal_candle_index = i
        # CONDIZIONI DI SELL
        # if (holding and last_signal_candle_index != i and
        #         df['RSI'].iloc[i - 1] < df['RSI2'].iloc[i - 1] and
        #         df['RSI'].iloc[i] > df['RSI2'].iloc[i]):
        if holding and last_signal_candle_index != i and rsi[i - 1] > rsi2[i - 1] and rsi[i] < rsi2[i]:
            sell_signals.append((index[i], float(closes[i])))
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

    index = df.index
    closes = df["Close"].to_numpy()
    psar = df["PSAR"].to_numpy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    lower_band = df["Lower_Band"].to_numpy()
    upper_band = df["Upper_Band"].to_numpy()
    for i in range(1, len(df)):
        if (
            not holding
            and last_signal_candle_index != i
            and lows[i] <= lower_band[i]
            and not (got_stop_loss and psar[i] > closes[i])
        ):
            buy_signals.append((index[i], float(lower_band[i])))
            holding = True
            last_signal_candle_index = i
            got_stop_loss = False
            stop_loss_price = lower_band[i] * (1 - stop_loss_decimal)
        if holding and last_signal_candle_index != i and highs[i] >= upper_band[i]:
            sell_signals.append((index[i], float(upper_band[i])))
            holding = False
            last_signal_candle_index = i
            stop_loss_price = None
            got_stop_loss = False
        if holding and stop_loss_price is not None and lows[i] < stop_loss_price and psar[i] > closes[i]:
            # devo vendere per STOP LOSS
            sell_signals.append((index[i], stop_loss_price))
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

    index = df.index
    closes = df["Close"].to_numpy()
    psar = df["PSAR"].to_numpy()
    lower_band = df["Lower_Band"].to_numpy()
    upper_band = df["Upper_Band"].to_numpy()
    for i in range(1, len(df)):
        if (
            not holding
            and last_signal_candle_index != i
            and closes[i] <= lower_band[i]
            and not (got_stop_loss and psar[i] > closes[i])
        ):
            buy_signals.append((index[i], float(closes[i])))
            holding = True
            last_signal_candle_index = i
            got_stop_loss = False
            stop_loss_price = float(closes[i]) * (1 - stop_loss_decimal)
        if holding and last_signal_candle_index != i and closes[i] >= upper_band[i]:
            sell_signals.append((index[i], float(closes[i])))
            holding = False
            last_signal_candle_index = i
            stop_loss_price = None
            got_stop_loss = False
        if holding and stop_loss_price is not None and closes[i] < stop_loss_price and psar[i] > closes[i]:
            # devo vendere per STOP LOSS
            sell_signals.append((index[i], float(closes[i])))
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
    index = df.index
    close = df["Close"].to_numpy()
    ema20, ema50, ema100 = (df[c].to_numpy() for c in ("EMA20", "EMA50", "EMA100"))
    for i in range(1, len(df)):
        ema50ema100up = ema20[i - 1] <= ema50[i - 1] and ema20[i] > ema50[i]
        ema50ema200up = ema20[i - 1] <= ema100[i - 1] and ema20[i] > ema100[i]
        ema100ema200up = ema50[i - 1] <= ema100[i - 1] and ema50[i] > ema100[i]

        ema50ema100down = ema20[i - 1] >= ema50[i - 1] and ema20[i] < ema50[i]
        ema50ema200down = ema20[i - 1] >= ema100[i - 1] and ema20[i] < ema100[i]
        ema100ema200down = ema50[i - 1] >= ema100[i - 1] and ema50[i] < ema100[i]

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
                buy_signals.append((index[i], float(close[i])))
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
                sell_signals.append((index[i], float(close[i])))
                holding = False

    return buy_signals, sell_signals


def close_bullish_ema_simulation(df, rsi_buy_limit: int = 50, rsi_sell_limit: int = 70):
    buy_signals = []
    sell_signals = []
    holding = False
    n = 30
    # Le fette `[i - n : i]` restano tali e quali: per i primi `i` l'inizio e' negativo e taglia
    # dalla coda, e `.all()` su una fetta vuota vale True. numpy segue le stesse regole di pandas.
    index = df.index
    close, low = df["Close"].to_numpy(), df["Low"].to_numpy()
    ema20, ema50, ema100 = (df[c].to_numpy() for c in ("EMA20", "EMA50", "EMA100"))
    rsi = df["RSI"].to_numpy()
    for i in range(1, len(df)):
        cond_ema = ((ema20[i - n : i] > ema50[i - n : i]) & (ema50[i - n : i] > ema100[i - n : i])).all()
        if (
            not holding
            and cond_ema
            and (ema20[i] > ema50[i] > ema100[i])  # trend rialzista nel breve termine
            # and df['ADX'].iloc[i] > 30  # conferma della forza del trend
            # and df['EMA50'].iloc[i] < df['Upper_Band3'].iloc[i]  # il prezzo oscilla attorno alla media lunga
            and close[i] > ema100[i]  # il prezzo sta sopra alla media lunga
            and rsi_buy_limit <= rsi[i] < rsi_sell_limit  # RSI compreso in una fascia che conferma il trend
            # controlli sulle candele precedenti
            and ((low[i - 1] < ema50[i - 1] < close[i - 1]) or (low[i - 1] < ema100[i - 1] < close[i - 1]))
            and ((ema50[i - 2] < low[i - 2] < close[i - 2]) or (ema100[i - 2] < low[i - 2] < close[i - 2]))
        ):
            buy_signals.append((index[i], float(close[i])))
            holding = True
        if holding and rsi[i] > rsi_sell_limit:
            sell_signals.append((index[i], float(close[i])))
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

    index = df.index
    closes = df["Close"].to_numpy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    lower_band = df["Lower_Band"].to_numpy()
    upper_band = df["Upper_Band"].to_numpy()
    for i in range(1, len(df)):
        if not holding and last_signal_candle_index != i and closes[i] >= upper_band[i]:
            buy_signals.append((index[i], float(closes[i])))
            holding = True
            last_signal_candle_index = i
            stop_loss_price = float(lower_band[i])
            take_profit_price = closes[i] + (closes[i] - stop_loss_price)
        if (
            holding
            and take_profit_price is not None
            and last_signal_candle_index != i
            and highs[i] >= take_profit_price
        ):
            # vengo per TAKE PROFIT
            sell_signals.append((index[i], take_profit_price))
            holding = False
            last_signal_candle_index = i
            stop_loss_price = None
            take_profit_price = None
        if holding and stop_loss_price is not None and last_signal_candle_index != i and lows[i] <= stop_loss_price:
            # devo vendere per STOP LOSS
            sell_signals.append((index[i], stop_loss_price))
            holding = False
            last_signal_candle_index = i
            stop_loss_price = None
            take_profit_price = None

    return buy_signals, sell_signals


def green_candles_simulation(df):
    buy_signals = []
    sell_signals = []
    holding = False

    index = df.index
    closes = df["Close"].to_numpy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    opens = df["Open"].to_numpy()
    for i in range(1, len(df)):
        if not holding and closes[i - 1] < opens[i - 1] and closes[i] > highs[i - 1]:
            buy_signals.append((index[i], float(closes[i])))
            holding = True
        if holding and closes[i - 1] > opens[i - 1] and closes[i] < lows[i - 1]:
            sell_signals.append((index[i], float(closes[i])))
            holding = False

    return buy_signals, sell_signals


def bullish_condition(df, i) -> bool:
    # cond_bullish = (df['EMA20'].iloc[i] > df['EMA2'].iloc[i] > df['EMA3'].iloc[i] and
    #                 df['RSI'].iloc[i] > df['RSI2'].iloc[i] > df['RSI3'].iloc[i] and
    #                 df['STOCH'].iloc[i] > df['STOCH_S'].iloc[i])

    # cond_bullish = df['Close'].iloc[i] >= df['Upper_Band'].iloc[i]
    cond_bullish = df["EMA20"].iloc[i] >= df["EMA100"].iloc[i]

    return cond_bullish


def bearish_condition(df, i) -> bool:
    # cond_bearish = (df['EMA20'].iloc[i] < df['EMA2'].iloc[i] < df['EMA3'].iloc[i] and
    #                 df['RSI'].iloc[i] < df['RSI2'].iloc[i] < df['RSI3'].iloc[i] and
    #                 df['STOCH'].iloc[i] < df['STOCH_S'].iloc[i])
    #
    # cond_bearish = df['Close'].iloc[i] <= df['Lower_Band'].iloc[i]
    cond_bearish = df["EMA20"].iloc[i] < df["EMA100"].iloc[i]

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

    index = df.index
    closes = df["Close"].to_numpy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    lower_band = df["Lower_Band"].to_numpy()
    upper_band = df["Upper_Band"].to_numpy()
    for i in range(1, len(df)):
        # cond_bullish = bullish_condition(df, i)
        cond_bullish = closes[i] >= upper_band[i]
        # cond_bearish = bearish_condition(df, i)
        cond_bearish = closes[i] <= lower_band[i]
        if cond_bullish:
            new_trend = "bullish"
        elif cond_bearish:
            new_trend = "bearish"
        else:
            new_trend = None

        if new_trend is not None and new_trend != current_trend:
            if not holding and new_trend == "bullish":
                buy_signals.append((index[i], float(closes[i])))
                holding = True
                stop_loss_price = lower_band[i]
                take_profit_price = closes[i] + (closes[i] - stop_loss_price) * 1.618

            if holding and new_trend == "bearish":
                sell_signals.append((index[i], float(closes[i])))
                holding = False

        if holding and take_profit_price is not None and highs[i] >= take_profit_price:
            # vengo per TAKE PROFIT
            sell_signals.append((index[i], take_profit_price))
            holding = False
            stop_loss_price = None
            take_profit_price = None

        if holding and stop_loss_price is not None and lows[i] <= stop_loss_price:
            # devo vendere per STOP LOSS
            sell_signals.append((index[i], stop_loss_price))
            holding = False
            stop_loss_price = None
            take_profit_price = None

            current_trend = new_trend

        # if not bearish_condition(df, i) and bearish_condition(df, i - 1):
        #     buy_signals.append((index[i], float(df['Close'].iloc[i])))
        #     holding = True
        #
        # if holding and not bullish_condition(df, i) and bullish_condition(df, i - 1):
        #     sell_signals.append((index[i], float(df['Close'].iloc[i])))
        #     holding = False

    return buy_signals, sell_signals


def trend_zone_simulation(df):
    buy_signals = []
    sell_signals = []
    holding = False
    index = df.index
    closes = df["Close"].to_numpy()
    ema20 = df["EMA20"].to_numpy()
    ema_long = df["EMA100"].to_numpy()
    for i in range(1, len(df)):
        if not holding and ema20[i] > ema_long[i]:
            buy_signals.append((index[i], float(closes[i])))
            holding = True

        if holding and ema20[i] <= ema_long[i]:
            sell_signals.append((index[i], float(closes[i])))
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


def ai_model_simulation(df, model, threshold: float = None, symbol: str = ""):
    """Strategia "AI Model": ingresso sul punteggio del modello, uscita sulle barriere.

    Il modello produce solo segnali di ingresso; l'uscita e' il take-profit, lo stop-loss o il
    limite temporale con cui sono state costruite le etichette. Rispettare quella corrispondenza
    e' cio' che rende il P&L qui sotto la traduzione diretta del win rate misurato in validation.

    I segnali risultano alternati per costruzione, che e' anche l'unico caso in cui
    l'accoppiamento per indice di `simulate_trading_with_commisions` ha senso.
    """
    threshold = threshold if threshold is not None else stored_decision_threshold()
    family = active_model_name()
    if family == "rl_model":
        # La politica RL emette direttamente la posizione, e il costo di cambiarla e' gia' dentro
        # l'obiettivo con cui e' stata addestrata: soglia e barriere qui non hanno posto.
        return rl_signals(model, df, symbol=symbol)
    if family == "swing_model":
        # Il modello a swing non emette una direzione: emette la prossimita' a un estremo locale,
        # e la forma misurata di quel segnale e' a U. L'uscita e' la stessa condizione
        # dell'ingresso letta al contrario, quindi qui non entrano ne' barriere ne' soglia.
        return swing_signals(df, model, symbol=symbol)
    if family == "leg_model":
        # Il modello delle gambe emette anche l'uscita (`P(giu)`), quindi non c'e' take profit:
        # e' `ml/signals.leg_signals` a decidere, e la ragione e' misurata li'.
        return leg_signals(df, model, threshold=threshold, symbol=symbol)
    if family == "policy_model":
        # La politica a tre azioni decide anche l'uscita, quindi le barriere qui non entrano.
        return policy_signals(df, model, threshold=threshold)
    if family == "meta_model":
        return meta_signals(df, model, threshold=threshold, **meta_parameters())
    return barrier_signals(df, model, threshold=threshold)
