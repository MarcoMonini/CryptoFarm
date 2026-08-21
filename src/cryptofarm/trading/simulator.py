import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema

from cryptofarm.ml.trainer import (
    load_signal_model,
)
from cryptofarm.paths import MODELS_DIR
from cryptofarm.trading.indicators import add_technical_indicator
from cryptofarm.trading.market_data import (
    get_market_data,
    get_market_data_between_dates,
)
from cryptofarm.trading.pnl import simulate_trading_with_commisions, simulate_trading_with_commisions_multiple_buy
from cryptofarm.trading.strategies import (
    ai_model_simulation,
    atr_buy_sell_simulation,
    buy_sell_limits_close_simulation,
    buy_sell_limits_simulation,
    close_atr_buy_sell_simulation,
    close_bullish_ema_simulation,
    close_ema_crossover_simulation,
    close_rsi_buy_sell_limits_simulation,
    green_candles_simulation,
    identify_trend_zones,
    simulate_candles,
    supertrend_simulation,
    tp_sl_simulation,
    trend_zone_simulation,
)

# Disattiva i FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)

MODEL_PATH = str(MODELS_DIR / "signal_model.joblib")


def trading_analysis(
    asset: str,
    interval: str,
    wallet: float,
    time_hours: int = 24,
    fee_percent: float = 0.1,  # Commissione % per ogni operazione (buy e sell)
    show: bool = True,
    step: float = 0.01,
    max_step: float = 0.4,
    atr_multiplier: float = 1.5,
    atr_window: int = 12,
    window_pivot: int = 80,
    rsi_window: int = 10,
    rsi_window2: int = 20,
    rsi_window3: int = 30,
    ema_window: int = 12,
    ema_window2: int = 24,
    ema_window3: int = 36,
    macd_short_window: int = 12,
    macd_long_window: int = 26,
    macd_signal_window: int = 9,
    kama_pow1: int = 2,
    kama_pow2: int = 30,
    rsi_buy_limit: int = 40,
    rsi_sell_limit: int = 60,
    macd_buy_limit: float = -0.4,
    macd_sell_limit: float = 0.4,
    num_cond: int = 1,
    stop_loss: int = 99,
    strategia: str = "",
    market_data: dict = None,
    # din_macd_div: float = 1.2, modello = None
):
    """
    Scarica le candele di 'asset' con intervallo 'interval' (tramite una funzione
    esterna get_market_data), calcola il SAR con i parametri 'step' e 'max_step',
    identifica segnali di acquisto/vendita, simula le operazioni in base al 'wallet'
    iniziale e restituisce un grafico Plotly con candlestick, SAR e segnali,
    oltre al DataFrame con tutte le operazioni, decurtando una commissione
    su ogni BUY e SELL (fee_percent).

    Parameters
    ----------
    asset : str
        Nome dell'asset (es. "BTCUSDT").
    interval : str
        Intervallo di tempo delle candele (es. "1h", "15m", ecc.).
    wallet : float
        Quantità di USDC/USDT a disposizione per le operazioni di trading.
    step : float
        Passo (step) per il calcolo del SAR (param. 'step' in PSARIndicator).
    max_step : float
        Valore massimo di step (param. 'max_step' in PSARIndicator).
    time_hours: int, optional
        tempo in ore che si vuole scaricare
    fee_percent : float, optional
        Percentuale di commissione per operazione (default 1.0, cioè 1%).

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Il grafico con candlestick, SAR e segnali di acquisto/vendita.
    trades_df : pandas.DataFrame
        Un DataFrame con tutte le operazioni effettuate, incluse informazioni
        su buy_time, sell_time, profit, volatilità del periodo, ecc.
    """

    # ======================================
    # Scarica i dati di mercato e calcola il SAR
    if market_data is None:
        # Otteniamo i dati di mercato (funzione esterna da definire)
        # df = get_market_data(asset=asset, interval=interval, limit=limit)
        df, actual_hours = get_market_data(asset=asset, interval=interval, time_hours=time_hours)
    else:
        df = market_data
        actual_hours = time_hours

    # Aggiungiamo una colonna per i massimi e i minimi relativi
    # # Utilizziamo i prezzi massimi ('High') e minimi ('Low')
    price_high = df["High"]
    price_low = df["Low"]
    # Trova gli indici dei massimi e minimi relativi
    order = int(window_pivot / 2)
    max_idx = argrelextrema(price_high.values, np.greater, order=order)[0]
    min_idx = argrelextrema(price_low.values, np.less, order=order)[0]
    # Inizializza gli array per massimi e minimi
    rel_max = []
    rel_min = []
    # Popola gli array con tuple (indice, prezzo)
    for i in min_idx:
        rel_min.append((df.index[i], df.loc[df.index[i], "Low"]))
    for i in max_idx:
        rel_max.append((df.index[i], df.loc[df.index[i], "High"]))

    # dinamic_atr = False
    # if strategia == "Dinamic ATR Bands" or strategia == "Dinamic Close ATR":
    #     dinamic_atr = True
    # df = calculate_relative_extrema(df)

    df = add_technical_indicator(
        df,
        step=step,
        max_step=max_step,
        rsi_window=rsi_window,
        rsi_window2=rsi_window2,
        rsi_window3=rsi_window3,
        ema_window=ema_window,
        ema_window2=ema_window2,
        ema_window3=ema_window3,
        macd_long_window=macd_long_window,
        macd_short_window=macd_short_window,
        macd_signal_window=macd_signal_window,
        atr_window=atr_window,
        atr_multiplier=atr_multiplier,
        kama_pow1=kama_pow1,
        kama_pow2=kama_pow2,
    )

    # ======================================
    # Identificazione dei segnali di acquisto e vendita in base alla strategia
    buy_signals = []
    sell_signals = []

    if strategia == "ATR Bands" or strategia == "Dinamic ATR Bands":
        buy_signals, sell_signals = atr_buy_sell_simulation(df=df, stop_loss_percent=stop_loss)

    if strategia == "Close ATR" or strategia == "Dinamic Close ATR":
        buy_signals, sell_signals = close_atr_buy_sell_simulation(df=df, stop_loss_percent=stop_loss)

    if strategia == "Buy/Sell Limits":
        buy_signals, sell_signals = buy_sell_limits_simulation(
            df=df,
            macd_buy_limit=macd_buy_limit,
            macd_sell_limit=macd_sell_limit,
            rsi_buy_limit=rsi_buy_limit,
            rsi_sell_limit=rsi_sell_limit,
            num_cond=num_cond,
        )

    if strategia == "Close Buy/Sell Limits":
        buy_signals, sell_signals = buy_sell_limits_close_simulation(
            df=df,
            macd_buy_limit=macd_buy_limit,
            macd_sell_limit=macd_sell_limit,
            rsi_buy_limit=rsi_buy_limit,
            rsi_sell_limit=rsi_sell_limit,
            num_cond=num_cond,
            stop_loss_percent=stop_loss,
        )

    if strategia == "ATR Live Trade":
        buy_signals, sell_signals = simulate_candles(
            raw_df=df,
            atr_window=atr_window,
            atr_multiplier=atr_multiplier,
            step=step,
            max_step=max_step,
            stop_loss_percent=stop_loss,
        )

    if strategia == "Close EMA Crossover":
        buy_signals, sell_signals = close_ema_crossover_simulation(df=df)

    if strategia == "Close Bullish EMA":
        buy_signals, sell_signals = close_bullish_ema_simulation(
            df=df, rsi_buy_limit=rsi_buy_limit, rsi_sell_limit=rsi_sell_limit
        )

    if strategia == "Close RSI Reverse":
        buy_signals, sell_signals = close_rsi_buy_sell_limits_simulation(df=df)

    if strategia == "Supertrend":
        buy_signals, sell_signals = supertrend_simulation(df=df)

    if strategia == "Trend Zones":
        buy_signals, sell_signals = trend_zone_simulation(df=df)

    if strategia == "TP/SL with ATR":
        buy_signals, sell_signals = tp_sl_simulation(df=df)

    if strategia == "Green Candles":
        buy_signals, sell_signals = green_candles_simulation(df=df)

    if strategia == "AI Model":
        buy_signals, sell_signals = ai_model_simulation(df=df, model=st.session_state["model"])

        # buy_signals.append((df[df['Prediction'] == 1].index, df[df['Prediction'] == 1]['Close']))
        # sell_signals.append((df[df['Prediction'] == 2].index, df[df['Prediction'] == 2]['Close']))

    # ======================================
    # Simulazione di trading con commissioni
    if strategia == "Close MACD Retest":  # or  strategia == "Trend Zones"
        operations = simulate_trading_with_commisions_multiple_buy(
            wallet=wallet, buy_signals=buy_signals, sell_signals=sell_signals, fee_percent=fee_percent
        )
    else:
        operations = simulate_trading_with_commisions(
            wallet=wallet, buy_signals=buy_signals, sell_signals=sell_signals, fee_percent=fee_percent
        )

    # ======================================
    # 4. Creazione del grafico
    rows = 2
    candlestick_height_px = 400
    indicators_height_px = candlestick_height_px / 2
    total_height = candlestick_height_px + ((rows - 1) * indicators_height_px)
    nominal_height = 1 / (rows + 1)
    candle_height = 2 * nominal_height
    row_heights = [candle_height] + [nominal_height] * (rows - 1)
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
        subplot_titles=(
            "Candlestick",
            "True and Relative Strength Index and Stochastic (TSI / RSI / STOCH)",
        ),
    )
    if show:
        trend_shapes = identify_trend_zones(df=df)
        index = 1
        # Candele (candlestick)
        fig.add_trace(
            go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=f"{asset}"
            ),
            row=index,
            col=1,
        )

        # Punti SAR (marker rossi)
        # fig.add_trace(go.Scatter(
        #     x=df.index,
        #     y=df['PSAR'],
        #     mode='markers',
        #     marker=dict(size=2, color='yellow', symbol='circle'),
        #     name='PSAR'
        # ),
        #     row=index, col=1
        # )

        # EMA SHORT
        fig.add_trace(
            go.Scatter(x=df.index, y=df["EMA20"], mode="lines", line=dict(color="Green", width=1), name="EMA SHORT"),
            row=index,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["EMA50"], mode="lines", line=dict(color="purple", width=1), name="EMA MED"),
            row=index,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["EMA100"], mode="lines", line=dict(color="coral", width=1), name="EMA LONG"),
            row=index,
            col=1,
        )

        fig.add_trace(
            go.Scatter(x=df.index, y=df["EMA200"], mode="lines", line=dict(color="Red", width=1), name="EMA OPEN"),
            row=index,
            col=1,
        )

        # KAMA
        fig.add_trace(
            go.Scatter(x=df.index, y=df["KAMA"], mode="lines", line=dict(color="yellow", width=1), name="KAMA"),
            row=index,
            col=1,
        )

        # Rolling ATR Bands
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Upper_Band"],
                mode="lines",
                line=dict(color="yellow", width=1, dash="dash"),
                name="Upper ATR",
            ),
            row=index,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Lower_Band"],
                mode="lines",
                line=dict(color="yellow", width=1, dash="dash"),
                name="Lower ATR",
            ),
            row=index,
            col=1,
        )

        # Massimi relativi
        if rel_max:
            max_times, max_prices = zip(*rel_max)
            fig.add_trace(
                go.Scatter(
                    x=max_times,
                    y=max_prices,
                    mode="markers",
                    marker=dict(size=10, color="red", symbol="square-open"),
                    name="Local Max",
                ),
                row=index,
                col=1,
            )
        # Minimi relativi
        if rel_min:
            min_times, min_prices = zip(*rel_min)
            fig.add_trace(
                go.Scatter(
                    x=min_times,
                    y=min_prices,
                    mode="markers",
                    marker=dict(size=10, color="green", symbol="square-open"),
                    name="Local Min",
                ),
                row=index,
                col=1,
            )

        # Segnali di acquisto
        if buy_signals:
            buy_times, buy_prices = zip(*buy_signals)
            fig.add_trace(
                go.Scatter(
                    x=buy_times,
                    y=buy_prices,
                    mode="markers",
                    marker=dict(size=14, color="green", symbol="triangle-up"),
                    name="Buy Signal",
                ),
                row=index,
                col=1,
            )

        # Segnali di vendita
        if sell_signals:
            sell_times, sell_prices = zip(*sell_signals)
            fig.add_trace(
                go.Scatter(
                    x=sell_times,
                    y=sell_prices,
                    mode="markers",
                    marker=dict(size=14, color="red", symbol="triangle-down"),
                    name="Sell Signal",
                ),
                row=index,
                col=1,
            )

        # STOCASTICO
        index += 1
        fig.add_trace(
            go.Scatter(x=df.index, y=df["STOCH"], mode="lines", line=dict(color="darkblue", width=1), name="STOCH"),
            row=index,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["STOCH_S"], mode="lines", line=dict(color="darkcyan", width=1), name="STOCH S"),
            row=index,
            col=1,
        )

        # TSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df["TSI"], mode="lines", line=dict(color="yellow", width=1), name="TSI"),
            row=index,
            col=1,
        )

        # RSI
        # fig.add_trace(go.Scatter(
        #     x=df.index, y=df['RSI_S'], mode='lines',
        #     line=dict(color='orange', width=1),
        #     name='RSI SMOOTH'
        # ), row=index, col=1)
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RSI"], mode="lines", line=dict(color="salmon", width=1), name="RSI SHORT"),
            row=index,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RSI2"], mode="lines", line=dict(color="pink", width=1), name="RSI MED"),
            row=index,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RSI3"], mode="lines", line=dict(color="purple", width=1), name="RSI LONG"),
            row=index,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[df.index.min(), df.index.max()],
                y=[rsi_sell_limit, rsi_sell_limit],
                mode="lines",
                line=dict(color="red", width=1, dash="dash"),
                name="Sell Limit",
            ),
            row=index,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[df.index.min(), df.index.max()],
                y=[rsi_buy_limit, rsi_buy_limit],
                mode="lines",
                line=dict(color="green", width=1, dash="dash"),
                name="Buy Limit",
            ),
            row=index,
            col=1,
        )

        # MACD
        # index += 1
        # fig.add_trace(go.Scatter(
        #     x=df.index, y=df['MACD_L'], mode='lines',
        #     line=dict(color='fuchsia', width=1, dash='dot'),
        #     name='MACD Line'
        # ), row=index, col=1)
        # fig.add_trace(go.Scatter(
        #     x=df.index, y=df['MACD_S'], mode='lines',
        #     line=dict(color='blue', width=1, dash='dot'),
        #     name='MACD Signal'
        # ), row=index, col=1)
        # fig.add_trace(go.Bar(
        #     x=df.index, y=df['MACD'], name='MACD',
        #     marker=dict(color='lightyellow')
        # ), row=index, col=1
        # )
        #
        # fig.add_trace(go.Scatter(
        #     x=[df.index.min(), df.index.max()], y=[macd_buy_limit, macd_buy_limit], mode='lines',
        #     line=dict(color='green', width=1, dash='dash'),
        #     name='Buy Limit'
        # ), row=index, col=1
        # )
        # fig.add_trace(go.Scatter(
        #     x=[df.index.min(), df.index.max()], y=[macd_sell_limit, macd_sell_limit], mode='lines',
        #     line=dict(color='red', width=1, dash='dash'),
        #     name='Sell Limit'
        # ), row=index, col=1)

        fig.update_layout(
            template="plotly_dark", xaxis_rangeslider_visible=False, height=total_height, shapes=trend_shapes
        )

    # ======================================
    # Creazione del DataFrame finale con le operazioni
    if operations:
        trades_df = pd.DataFrame(operations)
        # Aggiungiamo qualche metrica sul periodo analizzato
        apertura = df["Open"].iloc[0]  # Prezzo di apertura (prima candela)
        chiusura = df["Close"].iloc[-1]  # Prezzo di chiusura (ultima candela)
        high_max = df["High"].max()
        low_min = df["Low"].min()
        # Variazione percentuale (close finale su open iniziale)
        variazione = (chiusura - apertura) / apertura * 100
        # Volatilità: std dei rendimenti "Close-to-Close", in termini %
        volatilita = df["Close"].pct_change().std() * 100
        # Inseriamo questi valori su ogni riga del DataFrame trades_df.
        trades_df["massimo"] = high_max
        trades_df["minimo"] = low_min
        trades_df["variazione(%)"] = variazione
        trades_df["volatilita(%)"] = volatilita
    else:
        # Nessun trade effettuato
        trades_df = pd.DataFrame(
            columns=["Buy_Time", "Buy_Price", "Sell_Time", "Sell_Price", "Quantity", "Profit", "Wallet_After"]
        )

    print(
        f"{wallet} USDC su {asset}, fee={fee_percent}%, {interval}, strategia: {strategia}, "
        f"profitto totale={round(trades_df['Profit'].sum())} USD"
    )

    return fig, trades_df, actual_hours


if __name__ == "__main__":
    # ------------------------------
    # Configura il titolo della pagina e il logo
    st.set_page_config(
        page_title="CryptoFarm Simulator",  # Titolo della scheda del browser
        page_icon="📈",  # Icona (grafico che sale, simbolico per un mercato finanziario)
        layout="wide",  # Layout: "centered" o "wide"
        initial_sidebar_state="expanded",  # Stato iniziale della sidebar: "expanded", "collapsed", "auto"
    )
    if "df" not in st.session_state:
        st.session_state["df"] = None
    if "model" not in st.session_state:
        # load_signal_model trova da solo il formato del modello addestrato (gradient boosting
        # o rete), cosi' cambiare famiglia di modello non richiede di toccare la dashboard.
        st.session_state["model"] = load_signal_model()

    text_placeholder = st.empty()
    fig_placeholder = st.empty()
    st.sidebar.title("Market parameters")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        asset = st.text_input(label="Asset", placeholder="es. BTC, ETH, XRP...", max_chars=8, value="BTC")
        time_hours = st.number_input(label="Time Hours", min_value=0, value=240, step=24)

    with col2:
        currency = st.text_input(label="Currency", placeholder="es. USDC, USDT, EUR...", max_chars=8, value="USDC")
        interval = st.selectbox(
            label="Candle Interval", options=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"], index=3
        )

    symbol = asset + currency
    wallet = st.sidebar.number_input(label=f"Wallet ({currency})", min_value=0, value=100, step=1)
    st.sidebar.title("Indicators parameters")
    strategia = st.sidebar.selectbox(
        label="Strategia",
        options=[
            "-",
            "Close Buy/Sell Limits",
            "Close ATR",
            "Close Bullish EMA",
            "Close EMA Crossover",
            "Supetrend",
            "Trend Zones",
            "TP/SL with ATR",
            "Green Candles",
            "ATR Live Trade",
            "AI Model",
        ],
        index=0,
    )
    if st.sidebar.button("SIMULATE"):
        st.session_state["df"], _ = get_market_data(asset=symbol, interval=interval, time_hours=time_hours)

    col1, col2 = st.sidebar.columns(2)

    # step = col1.number_input(label="PSAR Step", min_value=0.001, max_value=1.000, value=0.01, step=0.001, format="%.3f")
    # max_step = col2.number_input(label="PSAR Max Step", min_value=0.01, max_value=1.0, value=0.4, step=0.01)
    atr_multiplier = col1.number_input(label="ATR Multiplier", min_value=0.1, max_value=50.0, value=1.6, step=0.1)
    atr_window = col2.number_input(label="ATR Window", min_value=1, max_value=100, value=5, step=1)

    col1, col2, col3 = st.sidebar.columns(3)
    rsi_window = col1.number_input(label="RSI Short", min_value=2, max_value=500, value=12, step=1)
    rsi_window2 = col2.number_input(label="Medium", min_value=2, max_value=500, value=24, step=1)
    rsi_window3 = col3.number_input(label="Long", min_value=2, max_value=500, value=36, step=1)

    ema_window = col1.number_input(label="EMA Short", min_value=1, max_value=500, value=10, step=1)
    ema_window2 = col2.number_input(label="Medium", min_value=1, max_value=500, value=50, step=1)
    ema_window3 = col3.number_input(label="Long", min_value=1, max_value=500, value=200, step=1)

    # macd_short_window = col1.number_input(label="MACD Short", min_value=0, max_value=500, value=12, step=1)
    # macd_long_window = col2.number_input(label="Long", min_value=0, max_value=500, value=26, step=1)
    # macd_signal_window = col3.number_input(label="Signal", min_value=0, max_value=500, value=9, step=1)

    col1, col2 = st.sidebar.columns(2)
    kama_pow1 = col1.number_input(label="KAMA Pow 1", min_value=1, max_value=1000, value=2, step=1)
    kama_pow2 = col2.number_input(label="Pow 2", min_value=1, max_value=1000, value=30, step=1)
    rsi_buy_limit = col1.number_input(label="RSI Buy limit", min_value=0, max_value=100, value=25, step=1)
    rsi_sell_limit = col2.number_input(label="RSI Sell limit", min_value=0, max_value=100, value=75, step=1)

    # macd_buy_limit = col1.number_input(label="MACD Buy Limit", min_value=-10.0, max_value=10.0, value=-2.5,
    # value=-0.66,
    #                                    step=0.01)
    # macd_sell_limit = col2.number_input(label="MACD Sell Limit", min_value=-10.0, max_value=10.0, value=2.5,
    # value=0.66,
    #                                    step=0.01)
    # din_macd_div = col1.number_input(label="ATR Dividend", min_value=-10.0, max_value=10.0, value=1.2,
    #                                  step=0.1)

    stop_loss = col2.number_input(label="Stop Loss %", min_value=0.1, max_value=100.0, value=99.0, step=1.0)

    num_cond = col1.number_input(label="Numero di condizioni", min_value=1, max_value=10, value=1, step=1)
    window_pivot = col2.number_input(label="Min-Max Window", min_value=2, max_value=500, value=100, step=2)

    if st.session_state["df"] is not None:
        if st.sidebar.button("SAVE DATA"):
            st.write(st.session_state["df"])

    csv_file = st.sidebar.text_input(label="CSV File", value="C:/Users/monini.m/Documents/market_data.csv")
    if st.sidebar.button("Read from CSV"):
        st.session_state["df"] = pd.read_csv(csv_file)
        st.session_state["df"].set_index("Open time", inplace=True)
        # Mantieni solo le colonne essenziali, converti a float
        st.session_state["df"] = st.session_state["df"][["Open", "High", "Low", "Close", "Volume"]].astype(float)

    show_graph = st.sidebar.checkbox(label="Show Graphs", value=1)

    if st.session_state["df"] is not None:
        fig, trades_df, actual_hours = trading_analysis(
            asset=symbol,
            interval=interval,
            wallet=wallet,  # Wallet iniziale
            # step=step,
            # max_step=max_step,
            time_hours=time_hours,
            fee_percent=0.1,  # %
            atr_multiplier=atr_multiplier,
            atr_window=atr_window,
            window_pivot=window_pivot,
            rsi_window=rsi_window,
            rsi_window2=rsi_window2,
            rsi_window3=rsi_window3,
            ema_window=ema_window,
            ema_window2=ema_window2,
            ema_window3=ema_window3,
            # macd_short_window=macd_short_window, macd_long_window=macd_long_window,
            # macd_signal_window=macd_signal_window,
            kama_pow1=kama_pow1,
            kama_pow2=kama_pow2,
            rsi_buy_limit=rsi_buy_limit,
            rsi_sell_limit=rsi_sell_limit,  # OK
            # macd_buy_limit=macd_buy_limit, macd_sell_limit=macd_sell_limit,  # NO, DA TOGLIERE
            num_cond=num_cond,
            stop_loss=stop_loss,
            strategia=strategia,
            # din_macd_div=din_macd_div,
            market_data=st.session_state["df"],
        )
        text_placeholder.subheader("Operations Report")

        if not trades_df.empty:
            # text_placeholder.write(trades_df)
            total_profit = trades_df["Profit"].sum()
            num_trades = len(trades_df)
            profitable_trades = trades_df[trades_df["Profit"] > 0]
            num_profitable = len(profitable_trades)
            win_rate = (num_profitable / num_trades * 100) if num_trades > 0 else 0.0
            text_placeholder.write(f"Total profit: {total_profit:.2f} {currency}, Winrate: {win_rate:.2f}%")
        else:
            text_placeholder.write("No operation performed.")
        if show_graph:
            fig_placeholder.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.sidebar.columns(2)
        start_date = col1.date_input(label="Start Date")
        end_date = col2.date_input(label="End Date")
        if st.sidebar.button("Get Data from Dates"):
            data, _ = get_market_data_between_dates(
                asset=symbol, interval=interval, start_date=start_date, end_date=end_date
            )
            st.write(data)
